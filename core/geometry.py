"""core/geometry.py -- CRS-safe geometry helpers (PHASE 5).

WHY THIS MODULE EXISTS
----------------------
A shape drawn on a Leaflet map arrives as GeoJSON in **EPSG:4326 lon/lat
degrees**. The raster it must be compared with lives in a possibly *projected*
CRS such as EPSG:32636 (metres of easting/northing). Degrees and metres are
different quantities, so the drawn shape MUST be transformed into the raster CRS
before it is intersected with the footprint, masked, or measured.

The traps this module exists to avoid:

1. Treating lon/lat as pixel/easting coordinates. `31.82, 31.20` is not
   `387640 E, 3431480 N` -- using one for the other moves the shape hundreds of
   kilometres and can still *look* plausible.
2. Area in degrees squared. One degree of longitude is ~111 km at the equator
   and ~95 km at 31 deg N, so a degree-square is not a square and its "area"
   means nothing. Areas are computed geodesically unless the CRS is projected.
3. Planar area in a projected CRS whose units are not metres (US survey feet,
   for example). The CRS axis unit conversion factor is applied.
4. Axis-order confusion. Everything here is always_xy=True: (lon, lat) in,
   (x, y) out.

Pure Python + shapely + pyproj. No Streamlit, no rasterio.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pyproj import CRS as PyCRS
from pyproj import Geod, Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

from .geo import MissingCRSError

CRS84 = "EPSG:4326"          # GeoJSON's mandated CRS: lon, lat in WGS84
_GEOD = Geod(ellps="WGS84")
POLYGONAL = (Polygon, MultiPolygon)


class GeometryError(Exception):
    """Base class for geometry problems that the UI can explain."""


class InvalidGeometryError(GeometryError):
    """A geometry that exists but cannot be used (empty, zero area, unparsable)."""


class UnsupportedGeometryError(GeometryError):
    """A geometry that is not an area (Point, LineString, ...)."""


# --------------------------------------------------------------------------- #
# CRS helpers
# --------------------------------------------------------------------------- #
def as_crs(value: Any) -> PyCRS:
    if value is None:
        raise MissingCRSError("No CRS available; geometry cannot be placed or measured.")
    return PyCRS.from_user_input(value)


def crs_label(value: Any) -> str:
    try:
        c = as_crs(value)
    except MissingCRSError:
        return "no CRS"
    epsg = c.to_epsg()
    return f"EPSG:{epsg}" if epsg else (c.name or str(c))


def is_geographic(value: Any) -> bool:
    return bool(as_crs(value).is_geographic)


def metres_per_unit(value: Any) -> float:
    """How many metres one unit of the CRS axis is (1.0 for metres, 0.3048 for feet)."""
    c = as_crs(value)
    try:
        factor = float(c.axis_info[0].unit_conversion_factor)
    except Exception:
        return 1.0
    return factor if factor > 0 else 1.0


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def parse_geometry(obj: Any) -> BaseGeometry:
    """GeoJSON Feature | geometry mapping | shapely object -> shapely geometry.

    Accepts anything Leaflet.Draw (or a user) is likely to hand us, including a
    FeatureCollection, in which case the LAST feature is taken (the most
    recently drawn one -- see `roi.pick_active_drawing`).
    """
    if obj is None:
        raise InvalidGeometryError("No geometry was provided.")
    if isinstance(obj, BaseGeometry):
        return obj
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if isinstance(obj, dict):
        kind = obj.get("type")
        if kind == "Feature":
            geom = obj.get("geometry")
            if not geom:
                raise InvalidGeometryError("GeoJSON Feature has no geometry.")
            return parse_geometry(geom)
        if kind == "FeatureCollection":
            features = obj.get("features") or []
            if not features:
                raise InvalidGeometryError("GeoJSON FeatureCollection is empty.")
            return parse_geometry(features[-1])
        if kind:
            try:
                return shapely_shape(obj)
            except Exception as exc:  # pragma: no cover - defensive
                raise InvalidGeometryError(f"Could not read GeoJSON {kind}: {exc}")
    raise InvalidGeometryError(f"Cannot interpret {type(obj).__name__} as a geometry.")


def is_polygonal(geom: BaseGeometry) -> bool:
    return isinstance(geom, POLYGONAL)


def ensure_polygonal(geom: BaseGeometry) -> BaseGeometry:
    """Reject anything that is not an area (a marker or a line is not an ROI)."""
    if isinstance(geom, POLYGONAL):
        return geom
    raise UnsupportedGeometryError(
        f"{geom.geom_type} is not an area. Draw a rectangle or a polygon."
    )


def polygonal_parts(geom: BaseGeometry) -> List[BaseGeometry]:
    """The polygonal, non-empty, non-degenerate components of any geometry."""
    parts: List[BaseGeometry] = []
    for part in _iter_parts(geom):
        if isinstance(part, POLYGONAL) and not part.is_empty and part.area > 0:
            parts.append(part)
    return parts


def _iter_parts(geom: BaseGeometry) -> Iterable[BaseGeometry]:
    """Components of any geometry: Polygon -> itself, Multi*/GeometryCollection -> parts.

    Note that MultiPolygon DOES have `.geoms` and must be unpacked, otherwise a
    two-part selection would be reported as one part.
    """
    if geom is None:
        return []
    if hasattr(geom, "geoms"):
        return list(geom.geoms)
    return [geom]


# --------------------------------------------------------------------------- #
# repair
# --------------------------------------------------------------------------- #
def repair(geom: BaseGeometry) -> Tuple[BaseGeometry, bool]:
    """Make an invalid polygon usable, REPORTING that we did so.

    Only `shapely.validation.make_valid` is used -- no silent buffering, no
    silent dropping of rings. If nothing polygonal survives, the geometry is
    rejected rather than quietly emptied.
    """
    if geom.is_valid:
        return geom, False
    fixed = make_valid(geom)
    if fixed.is_empty:
        raise InvalidGeometryError("The drawn shape is invalid and could not be repaired.")
    parts = polygonal_parts(fixed)
    if not parts:
        raise InvalidGeometryError(
            "The drawn shape is invalid and contains no usable area after repair."
        )
    merged = parts[0] if len(parts) == 1 else MultiPolygon(
        [p if isinstance(p, Polygon) else p for p in parts]
    )
    return merged, True


# --------------------------------------------------------------------------- #
# transformation
# --------------------------------------------------------------------------- #
def segmentize(geom: BaseGeometry, max_segment_length: float) -> BaseGeometry:
    """Split long edges so that a later CRS change follows the curve."""
    if max_segment_length and max_segment_length > 0 and not geom.is_empty:
        try:
            from shapely import segmentize as _segmentize

            return _segmentize(geom, max_segment_length)
        except Exception:  # pragma: no cover - older shapely
            return geom
    return geom


def transform_geometry(
    geom: BaseGeometry,
    src_crs: Any,
    dst_crs: Any,
    densify: bool = True,
    max_segment_degrees: float = 0.01,
) -> BaseGeometry:
    """Reproject a geometry with pyproj (always_xy=True).

    Straight edges are densified first (by default every ~0.01 deg, about
    1.1 km) because a straight line in lon/lat is not straight in a projection.
    The result is re-validated, since a projection can turn a valid ring into an
    invalid one at the poles or the antimeridian.
    """
    if src_crs is None or dst_crs is None:
        raise MissingCRSError("Both a source and a destination CRS are required.")
    src, dst = as_crs(src_crs), as_crs(dst_crs)
    if src.equals(dst):
        return geom
    if densify and not is_geographic(src):
        # densifying a projected geometry by degrees makes no sense; fall back
        max_segment_degrees = 0.0
    work = segmentize(geom, max_segment_degrees) if densify else geom
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    try:
        moved = shapely_transform(
            lambda xs, ys: transformer.transform(xs, ys), work
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise GeometryError(f"Transformation failed: {exc}")
    if not moved.is_valid:
        moved, _ = repair(moved)
    return moved


def to_wgs84(geom: BaseGeometry, src_crs: Any) -> BaseGeometry:
    return transform_geometry(geom, src_crs, CRS84)


# --------------------------------------------------------------------------- #
# area
# --------------------------------------------------------------------------- #
def planar_area_m2(geom: BaseGeometry, crs: Any) -> float:
    """Area in m2 from the geometry's own plane -- ONLY valid for projected CRS."""
    c = as_crs(crs)
    if c.is_geographic:
        raise GeometryError(
            "Planar area is not defined for a geographic CRS "
            "(degrees are not a length). Use geodesic_area_m2."
        )
    return float(geom.area) * metres_per_unit(c) ** 2


def geodesic_area_m2(geom: BaseGeometry, crs: Any = CRS84) -> float:
    """Ellipsoidal area in m2 -- correct for any CRS, including geographic ones."""
    in_wgs84 = geom if as_crs(crs).equals(CRS84) else to_wgs84(geom, crs)
    area, _perimeter = _GEOD.geometry_area_perimeter(in_wgs84)
    return abs(float(area))


def area_m2(
    geom: BaseGeometry, crs: Any, method: str = "auto"
) -> Tuple[float, str]:
    """(area in m2, human-readable method).

    auto     -> planar for a projected CRS (exact enough, and what Phase 6 will
                rasterise against), geodesic for a geographic CRS.
    geodesic -> always ellipsoidal.
    planar   -> always planar (raises for a geographic CRS).
    """
    c = as_crs(crs)
    if method == "geodesic" or (method == "auto" and c.is_geographic):
        return geodesic_area_m2(geom, crs), "geodesic (WGS84 ellipsoid)"
    if method == "planar" and c.is_geographic:
        raise GeometryError("Planar area requested for a geographic CRS.")
    unit = "metre" if abs(metres_per_unit(c) - 1.0) < 1e-9 else "non-metre (converted)"
    return planar_area_m2(geom, crs), f"planar ({crs_label(c)}, {unit})"


def format_area(area_m2_value: float) -> str:
    if area_m2_value is None:
        return "—"
    if area_m2_value < 10_000:
        return f"{area_m2_value:,.0f} m²"
    if area_m2_value < 1_000_000:
        return f"{area_m2_value / 10_000:,.2f} ha ({area_m2_value:,.0f} m²)"
    return f"{area_m2_value / 1e6:,.3f} km² ({area_m2_value / 10_000:,.1f} ha)"


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #
def geojson_of(geom: Optional[BaseGeometry]) -> Optional[Dict[str, Any]]:
    """Plain, JSON-safe GeoJSON geometry mapping (safe for session state)."""
    if geom is None:
        return None
    return json.loads(json.dumps(mapping(geom)))


def geometry_from_geojson(obj: Any) -> BaseGeometry:
    return parse_geometry(obj)


def wkt_of(geom: Optional[BaseGeometry]) -> Optional[str]:
    return None if geom is None else geom.wkt


def geometry_from_wkt(text: Optional[str]) -> Optional[BaseGeometry]:
    if not text:
        return None
    from shapely import wkt as shapely_wkt

    return shapely_wkt.loads(text)
