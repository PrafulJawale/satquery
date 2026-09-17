"""core/geo.py -- CRS handling, reprojection and footprints (PHASE 4).

WHY THIS MODULE EXISTS
----------------------
A web map (Leaflet/Folium) speaks two coordinate systems and no others:

    EPSG:4326  for coordinates and bounds   (lon/lat degrees)
    EPSG:3857  for the imagery underneath   (Web Mercator metres)

Our Sentinel-2 sample is in **EPSG:32636** (UTM zone 36N): easting 377200,
northing 3441820. Those are metres from a false origin, not degrees. Passing
them to Leaflet would place the scene nowhere on Earth — and worse, would look
plausible if someone merely "normalised" the numbers.

So we REPROJECT: resample the pixels into a Mercator-aligned grid. This is not
relabelling; the destination grid has its own width, height, transform and
pixel size (at 31 deg N a 10 m ground pixel becomes ~11.7 Mercator units).

THE RULE THIS MODULE ENFORCES
-----------------------------
    analysis      -> native CRS, native transform, native values
    visualisation -> a WebRaster: a DISPLAY COPY in the destination CRS

`WebRaster` always remembers where it came from (`source_crs`,
`source_transform`, `source_shape`) and is marked display-only. Nothing here
mutates the analytical result.

Pure Python + rasterio/shapely/pyproj. No Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject

from .preview import stretch_to_uint8

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"

DEFAULT_MIN_VALID_FRACTION = 0.5   # a destination pixel needs >=50% valid source


class GeoError(Exception):
    """Raised when a geographic operation is impossible (e.g. raster has no CRS)."""


class MissingCRSError(GeoError):
    pass


# --------------------------------------------------------------------------- #
# the display-only raster
# --------------------------------------------------------------------------- #
@dataclass
class WebRaster:
    """A raster resampled into a web-map CRS. DISPLAY ONLY.

    `array` is float32 with NaN where invalid, shaped (H, W) or (C, H, W).
    `mask` is the AND of band validity.
    """

    array: Any
    mask: Any
    transform: Any
    crs: Any
    bounds_wgs84: Tuple[float, float, float, float]
    source_crs: Any
    source_transform: Any
    source_shape: Tuple[int, int]
    resampling: str = "average"
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.array.shape[-2:])  # type: ignore[return-value]

    @property
    def leaflet_bounds(self) -> List[List[float]]:
        """[[lat_min, lon_min], [lat_max, lon_max]] -- the order Leaflet wants."""
        lon_min, lat_min, lon_max, lat_max = self.bounds_wgs84
        return [[lat_min, lon_min], [lat_max, lon_max]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "display_crs": str(self.crs),
            "source_crs": str(self.source_crs),
            "display_shape": list(self.shape),
            "source_shape": list(self.source_shape),
            "transform": tuple(self.transform)[:6],
            "source_transform": tuple(self.source_transform)[:6],
            "bounds_wgs84": list(self.bounds_wgs84),
            "leaflet_bounds": self.leaflet_bounds,
            "resampling": self.resampling,
            "valid_fraction": float(np.mean(self.mask)) if self.mask.size else 0.0,
            "min_valid_fraction": self.min_valid_fraction,
        }


# --------------------------------------------------------------------------- #
# coordinate helpers
# --------------------------------------------------------------------------- #
def pixel_to_crs(transform: Affine, col: float, row: float) -> Tuple[float, float]:
    """(col, row) -> (x, y) in the raster's own CRS. Written out longhand
    because `Affine.__mul__` is deprecated."""
    a, b, c, d, e, f = transform[:6]
    return (a * col + b * row + c, d * col + e * row + f)


def transform_coords(
    src_crs: Any, dst_crs: Any, xs: Sequence[float], ys: Sequence[float]
) -> Tuple[List[float], List[float]]:
    """Thin, always_xy=True wrapper over pyproj."""
    from pyproj import Transformer

    if src_crs is None:
        raise MissingCRSError("Source CRS is None; coordinates cannot be transformed.")
    transformer = Transformer.from_crs(CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True)
    xs_t, ys_t = transformer.transform(list(xs), list(ys))
    return list(xs_t), list(ys_t)


def pixel_to_lonlat(transform: Affine, crs: Any, col: float, row: float) -> Tuple[float, float]:
    """Pixel -> (longitude, latitude). The two-step chain: pixel -> CRS -> WGS84."""
    x, y = pixel_to_crs(transform, col, row)
    lons, lats = transform_coords(crs, WGS84, [x], [y])
    return float(lons[0]), float(lats[0])


def grid_bounds_wgs84(transform: Affine, crs: Any, width: int, height: int) -> Tuple[float, float, float, float]:
    """WGS84 bounding box of an axis-aligned grid.

    Exact for a Mercator destination grid (its edges are straight in Mercator
    and constant-latitude in WGS84). For a rotated source grid, use
    `raster_footprint` instead -- the bounding box alone is not the footprint.
    """
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    xs = [pixel_to_crs(transform, c, r)[0] for c, r in corners]
    ys = [pixel_to_crs(transform, c, r)[1] for c, r in corners]
    lons, lats = transform_coords(crs, WGS84, xs, ys)
    return (min(lons), min(lats), max(lons), max(lats))


# --------------------------------------------------------------------------- #
# footprint (the REAL outline, densified -- not just a bounding box)
# --------------------------------------------------------------------------- #
def raster_footprint(
    transform: Affine,
    crs: Any,
    width: int,
    height: int,
    segments_per_edge: int = 32,
):
    """Densified outline of the raster in EPSG:4326, as a shapely Polygon.

    Why densify: reprojecting a projected grid into lon/lat BENDS straight
    edges into curves. Four corners alone would understate the real coverage
    and, for a rotated raster, describe the wrong shape entirely.

    Works for rotated/sheared transforms: the corner polygon built from the
    affine is the true parallelogram of the grid, not an axis-aligned box.
    """
    from shapely.geometry import Polygon
    from shapely import segmentize

    if crs is None:
        raise MissingCRSError("Raster has no CRS; a footprint cannot be computed.")

    corners = [
        pixel_to_crs(transform, 0.0, 0.0),
        pixel_to_crs(transform, float(width), 0.0),
        pixel_to_crs(transform, float(width), float(height)),
        pixel_to_crs(transform, 0.0, float(height)),
    ]
    ring = Polygon(corners)

    # Segment length: split each edge into `segments_per_edge` pieces.
    coords = list(ring.exterior.coords)
    diag = max(
        math.dist(coords[0], coords[2] if len(coords) > 2 else coords[1]),
        1e-12,
    )
    max_segment_length = diag / max(int(segments_per_edge), 1)
    densified = segmentize(ring, max_segment_length)

    lons, lats = transform_coords(crs, WGS84, *zip(*list(densified.exterior.coords)))
    return Polygon(list(zip(lons, lats)))


def native_footprint(transform: Affine, width: int, height: int):
    """Exact outline of the raster grid **in its own CRS**, as a shapely Polygon.

    An affine maps the unit square onto a parallelogram, so in the raster's own
    CRS the extent is exactly the quadrilateral through the four grid corners --
    including rotation and shear. No densification is needed here (the edges are
    straight by definition); densification only matters once we leave this CRS,
    which is what `raster_footprint` does for display in EPSG:4326.

    This is the polygon Phase 5 intersects a user selection with, because the
    intersection must happen in the same CRS the pixels are indexed in.
    """
    from shapely.geometry import Polygon

    corners = [
        pixel_to_crs(transform, 0.0, 0.0),
        pixel_to_crs(transform, float(width), 0.0),
        pixel_to_crs(transform, float(width), float(height)),
        pixel_to_crs(transform, 0.0, float(height)),
    ]
    return Polygon(corners)


def footprint_feature(polygon, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GeoJSON Feature (dict) for a shapely polygon, safe to hand to folium."""
    import json

    geom = json.loads(_shapely_to_geojson(polygon))
    return {"type": "Feature", "properties": dict(properties or {}), "geometry": geom}


def _shapely_to_geojson(polygon) -> str:
    from shapely.geometry import mapping
    import json

    return json.dumps(mapping(polygon))


# --------------------------------------------------------------------------- #
# reprojection
# --------------------------------------------------------------------------- #
def destination_grid(
    src_crs: Any,
    dst_crs: Any,
    width: int,
    height: int,
    bounds: Sequence[float],
    max_pixels: Optional[int] = None,
) -> Tuple[Affine, int, int]:
    """Ask GDAL for a sensible destination grid, then cap its size.

    `calculate_default_transform` is the correct way to pick the destination
    transform: it accounts for the change in pixel size and shape between the
    two CRSs. Guessing "same width/height" would distort the image.
    """
    dst_transform, dst_w, dst_h = calculate_default_transform(
        CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), width, height, *bounds
    )
    if max_pixels and dst_w * dst_h > max_pixels:
        # Coarsen the resolution to respect the pixel budget, keeping GDAL's
        # origin. Round the grid UP: rounding to nearest can leave the grid a
        # fraction of a pixel SHORT of the true footprint, so the overlay would
        # sit slightly inside the outline it is supposed to cover.
        factor = math.sqrt((dst_w * dst_h) / max_pixels)
        dst_transform = Affine(
            dst_transform.a * factor, dst_transform.b * factor, dst_transform.c,
            dst_transform.d * factor, dst_transform.e * factor, dst_transform.f,
        )
        dst_w = max(1, math.ceil(dst_w / factor))
        dst_h = max(1, math.ceil(dst_h / factor))
    return dst_transform, dst_w, dst_h


def reproject_array(
    array: np.ndarray,
    src_transform: Affine,
    src_crs: Any,
    dst_crs: Any = WEB_MERCATOR,
    resampling: Resampling = Resampling.average,
    max_pixels: Optional[int] = None,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    nodata_aware: bool = True,
) -> WebRaster:
    """Resample an in-memory array into a web CRS.

    Nodata handling (the subtle part)
    ---------------------------------
    Feeding raw NaN into `average` resampling is unsafe: GDAL compares nodata
    with `==`, and NaN != NaN, so NaN pixels can be averaged in and poison
    whole output pixels -- making nodata bleed outward.

    Instead we reproject two grids and divide:
        filled  = value where valid, else 0        -- averaged  -> mean * fraction
        weight  = 1 where valid,  else 0           -- averaged  -> fraction
        result  = filled / weight  (valid where weight >= min_valid_fraction)

    That is a true mean over the VALID source pixels only.
    """
    if src_crs is None:
        raise MissingCRSError("Cannot reproject a raster that has no CRS.")

    arr = np.asarray(array)
    if arr.ndim == 2:
        stack = arr[np.newaxis, ...].astype(np.float32)
        squeeze = True
    elif arr.ndim == 3:
        stack = arr.astype(np.float32)
        squeeze = False
    else:
        raise ValueError(f"Expected a 2-D or 3-D array, got {arr.ndim}-D")

    src_height, src_width = stack.shape[-2:]
    mask = np.isfinite(stack)
    band_mask = mask.all(axis=0)                       # AND across bands

    src_transform = Affine(*src_transform[:6])
    bounds = array_bounds(src_height, src_width, src_transform)
    dst_transform, dst_w, dst_h = destination_grid(
        src_crs, dst_crs, src_width, src_height, bounds, max_pixels=max_pixels
    )

    dst_crs_obj = CRS.from_user_input(dst_crs)
    filled = np.where(mask, stack, 0.0).astype(np.float32)
    weight_src = band_mask.astype(np.float32)

    dst_stack = np.full((stack.shape[0], dst_h, dst_w), np.nan, dtype=np.float32)
    dst_weight = np.zeros((dst_h, dst_w), dtype=np.float32)

    src_crs_obj = CRS.from_user_input(src_crs)
    for b in range(stack.shape[0]):
        reproject(
            source=filled[b],
            destination=dst_stack[b],
            src_transform=src_transform,
            src_crs=src_crs_obj,
            dst_transform=dst_transform,
            dst_crs=dst_crs_obj,
            resampling=resampling,
            dst_nodata=np.nan,
        )
    reproject(
        source=weight_src,
        destination=dst_weight,
        src_transform=src_transform,
        src_crs=src_crs_obj,
        dst_transform=dst_transform,
        dst_crs=dst_crs_obj,
        resampling=Resampling.average,     # the weight grid is always averaged
        dst_nodata=0.0,
    )

    if nodata_aware:
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(dst_weight > 0, dst_stack / np.maximum(dst_weight, 1e-12), np.nan)
        out = np.where(dst_weight >= min_valid_fraction, out, np.nan).astype(np.float32)
    else:
        out = dst_stack

    dst_mask = np.isfinite(out).all(axis=0)
    bounds_wgs84 = grid_bounds_wgs84(dst_transform, dst_crs_obj, dst_w, dst_h)

    result = out[0] if squeeze else out
    return WebRaster(
        array=result,
        mask=dst_mask,
        transform=dst_transform,
        crs=dst_crs_obj,
        bounds_wgs84=tuple(round(float(v), 10) for v in bounds_wgs84),  # type: ignore[arg-type]
        source_crs=src_crs_obj,
        source_transform=src_transform,
        source_shape=(src_height, src_width),
        resampling=getattr(resampling, "name", str(resampling)),
        min_valid_fraction=min_valid_fraction,
    )


def reproject_bands(
    ds: rasterio.DatasetReader,
    band_indices: Sequence[int],
    dst_crs: Any = WEB_MERCATOR,
    resampling: Resampling = Resampling.average,
    max_pixels: Optional[int] = None,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
) -> WebRaster:
    """Read bands (masked) and reproject them together, so they stay aligned."""
    idx = list(band_indices)
    # Cast BEFORE filling: NaN cannot be stored in a uint16 masked array.
    stack = np.ma.filled(ds.read(idx, masked=True).astype(np.float32), np.nan)
    return reproject_array(
        stack,
        src_transform=ds.transform,
        src_crs=ds.crs,
        dst_crs=dst_crs,
        resampling=resampling,
        max_pixels=max_pixels,
        min_valid_fraction=min_valid_fraction,
    )


# --------------------------------------------------------------------------- #
# display encoding -- the ONLY place a WebRaster becomes pixels
# --------------------------------------------------------------------------- #
def web_rgba_from_bands(
    web: WebRaster,
    stretch_bounds: Sequence[Tuple[float, float]],
    channel_order: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    """Stretch reprojected bands to 8-bit RGBA, preserving a given stretch.

    `stretch_bounds` comes from the native composite, so the map shows exactly
    the same stretch as the Phase 2 panel. Invalid pixels -> alpha 0.
    """
    stack = web.array if web.array.ndim == 3 else web.array[np.newaxis, ...]
    planes = []
    for k, band_idx in enumerate(channel_order):
        lo, hi = stretch_bounds[band_idx]
        planes.append(stretch_to_uint8(stack[band_idx], web.mask, lo, hi))
    rgb = np.dstack(planes)
    alpha = np.where(web.mask, 255, 0).astype(np.uint8)
    rgb[~web.mask] = 0                    # never paint colour where data is missing
    return np.dstack([rgb, alpha])


def web_rgba_from_values(
    web: WebRaster,
    vmin: float,
    vmax: float,
    colormap: str = "RdYlGn",
) -> np.ndarray:
    """Colour a single-band WebRaster (e.g. NDVI) with transparency for invalid."""
    from .preview import ndvi_to_rgba

    return ndvi_to_rgba(web.array, web.mask, colormap=colormap, vmin=vmin, vmax=vmax)


def rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    """Encode an RGBA array as an optimised PNG (no temp files)."""
    from PIL import Image
    import io

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def lonlat_to_pixel(
    transform: Affine, crs: Any, lon: float, lat: float
) -> Tuple[Optional[int], Optional[int]]:
    """(longitude, latitude) -> (row, col) in the raster's own grid.

    The inverse of `pixel_to_lonlat`, used when a user clicks the map: we walk
    backwards WGS84 -> source CRS -> pixel, then read the value that is actually
    stored there. This is how the map proves its own georeferencing.
    """
    from rasterio.transform import rowcol

    if crs is None:
        raise MissingCRSError("Raster has no CRS; coordinates cannot be inverted.")
    xs, ys = transform_coords(WGS84, crs, [lon], [lat])
    try:
        rows, cols = rowcol(transform, xs[0], ys[0])
        return int(rows), int(cols)
    except Exception:
        return None, None
