"""core/raster.py -- GeoTIFF ingestion (PHASE 1).

WHAT THIS MODULE DOES
---------------------
Opens a GeoTIFF and returns a `RasterInfo` describing it: size, bands, dtypes,
CRS, affine transform, bounds, footprint in WGS84, nodata, tiling, overviews,
and a list of honest warnings.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* It does not display anything (that is `ui/`).
* It does not resample, resize or reproject the data (that is Phase 2/4).
* It does not guess band meanings (the user confirms them in Phase 3).

Only dependency: rasterio (+ numpy, which rasterio already brings in).

KEY CONCEPTS (read this once, it saves hours later)
---------------------------------------------------
1. CRS  -- Coordinate Reference System. "Which numbering system are these
   coordinates in?" EPSG:32643 = WGS 84 / UTM zone 43N (most of Maharashtra).
   EPSG:4326 = plain lon/lat degrees. Web maps (Leaflet/Folium) need EPSG:4326
   for coordinates, and EPSG:3857 (Web Mercator) for rasters underneath.

2. AFFINE TRANSFORM -- a 6-number recipe mapping (column, row) -> (x, y):
       x = a*col + b*row + c
       y = d*col + e*row + f
   For a normal north-up image: a = pixel width, e = -pixel height, b = d = 0.
   If b or d are non-zero the image is ROTATED: its bounding box in lon/lat is
   not a rectangle, so naive "corner" mapping is wrong. We detect that here.

3. BOUNDS vs FOOTPRINT -- `bounds` is the axis-aligned box; `footprint` is the
   actual outline. For rotated rasters (and for anything crossing a UTM zone or
   projected far from its central meridian) the outline is a curved quadrilateral.
   Projected rasters reprojected to lon/lat have CURVED edges, so we DENSIFY the
   outline (insert intermediate points along each edge) before transforming.
   Four corners alone would cut the true coverage short.

4. NODATA -- pixels that mean "no observation" (cloud gaps, scene edges...).
   If a file declares nodata=0 and you compute statistics without masking it,
   every background pixel is silently treated as a real measurement.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window

from .models import BandInfo, Coord, RasterInfo, SpatialInfo

WGS84 = "EPSG:4326"

# Trimmed subset of GDAL metadata keys we bother to show. Real scenes can carry
# hundreds of keys (RPCs, STAC JSON, processing history) that would swamp the UI.
_INTERESTING_TAGS = (
    "AREA_OR_POINT",
    "TIFFTAG_DATETIME",
    "TIFFTAG_IMAGEDESCRIPTION",
    "TIFFTAG_SOFTWARE",
    "TIFFTAG_ARTIST",
    "TIFFTAG_DOCUMENTNAME",
    "scale_factor",
    "add_offset",
    "_FillValue",
    "OVR_RESAMPLING_ALG",
)


# --------------------------------------------------------------------------- #
# Opening files
# --------------------------------------------------------------------------- #
def open_dataset(path: str) -> DatasetReader:
    """Open a GeoTIFF from disk / URL (GDAL virtual paths like /vsicurl/ work).

    Returns an OPEN dataset. Always use it as a context manager:
        with open_dataset(p) as ds: ...
    """
    return rasterio.open(path)


def open_upload(file_bytes: bytes) -> DatasetReader:
    """Open a GeoTIFF that arrived as raw bytes (Streamlit file_uploader).

    Why MemoryFile? Streamlit gives us a SpooledTemporaryFile-like object, not a
    path, and GDAL cannot read from a Python file object directly. MemoryFile
    hands the bytes to GDAL's in-memory VSI driver without touching disk.

    Caveat: this keeps the whole upload in RAM. Fine for prototype-sized
    GeoTIFFs (< a few hundred MB); a production system would spool to /vsimem
    or write to disk once and cache the path.
    """
    memfile = MemoryFile(file_bytes)
    # The MemoryFile must stay alive as long as the dataset does, so we attach
    # it to the dataset object and keep a module-level registry of open handles.
    ds = memfile.open()
    ds._satquery_memfile = memfile  # type: ignore[attr-defined]
    _OPEN_MEMFILES.append(memfile)
    return ds


# Keeps MemoryFile objects from being garbage-collected while datasets are open.
_OPEN_MEMFILES: List[MemoryFile] = []


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def pixel_size(transform: Affine) -> Tuple[float, float]:
    """(pixel width, pixel height) as positive numbers.

    `e` is negative for north-up images; `a` is the x step, `e` the y step.
    For rotated rasters we fall back to the affine's own scale components.
    """
    a, b, _c, d, e, _f = transform[:6]
    sx = math.hypot(a, b)
    sy = math.hypot(d, e)
    return sx, sy


def is_north_up(transform: Affine, eps: float = 1e-9) -> bool:
    """True when the transform has no rotation/skew (b == d == 0)."""
    _a, b, _c, d, _e, _f = transform[:6]
    return abs(b) < eps and abs(d) < eps


def outer_corners_native(ds: DatasetReader) -> List[Tuple[float, float]]:
    """The 4 outer corners of the raster grid, in the file's own CRS.

    Uses `rasterio.transform.xy` (rather than `transform * (col, row)`, which is
    deprecated) so rotated rasters give the true parallelogram corners instead
    of an axis-aligned box.
    """
    from rasterio.transform import xy

    t = ds.transform
    w, h = float(ds.width), float(ds.height)
    rows = [0.0, 0.0, h, h]
    cols = [0.0, w, w, 0.0]
    xs, ys = xy(t, rows, cols)
    return list(zip(xs, ys))


def densify_ring(ring: Sequence[Tuple[float, float]], points_per_edge: int = 16) -> List[Tuple[float, float]]:
    """Insert extra points along each edge of a polygon ring.

    Why: reprojecting a projected raster (UTM) into lon/lat bends straight
    lines into curves. Sampling only 4 corners understates the real footprint.
    16 points per edge is plenty for scene-sized rasters and costs nothing.
    """
    out: List[Tuple[float, float]] = []
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        for t in np.linspace(0.0, 1.0, points_per_edge, endpoint=False):
            out.append((x0 + (x1 - x0) * float(t), y0 + (y1 - y0) * float(t)))
    return out


def transform_coords(
    src_crs: Optional[CRS], xs: Sequence[float], ys: Sequence[float]
) -> Tuple[List[float], List[float]]:
    """Native CRS -> EPSG:4326 (lon, lat), densification-safe.

    rasterio.warp.transform uses (lon, lat) ordering regardless of the CRS's
    declared axis order, which is exactly the convention we standardise on.
    """
    if src_crs is None:
        raise ValueError("Raster has no CRS; coordinates cannot be transformed.")
    from rasterio.warp import transform as _warp_transform

    return _warp_transform(src_crs, WGS84, list(xs), list(ys))


def footprint_wgs84(ds: DatasetReader, points_per_edge: int = 16) -> Optional[List[Coord]]:
    """Densified outline of the raster in (lon, lat); None if it has no CRS."""
    if ds.crs is None:
        return None
    corners = outer_corners_native(ds)
    dense = densify_ring(corners, points_per_edge=points_per_edge)
    xs = [p[0] for p in dense]
    ys = [p[1] for p in dense]
    lons, lats = transform_coords(ds.crs, xs, ys)
    return list(zip(lons, lats))


def approx_area_km2(footprint: Optional[List[Coord]]) -> Optional[float]:
    """Geodesic area of a lon/lat polygon on the WGS84 ellipsoid.

    Uses pyproj if it is installed (it will be, from Phase 4 onwards). Falls
    back to a spherical excess calculation so Phase 1 has zero extra deps.
    """
    if not footprint or len(footprint) < 3:
        return None
    lons = [p[0] for p in footprint]
    lats = [p[1] for p in footprint]
    try:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        area_m2, _perimeter = geod.polygon_area_perimeter(lons, lats)
        return abs(area_m2) / 1e6
    except Exception:
        # Spherical fallback: shoelace on a mean-radius sphere. Accurate to
        # ~0.5% at scene scale -- good enough for a metadata panel.
        r = 6371.0088
        lat0 = math.radians(sum(lats) / len(lats))
        pts = [(math.radians(lo) * math.cos(lat0), math.radians(la)) for lo, la in zip(lons, lats)]
        s = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            s += x0 * y1 - x1 * y0
        return abs(s) * 0.5 * r * r


# --------------------------------------------------------------------------- #
# Metadata extraction
# --------------------------------------------------------------------------- #
def _safe_epsg(crs: Optional[CRS]) -> Optional[int]:
    if crs is None:
        return None
    try:
        return crs.to_epsg()
    except Exception:
        return None


def _linear_units(crs: Optional[CRS]) -> Optional[str]:
    if crs is None or not crs.is_projected:
        return None
    try:
        return crs.linear_units
    except Exception:
        return None


def read_band_info(ds: DatasetReader) -> Tuple[BandInfo, ...]:
    bands: List[BandInfo] = []
    for i in range(1, ds.count + 1):
        desc = ds.descriptions[i - 1] if ds.descriptions else None
        try:
            block = tuple(int(v) for v in ds.block_shapes[i - 1])
        except Exception:
            block = (-1, -1)
        nd = ds.nodatavals[i - 1]
        try:
            ci = ds.colorinterp[i - 1]
            ci_name = getattr(ci, "name", None)
        except Exception:
            ci_name = None
        bands.append(
            BandInfo(
                index=i,
                name=(desc.strip() if isinstance(desc, str) and desc.strip() else None),
                dtype=ds.dtypes[i - 1],
                nodata=(None if nd is None else float(nd)),
                block_shape=block,  # type: ignore[arg-type]
                color_interp=(str(ci_name).lower() if ci_name else None),
            )
        )
    return tuple(bands)


def read_spatial_info(ds: DatasetReader) -> SpatialInfo:
    crs = ds.crs
    t = ds.transform
    sx, sy = pixel_size(t)
    north_up = is_north_up(t)

    footprint = footprint_wgs84(ds)
    bounds_wgs: Optional[Tuple[float, float, float, float]] = None
    if footprint:
        lons = [p[0] for p in footprint]
        lats = [p[1] for p in footprint]
        bounds_wgs = (min(lons), min(lats), max(lons), max(lats))

    return SpatialInfo(
        has_crs=crs is not None,
        crs_epsg=_safe_epsg(crs),
        crs_name=(crs.to_string() if crs is not None else None),
        crs_wkt=(crs.to_wkt() if crs is not None else None),
        is_geographic=bool(crs is not None and crs.is_geographic),
        is_projected=bool(crs is not None and crs.is_projected),
        linear_units=_linear_units(crs),
        transform=tuple(float(v) for v in t[:6]),
        pixel_size_x=float(sx),
        pixel_size_y=float(sy),
        is_north_up=north_up,
        is_rotated=not north_up,
        bounds_native=(tuple(float(v) for v in ds.bounds) if crs is not None else None),  # type: ignore[arg-type]
        bounds_wgs84=bounds_wgs,
        footprint_wgs84=footprint,
        approx_area_km2=approx_area_km2(footprint),
    )


def _looks_like_cog(ds: DatasetReader) -> bool:
    """Cheap heuristic: Cloud-Optimised GeoTIFFs are tiled and have overviews.

    Not authoritative -- GDAL only records LAYOUT=COG for files it wrote with
    the COG driver. We use it to warn "windowed reads will be slow" and nothing
    more important than that.
    """
    try:
        layout = (ds.tags() or {}).get("LAYOUT") or (ds.tags(ns="IMAGE_STRUCTURE") or {}).get("LAYOUT")
        if str(layout).upper() == "COG":
            return True
        # `ds.is_tiled` is deprecated in rasterio 1.5; read it from the profile.
        tiled = bool((ds.profile or {}).get("tiled", False))
        return tiled and len(ds.overviews(1)) > 0 and max(ds.block_shapes[0]) >= 256
    except Exception:
        return False


def _is_tiled(ds: DatasetReader) -> bool:
    """`ds.is_tiled` is deprecated in rasterio >= 1.5; read it from the profile."""
    try:
        return bool((ds.profile or {}).get("tiled", False))
    except Exception:
        return False


def _build_warnings(ds: DatasetReader, spatial: SpatialInfo, bands: Sequence[BandInfo]) -> Tuple[str, ...]:
    """Caveats the UI must show. Silence here = the user is misled later."""
    w: List[str] = []

    if not spatial.has_crs:
        w.append(
            "NO CRS: this raster is not georeferenced. It can be displayed as a "
            "plain image but it cannot be placed on a map."
        )

    if spatial.is_rotated:
        w.append(
            "ROTATED TRANSFORM: the grid is not north-up (b or d != 0). Web-map "
            "overlays require reprojection to EPSG:3857/4326 first; a plain "
            "corner-to-corner image overlay would be geometrically wrong."
        )

    if spatial.has_crs and spatial.is_geographic and max(spatial.pixel_size_x, spatial.pixel_size_y) > 0.01:
        w.append(
            "COARSE GEOGRAPHIC GRID: pixel size is large in degrees; treat the "
            "reported pixel size as approximate ground resolution."
        )

    if all(b.nodata is None for b in bands):
        w.append(
            "NO NODATA DECLARED: fill/background pixels cannot be excluded "
            "automatically. Statistics will include them unless you set a "
            "nodata value explicitly."
        )
    elif any(b.nodata is not None for b in bands) and not all(b.nodata is not None for b in bands):
        w.append("NODATA declared on some bands only; check per-band values below.")

    if any(b.dtype in ("uint8", "int8") for b in bands) and len(bands) >= 3:
        w.append(
            "8-BIT DATA: this looks like a display/visual product (stretched for "
            "the human eye). Spectral indices such as NDVI computed from 8-bit "
            "visual products are NOT physically valid. Use analysis-ready "
            "reflectance (e.g. Sentinel-2 L2A, Landsat Collection-2 L2SP)."
        )

    if not _looks_like_cog(ds) and ds.width * ds.height > 20_000_000:
        w.append(
            "LARGE UNTILED RASTER (>20 MPx, no tiling/overviews): previews will "
            "be slow. Consider converting to a COG (gdal_translate -of COG)."
        )

    if abs(abs(spatial.pixel_size_x) - abs(spatial.pixel_size_y)) > 1e-6 * max(
        1.0, abs(spatial.pixel_size_x)
    ):
        w.append("NON-SQUARE PIXELS: x and y pixel sizes differ.")

    return tuple(w)


def read_metadata(
    ds: DatasetReader,
    source_label: str,
    source_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> RasterInfo:
    """Turn an open dataset into a fully described `RasterInfo`.

    This is the Phase 1 deliverable: a complete, honest, serialisable
    description of the file -- with no pixels read beyond a tiny sanity window.
    """
    if file_size_bytes is None and source_path:
        try:
            import os

            file_size_bytes = os.path.getsize(source_path)
        except OSError:
            file_size_bytes = None

    bands = read_band_info(ds)
    spatial = read_spatial_info(ds)
    dtypes = tuple(ds.dtypes)

    itemsize = max(np.dtype(d).itemsize for d in dtypes) if dtypes else 1
    full_read_mb = ds.width * ds.height * ds.count * itemsize / (1024 * 1024)

    tags = {}
    all_tags = ds.tags() or {}
    for k in _INTERESTING_TAGS:
        if k in all_tags:
            v = all_tags[k]
            tags[k] = v if len(v) <= 600 else v[:600] + " ...[truncated]"

    try:
        ovr = tuple(int(v) for v in ds.overviews(1))
    except Exception:
        ovr = ()

    try:
        block_shape = tuple(int(v) for v in ds.block_shapes[0])
    except Exception:
        block_shape = (-1, -1)

    profile = ds.profile or {}

    return RasterInfo(
        source_label=source_label,
        source_path=source_path,
        file_size_bytes=file_size_bytes,
        driver=ds.driver or "UNKNOWN",
        width=int(ds.width),
        height=int(ds.height),
        count=int(ds.count),
        dtypes=dtypes,
        bands=bands,
        spatial=spatial,
        nodata_per_band=tuple(b.nodata for b in bands),
        tiled=_is_tiled(ds),
        overview_levels=ovr,
        block_shape=block_shape,  # type: ignore[arg-type]
        compress=profile.get("compress"),
        interleave=profile.get("interleave"),
        looks_like_cog=_looks_like_cog(ds),
        tags=tags,
        warnings=_build_warnings(ds, spatial, bands),
        estimated_full_read_mb=round(full_read_mb, 2),
    )


def describe_path(path: str, label: Optional[str] = None) -> RasterInfo:
    """One-shot convenience: open, describe, close."""
    with open_dataset(path) as ds:
        return read_metadata(ds, source_label=label or path, source_path=path)


def describe_bytes(file_bytes: bytes, label: str) -> RasterInfo:
    """One-shot convenience for uploads."""
    with open_upload(file_bytes) as ds:
        return read_metadata(ds, source_label=label, file_size_bytes=len(file_bytes))


# --------------------------------------------------------------------------- #
# Pixel sanity check (reads a small window -- NOT a preview)
# --------------------------------------------------------------------------- #
def center_window(ds: DatasetReader, max_pixels: int = 64 * 64) -> Window:
    """A small window at the centre of the raster, capped at `max_pixels`."""
    side = int(math.sqrt(max_pixels))
    w = min(side, ds.width)
    h = min(side, ds.height)
    col_off = (ds.width - w) // 2
    row_off = (ds.height - h) // 2
    return Window(col_off=col_off, row_off=row_off, width=w, height=h)


def window_stats(
    ds: DatasetReader, band: int = 1, window: Optional[Window] = None, sample: int = 256
) -> Dict[str, Any]:
    """Read a small window and return basic statistics.

    Why this exists: "the file opened" is not the same as "the pixels are
    readable and the nodata convention behaves". This is a verification step,
    not a feature -- it costs one windowed read even on a 2 GB scene.

    `sample` decimates the read so we never pull a huge block for a statistic.
    """
    window = window or center_window(ds)
    out_shape = (
        1,
        max(1, min(int(window.height), sample)),
        max(1, min(int(window.width), sample)),
    )
    arr = ds.read(
        band,
        window=window,
        out_shape=out_shape,
        resampling=Resampling.nearest,
        masked=True,  # honours the declared nodata and returns a MaskedArray
    )
    masked = np.ma.masked_invalid(arr)
    valid = masked.compressed()

    return {
        "band": band,
        "window": {
            "col_off": int(window.col_off),
            "row_off": int(window.row_off),
            "width": int(window.width),
            "height": int(window.height),
        },
        "read_shape": list(arr.shape),
        "nodata": ds.nodatavals[band - 1] if band <= ds.count else None,
        "valid_pixels": int(valid.size),
        "masked_pixels": int(masked.count() is None or (masked.size - valid.size)),
        "min": (float(valid.min()) if valid.size else None),
        "max": (float(valid.max()) if valid.size else None),
        "mean": (float(valid.mean()) if valid.size else None),
        "std": (float(valid.std()) if valid.size else None),
        "all_nodata": bool(valid.size == 0),
    }
