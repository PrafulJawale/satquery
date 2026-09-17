"""core/statistics.py -- statistics over VALID pixels only (PHASE 3, extended in Phase 6).

One rule governs everything here: **invalid pixels never enter a statistic and
never become zero.** Nodata, NaN, infinity and undefined ratios are excluded by
mask, and every result reports how many pixels actually contributed.

Phase 6 will add zonal statistics (statistics over a drawn ROI); the helpers
here are the ones it will reuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rasterio import Affine, windows
from rasterio.features import geometry_mask

from .geometry import GeometryError, area_m2, crs_label, metres_per_unit


def describe_valid(
    array: np.ndarray,
    mask: Optional[np.ndarray] = None,
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
) -> Optional[Dict[str, Any]]:
    """Summary statistics over valid pixels.

    Returns **None** when there is not a single valid pixel. A caller must not
    invent numbers in that case -- that is how "mean NDVI = 0.0 over an empty
    scene" bugs are born.
    """
    arr = np.asarray(array)
    if mask is None:
        mask = np.isfinite(arr) if arr.dtype.kind == "f" else np.ones(arr.shape, dtype=bool)
    values = arr[mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        return None

    v = values.astype(np.float64)
    out: Dict[str, Any] = {
        "valid_pixels": int(v.size),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=0)),
        "sum": float(v.sum()),
    }
    pct = {f"p{int(p)}": float(np.percentile(v, p)) for p in percentiles}
    out["percentiles"] = pct
    return out


def count_summary(mask: np.ndarray) -> Dict[str, Any]:
    """total / valid / invalid / valid percentage from a boolean validity mask.

    Takes the MASK, never the data: passing a data array here would silently
    treat every non-finite-looking element as valid (this bug existed once --
    it made an all-invalid scene report 100% valid).
    """
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.size)
    valid = int(np.count_nonzero(mask))
    invalid = total - valid
    return {
        "total_pixels": total,
        "valid_pixels": valid,
        "invalid_pixels": invalid,
        "valid_percentage": (100.0 * valid / total) if total else 0.0,
        "invalid_percentage": (100.0 * invalid / total) if total else 0.0,
    }


def fraction_within(
    array: np.ndarray, mask: np.ndarray, low: float, high: float, include_high: bool = True
) -> float:
    """Fraction of VALID pixels inside [low, high] (or [low, high) when
    `include_high` is False).

    Half-open intervals matter when binning into classes: with closed intervals a
    pixel sitting exactly on a breakpoint is counted twice and the class shares
    no longer sum to 1.
    """
    vals = np.asarray(array)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    upper = vals <= high if include_high else vals < high
    return float(np.mean((vals >= low) & upper))


# =========================================================================== #
# PHASE 6 -- zonal statistics over a user-drawn ROI
# =========================================================================== #
# --------------------------------------------------------------------------- #
# PHASE 6 -- zonal statistics over a user-drawn ROI
#
# * The numbers come from the NATIVE analysis array and its validity mask. The
#   reprojected web-map raster is a display artefact: it is resampled with
#   `average` on a rotated grid, so its mean is biased and its standard
#   deviation is systematically too small.
# * Pixels are selected with `rasterio.features.geometry_mask(..., invert=True)`,
#   which honours rotation and shear because it works through the affine.
# * Pixel-inclusion convention: `all_touched=False` -- a pixel counts when its
#   CENTRE lies inside the polygon. This is unambiguous and unbiased, unlike
#   `all_touched=True`, which inflates area at every boundary.
# --------------------------------------------------------------------------- #

NO_PIXELS_MESSAGE = "No raster pixels fall inside the selected area."
NO_VALID_MESSAGE = "No valid NDVI pixels were found inside the selected area."


@dataclass
class ROINDVIStats:
    """Statistics of NATIVE NDVI inside one user-drawn region.

    `pixels_inside_roi` and `valid_pixels` are deliberately different numbers:
    the first is geometry (pixel centres inside the polygon), the second is
    geometry AND data validity. Conflating them is how "100% valid" bugs happen.
    """

    pixels_inside_roi: int = 0
    valid_pixels: int = 0
    invalid_pixels: int = 0
    valid_fraction: float = 0.0
    stats: Optional[Dict[str, Any]] = None
    area_m2: float = 0.0
    pixel_area_m2: float = 0.0
    pixel_width: float = 0.0
    pixel_height: float = 0.0
    valid_area_m2: float = 0.0
    crs: str = ""
    transform: Tuple[float, ...] = ()
    window: Optional[Tuple[int, int, int, int]] = None
    all_touched: bool = False
    message: str = ""
    warnings: Tuple[str, ...] = ()

    # -- Phase 11 ----------------------------------------------------------- #
    # `index_name` names WHICH index these statistics describe. It defaults to
    # "ndvi" so every Phase 6 caller and every existing test sees exactly what
    # it saw before.
    index_name: str = "ndvi"
    # The ROI-windowed index array and its validity mask, when the engine that
    # produced this result read them itself. Default None: Phase 6 keeps its
    # arrays in the app, and nothing serialises these (to_dict omits them).
    raster: Any = None
    mask: Any = None
    runtime_ms: float = 0.0

    @property
    def has_stats(self) -> bool:
        return self.stats is not None and self.valid_pixels > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixels_inside_roi": self.pixels_inside_roi,
            "valid_pixels": self.valid_pixels,
            "invalid_pixels": self.invalid_pixels,
            "valid_fraction": self.valid_fraction,
            "stats": self.stats,
            "area_m2": self.area_m2,
            "area_hectares": self.area_m2 / 10_000.0,
            "area_km2": self.area_m2 / 1_000_000.0,
            "pixel_area_m2": self.pixel_area_m2,
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "valid_area_m2": self.valid_area_m2,
            "valid_area_hectares": self.valid_area_m2 / 10_000.0,
            "crs": self.crs,
            "window": list(self.window) if self.window else None,
            "all_touched": self.all_touched,
            "index_name": self.index_name,
            "runtime_ms": round(float(self.runtime_ms), 2),
            "message": self.message,
            "warnings": list(self.warnings),
        }


def pixel_geometry(transform: Affine, crs: Any = None) -> Dict[str, float]:
    """Pixel size and area DERIVED FROM THE AFFINE (never assumed to be 10 m).

    For a general (rotated/sheared) affine the pixel edges are the column vector
    (a, d) and the row vector (b, e), so:
        width  = hypot(a, d)
        height = hypot(b, e)
        area   = |a*e - b*d|
    """
    a, b, _c, d, e, _f = transform[:6]
    unit = metres_per_unit(crs) if crs is not None else 1.0
    width = math.hypot(a, d) * unit
    height = math.hypot(b, e) * unit
    area = abs(a * e - b * d) * unit * unit
    return {"pixel_width": float(width), "pixel_height": float(height),
            "pixel_area_m2": float(area)}


def _roi_window(roi_bounds: Sequence[float], transform: Affine,
                height: int, width: int) -> Optional[windows.Window]:
    """Smallest integer pixel window covering the ROI, clipped to the raster.

    Grows outward (floor/ceil) so it can never crop the ROI by rounding inwards.
    """
    if None in roi_bounds:
        return None
    minx, miny, maxx, maxy = roi_bounds
    win = windows.from_bounds(minx, miny, maxx, maxy, transform=transform)
    row_off = max(0, int(math.floor(win.row_off)))
    col_off = max(0, int(math.floor(win.col_off)))
    row_end = min(height, int(math.ceil(win.row_off + win.height)))
    col_end = min(width, int(math.ceil(win.col_off + win.width)))
    if row_end <= row_off or col_end <= col_off:
        return None
    return windows.Window(col_off, row_off, col_end - col_off, row_end - row_off)


def roi_pixel_mask(
    roi_geometry: Any,
    transform: Affine,
    height: int,
    width: int,
    all_touched: bool = False,
) -> Tuple[Any, Optional[windows.Window]]:
    """Boolean mask (full raster shape) of pixel centres inside the ROI.

    The mask is built only over the ROI's window and then pasted into a
    full-shape array, so cost is O(ROI) for rasterisation while the caller keeps
    a full-shape mask to work with.
    """
    full = np.zeros((height, width), dtype=bool)
    window = _roi_window(roi_geometry.bounds, transform, height, width)
    if window is None:
        return full, None
    sub_transform = windows.transform(window, transform)
    sub = geometry_mask(
        [roi_geometry],
        out_shape=(int(window.height), int(window.width)),
        transform=sub_transform,
        all_touched=all_touched,
        invert=True,
    )
    r0 = int(window.row_off)
    c0 = int(window.col_off)
    full[r0:r0 + sub.shape[0], c0:c0 + sub.shape[1]] = sub
    return full, window


def calculate_roi_ndvi_stats(
    ndvi: Any,
    roi_geometry: Any,
    transform: Affine,
    valid_mask: Optional[Any] = None,
    *,
    crs: Any = None,
    roi_crs: Any = None,
    all_touched: bool = False,
    percentiles: Sequence[float] = (5, 25, 50, 75, 95),
) -> ROINDVIStats:
    """NDVI statistics over a user-selected region, from the NATIVE raster.

    Parameters
    ----------
    ndvi          native NDVI array (H, W), NaN where invalid.
    roi_geometry  shapely Polygon/MultiPolygon **already in the raster CRS**
                  (`ROISelection.geometry_raster_crs` from Phase 5).
    transform     the native affine.
    valid_mask    `AnalysisResult.mask`; if None, finite values are used.
    crs           raster CRS (for area and pixel units).
    roi_crs       the CRS the ROI claims to be in -- compared with `crs` so a
                  mismatch raises instead of silently mis-masking.
    all_touched   pixel-inclusion convention; False = pixel CENTRE inside.
    """
    if ndvi is None:
        raise ValueError("ndvi array is required")
    ndvi = np.asarray(ndvi)
    if ndvi.ndim != 2:
        raise ValueError(f"ndvi must be 2-D, got shape {ndvi.shape}")
    if roi_geometry is None or roi_geometry.is_empty:
        return ROINDVIStats(crs=crs_label(crs), transform=tuple(transform)[:6],
                            all_touched=all_touched, message=NO_PIXELS_MESSAGE)

    # --- CRS mismatch protection ----------------------------------------- #
    if crs is not None and roi_crs is not None:
        from pyproj import CRS as _CRS

        try:
            same = _CRS.from_user_input(crs).equals(_CRS.from_user_input(roi_crs))
        except Exception as exc:  # pragma: no cover - defensive
            raise GeometryError(f"Could not compare CRS values: {exc}")
        if not same:
            raise GeometryError(
                f"ROI is in {crs_label(roi_crs)} but the raster is in "
                f"{crs_label(crs)}. Transform the geometry into the raster CRS "
                "before computing statistics (see core.roi.select_roi)."
            )

    height, width = ndvi.shape
    geom = pixel_geometry(transform, crs)
    warnings_out: List[str] = []

    # --- ROI area (reuses Phase 5 core.geometry.area_m2) ------------------- #
    # Computed before the pixel test so that an ROI covering no pixel still
    # reports its own size instead of a misleading 0 m2.
    roi_area_m2 = 0.0
    area_method = ""
    try:
        roi_area_m2, area_method = area_m2(roi_geometry, crs)
    except GeometryError:
        roi_area_m2 = 0.0
        area_method = ""
    if area_method:
        warnings_out.append(f"Area method: {area_method}.")

    # --- geometry -> pixels (windowed: O(ROI), not O(scene)) -------------- #
    inside, window = roi_pixel_mask(roi_geometry, transform, height, width,
                                    all_touched=all_touched)
    pixels_inside = int(np.count_nonzero(inside))

    # warn if the ROI hangs off the raster (Phase 5 normally clips it away)
    if window is not None:
        rx0, ry0 = transform * (0, 0)
        rx1, ry1 = transform * (width, height)
        rmin_x, rmax_x = min(rx0, rx1), max(rx0, rx1)
        rmin_y, rmax_y = min(ry0, ry1), max(ry0, ry1)
        gx0, gy0, gx1, gy1 = roi_geometry.bounds
        if (gx0 < rmin_x - 1e-9 or gy0 < rmin_y - 1e-9
                or gx1 > rmax_x + 1e-9 or gy1 > rmax_y + 1e-9):
            warnings_out.append(
                "The ROI extends beyond the raster extent; only the part over "
                "the raster can be analysed."
            )

    if pixels_inside == 0:
        return ROINDVIStats(
            crs=crs_label(crs), transform=tuple(transform)[:6],
            area_m2=float(roi_area_m2),
            pixel_area_m2=geom["pixel_area_m2"], pixel_width=geom["pixel_width"],
            pixel_height=geom["pixel_height"],
            window=(int(window.row_off), int(window.col_off),
                    int(window.height), int(window.width)) if window else None,
            all_touched=all_touched, message=NO_PIXELS_MESSAGE,
            warnings=tuple(warnings_out),
        )

    # --- validity ---------------------------------------------------------- #
    if valid_mask is None:
        valid_mask = np.isfinite(ndvi)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != ndvi.shape:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match ndvi {ndvi.shape}"
        )

    valid = inside & valid_mask
    n_valid = int(np.count_nonzero(valid))
    n_invalid = pixels_inside - n_valid

    stats = describe_valid(ndvi, valid, percentiles=percentiles) if n_valid else None

    # --- area -------------------------------------------------------------- #
    # `roi_area_m2` / `area_method` come from the block above. If Phase 5 could
    # not measure the geometry, fall back to counting pixels: still honest, and
    # still derived from the affine rather than assumed.
    if not area_method:
        roi_area_m2 = float(pixels_inside) * geom["pixel_area_m2"]
        area_method = "derived from the pixel count and the affine"
        warnings_out.append(f"Area method: {area_method}.")
    valid_area = n_valid * geom["pixel_area_m2"]

    if stats is None:
        message = NO_VALID_MESSAGE
    else:
        message = (
            f"Mean NDVI {stats['mean']:.4f} over {n_valid:,} valid pixels "
            f"({100.0 * n_valid / pixels_inside:.2f}% of the pixels in the ROI)."
        )
        if n_invalid:
            warnings_out.append(
                f"{n_invalid:,} pixel(s) inside the ROI have no valid NDVI value "
                "and are excluded from every statistic."
            )
        if n_invalid == pixels_inside and pixels_inside:
            pass  # every pixel is invalid -> NO_VALID_MESSAGE already covers it

    return ROINDVIStats(
        pixels_inside_roi=pixels_inside,
        valid_pixels=n_valid,
        invalid_pixels=n_invalid,
        valid_fraction=(n_valid / pixels_inside) if pixels_inside else 0.0,
        stats=stats,
        area_m2=float(roi_area_m2),
        pixel_area_m2=geom["pixel_area_m2"],
        pixel_width=geom["pixel_width"],
        pixel_height=geom["pixel_height"],
        valid_area_m2=float(valid_area),
        crs=crs_label(crs),
        transform=tuple(transform)[:6],
        window=(int(window.row_off), int(window.col_off),
                int(window.height), int(window.width)) if window else None,
        all_touched=all_touched,
        message=message,
        warnings=tuple(warnings_out),
    )
