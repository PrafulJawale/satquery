"""Phase 8 -- spatial alignment: one common analysis grid, honest resampling.

THREE DIFFERENT RESOLUTIONS, KEPT HONEST
    native resolution      what the source actually measures (10 m / 30 m /
                           250 m / ~1 km)
    analysis resolution    the grid everything is resampled onto (default 30 m)
    effective resolution   what the result can actually resolve -- never finer
                           than the coarsest input that drove it

Resampling a 250 m soil map onto a 30 m grid does not create 30 m soil
information. The grid exists to align layers and to draw a map; every result
reports native resolutions alongside it (brief item #2).

MEMORY SAFETY (brief item #15)
    Nothing here ever reads a whole global raster. `read_into_grid()` opens the
    remote source and lets GDAL's WarpedVRT pull only the window that projects
    onto the analysis grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import bounds as window_bounds, transform as window_transform

__all__ = [
    "AnalysisGrid",
    "make_grid",
    "read_into_grid",
    "roi_mask",
    "compute_slope",
    "grid_signature",
    "DEFAULT_MAX_CELLS",
    "native_roi_grid",
]

DEFAULT_MAX_CELLS = 2_000_000        # brief item #15 -- keep the ~2 GB sandbox alive


@dataclass
class AnalysisGrid:
    """The common grid every factor layer is resampled onto."""

    crs: Any
    transform: Any
    width: int
    height: int
    resolution: float
    requested_resolution: float
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    note: str = ""
    buffer_cells: int = 0

    @property
    def cells(self) -> int:
        return int(self.width) * int(self.height)

    @property
    def coarsened(self) -> bool:
        return float(self.resolution) != float(self.requested_resolution)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crs": str(self.crs),
            "resolution_m": self.resolution,
            "requested_resolution_m": self.requested_resolution,
            "width": self.width, "height": self.height,
            "cells": self.cells,
            "bounds": list(self.bounds),
            "transform": list(self.transform)[:6],
            "coarsened": self.coarsened,
            "note": self.note,
        }


def make_grid(geometry: Any,
              crs: Any,
              requested_resolution: float = 30.0,
              max_cells: int = DEFAULT_MAX_CELLS,
              buffer_cells: int = 2) -> AnalysisGrid:
    """Snap a ROI bounding box outward onto a square grid in a METRIC CRS.

    If the ROI would exceed `max_cells`, the cell size is doubled (repeatedly)
    and the change is recorded -- never applied silently (brief item #14).
    """
    minx, miny, maxx, maxy = geometry.bounds
    res = float(requested_resolution)
    note = ""
    for _ in range(12):
        left = math.floor((minx - buffer_cells * res) / res) * res
        bottom = math.floor((miny - buffer_cells * res) / res) * res
        right = math.ceil((maxx + buffer_cells * res) / res) * res
        top = math.ceil((maxy + buffer_cells * res) / res) * res
        w = int(round((right - left) / res))
        h = int(round((top - bottom) / res))
        if w * h <= max_cells:
            break
        res *= 2.0
    else:  # pragma: no cover -- a 4096x ROI would be needed
        raise ValueError("ROI is too large to analyse even after coarsening.")

    if res != float(requested_resolution):
        note = (f"Requested analysis resolution {requested_resolution:g} m would "
                f"exceed the {max_cells:,}-cell limit; effective resolution is "
                f"{res:g} m.")

    transform = from_origin(left, top, res, res)
    return AnalysisGrid(crs=crs, transform=transform, width=w, height=h,
                        resolution=res, requested_resolution=float(requested_resolution),
                        bounds=(left, bottom, right, top), note=note,
                        buffer_cells=buffer_cells)


def native_roi_grid(transform: Any,
                    width: int,
                    height: int,
                    resolution: float,
                    crs: Any,
                    geometry: Any,
                    buffer_cells: int = 2) -> AnalysisGrid:
    """An AnalysisGrid snapped to a raster's NATIVE grid, covering only the ROI.

    ROI-FIRST: snapping outward (floor/ceil) means the ROI can never be clipped
    by rounding, and clamping to the raster means we never read out of bounds.
    The result stays on the source CRS and cell size, so anything read onto it
    is native data -- not a resampled copy.

    Shared by the Phase 10 temporal engine and Phase 11 index analyses, so the
    two cannot drift apart on how an ROI window is cut.
    """
    minx, miny, maxx, maxy = geometry.bounds
    res = float(resolution)
    win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=transform)
    col_off = max(0, int(math.floor(win.col_off - buffer_cells)))
    row_off = max(0, int(math.floor(win.row_off - buffer_cells)))
    col_end = min(int(width), int(math.ceil(win.col_off + win.width + buffer_cells)))
    row_end = min(int(height), int(math.ceil(win.row_off + win.height + buffer_cells)))
    w = max(1, col_end - col_off)
    h = max(1, row_end - row_off)
    window = rasterio.windows.Window(col_off, row_off, w, h)
    sub_transform = window_transform(window, transform)
    return AnalysisGrid(
        crs=crs,
        transform=sub_transform,
        width=w,
        height=h,
        resolution=res,
        requested_resolution=res,
        bounds=tuple(window_bounds(window, transform)),
        note="native grid, windowed to the ROI",
        buffer_cells=buffer_cells,
    )


def read_into_grid(url: str,
                   grid: AnalysisGrid,
                   resampling: Resampling,
                   band: int = 1,
                   src_nodata: Optional[float] = None,
                   dtype: str = "float32") -> np.ndarray:
    """Read only the part of a remote/local raster that covers the grid.

    Continuous layers -> Resampling.bilinear
    Categorical layers -> Resampling.nearest   (never bilinear on class codes)

    Returns a float array with NaN where the source has no data.
    """
    with rasterio.open(url) as src:
        nodata = src_nodata if src_nodata is not None else src.nodatavals[band - 1]
        with WarpedVRT(src,
                       crs=grid.crs,
                       transform=grid.transform,
                       width=grid.width,
                       height=grid.height,
                       resampling=resampling,
                       src_nodata=nodata,
                       nodata=np.nan,
                       dtype=dtype) as vrt:
            arr = vrt.read(band)
    arr = arr.astype("float32", copy=False)
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    return arr


def roi_mask(geometry: Any, grid: AnalysisGrid) -> np.ndarray:
    """True where a cell CENTRE lies inside the ROI (all_touched=False)."""
    return geometry_mask([geometry], out_shape=(grid.height, grid.width),
                         transform=grid.transform, invert=True, all_touched=False)


def compute_slope(dem: np.ndarray, transform: Any, crs: Any = None) -> np.ndarray:
    """Percent slope from a DEM in a METRIC CRS (Horn 3x3 method).

    Never call this on degree coordinates: the horizontal spacing would be in
    degrees and the gradient meaningless (brief item #13).

        a b c        dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize)
        d e f        dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize)
        g h i        slope% = hypot(dz/dx, dz/dy) * 100

    * pixel spacing is taken from the affine (assumes north-up, square cells,
      which `make_grid` guarantees);
    * edges are handled by replicating the border row/column, so the output has
      the same shape as the input. Consequence, stated rather than hidden:
      the outermost row/column sees one flat side and therefore returns
      about HALF the true gradient; the interior is exact.
    * nodata (NaN) in the DEM propagates to NaN in the slope.
    """
    cell = abs(float(transform.a))
    if cell == 0 or abs(abs(float(transform.a)) - abs(float(transform.e))) > 1e-9:
        raise ValueError("Slope needs a north-up grid with square cells.")
    if crs is not None:
        try:
            from rasterio.crs import CRS as _CRS
            if _CRS.from_user_input(crs).is_geographic:
                raise ValueError(
                    "Slope cannot be computed from a geographic (degree) grid: "
                    "reproject the DEM into a metric CRS first.")
        except ValueError:
            raise
        except Exception:  # pragma: no cover -- unparseable CRS is not our problem
            pass

    z = np.asarray(dem, dtype="float64")
    valid = np.isfinite(z)
    padded = np.pad(np.where(valid, z, 0.0), 1, mode="edge")
    vpad = np.pad(valid.astype("float64"), 1, mode="edge")

    a = padded[:-2, :-2]; b = padded[:-2, 1:-1]; c = padded[:-2, 2:]
    d = padded[1:-1, :-2];                    f = padded[1:-1, 2:]
    g = padded[2:, :-2];  h = padded[2:, 1:-1]; i = padded[2:, 2:]

    dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell)
    dzdy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cell)
    slope = np.hypot(dzdx, dzdy) * 100.0

    # a cell whose 3x3 neighbourhood contains nodata cannot be graded honestly
    hood = (vpad[:-2, :-2] + vpad[:-2, 1:-1] + vpad[:-2, 2:]
            + vpad[1:-1, :-2] + vpad[1:-1, 1:-1] + vpad[1:-1, 2:]
            + vpad[2:, :-2] + vpad[2:, 1:-1] + vpad[2:, 2:])
    slope[hood < 9.0] = np.nan
    slope[~valid] = np.nan
    return slope.astype("float32")


def grid_signature(grid: AnalysisGrid, extra: str = "") -> str:
    """Stable identity for caching: grid + anything else that changes the bytes."""
    import hashlib

    parts = [
        str(grid.crs),
        f"{grid.transform.a:.6f}", f"{grid.transform.e:.6f}",
        f"{grid.transform.c:.3f}", f"{grid.transform.f:.3f}",
        str(grid.width), str(grid.height), str(grid.resolution), extra,
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
