"""Phase 8 -- Copernicus DEM GLO-30 (30 m) for SLOPE and terrain context.

Elevation is deliberately NOT scored: no crop-specific elevation criterion was
established for cotton, and inventing one would be decoration (brief item #12).
The DEM is used for:
    * slope, computed in a METRIC CRS after reprojection (brief item #13)
    * terrain context (elevation range reported, never scored)

Working access path (discovered during Phase 8 research -- the COG_30 naming
returns 404 on this bucket):
    https://copernicus-dem-30m.s3.amazonaws.com/
        Copernicus_DSM_COG_10_N31_00_E031_00_DEM/
        Copernicus_DSM_COG_10_N31_00_E031_00_DEM.tif
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from ..alignment import AnalysisGrid, compute_slope
from .base import LayerResult, load_layer

__all__ = ["DATASET", "VERSION", "tile_ids_for_grid", "urls_for_grid",
           "fetch_dem", "fetch_slope", "LIMITATIONS"]

DATASET = "copernicus_dem"
VERSION = "GLO-30"
BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

LIMITATIONS = ("Digital SURFACE model (buildings and canopy included), not a "
               "bare-earth DTM; no nodata value is declared in the file; slope "
               "over very flat terrain is within DEM noise; acquisitions ~2011-2015.")


def tile_ids_for_grid(grid: AnalysisGrid) -> List[str]:
    """1-degree Copernicus tile ids (e.g. N31_00_E031_00) covering the grid."""
    left, bottom, right, top = transform_bounds(
        grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)
    ids: List[str] = []
    lat = int(np.floor(bottom))
    while lat <= top:
        lon = int(np.floor(left))
        while lon <= right:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            ids.append(f"{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00")
            lon += 1
        lat += 1
    return ids or ["N00_00_E000_00"]


def urls_for_grid(grid: AnalysisGrid) -> List[str]:
    out = []
    for tid in tile_ids_for_grid(grid):
        out.append(f"{BASE_URL}/Copernicus_DSM_COG_10_{tid}_DEM/"
                   f"Copernicus_DSM_COG_10_{tid}_DEM.tif")
    return out


def fetch_dem(grid: AnalysisGrid, use_cache: bool = True) -> LayerResult:
    urls = urls_for_grid(grid)
    return load_layer(
        dataset=DATASET, version=VERSION, variable="elevation", urls=urls,
        grid=grid, resampling=Resampling.bilinear,
        native_resolution="30 m (1/3600 deg)", native_crs="EPSG:4326",
        units="metres", temporal_period="static (TanDEM-X, ~2011-2015)",
        license="Copernicus Open Data Licence (free, attribution required)",
        limitations=LIMITATIONS,
        processing=(f"windowed read; bilinear reprojection to {grid.crs} at "
                    f"{grid.resolution:g} m (slope is computed AFTER reprojection)"),
        source_url="; ".join(urls), use_cache=use_cache)


def fetch_slope(grid: AnalysisGrid,
                use_cache: bool = True,
                inside: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """DEM -> slope in percent, in the metric analysis CRS."""
    dem = fetch_dem(grid, use_cache=use_cache)
    slope = compute_slope(dem.array, grid.transform, crs=grid.crs)
    elevation = dem.array if inside is None else dem.array[inside]
    elevation = elevation[np.isfinite(elevation)]
    slope_roi = slope if inside is None else slope[inside]
    slope_roi = slope_roi[np.isfinite(slope_roi)]
    return {
        "slope_pct": slope,
        "record": dem.record,
        "from_cache": dem.from_cache,
        "seconds": dem.seconds,
        "elevation_context": {
            "min_m": float(elevation.min()) if elevation.size else None,
            "max_m": float(elevation.max()) if elevation.size else None,
            "mean_m": float(elevation.mean()) if elevation.size else None,
        },
        "slope_context": {
            "mean_pct": float(slope_roi.mean()) if slope_roi.size else None,
            "max_pct": float(slope_roi.max()) if slope_roi.size else None,
        },
    }
