"""Phase 8 -- ESA WorldCover 2021 v200 (10 m land cover).

Used as a HARD CONSTRAINT, not as a suitability score: built-up, permanent
water, mangroves, snow/ice, moss and herbaceous wetland are exclusions.
A non-cropland pixel is NOT automatically unsuitable (brief item #9).

Resampling: NEAREST NEIGHBOUR. Bilinear interpolation of class codes would
invent classes that do not exist.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from ..alignment import AnalysisGrid
from .base import LayerResult, load_layer

__all__ = ["DATASET", "VERSION", "CLASS_NAMES", "tile_ids_for_grid", "fetch_land_cover",
           "land_cover_summary", "EXCLUDED_BY_DEFAULT"]

DATASET = "worldcover"
VERSION = "v200_2021"
BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

CLASS_NAMES: Dict[int, str] = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}

EXCLUDED_BY_DEFAULT: Sequence[int] = (50, 80, 95, 70, 100, 90)

LIMITATIONS = ("Thematic accuracy ~75% globally; class 40 'cropland' means ANY crop "
               "in 2021 (not cotton, not currently cultivated); v100 (2020) and "
               "v200 (2021) use different algorithms and must not be compared.")


def tile_ids_for_grid(grid: AnalysisGrid) -> List[str]:
    """3-degree WorldCover tile ids (e.g. N30E030) covering the grid bounds."""
    left, bottom, right, top = transform_bounds(
        grid.crs, "EPSG:4326", *grid.bounds, densify_pts=21)
    ids: List[str] = []
    lat = int(np.floor(bottom / 3.0) * 3)
    while lat <= top:
        lon = int(np.floor(left / 3.0) * 3)
        while lon <= right:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            ids.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
            lon += 3
        lat += 3
    return ids or [f"{'N' if bottom >= 0 else 'S'}{abs(int(np.floor(bottom / 3.0) * 3)):02d}"
                   f"{'E' if left >= 0 else 'W'}{abs(int(np.floor(left / 3.0) * 3)):03d}"]


def urls_for_grid(grid: AnalysisGrid) -> List[str]:
    return [f"{BASE_URL}/ESA_WorldCover_10m_2021_v200_{tid}_Map.tif"
            for tid in tile_ids_for_grid(grid)]


def fetch_land_cover(grid: AnalysisGrid, use_cache: bool = True) -> LayerResult:
    """Categorical land cover on the analysis grid (nearest neighbour)."""
    urls = urls_for_grid(grid)
    return load_layer(
        dataset=DATASET, version=VERSION, variable="land_cover", urls=urls,
        grid=grid, resampling=Resampling.nearest,
        native_resolution="10 m", native_crs="EPSG:4326", units="class code",
        temporal_period="2021 epoch",
        license="CC BY 4.0 (ESA WorldCover project; doi 10.5281/zenodo.7254221)",
        limitations=LIMITATIONS,
        processing="windowed read; nearest-neighbour reprojection (categorical)",
        source_url="; ".join(urls), use_cache=use_cache)


def land_cover_summary(classes: np.ndarray,
                       inside: np.ndarray,
                       policy: Dict[str, Any],
                       ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """(exclusion mask, summary) for the configured land-cover policy.

    `policy` comes from config/crops/<crop>.yml `land_cover`:
        exclude:            {code: reason}
        allow_with_context: {code: note}
    """
    exclude = {int(k): str(v) for k, v in (policy.get("exclude") or {}).items()}
    allow = {int(k): str(v) for k, v in (policy.get("allow_with_context") or {}).items()}

    valid = np.isfinite(classes) & inside
    mask = np.zeros(classes.shape, dtype=bool)
    for code in exclude:
        mask |= valid & (classes == code)

    n_valid = int(valid.sum())
    excluded = int(mask.sum())
    summary: Dict[str, Any] = {
        "excluded_fraction": (excluded / n_valid) if n_valid else 0.0,
        "excluded_cells": excluded,
        "valid_cells": n_valid,
        "dominant_class": None,
        "dominant_class_name": None,
        "reason": "",
        "allow_with_context": allow,
        "cropland_caveat": policy.get("cropland_caveat", ""),
    }
    if n_valid:
        vals, counts = np.unique(classes[valid], return_counts=True)
        dom = int(vals[int(np.argmax(counts))])
        summary["dominant_class"] = dom
        summary["dominant_class_name"] = CLASS_NAMES.get(dom, f"class {dom}")
        if dom in exclude:
            summary["reason"] = (f"the area is dominated by {CLASS_NAMES.get(dom, dom)} "
                                 f"({exclude[dom]}).")
    return mask, summary
