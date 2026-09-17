"""Phase 9, Checkpoint D -- real-data verification (NO UI, NO browser tests).

Runs the completed spatial-query engine against REAL satellite data over the
two approved AOIs and independently checks the result:

  AOI-1  the Phase 8 3x3 km ROI           (zero-water regression anchor)
  AOI-2  the full sample extent 687x687   (contains mapped permanent water)

Independent verification is done WITHOUT reusing the engine's intermediate
outputs where it matters:

  * WorldCover provenance, CRS, transform, dimensions, class counts
  * water proximity: brute-force nearest-water distance from cell coordinates,
    compared against the engine's mask (zero distance, boundary, units)
  * cotton: recomputed in plain NumPy from the Phase 8 class raster
  * combinations: recomputed in plain NumPy from the raw arrays
  * JRC Global Surface Water: independent cross-check (verification only)

Usage:  python scripts/verify_phase9.py [--skip-gsw] [--aoi AOI-1|AOI-2]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses import AnalysisContext, Status                       # noqa: E402
from analyses.crop_suitability import run_crop_suitability          # noqa: E402
from analyses.spatial_query import (                                # noqa: E402
    RESULT_INSUFFICIENT_DATA, RESULT_OK, RESULT_ZERO_MATCHES,
    SpatialQueryResult, run_spatial_query)
from core.alignment import make_grid                                # noqa: E402
from core.datasources import worldcover                             # noqa: E402
from core.geometry import as_crs                                    # noqa: E402
from core.roi import ROISelection                                   # noqa: E402
from core.router import parse_query                                 # noqa: E402
from core.spatial import (FALSE, INSUFFICIENT, TRUE, Grid,          # noqa: E402
                          GridMask, proximity_mask,
                          water_mask_from_land_cover)
from shapely.geometry import box                                    # noqa: E402

# --------------------------------------------------------------------------- #
# the AOIs (identical definitions to Phase 8 / Checkpoint A)
# --------------------------------------------------------------------------- #
#: AOI-1 -- the Phase 8 3x3 km ROI: 300 cells at 10 m, centred in the sample.
TEN_M_ORIGIN = (377200.0, 3462300.0)
AOI1_BOX = box(385940.0, 3450560.0, 388940.0, 3453560.0)      # 3000 x 3000 m

#: AOI-2 -- the full bundled sample extent: 20480 m -> 683 cells + 2x2 buffer.
AOI2_BOX = box(377200.0, 3441820.0, 397680.0, 3462300.0)      # 20480 x 20480 m

AOIS = {"AOI-1": AOI1_BOX, "AOI-2": AOI2_BOX}

#: The nine required verification cases.
CASES: List[Tuple[str, str]] = [
    ("1. cotton suitability alone", "Find areas suitable for cotton"),
    ("2. permanent water alone", "Find water"),
    ("3. near permanent water (default distance)", "Find areas near water"),
    ("4. cotton AND near permanent water", "Find cotton areas near water"),
    ("5. cropland AND near permanent water", "Find cropland near water"),
    ("6. cropland AND NOT permanent water", "Find cropland excluding water"),
    ("7. zero-match by construction", "Find cropland and water"),
    ("8. insufficient data (cotton)", "Find areas suitable for cotton"),
    ("9a. unsupported: irrigation", "Find cotton areas with irrigation"),
    ("9b. unsupported: flood", "Find cotton areas outside flood-prone zones"),
]

CLASS_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland",
    95: "Mangroves", 100: "Moss and lichen",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def roi_for(geometry) -> ROISelection:
    return ROISelection(is_valid=True, intersects_raster=True,
                        area_m2=float(geometry.area), raster_crs="EPSG:32636",
                        geometry_raster_crs=geometry, geometry_type="Polygon",
                        num_parts=1)


def rule(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def fmt(value: Any) -> str:
    return f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)


# --------------------------------------------------------------------------- #
# 1. dataset provenance
# --------------------------------------------------------------------------- #
def verify_worldcover(aoi_name: str, geometry) -> Dict[str, Any]:
    grid = make_grid(geometry, as_crs("EPSG:32636"),
                     requested_resolution=30.0, buffer_cells=2)
    t0 = time.perf_counter()
    layer = worldcover.fetch_land_cover(grid, use_cache=True)
    seconds = time.perf_counter() - t0
    array = np.asarray(layer.array)

    values, counts = np.unique(array, return_counts=True)
    classes = {int(v): int(c) for v, c in zip(values, counts)}
    inside = _roi_inside(geometry, grid)

    print(f"\n--- WorldCover: {aoi_name} ---")
    print(f"  source      : {layer.record.to_dict().get('source_url')}")
    print(f"  dataset     : {layer.record.to_dict().get('dataset')} "
          f"{layer.record.to_dict().get('version')}")
    print(f"  native      : {layer.record.to_dict().get('native_resolution')} m, "
          f"{layer.record.to_dict().get('native_crs')}, "
          f"{layer.record.to_dict().get('temporal_period')}")
    print(f"  resampling  : {layer.record.to_dict().get('resampling')} "
          f"(categorical)")
    print(f"  from cache  : {layer.from_cache}   read: {seconds:.2f}s")
    print(f"  grid        : {grid.height} x {grid.width} cells @ {grid.resolution} m"
          f"  ({grid.cells:,} cells)")
    print(f"  transform   : {tuple(round(float(v), 3) for v in tuple(grid.transform)[:6])}")
    print(f"  bounds      : {[round(float(b), 1) for b in grid.bounds]}")
    print("  class counts (full window):")
    for value, count in sorted(classes.items()):
        name = CLASS_NAMES.get(value, "nodata" if value == 0 else "?")
        print(f"      {value:>3} {name:<26} {count:>9,}  {count / array.size * 100:6.2f}%")
    print(f"  water (80)     : {classes.get(80, 0):,} cells")
    print(f"  cropland (40)  : {classes.get(40, 0):,} cells")
    print(f"  wetland (90)   : {classes.get(90, 0):,} cells  (NOT water)")
    print(f"  nodata (0/NaN) : {classes.get(0, 0):,} cells")
    print(f"  ROI cells      : {int(inside.sum()):,}")

    return {"grid": grid, "array": array, "record": layer.record.to_dict(),
            "classes": classes, "inside": inside, "seconds": seconds,
            "from_cache": layer.from_cache}


def _roi_inside(geometry, grid) -> np.ndarray:
    from core.alignment import roi_mask as _roi_mask
    return _roi_mask(geometry, grid)


# --------------------------------------------------------------------------- #
# 2. the nine cases
# --------------------------------------------------------------------------- #
def run_cases(aoi_name: str, geometry, wc: Dict[str, Any],
              cotton_runner) -> List[Dict[str, Any]]:
    context = AnalysisContext(roi=roi_for(geometry))
    rows: List[Dict[str, Any]] = []

    print(f"\n{'case':<44} {'status':<22} {'match':>9} {'insuff':>9} "
          f"{'match %':>8}")
    print("-" * 96)
    for label, query_text in CASES:
        parsed = parse_query(query_text)
        execution = run_spatial_query(context, parsed,
                                      suitability_runner=cotton_runner)
        result: Optional[SpatialQueryResult] = execution.result
        if result is None:
            print(f"{label:<44} {execution.status.value:<22} "
                  f"{'-':>9} {'-':>9} {'-':>8}")
            rows.append({"case": label, "query": query_text,
                         "status": execution.status.value,
                         "message": execution.message, "result": None})
            continue
        print(f"{label:<44} {execution.status.value:<22} "
              f"{result.matched_cell_count:>9,} "
              f"{result.insufficient_cell_count:>9,} "
              f"{result.matched_fraction * 100:>7.2f}%")
        rows.append({"case": label, "query": query_text,
                     "status": execution.status.value,
                     "result_status": result.status,
                     "expression": result.expression,
                     "matched_cell_count": result.matched_cell_count,
                     "non_matching_cell_count": result.non_matching_cell_count,
                     "insufficient_cell_count": result.insufficient_cell_count,
                     "analysed_cell_count": result.analysed_cell_count,
                     "matched_area_m2": result.matched_area_m2,
                     "matched_fraction": result.matched_fraction,
                     "insufficient_fraction": result.insufficient_fraction,
                     "message": result.message,
                     "warnings": list(result.warnings),
                     "performance": result.performance,
                     "grid": result.grid,
                     "condition_results": result.condition_results,
                     "condition_masks": {k: v.copy()
                                         for k, v in result.condition_masks.items()},
                     "result_mask": np.asarray(result.result_mask).copy()})
    return rows


# --------------------------------------------------------------------------- #
# 3. independent checks
# --------------------------------------------------------------------------- #
def brute_force_distance(water: np.ndarray, pixel: float,
                         rows: slice, cols: slice,
                         distance_m: float) -> np.ndarray:
    """Nearest-water distance by explicit coordinate arithmetic.

    Deliberately independent of `scipy.ndimage.distance_transform_edt`. The
    water cells considered are those in a box around the sample window expanded
    by (distance + window) cells, so every water cell that could be within
    `distance_m` of a sampled cell is included -- otherwise cells near the box
    edge would be given an over-estimated distance and the comparison would be
    unfair to the engine.
    """
    r0, r1 = rows.start, rows.stop
    c0, c1 = cols.start, cols.stop
    h, w = r1 - r0, c1 - c0
    margin = int(math.ceil(distance_m / pixel)) + max(h, w) + 2
    br0, br1 = max(0, r0 - margin), min(water.shape[0], r1 + margin)
    bc0, bc1 = max(0, c0 - margin), min(water.shape[1], c1 + margin)
    wy, wx = np.nonzero(water[br0:br1, bc0:bc1])
    if wy.size == 0:
        return np.full((h, w), np.inf)
    wy = wy + br0
    wx = wx + bc0
    ii, jj = np.mgrid[r0:r1, c0:c1]
    best = np.full((h, w), np.inf)
    # chunk the water cells to keep the temporary arrays bounded
    for start in range(0, wy.size, 512):
        cy, cx = wy[start:start + 512], wx[start:start + 512]
        dy = (ii[:, :, None] - cy[None, None, :]) * pixel
        dx = (jj[:, :, None] - cx[None, None, :]) * pixel
        best = np.minimum(best, np.sqrt(dx * dx + dy * dy).min(axis=2))
    return best


def _grid_from_dict(grid_dict: Dict[str, Any]):
    """Rebuild an AnalysisGrid from `SpatialQueryResult.grid`."""
    from rasterio.transform import Affine

    from core.alignment import AnalysisGrid
    transform = Affine(*[float(v) for v in grid_dict["transform"]])
    # `Grid.to_dict()` carries the transform, not the bounds, so derive them:
    # without real bounds the windowed read would ask for the wrong window.
    minx, maxy = transform * (0.0, 0.0)
    maxx, miny = transform * (float(grid_dict["width"]), float(grid_dict["height"]))
    bounds = (min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))
    return AnalysisGrid(
        crs=as_crs(grid_dict["crs"]), transform=transform,
        width=int(grid_dict["width"]), height=int(grid_dict["height"]),
        resolution=float(grid_dict["resolution_m"]),
        requested_resolution=float(grid_dict.get("requested_resolution_m")
                                   or grid_dict["resolution_m"]),
        bounds=bounds, note="", buffer_cells=0)


def _layer_at(grid_dict: Dict[str, Any], geometry) -> Tuple[Any, Any, Any]:
    """WorldCover (windowed, cached) + the ROI mask on the engine's own grid."""
    analysis_grid = _grid_from_dict(grid_dict)
    layer = worldcover.fetch_land_cover(analysis_grid, use_cache=True)
    return analysis_grid, np.asarray(layer.array), _roi_inside(geometry, analysis_grid)


def verify_proximity(aoi_name: str, case_row: Optional[Dict[str, Any]],
                     distance_m: float) -> Dict[str, Any]:
    """Independent brute-force check of the engine's proximity condition.

    The check runs on the SAME window the engine used (which may be the
    buffered one), so the comparison is like-for-like; the distances themselves
    are computed by explicit coordinate arithmetic, not by the EDT the engine
    uses.
    """
    print(f"\n--- proximity: {aoi_name} (independent brute-force check) ---")
    if case_row is None or case_row.get("grid") is None \
            or not case_row.get("condition_masks"):
        print("  no proximity case available: SKIPPED")
        return {"skipped": "no proximity case"}

    grid = Grid.from_analysis_grid(_grid_from_dict(case_row["grid"]))
    array = _layer_at(case_row["grid"], AOIS[aoi_name])[1]
    water = (array == 80)

    prox_names = [n for n in case_row["condition_masks"]
                  if "proximity" in n.lower() or n.lower().startswith("within")]
    if not prox_names:
        print("  no proximity condition in this case: SKIPPED")
        return {"skipped": "no proximity condition"}
    engine_state = case_row["condition_masks"][prox_names[0]]

    n_water = int(water.sum())
    print(f"  window                 : {grid.height} x {grid.width} cells "
          f"@ {grid.pixel_width_m:g} m")
    print(f"  requested distance     : {distance_m:g} m")
    print(f"  water cells in window  : {n_water:,}")
    if n_water == 0:
        truth = case_row.get("condition_results", [{}])[-1]
        print("  NO WATER in the source window: the engine must report zero "
              "matches and must NOT claim 'far from water'.")
        print(f"      engine matched cells      : "
              f"{case_row.get('matched_cell_count', 0):,}")
        print(f"      engine insufficient cells : "
              f"{case_row.get('insufficient_cell_count', 0):,}")
        print(f"      water_cells_found (provenance): "
              f"{truth.get('provenance', {}).get('water_cells_found')}")
        return {"no_water_in_window": True,
                "matched": case_row.get("matched_cell_count", 0),
                "insufficient": case_row.get("insufficient_cell_count", 0),
                "water_cells_found": truth.get("provenance", {}).get("water_cells_found")}

    wy, wx = np.nonzero(water)
    # Pick the water cell in the sparsest neighbourhood available, so the
    # sample window actually reaches past the threshold and the boundary
    # behaviour is exercised rather than skipped.
    half = 40
    if wy.size > 1:
        sample = np.random.default_rng(7).choice(
            wy.size, size=min(400, wy.size), replace=False)
        best_cells, best_count = (int(wy[len(wy) // 2]), int(wx[len(wx) // 2])), None
        for index in sample:
            r, c = int(wy[index]), int(wx[index])
            r_lo, r_hi = max(0, r - half), min(water.shape[0], r + half)
            c_lo, c_hi = max(0, c - half), min(water.shape[1], c + half)
            count = int(water[r_lo:r_hi, c_lo:c_hi].sum())
            if best_count is None or count < best_count:
                best_cells, best_count = (r, c), count
        cy, cx = best_cells
    else:
        cy, cx = int(wy[0]), int(wx[0])
    # +/- 40 cells = +/- 1200 m around a water cell, so the window straddles
    # the 1000 m threshold and boundary behaviour is actually sampled.
    r0, c0 = max(0, cy - half), max(0, cx - half)
    r0, c0 = min(r0, max(0, grid.height - 2 * half)), min(c0, max(0, grid.width - 2 * half))
    rows, cols = slice(r0, r0 + 2 * half), slice(c0, c0 + 2 * half)

    brute = brute_force_distance(water, grid.pixel_width_m, rows, cols,
                                 distance_m)
    sub_state = engine_state[rows, cols]
    expected_within = brute <= distance_m
    decidable = sub_state > INSUFFICIENT
    agree = int(np.count_nonzero((sub_state[decidable] == TRUE)
                                 == expected_within[decidable]))
    total = int(np.count_nonzero(decidable))
    zero_on_water = float(brute[np.nonzero(water[rows, cols])].max())

    just_outside = (brute > distance_m) & (brute <= distance_m + grid.pixel_width_m)
    just_inside = (brute <= distance_m) & (brute > distance_m - grid.pixel_width_m)
    inside_true = int(np.count_nonzero((sub_state == TRUE) & just_inside))
    outside_true = int(np.count_nonzero((sub_state == TRUE) & just_outside))

    print(f"  60x60 check window     : rows {r0}-{r0 + 60}, cols {c0}-{c0 + 60} "
          f"around water cell ({cy}, {cx})")
    print(f"  distance on water      : max {zero_on_water:.3f} m (expect 0.000)")
    print(f"  brute-force distances  : {np.nanmin(brute):.1f} - "
          f"{np.nanmax(brute):.1f} m (metres, 30 m cells)")
    print(f"  cells compared         : {total:,} (engine-decidable)")
    print(f"  agreement with brute force : {agree:,}/{total:,} "
          f"({agree / max(total, 1) * 100:.4f}%)")
    print(f"  boundary  just inside  : {int(np.count_nonzero(just_inside)):,} cells, "
          f"engine TRUE on {inside_true:,}")
    print(f"  boundary  just outside : {int(np.count_nonzero(just_outside)):,} cells, "
          f"engine TRUE on {outside_true:,}  (must be 0)")
    # unit sanity: walking k cells east from a water cell must give k * 30 m
    # (unless another water cell is even closer, which can only shorten it)
    probes = []
    for k in (1, 2, 5, 10, 20, 33):
        r, c = cy - r0, cx - c0 + k
        if 0 <= r < brute.shape[0] and 0 <= c < brute.shape[1]:
            probes.append((k, float(brute[r, c])))
    worst = max((abs(d - k * grid.pixel_width_m) for k, d in probes), default=0.0)
    violations = [k for k, d in probes if d > k * grid.pixel_width_m + 1e-6]
    k_word = "distance"
    print("  unit sanity (walking east from a water cell): " + ", ".join(
        f"{k} cells = {d:.1f} m" for k, d in probes))
    print(f"      a distance may only be SHORTER than k x {grid.pixel_width_m:g} m "
          f"(another water cell closer); "
          f"cells farther than that: {len(violations)} (must be 0)")

    return {
        "window_cells": [grid.height, grid.width],
        "pixel_m": grid.pixel_width_m,
        "requested_distance_m": distance_m,
        "water_cells_in_window": n_water,
        "max_distance_on_water_m": zero_on_water,
        "cells_compared": total,
        "agreement": agree,
        "agreement_pct": agree / max(total, 1) * 100.0,
        "just_inside_cells": int(np.count_nonzero(just_inside)),
        "just_inside_engine_true": inside_true,
        "just_outside_cells": int(np.count_nonzero(just_outside)),
        "just_outside_engine_true": outside_true,
        "brute_min_m": float(np.nanmin(brute)),
        "brute_max_m": float(np.nanmax(brute)),
        "orthogonal_probes": {str(k): d for k, d in probes},
        "orthogonal_max_deviation_m": worst,
        "orthogonal_violations": len(violations),
    }


def verify_combinations(aoi_name: str, geometry,
                        rows: List[Dict[str, Any]],
                        distance_m: float) -> Dict[str, Any]:
    """Recompute AND / NOT in plain NumPy from the RAW arrays, on the same
    window each case used, and compare cell by cell with the engine."""
    print(f"\n--- combinations: {aoi_name} (independent NumPy recomputation) ---")
    by_case = {r["case"]: r for r in rows}
    checks: Dict[str, Any] = {}

    def state(t: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros(t.shape, dtype=np.uint8)
        out[v] = 1
        out[v & t] = 2
        return out

    def compare(name: str, row: Optional[Dict[str, Any]],
                build_expected) -> None:
        if row is None or row.get("result_mask") is None \
                or row.get("grid") is None:
            checks[name] = "case missing"
            return
        _, array, inside = _layer_at(row["grid"], geometry)
        grid = Grid.from_analysis_grid(_grid_from_dict(row["grid"]))
        water_valid = (array > 0) & np.isfinite(array)
        water_true = water_valid & (array == 80)
        cropland_true = water_valid & (array == 40)
        engine_state = np.asarray(row["result_mask"])
        expected = build_expected(array, water_valid, water_true,
                                  cropland_true, grid, inside)
        same = bool(np.array_equal(engine_state[inside], expected[inside]))
        checks[name] = {
            "window": [grid.height, grid.width],
            "cells_compared": int(np.count_nonzero(inside)),
            "identical": same,
            "engine_true": int(np.count_nonzero(engine_state[inside] == TRUE)),
            "expected_true": int(np.count_nonzero(expected[inside] == TRUE)),
            "engine_insufficient": int(np.count_nonzero(
                engine_state[inside] == INSUFFICIENT)),
            "expected_insufficient": int(np.count_nonzero(
                expected[inside] == INSUFFICIENT)),
        }

    def _cotton_and_near(array, water_valid, water_true, cropland_true,
                         grid, inside):
        from analyses.spatial_query import expanded_analysis_grid
        from core.spatial import crop_mask
        row = by_case["4. cotton AND near permanent water"]
        name = [n for n in row["condition_masks"] if "cotton" in n.lower()]
        if not name:
            return np.zeros(array.shape, dtype=np.uint8)
        cotton_state = row["condition_masks"][name[0]]
        # replicate Decision 1: read a buffered window, compute proximity there,
        # then crop back to the analysis grid
        source = (row.get("performance", {}).get("window", {})
                  or {}).get("source_window_cells")
        if source and source[0] > grid.height:
            extra = (int(source[0]) - int(grid.height)) // 2
            big, _ = expanded_analysis_grid(_grid_from_dict(row["grid"]), extra)
            big_array = np.asarray(
                worldcover.fetch_land_cover(big, use_cache=True).array)
            prox_big = proximity_mask(
                water_mask_from_land_cover(
                    big_array, Grid.from_analysis_grid(big), water_class=80),
                distance_m)
            prox_state = crop_mask(prox_big, grid).state
        else:
            prox = proximity_mask(
                water_mask_from_land_cover(array, grid, water_class=80),
                distance_m)
            prox_state = state(prox.match, prox.valid)
        return np.where(
            (cotton_state == FALSE) | (prox_state == FALSE), FALSE,
            np.where((cotton_state == INSUFFICIENT)
                     | (prox_state == INSUFFICIENT), INSUFFICIENT, TRUE)
        ).astype(np.uint8)

    def _cropland_and_near(array, water_valid, water_true, cropland_true,
                           grid, inside):
        prox = proximity_mask(
            water_mask_from_land_cover(array, grid, water_class=80), distance_m)
        prox_state = state(prox.match, prox.valid)
        crop_state = state(cropland_true, water_valid)
        return np.where(
            (crop_state == FALSE) | (prox_state == FALSE), FALSE,
            np.where((crop_state == INSUFFICIENT)
                     | (prox_state == INSUFFICIENT), INSUFFICIENT, TRUE)
        ).astype(np.uint8)

    def _cropland_and_not_water(array, water_valid, water_true, cropland_true,
                                grid, inside):
        crop_state = state(cropland_true, water_valid)
        water_state = state(water_true, water_valid)
        not_water = np.where(water_state == TRUE, FALSE,
                             np.where(water_state == FALSE, TRUE,
                                      INSUFFICIENT)).astype(np.uint8)
        return np.where(
            (crop_state == FALSE) | (not_water == FALSE), FALSE,
            np.where((crop_state == INSUFFICIENT) | (not_water == INSUFFICIENT),
                     INSUFFICIENT, TRUE)).astype(np.uint8)

    compare("cotton AND near-water",
            by_case.get("4. cotton AND near permanent water"), _cotton_and_near)
    compare("cropland AND near-water",
            by_case.get("5. cropland AND near permanent water"),
            _cropland_and_near)
    compare("cropland AND NOT water",
            by_case.get("6. cropland AND NOT permanent water"),
            _cropland_and_not_water)

    # semantic regression: NOT water must not behave like NOT water proximity
    row = by_case.get("6. cropland AND NOT permanent water")
    if row is not None and row.get("grid") is not None:
        _, array, inside = _layer_at(row["grid"], geometry)
        grid = Grid.from_analysis_grid(_grid_from_dict(row["grid"]))
        valid = (array > 0) & np.isfinite(array)
        crop_state = state(valid & (array == 40), valid)
        water_state = state(valid & (array == 80), valid)
        prox = proximity_mask(
            water_mask_from_land_cover(array, grid, water_class=80), distance_m)
        prox_state = state(prox.match, prox.valid)
        not_water = np.where(water_state == TRUE, FALSE,
                             np.where(water_state == FALSE, TRUE, INSUFFICIENT))
        not_prox = np.where(prox_state == TRUE, FALSE,
                            np.where(prox_state == FALSE, TRUE, INSUFFICIENT))
        crop_and_not_water = np.where(
            (crop_state == FALSE) | (not_water == FALSE), FALSE,
            np.where((crop_state == INSUFFICIENT) | (not_water == INSUFFICIENT),
                     INSUFFICIENT, TRUE)).astype(np.uint8)
        crop_and_not_prox = np.where(
            (crop_state == FALSE) | (not_prox == FALSE), FALSE,
            np.where((crop_state == INSUFFICIENT) | (not_prox == INSUFFICIENT),
                     INSUFFICIENT, TRUE)).astype(np.uint8)
        checks["NOT water != NOT near-water"] = {
            "NOT water TRUE cells": int(np.count_nonzero(
                (crop_and_not_water == TRUE) & inside)),
            "NOT near-water TRUE cells": int(np.count_nonzero(
                (crop_and_not_prox == TRUE) & inside)),
            "identical": bool(np.array_equal(crop_and_not_water[inside],
                                             crop_and_not_prox[inside])),
        }

    for name, outcome in checks.items():
        if isinstance(outcome, str):
            print(f"  {name:<32} {outcome}")
            continue
        if "identical" in outcome and "window" in outcome:
            print(f"  {name:<32} identical={outcome['identical']}  "
                  f"engine TRUE={outcome['engine_true']:,}  "
                  f"recomputed TRUE={outcome['expected_true']:,}  "
                  f"({outcome['cells_compared']:,} ROI cells, window "
                  f"{outcome['window'][0]}x{outcome['window'][1]})")
        else:
            print(f"  {name:<32} {outcome}")
    return checks


def verify_cotton_independent(aoi_name: str, cotton_raster: np.ndarray,
                              rows: List[Dict[str, Any]],
                              inside: np.ndarray) -> Dict[str, Any]:
    """The cotton mask must be exactly (Phase 8 class >= 3), nothing else."""
    expected = np.zeros(cotton_raster.shape, dtype=np.uint8)
    valid = np.isfinite(cotton_raster) & (cotton_raster >= 1)
    expected[valid] = 1
    expected[valid & (cotton_raster >= 3)] = 2

    row = next((r for r in rows if r["case"].startswith("1.")), None)
    outcome: Dict[str, Any] = {}
    if row and row.get("condition_masks"):
        name = [n for n in row["condition_masks"] if "cotton" in n.lower()]
        if name:
            engine = row["condition_masks"][name[0]]
            outcome = {
                "identical": bool(np.array_equal(engine, expected)),
                "cells": int(engine.size),
                "engine_true": int(np.count_nonzero(engine == TRUE)),
                "expected_true": int(np.count_nonzero(expected == TRUE)),
                "engine_insufficient": int(np.count_nonzero(engine == INSUFFICIENT)),
                "expected_insufficient": int(np.count_nonzero(expected == INSUFFICIENT)),
                "phase8_classes_in_roi": {
                    str(int(v)): int(np.count_nonzero((cotton_raster == v) & inside))
                    for v in np.unique(cotton_raster)},
            }
    print(f"\n--- cotton: {aoi_name} (recomputed from the Phase 8 class raster) ---")
    if outcome:
        print(f"  engine mask == (class >= 3) : {outcome['identical']} "
              f"({outcome['cells']:,} cells)")
        print(f"  TRUE cells   : engine {outcome['engine_true']:,} / "
              f"expected {outcome['expected_true']:,}")
        print(f"  INSUFFICIENT : engine {outcome['engine_insufficient']:,} / "
              f"expected {outcome['expected_insufficient']:,}")
        print(f"  Phase 8 class counts inside ROI: "
              f"{outcome['phase8_classes_in_roi']}")
    return outcome


def verify_gsw(aoi_name: str, wc: Dict[str, Any]) -> Dict[str, Any]:
    """Optional independent cross-check against JRC Global Surface Water.

    GSW answers 'how often was water observed 1984-2021', which is a different
    question from WorldCover's 2021 snapshot, so disagreement is expected and is
    reported, not reconciled.
    """
    import urllib.request
    from rasterio.windows import from_bounds

    url = ("https://storage.googleapis.com/global-surface-water/downloads2021/"
           "occurrence/occurrence_30E_40Nv1_4_2021.tif")
    print(f"\n--- JRC GSW cross-check: {aoi_name} (verification only) ---")
    grid = wc["grid"]
    try:
        import rasterio
        import rasterio.warp
        from pyproj import Transformer

        bounds = grid.bounds
        transformer = Transformer.from_crs("EPSG:32636", "EPSG:4326",
                                           always_xy=True)
        minlon, minlat = transformer.transform(bounds[0], bounds[1])
        maxlon, maxlat = transformer.transform(bounds[2], bounds[3])
        out = np.zeros((grid.height, grid.width), dtype=np.uint8)
        with rasterio.open(url) as src:
            window = from_bounds(minlon, minlat, maxlon, maxlat, src.transform)
            rasterio.warp.reproject(
                source=rasterio.band(src, 1), destination=out,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=grid.transform, dst_crs=grid.crs,
                dst_nodata=255, resampling=rasterio.warp.Resampling.nearest)
        array = wc["array"]
        wc_water = array == 80
        gsw_any = (out > 0) & (out != 255)
        valid = gsw_any | (out == 0)
        agreement = {
            "gsw_cells_read": int(valid.sum()),
            "gsw_water_any_fraction": float(gsw_any.sum() / max(valid.sum(), 1)),
            "worldcover_water_fraction": float(wc_water.sum() / max(valid.sum(), 1)),
            "both": int(np.count_nonzero(wc_water & gsw_any)),
            "worldcover_only": int(np.count_nonzero(wc_water & ~gsw_any & valid)),
            "gsw_only": int(np.count_nonzero(~wc_water & gsw_any & valid)),
        }
        print(f"  GSW window            : {grid.height}x{grid.width} "
              f"(reprojected from EPSG:4326)")
        print(f"  GSW 'water observed'  : "
              f"{agreement['gsw_water_any_fraction'] * 100:.2f}% of cells")
        print(f"  WorldCover 2021 water : "
              f"{agreement['worldcover_water_fraction'] * 100:.2f}% of cells")
        print(f"  both                  : {agreement['both']:,} cells")
        print(f"  WorldCover only       : {agreement['worldcover_only']:,} cells")
        print(f"  GSW only (ephemeral)  : {agreement['gsw_only']:,} cells")
        print("  (GSW counts ANY observation 1984-2021, including ephemeral and")
        print("   flooded cropland; WorldCover is a single 2021 epoch. They are")
        print("   NOT expected to agree, and GSW is never used as the engine.)")
        return agreement
    except Exception as exc:                                # pragma: no cover
        print(f"  GSW cross-check unavailable: {type(exc).__name__}: {exc}")
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The queries a user actually types, run end to end on real data
# --------------------------------------------------------------------------- #
GALLERY: Tuple[str, ...] = (
    "Find cropland near water",
    "Find areas suitable for cotton near water",
    "Find cropland excluding water",
    "Find permanent water",
    "Find built-up areas near water",
    "Find cropland but not permanent water",
    "Where are the suitable agricultural areas near water?",
    "Find cotton land with reliable irrigation",
    "Find areas that are not flood-prone",
)


def run_gallery(aoi_name: str, geometry, cotton_runner) -> List[Dict[str, Any]]:
    """Every representative query through the router and the engine.

    Supported ones must return counts; unsupported ones must refuse without
    computing anything and must never be silently answered by a proxy.
    """
    context = AnalysisContext(roi=roi_for(geometry))
    print(f"\n--- {aoi_name}: representative user queries (real data) ---")
    rows: List[Dict[str, Any]] = []
    for query_text in GALLERY:
        parsed = parse_query(query_text)
        execution = run_spatial_query(context, parsed,
                                      suitability_runner=cotton_runner)
        conditions = ", ".join(
            ("NOT " if c.negate else "") + c.condition_type.value
            for c in (parsed.conditions if parsed else ()))
        row: Dict[str, Any] = {"query": query_text,
                               "conditions": conditions,
                               "execution_status": execution.status.value}
        if execution.result is None:
            message = " ".join((execution.message or "").split())
            print(f"  {query_text!r}")
            print(f"      -> {execution.status.value} "
                  f"({conditions or 'no conditions'})")
            print(f"         {message[:150]}")
            row.update({"result_status": None, "message": message,
                        "matched_cell_count": None,
                        "insufficient_cell_count": None,
                        "analysed_cell_count": None})
        else:
            r = execution.result
            print(f"  {query_text!r}")
            print(f"      -> {r.status}: {r.matched_cell_count:,} matched, "
                  f"{r.insufficient_cell_count:,} insufficient of "
                  f"{r.analysed_cell_count:,} analysed  [{conditions}]")
            row.update({"result_status": r.status,
                        "expression": r.expression,
                        "matched_cell_count": r.matched_cell_count,
                        "insufficient_cell_count": r.insufficient_cell_count,
                        "analysed_cell_count": r.analysed_cell_count,
                        "matched_fraction": r.matched_fraction,
                        "message": " ".join((r.message or "").split())})
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gsw", action="store_true")
    parser.add_argument("--aoi", choices=list(AOIS), default=None)
    args = parser.parse_args()

    tracemalloc.start()
    started = time.perf_counter()
    selected = {args.aoi: AOIS[args.aoi]} if args.aoi else AOIS
    evidence: Dict[str, Any] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "python": platform.python_version(),
                                "platform": platform.platform(),
                                "aois": {}}

    rule("PHASE 9 CHECKPOINT D -- REAL-DATA VERIFICATION")
    print("No UI, no map layer, no browser test. Real WorldCover + Phase 8 data.")

    # the configured proximity distance
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.spatial_query import load_spatial_config
    conditions_cfg = load_spatial_config("conditions")
    default_distance = float(conditions_cfg["water"]["default_proximity_m"])
    water_class = int(conditions_cfg["water"]["water_class"])
    print(f"\nConfigured semantics: cotton class >= 3, scenario = rainfed, "
          f"water class = {water_class}, cropland class = 40, "
          f"default proximity = {default_distance:g} m")

    for name, geometry in selected.items():
        rule(f"{name}  --  {geometry.bounds}  "
             f"({geometry.bounds[2] - geometry.bounds[0]:.0f} m square)")
        print(f"AOI area: {geometry.area / 1e6:.3f} km2   CRS: EPSG:32636")

        # -- 1. the source dataset ------------------------------------------ #
        wc = verify_worldcover(name, geometry)
        evidence["aois"][name] = {
            "bounds": list(geometry.bounds),
            "worldcover": {
                "record": wc["record"], "classes": wc["classes"],
                "grid": wc["grid"].to_dict(), "from_cache": wc["from_cache"],
                "seconds": wc["seconds"],
            },
        }

        # -- the Phase 8 engine, invoked once per AOI and reused ------------ #
        # The cotton screening does not depend on the spatial conditions, so
        # one real run per AOI is authoritative for every case that needs it.
        cache: Dict[str, Any] = {}

        def cotton_runner(context, query, **kwargs):
            if "execution" not in cache:
                t0 = time.perf_counter()
                cache["execution"] = run_crop_suitability(context, query, **kwargs)
                cache["seconds"] = time.perf_counter() - t0
                cache["raster"] = np.asarray(
                    cache["execution"].result.scenarios["rainfed"].suitability_raster)
                print(f"\n  [Phase 8 cotton engine: real run, "
                      f"{cache['seconds']:.1f}s, status="
                      f"{cache['execution'].status.value}]")
            return cache["execution"]

        # -- 2. the nine cases ---------------------------------------------- #
        rows = run_cases(name, geometry, wc, cotton_runner)
        evidence["aois"][name]["cases"] = [
            {k: v for k, v in row.items()
             if k not in ("result_mask", "condition_masks")} for row in rows]

        # -- 2b. the queries a user actually types -------------------------- #
        evidence["aois"][name]["gallery"] = run_gallery(
            name, geometry, cotton_runner)

        # -- 3. independent verification ------------------------------------ #
        evidence["aois"][name]["checks"] = {}
        evidence["aois"][name]["checks"]["proximity"] = verify_proximity(
            name, next((r for r in rows if r["case"].startswith("3.")), None),
            default_distance)
        evidence["aois"][name]["checks"]["combinations"] = verify_combinations(
            name, geometry, rows, default_distance)
        if "raster" in cache:
            evidence["aois"][name]["checks"]["cotton"] = verify_cotton_independent(
                name, cache["raster"], rows, wc["inside"])
        if not args.skip_gsw:
            evidence["aois"][name]["checks"]["gsw"] = verify_gsw(name, wc)

        # -- the required distinction --------------------------------------- #
        print(f"\n--- {name}: zero matches vs no water vs insufficient vs "
              f"unsupported ---")
        water_row = next((r for r in rows if r["case"].startswith("2.")), None)
        cotton_row = next((r for r in rows if r["case"].startswith("1.")), None)
        zero_row = next((r for r in rows if r["case"].startswith("7.")), None)
        combo_row = next((r for r in rows if r["case"].startswith("4.")), None)
        unsupported = [r for r in rows if r["status"] == "UNSUPPORTED_CONDITION"]
        in_window = wc["classes"].get(80, 0)
        print(f"  class-80 cells in the AOI source window : {in_window:,}"
              f"{'  <-- NO permanent water in the source' if in_window == 0 else ''}")
        if water_row and water_row.get("matched_cell_count") is not None:
            print(f"  (a) water alone      : matched "
                  f"{water_row['matched_cell_count']:,}, insufficient "
                  f"{water_row['insufficient_cell_count']:,}"
                  f"   -> {'zero matches, source has no water, all cells valid'
                         if in_window == 0 else 'positive water case'}")
        if zero_row and zero_row.get("matched_cell_count") is not None:
            print(f"  (b) impossible combo : matched "
                  f"{zero_row['matched_cell_count']:,}, insufficient "
                  f"{zero_row['insufficient_cell_count']:,}"
                  f"   -> zero valid matches (all cells decided)")
        if cotton_row and cotton_row.get("insufficient_cell_count") is not None:
            print(f"  (c) cotton alone     : matched "
                  f"{cotton_row['matched_cell_count']:,}, insufficient "
                  f"{cotton_row['insufficient_cell_count']:,}"
                  f"   -> insufficient data counted separately")
        if combo_row and combo_row.get("insufficient_cell_count") is not None:
            print(f"  (d) cotton AND near  : matched "
                  f"{combo_row['matched_cell_count']:,}, insufficient "
                  f"{combo_row['insufficient_cell_count']:,}")
        print(f"  (e) unsupported      : {len(unsupported)} queries refused with "
              f"no computation: "
              f"{[u['case'] for u in unsupported]}")

        # -- the headline messages ------------------------------------------ #
        print(f"\n--- {name}: engine wordings ---")
        for row in rows:
            if row.get("message") and row["case"].startswith(("2.", "4.", "7.")):
                print(f"  {row['case']}")
                print(f"      {row['message'][:300]}")
        for row in rows:
            for warning in row.get("warnings", []) or []:
                if "buffer" in warning.lower() or "window" in warning.lower():
                    print(f"  warning ({row['case']}): {warning}")
                    break

    elapsed = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rule("PERFORMANCE")
    print(f"  total wall time : {elapsed:.1f} s")
    print(f"  tracemalloc peak: {peak / 1e6:.1f} MB (current {current / 1e6:.1f} MB)")
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"  process peak RSS: {rss / 1024:.1f} MB")
        evidence["peak_rss_mb"] = rss / 1024
    except Exception:                                       # pragma: no cover
        pass
    evidence["elapsed_seconds"] = elapsed
    evidence["tracemalloc_peak_mb"] = peak / 1e6

    out_path = Path(__file__).resolve().parents[1] / "docs" / "PHASE9D_evidence.json"
    out_path.write_text(json.dumps(evidence, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nEvidence written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
