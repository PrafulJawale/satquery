"""Phase 9, Checkpoint C -- reproducible synthetic demonstration.

NO real data, NO network, NO UI. It builds a synthetic WorldCover-like layer
(30 m analysis grid, a block of class-80 water) and a synthetic Phase 8 cotton
class raster, injects them as providers, and prints:

  1. the structured query each sentence parses to;
  2. the resulting mask counts for the queries that are computable.

Run:  python scripts/demo_phase9_masks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses import AnalysisContext, Status                      # noqa: E402
from analyses.spatial_query import run_spatial_query               # noqa: E402
from core.alignment import make_grid                               # noqa: E402
from core.geometry import as_crs                                   # noqa: E402
from core.roi import ROISelection                                  # noqa: E402
from core.router import parse_query                                # noqa: E402
from core.spatial_query import parse_spatial_query                 # noqa: E402
from shapely.geometry import box                                   # noqa: E402

X0, Y0, SIZE = 377200.0, 3441820.0, 3000.0


# -- synthetic providers ---------------------------------------------------- #
class _Record:
    source = "ESA WorldCover 2021 v200 (SYNTHETIC stand-in)"
    native_resolution_m = 10.0

    def to_dict(self):
        return {"source": self.source,
                "native_resolution_m": self.native_resolution_m}


class _Layer:
    def __init__(self, array):
        self.array = array
        self.record = _Record()


class _Scenario:
    def __init__(self, raster):
        self.suitability_raster = raster
        self.native_resolutions = {"worldclim": 1000.0, "soilgrids": 250.0}
        self.provenance = [{"source": "SYNTHETIC", "factor": "ph"}]
        self.analysis_resolution = 30.0


class _Screening:
    def __init__(self, grid, raster):
        self.grid = grid
        self.scenarios = {"rainfed": _Scenario(raster)}


def build_layers(size: int):
    """Cropland with a 4x4 block of permanent water in the middle."""
    land_cover = np.full((size, size), 40, dtype=np.uint8)
    centre = size // 2
    land_cover[centre:centre + 4, centre:centre + 4] = 80
    land_cover[0:8, 0:8] = 0                      # a patch of nodata
    # cotton classes: 4 (highly) except a band of 2 (marginal) and 0 (nodata)
    cotton = np.full((size, size), 4, dtype=np.uint8)
    cotton[0:20, :] = 2
    cotton[20:24, :] = 0
    return land_cover, cotton


def main() -> int:
    roi = ROISelection(is_valid=True, intersects_raster=True,
                       area_m2=SIZE * SIZE, raster_crs="EPSG:32636",
                       geometry_raster_crs=box(X0, Y0, X0 + SIZE, Y0 + SIZE),
                       geometry_type="Polygon", num_parts=1)
    grid = make_grid(roi.geometry_raster_crs, as_crs(roi.raster_crs),
                     requested_resolution=30.0, buffer_cells=2)
    land_cover, cotton = build_layers(grid.height)
    context = AnalysisContext(roi=roi)

    print("=" * 78)
    print("PHASE 9 CHECKPOINT C -- SYNTHETIC DEMONSTRATION (no real data)")
    print(f"analysis grid: {grid.height}x{grid.width} cells at 30 m "
          f"({grid.height * grid.width:,} cells, "
          f"cell area {900.0:,.0f} m2)")
    print("=" * 78)

    print("\n1. STRUCTURED QUERIES (parser output)")
    print("-" * 78)
    for query in ("Find areas suitable for cotton",
                  "Find cropland",
                  "Find cropland near water",
                  "Find cropland within 500 m of water",
                  "Find cropland excluding water",
                  "Find cropland excluding areas near water",
                  "Find cotton areas near water but not built-up",
                  "Find cotton areas with irrigation",
                  "Can I grow cotton with irrigation?"):
        parsed = parse_spatial_query(query)
        conditions = ", ".join(c.label for c in parsed.conditions) or "-"
        print(f"  {query:<46} {parsed.status.value:<20}")
        print(f"      {parsed.expression() or conditions}")

    print("\n2. SPATIAL MASK RESULTS (engine output, synthetic layers)")
    print("-" * 78)
    header = f"  {'query':<44} {'match':>8} {'area km2':>9} {'match %':>8} {'insuff %':>9}"
    print(header)

    for query in ("Find cropland",
                  "Find water",
                  "Find cropland near water",
                  "Find cropland within 500 m of water",
                  "Find cropland excluding water",
                  "Find cropland excluding areas near water",
                  "Find cotton areas",
                  "Find cotton areas near water",
                  "Find cotton areas with irrigation"):
        parsed = parse_query(query)
        execution = run_spatial_query(
            context, parsed,
            suitability_runner=lambda ctx, q, **kw: _fake_run(
                q, grid, cotton),
            land_cover_fetcher=lambda g, use_cache=True: _Layer(
                build_layers(g.height)[0]))
        result = execution.result
        if result is None:
            print(f"  {query:<44} {execution.status.value:>8} "
                  f"{'—':>9} {'—':>8} {'—':>9}")
            continue
        total = max(result.analysed_cell_count, 1)
        print(f"  {query:<44} {result.matched_cell_count:>8,} "
              f"{result.matched_area_m2 / 1e6:>9.3f} "
              f"{result.matched_fraction * 100:>7.1f}% "
              f"{result.insufficient_cell_count / (total + result.insufficient_cell_count) * 100:>8.1f}%")

    print("\n3. ONE RESULT IN FULL")
    print("-" * 78)
    parsed = parse_query("Find cropland near water")
    execution = run_spatial_query(
        context, parsed,
        land_cover_fetcher=lambda g, use_cache=True: _Layer(
            build_layers(g.height)[0]))
    print(execution.message)
    print()
    for condition in execution.result.condition_results:
        print(f"  - {condition['name']}: "
              f"{condition['counts']['matching_cells']:,} cells match, "
              f"{condition['counts']['insufficient_cells']:,} insufficient")
    print(f"  grid: {execution.result.grid['width']}x"
          f"{execution.result.grid['height']} cells, "
          f"{execution.result.grid['pixel_size_m']} m")
    print(f"  {execution.result.effective_resolution_note}")
    print()
    return 0


def _fake_run(query, grid, raster):
    """Stand in for the Phase 8 cotton engine."""
    from analyses.base import AnalysisExecution
    from core.router import Intent
    return AnalysisExecution(
        intent=Intent.CROP_SUITABILITY, status=Status.OK,
        query=query.original_query, normalized_query=query.normalized_query,
        confidence=query.confidence, explanation=query.explanation,
        matched=query.matched, message="SYNTHETIC",
        result=_Screening(grid, raster))


if __name__ == "__main__":
    raise SystemExit(main())
