"""Phase 7 verification harness: end-to-end query understanding + routing.

Run:
    python scripts/verify_phase7.py

Exits 0 only if every check passes.

It proves, on the REAL Sentinel-2 sample, that

    1. a natural-language query is parsed into a structured, explainable intent;
    2. the registry binds that intent to the existing Phase 6 engine;
    3. the statistics returned THROUGH THE ROUTER are identical to calling the
       Phase 6 engine directly, and to an independent numpy recomputation;
    4. the router performs no raster mathematics of its own;
    5. missing context, unsupported and ambiguous requests are reported, never
       guessed and never fabricated.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from rasterio import Affine
from shapely.geometry import box

from analyses import (
    AnalysisContext,
    Intent,
    NdviContext,
    Status,
    available_specs,
    get_spec,
    planned_specs,
    route,
    suggestions,
)
from core.indices import ndvi_from_dataset
from core.raster import open_dataset
from core.roi import ROISelection
from core.router import parse_query
from core.statistics import calculate_roi_ndvi_stats

CHECKS: list[tuple[bool, str]] = []
SCENE = REPO_ROOT / "data" / "sample" / "s2_s2b-36ruv-20230806-0-l2a_2048px.tif"


def check(ok: bool, label: str) -> None:
    CHECKS.append((bool(ok), label))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> int:
    print("=" * 78)
    print("PHASE 7 - END-TO-END QUERY UNDERSTANDING + ANALYSIS ROUTER")
    print("=" * 78)

    # ------------------------------------------------------------------ #
    print("\n1. The router module performs no raster mathematics")
    source = (REPO_ROOT / "core" / "router.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"numpy", "rasterio", "shapely", "pyproj", "geopandas", "matplotlib", "pandas"}
    check(not (imported & banned), f"core/router.py imports no raster library ({sorted(imported)})")
    check("calculate_roi_ndvi_stats" not in source,
          "core/router.py never references an analysis function")

    sub = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.');\n"
         "import core.router\n"
         "loaded=[m for m in ('numpy','rasterio','shapely','pyproj') if m in sys.modules]\n"
         "print('LOADED:', loaded)\n"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    check(sub.returncode == 0 and "LOADED: []" in sub.stdout,
          f"a clean 'import core.router' loads no raster stack ({sub.stdout.strip()})")

    # ------------------------------------------------------------------ #
    print("\n2. Registry: one place binds intents to engines")
    check(set(available_specs()) == {Intent.NDVI_ROI_STATS},
          "exactly one executable intent is registered")
    check(all(spec.handler is None for spec in planned_specs().values()),
          f"planned intents have no handler: {[i.value for i in planned_specs()]}")
    check(bool(suggestions()), f"suggested queries come from the registry: {suggestions(limit=2)}")

    # ------------------------------------------------------------------ #
    print("\n3. Query understanding (deterministic, explainable)")
    cases = [
        ("What is the NDVI of this area?", Intent.NDVI_ROI_STATS),
        ("Calculate the vegetation index here.", Intent.NDVI_ROI_STATS),
        ("Show vegetation health in this area.", Intent.NDVI_ROI_STATS),
        ("Analyze the vegetation in this selected area.", Intent.NDVI_ROI_STATS),
        ("   WHAT is   the NDVI  of this area??? ", Intent.NDVI_ROI_STATS),
        ("Can I grow cotton here?", Intent.CROP_SUITABILITY),
        ("Show flood areas.", Intent.FLOOD_CHANGE),
        ("How has the vegetation changed since 2023?", Intent.VEGETATION_CHANGE),
        ("What is the weather here?", Intent.UNKNOWN),
        ("Tell me about this area.", Intent.UNKNOWN),
    ]
    for query, expected in cases:
        parsed = parse_query(query)
        check(parsed.intent is expected,
              f"{query!r} -> {parsed.intent.value} (confidence {parsed.confidence:.2f})")
        if expected is Intent.NDVI_ROI_STATS:
            check(bool(parsed.matched), f"   explainable: matched {parsed.matched}")

    # ------------------------------------------------------------------ #
    print("\n4. End-to-end on the real Sentinel-2 sample")
    with open_dataset(str(SCENE)) as ds:
        result, _spec, _report = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        ndvi = np.asarray(result.array, dtype="float32")
        mask = np.asarray(result.mask, dtype=bool)
        transform = Affine(*tuple(result.transform))
        crs = result.crs
    height, width = ndvi.shape
    print(f"    scene: {SCENE.name}  {width}x{height}  {crs.to_string()}")

    minx, miny = transform * (0, height)
    maxx, maxy = transform * (width, 0)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    geom = box(cx - 500, cy - 500, cx + 500, cy + 500)
    roi = ROISelection(is_valid=True, intersects_raster=True, area_m2=float(geom.area),
                       raster_crs="EPSG:32636", geometry_raster_crs=geom,
                       geometry_type="Polygon", num_parts=1)
    context = AnalysisContext(
        roi=roi,
        ndvi=NdviContext(array=ndvi, mask=mask, crs=crs, transform=transform,
                         bands=dict(result.bands_used or {}), source_label=SCENE.name),
        ndvi_confirmed=True, raster_label=SCENE.name,
    )

    execution = route("What is the NDVI of this area?", context)
    direct = calculate_roi_ndvi_stats(ndvi, geom, transform, mask, crs=crs, roi_crs="EPSG:32636")

    print(f"    query    : \"What is the NDVI of this area?\"")
    print(f"    intent   : {execution.intent.value} (confidence {execution.confidence:.2f})")
    print(f"    status   : {execution.status.value}")
    print(f"    answer   : {execution.message}")
    if execution.result and execution.result.stats:
        st = execution.result.stats
        p = st["percentiles"]
        print(f"    stats    : inside={execution.result.pixels_inside_roi:,} "
              f"valid={execution.result.valid_pixels:,} "
              f"mean={st['mean']:.4f} median={st['median']:.4f} std={st['std']:.4f}")
        print(f"               min={st['min']:.4f} max={st['max']:.4f} "
              f"P5={p['p5']:.4f} P95={p['p95']:.4f}")
        print(f"    area     : {execution.result.area_m2 / 10_000:,.2f} ha, "
              f"valid {execution.result.valid_area_m2 / 10_000:,.2f} ha")

    check(execution.status is Status.OK, "the routed query executed")
    check(execution.result.pixels_inside_roi == direct.pixels_inside_roi == 10_000,
          f"pixel count identical to the direct engine call "
          f"({execution.result.pixels_inside_roi:,})")
    check(execution.result.valid_pixels == direct.valid_pixels,
          "valid-pixel count identical to the direct engine call")
    check(execution.to_dict()["result"] == direct.to_dict(),
          "the whole structured result is identical to calling Phase 6 directly")
    check(execution.provenance["engine"].startswith("core.statistics"),
          "provenance records the engine that produced the numbers")
    check(execution.provenance["native_resolution_m"] == [10.0, 10.0],
          "provenance records the native resolution (10 m x 10 m)")

    # independent recomputation: matplotlib Path + numpy, inside this process
    import matplotlib.path as mpath

    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    a, b, c, d, e, f = tuple(transform)[:6]
    xs = a * cols + b * rows + c
    ys = d * cols + e * rows + f
    inside = mpath.Path(np.asarray(geom.exterior.coords, float)).contains_points(
        np.column_stack([xs.ravel(), ys.ravel()]), radius=0.0).reshape(height, width)
    vals = ndvi[inside & mask]
    check(int(inside.sum()) == execution.result.pixels_inside_roi,
          f"pixel count matches an independent point-in-polygon test ({int(inside.sum()):,})")
    # float32 sums depend on accumulation order, so agreement is checked at
    # float32 precision (~1e-7 relative), not at 1e-9.
    delta = abs(float(vals.mean()) - float(execution.result.stats["mean"]))
    check(delta < 1e-6,
          f"mean NDVI matches the independent recomputation "
          f"({vals.mean():.9f} vs {execution.result.stats['mean']:.9f}, delta {delta:.2e})")
    check(abs(float(np.percentile(vals, 95)) - execution.result.stats["percentiles"]["p95"]) < 1e-6,
          "P95 matches the independent recomputation")

    # ------------------------------------------------------------------ #
    print("\n5. Context validation and honesty")
    no_roi = route("What is the NDVI of this area?",
                   AnalysisContext(roi=None, ndvi=context.ndvi, ndvi_confirmed=True))
    check(no_roi.status is Status.NEEDS_ROI and no_roi.result is None,
          f"no ROI -> {no_roi.status.value}: {no_roi.message}")

    unconfirmed = route("What is the NDVI of this area?",
                        AnalysisContext(roi=roi, ndvi=context.ndvi, ndvi_confirmed=False))
    check(unconfirmed.status is Status.NEEDS_NDVI_CONFIRMATION and unconfirmed.result is None,
          f"NDVI not confirmed -> {unconfirmed.status.value}: {unconfirmed.message}")

    tiny = box(minx - 50_000, miny - 50_000, minx - 49_000, miny - 49_000)
    empty_roi = ROISelection(is_valid=True, intersects_raster=True, area_m2=float(tiny.area),
                             raster_crs="EPSG:32636", geometry_raster_crs=tiny)
    empty = route("What is the NDVI of this area?",
                  AnalysisContext(roi=empty_roi, ndvi=context.ndvi, ndvi_confirmed=True))
    check(empty.status is Status.NO_VALID_PIXELS and empty.result.valid_pixels == 0,
          f"no pixels -> {empty.status.value}: {empty.message}")

    for query, intent in (("Can I grow cotton here?", Intent.CROP_SUITABILITY),
                          ("Show flood areas.", Intent.FLOOD_CHANGE),
                          ("How has the vegetation changed?", Intent.VEGETATION_CHANGE)):
        ex = route(query, context)
        check(ex.status is Status.UNSUPPORTED and ex.result is None,
              f"{query!r} -> {ex.status.value}: {ex.message}")
        check(get_spec(intent) is not None and get_spec(intent).handler is None,
              f"   {intent.value} is declared but has no engine")

    vague = route("Tell me about this area.", context)
    check(vague.status is Status.UNKNOWN and vague.result is None,
          f"ambiguous -> {vague.status.value}: {vague.message[:70]}...")
    check("NDVI" in vague.message, "the ambiguous answer points at what IS available")

    weather = route("What is the weather here?", context)
    check(weather.status is Status.UNKNOWN and weather.result is None,
          "an out-of-domain question produces no result")

    # ------------------------------------------------------------------ #
    print("\n6. Scientific honesty of the answer wording")
    blob = (execution.message + " " + " ".join(execution.warnings)).lower()
    check(not any(w in execution.message.lower()
                  for w in ("healthy", "suitab", "yield", "disease", "flood")),
          "the answer itself uses no health/suitability/yield wording")
    check("vegetation" in blob or "ndvi" in blob, "the answer says what was measured (NDVI)")
    check(any("not a crop-health" in w for w in execution.warnings),
          "the answer states it is not a crop-health diagnosis")

    failed = [label for ok, label in CHECKS if not ok]
    print("\n" + "=" * 78)
    print(f"PHASE 7 RESULT: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for label in failed:
        print(f"  FAILED: {label}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
