"""Phase 8 -- PERFORMANCE report (brief item #27).

Measures the stages SEPARATELY, because they cost different things:

    source access        rasterio opening a remote COG/VRT (network, headers)
    windowed read        pulling only the AOI window out of that source
    reprojection         warping that window onto the common analysis grid
    factor computation   masks, slope, GDD, texture class, ROI means
    suitability          memberships, weighted score, gates, classification
    map overlay          the DISPLAY-ONLY reprojection to web mercator
    total                end to end, plus peak memory

Run (cached layers):   python scripts/performance_phase8.py
Run (cold, network):   python scripts/performance_phase8.py --cold

Writes artifacts/phase8_performance.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                            # noqa: E402
from rasterio import Affine                                   # noqa: E402
from rasterio.crs import CRS                                  # noqa: E402
from rasterio.enums import Resampling                         # noqa: E402
from shapely.geometry import box                              # noqa: E402

from analyses import AnalysisContext                          # noqa: E402
from analyses.crop_suitability import run_crop_suitability    # noqa: E402
from core.datasources.base import CACHE_ROOT                  # noqa: E402
from core.geo import reproject_array, WEB_MERCATOR            # noqa: E402
from core.router import parse_query                           # noqa: E402
from core.roi import ROISelection                             # noqa: E402
from ui.map import suitability_rgba                           # noqa: E402

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# --- the 3 x 3 km Nile Delta ROI used by the verifier ----------------------- #
TEN_M = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3462300.0)
C0, R0, C1, R1 = 1024 - 150, 1024 - 150, 1024 + 150, 1024 + 150
X0, Y0 = TEN_M * (C0, R1)
X1, Y1 = TEN_M * (C1, R0)
ROI_BOX = box(min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1))


def make_roi() -> ROISelection:
    return ROISelection(is_valid=True, intersects_raster=True,
                        area_m2=float(ROI_BOX.area), raster_crs="EPSG:32636",
                        geometry_raster_crs=ROI_BOX, geometry_type="Polygon",
                        num_parts=1)


def rss_mb() -> float:
    """Resident set size in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure_map_overlay(result) -> dict:
    """Time the DISPLAY-ONLY step: classes -> web mercator -> RGBA."""
    primary = result.scenarios["rainfed"]
    t0 = time.perf_counter()
    web = reproject_array(
        np.asarray(primary.suitability_raster).astype("float32"),
        src_transform=Affine(*tuple(primary.raster_transform)[:6]),
        src_crs=CRS.from_user_input(str(primary.raster_crs)),
        dst_crs=WEB_MERCATOR, resampling=Resampling.nearest,
        max_pixels=2_000_000)
    arr = np.asarray(web.array)
    codes = np.where(np.isfinite(arr), np.rint(arr), 0).astype("int16")
    rgba = suitability_rgba(codes)
    rgba[~np.asarray(web.mask)] = (0, 0, 0, 0)
    seconds = time.perf_counter() - t0
    del rgba, web, arr, codes
    gc.collect()
    return {"map_overlay_seconds": round(seconds, 3),
            "map_overlay_pixels": int(np.asarray(primary.suitability_raster).size)}


def measure_render(result) -> dict:
    """Time rendering the panel in a real Streamlit runtime."""
    from streamlit.testing.v1 import AppTest

    holder = {}

    def _feed(r):
        holder["r"] = r

    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
from ui.components import render_crop_suitability
from performance_helper import RESULT
render_crop_suitability(RESULT)
'''
    # AppTest runs in-process, so pass the object through a module.
    mod = sys.modules.get("performance_helper")
    if mod is None:
        mod = type(sys)("performance_helper")
        sys.modules["performance_helper"] = mod
    mod.RESULT = result
    t0 = time.perf_counter()
    at = AppTest.from_string(script, default_timeout=600).run()
    seconds = time.perf_counter() - t0
    return {"render_seconds": round(seconds, 3),
            "render_raised": bool(at.exception)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true",
                    help="delete the external-data cache first (network run)")
    args = ap.parse_args()

    print("=" * 78)
    print("SatQuery AI -- Phase 8 PERFORMANCE")
    print("=" * 78)

    baseline_rss = rss_mb()
    if args.cold:
        print(f"\n--cold: removing {CACHE_ROOT} (every layer will be refetched)")
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)

    tracemalloc.start()
    t0 = time.perf_counter()
    execution = run_crop_suitability(
        AnalysisContext(roi=make_roi()), parse_query("Can I grow cotton here?"))
    wall = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if not execution.ok:
        print(f"  run failed: {execution.message}")
        return 1

    r = execution.result
    perf = dict(r.performance)
    perf["wall_seconds"] = round(wall, 3)
    perf["peak_rss_mb"] = round(rss_mb(), 1)
    perf["baseline_rss_mb"] = round(baseline_rss, 1)
    perf["tracemalloc_peak_mb"] = round(peak / 1024 / 1024, 1)
    perf["cold_cache"] = bool(args.cold)
    perf.update(measure_map_overlay(r))
    perf.update(measure_render(r))
    perf["python"] = platform.python_version()
    perf["platform"] = platform.platform()
    perf["grid"] = f"{r.grid.width} x {r.grid.height} @ {r.grid.resolution:g} m"
    perf["cells"] = int(r.grid.cells)

    order = [
        ("grid_seconds", "analysis grid"),
        ("source_access_seconds", "source access (opening remote sources)"),
        ("windowed_read_seconds", "windowed read + reprojection"),
        ("cache_read_seconds", "read back from local cache"),
        ("landcover_seconds", "  of which: land cover"),
        ("soil_seconds", "  of which: soil"),
        ("climate_seconds", "  of which: climate"),
        ("terrain_seconds", "  of which: terrain"),
        ("factor_seconds", "factor computation"),
        ("suitability_seconds", "suitability scoring + classification"),
        ("map_overlay_seconds", "map overlay (display only)"),
        ("render_seconds", "panel render (AppTest)"),
        ("total_seconds", "engine total (as reported)"),
        ("wall_seconds", "wall clock (as measured here)"),
    ]
    print(f"\n  grid: {perf['grid']}  ({perf['cells']:,} cells)   "
          f"cold cache: {perf['cold_cache']}")
    print(f"  {'stage':<42}{'seconds':>10}")
    print("  " + "-" * 52)
    for key, label in order:
        if key in perf:
            print(f"  {label:<42}{perf[key]:>10.3f}")
    print("  " + "-" * 52)
    print(f"  {'peak process RSS (MB)':<42}{perf['peak_rss_mb']:>10.1f}")
    print(f"  {'  (baseline after imports, MB)':<42}{perf['baseline_rss_mb']:>10.1f}")
    print(f"  {'tracemalloc peak (MB)':<42}{perf['tracemalloc_peak_mb']:>10.1f}")
    print(f"  {'map overlay pixels':<42}{perf['map_overlay_pixels']:>10,}")

    out = ARTIFACTS / ("phase8_performance_cold.json" if args.cold
                       else "phase8_performance.json")
    out.write_text(json.dumps(perf, indent=2), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
