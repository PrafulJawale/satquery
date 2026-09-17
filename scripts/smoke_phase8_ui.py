"""Phase 8 -- UI smoke test.

Render the NEW Phase 8 panel with a REAL engine result (cached external data,
so this is fast) inside a real Streamlit runtime, and exercise the map overlay
helper. It exists because the browser test is slow: this catches attribute-name
and Streamlit-API mistakes in seconds instead of minutes.

Run:  python scripts/smoke_phase8_ui.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                            # noqa: E402
from rasterio import Affine                                   # noqa: E402
from rasterio.crs import CRS                                  # noqa: E402
from rasterio.enums import Resampling                         # noqa: E402

from analyses import AnalysisContext                          # noqa: E402
from analyses.crop_suitability import run_crop_suitability    # noqa: E402
from core.geo import reproject_array, WEB_MERCATOR            # noqa: E402
from core.router import parse_query                           # noqa: E402
from core.roi import ROISelection                             # noqa: E402
from shapely.geometry import box                              # noqa: E402
from ui.map import suitability_rgba                           # noqa: E402

CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    CHECKS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}", flush=True)


# --- the same 3 x 3 km Nile Delta ROI the verifier uses (cached layers) ----- #
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


def main() -> int:
    print("=" * 78)
    print("Phase 8 UI SMOKE TEST -- real result object, real Streamlit runtime")
    print("=" * 78)

    print("\n1. run the engine (cached external layers)")
    t0 = time.perf_counter()
    execution = run_crop_suitability(
        AnalysisContext(roi=make_roi()), parse_query("Can I grow cotton here?"))
    print(f"   engine: {execution.status.value} in {time.perf_counter() - t0:.1f} s")
    check(execution.ok, "the engine produced a result")
    if not execution.ok:
        print(f"   message: {execution.message}")
        return 1
    result = execution.result

    print("\n2. render the panel inside a real Streamlit runtime")
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
from analyses.crop_suitability import run_crop_suitability
from analyses import AnalysisContext
from core.router import parse_query
from core.roi import ROISelection
from shapely.geometry import box
from rasterio import Affine
from ui.components import render_crop_suitability

TEN_M = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3462300.0)
C0, R0, C1, R1 = 1024 - 150, 1024 - 150, 1024 + 150, 1024 + 150
X0, Y0 = TEN_M * (C0, R1)
X1, Y1 = TEN_M * (C1, R0)
ROI_BOX = box(min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1))
roi = ROISelection(is_valid=True, intersects_raster=True,
                   area_m2=float(ROI_BOX.area), raster_crs="EPSG:32636",
                   geometry_raster_crs=ROI_BOX, geometry_type="Polygon",
                   num_parts=1)
ex = run_crop_suitability(AnalysisContext(roi=roi), parse_query("Can I grow cotton here?"))
render_crop_suitability(ex.result)
'''
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(script, default_timeout=900).run()
    if at.exception:
        for e in at.exception:
            print(f"   !! {e.value}")
    check(not at.exception, "render_crop_suitability ran without raising")
    def all_text(at) -> str:
        """Every rendered element -- metrics and callouts live outside markdown."""
        parts: list[str] = []
        for attr in ("title", "header", "subheader", "markdown", "caption",
                     "text", "warning", "error", "info", "success"):
            for el in getattr(at, attr, []):
                parts.append(str(getattr(el, "value", "")))
        for el in getattr(at, "metric", []):
            parts.append(f"{getattr(el, 'label', '')} {getattr(el, 'value', '')}")
        for el in getattr(at, "dataframe", []):
            parts.append(str(getattr(el, "value", "")))
        return " ".join(parts)

    text = all_text(at)
    low = text.lower()
    check("Screening class" in text, "the screening class is rendered as a metric")
    check("Score" in text and "Confidence" in text,
          "score and confidence are rendered as metrics")
    check("experimental" in low, "the experimental framing is kept in the panel")
    check("Insufficient data" in text and "Unsuitable" in text,
          "more than one screening class name appears (legend / fractions)")
    check("Computed limiting factors" in text,
          "computed limiting factors are their own section")
    check("Membership" in text, "per-factor memberships are shown")
    check("not assessed" in low, "unassessed constraints are stated as such")
    check("salinity" in low, "salinity is named as not assessed")
    check("irrigation" in low, "irrigation availability is named as not assessed")
    check("growing-season" in low, "water is reported for the growing season")
    # AppTest does not expose container LABELS, only their children, so the
    # explainers are identified by the text they contain.
    check("weighted mean" in text and "Critical-factor veto" in text,
          "the 'Why?' explainer states the model and the gates")
    check("EXPERIMENTAL" in text,
          "the threshold-provenance table is inside the explainer")
    check("Datasets" in text, "the 'Data & methodology' section lists datasets")
    check("Climatological" in text, "the temporal basis is stated")

    print("\n3. the map overlay helper")
    primary = result.scenarios.get("rainfed") or next(iter(result.scenarios.values()))
    codes = np.asarray(primary.suitability_raster)
    check(codes.ndim == 2, f"the suitability raster is 2-D ({codes.shape})")
    rgba = suitability_rgba(codes.astype("float32"))
    check(rgba.shape == codes.shape + (4,), f"RGBA shape {rgba.shape}")
    check(set(np.unique(rgba[:, :, 3]).tolist()) <= {0, 95, 195, 205, 215},
          "opacity is per-class (215/205/195/205/95), nothing invented")
    check(int(rgba[..., 3][np.isfinite(codes) & (codes == 0)].max() or 0) in (0, 95)
          if np.isfinite(codes).any() else True,
          "'Insufficient data' cells are drawn faint, never as Unsuitable")

    web = reproject_array(codes.astype("float32"),
                          src_transform=Affine(*tuple(primary.raster_transform)[:6]),
                          src_crs=CRS.from_user_input(str(primary.raster_crs)),
                          dst_crs=WEB_MERCATOR,
                          resampling=Resampling.nearest, max_pixels=2_000_000)
    check("leaflet_bounds" in web.to_dict(), "the web copy carries leaflet bounds")
    web_codes = np.rint(web.array).astype("int16")
    check(set(np.unique(web_codes[np.asarray(web.mask)]).tolist())
          <= {0, 1, 2, 3, 4},
          "nearest-neighbour reprojection introduced no new class codes")

    print("\n" + "=" * 78)
    passed = sum(1 for ok, _ in CHECKS if ok)
    failed = len(CHECKS) - passed
    print(f"SMOKE RESULT: {passed} passed, {failed} failed")
    for ok, msg in CHECKS:
        if not ok:
            print(f"  FAILED: {msg}")
    print("=" * 78)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
