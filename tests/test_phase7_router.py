"""Phase 7 -- router, registry and structured-result tests.

The router is tested with SYNTHETIC data only: no internet, no real GeoTIFF, no
Streamlit. The point of these tests is language behaviour and plumbing -- the
numerical correctness of the NDVI engine is Phase 6's job and is re-proved by
the end-to-end test in `tests/test_phase7_end_to_end.py`.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from analyses import (
    REGISTRY,
    AnalysisContext,
    AnalysisExecution,
    Intent,
    NdviContext,
    Status,
    available_specs,
    get_spec,
    planned_specs,
    route,
    suggestions,
)
from analyses import ndvi as ndvi_engine
from core.roi import ROISelection
from core.router import (
    CONFIDENCE_THRESHOLD,
    PLANNED_INTENTS,
    Intent as RouterIntent,
    normalize,
    parse_query,
)
from core.statistics import NO_PIXELS_MESSAGE, NO_VALID_MESSAGE, calculate_roi_ndvi_stats

REPO_ROOT = Path(__file__).resolve().parent.parent

# A hand-checkable 10 x 10 grid: value = index / 100  (0.00 .. 0.99)
GRID = (np.arange(100, dtype="float32") / 100.0).reshape(10, 10)
TEN_M = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
UTM36 = CRS.from_epsg(32636)
FULL_MASK = np.ones((10, 10), dtype=bool)


def box_roi(col0: float, row0: float, col1: float, row1: float) -> box:
    """Pixel corner coordinates -> a CRS box in the raster CRS."""
    x0, y0 = TEN_M * (col0, row1)
    x1, y1 = TEN_M * (col1, row0)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def make_roi(geom=None) -> ROISelection:
    geom = box_roi(1, 1, 6, 6) if geom is None else geom
    return ROISelection(
        is_valid=True, intersects_raster=True, area_m2=float(geom.area),
        raster_crs="EPSG:32636", geometry_raster_crs=geom,
        geometry_type="Polygon", num_parts=1,
    )


def make_context(roi=None, array=GRID, mask=FULL_MASK, confirmed=True):
    return AnalysisContext(
        roi=roi,
        ndvi=NdviContext(array=array, mask=mask, crs=UTM36, transform=TEN_M,
                         bands={"red": {"index": 3}, "nir": {"index": 4}},
                         source_label="synthetic.tif"),
        ndvi_confirmed=confirmed,
    )


# --------------------------------------------------------------------------- #
# 1-6. supported NDVI queries
# --------------------------------------------------------------------------- #
def test_exact_supported_ndvi_query():
    parsed = parse_query("What is the NDVI of this area?")
    assert parsed.intent is RouterIntent.NDVI_ROI_STATS
    assert parsed.confidence >= CONFIDENCE_THRESHOLD
    assert parsed.is_actionable
    assert parsed.required_context == ("roi", "ndvi_confirmed")


@pytest.mark.parametrize("query", [
    "What is the NDVI of this area?",
    "Calculate the vegetation index here.",
    "Show vegetation health in this area.",
    "Analyze the vegetation in this selected area.",
    "Show me the NDVI statistics.",
    "what is the ndvi here",
    "ndvi stats please",
    "measure the vegetation in this region",
    "calculate the mean NDVI of the selected polygon",
])
def test_ndvi_phrasings(query):
    parsed = parse_query(query)
    assert parsed.intent is RouterIntent.NDVI_ROI_STATS, query
    assert parsed.matched, "every accepted intent must be explainable"


def test_case_insensitive_input():
    assert parse_query("WHAT IS THE NDVI OF THIS AREA?").intent is RouterIntent.NDVI_ROI_STATS
    assert parse_query("What Is The NdVi Here").intent is RouterIntent.NDVI_ROI_STATS


def test_extra_whitespace_and_punctuation():
    messy = "   \t What   is   the  NDVI   of this area???  \n"
    parsed = parse_query(messy)
    assert parsed.intent is RouterIntent.NDVI_ROI_STATS
    assert parsed.normalized_query == "what is the ndvi of this area"


def test_vegetation_index_phrasing():
    parsed = parse_query("Calculate the vegetation index here")
    assert parsed.intent is RouterIntent.NDVI_ROI_STATS
    assert "ndvi" in parsed.matched or "vegetation index" in parsed.matched


def test_vegetation_health_phrasing():
    parsed = parse_query("Show vegetation health in this area")
    assert parsed.intent is RouterIntent.NDVI_ROI_STATS
    assert "vegetation health" in parsed.matched


# --------------------------------------------------------------------------- #
# 7-10. unsupported / ambiguous
# --------------------------------------------------------------------------- #
def test_crop_query_is_understood_and_now_executable():
    """Phase 8: cotton graduated from 'understood but unbuildable' to a real engine.

    The parser changed by one line (CROP_SUITABILITY left PLANNED_INTENTS); the
    routing logic did not. The engine itself is exercised with synthetic context
    in tests/test_phase8_router.py so this test needs no network.
    """
    parsed = parse_query("Can I grow cotton here?")
    assert parsed.intent is RouterIntent.CROP_SUITABILITY
    assert not parsed.is_planned
    assert parsed.is_actionable
    assert get_spec(parsed.intent).available


def test_unsupported_flood_query():
    parsed = parse_query("Show flood areas.")
    assert parsed.intent is RouterIntent.FLOOD_CHANGE
    execution = route("Show flood areas.", make_context(roi=make_roi()))
    assert execution.status is Status.UNSUPPORTED
    assert "flood" in execution.message.lower()
    assert execution.result is None


def test_unsupported_weather_query():
    parsed = parse_query("What is the weather here?")
    assert parsed.intent is RouterIntent.UNKNOWN
    execution = route("What is the weather here?", make_context(roi=make_roi()))
    assert execution.status is Status.UNKNOWN
    assert execution.result is None
    assert "NDVI" in execution.message          # points at what IS available


@pytest.mark.parametrize("query", [
    "Tell me about this area.",
    "show me the data",
    "hello",
    "",
])
def test_ambiguous_queries_stay_unknown(query):
    parsed = parse_query(query)
    assert parsed.intent is RouterIntent.UNKNOWN, query
    assert not parsed.is_actionable
    execution = route(query, make_context(roi=make_roi()))
    assert execution.status is Status.UNKNOWN
    assert execution.result is None


def test_vegetation_change_is_not_silently_ndvi():
    """A change request must not be answered with a single-date statistic.

    Phase 10 gave VEGETATION_CHANGE a real engine, so the invariant is now
    sharper than "unsupported": the request must reach a TEMPORAL engine and,
    with no dates chosen, must ask for them. What must never happen -- in any
    phase -- is a single-date NDVI mean being returned for a change question.
    """
    parsed = parse_query("How has the vegetation changed since 2023?")
    assert parsed.intent is RouterIntent.VEGETATION_CHANGE
    execution = route("How has the vegetation changed since 2023?",
                      make_context(roi=make_roi()))
    assert execution.intent is Intent.VEGETATION_CHANGE
    assert execution.intent is not Intent.NDVI_ROI_STATS
    assert execution.status is Status.NEEDS_TWO_DATES
    assert execution.result is None


# --------------------------------------------------------------------------- #
# 11-13. context validation
# --------------------------------------------------------------------------- #
def test_no_roi_context():
    execution = route("What is the NDVI of this area?", make_context(roi=None))
    assert execution.intent is Intent.NDVI_ROI_STATS      # understood ...
    assert execution.status is Status.NEEDS_ROI           # ... but cannot run
    assert execution.result is None
    assert execution.message == "Please select an area on the map first."


def test_unusable_roi_is_treated_as_no_roi():
    broken = ROISelection(is_valid=False, intersects_raster=False, area_m2=0.0,
                          raster_crs="EPSG:32636")
    execution = route("What is the NDVI of this area?", make_context(roi=broken))
    assert execution.status is Status.NEEDS_ROI


def test_ndvi_not_confirmed():
    execution = route("What is the NDVI of this area?",
                      make_context(roi=make_roi(), confirmed=False))
    assert execution.status is Status.NEEDS_NDVI_CONFIRMATION
    assert execution.result is None
    assert "confirm" in execution.message.lower()


def test_ndvi_confirmed_but_engine_input_missing():
    context = AnalysisContext(roi=make_roi(), ndvi=None, ndvi_confirmed=True)
    execution = route("What is the NDVI of this area?", context)
    assert execution.status is Status.NEEDS_NDVI_CONFIRMATION
    assert execution.result is None


def test_empty_roi_with_no_pixels():
    far = box(0.0, 0.0, 10.0, 10.0)          # nowhere near the grid
    execution = route("What is the NDVI of this area?", make_context(roi=make_roi(far)))
    assert execution.status is Status.NO_VALID_PIXELS
    assert execution.message == NO_PIXELS_MESSAGE
    assert execution.result is not None
    assert execution.result.valid_pixels == 0


def test_roi_without_valid_pixels():
    data = GRID.copy()
    data[:, :] = np.nan
    execution = route("What is the NDVI of this area?",
                      make_context(roi=make_roi(), array=data, mask=np.zeros((10, 10), bool)))
    assert execution.status is Status.NO_VALID_PIXELS
    assert execution.message == NO_VALID_MESSAGE
    assert execution.result.pixels_inside_roi > 0     # geometry counted ...
    assert execution.result.valid_pixels == 0         # ... data absent
    assert execution.result.stats is None             # never zeros


def test_successful_execution_reports_observed_ndvi_only():
    execution = route("What is the NDVI of this area?", make_context(roi=make_roi()))
    assert execution.status is Status.OK
    assert execution.ok
    assert execution.result.valid_pixels == 25
    assert "mean NDVI" in execution.message
    low = execution.message.lower()
    assert not any(w in low for w in ("healthy", "suitab", "yield", "disease"))
    assert any("not a crop-health" in w for w in execution.warnings)


# --------------------------------------------------------------------------- #
# 14-15. registry
# --------------------------------------------------------------------------- #
def test_registry_lookup():
    spec = get_spec(Intent.NDVI_ROI_STATS)
    assert spec is not None and spec.available
    assert spec.handler is not None
    assert spec.requires == ("roi", "ndvi_confirmed")
    assert not get_spec(Intent.UNKNOWN)


def test_planned_intents_have_no_handler():
    for intent in PLANNED_INTENTS:
        spec = get_spec(intent)
        assert spec is not None, f"{intent} must be declared"
        assert spec.handler is None, f"{intent} must NOT be executable yet"
        assert not spec.available
        assert spec.unavailable_message.endswith("not available yet.")
    assert set(planned_specs()) == set(PLANNED_INTENTS)
    # Phase 8: crop suitability graduated from planned to available; Phase 9:
    # the multi-condition spatial query did the same; Phase 10: vegetation
    # change (and the two new temporal intents) graduated too. The remaining
    # planned intent is the one still without an engine.
    # Phase 11: NDWI graduated to an implemented intent (it shares the generic
    # index engine with NDVI), and TEMPORAL_NDWI was ADDED to the planned set,
    # so that "compare NDWI before and after" is refused by name instead of
    # being answered with an NDVI difference.
    # Phase 12: multi-condition composition graduated; it composes existing
    # engines and adds no new index, so the planned set is untouched.
    assert set(planned_specs()) == {Intent.FLOOD_CHANGE, Intent.TEMPORAL_NDWI}
    # Phase 12: composed multi-condition queries graduated to an implemented
    # intent. Nothing left the planned set and nothing else graduated: the
    # census below is exhaustive on purpose -- a new engine must be declared
    # here, in the open, and not slip in silently.
    assert set(available_specs()) == {Intent.NDVI_ROI_STATS,
                                      Intent.CROP_SUITABILITY,
                                      Intent.SPATIAL_QUERY,
                                      Intent.NDVI_CHANGE_ROI,
                                      Intent.TEMPORAL_COMPARISON,
                                      Intent.VEGETATION_CHANGE,
                                      Intent.NDWI_ROI_STATS,
                                      Intent.MULTI_CONDITION}


def test_suggestions_come_from_the_registry():
    available = suggestions()
    assert available
    for example in available:
        assert any(example in spec.example_queries for spec in available_specs().values())


def test_unknown_intent_is_not_registered():
    execution = route("hello there", make_context(roi=make_roi()))
    assert execution.intent is Intent.UNKNOWN
    assert execution.status is Status.UNKNOWN
    assert get_spec(Intent.UNKNOWN) is None
    assert execution.provenance.get("engine") is None


# --------------------------------------------------------------------------- #
# 16. the router performs no raster mathematics
# --------------------------------------------------------------------------- #
FORBIDDEN = {"numpy", "rasterio", "shapely", "pyproj", "geopandas", "matplotlib",
             "pandas", "xarray", "rioxarray", "cv2", "skimage"}


def test_router_module_imports_no_raster_libraries():
    source = (REPO_ROOT / "core" / "router.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & FORBIDDEN), f"core/router.py imports {imported & FORBIDDEN}"
    assert "core.statistics" not in source and "calculate_roi_ndvi_stats" not in source


def test_importing_the_router_does_not_load_raster_libraries():
    """In a CLEAN interpreter, importing the router pulls in no raster stack.

    It has to be a subprocess: pytest has already imported numpy by the time
    this file runs, so an in-process sys.modules check would prove nothing.
    """
    code = (
        "import sys; sys.path.insert(0, '.');\n"
        "import core.router\n"
        "loaded = [m for m in ('numpy', 'rasterio', 'shapely', 'pyproj', 'geopandas')\n"
        "          if m in sys.modules]\n"
        "print('LOADED:', loaded)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOADED: []" in result.stdout


def test_the_router_only_routes(monkeypatch):
    """The engine is called with the NATIVE array; the router adds no numbers."""
    seen = {}

    def fake_engine(array, geometry, transform, valid_mask, **kwargs):
        seen.update(array=array, geometry=geometry, transform=transform,
                    mask=valid_mask, kwargs=kwargs)
        return calculate_roi_ndvi_stats(array, geometry, transform, valid_mask, **kwargs)

    monkeypatch.setattr(ndvi_engine, "calculate_roi_ndvi_stats", fake_engine)
    context = make_context(roi=make_roi())
    execution = route("What is the NDVI of this area?", context)

    assert seen["array"] is GRID                       # the same native object
    assert seen["transform"] is TEN_M
    assert seen["mask"] is FULL_MASK
    assert seen["geometry"] is context.roi.geometry_raster_crs
    assert execution.status is Status.OK
    assert execution.provenance["engine"].startswith("core.statistics")


# --------------------------------------------------------------------------- #
# 17. structured result
# --------------------------------------------------------------------------- #
def test_structured_result_correctness():
    execution = route("What is the NDVI of this area?", make_context(roi=make_roi()))
    assert isinstance(execution, AnalysisExecution)

    payload = execution.to_dict()
    for key in ("intent", "status", "query", "normalized_query", "confidence",
                "explanation", "matched", "message", "warnings", "provenance",
                "result"):
        assert key in payload
    assert payload["intent"] == "NDVI_ROI_STATS"
    assert payload["status"] == "OK"
    assert payload["matched"]
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["result"]["valid_pixels"] == 25
    assert payload["provenance"]["engine"] == "core.statistics.calculate_roi_ndvi_stats (Phase 6)"
    assert payload["provenance"]["router"]["intent"] == "NDVI_ROI_STATS"
    assert payload["provenance"]["crs"] == "EPSG:32636"
    assert payload["provenance"]["native_resolution_m"] == [10.0, 10.0]
    # every number in the payload is JSON-serialisable (no numpy leakage)
    import json

    json.dumps(payload)


def test_failed_execution_carries_no_result():
    context = make_context(roi=None)
    payload = route("What is the NDVI of this area?", context).to_dict()
    assert payload["result"] is None
    assert payload["status"] == "NEEDS_ROI"
    assert payload["provenance"]["engine"] is None


def test_normalize_is_stable():
    assert normalize("  What   IS the NDVI?? ") == "what is the ndvi"
    assert normalize("") == ""
    assert normalize(None) == ""
    assert normalize("Calculate the vegetation index here.") == "calculate ndvi here"


# --------------------------------------------------------------------------- #
# 18. the architecture is extensible: a future engine is a registry entry
# --------------------------------------------------------------------------- #
def test_a_planned_intent_becomes_live_by_registering_a_handler(monkeypatch):
    """Demonstrates the three-step extension, end to end.

    Step 1: write the analysis function (here: a stand-in that returns an
            AnalysisExecution like a real engine would).
    Step 2: register it against the intent.
    Step 3: the query patterns already exist.

    Nothing else changes: core/router.py, app.py and the UI are untouched, and
    the UI's suggestion line picks the example queries straight from REGISTRY.
    """
    calls = {}

    def run_crop_suitability(context: AnalysisContext, query: QueryIntent):
        calls["context"] = context
        calls["query"] = query
        return AnalysisExecution(
            intent=Intent.CROP_SUITABILITY, status=Status.OK, query=query.original_query,
            normalized_query=query.normalized_query, confidence=query.confidence,
            explanation=query.explanation, matched=query.matched,
            result={"suitability": "demonstration only"},
            message="Demonstration engine ran.",
        )

    spec = get_spec(Intent.CROP_SUITABILITY)
    monkeypatch.setattr(spec, "handler", run_crop_suitability)

    context = make_context(roi=make_roi())
    execution = route("Can I grow cotton here?", context)

    assert execution.status is Status.OK                      # it ran
    assert execution.result == {"suitability": "demonstration only"}
    assert calls["context"] is context                        # same data object
    assert calls["query"].intent is Intent.CROP_SUITABILITY
    assert spec.available                                     # now advertised
    assert "Can I grow cotton here?" in suggestions(limit=8)

    # ... and removing the handler puts it straight back to "not available yet"
    monkeypatch.setattr(spec, "handler", None)
    execution = route("Can I grow cotton here?", context)
    assert execution.status is Status.UNSUPPORTED
    assert execution.result is None
