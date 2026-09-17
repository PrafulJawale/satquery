"""Phase 8 -- the crop-suitability intent through the Phase 7 router.

No network and no real raster: these tests prove the PLUMBING.
    * cotton questions reach the CROP_SUITABILITY engine
    * any other crop is refused with the exact wording, not silently screened
    * the router itself was not rewritten -- only the registry changed
    * NDVI still routes exactly as it did in Phase 7
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
    AnalysisContext,
    AnalysisExecution,
    Intent,
    Status,
    available_specs,
    get_spec,
    planned_specs,
    route,
    suggestions,
)
from analyses.crop_suitability import (
    NO_CROP_MESSAGE,
    OTHER_CROP_MESSAGE,
    SCREENING_DISCLAIMER,
    detect_crop,
    run_crop_suitability,
)
from core.router import Intent as RouterIntent
from core.spatial_query import ConditionType
from core.router import PLANNED_INTENTS, parse_query
from core.roi import ROISelection

REPO_ROOT = Path(__file__).resolve().parent.parent

TEN_M = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
UTM36 = CRS.from_epsg(32636)


def _box_roi(col0: float, row0: float, col1: float, row1: float) -> box:
    x0, y0 = TEN_M * (col0, row1)
    x1, y1 = TEN_M * (col1, row0)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def make_roi(geom=None) -> ROISelection:
    geom = _box_roi(1, 1, 6, 6) if geom is None else geom
    return ROISelection(is_valid=True, intersects_raster=True,
                        area_m2=float(geom.area), raster_crs="EPSG:32636",
                        geometry_raster_crs=geom, geometry_type="Polygon",
                        num_parts=1)


def make_context(roi=None) -> AnalysisContext:
    return AnalysisContext(roi=roi)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query", [
    "Can I grow cotton here?",
    "Is cotton suitable here?",
    "Is this area suitable for cotton?",
    "can i grow cotton in this area",
    "Is this land suitable for growing cotton?",
])
def test_cotton_queries_route_to_crop_suitability(query):
    parsed = parse_query(query)
    assert parsed.intent is RouterIntent.CROP_SUITABILITY
    assert parsed.is_actionable
    assert not parsed.is_planned                      # no longer a planned intent
    spec = get_spec(parsed.intent)
    assert spec is not None and spec.available
    assert spec.requires == ("roi",)                  # NDVI is NOT required


@pytest.mark.parametrize("query", [
    "Where can I grow cotton in this region?",
    "Where can I grow cotton?",
])
def test_where_questions_are_phase9_spatial_selections(query):
    """APPROVED PHASE 9 CHANGE (Checkpoint B) -- routing only, not semantics.

    "Where ...?" asks for GEOGRAPHY, so Checkpoint B routes it to SPATIAL_QUERY
    ("Where can I grow cotton?" is listed there as a cotton-suitability
    example). This query therefore moved OUT of
    `test_cotton_queries_route_to_crop_suitability` above.

    The cotton SEMANTICS are unchanged: the condition still demands Phase 8
    class >= 3 under the rainfed scenario, evaluated by the same engine.
    """
    parsed = parse_query(query)
    assert parsed.intent is RouterIntent.SPATIAL_QUERY
    assert len(parsed.conditions) == 1
    condition = parsed.conditions[0]
    assert condition.condition_type is ConditionType.CROP_SUITABILITY
    assert condition.parameters["crop"] == "cotton"
    assert condition.parameters["min_class"] == 3
    assert condition.parameters["scenario"] == "rainfed"


def test_crop_suitability_is_registered_and_no_longer_planned():
    assert Intent.CROP_SUITABILITY in available_specs()
    assert Intent.CROP_SUITABILITY not in planned_specs()
    assert Intent.CROP_SUITABILITY not in PLANNED_INTENTS
    # Phase 10: VEGETATION_CHANGE graduated to an implemented intent (it shares
    # the temporal NDVI engine), so only flood/water change remained planned.
    # Phase 11: TEMPORAL_NDWI was added to the planned set -- NDWI exists for a
    # single date, and a two-date NDWI comparison must not be answered with an
    # NDVI comparison. Flood/water change is still planned for the same reason.
    # Phase 12: multi-condition composition is implemented. It introduces no
    # new index and no new change type, so FLOOD_CHANGE and TEMPORAL_NDWI stay
    # planned -- the system still refuses "did flooding happen?" by name.
    assert set(planned_specs()) == {Intent.FLOOD_CHANGE, Intent.TEMPORAL_NDWI}
    assert Intent.MULTI_CONDITION in available_specs()
    assert any("cotton" in q.lower() for q in suggestions(limit=8))


def test_ndvi_routing_is_unchanged_by_phase_8():
    parsed = parse_query("What is the NDVI of this area?")
    assert parsed.intent is RouterIntent.NDVI_ROI_STATS
    assert parsed.required_context == ("roi", "ndvi_confirmed")
    assert get_spec(Intent.NDVI_ROI_STATS).available


# --------------------------------------------------------------------------- #
# crop support policy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query,crop,other", [
    ("Can I grow cotton here?", "cotton", None),
    ("Is this area suitable for cotton?", "cotton", None),
    ("Can I grow rice here?", None, "rice"),
    ("Is wheat suitable here?", None, "wheat"),
    ("Can we grow sugarcane in this region?", None, "sugarcane"),
    ("which crop suits this land", None, None),
])
def test_detect_crop(query, crop, other):
    assert detect_crop(parse_query(query).normalized_query) == (crop, other)


@pytest.mark.parametrize("query", [
    "Can I grow rice here?",
    "Is wheat suitable here?",
    "Can I grow maize in this area?",
    "Is this land good for sugarcane?",
])
def test_unsupported_crop_is_refused_before_any_fetching(query):
    execution = run_crop_suitability(make_context(roi=make_roi()), parse_query(query))
    assert execution.status is Status.UNSUPPORTED_CROP
    assert execution.message == OTHER_CROP_MESSAGE
    assert execution.result is None                   # nothing was computed
    assert SCREENING_DISCLAIMER in execution.warnings


def test_crop_question_without_naming_a_crop_is_answered_with_the_supported_one():
    execution = run_crop_suitability(make_context(roi=make_roi()),
                                     parse_query("which crop suits this land"))
    assert execution.status is Status.UNSUPPORTED_CROP
    assert execution.message == NO_CROP_MESSAGE
    assert "cotton" in execution.message


def test_no_roi_still_asks_for_one_before_mentioning_crops():
    execution = run_crop_suitability(make_context(roi=None),
                                     parse_query("Can I grow cotton here?"))
    assert execution.status is Status.NEEDS_ROI
    assert execution.result is None


def test_unsupported_crop_through_the_router_end_to_end():
    execution = route("Can I grow rice here?", make_context(roi=make_roi()))
    assert execution.intent is Intent.CROP_SUITABILITY
    assert execution.status is Status.UNSUPPORTED_CROP
    assert "Only cotton suitability is currently supported." in execution.message


# --------------------------------------------------------------------------- #
# the router stays Streamlit-free and free of raster libraries
# --------------------------------------------------------------------------- #
def test_router_still_imports_no_raster_libraries():
    code = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); import core.router; "
         "bad = [m for m in ('numpy', 'rasterio', 'shapely', 'pyproj', 'streamlit') "
         "if m in sys.modules]; print('BAD:' + ','.join(bad) if bad else 'CLEAN')"],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert "CLEAN" in code.stdout, code.stdout + code.stderr


def test_the_router_file_was_not_rewritten_by_phase_8():
    src = (REPO_ROOT / "core" / "router.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for node in tree.body:                       # module-level bindings
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    for expected in ("parse_query", "normalize", "tokenize", "QueryIntent",
                     "Intent", "PLANNED_INTENTS", "REQUIRED_CONTEXT",
                     "CONFIDENCE_THRESHOLD", "supported_examples"):
        assert expected in names
    assert "CROP_SUITABILITY" in src          # vocabulary
    assert "crop" in src.lower()              # and only vocabulary
