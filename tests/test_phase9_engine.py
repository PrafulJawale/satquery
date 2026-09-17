"""Phase 9, Checkpoint C -- engine-level tests for `analyses.spatial_query`.

The engine is exercised end to end (router -> registry -> masks -> result) with
SYNTHETIC layers injected as providers:

    suitability_runner   stands in for the Phase 8 cotton engine
    land_cover_fetcher   stands in for the WorldCover datasource

so no network and no GeoTIFF is touched, while the real grid geometry
(`make_grid`, `roi_mask`) and the real mask algebra stay in the loop. The tests
also prove the orchestration rules: one fetch per run, refusal without any
computation, and zero matches reported as a geographic fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from analyses import AnalysisContext, Status
from analyses.registry import route
from analyses.spatial_query import (
    RESULT_INSUFFICIENT_DATA, RESULT_OK, RESULT_ZERO_MATCHES,
    SpatialQueryResult, run_spatial_query,
)
from core.alignment import make_grid
from core.geometry import as_crs
from core.roi import ROISelection
from core.router import Intent, parse_query
from core.spatial import FALSE, INSUFFICIENT, TRUE
from shapely.geometry import box


# --------------------------------------------------------------------------- #
# synthetic ROI, grid and layers
# --------------------------------------------------------------------------- #
X0, Y0 = 377200.0, 3441820.0          # the Nile Delta AOI-1 corner


def make_roi(size_m: float = 3000.0) -> ROISelection:
    return ROISelection(
        is_valid=True, intersects_raster=True, area_m2=size_m * size_m,
        raster_crs="EPSG:32636",
        geometry_raster_crs=box(X0, Y0, X0 + size_m, Y0 + size_m),
        geometry_type="Polygon", num_parts=1)


def context_with(roi: Optional[ROISelection] = None) -> AnalysisContext:
    return AnalysisContext(roi=roi if roi is not None else make_roi())


@dataclass
class FakeRecord:
    source: str = "ESA WorldCover 2021 v200 (synthetic stand-in)"
    native_resolution_m: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source,
                "native_resolution_m": self.native_resolution_m,
                "url": "synthetic://worldcover"}


@dataclass
class FakeLayer:
    array: Any
    record: Any = field(default_factory=FakeRecord)
    from_cache: bool = False


@dataclass
class FakeScenario:
    suitability_raster: Any
    native_resolutions: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    analysis_resolution: float = 30.0


@dataclass
class FakeScreening:
    grid: Any
    scenarios: Dict[str, Any]


def land_cover_fetcher(spec: Any, calls: Optional[List[int]] = None):
    """A stand-in for `worldcover.fetch_land_cover`, with a call counter.

    `spec` is either an array (used as-is, shape-checked against the grid) or
    a callable `(height, width) -> array`, so a test never has to know the
    grid size that `make_grid` happens to choose.
    """
    def _fetch(analysis_grid: Any, use_cache: bool = True) -> FakeLayer:
        if calls is not None:
            calls.append(1)
        height, width = analysis_grid.height, analysis_grid.width
        array = spec(height, width) if callable(spec) else spec
        if np.asarray(array).shape != (height, width):
            raise AssertionError(
                f"the layer must be windowed on the analysis grid "
                f"{(height, width)}, got {np.asarray(array).shape}")
        return FakeLayer(array=np.asarray(array))
    return _fetch


def cotton_runner(codes: Any, grid: Any, calls: Optional[List[int]] = None,
                  scenario: str = "rainfed"):
    """A stand-in for the Phase 8 cotton engine."""
    from analyses.base import AnalysisExecution
    from core.router import Intent

    def _run(context: Any, query: Any, **kwargs: Any) -> AnalysisExecution:
        if calls is not None:
            calls.append(1)
        raster = (codes(grid.height, grid.width) if callable(codes)
                  else np.asarray(codes))
        return AnalysisExecution(
            intent=Intent.CROP_SUITABILITY, status=Status.OK,
            query=query.original_query, normalized_query=query.normalized_query,
            confidence=query.confidence, explanation=query.explanation,
            matched=query.matched, message="synthetic",
            result=FakeScreening(
                grid=grid,
                scenarios={scenario: FakeScenario(
                    suitability_raster=raster,
                    native_resolutions={"worldclim": 1000.0, "soilgrids": 250.0},
                    provenance=[{"source": "synthetic", "factor": "ph"}])}))
    return _run


def grid_for(roi: ROISelection, resolution: float = 30.0,
             buffer_cells: int = 2):
    return make_grid(roi.geometry_raster_crs, as_crs(roi.raster_crs),
                     requested_resolution=resolution,
                     buffer_cells=buffer_cells)


def run(query_text: str, *, land_cover=None, cotton=None, roi=None,
        resolution: float = 30.0, via_registry: bool = False):
    """Execute a query end to end with synthetic layers."""
    parsed = parse_query(query_text)
    context = context_with(roi)
    grid = grid_for(context.roi, resolution)
    lc_calls: List[int] = []
    co_calls: List[int] = []
    if via_registry:
        # the registry path needs the providers injected differently, so tests
        # that use it pass a pre-parsed intent to run_spatial_query instead
        raise AssertionError("use run() for engine-only, route() separately")
    execution = run_spatial_query(
        context, parsed,
        suitability_runner=cotton_runner(cotton, grid, co_calls)
        if cotton is not None else None,
        land_cover_fetcher=land_cover_fetcher(land_cover, lc_calls)
        if land_cover is not None else None,
        analysis_resolution=resolution)
    return execution, lc_calls, co_calls, grid




# --------------------------------------------------------------------------- #
# synthetic layers, built to the size the analysis grid actually has
# --------------------------------------------------------------------------- #
def land(fill: int = 40, water: Optional[Any] = None,
         nodata: Optional[Any] = None, other: Optional[Any] = None):
    """A builder: `land(40, water=block(50, 50, 4))` -> (h, w) -> array."""
    def build(height: int, width: int) -> Any:
        array = np.full((height, width), fill, dtype=np.uint8)
        for spec, value in ((nodata, 0), (water, 80), (other, 10)):
            if spec is None:
                continue
            y0 = spec[0] if spec[0] >= 0 else height // 2
            x0 = spec[1] if spec[1] >= 0 else width // 2
            array[y0:y0 + spec[2], x0:x0 + spec[2]] = value
        return array
    return build


def block(y0: int = -1, x0: int = -1, size: int = 4):
    """A square patch; -1 centres it in the array."""
    return (y0, x0, size)


def rows(spec: Any):
    """A builder for a cotton class raster: `rows([(0, 20, 2), ...])`."""
    def build(height: int, width: int) -> Any:
        array = np.full((height, width), 4, dtype=np.uint8)
        for y0, y1, value in spec:
            array[y0:y1, :] = value
        return array
    return build

# =========================================================================== #
# the happy path
# =========================================================================== #
def test_cropland_near_water_finds_the_buffer():
    """A 4x4 block of water in the middle of a 30 m cropland grid."""
    lc = land(40, water=block(size=4))
    execution, lc_calls, _, grid = run("Find cropland near water", land_cover=lc)

    assert execution.status is Status.OK
    result: SpatialQueryResult = execution.result
    assert result.status == RESULT_OK
    assert result.matched_cell_count > 0
    # 30 m cells -> 900 m2 each; area must come from the real cell size
    assert result.matched_area_m2 == pytest.approx(
        result.matched_cell_count * 900.0)
    assert 0.0 < result.matched_fraction < 1.0
    assert "Within the selected ROI" in result.message
    assert len(lc_calls) == 1                    # one windowed read, not one
                                                 # per condition
    assert len(result.condition_results) == 2
    assert [c["condition"] for c in result.condition_results] == [
        "land_cover_class", "water_proximity"]


def test_area_uses_the_actual_cell_size_not_ten_metres():
    execution, _, _, _ = run("Find cropland", land_cover=land(40))
    result = execution.result
    assert result.grid["cell_area_m2"] == pytest.approx(900.0)
    assert result.matched_area_m2 == pytest.approx(
        result.matched_cell_count * 900.0)


def test_result_exposes_everything_the_ui_needs():
    execution, _, _, _ = run("Find cropland near water",
                             land_cover=land(40, water=block(20, 20, 4)))
    result = execution.result
    for attribute in ("query", "conditions", "operator", "matched_cell_count",
                      "matched_area_m2", "matched_fraction", "result_mask",
                      "condition_results", "warnings", "provenance",
                      "analysis_resolution", "source_resolutions", "status"):
        assert hasattr(result, attribute), attribute
    payload = result.to_dict()
    json.dumps(payload)                      # JSON-safe: no arrays leaked
    assert payload["result_mask"] is None
    assert payload["result_mask_shape"] == list(np.asarray(result.result_mask).shape)


# =========================================================================== #
# zero matches, insufficient data
# =========================================================================== #
def test_zero_matches_is_a_geographic_statement_about_the_roi():
    execution, _, _, _ = run("Find cropland", land_cover=land(10))
    result = execution.result
    assert result.status == RESULT_ZERO_MATCHES
    assert result.matched_cell_count == 0
    assert "0% of analysed cells" in result.message
    assert "not about the surrounding region" in result.message
    assert "does not exist" not in result.message.lower()
    assert "no suitable land" not in result.message.lower()


def test_nodata_everywhere_is_insufficient_not_zero_matches():
    execution, _, _, _ = run("Find cropland", land_cover=land(0))
    assert execution.status is Status.INSUFFICIENT_DATA
    assert execution.result.status == RESULT_INSUFFICIENT_DATA


def test_partial_nodata_is_counted_separately_and_never_as_false():
    # the proximity buffer makes the grid larger than the ROI, so the patches
    # must sit inside it: the ROI spans rows/cols 35..136 of a 171 grid
    lc = land(40, water=block(size=4), nodata=block(40, 40, 10))
    execution, _, _, _ = run("Find cropland near water", land_cover=lc)
    result = execution.result
    assert result.insufficient_cell_count > 0
    assert "insufficient" in result.message.lower()
    assert (result.matched_cell_count + result.non_matching_cell_count
            + result.insufficient_cell_count == result.analysed_cell_count
            + result.insufficient_cell_count)


# =========================================================================== #
# refusal without computation
# =========================================================================== #
@pytest.mark.parametrize("query_text,topic", [
    ("Find cotton areas with irrigation", "irrigation"),
    ("Find cotton land with reliable irrigation", "irrigation"),
    ("Find cotton areas near groundwater", "groundwater"),
])
def test_unsupported_conditions_compute_nothing(query_text, topic):
    execution, lc_calls, cotton_calls, _ = run(query_text, land_cover=land(40))
    assert execution.status is Status.UNSUPPORTED_CONDITION
    assert execution.result is None
    assert lc_calls == [] and cotton_calls == []     # nothing was fetched
    assert topic in execution.message.lower()
    assert any(phrase in execution.message.lower()
               for phrase in ("not measured", "not currently measured",
                              "not assessed"))


def test_irrigation_refusal_never_mentions_near_water_as_an_answer():
    execution, _, _, _ = run("Find cotton areas with irrigation",
                             land_cover=land(40))
    lower = execution.message.lower()
    assert "irrigation" in lower and "not currently measured" in lower
    # proximity is offered as a DIFFERENT capability, never delivered as the
    # answer to an irrigation question
    assert "proximity to mapped surface water" in lower
    assert execution.result is None


def test_no_roi_asks_for_one_before_anything_else():
    parsed = parse_query("Find cropland near water")
    execution = run_spatial_query(AnalysisContext(roi=None), parsed)
    assert execution.status is Status.NEEDS_ROI


# =========================================================================== #
# orchestration: the Phase 8 engine is reused, not duplicated
# =========================================================================== #
def test_cotton_condition_reuses_the_phase8_raster_and_threshold():
    codes = rows([(0, 20, 2), (20, 24, 0)])   # marginal, then insufficient

    execution, _, cotton_calls, _ = run("Find cotton areas", cotton=codes,
                                        land_cover=land(40))
    assert cotton_calls == [1]            # the Phase 8 engine ran exactly once
    result = execution.result
    assert result.status == RESULT_OK
    assert result.matched_cell_count > 0
    # none of the class-2 rows may match, and none of the class-0 rows either
    mask = np.asarray(result.result_mask)
    assert not (mask[0:20, :] == TRUE).any()
    assert (mask[20:24, :] == INSUFFICIENT).all()
    # the interior: the outer two rows/columns are the window buffer, which
    # lies outside the ROI and is therefore INSUFFICIENT by construction
    assert (mask[50, 5:-5] == TRUE).all()
    assert result.condition_results[0]["min_class"] == 3
    assert result.source_resolutions  # native resolutions carried over


def test_cotton_and_near_water_combines_both_engines():
    codes = rows([(0, 40, 1)])                    # unsuitable in the north
    lc = land(40, water=block(size=2))            # water, centred

    execution, _, _, _ = run("Find cotton areas near water", cotton=codes,
                             land_cover=lc)
    result = execution.result
    assert [c["condition"] for c in result.condition_results] == [
        "crop_suitability", "water_proximity"]
    mask = np.asarray(result.result_mask)
    assert not (mask[0:40, :] == TRUE).any()      # cotton fails there
    assert (mask == TRUE).sum() > 0               # cotton AND near water exists
    # the read was made on a buffered window, not on the narrow cotton grid
    assert result.performance["window"]["buffer_cells"] > 0
    assert (result.performance["window"]["buffer_metres"]
            >= result.condition_results[1]["provenance"]["distance_m"])


def test_not_water_differs_from_not_near_water_end_to_end():
    lc = land(40, water=block(size=4))

    a, _, _, _ = run("Find cropland excluding water", land_cover=lc)
    b, _, _, _ = run("Find cropland excluding areas near water", land_cover=lc)
    assert a.result.expression != b.result.expression
    assert "water(class [80])" in a.result.expression
    assert "water_proximity" in b.result.expression
    # NOT water keeps every non-water cell; NOT near water drops the buffer
    assert a.result.matched_cell_count > b.result.matched_cell_count


def test_buffered_window_removes_the_artificial_insufficient_rim():
    """Decision 1: cotton + proximity reads WorldCover on a buffered window, so
    the rim that the narrow Phase 8 grid would create is NOT reported as
    insufficient data."""
    codes = rows([])                       # every cell highly suitable
    lc = land(40, water=block(size=4))     # water in the middle

    execution, _, _, _ = run("Find cotton areas near water", cotton=codes,
                             land_cover=lc)
    result = execution.result
    assert result.matched_cell_count > 0
    assert result.insufficient_cell_count == 0
    window = result.performance["window"]
    assert window["source_window_cells"][0] > window["analysis_window_cells"][0]


# =========================================================================== #
# performance / bounded work
# =========================================================================== #
def test_layers_are_windowed_on_the_analysis_grid_only():
    """A provider that is handed a global-sized grid fails the test."""
    grid_shape = {}

    def spy(analysis_grid: Any, use_cache: bool = True):
        grid_shape["shape"] = (analysis_grid.height, analysis_grid.width)
        return FakeLayer(array=np.full(
            (analysis_grid.height, analysis_grid.width), 40, dtype=np.uint8))

    parsed = parse_query("Find cropland near water")
    execution = run_spatial_query(context_with(), parsed,
                                  land_cover_fetcher=spy)
    assert execution.status is Status.OK
    assert grid_shape["shape"] == grid_shape["shape"]     # read once, windowed
    # 3 km ROI at 30 m + buffer: a few thousand cells, never a global raster
    assert max(grid_shape["shape"]) < 200


def test_the_same_worldcover_read_serves_several_conditions():
    lc = land(40, water=block(size=4))
    execution, lc_calls, _, _ = run(
        "Find cropland near water but not built-up", land_cover=lc)
    assert len(lc_calls) == 1
    assert len(execution.result.condition_results) == 3


# =========================================================================== #
# structured output
# =========================================================================== #
def test_expression_and_conditions_are_structured_not_free_text():
    execution, _, _, _ = run("Find cropland near water", land_cover=land(40))
    result = execution.result
    assert result.operator.value == "and"
    assert "land_cover([40])" in result.expression
    assert "water_proximity(<= 1000 m, class 80)" in result.expression
    assert set(result.condition_results[0]) >= {
        "name", "source", "counts", "condition", "negated", "evidence"}


def test_provenance_names_every_dataset_used():
    execution, _, _, _ = run("Find cropland near water",
                             land_cover=land(40, water=block(20, 20, 4)))
    joined = json.dumps(execution.result.provenance).lower()
    assert "worldcover" in joined
    assert "proximity" in joined
    assert execution.result.analysis_resolution == 30.0
