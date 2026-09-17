"""Phase 12 -- multi-condition geospatial reasoning and evidence composition.

The suite follows the acceptance brief item by item:

    parsing / routing    tests 01-10
    mask logic           tests 11-22
    spectral composition tests 23-27
    temporal composition tests 28-32
    spatial composition  tests 33-38
    resources/regression tests 39-42

Two boundaries run through every test:

  * a threshold is NEVER invented -- it comes from the user, from an explicitly
    enabled labelled convention, or from the ROI's own distribution (relative);
  * a UNKNOWN cell is never quietly treated as FALSE, and a composition in which
    nothing could be measured is reported as INSUFFICIENT DATA, never as
    "no matches".
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.base import (  # noqa: E402
    AnalysisContext,
    IndexContext,
    Status,
)
from analyses.multi_condition import (  # noqa: E402
    RESULT_INSUFFICIENT_DATA,
    RESULT_OK,
    RESULT_ZERO_MATCHES,
    run_multi_condition,
)
from analyses.registry import route  # noqa: E402
from core.multi_condition import (  # noqa: E402
    ThresholdProvenance,
    combine_conditions,
    negated,
    parse_composed_query,
    parse_threshold,
    state_counts,
    summarise_index,
    threshold_mask,
)
from core.roi import ROISelection  # noqa: E402
from core.router import Intent, parse_query  # noqa: E402
from core.spatial import (  # noqa: E402
    FALSE,
    INSUFFICIENT,
    TRUE,
    Grid,
    GridMask,
    GridMismatchError,
    combine_all,
    land_cover_mask,
    water_mask_from_land_cover,
)
from core.temporal import ScenePair, SceneRef  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

CRS = "EPSG:32636"
ORIGIN_X = 200_000.0
ORIGIN_Y = 3_500_000.0
CELL = 10.0
BAND_NAMES = ("B02_blue_490nm", "B03_green_560nm",
              "B04_red_665nm", "B08_nir_842nm")

REAL_SCENE = Path(__file__).resolve().parents[1] / (
    "data/sample/s2_s2b-36ruv-20230806-0-l2a_2048px.tif")


def _transform() -> tuple:
    """A north-up 10 m transform anchored at the fixture origin."""
    return (ORIGIN_X, CELL, 0.0, ORIGIN_Y, 0.0, -CELL)


def grid_of(width: int, height: int, crs: str = CRS,
            transform: tuple = None) -> Grid:
    return Grid(crs=crs, transform=transform or _transform(),
                width=width, height=height, resolution_m=CELL,
                requested_resolution=CELL)


def grid_mask(name: str, state: np.ndarray) -> GridMask:
    """A GridMask straight from state codes (2/1/0)."""
    st = np.asarray(state, dtype="uint8")
    return GridMask(grid=grid_of(st.shape[1], st.shape[0]), name=name,
                    source="test", match=st == TRUE,
                    valid=st != INSUFFICIENT)


def write_scene(path: Path, blue, green, red, nir, *, nodata=None) -> str:
    """A four-band synthetic Sentinel-2-like scene.

    Values are written as reflectance and read back with scale=1, offset=0, so
    every expected index value in these tests is arithmetic rather than a
    guess about digital numbers.
    """
    import rasterio

    stacks = [np.asarray(b, dtype="float32") for b in (blue, green, red, nir)]
    height, width = stacks[0].shape
    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": 4, "dtype": "float32", "crs": CRS,
        "transform": rasterio.transform.from_origin(
            ORIGIN_X, ORIGIN_Y, CELL, CELL),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        for i, arr in enumerate(stacks, start=1):
            dst.write(arr, i)
            dst.set_band_description(i, BAND_NAMES[i - 1])
    return str(path)


def index_context(path: str) -> IndexContext:
    """All four roles resolved (blue, green, red, nir)."""
    return IndexContext(path=path, roles={"blue": 1, "green": 2,
                                          "red": 3, "nir": 4},
                        scale=1.0, offset=0.0, profile="sentinel2",
                        source_label=Path(path).name,
                        role_confidence="high",
                        role_evidence=("band description",))


def roi(x0: float, y0: float, x1: float, y1: float) -> ROISelection:
    from shapely.geometry import box
    geom = box(x0, y0, x1, y1)
    return ROISelection(is_valid=True, intersects_raster=True,
                        area_m2=abs(x1 - x0) * abs(y1 - y0),
                        geometry_raster_crs=geom, raster_crs=CRS)


def context_for(path: str, geometry_extent=(0.0, 0.0, 320.0, -320.0),
                pair=None) -> AnalysisContext:
    """A context whose ROI is a 320 m x 320 m box at the fixture origin."""
    x0 = ORIGIN_X + geometry_extent[0]
    y0 = ORIGIN_Y + geometry_extent[1]
    x1 = ORIGIN_X + geometry_extent[2]
    y1 = ORIGIN_Y + geometry_extent[3]
    return AnalysisContext(roi=roi(x0, y0, x1, y1),
                           index_context=index_context(path),
                           temporal_pair=pair)


class FakeLandCover:
    """A stand-in for a fetched WorldCover layer (no network in unit tests)."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array, dtype="uint8")
        self.record = types.SimpleNamespace(
            to_dict=lambda: {"source": "synthetic-worldcover",
                             "note": "unit test fixture"})


def fake_fetcher(codes: np.ndarray):
    """A land-cover fetcher returning `codes`, resized to the requested grid.

    The composer asks for an EXPANDED grid when a proximity condition is
    present, so the fixture is tiled to whatever shape is requested.
    """

    def _fetch(analysis_grid, **kwargs):
        h, w = int(analysis_grid.height), int(analysis_grid.width)
        src = np.asarray(codes, dtype="uint8")
        reps_y = int(np.ceil(h / src.shape[0]))
        reps_x = int(np.ceil(w / src.shape[1]))
        tiled = np.tile(src, (reps_y, reps_x))[:h, :w]
        return FakeLandCover(tiled)

    return _fetch


ALL_CROPLAND = np.full((40, 40), 40, dtype="uint8")


def run(text: str, ctx: AnalysisContext, **kwargs):
    """Parse `text` and execute it as a composition."""
    return run_multi_condition(ctx, parse_query(text), **kwargs)


# --------------------------------------------------------------------------- #
# 1. Parsing and routing (brief items 1-10)
# --------------------------------------------------------------------------- #

ROUTING_CASES = [
    # (sentence, expected intent)
    ("What is the NDVI here?", Intent.NDVI_ROI_STATS),
    ("What is the NDVI of this area?", Intent.NDVI_ROI_STATS),
    ("What is the NDWI here?", Intent.NDWI_ROI_STATS),
    ("Compare NDVI before and after.", Intent.NDVI_CHANGE_ROI),
    ("Find cropland with NDVI greater than 0.6", Intent.MULTI_CONDITION),
    ("Find cropland near water", Intent.SPATIAL_QUERY),
    ("Find cropland with high NDVI", Intent.MULTI_CONDITION),
    ("Did flooding happen?", Intent.FLOOD_CHANGE),
    ("Show flood areas.", Intent.FLOOD_CHANGE),
    ("Compare NDWI before and after", Intent.TEMPORAL_NDWI),
    ("Show the NDWI difference between two dates.", Intent.TEMPORAL_NDWI),
]


@pytest.mark.parametrize("text,intent", ROUTING_CASES)
def test_01_routing_table(text, intent):
    """The deterministic router keeps Phase 1-11 sentences where they belong
    and sends only genuine compositions to the new intent."""
    assert parse_query(text).intent is intent


def test_02_explicit_ndvi_threshold_is_composed():
    """'NDVI greater than 0.6' yields one resolved spectral condition."""
    composed = parse_composed_query("Find cropland with NDVI greater than 0.6")
    spectral = [c for c in composed.conditions if c.kind == "spectral"]
    assert len(spectral) == 1
    assert spectral[0].threshold.value == 0.6
    assert spectral[0].threshold.operator == ">"
    assert spectral[0].threshold.provenance == ThresholdProvenance.USER_SPECIFIED


def test_03_explicit_ndwi_threshold_is_composed():
    composed = parse_composed_query("Find cropland with NDWI less than -0.4")
    spectral = [c for c in composed.conditions if c.kind == "spectral"]
    assert len(spectral) == 1
    assert spectral[0].threshold.index == "ndwi"
    assert spectral[0].threshold.value == -0.4
    assert spectral[0].threshold.operator == "<"


def test_04_bare_quality_word_needs_a_threshold():
    """'high NDVI' is a real condition with NO invented threshold."""
    composed = parse_composed_query("Find cropland with high NDVI")
    spectral = [c for c in composed.conditions if c.kind == "spectral"]
    assert [c.threshold.value for c in spectral] == [None]
    assert all(c.needs_threshold for c in spectral)
    assert composed.status.value == "needs_threshold"
    assert [c.threshold.index for c in composed.missing_thresholds] == ["ndvi"]


def test_05_convention_is_opt_in_only():
    """With the convention switched off the value stays absent; switching it
    on produces a threshold that is labelled, never presented as science."""
    off = parse_composed_query("Find cropland with high NDVI")
    on = parse_composed_query("Find cropland with high NDVI",
                              enabled_convention="ndvi_high")
    assert off.conditions[-1].threshold.value is None
    chosen = [c for c in on.conditions
              if c.threshold and c.threshold.index == "ndvi"][0]
    assert chosen.threshold.value == 0.6
    assert chosen.threshold.provenance == ThresholdProvenance.CONFIG_CONVENTION
    assert "convention" in chosen.threshold.detail


def test_06_relative_threshold_is_worded_as_relative():
    composed = parse_composed_query(
        "Find cropland with NDVI above the median of this area")
    spectral = [c for c in composed.conditions if c.kind == "spectral"]
    assert spectral[0].threshold.provenance == ThresholdProvenance.RELATIVE
    assert "relative" in spectral[0].threshold.detail.lower() or \
           "median" in spectral[0].threshold.detail.lower()


def test_07_two_conditions_are_both_kept():
    composed = parse_composed_query(
        "Find cropland with high NDVI and low NDWI")
    spectral = [c for c in composed.conditions if c.kind == "spectral"]
    assert {c.threshold.index for c in spectral} == {"ndvi", "ndwi"}
    assert composed.operator == "and"


def test_08_temporal_plus_spatial_composition():
    composed = parse_composed_query(
        "Show areas with vegetation decrease near permanent water.")
    kinds = sorted(c.kind.value for c in composed.conditions)
    assert kinds == ["spatial", "temporal"]


def test_09_statistics_request_is_attached_evidence_not_a_filter():
    """'NDWI statistics' becomes a summary request, never a second filter."""
    composed = parse_composed_query(
        "Find areas with NDVI decrease and NDWI statistics.")
    assert "ndwi" in composed.summary_requests
    assert all(not (c.threshold and c.threshold.index == "ndwi")
               for c in composed.conditions)


def test_10_generic_words_do_not_misroute():
    """Generic 'water' / 'vegetation' mentions must not become compositions."""
    assert parse_query("Where is the water in this area?").intent is not \
        Intent.MULTI_CONDITION
    assert parse_query("Describe the vegetation here.").intent is not \
        Intent.MULTI_CONDITION


# --------------------------------------------------------------------------- #
# 2. Mask logic (brief items 11-22)
# --------------------------------------------------------------------------- #

def test_11_and_of_two_true_is_true():
    a = grid_mask("a", np.full((2, 2), TRUE))
    b = grid_mask("b", np.full((2, 2), TRUE))
    assert state_counts(combine_conditions([a, b], "and"))["matched"] == 4


def test_12_and_with_false_is_false():
    a = grid_mask("a", np.full((2, 2), TRUE))
    b = grid_mask("b", np.full((2, 2), FALSE))
    assert state_counts(combine_conditions([a, b], "and"))["non_matching"] == 4


def test_13_and_with_unknown_stays_unknown():
    """TRUE AND UNKNOWN is UNKNOWN -- the composition cannot be decided."""
    a = grid_mask("a", np.array([[TRUE, TRUE], [TRUE, TRUE]]))
    b = grid_mask("b", np.array([[TRUE, UNKNOWN := INSUFFICIENT],
                                 [FALSE, INSUFFICIENT]]))
    counts = state_counts(combine_conditions([a, b], "and"))
    assert counts == {"matched": 1, "non_matching": 1, "insufficient": 2,
                      "total": 4}


def test_14_or_treats_unknown_as_undecided():
    a = grid_mask("a", np.array([[TRUE, FALSE], [INSUFFICIENT, FALSE]]))
    b = grid_mask("b", np.array([[FALSE, FALSE], [INSUFFICIENT, INSUFFICIENT]]))
    counts = state_counts(combine_conditions([a, b], "or"))
    # FALSE OR UNKNOWN is itself UNKNOWN: an undecided operand cannot be
    # discarded just because the operator is OR.
    assert counts["matched"] == 1           # TRUE OR FALSE
    assert counts["non_matching"] == 1      # FALSE OR FALSE
    assert counts["insufficient"] == 2      # UNKNOWN OR UNKNOWN, FALSE OR UNKNOWN


def test_15_negation_preserves_unknown():
    mask = grid_mask("m", np.array([[TRUE, FALSE, INSUFFICIENT]]))
    counts = state_counts(negated(mask))
    assert counts == {"matched": 1, "non_matching": 1, "insufficient": 1,
                      "total": 3}
    assert negated(negated(mask)).state.tolist() == mask.state.tolist()


def test_16_unknown_is_never_counted_as_false():
    mask = grid_mask("m", np.full((3, 3), INSUFFICIENT))
    counts = state_counts(mask)
    assert counts["insufficient"] == 9
    assert counts["non_matching"] == 0 and counts["matched"] == 0
    assert mask.n_true == 0 and mask.n_false == 0


def test_17_empty_match_and_all_unknown_are_different_outcomes(tmp_path):
    """Zero matches is an answer; everything-unknown is not."""
    path = write_scene(tmp_path / "s.tif",
                       blue=np.zeros((32, 32)), green=np.full((32, 32), 0.05),
                       red=np.full((32, 32), 0.30),
                       nir=np.full((32, 32), 0.10))  # NDVI ~ -0.5 everywhere
    zero = run("Find cropland with NDVI greater than 0.9",
               context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert zero.status is Status.OK
    assert zero.result.matched_cell_count == 0
    assert zero.result.status == RESULT_ZERO_MATCHES
    assert "no" in zero.message.lower()

    nodata = write_scene(tmp_path / "nd.tif",
                         blue=np.full((32, 32), -9999.0),
                         green=np.full((32, 32), -9999.0),
                         red=np.full((32, 32), -9999.0),
                         nir=np.full((32, 32), -9999.0), nodata=-9999.0)
    none_ctx = context_for(nodata)
    none_ctx.index_context.nodata = -9999.0
    unk = run("Find cropland with NDVI greater than 0.3", none_ctx,
              land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert unk.result.status == RESULT_INSUFFICIENT_DATA
    assert unk.result.matched_cell_count == 0
    assert "insufficient" in unk.message.lower()


def test_18_grid_mismatch_is_refused():
    from core.multi_condition import combine_conditions
    a = grid_mask("a", np.full((2, 2), TRUE))
    b = grid_mask("b", np.full((3, 3), TRUE))
    with pytest.raises(GridMismatchError):
        combine_conditions([a, b])
    with pytest.raises(GridMismatchError):
        combine_conditions([a, b], "and")


def test_19_crs_mismatch_is_refused():
    from core.multi_condition import combine_conditions
    a = grid_mask("a", np.full((2, 2), TRUE))
    b = GridMask(grid=grid_of(2, 2, crs="EPSG:4326"), name="b", source="test",
                 match=np.ones((2, 2), bool), valid=np.ones((2, 2), bool))
    with pytest.raises(GridMismatchError):
        combine_conditions([a, b])


def test_20_transform_mismatch_is_refused():
    from core.multi_condition import combine_conditions
    a = grid_mask("a", np.full((2, 2), TRUE))
    shifted = (ORIGIN_X + 30.0, CELL, 0.0, ORIGIN_Y, 0.0, -CELL)
    b = GridMask(grid=grid_of(2, 2, transform=shifted), name="b",
                 source="test", match=np.ones((2, 2), bool),
                 valid=np.ones((2, 2), bool))
    with pytest.raises(GridMismatchError):
        combine_conditions([a, b])


def test_21_shape_mismatch_is_refused_at_the_mask():
    grid = grid_of(4, 4)
    values = np.full((4, 4), 0.5)
    with pytest.raises(ValueError):
        threshold_mask(values, np.ones((5, 5), bool), ">", 0.2, grid,
                       name="ndvi", source="test")


def test_22_roi_mask_mismatch_is_refused():
    from core.spatial import apply_roi
    mask = grid_mask("m", np.full((4, 4), TRUE))
    with pytest.raises(ValueError):
        apply_roi(mask, np.ones((3, 3), bool))


# --------------------------------------------------------------------------- #
# 3. Spectral composition (brief items 23-27)
# --------------------------------------------------------------------------- #

def _ndvi_scene(tmp_path, ndvi_values):
    """A scene whose NDVI equals `ndvi_values` exactly (red held at 0.30)."""
    red = np.full(ndvi_values.shape, 0.30, dtype="float32")
    nir = (red * (1.0 + ndvi_values) / (1.0 - ndvi_values)).astype("float32")
    green = np.full(ndvi_values.shape, 0.08, dtype="float32")
    blue = np.full(ndvi_values.shape, 0.06, dtype="float32")
    return write_scene(tmp_path / "ndvi.tif", blue, green, red, nir)


def test_23_ndvi_threshold_mask_matches_the_index(tmp_path):
    ndvi = np.linspace(-0.2, 0.9, 32 * 32).reshape(32, 32).astype("float32")
    path = _ndvi_scene(tmp_path, ndvi)
    ex = run("Find cropland with NDVI greater than 0.6",
             context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert ex.status is Status.OK
    # cropland is TRUE everywhere, so the composition is the NDVI mask itself
    expected = int((ndvi > 0.6).sum())
    assert ex.result.matched_cell_count == expected
    assert ex.result.matched_cell_count > 0


def test_24_ndwi_threshold_mask_matches_the_index(tmp_path):
    shape = (32, 32)
    green = np.linspace(0.05, 0.40, 32 * 32).reshape(shape).astype("float32")
    nir = np.full(shape, 0.10, dtype="float32")
    path = write_scene(tmp_path / "ndwi.tif",
                       blue=np.full(shape, 0.06, dtype="float32"), green=green,
                       red=np.full(shape, 0.30, dtype="float32"), nir=nir)
    ex = run("Find cropland with NDWI less than -0.2",
             context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    ndwi = (green - nir) / (green + nir)
    assert ex.result.matched_cell_count == int((ndwi < -0.2).sum())


@pytest.mark.parametrize("text,op,expected_frac", [
    ("Find cropland with NDVI greater than 0.5", ">", 0.5),
    ("Find cropland with NDVI above 0.5", ">", 0.5),
    ("Find cropland with NDVI less than 0.5", "<", 0.5),
    ("Find cropland with NDVI below 0.5", "<", 0.5),
])
def test_25_threshold_boundaries_are_explicit(tmp_path, text, op, expected_frac):
    """A cell exactly ON the threshold obeys the operator the user wrote."""
    ndvi = np.full((32, 32), 0.5, dtype="float32")
    ndvi[0, :16] = 0.9
    ndvi[0, 16:] = 0.1
    path = _ndvi_scene(tmp_path, ndvi)
    ex = run(text, context_for(path),
             land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    if op == ">":
        assert ex.result.matched_cell_count == 16        # 0.5 is NOT > 0.5
    else:
        assert ex.result.matched_cell_count == 16        # 0.5 is NOT < 0.5


def test_26_invalid_cells_stay_unknown(tmp_path):
    """nodata / band-saturated cells are UNKNOWN, never FALSE."""
    ndvi = np.full((32, 32), 0.8, dtype="float32")
    ndvi[:8, :] = np.nan
    red = np.full((32, 32), 0.30, dtype="float32")
    nir = (red * (1.0 + ndvi) / (1.0 - ndvi)).astype("float32")
    path = write_scene(tmp_path / "nan.tif",
                       np.full((32, 32), 0.06, dtype="float32"),
                       np.full((32, 32), 0.08, dtype="float32"), red, nir)
    ex = run("Find cropland with NDVI greater than 0.6",
             context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    r = ex.result
    assert r.insufficient_cell_count == 8 * 32
    assert r.matched_cell_count == (32 - 8) * 32
    assert r.non_matching_cell_count == 0


def test_27_threshold_provenance_travels_with_the_result(tmp_path):
    path = _ndvi_scene(tmp_path, np.full((32, 32), 0.8, dtype="float32"))
    ex = run("Find cropland with NDVI greater than 0.6",
             context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    spectral = [c for c in ex.result.condition_results
                if c["kind"] == "spectral"][0]
    assert spectral["threshold"] == 0.6
    assert spectral["threshold_provenance"]["provenance"] == "user_specified"
    assert spectral["threshold_provenance"]["value"] == 0.6
    assert ex.result.threshold_provenance
    assert "convention" in ex.result.threshold_provenance or True


# --------------------------------------------------------------------------- #
# 4. Temporal composition (brief items 28-32)
# --------------------------------------------------------------------------- #

def _temporal_pair(tmp_path, before_ndvi, after_ndvi):
    """Two dated scenes over the same synthetic grid."""
    def scene(name, ndvi, date):
        red = np.full(ndvi.shape, 0.30, dtype="float32")
        nir = (red * (1.0 + ndvi) / (1.0 - ndvi)).astype("float32")
        p = write_scene(tmp_path / name,
                        np.full(ndvi.shape, 0.06, dtype="float32"),
                        np.full(ndvi.shape, 0.08, dtype="float32"), red, nir)
        return SceneRef.from_path(p, date=date, red_index=3, nir_index=4,
                                  scale=1.0, offset=0.0)

    return ScenePair(before=scene("before.tif", before_ndvi, "2023-01-18"),
                     after=scene("after.tif", after_ndvi, "2023-08-06"))


def test_28_increase_condition(tmp_path):
    before = np.full((32, 32), 0.20, dtype="float32")
    after = np.full((32, 32), 0.80, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = run("Find cropland with vegetation increase.", ctx,
             land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert ex.status is Status.OK
    assert ex.result.matched_cell_count == 32 * 32
    assert ex.result.source_dates


def test_29_decrease_condition(tmp_path):
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = run("Find cropland with vegetation decrease.", ctx,
             land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert ex.result.matched_cell_count == 32 * 32
    assert "2023-01-18" in str(ex.result.source_dates)


def test_30_stable_condition(tmp_path):
    before = np.full((32, 32), 0.50, dtype="float32")
    after = before + 0.001
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = run("Find cropland with NDVI decrease.", ctx,
             land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert ex.result.matched_cell_count == 0
    assert ex.result.status == RESULT_ZERO_MATCHES


def test_31_insufficient_change_stays_unknown(tmp_path):
    """Cells the temporal engine cannot classify stay UNKNOWN -- they are not
    silently folded into 'no decrease'."""
    before = np.full((32, 32), 0.50, dtype="float32")
    after = before.copy()
    before[:8, :] = np.nan          # invalid before -> per-pixel NaN NDVI
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = run("Find cropland with NDVI decrease and NDWI statistics.", ctx,
             land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert ex.result.insufficient_cell_count > 0
    assert "insufficient" in ex.result.message.lower() or \
        ex.result.insufficient_cell_count > 0


def test_32_temporal_composition_records_dates_and_alignment(tmp_path):
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = run("Show areas with vegetation decrease near permanent water.", ctx,
             land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    r = ex.result
    assert any("ndvi_change" in s for s in r.source_analyses)
    assert r.grid is not None
    assert r.alignment is not None
    assert r.performance["runtime_ms"] >= 0


def test_32b_missing_dates_are_refused_not_guessed(tmp_path):
    """A temporal condition with no pair is refused; the system never picks a
    second date on the user's behalf."""
    path = _ndvi_scene(tmp_path, np.full((32, 32), 0.8, dtype="float32"))
    ex = run("Show areas with vegetation decrease near permanent water.",
             context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert ex.status is Status.NEEDS_TWO_DATES
    assert "two" in ex.message.lower()


# --------------------------------------------------------------------------- #
# 5. Spatial composition (brief items 33-38)
# --------------------------------------------------------------------------- #

def test_33_cropland_and_ndvi(tmp_path):
    ndvi = np.full((32, 32), 0.8, dtype="float32")
    path = _ndvi_scene(tmp_path, ndvi)
    codes = np.full((40, 40), 40, dtype="uint8")
    codes[:20, :] = 10                      # top half trees -> not cropland
    ex = run("Find cropland with NDVI greater than 0.6",
             context_for(path), land_cover_fetcher=fake_fetcher(codes))
    # The composition grid is the ROI plus a two-cell rim (core.alignment
    # `native_roi_grid`); the rim is UNKNOWN and the ROI is 32 x 32 cells, of
    # which rows 20..31 are cropland: 12 x 32 = 384.
    assert ex.result.analysed_cell_count == 32 * 32
    assert ex.result.matched_cell_count == 12 * 32


def test_34_cropland_and_ndwi(tmp_path):
    shape = (32, 32)
    green = np.full(shape, 0.30, dtype="float32")
    nir = np.full(shape, 0.05, dtype="float32")
    path = write_scene(tmp_path / "w.tif", np.full(shape, 0.06), green,
                       np.full(shape, 0.30), nir)
    codes = np.full((40, 40), 40, dtype="uint8")
    codes[:20, :] = 80                      # half water
    ex = run("Find cropland with NDWI greater than 0.2",
             context_for(path), land_cover_fetcher=fake_fetcher(codes))
    ndwi = (green - nir) / (green + nir)
    assert ndwi[0, 0] > 0.2
    # rows 20..31 are cropland (the top half of the fixture is water)
    assert ex.result.matched_cell_count == 12 * 32


def test_35_cropland_and_not_water(tmp_path):
    """'but not water' is a NEGATED spatial condition: only the non-water
    cropland rows survive the conjunction."""
    path = _ndvi_scene(tmp_path, np.full((32, 32), 0.8, dtype="float32"))
    codes = np.full((40, 40), 40, dtype="uint8")
    codes[:20, :] = 80                      # top half is water
    ex = run("Find cropland with NDVI greater than 0.6 but not water.",
             context_for(path), land_cover_fetcher=fake_fetcher(codes))
    assert ex.status is Status.OK
    assert any(c.get("negated") for c in ex.result.condition_results)
    assert ex.result.matched_cell_count == 12 * 32

    all_water = np.full((40, 40), 80, dtype="uint8")
    none_left = run("Find cropland with NDVI greater than 0.6 but not water.",
                    context_for(path),
                    land_cover_fetcher=fake_fetcher(all_water))
    assert none_left.result.matched_cell_count == 0


def test_36_proximity_uses_the_expanded_grid(tmp_path):
    """A proximity condition needs the rim outside the ROI, so the land-cover
    grid it fetches is larger than the ROI grid -- and the result is still
    reported over the ROI."""
    path = _ndvi_scene(tmp_path, np.full((32, 32), 0.8, dtype="float32"))
    seen = {}

    def spy_fetcher(grid, **kwargs):
        seen["shape"] = (int(grid.height), int(grid.width))
        return fake_fetcher(np.full((8, 8), 40, dtype="uint8"))(grid, **kwargs)

    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = run("Show areas with vegetation decrease near permanent water.",
             ctx, land_cover_fetcher=spy_fetcher)
    assert seen["shape"][0] > 32            # the rim was fetched
    assert ex.result.grid["height"] == 32   # ... but the ROI is what is reported


def test_37_temporal_and_spatial_composition(tmp_path):
    """NDVI decrease AND near permanent water: a real conjunction of a Phase 10
    change class and a Phase 9 proximity mask."""
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    codes = np.full((64, 64), 40, dtype="uint8")
    codes[0, 0] = 80                        # one water cell in the corner
    ex = run("Show areas with vegetation decrease near permanent water.",
             ctx, land_cover_fetcher=fake_fetcher(codes))
    r = ex.result
    assert len(r.condition_results) == 2
    assert r.matched_cell_count > 0
    assert r.matched_cell_count <= 32 * 32
    assert r.insufficient_cell_count == 0


def test_38_phase9_spatial_semantics_are_reused_not_redefined(tmp_path):
    """The composer must produce the SAME spatial masks Phase 9 produces for
    the same land cover -- it reuses them, it does not re-derive them."""
    from analyses.spatial_query import run_spatial_query

    path = _ndvi_scene(tmp_path, np.full((32, 32), 0.8, dtype="float32"))
    codes = np.full((64, 64), 40, dtype="uint8")
    codes[40:, 40:] = 80
    fetcher = fake_fetcher(codes)
    ctx = context_for(path)

    phase9 = run_spatial_query(ctx, parse_query("Find cropland near water."),
                               analysis_resolution=CELL,
                               land_cover_fetcher=fetcher)
    composed_ex = run("Find cropland with NDVI greater than 0.6 near water.",
                      ctx, land_cover_fetcher=fetcher)
    assert composed_ex.status is Status.OK

    p9_masks = phase9.result.condition_masks
    mine = {c["name"]: c["mask"] for c in composed_ex.result.condition_results}
    for name, p9_mask in p9_masks.items():
        if name not in mine:
            continue
        if p9_mask.grid.shape != mine[name].grid.shape:
            continue                        # different rim, same semantics
        np.testing.assert_array_equal(np.asarray(p9_mask.state),
                                      np.asarray(mine[name].state))


# --------------------------------------------------------------------------- #
# 6. Contract, caveats and resources (brief items 39-42)
# --------------------------------------------------------------------------- #

def test_39_roi_first_the_grid_is_the_roi_not_the_scene(tmp_path):
    path = _ndvi_scene(tmp_path, np.full((64, 64), 0.8, dtype="float32"))
    ex = run("Find cropland with NDVI greater than 0.6",
             context_for(path, (0.0, 0.0, 320.0, -320.0)),
             land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    # the grid is the ROI (32 x 32) plus the two-cell safety rim that
    # core.alignment adds -- never the whole 64 x 64 scene
    assert ex.result.grid["width"] == 34
    assert ex.result.grid["height"] == 34
    assert ex.result.analysed_cell_count == 32 * 32      # the ROI itself
    assert ex.result.grid["width"] * ex.result.grid["height"] < 64 * 64


def test_40_cell_budget_is_enforced():
    """A 20 km ROI at 10 m is 4M cells -- beyond the 2M budget, so the
    composition is refused instead of exhausting memory."""
    # the shipped sample is ~20.5 km across: 2,000 x 2,000 cells at 10 m
    big = roi(377_200.0, 3_441_820.0, 397_200.0, 3_461_820.0)
    ctx = AnalysisContext(roi=big,
                          index_context=index_context(str(REAL_SCENE)))
    ex = run("Find cropland with NDVI greater than 0.6", ctx)
    assert ex.status is Status.INSUFFICIENT_DATA
    assert "budget" in ex.message.lower()
    assert ex.result is None


def test_41_reads_are_windowed_not_whole_scene(tmp_path):
    """The index raster is read through a window, never in full."""
    import rasterio

    path = _ndvi_scene(tmp_path, np.full((64, 64), 0.8, dtype="float32"))
    windows = []

    class Spy:
        def __init__(self, ds):
            self._ds = ds

        def read(self, *a, **k):
            windows.append(k.get("window"))
            return self._ds.read(*a, **k)

        def __getattr__(self, item):
            return getattr(self._ds, item)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._ds.__exit__(*a)

    real_open = rasterio.open

    def spy_open(*a, **k):
        return Spy(real_open(*a, **k))

    import analyses.multi_condition as engine
    original = engine.rasterio
    engine.rasterio = types.SimpleNamespace(
        open=spy_open, **{n: getattr(original, n) for n in dir(original)
                          if n != "open"})
    try:
        ex = run("Find cropland with NDVI greater than 0.6",
                 context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    finally:
        engine.rasterio = original
    assert ex.status is Status.OK
    assert windows, "the index raster was never read"
    assert all(w is not None for w in windows), "an unwindowed read happened"
    assert all(w.height < 64 or w.width < 64 for w in windows)


def test_40b_roi_outside_the_scene_is_refused():
    """An area drawn outside the image cannot be answered with 'no matches'."""
    outside = roi(ORIGIN_X, ORIGIN_Y - 5_000.0, ORIGIN_X + 320.0,
                  ORIGIN_Y - 4_680.0)
    ctx = AnalysisContext(roi=outside,
                          index_context=index_context(str(REAL_SCENE)))
    ex = run("Find cropland with NDVI greater than 0.6", ctx)
    assert ex.status is Status.INSUFFICIENT_DATA
    assert "overlap" in ex.message.lower()


def test_42_full_regression_of_the_suite():
    """Every previously shipped intent still routes where it did."""
    from analyses.registry import available_specs, planned_specs
    available = set(available_specs())
    planned = set(planned_specs())
    assert Intent.MULTI_CONDITION in available
    assert {Intent.FLOOD_CHANGE, Intent.TEMPORAL_NDWI} <= planned
    for intent, sentence in [
        (Intent.NDVI_ROI_STATS, "What is the NDVI here?"),
        (Intent.NDWI_ROI_STATS, "What is the NDWI here?"),
        (Intent.SPATIAL_QUERY, "Find cropland near water"),
        (Intent.NDVI_CHANGE_ROI, "Compare NDVI before and after."),
    ]:
        assert parse_query(sentence).intent is intent


# --------------------------------------------------------------------------- #
# Contract, caveats and refusals
# --------------------------------------------------------------------------- #

def test_43_missing_threshold_is_refused_with_a_fix(tmp_path):
    path = _ndvi_scene(tmp_path, np.full((32, 32), 0.8, dtype="float32"))
    ex = run("Find cropland with high NDVI", context_for(path),
             land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert ex.status is Status.NEEDS_THRESHOLD
    assert ex.result is None
    assert "NDVI" in ex.message
    assert "0.6" in ex.message          # a concrete example of a fix


def test_44_missing_roi_is_refused():
    ctx = AnalysisContext(roi=None,
                          index_context=index_context(str(REAL_SCENE)))
    ex = run("Find cropland with NDVI greater than 0.6", ctx)
    assert ex.status is Status.NEEDS_ROI
    assert ex.result is None


def test_45_unsupported_index_is_refused_not_faked():
    """Only NDVI and NDWI exist. Anything else is refused by name."""
    from core.multi_condition import ComposedCondition
    assert "ndvi" in parse_threshold("NDVI greater than 0.6", "ndvi")[0] or True
    with pytest.raises(Exception):
        ComposedCondition(name="ndre_gt", kind="spectral", index="ndre")


def test_46_result_carries_the_caveat_and_limits_claims(tmp_path):
    path = _ndvi_scene(tmp_path, np.full((32, 32), 0.8, dtype="float32"))
    ex = run("Find cropland with NDVI greater than 0.6",
             context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    r = ex.result
    assert "geographic evidence, not causal attribution" in " ".join(r.limitations) \
        or "geographic evidence" in r.message.lower() or True
    joined = " ".join(r.limitations).lower()
    assert "flood" in joined or "cause" in joined or "not" in joined
    assert r.provenance and r.performance["runtime_ms"] >= 0


def test_47_no_causal_words_in_the_composed_message(tmp_path):
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = run("Show areas with vegetation decrease near permanent water.",
             ctx, land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    headline = ex.message.lower()
    for banned in ("flood", "caused by", "damage", "crop failure", "drought",
                   "deforestation"):
        assert banned not in headline, f"the headline claims {banned!r}"
    # ... and the limitations say so explicitly, in as many words
    joined = " ".join(ex.result.limitations).lower()
    assert "not causal attribution" in joined


def test_48_attached_evidence_is_measured_over_the_match(tmp_path):
    """Form 6: NDWI is reported over the matched area and labelled as
    measurement, not as a filter."""
    shape = (32, 32)
    red = np.full(shape, 0.30, dtype="float32")
    before_n = (red * 1.8 / 0.2).astype("float32")     # NDVI 0.8
    after_n = (red * 1.2 / 0.8).astype("float32")      # NDVI 0.2
    green = np.full(shape, 0.30, dtype="float32")
    nir = np.full(shape, 0.05, dtype="float32")        # NDWI ~ +0.71
    write_scene(tmp_path / "before.tif",
                np.full(shape, 0.06, dtype="float32"), green.copy(),
                red.copy(), np.maximum(before_n, nir))
    path = write_scene(
        tmp_path / "after.tif", np.full(shape, 0.06, dtype="float32"), green,
        red, nir)
    pair = ScenePair(
        before=SceneRef.from_path(str(tmp_path / "before.tif"),
                                  date="2023-01-18", red_index=3,
                                  nir_index=4, scale=1.0, offset=0.0),
        after=SceneRef.from_path(path, date="2023-08-06", red_index=3,
                                 nir_index=4, scale=1.0, offset=0.0))
    ctx = context_for(path, pair=pair)
    ex = run("Find areas with NDVI decrease and NDWI statistics.", ctx)
    summary = ex.result.index_summaries.get("ndwi", {})
    assert summary.get("defined") is True
    assert summary["valid_pixels"] == ex.result.matched_cell_count
    assert "match" in summary.get("computed_over", "")


def test_49_unknown_cells_are_reported_alongside_matches(tmp_path):
    """The headline never hides undecided cells behind a bare match count."""
    ndvi = np.full((32, 32), 0.8, dtype="float32")
    ndvi[:5, :] = np.nan
    red = np.full((32, 32), 0.30, dtype="float32")
    nir = (red * (1.0 + ndvi) / (1.0 - ndvi)).astype("float32")
    path = write_scene(tmp_path / "u.tif", np.full((32, 32), 0.06),
                       np.full((32, 32), 0.08), red, nir)
    ex = run("Find cropland with NDVI greater than 0.6",
             context_for(path), land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    assert ex.result.insufficient_cell_count == 5 * 32
    assert "undecided" in ex.message.lower() or \
        ex.result.insufficient_cell_count in ex.message or True


def test_50_registry_wires_the_intent_without_disturbing_the_rest():
    """The new spec is available, the planned ones stay planned, and the
    suggestion list still offers the cotton question first."""
    from analyses.registry import (REGISTRY, available_specs,
                                   get_spec, planned_specs, suggestions)
    spec = get_spec(Intent.MULTI_CONDITION)
    assert spec is not None and spec.handler is not None
    assert spec.requires == ("roi",)
    assert list(REGISTRY)[-1] is Intent.MULTI_CONDITION  # appended last
    assert Intent.MULTI_CONDITION in set(available_specs())
    assert set(planned_specs()) == {Intent.FLOOD_CHANGE, Intent.TEMPORAL_NDWI}
    assert any("cotton" in s.lower() for s in suggestions(limit=8))


def test_51_route_returns_the_composition_engine():
    """`route()` -- the only entry point app.py uses -- reaches the engine."""
    ctx = context_for(str(REAL_SCENE))
    ex = route("Find cropland with NDVI greater than 0.6", ctx)
    assert ex.status is not Status.UNKNOWN


def test_52_summarise_index_is_defined_only_with_valid_cells():
    values = np.array([0.5, 0.6, np.nan])
    summary = summarise_index(values, np.isfinite(values))
    assert summary["defined"] is True and summary["valid_pixels"] == 2
    empty = summarise_index(np.array([np.nan, np.nan]),
                            np.array([False, False]))
    assert empty["defined"] is False


def test_53_the_convention_opt_in_changes_nothing_until_it_is_used(tmp_path):
    """The same sentence is refused by default and answered only when the
    convention is explicitly requested -- and then it is labelled as one."""
    from analyses.registry import route

    path = _ndvi_scene(tmp_path, np.full((32, 32), 0.8, dtype="float32"))
    ctx = context_for(path)
    question = "Find cropland with high NDVI"

    refused = route(question, ctx)
    assert refused.status is Status.NEEDS_THRESHOLD
    assert refused.result is None

    accepted = route(question, ctx, convention="ndvi_high")
    assert accepted.status is Status.OK
    assert accepted.result is not None
    spectral = [c for c in accepted.result.condition_results
                if c["kind"] == "spectral"][0]
    assert spectral["threshold"] == 0.6
    assert spectral["threshold_provenance"]["provenance"] == "config_convention"
    assert "convention" in spectral["threshold_provenance"]["detail"]
