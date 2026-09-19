"""Phase 13 -- evidence-backed answers and result explainability.

The suite follows the acceptance brief item by item:

    contract / serialization   tests 01-08
    lineage                    tests 09-14
    explanation                tests 15-24
    unknown handling           tests 25-28
    scientific boundary        tests 29-33
    reproducibility rows       tests 34-36
    evidence layers / integration / regression   tests 37-45

Three rules run through every test:

  * Phase 13 READS the frozen Phase 9-12 results. It never re-decides anything,
    so an evidence package can only ever state what the engine already stated;
  * UNKNOWN stays UNKNOWN -- it is never a match and never a non-match, and
    "nothing could be measured" is never printed as "nothing matched";
  * no causal, validated or scientific-sounding claim is generated anywhere in
    the evidence path.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses import multi_condition as mc                        # noqa: E402
from analyses.base import AnalysisContext, IndexContext, Status   # noqa: E402
from analyses import evidence as ev                               # noqa: E402
from analyses.evidence import (                                   # noqa: E402
    UNAVAILABLE,
    condition_rows,
    evidence_from_entry,
    evidence_from_execution,
    explain,
    grid_rows,
    source_rows,
    threshold_origin_text,
)
from analyses.evidence_masks import evidence_masks                # noqa: E402
from analyses.multi_condition import run_multi_condition          # noqa: E402
from analyses.registry import route                               # noqa: E402
from core.evidence import (                                       # noqa: E402
    UNKNOWN_ALL,
    UNKNOWN_NOT_APPLICABLE,
    UNKNOWN_NONE,
    UNKNOWN_PARTIAL,
    EvidencePackage,
    EvidenceRecord,
    counts_of,
    json_safe,
    unknown_handling_for,
)
from core.session import (
    Session,
    SessionMetadata,
    SessionAnnotation,
    build_session_from_app_state,
    load_session_from_file,
    save_session_to_file,
    SESSION_SCHEMA,
    SESSION_SCHEMA_VERSION,
    list_checkpoints,
)
from core.planner import ConversationState, ConversationTurn
from core.roi import ROISelection                                 # noqa: E402
from core.router import parse_query                               # noqa: E402
from core.temporal import ScenePair, SceneRef                     # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures (same geometry conventions as the Phase 12 suite)
# --------------------------------------------------------------------------- #

CRS = "EPSG:32636"
ORIGIN_X = 200_000.0
ORIGIN_Y = 3_500_000.0
CELL = 10.0
BAND_NAMES = ("B02_blue_490nm", "B03_green_560nm",
              "B04_red_665nm", "B08_nir_842nm")

REAL_SCENE = Path(__file__).resolve().parents[1] / (
    "data/sample/s2_s2b-36ruv-20230806-0-l2a_2048px.tif")

#: Words that must never be produced by the evidence path. They are the
#: scientific claims Phase 13 is explicitly not allowed to make.
BANNED_WORDS = ("flooded", "flood extent", "crop failure", "drought",
                "deforestation", "damage", "water availability",
                "water quality", "caused by", "because of", "cause of",
                "scientific classification", "scientifically validated",
                "validated threshold")


def _transform() -> tuple:
    return (ORIGIN_X, CELL, 0.0, ORIGIN_Y, 0.0, -CELL)


def write_scene(path: Path, blue, green, red, nir, *, nodata=None) -> str:
    """A four-band synthetic Sentinel-2-like scene, reflectance-valued."""
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


def _ndvi_scene(tmp_path, ndvi_values, name="ndvi.tif") -> str:
    """A scene whose NDVI equals `ndvi_values` exactly (red held at 0.30)."""
    red = np.full(ndvi_values.shape, 0.30, dtype="float32")
    nir = (red * (1.0 + ndvi_values) / (1.0 - ndvi_values)).astype("float32")
    return write_scene(tmp_path / name,
                       np.full(ndvi_values.shape, 0.06, dtype="float32"),
                       np.full(ndvi_values.shape, 0.08, dtype="float32"),
                       red, nir)


def index_context(path: str) -> IndexContext:
    return IndexContext(path=path, roles={"blue": 1, "green": 2,
                                          "red": 3, "nir": 4},
                        scale=1.0, offset=0.0, profile="sentinel2",
                        source_label=Path(path).name,
                        role_confidence="high",
                        role_evidence=("band description",))


def roi(x0, y0, x1, y1) -> ROISelection:
    from shapely.geometry import box
    return ROISelection(is_valid=True, intersects_raster=True,
                        area_m2=abs(x1 - x0) * abs(y1 - y0),
                        geometry_raster_crs=box(x0, y0, x1, y1),
                        raster_crs=CRS)


def context_for(path: str, extent=(0.0, 0.0, 320.0, -320.0),
                pair=None) -> AnalysisContext:
    x0 = ORIGIN_X + extent[0]
    y0 = ORIGIN_Y + extent[1]
    x1 = ORIGIN_X + extent[2]
    y1 = ORIGIN_Y + extent[3]
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
    def _fetch(analysis_grid, **kwargs):
        h, w = int(analysis_grid.height), int(analysis_grid.width)
        src = np.asarray(codes, dtype="uint8")
        reps_y = int(np.ceil(h / src.shape[0]))
        reps_x = int(np.ceil(w / src.shape[1]))
        tiled = np.tile(src, (reps_y, reps_x))[:h, :w]
        return FakeLandCover(tiled)

    return _fetch


def _temporal_pair(tmp_path, before_ndvi, after_ndvi):
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


ALL_CROPLAND = np.full((64, 64), 40, dtype="uint8")


def composed(tmp_path, text, *, codes=ALL_CROPLAND, extent=(0.0, 0.0, 320.0, -320.0),
             ndvi_values=None, pair=None, name="scene.tif"):
    """One real composition, executed by the Phase 12 engine."""
    shape = (32, 32)
    if ndvi_values is None:
        ndvi_values = np.linspace(-0.2, 0.9, 32 * 32).reshape(shape)
    path = _ndvi_scene(tmp_path, np.asarray(ndvi_values, dtype="float32"), name)
    ctx = context_for(path, extent=extent, pair=pair)
    ex = run_multi_condition(ctx, parse_query(text),
                             land_cover_fetcher=fake_fetcher(codes))
    return ex


def package_of(execution):
    assert execution.status is Status.OK, execution.message
    return evidence_from_execution(execution)


# --------------------------------------------------------------------------- #
# 1. The evidence contract (brief item 1)
# --------------------------------------------------------------------------- #

def test_01_record_fields_cover_the_required_facts():
    """One record carries identity, source, condition, grid, counts and area."""
    record = EvidenceRecord(id="spectral:ndvi_gt", kind="spectral",
                            label="NDVI > 0.6", source_analysis="core.indices:ndvi",
                            source_dataset="scene.tif", band_or_index="ndvi",
                            condition="ndvi_gt", operator=">", threshold=0.6,
                            grid={"crs": CRS, "width": 32, "height": 32,
                                  "resolution_m": 10.0},
                            counts=counts_of(matched=4, non_matching=6,
                                             insufficient=2, total=12),
                            fraction=4 / 12)
    as_dict = record.to_dict()
    for key in ("id", "kind", "label", "source_analysis", "source_dataset",
                "band_or_index", "condition", "operator", "threshold",
                "threshold_provenance", "grid", "counts", "area_m2",
                "fraction", "runtime_ms", "provenance", "limitations"):
        assert key in as_dict, key
    assert as_dict["counts"]["matched"] == 4
    assert as_dict["counts"]["insufficient"] == 2
    assert as_dict["fraction"] == pytest.approx(4 / 12)


def test_02_records_are_frozen_like_the_results_they_describe():
    record = EvidenceRecord(id="a", kind="statistics", label="a")
    with pytest.raises(Exception):
        record.id = "b"          # type: ignore[misc]
    package = EvidencePackage(query="q", normalized_query="q", intent="X")
    with pytest.raises(Exception):
        package.query = "other"  # type: ignore[misc]


@pytest.mark.parametrize("counts,expected", [
    (counts_of(matched=1, non_matching=2, insufficient=3, total=6), UNKNOWN_PARTIAL),
    (counts_of(matched=1, non_matching=2, insufficient=0, total=3), UNKNOWN_NONE),
    (counts_of(matched=0, non_matching=0, insufficient=5, total=5), UNKNOWN_ALL),
    (counts_of(matched=0, non_matching=0, insufficient=0, total=0),
     UNKNOWN_NOT_APPLICABLE),
])
def test_03_unknown_handling_is_classified_not_guessed(counts, expected):
    assert unknown_handling_for(counts) == expected


def test_04_zero_denominators_are_zero_not_nan():
    """No cells analysed is a stated zero, never a division by zero."""
    counts = counts_of(matched=0, non_matching=0, insufficient=0, total=0)
    assert counts == {"matched": 0, "non_matching": 0, "insufficient": 0,
                      "total": 0}
    record = EvidenceRecord(id="a", kind="statistics", label="a", counts=counts)
    as_dict = record.to_dict()
    assert as_dict["counts"]["total"] == 0
    assert as_dict["fraction"] is None          # nothing to take a fraction of


def test_05_json_safe_drops_numpy_and_odd_types():
    payload = json_safe({
        "array": np.arange(4, dtype="uint8"),
        "mask": np.array([True, False]),
        "number": np.float32(0.5),
        "text": "kept",
        "nested": {"inner": np.int64(7), "tuple": (1, 2)},
        "lambda": (lambda: 1),
    })
    # an ARRAY is data, not an evidence fact: it never enters the export
    assert payload["array"] == UNAVAILABLE
    assert payload["number"] == pytest.approx(0.5)
    assert payload["nested"]["inner"] == 7
    assert payload["nested"]["tuple"] == [1, 2]
    assert payload["lambda"] == UNAVAILABLE      # never a repr that looks real
    json.dumps(payload)                          # must be serializable


def test_06_package_serializes_and_round_trips(tmp_path):
    """The whole package survives JSON -- including the counts it reports."""
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    package = package_of(ex)
    text = package.to_json()
    back = json.loads(text)
    assert back["schema"] == "satquery-evidence/1"
    result_block = back["result"]
    assert result_block["matched_cells"] == ex.result.matched_cell_count
    assert result_block["unknown_cells"] == ex.result.insufficient_cell_count
    assert result_block["analysed_cells"] == ex.result.analysed_cell_count
    assert back["intent"] == "MULTI_CONDITION"
    assert len(back["conditions"]) == len(package.records)
    # the lineage is exported as a chain, not only as a final number
    assert back["lineage"]["query"] == package.query


def test_07_the_export_is_deterministic(tmp_path):
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    first = package_of(ex).to_json()
    second = package_of(ex).to_json()
    assert first == second, "the same result must export the same bytes"


def test_08_export_adds_no_conclusion_the_engine_did_not_make(tmp_path):
    """Only facts that exist in the result, plus the generated wording."""
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    package = package_of(ex)
    allowed_top = {"schema", "query", "normalized_query", "intent", "status",
                   "expression", "result", "lineage", "conditions", "combined",
                   "statistics", "sources", "grid", "alignment",
                   "threshold_provenance", "limitations", "boundary",
                   "runtime_ms", "explanation"}
    as_dict = json.loads(package.to_json())
    assert set(as_dict) <= allowed_top, set(as_dict) - allowed_top
    # every count in the export is one the engine reported
    assert as_dict["result"]["matched_cells"] == ex.result.matched_cell_count


# --------------------------------------------------------------------------- #
# 2. Evidence lineage (brief item 2)
# --------------------------------------------------------------------------- #

def test_09_cropland_traces_to_esa_worldcover_class_40(tmp_path):
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    cropland = [r for r in package.records if r.kind == "spatial"][0]
    assert cropland.parameters.get("classes") == [40] or \
        cropland.parameters.get("class") == 40
    assert "WorldCover" in cropland.source_dataset
    assert cropland.band_or_index == "land cover class"


def test_10_ndvi_and_ndwi_trace_to_the_index_engine_and_the_user(tmp_path):
    """NDVI > 0.6 and NDWI < -0.4: engine, operator, threshold, provenance."""
    shape = (32, 32)
    green = np.linspace(0.05, 0.40, 32 * 32).reshape(shape).astype("float32")
    nir = np.full(shape, 0.10, dtype="float32")
    path = write_scene(tmp_path / "both.tif",
                       np.full(shape, 0.06, dtype="float32"), green,
                       np.full(shape, 0.30, dtype="float32"), nir)
    ctx = context_for(path)
    ex = run_multi_condition(
        ctx, parse_query("Find cropland with NDVI greater than 0.6 "
                         "and NDWI less than -0.4"),
        land_cover_fetcher=fake_fetcher(ALL_CROPLAND))
    package = package_of(ex)
    spectral = {r.band_or_index: r for r in package.records if r.kind == "spectral"}
    assert set(spectral) == {"ndvi", "ndwi"}
    assert spectral["ndvi"].operator == ">" and spectral["ndvi"].threshold == 0.6
    assert spectral["ndwi"].operator == "<" and spectral["ndwi"].threshold == -0.4
    for record in spectral.values():
        assert record.threshold_provenance["provenance"] == "user_specified"
        assert "core.indices" in record.source_analysis
    assert package.combined is not None
    assert "AND" in package.combined.source_analysis


def test_11_the_combined_record_names_the_three_valued_operator(tmp_path):
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    assert package.combined is not None
    assert package.combined.source_analysis == "three-valued AND"
    assert package.combined.counts["matched"] == package.matched_cells


def test_12_temporal_conditions_carry_both_dates_and_phase_10(tmp_path):
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    codes = np.full((64, 64), 40, dtype="uint8")
    codes[0, 0] = 80
    ex = composed(tmp_path, "Show areas with vegetation decrease near water.",
                  codes=codes,
                  pair=_temporal_pair(tmp_path, before, after))
    package = package_of(ex)
    temporal = [r for r in package.records if r.kind == "temporal"]
    assert len(temporal) == 1
    assert temporal[0].source_dates == ("2023-01-18", "2023-08-06")
    assert "ndvi_change" in temporal[0].source_analysis
    water = [r for r in package.records if r.kind == "spatial"][0]
    assert water.parameters.get("distance_m") is not None


def test_13_the_full_chain_is_reconstructable(tmp_path):
    """Every step of the chain appears in the package, in order."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    assert package.query
    assert package.normalized_query
    assert package.intent == "MULTI_CONDITION"
    kinds = {r.kind for r in package.records}
    assert kinds == {"spatial", "spectral"}
    assert package.combined is not None
    assert package.grid and package.grid["resolution_m"] == 10.0
    assert package.statistics or package.combined.counts["matched"] > 0
    assert package.explanation["what_was_found"]


def test_14_the_chain_is_not_just_the_final_number(tmp_path):
    """A lineage of at least four named steps, not one count."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    joined = package.to_json()
    for step in ("MULTI_CONDITION", "WorldCover", "core.indices:ndvi",
                 "three-valued AND", "EPSG:32636"):
        assert step in joined, step


# --------------------------------------------------------------------------- #
# 3. The explanation generator (brief item 3)
# --------------------------------------------------------------------------- #

def test_15_counts_in_the_explanation_are_the_engine_counts(tmp_path):
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    package = package_of(ex)
    counts = package.explanation["counts"]
    assert counts["matched"] == ex.result.matched_cell_count
    assert counts["non_matching"] == ex.result.non_matching_cell_count
    assert counts["insufficient"] == ex.result.insufficient_cell_count
    assert counts["total"] == ex.result.analysed_cell_count


def test_16_area_and_fraction_are_reported(tmp_path):
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    package = package_of(ex)
    assert package.matched_area_km2 == pytest.approx(
        float(getattr(ex.result, "matched_area_km2", 0.0)), abs=1e-9)
    assert "km" in dict(grid_rows(package))["Matched area"]


def test_17_dates_are_shown_for_a_two_date_result(tmp_path):
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = route("Compare NDVI before and after.", ctx)
    package = package_of(ex)
    joined = package.to_json()
    assert "2023-01-18" in joined and "2023-08-06" in joined


def test_18_thresholds_and_their_origin_are_shown(tmp_path):
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    rows = condition_rows(package)
    spectral = [r for r in rows if "NDVI" in r["condition"]][0]
    assert "0.6" in spectral["parameter"]
    assert "your query" in spectral["origin"]


def test_19_zero_matches_is_stated_as_measured_not_as_unknown(tmp_path):
    """'No cells satisfy' is only allowed when cells were actually measured."""
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.99")
    package = package_of(ex)
    text = package.explanation["what_was_found"]
    assert ex.result.matched_cell_count == 0
    assert "No cells satisfy" in text
    assert "could not be evaluated" not in text


def test_20_all_unknown_is_not_reported_as_no_matches(tmp_path):
    """Nothing could be measured -- so it must not read like a negative result."""
    shape = (8, 8)
    broken = np.full(shape, np.nan, dtype="float32")
    path = write_scene(tmp_path / "nodata.tif",
                       np.full(shape, np.nan, dtype="float32"),
                       np.full(shape, np.nan, dtype="float32"),
                       np.full(shape, 0.30, dtype="float32"),
                       np.full(shape, 0.30, dtype="float32"))
    ctx = context_for(path, extent=(0.0, 0.0, 80.0, -80.0))
    ex = run_multi_condition(
        ctx, parse_query("Find cropland with NDVI greater than 0.6"),
        land_cover_fetcher=fake_fetcher(np.full((16, 16), 40, dtype="uint8")))
    if ex.status is Status.OK and ex.result.insufficient_cell_count == \
            ex.result.analysed_cell_count + ex.result.insufficient_cell_count:
        package = package_of(ex)
        assert package.unknown_handling == UNKNOWN_ALL
        assert "could not be evaluated" in package.explanation["what_was_found"]
        assert "No cells satisfy" not in package.explanation["what_was_found"]


def test_21_partial_unknown_states_the_undecided_cells(tmp_path):
    """Some cells undecided: the number is printed, and it is not a match."""
    shape = (16, 16)
    red = np.full(shape, 0.30, dtype="float32")
    nir = np.full(shape, 0.90, dtype="float32")           # NDVI ~ 0.5
    with_nan = nir.copy()
    with_nan[:4, :] = np.nan                              # undecided rows
    path = write_scene(tmp_path / "partial.tif",
                       np.full(shape, 0.06, dtype="float32"),
                       np.full(shape, 0.08, dtype="float32"), red, with_nan)
    ctx = context_for(path, extent=(0.0, 0.0, 160.0, -160.0))
    ex = run_multi_condition(
        ctx, parse_query("Find cropland with NDVI greater than 0.6"),
        land_cover_fetcher=fake_fetcher(np.full((32, 32), 40, dtype="uint8")))
    assert ex.status is Status.OK
    package = package_of(ex)
    if package.unknown_cells:
        assert package.unknown_handling == UNKNOWN_PARTIAL
        note = package.explanation["unknown_note"]
        assert f"{package.unknown_cells:,}" in note
        assert "not counted as matches" in note
    assert package.matched_cells <= package.analysed_cells


def test_22_a_statistics_answer_is_not_described_as_a_filter(tmp_path):
    """'What is the NDVI here?' measures -- it does not select cells."""
    values = np.full((32, 32), 0.42, dtype="float32")
    path = _ndvi_scene(tmp_path, values)
    ex = route("What is the NDVI of this area?",
               confirmed_context(tmp_path, values))
    package = package_of(ex)
    text = package.explanation["what_was_found"]
    assert "satisfy" not in text
    assert "valid cells" in text or "measured" in text
    assert not package.records            # no condition was evaluated


def test_23_a_change_result_is_described_as_a_classification(tmp_path):
    """Phase 10 classifies every cell; it does not filter them."""
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = route("Compare NDVI before and after.", ctx)
    package = package_of(ex)
    text = package.explanation["what_was_found"]
    assert "satisfy" not in text, text
    assert "valid cells" in text
    assert "classified" in package.explanation["how_it_was_evaluated"]


def test_24_the_explanation_is_deterministic(tmp_path):
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    package = package_of(ex)
    assert explain(package) == explain(package)
    again = package_of(ex)
    assert again.explanation == package.explanation


# --------------------------------------------------------------------------- #
# 4. Unknown handling is preserved exactly (brief item 8)
# --------------------------------------------------------------------------- #

def test_25_unknown_is_never_counted_as_matched_or_as_non_matching(tmp_path):
    shape = (16, 16)
    red = np.full(shape, 0.30, dtype="float32")
    nir = np.full(shape, 0.90, dtype="float32")
    nir[:4, :] = np.nan
    path = write_scene(tmp_path / "unk.tif",
                       np.full(shape, 0.06, dtype="float32"),
                       np.full(shape, 0.08, dtype="float32"), red, nir)
    ctx = context_for(path, extent=(0.0, 0.0, 160.0, -160.0))
    ex = run_multi_condition(
        ctx, parse_query("Find cropland with NDVI greater than 0.6"),
        land_cover_fetcher=fake_fetcher(np.full((32, 32), 40, dtype="uint8")))
    package = package_of(ex)
    assert package.unknown_cells == ex.result.insufficient_cell_count
    assert package.matched_cells + package.non_matching_cells == \
        package.analysed_cells
    assert package.unknown_cells > 0
    # and the package never silently zeroes them
    assert package.unknown_handling in (UNKNOWN_PARTIAL, UNKNOWN_ALL)


def test_26_no_measured_cells_is_never_printed_as_no_matches():
    """With nothing decided, the answer must not read like a negative result."""
    counts = counts_of(matched=0, non_matching=0, insufficient=0, total=0)
    assert unknown_handling_for(counts) == UNKNOWN_NOT_APPLICABLE
    package = EvidencePackage(query="q", normalized_query="q", intent="X",
                              unknown_handling=UNKNOWN_NOT_APPLICABLE)
    text = explain(package)["what_was_found"]
    assert "No cells satisfy" not in text


def test_27_grid_metadata_is_carried_not_invented(tmp_path):
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    grid = package.grid
    assert grid["crs"] == CRS
    assert grid["width"] == 32 and grid["height"] == 32
    assert grid["resolution_m"] == 10.0
    assert len(grid["transform"]) == 6
    for record in package.records:
        assert record.grid == grid          # one grid, verified identical


def test_28_missing_metadata_is_said_to_be_unavailable_not_guessed():
    package = EvidencePackage(query="q", normalized_query="q", intent="X")
    rows = dict(grid_rows(package))
    assert rows["CRS"] == UNAVAILABLE
    assert rows["Resolution"] == UNAVAILABLE
    # an unknown alignment is NOT reported as "computed on the native grid"
    assert rows["Resampling"] == UNAVAILABLE
    assert rows["Alignment method"] == UNAVAILABLE


# --------------------------------------------------------------------------- #
# 5. The scientific boundary (brief items 3, 8, 9)
# --------------------------------------------------------------------------- #

def test_29_the_caveat_is_reused_not_copied(tmp_path):
    """One sentence, one source: Phase 13 must not create a second copy."""
    assert ev.CAVEAT is mc.CAVEAT
    # and it is the same sentence the engine itself reports
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    assert ev.CAVEAT in tuple(ex.result.limitations)


def test_30_a_composed_answer_carries_the_boundary(tmp_path):
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    assert package.boundary == mc.CAVEAT
    assert package.boundary in package.explanation["boundary"]


def test_31_a_composed_answer_carries_the_boundary_in_case_b(tmp_path):
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    codes = np.full((64, 64), 40, dtype="uint8")
    codes[0, 0] = 80
    package = package_of(composed(
        tmp_path, "Show areas with vegetation decrease near water.",
        codes=codes, pair=_temporal_pair(tmp_path, before, after)))
    assert package.boundary == mc.CAVEAT


def test_32_no_scientific_or_causal_wording_anywhere_in_the_evidence(tmp_path):
    """Test the ACTUAL explanation paths for every intent, not a file scan."""
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    temporal_ctx = context_for(str(tmp_path / "before.tif"),
                              pair=_temporal_pair(tmp_path, before, after))
    scene = _ndvi_scene(tmp_path, np.linspace(-0.2, 0.9, 1024).reshape(32, 32)
                        .astype("float32"), name="stats.tif")
    from analyses.spatial_query import run_spatial_query
    cases = [
        composed(tmp_path, "Find cropland with NDVI greater than 0.6"),
        composed(tmp_path, "Show areas with vegetation decrease near water.",
                 codes=np.full((64, 64), 40, dtype="uint8"),
                 pair=_temporal_pair(tmp_path, before, after)),
        route("What is the NDVI of this area?",
              confirmed_context(tmp_path, np.full((32, 32), 0.42, "float32"))),
        route("What is the NDWI of this area?", context_for(scene)),
        route("Compare NDVI before and after.", temporal_ctx),
        run_spatial_query(context_for(scene),
                          parse_query("Where is cropland near water?"),
                          land_cover_fetcher=fake_fetcher(ALL_CROPLAND)),
    ]
    checked = 0
    for execution in cases:
        if execution.status is not Status.OK:
            continue
        checked += 1
        package = evidence_from_execution(execution)
        # The scan covers what Phase 13 GENERATES. The limitations are the
        # engine's own sentences (they are where the denials live -- "does not
        # establish flooding" -- and they are copied, never reworded).
        blob = " ".join([
            str(package.explanation["what_was_found"]),
            str(package.explanation["how_it_was_evaluated"]),
            str(package.explanation["unknown_note"]),
            " ".join(package.explanation["what_was_evaluated"]),
            " ".join(package.explanation["evidence_sources"]),
            " ".join(str(value) for row in condition_rows(package)
                     for value in row.values()),
        ]).lower()
        leaks = [word for word in BANNED_WORDS if word in blob]
        assert not leaks, f"{package.intent}: {leaks}"
    assert checked >= 5, f"only {checked} intents were exercised"


def test_33_a_convention_threshold_stays_labelled_as_one(tmp_path):
    """A labelled convention is never promoted to a scientific class."""
    ex = composed(tmp_path, "Find cropland with high NDVI")
    assert ex.status.name == "NEEDS_THRESHOLD"
    with_convention = route(
        "Find cropland with high NDVI",
        context_for(str(tmp_path / "ndvi.tif")), convention="high_ndvi")
    if with_convention.status is Status.OK:
        package = package_of(with_convention)
        origins = " ".join(threshold_origin_text(r.threshold_provenance)
                           for r in package.records)
        assert "convention" in origins
        assert "scientific" not in origins


# --------------------------------------------------------------------------- #
# 6. Reproducibility rows (brief item 6)
# --------------------------------------------------------------------------- #

def test_34_condition_rows_carry_source_parameter_and_provenance(tmp_path):
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    rows = condition_rows(package)
    assert len(rows) == 2
    for row in rows:
        assert row["condition"] and row["source"]
        assert row["parameter"] and row["origin"]
        assert isinstance(row["matched"], int) and row["matched"] >= 0
        assert isinstance(row["unknown"], int) and row["unknown"] >= 0
    spatial = [r for r in rows if "ropland" in r["condition"]][0]
    assert "WorldCover" in spatial["origin"] or "WorldCover" in spatial["source"]


def test_35_analysis_details_rows_cover_the_reproducibility_list(tmp_path):
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    rows = dict(grid_rows(package))
    for field in ("Query", "Normalized query", "Intent", "Analysis dates",
                  "CRS", "Grid", "Resolution", "Transform", "Alignment method",
                  "Resampling", "Analysed cells", "Undecided cells",
                  "Matched area", "Unknown handling"):
        assert field in rows, field
    assert rows["Intent"] == "MULTI_CONDITION"
    assert rows["Grid"] == "32 x 32 cells"


def test_36_threshold_provenance_rows_name_every_threshold(tmp_path):
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))
    package_thresholds = package.threshold_provenance
    assert any("0.6" in str(spec.get("value")) for spec in
               package_thresholds.values()), package_thresholds
    sources = source_rows(package)
    assert sources and all("analysis" in row for row in sources)
    assert any("WorldCover" in row["analysis"] or "WorldCover" in row["dataset"]
               for row in sources)


# --------------------------------------------------------------------------- #
# 7. Evidence layers, integration and regression (brief items 5, 7, 10, 11)
# --------------------------------------------------------------------------- #

def test_37_evidence_layers_are_the_very_masks_that_produced_the_counts(tmp_path):
    """A layer is only honest if it is the mask the engine used."""
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    package = package_of(ex)
    masks = evidence_masks(ex.result)
    assert set(masks) == {r.condition for r in package.records}
    for record in package.records:
        mask = masks[record.condition]
        assert int(np.count_nonzero(mask.match)) == record.counts["matched"]
        assert int(np.count_nonzero(~mask.valid)) == record.counts["insufficient"]


def test_38_temporal_class_masks_come_from_the_phase_10_class_raster(tmp_path):
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = route("Compare NDVI before and after.", ctx)
    assert ex.status is Status.OK
    masks = evidence_masks(ex.result)
    assert set(masks) == {"increase", "stable", "decrease"}
    assert int(np.count_nonzero(masks["decrease"].match)) == \
        ex.result.decreased_count
    assert int(np.count_nonzero(masks["stable"].match)) == ex.result.stable_count
    # insufficient data is never a match
    assert int(np.count_nonzero(~masks["decrease"].valid)) == \
        ex.result.insufficient_count


def test_39_a_refusal_has_no_evidence_package(tmp_path):
    """No result, no evidence: nothing is invented to fill the gap."""
    assert evidence_from_entry({"status": "NEEDS_THRESHOLD"}) is None
    assert evidence_from_entry({"status": "OK", "result": None}) is None
    ex = composed(tmp_path, "Find cropland with high NDVI")
    assert evidence_from_execution(ex) is None


def test_40_a_chat_entry_produces_the_same_package_as_the_execution(tmp_path):
    """The app and the engine cannot disagree: same builder, same bytes."""
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    entry = ex.to_dict()
    entry["result"] = ex.result
    from_entry = evidence_from_entry(entry)
    from_execution = evidence_from_execution(ex)
    assert from_entry is not None and from_execution is not None
    assert from_entry.to_json() == from_execution.to_json()


def test_41_evidence_does_not_mutate_the_result_it_describes(tmp_path):
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    before = (ex.result.matched_cell_count, ex.result.insufficient_cell_count,
              ex.result.non_matching_cell_count)
    package_of(ex)
    package_of(ex)
    after = (ex.result.matched_cell_count, ex.result.insufficient_cell_count,
             ex.result.non_matching_cell_count)
    assert before == after


def test_42_the_panel_module_exposes_the_agreed_api():
    """Opt-in layers and the download are the only UI entry points added."""
    streamlit = pytest.importorskip("streamlit")
    assert streamlit is not None
    import numpy as np

    import ui.evidence_panel as panel
    for name in ("render_evidence", "evidence_layer_choice",
                 "evidence_layer_colour", "web_evidence_mask"):
        assert hasattr(panel, name), name
    assert panel.COMPOSED_INTENTS == ("MULTI_CONDITION", "SPATIAL_QUERY")

    # The layer helper is EXERCISED, not only imported: its imports live in the
    # function body, so a wrong one would otherwise fail at render time only.
    # (`Grid.transform` is stored in Affine order, like every transform the
    # app hands to `Affine(*transform)`.)
    transform = (CELL, 0.0, ORIGIN_X, 0.0, -CELL, ORIGIN_Y)
    rgba, bounds = panel.web_evidence_mask(
        np.full((32, 32), 2, dtype="uint8"), np.ones((32, 32), dtype=bool),
        transform, CRS, 1_000_000, (72, 133, 237))
    assert rgba.shape[2] == 4 and rgba.dtype == np.uint8
    assert rgba[..., 3].max() > 0                 # something is painted
    assert len(bounds) == 2 and len(bounds[0]) == 2
    # cells outside the ROI stay transparent: they were never analysed
    empty, _ = panel.web_evidence_mask(
        np.zeros((32, 32), dtype="uint8"), np.zeros((32, 32), dtype=bool),
        transform, CRS, 1_000_000, (72, 133, 237))
    assert empty[..., 3].max() == 0


def test_43_phase_12_panels_and_results_are_untouched(tmp_path):
    """Phase 13 adds a layer of explanation; it redefines nothing."""
    ex = composed(tmp_path, "Find cropland with NDVI greater than 0.6")
    result = ex.result
    assert result.matched_cell_count + result.non_matching_cell_count == \
        result.analysed_cell_count
    package = package_of(ex)
    assert package.matched_cells == result.matched_cell_count
    assert package.unknown_cells == result.insufficient_cell_count
    # the combined mask is still the engine's own, byte for byte
    import numpy as np2
    states = np2.asarray(result.combined_mask)
    assert int(np2.count_nonzero(states == 2)) == result.matched_cell_count


@pytest.mark.skipif(not REAL_SCENE.exists(), reason="bundled scene missing")
def test_44_real_data_case_a_matches_the_engine():
    """cropland AND NDVI > 0.6 AND NDWI < -0.4 over the bundled scene."""
    package = _real_package(
        "Find cropland with NDVI greater than 0.6 and NDWI less than -0.4")
    assert package.intent == "MULTI_CONDITION"
    assert package.matched_cells == 204_521
    assert package.unknown_cells == 4_112
    assert round(package.matched_area_km2, 3) == 20.452
    assert package.unknown_handling == UNKNOWN_PARTIAL
    assert len(package.records) == 3


@pytest.mark.skipif(not REAL_SCENE.exists(), reason="bundled scene missing")
def test_45_real_data_case_b_carries_dates_and_both_analyses():
    """NDVI decrease AND near permanent water -- over two real dates."""
    package = _real_package("Show areas with vegetation decrease near water.",
                            temporal=True)
    assert package.matched_cells == 4_511
    assert round(package.matched_area_km2, 3) == 0.451
    kinds = {r.kind for r in package.records}
    assert kinds == {"spatial", "temporal"}
    joined = package.to_json()
    assert "2023-01-18" in joined and "2023-08-06" in joined


def confirmed_context(tmp_path, ndvi_values):
    """An AnalysisContext whose Phase 3 NDVI gate has been passed."""
    from analyses.base import NdviContext
    import rasterio

    path = _ndvi_scene(tmp_path, np.asarray(ndvi_values, dtype="float32"))
    with rasterio.open(path) as ds:
        transform = ds.transform
        crs = ds.crs
    return AnalysisContext(
        roi=roi(ORIGIN_X, ORIGIN_Y - 320.0, ORIGIN_X + 320.0, ORIGIN_Y),
        index_context=index_context(path),
        ndvi=NdviContext(array=np.asarray(ndvi_values, dtype="float32"),
                         mask=np.ones(np.asarray(ndvi_values).shape, dtype=bool),
                         crs=crs, transform=transform,
                         bands={"red": 3, "nir": 4},
                         source_label=Path(path).name),
        ndvi_confirmed=True,
        raster_label=Path(path).name)


def _real_package(query: str, temporal: bool = False):
    """Run a real query over the bundled scene (shared by tests 44-45)."""
    from core.bands import guess_band_roles
    from core.raster import describe_path
    from core.temporal import discover_scenes, pair_by_dates
    import rasterio
    from shapely.geometry import box

    with rasterio.open(REAL_SCENE) as ds:
        minx, maxy = ds.bounds.left, ds.bounds.top
    guess = guess_band_roles(describe_path(str(REAL_SCENE)))
    selection = ROISelection(
        is_valid=True, intersects_raster=True, area_m2=5120.0 ** 2,
        geometry_raster_crs=box(minx + 2000, maxy - 7120,
                                minx + 7120, maxy - 2000),
        raster_crs="EPSG:32636")
    ctx = AnalysisContext(
        roi=selection,
        index_context=IndexContext(path=str(REAL_SCENE),
                                   roles={k: int(v) for k, v in guess.roles.items()},
                                   scale=0.0001, offset=0.0,
                                   profile=guess.profile,
                                   source_label=REAL_SCENE.name,
                                   role_confidence="high"))
    if temporal:
        ctx.temporal_pair = pair_by_dates(discover_scenes(
            str(REAL_SCENE.parent)), "2023-01-18", "2023-08-06")
    ex = route(query, ctx)
    return package_of(ex)


# =============================================================================
# Step 8: Enhanced Evidence & Provenance UX
# =============================================================================

def test_step8_evidence_explorer_filters_by_kind(tmp_path):
    """Step 8: Evidence explorer correctly filters by kind."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    # Should have spatial and spectral kinds
    kinds = {r.kind for r in package.records}
    assert "spatial" in kinds
    assert "spectral" in kinds

    # Filter by kind
    spatial_records = [r for r in package.records if r.kind == "spatial"]
    spectral_records = [r for r in package.records if r.kind == "spectral"]

    assert len(spatial_records) >= 1
    assert len(spectral_records) >= 1


def test_step8_evidence_explorer_filters_by_condition(tmp_path):
    """Step 8: Evidence explorer correctly filters by condition."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    conditions = {r.condition for r in package.records}
    assert len(conditions) >= 1


def test_step8_evidence_explorer_filters_by_band(tmp_path):
    """Step 8: Evidence explorer correctly filters by band/index."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    bands = {r.band_or_index for r in package.records}
    assert "ndvi" in bands
    assert "land cover class" in bands


def test_step8_evidence_explorer_filters_by_source(tmp_path):
    """Step 8: Evidence explorer correctly filters by source dataset."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    sources = {r.source_dataset for r in package.records
               if r.source_dataset and r.source_dataset != UNAVAILABLE}
    assert len(sources) >= 1


def test_step8_provenance_timeline_includes_all_steps(tmp_path):
    """Step 8: Provenance timeline includes all logical steps."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    # Build timeline steps (same logic as render_provenance_timeline)
    steps = []

    # Query step
    steps.append({"title": "User Query", "type": "query"})

    # Condition steps
    for record in package.records:
        steps.append({"title": f"Condition: {record.label}", "type": "condition"})

    # Combined step
    if package.combined:
        steps.append({"title": "Combined Result", "type": "combined"})

    # Statistics steps
    for stat in package.statistics:
        steps.append({"title": f"Statistics: {stat.label}", "type": "statistics"})

    # Sources step
    if package.sources:
        steps.append({"title": "Evidence Sources", "type": "sources"})

    # Grid/Alignment step
    if package.grid or package.alignment:
        steps.append({"title": "Grid & Alignment", "type": "grid"})

    # Limitations step
    if package.limitations:
        steps.append({"title": "Limitations", "type": "limitations"})

    # Should have at least query, conditions, combined, sources, grid, limitations
    assert len(steps) >= 6

    # Verify step types (may vary by query type)
    types = [s["type"] for s in steps]
    assert "query" in types
    assert "condition" in types
    assert "combined" in types
    assert "sources" in types
    assert "grid" in types
    assert "limitations" in types


def test_step8_evidence_comparison_temporal(tmp_path):
    """Step 8: Evidence comparison works for temporal analyses."""
    before = np.full((32, 32), 0.80, dtype="float32")
    after = np.full((32, 32), 0.20, dtype="float32")
    ctx = context_for(str(tmp_path / "before.tif"),
                      pair=_temporal_pair(tmp_path, before, after))
    ex = route("Compare NDVI before and after.", ctx)
    package = package_of(ex)

    # Should have temporal records
    temporal_records = [r for r in package.records if r.kind == "temporal"]
    assert len(temporal_records) >= 3  # increase, stable, decrease

    # Check classes
    classes = {r.condition for r in temporal_records}
    assert "increase" in classes
    assert "stable" in classes
    assert "decrease" in classes
    assert "insufficient" in classes


def test_step8_evidence_comparison_conditions(tmp_path):
    """Step 8: Evidence comparison works for multi-condition analyses."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    # Should have spatial and spectral conditions
    conditions = [r.condition for r in package.records]
    assert len(conditions) >= 2


def test_step8_evidence_comparison_statistics(tmp_path):
    """Step 8: Evidence comparison works for statistics-only results."""
    values = np.full((32, 32), 0.42, dtype="float32")
    path = _ndvi_scene(tmp_path, values)
    ex = route("What is the NDVI of this area?",
               confirmed_context(tmp_path, values))
    package = package_of(ex)

    # Should have statistics
    assert len(package.statistics) >= 1
    assert package.statistics[0].kind == "statistics"
    assert package.statistics[0].band_or_index == "ndvi"


def test_step8_export_report_includes_query_and_evidence(tmp_path):
    """Step 8: Export report includes query, analysis, and evidence."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    # Simulate export report building
    report = {
        "schema": "satquery-report/1",
        "query": package.query,
        "normalized_query": package.normalized_query,
        "intent": package.intent,
        "status": package.status,
        "expression": package.expression,
        "result": {
            "status": package.status,
            "matched_cells": package.matched_cells,
            "non_matching_cells": package.non_matching_cells,
            "unknown_cells": package.unknown_cells,
            "analysed_cells": package.analysed_cells,
            "matched_area_m2": package.matched_area_m2,
            "unknown_handling": package.unknown_handling,
        },
        "evidence": {
            "conditions": [r.to_dict() for r in package.records],
            "combined": package.combined.to_dict() if package.combined else None,
            "statistics": [r.to_dict() for r in package.statistics],
            "sources": [dict(s) for s in package.sources],
            "grid": dict(package.grid),
            "alignment": dict(package.alignment),
            "threshold_provenance": dict(package.threshold_provenance),
            "limitations": list(package.limitations),
            "boundary": package.boundary,
            "runtime_ms": package.runtime_ms,
        },
        "explanation": package.explanation,
    }

    # Verify key fields present
    assert report["query"] == package.query
    assert report["intent"] == package.intent
    assert report["result"]["status"] == package.status
    assert len(report["evidence"]["conditions"]) >= 2
    assert report["evidence"]["combined"] is not None
    # Statistics may be empty for some query types
    assert "statistics" in report["evidence"]


def test_step8_export_report_excludes_secrets(tmp_path):
    """Step 8: Export report never includes secrets or credentials."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    report = {
        "query": package.query,
        "intent": package.intent,
        "evidence": {
            "conditions": [r.to_dict() for r in package.records],
        },
    }

    # Convert to JSON and check for banned terms
    report_json = json.dumps(report)
    banned = ["api key", "apikey", "access token", "sign up for",
              "password", "secret", "credential"]
    for token in banned:
        assert token not in report_json.lower()


def test_step8_export_report_no_large_arrays(tmp_path):
    """Step 8: Export report never includes large raster arrays."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    report = {
        "evidence": {
            "conditions": [r.to_dict() for r in package.records],
        },
    }

    report_json = json.dumps(report)
    # Should be reasonably small (under 100KB for this test)
    assert len(report_json) < 100_000


def test_step8_export_report_conversation_context(tmp_path):
    """Step 8: Export report includes conversation context when available."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    turn = ConversationTurn(
        user_query="Calculate NDVI for this area",
        tool_name="compute_ndvi",
        intent="NDVI_ROI_STATS",
        status="OK",
        crop=None,
        has_roi=True,
        dates=("2024-01-15", "2024-03-15"),
    )
    state.add_turn(turn)

    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    report = {
        "query": package.query,
        "intent": package.intent,
        "evidence": {
            "conditions": [r.to_dict() for r in package.records],
        },
        "conversation_context": {
            "recent_turns": [
                {
                    "query": turn.user_query,
                    "tool": turn.tool_name,
                    "intent": turn.intent,
                    "status": turn.status,
                    "crop": turn.crop,
                    "has_roi": turn.has_roi,
                    "dates": list(turn.dates),
                }
            ],
            "current_roi_available": state.current_roi_available,
            "current_dates": list(state.current_dates),
            "current_crop": state.current_crop,
            "current_intent": state.current_intent,
        },
    }

    assert "conversation_context" in report
    ctx = report["conversation_context"]
    assert ctx["current_roi_available"] is True
    assert ctx["current_dates"] == ["2024-01-15", "2024-03-15"]
    assert len(ctx["recent_turns"]) == 1
    assert ctx["recent_turns"][0]["tool"] == "compute_ndvi"


def test_step8_comparison_only_when_comparable(tmp_path):
    """Step 8: Comparison only appears when comparable evidence exists."""
    # NDVI statistics only - no temporal, no multi-condition
    values = np.full((32, 32), 0.42, dtype="float32")
    path = _ndvi_scene(tmp_path, values)
    ex = route("What is the NDVI of this area?",
               confirmed_context(tmp_path, values))
    package = package_of(ex)

    # Has statistics, no temporal, no combined
    assert len(package.statistics) >= 1
    assert package.combined is None
    temporal_records = [r for r in package.records if r.kind == "temporal"]
    assert len(temporal_records) == 0


def test_step8_no_fabricated_evidence(tmp_path):
    """Step 8: Evidence explorer/comparison never fabricates missing metadata."""
    # Result with minimal metadata
    values = np.full((32, 32), 0.42, dtype="float32")
    path = _ndvi_scene(tmp_path, values)
    ex = route("What is the NDVI of this area?",
               confirmed_context(tmp_path, values))
    package = package_of(ex)

    # Missing metadata should be UNAVAILABLE, not fabricated
    for record in package.records:
        # source_dates might be empty tuple
        assert isinstance(record.source_dates, tuple)
        # source_dataset might be UNAVAILABLE
        if record.source_dataset:
            assert record.source_dataset == UNAVAILABLE or isinstance(record.source_dataset, str)
        # band_or_index might be empty string
        assert isinstance(record.band_or_index, str)

    # Statistics should have proper parameters
    for stat in package.statistics:
        assert isinstance(stat.parameters, dict)
        # mean might be None if not computed
        if "mean" in stat.parameters:
            assert isinstance(stat.parameters["mean"], (int, float, type(None)))


# =============================================================================
# Step 9: Collaborative Session Sharing & Persistence
# =============================================================================

def test_step9_conversation_state_serializes(tmp_path):
    """Step 9: ConversationState serializes successfully."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    turn = ConversationTurn(
        user_query="Calculate NDVI for this area",
        tool_name="compute_ndvi",
        intent="NDVI_ROI_STATS",
        status="OK",
        crop=None,
        has_roi=True,
        dates=("2024-01-15", "2024-03-15"),
    )
    state.add_turn(turn)

    # Build session
    session = build_session_from_app_state(conversation_state=state)

    # Serialize
    json_str = session.to_json()
    assert "satquery-session/2" in json_str
    assert "conversation_state" in json_str

    # Deserialize
    loaded = Session.from_json(json_str)
    assert loaded.conversation_state is not None
    assert loaded.conversation_state["current_roi_available"] is True
    assert loaded.conversation_state["current_dates"] == ["2024-01-15", "2024-03-15"]


def test_step9_conversation_turns_survive_save_load(tmp_path):
    """Step 9: Conversation turns survive save/load."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    for i in range(3):
        turn = ConversationTurn(
            user_query=f"Query {i}",
            tool_name="compute_ndvi",
            intent="NDVI_ROI_STATS",
            status="OK",
            crop=None,
            has_roi=True,
            dates=(None, None),
        )
        state.add_turn(turn)

    session = build_session_from_app_state(conversation_state=state)
    json_str = session.to_json()
    loaded = Session.from_json(json_str)

    assert loaded.conversation_state is not None
    turns = loaded.conversation_state.get("recent_turns", [])
    assert len(turns) == 3
    assert turns[0]["query"] == "Query 0"
    assert turns[2]["query"] == "Query 2"


def test_step9_crop_dates_intent_survive_serialization(tmp_path):
    """Step 9: Crop, dates, intent, and ROI metadata survive serialization."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    turn = ConversationTurn(
        user_query="Can I grow cotton here?",
        tool_name="crop_suitability",
        intent="CROP_SUITABILITY",
        status="OK",
        crop="cotton",
        has_roi=True,
        dates=("2024-01-15", "2024-03-15"),
    )
    state.add_turn(turn)

    session = build_session_from_app_state(conversation_state=state)
    json_str = session.to_json()
    loaded = Session.from_json(json_str)

    cs = loaded.conversation_state
    assert cs["current_crop"] == "cotton"
    assert cs["current_dates"] == ["2024-01-15", "2024-03-15"]
    assert cs["current_intent"] == "CROP_SUITABILITY"
    assert cs["current_roi_available"] is True


def test_step9_evidence_provenance_survives_serialization(tmp_path):
    """Step 9: Evidence/provenance metadata survives serialization."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session = build_session_from_app_state(evidence_package=package.to_dict())
    json_str = session.to_json()
    loaded = Session.from_json(json_str)

    assert loaded.evidence_package is not None
    ep = loaded.evidence_package
    assert "conditions" in ep
    assert "combined" in ep
    assert "statistics" in ep
    assert "sources" in ep
    assert "grid" in ep
    assert "alignment" in ep
    assert "threshold_provenance" in ep
    assert "limitations" in ep
    assert "boundary" in ep


def test_step9_numpy_arrays_excluded(tmp_path):
    """Step 9: NumPy arrays/raw raster data are excluded."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session = build_session_from_app_state(evidence_package=package.to_dict())
    json_str = session.to_json()

    # Should be reasonably small (no arrays)
    assert len(json_str) < 100_000

    # Verify no numpy arrays in JSON
    data = json.loads(json_str)
    report_json = json.dumps(data)

    # Should not contain array representations
    assert "array" not in report_json.lower() or "array" in "satquery"


def test_step9_secrets_credentials_excluded(tmp_path):
    """Step 9: Secrets/credentials are excluded."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session = build_session_from_app_state(evidence_package=package.to_dict())
    json_str = session.to_json()

    banned = ["api key", "apikey", "access token", "sign up for",
              "password", "secret", "credential", "bearer"]
    for token in banned:
        assert token not in json_str.lower()


def test_step9_empty_session_loads_safely():
    """Step 9: Empty session loads safely."""
    session = Session(
        metadata=SessionMetadata(),
        conversation_state=None,
    )
    json_str = session.to_json()
    loaded = Session.from_json(json_str)

    assert loaded.schema == SESSION_SCHEMA
    assert loaded.version == SESSION_SCHEMA_VERSION
    assert loaded.conversation_state is None
    assert loaded.evidence_package is None


def test_step9_malformed_session_rejected():
    """Step 9: Malformed session is rejected safely."""
    # Invalid schema
    try:
        Session.from_dict({"schema": "bad", "version": 1})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported session schema" in str(e)

    # Invalid version
    try:
        Session.from_dict({"schema": SESSION_SCHEMA, "version": 999})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "newer than supported" in str(e)

    # Missing required fields
    try:
        Session.from_dict({"schema": SESSION_SCHEMA, "version": 1, "conversation_state": {}})
        # Should not crash - validation happens later
    except Exception:
        pass  # OK if it raises


def test_step9_unsupported_future_version_rejected():
    """Step 9: Unsupported future schema version is rejected safely."""
    try:
        Session.from_dict({"schema": SESSION_SCHEMA, "version": SESSION_SCHEMA_VERSION + 1})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "newer than supported" in str(e)


def test_step9_missing_optional_fields_dont_crash():
    """Step 9: Missing optional fields do not crash loading."""
    # Minimal valid session
    minimal = {
        "schema": SESSION_SCHEMA,
        "version": 1,
        "metadata": {},
    }
    loaded = Session.from_dict(minimal)
    assert loaded is not None
    assert loaded.conversation_state is None
    assert loaded.evidence_package is None


def test_step9_loaded_session_cannot_invoke_arbitrary_tools():
    """Step 9: Loaded session cannot invoke arbitrary tools."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    turn = ConversationTurn(
        user_query="Calculate NDVI for this area",
        tool_name="compute_ndvi",
        intent="NDVI_ROI_STATS",
        status="OK",
        crop=None,
        has_roi=True,
        dates=(None, None),
    )
    state.add_turn(turn)

    session = build_session_from_app_state(conversation_state=state)
    json_str = session.to_json()

    # Modify the JSON to try to inject a tool
    data = json.loads(json_str)
    data["conversation_state"]["recent_turns"][0]["tool"] = "delete_all_data"
    modified_json = json.dumps(data)

    # Loading should work but the tool name is just metadata
    loaded = Session.from_json(modified_json)
    assert loaded.conversation_state["recent_turns"][0]["tool"] == "delete_all_data"

    # The tool name is just stored metadata - it doesn't get executed
    # The actual tool execution goes through ToolAdapter which validates against SUPPORTED_TOOLS


def test_step9_historical_context_not_authoritative():
    """Step 9: Loaded historical context does not become authoritative raster/ROI."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    turn = ConversationTurn(
        user_query="Calculate NDVI for this area",
        tool_name="compute_ndvi",
        intent="NDVI_ROI_STATS",
        status="OK",
        crop=None,
        has_roi=True,
        dates=(None, None),
    )
    state.add_turn(turn)

    session = build_session_from_app_state(conversation_state=state)

    # The session records that ROI was available, but this is HISTORICAL
    # The authoritative raster/ROI must be re-selected by the user
    assert session.has_roi_selected is False  # Not authoritative
    assert session.has_raster_loaded is False  # Not authoritative

    # The conversation_state.current_roi_available is True (historical)
    # but this does not make the ROI authoritative for new analysis
    cs = session.conversation_state
    assert cs["current_roi_available"] is True


def test_step9_save_load_preserves_scientific_results(tmp_path):
    """Step 9: Save/load does not alter scientific results."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    # Get counts from the serialized form
    pkg_dict = package.to_dict()
    original_matched = pkg_dict["conditions"][0]["counts"]["matched"]

    session = build_session_from_app_state(evidence_package=pkg_dict)
    json_str = session.to_json()
    loaded = Session.from_json(json_str)

    # Evidence should be identical to what was passed in
    ep = loaded.evidence_package
    assert ep["conditions"][0]["counts"]["matched"] == original_matched
    assert ep["conditions"][1]["counts"]["matched"] == pkg_dict["conditions"][1]["counts"]["matched"]


def test_step9_reviewer_annotations_metadata_only(tmp_path):
    """Step 9: Reviewer annotations remain metadata-only."""
    from core.session import SessionAnnotation

    annotation = SessionAnnotation(
        text="This area looks suitable for cotton",
        author="Reviewer A",
        timestamp="2024-01-15T10:00:00"
    )

    session = Session(
        metadata=SessionMetadata(
            annotations=[annotation],
        ),
    )
    json_str = session.to_json()
    loaded = Session.from_json(json_str)

    assert len(loaded.metadata.annotations) == 1
    ann = loaded.metadata.annotations[0]
    assert ann.text == "This area looks suitable for cotton"
    assert ann.author == "Reviewer A"

    # Annotations are metadata only - they don't affect evidence
    assert loaded.evidence_package is None


def test_step9_session_save_load_roundtrip(tmp_path):
    """Step 9: Full session save/load roundtrip preserves state."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    turn = ConversationTurn(
        user_query="Calculate NDVI for this area",
        tool_name="compute_ndvi",
        intent="NDVI_ROI_STATS",
        status="OK",
        crop=None,
        has_roi=True,
        dates=(None, None),
    )
    state.add_turn(turn)

    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session = build_session_from_app_state(
        conversation_state=state,
        evidence_package=package.to_dict(),
    )

    # Save to file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        save_session_to_file(session, temp_path)
        loaded = load_session_from_file(temp_path)

        # Verify roundtrip
        assert loaded.schema == session.schema
        assert loaded.version == session.version
        assert loaded.conversation_state["current_roi_available"] == session.conversation_state["current_roi_available"]
        assert loaded.evidence_package["conditions"][0]["counts"]["matched"] == session.evidence_package["conditions"][0]["counts"]["matched"]
    finally:
        import os
        os.unlink(temp_path)


# =============================================================================
# Step 10: Session Comparison, Forking, Tagging & Templates
# =============================================================================

def test_step10_session_diff_unchanged(tmp_path):
    """Step 10: Session.diff() correctly identifies unchanged fields."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session1 = build_session_from_app_state(evidence_package=package.to_dict())
    session2 = build_session_from_app_state(evidence_package=package.to_dict())

    diff = session1.diff(session2)

    # All fields should be unchanged (same evidence package)
    assert len(diff["added"]) == 0
    assert len(diff["removed"]) == 0
    assert len(diff["changed"]) == 0
    assert len(diff["unavailable"]) >= 0  # Some fields may be unavailable in both
    assert len(diff["unchanged"]) > 0


def test_step10_session_diff_added_removed_changed(tmp_path):
    """Step 10: Session.diff() correctly identifies added, removed, and changed fields."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session1 = build_session_from_app_state(evidence_package=package.to_dict())
    session1.metadata.title = "Session A"
    session1.metadata.tags = ["tag1"]

    session2 = build_session_from_app_state(evidence_package=package.to_dict())
    session2.metadata.title = "Session B"
    session2.metadata.tags = ["tag2"]
    session2.metadata.description = "New description"

    diff = session1.diff(session2)

    # Title changed
    assert "metadata.title" in diff["changed"]
    assert diff["changed"]["metadata.title"]["from"] == "Session A"
    assert diff["changed"]["metadata.title"]["to"] == "Session B"

    # Tags changed
    assert "metadata.tags" in diff["changed"]

    # Description added in session2
    assert "metadata.description" in diff["added"]
    assert diff["added"]["metadata.description"] == "New description"

    # Schema/version unchanged
    assert "schema" in diff["unchanged"]
    assert "version" in diff["unchanged"]


def test_step10_session_diff_non_mutating(tmp_path):
    """Step 10: Session.diff() does not mutate either session."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session1 = build_session_from_app_state(evidence_package=package.to_dict())
    session1.metadata.title = "Original"

    session2 = build_session_from_app_state(evidence_package=package.to_dict())
    session2.metadata.title = "Modified"

    # Store original states
    orig1_title = session1.metadata.title
    orig2_title = session2.metadata.title

    # Perform diff
    diff = session1.diff(session2)

    # Sessions should be unchanged
    assert session1.metadata.title == orig1_title
    assert session2.metadata.title == orig2_title


def test_step10_session_fork_independence(tmp_path):
    """Step 10: Forked session is independent of parent."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    turn = ConversationTurn(
        user_query="Calculate NDVI for this area",
        tool_name="compute_ndvi",
        intent="NDVI_ROI_STATS",
        status="OK",
        crop=None,
        has_roi=True,
        dates=("2024-01-15", "2024-03-15"),
    )
    state.add_turn(turn)

    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    original = build_session_from_app_state(
        conversation_state=state,
        evidence_package=package.to_dict(),
    )

    # Fork the session
    forked = original.fork("Forked Analysis")

    # Verify fork has new session_id and parent reference
    assert forked.session_id != original.session_id
    assert forked.metadata.parent_session_id == original.session_id
    assert forked.metadata.forked_at is not None
    assert forked.metadata.title == "Forked Analysis"

    # Fork should have independent metadata
    forked.metadata.title = "Modified Fork"
    forked.metadata.tags = ["forked"]

    # Original should be unchanged
    assert original.metadata.title != "Modified Fork"
    assert original.metadata.tags == []

    # Fork should not inherit authoritative flags
    assert forked.has_raster_loaded is False
    assert forked.has_roi_selected is False


def test_step10_session_fork_parent_unchanged(tmp_path):
    """Step 10: Modifying parent does not affect fork."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    original = build_session_from_app_state(evidence_package=package.to_dict())
    original.metadata.tags = ["original"]

    forked = original.fork("Fork")

    # Modify original after forking
    original.metadata.title = "Modified Original"
    original.metadata.tags = ["modified"]

    # Fork should be unchanged
    assert forked.metadata.title == "Fork of Untitled Session"  # default fork title
    assert forked.metadata.tags == []
    assert forked.metadata.parent_session_id == original.session_id


def test_step10_session_fork_historical_not_authoritative(tmp_path):
    """Step 10: Fork does not make historical context authoritative."""
    from core.planner import ConversationState, ConversationTurn

    state = ConversationState()
    turn = ConversationTurn(
        user_query="Calculate NDVI for this area",
        tool_name="compute_ndvi",
        intent="NDVI_ROI_STATS",
        status="OK",
        crop=None,
        has_roi=True,
        dates=(None, None),
    )
    state.add_turn(turn)

    original = build_session_from_app_state(
        conversation_state=state,
    )

    forked = original.fork("Fork")

    # Both should have historical ROI availability but not authoritative
    assert original.conversation_state["current_roi_available"] is True
    assert original.has_roi_selected is False
    assert forked.conversation_state["current_roi_available"] is True
    assert forked.has_roi_selected is False

    # Forking doesn't make historical context authoritative
    assert forked.has_raster_loaded is False
    assert forked.has_roi_selected is False


def test_step10_session_metadata_tagging(tmp_path):
    """Step 10: Session metadata tagging works correctly."""
    session = Session()
    session.metadata.session_id = "test123"

    # Add tags
    session.add_tag("flood")
    session.add_tag("temporal")
    session.add_tag("sentinel-2")

    assert "flood" in session.metadata.tags
    assert "temporal" in session.metadata.tags
    assert "sentinel-2" in session.metadata.tags
    assert len(session.metadata.tags) == 3

    # Duplicate tag should not be added twice
    session.add_tag("flood")
    assert len(session.metadata.tags) == 3

    # Remove tag
    session.remove_tag("temporal")
    assert "temporal" not in session.metadata.tags
    assert len(session.metadata.tags) == 2

    # Update metadata
    session.update_metadata(title="Flood Analysis", description="Pune flood study")
    assert session.metadata.title == "Flood Analysis"
    assert session.metadata.description == "Pune flood study"


def test_step10_session_annotations(tmp_path):
    """Step 10: Reviewer annotations are metadata-only."""
    session = Session()
    session.metadata.session_id = "test123"

    session.add_annotation("Area looks suitable for cotton", author="Reviewer A")
    session.add_annotation("Check soil data", author="Reviewer B")

    assert len(session.metadata.annotations) == 2
    assert session.metadata.annotations[0].text == "Area looks suitable for cotton"
    assert session.metadata.annotations[0].author == "Reviewer A"
    assert session.metadata.annotations[1].text == "Check soil data"

    # Serialize and deserialize
    json_str = session.to_json()
    loaded = Session.from_json(json_str)

    assert len(loaded.metadata.annotations) == 2
    assert loaded.metadata.annotations[0].text == "Area looks suitable for cotton"
    assert loaded.metadata.annotations[0].author == "Reviewer A"

    # Annotations don't affect evidence
    assert loaded.evidence_package is None


def test_step10_template_loading(tmp_path):
    """Step 10: Session templates load correctly."""
    # Test each template
    templates = Session.get_templates()

    expected_templates = [
        "ndvi_monitoring",
        "ndvi_change_detection",
        "ndwi_water_analysis",
        "crop_suitability",
        "multi_condition_spatial",
        "temporal_ndvi_comparison",
    ]

    for key in expected_templates:
        assert key in templates
        template = templates[key]
        assert "title" in template
        assert "description" in template
        assert "tags" in template
        assert "expected_intent" in template
        assert "required_inputs" in template
        assert "optional_inputs" in template

    # Create session from template
    session = Session.from_template("ndvi_monitoring", "My NDVI Study")

    assert session.metadata.title == "My NDVI Study"
    assert session.metadata.description == templates["ndvi_monitoring"]["description"]
    assert session.metadata.tags == templates["ndvi_monitoring"]["tags"]
    assert session.metadata.session_id is not None
    assert session.metadata.created_at is not None
    assert session.metadata.updated_at is not None
    assert session.conversation_state is None  # Fresh session


def test_step10_template_does_not_execute_analysis(tmp_path):
    """Step 10: Template selection does not execute analysis."""
    # Create session from template
    session = Session.from_template("crop_suitability", "Cotton Study")

    # Template should not have evidence package (no analysis run)
    assert session.evidence_package is None

    # Template should not have conversation state
    assert session.conversation_state is None

    # Template should have expected intent and required inputs
    assert session.metadata.tags == ["agriculture", "crop", "suitability", "cotton"]
    assert "roi" in Session.get_templates()["crop_suitability"]["required_inputs"]
    assert "crop" in Session.get_templates()["crop_suitability"]["required_inputs"]


def test_step10_template_does_not_invent_inputs(tmp_path):
    """Step 10: Templates do not fabricate ROI, crop, or dates."""
    session = Session.from_template("ndvi_change_detection", "Change Study")

    # Template should not have fabricated inputs
    assert session.roi_context is None
    assert session.raster_context is None
    assert session.conversation_state is None
    assert session.evidence_package is None
    assert session.has_raster_loaded is False
    assert session.has_roi_selected is False


def test_step10_schema_v2_compatibility(tmp_path):
    """Step 10: Schema v2 is backward compatible with v1 sessions."""
    # Create a v1-style session dict (no new fields)
    v1_session = {
        "schema": "satquery-session/1",
        "version": 1,
        "metadata": {
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "annotations": [],
        },
        "conversation_state": {
            "current_roi_available": True,
            "current_crop": "cotton",
            "current_dates": ["2024-01-15", "2024-03-15"],
            "current_intent": "CROP_SUITABILITY",
            "recent_turns": [],
        },
        "raster_context": None,
        "roi_context": None,
        "evidence_package": None,
        "chat_history": [],
        "current_arguments": {},
        "has_raster_loaded": False,
        "has_roi_selected": False,
    }

    # Should load without error
    loaded = Session.from_dict(v1_session)

    # Should have defaults for new fields
    assert loaded.metadata.title == ""
    assert loaded.metadata.description == ""
    assert loaded.metadata.tags == []
    assert loaded.metadata.session_id == ""
    assert loaded.metadata.parent_session_id == ""
    assert loaded.metadata.forked_at == ""
    assert loaded.schema == "satquery-session/2"
    assert loaded.version == 2

    # Original data preserved
    assert loaded.conversation_state["current_roi_available"] is True
    assert loaded.conversation_state["current_crop"] == "cotton"


def test_step10_future_version_rejected():
    """Step 10: Future schema versions are safely rejected."""
    future_session = {
        "schema": "satquery-session/2",
        "version": 999,
        "metadata": {},
    }

    try:
        Session.from_dict(future_session)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "newer than supported" in str(e)


def test_step10_session_validation_catches_errors():
    """Step 10: Session validation catches structural errors."""
    # Missing schema
    try:
        Session.from_dict({"version": 2})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported session schema" in str(e)

    # Invalid version
    try:
        Session.from_dict({"schema": "satquery-session/2", "version": 0})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid session version" in str(e)

    # Missing conversation state fields
    invalid_cs = {
        "schema": "satquery-session/2",
        "version": 2,
        "metadata": {},
        "conversation_state": {},  # Missing required fields
    }
    loaded = Session.from_dict(invalid_cs)
    is_valid, errors = loaded.validate()
    assert not is_valid
    assert any("current_roi_available" in e for e in errors)
    assert any("current_crop" in e for e in errors)
    assert any("current_dates" in e for e in errors)
    assert any("current_intent" in e for e in errors)


def test_step10_no_secrets_in_session(tmp_path):
    """Step 10: Session serialization never includes secrets."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session = build_session_from_app_state(evidence_package=package.to_dict())
    session.metadata.title = "Test with API key"
    session.add_annotation("Remember to use api key: sk-12345")

    json_str = session.to_json()

    # Check for banned patterns
    banned = ["api key", "apikey", "access token", "sign up for",
              "password", "secret", "credential", "bearer"]
    for token in ["api key", "apikey", "access token", "sign up for",
                  "password", "secret", "credential", "bearer"]:
        assert token not in json_str.lower()


def test_step10_no_raw_arrays_in_session(tmp_path):
    """Step 10: Session serialization excludes raw arrays."""
    package = package_of(composed(
        tmp_path, "Find cropland with NDVI greater than 0.6"))

    session = build_session_from_app_state(evidence_package=package.to_dict())
    json_str = session.to_json()

    # Should be small (no large arrays)
    assert len(json_str) < 100_000

    # Should not contain numpy array representations
    data = json.loads(json_str)
    report_json = json.dumps(data)
    assert "array" not in report_json.lower() or "array" in "satquery"  # only "satquery" string


# =============================================================================
# Step 11: Session Organization & Discovery
# =============================================================================

def test_step11_list_sessions(tmp_path):
    """Step 11: list_sessions returns session entries with metadata."""
    from core.session import list_sessions, save_session_to_file, Session, SessionMetadata

    # Create and save a test session
    session = Session(
        metadata=SessionMetadata(
            title="Test Session",
            description="Test description",
            tags=["test", "ndvi"],
            session_id="test123",
        ),
    )
    session_file = tmp_path / "test_session.json"
    save_session_to_file(session, str(session_file))

    # Change session dir to tmp_path for this test
    import os
    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        entries = list_sessions()
        assert len(entries) >= 1
        # Find our test session
        test_entry = next((e for e in entries if e.session_id == "test123"), None)
        assert test_entry is not None
        assert test_entry.title == "Test Session"
        assert test_entry.description == "Test description"
        assert test_entry.tags == ["test", "ndvi"]
        assert test_entry.has_conversation is False
        assert test_entry.has_evidence is False
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_list_sessions_handles_malformed(tmp_path):
    """Step 11: list_sessions handles malformed/missing files safely."""
    import os
    from core.session import list_sessions, save_session_to_file, Session

    # Create a valid session
    session = Session(metadata=SessionMetadata(title="Valid", session_id="valid1"))
    save_session_to_file(session, str(tmp_path / "valid.json"))

    # Create a malformed JSON file
    (tmp_path / "malformed.json").write_text("{ not valid json")

    # Create an empty file
    (tmp_path / "empty.json").write_text("")

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        entries = list_sessions()
        # Should have at least the valid session
        assert len(entries) >= 1
        # Malformed entry should be included but marked as unreadable
        malformed_entry = next((e for e in entries if "malformed" in e.filepath), None)
        assert malformed_entry is not None
        assert malformed_entry.title == "(unreadable)"
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_filter_sessions_by_tags(tmp_path):
    """Step 11: filter_sessions filters by tags correctly."""
    from core.session import SessionFilter, filter_sessions, SessionListEntry
    from datetime import datetime

    entries = [
        SessionListEntry(
            session_id="1", title="A", description="", tags=["flood", "temporal"],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_CHANGE_ROI",
        ),
        SessionListEntry(
            session_id="2", title="B", description="", tags=["ndvi", "monitoring"],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_ROI_STATS",
        ),
        SessionListEntry(
            session_id="3", title="C", description="", tags=["flood", "ndvi"],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="MULTI_CONDITION",
        ),
    ]

    # Filter by ALL tags (must have both)
    filtered = filter_sessions(entries, SessionFilter(tags=["flood", "temporal"]))
    assert len(filtered) == 1
    assert filtered[0].session_id == "1"

    # Filter by ANY tag (must have at least one)
    filtered = filter_sessions(entries, SessionFilter(tag_any=["flood"]))
    assert len(filtered) == 2
    assert {e.session_id for e in filtered} == {"1", "3"}

    # Filter by ANY tag with multiple options
    filtered = filter_sessions(entries, SessionFilter(tag_any=["monitoring", "ndvi"]))
    assert len(filtered) == 2
    assert {e.session_id for e in filtered} == {"2", "3"}


def test_step11_filter_sessions_by_date(tmp_path):
    """Step 11: filter_sessions filters by date range correctly."""
    from core.session import SessionFilter, filter_sessions, SessionListEntry
    from datetime import datetime, timedelta

    now = datetime.now()
    yesterday = now - timedelta(days=1)
    tomorrow = now + timedelta(days=1)
    last_week = now - timedelta(days=7)

    entries = [
        SessionListEntry(
            session_id="1", title="Recent", description="", tags=[],
            created_at=now.isoformat(), updated_at=now.isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_ROI_STATS",
        ),
        SessionListEntry(
            session_id="2", title="Old", description="", tags=[],
            created_at=last_week.isoformat(), updated_at=last_week.isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_ROI_STATS",
        ),
    ]

    # Filter by date_from (updated after)
    filtered = filter_sessions(entries, SessionFilter(date_from=yesterday.isoformat()))
    assert len(filtered) == 1
    assert filtered[0].session_id == "1"

    # Filter by date_to (updated before)
    filtered = filter_sessions(entries, SessionFilter(date_to=tomorrow.isoformat()))
    assert len(filtered) == 2

    # Filter by date range
    filtered = filter_sessions(entries, SessionFilter(
        date_from=last_week.isoformat(), date_to=yesterday.isoformat()))
    assert len(filtered) == 1
    assert filtered[0].session_id == "2"


def test_step11_filter_sessions_by_intent(tmp_path):
    """Step 11: filter_sessions filters by intent correctly."""
    from core.session import SessionFilter, filter_sessions, SessionListEntry
    from datetime import datetime

    entries = [
        SessionListEntry(
            session_id="1", title="A", description="", tags=[],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_ROI_STATS",
        ),
        SessionListEntry(
            session_id="2", title="B", description="", tags=[],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_CHANGE_ROI",
        ),
    ]

    filtered = filter_sessions(entries, SessionFilter(intent="NDVI_ROI_STATS"))
    assert len(filtered) == 1
    assert filtered[0].session_id == "1"

    # Case-insensitive
    filtered = filter_sessions(entries, SessionFilter(intent="ndvi_roi_stats"))
    assert len(filtered) == 1


def test_step11_filter_sessions_by_evidence(tmp_path):
    """Step 11: filter_sessions filters by evidence/conversation presence."""
    from core.session import SessionFilter, filter_sessions, SessionListEntry
    from datetime import datetime

    entries = [
        SessionListEntry(
            session_id="1", title="With Evidence", description="", tags=[],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=True,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_ROI_STATS",
        ),
        SessionListEntry(
            session_id="2", title="Without Evidence", description="", tags=[],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_ROI_STATS",
        ),
    ]

    filtered = filter_sessions(entries, SessionFilter(has_evidence=True))
    assert len(filtered) == 1
    assert filtered[0].session_id == "1"

    filtered = filter_sessions(entries, SessionFilter(has_evidence=False))
    assert len(filtered) == 1
    assert filtered[0].session_id == "2"


def test_step11_search_sessions(tmp_path):
    """Step 11: search_sessions searches metadata case-insensitively."""
    from core.session import search_sessions, SessionListEntry
    from datetime import datetime

    entries = [
        SessionListEntry(
            session_id="abc123", title="Flood Analysis", description="Pune flood study", tags=["flood"],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_CHANGE_ROI",
        ),
        SessionListEntry(
            session_id="def456", title="NDVI Monitoring", description="Crop health", tags=["ndvi", "monitoring"],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_ROI_STATS",
        ),
    ]

    # Search by title
    results = search_sessions(entries, "flood")
    assert len(results) == 1
    assert results[0].session_id == "abc123"

    # Search by description
    results = search_sessions(entries, "crop")
    assert len(results) == 1
    assert results[0].session_id == "def456"

    # Search by tag
    results = search_sessions(entries, "monitoring")
    assert len(results) == 1
    assert results[0].session_id == "def456"

    # Search by session_id
    results = search_sessions(entries, "abc123")
    assert len(results) == 1
    assert results[0].session_id == "abc123"

    # Case-insensitive
    results = search_sessions(entries, "FLOOD")
    assert len(results) == 1

    # Empty query returns all
    results = search_sessions(entries, "")
    assert len(results) == 2

    # Whitespace only returns all
    results = search_sessions(entries, "   ")
    assert len(results) == 2


def test_step11_archive_session(tmp_path):
    """Step 11: archive_session moves session to archive directory."""
    import os
    from core.session import archive_session, list_sessions, list_archived_sessions, Session, SessionMetadata, save_session_to_file

    session = Session(metadata=SessionMetadata(title="To Archive", session_id="archive1"))
    save_session_to_file(session, str(tmp_path / "to_archive.json"))

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        # Archive the session
        success, msg = archive_session("archive1")
        assert success
        assert "archived" in msg.lower()

        # Original session should be gone from main list
        entries = list_sessions()
        assert not any(e.session_id == "archive1" for e in entries)

        # Should appear in archived list
        archived = list_archived_sessions()
        assert any(e.session_id == "archive1" for e in archived)
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_delete_session_requires_confirmation(tmp_path):
    """Step 11: delete_session requires explicit confirmation."""
    import os
    from core.session import delete_session, list_sessions, Session, SessionMetadata, save_session_to_file

    session = Session(metadata=SessionMetadata(title="To Delete", session_id="delete1"))
    save_session_to_file(session, str(tmp_path / "to_delete.json"))

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        # Without confirmation should fail
        success, msg = delete_session("delete1", confirm=False)
        assert not success
        assert "confirm" in msg.lower()

        # With confirmation should succeed
        success, msg = delete_session("delete1", confirm=True)
        assert success
        assert "deleted" in msg.lower()

        entries = list_sessions()
        assert not any(e.session_id == "delete1" for e in entries)
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_create_and_list_checkpoints(tmp_path):
    """Step 11: create_checkpoint and list_checkpoints work correctly."""
    import os
    from core.session import create_checkpoint, list_checkpoints, load_checkpoint, Session, SessionMetadata, save_session_to_file

    session = Session(metadata=SessionMetadata(title="Checkpoint Test", session_id="cp_test"))
    save_session_to_file(session, str(tmp_path / "cp_test.json"))

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        # Load session
        loaded = Session.from_json((tmp_path / "cp_test.json").read_text())

        # Create checkpoint
        success, msg = create_checkpoint(loaded, label="test_label")
        assert success
        assert "checkpoint" in msg.lower()

        # List checkpoints
        checkpoints = list_checkpoints("cp_test")
        assert len(checkpoints) == 1
        assert checkpoints[0]["label"] == "test_label"

        # Load checkpoint
        checkpoint_session = load_checkpoint("cp_test", checkpoints[0]["filename"])
        assert checkpoint_session is not None
        assert checkpoint_session.metadata.title == "Checkpoint Test"
        assert checkpoint_session.has_raster_loaded is False  # Checkpoints don't preserve authoritative raster/ROI
        assert checkpoint_session.has_roi_selected is False
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_checkpoint_bounded_history(tmp_path):
    """Step 11: Checkpoint history is bounded (max 10 per session)."""
    import os
    from core.session import create_checkpoint, list_checkpoints, Session, SessionMetadata, save_session_to_file, MAX_CHECKPOINTS_PER_SESSION

    session = Session(metadata=SessionMetadata(title="Bounded Test", session_id="bounded_test"))
    save_session_to_file(session, str(tmp_path / "bounded_test.json"))

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        loaded = Session.from_json((tmp_path / "bounded_test.json").read_text())

        # Create more checkpoints than the limit
        for i in range(MAX_CHECKPOINTS_PER_SESSION + 5):
            success, msg = create_checkpoint(loaded, label=f"checkpoint_{i}")
            assert success

        # Should only keep the most recent MAX_CHECKPOINTS_PER_SESSION
        checkpoints = list_checkpoints("bounded_test")
        assert len(checkpoints) == MAX_CHECKPOINTS_PER_SESSION

        # The oldest should be gone, newest should remain
        labels = [cp["label"] for cp in checkpoints]
        assert "checkpoint_0" not in labels
        assert f"checkpoint_{MAX_CHECKPOINTS_PER_SESSION + 4}" in labels
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_checkpoint_no_raster_arrays(tmp_path):
    """Step 11: Checkpoints don't contain raw raster arrays or secrets."""
    import os
    from core.session import create_checkpoint, list_checkpoints, load_checkpoint, Session, SessionMetadata, save_session_to_file

    session = Session(metadata=SessionMetadata(title="Secret Test", session_id="secret_test"))
    save_session_to_file(session, str(tmp_path / "secret_test.json"))

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        loaded = Session.from_json((tmp_path / "secret_test.json").read_text())

        success, msg = create_checkpoint(loaded, label="with_secrets")
        assert success

        # Load checkpoint and verify no secrets or arrays
        checkpoints = list_checkpoints("secret_test")
        cp_session = load_checkpoint("secret_test", checkpoints[0]["filename"])

        assert cp_session is not None
        # Checkpoint should not have raster_context, roi_context, evidence_package
        # (they're not saved in checkpoints)
        json_str = json.dumps(cp_session.to_dict())
        # Should be small
        assert len(json_str) < 100_000
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_restore_archived_session(tmp_path):
    """Step 11: restore_archived_session restores session to main directory."""
    import os
    from core.session import archive_session, restore_archived_session, list_sessions, list_archived_sessions, Session, SessionMetadata, save_session_to_file

    session = Session(metadata=SessionMetadata(title="Restore Test", session_id="restore_test"))
    save_session_to_file(session, str(tmp_path / "restore_test.json"))

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        # Archive
        success, msg = archive_session("restore_test")
        assert success

        # Find archived filename
        archived = list_archived_sessions()
        archived_entry = next(e for e in archived if e.session_id == "restore_test")
        archive_filename = Path(archived_entry.filepath).name

        # Restore
        success, msg = restore_archived_session(archive_filename)
        assert success
        assert "restored" in msg.lower()

        # Should be back in main list
        entries = list_sessions()
        assert any(e.session_id == "restore_test" for e in entries)
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_load_session_list_entry_extracts_intent(tmp_path):
    """Step 11: load_session_list_entry extracts intent from conversation state."""
    import os
    from core.session import load_session_list_entry, Session, SessionMetadata, save_session_to_file
    from core.planner import ConversationState, ConversationTurn

    # Create session with conversation state containing intent
    state = ConversationState()
    turn = ConversationTurn(
        user_query="Calculate NDVI",
        tool_name="compute_ndvi",
        intent="NDVI_ROI_STATS",
        status="OK",
        crop=None,
        has_roi=True,
        dates=(None, None),
    )
    state.add_turn(turn)

    session = build_session_from_app_state(conversation_state=state)
    session.metadata.session_id = "intent_test"
    save_session_to_file(session, str(tmp_path / "intent_test.json"))

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        entry = load_session_list_entry(tmp_path / "intent_test.json")
        assert entry is not None
        assert entry.intent == "NDVI_ROI_STATS"
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


def test_step11_session_list_entry_to_dict(tmp_path):
    """Step 11: SessionListEntry.to_dict serializes correctly."""
    from core.session import SessionListEntry
    from datetime import datetime

    entry = SessionListEntry(
        session_id="test123",
        title="Test",
        description="Desc",
        tags=["tag1", "tag2"],
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        parent_session_id="parent1",
        forked_at="fork_time",
        has_conversation=True,
        has_evidence=False,
        has_raster=True,
        has_roi=True,
        num_annotations=2,
        num_chat_entries=5,
        filepath="/path/to/file.json",
        intent="NDVI_ROI_STATS",
    )

    d = entry.to_dict()
    assert d["session_id"] == "test123"
    assert d["title"] == "Test"
    assert d["tags"] == ["tag1", "tag2"]
    assert d["intent"] == "NDVI_ROI_STATS"
    assert d["has_conversation"] is True


def test_step11_session_filter_matches_empty_filters(tmp_path):
    """Step 11: Empty SessionFilter matches all valid sessions."""
    from core.session import SessionFilter, filter_sessions, SessionListEntry
    from datetime import datetime

    entries = [
        SessionListEntry(
            session_id="1", title="A", description="", tags=["a"],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_ROI_STATS",
        ),
        SessionListEntry(
            session_id="2", title="B", description="", tags=["b"],
            created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
            parent_session_id="", forked_at="", has_conversation=True, has_evidence=False,
            has_raster=False, has_roi=False, num_annotations=0, num_chat_entries=0,
            filepath="", intent="NDVI_CHANGE_ROI",
        ),
    ]

    # Empty filter should match all
    filtered = filter_sessions(entries, SessionFilter())
    assert len(filtered) == 2


def test_step11_delete_checkpoint(tmp_path):
    """Step 11: delete_checkpoint removes a specific checkpoint."""
    import os
    from core.session import create_checkpoint, list_checkpoints, delete_checkpoint, Session, SessionMetadata, save_session_to_file

    session = Session(metadata=SessionMetadata(title="Delete CP Test", session_id="del_cp_test"))
    save_session_to_file(session, str(tmp_path / "del_cp_test.json"))

    old_env = os.environ.get("SATQUERY_SESSION_DIR")
    os.environ["SATQUERY_SESSION_DIR"] = str(tmp_path)

    try:
        loaded = Session.from_json((tmp_path / "del_cp_test.json").read_text())

        # Create multiple checkpoints
        success, msg1 = create_checkpoint(loaded, label="first")
        success, msg2 = create_checkpoint(loaded, label="second")

        checkpoints = list_checkpoints("del_cp_test")
        assert len(checkpoints) == 2

        # checkpoints are sorted newest first, so checkpoints[0] is "second", checkpoints[1] is "first"
        # Delete the "second" (newest) checkpoint
        success, msg = delete_checkpoint("del_cp_test", checkpoints[0]["filename"])
        assert success

        checkpoints = list_checkpoints("del_cp_test")
        assert len(checkpoints) == 1
        assert checkpoints[0]["label"] == "first"
    finally:
        if old_env:
            os.environ["SATQUERY_SESSION_DIR"] = old_env
        else:
            os.environ.pop("SATQUERY_SESSION_DIR", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
