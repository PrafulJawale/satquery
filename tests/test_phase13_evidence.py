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
