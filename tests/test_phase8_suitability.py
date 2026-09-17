"""Phase 8 -- SYNTHETIC tests of the suitability model (no network, no real data).

Everything here is hand-calculable: the factor values are chosen so the expected
score can be worked out on paper from config/crops/cotton.yml.

    weights   temperature 0.30 | precipitation 0.30 | pH 0.15 | texture 0.15 | slope 0.10
    classes   >=0.75 highly | >=0.55 moderate | >=0.35 marginal | <0.35 unsuitable
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest

from core.suitability import (
    CLASS_CODES,
    CLASS_LABELS,
    SuitabilityResult,
    approximate_gdd,
    assess_suitability,
    classify_score,
    load_crop_config,
    score_factors,
    score_layer_stack,
    texture_membership,
    trapezoid_membership,
    usda_texture_class,
)

CFG = load_crop_config("cotton")

# --- hand-picked factor sets ------------------------------------------------ #
# texture: sand 40 / silt 35 / clay 25 -> USDA "loam" -> membership 1.00
LOAM = {"sand_pct": 40.0, "silt_pct": 35.0, "clay_pct": 25.0}
# texture: sand 90 / silt 5 / clay 5 -> USDA "sand" -> membership 0.10
SANDY = {"sand_pct": 90.0, "silt_pct": 5.0, "clay_pct": 5.0}

OPTIMAL: Dict[str, Any] = {
    "growing_season_temperature": 1.0,       # arrives as a membership
    "growing_season_precipitation": 600.0,   # mm, inside optimum 437-700
    "soil_reaction": 7.0,                    # pH, inside optimum 6.0-7.5
    "terrain_slope": 0.5,                    # %, inside optimum 0-2
    **LOAM,
}

POOR: Dict[str, Any] = {
    "growing_season_temperature": 0.10,
    "growing_season_precipitation": 300.0,   # (300-262)/(437-262) = 0.2171
    "soil_reaction": 5.2,                    # (5.2-5.0)/(6.0-5.0) = 0.2000
    "terrain_slope": 7.0,                    # (8-7)/(8-2)         = 0.1667
    **SANDY,                                 # sand                = 0.1000
}

SEASON = [4, 5, 6, 7, 8, 9, 10]


def _weights() -> Dict[str, float]:
    return dict(CFG["weights"]["values"])


def test_config_weights_are_the_documented_experimental_scheme():
    w = _weights()
    assert w == {"growing_season_temperature": 0.30,
                 "growing_season_precipitation": 0.30,
                 "soil_reaction": 0.15, "soil_texture": 0.15, "terrain_slope": 0.10}
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    assert CFG["weights"]["threshold_status"] == "experimental"
    assert CFG["growing_season"]["months"] == SEASON


# --------------------------------------------------------------------------- #
# 1-3. membership, optimum, veto
# --------------------------------------------------------------------------- #
def test_1_all_factors_optimal_score_one_highly_suitable():
    res = assess_suitability(CFG, "rainfed", dict(OPTIMAL))
    assert res.score == pytest.approx(1.0, abs=1e-9)
    assert res.classification == "Highly suitable"
    assert res.status == "OK"
    assert res.confidence == "High"
    assert not res.missing_factors
    assert not res.computed_limiting_factors


def test_2_all_factors_poor_is_unsuitable_by_score_not_by_veto():
    res = assess_suitability(CFG, "rainfed", dict(POOR))
    expected = (0.30 * 0.10 + 0.30 * (38 / 175) + 0.15 * 0.20
                + 0.15 * 0.10 + 0.10 * (1 / 6))
    assert res.score == pytest.approx(expected, abs=1e-9)
    assert res.score < 0.35
    assert res.classification == "Unsuitable"
    # the veto did NOT fire -- every factor is inside its absolute range
    assert not any("veto" in w for w in res.warnings)
    # the strongest computed limitation is the weakest factor
    assert res.computed_limiting_factors[0]["factor"] == "growing_season_temperature"


def test_3_critical_factor_veto_overrides_a_high_weighted_score():
    values = dict(OPTIMAL)
    values["growing_season_precipitation"] = 100.0      # below 262 mm absolute
    res = assess_suitability(CFG, "rainfed", values)
    # weighted mean would be 0.70 (everything else optimal, water at 0)
    assert res.score == pytest.approx(0.70, abs=1e-9)
    assert res.classification == "Unsuitable"           # veto wins
    assert any("Critical-factor veto" in w for w in res.warnings)
    assert res.computed_limiting_factors[0]["factor"] == "growing_season_precipitation"


# --------------------------------------------------------------------------- #
# 4-6. missing data
# --------------------------------------------------------------------------- #
def test_4_optional_factor_missing_renormalises_weights_and_still_scores():
    values = dict(OPTIMAL)
    values.pop("terrain_slope")                      # supporting factor
    res = assess_suitability(CFG, "rainfed", values)
    assert res.score == pytest.approx(1.0, abs=1e-9)  # (0.30+0.30+0.15+0.15)/0.90
    assert res.classification == "Highly suitable"
    assert res.status == "PARTIAL_DATA"
    assert res.missing_factors == ["terrain_slope"]
    assert res.confidence == "Medium"
    assert any("terrain_slope" in a for a in res.assumptions) is False


def test_5_critical_factor_missing_gives_no_score_at_all():
    values = dict(OPTIMAL)
    values.pop("growing_season_precipitation")       # critical factor
    res = assess_suitability(CFG, "rainfed", values)
    assert res.score is None
    assert res.classification == "Insufficient data"
    assert res.status == "INSUFFICIENT_DATA"
    assert any("A zero was NOT substituted" in w for w in res.warnings)
    assert "growing_season_precipitation" in res.missing_factors


def test_6_all_nodata_factor_is_missing_never_zero():
    values = dict(OPTIMAL)
    values["soil_reaction"] = float("nan")
    factors = score_factors(CFG, values)
    assert factors["soil_reaction"].status == "nodata"
    assert factors["soil_reaction"].membership is None or math.isnan(
        float(np.asarray(factors["soil_reaction"].membership)))
    res = assess_suitability(CFG, "rainfed", values)
    assert res.status == "INSUFFICIENT_DATA"          # pH is critical
    assert res.score is None


# --------------------------------------------------------------------------- #
# 7. hard land-cover constraint
# --------------------------------------------------------------------------- #
def test_7_land_cover_hard_constraint_excludes_without_a_score():
    res = assess_suitability(
        CFG, "rainfed", dict(OPTIMAL),
        landcover={"excluded_fraction": 0.8, "dominant_class": 50,
                   "reason": "the area is dominated by Built-up."})
    assert res.classification == "Unsuitable"
    assert res.score == 0.0
    assert res.hard_constraint["type"] == "land_cover"
    assert "Built-up" in res.hard_constraint["reason"]
    # it is an exclusion, not a computed limitation
    assert all(f["factor"] != "land_cover" for f in res.computed_limiting_factors)


def test_7b_partial_land_cover_exclusion_does_not_exclude_the_whole_area():
    res = assess_suitability(
        CFG, "rainfed", dict(OPTIMAL),
        landcover={"excluded_fraction": 0.2, "dominant_class": 40, "reason": ""})
    assert res.classification == "Highly suitable"
    assert res.hard_constraint is None


# --------------------------------------------------------------------------- #
# 8. class boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("score,expected", [
    (0.75, "Highly suitable"), (0.749, "Moderately suitable"),
    (0.55, "Moderately suitable"), (0.549, "Marginal"),
    (0.35, "Marginal"), (0.349, "Unsuitable"), (0.0, "Unsuitable"),
])
def test_8_class_boundaries(score, expected):
    assert CLASS_LABELS[int(classify_score(score, CFG))] == expected
    assert CFG["classes"]["threshold_status"] == "experimental"


# --------------------------------------------------------------------------- #
# 9. weight renormalisation (hand-calculated)
# --------------------------------------------------------------------------- #
def test_9_weight_renormalisation_with_two_factors_missing():
    values = dict(OPTIMAL)
    values.pop("terrain_slope")            # 0.10
    values["growing_season_temperature"] = 0.4
    res = assess_suitability(CFG, "rainfed", values)
    # scored: temperature 0.4 (w 0.30), precipitation 1.0 (0.30),
    #         pH 1.0 (0.15), texture 1.0 (0.15) -> total weight 0.90
    expected = (0.30 * 0.4 + 0.30 * 1.0 + 0.15 * 1.0 + 0.15 * 1.0) / 0.90
    assert res.score == pytest.approx(expected, abs=1e-9)
    assert res.status == "PARTIAL_DATA"


# --------------------------------------------------------------------------- #
# 10. assumed irrigation scenario
# --------------------------------------------------------------------------- #
def test_10_assumed_irrigation_removes_water_and_caps_the_class():
    scfg = CFG["scenarios"]["irrigation_assumed"]
    res = assess_suitability(CFG, "irrigation_assumed", dict(OPTIMAL),
                             scenario_cfg=scfg)
    # water is not scored: (0.30 + 0.15 + 0.15 + 0.10) / 0.70 = 1.0
    assert res.score == pytest.approx(1.0, abs=1e-9)
    assert res.assumed_factors == ["growing_season_precipitation"]
    assert res.classification == "Moderately suitable"       # capped
    assert res.confidence == "Low"
    assert any("capped" in w for w in res.warnings)
    assert any("hypothetical" in a.lower() or "not assessed" in a.lower()
               for a in res.assumptions)


def test_10b_assumed_irrigation_keeps_other_computed_limitations():
    values = dict(OPTIMAL)
    values["soil_reaction"] = 5.2                     # pH membership 0.2
    scfg = CFG["scenarios"]["irrigation_assumed"]
    res = assess_suitability(CFG, "irrigation_assumed", values, scenario_cfg=scfg)
    expected = (0.30 * 1.0 + 0.15 * 0.2 + 0.15 * 1.0 + 0.10 * 1.0) / 0.70
    assert res.score == pytest.approx(expected, abs=1e-9)
    assert res.computed_limiting_factors[0]["factor"] == "soil_reaction"
    assert res.classification != "Highly suitable"


# --------------------------------------------------------------------------- #
# 11-12. insufficient data and class fractions over a raster
# --------------------------------------------------------------------------- #
def test_11_insufficient_data_when_a_whole_layer_is_absent():
    layers = {
        "growing_season_temperature": np.full((4, 4), 1.0, dtype="float32"),
        "growing_season_precipitation": None,          # layer missing entirely
        "phh2o": np.full((4, 4), 7.0, dtype="float32"),
        **{k: np.full((4, 4), v, dtype="float32")
           for k, v in LOAM.items()},
        "terrain_slope": np.full((4, 4), 0.5, dtype="float32"),
    }
    out = score_layer_stack(CFG, layers, inside=np.ones((4, 4), dtype=bool))
    assert np.all(out["class_codes"] == CLASS_CODES["insufficient"])
    assert out["valid_fraction"] == 0.0


def test_12_class_fractions_sum_to_one_over_the_roi():
    h, w = 8, 8
    inside = np.zeros((h, w), dtype=bool)
    inside[2:6, 2:6] = True                            # 16 of 64 cells
    precip = np.full((h, w), 600.0, dtype="float32")
    precip[inside] = 600.0
    precip[3, 3] = 100.0                               # one cell vetoed
    layers = {
        "growing_season_temperature": np.full((h, w), 1.0, dtype="float32"),
        "growing_season_precipitation": precip,
        "phh2o": np.full((h, w), 7.0, dtype="float32"),
        **{k: np.full((h, w), v, dtype="float32") for k, v in LOAM.items()},
        "terrain_slope": np.full((h, w), 0.5, dtype="float32"),
    }
    out = score_layer_stack(CFG, layers, inside=inside)
    assert out["class_codes"][~inside].size == 48
    assert np.all(out["class_codes"][~inside] == CLASS_CODES["insufficient"])
    assert out["class_codes"][3, 3] == CLASS_CODES["unsuitable"]
    assert out["class_codes"][2, 2] == CLASS_CODES["highly"]
    assert math.isclose(sum(out["class_fractions"].values()), 1.0, abs_tol=1e-9)
    # 15 of 16 ROI cells are highly suitable (one is vetoed)
    assert math.isclose(out["class_fractions"]["Highly suitable"], 15 / 16, abs_tol=1e-9)
    assert math.isclose(out["class_fractions"]["Unsuitable"], 1 / 16, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# 13. ROI clipping
# --------------------------------------------------------------------------- #
def test_13_cells_outside_the_roi_are_not_scored():
    h = w = 6
    inside = np.zeros((h, w), dtype=bool)
    inside[1:4, 1:4] = True
    layers = {
        "growing_season_temperature": np.full((h, w), 1.0, dtype="float32"),
        "growing_season_precipitation": np.full((h, w), 600.0, dtype="float32"),
        "phh2o": np.full((h, w), 7.0, dtype="float32"),
        **{k: np.full((h, w), v, dtype="float32") for k, v in LOAM.items()},
        "terrain_slope": np.full((h, w), 0.5, dtype="float32"),
    }
    out = score_layer_stack(CFG, layers, inside=inside)
    assert int(np.count_nonzero(out["class_codes"] == CLASS_CODES["insufficient"])) == 27
    assert out["valid_fraction"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 14. coarse-source warning
# --------------------------------------------------------------------------- #
def test_14_coarse_layers_produce_a_resolution_warning():
    res = assess_suitability(
        CFG, "rainfed", dict(OPTIMAL),
        native_resolutions={"soil": "250 m", "climate": "~1 km",
                            "land_cover": "10 m", "topography": "30 m"},
        analysis_resolution=30.0)
    joined = " ".join(res.warnings)
    assert "250 m" in joined and "~1 km" in joined
    assert "does not create detail" in joined
    assert "10 m" not in joined                      # 10 m is not 4x coarser


def test_14b_grid_coarsening_is_disclosed_not_silent():
    res = assess_suitability(
        CFG, "rainfed", dict(OPTIMAL),
        analysis_resolution=60.0, requested_resolution=30.0,
        effective_resolution_note="Requested 30 m exceeded the cell cap; using 60 m.")
    assert any("cell cap" in w for w in res.warnings)


# --------------------------------------------------------------------------- #
# texture and membership maths
# --------------------------------------------------------------------------- #
def test_texture_proxy_hits_all_twelve_usda_classes():
    expected = {
        (90, 5, 5): 1, (85, 10, 5): 2, (70, 20, 10): 3, (40, 40, 20): 4,
        (20, 55, 25): 5, (5, 85, 10): 6, (60, 15, 25): 7, (30, 35, 35): 8,
        (10, 55, 35): 9, (50, 10, 40): 10, (5, 45, 50): 11, (20, 20, 60): 12,
    }
    for triple, code in expected.items():
        got = usda_texture_class(*triple)
        assert int(got) == code, f"{triple} -> {got}, expected {code}"
    assert CFG["factors"]["soil_texture"]["threshold_status"] == "experimental"
    assert "proxy" in CFG["factors"]["soil_texture"]["adaptation"].lower()


def test_trapezoid_membership_is_hand_calculable():
    assert trapezoid_membership(7.1, 5.0, 6.0, 7.5, 9.5) == pytest.approx(1.0)
    assert trapezoid_membership(5.5, 5.0, 6.0, 7.5, 9.5) == pytest.approx(0.5)
    assert trapezoid_membership(4.0, 5.0, 6.0, 7.5, 9.5) == pytest.approx(0.0)
    assert trapezoid_membership(8.5, 5.0, 6.0, 7.5, 9.5) == pytest.approx(0.5)
    assert trapezoid_membership(10.0, 5.0, 6.0, 7.5, 9.5) == pytest.approx(0.0)
    # degenerate rising limb ("flat is optimal")
    assert trapezoid_membership(0.0, 0.0, 0.0, 2.0, 8.0) == pytest.approx(1.0)
    assert trapezoid_membership(5.0, 0.0, 0.0, 2.0, 8.0) == pytest.approx(0.5)   # (8-5)/(8-2)
    # nodata propagates, never becomes 0
    assert math.isnan(float(trapezoid_membership(float("nan"), 0, 1, 2, 3)))


def test_texture_membership_uses_the_configured_class_scores():
    scores = CFG["factors"]["soil_texture"]["class_scores"]
    mem, code = texture_membership(40.0, 35.0, 25.0, scores)     # loam
    assert int(code) == 4
    assert mem == pytest.approx(scores["loam"])
    mem2, code2 = texture_membership(90.0, 5.0, 5.0, scores)     # sand
    assert int(code2) == 1
    assert mem2 == pytest.approx(scores["sand"])


# --------------------------------------------------------------------------- #
# the approximate GDD (context metric)
# --------------------------------------------------------------------------- #
def test_approximate_gdd_is_hand_calculable_and_labelled():
    # every month: tmin 20, tmax 30 -> mean 25 -> 25 - 15.6 = 9.4 DD/day
    days = [30, 31, 30, 31, 31, 30, 31]              # Apr..Oct
    expected = sum(9.4 * d for d in days)            # 2011.6
    gdd = approximate_gdd([20.0] * 7, [30.0] * 7, SEASON)
    assert gdd["value"] == pytest.approx(expected, abs=0.05)
    assert gdd["base_temp_c"] == 15.6
    assert "not measured" in gdd["warning"]


def test_approximate_gdd_truncates_at_the_base_and_never_goes_negative():
    # tmin below base is raised to the base: (15.6 + 20)/2 - 15.6 = 2.2 DD/day
    gdd = approximate_gdd([10.0], [20.0], [5])
    assert gdd["value"] == pytest.approx(2.2 * 31, abs=0.05)
    # colder than the base contributes nothing, never a negative
    gdd2 = approximate_gdd([5.0], [10.0], [1])
    assert gdd2["value"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# reporting: evidence vs assumptions
# --------------------------------------------------------------------------- #
def test_missing_factors_are_never_reported_as_computed_limitations():
    values = dict(OPTIMAL)
    values.pop("terrain_slope")
    res = assess_suitability(CFG, "rainfed", values)
    assert all(f["factor"] != "terrain_slope" for f in res.computed_limiting_factors)
    assert "terrain_slope" in res.missing_factors
    # unassessed constraints (salinity, irrigation) are assumptions, not factors
    joined = " ".join(res.assumptions).lower()
    assert "salinity" in joined and "irrigation" in joined
    assert all("salinity" not in f["factor"] for f in res.computed_limiting_factors)


def test_result_exposes_the_fields_the_brief_requires():
    res = assess_suitability(
        CFG, "rainfed", dict(OPTIMAL),
        precipitation_context={"annual_precipitation": 73.0,
                               "growing_season_precipitation": 600.0,
                               "months": SEASON, "approx_gdd": None},
        native_resolutions={"soil": "250 m"}, analysis_resolution=30.0)
    d = res.to_dict()
    for key in ("crop", "scenario", "score", "classification", "confidence",
                "factor_scores", "factor_values", "computed_limiting_factors",
                "missing_factors", "assumed_factors", "annual_precipitation",
                "growing_season_precipitation", "growing_season_months",
                "native_resolutions", "analysis_resolution", "valid_fraction",
                "class_fractions", "provenance", "threshold_provenance",
                "assumptions", "warnings", "temporal_basis"):
        assert key in d, key
    assert d["annual_precipitation"] == 73.0
    assert d["growing_season_precipitation"] == 600.0
    assert d["growing_season_months"] == SEASON
    assert d["temporal_basis"] == "climatological + static (mixed)"
    # raw arrays are never serialised into the response
    assert "suitability_raster" not in d


def test_threshold_provenance_is_machine_readable():
    rows = {r["item"]: r for r in assess_suitability(CFG, "rainfed", dict(OPTIMAL)).threshold_provenance}
    assert set(rows) >= {"factor:soil_reaction", "factor:soil_texture",
                         "weights", "classes", "growing_season_months"}
    assert rows["factor:soil_reaction"]["threshold_status"] == "literature-backed"
    assert rows["weights"]["threshold_status"] == "experimental"
    assert rows["factor:soil_texture"]["threshold_status"] == "experimental"
    assert "FAO" in rows["factor:soil_reaction"]["threshold_source"]
    assert rows["growing_season_months"]["threshold_values"]["months"] == SEASON
