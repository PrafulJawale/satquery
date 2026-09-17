"""Phase 9 / CHECKPOINT B -- synthetic tests for the spatial-query models + parser.

Everything here is hand-checkable: no rasters, no network, no external data.
These tests exist to prove the parser's semantics BEFORE any geography is
computed (Checkpoint C), and in particular to prove the two safety rules:

    "flood-prone"  must NEVER become "permanent water"
    "irrigation"   must NEVER become "near water"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.router import Intent, parse_query
from core.spatial_query import (
    detect_unsupported_requirements,
    ConditionStatus,
    ConditionType,
    Operator,
    RequestedOutput,
    SpatialQueryStatus,
    load_spatial_config,
    parse_distance,
    parse_spatial_query,
)

ROOT = Path(__file__).resolve().parent.parent


def conditions_of(query: str):
    return parse_spatial_query(query).conditions


def types_of(query: str):
    return [c.condition_type for c in parse_spatial_query(query).conditions]


def one(query: str):
    conds = parse_spatial_query(query).conditions
    assert len(conds) == 1, f"expected exactly one condition, got {conds}"
    return conds[0]


# =========================================================================== #
# 1. cotton suitability
# =========================================================================== #
@pytest.mark.parametrize("query", [
    "Find areas suitable for cotton",
    "Find suitable cotton areas",
    "Where can I grow cotton?",
    "Find cotton land",
])
def test_cotton_condition_uses_the_phase8_class_threshold(query):
    c = one(query)
    assert c.condition_type is ConditionType.CROP_SUITABILITY
    assert c.parameters["crop"] == "cotton"
    assert c.parameters["min_class"] == 3          # Moderately suitable or better
    assert c.parameters["scenario"] == "rainfed"   # never the assumed scenario
    assert c.required_analysis == "crop_suitability"
    assert c.status is ConditionStatus.SUPPORTED
    assert not c.negate


def test_cotton_threshold_is_exactly_the_config_value():
    cfg = load_spatial_config("conditions")
    assert one("Find areas suitable for cotton").parameters["min_class"] == int(
        cfg["cotton"]["min_suitability_class"])


def test_marginal_is_not_suitable():
    """Class 3 is the floor: Marginal (2) must never be treated as suitable."""
    cfg = load_spatial_config("conditions")
    assert int(cfg["cotton"]["min_suitability_class"]) >= 3


# =========================================================================== #
# 2. equivalent queries -> equivalent structure
# =========================================================================== #
def test_equivalent_queries_produce_equivalent_conditions():
    a = parse_spatial_query("Find areas suitable for cotton near water")
    b = parse_spatial_query("Show cotton areas near water suitable for cotton")
    assert [c.to_dict()["label"] for c in a.conditions] == \
           [c.to_dict()["label"] for c in b.conditions]
    assert a.operator is b.operator is Operator.AND
    assert a.expression() == b.expression()


def test_canonical_cotton_and_water_expression():
    q = parse_spatial_query("Find areas suitable for cotton near water")
    assert q.is_executable
    assert q.status is SpatialQueryStatus.OK
    assert types_of("Find areas suitable for cotton near water") == [
        ConditionType.CROP_SUITABILITY, ConditionType.WATER_PROXIMITY]
    expression = q.expression()
    assert "crop_suitability(cotton, class >= 3, rainfed)" in expression
    assert "water_proximity(" in expression and "class 80" in expression
    assert " AND " in expression


# =========================================================================== #
# 3. cropland (a land-cover class -- never a suitability statement)
# =========================================================================== #
@pytest.mark.parametrize("query", ["Find cropland", "Show agricultural land",
                                   "Find farmland"])
def test_cropland_is_a_land_cover_class(query):
    c = one(query)
    assert c.condition_type is ConditionType.LAND_COVER_CLASS
    assert c.parameters["classes"] == [40]
    assert c.required_analysis == "worldcover"


def test_cropland_is_not_suitability():
    """A separate condition with separate evidence -- never conflated."""
    assert ConditionType.CROP_SUITABILITY not in types_of("Find cropland")


# =========================================================================== #
# 4. water proximity: default and explicit distances
# =========================================================================== #
def test_default_distance_comes_from_configuration():
    cfg = load_spatial_config("conditions")
    c = one("Find areas near water")
    assert c.condition_type is ConditionType.WATER_PROXIMITY
    assert c.parameters["distance_m"] == int(cfg["water"]["default_proximity_m"])
    assert c.parameters["water_class"] == 80          # permanent water only
    assert c.parameters["distance_evidence"] is None  # not stated by the user


@pytest.mark.parametrize("query,expected", [
    ("Find cotton areas within 500 m of water", 500),
    ("Find cotton areas within 1 km of water", 1000),
    ("Find cotton areas within 2 km of water", 2000),
    ("Find cotton areas within 750 metres of water", 750),
])
def test_explicit_distance_is_parsed_to_metres(query, expected):
    c = [x for x in parse_spatial_query(query).conditions
         if x.condition_type is ConditionType.WATER_PROXIMITY][0]
    assert c.parameters["distance_m"] == expected


def test_absurd_distance_is_clamped_and_reported():
    cfg = load_spatial_config("conditions")
    q = parse_spatial_query("Find cotton areas within 500 km of water")
    c = [x for x in q.conditions
         if x.condition_type is ConditionType.WATER_PROXIMITY][0]
    assert c.parameters["distance_m"] == int(cfg["water"]["max_proximity_m"])
    assert q.warnings, "the clamp must be reported, not applied silently"


def test_parse_distance_units_and_defaults():
    units = load_spatial_config("conditions")["distance_units"]
    assert parse_distance("within 2 km of water", 1000, units, 30, 10000)[0] == 2000
    assert parse_distance("no number here", 1000, units, 30, 10000)[0] == 1000
    metres, evidence, warning = parse_distance("within 10 m", 1000, units, 30, 10000)
    assert metres == 30 and warning and evidence == "10 m"


def test_distance_convention_is_carried_with_the_condition():
    c = one("Find areas near water")
    assert "cell centre" in c.parameters["distance_convention"]
    assert "class-80" in c.parameters["distance_convention"]


# =========================================================================== #
# 5. operators
# =========================================================================== #
def test_and_is_the_default_operator():
    q = parse_spatial_query("Find cropland suitable for cotton")
    assert q.operator is Operator.AND
    assert len(q.conditions) == 2


def test_or_is_recognised():
    q = parse_spatial_query("Find cropland or cotton areas")
    assert q.operator is Operator.OR
    assert " OR " in q.expression()


def test_mixed_and_or_is_refused_not_guessed():
    q = parse_spatial_query("Find cropland and cotton areas or water")
    assert q.status is SpatialQueryStatus.NEEDS_CLARIFICATION
    assert not q.is_executable


@pytest.mark.parametrize("query", [
    "Find cotton areas but not built-up",
    "Find cotton areas excluding built-up",
    "Find cotton areas outside built-up areas",
    "Find cotton areas away from built-up",
])
def test_negation_forms_attach_to_the_following_condition(query):
    q = parse_spatial_query(query)
    negated = [c for c in q.conditions if c.negate]
    assert len(negated) == 1
    assert negated[0].parameters["classes"] == [50]
    assert "NOT land_cover([50])" in q.expression()


def test_not_water_is_expressed_as_a_negated_condition():
    """'excluding water' negates the WATER CLASS (see also the proximity twin
    below, which is a different condition with a different meaning)."""
    q = parse_spatial_query("Find cropland excluding water")
    water = [c for c in q.conditions
             if c.condition_type is ConditionType.WATER][0]
    assert water.negate is True
    assert q.is_executable


def test_an_unattachable_not_is_refused():
    """A negation the grammar cannot place is clarified, not guessed."""
    q = parse_spatial_query("Find areas not suitable for cotton")
    assert q.status is SpatialQueryStatus.NEEDS_CLARIFICATION
    assert not q.is_executable


# =========================================================================== #
# 6. unsupported conditions abort -- and are NEVER proxied
# =========================================================================== #
@pytest.mark.parametrize("query,topic", [
    ("cotton outside flood-prone areas", "flood"),
    ("Find cotton areas with reliable irrigation", "irrigation"),
    ("cotton with reliable irrigation", "irrigation"),
    ("cotton where irrigation is guaranteed", "irrigation"),
    ("cotton near groundwater", "groundwater"),
    ("Find cotton land on saline soil", "salinity"),
])
def test_unsupported_conditions_abort_the_query(query, topic):
    q = parse_spatial_query(query)
    assert q.status is SpatialQueryStatus.NEEDS_CLARIFICATION
    assert not q.is_executable
    assert q.supported_conditions == ()
    assert [c.parameters["topic"] for c in q.unsupported_conditions] == [topic]


def test_flood_prone_does_not_become_permanent_water():
    """The central safety rule of Phase 9."""
    q = parse_spatial_query("cotton outside flood-prone areas")
    assert ConditionType.WATER_PROXIMITY not in types_of(
        "cotton outside flood-prone areas")
    assert all(c.condition_type is ConditionType.UNSUPPORTED
               for c in q.conditions)
    assert "flood" in q.warnings[0].lower()


def test_irrigation_does_not_become_near_water():
    """The second safety rule: no proxy for a water SUPPLY."""
    q = parse_spatial_query("Find cotton land with reliable irrigation")
    assert ConditionType.WATER_PROXIMITY not in types_of(
        "Find cotton land with reliable irrigation")
    unsupported = q.unsupported_conditions[0]
    assert unsupported.parameters["nearest_supported"] == "water_proximity"
    assert "irrigation availability is not currently measured" in unsupported.note


def test_unsupported_plus_supported_still_aborts_everything():
    """No partial answer: an unsupported condition cancels the whole query."""
    q = parse_spatial_query("Find cotton areas near water with reliable irrigation")
    assert q.status is SpatialQueryStatus.NEEDS_CLARIFICATION
    assert q.supported_conditions == ()


def test_wetland_is_not_water():
    """Class 90 is not class 80 -- the config must say so."""
    cfg = load_spatial_config("conditions")
    assert int(cfg["water"]["water_class"]) == 80
    assert int(cfg["land_cover"]["wetland_class"]) == 90
    assert int(cfg["land_cover"]["wetland_class"]) != int(cfg["water"]["water_class"])


# =========================================================================== #
# 7. ambiguity is recorded, and never implies irrigation
# =========================================================================== #
def test_ambiguous_near_water_is_interpreted_and_announced():
    q = parse_spatial_query("Can I grow cotton near water?")
    assert q.status is SpatialQueryStatus.OK
    assert {c.condition_type for c in q.conditions} == {
        ConditionType.CROP_SUITABILITY, ConditionType.WATER_PROXIMITY}
    joined = " ".join(q.notes).lower()
    assert "not irrigation availability" in joined


def test_water_condition_always_carries_the_not_irrigation_meaning():
    q = parse_spatial_query("Find areas near water")
    assert any("not irrigation" in n.lower() or "not irrigation" in n.lower()
               for n in q.notes)


# =========================================================================== #
# 8. malformed / empty input
# =========================================================================== #
@pytest.mark.parametrize("query", ["", "   ", "asdfgh qwerty", "?? !!"])
def test_nothing_recognisable_is_reported_not_guessed(query):
    q = parse_spatial_query(query)
    assert q.status is SpatialQueryStatus.NO_CONDITIONS
    assert q.conditions == ()
    assert not q.is_executable


# =========================================================================== #
# 9. the model itself
# =========================================================================== #
def test_result_is_json_safe_and_carries_no_arrays():
    payload = json.dumps(parse_spatial_query(
        "Find cotton areas near water but not built-up").to_dict())
    assert "SpatialCondition" not in payload
    assert "numpy" not in payload
    assert json.loads(payload)["operator"] == "and"


def test_requested_output_defaults_to_areas():
    q = parse_spatial_query("Find cropland")
    assert q.requested_output is RequestedOutput.AREAS


def test_no_raster_or_ui_logic_in_the_model_module():
    """Architecture rule: models + parser only (mirrors the router rule)."""
    src = (ROOT / "core" / "spatial_query.py").read_text(encoding="utf-8")
    for forbidden in ("import numpy", "import rasterio", "import streamlit",
                      "import folium", "distance_transform_edt"):
        assert forbidden not in src, f"{forbidden} does not belong in the model"


def test_semantic_defaults_come_from_configuration_not_python():
    """Change the config -> the parse changes. Nothing is baked into the code."""
    custom = load_spatial_config("conditions")
    custom["water"]["default_proximity_m"] = 250
    custom["cotton"]["min_suitability_class"] = 4
    custom["water"]["water_class"] = 95
    q = parse_spatial_query("Find areas suitable for cotton near water",
                            config=custom)
    cotton = [c for c in q.conditions
              if c.condition_type is ConditionType.CROP_SUITABILITY][0]
    water = [c for c in q.conditions
             if c.condition_type is ConditionType.WATER_PROXIMITY][0]
    assert cotton.parameters["min_class"] == 4
    assert water.parameters["distance_m"] == 250
    assert water.parameters["water_class"] == 95


# =========================================================================== #
# 10. router integration + Phase 7/8 regression
# =========================================================================== #
def test_router_identifies_spatial_queries_and_fills_conditions():
    parsed = parse_query("Find areas suitable for cotton near water.")
    assert parsed.intent is Intent.SPATIAL_QUERY
    assert parsed.is_actionable and not parsed.is_planned
    assert parsed.required_context == ("roi",)
    assert len(parsed.conditions) == 2
    assert parsed.to_dict()["conditions"][0]["condition_type"] == "crop_suitability"


@pytest.mark.parametrize("query,intent", [
    ("Can I grow cotton here?", Intent.CROP_SUITABILITY),      # Phase 8 unchanged
    ("Is this land suitable for cotton", Intent.CROP_SUITABILITY),
    ("Can I grow rice here?", Intent.CROP_SUITABILITY),
    ("What is the NDVI of this area?", Intent.NDVI_ROI_STATS),  # Phase 6 unchanged
    ("Show flood areas.", Intent.FLOOD_CHANGE),                 # Phase 7 unchanged
])
def test_existing_intents_are_not_stolen(query, intent):
    parsed = parse_query(query)
    assert parsed.intent is intent
    assert parsed.conditions == ()          # conditions belong to spatial queries


def test_ambiguous_verdict_question_reaches_the_spatial_parser():
    parsed = parse_query("Can I grow cotton near water?")
    assert parsed.intent is Intent.SPATIAL_QUERY
    assert any(c.condition_type is ConditionType.WATER_PROXIMITY
               for c in parsed.conditions)


def test_unsupported_question_routes_to_the_spatial_parser_and_stops():
    parsed = parse_query("Find cotton land with reliable irrigation.")
    assert parsed.intent is Intent.SPATIAL_QUERY
    assert parsed.conditions
    assert all(c.status is ConditionStatus.UNSUPPORTED
               for c in parsed.conditions)


def test_route_reaches_the_engine_now_that_it_exists():
    """Checkpoint C registered the engine: the query now reaches it and degrades
    to NEEDS_ROI instead of "not available yet" -- still never fabricated."""
    from analyses import AnalysisContext, Status
    from analyses.registry import route

    execution = route("Find areas suitable for cotton near water.",
                      AnalysisContext(roi=None))
    assert execution.intent is Intent.SPATIAL_QUERY
    assert execution.status is Status.NEEDS_ROI
    assert execution.result is None


# =========================================================================== #
# 11. CORRECTION 1 -- WATER is not WATER_PROXIMITY
# =========================================================================== #
def test_excluding_water_is_the_water_class_not_a_distance():
    """'excluding water' = NOT class 80, NOT 'farther than 1 km from water'."""
    q = parse_spatial_query("Find cropland excluding water")
    assert q.is_executable
    water = [c for c in q.conditions if c.condition_type is ConditionType.WATER]
    assert len(water) == 1
    assert water[0].negate is True
    assert water[0].parameters["classes"] == [80]
    assert ConditionType.WATER_PROXIMITY not in types_of(
        "Find cropland excluding water")
    assert "distance" not in water[0].label.lower()


def test_near_water_is_proximity_not_the_water_class():
    q = parse_spatial_query("Find cropland near water")
    prox = [c for c in q.conditions
            if c.condition_type is ConditionType.WATER_PROXIMITY]
    assert len(prox) == 1
    assert prox[0].negate is False
    assert prox[0].parameters["distance_m"] == int(
        load_spatial_config("conditions")["water"]["default_proximity_m"])
    assert ConditionType.WATER not in types_of("Find cropland near water")


@pytest.mark.parametrize("query", [
    "Find cropland not water",
    "Find cropland and not water",
    "Find cropland outside water",
    "Find cotton areas but not water",
])
def test_negated_water_forms(query):
    q = parse_spatial_query(query)
    water = [c for c in q.conditions if c.condition_type is ConditionType.WATER]
    assert len(water) == 1 and water[0].negate is True
    assert ConditionType.WATER_PROXIMITY not in [c.condition_type
                                                 for c in q.conditions]


def test_not_water_is_not_not_water_proximity():
    """The semantic regression the correction is about."""
    a = parse_spatial_query("Find cropland excluding water")
    b = parse_spatial_query("Find cropland excluding areas near water")
    types_a = [c.condition_type for c in a.conditions]
    types_b = [c.condition_type for c in b.conditions]
    assert ConditionType.WATER in types_a
    assert ConditionType.WATER_PROXIMITY not in types_a
    assert ConditionType.WATER_PROXIMITY in types_b
    assert a.expression() != b.expression()
    assert "water(class [80])" in a.expression()
    assert "water_proximity(" in b.expression()


def test_bare_water_is_the_water_class():
    q = parse_spatial_query("Find water")
    assert [c.condition_type for c in q.conditions] == [ConditionType.WATER]
    assert q.conditions[0].parameters["classes"] == [80]
    assert q.conditions[0].negate is False


def test_water_condition_states_what_it_is_not():
    c = one("Find water")
    joined = c.interpretation.lower()
    assert "not flooding" in joined and "not irrigation" in joined


# =========================================================================== #
# 12. CORRECTION 2 -- explicit irrigation blocks verdict queries too
# =========================================================================== #
@pytest.mark.parametrize("query", [
    "Can I grow cotton with irrigation?",
    "Can I grow cotton if irrigation is available?",
    "Is this suitable for irrigated cotton?",
    "Find cotton land with reliable irrigation",
    "Find cotton areas with irrigation",
])
def test_irrigation_is_detected_in_both_question_forms(query):
    blocked = detect_unsupported_requirements(query)
    assert [c.parameters["topic"] for c in blocked] == ["irrigation"]


@pytest.mark.parametrize("query", [
    "Can I grow cotton with irrigation?",
    "Is this suitable for irrigated cotton?",
    "Find cotton areas with irrigation",
])
def test_irrigation_blocks_execution_before_any_engine_runs(query):
    """No rainfed verdict may be returned for an irrigation question."""
    from analyses import AnalysisContext, Status
    from analyses.registry import route

    # no ROI at all: the block must be reported, not silently deferred
    execution = route(query, AnalysisContext(roi=None))
    assert execution.status is Status.UNSUPPORTED_CONDITION
    assert execution.result is None
    assert "rainfed" in (execution.message or "").lower()
    assert "irrigation" in (execution.message or "").lower()


def test_irrigation_is_never_turned_into_near_water():
    q = parse_query("Can I grow cotton with irrigation?")
    assert ConditionType.WATER_PROXIMITY not in [c.condition_type
                                                 for c in q.conditions]
    blocked = q.blocked_by
    assert blocked and blocked[0].parameters["topic"] == "irrigation"
    assert blocked[0].parameters["nearest_supported"] == "water_proximity"
    assert "not measured" in blocked[0].note.lower()
    assert "near water" not in blocked[0].note.lower().replace(
        "proximity to mapped surface water", "")


def test_a_plain_cotton_verdict_is_not_blocked():
    parsed = parse_query("Can I grow cotton here?")
    assert parsed.blocked_by == ()
    assert parsed.intent is Intent.CROP_SUITABILITY


def test_flood_keeps_its_own_path_and_message():
    """Flood is deliberately NOT a verdict blocker: Phase 7 owns it."""
    from analyses import AnalysisContext
    from analyses.registry import route

    parsed = parse_query("Show flood areas.")
    assert parsed.blocked_by == ()
    assert parsed.intent is Intent.FLOOD_CHANGE
    assert "flood" in route("Show flood areas.",
                            AnalysisContext(roi=None)).message.lower()


# =========================================================================== #
# 13. CORRECTION 3 -- the re-routed "Where …?" keeps the Phase 8 semantics
# =========================================================================== #
def test_where_question_preserves_the_phase8_screening_contract():
    parsed = parse_query("Where can I grow cotton in this region?")
    assert parsed.intent is Intent.SPATIAL_QUERY
    condition = parsed.conditions[0]
    assert condition.condition_type is ConditionType.CROP_SUITABILITY
    assert condition.parameters == {"crop": "cotton", "min_class": 3,
                                    "scenario": "rainfed"}
    # it delegates to the Phase 8 engine -- it does not re-implement it
    assert condition.required_analysis == "crop_suitability"
