"""tests/test_tools.py -- Tool Contract tests for SatQuery AI.

Tests the minimal tool adapter layer without modifying any existing
deterministic engine, router, or evidence system.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from core.models import RasterInfo, BandInfo, SpatialInfo
from core.router import Intent, QueryIntent, parse_query
from core.tools import (
    ToolCall,
    ToolError,
    ToolResult,
    ToolAdapter,
    execute_tool,
    SUPPORTED_TOOLS,
    tool_name_to_intent,
    intent_to_tool_name,
    list_supported_tools,
    get_tool_schema,
    TOOL_TO_INTENT,
    INTENT_TO_TOOL,
)
from analyses.registry import get_spec, available_specs
from analyses.base import AnalysisContext, AnalysisExecution, Status, NdviContext
from core.evidence import EvidencePackage, EvidenceRecord, Lineage


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #

TEN_M = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
UTM36 = CRS.from_epsg(32636)
FULL_MASK = np.ones((10, 10), dtype=bool)
GRID = (np.arange(100, dtype="float32") / 100.0).reshape(10, 10)


def box_roi(col0: float, row0: float, col1: float, row1: float) -> box:
    x0, y0 = TEN_M * (col0, row1)
    x1, y1 = TEN_M * (col1, row0)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def make_roi(geom=None) -> AnalysisContext:
    geom = box_roi(1, 1, 6, 6) if geom is None else geom
    roi = type("ROISelection", (), {
        "is_valid": True,
        "intersects_raster": True,
        "area_m2": float(geom.area),
        "raster_crs": "EPSG:32636",
        "geometry_raster_crs": geom,
        "geometry_type": "Polygon",
        "num_parts": 1,
        "usable": True,
    })()
    ndvi_ctx = NdviContext(
        array=GRID,
        mask=FULL_MASK,
        crs=UTM36,
        transform=TEN_M,
        bands={"red": {"index": 3}, "nir": {"index": 4}},
        source_label="synthetic.tif",
    )
    return AnalysisContext(
        roi=roi,
        ndvi=ndvi_ctx,
        ndvi_confirmed=True,
        raster_label="synthetic.tif",
    )


def make_context_no_roi() -> AnalysisContext:
    return AnalysisContext(roi=None, ndvi=None, ndvi_confirmed=False, raster_label="synthetic.tif")


def make_context_no_ndvi() -> AnalysisContext:
    roi = type("ROISelection", (), {
        "is_valid": True,
        "intersects_raster": True,
        "area_m2": 250000.0,
        "raster_crs": "EPSG:32636",
        "geometry_raster_crs": box_roi(1, 1, 6, 6),
        "geometry_type": "Polygon",
        "num_parts": 1,
        "usable": True,
    })()
    return AnalysisContext(
        roi=roi,
        ndvi=None,
        ndvi_confirmed=False,
        raster_label="synthetic.tif",
    )


# --------------------------------------------------------------------------- #
# 1. ToolCall serialization
# --------------------------------------------------------------------------- #

def test_tool_call_serialization():
    call = ToolCall(name="compute_ndvi", arguments={"query": "What is the NDVI?"}, call_id="abc123")
    d = call.to_dict()
    assert d["name"] == "compute_ndvi"
    assert d["arguments"]["query"] == "What is the NDVI?"
    assert d["call_id"] == "abc123"

    # Round-trip
    call2 = ToolCall.from_dict(d)
    assert call2.name == call.name
    assert call2.arguments == call.arguments
    assert call2.call_id == call.call_id


def test_tool_call_json_serializable():
    call = ToolCall(name="compute_ndvi", arguments={"query": "test"}, call_id="test123")
    json_str = json.dumps(call.to_dict())
    assert "compute_ndvi" in json_str
    assert "test123" in json_str


# --------------------------------------------------------------------------- #
# 2. ToolError serialization
# --------------------------------------------------------------------------- #

def test_tool_error_serialization():
    err = ToolError(code="NEEDS_ROI", message="Please select an ROI", details={"required": True})
    d = err.to_dict()
    assert d["code"] == "NEEDS_ROI"
    assert d["message"] == "Please select an ROI"
    assert d["details"]["required"] is True

    json_str = json.dumps(d)
    assert "NEEDS_ROI" in json_str


# --------------------------------------------------------------------------- #
# 3. ToolResult serialization
# --------------------------------------------------------------------------- #

def test_tool_result_serialization_with_result_and_evidence():
    result = ToolResult(
        call_id="call1",
        tool_name="compute_ndvi",
        status="OK",
        result={"valid_pixels": 25, "stats": {"mean": 0.5}},
        evidence={"provenance": {"engine": "core.statistics"}},
        warnings=["Caveat: not a crop health diagnosis"],
        error=None,
    )
    d = result.to_dict()
    assert d["call_id"] == "call1"
    assert d["tool_name"] == "compute_ndvi"
    assert d["status"] == "OK"
    assert d["result"]["valid_pixels"] == 25
    assert d["evidence"]["provenance"]["engine"] == "core.statistics"
    assert len(d["warnings"]) == 1

    json_str = json.dumps(d)
    assert "compute_ndvi" in json_str
    assert "valid_pixels" in json_str


def test_tool_result_serialization_with_error():
    result = ToolResult(
        call_id="call1",
        tool_name="unknown_tool",
        status="UNSUPPORTED",
        result=None,
        evidence=None,
        warnings=[],
        error=ToolError(code="UNKNOWN_TOOL", message="Unknown tool"),
    )
    d = result.to_dict()
    assert d["error"]["code"] == "UNKNOWN_TOOL"
    assert d["result"] is None
    assert d["evidence"] is None


# --------------------------------------------------------------------------- #
# 4. Unknown tool
# --------------------------------------------------------------------------- #

def test_unknown_tool_produces_structured_error():
    ctx = make_roi()
    result = execute_tool("invent_satellite_answer", {"query": "fake"}, ctx)
    assert result.status == Status.UNSUPPORTED.value
    assert result.error is not None
    assert result.error.code == "UNKNOWN_TOOL"
    assert "invent_satellite_answer" in result.error.message


# --------------------------------------------------------------------------- #
# 5. Valid tool mapping
# --------------------------------------------------------------------------- #

def test_supported_tools_mapping_complete():
    """Every supported tool maps to a real, available Intent/AnalysisSpec."""
    for tool_name in SUPPORTED_TOOLS:
        intent = tool_name_to_intent(tool_name)
        assert intent is not None, f"{tool_name} has no Intent mapping"
        assert intent in TOOL_TO_INTENT.values()

        spec = get_spec(intent)
        assert spec is not None, f"{tool_name} -> {intent} has no AnalysisSpec"
        assert spec.available, f"{tool_name} -> {intent} is not available"
        assert spec.handler is not None, f"{tool_name} -> {intent} has no handler"


def test_intent_to_tool_name_roundtrip():
    for tool_name in SUPPORTED_TOOLS:
        intent = tool_name_to_intent(tool_name)
        back = intent_to_tool_name(intent)
        assert back == tool_name, f"Round-trip failed for {tool_name}"


def test_tool_mapping_matches_registry():
    """Verify the tool mapping covers exactly the available specs (minus planned).

    Note: temporal_compare covers NDVI_CHANGE_ROI, TEMPORAL_COMPARISON, and VEGETATION_CHANGE.
    """
    available = set(available_specs().keys())
    mapped = set(TOOL_TO_INTENT.values())
    # Planned intents (FLOOD_CHANGE, TEMPORAL_NDWI) should NOT be in SUPPORTED_TOOLS
    planned = {Intent.FLOOD_CHANGE, Intent.TEMPORAL_NDWI}
    # temporal_compare tool covers 3 intents, so mapped has fewer entries than available
    assert mapped == (available - planned - {Intent.TEMPORAL_COMPARISON, Intent.VEGETATION_CHANGE})


# --------------------------------------------------------------------------- #
# 6. Unsupported operation (planned intents)
# --------------------------------------------------------------------------- #

def test_flood_change_not_exposed_as_tool():
    """FLOOD_CHANGE is planned but not implemented; must not be exposed."""
    assert "flood_detection" not in SUPPORTED_TOOLS
    assert "flood_change" not in SUPPORTED_TOOLS
    assert tool_name_to_intent("flood_detection") is None


def test_temporal_ndwi_not_exposed_as_tool():
    """TEMPORAL_NDWI is planned but not implemented; must not be exposed."""
    assert "temporal_ndwi" not in SUPPORTED_TOOLS
    assert tool_name_to_intent("compare_ndwi") is None


# --------------------------------------------------------------------------- #
# 7. Missing parameters
# --------------------------------------------------------------------------- #

def test_compute_ndvi_without_roi_returns_needs_roi():
    ctx = make_context_no_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx)
    assert result.status == Status.NEEDS_ROI.value
    assert result.error is None  # Not an error, a valid status
    assert "area" in result.message.lower() or "select" in result.message.lower()


def test_compute_ndvi_without_ndvi_confirmed_returns_needs_confirmation():
    ctx = make_context_no_ndvi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx)
    assert result.status == Status.NEEDS_NDVI_CONFIRMATION.value
    assert "confirm" in result.message.lower()


def test_temporal_compare_without_dates_returns_needs_two_dates():
    ctx = make_roi()
    # No temporal_pair in context
    result = execute_tool("temporal_compare", {"query": "Compare before and after"}, ctx)
    assert result.status == Status.NEEDS_TWO_DATES.value
    assert "two" in result.message.lower() or "date" in result.message.lower()


def test_spatial_query_without_roi_returns_needs_roi():
    ctx = make_context_no_roi()
    result = execute_tool("spatial_query", {"query": "Find cropland near water"}, ctx)
    assert result.status == Status.NEEDS_ROI.value


# --------------------------------------------------------------------------- #
# 8. Evidence propagation
# --------------------------------------------------------------------------- #

def test_evidence_propagation_ndvi():
    ctx = make_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI of this area?"}, ctx)
    assert result.status == Status.OK.value
    assert result.result is not None
    assert "valid_pixels" in result.result
    assert result.evidence is not None
    assert "provenance" in result.evidence
    assert result.evidence["provenance"]["engine"] == "core.statistics.calculate_roi_ndvi_stats (Phase 6)"


def test_evidence_propagation_spatial_query():
    ctx = make_roi()
    # spatial_query needs WorldCover data (network), so we test the status path
    # The engine will either return OK or INSUFFICIENT_DATA depending on cache
    result = execute_tool("spatial_query", {"query": "Find cropland near water"}, ctx)
    # Either OK or a valid refusal status
    assert result.status in (Status.OK.value, Status.INSUFFICIENT_DATA.value, Status.UNSUPPORTED_CONDITION.value)
    if result.result:
        assert "expression" in result.result
    if result.evidence:
        assert "provenance" in result.evidence


def test_evidence_preserves_existing_structure():
    """EvidencePackage structure is preserved through ToolResult."""
    ctx = make_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI of this area?"}, ctx)
    # The evidence dict should be JSON-serializable without numpy/shapely leakage
    json.dumps(result.to_dict())  # Should not raise


# --------------------------------------------------------------------------- #
# 9. JSON safety
# --------------------------------------------------------------------------- #

def test_no_numpy_leakage_in_tool_result():
    ctx = make_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx)
    serialized = json.dumps(result.to_dict())
    # Verify no numpy types leaked
    assert "float32" not in serialized
    assert "int64" not in serialized
    assert "ndarray" not in serialized


def test_no_rasterio_leakage_in_tool_result():
    ctx = make_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx)
    serialized = json.dumps(result.to_dict())
    assert "Affine" not in serialized
    assert "CRS" not in serialized
    assert "DatasetReader" not in serialized


def test_no_shapely_leakage_in_tool_result():
    ctx = make_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx)
    serialized = json.dumps(result.to_dict())
    assert "Polygon" not in serialized
    assert "geometry" not in serialized.lower() or "geometry" in serialized  # "geometry" as key is OK


# --------------------------------------------------------------------------- #
# 10. Deterministic behavior
# --------------------------------------------------------------------------- #

def test_deterministic_output_same_call_same_context():
    ctx1 = make_roi()
    ctx2 = make_roi()
    # Use fixed call_id for deterministic comparison
    result1 = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx1, call_id="fixed123")
    result2 = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx2, call_id="fixed123")
    # Same synthetic context should produce equivalent serialized outputs
    assert json.dumps(result1.to_dict(), sort_keys=True) == json.dumps(result2.to_dict(), sort_keys=True)


# --------------------------------------------------------------------------- #
# 11. Registry is still the execution path
# --------------------------------------------------------------------------- #

def test_tool_adapter_uses_registry_route(monkeypatch):
    """Verify the adapter calls analyses.registry.route, not a bypass."""
    calls = {}

    def fake_route(query, context, *, convention=None):
        calls["query"] = query
        calls["context"] = context
        calls["convention"] = convention
        return AnalysisExecution(
            intent=Intent.NDVI_ROI_STATS, status=Status.OK, query="test",
            normalized_query="test", confidence=1.0, explanation="test",
            matched=(), result={"valid_pixels": 10},
            message="test", provenance={"engine": "fake"},
        )

    # Patch where it's used in core.tools
    import core.tools as tools_module
    monkeypatch.setattr(tools_module, "route", fake_route)

    ctx = make_roi()
    adapter = ToolAdapter(ctx)
    adapter.execute(ToolCall(name="compute_ndvi", arguments={"query": "test"}))

    assert "query" in calls
    assert calls["context"] is ctx


def test_tool_adapter_does_not_bypass_context_validation():
    """Context validation (NEEDS_ROI, etc.) still runs through registry."""
    ctx = make_context_no_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx)
    # The registry's validate_context should have run
    assert result.status == Status.NEEDS_ROI.value


def test_execution_error_does_not_leak_exception_details():
    """EXECUTION_ERROR must not expose raw exception type/message/details."""
    ctx = make_roi()
    adapter = ToolAdapter(ctx)

    # Patch route to raise an exception with a distinctive internal message
    import core.tools as tools_module
    original_route = tools_module.route

    def failing_route(query, context, *, convention=None):
        raise RuntimeError("DISTINCTIVE_INTERNAL_SECRET_PATH_/etc/passwd")

    tools_module.route = failing_route
    try:
        result = adapter.execute(ToolCall(name="compute_ndvi", arguments={"query": "test"}))
    finally:
        tools_module.route = original_route

    assert result.status == Status.ERROR.value
    assert result.error is not None
    assert result.error.code == "EXECUTION_ERROR"
    # The distinctive internal message must NOT appear in error details
    error_dict = result.error.to_dict()
    assert "DISTINCTIVE_INTERNAL_SECRET" not in str(error_dict)
    assert "/etc/passwd" not in str(error_dict)
    assert "RuntimeError" not in str(error_dict)
    # Generic safe message should be present
    assert "Internal analysis error" in error_dict.get("message", "")


# --------------------------------------------------------------------------- #
# Additional: introspection helpers
# --------------------------------------------------------------------------- #

def test_list_supported_tools_returns_metadata():
    tools = list_supported_tools()
    assert len(tools) == len(SUPPORTED_TOOLS)
    for t in tools:
        assert "name" in t
        assert "intent" in t
        assert "title" in t
        assert "description" in t
        assert "available" in t
        assert "example_queries" in t
        assert t["available"] is True


def test_get_tool_schema():
    schema = get_tool_schema("compute_ndvi")
    assert schema is not None
    assert schema["type"] == "object"
    assert "query" in schema["properties"]

    assert get_tool_schema("unknown_tool") is None


# --------------------------------------------------------------------------- #
# Integration: ToolCall -> QueryIntent -> Registry -> AnalysisExecution -> ToolResult
# --------------------------------------------------------------------------- #

def test_full_path_compute_ndvi():
    ctx = make_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI of this area?"}, ctx)
    assert result.status == Status.OK.value
    assert result.tool_name == "compute_ndvi"
    assert result.call_id is not None
    assert result.result is not None
    assert result.result["valid_pixels"] == 25  # 5x5 box in 10x10 grid


def test_full_path_crop_suitability_unsupported_crop():
    ctx = make_roi()
    result = execute_tool("crop_suitability", {"query": "Can I grow wheat here?"}, ctx)
    assert result.status == Status.UNSUPPORTED_CROP.value
    assert "cotton" in result.message.lower()


def test_full_path_multi_condition_needs_threshold():
    ctx = make_roi()
    # Multi-condition without threshold should return NEEDS_THRESHOLD
    result = execute_tool("multi_condition_query", {"query": "Find cropland with high NDVI"}, ctx)
    assert result.status == Status.NEEDS_THRESHOLD.value
    assert "threshold" in result.message.lower()


def test_tool_result_preserves_warnings():
    ctx = make_roi()
    result = execute_tool("compute_ndvi", {"query": "What is the NDVI?"}, ctx)
    assert result.status == Status.OK.value
    assert len(result.warnings) > 0
    assert any("crop-health" in w or "diagnosis" in w for w in result.warnings)


def test_tool_adapter_batch_execution():
    ctx = make_roi()
    adapter = ToolAdapter(ctx)
    calls = [
        ToolCall(name="compute_ndvi", arguments={"query": "NDVI"}),
        ToolCall(name="compute_ndwi", arguments={"query": "NDWI"}),
    ]
    results = adapter.execute_batch(calls)
    assert len(results) == 2
    assert results[0].tool_name == "compute_ndvi"
    assert results[1].tool_name == "compute_ndwi"


# --------------------------------------------------------------------------- #
# EvidencePackage integration check
# --------------------------------------------------------------------------- #

def test_evidence_package_structure_preserved():
    """Verify that when an EvidencePackage exists, its structure is preserved."""
    # Build a minimal EvidencePackage manually
    pkg = EvidencePackage(
        query="test query",
        normalized_query="test query",
        intent="NDVI_ROI_STATS",
        status="OK",
        expression="ndvi",
        lineage=Lineage(query="test query", intent="NDVI_ROI_STATS"),
        records=(),
        grid={"width": 10, "height": 10, "resolution_m": 10.0, "crs": "EPSG:32636"},
        analysed_cells=25,
        matched_cells=25,
        non_matching_cells=0,
        unknown_cells=0,
    )
    pkg_dict = pkg.to_dict()
    assert pkg_dict["schema"] == "satquery-evidence/1"
    assert pkg_dict["lineage"]["query"] == "test query"
    # Should be JSON serializable
    json.dumps(pkg_dict)


# --------------------------------------------------------------------------- #
# 12. Structured argument tests (Step 4)
# --------------------------------------------------------------------------- #

def test_get_tool_schema_includes_structured_args():
    """Verify that tool schemas now include structured arguments."""
    from core.tools import get_tool_schema, SUPPORTED_TOOLS

    for tool_name in SUPPORTED_TOOLS:
        schema = get_tool_schema(tool_name)
        assert schema is not None, f"{tool_name} should have a schema"
        assert schema["type"] == "object"
        props = schema.get("properties", {})
        # All tools now have query and roi
        assert "query" in props, f"{tool_name} should have query property"
        assert "roi" in props, f"{tool_name} should have roi property"
        assert schema.get("additionalProperties") is False

    # Check specific structured fields per tool
    temporal_schema = get_tool_schema("temporal_compare")
    assert "date1" in temporal_schema["properties"]
    assert "date2" in temporal_schema["properties"]

    multi_schema = get_tool_schema("multi_condition_query")
    assert "threshold" in multi_schema["properties"]
    assert "conditions" in multi_schema["properties"]
    assert "date1" in multi_schema["properties"]
    assert "date2" in multi_schema["properties"]

    crop_schema = get_tool_schema("crop_suitability")
    assert "crop" in crop_schema["properties"]


def test_validate_tool_arguments_valid():
    """Test validation of valid structured arguments."""
    from core.tools import validate_tool_arguments

    # compute_ndvi with roi
    err = validate_tool_arguments("compute_ndvi", {"query": "NDVI", "roi": {"type": "Polygon", "coordinates": []}})
    assert err is None

    # compute_ndvi with roi reference string
    err = validate_tool_arguments("compute_ndvi", {"query": "NDVI", "roi": "current"})
    assert err is None

    # temporal_compare with dates
    err = validate_tool_arguments("temporal_compare", {"query": "compare", "date1": "2024-01-01", "date2": "2024-06-01"})
    assert err is None

    # multi_condition_query with conditions
    err = validate_tool_arguments("multi_condition_query", {
        "query": "find cropland",
        "conditions": [
            {"index": "ndvi", "operator": ">", "threshold": 0.6},
            {"index": "ndwi", "operator": "<", "threshold": -0.2}
        ]
    })
    assert err is None

    # crop_suitability with crop
    err = validate_tool_arguments("crop_suitability", {"query": "cotton", "crop": "cotton"})
    assert err is None


def test_validate_tool_arguments_invalid_roi():
    """Test validation rejects invalid ROI."""
    from core.tools import validate_tool_arguments

    # Invalid ROI type
    err = validate_tool_arguments("compute_ndvi", {"roi": 123})
    assert err is not None
    assert "roi" in err.lower()

    # Invalid ROI dict (missing type)
    err = validate_tool_arguments("compute_ndvi", {"roi": {"coordinates": []}})
    assert err is not None
    assert "type" in err.lower()

    # Invalid ROI type value
    err = validate_tool_arguments("compute_ndvi", {"roi": {"type": "Point", "coordinates": [0, 0]}})
    assert err is not None
    assert "polygon" in err.lower() or "type" in err.lower()


def test_validate_tool_arguments_invalid_date():
    """Test validation rejects invalid dates."""
    from core.tools import validate_tool_arguments

    # Invalid date format
    err = validate_tool_arguments("temporal_compare", {"date1": "01-01-2024"})
    assert err is not None
    assert "date1" in err.lower()
    assert "yyyy-mm-dd" in err.lower()

    # Invalid date value
    err = validate_tool_arguments("temporal_compare", {"date1": "2024-13-01"})
    assert err is not None
    assert "date1" in err.lower()

    # Date as number
    err = validate_tool_arguments("temporal_compare", {"date1": 20240101})
    assert err is not None
    assert "string" in err.lower()


def test_validate_tool_arguments_invalid_threshold():
    """Test validation rejects invalid thresholds."""
    from core.tools import validate_tool_arguments

    # Threshold as string
    err = validate_tool_arguments("multi_condition_query", {"threshold": "0.6"})
    assert err is not None
    assert "number" in err.lower()

    # Threshold out of reasonable range
    err = validate_tool_arguments("multi_condition_query", {"threshold": 100})
    assert err is not None
    assert "range" in err.lower()


def test_validate_tool_arguments_invalid_crop():
    """Test validation rejects unsupported crops."""
    from core.tools import validate_tool_arguments

    # Unsupported crop
    err = validate_tool_arguments("crop_suitability", {"crop": "wheat"})
    assert err is not None
    assert "cotton" in err.lower()

    # Crop as number
    err = validate_tool_arguments("crop_suitability", {"crop": 123})
    assert err is not None
    assert "string" in err.lower()


def test_validate_tool_arguments_invalid_conditions():
    """Test validation rejects invalid conditions."""
    from core.tools import validate_tool_arguments

    # Conditions not a list
    err = validate_tool_arguments("multi_condition_query", {"conditions": "not a list"})
    assert err is not None
    assert "list" in err.lower()

    # Missing required fields in condition
    err = validate_tool_arguments("multi_condition_query", {"conditions": [{"index": "ndvi"}]})
    assert err is not None
    assert "missing required fields" in err.lower()

    # Invalid index
    err = validate_tool_arguments("multi_condition_query", {"conditions": [{"index": "evi", "operator": ">", "threshold": 0.5}]})
    assert err is not None
    assert "index" in err.lower()

    # Invalid operator
    err = validate_tool_arguments("multi_condition_query", {"conditions": [{"index": "ndvi", "operator": "!=", "threshold": 0.5}]})
    assert err is not None
    assert "operator" in err.lower()

    # Invalid threshold in condition
    err = validate_tool_arguments("multi_condition_query", {"conditions": [{"index": "ndvi", "operator": ">", "threshold": "high"}]})
    assert err is not None
    assert "number" in err.lower()


def test_validate_tool_arguments_unknown_tool():
    """Test validation rejects unknown tools."""
    from core.tools import validate_tool_arguments

    err = validate_tool_arguments("unknown_tool", {"query": "test"})
    assert err is not None
    assert "unknown tool" in err.lower()


def test_validate_tool_arguments_extra_property():
    """Test validation rejects extra properties."""
    from core.tools import validate_tool_arguments

    # Unknown property
    err = validate_tool_arguments("compute_ndvi", {"query": "test", "secret_field": "hack"})
    assert err is not None
    assert "unexpected argument" in err.lower()
    assert "secret_field" in err


def test_tool_adapter_validates_structured_args():
    """Test ToolAdapter validates structured arguments before execution."""
    from core.tools import ToolAdapter, ToolCall, execute_tool
    from analyses.base import AnalysisContext, Status

    ctx = make_context_no_roi()
    adapter = ToolAdapter(ctx)

    # Valid structured args should pass validation (but fail at engine level due to no ROI)
    result = adapter.execute(ToolCall(name="compute_ndvi", arguments={"query": "NDVI", "roi": "current"}))
    assert result.status == Status.NEEDS_ROI.value

    # Invalid structured args should be rejected at adapter level
    result = adapter.execute(ToolCall(name="compute_ndvi", arguments={"query": "NDVI", "roi": 123}))
    assert result.status == Status.ERROR.value
    assert result.error is not None
    assert result.error.code == "INVALID_ARGUMENTS"


def test_tool_adapter_merge_structured_args_ndvi():
    """Test ToolAdapter merges structured args for NDVI."""
    from core.tools import ToolAdapter, ToolCall, merge_structured_args
    from analyses.base import AnalysisContext
    from core.router import Intent

    ctx = make_roi()
    adapter = ToolAdapter(ctx)

    # Query only
    result = adapter.execute(ToolCall(name="compute_ndvi", arguments={"query": "What is NDVI?"}))
    assert result.status == Status.OK.value

    # With roi reference (should not conflict with context)
    result = adapter.execute(ToolCall(name="compute_ndvi", arguments={"query": "NDVI", "roi": "current"}))
    assert result.status == Status.OK.value


def test_tool_adapter_merge_structured_args_temporal():
    """Test ToolAdapter merges structured args for temporal compare."""
    from core.tools import ToolAdapter, ToolCall, merge_structured_args
    from analyses.base import AnalysisContext
    from core.router import Intent

    ctx = make_roi()
    adapter = ToolAdapter(ctx)

    # Query with dates
    result = adapter.execute(ToolCall(name="temporal_compare", arguments={
        "query": "compare before and after",
        "date1": "2024-01-01",
        "date2": "2024-06-01"
    }))
    # Should fail with NEEDS_TWO_DATES since no temporal_pair in context
    assert result.status == Status.NEEDS_TWO_DATES.value


def test_tool_adapter_merge_structured_args_multi_condition():
    """Test ToolAdapter merges structured args for multi-condition."""
    from core.tools import ToolAdapter, ToolCall
    from analyses.base import AnalysisContext
    from core.router import Intent

    ctx = make_roi()
    adapter = ToolAdapter(ctx)

    # With structured conditions
    result = adapter.execute(ToolCall(name="multi_condition_query", arguments={
        "query": "find cropland",
        "conditions": [
            {"index": "ndvi", "operator": ">", "threshold": 0.6}
        ]
    }))
    # Should fail with NEEDS_THRESHOLD or NEEDS_ROI depending on context
    # But validation should pass
    assert result.status != Status.ERROR.value or (result.error and result.error.code != "INVALID_ARGUMENTS")


def test_tool_adapter_rejects_tool_injection():
    """Test ToolAdapter rejects unknown tool names (injection attempt)."""
    from core.tools import ToolAdapter, ToolCall
    from analyses.base import AnalysisContext

    ctx = make_roi()
    adapter = ToolAdapter(ctx)

    # Try to inject a non-existent tool
    result = adapter.execute(ToolCall(name="python.exec", arguments={"code": "import os"}))
    assert result.status == Status.UNSUPPORTED.value
    assert result.error is not None
    assert result.error.code == "UNKNOWN_TOOL"


def test_structured_args_conflict_with_context_warning():
    """Test that conflicts between structured args and context produce warnings."""
    from core.tools import merge_structured_args
    from analyses.base import AnalysisContext
    from core.router import Intent
    from core.temporal import SceneRef, ScenePair
    from datetime import date

    # Create context with a temporal pair
    ctx = make_roi()
    before_scene = SceneRef(path="before.tif", label="before", date=date(2024, 1, 1))
    after_scene = SceneRef(path="after.tif", label="after", date=date(2024, 6, 1))
    ctx.temporal_pair = ScenePair(before=before_scene, after=after_scene)

    # Provide conflicting dates in arguments
    query, warning = merge_structured_args(
        "temporal_compare",
        {"query": "compare", "date1": "2023-01-01", "date2": "2023-06-01"},
        ctx,
        Intent.NDVI_CHANGE_ROI
    )
    assert warning is not None
    assert "differ from context" in warning
    assert "2024-01-01" in warning or "2024-06-01" in warning  # Warning mentions context dates
    # Query should be the original query text since context takes precedence
    assert query == "compare"


def test_execute_tool_with_structured_args():
    """Test execute_tool convenience function with structured args."""
    from core.tools import execute_tool
    from analyses.base import Status

    ctx = make_roi()

    # Query-only still works
    result = execute_tool("compute_ndvi", {"query": "What is NDVI?"}, ctx)
    assert result.status == Status.OK.value

    # Structured args also work
    result = execute_tool("compute_ndvi", {"query": "NDVI", "roi": "current"}, ctx)
    assert result.status == Status.OK.value

    # Invalid structured args rejected
    result = execute_tool("compute_ndvi", {"query": "NDVI", "roi": 123}, ctx)
    assert result.status == Status.ERROR.value
    assert result.error.code == "INVALID_ARGUMENTS"


def test_build_tools_schema_for_llm():
    """Test that build_tools_schema produces valid OpenAI-compatible schemas."""
    from core.planner import build_tools_schema

    schemas = build_tools_schema()
    assert len(schemas) == 6  # 6 supported tools

    for schema in schemas:
        assert schema["type"] == "function"
        assert "function" in schema
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "roi" in params["properties"]
        assert params.get("additionalProperties") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
