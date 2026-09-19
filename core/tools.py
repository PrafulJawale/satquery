"""core/tools.py -- Tool Contract Layer for SatQuery AI.

This module provides a minimal adapter between a future LLM planner and the
existing deterministic analysis engine. It does NOT replace or rewrite any
existing analysis logic. It only wraps the existing registry/router/execution
path with a tool-oriented interface.

The existing path remains:
    ToolCall -> ToolAdapter -> analyses.registry.route() -> AnalysisExecution
                                                              -> EvidencePackage
                                                              -> ToolResult

No LLM, no agent framework, no new dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.router import Intent, QueryIntent, parse_query
from core.evidence import EvidencePackage, json_safe
from analyses.base import AnalysisContext, AnalysisExecution, Status
from analyses.registry import route, get_spec, available_specs, REGISTRY


__all__ = [
    "ToolCall",
    "ToolError",
    "ToolResult",
    "ToolAdapter",
    "SUPPORTED_TOOLS",
    "tool_name_to_intent",
    "intent_to_tool_name",
    "validate_tool_arguments",
    "merge_structured_args",
]


# --------------------------------------------------------------------------- #
# Tool <-> Intent Mapping
# --------------------------------------------------------------------------- #

# Mapping from LLM-facing tool names to existing SatQuery Intents.
# Only tools with a corresponding implemented AnalysisSpec are included.
# Planned intents (FLOOD_CHANGE, TEMPORAL_NDWI) are deliberately excluded.
SUPPORTED_TOOLS: Tuple[str, ...] = (
    "compute_ndvi",
    "compute_ndwi",
    "temporal_compare",
    "spatial_query",
    "multi_condition_query",
    "crop_suitability",
    # "inspect_raster"  -- not exposed: no stable execution ID mechanism yet
    # "get_evidence"    -- not exposed: no stable execution reference mechanism yet
)

# Explicit mapping: tool name -> Intent enum
TOOL_TO_INTENT: Dict[str, Intent] = {
    "compute_ndvi": Intent.NDVI_ROI_STATS,
    "compute_ndwi": Intent.NDWI_ROI_STATS,
    "temporal_compare": Intent.NDVI_CHANGE_ROI,   # also covers TEMPORAL_COMPARISON, VEGETATION_CHANGE
    "spatial_query": Intent.SPATIAL_QUERY,
    "multi_condition_query": Intent.MULTI_CONDITION,
    "crop_suitability": Intent.CROP_SUITABILITY,
}

# Reverse mapping for serialization/debugging
INTENT_TO_TOOL: Dict[Intent, str] = {v: k for k, v in TOOL_TO_INTENT.items()}


def tool_name_to_intent(tool_name: str) -> Optional[Intent]:
    """Map a tool name to its corresponding Intent, or None if unknown."""
    return TOOL_TO_INTENT.get(tool_name)


def intent_to_tool_name(intent: Intent) -> Optional[str]:
    """Map an Intent to its primary tool name, or None if not exposed as a tool."""
    return INTENT_TO_TOOL.get(intent)


# --------------------------------------------------------------------------- #
# ToolCall
# --------------------------------------------------------------------------- #

@dataclass
class ToolCall:
    """A single tool invocation requested by an LLM planner.

    This is the input contract. The LLM produces this; the adapter validates
    and executes it against the existing deterministic engine.
    """

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.call_id is None:
            self.call_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization for LLM communication."""
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Deserialize from LLM-produced JSON."""
        return cls(
            name=str(data.get("name", "")),
            arguments=dict(data.get("arguments", {}) or {}),
            call_id=data.get("call_id"),
        )


# --------------------------------------------------------------------------- #
# ToolError
# --------------------------------------------------------------------------- #

@dataclass
class ToolError:
    """Structured error for tool execution failures.

    Codes align with existing SatQuery Status values where possible.
    """

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            out["details"] = self.details
        return out


# Standard error codes (matching existing SatQuery semantics where possible)
ERROR_CODES = {
    "UNKNOWN_TOOL": "The requested tool is not recognized or not available.",
    "INVALID_ARGUMENTS": "The tool arguments are invalid or malformed.",
    "MISSING_ARGUMENT": "A required argument is missing.",
    "UNSUPPORTED": "The requested operation is not supported by the current engine.",
    "NEEDS_ROI": "An area of interest (ROI) must be selected first.",
    "NEEDS_NDVI_CONFIRMATION": "Band mapping must be confirmed before running index analysis.",
    "NEEDS_TWO_DATES": "Two dated acquisitions are required for temporal comparison.",
    "NEEDS_THRESHOLD": "A spectral threshold was requested without a numeric value.",
    "INSUFFICIENT_DATA": "Critical input data is missing or coverage is insufficient.",
    "NO_TEMPORAL_OVERLAP": "The two scenes do not share common ground or the ROI is not covered by both.",
    "NO_VALID_PIXELS": "No valid pixels were found for the requested analysis.",
    "EXECUTION_ERROR": "The analysis failed due to an internal error.",
}


# --------------------------------------------------------------------------- #
# ToolResult
# --------------------------------------------------------------------------- #

@dataclass
class ToolResult:
    """The result envelope for a tool execution.

    Wraps the existing AnalysisExecution and EvidencePackage without
    replacing them. The LLM receives this structured result.
    """

    call_id: Optional[str]
    tool_name: str
    status: str                           # matches Status enum values
    result: Optional[Dict[str, Any]]      # AnalysisExecution.result.to_dict()
    evidence: Optional[Dict[str, Any]]    # EvidencePackage.to_dict() when available
    warnings: List[str] = field(default_factory=list)
    error: Optional[ToolError] = None
    message: str = ""                     # Human-readable message from AnalysisExecution

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization for LLM consumption."""
        out: Dict[str, Any] = {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "result": self.result,
            "evidence": self.evidence,
            "warnings": list(self.warnings),
            "message": self.message,
        }
        if self.error:
            out["error"] = self.error.to_dict()
        return out


# --------------------------------------------------------------------------- #
# ToolAdapter
# --------------------------------------------------------------------------- #

class ToolAdapter:
    """Adapts ToolCall -> existing AnalysisExecution -> ToolResult.

    This is the ONLY execution path. It delegates to analyses.registry.route()
    which validates context, checks requirements, and runs the deterministic engine.
    """

    def __init__(self, context: AnalysisContext) -> None:
        self.context = context

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call and return a ToolResult."""
        call_id = tool_call.call_id
        tool_name = tool_call.name

        # 1. Unknown tool
        intent = tool_name_to_intent(tool_name)
        if intent is None:
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=Status.UNSUPPORTED.value,
                result=None,
                evidence=None,
                warnings=[],
                message="",
                error=ToolError(
                    code="UNKNOWN_TOOL",
                    message=f"Unknown tool: '{tool_name}'. Supported tools: {', '.join(SUPPORTED_TOOLS)}",
                ),
            )

        # 2. Validate structured arguments
        validation_error = validate_tool_arguments(tool_name, tool_call.arguments)
        if validation_error:
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=Status.ERROR.value,
                result=None,
                evidence=None,
                warnings=[],
                message="",
                error=ToolError(
                    code="INVALID_ARGUMENTS",
                    message=validation_error,
                    details={"arguments": tool_call.arguments},
                ),
            )

        # 3. Build the query string for the existing router, merging structured args
        query_string, merge_warning = merge_structured_args(
            tool_name, tool_call.arguments, self.context, intent
        )
        if query_string is None:
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=Status.ERROR.value,
                result=None,
                evidence=None,
                warnings=[],
                message="",
                error=ToolError(
                    code="INVALID_ARGUMENTS",
                    message=f"Could not build valid query string for tool '{tool_name}'",
                    details={"arguments": tool_call.arguments},
                ),
            )

        # 4. Route through existing registry (the deterministic path)
        try:
            execution = route(query_string, self.context)
        except Exception:
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                status=Status.ERROR.value,
                result=None,
                evidence=None,
                warnings=[],
                message="",
                error=ToolError(
                    code="EXECUTION_ERROR",
                    message="Internal analysis error",
                    details={"exception": "Internal analysis error"},
                ),
            )

        # 5. Convert AnalysisExecution -> ToolResult
        result = self._execution_to_tool_result(call_id, tool_name, execution)
        if merge_warning:
            # Add merge warning to result warnings
            result.warnings = list(result.warnings) + [merge_warning]
        return result

    def execute_batch(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls sequentially."""
        return [self.execute(tc) for tc in tool_calls]

    def _execution_to_tool_result(
        self,
        call_id: Optional[str],
        tool_name: str,
        execution: AnalysisExecution,
    ) -> ToolResult:
        """Convert AnalysisExecution to ToolResult, preserving evidence."""
        # Extract result dict (excludes pixel arrays by design)
        result_dict = None
        if execution.result is not None and hasattr(execution.result, "to_dict"):
            result_dict = execution.result.to_dict()
        elif isinstance(execution.result, dict):
            result_dict = execution.result

        # Extract evidence if available
        evidence_dict = None
        if execution.provenance and isinstance(execution.provenance, dict):
            # The execution.provenance carries the engine's provenance.
            # For full EvidencePackage, the UI builds it in analyses/evidence.py
            # from the AnalysisExecution. Here we pass what we have.
            evidence_dict = {
                "provenance": execution.provenance,
                "status": execution.status.value,
                "message": execution.message,
                "warnings": list(execution.warnings),
            }
            # If the result itself has provenance (e.g., MultiConditionResult),
            # merge it in
            if result_dict and "provenance" in result_dict:
                evidence_dict["result_provenance"] = result_dict["provenance"]
            if result_dict and "conditions" in result_dict:
                evidence_dict["conditions"] = result_dict["conditions"]

        # Map Status enum to string
        status_str = execution.status.value if isinstance(execution.status, Status) else str(execution.status)

        return ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            status=status_str,
            result=result_dict,
            evidence=evidence_dict,
            warnings=list(execution.warnings) if execution.warnings else [],
            message=execution.message or "",
            error=None,
        )


# --------------------------------------------------------------------------- #
# Structured argument validation
# --------------------------------------------------------------------------- #

def _validate_roi_arg(arg: Any) -> Optional[str]:
    """Validate ROI argument (GeoJSON-like dict or reference string)."""
    if isinstance(arg, str):
        # Reference to existing ROI in context
        return None
    if isinstance(arg, dict):
        # GeoJSON geometry or feature
        if "type" not in arg:
            return "ROI dict must have a 'type' field (GeoJSON geometry)"
        if arg["type"] not in ("Polygon", "MultiPolygon", "Feature", "FeatureCollection"):
            return f"ROI type must be Polygon, MultiPolygon, Feature, or FeatureCollection, got {arg['type']}"
        return None
    return "ROI must be a string reference or GeoJSON dict"


def _validate_date_arg(arg: Any, field_name: str) -> Optional[str]:
    """Validate date argument (ISO format string)."""
    if not isinstance(arg, str):
        return f"{field_name} must be a string in ISO format (YYYY-MM-DD)"
    import re
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", arg):
        return f"{field_name} must be in YYYY-MM-DD format"
    # Additional validation: check it's a valid date
    try:
        from datetime import date
        date.fromisoformat(arg)
    except ValueError:
        return f"{field_name} is not a valid date"
    return None


def _validate_threshold_arg(arg: Any) -> Optional[str]:
    """Validate threshold argument (numeric)."""
    if not isinstance(arg, (int, float)):
        return "threshold must be a number"
    if not (-10 <= arg <= 10):
        return "threshold should be in a reasonable range for spectral indices"
    return None


def _validate_crop_arg(arg: Any) -> Optional[str]:
    """Validate crop argument."""
    if not isinstance(arg, str):
        return "crop must be a string"
    supported = ("cotton",)
    if arg.lower() not in supported:
        return f"crop must be one of {supported}, got '{arg}'"
    return None


def _validate_condition_arg(arg: Any) -> Optional[str]:
    """Validate a single condition dict for multi_condition_query."""
    if not isinstance(arg, dict):
        return "condition must be an object"
    required = {"index", "operator", "threshold"}
    missing = required - set(arg.keys())
    if missing:
        return f"condition missing required fields: {missing}"
    # Validate index
    index = arg.get("index")
    if index not in ("ndvi", "ndwi"):
        return f"condition index must be 'ndvi' or 'ndwi', got '{index}'"
    # Validate operator
    operator = arg.get("operator")
    if operator not in (">", ">=", "<", "<="):
        return f"condition operator must be one of >, >=, <, <=, got '{operator}'"
    # Validate threshold
    err = _validate_threshold_arg(arg.get("threshold"))
    if err:
        return f"condition threshold: {err}"
    return None


def validate_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    """Validate structured arguments for a tool.

    Returns None if valid, error message string if invalid.
    """
    if tool_name not in SUPPORTED_TOOLS:
        return f"Unknown tool: '{tool_name}'"

    # Check for unknown properties
    schema = get_tool_schema(tool_name)
    if schema and not schema.get("additionalProperties", True):
        allowed = set(schema.get("properties", {}).keys())
        for key in arguments:
            if key not in allowed:
                return f"Unexpected argument: '{key}'. Allowed: {sorted(allowed)}"

    # Validate specific fields
    if "roi" in arguments:
        err = _validate_roi_arg(arguments["roi"])
        if err:
            return f"roi: {err}"

    if "date1" in arguments:
        err = _validate_date_arg(arguments["date1"], "date1")
        if err:
            return err

    if "date2" in arguments:
        err = _validate_date_arg(arguments["date2"], "date2")
        if err:
            return err

    if "threshold" in arguments:
        err = _validate_threshold_arg(arguments["threshold"])
        if err:
            return err

    if "crop" in arguments:
        err = _validate_crop_arg(arguments["crop"])
        if err:
            return err

    if "conditions" in arguments:
        conditions = arguments["conditions"]
        if not isinstance(conditions, list):
            return "conditions must be a list"
        for i, cond in enumerate(conditions):
            err = _validate_condition_arg(cond)
            if err:
                return f"conditions[{i}]: {err}"

    # Type checking for known properties
    properties = schema.get("properties", {}) if schema else {}
    for key, value in arguments.items():
        if key in properties:
            prop_schema = properties[key]
            expected_type = prop_schema.get("type")
            if expected_type == "string" and not isinstance(value, str):
                return f"Argument '{key}' must be a string, got {type(value).__name__}"
            if expected_type == "number" and not isinstance(value, (int, float)):
                return f"Argument '{key}' must be a number, got {type(value).__name__}"
            if expected_type == "integer" and not isinstance(value, int):
                return f"Argument '{key}' must be an integer, got {type(value).__name__}"
            if expected_type == "boolean" and not isinstance(value, bool):
                return f"Argument '{key}' must be a boolean, got {type(value).__name__}"
            if expected_type == "array" and not isinstance(value, list):
                return f"Argument '{key}' must be an array, got {type(value).__name__}"
            if expected_type == "object" and not isinstance(value, dict):
                return f"Argument '{key}' must be an object, got {type(value).__name__}"

    return None


def merge_structured_args(
    tool_name: str,
    arguments: Dict[str, Any],
    context: AnalysisContext,
    intent: Intent,
) -> Tuple[Optional[str], Optional[str]]:
    """Merge structured arguments into a query string for the router.

    Returns (query_string, warning_message).
    - query_string: the natural language query to pass to the router
    - warning_message: any conflict warnings between args and context

    Precedence: explicit validated args > existing context > engine defaults
    Conflicts between args and authoritative context are reported as warnings.
    """
    warnings = []
    query_parts = []

    # Start with the natural language query if provided
    query_text = arguments.get("query", "")
    if query_text:
        query_parts.append(query_text)

    # Handle ROI
    if "roi" in arguments:
        roi_arg = arguments["roi"]
        if isinstance(roi_arg, str):
            if roi_arg == "context":
                # Special signal to use the authoritative AnalysisContext ROI
                # No warning needed - this is the expected pattern for conversation inheritance
                if not context.has_roi:
                    warnings.append("roi inheritance requested but no ROI in context")
            else:
                # Reference to existing ROI - context should have it
                if not context.has_roi:
                    warnings.append("roi reference provided but no ROI in context")
        elif isinstance(roi_arg, dict):
            # GeoJSON geometry - would need to be converted to a selection
            # For now, we note it but the router works from context
            if context.has_roi:
                warnings.append("roi argument provided but context already has ROI; context takes precedence")
        # ROI is handled by context, not by query string

    # Handle dates for temporal_compare
    if tool_name == "temporal_compare":
        date1 = arguments.get("date1")
        date2 = arguments.get("date2")
        if date1 and date2:
            if context.has_temporal_pair:
                pair = context.temporal_pair
                if pair and pair.before and pair.after:
                    existing_d1 = pair.before.date.isoformat() if pair.before.date else None
                    existing_d2 = pair.after.date.isoformat() if pair.after.date else None
                    if existing_d1 != date1 or existing_d2 != date2:
                        warnings.append(f"date arguments ({date1}, {date2}) differ from context ({existing_d1}, {existing_d2}); context takes precedence")
            else:
                # Add date info to query to help router
                query_parts.append(f"between {date1} and {date2}")
        elif date1 or date2:
            warnings.append("both date1 and date2 must be provided together")

    # Handle threshold for multi_condition_query
    if tool_name == "multi_condition_query":
        threshold = arguments.get("threshold")
        conditions = arguments.get("conditions")
        if conditions:
            # Structured conditions - build a query fragment
            cond_strs = []
            for cond in conditions:
                idx = cond.get("index", "NDVI")
                op = cond.get("operator", ">")
                thresh = cond.get("threshold")
                cond_strs.append(f"{idx} {op} {thresh}")
            if cond_strs:
                query_parts.append("with " + " and ".join(cond_strs))
        elif threshold is not None:
            # Simple threshold - add to query
            query_parts.append(f"with threshold {threshold}")

    # Handle crop for crop_suitability
    if tool_name == "crop_suitability":
        crop = arguments.get("crop")
        if crop:
            query_parts.append(f"for {crop}")

    # If no query parts at all, use canonical
    if not query_parts:
        query_text = _canonical_query_for_intent_static(intent)
    else:
        query_text = " ".join(query_parts)

    warning_msg = "; ".join(warnings) if warnings else None
    return query_text, warning_msg


def _canonical_query_for_intent_static(intent: Intent) -> str:
    """Static version of canonical query generation."""
    canonical = {
        Intent.NDVI_ROI_STATS: "What is the NDVI of this area?",
        Intent.NDWI_ROI_STATS: "What is the NDWI of this area?",
        Intent.CROP_SUITABILITY: "Can I grow cotton here?",
        Intent.SPATIAL_QUERY: "Find cropland near water",
        Intent.NDVI_CHANGE_ROI: "Compare NDVI between these two dates",
        Intent.TEMPORAL_COMPARISON: "Compare before and after",
        Intent.VEGETATION_CHANGE: "How has the vegetation changed?",
        Intent.MULTI_CONDITION: "Find cropland with NDVI greater than 0.6",
    }
    return canonical.get(intent, "")


# --------------------------------------------------------------------------- #
# Convenience function for direct tool execution (for tests)
# --------------------------------------------------------------------------- #

def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    context: AnalysisContext,
    call_id: Optional[str] = None,
) -> ToolResult:
    """Execute a single tool by name with arguments.

    This is a convenience wrapper for tests and direct usage.
    """
    adapter = ToolAdapter(context)
    call = ToolCall(name=tool_name, arguments=arguments, call_id=call_id)
    return adapter.execute(call)


# --------------------------------------------------------------------------- #
# Introspection helpers
# --------------------------------------------------------------------------- #

def list_supported_tools() -> List[Dict[str, Any]]:
    """Return metadata about all supported tools for LLM function calling schemas."""
    tools = []
    for tool_name in SUPPORTED_TOOLS:
        intent = tool_name_to_intent(tool_name)
        spec = get_spec(intent) if intent else None
        tools.append({
            "name": tool_name,
            "intent": intent.value if intent else None,
            "title": spec.title if spec else "",
            "description": spec.description if spec else "",
            "available": spec.available if spec else False,
            "example_queries": list(spec.example_queries) if spec else [],
        })
    return tools


def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get the JSON schema for a specific tool's arguments.

    Schemas now include structured arguments where the deterministic engine
    can safely consume them. The 'query' field remains supported for
    backward compatibility and natural language queries.
    """
    if tool_name not in SUPPORTED_TOOLS:
        return None

    # Get description from registry spec
    intent = tool_name_to_intent(tool_name)
    spec = get_spec(intent) if intent else None
    description = spec.description if spec else ""

    schemas = {
        "compute_ndvi": {
            "type": "object",
            "description": description,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for the analysis",
                },
                "roi": {
                    "type": ["object", "string"],
                    "description": "Optional ROI as GeoJSON geometry or reference string",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "compute_ndwi": {
            "type": "object",
            "description": description,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for the analysis",
                },
                "roi": {
                    "type": ["object", "string"],
                    "description": "Optional ROI as GeoJSON geometry or reference string",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "temporal_compare": {
            "type": "object",
            "description": description,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for the analysis",
                },
                "roi": {
                    "type": ["object", "string"],
                    "description": "Optional ROI as GeoJSON geometry or reference string",
                },
                "date1": {
                    "type": "string",
                    "format": "date",
                    "description": "First acquisition date in YYYY-MM-DD format",
                },
                "date2": {
                    "type": "string",
                    "format": "date",
                    "description": "Second acquisition date in YYYY-MM-DD format",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "spatial_query": {
            "type": "object",
            "description": description,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for the analysis",
                },
                "roi": {
                    "type": ["object", "string"],
                    "description": "Optional ROI as GeoJSON geometry or reference string",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "multi_condition_query": {
            "type": "object",
            "description": description,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for the analysis",
                },
                "roi": {
                    "type": ["object", "string"],
                    "description": "Optional ROI as GeoJSON geometry or reference string",
                },
                "threshold": {
                    "type": "number",
                    "description": "Simple threshold value (for single condition)",
                },
                "conditions": {
                    "type": "array",
                    "description": "List of structured conditions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "string", "enum": ["ndvi", "ndwi"]},
                            "operator": {"type": "string", "enum": [">", ">=", "<", "<="]},
                            "threshold": {"type": "number"},
                        },
                        "required": ["index", "operator", "threshold"],
                        "additionalProperties": False,
                    },
                },
                "date1": {
                    "type": "string",
                    "format": "date",
                    "description": "First acquisition date for temporal conditions",
                },
                "date2": {
                    "type": "string",
                    "format": "date",
                    "description": "Second acquisition date for temporal conditions",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "crop_suitability": {
            "type": "object",
            "description": description,
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for the analysis",
                },
                "roi": {
                    "type": ["object", "string"],
                    "description": "Optional ROI as GeoJSON geometry or reference string",
                },
                "crop": {
                    "type": "string",
                    "enum": ["cotton"],
                    "description": "Crop to evaluate (only cotton supported)",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    }
    return schemas.get(tool_name)
