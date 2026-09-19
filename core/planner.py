"""core/planner.py -- LLM Planner Boundary for SatQuery AI.

This module provides a minimal, provider-neutral boundary between an LLM planner
and the existing deterministic SatQuery engine. It does NOT replace the router,
the engines, or the evidence system. It only wraps the existing ToolAdapter
with an LLM-facing planning interface.

Architecture:
    User Query
        ↓
    LLMPlanner.plan()  -- understands intent, extracts params, builds ToolCall
        ↓
    validate_tool_call()  -- rejects unknown tools, malformed args
        ↓
    ToolAdapter.execute()  -- existing deterministic path
        ↓
    ToolResult

The LLM may ONLY:
    * understand natural language
    * select an existing tool from SUPPORTED_TOOLS
    * extract parameters from the query
    * construct a valid ToolCall
    * ask for clarification
    * interpret returned deterministic evidence

The LLM MUST NOT:
    * calculate indices or statistics
    * invent numerical observations, dates, thresholds, band mappings
    * override UNSUPPORTED, NEEDS_ROI, NEEDS_NDVI_CONFIRMATION, etc.
    * execute tools directly
    * access raster objects
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from collections import deque

from core.tools import (
    ToolCall,
    ToolError,
    ToolResult,
    ToolAdapter,
    SUPPORTED_TOOLS,
    tool_name_to_intent,
    get_tool_schema,
    list_supported_tools,
    ERROR_CODES,
    validate_tool_arguments,
)
from analyses.base import AnalysisContext, Status


# --------------------------------------------------------------------------- #
# Conversation State (Step 5: Multi-turn context)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in the conversation."""
    user_query: str
    tool_name: Optional[str] = None
    intent: Optional[str] = None
    status: Optional[str] = None
    crop: Optional[str] = None
    has_roi: bool = False
    dates: Tuple[Optional[str], Optional[str]] = (None, None)


@dataclass
class ConversationState:
    """Lightweight in-memory conversation state for multi-turn context.

    Stores only compact metadata needed for follow-up resolution.
    Does NOT store raster arrays, evidence packages, or large objects.
    """

    # Bounded history of recent turns (max 5 turns)
    _turns: deque = field(default_factory=lambda: deque(maxlen=5))

    # Current authoritative context reference (not copied, just tracked)
    current_roi_available: bool = False
    current_dates: Tuple[Optional[str], Optional[str]] = (None, None)
    current_crop: Optional[str] = None
    current_intent: Optional[str] = None

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a completed turn to history."""
        self._turns.append(turn)

        # Update current context from successful tool calls
        if turn.status in ("OK", "PARTIAL_DATA", "INSUFFICIENT_DATA", "UNSUPPORTED_CROP"):
            if turn.has_roi:
                self.current_roi_available = True
            if turn.crop:
                self.current_crop = turn.crop
            if turn.dates != (None, None):
                self.current_dates = turn.dates
            if turn.intent:
                self.current_intent = turn.intent

    def get_recent_turns(self, n: int = 3) -> List[ConversationTurn]:
        """Get the most recent N turns."""
        return list(self._turns)[-n:]

    def get_context_summary(self) -> Dict[str, Any]:
        """Get a compact summary for LLM context."""
        recent = self.get_recent_turns(3)
        return {
            "recent_turns": [
                {
                    "query": t.user_query,
                    "tool": t.tool_name,
                    "intent": t.intent,
                    "status": t.status,
                    "crop": t.crop,
                    "has_roi": t.has_roi,
                }
                for t in recent
            ],
            "current_roi_available": self.current_roi_available,
            "current_dates": list(self.current_dates),
            "current_crop": self.current_crop,
            "current_intent": self.current_intent,
        }

    def resolve_references(self, query: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Resolve conversational references in the query.

        Returns (resolved_query, inherited_args) where inherited_args contains
        context that should be applied to tool arguments.

        Precedence: explicit current query > conversation context > ask for clarification
        """
        q_lower = query.lower().strip()
        inherited = {}

        # Check for spatial references
        spatial_refs = ["this area", "here", "same area", "same place", "that area", "the area"]
        has_spatial_ref = any(ref in q_lower for ref in spatial_refs)

        # Check for crop references
        crop_refs = ["same crop", "that crop", "the crop"]
        has_crop_ref = any(ref in q_lower for ref in crop_refs)

        # Check for temporal references
        temporal_refs = [
            "same dates", "same period", "same timeframe",
            "previous date", "earlier date", "earlier",
            "later date", "later", "the other date",
            "compare it with the previous", "compare with the previous",
            "compare with the other"
        ]
        has_temporal_ref = any(ref in q_lower for ref in temporal_refs)

        # Check for analysis type references
        analysis_refs = ["what about", "same analysis", "compare it", "show the same"]
        has_analysis_ref = any(ref in q_lower for ref in analysis_refs)

        # Only inherit if we have context and the query seems to reference it
        if has_spatial_ref and self.current_roi_available:
            inherited["roi"] = "context"  # Signal to use context ROI

        if has_crop_ref and self.current_crop:
            inherited["crop"] = self.current_crop

        if has_temporal_ref and self.current_dates != (None, None):
            # Inherit previous dates when temporal reference is detected
            inherited["date1"] = self.current_dates[0]
            inherited["date2"] = self.current_dates[1]

        # For analysis type changes (NDVI -> NDWI, etc.), don't inherit intent
        # but do inherit spatial context if referenced

        return query, inherited if inherited else None

    def apply_inherited_args(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Apply inherited arguments from conversation state to tool arguments.

        Precedence: explicit current arguments > inherited conversation context
        Only applies safe, deterministic context (ROI, dates, crop).
        """
        # Resolve references from the user query (already done in plan(), but we re-check
        # here to ensure the inherited args match the actual tool being called)
        # We don't have the query here, so we use the stored current context directly

        result = dict(arguments)  # Copy to avoid mutation

        # Apply ROI inheritance if not explicitly provided and context has ROI
        if "roi" not in result and self.current_roi_available:
            # Check if this tool type can use ROI
            roi_tools = ("compute_ndvi", "compute_ndwi", "temporal_compare",
                         "spatial_query", "multi_condition_query", "crop_suitability")
            if tool_name in roi_tools:
                result["roi"] = "context"

        # Apply date inheritance for temporal tools if not explicitly provided
        if tool_name == "temporal_compare":
            if "date1" not in result and "date2" not in result and self.current_dates != (None, None):
                if self.current_dates[0] and self.current_dates[1]:
                    result["date1"] = self.current_dates[0]
                    result["date2"] = self.current_dates[1]

        # Apply crop inheritance for crop_suitability if not explicitly provided
        if tool_name == "crop_suitability" and "crop" not in result and self.current_crop:
            result["crop"] = self.current_crop

        return result

    def clear(self) -> None:
        """Clear all conversation state."""
        self._turns.clear()
        self.current_roi_available = False
        self.current_dates = (None, None)
        self.current_crop = None
        self.current_intent = None


# --------------------------------------------------------------------------- #
# Provider Interface (minimal, OpenAI-compatible)
# --------------------------------------------------------------------------- #

@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface for an LLM provider. Implementations must be stateless."""

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call the LLM and return the raw response dict.

        The response must contain at least:
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": str | None,
                        "tool_calls": [
                            {
                                "id": str,
                                "type": "function",
                                "function": {"name": str, "arguments": str}
                            }
                        ] | None
                    }
                }
            ]
        }
        """
        ...


class MockLLMProvider:
    """Deterministic mock provider for testing and offline development.

    Returns a valid ToolCall for recognized intents, or a clarification
    when the query is ambiguous. Never executes tools.
    """

    def __init__(self, *, force_tool: Optional[str] = None, force_clarify: bool = False):
        self.force_tool = force_tool
        self.force_clarify = force_clarify

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Extract user query from messages
        user_query = ""
        for m in messages:
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break

        q = user_query.lower()

        # Forced responses for testing
        if self.force_clarify:
            return self._clarification_response(
                "I need to know which area you want to analyze. Please select an ROI on the map first."
            )
        if self.force_tool:
            return self._tool_call_response(self.force_tool, {"query": user_query})

        # Simple keyword-based routing for the mock
        # Check multi-condition first (more specific patterns)
        if any(kw in q for kw in ("ndvi greater", "ndvi less", "ndwi greater", "ndwi less", "threshold", "with ndvi", "with ndwi")):
            return self._tool_call_response("multi_condition_query", {"query": user_query})
        if any(kw in q for kw in ("ndvi", "vegetation index", "greenness", "crop health")):
            return self._tool_call_response("compute_ndvi", {"query": user_query})
        if any(kw in q for kw in ("ndwi", "water index", "water content")):
            return self._tool_call_response("compute_ndwi", {"query": user_query})
        if any(kw in q for kw in ("compare", "before and after", "change", "difference", "temporal")):
            return self._tool_call_response("temporal_compare", {"query": user_query})
        if any(kw in q for kw in ("find", "where", "near", "cropland", "suitable area")):
            return self._tool_call_response("spatial_query", {"query": user_query})
        if any(kw in q for kw in ("suitab", "grow cotton", "grow wheat", "can i grow", "best crop")):
            return self._tool_call_response("crop_suitability", {"query": user_query})

        # Default: ask for clarification
        return self._clarification_response(
            "I'm not sure what analysis you'd like. Try asking about NDVI, NDWI, "
            "temporal comparison, spatial queries, or crop suitability."
        )

    def _tool_call_response(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_mock_1",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments)
                        }
                    }]
                }
            }]
        }

    def _clarification_response(self, message: str) -> Dict[str, Any]:
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": message,
                    "tool_calls": None
                }
            }]
        }


# --------------------------------------------------------------------------- #
# Planner Data Structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClarificationRequest:
    """The planner needs more information before it can produce a ToolCall."""

    message: str
    """Human-readable explanation of what is needed."""

    missing: Tuple[str, ...] = ()
    """Optional structured list of missing context (e.g., "roi", "ndvi_confirmation")."""


@dataclass(frozen=True)
class PlannerError:
    """The planner itself failed (not the tool execution)."""

    code: str
    """Error code: PLANNER_UNAVAILABLE, PLANNER_TIMEOUT, PLANNER_MALFORMED_OUTPUT, PLANNER_UNKNOWN_TOOL."""

    message: str
    """Human-readable message."""

    details: Optional[Dict[str, Any]] = None
    """Additional context for debugging."""


PLANNER_ERROR_CODES = {
    "PLANNER_UNAVAILABLE": "The LLM planner is not configured or unavailable.",
    "PLANNER_TIMEOUT": "The planner did not respond in time.",
    "PLANNER_MALFORMED_OUTPUT": "The planner returned output that could not be parsed.",
    "PLANNER_UNKNOWN_TOOL": "The planner selected a tool that does not exist.",
    "PLANNER_INVALID_ARGUMENTS": "The planner produced invalid arguments for the selected tool.",
    "PLANNER_NO_TOOL_CALL": "The planner did not produce a tool call or clarification.",
}


@dataclass
class PlannerResponse:
    """Result of the planning step.

    Exactly one of tool_call, clarification, or error will be set.
    """

    tool_call: Optional[ToolCall] = None
    clarification: Optional[ClarificationRequest] = None
    error: Optional[PlannerError] = None

    @property
    def has_tool_call(self) -> bool:
        return self.tool_call is not None

    @property
    def has_clarification(self) -> bool:
        return self.clarification is not None

    @property
    def has_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.tool_call:
            out["tool_call"] = self.tool_call.to_dict()
        if self.clarification:
            out["clarification"] = {
                "message": self.clarification.message,
                "missing": list(self.clarification.missing),
            }
        if self.error:
            out["error"] = {
                "code": self.error.code,
                "message": self.error.message,
                "details": self.error.details,
            }
        return out


# --------------------------------------------------------------------------- #
# System Prompt Construction
# --------------------------------------------------------------------------- #

def build_system_prompt(
    context: AnalysisContext,
    conversation_state: Optional[ConversationState] = None
) -> str:
    """Build the system prompt for the LLM planner.

    Includes:
    - Role definition
    - Available tools with schemas
    - Current analysis context (what data is available)
    - Conversation context (for multi-turn follow-ups)
    - Hard constraints (what the LLM must never do)
    """
    tools_meta = list_supported_tools()
    tools_desc = []
    for t in tools_meta:
        schema = get_tool_schema(t["name"])
        params = schema.get("properties", {}) if schema else {}
        param_desc = ", ".join(f"{k}: {v.get('description', '')}" for k, v in params.items())
        tools_desc.append(f"- {t['name']}: {t['description']} (params: {param_desc or 'none'})")

    # Context awareness
    has_roi = context.has_roi
    has_ndvi = context.has_ndvi
    has_temporal = context.has_temporal_pair
    has_index_ctx = context.has_index_context

    context_lines = [
        f"ROI selected: {'yes' if has_roi else 'no'}",
        f"NDVI confirmed: {'yes' if has_ndvi else 'no'}",
        f"Temporal pair selected: {'yes' if has_temporal else 'no'}",
        f"Index context (band roles resolved): {'yes' if has_index_ctx else 'no'}",
    ]

    # Add conversation context if available
    if conversation_state:
        summary = conversation_state.get_context_summary()
        if summary["recent_turns"]:
            conv_lines = ["CONVERSATION CONTEXT (for reference resolution only):"]
            for turn in summary["recent_turns"]:
                conv_lines.append(
                    f"  - User: {turn['query']} -> Tool: {turn['tool'] or 'none'} "
                    f"(intent: {turn['intent'] or 'none'}, status: {turn['status'] or 'none'})"
                )
            if summary["current_roi_available"]:
                conv_lines.append("  - ROI available from previous turn: yes")
            if summary["current_crop"]:
                conv_lines.append(f"  - Crop from previous turn: {summary['current_crop']}")
            if summary["current_dates"] != [None, None]:
                conv_lines.append(f"  - Dates from previous turn: {summary['current_dates']}")
            context_lines.extend(conv_lines)

    return f"""You are the SatQuery AI planner. Your ONLY job is to convert a user's natural-language question into a valid tool call for the SatQuery geospatial analysis engine.

AVAILABLE TOOLS:
{chr(10).join(tools_desc)}

CURRENT CONTEXT:
{chr(10).join(context_lines)}

HARD CONSTRAINTS (violation = planner error):
1. You may ONLY call tools from the AVAILABLE TOOLS list above.
2. You may NOT invent tools, parameters, or numerical values.
3. You may NOT calculate NDVI, NDWI, statistics, or any geospatial computation.
4. You may NOT override engine statuses: NEEDS_ROI, NEEDS_NDVI_CONFIRMATION, NEEDS_TWO_DATES, NEEDS_THRESHOLD, UNSUPPORTED, INSUFFICIENT_DATA.
5. If the user asks for something requiring missing context (e.g., NDVI without ROI), you MUST either:
   a) Produce the tool call anyway (the engine will return the proper status), OR
   b) Return a clarification request asking for the missing context.
6. If the query is ambiguous or you cannot determine intent, return a clarification request.
7. All tool arguments must be valid JSON matching the tool's schema.

CONVERSATION CONTEXT RULES:
- Previous turns are for REFERENCE RESOLUTION ONLY (e.g., "this area" -> previous ROI)
- NEVER use conversation history to invent tools, parameters, or numerical values
- NEVER use conversation history to override engine statuses
- Explicit information in the CURRENT query ALWAYS overrides conversation context
- If context is missing or ambiguous, ask for clarification

RESPONSE FORMAT:
You must respond with EITHER:
- A function/tool call (using the provided function calling interface), OR
- A plain text message asking for clarification (no tool call).

Do not include explanations, apologies, or conversational filler in tool calls.
"""


def build_tools_schema() -> List[Dict[str, Any]]:
    """Build the OpenAI-compatible tools schema for function calling."""
    tools = []
    for tool_name in SUPPORTED_TOOLS:
        schema = get_tool_schema(tool_name)
        if schema:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": schema.get("description", ""),
                    "parameters": schema,
                }
            })
    return tools


# --------------------------------------------------------------------------- #
# Planner Core
# --------------------------------------------------------------------------- #

class LLMPlanner:
    """LLM-powered planner that converts user queries to validated ToolCalls.

    This class is the boundary between the LLM and the deterministic engine.
    It handles:
    - Prompt construction with context
    - LLM provider communication
    - Output parsing and validation
    - Fallback to deterministic router on planner failure
    - Conversation context for multi-turn reference resolution
    """

    def __init__(
        self,
        provider: LLMProvider,
        context: AnalysisContext,
        *,
        fallback_to_deterministic: bool = True,
        conversation_state: Optional[ConversationState] = None,
    ) -> None:
        self.provider = provider
        self.context = context
        self.fallback_to_deterministic = fallback_to_deterministic
        self.conversation_state = conversation_state

    def plan(self, user_query: str) -> PlannerResponse:
        """Plan a tool call from a user query.

        Returns a PlannerResponse containing either:
        - tool_call: a validated ToolCall ready for ToolAdapter.execute()
        - clarification: a ClarificationRequest to show the user
        - error: a PlannerError if planning failed
        """
        if not user_query or not user_query.strip():
            return PlannerResponse(
                clarification=ClarificationRequest(
                    message="Please provide a question about the satellite imagery.",
                    missing=(),
                )
            )

        # Resolve conversational references if conversation state is available
        inherited_args = None
        if self.conversation_state:
            _, inherited_args = self.conversation_state.resolve_references(user_query)

        # Build messages for the LLM
        messages = [
            {"role": "system", "content": build_system_prompt(self.context, self.conversation_state)},
            {"role": "user", "content": user_query.strip()},
        ]

        tools_schema = build_tools_schema()

        try:
            response = self.provider.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                tools=tools_schema,
                tool_choice="auto",
            )
        except Exception as e:
            return PlannerResponse(
                error=PlannerError(
                    code="PLANNER_UNAVAILABLE",
                    message="The planner could not be reached.",
                    details={"exception": type(e).__name__},
                )
            )

        return self._parse_response(response, inherited_args)

    def _parse_response(self, response: Dict[str, Any], inherited_args: Optional[Dict[str, Any]] = None) -> PlannerResponse:
        """Parse and validate the LLM response."""
        try:
            choices = response.get("choices", [])
            if not choices:
                return PlannerResponse(
                    error=PlannerError(
                        code="PLANNER_MALFORMED_OUTPUT",
                        message="LLM response has no choices.",
                        details={"response": response},
                    )
                )

            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls")
            content = message.get("content")

            # Case 1: LLM returned a tool call
            if tool_calls:
                if len(tool_calls) != 1:
                    return PlannerResponse(
                        error=PlannerError(
                            code="PLANNER_MALFORMED_OUTPUT",
                            message=f"Expected exactly one tool call, got {len(tool_calls)}.",
                            details={"tool_calls": tool_calls},
                        )
                    )

                tc = tool_calls[0]
                func = tc.get("function", {})
                tool_name = func.get("name")
                arguments_str = func.get("arguments", "{}")

                # Validate tool name
                if tool_name not in SUPPORTED_TOOLS:
                    return PlannerResponse(
                        error=PlannerError(
                            code="PLANNER_UNKNOWN_TOOL",
                            message=f"Unknown tool: '{tool_name}'. Supported: {', '.join(SUPPORTED_TOOLS)}",
                            details={"tool_name": tool_name, "supported": list(SUPPORTED_TOOLS)},
                        )
                    )

                # Parse arguments
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError as e:
                    return PlannerResponse(
                        error=PlannerError(
                            code="PLANNER_INVALID_ARGUMENTS",
                            message=f"Tool arguments are not valid JSON: {e}",
                            details={"arguments_str": arguments_str},
                        )
                    )

                # Apply inherited conversation context (explicit LLM args take precedence)
                # Always apply if conversation state has relevant context
                if self.conversation_state:
                    arguments = self.conversation_state.apply_inherited_args(tool_name, arguments)

                # Validate arguments against schema
                schema = get_tool_schema(tool_name)
                if schema:
                    validation_error = self._validate_arguments(arguments, schema)
                    if validation_error:
                        return PlannerResponse(
                            error=PlannerError(
                                code="PLANNER_INVALID_ARGUMENTS",
                                message=validation_error,
                                details={"arguments": arguments, "schema": schema},
                            )
                        )

                # Semantic validation of structured arguments
                semantic_error = validate_tool_arguments(tool_name, arguments)
                if semantic_error:
                    return PlannerResponse(
                        error=PlannerError(
                            code="PLANNER_INVALID_ARGUMENTS",
                            message=semantic_error,
                            details={"arguments": arguments},
                        )
                    )

                # Build validated ToolCall
                call_id = tc.get("id")
                tool_call = ToolCall(name=tool_name, arguments=arguments, call_id=call_id)
                return PlannerResponse(tool_call=tool_call)

            # Case 2: LLM returned a clarification (plain text)
            if content and content.strip():
                return PlannerResponse(
                    clarification=ClarificationRequest(
                        message=content.strip(),
                        missing=(),
                    )
                )

            # Case 3: Neither tool call nor content
            return PlannerResponse(
                error=PlannerError(
                    code="PLANNER_NO_TOOL_CALL",
                    message="The planner did not produce a tool call or clarification.",
                    details={"response": response},
                )
            )

        except Exception as e:
            return PlannerResponse(
                error=PlannerError(
                    code="PLANNER_MALFORMED_OUTPUT",
                    message=f"Failed to parse planner response: {e}",
                    details={"exception": type(e).__name__, "response": response},
                )
            )

    def _validate_arguments(self, arguments: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against the tool's JSON schema.

        Returns None if valid, error message string if invalid.
        """
        # First check the tool name from the schema
        # The schema doesn't directly contain tool name, so we use the schema's properties
        # to infer which tool it might be, but it's simpler to just do schema validation
        # and also call the new validate_tool_arguments for semantic validation.

        # Check required properties
        required = schema.get("required", [])
        for req in required:
            if req not in arguments:
                return f"Missing required argument: '{req}'"

        # Check for extra properties if additionalProperties is false
        if not schema.get("additionalProperties", True):
            allowed = set(schema.get("properties", {}).keys())
            for key in arguments:
                if key not in allowed:
                    return f"Unexpected argument: '{key}'. Allowed: {sorted(allowed)}"

        # Basic type checking for known properties
        properties = schema.get("properties", {})
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


# --------------------------------------------------------------------------- #
# Execution with Fallback
# --------------------------------------------------------------------------- #

def execute_with_fallback(
    user_query: str,
    context: AnalysisContext,
    planner: Optional[LLMPlanner] = None,
    conversation_state: Optional[ConversationState] = None,
) -> Tuple[ToolResult, bool]:
    """Execute a user query with planner + fallback to deterministic router.

    Returns:
        (ToolResult, used_fallback)
        - used_fallback=True means the deterministic router was used
        - used_fallback=False means the planner succeeded

    The fallback path:
    1. If planner is None or returns an error -> use deterministic router
    2. If planner returns clarification -> return it as a ToolResult with status NEEDS_CLARIFICATION
    3. If planner returns a valid ToolCall -> execute via ToolAdapter
    4. If ToolAdapter execution fails with EXECUTION_ERROR -> try deterministic router

    If conversation_state is provided, it will be updated with the turn result.
    """
    # No planner available or disabled -> direct deterministic route
    if planner is None:
        result = _deterministic_fallback(user_query, context)
        if conversation_state:
            _update_conversation_state(conversation_state, user_query, result)
        return result, True

    # Try planner
    plan_response = planner.plan(user_query)

    # Planner error -> fallback
    if plan_response.has_error:
        result = _deterministic_fallback(user_query, context)
        if conversation_state:
            _update_conversation_state(conversation_state, user_query, result)
        return result, True

    # Planner clarification -> return as special ToolResult
    if plan_response.has_clarification:
        clar = plan_response.clarification
        result = ToolResult(
            call_id=None,
            tool_name="clarification",
            status="NEEDS_CLARIFICATION",
            result=None,
            evidence=None,
            warnings=[],
            message=clar.message,
            error=None,
        )
        if conversation_state:
            _update_conversation_state(conversation_state, user_query, result)
        return result, False

    # Planner tool call -> execute
    if plan_response.has_tool_call:
        adapter = ToolAdapter(context)
        tool_result = adapter.execute(plan_response.tool_call)

        # If execution failed with internal error, try fallback
        if tool_result.status == Status.ERROR.value and tool_result.error and tool_result.error.code == "EXECUTION_ERROR":
            fallback_result = _deterministic_fallback(user_query, context)
            if conversation_state:
                _update_conversation_state(conversation_state, user_query, fallback_result)
            return fallback_result, True

        if conversation_state:
            _update_conversation_state(conversation_state, user_query, tool_result)
        return tool_result, False

    # Should not reach here
    result = _deterministic_fallback(user_query, context)
    if conversation_state:
        _update_conversation_state(conversation_state, user_query, result)
    return result, True


def _update_conversation_state(
    conversation_state: ConversationState,
    user_query: str,
    tool_result: ToolResult
) -> None:
    """Update conversation state with the completed turn."""
    # Extract crop from tool result if available
    crop = None
    if tool_result.result and isinstance(tool_result.result, dict):
        crop = tool_result.result.get("crop")

    # Check if ROI was available in context (we infer from tool name and status)
    has_roi = tool_result.status not in ("NEEDS_ROI", "NEEDS_NDVI_CONFIRMATION")

    # Extract dates if temporal
    dates = (None, None)
    if tool_result.tool_name == "temporal_compare":
        # Could extract from result if available
        pass

    turn = ConversationTurn(
        user_query=user_query,
        tool_name=tool_result.tool_name,
        intent=tool_result.tool_name,  # tool_name maps to intent
        status=tool_result.status,
        crop=crop,
        has_roi=has_roi,
        dates=dates,
    )
    conversation_state.add_turn(turn)


def _deterministic_fallback(user_query: str, context: AnalysisContext) -> ToolResult:
    """Execute using the existing deterministic router directly."""
    from analyses.registry import route
    from core.tools import execute_tool

    execution = route(user_query, context)

    # Convert AnalysisExecution to ToolResult (similar to ToolAdapter)
    call_id = None
    tool_name = intent_to_tool_name(execution.intent) or "deterministic_router"

    # Extract result dict
    result_dict = None
    if execution.result is not None and hasattr(execution.result, "to_dict"):
        result_dict = execution.result.to_dict()
    elif isinstance(execution.result, dict):
        result_dict = execution.result

    # Extract evidence
    evidence_dict = None
    if execution.provenance and isinstance(execution.provenance, dict):
        evidence_dict = {
            "provenance": execution.provenance,
            "status": execution.status.value,
            "message": execution.message,
            "warnings": list(execution.warnings),
        }
        if result_dict and "provenance" in result_dict:
            evidence_dict["result_provenance"] = result_dict["provenance"]
        if result_dict and "conditions" in result_dict:
            evidence_dict["conditions"] = result_dict["conditions"]

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


# Re-export for convenience
from core.tools import intent_to_tool_name  # noqa: E402

__all__ = [
    "LLMProvider",
    "MockLLMProvider",
    "ClarificationRequest",
    "PlannerError",
    "PlannerResponse",
    "LLMPlanner",
    "build_system_prompt",
    "build_tools_schema",
    "execute_with_fallback",
    "PLANNER_ERROR_CODES",
    "ConversationState",
    "ConversationTurn",
]
