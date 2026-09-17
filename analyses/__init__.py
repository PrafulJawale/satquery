"""Phase 7 -- the analysis layer.

    core/router.py      understands the SENTENCE      (no mathematics)
    analyses/registry.py binds an INTENT to an ENGINE  (the only such place)
    analyses/<engine>.py does the ANALYSIS             (no knowledge of text)

`route(query, context)` is the only function `app.py` needs to call.
"""

from __future__ import annotations

from core.router import Intent, QueryIntent, parse_query

from .base import (
    AnalysisContext,
    AnalysisExecution,
    AnalysisSpec,
    IndexContext,
    NdviContext,
    Status,
)
from .multi_condition import MultiConditionResult, run_multi_condition
from .registry import (
    REGISTRY,
    available_specs,
    get_spec,
    planned_specs,
    route,
    suggestions,
    validate_context,
)

__all__ = [
    "Intent",
    "QueryIntent",
    "parse_query",
    "AnalysisContext",
    "AnalysisExecution",
    "AnalysisSpec",
    "IndexContext",
    "NdviContext",
    "Status",
    "MultiConditionResult",
    "run_multi_condition",
    "REGISTRY",
    "available_specs",
    "planned_specs",
    "get_spec",
    "route",
    "suggestions",
    "validate_context",
]
