"""Phase 7 -- the analysis registry and the single routing entry point.

THE ONLY PLACE WHERE AN INTENT IS BOUND TO CODE.

    QUERY -> parse_query -> QueryIntent -> registry -> context check -> handler

Adding a future analysis (crop suitability, flood change, vegetation change,
NDWI, multi-condition queries) means adding ONE entry here plus its engine
module. `core/router.py`, `app.py` and the UI do not change.

There is deliberately no `if "ndvi" in query:` anywhere in the app.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from core.router import (
    PLANNED_INTENTS,
    Intent,
    QueryIntent,
    parse_query,
)
from core.router import REQUIRED_CONTEXT

from .base import (
    CONTEXT_MESSAGES,
    AnalysisContext,
    AnalysisExecution,
    AnalysisSpec,
    Status,
)
from .crop_suitability import run_crop_suitability
from .spatial_query import run_spatial_query
from .ndvi import run_ndvi_roi_stats
from .ndvi_change import run_ndvi_change
from .multi_condition import run_multi_condition
from .ndwi import run_ndwi_roi_stats

__all__ = [
    "REGISTRY",
    "get_spec",
    "available_specs",
    "planned_specs",
    "suggestions",
    "route",
]


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
REGISTRY: Dict[Intent, AnalysisSpec] = {
    Intent.NDVI_ROI_STATS: AnalysisSpec(
        intent=Intent.NDVI_ROI_STATS,
        title="NDVI statistics for a selected area",
        description=("Zonal statistics of the native NDVI inside the drawn "
                     "region: pixel counts, mean/median/std, percentiles."),
        requires=REQUIRED_CONTEXT[Intent.NDVI_ROI_STATS],
        handler=run_ndvi_roi_stats,
        example_queries=(
            "What is the NDVI of this area?",
            "Calculate the vegetation index here.",
            "Show vegetation health in this area.",
            "Analyze the vegetation in this selected area.",
        ),
    ),
    Intent.CROP_SUITABILITY: AnalysisSpec(
        intent=Intent.CROP_SUITABILITY,
        title="Crop suitability screening (experimental, cotton only)",
        description=(
            "Experimental crop-suitability screening for cotton: climatological "
            "temperature and growing-season rainfall, soil pH and texture, slope "
            "and land-cover constraints, combined with an explicit, configurable "
            "weighting scheme. Not a recommendation or yield prediction."),
        requires=REQUIRED_CONTEXT[Intent.CROP_SUITABILITY],
        handler=run_crop_suitability,
        unavailable_message="Crop suitability analysis is not available yet.",
        example_queries=(
            "Can I grow cotton here?",
            "Is cotton suitable here?",
            "Is this area suitable for cotton?",
            "Where can I grow cotton in this region?",
        ),
    ),
    # Phase 9: an orchestrator, not a new model. It combines conditions
    # produced by the engines above (cotton screening) and by the WorldCover
    # datasource. Example queries are added with the UI (Checkpoint D).
    Intent.SPATIAL_QUERY: AnalysisSpec(
        intent=Intent.SPATIAL_QUERY,
        title="Multi-condition spatial query (Phase 9, experimental)",
        description=(
            "Combines mappable conditions over the selected area -- cotton "
            "suitability classes from the Phase 8 screening, WorldCover land-"
            "cover classes, mapped permanent water and distance to it -- with "
            "explicit three-valued logic (match / no match / insufficient "
            "data). Conditions the system does not measure (irrigation, flood "
            "risk, groundwater, salinity) are refused, never approximated."),
        requires=REQUIRED_CONTEXT[Intent.SPATIAL_QUERY],
        handler=run_spatial_query,
        unavailable_message="Spatial query analysis is not available yet.",
        example_queries=(
            "Find cropland near water",
            "Find areas suitable for cotton near water",
            "Find cropland excluding water",
        ),
    ),
    Intent.FLOOD_CHANGE: AnalysisSpec(
        intent=Intent.FLOOD_CHANGE,
        title="Flood / water change detection",
        description="Not available yet — needs a second date and NDWI.",
        unavailable_message="Flood analysis is not available yet.",
        example_queries=("Show flood areas.", "Where did the water extent change?"),
    ),
    # Phase 10: the temporal NDVI engine. Three intents share it -- they differ
    # only in how the user phrases the request.
    Intent.NDVI_CHANGE_ROI: AnalysisSpec(
        intent=Intent.NDVI_CHANGE_ROI,
        title="NDVI change between two dates (Phase 10)",
        description=(
            "Compares NDVI for the SAME selected area across two dated "
            "acquisitions on one aligned grid, and reports the mean/median "
            "change plus the share of cells that increased, stayed stable or "
            "decreased at a configurable display threshold. Change is computed "
            "only where BOTH dates are valid; the result states the alignment "
            "used and never attributes the change to a cause."),
        requires=REQUIRED_CONTEXT[Intent.NDVI_CHANGE_ROI],
        handler=run_ndvi_change,
        example_queries=(
            "Compare NDVI between these two dates.",
            "Show the NDVI difference for this area.",
            "What is the NDVI change here?",
        ),
    ),
    Intent.TEMPORAL_COMPARISON: AnalysisSpec(
        intent=Intent.TEMPORAL_COMPARISON,
        title="Before / after comparison of two acquisitions (Phase 10)",
        description=(
            "The generic before-and-after question: compares the vegetation "
            "index of the selected area between the chosen earlier and later "
            "acquisition. Both dates must be selected; the system never picks "
            "one on the user's behalf."),
        requires=REQUIRED_CONTEXT[Intent.TEMPORAL_COMPARISON],
        handler=run_ndvi_change,
        example_queries=(
            "Compare before and after.",
            "How has this area changed between these two dates?",
        ),
    ),
    # Phase 11: NDWI CHANGE is recognised and refused, so that "compare NDWI
    # before and after" is never answered with an NDVI difference.
    Intent.TEMPORAL_NDWI: AnalysisSpec(
        intent=Intent.TEMPORAL_NDWI,
        title="NDWI change between two dates",
        description=("Not available yet — NDWI is implemented for a single date "
                     "only. A two-date NDWI comparison is not computed, and is "
                     "never substituted with an NDVI comparison."),
        # Order matters: the registry contract (asserted by
        # tests/test_phase7_router.py) is that an unavailable message ENDS with
        # "not available yet." -- the explanation comes first.
        unavailable_message=("NDWI is implemented for a single date. NDWI change "
                             "between two dates is not available yet."),
        example_queries=("Compare NDWI before and after.",),
    ),
    Intent.VEGETATION_CHANGE: AnalysisSpec(
        intent=Intent.VEGETATION_CHANGE,
        title="Vegetation change between two dates (Phase 10)",
        description=(
            "Reports where the vegetation index increased, stayed stable or "
            "decreased inside the selected area between two dated "
            "acquisitions. A decrease is a decrease in the measured INDEX: it "
            "is not, on its own, evidence of deforestation, crop failure, "
            "drought or flooding, all of which need data this app does not "
            "have."),
        requires=REQUIRED_CONTEXT[Intent.VEGETATION_CHANGE],
        handler=run_ndvi_change,
        example_queries=(
            "How has the vegetation changed in this area?",
            "Find vegetation loss.",
            "Which parts of this area experienced vegetation decrease?",
        ),
    ),
    # Phase 11: the second index, computed by the SAME engine as NDVI
    # (core.indices.compute_index) from config/indices/ndwi.yml.
    #
    # APPENDED LAST, deliberately: suggestions() walks this registry in
    # insertion order, and the Phase 8 cotton examples must remain inside
    # suggestions(limit=8) (asserted in tests/test_phase7_router.py).
    Intent.NDWI_ROI_STATS: AnalysisSpec(
        intent=Intent.NDWI_ROI_STATS,
        title="NDWI — water index for a selected area (Phase 11)",
        description=(
            "Zonal statistics of the native NDWI inside the drawn region: pixel "
            "counts, mean/median/std/min/max over the valid pixels only. NDWI is "
            "a spectral index — it is not flood detection, not a water mask, and "
            "not a water-quality measurement. No water/non-water threshold is "
            "applied."),
        requires=REQUIRED_CONTEXT[Intent.NDWI_ROI_STATS],
        handler=run_ndwi_roi_stats,
        example_queries=(
            "What is the NDWI of this area?",
            "Calculate NDWI here.",
            "Show the water index for this ROI.",
        ),
    ),
    # Phase 12: composition of evidence that already exists. APPENDED LAST --
    # suggestions() walks this registry in insertion order and the Phase 8
    # cotton examples must stay inside suggestions(limit=8).
    Intent.MULTI_CONDITION: AnalysisSpec(
        intent=Intent.MULTI_CONDITION,
        title="Multi-condition evidence composition (Phase 12)",
        description=(
            "Combines conditions that already have explicit support: a Phase 9 "
            "spatial condition (cropland, permanent water, near water), a "
            "Phase 11 spectral threshold (NDVI / NDWI, native index values) and "
            "a Phase 10 NDVI change class. Three-valued throughout: a cell that "
            "could not be measured stays UNKNOWN and is never counted as a "
            "non-match. A combination is geographic evidence, not causal "
            "attribution."),
        requires=REQUIRED_CONTEXT[Intent.MULTI_CONDITION],
        handler=run_multi_condition,
        example_queries=(
            "Find cropland with NDVI greater than 0.6.",
            "Show areas with vegetation decrease near permanent water.",
            "Find cropland with NDWI less than -0.4.",
        ),
    ),
}


def get_spec(intent: Intent) -> Optional[AnalysisSpec]:
    return REGISTRY.get(intent)


def available_specs() -> Dict[Intent, AnalysisSpec]:
    return {i: s for i, s in REGISTRY.items() if s.available}


def planned_specs() -> Dict[Intent, AnalysisSpec]:
    return {i: s for i, s in REGISTRY.items() if not s.available}


def suggestions(limit: int = 4) -> List[str]:
    """Example queries, generated from the registry -- never hard-coded in the UI."""
    out: List[str] = []
    for spec in REGISTRY.values():
        if not spec.available:
            continue
        for example in spec.example_queries:
            if example not in out:
                out.append(example)
    return out[:limit]


# --------------------------------------------------------------------------- #
# context validation
# --------------------------------------------------------------------------- #
def _check_requirement(name: str, context: AnalysisContext) -> bool:
    if name == "roi":
        return context.has_roi
    if name == "ndvi_confirmed":
        return context.has_ndvi
    # An unknown requirement is a programming error, not a context problem.
    raise KeyError(f"Unknown context requirement: {name!r}")


def validate_context(spec: AnalysisSpec,
                     context: AnalysisContext) -> Tuple[Status, str]:
    """Return (status, message): (OK, '') when everything required is present."""
    for requirement in spec.requires:
        if not _check_requirement(requirement, context):
            return (Status.NEEDS_ROI if requirement == "roi"
                    else Status.NEEDS_NDVI_CONFIRMATION,
                    CONTEXT_MESSAGES.get(requirement, "Missing context."))
    return Status.OK, ""


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def route(query: str,
          context: AnalysisContext,
          *,
          convention: Optional[str] = None) -> AnalysisExecution:
    """Text + context -> AnalysisExecution. The whole of Phase 7 in one call.

    The router decides WHAT is being asked; this function decides whether the
    system CAN answer it, and only then calls the engine.

    `convention` is the Phase 12 opt-in: the name of a labelled threshold
    convention (from config/multi/thresholds.yml) that the USER has explicitly
    switched on for this one question. It is never applied by default and it
    never changes the routing -- it only fills in a threshold the engine would
    otherwise refuse to invent.
    """
    parsed: QueryIntent = parse_query(query)

    if convention and parsed.intent is Intent.MULTI_CONDITION:
        # QueryIntent is frozen: the opt-in produces a NEW intent instead of
        # mutating the parsed one. A failed opt-in leaves the question exactly
        # as the parser found it -- it never becomes a free threshold.
        try:
            from dataclasses import replace

            from core.multi_condition import parse_composed_query

            parsed = replace(
                parsed,
                composed=parse_composed_query(parsed.original_query,
                                              enabled_convention=convention))
        except Exception:        # an opt-in must never break the refusal path
            pass

    # 1. the parser could not resolve the request -- never guess
    if parsed.intent is Intent.UNKNOWN or not parsed.is_actionable:
        return AnalysisExecution(
            intent=Intent.UNKNOWN, status=Status.UNKNOWN, query=str(query or ""),
            normalized_query=parsed.normalized_query, confidence=parsed.confidence,
            explanation=parsed.explanation, matched=parsed.matched,
            message=_unknown_message(parsed),
            provenance={"engine": None},
        )

    # 3b. Phase 9: an unmeasurable requirement blocks the engine. Checked AFTER
    #     the context check (an area must still be selected) and BEFORE the
    #     engine runs, so an irrigation question can never be answered with a
    #     rainfed verdict.
    if parsed.blocked_by:
        detail = " ".join(
            str(getattr(c, "note", "") or "") for c in parsed.blocked_by).strip()
        return AnalysisExecution(
            intent=parsed.intent, status=Status.UNSUPPORTED_CONDITION,
            query=str(query or ""), normalized_query=parsed.normalized_query,
            confidence=parsed.confidence, explanation=parsed.explanation,
            matched=parsed.matched, message=detail,
            provenance={"engine": None,
                        "blocked_by": [c.to_dict() for c in parsed.blocked_by]},
        )

    spec = get_spec(parsed.intent)

    # 2. an intent with no registered engine
    if spec is None or not spec.available:
        return AnalysisExecution(
            intent=parsed.intent, status=Status.UNSUPPORTED, query=str(query or ""),
            normalized_query=parsed.normalized_query, confidence=parsed.confidence,
            explanation=parsed.explanation, matched=parsed.matched,
            message=(spec.unavailable_message if spec else
                     "That analysis is not available yet."),
            provenance={"engine": None, "planned": parsed.intent in PLANNED_INTENTS},
        )

    # 3. required context missing -> ask for it, do NOT run something else
    status, message = validate_context(spec, context)
    if status is not Status.OK:
        return AnalysisExecution(
            intent=parsed.intent, status=status, query=str(query or ""),
            normalized_query=parsed.normalized_query, confidence=parsed.confidence,
            explanation=parsed.explanation, matched=parsed.matched,
            message=message, provenance={"engine": None},
        )

    # 4. the engine does the mathematics
    try:
        execution = spec.handler(context, parsed)          # type: ignore[misc]
    except Exception as exc:                                # pragma: no cover
        return AnalysisExecution(
            intent=parsed.intent, status=Status.ERROR, query=str(query or ""),
            normalized_query=parsed.normalized_query, confidence=parsed.confidence,
            explanation=parsed.explanation, matched=parsed.matched,
            message=f"The analysis failed: {exc}",
            warnings=("Nothing was reported because the analysis did not "
                      "complete — a partial or default value would be "
                      "fabricated.",),
            provenance={"engine": getattr(spec.handler, "__module__", ""),
                        "error": f"{type(exc).__name__}: {exc}"},
        )
    return execution


def _unknown_message(parsed: QueryIntent) -> str:
    base = ("I can currently analyze NDVI (a vegetation index) for a selected "
            "area.")
    hint = " Try asking: " + suggestions(limit=1)[0] if suggestions(limit=1) else ""
    if parsed.matched:
        return (f"I am not sure what analysis you mean "
                f"(weak match on {', '.join(repr(m) for m in parsed.matched)}). "
                + base + hint)
    return f"I could not match that to an available analysis. {base}{hint}"
