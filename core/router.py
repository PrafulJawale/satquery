"""Phase 7 -- deterministic query understanding (the ROUTER).

WHAT THIS MODULE IS
    It turns a sentence into a structured intent. That is all it does.

WHAT THIS MODULE IS NOT
    It is not an analysis engine. There is deliberately no numpy, rasterio,
    shapely or pyproj import anywhere in this file -- the router must never be
    able to compute a pixel value, and a test asserts that
    (`tests/test_phase7_router.py::test_router_imports_no_raster_libraries`).

    The separation matters because the two halves fail in different ways: the
    parser can be wrong about *language*, the engine can be wrong about *the
    world*. Mixing them makes both unverifiable.

WHY DETERMINISTIC AND NOT AN LLM (for now)
    * every answer can be explained ("pattern 'vegetation index' matched");
    * the behaviour is exactly testable, including the ambiguous cases;
    * no network, no key, no cost, no hallucination;
    * "can I grow cotton here?" cannot be answered, so it must not be invented.

HOW AN LLM REPLACES THIS LATER
    Anything with the same signature can take over:

        parse_query(text) -> QueryIntent

    A future `parse_query_llm()` would be constrained to emit only the members
    of `Intent` that the registry already knows, use this parser as a fallback /
    cross-check, and change nothing else: not `app.py`, not the UI, not any
    engine, not any existing test.

Matching is EVIDENCE-SCORED, not string equality:
    phrase hit  = 3 points
    token  hit  = 1 point
    negative    = -2 points      (e.g. "change" vetoes the single-date NDVI)
    confidence  = clamp(score / 4)
    accepted    when confidence >= THRESHOLD and the winner is not tied
    otherwise   -> Intent.UNKNOWN  (a guess is worse than a question)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Intent",
    "QueryIntent",
    "PLANNED_INTENTS",
    "CONFIDENCE_THRESHOLD",
    "normalize",
    "parse_query",
    "tokenize",
    "supported_examples",
]


class Intent(str, Enum):
    """The complete set of requests the system can talk about.

    Only intents with a registered handler are executable. Adding a member here
    is step 1 of adding a future analysis; step 2 is `analyses/registry.py`.
    """

    # -- implemented ------------------------------------------------------- #
    NDVI_ROI_STATS = "NDVI_ROI_STATS"
    # Phase 9: geographic selection built from several supported conditions.
    # The CONDITION GRAMMAR lives in core/spatial_query.py (+ config/spatial/);
    # the router only decides that the sentence is this kind of request.
    SPATIAL_QUERY = "SPATIAL_QUERY"
    # Phase 10: temporal comparison. TWO names, ONE engine, because the two
    # phrasings are genuinely different requests for the same computation:
    #   NDVI_CHANGE_ROI    -- "compare NDVI between these two dates" (index-first)
    #   TEMPORAL_COMPARISON -- "compare before and after" (time-first)
    # VEGETATION_CHANGE (below) keeps its Phase 7 vocabulary ("how has the
    # vegetation changed", "vegetation loss") and now has an engine too.
    NDVI_CHANGE_ROI = "NDVI_CHANGE_ROI"
    TEMPORAL_COMPARISON = "TEMPORAL_COMPARISON"
    # Phase 11: a second spectral index over a selected area. Same contract as
    # NDVI_ROI_STATS (ROI required, native index values), different index.
    NDWI_ROI_STATS = "NDWI_ROI_STATS"
    # Phase 12: COMPOSITION of evidence that already exists -- a spatial
    # condition (Phase 9) with a spectral threshold (Phase 11) and/or a temporal
    # change class (Phase 10). It computes no new science; it intersects masks
    # that other phases produced.
    MULTI_CONDITION = "MULTI_CONDITION"

    # -- planned: recognised vocabulary, NO handler, never fabricates ------ #
    CROP_SUITABILITY = "CROP_SUITABILITY"
    FLOOD_CHANGE = "FLOOD_CHANGE"
    # Phase 10: VEGETATION_CHANGE moved to IMPLEMENTED (it shares the temporal
    # NDVI engine). FLOOD_CHANGE stays planned: flood/water change needs a
    # second date AND a water index, and a flood claim from NDVI alone would be
    # fabricated, so it must keep returning UNSUPPORTED.
    VEGETATION_CHANGE = "VEGETATION_CHANGE"
    # Phase 11: NDWI *between two dates* is NOT implemented. It gets its own
    # planned intent so that "compare NDWI before and after" is refused by name
    # instead of being answered with an NDVI change (a silent substitution would
    # be a fabricated result).
    TEMPORAL_NDWI = "TEMPORAL_NDWI"

    # -- parser outcome ---------------------------------------------------- #
    UNKNOWN = "UNKNOWN"


#: Intents the parser understands but that have no engine yet. Anything routed
#: here returns the registry's `unavailable_message` and computes nothing.
PLANNED_INTENTS: Tuple[Intent, ...] = (
    # Phase 10: vegetation change is IMPLEMENTED -- it shares the temporal NDVI
    # engine with NDVI_CHANGE_ROI and TEMPORAL_COMPARISON. Flood / water change
    # stays planned: it needs a second date AND a water index, and answering it
    # from NDVI alone would be a fabricated claim.
    Intent.FLOOD_CHANGE,
    # Phase 11: NDWI is implemented as a single-date index, but NDWI CHANGE is
    # not. Refused by name rather than answered with an NDVI difference.
    Intent.TEMPORAL_NDWI,
)

#: What each executable intent needs before it may run. Validated in
#: `analyses/registry.py` -> `route()`; the router itself only declares it.
REQUIRED_CONTEXT: Dict[Intent, Tuple[str, ...]] = {
    Intent.NDVI_ROI_STATS: ("roi", "ndvi_confirmed"),
    # Crop suitability needs an area and nothing else: every environmental layer
    # is fetched by the engine itself, for the ROI, when it runs.
    Intent.CROP_SUITABILITY: ("roi",),
    # A spatial query is geography inside the drawn area -- it needs one.
    Intent.SPATIAL_QUERY: ("roi",),
    # Phase 10: a temporal comparison needs a selected area. The two DATES are
    # deliberately NOT a context requirement: the engine can say which of them
    # is missing (NEEDS_TWO_DATES), which a generic context message cannot.
    Intent.NDVI_CHANGE_ROI: ("roi",),
    Intent.TEMPORAL_COMPARISON: ("roi",),
    Intent.VEGETATION_CHANGE: ("roi",),
    # Phase 11: an index needs a selected area, exactly like NDVI. The BAND
    # ROLES are not a context requirement: the engine can name the missing role
    # (UNSUPPORTED), which a generic context message cannot.
    Intent.NDWI_ROI_STATS: ("roi",),
    # Phase 12: a composition is geography inside the drawn area, like every
    # other spatial analysis. The THRESHOLD and the two DATES are deliberately
    # not context requirements: the engine names what is missing
    # (NEEDS_THRESHOLD / NEEDS_TWO_DATES), which a generic message cannot.
    Intent.MULTI_CONDITION: ("roi",),
}

#: Phase 12: intents whose sentences MAY be a composition. Anything outside this
#: set is left exactly where Phases 1-11 put it.
COMPOSITION_CANDIDATES = (
    Intent.SPATIAL_QUERY,
    Intent.MULTI_CONDITION,
    Intent.NDVI_ROI_STATS,
    Intent.NDWI_ROI_STATS,
    Intent.NDVI_CHANGE_ROI,
    Intent.TEMPORAL_COMPARISON,
    Intent.VEGETATION_CHANGE,
)

CONFIDENCE_THRESHOLD: float = 0.34        # one phrase (0.75) or two tokens (0.5)
_PHRASE_WEIGHT: int = 3
_TOKEN_WEIGHT: int = 1
_NEGATIVE_WEIGHT: int = -2
_NORMALISER_MAX: float = 4.0


# --------------------------------------------------------------------------- #
# intent model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueryIntent:
    """The structured result of understanding a sentence.

    Every field is data -- no raster, no map, no widget state. The UI can print
    all of it, which is what makes the router auditable instead of magic.
    """

    intent: Intent
    confidence: float
    original_query: str
    normalized_query: str
    matched: Tuple[str, ...] = ()
    explanation: str = ""
    required_context: Tuple[str, ...] = ()
    rejected: Tuple[str, ...] = ()
    #: Phase 9: the structured spatial conditions, when this is a SPATIAL_QUERY.
    #: Populated by `core.spatial_query.parse_spatial_query`; empty otherwise.
    conditions: Tuple[Any, ...] = ()
    #: Phase 12: the parsed composition (core.multi_condition.ComposedQuery)
    #: when this is a MULTI_CONDITION request; None otherwise.
    composed: Any = None
    #: Phase 9: recognised requirements this system does NOT measure
    #: (irrigation, groundwater, salinity). They block execution of ANY engine:
    #: an explicit irrigation question must not be answered with a rainfed
    #: verdict. Empty when nothing unsupported was asked for.
    blocked_by: Tuple[Any, ...] = ()

    @property
    def is_actionable(self) -> bool:
        """True when the parser believes it understood the request."""
        return self.intent is not Intent.UNKNOWN and self.confidence >= CONFIDENCE_THRESHOLD

    @property
    def is_planned(self) -> bool:
        return self.intent in PLANNED_INTENTS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(float(self.confidence), 3),
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "matched": list(self.matched),
            "explanation": self.explanation,
            "required_context": list(self.required_context),
            "rejected": list(self.rejected),
            "actionable": self.is_actionable,
            "conditions": [c.to_dict() if hasattr(c, "to_dict") else c
                           for c in self.conditions],
            "blocked_by": [c.to_dict() if hasattr(c, "to_dict") else c
                           for c in self.blocked_by],
        }


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^\w\s\-/]")           # drop ? . ! ' " etc.
_WS = re.compile(r"\s+")
_DASHES = re.compile(r"[_/]+")

#: Spellings that mean the same thing. Applied to the normalised text so the
#: pattern tables below can stay short and readable.
_SYNONYMS: Tuple[Tuple[str, str], ...] = (
    ("normalized difference vegetation index", "ndvi"),
    ("normalised difference vegetation index", "ndvi"),
    ("vegetation index", "ndvi"),
    ("vegetation indices", "ndvi"),
    ("vegetation statistic", "ndvi statistics"),
    ("vegetation stats", "ndvi statistics"),
    ("ndvi stats", "ndvi statistics"),
    ("greenness index", "ndvi"),
    ("crop vigour", "crop vigor"),
    ("analyses", "analyze"),
    ("analyse", "analyze"),
    ("analysis", "analyze"),
    ("calculate the", "calculate"),
    ("computed", "calculate"),
    ("compute", "calculate"),
)


def normalize(text: str) -> str:
    """Lower-case, de-punctuate and de-duplicate whitespace; apply synonyms.

    '  What IS the   NDVI of this area?? ' -> 'what is the ndvi of this area'
    'Calculate the vegetation index here.' -> 'calculate ndvi here'
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", str(text)).lower().strip()
    out = _DASHES.sub(" ", out)
    out = _PUNCT.sub(" ", out)
    out = _WS.sub(" ", out).strip()
    for src, dst in _SYNONYMS:
        out = out.replace(src, dst)
    return _WS.sub(" ", out).strip()


def tokenize(normalized: str) -> List[str]:
    return [t for t in normalized.split() if t]


# --------------------------------------------------------------------------- #
# pattern tables  (the only place query vocabulary lives)
# --------------------------------------------------------------------------- #
_PATTERN_TABLE: Dict[Intent, Dict[str, Sequence[str]]] = {
    Intent.NDVI_ROI_STATS: {
        "phrases": (
            "ndvi",
            "ndvi statistics",
            "ndvi value",
            "ndvi values",
            "mean ndvi",
            "ndvi of",
            "vegetation health",
            "health of the vegetation",
            "vegetation analysis",
            "analyze the vegetation",
            "analyze vegetation",
            "vegetation in this",
            "vegetation for this",
            "plant health",
            "crop health index",
        ),
        "tokens": (
            "ndvi",
            "vegetation",
            "vegetative",
            "greenness",
            "vigor",
            "health",
            "index",
            "indices",
            "statistics",
            "stats",
            "analyze",
            "calculate",
            "show",
            "measure",
            "mean",
        ),
        # a single-date statistic cannot answer these -- say so instead of lying
        "negative": (
            "change",
            "changed",
            "changes",
            "difference",
            "delta",
            "trend",
            "temporal",
            "multi-date",
            "between 20",
            "before and after",
            "compare",
            "flood",
            "flooded",
            "inundation",
            "weather",
            "rainfall",
            "temperature",
            "soil",
            "suitability",
            "suitable",
            "yield",
            "harvest",
            "disease",
            "pest",
        ),
    },
    # Phase 9 -- geographic SELECTION ("find/show/where/which areas …"), as
    # opposed to the yes/no screening verdict that CROP_SUITABILITY answers.
    # Vocabulary only; the conditions themselves are parsed in
    # core/spatial_query.py from config/spatial/patterns.yml.
    Intent.SPATIAL_QUERY: {
        "phrases": (
            "find areas", "find area", "find land", "find cropland",
            "find cotton", "find suitable", "show areas", "show land",
            "show cropland", "show cotton", "show agricultural land",
            "where are", "where can i", "where can i grow", "which areas",
            "which parts", "areas suitable", "suitable areas", "areas where",
            "agricultural land", "cropland",
            "near water", "near the water", "near a water",
            "near permanent water", "close to water", "next to water",
            "cotton near", "cotton near water", "where irrigation",
            "within 1 km of water", "within 500 m of water",
            "within 2 km of water", "within 1 km of", "within 500 m of",
            "within 2 km of",
            "but not", "excluding", "outside flood", "flood-prone",
            "flood prone",
        ),
        "tokens": (
            "find", "areas", "where", "near", "water", "cropland",
            "agricultural", "excluding", "outside", "within", "built",
            "cotton", "irrigation",
        ),
        # a selection question is not a verdict and not a change detection
        "negative": (),
    },

    # Phase 10 -- the NDVI-difference request. Index-first phrasing: the user
    # names NDVI and a change. Deliberately does NOT use bare "change" as a
    # token, so a single-date NDVI question is never mistaken for this.
    Intent.NDVI_CHANGE_ROI: {
        "phrases": (
            "ndvi change", "change in ndvi", "change of ndvi", "the ndvi change",
            "ndvi difference", "difference in ndvi", "difference of ndvi",
            "ndvi difference between", "difference between the ndvi",
            "delta ndvi", "ndvi delta", "change in the ndvi",
            "compare ndvi", "compare the ndvi", "ndvi compared",
            "ndvi before and after", "ndvi between", "ndvi between two dates",
            "ndvi between these two dates", "ndvi increased", "ndvi decreased",
            "ndvi went up", "ndvi went down",
        ),
        "tokens": (
            "ndvi", "delta", "difference",
        ),
        # a two-date difference is not the same question as a spatial selection
        "negative": (
            "find areas", "find land", "where are", "which areas",
            "suitable", "suitability",
        ),
    },
    # Phase 10 -- the generic before/after request. Time-first phrasing: the
    # user describes an interval without naming an index.
    Intent.TEMPORAL_COMPARISON: {
        "phrases": (
            "compare before and after", "before and after",
            "compare before", "compare after",
            "between two dates", "between these two dates", "between the two dates",
            "compare two dates", "compare the two dates",
            "compare two scenes", "compare the two scenes",
            "two acquisitions", "compare acquisitions",
            "temporal comparison", "temporal change", "temporal difference",
            "how has this area changed", "how has the area changed",
            "how has it changed", "how has the land changed",
            "changed between", "change between",
        ),
        "tokens": (
            "temporal", "compare", "before", "after", "interval",
        ),
        # flood / water change is FLOOD_CHANGE and stays unsupported
        "negative": (
            "flood", "flooded", "flooding", "inundation", "ndwi",
        ),
    },
    Intent.CROP_SUITABILITY: {
        "phrases": (
            "crop suitability",
            "suitable for",
            "can i grow",
            "can we grow",
            "grow cotton",
            "grow wheat",
            "grow rice",
            "which crop",
            "what crop",
            "best crop",
            "crop recommendation",
            "is this land suitable",
        ),
        "tokens": (
            "suitability",
            "suitable",
            "cotton",
            "wheat",
            "rice",
            "maize",
            "sugarcane",
            "crop",
            "grow",
            "planting",
            "yield",
            "fertiliser",
            "fertilizer",
        ),
        "negative": (),
    },
    Intent.FLOOD_CHANGE: {
        "phrases": (
            "flood",
            "flooded area",
            "flooded areas",
            "flood extent",
            "flood change",
            "flood detection",
            "inundation",
            "water extent",
        ),
        "tokens": (
            "flood",
            "flooded",
            "flooding",
            "inundation",
            "inundated",
            # Phase 11: "ndwi" stays a TOKEN here (weight 1) so that a flood
            # question that merely mentions NDWI still reads as a flood question
            # and stays unsupported. It is no longer a PHRASE (weight 3), because
            # a plain NDWI question is an index question, not a flood one.
            "ndwi",
            "waterlogging",
        ),
        "negative": (),
    },
    # Phase 11 -- NDWI: an INDEX question, not a flood question and not a water
    # map. "water" alone is deliberately NOT a token: "cropland near water" and
    # "water proximity" are Phase 9 spatial queries and must keep routing there.
    Intent.NDWI_ROI_STATS: {
        "phrases": (
            "ndwi",
            "the ndwi",
            "water index", "the water index",
            "normalised difference water index",
            "normalized difference water index",
            "what is the ndwi", "what is the water index",
            "calculate ndwi", "calculate the water index",
            "show ndwi", "show the ndwi", "show the water index",
            "ndwi of this", "ndwi for this", "ndwi in this",
            "ndwi here", "ndwi of the", "ndwi for the",
            "water index of this", "water index for this",
            "water index here", "water index in this",
        ),
        "tokens": (
            "ndwi",
        ),
        # Anything below turns the request into something Phase 11 does not do:
        # flood mapping, water extraction, or a comparison across dates.
        "negative": (
            "flood", "flooded", "flooding", "inundation", "inundated",
            "extent", "quality", "irrigation", "groundwater", "depth",
            "segmentation", "extract", "extraction", "mask",
            "change", "changed", "compare", "before", "after",
            "difference", "delta", "between", "two dates",
            "trend", "time series", "timeseries",
            "near", "within", "proximity",
        ),
    },
    # Phase 11 -- NDWI between two dates: RECOGNISED and REFUSED. It exists so
    # the sentence cannot fall through to the NDVI change engine.
    Intent.TEMPORAL_NDWI: {
        "phrases": (
            "compare ndwi", "compare the ndwi", "compare the water index",
            "ndwi change", "change in ndwi", "change of ndwi",
            "change in the water index", "water index change",
            "ndwi before and after", "ndwi between",
            "ndwi difference", "difference in ndwi",
            "ndwi difference between", "ndwi difference for",
            "ndwi change between", "ndwi between two dates",
            "what is the ndwi change", "what is the water index change",
            "ndwi change here", "ndwi change in", "show the ndwi change",
            "show ndwi change", "water index change in",
            "temporal ndwi", "ndwi trend", "ndwi over time",
        ),
        "tokens": (
            "ndwi", "temporal",
        ),
        "negative": (),
    },
    # Phase 10: IMPLEMENTED. This is the vocabulary of "how has the vegetation
    # changed" / "find vegetation loss" -- the phrasing that names vegetation
    # rather than an index. "which parts" is listed because "which parts of this
    # area experienced vegetation decrease" is a temporal question, and without
    # it that sentence would tie with the spatial-query intent.
    Intent.VEGETATION_CHANGE: {
        "phrases": (
            "vegetation change",
            "change in vegetation",
            "change of vegetation",
            "vegetation loss",
            "loss of vegetation",
            "vegetation trend",
            "deforestation",
            "how has the vegetation",
            "how has vegetation changed",
            "how has the vegetation changed",
            "has the vegetation changed",
            "vegetation changed",
            "vegetation difference",
            "vegetation decrease",
            "decrease in vegetation",
            "vegetation decline",
            "decline in vegetation",
            "vegetation decreased",
            "vegetation increased",
            "which parts",
        ),
        "tokens": (
            "deforestation",
            "regrowth",
        ),
        "negative": (),
    },
}


def _score(normalized: str, spec: Dict[str, Sequence[str]]) -> Tuple[int, List[str], List[str]]:
    """Return (score, matched patterns, matched negatives) for one intent."""
    score = 0
    matched: List[str] = []
    covered: set[str] = set()

    for phrase in spec.get("phrases", ()):
        if phrase in normalized:
            score += _PHRASE_WEIGHT
            matched.append(phrase)
            covered.update(tokenize(phrase))

    for token in spec.get("tokens", ()):
        if token in covered:
            continue
        if re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", normalized):
            score += _TOKEN_WEIGHT
            matched.append(token)

    rejected: List[str] = []
    for bad in spec.get("negative", ()):
        if bad in normalized:
            score += _NEGATIVE_WEIGHT
            rejected.append(bad)

    return score, matched, rejected


def _explain(intent: Intent, matched: Sequence[str], rejected: Sequence[str],
             confidence: float) -> str:
    if intent is Intent.UNKNOWN:
        return ("No registered pattern matched strongly enough, or two intents "
                "tied. The system asks instead of guessing.")
    bits = [f"matched {', '.join(repr(m) for m in matched)}"] if matched else ["no explicit pattern"]
    if rejected:
        bits.append(f"rejected by {', '.join(repr(r) for r in rejected)}")
    return f"{intent.value}: {'; '.join(bits)} -> confidence {confidence:.2f}."


def parse_query(query: str) -> QueryIntent:
    """The one public entry point: text -> QueryIntent."""
    original = str(query or "")
    normalized = normalize(original)

    if not normalized:
        return QueryIntent(
            intent=Intent.UNKNOWN, confidence=0.0, original_query=original,
            normalized_query=normalized, matched=(), explanation="Empty query.",
            required_context=(),
        )

    scored: List[Tuple[int, Intent, List[str], List[str]]] = []
    for intent, spec in _PATTERN_TABLE.items():
        score, matched, rejected = _score(normalized, spec)
        if score > 0 and matched:
            scored.append((score, intent, matched, rejected))

    if not scored:
        return QueryIntent(
            intent=Intent.UNKNOWN, confidence=0.0, original_query=original,
            normalized_query=normalized, matched=(),
            explanation=_explain(Intent.UNKNOWN, (), (), 0.0),
            required_context=(),
        )

    scored.sort(key=lambda row: row[0], reverse=True)
    best, runner_up = scored[0], (scored[1] if len(scored) > 1 else None)
    score, intent, matched, rejected = best

    # A tie means the sentence is genuinely ambiguous: refuse to guess.
    if runner_up is not None and runner_up[0] == score:
        return QueryIntent(
            intent=Intent.UNKNOWN, confidence=0.0, original_query=original,
            normalized_query=normalized,
            matched=tuple(matched), rejected=tuple(rejected),
            explanation=(f"Ambiguous: '{intent.value}' and '{runner_up[1].value}' "
                         "scored equally. The system asks instead of guessing."),
            required_context=(),
        )

    confidence = max(0.0, min(1.0, score / _NORMALISER_MAX))
    if confidence < CONFIDENCE_THRESHOLD:
        return QueryIntent(
            intent=Intent.UNKNOWN, confidence=confidence, original_query=original,
            normalized_query=normalized, matched=tuple(matched),
            explanation=("Weak match "
                         f"({', '.join(repr(m) for m in matched)} -> {confidence:.2f}), "
                         "below the acceptance threshold."),
            required_context=(),
        )

    # Phase 9: an unsupported requirement (irrigation, groundwater, salinity)
    # blocks execution whatever the intent is -- the engines must never answer
    # an irrigation question with a rainfed result.
    from .spatial_query import detect_unsupported_requirements

    blocked = detect_unsupported_requirements(original)
    explanation = _explain(intent, matched, rejected, confidence)
    if blocked:
        explanation = (
            f"{explanation} Blocked: "
            f"{', '.join(sorted({str(c.parameters.get('topic')) for c in blocked}))} "
            "is requested but not measured.").strip()

    # Phase 12: composition is decided by WHAT THE SENTENCE STATES, not by
    # pattern scores. A sentence is a composition when it names a spectral
    # threshold or combines a temporal change class with something else; a
    # purely spatial sentence stays a Phase 9 query, and a two-date comparison
    # stays with Phase 10 (or with the TEMPORAL_NDWI refusal).
    composed = None
    if intent in COMPOSITION_CANDIDATES:
        from .multi_condition import parse_composed_query    # local: see below

        composed = parse_composed_query(original)
        if composed.is_composition:
            intent = Intent.MULTI_CONDITION
            matched = tuple(matched) + ("composition",)
            explanation = (
                f"{explanation} Composed: {composed.expression}.").strip()
        elif intent is Intent.MULTI_CONDITION:
            # Scored as a composition but nothing spectral/temporal was stated:
            # it is an ordinary Phase 9 spatial question.
            intent = Intent.SPATIAL_QUERY

    parsed = QueryIntent(
        intent=intent, confidence=confidence, original_query=original,
        normalized_query=normalized, matched=tuple(matched),
        rejected=tuple(rejected), explanation=explanation,
        required_context=REQUIRED_CONTEXT.get(intent, ()),
        blocked_by=blocked,
        composed=composed,
    )

    # Phase 9: fill in the structured conditions for a spatial query. Imported
    # here (not at module scope) because core.spatial_query imports `normalize`
    # from this module -- a lazy import keeps the two modules acyclic.
    if intent is Intent.SPATIAL_QUERY:
        from dataclasses import replace

        from .spatial_query import parse_spatial_query

        spatial = parse_spatial_query(original)
        explanation = parsed.explanation
        if spatial.status.value != "ok":
            detail = "; ".join(spatial.warnings + spatial.notes)[:200]
            explanation = (f"{explanation} Spatial parsing: "
                           f"{spatial.status.value} ({detail})").strip()
        # QueryIntent is frozen: build a new one carrying the conditions.
        return replace(parsed, conditions=tuple(spatial.conditions),
                       explanation=explanation)

    return parsed


def supported_examples() -> Tuple[str, ...]:
    """Example phrasings, used for UI hints and tests (still no engine here)."""
    return (
        "What is the NDVI of this area?",
        "Calculate the vegetation index here.",
        "Show vegetation health in this area.",
        "Analyze the vegetation in this selected area.",
    )
