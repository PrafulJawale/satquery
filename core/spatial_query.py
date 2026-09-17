"""Phase 9 -- multi-condition spatial query: MODELS and DETERMINISTIC PARSER.

Scope of this module (Checkpoint B): turn a sentence into a structured
`SpatialQuery`. It contains **no raster processing, no distance transform, no
data access and no Streamlit** -- those belong to later checkpoints.

The architecture it serves:

    raw query
        -> core/router.py            decides: this is a SPATIAL_QUERY
        -> core/spatial_query.py     builds the structured SpatialQuery
        -> analyses/registry.py      (Checkpoint C) binds it to an engine
        -> core/spatial.py           (Checkpoint C) combines the masks

Two rules govern everything here:

1. **No free-form strings survive parsing.** After `parse_spatial_query()` the
   request is enums, class codes and metre values -- never prose.
2. **An unsupported condition aborts the query.** "flood-prone" never becomes
   "water", and "irrigation" never becomes "near water". The parser reports
   what it cannot do instead of substituting a proxy.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .router import normalize

__all__ = [
    "SPATIAL_CONFIG_DIR",
    "ConditionType",
    "ConditionStatus",
    "SpatialQueryStatus",
    "Operator",
    "RequestedOutput",
    "SpatialCondition",
    "SpatialQuery",
    "load_spatial_config",
    "parse_distance",
    "parse_spatial_query",
    "detect_unsupported_requirements",
    "VERDICT_BLOCKING_TOPICS",
]

SPATIAL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "spatial")


# =========================================================================== #
# configuration
# =========================================================================== #
def load_spatial_config(name: str = "conditions",
                        base_dir: str = SPATIAL_CONFIG_DIR) -> Dict[str, Any]:
    """Load config/spatial/<name>.yml (mirrors core.suitability.load_crop_config)."""
    path = os.path.join(base_dir, f"{name}.yml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Spatial configuration is not a mapping: {path}")
    return cfg


# =========================================================================== #
# enums -- the vocabulary of a spatial query
# =========================================================================== #
class ConditionType(str, Enum):
    """What kind of evidence a condition needs."""

    CROP_SUITABILITY = "crop_suitability"      # Phase 8 screening class
    LAND_COVER_CLASS = "land_cover_class"      # a WorldCover class
    # WATER = the cell IS mapped permanent water (WorldCover class 80).
    # WATER_PROXIMITY = the cell is WITHIN N METRES of such a cell.
    # They are not interchangeable: "excluding water" is the first,
    # "near water" is the second, and NOT-water is not NOT-near-water.
    WATER = "water"
    WATER_PROXIMITY = "water_proximity"
    UNSUPPORTED = "unsupported"                # recognised, but not measurable


class ConditionStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"    # recognised, refused, never proxied


class SpatialQueryStatus(str, Enum):
    OK = "ok"                                  # at least one executable condition
    NEEDS_CLARIFICATION = "needs_clarification"  # unsupported / mixed operators
    NO_CONDITIONS = "no_conditions"            # nothing recognisable was asked
    NEEDS_ROI = "needs_roi"                    # parsed fine, but no area selected


class Operator(str, Enum):
    """How the conditions are combined.

    `NOT` exists for completeness but is stored **per condition**
    (`SpatialCondition.negate`): "cotton AND NOT built-up" is not an operator
    between two queries, it is a negation of one of them.
    """

    AND = "and"
    OR = "or"
    NOT = "not"


class RequestedOutput(str, Enum):
    """What the user wants back (the MVP always computes the same masks)."""

    AREAS = "areas"        # where the matching cells are + how much (default)
    COUNT = "count"        # how many / how much, emphasised over the map


# =========================================================================== #
# models
# =========================================================================== #
@dataclass(frozen=True)
class SpatialCondition:
    """One structured condition. Carries no rasters and does no computation."""

    condition_type: ConditionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    #: which engine/datasource can answer it ("crop_suitability", "worldcover")
    required_analysis: Optional[str] = None
    status: ConditionStatus = ConditionStatus.SUPPORTED
    negate: bool = False
    #: the phrases in the user's sentence that produced this condition
    evidence: Tuple[str, ...] = ()
    #: one sentence for the UI: what this condition means, in plain words
    interpretation: str = ""
    #: why it is unsupported, or what could be done instead
    note: str = ""

    # -- helpers ----------------------------------------------------------- #
    @property
    def label(self) -> str:
        prefix = "NOT " if self.negate else ""
        if self.condition_type is ConditionType.CROP_SUITABILITY:
            return (f"{prefix}cotton suitability (class >= "
                    f"{self.parameters.get('min_class')}, "
                    f"{self.parameters.get('scenario')})")
        if self.condition_type is ConditionType.LAND_COVER_CLASS:
            return (f"{prefix}land cover class "
                    f"{self.parameters.get('classes')} "
                    f"({self.parameters.get('class_name', '')})")
        if self.condition_type is ConditionType.WATER:
            return (f"{prefix}water (WorldCover class "
                    f"{self.parameters.get('classes')})")
        if self.condition_type is ConditionType.WATER_PROXIMITY:
            return (f"{prefix}within {self.parameters.get('distance_m')} m of "
                    f"mapped permanent water (class "
                    f"{self.parameters.get('water_class')})")
        return f"{prefix}unsupported condition"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_type": self.condition_type.value,
            "parameters": dict(self.parameters),
            "required_analysis": self.required_analysis,
            "status": self.status.value,
            "negate": self.negate,
            "evidence": list(self.evidence),
            "interpretation": self.interpretation,
            "note": self.note,
            "label": self.label,
        }


@dataclass
class SpatialQuery:
    """The structured, auditable form of a spatial request."""

    original_query: str
    normalized_query: str = ""
    operator: Operator = Operator.AND
    conditions: Tuple[SpatialCondition, ...] = ()
    status: SpatialQueryStatus = SpatialQueryStatus.OK
    requested_output: RequestedOutput = RequestedOutput.AREAS
    warnings: List[str] = field(default_factory=list)
    #: interpretation notes the UI MUST show (water != irrigation, etc.)
    notes: List[str] = field(default_factory=list)

    # -- derived views ------------------------------------------------------ #
    @property
    def is_executable(self) -> bool:
        """True only when every condition is supported and at least one exists."""
        return (self.status is SpatialQueryStatus.OK
                and bool(self.conditions)
                and all(c.status is ConditionStatus.SUPPORTED
                        for c in self.conditions))

    @property
    def supported_conditions(self) -> Tuple[SpatialCondition, ...]:
        return tuple(c for c in self.conditions
                     if c.status is ConditionStatus.SUPPORTED)

    @property
    def unsupported_conditions(self) -> Tuple[SpatialCondition, ...]:
        return tuple(c for c in self.conditions
                     if c.status is ConditionStatus.UNSUPPORTED)

    def expression(self) -> str:
        """The combination as one auditable line, e.g.
        `crop_suitability(cotton, class >= 3) AND water_proximity(<= 1000 m)`."""
        if not self.conditions:
            return "(no conditions recognised)"
        parts: List[str] = []
        for c in self.conditions:
            if c.condition_type is ConditionType.CROP_SUITABILITY:
                inner = (f"crop_suitability({c.parameters.get('crop')}, "
                         f"class >= {c.parameters.get('min_class')}, "
                         f"{c.parameters.get('scenario')})")
            elif c.condition_type is ConditionType.LAND_COVER_CLASS:
                inner = f"land_cover({c.parameters.get('classes')})"
            elif c.condition_type is ConditionType.WATER:
                inner = f"water(class {c.parameters.get('classes')})"
            elif c.condition_type is ConditionType.WATER_PROXIMITY:
                inner = (f"water_proximity(<= {c.parameters.get('distance_m')} m, "
                         f"class {c.parameters.get('water_class')})")
            else:
                inner = f"UNSUPPORTED({c.parameters.get('topic', 'condition')})"
            parts.append(f"NOT {inner}" if c.negate else inner)
        joiner = f" {self.operator.value.upper()} "
        return joiner.join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "operator": self.operator.value,
            "status": self.status.value,
            "requested_output": self.requested_output.value,
            "executable": self.is_executable,
            "expression": self.expression(),
            "conditions": [c.to_dict() for c in self.conditions],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


# =========================================================================== #
# parsing helpers
# =========================================================================== #
_DISTANCE_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*"
    r"(km|kilometres?|kilometers?|m|metres?|meters?)(?![\w])")


def parse_distance(text: str, default_m: float, units: Dict[str, float],
                   min_m: float, max_m: float
                   ) -> Tuple[float, Optional[str], Optional[str]]:
    """Return (metres, evidence, warning).

    The default comes from configuration; an explicit "1 km" / "500 m" in the
    sentence overrides it. Absurd values are clamped and the clamp is reported
    so the UI can say what distance was actually used.
    """
    match = _DISTANCE_RE.search(text or "")
    if not match:
        # no distance stated: the configured default, reported as such
        default = float(default_m)
        return (int(default) if default.is_integer() else default), None, None

    value = float(match.group(1))
    unit = match.group(2)
    factor = 1.0
    for key, mult in units.items():
        if unit == key or unit.startswith(key):
            factor = float(mult)
            break
    metres = value * factor
    evidence = match.group(0).strip()

    warning = None
    if metres < min_m:
        warning = (f"The requested distance {evidence} is smaller than one "
                   f"analysis cell; {min_m:g} m was used instead.")
        metres = float(min_m)
    elif metres > max_m:
        warning = (f"The requested distance {evidence} exceeds the supported "
                   f"maximum; {max_m:g} m was used instead.")
        metres = float(max_m)
    if float(metres).is_integer():
        metres = int(metres)
    return metres, evidence, warning


def _matches(text: str, phrase: str) -> bool:
    """Match a vocabulary entry.

    Multi-word phrases are matched as substrings ("but not"), single words on
    WORD BOUNDARIES -- otherwise "or" would match inside "for" and every
    "suitable for cotton" query would be parsed as an OR.
    """
    phrase = phrase.strip()
    if not phrase:
        return False
    if " " in phrase:
        return phrase in text
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text) is not None


def _vocab(spec: Any) -> Tuple[str, ...]:
    """Accept either a plain list of phrases or {"phrases": [...]}."""
    if spec is None:
        return ()
    if isinstance(spec, dict):
        return tuple(str(v) for v in spec.get("phrases", ()))
    return tuple(str(v) for v in spec)


def _find(text: str, phrases: Sequence[str], tokens: Sequence[str]
          ) -> Tuple[Optional[int], Tuple[str, ...]]:
    """Earliest match position and the matched vocabulary (phrases win).

    Tokens are matched on word boundaries, so "somewhere" never matches "where"
    and "irrigation" is never hidden inside another word.
    """
    hits: List[Tuple[int, str]] = []
    for phrase in phrases:
        pos = text.find(phrase)
        if pos >= 0:
            hits.append((pos, phrase))
    for token in tokens:
        if token in ("cotton",):                    # bare nouns count
            pos = text.find(token)
            if pos >= 0:
                hits.append((pos, token))
            continue
        m = re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", text)
        if m:
            hits.append((m.start(), token))
    if not hits:
        return None, ()
    hits.sort(key=lambda row: row[0])
    best_pos = hits[0][0]
    matched = tuple(sorted({word for pos, word in hits if pos == best_pos}))
    return best_pos, matched


def _negated(text: str, position: int, negations: Sequence[str],
             matched_word: str = "") -> Tuple[bool, Tuple[str, ...]]:
    """True when a negation phrase governs the condition at `position`.

    Two cases, because a negation can either precede the condition
    ("but not built-up") or OVERLAP it ("not water" -- the word "water" is part
    of the negation phrase itself).
    """
    window = text[max(0, position - 40):position]
    segment = (text[:position] + matched_word).rstrip() if matched_word \
        else text[:position].rstrip()
    found = tuple(n.strip() for n in negations
                  if n.strip() and (n.strip() in window
                                    or segment.endswith(n.strip())))
    return bool(found), found


#: Topics that must block a SUITABILITY VERDICT as well as a spatial query.
#: Flood is deliberately absent: it already has its own intent (FLOOD_CHANGE)
#: and its own "not available yet" message, which must not change.
VERDICT_BLOCKING_TOPICS: Tuple[str, ...] = ("irrigation", "groundwater", "salinity")


def detect_unsupported_requirements(
        text: str,
        patterns: Optional[Dict[str, Any]] = None,
        topics: Optional[Sequence[str]] = None) -> Tuple[SpatialCondition, ...]:
    """Recognised-but-unmeasurable requirements, for ANY kind of question.

    Used by the router so that "Can I grow cotton with irrigation?" is refused
    BEFORE the crop-suitability engine runs, instead of quietly returning a
    rainfed verdict. Never substitutes a proxy: irrigation is not "near water".
    """
    pat = patterns if patterns is not None else load_spatial_config("patterns")
    unsupported_cfg = pat.get("unsupported", {})
    wanted = tuple(topics) if topics is not None else VERDICT_BLOCKING_TOPICS

    found: List[SpatialCondition] = []
    for topic, spec in unsupported_cfg.items():
        if topic not in wanted:
            continue
        pos, matched = _find(normalize(str(text or "")), spec.get("phrases", ()),
                             spec.get("tokens", ()))
        if pos is None:
            continue
        found.append(SpatialCondition(
            condition_type=ConditionType.UNSUPPORTED,
            parameters={"topic": topic,
                        "nearest_supported": spec.get("nearest_supported")},
            status=ConditionStatus.UNSUPPORTED,
            evidence=matched,
            interpretation=f"'{matched[0] if matched else topic}' is not measured.",
            note=" ".join(str(spec.get("blocking_message")
                              or spec.get("message", "")).split()),
        ))
    return tuple(found)


# =========================================================================== #
# the parser
# =========================================================================== #
def parse_spatial_query(text: str, config: Optional[Dict[str, Any]] = None,
                        patterns: Optional[Dict[str, Any]] = None
                        ) -> SpatialQuery:
    """Sentence -> SpatialQuery. Deterministic; never guesses a proxy."""
    original = str(text or "")
    normalized = normalize(original)
    cfg = config if config is not None else load_spatial_config("conditions")
    pat = patterns if patterns is not None else load_spatial_config("patterns")

    cond_cfg = pat.get("conditions", {})
    unsupported_cfg = pat.get("unsupported", {})
    operators = pat.get("operators", {})
    negations = _vocab(operators.get("negation"))
    or_phrases = _vocab(operators.get("or"))
    and_phrases = _vocab(operators.get("and"))

    warnings: List[str] = []
    notes: List[str] = []

    # ---- 1. unsupported conditions abort everything ----------------------- #
    # Checked FIRST so that "cotton outside flood-prone areas" can never be
    # quietly rewritten as "cotton outside permanent water".
    unsupported: List[SpatialCondition] = []
    for topic, spec in unsupported_cfg.items():
        pos, matched = _find(normalized, spec.get("phrases", ()),
                             spec.get("tokens", ()))
        if pos is None:
            continue
        message = " ".join(str(spec.get("message", "")).split())
        unsupported.append(SpatialCondition(
            condition_type=ConditionType.UNSUPPORTED,
            parameters={"topic": topic,
                        "nearest_supported": spec.get("nearest_supported")},
            required_analysis=None,
            status=ConditionStatus.UNSUPPORTED,
            evidence=matched,
            interpretation=f"'{matched[0] if matched else topic}' is not measured by this system.",
            note=message,
        ))

    if unsupported:
        return SpatialQuery(
            original_query=original, normalized_query=normalized,
            status=SpatialQueryStatus.NEEDS_CLARIFICATION,
            conditions=tuple(unsupported),
            warnings=[c.note for c in unsupported if c.note],
            notes=[("No computation was performed: an unsupported condition "
                    "is never replaced by a proxy.")],
        )

    # ---- 2. supported conditions (fixed order -> reproducible structure) -- #
    water_cfg = cfg.get("water", {})
    land_cfg = cfg.get("land_cover", {})
    cotton_cfg = cfg.get("cotton", {})

    distance_m, distance_evidence, distance_warning = parse_distance(
        normalized,
        default_m=float(water_cfg.get("default_proximity_m", 1000)),
        units=dict(cfg.get("distance_units", {})),
        min_m=float(water_cfg.get("min_proximity_m", 30)),
        max_m=float(water_cfg.get("max_proximity_m", 10000)))
    if distance_warning:
        warnings.append(distance_warning)

    found: List[Tuple[str, SpatialCondition]] = []   # (sort key, condition)

    # 2a. cotton suitability
    pos, matched = _find(normalized, cond_cfg.get("cotton_suitability", {}).get("phrases", ()),
                         cond_cfg.get("cotton_suitability", {}).get("tokens", ()))
    if pos is not None:
        neg, neg_words = _negated(normalized, pos, negations, matched[0] if matched else "")
        found.append(("1_cotton", SpatialCondition(
            condition_type=ConditionType.CROP_SUITABILITY,
            parameters={"crop": cotton_cfg.get("crop", "cotton"),
                        "min_class": int(cotton_cfg.get("min_suitability_class", 3)),
                        "scenario": cotton_cfg.get("scenario", "rainfed")},
            required_analysis=cotton_cfg.get("required_analysis", "crop_suitability"),
            negate=neg,
            evidence=matched + neg_words,
            interpretation=" ".join(str(cotton_cfg.get("interpretation", "")).split()),
        )))

    # 2b. cropland (a land-cover class -- NOT a suitability statement)
    pos, matched = _find(normalized, cond_cfg.get("cropland", {}).get("phrases", ()),
                         cond_cfg.get("cropland", {}).get("tokens", ()))
    if pos is not None:
        neg, neg_words = _negated(normalized, pos, negations, matched[0] if matched else "")
        found.append(("2_cropland", SpatialCondition(
            condition_type=ConditionType.LAND_COVER_CLASS,
            parameters={"classes": [int(land_cfg.get("cropland_class", 40))],
                        "class_name": land_cfg.get("class_names", {}).get(
                            int(land_cfg.get("cropland_class", 40)), "Cropland")},
            required_analysis=land_cfg.get("required_analysis", "worldcover"),
            negate=neg,
            evidence=matched + neg_words,
            interpretation=" ".join(str(land_cfg.get("cropland_interpretation", "")).split()),
        )))

    # 2c. water: PROXIMITY needs a cue + water; the bare word means the CLASS.
    water_word: Optional[Tuple[int, str]] = None
    for candidate in list(cond_cfg.get("water", {}).get("phrases", ())) + ["water"]:
        m = re.search(rf"(?<![a-z]){re.escape(str(candidate))}(?![a-z])", normalized)
        if m:
            water_word = (m.start(), str(candidate))
            break

    cue_pos: Optional[int] = None
    cue_word: Optional[str] = None
    for cue in list(cond_cfg.get("water_proximity", {}).get("phrases", ())) + \
            list(cond_cfg.get("water_proximity", {}).get("tokens", ())):
        cue = str(cue).strip()
        if not cue:
            continue
        m = (re.search(re.escape(cue), normalized) if " " in cue else
             re.search(rf"(?<![a-z]){re.escape(cue)}(?![a-z])", normalized))
        if m and (cue_pos is None or m.start() < cue_pos):
            cue_pos, cue_word = m.start(), cue

    pos: Optional[int] = None
    matched: Tuple[str, ...] = ()
    if water_word is not None and cue_pos is not None:
        pos = min(water_word[0], cue_pos)
        matched = tuple(sorted({cue_word or "", water_word[1]} - {""}))
    elif water_word is not None:
        pos, matched = water_word[0], (water_word[1],)

    if pos is not None and cue_pos is not None:
        neg, neg_words = _negated(normalized, pos, negations, matched[0] if matched else "")
        found.append(("3_water", SpatialCondition(
            condition_type=ConditionType.WATER_PROXIMITY,
            parameters={"distance_m": distance_m,
                        "water_class": int(water_cfg.get("water_class", 80)),
                        "distance_evidence": distance_evidence,
                        "distance_convention": " ".join(
                            str(water_cfg.get("distance_convention", "")).split())},
            required_analysis=water_cfg.get("required_analysis", "worldcover"),
            negate=neg,
            evidence=matched + neg_words,
            interpretation=" ".join(str(water_cfg.get("interpretation", "")).split()),
        )))
        notes.append(" ".join(str(water_cfg.get("interpretation", "")).split()))
    else:
        # 2c'. WATER ITSELF -- the cell is mapped permanent water (class 80).
        # "near water" selects a buffer around water; "excluding water" removes
        # the water cells. Conflating the two would silently change the meaning
        # of the query.
        pos_w, matched_w = pos, matched
        if pos_w is not None:
            neg, neg_words = _negated(normalized, pos_w, negations, matched_w[0] if matched_w else "")
            found.append(("3_water", SpatialCondition(
                condition_type=ConditionType.WATER,
                parameters={"classes": [int(water_cfg.get("water_class", 80))],
                            "class_name": land_cfg.get("class_names", {}).get(
                                int(water_cfg.get("water_class", 80)),
                                "Permanent water bodies")},
                required_analysis=water_cfg.get("required_analysis", "worldcover"),
                negate=neg,
                evidence=matched_w + neg_words,
                interpretation=(" ".join(
                    str(water_cfg.get("water_class_interpretation", "")).split())
                    or ("The cell itself is mapped as permanent surface water "
                        "(WorldCover class 80, 2021). It is not flooding, not "
                        "irrigation and not soil moisture.")),
            )))

    # 2d. built-up (usually negated: "but not built-up")
    pos, matched = _find(normalized, cond_cfg.get("built_up", {}).get("phrases", ()),
                         cond_cfg.get("built_up", {}).get("tokens", ()))
    if pos is not None:
        neg, neg_words = _negated(normalized, pos, negations, matched[0] if matched else "")
        found.append(("4_builtup", SpatialCondition(
            condition_type=ConditionType.LAND_COVER_CLASS,
            parameters={"classes": [int(land_cfg.get("built_up_class", 50))],
                        "class_name": land_cfg.get("class_names", {}).get(
                            int(land_cfg.get("built_up_class", 50)), "Built-up")},
            required_analysis=land_cfg.get("required_analysis", "worldcover"),
            negate=neg,
            evidence=matched + neg_words,
            interpretation=("Built-up areas are mapped land cover (2021), not a "
                            "suitability judgement."),
        )))

    conditions = tuple(c for _, c in sorted(found, key=lambda row: row[0]))

    if not conditions:
        return SpatialQuery(
            original_query=original, normalized_query=normalized,
            status=SpatialQueryStatus.NO_CONDITIONS,
            warnings=["No supported spatial condition was recognised in the "
                      "question."],
        )

    # ---- 3. operators ----------------------------------------------------- #
    has_or = any(_matches(normalized, p) for p in or_phrases)
    has_and = any(_matches(normalized, p) for p in and_phrases)
    operator = Operator.OR if (has_or and len(conditions) > 1) else Operator.AND

    if has_or and has_and:
        return SpatialQuery(
            original_query=original, normalized_query=normalized,
            operator=Operator.AND, conditions=conditions,
            status=SpatialQueryStatus.NEEDS_CLARIFICATION,
            warnings=warnings, notes=notes + [
                "The question mixes AND and OR. I will not guess the order of "
                "operations: please ask one combination at a time."],
        )

    # ---- 4. a bare "not" outside the recognised grammar is ambiguous ------ #
    # A "not" the grammar could not attach to a condition is refused, not guessed.
    not_count = len(re.findall(r"(?<![a-z])not(?![a-z])", normalized))
    negated_count = sum(1 for c in conditions if c.negate)
    if not_count > negated_count:
        # a "not" the grammar does not understand: refuse instead of guessing
        return SpatialQuery(
            original_query=original, normalized_query=normalized,
            operator=operator, conditions=conditions,
            status=SpatialQueryStatus.NEEDS_CLARIFICATION,
            warnings=warnings, notes=notes + [
                "The question contains a negation I cannot attach to a "
                "condition with confidence. Please phrase it as "
                "'excluding …', 'outside …' or 'but not …'."],
        )

    # ---- 5. ambiguity the parser MAY resolve, but must announce ----------- #
    ambiguity = pat.get("ambiguity", {})
    for key, spec in ambiguity.items():
        pos, matched = _find(normalized, spec.get("phrases", ()), ())
        if pos is not None:
            note = " ".join(str(spec.get("note", "")).split())
            if note and note not in notes:
                notes.append(note)

    return SpatialQuery(
        original_query=original, normalized_query=normalized,
        operator=operator, conditions=conditions,
        status=SpatialQueryStatus.OK, warnings=warnings, notes=notes,
    )
