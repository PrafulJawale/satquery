"""Phase 12 -- composed (multi-condition) geospatial queries: PARSING and MASKS.

WHAT THIS MODULE IS
-------------------
The vocabulary and the mask algebra for combining evidence that ALREADY
exists:

    spatial  (Phase 9:  cropland, permanent water, near water, NOT water)
    spectral (Phase 11: NDVI / NDWI thresholded on native index rasters)
    temporal (Phase 10: NDVI increase / stable / decrease classes)

It computes no science of its own. Spatial conditions are produced by Phase 9's
own parser (so "cropland" and "near water" cannot drift from Phase 9), the index
arithmetic belongs to `core.indices`, the change classes to
`analyses.ndvi_change`, and the three-valued combination to `core.spatial`.

WHAT IT REFUSES TO DO
---------------------
* Invent a threshold. "high NDVI" with no number is a NEEDS_THRESHOLD outcome,
  not a hidden 0.6. Conventions exist in config/multi/thresholds.yml and are
  applied only when the user enables one, always labelled as a convention.
* Combine masks that do not share one grid. `require_compatible()` decides;
  a mismatch is an error, never a silent resample.
* Turn a combination into a cause. The caveats live in the configuration and
  travel with every result.

LAYERING
--------
This module imports `core.spatial` (masks) and `core.spatial_query` (the Phase 9
parser). It must NOT import `analyses.*` -- the engines import it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.spatial import Grid, GridMask, combine_all, negate

MULTI_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "multi"
)


# =========================================================================== #
# vocabulary
# =========================================================================== #
class ConditionKind(str, Enum):
    SPATIAL = "spatial"        # delegated to Phase 9's parser
    SPECTRAL = "spectral"      # NDVI / NDWI threshold
    TEMPORAL = "temporal"      # Phase 10 NDVI change class


class ComposedQueryStatus(str, Enum):
    OK = "ok"
    NO_CONDITIONS = "no_conditions"
    NEEDS_THRESHOLD = "needs_threshold"
    NEEDS_CLARIFICATION = "needs_clarification"


class ThresholdProvenance(str, Enum):
    USER_SPECIFIED = "user_specified"
    CONFIG_CONVENTION = "config_convention"
    RELATIVE = "relative"


#: Temporal class codes, mirroring config/temporal/ndvi_change.yml. Kept here
#: (rather than imported from analyses.ndvi_change) to respect the layering
#: rule; tests/test_phase12_multi_condition.py asserts they agree with the file.
TEMPORAL_CLASS_CODES: Dict[str, int] = {
    "insufficient": 0,
    "decrease": 1,
    "stable": 2,
    "increase": 3,
}
TEMPORAL_CODE_NAMES: Dict[int, str] = {v: k for k, v in TEMPORAL_CLASS_CODES.items()}

#: Index aliases -> the canonical index name from config/indices/<name>.yml
INDEX_ALIASES: Dict[str, str] = {
    "ndvi": "ndvi",
    "vegetation index": "ndvi",
    "ndwi": "ndwi",
    "water index": "ndwi",
    "normalised difference water index": "ndwi",
    "normalized difference water index": "ndwi",
}

#: Ordered longest-first so "vegetation index" wins over "ndvi"-style tokens.
_INDEX_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(INDEX_ALIASES, key=len, reverse=True)),
    re.IGNORECASE,
)

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")

_OPERATORS: Tuple[Tuple[str, str], ...] = (
    (r"greater\s+than\s+or\s+equal\s+to|at\s+least|>=", ">="),
    (r"less\s+than\s+or\s+equal\s+to|at\s+most|<=", "<="),
    (r"greater\s+than|above|over|higher\s+than|more\s+than|>", ">"),
    (r"less\s+than|below|under|lower\s+than|<", "<"),
)

_TEMPORAL_WORDS: Dict[str, str] = {
    "increase": "increase", "increased": "increase", "increasing": "increase",
    "gain": "increase", "greener": "increase",
    "decrease": "decrease", "decreased": "decrease", "decreasing": "decrease",
    "decline": "decrease", "loss": "decrease", "reduction": "decrease",
    "stable": "stable", "unchanged": "stable", "no change": "stable",
}

#: Words that suggest a threshold without supplying one.
_RELATIVE_WORDS = ("above the median", "below the median", "above median",
                   "below median", "above average", "below average")

#: Temporal phrasings that name no index ("vegetation decrease"). Recognised
#: here so they can be COMPOSED with spatial conditions; they are still produced
#: by Phase 10's engine, never recomputed.
_TEMPORAL_PHRASES: Tuple[Tuple[str, str], ...] = (
    (r"\bvegetation\s+(?:decrease|decline|loss|reduction|drop)\b", "decrease"),
    (r"\bvegetation\s+(?:increase|gain|growth|greening)\b", "increase"),
    (r"\bvegetation\s+(?:stable|unchanged)\b", "stable"),
    (r"\bgreenness\s+(?:decrease|loss|decline)\b", "decrease"),
    (r"\bgreenness\s+(?:increase|gain)\b", "increase"),
)

#: Sentences that are a TWO-DATE comparison, not a composition. "Compare NDWI
#: before and after" must reach the temporal-NDWI intent (which refuses it),
#: never be read as "NDWI, threshold unspecified".
_TEMPORAL_COMPARISON_CUES: Tuple[str, ...] = (
    r"\bbefore\s+and\s+after\b",
    r"\bbetween\s+(?:these\s+|the\s+)?two\s+dates\b",
    r"\bbetween\s+the\s+two\s+scenes\b",
    r"\bchange\s+between\b",
    r"\bdifference\s+between\b",
    r"\bcompare\b",
)

#: Quality words that turn a bare index mention into a threshold request
#: ("high NDVI"). Without one of these -- "What is the NDVI of this area?" --
#: the sentence is a statistics question, not a condition, and NO spectral
#: condition is created. That is what keeps Phase 3 / Phase 11 routing intact.
_QUALITY_WORDS: Tuple[str, ...] = (
    "high", "low", "dense", "sparse", "strong", "weak", "healthy", "poor",
    "green", "bare", "more", "less",
)

#: Query forms that ask for STATISTICS over the matched area rather than acting
#: as an additional filter (e.g. "NDVI decrease and NDWI statistics").
_SUMMARY_PATTERN = re.compile(
    r"\b(ndvi|ndwi)\b[^.;]{0,24}?\b(statistics|stats|summary|mean|average|values)\b",
    re.IGNORECASE)


def load_threshold_config(name: str = "thresholds",
                          base_dir: str = MULTI_CONFIG_DIR) -> Dict[str, Any]:
    import yaml

    with open(os.path.join(base_dir, f"{name}.yml"), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg if isinstance(cfg, dict) else {}


# =========================================================================== #
# the condition model
# =========================================================================== #
@dataclass(frozen=True)
class ThresholdSpec:
    """A threshold and, always, where it came from."""

    index: str
    operator: str = ">"
    value: Optional[float] = None
    provenance: str = ThresholdProvenance.USER_SPECIFIED.value
    #: e.g. "your query", "convention:ndvi_high", "relative:median"
    detail: str = ""
    note: str = ""

    @property
    def is_defined(self) -> bool:
        return self.value is not None and self.operator in (">", ">=", "<", "<=")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "operator": self.operator,
            "value": self.value,
            "provenance": self.provenance,
            "detail": self.detail,
            "note": self.note,
            "defined": self.is_defined,
        }


@dataclass(frozen=True)
class ComposedCondition:
    """One condition of a composed query. Carries no rasters."""

    kind: ConditionKind
    name: str                       # "cropland" | "ndvi_gt" | "ndvi_decrease"
    label: str                      # one line for the UI
    source_analysis: str            # "worldcover" | "core.indices:ndvi" | "analyses.ndvi_change"
    parameters: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[ThresholdSpec] = None
    negate: bool = False
    #: the Phase 9 SpatialCondition, when kind is SPATIAL (never re-derived)
    spatial_condition: Optional[Any] = None
    evidence: Tuple[str, ...] = ()
    interpretation: str = ""
    limitations: Tuple[str, ...] = ()

    @property
    def needs_threshold(self) -> bool:
        return self.kind is ConditionKind.SPECTRAL and (
            self.threshold is None or not self.threshold.is_defined)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "label": self.label,
            "source_analysis": self.source_analysis,
            "parameters": dict(self.parameters),
            "threshold": self.threshold.to_dict() if self.threshold else None,
            "negate": self.negate,
            "evidence": list(self.evidence),
            "interpretation": self.interpretation,
            "limitations": list(self.limitations),
            "needs_threshold": self.needs_threshold,
        }


@dataclass
class ComposedQuery:
    original_query: str = ""
    normalized_query: str = ""
    conditions: Tuple[ComposedCondition, ...] = ()
    operator: str = "and"
    status: ComposedQueryStatus = ComposedQueryStatus.NO_CONDITIONS
    #: indices asked for as EVIDENCE (statistics over the matched area), not
    #: as filters -- e.g. "NDVI decrease and NDWI statistics"
    summary_requests: Tuple[str, ...] = ()
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def spectral_conditions(self) -> Tuple[ComposedCondition, ...]:
        return tuple(c for c in self.conditions if c.kind is ConditionKind.SPECTRAL)

    @property
    def temporal_conditions(self) -> Tuple[ComposedCondition, ...]:
        return tuple(c for c in self.conditions if c.kind is ConditionKind.TEMPORAL)

    @property
    def spatial_conditions(self) -> Tuple[ComposedCondition, ...]:
        return tuple(c for c in self.conditions if c.kind is ConditionKind.SPATIAL)

    @property
    def missing_thresholds(self) -> Tuple[ComposedCondition, ...]:
        return tuple(c for c in self.conditions if c.needs_threshold)

    @property
    def is_composition(self) -> bool:
        """True when the sentence is MORE than an existing single analysis.

        * a spectral threshold: only Phase 12 can produce a threshold map, so
          any spectral condition is a composition;
        * a temporal change class: only when it is combined with something
          else. A lone "find vegetation loss" is Phase 10's job and stays
          there -- Phase 12 must not shadow it.
        """
        if self.spectral_conditions:
            return True
        if self.temporal_conditions and (
                len(self.conditions) >= 2 or self.summary_requests):
            return True
        return False

    @property
    def expression(self) -> str:
        joiner = " AND " if self.operator == "and" else " OR "
        return joiner.join(
            (f"NOT {c.name}" if c.negate else c.name) for c in self.conditions)


# =========================================================================== #
# parsing
# =========================================================================== #
def parse_threshold(text: str, index: str, mention_pos: int = 0) -> Tuple[Optional[str], Optional[float], Tuple[str, ...]]:
    """(operator, value, evidence) for a threshold near an index mention.

    Looks in a window after the mention (the usual phrasing, "NDVI > 0.6") and
    a small window before it ("greater than 0.6 NDVI"). Returns (None, None, ())
    when no number is attached -- which is the signal to ask, not to guess.
    """
    window = text[mention_pos: mention_pos + 60]
    for pattern, symbol in _OPERATORS:
        m = re.search(pattern, window, re.IGNORECASE)
        if not m:
            continue
        after = window[m.end(): m.end() + 24]
        n = _NUMBER.search(after)
        if n:
            return symbol, float(n.group(0)), (m.group(0).strip(), n.group(0))
    before = text[max(0, mention_pos - 40): mention_pos]
    for pattern, symbol in _OPERATORS:
        m = re.search(pattern, before, re.IGNORECASE)
        if not m:
            continue
        n = _NUMBER.search(before[m.end():])
        if n:
            return symbol, float(n.group(0)), (m.group(0).strip(), n.group(0))
    return None, None, ()


def _relative_request(text: str) -> Optional[str]:
    low = text.lower()
    for phrase in _RELATIVE_WORDS:
        if phrase in low:
            return phrase.replace("above the ", "").replace("below the ", "") \
                         .replace("above ", "").replace("below ", "")
    return None


def parse_composed_query(text: str,
                         config: Optional[Dict[str, Any]] = None,
                         enabled_convention: Optional[str] = None) -> ComposedQuery:
    """Extract every condition a composed query states.

    Spatial conditions come from Phase 9's own parser, so their meaning is
    identical to a pure Phase 9 query. Spectral and temporal conditions are
    recognised here. A spectral condition without a number is kept and flagged
    (`needs_threshold`) instead of being assigned a default.
    """
    cfg = config if config is not None else load_threshold_config()
    conventions = cfg.get("conventions") or {}
    low = text.lower()

    conditions: List[ComposedCondition] = []
    warnings: List[str] = []
    notes: List[str] = []

    # A two-date comparison is NOT a composition: return no conditions so the
    # router leaves the sentence to Phase 10 (NDVI) or to the TEMPORAL_NDWI
    # refusal. Without this, "Compare NDWI before and after" would be read as an
    # NDWI threshold request.
    has_comparison_cue = any(
        re.search(cue, low) for cue in _TEMPORAL_COMPARISON_CUES)
    has_change_word = any(
        re.search(rf"\b{re.escape(w)}\b", low) for w in _TEMPORAL_WORDS)
    if has_comparison_cue and not has_change_word:
        return ComposedQuery(original_query=text,
                             normalized_query=" ".join(low.split()),
                             status=ComposedQueryStatus.NO_CONDITIONS,
                             notes=["two-date comparison: not a composition"])

    # ---- 1. spatial conditions, straight from Phase 9 --------------------- #
    from core.spatial_query import parse_spatial_query     # local: see LAYERING

    spatial = parse_spatial_query(text)
    for condition in spatial.conditions:
        unsupported = getattr(condition, "status", None)
        if unsupported is not None and str(getattr(unsupported, "value", "")) == "unsupported":
            conditions.append(ComposedCondition(
                kind=ConditionKind.SPATIAL,
                name=getattr(condition, "condition_type", None).value
                if getattr(condition, "condition_type", None) else "unsupported",
                label=getattr(condition, "label", "") or "unsupported condition",
                source_analysis=str(getattr(condition, "required_analysis", "") or ""),
                parameters=dict(getattr(condition, "parameters", {}) or {}),
                spatial_condition=condition,
                evidence=tuple(getattr(condition, "evidence", ()) or ()),
                interpretation=getattr(condition, "interpretation", "") or "",
                limitations=(getattr(condition, "note", "") or "",),
            ))
            continue
        conditions.append(ComposedCondition(
            kind=ConditionKind.SPATIAL,
            name=(getattr(condition, "condition_type", None).value
                  if getattr(condition, "condition_type", None) else "spatial"),
            label=getattr(condition, "label", "") or "spatial condition",
            source_analysis=str(getattr(condition, "required_analysis", "") or "worldcover"),
            parameters=dict(getattr(condition, "parameters", {}) or {}),
            negate=bool(getattr(condition, "negate", False)),
            spatial_condition=condition,
            evidence=tuple(getattr(condition, "evidence", ()) or ()),
            interpretation=getattr(condition, "interpretation", "") or "",
        ))

    # ---- 2. spectral + temporal conditions ------------------------------- #
    summary_requests: List[str] = []
    for match in _INDEX_PATTERN.finditer(low):
        index = INDEX_ALIASES[match.group(0).lower()]
        start, end = match.span()

        after = low[end: end + 40]
        # "NDWI statistics" is a request for EVIDENCE, not a filter
        if re.match(r"\s*(statistics|stats|summary|mean|average|values)\b", after):
            summary_requests.append(index)
            continue

        temporal_word = next(
            (w for w in _TEMPORAL_WORDS
             if re.search(rf"\b{re.escape(w)}\b", after[:28])), None)
        if temporal_word:
            cls = _TEMPORAL_WORDS[temporal_word]
            conditions.append(ComposedCondition(
                kind=ConditionKind.TEMPORAL,
                name=f"ndvi_{cls}",
                label=(f"NDVI change class = {cls} "
                       f"(Phase 10 ±threshold, two dates)"),
                source_analysis="analyses.ndvi_change",
                parameters={"class": cls,
                            "class_code": TEMPORAL_CLASS_CODES[cls]},
                evidence=(match.group(0), temporal_word),
                interpretation=(f"Cells whose NDVI change class between the two "
                                f"selected dates is {cls}."),
                limitations=("A change class describes the index only; it does "
                             "not identify a cause.",),
            ))
            continue

        operator, value, evidence = parse_threshold(low, index, end)
        relative = _relative_request(low[max(0, start - 40): end + 60])
        # A bare "NDVI" with no number, no quality word and no relative rule is
        # a STATISTICS question ("what is the NDVI here?"), not a condition.
        near = low[max(0, start - 24): end + 24]
        if not (value is not None or relative
                or any(w in near.split() for w in _QUALITY_WORDS)):
            continue

        if value is None and enabled_convention:
            key = f"{index}_{'high' if 'high' in low[max(0, start - 12): end + 20] else 'low'}"
            convention = conventions.get(key)
            if convention:
                operator = str(convention.get("operator", ">"))
                value = float(convention["value"])
                conditions.append(ComposedCondition(
                    kind=ConditionKind.SPECTRAL,
                    name=f"{index}_{'gt' if operator.startswith('>') else 'lt'}",
                    label=f"{index.upper()} {operator} {value:g}  (convention: {key})",
                    source_analysis=f"core.indices:{index}",
                    threshold=ThresholdSpec(
                        index=index, operator=operator, value=value,
                        provenance=ThresholdProvenance.CONFIG_CONVENTION.value,
                        detail=f"convention:{key}",
                        note=str(convention.get("note", "")).strip()),
                    parameters={"convention": key},
                    evidence=(match.group(0), key),
                    interpretation=(f"{index.upper()} {operator} {value:g}, from the "
                                    f"enabled display/query convention '{key}'."),
                    limitations=("This threshold is a display/query convention, "
                                 "not a validated scientific classification.",),
                ))
                continue

        if value is None and relative:
            conditions.append(ComposedCondition(
                kind=ConditionKind.SPECTRAL,
                name=f"{index}_relative",
                label=f"{index.upper()} relative to the selected area ({relative})",
                source_analysis=f"core.indices:{index}",
                threshold=ThresholdSpec(
                    index=index, operator=">", value=None,
                    provenance=ThresholdProvenance.RELATIVE.value,
                    detail=f"relative:{relative}",
                    note=("Computed from the selected area only; relative to "
                          "that area, never an absolute land-cover rule.")),
                parameters={"relative": relative},
                evidence=(match.group(0), relative),
                interpretation=(f"{index.upper()} above/below the {relative} of "
                                f"the SELECTED AREA."),
                limitations=("Relative to the selected area only; it says "
                             "nothing about any absolute class.",),
            ))
            continue

        # No number, no convention, no relative request: flag it, do not invent.
        conditions.append(ComposedCondition(
            kind=ConditionKind.SPECTRAL,
            name=f"{index}_{'gt' if (operator or '>').startswith('>') else 'lt'}"
                 if value is not None else f"{index}_threshold",
            label=(f"{index.upper()} {operator} {value:g}"
                   if value is not None
                   else f"{index.upper()} threshold (not specified)"),
            source_analysis=f"core.indices:{index}",
            threshold=ThresholdSpec(
                index=index, operator=operator or ">", value=value,
                provenance=ThresholdProvenance.USER_SPECIFIED.value,
                detail=("your query" if value is not None else "missing")),
            evidence=(match.group(0),) + tuple(evidence),
            interpretation=(f"{index.upper()} was asked for without a threshold."),
            limitations=("No threshold was supplied and none will be invented.",),
        ))

    # ---- 2b. temporal phrasings that name no index ------------------------- #
    claimed = [m.span() for m in _INDEX_PATTERN.finditer(low)]
    for pattern, cls in _TEMPORAL_PHRASES:
        for m in re.finditer(pattern, low):
            if any(s <= m.start() < e for s, e in claimed):
                continue                    # already handled as index + temporal
            conditions.append(ComposedCondition(
                kind=ConditionKind.TEMPORAL,
                name=f"ndvi_{cls}",
                label=f"NDVI change class = {cls} (Phase 10 ±threshold, two dates)",
                source_analysis="analyses.ndvi_change",
                parameters={"class": cls, "class_code": TEMPORAL_CLASS_CODES[cls]},
                evidence=(m.group(0),),
                interpretation=(f"Cells whose NDVI change class between the two "
                                f"selected dates is {cls}."),
                limitations=("A change class describes the index only; it does "
                             "not identify a cause.",),
            ))
            claimed.append(m.span())

    # ---- 3. status --------------------------------------------------------- #
    if not conditions and not summary_requests:
        status = ComposedQueryStatus.NO_CONDITIONS
    elif any(c.needs_threshold for c in conditions):
        status = ComposedQueryStatus.NEEDS_THRESHOLD
        for c in conditions:
            if c.needs_threshold:
                warnings.append(
                    f"'{c.name.split('_')[0].upper()}' was asked for without a "
                    f"threshold. Ask for e.g. "
                    f"{c.name.split('_')[0].upper()} greater than 0.6, or enable "
                    f"a labelled convention in the UI.")
    else:
        status = ComposedQueryStatus.OK

    unsupported = [c for c in conditions
                   if c.kind is ConditionKind.SPATIAL and c.name == "unsupported"]
    if unsupported:
        status = ComposedQueryStatus.NEEDS_CLARIFICATION
        notes.extend(c.interpretation or c.label for c in unsupported)

    return ComposedQuery(
        original_query=text,
        normalized_query=" ".join(low.split()),
        conditions=tuple(conditions),
        operator="or" if re.search(r"\bor\b|either", low) else "and",
        status=status,
        summary_requests=tuple(dict.fromkeys(summary_requests)),
        warnings=warnings,
        notes=notes,
    )


# =========================================================================== #
# masks
# =========================================================================== #
def threshold_mask(values: Any,
                   valid: Any,
                   operator: str,
                   threshold: float,
                   grid: Grid,
                   *,
                   name: str,
                   source: str,
                   **provenance: Any) -> GridMask:
    """A continuous index raster -> three-valued mask on one grid.

    Invalid pixels (NaN, nodata, guarded denominator) stay INSUFFICIENT: they
    are unknown, never "not matching".
    """
    values = np.asarray(values, dtype="float32")
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError(
            f"threshold mask '{name}': values {values.shape} vs valid {valid.shape}")
    if tuple(values.shape) != tuple(grid.shape):
        raise ValueError(
            f"threshold mask '{name}': array {values.shape} is not on the "
            f"grid {grid.shape}")

    ops = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }
    if operator not in ops:
        raise ValueError(f"unsupported operator {operator!r}")

    usable = valid & np.isfinite(values)
    match = np.zeros(values.shape, dtype=bool)
    match[usable] = ops[operator](values[usable], float(threshold))
    return GridMask(
        grid=grid, name=name, source=source,
        match=match, valid=usable,
        provenance={"operator": operator, "threshold": float(threshold), **provenance},
    )


def temporal_class_mask(class_raster: Any,
                        grid: Grid,
                        *,
                        wanted: Sequence[str],
                        name: str,
                        source: str = "analyses.ndvi_change",
                        **provenance: Any) -> GridMask:
    """Phase 10 change classes -> three-valued mask.

    Class 0 (insufficient) is UNKNOWN, never FALSE: a cell that could not be
    compared did not "not decrease".
    """
    codes = np.asarray(class_raster)
    if tuple(codes.shape) != tuple(grid.shape):
        raise ValueError(
            f"temporal mask '{name}': class raster {codes.shape} is not on the "
            f"grid {grid.shape}")
    wanted_codes = [TEMPORAL_CLASS_CODES[w] for w in wanted]
    valid = np.isin(codes, list(TEMPORAL_CODE_NAMES)) & (codes != TEMPORAL_CLASS_CODES["insufficient"])
    match = valid & np.isin(codes, wanted_codes)
    return GridMask(
        grid=grid, name=name, source=source,
        match=match, valid=valid,
        provenance={"wanted_classes": list(wanted),
                    "wanted_codes": wanted_codes, **provenance},
    )


def combine_conditions(masks: Sequence[GridMask], operator: str = "and") -> GridMask:
    """AND / OR over GridMasks, three-valued. Unknown is never silently dropped."""
    from core.spatial import Operator

    op = Operator.OR if str(operator).lower().startswith("or") else Operator.AND
    return combine_all(list(masks), op)


def negated(mask: GridMask) -> GridMask:
    """NOT with three-valued semantics (NA stays NA)."""
    return negate(mask)


def state_counts(mask: GridMask) -> Dict[str, int]:
    from core.spatial import FALSE, INSUFFICIENT, TRUE

    state = np.asarray(mask.state)
    return {
        "matched": int(np.count_nonzero(state == TRUE)),
        "non_matching": int(np.count_nonzero(state == FALSE)),
        "insufficient": int(np.count_nonzero(state == INSUFFICIENT)),
        "total": int(state.size),
    }


def summarise_index(values: Any, valid: Any, inside: Any = None) -> Dict[str, Any]:
    """Statistics of one index over a set of cells (attached evidence)."""
    values = np.asarray(values, dtype="float32")
    mask = np.asarray(valid, dtype=bool)
    if inside is not None:
        mask = mask & np.asarray(inside, dtype=bool)
    mask = mask & np.isfinite(values)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return {"valid_pixels": 0, "defined": False}
    v = values[mask]
    return {
        "valid_pixels": n,
        "defined": True,
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "std": float(v.std()),
    }


__all__ = [
    "MULTI_CONFIG_DIR",
    "ConditionKind",
    "ComposedQueryStatus",
    "ThresholdProvenance",
    "ThresholdSpec",
    "ComposedCondition",
    "ComposedQuery",
    "TEMPORAL_CLASS_CODES",
    "TEMPORAL_CODE_NAMES",
    "INDEX_ALIASES",
    "load_threshold_config",
    "parse_threshold",
    "parse_composed_query",
    "threshold_mask",
    "temporal_class_mask",
    "combine_conditions",
    "negated",
    "state_counts",
    "summarise_index",
]
