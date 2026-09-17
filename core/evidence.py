"""Phase 13 -- the evidence contract.

This module defines HOW A RESULT IS DESCRIBED, never what it means. It holds
facts: what was measured, from which source, on which grid, with which
threshold, and how many cells ended up in each of the three states.

Layering rule (the same one `core/multi_condition.py` obeys): nothing here may
import from `analyses.*`. The builders that read engine results live in
`analyses/evidence.py`.

Three rules shape every structure below:

1. **Facts, not interpretations.** A record never stores a conclusion, and it
   never stores an array -- the rasters stay in the result object that produced
   them. That is what keeps a package small enough to export and honest enough
   to audit.
2. **Nothing is invented.** A fact that is not available is the string
   `"unavailable"`, never a plausible substitute.
3. **Three-valued, always.** `insufficient` is its own counter beside `matched`
   and `non_matching`, so an undecided cell can never be reported as a
   non-match further down the line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Version of the JSON shape. Bumped only when a field changes meaning.
EVIDENCE_SCHEMA = "satquery-evidence/1"

#: Stored wherever a fact genuinely does not exist. Never a guess, never a zero.
UNAVAILABLE = "unavailable"

#: The three-valued states, restated here so an evidence record is readable on
#: its own. The values are Phase 9's (2/1/0) and are not redefined.
STATE_MATCH = 2
STATE_NO_MATCH = 1
STATE_UNKNOWN = 0

#: How much of the answer could not be decided.
UNKNOWN_NONE = "none_unknown"            # every measured cell was decided
UNKNOWN_PARTIAL = "partial_unknown"      # some cells undecided, others decided
UNKNOWN_ALL = "all_unknown"              # nothing could be evaluated
UNKNOWN_NOT_APPLICABLE = "not_applicable"  # the result carries no per-cell states


@dataclass(frozen=True)
class EvidenceRecord:
    """One measured thing: a condition, a combined mask or a statistic.

    `id` is deterministic (`"{kind}:{name}"`, de-duplicated with a counter) so
    two runs of the same question produce byte-identical packages.
    """

    id: str
    kind: str                       # spatial | spectral | temporal | combined | statistics
    label: str                      # as measured, never reworded
    source_analysis: str = UNAVAILABLE
    source_dataset: str = UNAVAILABLE
    source_dates: Tuple[str, ...] = ()
    band_or_index: str = ""
    condition: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    operator: Optional[str] = None
    threshold: Optional[float] = None
    threshold_provenance: Optional[Dict[str, Any]] = None
    negated: bool = False
    grid: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    area_m2: Optional[float] = None
    fraction: Optional[float] = None
    runtime_ms: Optional[float] = None
    limitations: Tuple[str, ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)

    # -- derived, but still facts ------------------------------------------ #
    @property
    def matched(self) -> int:
        return int(self.counts.get("matched", 0))

    @property
    def non_matching(self) -> int:
        return int(self.counts.get("non_matching", 0))

    @property
    def unknown(self) -> int:
        return int(self.counts.get("insufficient", 0))

    @property
    def total(self) -> int:
        return int(self.counts.get("total", 0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "source_analysis": self.source_analysis,
            "source_dataset": self.source_dataset,
            "source_dates": list(self.source_dates),
            "band_or_index": self.band_or_index,
            "condition": self.condition,
            "parameters": dict(self.parameters),
            "operator": self.operator,
            "threshold": self.threshold,
            "threshold_provenance": dict(self.threshold_provenance or {}),
            "negated": self.negated,
            "grid": dict(self.grid),
            "counts": {
                "matched": self.matched,
                "non_matching": self.non_matching,
                "insufficient": self.unknown,
                "total": self.total,
            },
            "area_m2": self.area_m2,
            "fraction": self.fraction,
            "runtime_ms": self.runtime_ms,
            "limitations": list(self.limitations),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class Lineage:
    """The chain that produced an answer, link by link.

    Every field is a value, not a description: the lineage can be printed as a
    list or asserted against in a test.
    """

    query: str = ""
    normalized_query: str = ""
    intent: str = ""
    conditions: Tuple[str, ...] = ()
    sources: Tuple[str, ...] = ()
    grid: Dict[str, Any] = field(default_factory=dict)
    masks: Tuple[str, ...] = ()
    combined: str = ""
    statistics: Tuple[str, ...] = ()
    answer: str = ""

    def steps(self) -> List[Tuple[str, str]]:
        """(label, value) pairs, in the order the answer was derived."""
        grid = self.grid or {}
        grid_text = (
            f"{grid.get('width', UNAVAILABLE)} x {grid.get('height', UNAVAILABLE)}"
            f" cells at {grid.get('resolution_m', UNAVAILABLE)} m"
            f" ({grid.get('crs', UNAVAILABLE)})" if grid else UNAVAILABLE)
        return [
            ("User query", self.query or UNAVAILABLE),
            ("Normalized query", self.normalized_query or UNAVAILABLE),
            ("Intent", self.intent or UNAVAILABLE),
            ("Conditions", " AND ".join(self.conditions) if self.conditions
             else UNAVAILABLE),
            ("Source analyses", ", ".join(self.sources) if self.sources
             else UNAVAILABLE),
            ("Grid / ROI", grid_text),
            ("Condition masks", ", ".join(self.masks) if self.masks
             else UNAVAILABLE),
            ("Combined mask", self.combined or UNAVAILABLE),
            ("Statistics", ", ".join(self.statistics) if self.statistics
             else UNAVAILABLE),
            ("Final answer", self.answer or UNAVAILABLE),
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "conditions": list(self.conditions),
            "sources": list(self.sources),
            "grid": dict(self.grid),
            "masks": list(self.masks),
            "combined": self.combined,
            "statistics": list(self.statistics),
            "answer": self.answer,
        }


@dataclass(frozen=True)
class EvidencePackage:
    """The auditable package for one answer.

    It contains facts and the deterministic explanation blocks derived from
    them. It contains no generated conclusion beyond that explanation.
    """

    query: str = ""
    normalized_query: str = ""
    intent: str = ""
    status: str = ""
    expression: str = ""
    lineage: Lineage = field(default_factory=Lineage)
    records: Tuple[EvidenceRecord, ...] = ()
    combined: Optional[EvidenceRecord] = None
    statistics: Tuple[EvidenceRecord, ...] = ()
    sources: Tuple[Dict[str, Any], ...] = ()
    grid: Dict[str, Any] = field(default_factory=dict)
    alignment: Dict[str, Any] = field(default_factory=dict)
    threshold_provenance: Dict[str, Any] = field(default_factory=dict)
    analysed_cells: int = 0
    matched_cells: int = 0
    non_matching_cells: int = 0
    unknown_cells: int = 0
    matched_area_m2: float = 0.0
    unknown_handling: str = UNKNOWN_NOT_APPLICABLE
    limitations: Tuple[str, ...] = ()
    boundary: str = ""
    runtime_ms: Optional[float] = None
    explanation: Dict[str, Any] = field(default_factory=dict)

    # -- the three questions a reader actually asks ------------------------- #
    @property
    def all_unknown(self) -> bool:
        """True when nothing could be evaluated -- NOT the same as no matches."""
        return self.unknown_handling == UNKNOWN_ALL

    @property
    def has_matches(self) -> bool:
        return self.matched_cells > 0

    @property
    def matched_area_km2(self) -> float:
        return self.matched_area_m2 / 1.0e6

    def to_dict(self) -> Dict[str, Any]:
        """The machine-readable package. JSON-safe by construction."""
        return {
            "schema": EVIDENCE_SCHEMA,
            "query": self.query,
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "status": self.status,
            "expression": self.expression,
            "result": {
                "status": self.status,
                "matched_cells": self.matched_cells,
                "non_matching_cells": self.non_matching_cells,
                "unknown_cells": self.unknown_cells,
                "analysed_cells": self.analysed_cells,
                "matched_area_m2": self.matched_area_m2,
                "unknown_handling": self.unknown_handling,
            },
            "lineage": self.lineage.to_dict(),
            "conditions": [r.to_dict() for r in self.records],
            "combined": self.combined.to_dict() if self.combined else None,
            "statistics": [r.to_dict() for r in self.statistics],
            "sources": [dict(s) for s in self.sources],
            "grid": dict(self.grid),
            "alignment": dict(self.alignment),
            "threshold_provenance": dict(self.threshold_provenance),
            "limitations": list(self.limitations),
            "boundary": self.boundary,
            "runtime_ms": self.runtime_ms,
            "explanation": dict(self.explanation),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Deterministic JSON: sorted keys, no arrays, no engine objects."""
        return json.dumps(json_safe(self.to_dict()), indent=indent,
                          sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #
def json_safe(value: Any) -> Any:
    """Recursively convert `value` into something `json.dumps` accepts.

    Results carry numpy scalars, tuples, CRS objects, datetimes and (in
    `MultiConditionResult.to_dict()`) live mask objects. None of those belong in
    an export, so scalars are stringified and anything unrecognised becomes
    `"unavailable"` rather than crashing the download.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return UNAVAILABLE
    module = type(value).__module__ or ""
    if module.startswith("numpy"):
        # A scalar is a fact; an array is data. Arrays stay in the result that
        # owns them -- an export is an audit trail, not a raster copy.
        if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
            try:
                return json_safe(value.item())
            except Exception:
                return UNAVAILABLE
        return UNAVAILABLE
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return json_safe(value.to_dict())
        except Exception:
            return UNAVAILABLE
    if hasattr(value, "isoformat") and not hasattr(value, "year"):
        try:
            return str(value)
        except Exception:
            return UNAVAILABLE
    if hasattr(value, "year"):      # date / datetime
        return value.isoformat()
    # Anything else (functions, live engine objects, file handles) is not a
    # fact the reader can audit, so it is reported as missing -- never printed
    # as a repr that looks like data.
    return UNAVAILABLE


# --------------------------------------------------------------------------- #
# helpers used by the builders
# --------------------------------------------------------------------------- #
def counts_of(matched: Any = 0, non_matching: Any = 0,
              insufficient: Any = 0, total: Any = 0) -> Dict[str, int]:
    """One normalised count block, so every record counts the same way."""
    m, n, u = int(matched or 0), int(non_matching or 0), int(insufficient or 0)
    t = int(total or 0) or (m + n + u)
    return {"matched": m, "non_matching": n, "insufficient": u, "total": t}


def unknown_handling_for(counts: Dict[str, int]) -> str:
    """Classify how much of the answer is undecided.

    `all_unknown` is returned ONLY when nothing was decided -- that is the
    distinction that stops "we could not measure this" being printed as
    "nothing matched".
    """
    matched = int(counts.get("matched", 0))
    non_matching = int(counts.get("non_matching", 0))
    unknown = int(counts.get("insufficient", 0))
    total = int(counts.get("total", 0)) or (matched + non_matching + unknown)
    if total <= 0:
        return UNKNOWN_NOT_APPLICABLE
    if matched == 0 and non_matching == 0 and unknown > 0:
        return UNKNOWN_ALL
    if unknown > 0:
        return UNKNOWN_PARTIAL
    return UNKNOWN_NONE


__all__ = [
    "EVIDENCE_SCHEMA",
    "UNAVAILABLE",
    "STATE_MATCH",
    "STATE_NO_MATCH",
    "STATE_UNKNOWN",
    "UNKNOWN_ALL",
    "UNKNOWN_NONE",
    "UNKNOWN_NOT_APPLICABLE",
    "UNKNOWN_PARTIAL",
    "EvidencePackage",
    "EvidenceRecord",
    "Lineage",
    "counts_of",
    "json_safe",
    "unknown_handling_for",
]
