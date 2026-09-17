"""Phase 7 -- the contract between the router, the engines and the UI.

Three ideas live here:

    AnalysisContext    what the app currently has (ROI, NDVI, gate state)
    AnalysisExecution  what came back (structured, never a string scraped
                       from a widget)
    AnalysisSpec       what a registered intent is allowed to do

Engine modules (analyses/ndvi.py and its future siblings) import this and
nothing from `app.py`; they never touch `st.session_state`. That is what makes
them testable without a browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.router import Intent, QueryIntent

__all__ = [
    "Status",
    "NdviContext",
    "AnalysisContext",
    "AnalysisExecution",
    "AnalysisSpec",
]


class Status(str, Enum):
    """Why an execution ended the way it did.

    Deliberately separate from `Intent`: the intent is *what the user asked*,
    the status is *whether we could answer it*.
    """

    OK = "OK"
    NEEDS_ROI = "NEEDS_ROI"
    NEEDS_NDVI_CONFIRMATION = "NEEDS_NDVI_CONFIRMATION"
    NO_VALID_PIXELS = "NO_VALID_PIXELS"
    # -- Phase 10: temporal comparison --------------------------------------- #
    # NEEDS_TWO_DATES      = a before/after pair is missing, undated, or not
    #                        two DIFFERENT dates. Never resolved by guessing a date.
    # NO_TEMPORAL_OVERLAP  = the two scenes share no ground (or the ROI is not
    #                        inside both), so there is nothing to difference.
    NEEDS_TWO_DATES = "NEEDS_TWO_DATES"
    NO_TEMPORAL_OVERLAP = "NO_TEMPORAL_OVERLAP"
    UNSUPPORTED = "UNSUPPORTED"        # intent exists, engine does not (yet)
    UNKNOWN = "UNKNOWN"                # parser could not resolve the request
    ERROR = "ERROR"
    # -- Phase 12: a composed query named an index but gave no threshold ----- #
    # Distinct from UNSUPPORTED (the analysis does not exist) and from
    # NEEDS_NDVI_CONFIRMATION (a data-trust gate): here the analysis exists and
    # runs the moment the user states a number. The message says which index
    # needs one and shows a phrasing that works.
    NEEDS_THRESHOLD = "NEEDS_THRESHOLD"
    # -- Phase 8: data completeness of an analysis that DID run ------------- #
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"   # a critical input is missing: no score
    PARTIAL_DATA = "PARTIAL_DATA"             # scored without optional inputs
    UNSUPPORTED_CROP = "UNSUPPORTED_CROP"     # understood crop, no configuration
    # Phase 9: the user asked for something the engines do not measure
    # (irrigation, groundwater, salinity). Nothing was computed, on purpose.
    UNSUPPORTED_CONDITION = "UNSUPPORTED_CONDITION"


#: Messages for the context failures. Kept here so tests can assert on them and
#: the UI never invents its own wording.
CONTEXT_MESSAGES: Dict[str, str] = {
    "roi": "Please select an area on the map first.",
    "ndvi_confirmed": ("Please confirm the detected satellite bands before running "
                       "NDVI analysis."),
}


@dataclass
class NdviContext:
    """The native NDVI the ROI analysis is allowed to measure.

    Mirrors `AnalysisResult`: NATIVE array, NATIVE mask, NATIVE georeferencing.
    The reprojected web-mercator copy is display-only and is never passed here.
    """

    array: Any
    mask: Any
    crs: Any
    transform: Any
    bands: Dict[str, Any] = field(default_factory=dict)
    source_label: str = ""


@dataclass
class IndexContext:
    """Phase 11: what a generic index analysis needs in order to READ data.

    Deliberately NOT a copy of `NdviContext`. NDVI's context carries a ready-
    made array because Phase 3 computes the whole scene up front; an index
    analysis is ROI-first, so it needs the SOURCE plus resolved band ROLES and
    reads only the window the ROI covers.

    `roles` maps a role name ("green", "nir") to a 1-based band index. It is
    filled only from evidence-backed band metadata -- never guessed, never
    defaulted by position.
    """

    path: str = ""
    roles: Dict[str, int] = field(default_factory=dict)
    scale: float = 1.0
    offset: float = 0.0
    is_reflectance: bool = True
    profile: Optional[str] = None
    source_label: str = ""
    reflectance_source: str = "detected"
    role_confidence: str = "none"
    role_evidence: Tuple[str, ...] = ()

    def band(self, role: str) -> Optional[int]:
        return self.roles.get(role)

    def has_roles(self, *roles: str) -> bool:
        return all(self.roles.get(r) for r in roles)


@dataclass
class AnalysisContext:
    """Everything an analysis may use. Built by `app.py`, consumed by engines."""

    roi: Optional[Any] = None                 # core.roi.ROISelection | None
    ndvi: Optional[NdviContext] = None
    ndvi_confirmed: bool = False
    raster_label: str = ""
    # Phase 10: the two acquisitions of a temporal comparison
    # (core.temporal.ScenePair). Optional, so every Phase 1-9 caller is unchanged.
    temporal_pair: Optional[Any] = None
    # Phase 11: the source + resolved band roles for a generic index analysis
    # (IndexContext). Optional, so every Phase 1-10 caller is unchanged.
    index_context: Optional[Any] = None

    @property
    def has_index_context(self) -> bool:
        ctx = self.index_context
        return ctx is not None and bool(getattr(ctx, "path", "")) and bool(
            getattr(ctx, "roles", None)
        )

    @property
    def has_temporal_pair(self) -> bool:
        return self.temporal_pair is not None and bool(
            getattr(self.temporal_pair, "complete", False)
        )

    @property
    def has_roi(self) -> bool:
        return self.roi is not None and bool(getattr(self.roi, "usable", False))

    @property
    def has_ndvi(self) -> bool:
        return self.ndvi is not None and self.ndvi_confirmed


@dataclass
class AnalysisExecution:
    """The structured answer the UI renders.

    `result` is engine-specific (for NDVI: `ROINDVIStats`); everything else has
    the same shape for every analysis, now and in the future.
    """

    intent: Intent
    status: Status
    query: str
    normalized_query: str = ""
    confidence: float = 0.0
    explanation: str = ""
    matched: Tuple[str, ...] = ()
    result: Any = None
    message: str = ""
    warnings: Tuple[str, ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is Status.OK and self.result is not None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "intent": self.intent.value,
            "status": self.status.value,
            "query": self.query,
            "normalized_query": self.normalized_query,
            "confidence": round(float(self.confidence), 3),
            "explanation": self.explanation,
            "matched": list(self.matched),
            "message": self.message,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "result": self.result.to_dict() if hasattr(self.result, "to_dict") else None,
        }
        return out


@dataclass
class AnalysisSpec:
    """One row of the registry.

    `handler=None` means "we understand this question, we cannot answer it yet".
    That explicit hole is what stops the system from hallucinating an answer,
    and it is exactly where a future engine is plugged in.
    """

    intent: Intent
    title: str
    description: str
    requires: Tuple[str, ...] = ()
    handler: Optional[Callable[["AnalysisContext", QueryIntent], AnalysisExecution]] = None
    unavailable_message: str = ""
    example_queries: Tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.handler is not None
