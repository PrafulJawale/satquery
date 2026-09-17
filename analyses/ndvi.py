"""Phase 7 -- NDVI adapter: the bridge from an intent to the Phase 6 engine.

This module contains NO mathematics. It knows how to:

    * check that the context really has what NDVI needs;
    * call `core.statistics.calculate_roi_ndvi_stats` with the NATIVE array;
    * turn the result into an `AnalysisExecution` the UI can render.

Every number it reports was computed by Phase 6. If Phase 6 changes, this
adapter does not need to.
"""

from __future__ import annotations

from typing import Any, Dict

from core.router import Intent, QueryIntent
from core.statistics import (
    NO_PIXELS_MESSAGE,
    NO_VALID_MESSAGE,
    calculate_roi_ndvi_stats,
)

from .base import AnalysisContext, AnalysisExecution, Status

__all__ = ["run_ndvi_roi_stats", "NDVI_ANSWER_TEMPLATE", "NDVI_CAVEAT"]

#: Wording rule for the whole prototype: NDVI is a *vegetation index* value.
#: It is never called crop health, yield, disease or suitability.
NDVI_ANSWER_TEMPLATE = (
    "Selected area has a mean NDVI of **{mean:.4f}** over {n:,} valid pixels "
    "({pct:.2f}% of the pixels inside the selection)."
)
NDVI_CAVEAT = (
    "This is a measurement of the selected pixels only — an observed NDVI "
    "value, not a crop-health, yield or disease diagnosis."
)


def run_ndvi_roi_stats(context: AnalysisContext, query: QueryIntent) -> AnalysisExecution:
    """Execute NDVI_ROI_STATS by delegating to the Phase 6 analysis engine."""
    roi = context.roi
    ndvi = context.ndvi

    # --- context validation (never silently substitute another analysis) --- #
    if roi is None or not getattr(roi, "usable", False):
        return AnalysisExecution(
            intent=Intent.NDVI_ROI_STATS, status=Status.NEEDS_ROI, query=query.original_query,
            normalized_query=query.normalized_query, confidence=query.confidence,
            explanation=query.explanation, matched=query.matched,
            message="Please select an area on the map first.",
        )
    if ndvi is None or not context.ndvi_confirmed:
        return AnalysisExecution(
            intent=Intent.NDVI_ROI_STATS, status=Status.NEEDS_NDVI_CONFIRMATION,
            query=query.original_query, normalized_query=query.normalized_query,
            confidence=query.confidence, explanation=query.explanation,
            matched=query.matched,
            message=("Please confirm the detected satellite bands before running "
                     "NDVI analysis."),
        )

    # --- the analysis: Phase 6 does the mathematics ------------------------ #
    stats = calculate_roi_ndvi_stats(
        ndvi.array,
        roi.geometry_raster_crs,
        ndvi.transform,
        ndvi.mask,
        crs=ndvi.crs,
        roi_crs=roi.raster_crs,
    )

    provenance: Dict[str, Any] = {
        "engine": "core.statistics.calculate_roi_ndvi_stats (Phase 6)",
        "native_resolution_m": [stats.pixel_width, stats.pixel_height],
        "crs": stats.crs,
        "raster_window": list(stats.window) if stats.window else None,
        "bands": dict(ndvi.bands or {}),
        "source": ndvi.source_label,
        "router": {
            "intent": Intent.NDVI_ROI_STATS.value,
            "confidence": round(float(query.confidence), 3),
            "matched": list(query.matched),
            "explanation": query.explanation,
        },
        "pixels_inside_roi": stats.pixels_inside_roi,
        "valid_pixels": stats.valid_pixels,
    }

    # --- honest reporting of the empty cases ------------------------------- #
    if stats.valid_pixels == 0:
        return AnalysisExecution(
            intent=Intent.NDVI_ROI_STATS, status=Status.NO_VALID_PIXELS,
            query=query.original_query, normalized_query=query.normalized_query,
            confidence=query.confidence, explanation=query.explanation,
            matched=query.matched, result=stats,
            message=(NO_PIXELS_MESSAGE if stats.pixels_inside_roi == 0 else NO_VALID_MESSAGE),
            warnings=stats.warnings, provenance=provenance,
        )

    return AnalysisExecution(
        intent=Intent.NDVI_ROI_STATS, status=Status.OK,
        query=query.original_query, normalized_query=query.normalized_query,
        confidence=query.confidence, explanation=query.explanation,
        matched=query.matched, result=stats,
        message=NDVI_ANSWER_TEMPLATE.format(
            mean=float(stats.stats["mean"]),
            n=int(stats.valid_pixels),
            pct=100.0 * float(stats.valid_fraction),
        ),
        warnings=tuple(stats.warnings) + (NDVI_CAVEAT,),
        provenance=provenance,
    )
