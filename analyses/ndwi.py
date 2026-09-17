"""Phase 11 -- the NDWI ROI analysis: an index, not a flood map.

WHAT THIS ENGINE IS
-------------------
An adapter, in the same shape as `analyses/ndvi.py`: it validates the context,
asks `core.indices` to compute the index on the ROI window, and turns the
numbers into an `AnalysisExecution`. It contains no arithmetic of its own.

NDWI = (GREEN - NIR) / (GREEN + NIR)                        [McFeeters 1996]

with the same denominator guard the NDVI engine has used since Phase 3
(|GREEN + NIR| < 1e-6 is UNDEFINED, not zero). The formula, the band roles, the
guard, the range, the colormap and the wording all come from
config/indices/ndwi.yml -- none of them is written in Python.

WHAT THIS ENGINE IS NOT
-----------------------
* **Not flood detection.** NDWI is a spectral index. Water-like values can come
  from built-up surfaces, shade, cloud or shadow, and a change in NDWI between
  two dates is not a flood. The result message and the map legend both say so.
* **Not water extraction.** There is no "water if NDWI > X" rule anywhere in
  Phase 11. Turning a continuous index into a water mask is a separate scientific
  claim that needs its own phase, its own justification and its own tests.
* **Not a temporal analysis.** "Compare NDWI before and after" is routed to a
  planned, handler-less intent and returns UNSUPPORTED.

ROI-FIRST
---------
Only the pixel window covering the ROI is read -- never the whole 2048x2048
scene -- via `core.alignment.native_roi_grid`. A selection bigger than the
2,000,000-cell budget is REFUSED with an explanation, never silently coarsened.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from core.alignment import DEFAULT_MAX_CELLS, native_roi_grid, roi_mask
from core.index_definitions import get_index
from core.indices import index_from_dataset
from core.raster import open_dataset
from core.reflectance import ReflectanceSpec
from core.router import Intent, QueryIntent
from core.statistics import (
    NO_PIXELS_MESSAGE,
    NO_VALID_MESSAGE,
    ROINDVIStats,
    calculate_roi_ndvi_stats,
)

from .base import AnalysisContext, AnalysisExecution, Status

INDEX_NAME = "ndwi"

#: Wording rule: NDWI is a spectral index, never a statement about water bodies.
NDWI_ANSWER_TEMPLATE = (
    "Selected area has a mean NDWI of **{mean:.4f}** over {n:,} valid pixels "
    "({pct:.2f}% of the pixels inside the selection)."
)

#: The caveat travels with every result and is also printed on the legend.
NDWI_CAVEAT = (
    "NDWI is a spectral index. This result does not by itself establish flood "
    "extent, water availability, or water quality."
)

#: Shown when the ROI is larger than the cell budget -- a refusal, not a rescale.
OVERSIZED_MESSAGE = (
    "The selected area is too large to analyse at this resolution "
    "({cells:,} cells, budget {budget:,}). Draw a smaller area: coarsening it "
    "silently would change the question being asked."
)


def _execution(query: QueryIntent, status: Status, message: str, *,
               warnings: Any = (), result: Any = None,
               provenance: Optional[Dict[str, Any]] = None) -> AnalysisExecution:
    return AnalysisExecution(
        intent=query.intent, status=status, query=query.original_query,
        normalized_query=query.normalized_query, confidence=query.confidence,
        explanation=query.explanation, matched=query.matched, message=message,
        result=result, warnings=tuple(warnings),
        provenance=provenance if provenance is not None else {"engine": __name__},
    )


def missing_roles_message(definition: Any, available: Any) -> str:
    """Why an index cannot be computed, naming the roles that are absent."""
    have = ", ".join(sorted(available)) or "none"
    return (
        f"{definition.short_name} needs the {', '.join(definition.roles)} band "
        f"roles and only {have} could be established for this raster. The band "
        f"roles are resolved from band metadata and are never guessed, so no "
        f"{definition.short_name} value is reported."
    )


def run_ndwi_roi_stats(context: AnalysisContext, query: QueryIntent) -> AnalysisExecution:
    """Execute NDWI_ROI_STATS: NDWI statistics for the selected area."""
    definition = get_index(INDEX_NAME)
    started = time.perf_counter()

    # --- 1. an area must be selected --------------------------------------- #
    roi = getattr(context, "roi", None)
    if roi is None or not getattr(roi, "usable", False):
        return _execution(query, Status.NEEDS_ROI,
                          "Please select an area on the map first.")

    # --- 2. the band roles must be KNOWN, not guessed ----------------------- #
    ctx = getattr(context, "index_context", None)
    if ctx is None or not getattr(ctx, "path", ""):
        return _execution(query, Status.UNSUPPORTED,
                          "No raster with resolvable band roles is available for "
                          "an index analysis.")
    if not ctx.has_roles(*definition.roles):
        return _execution(query, Status.UNSUPPORTED,
                          missing_roles_message(definition, ctx.roles.keys()))

    geometry = getattr(roi, "geometry_raster_crs", None) or getattr(roi, "geometry", None)
    if geometry is None:
        return _execution(query, Status.NEEDS_ROI,
                          "The selected area has no usable geometry. Draw it again.")

    roi_crs = getattr(roi, "raster_crs", None)

    # --- 3. read ONLY the ROI window, compute, and clip to the ROI ---------- #
    try:
        with open_dataset(ctx.path) as ds:
            width, height = ds.width, ds.height
            res = float(ds.res[0])
            transform = ds.transform
            crs = ds.crs

            grid = native_roi_grid(transform, width, height, res, crs, geometry)
            if grid.cells > DEFAULT_MAX_CELLS:
                # A refusal still carries the numbers it reasoned about, so the
                # UI can say HOW big the request was instead of just "too big".
                return _execution(
                    query, Status.INSUFFICIENT_DATA,
                    OVERSIZED_MESSAGE.format(cells=grid.cells, budget=DEFAULT_MAX_CELLS),
                    provenance={
                        "engine": __name__,
                        "index": definition.to_dict(),
                        "roi_cells": int(grid.cells),
                        "cell_budget": int(DEFAULT_MAX_CELLS),
                        "refused": "roi_exceeds_cell_budget",
                    })

            win = rasterio.windows.from_bounds(*grid.bounds, transform=transform)
            col_off = max(0, int(round(win.col_off)))
            row_off = max(0, int(round(win.row_off)))
            w = max(1, min(int(round(win.width)), width - col_off))
            h = max(1, min(int(round(win.height)), height - row_off))
            window = Window(col_off, row_off, w, h)

            spec = ReflectanceSpec(
                scale=float(ctx.scale), offset=float(ctx.offset),
                profile=ctx.profile, source=ctx.reflectance_source,
                is_reflectance=bool(ctx.is_reflectance),
            )
            result = index_from_dataset(
                ds, definition,
                {role: int(ctx.roles[role]) for role in definition.roles},
                window=window, reflectance=spec, profile=ctx.profile,
                provenance={"source": ctx.source_label},
            )

            inside = roi_mask(geometry, grid)
            combined = np.asarray(result.mask) & inside
            arr = np.where(combined, np.asarray(result.array), np.nan).astype("float32")

        stats = calculate_roi_ndvi_stats(
            arr, geometry, grid.transform, combined,
            crs=grid.crs, roi_crs=roi_crs,
        )
        stats.index_name = INDEX_NAME
        stats.raster = arr
        stats.mask = combined
        stats.runtime_ms = (time.perf_counter() - started) * 1000.0
    except ValueError as exc:
        # CRS mismatch and geometry problems are contract violations, not crashes
        return _execution(query, Status.ERROR, f"The analysis could not be run: {exc}")
    except Exception as exc:                     # never show a traceback to the user
        return _execution(query, Status.ERROR,
                          f"The NDWI analysis failed: {exc}")

    # --- 4. provenance ------------------------------------------------------- #
    provenance: Dict[str, Any] = {
        "engine": __name__,
        "index": definition.to_dict(),
        "formula": definition.formula,
        "citation": definition.citation,
        "min_denominator": definition.min_denominator,
        "bands": {role: {"index": int(ctx.roles[role]),
                         "band_id": definition.band_id(role, ctx.profile)}
                  for role in definition.roles},
        "reflectance": {"scale": float(ctx.scale), "offset": float(ctx.offset),
                        "source": ctx.reflectance_source,
                        "is_reflectance": bool(ctx.is_reflectance)},
        "role_resolution": {"confidence": ctx.role_confidence,
                            "evidence": list(ctx.role_evidence)},
        "source": ctx.source_label,
        "native_resolution_m": [float(res), float(res)],
        "crs": str(crs),
        "roi_window": [int(col_off), int(row_off), int(w), int(h)],
        "roi_cells": int(grid.cells),
        "valid_pixels": int(stats.valid_pixels),
        "runtime_ms": round(stats.runtime_ms, 2),
        "router": {
            "intent": query.intent.value,
            "confidence": round(float(query.confidence), 3),
            "matched": list(query.matched),
            "explanation": query.explanation,
        },
    }

    # --- 5. honest reporting of the empty cases ------------------------------ #
    if stats.valid_pixels == 0:
        return _execution(
            query, Status.NO_VALID_PIXELS,
            NO_PIXELS_MESSAGE if stats.pixels_inside_roi == 0 else NO_VALID_MESSAGE,
            warnings=tuple(stats.warnings) + (NDWI_CAVEAT,),
            result=stats, provenance=provenance,
        )

    return _execution(
        query, Status.OK,
        NDWI_ANSWER_TEMPLATE.format(
            mean=float(stats.stats["mean"]),
            n=int(stats.valid_pixels),
            pct=100.0 * float(stats.valid_fraction),
        ),
        warnings=tuple(stats.warnings) + (NDWI_CAVEAT,),
        result=stats, provenance=provenance,
    )


__all__ = [
    "run_ndwi_roi_stats",
    "NDWI_ANSWER_TEMPLATE",
    "NDWI_CAVEAT",
    "OVERSIZED_MESSAGE",
    "missing_roles_message",
    "INDEX_NAME",
]
