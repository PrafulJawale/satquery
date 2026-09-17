"""Phase 9, Checkpoint C -- the multi-condition spatial query engine.

This module is an ORCHESTRATOR, not a new model and not a copy of anything:

    Spatial Query
        |-- condition "suitable cotton" -> the EXISTING Phase 8 engine
        |-- condition "cropland"        -> the EXISTING WorldCover datasource
        |-- condition "water"           -> the EXISTING WorldCover datasource
        `-- condition "near water"      -> core.spatial.proximity_mask (EDT)
                 |
                 v
        three-valued mask combination (core.spatial)
                 |
                 v
        SpatialQueryResult

Rules that shape the code below:

* **Nothing is computed for an unsupported condition.** Irrigation, flood risk,
  groundwater or salinity abort the query with an explanation; they are never
  proxied (irrigation is not "near water").
* **Missing data is never FALSE.** A cell with no data stays INSUFFICIENT all
  the way through the combination and is counted separately.
* **Zero matches is a result, not a failure.** The wording is "0% of the
  analysed cells satisfy all requested conditions" -- never "no such land
  exists", which would be a claim about the world rather than about the ROI.
* **Distances are metres**, computed on the projected analysis grid with the
  real cell size; never degrees, never a bare pixel count.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.alignment import make_grid, roi_mask
from core.datasources import worldcover
from core.geometry import as_crs
from core.router import Intent, QueryIntent
from core.spatial import (
    FALSE, INSUFFICIENT, TRUE, Grid, GridMask, GridMismatchError, apply_roi,
    combine_all, crop_mask, land_cover_mask, mask_from_class_raster, negate,
    proximity_mask, water_mask_from_land_cover,
)
from core.spatial_query import (
    ConditionStatus, ConditionType, Operator, SpatialCondition, SpatialQuery,
    SpatialQueryStatus, parse_spatial_query,
)

from .base import AnalysisContext, AnalysisExecution, Status

#: The finer-grained outcome carried by `SpatialQueryResult.status`.
RESULT_OK = "ok"
RESULT_ZERO_MATCHES = "zero_matches"
RESULT_INSUFFICIENT_DATA = "insufficient_data"

SCREENING_NOTE = (
    "This is a screening result from mapped datasets, not a recommendation "
    "and not a prediction."
)


# =========================================================================== #
# the result
# =========================================================================== #
@dataclass
class SpatialQueryResult:
    """Everything the UI needs to explain the answer -- and nothing it does not.

    `result_mask` holds the three-valued codes (2 match / 1 no / 0 unknown) on
    the analysis grid; it is deliberately excluded from `to_dict()`.
    """

    query: str
    normalized_query: str
    conditions: Tuple[SpatialCondition, ...]
    operator: Operator
    status: str = RESULT_OK
    message: str = ""

    # -- the geography ------------------------------------------------------ #
    result_mask: Optional[Any] = None
    #: name -> three-valued codes (2/1/0) per condition, for verification and
    #: for the future map layer. Excluded from `to_dict()`.
    condition_masks: Dict[str, Any] = field(default_factory=dict)
    grid: Optional[Dict[str, Any]] = None
    roi_mask: Optional[Any] = None

    # -- the numbers -------------------------------------------------------- #
    matched_cell_count: int = 0
    non_matching_cell_count: int = 0
    insufficient_cell_count: int = 0
    analysed_cell_count: int = 0
    matched_area_m2: float = 0.0
    matched_fraction: float = 0.0
    insufficient_fraction: float = 0.0

    # -- provenance --------------------------------------------------------- #
    condition_results: List[Dict[str, Any]] = field(default_factory=list)
    analysis_resolution: Optional[float] = None
    requested_resolution: Optional[float] = None
    source_resolutions: Dict[str, Any] = field(default_factory=dict)
    effective_resolution_note: str = ""
    warnings: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    performance: Dict[str, Any] = field(default_factory=dict)

    # -- views -------------------------------------------------------------- #
    @property
    def expression(self) -> str:
        """The structured query, e.g. `cotton(class >= 3) AND NOT water(80)`."""
        parts: List[str] = []
        for condition in self.conditions:
            text = _condition_expression(condition)
            parts.append(f"NOT {text}" if condition.negate else text)
        joiner = " AND " if self.operator is Operator.AND else " OR "
        return joiner.join(parts)

    @property
    def matched_area_km2(self) -> float:
        return self.matched_area_m2 / 1.0e6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "expression": self.expression,
            "operator": self.operator.value,
            "status": self.status,
            "message": self.message,
            "conditions": [c.to_dict() for c in self.conditions],
            "condition_results": list(self.condition_results),
            "matched_cell_count": self.matched_cell_count,
            "non_matching_cell_count": self.non_matching_cell_count,
            "insufficient_cell_count": self.insufficient_cell_count,
            "analysed_cell_count": self.analysed_cell_count,
            "matched_area_m2": self.matched_area_m2,
            "matched_area_km2": self.matched_area_km2,
            "matched_fraction": self.matched_fraction,
            "insufficient_fraction": self.insufficient_fraction,
            "result_mask": None,            # arrays stay out of the payload
            "condition_mask_names": sorted(self.condition_masks),
            "result_mask_shape": (None if self.result_mask is None
                                  else list(np.asarray(self.result_mask).shape)),
            "analysis_resolution": self.analysis_resolution,
            "requested_resolution": self.requested_resolution,
            "source_resolutions": dict(self.source_resolutions),
            "effective_resolution_note": self.effective_resolution_note,
            "grid": self.grid,
            "warnings": list(self.warnings),
            "provenance": list(self.provenance),
            "performance": dict(self.performance),
        }


def _condition_expression(condition: SpatialCondition) -> str:
    if condition.condition_type is ConditionType.CROP_SUITABILITY:
        return (f"cotton_suitability({condition.parameters.get('crop')}, "
                f"class >= {condition.parameters.get('min_class')}, "
                f"{condition.parameters.get('scenario')})")
    if condition.condition_type is ConditionType.LAND_COVER_CLASS:
        return f"land_cover({condition.parameters.get('classes')})"
    if condition.condition_type is ConditionType.WATER:
        return f"water(class {condition.parameters.get('classes')})"
    if condition.condition_type is ConditionType.WATER_PROXIMITY:
        return (f"water_proximity(<= {condition.parameters.get('distance_m')} m"
                f", class {condition.parameters.get('water_class')})")
    return f"unsupported({condition.parameters.get('topic')})"


# =========================================================================== #
# the engine
# =========================================================================== #
def _default_suitability_runner(context: AnalysisContext, query: QueryIntent,
                                **kwargs: Any) -> AnalysisExecution:
    """The Phase 8 cotton engine, reused as-is (never re-implemented)."""
    from analyses.crop_suitability import run_crop_suitability
    return run_crop_suitability(context, query, **kwargs)


def expanded_analysis_grid(grid: Any, extra_cells: int,
                           max_cells: int = 2_000_000) -> Tuple[Any, int]:
    """The same grid, extended by `extra_cells` on every side.

    The transform coefficients are kept and only the origin moves by whole
    cells, so the expanded window is pixel-aligned with the original and the
    result can be cropped back exactly (Decision 1). The window is capped at
    `max_cells`; if the cap bites, the caller must warn that the proximity rim
    is undecidable again.
    """
    if extra_cells <= 0:
        return grid, 0
    import dataclasses
    from rasterio.transform import Affine

    applied = int(extra_cells)
    while applied > 0 and ((grid.width + 2 * applied)
                           * (grid.height + 2 * applied) > max_cells):
        applied //= 2
    if applied <= 0:
        return grid, 0

    a, b, c, d, e, f = (float(v) for v in tuple(grid.transform)[:6])
    width, height = int(grid.width) + 2 * applied, int(grid.height) + 2 * applied
    transform = Affine(a, b, c - applied * a, d, e, f - applied * e)
    minx = c - applied * a
    maxx = minx + width * a
    maxy = f - applied * e
    miny = maxy + height * e
    return (dataclasses.replace(
        grid, transform=transform, width=width, height=height,
        bounds=(minx, miny, maxx, maxy),
        buffer_cells=int(getattr(grid, "buffer_cells", 0)) + applied,
        note=(grid.note or "")), applied)


def _default_land_cover_fetcher(analysis_grid: Any, use_cache: bool = True) -> Any:
    """The existing WorldCover datasource, windowed on the analysis grid."""
    return worldcover.fetch_land_cover(analysis_grid, use_cache=use_cache)


def run_spatial_query(context: AnalysisContext,
                      query: QueryIntent,
                      *,
                      use_cache: bool = True,
                      analysis_resolution: Optional[float] = None,
                      suitability_runner: Optional[Callable[..., Any]] = None,
                      land_cover_fetcher: Optional[Callable[..., Any]] = None,
                      progress: Optional[Callable[[str], None]] = None,
                      ) -> AnalysisExecution:
    """Execute a structured multi-condition spatial query over the ROI."""
    started = time.perf_counter()
    timings: Dict[str, float] = {}
    warnings: List[str] = list(getattr(query, "conditions_warnings", ()) or ())

    # ---- 1. the structured query ------------------------------------------ #
    parsed: SpatialQuery = _structured_query(query)
    warnings.extend(parsed.warnings)

    if parsed.status is SpatialQueryStatus.NO_CONDITIONS or not parsed.conditions:
        return _execution(query, Status.UNKNOWN,
                          "I could not recognise a mappable condition there. "
                          "Try asking for areas that are, for example, "
                          "suitable for cotton, cropland, or near water.",
                          warnings=warnings)

    unsupported = parsed.unsupported_conditions
    if unsupported:
        detail = " ".join(str(c.note or c.interpretation).strip()
                          for c in unsupported)
        return _execution(
            query, Status.UNSUPPORTED_CONDITION,
            f"{detail} Nothing was computed: a proxy would be misleading.",
            warnings=warnings,
            provenance={"engine": __name__,
                        "blocked_by": [c.to_dict() for c in unsupported]})

    if parsed.status is not SpatialQueryStatus.OK:
        notes = " ".join(parsed.notes)
        return _execution(query, Status.UNKNOWN,
                          f"I need one clarification before mapping that: {notes}"
                          if notes else
                          "I could not resolve that into mappable conditions.",
                          warnings=warnings)

    conditions = list(parsed.supported_conditions)
    if not conditions:
        return _execution(query, Status.UNKNOWN,
                          "No supported condition survived parsing, so nothing "
                          "was mapped.", warnings=warnings)

    # ---- 2. the ROI ------------------------------------------------------- #
    roi = context.roi
    if roi is None or not getattr(roi, "usable", False):
        return _execution(query, Status.NEEDS_ROI,
                          "Please select an area on the map first.",
                          warnings=warnings)

    # ---- 3. the analysis grid --------------------------------------------- #
    # A proximity condition needs enough window around the ROI for the distance
    # to be decidable: any water within N metres must be inside the window.
    max_distance = max([float(c.parameters.get("distance_m", 0) or 0)
                        for c in conditions
                        if c.condition_type is ConditionType.WATER_PROXIMITY]
                       or [0.0])
    resolution = float(analysis_resolution or 30.0)
    wanted_buffer = (int(math.ceil(max_distance / resolution)) + 1
                     if max_distance > 0 else 2)

    runner = suitability_runner or _default_suitability_runner
    fetcher = land_cover_fetcher or _default_land_cover_fetcher

    needs_cotton = any(c.required_analysis == "crop_suitability"
                       for c in conditions)
    screening = None
    grid: Optional[Grid] = None
    analysis_grid: Any = None

    if needs_cotton:
        if progress:
            progress("Running the cotton suitability screening")
        t0 = time.perf_counter()
        execution = runner(context, query, use_cache=use_cache)
        timings["suitability_seconds"] = time.perf_counter() - t0
        if execution is None or execution.result is None:
            # the Phase 8 engine explains why (no ROI, unknown crop, no data)
            return execution if execution is not None else _execution(
                query, Status.ERROR, "The cotton screening did not run.")
        screening = execution.result
        warnings.extend(w for w in (execution.warnings or ()) if w
                        not in warnings and w != SCREENING_NOTE)
        analysis_grid = screening.grid
        if analysis_grid is None:
            return _execution(query, Status.INSUFFICIENT_DATA,
                              "The cotton screening produced no analysis grid, "
                              "so no spatial conditions could be mapped.",
                              warnings=warnings)
        grid = Grid.from_analysis_grid(analysis_grid)
    else:
        t0 = time.perf_counter()
        crs = as_crs(roi.raster_crs)
        analysis_grid = make_grid(roi.geometry_raster_crs, crs,
                                  requested_resolution=resolution,
                                  buffer_cells=wanted_buffer)
        timings["grid_seconds"] = time.perf_counter() - t0
        if analysis_grid.note:
            warnings.append(analysis_grid.note)
        grid = Grid.from_analysis_grid(analysis_grid)

    # ---- 3b. the source window -------------------------------------------- #
    # Phase 8 owns the analysis grid and must not be touched, so when a
    # proximity condition needs more room than the Phase 8 buffer gives, the
    # WORLDCOVER READ is made on a wider, pixel-aligned window and cropped back
    # to the analysis grid (Decision 1). The buffer is the requested distance
    # plus a 2-cell margin for alignment/reprojection.
    layer_grid = analysis_grid
    layer_grid_obj: Optional[Grid] = grid
    buffer_applied = 0
    window_info: Dict[str, Any] = {}
    if needs_cotton and max_distance > 0:
        wanted = int(math.ceil(max_distance / resolution)) + 2
        layer_grid, buffer_applied = expanded_analysis_grid(analysis_grid, wanted)
        layer_grid_obj = Grid.from_analysis_grid(layer_grid)
        window_info.update({
            "buffer_cells": buffer_applied,
            "source_window_cells": [layer_grid.height, layer_grid.width],
            "analysis_window_cells": [grid.height, grid.width],
            "buffer_metres": buffer_applied * grid.resolution_m,
        })
        if buffer_applied < wanted:
            warnings.append(
                f"The proximity buffer was capped at {buffer_applied} cells "
                f"(wanted {wanted}) to keep the window bounded, so cells within "
                f"{max_distance:.0f} m of the window edge are reported as "
                "insufficient data rather than assumed to be far from water.")
        else:
            warnings.append(
                f"Permanent-water proximity was computed on a buffered "
                f"WorldCover window ({layer_grid.height}x{layer_grid.width} "
                f"cells, +{buffer_applied} cells per side = "
                f"{buffer_applied * grid.resolution_m:.0f} m) and cropped back "
                f"to the {grid.height}x{grid.width} analysis grid.")

    inside = roi_mask(roi.geometry_raster_crs, analysis_grid)
    timings["roi_mask_seconds"] = time.perf_counter() - started
    if not inside.any():
        return _execution(query, Status.INSUFFICIENT_DATA,
                          "The selected area covers no analysis cell, so "
                          "nothing could be mapped.", warnings=warnings)

    # ---- 4. one mask per condition ---------------------------------------- #
    masks: List[GridMask] = []
    condition_results: List[Dict[str, Any]] = []
    condition_masks: Dict[str, Any] = {}
    provenance: List[Dict[str, Any]] = []
    source_resolutions: Dict[str, Any] = {}

    # One windowed WorldCover read serves every land-cover-derived condition,
    # and the class-80 mask derived from it is reused by every proximity
    # condition. Nothing is read twice and nothing global is loaded.
    land_cover_layer: Any = None
    water_mask_cache: Optional[GridMask] = None

    def get_land_cover() -> Any:
        nonlocal land_cover_layer
        if land_cover_layer is None:
            if progress:
                progress("Fetching ESA WorldCover land cover (10 m)")
            t0 = time.perf_counter()
            land_cover_layer = fetcher(layer_grid, use_cache=use_cache)
            timings.setdefault("landcover_seconds",
                               time.perf_counter() - t0)
        return land_cover_layer

    def get_water_mask() -> GridMask:
        nonlocal water_mask_cache
        if water_mask_cache is None:
            layer = get_land_cover()
            array = layer.array if hasattr(layer, "array") else layer
            t0 = time.perf_counter()
            water_mask_cache = water_mask_from_land_cover(
                array, layer_grid_obj, water_class=80, name="water(class 80)",
                source="worldcover")
            timings.setdefault("water_mask_seconds", time.perf_counter() - t0)
        return water_mask_cache

    for condition in conditions:
        # The cotton raster lives on the analysis grid (Phase 8 owns it); only
        # land-cover-derived conditions are computed on the buffered window.
        condition_grid = (grid if condition.required_analysis == "crop_suitability"
                          else layer_grid_obj)
        mask, summary, condition_provenance, resolutions = _build_condition_mask(
            condition, grid=condition_grid, screening=screening,
            land_cover_provider=get_land_cover, water_provider=get_water_mask)
        # crop back to the analysis grid when the source window was buffered
        if condition_grid is not grid:
            mask = crop_mask(mask, grid)
        masks.append(mask)
        condition_masks[mask.name] = mask.state
        condition_results.append(summary)
        provenance.extend(condition_provenance)
        source_resolutions.update(resolutions)

    # ---- 5. combine -------------------------------------------------------- #
    try:
        combined = combine_all(masks, parsed.operator)
    except GridMismatchError as exc:
        return _execution(query, Status.INSUFFICIENT_DATA,
                          f"The requested conditions could not be combined: {exc}",
                          warnings=warnings)

    combined = apply_roi(combined, inside)
    # Count inside the ROI only: the window buffer around it is not part of the
    # answer and must not be reported as missing data.
    counts = combined.counts(scope=inside)

    # ---- 6. the result ---------------------------------------------------- #
    matched = int(counts["matching_cells"])
    analysed = int(counts["valid_cells"])
    insufficient = int(counts["insufficient_cells"])

    if analysed == 0:
        result_status = RESULT_INSUFFICIENT_DATA
    elif matched == 0:
        result_status = RESULT_ZERO_MATCHES
    else:
        result_status = RESULT_OK

    timings["total_seconds"] = time.perf_counter() - started

    result = SpatialQueryResult(
        query=query.original_query,
        normalized_query=query.normalized_query,
        conditions=tuple(conditions),
        operator=parsed.operator,
        status=result_status,
        result_mask=combined.state,
        roi_mask=inside,
        grid=grid.to_dict() if grid else None,
        matched_cell_count=matched,
        non_matching_cell_count=int(counts["non_matching_cells"]),
        insufficient_cell_count=insufficient,
        analysed_cell_count=analysed,
        matched_area_m2=float(counts["matched_area_m2"]),
        matched_fraction=float(counts["matched_fraction"]),
        insufficient_fraction=float(counts["insufficient_fraction"]),
        condition_results=condition_results,
        condition_masks=condition_masks,
        analysis_resolution=(grid.resolution_m if grid else None),
        requested_resolution=resolution,
        source_resolutions=source_resolutions,
        effective_resolution_note=_resolution_note(grid, source_resolutions),
        warnings=list(warnings),
        provenance=provenance,
        performance={"timings": timings,
                     "cells": int(combined.n_cells),
                     "conditions": len(conditions),
                     "window": dict(window_info)},
    )
    result.message = compose_message(result)

    status = (Status.INSUFFICIENT_DATA if result_status == RESULT_INSUFFICIENT_DATA
              else Status.OK)
    return _execution(query, status, result.message, warnings=warnings,
                      result=result,
                      provenance={"engine": __name__,
                                  "conditions": [c.to_dict() for c in conditions],
                                  "operator": parsed.operator.value})


# --------------------------------------------------------------------------- #
# one condition -> one three-valued mask
# --------------------------------------------------------------------------- #
def _build_condition_mask(condition: SpatialCondition, *, grid: Grid,
                          screening: Any,
                          land_cover_provider: Callable[[], Any],
                          water_provider: Callable[[], GridMask],
                          ) -> Tuple[GridMask, Dict[str, Any],
                                     List[Dict[str, Any]], Dict[str, Any]]:
    """One condition -> one three-valued mask on the shared analysis grid."""
    provenance: List[Dict[str, Any]] = []
    resolutions: Dict[str, Any] = {}

    if condition.condition_type is ConditionType.CROP_SUITABILITY:
        scenario = str(condition.parameters.get("scenario", "rainfed"))
        min_class = int(condition.parameters.get("min_class", 3))
        screening_result = (screening.scenarios.get(scenario)
                            if screening is not None else None)
        if screening_result is None or screening_result.suitability_raster is None:
            raise ValueError(
                f"the cotton screening produced no '{scenario}' raster")
        mask = mask_from_class_raster(
            screening_result.suitability_raster, grid, min_class=min_class,
            name=condition.label, source="phase8:cotton_suitability",
            scenario=scenario,
            crop=condition.parameters.get("crop"))
        resolutions.update(screening_result.native_resolutions or {})
        for entry in (screening_result.provenance or []):
            provenance.append({**entry, "used_for": "cotton suitability"})
        summary_extra = {
            "scenario": scenario,
            "min_class": min_class,
            "class_label": f"class >= {min_class}",
            "analysis_resolution_m": screening_result.analysis_resolution,
        }
    else:
        # every remaining condition is derived from the WorldCover layer
        layer = land_cover_provider()
        array = layer.array if hasattr(layer, "array") else layer
        record = (layer.record.to_dict() if hasattr(layer, "record")
                  else {"source": "ESA WorldCover 2021 v200"})
        provenance.append({**record, "used_for": condition.condition_type.value,
                           "role": "categorical (nearest neighbour)"})
        resolutions["worldcover"] = record.get("native_resolution_m", 10.0)

        if condition.condition_type is ConditionType.WATER:
            mask = water_mask_from_land_cover(
                array, grid,
                water_class=int((condition.parameters.get("classes") or [80])[0]),
                name=condition.label, source="worldcover")
        elif condition.condition_type is ConditionType.LAND_COVER_CLASS:
            mask = land_cover_mask(
                array, grid,
                classes_wanted=list(condition.parameters.get("classes") or []),
                name=condition.label, source="worldcover")
        elif condition.condition_type is ConditionType.WATER_PROXIMITY:
            base = water_provider()
            mask = proximity_mask(
                base, float(condition.parameters.get("distance_m", 1000.0)),
                name=condition.label)
            provenance.append({
                "source": "derived from ESA WorldCover class 80",
                "used_for": "water proximity",
                "method": mask.provenance.get("method"),
                "distance_m": mask.provenance.get("distance_m"),
            })
        else:                                        # pragma: no cover
            raise ValueError(f"unsupported condition: {condition.condition_type}")
        summary_extra = {"source": "ESA WorldCover 2021 v200 (10 m)"}

    if condition.negate:
        mask = negate(mask)

    summary = mask.to_dict()
    summary.update(summary_extra)
    summary.update({
        "condition": condition.condition_type.value,
        "negated": bool(condition.negate),
        "evidence": list(condition.evidence),
        "interpretation": condition.interpretation,
        "status": condition.status.value,
    })
    return mask, summary, provenance, resolutions


# --------------------------------------------------------------------------- #
# wording -- generated from computed evidence only
# --------------------------------------------------------------------------- #
def compose_message(result: SpatialQueryResult) -> str:
    """The user-facing sentence. Zero matches is stated as a geographic fact
    about the ROI, never as 'no such land exists'."""
    pct = result.matched_fraction * 100.0
    headline = (
        f"Within the selected ROI, {pct:.1f}% of analysed cells "
        f"({result.matched_cell_count:,} cells, "
        f"{result.matched_area_km2:.3f} km²) satisfy all requested conditions "
        f"— {result.expression}.")

    if result.status == RESULT_ZERO_MATCHES:
        headline = (
            "Within the selected ROI, 0% of analysed cells satisfy all "
            f"requested conditions — {result.expression}. "
            "That is a statement about the selected area, not about the "
            "surrounding region.")

    parts = [headline]

    if result.insufficient_cell_count:
        parts.append(
            f"For {result.insufficient_fraction * 100:.1f}% of cells the "
            "required data was missing or undecidable (insufficient data), so "
            "no claim is made about them; they are counted as neither "
            "matching nor non-matching.")

    if result.condition_results:
        detail = "; ".join(
            f"{c['name']}: {c['counts']['matching_cells']:,} cells"
            for c in result.condition_results)
        parts.append(f"Per condition: {detail}.")

    parts.append(SCREENING_NOTE)
    return " ".join(parts)


def _resolution_note(grid: Optional[Grid],
                     source_resolutions: Dict[str, Any]) -> str:
    """The Phase 8 native/effective distinction, carried over."""
    if grid is None:
        return ""
    native = sorted({float(v) for v in source_resolutions.values()
                     if isinstance(v, (int, float))})
    if not native:
        return f"Analysed at {grid.resolution_m:g} m."
    return (f"Analysed at {grid.resolution_m:g} m; inputs are native "
            f"{min(native):g}–{max(native):g} m, so finer detail than the "
            f"analysis grid is not resolved.")


def _structured_query(query: QueryIntent) -> SpatialQuery:
    """The structured query: reuse the router's, else parse the text."""
    conditions = tuple(getattr(query, "conditions", ()) or ())
    if conditions:
        return parse_spatial_query(query.original_query)
    return parse_spatial_query(query.original_query)


def _execution(query: QueryIntent, status: Status, message: str, *,
               warnings: Sequence[str] = (), result: Any = None,
               provenance: Optional[Dict[str, Any]] = None) -> AnalysisExecution:
    return AnalysisExecution(
        intent=Intent.SPATIAL_QUERY, status=status, query=query.original_query,
        normalized_query=query.normalized_query, confidence=query.confidence,
        explanation=query.explanation, matched=query.matched, message=message,
        result=result, warnings=tuple(warnings),
        provenance=provenance if provenance is not None else {"engine": __name__},
    )


__all__ = [
    "SpatialQueryResult", "run_spatial_query", "compose_message",
    "RESULT_OK", "RESULT_ZERO_MATCHES", "RESULT_INSUFFICIENT_DATA",
]
