"""Phase 12 -- multi-condition evidence composition.

WHAT THIS ENGINE DOES
---------------------
It intersects evidence that ALREADY exists and is ALREADY tested:

    spatial   Phase 9   cropland / permanent water / near water / NOT water
    spectral  Phase 11  NDVI or NDWI, thresholded on a native index raster
    temporal  Phase 10  NDVI increase / stable / decrease classes

It computes no new science. Every mask comes from another phase's engine or
from `core.spatial`, and the combination is three-valued throughout.

WHAT IT MUST NEVER DO
---------------------
* Invent a threshold. A spectral condition with no number stops the run with
  NEEDS_THRESHOLD and names the index.
* Combine masks that do not share one grid. Grids are made identical BY
  CONSTRUCTION (one composition grid, chosen before any mask exists) and
  `require_compatible()` still verifies before combining.
* Collapse UNKNOWN into FALSE. A cell that could not be measured is amber,
  never grey, and is never counted as a non-match.
* Turn a combination into a cause. Every result carries the caveat:
  "A combined condition is geographic evidence, not causal attribution."
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from core.alignment import DEFAULT_MAX_CELLS, native_roi_grid, roi_mask
from core.index_definitions import get_index
from core.indices import index_from_dataset
from core.multi_condition import (
    ComposedCondition,
    ComposedQuery,
    ConditionKind,
    ThresholdProvenance,
    combine_conditions,
    load_threshold_config,
    negated,
    parse_composed_query,
    state_counts,
    summarise_index,
    temporal_class_mask,
    threshold_mask,
)
from core.reflectance import ReflectanceSpec
from core.router import Intent, QueryIntent
from core.spatial import Grid, GridMask, apply_roi, crop_mask
from core.spatial import (
    land_cover_mask,
    proximity_mask,
    water_mask_from_land_cover,
)

from .base import AnalysisContext, AnalysisExecution, Status

#: The limitations, with the boundary statement first and de-duplicated.
CAVEAT_FIRST: Tuple[str, ...] = ()

RESULT_OK = "ok"
RESULT_ZERO_MATCHES = "zero_matches"
RESULT_INSUFFICIENT_DATA = "insufficient_data"

#: The mandatory caveat. It is repeated in the panel and on the legend.
#: The boundary statement. It leads the limitations so it can never be
#: scrolled past, and it is stored as a whole sentence so the UI, the map
#: legend and the docs cannot drift apart on the wording.
CAVEAT = ("A combined condition is geographic evidence, "
          "not causal attribution.")

#: Claims that may appear ONLY inside the caveat or the limitations.
FORBIDDEN_CLAIMS = (
    "proves", "proof of", "caused by", "because of flooding",
    "is flooded", "flooded area", "water scarcity", "crop failure",
    "drought", "deforestation",
)


@dataclass
class MultiConditionResult:
    """The composed answer: what was asked, what matched, what could not be."""

    query: str = ""
    normalized_query: str = ""
    conditions: Tuple[ComposedCondition, ...] = ()
    operator: str = "and"
    status: str = RESULT_OK
    message: str = ""

    # -- the geography ------------------------------------------------------ #
    combined_mask: Any = None            # three-valued codes (2/1/0)
    combined_gridmask: Any = None
    condition_masks: Dict[str, Any] = field(default_factory=dict)
    grid: Optional[Dict[str, Any]] = None
    roi_mask: Any = None

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
    index_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_analyses: Tuple[str, ...] = ()
    source_dates: Dict[str, str] = field(default_factory=dict)
    alignment: Dict[str, Any] = field(default_factory=dict)
    threshold_provenance: Dict[str, Any] = field(default_factory=dict)
    analysis_resolution: Optional[float] = None
    source_resolutions: Dict[str, Any] = field(default_factory=dict)
    limitations: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)

    # -- views -------------------------------------------------------------- #
    @property
    def expression(self) -> str:
        joiner = " AND " if self.operator == "and" else " OR "
        return joiner.join(
            (f"NOT {c.label}" if c.negate else c.label) for c in self.conditions)

    @property
    def matched_area_km2(self) -> float:
        return self.matched_area_m2 / 1.0e6

    @property
    def all_unknown(self) -> bool:
        return self.insufficient_cell_count > 0 and self.analysed_cell_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "expression": self.expression,
            "operator": self.operator,
            "status": self.status,
            "message": self.message,
            "conditions": [c.to_dict() for c in self.conditions],
            "matched_cell_count": self.matched_cell_count,
            "non_matching_cell_count": self.non_matching_cell_count,
            "insufficient_cell_count": self.insufficient_cell_count,
            "analysed_cell_count": self.analysed_cell_count,
            "matched_area_m2": self.matched_area_m2,
            "matched_area_km2": self.matched_area_km2,
            "matched_fraction": self.matched_fraction,
            "insufficient_fraction": self.insufficient_fraction,
            "condition_results": list(self.condition_results),
            "index_summaries": dict(self.index_summaries),
            "source_analyses": list(self.source_analyses),
            "source_dates": dict(self.source_dates),
            "alignment": dict(self.alignment),
            "threshold_provenance": dict(self.threshold_provenance),
            "analysis_resolution": self.analysis_resolution,
            "source_resolutions": dict(self.source_resolutions),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "performance": dict(self.performance),
        }


def _execution(query: QueryIntent, status: Status, message: str, *,
               result: Any = None, warnings: Any = (),
               provenance: Optional[Dict[str, Any]] = None) -> AnalysisExecution:
    return AnalysisExecution(
        intent=query.intent, status=status, query=query.original_query,
        normalized_query=query.normalized_query, confidence=query.confidence,
        explanation=query.explanation, matched=query.matched, message=message,
        result=result, warnings=tuple(warnings),
        provenance=provenance if provenance is not None else {"engine": __name__},
    )


def _analysis_grid(grid: Grid) -> Any:
    """core.spatial.Grid -> core.alignment.AnalysisGrid (for datasource reads)."""
    from core.alignment import AnalysisGrid
    from rasterio.transform import Affine

    # `tuple(Affine)` yields NINE values (the full 3x3 matrix); Phase 9's
    # expanded_analysis_grid slices to six for exactly this reason.
    a, b, c, d, e, f = (float(v) for v in tuple(grid.transform)[:6])
    return AnalysisGrid(
        crs=grid.crs, transform=Affine(a, b, c, d, e, f),
        width=int(grid.width), height=int(grid.height),
        resolution=float(grid.resolution_m),
        requested_resolution=float(grid.resolution_m),
        bounds=(float(c), float(f) + float(e) * grid.height,
                float(c) + float(a) * grid.width, float(f)),
        note=grid.note or "composition grid",
    )


def _grid_bounds(grid: Grid) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of a `core.spatial.Grid`.

    `Grid` deliberately carries no bounds field (it derives everything from the
    transform), so they are computed here -- the same way Phase 9's
    `expanded_analysis_grid` does it.
    """
    a, b, c, d, e, f = (float(v) for v in tuple(grid.transform)[:6])
    minx, maxy = c, f
    return (minx, f + e * int(grid.height), c + a * int(grid.width), maxy)


CAVEAT_FIRST = (CAVEAT,)


def _overlaps(geometry: Any, bounds: Any) -> bool:
    """True when the ROI touches the raster extent at all.

    Without this, a ROI drawn outside the image would produce a 1x1 grid that
    "passes" the cell budget and then reports a meaningless zero-match answer.
    """
    from rasterio.coords import BoundingBox
    left, bottom, right, top = (bounds.left, bounds.bottom,
                                bounds.right, bounds.top)
    try:
        from shapely.geometry import box as _box
        return bool(_box(left, bottom, right, top).intersects(geometry))
    except Exception:
        return not (geometry.bounds[2] <= left or geometry.bounds[0] >= right
                    or geometry.bounds[3] <= bottom
                    or geometry.bounds[1] >= top)


def _pixel_area(transform: Any) -> float:
    a, b, _c, d, e, _f = (float(v) for v in tuple(transform)[:6])
    return abs(a * e - b * d)


def _grid_within_budget(grid: Grid) -> bool:
    return int(grid.width) * int(grid.height) <= DEFAULT_MAX_CELLS


def run_multi_condition(context: AnalysisContext,
                        query: QueryIntent,
                        *,
                        use_cache: bool = True,
                        land_cover_fetcher: Optional[Any] = None) -> AnalysisExecution:
    """...

    `land_cover_fetcher` is a testability seam, not a configuration option: it
    lets the unit tests supply a synthetic WorldCover layer instead of reaching
    the network. It defaults to the real datasource, exactly as Phase 9 does.
    """
    """Compose existing evidence for the selected area. No new science."""
    started = time.perf_counter()
    cfg = load_threshold_config()
    composed: ComposedQuery = (getattr(query, "composed", None)
                               or parse_composed_query(query.original_query, cfg))

    # ---- 1. an area must be selected --------------------------------------- #
    roi = getattr(context, "roi", None)
    if roi is None or not getattr(roi, "usable", False):
        return _execution(query, Status.NEEDS_ROI,
                          "Please select an area on the map first.")

    geometry = getattr(roi, "geometry_raster_crs", None) or getattr(roi, "geometry", None)
    if geometry is None:
        return _execution(query, Status.NEEDS_ROI,
                          "The selected area has no usable geometry. Draw it again.")

    # ---- 2. nothing that Phase 9 refuses may be computed -------------------- #
    refused = [c for c in composed.conditions
               if c.kind is ConditionKind.SPATIAL and c.name == "unsupported"]
    if refused:
        return _execution(
            query, Status.UNSUPPORTED_CONDITION,
            "; ".join(c.interpretation or c.label for c in refused) or
            "That condition is not measured by this system, so nothing was computed.")

    # ---- 3. no threshold is invented --------------------------------------- #
    missing = composed.missing_thresholds
    if missing:
        names = ", ".join(sorted({(c.threshold.index.upper() if c.threshold else "index")
                                  for c in missing}))
        return _execution(
            query, Status.NEEDS_THRESHOLD,
            f"{names} was asked for without a threshold, and no threshold will "
            f"be invented. Ask for e.g. "
            f"{names.split(',')[0].strip()} greater than 0.6, use a relative "
            f"form ('above the median of this area'), or enable a labelled "
            f"convention in the UI.",
            warnings=tuple(composed.warnings))

    conditions = composed.conditions
    if not conditions:
        return _execution(query, Status.UNKNOWN,
                          "No condition could be recognised in that request.")

    # ---- 4. the composition grid: ONE grid, chosen before any mask --------- #
    index_ctx = getattr(context, "index_context", None)
    spectral = composed.spectral_conditions
    temporal = composed.temporal_conditions

    if spectral and (index_ctx is None or not getattr(index_ctx, "path", "")):
        return _execution(
            query, Status.UNSUPPORTED,
            "An index threshold was asked for, but no raster with resolved "
            "band roles is available to compute the index from.")

    grid: Optional[Grid] = None
    temporal_result = None
    alignment: Dict[str, Any] = {}
    source_dates: Dict[str, str] = {}
    source_analyses: List[str] = []
    warnings: List[str] = list(composed.warnings)

    # A temporal condition fixes the grid: Phase 10 owns the change raster, so
    # the composition adopts ITS grid and everything else is brought onto it.
    if temporal:
        pair = getattr(context, "temporal_pair", None)
        if pair is None:
            return _execution(
                query, Status.NEEDS_TWO_DATES,
                "A vegetation-change condition needs two dated acquisitions. "
                "Select both dates -- the system never chooses one for you.")
        from analyses.ndvi_change import compare_ndvi

        temporal_result, t_status, t_message, t_warnings = compare_ndvi(
            pair, geometry, getattr(roi, "raster_crs", None))
        if temporal_result is None:
            return _execution(query, t_status, t_message, warnings=t_warnings)
        warnings.extend(t_warnings)
        class_raster = np.asarray(temporal_result.class_raster)
        grid = Grid(crs=temporal_result.crs,
                    transform=tuple(float(v) for v in tuple(temporal_result.transform)[:6]),
                    width=int(class_raster.shape[1]),
                    height=int(class_raster.shape[0]),
                    resolution_m=float(temporal_result.resolution or 10.0),
                    note="Phase 10 temporal grid (native, ROI window)")
        alignment["temporal"] = dict(temporal_result.alignment or {})
        alignment["grid_source"] = "phase10_temporal"
        source_dates["before"] = str(temporal_result.before_date or "")
        source_dates["after"] = str(temporal_result.after_date or "")
        source_analyses.append("analyses.ndvi_change")

    # otherwise the native Sentinel-2 grid over the ROI
    if grid is None:
        try:
            with rasterio.open(index_ctx.path) as ds:
                if not _overlaps(geometry, ds.bounds):
                    return _execution(
                        query, Status.INSUFFICIENT_DATA,
                        "The selected area does not overlap the loaded scene, "
                        "so no condition could be evaluated over it. Draw the "
                        "area inside the image, or load a scene that covers "
                        "it.",
                        provenance={"engine": __name__})
                analysis_grid = native_roi_grid(
                    ds.transform, ds.width, ds.height, float(ds.res[0]),
                    ds.crs, geometry)
        except Exception as exc:
            return _execution(query, Status.ERROR,
                              f"The analysis grid could not be built: {exc}")
        grid = Grid.from_analysis_grid(analysis_grid)
        alignment["grid_source"] = "native_roi_grid"

    if not _grid_within_budget(grid):
        return _execution(
            query, Status.INSUFFICIENT_DATA,
            f"The selected area is too large to compose at this resolution "
            f"({int(grid.width) * int(grid.height):,} cells, budget "
            f"{DEFAULT_MAX_CELLS:,}). Draw a smaller area: coarsening it "
            f"silently would change the question being asked.",
            provenance={"engine": __name__,
                        "roi_cells": int(grid.width) * int(grid.height),
                        "cell_budget": int(DEFAULT_MAX_CELLS),
                        "refused": "roi_exceeds_cell_budget"})

    analysis_grid = _analysis_grid(grid)
    inside = roi_mask(geometry, analysis_grid)
    masks: List[GridMask] = []
    condition_results: List[Dict[str, Any]] = []
    condition_masks: Dict[str, Any] = {}
    source_resolutions: Dict[str, Any] = {}
    provenance_entries: List[Dict[str, Any]] = []
    threshold_provenance: Dict[str, Any] = {}

    # ---- 5. spatial masks (Phase 9 primitives, Shared WorldCover read) ----- #
    spatial_conditions = composed.spatial_conditions
    land_cover_layer = None
    water_cache: Optional[GridMask] = None
    layer_grid = grid
    layer_analysis_grid = analysis_grid

    def get_land_cover() -> Any:
        nonlocal land_cover_layer
        if land_cover_layer is None:
            if land_cover_fetcher is not None:
                land_cover_layer = land_cover_fetcher(layer_analysis_grid)
            else:
                from core.datasources import worldcover
                land_cover_layer = worldcover.fetch_land_cover(
                    layer_analysis_grid, use_cache=use_cache)
        return land_cover_layer

    def get_water() -> GridMask:
        nonlocal water_cache
        if water_cache is None:
            layer = get_land_cover()
            array = layer.array if hasattr(layer, "array") else layer
            water_cache = water_mask_from_land_cover(
                array, layer_grid, water_class=80,
                name="water(class 80)", source="worldcover")
        return water_cache

    # proximity needs a buffered window so cells near the ROI edge are decidable
    max_distance = max(
        [float(c.parameters.get("distance_m", 1000.0))
         for c in spatial_conditions
         if c.name == "water_proximity"] or [0.0])
    if max_distance > 0:
        from analyses.spatial_query import expanded_analysis_grid
        extra = int(np.ceil(max_distance / float(grid.resolution_m))) + 1
        expanded, applied = expanded_analysis_grid(analysis_grid, extra)
        layer_analysis_grid = expanded
        layer_grid = Grid.from_analysis_grid(expanded)
        if applied < extra and applied > 0:
            warnings.append(
                f"Water proximity: the search window was reduced to "
                f"{applied} cells by the {DEFAULT_MAX_CELLS:,}-cell limit, so "
                f"cells within {max_distance:g} m of the ROI edge may be "
                f"reported as insufficient data.")

    if spatial_conditions:
        source_analyses.append("ESA WorldCover 2021 v200")
        source_resolutions["worldcover"] = 10.0

    for condition in spatial_conditions:
        spatial = condition.spatial_condition
        params = dict(condition.parameters or {})
        if condition.name == "water_proximity":
            base = get_water()
            mask = proximity_mask(base, float(params.get("distance_m", 1000.0)),
                                  name=condition.label)
            provenance_entries.append({
                "source": "derived from ESA WorldCover class 80",
                "used_for": "water proximity",
                "distance_m": params.get("distance_m", 1000.0),
                "method": mask.provenance.get("method")})
        elif condition.name == "water":
            layer = get_land_cover()
            array = layer.array if hasattr(layer, "array") else layer
            mask = water_mask_from_land_cover(
                array, layer_grid,
                water_class=int((params.get("classes") or [80])[0]),
                name=condition.label, source="worldcover")
            provenance_entries.append({"source": "ESA WorldCover class 80",
                                       "used_for": "permanent water"})
        elif condition.name == "land_cover_class":
            layer = get_land_cover()
            array = layer.array if hasattr(layer, "array") else layer
            mask = land_cover_mask(
                array, layer_grid,
                classes_wanted=list(params.get("classes") or []),
                name=condition.label, source="worldcover")
            provenance_entries.append({
                "source": f"ESA WorldCover classes {params.get('classes')}",
                "used_for": condition.parameters.get("class_name", "land cover")})
        else:
            return _execution(
                query, Status.UNSUPPORTED_CONDITION,
                f"The spatial condition '{condition.label}' is not supported by "
                f"composition; nothing was computed.")

        # back to the composition grid, then keep only the ROI
        if layer_grid is not grid:
            mask = crop_mask(mask, grid)
        mask = apply_roi(mask, inside)
        if condition.negate:
            mask = negated(mask)
        masks.append(mask)
        condition_masks[mask.name] = mask.state
        counts = state_counts(mask)
        condition_results.append({
            "kind": "spatial", "name": condition.name, "label": condition.label,
            "mask": mask, "negated": bool(condition.negate),
            "source": "ESA WorldCover 2021 v200",
            "negate": condition.negate, **counts,
            "provenance": dict(mask.provenance)})

    # ---- 6. spectral masks (Phase 11 engine, native, ROI window) ----------- #
    for condition in spectral:
        definition = get_index(condition.threshold.index)
        roles = {role: int(index_ctx.roles[role]) for role in definition.roles
                 if role in (index_ctx.roles or {})}
        if not set(definition.roles) <= set(roles):
            return _execution(
                query, Status.UNSUPPORTED,
                f"{definition.short_name} needs the "
                f"{', '.join(definition.roles)} band roles and only "
                f"{', '.join(roles) or 'none'} could be established.")
        try:
            with rasterio.open(index_ctx.path) as ds:
                win = rasterio.windows.from_bounds(
                    *_grid_bounds(grid), transform=ds.transform)
                col_off = max(0, int(round(win.col_off)))
                row_off = max(0, int(round(win.row_off)))
                w = max(1, min(int(round(win.width)), ds.width - col_off))
                h = max(1, min(int(round(win.height)), ds.height - row_off))
                spec = ReflectanceSpec(
                    scale=float(index_ctx.scale), offset=float(index_ctx.offset),
                    profile=index_ctx.profile,
                    source=index_ctx.reflectance_source, is_reflectance=True)
                idx = index_from_dataset(
                    ds, definition, roles, window=Window(col_off, row_off, w, h),
                    reflectance=spec, profile=index_ctx.profile,
                    provenance={"source": index_ctx.source_label})
        except Exception as exc:
            return _execution(query, Status.ERROR,
                              f"The {definition.short_name} raster could not be "
                              f"read for this area: {exc}")

        if tuple(np.asarray(idx.array).shape) != tuple(grid.shape):
            return _execution(
                query, Status.ERROR,
                f"The {definition.short_name} window "
                f"{np.asarray(idx.array).shape} does not match the composition "
                f"grid {grid.shape}; the two were not combined.")

        threshold = float(condition.threshold.value)
        mask = threshold_mask(
            idx.array, idx.mask, condition.threshold.operator, threshold, grid,
            name=condition.label,
            source=f"core.indices:{definition.name}",
            dataset=str(index_ctx.source_label),
            reflectance_scale=float(index_ctx.scale),
            reflectance_offset=float(index_ctx.offset))
        mask = apply_roi(mask, inside)
        if condition.negate:
            mask = negated(mask)
        masks.append(mask)
        condition_masks[mask.name] = mask.state
        threshold_provenance[condition.name] = condition.threshold.to_dict()
        source_analyses.append(f"core.indices:{definition.name}")
        source_resolutions[definition.name] = float(grid.resolution_m)
        provenance_entries.append({
            "source": f"{definition.short_name} from "
                      f"{index_ctx.source_label}",
            "used_for": f"{definition.short_name} threshold",
            "formula": definition.formula,
            "threshold": condition.threshold.to_dict()})
        condition_results.append({
            "kind": "spectral", "name": condition.name, "label": condition.label,
            "mask": mask, "negated": bool(condition.negate),
            "source": f"core.indices:{definition.name}",
            "operator": condition.threshold.operator,
            "threshold": threshold,
            "threshold_provenance": condition.threshold.to_dict(),
            "negate": condition.negate, **state_counts(mask),
            "provenance": dict(mask.provenance)})

    # ---- 7. temporal masks (Phase 10 classes) ------------------------------ #
    for condition in temporal:
        wanted = str(condition.parameters.get("class", "stable"))
        mask = temporal_class_mask(
            temporal_result.class_raster, grid, wanted=[wanted],
            name=condition.label,
            before_date=str(temporal_result.before_date or ""),
            after_date=str(temporal_result.after_date or ""),
            thresholds=dict(temporal_result.thresholds or {}))
        mask = apply_roi(mask, inside)
        if condition.negate:
            mask = negated(mask)
        masks.append(mask)
        condition_masks[mask.name] = mask.state
        source_resolutions["ndvi_change"] = float(grid.resolution_m)
        provenance_entries.append({
            "source": "analyses.ndvi_change",
            "used_for": "NDVI change class",
            "before_date": str(temporal_result.before_date or ""),
            "after_date": str(temporal_result.after_date or ""),
            "thresholds": dict(temporal_result.thresholds or {})})
        condition_results.append({
            "kind": "temporal", "name": condition.name, "label": condition.label,
            "mask": mask, "negated": bool(condition.negate),
            "source": "analyses.ndvi_change",
            "class": wanted,
            "before_date": str(temporal_result.before_date or ""),
            "after_date": str(temporal_result.after_date or ""),
            "negate": condition.negate, **state_counts(mask),
            "provenance": dict(mask.provenance)})

    if not masks:
        return _execution(query, Status.UNKNOWN,
                          "No measurable condition was found in that request.")

    # ---- 8. combine (three-valued, one verified grid) ---------------------- #
    try:
        combined = combine_conditions(masks, composed.operator)
    except ValueError as exc:                     # GridMismatchError subclasses it
        return _execution(query, Status.ERROR,
                          f"The conditions could not be combined: {exc}")

    counts = state_counts(combined)
    pixel_area = _pixel_area(grid.transform)
    analysed = counts["matched"] + counts["non_matching"]
    matched_area = counts["matched"] * pixel_area

    # ---- 9. index summaries requested as EVIDENCE over the matched cells --- #
    index_summaries: Dict[str, Dict[str, Any]] = {}
    for name in composed.summary_requests:
        try:
            definition = get_index(name)
            roles = {role: int(index_ctx.roles[role]) for role in definition.roles
                     if role in (index_ctx.roles or {})}
            if not set(definition.roles) <= set(roles) or index_ctx is None:
                index_summaries[name] = {
                    "defined": False,
                    "reason": f"the {definition.short_name} band roles are not "
                              f"available for this raster"}
                continue
            with rasterio.open(index_ctx.path) as ds:
                win = rasterio.windows.from_bounds(
                    *_grid_bounds(grid), transform=ds.transform)
                col_off = max(0, int(round(win.col_off)))
                row_off = max(0, int(round(win.row_off)))
                w = max(1, min(int(round(win.width)), ds.width - col_off))
                h = max(1, min(int(round(win.height)), ds.height - row_off))
                idx = index_from_dataset(
                    ds, definition, roles, window=Window(col_off, row_off, w, h),
                    reflectance=ReflectanceSpec(
                        scale=float(index_ctx.scale),
                        offset=float(index_ctx.offset),
                        profile=index_ctx.profile,
                        source=index_ctx.reflectance_source, is_reflectance=True))
            matched_cells = np.asarray(combined.match) & np.asarray(combined.valid)
            summary = summarise_index(idx.array, idx.mask, matched_cells)
            summary["computed_over"] = "cells matching the combined condition"
            summary["index"] = name
            index_summaries[name] = summary
            source_analyses.append(f"core.indices:{definition.name}")
        except Exception as exc:                   # evidence must never break the run
            index_summaries[name] = {"defined": False, "reason": str(exc)[:160]}

    # ---- 10. the result ---------------------------------------------------- #
    if analysed == 0:
        status = RESULT_INSUFFICIENT_DATA
        message = ("No cell of the selected area could be decided: every "
                   "condition was insufficient for every cell, so this is "
                   "UNKNOWN, not 'no matches'.")
    elif counts["matched"] == 0:
        status = RESULT_ZERO_MATCHES
        message = (f"No cells match {composed.expression} "
                   f"({counts['non_matching']:,} cells were measured and did "
                   f"not match, {counts['insufficient']:,} could not be "
                   f"decided).")
    else:
        status = RESULT_OK
        area = matched_area / 1.0e6
        message = (f"{counts['matched']:,} cells ({area:.3f} km²) match "
                   f"{composed.expression} over {analysed:,} measured cells "
                   f"({counts['insufficient']:,} undecided).")

    result = MultiConditionResult(
        query=query.original_query,
        normalized_query=composed.normalized_query,
        conditions=conditions,
        operator=composed.operator,
        status=status,
        message=message,
        combined_mask=np.asarray(combined.state),
        combined_gridmask=combined,
        condition_masks=condition_masks,
        grid={"crs": str(grid.crs), "transform": list(grid.transform)[:6],
              "width": int(grid.width), "height": int(grid.height),
              "resolution_m": float(grid.resolution_m)},
        roi_mask=inside,
        matched_cell_count=counts["matched"],
        non_matching_cell_count=counts["non_matching"],
        insufficient_cell_count=counts["insufficient"],
        analysed_cell_count=analysed,
        matched_area_m2=float(matched_area),
        matched_fraction=(counts["matched"] / analysed) if analysed else 0.0,
        insufficient_fraction=(counts["insufficient"] / counts["total"])
        if counts["total"] else 0.0,
        condition_results=condition_results,
        index_summaries=index_summaries,
        source_analyses=tuple(dict.fromkeys(source_analyses)),
        source_dates=source_dates,
        alignment=alignment,
        threshold_provenance=threshold_provenance,
        analysis_resolution=float(grid.resolution_m),
        source_resolutions=source_resolutions,
        # the boundary is stated once, at the end, and never duplicated
        limitations=CAVEAT_FIRST + tuple(
            item for item in (cfg.get("limitations") or ())
            if item.strip() != CAVEAT),
        warnings=tuple(warnings),
        provenance={
            "engine": __name__,
            "config_version": cfg.get("version", ""),
            "conditions": [c.to_dict() for c in conditions],
            "sources": provenance_entries,
            "thresholds": threshold_provenance,
            "alignment": alignment,
            "router": {
                "intent": query.intent.value,
                "confidence": round(float(query.confidence), 3),
                "matched": list(query.matched),
                "explanation": query.explanation,
            },
        },
        performance={
            "runtime_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "grid_cells": int(grid.width) * int(grid.height),
        },
    )
    return _execution(query, Status.OK, message, result=result,
                      warnings=result.warnings, provenance=result.provenance)


__all__ = [
    "MultiConditionResult",
    "run_multi_condition",
    "RESULT_OK",
    "RESULT_ZERO_MATCHES",
    "RESULT_INSUFFICIENT_DATA",
    "CAVEAT",
]
