"""Phase 10 -- the temporal NDVI change engine.

WHAT THIS ENGINE DOES
---------------------
    one ROI + two dated acquisitions
        -> validate the pair            (core.temporal.validate_pair)
        -> agree on a grid              (core.temporal.plan_alignment)
        -> NDVI for each date           (core.indices.compute_ndvi -- the SAME
                                         function the single-date analysis uses,
                                         so the two engines cannot diverge)
        -> delta = after - before, ONLY where both are valid
        -> classify with configurable thresholds
        -> NDVIChangeResult + provenance

WHAT THIS ENGINE REFUSES TO DO
------------------------------
* **Subtract arrays on different grids.** Alignment is explicit and recorded;
  the identical-grid case is detected, not assumed.
* **Invent a date.** A missing or ambiguous date returns NEEDS_TWO_DATES. The
  engine never substitutes "the nearest available acquisition".
* **Report on an area one scene does not cover.** That is a coverage failure,
  not a small number.
* **Claim a cause.** Every message is a statement about the *index*: "vegetation
  index decreased". Never "the crop failed", "this was flooded", "deforestation".
  Those are questions for other data, and Phase 10 has none of it.

CAUTIOUS BY CONSTRUCTION
------------------------
If the valid-pixel fraction inside the ROI falls below the configured minimum,
the engine returns INSUFFICIENT_DATA **without statistics**. Reporting a mean
computed on 12% of the area would look like an answer while being a biased one.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform

from core.alignment import AnalysisGrid, native_roi_grid, roi_mask
from core.indices import compute_ndvi
from core.reflectance import ReflectanceSpec
from core.router import Intent, QueryIntent
from core.temporal import (
    RESAMPLING_NOTE,
    AlignmentRecord,
    ScenePair,
    SceneRef,
    load_temporal_config,
    plan_alignment,
    reproject_geometry,
    validate_pair,
)

from .base import AnalysisContext, AnalysisExecution, Status

# --------------------------------------------------------------------------- #
# change classes
# --------------------------------------------------------------------------- #
#: Codes stored in `class_raster`. Insufficient data is 0 and is NEVER counted
#: as "stable": an unknown is not a zero.
CHANGE_INSUFFICIENT = 0
CHANGE_DECREASE = 1
CHANGE_STABLE = 2
CHANGE_INCREASE = 3

CHANGE_CLASS_NAMES = {
    CHANGE_INCREASE: "Increase",
    CHANGE_STABLE: "Stable",
    CHANGE_DECREASE: "Decrease",
    CHANGE_INSUFFICIENT: "Insufficient data",
}


# --------------------------------------------------------------------------- #
# result
# --------------------------------------------------------------------------- #
@dataclass
class NDVIChangeResult:
    """The structured outcome of one before/after NDVI comparison."""

    before_date: Optional[str] = None
    after_date: Optional[str] = None
    before_scene: str = ""
    after_scene: str = ""

    valid_pixel_count: int = 0
    roi_cell_count: int = 0

    before_mean: Optional[float] = None
    after_mean: Optional[float] = None
    delta_mean: Optional[float] = None
    before_median: Optional[float] = None
    after_median: Optional[float] = None
    delta_median: Optional[float] = None
    before_std: Optional[float] = None
    after_std: Optional[float] = None
    delta_std: Optional[float] = None
    delta_min: Optional[float] = None
    delta_max: Optional[float] = None

    increased_count: int = 0
    decreased_count: int = 0
    stable_count: int = 0
    insufficient_count: int = 0

    change_raster: Any = None          # dNDVI float32, NaN where not comparable
    class_raster: Any = None           # int8 class codes (see CHANGE_* above)
    before_raster: Any = None          # NDVI before, NaN where invalid
    after_raster: Any = None           # NDVI after,  NaN where invalid
    roi_mask: Any = None               # True where a cell was INSIDE the selection

    crs: Any = None
    transform: Any = None
    resolution: Optional[float] = None

    thresholds: Dict[str, Any] = field(default_factory=dict)
    alignment: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    limitations: Tuple[str, ...] = ()

    # -- derived ----------------------------------------------------------- #
    @property
    def compared_count(self) -> int:
        return self.increased_count + self.decreased_count + self.stable_count

    def fraction(self, count: int) -> float:
        """Share of the COMPARED pixels -- not of the ROI, which also holds
        pixels that could not be compared."""
        return float(count) / self.compared_count if self.compared_count else 0.0

    @property
    def valid_fraction_of_roi(self) -> float:
        return float(self.valid_pixel_count) / self.roi_cell_count if self.roi_cell_count else 0.0

    @property
    def net_direction(self) -> str:
        """A one-word, non-causal description of the balance of change."""
        if self.delta_mean is None or self.compared_count == 0:
            return "unknown"
        if self.increased_count == 0 and self.decreased_count == 0:
            return "stable"
        if self.increased_count > self.decreased_count * 1.1:
            return "increase"
        if self.decreased_count > self.increased_count * 1.1:
            return "decrease"
        return "mixed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before_date": self.before_date,
            "after_date": self.after_date,
            "before_scene": self.before_scene,
            "after_scene": self.after_scene,
            "valid_pixel_count": self.valid_pixel_count,
            "roi_cell_count": self.roi_cell_count,
            "valid_fraction_of_roi": round(self.valid_fraction_of_roi, 6),
            "before_mean": self.before_mean,
            "after_mean": self.after_mean,
            "delta_mean": self.delta_mean,
            "before_median": self.before_median,
            "after_median": self.after_median,
            "delta_median": self.delta_median,
            "before_std": self.before_std,
            "after_std": self.after_std,
            "delta_std": self.delta_std,
            "delta_min": self.delta_min,
            "delta_max": self.delta_max,
            "increased_count": self.increased_count,
            "decreased_count": self.decreased_count,
            "stable_count": self.stable_count,
            "insufficient_count": self.insufficient_count,
            "increased_pct": round(self.fraction(self.increased_count) * 100, 2),
            "decreased_pct": round(self.fraction(self.decreased_count) * 100, 2),
            "stable_pct": round(self.fraction(self.stable_count) * 100, 2),
            "net_direction": self.net_direction,
            "resolution": self.resolution,
            "thresholds": dict(self.thresholds),
            "alignment": dict(self.alignment),
            "provenance": dict(self.provenance),
            "limitations": list(self.limitations),
        }


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _native_roi_grid(scene: SceneRef, geometry: Any, buffer_cells: int = 2) -> AnalysisGrid:
    """An AnalysisGrid snapped to the scene's NATIVE grid, covering the ROI.

    Phase 11: delegates to `core.alignment.native_roi_grid`, which is now the
    single implementation of "cut the ROI window on the native grid". Behaviour
    is unchanged -- the 44 Phase 10 tests are the gate.
    """
    return native_roi_grid(
        scene.transform, scene.width, scene.height, scene.resolution,
        scene.crs, geometry, buffer_cells=buffer_cells,
    )


def _read_bands(scene: SceneRef,
                grid: AnalysisGrid,
                identical: bool,
                resampling: str = "nearest") -> Tuple[np.ndarray, np.ndarray]:
    """Read this scene's red and NIR bands onto `grid`, as float32 DN.

    * `identical`  -> a plain windowed read. No resampling: the grid IS the
      scene's native grid, so the returned pixels are sensor values.
    * otherwise    -> a WarpedVRT read onto the common grid.
    """
    from core.alignment import read_into_grid

    method = Resampling.nearest if str(resampling).lower() == "nearest" else Resampling.bilinear
    if identical:
        with rasterio.open(scene.path) as ds:
            win = rasterio.windows.from_bounds(*grid.bounds, transform=scene.transform)
            col_off = max(0, int(round(win.col_off)))
            row_off = max(0, int(round(win.row_off)))
            width = max(1, min(int(round(win.width)), scene.width - col_off))
            height = max(1, min(int(round(win.height)), scene.height - row_off))
            window = Window(col_off, row_off, width, height)
            red = ds.read(scene.red_index, window=window).astype("float32")
            nir = ds.read(scene.nir_index, window=window).astype("float32")
        return red, nir

    red = read_into_grid(scene.path, grid, method, band=scene.red_index,
                         src_nodata=scene.nodata, dtype="float32")
    nir = read_into_grid(scene.path, grid, method, band=scene.nir_index,
                         src_nodata=scene.nodata, dtype="float32")
    return red, nir


# --------------------------------------------------------------------------- #
# the comparison itself
# --------------------------------------------------------------------------- #
def compare_ndvi(pair: ScenePair,
                 geometry: Any,
                 geometry_crs: Any = None,
                 config: Optional[Dict[str, Any]] = None
                 ) -> Tuple[Optional[NDVIChangeResult], Status, str, Tuple[str, ...]]:
    """Compare NDVI between two dated acquisitions over one ROI.

    Returns `(result, status, message, warnings)`. `result` is not None only
    when `status is Status.OK`. Everything else is a reason, in words, plus the
    numbers that justify it where there are any.

    This function is Streamlit-free and side-effect free: the unit tests call it
    directly with synthetic SceneRefs.
    """
    cfg = config or load_temporal_config()
    ndvi_cfg = cfg.get("ndvi", {}) or {}
    chg_cfg = cfg.get("change", {}) or {}

    status, message, warnings = validate_pair(pair, geometry, geometry_crs, cfg)
    if status.value != "OK":
        return None, Status(status.value), message, warnings

    before, after = pair.before, pair.after
    warnings = list(warnings)
    started = time.perf_counter()

    # -- 1. agree on a grid ------------------------------------------------- #
    alignment: AlignmentRecord = plan_alignment(before, after, geometry, cfg)
    identical = alignment.method == "identical_grid"

    if identical:
        grid = _native_roi_grid(before, geometry,
                                buffer_cells=int((cfg.get("alignment", {}) or {}).get("buffer_cells", 2)))
        # Both scenes share one transform, so this window is valid for both.
        grid_geometry = geometry
    else:
        from affine import Affine

        grid = AnalysisGrid(
            crs=alignment.crs,
            transform=Affine(*alignment.transform),
            width=alignment.width,
            height=alignment.height,
            resolution=float(alignment.resolution),
            requested_resolution=float(alignment.resolution),
            note=alignment.note,
        )
        grid_geometry = geometry
        if geometry_crs is not None:
            try:
                from rasterio.crs import CRS

                if CRS.from_user_input(geometry_crs) != CRS.from_user_input(grid.crs):
                    grid_geometry = reproject_geometry(geometry, geometry_crs, grid.crs)
            except Exception:
                pass

    if grid.cells > int((cfg.get("alignment", {}) or {}).get("max_cells", 2_000_000)):
        return (None, Status.INSUFFICIENT_DATA,
                "The selected area is too large to compare at this resolution. "
                "Draw a smaller area.", tuple(warnings))

    # -- 2. NDVI for each date (same function as the single-date engine) ----- #
    min_denominator = float(ndvi_cfg.get("min_denominator", 1e-6))
    before_ndvi, after_ndvi = None, None
    for role, scene in (("before", before), ("after", after)):
        red, nir = _read_bands(scene, grid, identical,
                               resampling=str(alignment.resampling or "nearest"))
        spec = ReflectanceSpec(scale=float(scene.scale), offset=float(scene.offset),
                               source="scene_provenance", is_reflectance=True,
                               verified=True,
                               evidence=(f"scale={scene.scale:g} recorded in the scene provenance",))
        if red.shape != nir.shape:                      # cannot happen; be explicit
            return (None, Status.ERROR,
                    f"The red and near-infrared bands of the {role} scene could not be "
                    f"read onto the same grid.", tuple(warnings))
        result = compute_ndvi(red, nir,
                              transform=grid.transform, crs=grid.crs,
                              red_nodata=scene.nodata, nir_nodata=scene.nodata,
                              reflectance=spec, min_denominator=min_denominator,
                              red_label=f"{scene.label} red (band {scene.red_index})",
                              nir_label=f"{scene.label} nir (band {scene.nir_index})")
        if role == "before":
            before_ndvi = result
        else:
            after_ndvi = result

    # -- 3. the ROI, expressed on this grid --------------------------------- #
    roi = roi_mask(grid_geometry, grid)
    roi_cell_count = int(roi.sum())
    if roi_cell_count == 0:
        return (None, Status.NO_VALID_PIXELS,
                "The selected area contains no analysis cell at this resolution.",
                tuple(warnings))

    # -- 4. change ONLY where both dates are valid --------------------------- #
    valid = np.asarray(before_ndvi.mask) & np.asarray(after_ndvi.mask) & roi
    valid_pixel_count = int(valid.sum())

    if valid_pixel_count == 0:
        return (None, Status.NO_VALID_PIXELS,
                "No pixel inside the selected area has a valid NDVI value in BOTH "
                "acquisitions, so no change can be measured.", tuple(warnings))

    min_fraction = float(chg_cfg.get("min_valid_fraction", 0.60))
    min_pixels = int(chg_cfg.get("min_valid_pixels", 4))
    fraction = valid_pixel_count / roi_cell_count

    # TWO separate refusals, each explained by the rule it actually broke --
    # a coverage shortfall and a sample-size shortfall are not the same thing.
    if fraction < min_fraction:
        return (None, Status.INSUFFICIENT_DATA,
                f"Only {fraction * 100:.1f}% of the selected area has a valid NDVI "
                f"value in both acquisitions ({valid_pixel_count:,} of "
                f"{roi_cell_count:,} cells); at least {min_fraction * 100:.0f}% is "
                f"required. No change statistics are reported for a partial area.",
                tuple(warnings))
    if valid_pixel_count < min_pixels:
        return (None, Status.INSUFFICIENT_DATA,
                f"Only {valid_pixel_count:,} comparable cell(s) inside the selected "
                f"area hold a valid NDVI value on both dates; at least {min_pixels} "
                f"are required before any statistics are reported.",
                tuple(warnings))

    b = np.asarray(before_ndvi.array, dtype="float32")
    a = np.asarray(after_ndvi.array, dtype="float32")
    delta = np.where(valid, a - b, np.nan).astype("float32")

    # -- 5. classify --------------------------------------------------------- #
    inc_t = float(chg_cfg.get("increase_threshold", 0.10))
    dec_t = float(chg_cfg.get("decrease_threshold", 0.10))
    dv = delta[valid]
    classes = np.zeros(delta.shape, dtype="int8")          # 0 = insufficient
    classes[valid & (delta >= inc_t)] = CHANGE_INCREASE
    classes[valid & (delta <= -dec_t)] = CHANGE_DECREASE
    classes[valid & (delta > -dec_t) & (delta < inc_t)] = CHANGE_STABLE

    increased = int(np.count_nonzero(classes == CHANGE_INCREASE))
    decreased = int(np.count_nonzero(classes == CHANGE_DECREASE))
    stable = int(np.count_nonzero(classes == CHANGE_STABLE))
    insufficient = int(roi_cell_count - valid_pixel_count)

    # -- 6. provenance -------------------------------------------------------- #
    provenance = {
        "engine": __name__,
        "config_version": cfg.get("version", "unknown"),
        "formula_ndvi": ndvi_cfg.get("formula", "(NIR - RED) / (NIR + RED)"),
        "formula_delta": chg_cfg.get("delta_formula", "NDVI_after - NDVI_before"),
        "reflectance": {
            "before": {"scale": before.scale, "offset": before.offset,
                       "source": "scene provenance"},
            "after": {"scale": after.scale, "offset": after.offset,
                      "source": "scene provenance"},
        },
        "bands": {
            "before": {"red": before.red_index, "nir": before.nir_index,
                       "names": list(before.band_names)},
            "after": {"red": after.red_index, "nir": after.nir_index,
                      "names": list(after.band_names)},
        },
        "scenes": {"before": before.to_dict(), "after": after.to_dict()},
        "roi_cells": roi_cell_count,
        "valid_fraction_of_roi": round(fraction, 6),
        "runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "resampling_adds_no_information": alignment.resampled,
    }

    result = NDVIChangeResult(
        before_date=before.date.isoformat() if before.date else None,
        after_date=after.date.isoformat() if after.date else None,
        before_scene=before.label,
        after_scene=after.label,
        valid_pixel_count=valid_pixel_count,
        roi_cell_count=roi_cell_count,
        before_mean=float(np.mean(b[valid])),
        after_mean=float(np.mean(a[valid])),
        delta_mean=float(np.mean(dv)),
        before_median=float(np.median(b[valid])),
        after_median=float(np.median(a[valid])),
        delta_median=float(np.median(dv)),
        before_std=float(np.std(b[valid])),
        after_std=float(np.std(a[valid])),
        delta_std=float(np.std(dv)),
        delta_min=float(np.min(dv)),
        delta_max=float(np.max(dv)),
        increased_count=increased,
        decreased_count=decreased,
        stable_count=stable,
        insufficient_count=insufficient,
        change_raster=delta,
        class_raster=classes,
        before_raster=np.where(valid, b, np.nan).astype("float32"),
        after_raster=np.where(valid, a, np.nan).astype("float32"),
        roi_mask=roi,
        crs=grid.crs,
        transform=grid.transform,
        resolution=float(grid.resolution),
        thresholds={
            "increase": inc_t,
            "decrease": dec_t,
            "rationale": str(chg_cfg.get("threshold_rationale", "")).strip(),
            "min_valid_fraction": min_fraction,
            "min_valid_pixels": min_pixels,
        },
        alignment=alignment.to_dict(),
        provenance=provenance,
        limitations=tuple(str(x) for x in (cfg.get("limitations", []) or [])),
    )
    if alignment.resampled:
        warnings.append(RESAMPLING_NOTE)
    return result, Status.OK, "", tuple(warnings)


# --------------------------------------------------------------------------- #
# wording
# --------------------------------------------------------------------------- #
def compose_message(result: NDVIChangeResult,
                    config: Optional[Dict[str, Any]] = None) -> str:
    """The textual summary shown next to the map.

    Every sentence is a statement about the INDEX. The causal caveat is not
    optional and is always appended: an NDVI difference is evidence of a
    difference, never of a reason.
    """
    cfg = config or load_temporal_config()
    wording = cfg.get("causal_wording", {}) or {}
    d = result.to_dict()
    lines = [
        f"NDVI compared for the selected area between {result.before_date} "
        f"({result.before_scene}) and {result.after_date} ({result.after_scene}).",
        f"Mean NDVI moved from {result.before_mean:+.4f} to {result.after_mean:+.4f} "
        f"(mean change {result.delta_mean:+.4f}, median {result.delta_median:+.4f}) "
        f"over {result.valid_pixel_count:,} comparable cells "
        f"({d['valid_fraction_of_roi'] * 100:.1f}% of the selected area).",
        f"Change classes at the ±{result.thresholds.get('increase', 0.1):g} display "
        f"threshold: increase {d['increased_pct']:.1f}%, stable {d['stable_pct']:.1f}%, "
        f"decrease {d['decreased_pct']:.1f}%.",
    ]
    direction = result.net_direction
    if direction == "increase":
        lines.append(str(wording.get("increase", "Vegetation index increased.")))
    elif direction == "decrease":
        lines.append(str(wording.get("decrease", "Vegetation index decreased.")))
        lines.append(str(wording.get("decrease_interpretation",
                                     "The observed change is consistent with reduced "
                                     "vegetation signal.")))
    else:
        lines.append(str(wording.get("stable",
                                     "Vegetation index showed no consistent change "
                                     "above the display threshold.")))
    lines.append(str(wording.get("cause_caveat",
                                 "Additional data is required to identify the cause.")))
    if result.alignment.get("resampled"):
        lines.append(RESAMPLING_NOTE)
    return "\n\n".join(line for line in lines if line)


# --------------------------------------------------------------------------- #
# engine entry point (called by analyses.registry.route)
# --------------------------------------------------------------------------- #
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


def run_ndvi_change(context: AnalysisContext, query: QueryIntent) -> AnalysisExecution:
    """Phase 10 entry point: `NDVI_CHANGE_ROI` / `TEMPORAL_COMPARISON`."""
    roi = getattr(context, "roi", None)
    if roi is None or not getattr(roi, "usable", False):
        return _execution(query, Status.NEEDS_ROI,
                          "Please select an area on the map first.")

    pair = getattr(context, "temporal_pair", None)
    if pair is None or not getattr(pair, "complete", False):
        return _execution(query, Status.NEEDS_TWO_DATES,
                          "Two acquisitions are required. Choose a before scene and "
                          "an after scene (dates are never selected automatically).")

    geometry = getattr(roi, "geometry_raster_crs", None) or getattr(roi, "geometry", None)
    if geometry is None:
        return _execution(query, Status.NEEDS_ROI,
                          "The selected area has no usable geometry. Draw it again.")

    geometry_crs = getattr(roi, "raster_crs", None) or None
    try:
        result, status, message, warnings = compare_ndvi(
            pair, geometry, geometry_crs=geometry_crs
        )
    except Exception as exc:                      # never let a stack trace reach the UI
        return _execution(query, Status.ERROR,
                          f"The comparison could not be computed: {exc}")

    if result is None or status is not Status.OK:
        return _execution(query, status, message, warnings=warnings)

    return _execution(
        query, Status.OK, compose_message(result),
        warnings=warnings, result=result,
        provenance={"engine": __name__,
                    "intent": query.intent.value,
                    **{k: v for k, v in result.provenance.items() if k != "engine"}},
    )


__all__ = [
    "NDVIChangeResult",
    "compare_ndvi",
    "compose_message",
    "run_ndvi_change",
    "CHANGE_INCREASE", "CHANGE_STABLE", "CHANGE_DECREASE", "CHANGE_INSUFFICIENT",
    "CHANGE_CLASS_NAMES",
]
