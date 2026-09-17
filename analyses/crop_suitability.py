"""Phase 8 -- the CROP_SUITABILITY engine (registered with the Phase 7 router).

    QUERY -> Intent.CROP_SUITABILITY -> this module -> CropSuitabilityScreening

WHAT IT PRODUCES
    An EXPERIMENTAL CROP-SUITABILITY SCREENING for cotton, and nothing else:
    not a recommendation, not a yield estimate, not a soil diagnosis and not a
    statement about irrigation availability.

HOW IT WORKS
    ROI -> native analysis grid -> factor layers (windowed, cached, provenance)
        -> per-cell scoring  -> suitability raster (map)
        -> ROI-level summary -> score, class, limiting factors (answer)

    Both scenarios (rainfed and irrigation-assumed) reuse the SAME factor
    layers: the second one only changes how the water factor is treated.

Every sentence in the answer is generated from computed values. Nothing is
hard-coded, and a missing factor is reported as a limitation, never as a
computed limitation (brief item #19).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.alignment import AnalysisGrid, make_grid, roi_mask
from core.datasources import copernicus_dem, soilgrids, worldclim, worldcover
from core.datasources.base import stage_totals   # brief item #27
from core.suitability import (
    SuitabilityResult,
    assess_suitability,
    load_crop_config,
    score_layer_stack,
)
from core.router import Intent, QueryIntent

from .base import AnalysisContext, AnalysisExecution, Status

__all__ = [
    "CropSuitabilityScreening",
    "run_crop_suitability",
    "detect_crop",
    "SUPPORTED_CROPS",
    "OTHER_CROP_MESSAGE",
    "NO_CROP_MESSAGE",
    "SCREENING_DISCLAIMER",
]

SUPPORTED_CROPS = ("cotton",)

#: Crop words we can name but not screen. Anything here routes to a refusal
#: instead of being quietly treated as cotton.
_OTHER_CROPS: Tuple[str, ...] = (
    "rice", "wheat", "maize", "corn", "sugarcane", "sugar cane", "soybean",
    "soya", "barley", "millet", "sorghum", "potato", "potatoes", "tomato",
    "groundnut", "peanut", "sunflower", "mustard", "lentil", "chickpea",
    "tea", "coffee", "banana", "mango", "grape", "grapes", "onion", "garlic",
    "pulses", "oat", "oats", "rye", "cottonseed",
)

OTHER_CROP_MESSAGE = "Only cotton suitability is currently supported."
NO_CROP_MESSAGE = ("Only cotton suitability is currently supported — try asking "
                   "“Can I grow cotton here?”.")

SCREENING_DISCLAIMER = (
    "This is an experimental screening of long-term, static public data. It is "
    "not a crop recommendation, a yield prediction, a soil diagnosis or an "
    "irrigation-availability statement, and it must be validated with local "
    "agronomic and field information before it means anything on the ground."
)


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class CropSuitabilityScreening:
    """Everything the UI needs. Arrays stay here; `to_dict()` excludes them."""

    crop: str = "cotton"
    scenarios: Dict[str, SuitabilityResult] = field(default_factory=dict)
    rasters: Dict[str, Any] = field(default_factory=dict)
    score_rasters: Dict[str, Any] = field(default_factory=dict)
    grid: Optional[AnalysisGrid] = None
    performance: Dict[str, Any] = field(default_factory=dict)
    land_cover: Dict[str, Any] = field(default_factory=dict)
    monthly_stats: Dict[int, Dict[str, Optional[float]]] = field(default_factory=dict)
    elevation_context: Dict[str, Any] = field(default_factory=dict)
    slope_context: Dict[str, Any] = field(default_factory=dict)
    growing_season_months: List[int] = field(default_factory=list)
    config_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view -- deliberately WITHOUT the raw rasters (#18)."""
        return {
            "crop": self.crop,
            "config_version": self.config_version,
            "scenarios": {k: v.to_dict() for k, v in self.scenarios.items()},
            "grid": self.grid.to_dict() if self.grid else None,
            "performance": self.performance,
            "land_cover": self.land_cover,
            "monthly_stats": {str(k): v for k, v in self.monthly_stats.items()},
            "elevation_context": self.elevation_context,
            "slope_context": self.slope_context,
            "growing_season_months": list(self.growing_season_months),
        }


# --------------------------------------------------------------------------- #
# crop detection
# --------------------------------------------------------------------------- #
def detect_crop(normalized_query: str) -> Tuple[Optional[str], Optional[str]]:
    """(crop_key, unsupported_crop_word). cotton -> ("cotton", None)."""
    text = (normalized_query or "").lower()
    if "cotton" in text:
        return "cotton", None
    for word in _OTHER_CROPS:
        if word in text:
            return None, word
    return None, None


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
def run_crop_suitability(context: AnalysisContext,
                         query: QueryIntent,
                         config_dir: str = "config/crops",
                         use_cache: bool = True,
                         progress: Optional[Callable[[str], None]] = None,
                         analysis_resolution: Optional[float] = None) -> AnalysisExecution:
    """Execute CROP_SUITABILITY for the ROI in `context`."""
    roi = context.roi
    if roi is None or not getattr(roi, "usable", False):
        return _execution(query, Status.NEEDS_ROI,
                          "Please select an area on the map first.")

    crop, other = detect_crop(query.normalized_query)
    if other:
        return _execution(query, Status.UNSUPPORTED_CROP, OTHER_CROP_MESSAGE)
    if crop is None:
        return _execution(query, Status.UNSUPPORTED_CROP, NO_CROP_MESSAGE)

    try:
        cfg = load_crop_config(crop, base_dir=config_dir)
    except FileNotFoundError:
        return _execution(query, Status.UNSUPPORTED_CROP,
                          f"No screening configuration exists for '{crop}'.")

    timings: Dict[str, float] = {}
    warnings: List[str] = []
    stage_totals(reset=True)          # brief item #27: per-stage accounting
    t_start = time.perf_counter()

    # ---- 1. analysis grid -------------------------------------------------- #
    from core.geometry import as_crs
    crs = as_crs(roi.raster_crs)
    requested = float(analysis_resolution or 30.0)
    grid = make_grid(roi.geometry_raster_crs, crs, requested_resolution=requested)
    timings["grid_seconds"] = time.perf_counter() - t_start
    if grid.note:
        warnings.append(grid.note)

    inside = roi_mask(roi.geometry_raster_crs, grid)
    if not inside.any():
        return _execution(query, Status.INSUFFICIENT_DATA,
                          "The selected area covers no analysis cell, so nothing "
                          "could be screened.")

    screening = CropSuitabilityScreening(crop=crop, grid=grid,
                                         config_version=str(cfg.get("config_version", "")))

    # ---- 2. factor layers (each failure degrades, none fabricates) --------- #
    def note(msg: str) -> None:
        if progress:
            try:
                progress(msg)
            except Exception:  # pragma: no cover
                pass

    layers: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []

    note("Fetching ESA WorldCover land cover (10 m)")
    lc = _safe(lambda: worldcover.fetch_land_cover(grid, use_cache=use_cache),
               timings, "landcover_seconds", warnings, "land cover")
    if lc is not None:
        records.append(lc.record.to_dict())
        policy = cfg.get("land_cover", {})
        hard_mask, summary = worldcover.land_cover_summary(lc.array, inside, policy)
        screening.land_cover = summary
        if summary.get("excluded_fraction", 0.0) > 0:
            warnings.append(
                f"{summary['excluded_fraction'] * 100:.1f}% of the area is a hard "
                f"land-cover constraint ({summary.get('dominant_class_name') or 'excluded classes'}) "
                "and is screened as Unsuitable, not scored.")
    else:
        hard_mask = np.zeros((grid.height, grid.width), dtype=bool)
        warnings.append("Land cover could not be retrieved, so no hard land-cover "
                        "constraint was applied.")

    note("Fetching ISRIC SoilGrids pH and texture (250 m)")
    soil = _safe(lambda: soilgrids.fetch_texture_and_ph(grid, use_cache=use_cache),
                 timings, "soil_seconds", warnings, "soil")
    if soil is not None:
        records.extend(r.to_dict() for r in soil.get("records", []))
        for key in ("phh2o", "clay_pct", "sand_pct", "silt_pct"):
            if soil.get(key) is not None:
                layers[key] = soil[key]
        soil["records"] = []

    season = [int(m) for m in cfg.get("growing_season", {}).get("months", [])]
    screening.growing_season_months = season

    note("Fetching WorldClim monthly climatologies (1970-2000)")
    climate = _safe(
        lambda: worldclim.fetch_climate(
            grid, season, cfg["factors"]["growing_season_temperature"],
            inside=inside, use_cache=use_cache, progress=progress),
        timings, "climate_seconds", warnings, "climate")
    if climate is not None:
        records.extend(r.to_dict() for r in climate.get("records", []))
        layers["growing_season_precipitation"] = climate["growing_season_precipitation"]
        layers["growing_season_temperature"] = climate["growing_season_temperature"]
        screening.monthly_stats = climate.get("monthly_stats", {})
        warnings.extend(climate.get("warnings", []))
        timings["climate_precip_seconds"] = climate.get("timings", {}).get("precipitation", 0.0)
        timings["climate_temperature_seconds"] = climate.get("timings", {}).get("temperature", 0.0)
        climate["records"] = []

    note("Fetching Copernicus DEM and computing slope (30 m)")
    terrain = _safe(lambda: copernicus_dem.fetch_slope(grid, use_cache=use_cache,
                                                       inside=inside),
                    timings, "terrain_seconds", warnings, "terrain")
    if terrain is not None:
        layers["terrain_slope"] = terrain["slope_pct"]
        screening.elevation_context = terrain.get("elevation_context", {})
        screening.slope_context = terrain.get("slope_context", {})
        if terrain.get("record") is not None:
            records.append(terrain["record"].to_dict())
        terrain["slope_pct"] = None

    # ---- 3. ROI-level values (means over the ROI, factors first) ----------- #
    t_factors = time.perf_counter()
    def roi_mean(arr: Any) -> Optional[float]:
        if arr is None:
            return None
        a = np.asarray(arr, dtype="float64")
        vals = a[inside & np.isfinite(a)]
        return float(vals.mean()) if vals.size else None

    roi_values: Dict[str, Optional[float]] = {
        "growing_season_precipitation": roi_mean(layers.get("growing_season_precipitation")),
        "growing_season_temperature": roi_mean(layers.get("growing_season_temperature")),
        "soil_reaction": roi_mean(layers.get("phh2o")),
        "soil_texture": roi_mean(layers.get("clay_pct")),       # presence check
        "sand_pct": roi_mean(layers.get("sand_pct")),
        "silt_pct": roi_mean(layers.get("silt_pct")),
        "clay_pct": roi_mean(layers.get("clay_pct")),
        "terrain_slope": roi_mean(layers.get("terrain_slope")),
    }
    if roi_values["soil_texture"] is None:
        roi_values["soil_texture"] = None

    annual_precip = roi_mean(
        climate.get("annual_precipitation") if climate else None)

    # depth/salinity/drainage caveat attached to flat + heavy texture
    if (roi_values.get("terrain_slope") is not None and roi_values["terrain_slope"] < 1.0
            and (roi_values.get("clay_pct") or 0) >= 30):
        warnings.append(
            "The area is very flat with a clay-rich topsoil, so natural drainage "
            "may be poor. Drainage was not measured; this is a risk flag derived "
            "from slope and texture, not a scored factor.")

    # ---- 4. score both scenarios ------------------------------------------ #
    scenarios_cfg = cfg.get("scenarios", {})
    precip_context = {
        "annual_precipitation": annual_precip,
        "growing_season_precipitation": roi_values.get("growing_season_precipitation"),
        "months": season,
        "approx_gdd": (climate or {}).get("approx_gdd"),
    }

    timings["factor_seconds"] = time.perf_counter() - t_factors
    t_score = time.perf_counter()

    for name, scfg in scenarios_cfg.items():
        treatment = str(scfg.get("water_treatment", "scored"))
        stack = score_layer_stack(cfg, layers, hard_mask=hard_mask,
                                  water_treatment=treatment, inside=inside)
        result = assess_suitability(
            cfg, name, roi_values,
            scenario_cfg=scfg,
            landcover=screening.land_cover if name == "rainfed" else None,
            precipitation_context=precip_context,
            native_resolutions={
                "land_cover": "10 m", "soil": "250 m", "climate": "~1 km",
                "topography": "30 m"},
            analysis_resolution=grid.resolution,
            requested_resolution=grid.requested_resolution,
            effective_resolution_note=grid.note,
            provenance=records,
        )
        result.valid_fraction = stack["valid_fraction"]
        result.class_fractions = stack["class_fractions"]
        result.suitability_raster = stack["class_codes"]
        result.score_raster = stack["score"]
        result.raster_transform = grid.transform
        result.raster_crs = grid.crs
        screening.scenarios[name] = result
        screening.rasters[name] = stack["class_codes"]
        screening.score_rasters[name] = stack["score"]

    timings["suitability_seconds"] = time.perf_counter() - t_score
    stages = stage_totals()
    timings["source_access_seconds"] = stages["open"]
    timings["windowed_read_seconds"] = stages["read"]
    timings["cache_read_seconds"] = stages["cache"]
    timings["total_seconds"] = time.perf_counter() - t_start
    screening.performance = {k: round(v, 3) for k, v in timings.items()}
    screening.performance["cells"] = grid.cells
    screening.performance["analysis_resolution_m"] = grid.resolution
    for key in ("landcover_seconds", "soil_seconds", "climate_seconds", "terrain_seconds"):
        timings.setdefault(key, 0.0)

    primary = screening.scenarios.get("rainfed") or next(iter(screening.scenarios.values()))
    message = compose_answer(cfg, screening, primary)

    status = Status.OK if primary.status == "OK" else (
        Status.PARTIAL_DATA if primary.status == "PARTIAL_DATA" else Status.INSUFFICIENT_DATA)

    all_warnings = list(dict.fromkeys(list(warnings) + list(primary.warnings)))
    all_warnings.append(SCREENING_DISCLAIMER)

    return AnalysisExecution(
        intent=Intent.CROP_SUITABILITY, status=status,
        query=query.original_query, normalized_query=query.normalized_query,
        confidence=query.confidence, explanation=query.explanation,
        matched=query.matched, result=screening,
        message=message, warnings=tuple(all_warnings),
        provenance={
            "engine": __name__,
            "crop": crop,
            "config": f"config/crops/{crop}.yml (v{cfg.get('config_version')})",
            "grid": grid.to_dict(),
            "datasets": records,
            "performance": screening.performance,
            "router": {"intent": Intent.CROP_SUITABILITY.value,
                       "confidence": round(float(query.confidence), 3),
                       "matched": list(query.matched)},
        },
    )


def _safe(fn: Callable[[], Any], timings: Dict[str, float], key: str,
          warnings: List[str], label: str) -> Optional[Any]:
    """Run a fetch; on failure record a warning and return None (never fabricate)."""
    t0 = time.perf_counter()
    try:
        return fn()
    except Exception as exc:  # pragma: no cover -- network dependent
        warnings.append(f"The {label} layer could not be retrieved "
                        f"({type(exc).__name__}: {exc}). It is reported as "
                        "missing data, not as a value.")
        return None
    finally:
        timings[key] = time.perf_counter() - t0


def _execution(query: QueryIntent, status: Status, message: str) -> AnalysisExecution:
    return AnalysisExecution(
        intent=Intent.CROP_SUITABILITY, status=status, query=query.original_query,
        normalized_query=query.normalized_query, confidence=query.confidence,
        explanation=query.explanation, matched=query.matched,
        message=message, result=None, warnings=(SCREENING_DISCLAIMER,),
        provenance={"engine": __name__},
    )


# --------------------------------------------------------------------------- #
# answer composition -- generated from computed evidence only
# --------------------------------------------------------------------------- #
def compose_answer(cfg: Dict[str, Any],
                   screening: CropSuitabilityScreening,
                   res: SuitabilityResult) -> str:
    """Build the chat answer. Every sentence comes from a computed value."""
    scfg = cfg.get("scenarios", {}).get(res.scenario, {})
    lines: List[str] = []

    lines.append(f"**Experimental crop-suitability screening — cotton · "
                 f"{scfg.get('label', res.scenario)}**")

    if res.score is None:
        lines.append("**Insufficient data** — no score was computed.")
        for w in res.warnings[:3]:
            lines.append(f"- {w}")
        lines.append("_No value is reported rather than a defaulted one._")
        return "\n".join(lines)

    lines.append(f"**{res.classification}** · score {res.score:.2f} (0–1) · "
                 f"confidence **{res.confidence}**")

    if scfg.get("hypothetical"):
        lines.append("_" + str(scfg.get("required_wording", "")).strip() + "_")

    # limiting factors (computed only)
    if res.computed_limiting_factors:
        top = res.computed_limiting_factors[0]
        lines.append(
            f"**Strongest computed limitation:** {top['label'].lower()} — "
            f"{_fmt(top['value'])} {top['unit']} against the configured screening "
            f"range of {_fmt(top['absolute_range'][0])}–{_fmt(top['absolute_range'][1])} "
            f"{top['unit']} (optimum {_fmt(top['optimum_range'][0])}–"
            f"{_fmt(top['optimum_range'][1])}).")
        if len(res.computed_limiting_factors) > 1:
            second = res.computed_limiting_factors[1]
            lines.append(f"Next: {second['label'].lower()} "
                         f"({_fmt(second['value'])} {second['unit']}).")
    else:
        lines.append("No computed factor was limiting at this location.")

    # water reporting (annual AND seasonal, never silently swapped)
    if res.growing_season_precipitation is not None:
        months = ", ".join(_month_name(m) for m in res.growing_season_months[:3])
        lines.append(
            f"Water: growing-season precipitation ({res.growing_season_months[0]}–"
            f"{res.growing_season_months[-1]}, i.e. {months}…) is "
            f"**{res.growing_season_precipitation:,.0f} mm**; annual total is "
            f"**{res.annual_precipitation:,.0f} mm**. The seasonal figure is the "
            "one that is scored.")

    # factors that are fine
    good = [f for f in res.factors if f.status == "scored"
            and f.membership is not None and f.membership >= 0.999]
    if good:
        names = ", ".join(f.label.lower() for f in good)
        lines.append(f"Within the configured preferred range: {names}.")

    # unassessed constraints are limitations, never computed limitations
    if res.assumptions:
        lines.append("Not assessed in this screening: " + "; ".join(
            a.split(":")[0].lower() for a in res.assumptions[:3]) + ".")

    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) < 1000 else f"{value:,.0f}"


def _month_name(month: int) -> str:
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month - 1]
