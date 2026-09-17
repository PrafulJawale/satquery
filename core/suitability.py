"""Phase 8 -- the crop-suitability MODEL (no raster IO, no Streamlit).

WHAT THIS MODULE IS
    The deterministic core of the experimental screening:

        factor value -> membership -> weighted mean -> explicit gates -> class

    It works on scalars AND on numpy arrays with the same code path, so the
    per-cell map and the ROI-level summary can never disagree about the
    mathematics: the ROI summary is produced by aggregating each factor and then
    calling *the same functions* on a scalar.

WHAT THIS MODULE IS NOT
    * not a crop recommendation engine
    * not a yield model
    * not machine learning  (there is no labelled dataset, and training on our
      own output would be circular)
    * not a dataset loader   (that is core/datasources/)

Every threshold it uses comes from config/crops/<crop>.yml and is returned in
`threshold_provenance` so a reader can see, per number, where it came from and
whether it is literature-backed or experimental.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


__all__ = [
    "CROP_CONFIG_DIR",
    "load_crop_config",
    "trapezoid_membership",
    "usda_texture_class",
    "texture_membership",
    "approximate_gdd",
    "FactorAssessment",
    "SuitabilityResult",
    "CLASS_CODES",
    "CLASS_LABELS",
    "score_factors",
    "score_layer_stack",
    "classify_score",
    "assess_suitability",
]

CROP_CONFIG_DIR = "config/crops"

# Suitability class codes used in the raster that reaches the map.
CLASS_CODES: Dict[str, int] = {
    "insufficient": 0,
    "unsuitable": 1,
    "marginal": 2,
    "moderate": 3,
    "highly": 4,
}
CLASS_LABELS: Dict[int, str] = {
    0: "Insufficient data",
    1: "Unsuitable",
    2: "Marginal",
    3: "Moderately suitable",
    4: "Highly suitable",
}

USDA_CLASSES: Tuple[str, ...] = (
    "unknown",
    "sand",
    "loamy_sand",
    "sandy_loam",
    "loam",
    "silt_loam",
    "silt",
    "sandy_clay_loam",
    "clay_loam",
    "silty_clay_loam",
    "sandy_clay",
    "silty_clay",
    "clay",
)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def load_crop_config(crop: str = "cotton", base_dir: str = CROP_CONFIG_DIR) -> Dict[str, Any]:
    """Load config/crops/<crop>.yml. Raises FileNotFoundError for unknown crops."""
    import os

    path = os.path.join(base_dir, f"{crop.lower()}.yml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Crop configuration is not a mapping: {path}")
    return cfg


# --------------------------------------------------------------------------- #
# membership
# --------------------------------------------------------------------------- #
def trapezoid_membership(
    value: Any,
    abs_min: float,
    opt_low: float,
    opt_high: float,
    abs_max: float,
) -> Any:
    """Trapezoidal fuzzy membership in [0, 1]. Works on scalars and arrays.

        s = 0                                    v < abs_min or v > abs_max
        s = (v - abs_min) / (opt_low - abs_min)  abs_min <= v < opt_low
        s = 1                                    opt_low <= v <= opt_high
        s = (abs_max - v) / (abs_max - opt_high) opt_high < v <= abs_max

    NaN / None propagate as NaN -- a missing value is never scored as zero.
    Degenerate ramps (opt_low == abs_min, i.e. "flat is optimal") return 1 below
    opt_high instead of dividing by zero.
    """
    v = np.asarray(value, dtype="float64")
    s = np.full(v.shape, np.nan, dtype="float64")

    inside = np.isfinite(v)
    outside = inside & ((v < abs_min) | (v > abs_max))
    s[outside] = 0.0

    # rising limb
    if opt_low > abs_min:
        m = inside & (v >= abs_min) & (v < opt_low)
        s[m] = (v[m] - abs_min) / (opt_low - abs_min)
    else:
        m = inside & (v >= abs_min) & (v < opt_low)
        s[m] = 1.0

    # optimum plateau
    m = inside & (v >= opt_low) & (v <= opt_high)
    s[m] = 1.0

    # falling limb
    if abs_max > opt_high:
        m = inside & (v > opt_high) & (v <= abs_max)
        s[m] = (abs_max - v[m]) / (abs_max - opt_high)
    else:
        m = inside & (v > opt_high) & (v <= abs_max)
        s[m] = 1.0

    return s if s.ndim else float(s)


# --------------------------------------------------------------------------- #
# soil texture
# --------------------------------------------------------------------------- #
def usda_texture_class(sand: Any, silt: Any, clay: Any) -> Any:
    """USDA texture class code (1..12) from sand/silt/clay percentages.

    Standard USDA texture-triangle decision rules (NRCS / Soil Survey Manual).
    Returns 0 ("unknown") where the three fractions do not describe a soil.
    Array-aware.
    """
    sa = np.asarray(sand, dtype="float64")
    si = np.asarray(silt, dtype="float64")
    cl = np.asarray(clay, dtype="float64")

    total = sa + si + cl
    ok = np.isfinite(total) & (total > 0)
    # normalise to 100 so small prediction noise cannot push a point out of the triangle
    with np.errstate(invalid="ignore", divide="ignore"):
        sa = np.where(ok, sa * 100.0 / np.where(ok, total, 1.0), np.nan)
        si = np.where(ok, si * 100.0 / np.where(ok, total, 1.0), np.nan)
        cl = np.where(ok, cl * 100.0 / np.where(ok, total, 1.0), np.nan)

    conds = [
        ok & (si + 1.5 * cl < 15),                                              # sand
        ok & (si + 1.5 * cl >= 15) & (si + 2 * cl < 30),                        # loamy sand
        ok & (((cl >= 7) & (cl < 20) & (sa > 52) & (si + 2 * cl >= 30))
              | ((cl < 7) & (si < 50) & (si + 2 * cl >= 30))),                  # sandy loam
        ok & (cl >= 7) & (cl < 27) & (si >= 28) & (si < 50) & (sa <= 52),       # loam
        ok & (((si >= 50) & (cl >= 12) & (cl < 27))
              | ((si >= 50) & (si < 80) & (cl < 12))),                          # silt loam
        ok & (si >= 80) & (cl < 12),                                            # silt
        ok & (cl >= 20) & (cl < 35) & (si < 28) & (sa > 45),                    # sandy clay loam
        ok & (cl >= 27) & (cl < 40) & (sa > 20) & (sa <= 45),                   # clay loam
        ok & (cl >= 27) & (cl < 40) & (sa <= 20),                               # silty clay loam
        ok & (cl >= 35) & (sa > 45),                                            # sandy clay
        ok & (cl >= 40) & (si >= 40),                                           # silty clay
        ok & (cl >= 40) & (sa <= 45) & (si < 40),                               # clay
    ]
    out = np.zeros(sa.shape, dtype="int16")
    for code, cond in enumerate(conds, start=1):
        out = np.where(cond & (out == 0), code, out)
    return out if out.ndim else int(out)


def texture_membership(sand: Any, silt: Any, clay: Any,
                       class_scores: Dict[str, float]) -> Tuple[Any, Any]:
    """(membership, class_code) for a texture triple. Array-aware."""
    codes = np.asarray(usda_texture_class(sand, silt, clay))
    table = np.zeros(len(USDA_CLASSES), dtype="float64")
    for name, score in (class_scores or {}).items():
        if name in USDA_CLASSES:
            table[USDA_CLASSES.index(name)] = float(score)
    table[0] = np.nan                       # unknown texture -> not scored
    mem = table[np.clip(codes, 0, len(USDA_CLASSES) - 1)]
    if np.asarray(codes).ndim == 0:
        return float(mem), int(codes)
    return mem, codes


# --------------------------------------------------------------------------- #
# temperature context: an explicitly approximate degree-day sum
# --------------------------------------------------------------------------- #
def approximate_gdd(monthly_tmin: Sequence[float],
                    monthly_tmax: Sequence[float],
                    months: Sequence[int],
                    base_temp: float = 15.6,
                    days_in_month: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """APPROXIMATE growing degree-day sum from MONTHLY climatological means.

    This is NOT measured daily GDD. Monthly means remove day-to-day variability,
    so the result is a coarse thermal-sum indicator, nothing more.

    Formula, per month m in the season:

        Tmin' = max(Tmin_m, Tbase)          (daily DD practice: truncate at base)
        Tmean'= (Tmin' + Tmax_m) / 2
        DD_m  = max(0, Tmean' - Tbase) * n_m
        GDD   = sum over the season of DD_m

    * base temperature 15.6 degC (60 degF) -- the standard cotton base.
    * months whose adjusted mean is at or below the base contribute 0 (never
      negative: cotton does not "un-grow").
    * n_m = calendar days of each month (default: non-leap year).
    * no upper cut-off is applied (some cotton DD methods cap at 32-35 degC);
      the monthly-mean input is already too smooth for a daily ceiling to mean
      anything, and applying one here would imply a precision we do not have.
    """
    days = days_in_month or {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                             7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    total = 0.0
    per_month: List[Dict[str, Any]] = []
    for month, tmin, tmax in zip(months, monthly_tmin, monthly_tmax):
        if tmin is None or tmax is None or not (np.isfinite(tmin) and np.isfinite(tmax)):
            per_month.append({"month": month, "dd": None, "days": days.get(month, 30)})
            continue
        tmin_eff = max(float(tmin), base_temp)
        tmean = (tmin_eff + float(tmax)) / 2.0
        dd = max(0.0, tmean - base_temp) * days.get(month, 30)
        total += dd
        per_month.append({"month": month, "dd": round(dd, 2), "days": days.get(month, 30),
                          "tmin_used": round(tmin_eff, 3), "tmax_used": round(float(tmax), 3)})
    return {
        "value": round(total, 1),
        "base_temp_c": base_temp,
        "months": list(months),
        "per_month": per_month,
        "label": "approximate seasonal temperature sum (monthly means)",
        "warning": ("Approximated from monthly climatological means, not measured "
                    "from daily temperatures. Reported as context only -- it is "
                    "not scored and it is not a daily GDD total."),
    }


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class FactorAssessment:
    """One factor, as assessed -- carrying its own evidence status."""

    name: str
    label: str
    category: str                     # critical | supporting | hard_constraint
    unit: str = ""
    value: Optional[float] = None
    membership: Optional[float] = None
    status: str = "scored"            # scored | missing | nodata | assumed
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "label": self.label, "category": self.category,
            "unit": self.unit,
            "value": None if self.value is None or not np.isfinite(self.value) else round(float(self.value), 4),
            "membership": (None if self.membership is None or not np.isfinite(self.membership)
                           else round(float(self.membership), 4)),
            "status": self.status, "note": self.note,
        }


@dataclass
class SuitabilityResult:
    """The structured screening result (#18 of the Phase 8 brief)."""

    crop: str = "cotton"
    scenario: str = "rainfed"
    score: Optional[float] = None
    classification: str = "Insufficient data"
    confidence: str = "Low"
    status: str = "OK"                       # OK | PARTIAL_DATA | INSUFFICIENT_DATA

    factors: List[FactorAssessment] = field(default_factory=list)
    factor_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    factor_values: Dict[str, Optional[float]] = field(default_factory=dict)

    computed_limiting_factors: List[Dict[str, Any]] = field(default_factory=list)
    missing_factors: List[str] = field(default_factory=list)
    assumed_factors: List[str] = field(default_factory=list)
    hard_constraint: Optional[Dict[str, Any]] = None

    annual_precipitation: Optional[float] = None
    growing_season_precipitation: Optional[float] = None
    growing_season_months: List[int] = field(default_factory=list)
    approx_gdd: Optional[Dict[str, Any]] = None

    native_resolutions: Dict[str, Any] = field(default_factory=dict)
    analysis_resolution: Optional[float] = None
    requested_resolution: Optional[float] = None
    effective_resolution_note: str = ""

    valid_fraction: float = 0.0
    class_fractions: Dict[str, float] = field(default_factory=dict)

    provenance: List[Dict[str, Any]] = field(default_factory=list)
    threshold_provenance: List[Dict[str, Any]] = field(default_factory=list)

    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    temporal_basis: str = "climatological + static (mixed)"

    suitability_raster: Optional[Any] = None     # int8 class codes, native analysis grid
    score_raster: Optional[Any] = None
    raster_transform: Any = None
    raster_crs: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view for the UI/API. Raw arrays are deliberately excluded."""
        return {
            "crop": self.crop, "scenario": self.scenario,
            "score": None if self.score is None else round(float(self.score), 4),
            "classification": self.classification, "confidence": self.confidence,
            "status": self.status,
            "factors": [f.to_dict() for f in self.factors],
            "factor_scores": {k: (None if v is None else round(float(v), 4))
                              for k, v in self.factor_scores.items()},
            "factor_values": {k: (None if v is None else round(float(v), 4))
                              for k, v in self.factor_values.items()},
            "computed_limiting_factors": self.computed_limiting_factors,
            "missing_factors": list(self.missing_factors),
            "assumed_factors": list(self.assumed_factors),
            "hard_constraint": self.hard_constraint,
            "annual_precipitation": self.annual_precipitation,
            "growing_season_precipitation": self.growing_season_precipitation,
            "growing_season_months": list(self.growing_season_months),
            "approx_gdd": self.approx_gdd,
            "native_resolutions": self.native_resolutions,
            "analysis_resolution": self.analysis_resolution,
            "requested_resolution": self.requested_resolution,
            "effective_resolution_note": self.effective_resolution_note,
            "valid_fraction": round(float(self.valid_fraction), 4),
            "class_fractions": {k: round(float(v), 4) for k, v in self.class_fractions.items()},
            "provenance": self.provenance,
            "threshold_provenance": self.threshold_provenance,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "temporal_basis": self.temporal_basis,
        }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_factors(cfg: Dict[str, Any],
                  values: Dict[str, Optional[float]],
                  scenario_water_treatment: str = "scored",
                  ) -> Dict[str, FactorAssessment]:
    """Turn raw factor values into memberships. Pure, array-safe, no IO.

    `values` may contain scalars or arrays (per-cell layers). A factor that is
    absent from `values`, or whose value is NaN, is reported as
    missing/nodata -- never as 0.
    """
    factors_cfg = cfg.get("factors", {})
    out: Dict[str, FactorAssessment] = {}

    for name, spec in factors_cfg.items():
        unit = str(spec.get("unit", ""))
        # a factor may be fed by a source layer with a different name
        # (soil_reaction <- phh2o); the factor's own name always wins.
        key = name if name in values else str(spec.get("layer", name))
        value = values.get(key, None)
        fa = FactorAssessment(name=name, label=str(spec.get("label", name)),
                              category=str(spec.get("category", "supporting")), unit=unit)

        # soil texture is DERIVED from sand/silt/clay: it owns no single value,
        # so it is handled before the "is a value present?" test below.
        if name == "soil_texture" or spec.get("method") == "usda_class_then_score":
            inputs = list(spec.get("inputs") or ["sand_pct", "silt_pct", "clay_pct"])
            if not all(values.get(k) is not None for k in inputs):
                fa.status = "missing"
                out[name] = fa
                continue
            sand = np.asarray(values.get(inputs[0]), dtype="float64")
            silt = np.asarray(values.get(inputs[1]), dtype="float64")
            clay = np.asarray(values.get(inputs[2]), dtype="float64")
            mem, codes = texture_membership(sand, silt, clay, spec.get("class_scores", {}))
            fa.membership = mem
            fa.value = _scalar(clay)
            codes_arr = np.asarray(codes)
            dominant = int(np.nanmax(codes_arr)) if codes_arr.size else 0
            fa.note = (f"USDA texture class: {USDA_CLASSES[dominant]}; "
                       "texture proxy from sand/silt/clay, not a full soil-physical assessment")
            out[name] = fa
            continue

        if name == "growing_season_precipitation" and scenario_water_treatment == "assumed":
            fa.status = "assumed"
            fa.value = None if value is None else _scalar(value)
            fa.note = ("Water supply is ASSUMED adequate in this scenario; the "
                       "factor is not scored and its weight is removed.")
            out[name] = fa
            continue

        if value is None:
            fa.status = "missing"
            out[name] = fa
            continue

        arr = np.asarray(value, dtype="float64")
        if arr.size and not np.any(np.isfinite(arr)):
            fa.status = "nodata"
            fa.value = None
            out[name] = fa
            continue

        # some layers arrive ALREADY expressed as a membership (e.g. the
        # growing-season temperature, which averages per-month memberships)
        if str(spec.get("input", "value")) == "membership":
            fa.membership = np.clip(arr, 0.0, 1.0)
            fa.value = _scalar(arr)
            fa.unit = "membership (0-1)"
            fa.note = str(spec.get("input_note", "Value arrives pre-normalised as a membership."))
            out[name] = fa
            continue

        opt = spec.get("optimum", [0, 1])
        abso = spec.get("absolute", [0, 1])
        mem = trapezoid_membership(arr, float(abso[0]), float(opt[0]),
                                   float(opt[1]), float(abso[1]))
        fa.membership = mem
        fa.value = _scalar(arr)
        out[name] = fa

    return out


def _metres(text: str) -> Optional[float]:
    """'250 m' -> 250.0, '~1 km' -> 1000.0. None when it cannot be parsed."""
    import re

    t = text.strip().lower().replace("~", "").replace("about", "").strip()
    try:
        if "km" in t:
            return float(re.sub(r"[^0-9.]", "", t.split("km")[0]) or 0) * 1000.0
        if "m" in t:
            return float(re.sub(r"[^0-9.]", "", t.split("m")[0]) or 0)
        return float(re.sub(r"[^0-9.]", "", t) or 0)
    except ValueError:
        return None


def _scalar(arr: Any) -> Optional[float]:
    a = np.asarray(arr, dtype="float64")
    if a.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        mean = float(np.nanmean(a)) if a.size > 1 else float(a.reshape(-1)[0])
    return mean if np.isfinite(mean) else None


def classify_score(score: Any, cfg: Dict[str, Any]) -> Any:
    """Score -> class code. Array-aware. NaN -> Insufficient data."""
    classes = cfg.get("classes", {})
    hi = float(classes.get("highly_suitable", 0.75))
    mo = float(classes.get("moderately_suitable", 0.55))
    ma = float(classes.get("marginal", 0.35))
    s = np.asarray(score, dtype="float64")
    out = np.zeros(s.shape, dtype="int8")                       # insufficient
    out = np.where(np.isfinite(s) & (s < ma), CLASS_CODES["unsuitable"], out)
    out = np.where(np.isfinite(s) & (s >= ma) & (s < mo), CLASS_CODES["marginal"], out)
    out = np.where(np.isfinite(s) & (s >= mo) & (s < hi), CLASS_CODES["moderate"], out)
    out = np.where(np.isfinite(s) & (s >= hi), CLASS_CODES["highly"], out)
    return out if out.ndim else int(out)


# --------------------------------------------------------------------------- #
# per-cell scoring (the array twin of assess_suitability)
# --------------------------------------------------------------------------- #
def score_layer_stack(cfg: Dict[str, Any],
                      layers: Dict[str, Any],
                      hard_mask: Any = None,
                      water_treatment: str = "scored",
                      inside: Any = None) -> Dict[str, Any]:
    """Score every cell of the analysis grid with the SAME rules as the ROI summary.

    Returns score (float32, NaN where it cannot be computed), class codes (int8)
    and the per-cell membership stack.
    """
    factors_cfg = cfg.get("factors", {})
    weights = dict(cfg.get("weights", {}).get("values", {}))

    memberships: Dict[str, np.ndarray] = {}
    for name, spec in factors_cfg.items():
        if name == "growing_season_precipitation" and water_treatment == "assumed":
            continue
        key = name if name in layers else str(spec.get("layer", name))
        arr = layers.get(key, None)
        if arr is None:
            continue
        a = np.asarray(arr, dtype="float64")
        if spec.get("method") == "usda_class_then_score":
            inputs = list(spec.get("inputs") or ["sand_pct", "silt_pct", "clay_pct"])
            if not all(k in layers for k in inputs):
                continue
            m, _ = texture_membership(layers[inputs[0]], layers[inputs[1]],
                                      layers[inputs[2]], spec.get("class_scores", {}))
        elif str(spec.get("input", "value")) == "membership":
            m = np.clip(a, 0.0, 1.0)
        else:
            opt = spec.get("optimum", [0, 1])
            abso = spec.get("absolute", [0, 1])
            m = trapezoid_membership(a, float(abso[0]), float(opt[0]),
                                     float(opt[1]), float(abso[1]))
        memberships[name] = np.asarray(m, dtype="float64")

    num = np.zeros_like(next(iter(memberships.values()))) if memberships else None
    den = np.zeros_like(num) if num is not None else None
    for name, m in memberships.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        finite = np.isfinite(m)
        num += w * np.where(finite, m, 0.0)
        den += w * finite

    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    # gate 2 -- critical veto (membership exactly 0)
    veto = np.zeros(score.shape, dtype=bool)
    for name, m in memberships.items():
        if str(factors_cfg.get(name, {}).get("category", "")) == "critical":
            veto |= np.isfinite(m) & (m <= 0.0)

    # gate 3 -- a critical factor with no data at all cannot be scored
    critical_missing = np.zeros(score.shape, dtype=bool)
    for name, spec in factors_cfg.items():
        if str(spec.get("category", "")) != "critical":
            continue
        if name not in memberships:
            critical_missing |= np.ones(score.shape, dtype=bool)
        else:
            critical_missing |= ~np.isfinite(memberships[name])

    codes = classify_score(score, cfg)
    codes = np.where(veto, CLASS_CODES["unsuitable"], codes)
    codes = np.where(critical_missing, CLASS_CODES["insufficient"], codes)
    if hard_mask is not None:
        codes = np.where(hard_mask, CLASS_CODES["unsuitable"], codes)

    if inside is not None:
        codes = np.where(inside, codes, CLASS_CODES["insufficient"])
        valid_cells = int(np.count_nonzero(inside))
        scored_cells = int(np.count_nonzero(inside & np.isfinite(score) & ~critical_missing))
    else:
        valid_cells = int(score.size)
        scored_cells = int(np.count_nonzero(np.isfinite(score) & ~critical_missing))

    # fractions are reported over the ROI only: cells outside the drawn area are
    # "insufficient" by definition and must not dilute the percentages
    roi_codes = codes[inside] if inside is not None else codes
    fractions: Dict[str, float] = {}
    if roi_codes.size:
        for code, label in CLASS_LABELS.items():
            fractions[label] = float(np.count_nonzero(roi_codes == code) / roi_codes.size)

    return {
        "score": score.astype("float32"),
        "class_codes": codes.astype("int8"),
        "memberships": memberships,
        "veto": veto,
        "critical_missing": critical_missing,
        "valid_fraction": (scored_cells / valid_cells) if valid_cells else 0.0,
        "class_fractions": fractions,
    }


# --------------------------------------------------------------------------- #
# the assessment
# --------------------------------------------------------------------------- #
def assess_suitability(
    cfg: Dict[str, Any],
    scenario: str,
    values: Dict[str, Optional[float]],
    scenario_cfg: Optional[Dict[str, Any]] = None,
    landcover: Optional[Dict[str, Any]] = None,
    precipitation_context: Optional[Dict[str, Any]] = None,
    native_resolutions: Optional[Dict[str, Any]] = None,
    analysis_resolution: Optional[float] = None,
    requested_resolution: Optional[float] = None,
    effective_resolution_note: str = "",
    provenance: Optional[List[Dict[str, Any]]] = None,
    extra_assumptions: Optional[List[str]] = None,
) -> SuitabilityResult:
    """Score one unit (a cell or an aggregated ROI) and explain the outcome.

    The four gates are applied in a fixed order and reported separately:
      1. hard land-cover constraint  -> Unsuitable
      2. critical-factor veto (s=0)  -> Unsuitable
      3. missing critical factor     -> INSUFFICIENT_DATA (no score)
      4. assumed mandatory factor    -> class capped, confidence Low
    """
    scenario_cfg = scenario_cfg or cfg.get("scenarios", {}).get(scenario, {})
    treatments = str(scenario_cfg.get("water_treatment", "scored"))

    res = SuitabilityResult(
        crop=str(cfg.get("crop", "cotton")),
        scenario=scenario,
        analysis_resolution=analysis_resolution,
        requested_resolution=requested_resolution,
        effective_resolution_note=effective_resolution_note,
        native_resolutions=dict(native_resolutions or {}),
        provenance=list(provenance or []),
    )
    res.threshold_provenance = threshold_provenance(cfg)

    # ---- factor scoring ---------------------------------------------------- #
    assessments = score_factors(cfg, values, treatments)
    res.factors = list(assessments.values())
    res.factor_scores = {n: (None if a.membership is None else _scalar(a.membership))
                         for n, a in assessments.items()}
    res.factor_values = {n: a.value for n, a in assessments.items()}
    # the texture inputs are reported too, so the proxy can be checked by hand
    for extra in ("sand_pct", "silt_pct", "clay_pct"):
        if extra in values:
            res.factor_values[extra] = _scalar(values[extra])
    res.missing_factors = [n for n, a in assessments.items() if a.status in ("missing", "nodata")]
    res.assumed_factors = [n for n, a in assessments.items() if a.status == "assumed"]

    # ---- precipitation context (always reported, never silently swapped) --- #
    if precipitation_context:
        res.annual_precipitation = precipitation_context.get("annual_precipitation")
        res.growing_season_precipitation = precipitation_context.get("growing_season_precipitation")
        res.growing_season_months = list(precipitation_context.get("months", []))
        res.approx_gdd = precipitation_context.get("approx_gdd")

    # ---- gate 1: hard land-cover constraint -------------------------------- #
    if landcover and landcover.get("excluded_fraction", 0.0) > 0.5:
        res.hard_constraint = {
            "type": "land_cover",
            "dominant_class": landcover.get("dominant_class"),
            "reason": landcover.get("reason", "Excluded land-cover class dominates the area."),
        }
        res.classification = "Unsuitable"
        res.score = 0.0
        res.confidence = "Medium"
        res.warnings.append(
            f"Hard constraint: {landcover.get('reason', 'excluded land cover')} "
            "This is an exclusion, not a suitability score.")
        _finish(res, cfg)
        return res

    # ---- gate 3: missing critical factor (checked before scoring) ---------- #
    critical_missing = [
        n for n, a in assessments.items()
        if a.category == "critical" and a.status in ("missing", "nodata")
    ]
    if critical_missing:
        res.status = "INSUFFICIENT_DATA"
        res.classification = "Insufficient data"
        res.score = None
        res.confidence = "Low"
        for n in critical_missing:
            res.warnings.append(
                f"No score was computed: the critical factor '{n}' has no data "
                "for this area. A zero was NOT substituted.")
        _finish(res, cfg)
        return res

    # ---- weighted mean over available factors ------------------------------ #
    weights_cfg = dict(cfg.get("weights", {}).get("values", {}))
    used: Dict[str, float] = {}
    total_w = 0.0
    for n, a in assessments.items():
        if a.status != "scored" or a.membership is None:
            continue
        mem = _scalar(a.membership) if not np.isscalar(a.membership) else float(a.membership)
        if mem is None or not np.isfinite(mem):
            continue
        w = float(weights_cfg.get(n, 0.0))
        if w <= 0:
            continue
        used[n] = mem
        total_w += w

    if not used or total_w <= 0:
        res.status = "INSUFFICIENT_DATA"
        res.classification = "Insufficient data"
        res.score = None
        res.confidence = "Low"
        res.warnings.append("No scored factor had usable data, so no score was computed.")
        _finish(res, cfg)
        return res

    res.score = sum(weights_cfg.get(n, 0.0) * m for n, m in used.items()) / total_w

    # ---- gate 2: critical-factor veto -------------------------------------- #
    vetoed = [
        n for n, a in assessments.items()
        if a.category == "critical" and a.status == "scored"
        and a.membership is not None and _scalar(a.membership) == 0.0
    ]
    if vetoed:
        res.classification = "Unsuitable"
        res.warnings.append(
            "Critical-factor veto: " + ", ".join(vetoed) +
            " is outside the absolute tolerance range for cotton, so the area is "
            "screened as Unsuitable regardless of the weighted score "
            f"({res.score:.2f}).")
    else:
        res.classification = CLASS_LABELS[int(classify_score(res.score, cfg))]

    # ---- gate 4: unverified-input cap -------------------------------------- #
    if res.assumed_factors or scenario_cfg.get("hypothetical"):
        max_class = str(scenario_cfg.get("max_class", "Moderately suitable"))
        order = ["Unsuitable", "Marginal", "Moderately suitable", "Highly suitable"]
        if order.index(res.classification) > order.index(max_class) if res.classification in order else False:
            res.warnings.append(
                f"Class capped at '{max_class}': this scenario assumes a mandatory "
                "factor, so it cannot reach a higher screening class.")
            res.classification = max_class
        res.confidence = str(scenario_cfg.get("confidence", "Low"))

    # ---- limiting factors (computed only, never from missing data) --------- #
    scored = [(n, _scalar(a.membership)) for n, a in assessments.items()
              if a.status == "scored" and a.membership is not None]
    scored = [(n, m) for n, m in scored if m is not None and np.isfinite(m)]
    scored.sort(key=lambda kv: kv[1])
    for n, m in scored[:2]:
        if m >= 0.999:
            continue
        spec = cfg.get("factors", {}).get(n, {})
        opt = spec.get("optimum", [None, None])
        abso = spec.get("absolute", [None, None])
        res.computed_limiting_factors.append({
            "factor": n,
            "label": str(spec.get("label", n)),
            "value": res.factor_values.get(n),
            "unit": str(spec.get("unit", "")),
            "membership": round(float(m), 4),
            "optimum_range": list(opt),
            "absolute_range": list(abso),
            "weight": weights_cfg.get(n),
        })

    # ---- confidence (when not already set by a gate) ----------------------- #
    if res.confidence == "Low" and not (res.assumed_factors or scenario_cfg.get("hypothetical")):
        if res.missing_factors:
            res.status = "PARTIAL_DATA"
            res.confidence = "Medium"
        elif res.valid_fraction and res.valid_fraction < 0.9:
            res.confidence = "Medium"
        else:
            res.confidence = "Medium" if res.classification == "Marginal" else "High"

    if res.missing_factors and res.status == "OK":
        res.status = "PARTIAL_DATA"

    # ---- resolution honesty (#2 of the brief) ------------------------------ #
    if analysis_resolution:
        for name, native in (native_resolutions or {}).items():
            native_m = _metres(str(native))
            if native_m is not None and native_m >= analysis_resolution * 4:
                res.warnings.append(
                    f"The {name} layer is native {native} but is shown on a "
                    f"{analysis_resolution:g} m analysis grid. Resampling does not "
                    "create detail: effective information resolution remains "
                    f"~{native}.")
    if effective_resolution_note:
        res.warnings.append(effective_resolution_note)

    for a in (extra_assumptions or []):
        res.assumptions.append(a)

    _finish(res, cfg)
    return res


def _finish(res: SuitabilityResult, cfg: Dict[str, Any]) -> None:
    """Attach the standing assumptions and unassessed-constraint caveats."""
    for item in cfg.get("unassessed_constraints", []) or []:
        res.assumptions.append(
            f"{item.get('label', 'Unknown constraint')}: not assessed -- "
            f"{item.get('reason', '')}")
    if res.temporal_basis:
        res.assumptions.append(
            "This screening combines a climatological normal with static soil, "
            "terrain and land-cover layers. It is not a forecast for a specific "
            "season or year.")


def threshold_provenance(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Machine-readable provenance for every threshold used (#6, #17)."""
    out: List[Dict[str, Any]] = []
    for name, spec in (cfg.get("factors") or {}).items():
        out.append({
            "item": f"factor:{name}",
            "threshold_type": "optimum+absolute",
            "threshold_values": {
                "optimum": list(spec.get("optimum", [])),
                "absolute": list(spec.get("absolute", [])),
            },
            "threshold_status": spec.get("threshold_status", "experimental"),
            "threshold_source": " ".join(str(spec.get("threshold_source", "")).split()),
            "adaptation": " ".join(str(spec.get("adaptation", "")).split()) or None,
        })
    w = cfg.get("weights", {}) or {}
    out.append({
        "item": "weights",
        "threshold_type": "weighting scheme",
        "threshold_values": dict(w.get("values", {})),
        "threshold_status": w.get("threshold_status", "experimental"),
        "threshold_source": " ".join(str(w.get("threshold_source", "")).split()),
        "adaptation": None,
    })
    c = cfg.get("classes", {}) or {}
    out.append({
        "item": "classes",
        "threshold_type": "classification breaks",
        "threshold_values": {"highly_suitable": c.get("highly_suitable"),
                             "moderately_suitable": c.get("moderately_suitable"),
                             "marginal": c.get("marginal")},
        "threshold_status": c.get("threshold_status", "experimental"),
        "threshold_source": " ".join(str(c.get("threshold_source", "")).split()),
        "adaptation": None,
    })
    g = cfg.get("growing_season", {}) or {}
    out.append({
        "item": "growing_season_months",
        "threshold_type": "temporal window",
        "threshold_values": {"months": list(g.get("months", []))},
        "threshold_status": g.get("threshold_status", "experimental"),
        "threshold_source": " ".join(str(g.get("threshold_source", "")).split()),
        "adaptation": " ".join(str(g.get("limitation", "")).split()) or None,
    })
    return out
