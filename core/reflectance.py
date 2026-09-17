"""core/reflectance.py -- DN -> surface reflectance (PHASE 3).

WHY THIS MODULE EXISTS
----------------------
NDVI is a ratio of *reflectances*. Most archives ship integers (digital
numbers, DN) and the conversion lives in metadata that is frequently wrong.
Concretely, for our own Sentinel-2 sample the STAC catalogue advertises

    scale: 0.0001,  offset: -0.1

and applying that offset makes the median pixel **-0.045 reflectance** --
physically impossible. (Older Sentinel-2 baselines really did use
BOA_ADD_OFFSET = -1000 DN; baseline 05.00+ sets it to 0.)

So this module does three things, in order:

    1. PROPOSE a spec   -- from the file itself, else from a sensor profile
    2. VALIDATE it      -- apply it to real pixels and check physics
    3. RECORD evidence  -- what was used, what was rejected, and why

DESIGN NOTE (important, and non-obvious)
----------------------------------------
For a *multiplicative-only* conversion (offset = 0) NDVI is mathematically
identical whether you compute it from DN or from reflectance, because the scale
factor cancels in (NIR-Red)/(NIR+Red). We still convert to reflectance, for
three reasons:
  * it makes the offset assumption explicit and testable instead of implicit;
  * sensors with a real offset (Landsat C2: rho = DN*2.75e-5 - 0.2) NEED it;
  * later phases compare absolute reflectance against thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Physical plausibility bounds for bottom-of-atmosphere reflectance.
NEGATIVE_TOLERANCE = -0.01   # allow tiny negative noise
IMPLAUSIBLE_MAX = 1.6        # BOA reflectance above this means the scale is wrong
NEGATIVE_FRACTION_LIMIT = 0.01   # >1% implausible negative pixels -> reject offset

# Sensor profiles. `verified=True` means we have empirically checked the pair
# against real data (see docs/PHASE2.md and docs/PHASE3.md).
SENSOR_PROFILES: Dict[str, Dict[str, Any]] = {
    "sentinel-2-l2a": {
        "scale": 1.0 / 10000.0,
        "offset": 0.0,
        "note": ("Sentinel-2 L2A: rho = DN/10000. Older baselines used "
                 "BOA_ADD_OFFSET = -1000 DN (=-0.1); baseline 05.00+ uses 0. "
                 "Verified empirically on S2B_36RUV_20230806 (offset rejected)."),
        "verified": True,
    },
    "landsat-c2-l2": {
        "scale": 2.75e-5,
        "offset": -0.2,
        "note": ("Landsat 8/9 Collection-2 L2SP/L2SR: rho = DN * 2.75e-5 - 0.2. "
                 "The offset is REAL here -- unlike Sentinel-2 baseline 05. "
                 "NOT yet verified against a real Landsat file in this project."),
        "verified": False,
    },
}


@dataclass(frozen=True)
class ReflectanceSpec:
    """How to turn stored integers into surface reflectance."""

    scale: float = 1.0
    offset: float = 0.0
    profile: Optional[str] = None
    source: str = "raw"            # file_tags | sensor_profile | raw | user_supplied
    is_reflectance: bool = True    # False => values are DN / unknown units
    verified: bool = False
    evidence: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def apply(self, dn: np.ndarray) -> np.ndarray:
        """rho = DN * scale + offset, computed in float32."""
        return dn.astype(np.float32) * np.float32(self.scale) + np.float32(self.offset)

    @property
    def label(self) -> str:
        if not self.is_reflectance:
            return "raw digital numbers (no reflectance conversion)"
        txt = f"rho = DN x {self.scale:g}"
        if self.offset:
            txt += f" {self.offset:+g}"
        return txt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "offset": self.offset,
            "profile": self.profile,
            "source": self.source,
            "is_reflectance": self.is_reflectance,
            "verified": self.verified,
            "label": self.label,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
        }


RAW_SPEC = ReflectanceSpec(
    scale=1.0, offset=0.0, source="raw", is_reflectance=False,
    warnings=("No reflectance scaling known: values are treated as raw digital numbers. "
              "NDVI remains valid IF both bands share the same scaling and there is no "
              "additive offset -- but this has not been verified."),
)


# --------------------------------------------------------------------------- #
# proposal
# --------------------------------------------------------------------------- #
def spec_from_profile(profile: Optional[str]) -> Optional[ReflectanceSpec]:
    """ReflectanceSpec from a known sensor profile, or None."""
    if not profile or profile not in SENSOR_PROFILES:
        return None
    p = SENSOR_PROFILES[profile]
    return ReflectanceSpec(
        scale=float(p["scale"]),
        offset=float(p["offset"]),
        profile=profile,
        source="sensor_profile",
        is_reflectance=True,
        verified=bool(p["verified"]),
        evidence=(f"sensor profile '{profile}': {p['note']}",),
    )


def spec_from_file(ds: Any) -> Optional[ReflectanceSpec]:
    """ReflectanceSpec from scaling the FILE declares (GDAL scale/offset or tags).

    True only when the file actually declares something other than (1.0, 0.0);
    GDAL reports (1.0, 0.0) by default, which is not evidence of anything.
    """
    scale = offset = None
    try:
        scales = ds.scales
        offsets = ds.offsets
        if scales and scales[0] is not None:
            scale = float(scales[0])
        if offsets and offsets[0] is not None:
            offset = float(offsets[0])
    except Exception:
        pass

    tags = {}
    try:
        tags = ds.tags() or {}
    except Exception:
        pass
    if scale is None:
        for key in ("scale_factor", "SCALE_FACTOR", "scale", "SCALE"):
            if key in tags:
                scale = float(tags[key])
                break
    if offset is None:
        for key in ("add_offset", "ADD_OFFSET", "offset", "OFFSET"):
            if key in tags:
                offset = float(tags[key])
                break

    if scale is None and offset is None:
        return None
    scale = 1.0 if scale is None else scale
    offset = 0.0 if offset is None else offset
    if scale == 1.0 and offset == 0.0:
        return None
    return ReflectanceSpec(
        scale=scale, offset=offset, source="file_tags", is_reflectance=True,
        evidence=(f"the file declares scale={scale:g}, offset={offset:g}",),
    )


# --------------------------------------------------------------------------- #
# validation -- the part that stops us trusting bad metadata
# --------------------------------------------------------------------------- #
def validate_spec(spec: ReflectanceSpec, dn_valid: np.ndarray, max_sample: int = 200_000) -> Tuple[ReflectanceSpec, Dict[str, Any]]:
    """Apply `spec` to a sample of REAL pixels and check the physics.

    Returns (possibly corrected spec, report). The correction is deliberately
    narrow: if the reflectance comes out negative for a large share of pixels
    AND an offset was applied, we drop the offset and re-test. That is the
    Sentinel-2 stale-metadata case, caught empirically rather than by trusting
    (or ignoring) the catalogue.
    """
    dn = np.asarray(dn_valid).ravel().astype(np.float32)
    if dn.size > max_sample:
        rng = np.random.default_rng(0)
        dn = dn[rng.choice(dn.size, max_sample, replace=False)]

    if dn.size == 0:
        return spec, {"sample_pixels": 0, "negative_fraction": None,
                      "implausible_high_fraction": None, "offset_rejected": False,
                      "median_reflectance": None}

    def assess(candidate: ReflectanceSpec) -> Dict[str, Any]:
        rho = candidate.apply(dn)
        rho = rho[np.isfinite(rho)]
        if rho.size == 0:
            return {"negative_fraction": 1.0, "implausible_high_fraction": 0.0,
                    "median_reflectance": None}
        return {
            "negative_fraction": float(np.mean(rho < NEGATIVE_TOLERANCE)),
            "implausible_high_fraction": float(np.mean(rho > IMPLAUSIBLE_MAX)),
            "median_reflectance": float(np.median(rho)),
        }

    report = assess(spec)
    offset_rejected = False
    warnings = list(spec.warnings)
    evidence = list(spec.evidence)

    if report["negative_fraction"] > NEGATIVE_FRACTION_LIMIT and spec.offset != 0.0:
        retry = replace(spec, offset=0.0)
        retry_report = assess(retry)
        if retry_report["negative_fraction"] <= report["negative_fraction"]:
            offset_rejected = True
            spec = retry
            report = retry_report
            warnings.append(
                f"Offset {spec.offset:+g} rejected: it made "
                f"{100 * report['negative_fraction']:.1f}% of sampled pixels negative, "
                f"which is physically impossible for surface reflectance. Offset set to 0 "
                f"after empirical validation."
            )
            evidence.append("offset validated empirically against pixel values and rejected")
    elif report["negative_fraction"] > NEGATIVE_FRACTION_LIMIT:
        warnings.append(
            f"{100 * report['negative_fraction']:.1f}% of sampled pixels are below "
            f"{NEGATIVE_TOLERANCE} reflectance even with offset 0 -- check that this band "
            f"is really surface reflectance and not a raw or mis-scaled product."
        )

    if report["implausible_high_fraction"] > NEGATIVE_FRACTION_LIMIT:
        warnings.append(
            f"{100 * report['implausible_high_fraction']:.1f}% of sampled pixels exceed "
            f"{IMPLAUSIBLE_MAX} reflectance -- the scale factor may be wrong for this product."
        )

    spec = replace(spec, warnings=tuple(warnings), evidence=tuple(evidence))
    report.update({"sample_pixels": int(dn.size), "offset_rejected": offset_rejected})
    return spec, report


def detect_reflectance_spec(
    ds: Any = None,
    profile: Optional[str] = None,
    dn_sample: Optional[np.ndarray] = None,
) -> Tuple[ReflectanceSpec, Dict[str, Any]]:
    """Propose a spec (file -> profile -> raw) and validate it against pixels.

    `dn_sample` should already exclude nodata; when it is None we try to read a
    small decimated sample from `ds`.
    """
    spec = spec_from_file(ds) if ds is not None else None
    if spec is None:
        spec = spec_from_profile(profile)
    if spec is None:
        spec = RAW_SPEC
        spec = replace(spec, profile=profile)

    if dn_sample is None and ds is not None:
        dn_sample = _sample_valid_dn(ds)

    if dn_sample is None or np.asarray(dn_sample).size == 0:
        return spec, {"sample_pixels": 0, "negative_fraction": None,
                      "implausible_high_fraction": None, "offset_rejected": False,
                      "median_reflectance": None, "validated": False}

    spec, report = validate_spec(spec, dn_sample)
    report["validated"] = True
    report["spec"] = spec.to_dict()
    return spec, report


def _sample_valid_dn(ds: Any, max_pixels: int = 100_000) -> Optional[np.ndarray]:
    """A small sample of valid DN values, used only to validate the scaling."""
    try:
        from .preview import decimated_read, valid_mask

        stack, _ = decimated_read(ds, (1,), max_pixels=max_pixels)
        mask = valid_mask(stack[0], ds.nodatavals[0])
        return stack[0][mask]
    except Exception:
        return None
