"""core/bands.py -- band inspection and role guessing (PHASE 2).

PROBLEM
-------
"Which band is near-infrared?" sounds trivial and is not. Band meaning is
*metadata*, and it is frequently missing, misleading, or wrong:

  * our `RGB.byte.tif` has no band descriptions at all (only `colorinterp`);
  * on Earth Search v1 the STAC asset key `nir08` is B8A (**20 m**), while the
    10 m NIR band B08 is under the key `nir`;
  * `rgb1_fake_nir_epsg3857.tif` says "nir" in its filename and has ONE band;
  * Landsat names bands `SR_B4` (red) while Sentinel-2 uses `B04`.

So this module does **guessing with evidence**, never silent assumption. Every
guess returns its confidence and the evidence behind it, the UI shows both, and
the user confirms or overrides. Phase 3 will refuse to compute NDVI from an
unconfirmed or missing NIR band.

WHAT WE USE AS EVIDENCE (best first)
------------------------------------
1. GDAL band **descriptions** (`B08_nir_842nm`, `SR_B5`, `blue`, ...)
2. GDAL **colorinterp** (`red`, `green`, `blue`) -- the only clue some files have
3. **Band-count profiles** (13 bands = Sentinel-2 SAFE stack) -- weak on its own
4. Filename tokens -- weakest, and only used as a hint we print, never decide

Also carries the reflectance scale/offset used in Phase 3, because the sensor
profile that identifies the bands is the same one that knows the scaling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import RasterInfo

# --------------------------------------------------------------------------- #
# wavelength reference tables
# --------------------------------------------------------------------------- #
# Sentinel-2 MSI band id -> (role, central wavelength nm, ground sampling m)
SENTINEL2_BANDS: Dict[str, Tuple[str, int, int]] = {
    "B01": ("coastal", 443, 60),
    "B02": ("blue", 490, 10),
    "B03": ("green", 560, 10),
    "B04": ("red", 665, 10),
    "B05": ("rededge1", 705, 20),
    "B06": ("rededge2", 740, 20),
    "B07": ("rededge3", 783, 20),
    "B08": ("nir", 842, 10),
    "B8A": ("nir08", 865, 20),
    "B09": ("nir09", 945, 60),
    "B11": ("swir16", 1610, 20),
    "B12": ("swir22", 2190, 20),
}

# Landsat 8/9 Collection-2 Level-2 band id -> (role, wavelength nm, GSD m)
LANDSAT89_BANDS: Dict[str, Tuple[str, int, int]] = {
    "SR_B1": ("coastal", 443, 30),
    "SR_B2": ("blue", 482, 30),
    "SR_B3": ("green", 562, 30),
    "SR_B4": ("red", 655, 30),
    "SR_B5": ("nir", 865, 30),
    "SR_B6": ("swir16", 1609, 30),
    "SR_B7": ("swir22", 2201, 30),
}

# plain-word tokens -> role
WORD_TOKENS: Dict[str, str] = {
    "coastal": "coastal", "aerosol": "coastal", "ultrablue": "coastal",
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir": "nir", "nearinfrared": "nir", "infrared": "nir",
    "nir08": "nir08", "narrownir": "nir08", "rededge": "rededge1",
    "swir16": "swir16", "swir1": "swir16", "swircirrus": "nir09",
    "swir22": "swir22", "swir2": "swir22",
    "cirrus": "nir09",
    "panchromatic": "pan", "pan": "pan",
    "thermal": "thermal", "lwir": "thermal",
    "scl": "scl", "qa": "qa", "cloud": "scl",
}

# sensor profile -> (scale, offset) to convert DN to surface reflectance.
# Phase 3 needs this; it lives here because it belongs to the same profile
# that identifies the bands.
REFLECTANCE_PROFILES: Dict[str, Tuple[float, float]] = {
    # Sentinel-2 L2A: reflectance = DN / 10000. Older baselines advertised a
    # -0.1 offset (BOA_ADD_OFFSET = -1000); baseline 05.00+ sets it to 0.
    # We default to 0 and let the caller override -- see docs/PHASE2.md.
    "sentinel-2-l2a": (1.0 / 10000.0, 0.0),
    # Landsat C2 L2SP/L2SR: reflectance = DN * 2.75e-5 - 0.2  (offset is REAL)
    "landsat-c2-l2": (2.75e-5, -0.2),
}


# --------------------------------------------------------------------------- #
# result type
# --------------------------------------------------------------------------- #
@dataclass
class BandRoleGuess:
    """A guess about band meaning, with the evidence that produced it.

    `roles` maps a role name ("red", "nir", ...) to a 1-based band index.
    `confidence` is one of: "high", "medium", "low", "none".
    """

    roles: Dict[str, int] = field(default_factory=dict)
    confidence: str = "none"
    profile: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reflectance_scale: Optional[float] = None
    reflectance_offset: Optional[float] = None

    def band(self, role: str) -> Optional[int]:
        return self.roles.get(role)

    @property
    def has_nir(self) -> bool:
        return "nir" in self.roles

    @property
    def has_rgb(self) -> bool:
        return all(r in self.roles for r in ("red", "green", "blue"))

    @property
    def needs_confirmation(self) -> bool:
        """Phase 3 gates NDVI on this: low-confidence guesses must be confirmed."""
        return self.confidence in ("none", "low")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roles": dict(self.roles),
            "confidence": self.confidence,
            "profile": self.profile,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "reflectance_scale": self.reflectance_scale,
            "reflectance_offset": self.reflectance_offset,
            "has_nir": self.has_nir,
            "has_rgb": self.has_rgb,
            "needs_confirmation": self.needs_confirmation,
        }


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def _normalise(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _tokens(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [t for t in re.split(r"[^a-zA-Z0-9]+", str(text).lower()) if t]


def _match_sentinel2(desc: str) -> Optional[Tuple[str, int]]:
    """`B08_nir_842nm` -> ("nir", 842). Returns (role, wavelength) or None."""
    for token in _tokens(desc):
        key = token.upper()                      # B02, B8A, B08, ...
        if key in SENTINEL2_BANDS:
            role, wl, _gsd = SENTINEL2_BANDS[key]
            return role, wl
        # B8 / B08 written as bare numbers, e.g. "band 08 nir"
        m = re.fullmatch(r"B?0?(\d{1,2})A?", key)
        if m:
            key2 = f"B{int(m.group(1)):02d}"
            if key2 in SENTINEL2_BANDS:
                role, wl, _gsd = SENTINEL2_BANDS[key2]
                return role, wl
    return None


def _match_landsat(desc: str) -> Optional[Tuple[str, int]]:
    norm = _normalise(desc)
    for key, (role, wl, _gsd) in LANDSAT89_BANDS.items():
        if _normalise(key) in norm:
            return role, wl
    return None


def _match_words(desc: str) -> Optional[str]:
    """`blue`, `red`, `nir`, `SWIR-1` -> role."""
    norm = _normalise(desc)
    if not norm:
        return None
    for token, role in WORD_TOKENS.items():
        key = _normalise(token)
        if key and key in norm:
            return role
    return None


def _match_wavelength(desc: str) -> Optional[str]:
    """A literal wavelength in the description (e.g. `842nm`) is strong evidence."""
    m = re.search(r"(\d{3,4})\s*nm", str(desc).lower())
    if not m:
        return None
    wl = int(m.group(1))
    best, best_diff = None, 40
    for role, ref_wl, _gsd in list(SENTINEL2_BANDS.values()) + list(LANDSAT89_BANDS.values()):
        diff = abs(ref_wl - wl)
        if diff < best_diff:
            best, best_diff = role, diff
    return best


# --------------------------------------------------------------------------- #
# the guess itself
# --------------------------------------------------------------------------- #
def guess_band_roles(info: RasterInfo) -> BandRoleGuess:
    """Inspect `RasterInfo` and guess which band is which. Never silent."""
    guess = BandRoleGuess()
    roles: Dict[str, int] = {}

    # ---- pass 1: explicit descriptions ------------------------------------ #
    matched = 0
    for b in info.bands:
        role: Optional[str] = None
        why = ""
        s2 = _match_sentinel2(b.name or "")
        if s2:
            role, _wl = s2
            why = f"band {b.index}: description '{b.name}' matches a Sentinel-2 band id"
        if role is None:
            ls = _match_landsat(b.name or "")
            if ls:
                role, _wl = ls
                why = f"band {b.index}: description '{b.name}' matches a Landsat 8/9 band id"
        if role is None:
            role = _match_wavelength(b.name or "")
            if role:
                why = f"band {b.index}: description '{b.name}' contains a central wavelength"
        if role is None:
            role = _match_words(b.name or "")
            if role:
                why = f"band {b.index}: description '{b.name}' contains the word '{role}'"
        if role:
            roles.setdefault(role, b.index)
            matched += 1
            guess.evidence.append(why)

    if matched:
        # Sentinel-2 descriptions also tell us the reflectance scaling.
        if any(_match_sentinel2(b.name or "") for b in info.bands):
            guess.profile = "sentinel-2-l2a"
        elif any(_match_landsat(b.name or "") for b in info.bands):
            guess.profile = "landsat-c2-l2"
        if guess.profile and guess.profile in REFLECTANCE_PROFILES:
            guess.reflectance_scale, guess.reflectance_offset = REFLECTANCE_PROFILES[guess.profile]
            guess.evidence.append(
                f"profile '{guess.profile}' implies reflectance = DN × {guess.reflectance_scale:g} "
                f"{'+ ' + str(guess.reflectance_offset) if guess.reflectance_offset else ''}".strip()
            )

    # ---- pass 2: color interpretation (the only clue in many files) ------- #
    if not {"red", "green", "blue"} <= set(roles):
        ci_roles = {}
        for b in info.bands:
            ci = b.color_interp
            if ci and ci.lower() in ("red", "green", "blue"):
                ci_roles[ci.lower()] = b.index
        if len(ci_roles) == 3:
            for role, idx in ci_roles.items():
                roles.setdefault(role, idx)
            guess.evidence.append(
                f"GDAL color interpretation marks bands "
                f"{ci_roles.get('red')},{ci_roles.get('green')},{ci_roles.get('blue')} as red,green,blue"
            )

    # ---- pass 3: band-count profiles (weak evidence) ---------------------- #
    if not roles and info.count == 13:
        order = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
                 "nir", "nir08", "nir09", "swir16", "swir22", "scl"]
        roles = {r: i + 1 for i, r in enumerate(order)}
        guess.profile = guess.profile or "sentinel-2-l2a"
        guess.evidence.append(
            "13 unnamed bands: assumed to be a Sentinel-2 stack in standard order. "
            "This is a WEAK assumption -- confirm it before trusting any index."
        )
        guess.warnings.append("Band meaning was assumed from band count only.")

    # ---- pass 4: last resort ---------------------------------------------- #
    if not roles and info.count >= 3:
        roles = {"red": 1, "green": 2, "blue": 3}
        guess.evidence.append(
            f"No usable band metadata: ASSUMED band 1=red, 2=green, 3=blue. "
            f"This is a guess, not knowledge."
        )
        guess.warnings.append(
            "Band meaning is unknown. The order red,green,blue is assumed for display only."
        )

    guess.roles = roles

    # ---- confidence ------------------------------------------------------- #
    has_rgb = {"red", "green", "blue"} <= set(roles)
    if has_rgb and "nir" in roles and any("Sentinel-2" in e or "Landsat" in e for e in guess.evidence):
        guess.confidence = "high"
    elif has_rgb and "nir" in roles:
        guess.confidence = "medium"
    elif has_rgb:
        guess.confidence = "medium" if any("color interpretation" in e for e in guess.evidence) else "low"
        if "nir" not in roles:
            guess.warnings.append(
                "No near-infrared band identified: NDVI/NDWI and false-colour composites "
                "need NIR, so they are unavailable unless you assign it manually."
            )
    else:
        guess.confidence = "low" if roles else "none"

    if info.count < 4 and "nir" not in roles:
        guess.warnings.append(
            f"Only {info.count} band(s): a multispectral file normally has at least 4 "
            "(blue, green, red, NIR)."
        )
    return guess


def describe_guess(guess: BandRoleGuess) -> str:
    """One-line human summary, used in the UI banner."""
    if not guess.roles:
        return "Band meaning could not be determined."
    parts = [f"{role}=band {idx}" for role, idx in sorted(guess.roles.items(), key=lambda kv: kv[1])]
    return f"{guess.confidence.upper()} confidence ({guess.profile or 'no profile'}): " + ", ".join(parts)
