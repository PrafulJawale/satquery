"""core/indices.py -- spectral index engine (PHASE 3: NDVI; later NDWI/MNDWI).

ARCHITECTURE RULE
-----------------
This module is **pure computation**. It never imports Streamlit, never renders
and never talks to a UI. It accepts arrays (or a rasterio dataset) and returns a
structured `AnalysisResult`.

That is deliberate: in Phase 8 the natural-language query router will call
exactly these functions.

    "show vegetation health here"  ->  router  ->  compute_ndvi(...)  ->  AnalysisResult
                                                                          -> map + sentence

The LLM picks WHICH analysis to run and never computes the values itself.

NDVI
----
    NDVI = (NIR - Red) / (NIR + Red)          [Rouse et al. 1974; Tucker 1979]

Theoretically bounded to [-1, 1] for reflectance inputs. Values outside that
range are reported as a warning rather than silently clipped, because out-of-range
NDVI is a *symptom*: wrong scaling, wrong bands, or unmasked clouds.

INVALID PIXELS
--------------
A pixel is invalid if it is nodata, NaN, +/-inf, or has a denominator of
(near) zero. Invalid pixels are stored as NaN **and** flagged in a boolean mask.
They are never turned into 0.0 -- a zero would be read as "bare rock" by any
downstream consumer, including a future classifier.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import Affine

from .index_definitions import IndexDefinition, get_index
from .models import AnalysisResult
from .reflectance import RAW_SPEC, ReflectanceSpec, detect_reflectance_spec
from .statistics import count_summary, describe_valid, fraction_within

NDVI_NAME = "ndvi"
NDVI_LABEL = "NDVI (Normalised Difference Vegetation Index)"
NDVI_METHOD = "NDVI = (NIR - Red) / (NIR + Red)   [Rouse et al. 1974; Tucker 1979]"
NDVI_DESCRIPTION = (
    "A normalised ratio of near-infrared and red surface reflectance. Dense, "
    "photosynthetically active vegetation reflects NIR strongly while absorbing red "
    "light, giving high positive values; water absorbs NIR and gives negative values."
)
NDVI_RANGE: Tuple[float, float] = (-1.0, 1.0)
DEFAULT_MIN_DENOMINATOR = 1e-6


# --------------------------------------------------------------------------- #
# masking
# --------------------------------------------------------------------------- #
def build_valid_mask(
    *bands: np.ndarray, nodatas: Sequence[Optional[float]]
) -> np.ndarray:
    """AND of per-band validity: finite, not nodata.

    `nodatas` is one entry per band (None means the band declares no nodata).
    NaN nodata is handled: any NaN pixel is invalid regardless.
    """
    if not bands:
        raise ValueError("build_valid_mask needs at least one band")
    mask = np.ones(bands[0].shape, dtype=bool)
    for band, nodata in zip(bands, nodatas):
        band = np.asarray(band)
        m = np.isfinite(band)                       # kills NaN and +/-inf
        if nodata is not None:
            if isinstance(nodata, float) and np.isnan(nodata):
                m = m & ~np.isnan(band)
            else:
                m = m & (band != nodata)
        mask = mask & m
    return mask


# --------------------------------------------------------------------------- #
# the generic index engine (Phase 11)
# --------------------------------------------------------------------------- #
def compute_index(
    definition: IndexDefinition,
    bands: Dict[str, Any],
    *,
    transform: Optional[Affine] = None,
    crs: Any = None,
    nodatas: Optional[Dict[str, Optional[float]]] = None,
    labels: Optional[Dict[str, str]] = None,
    reflectance: Optional[ReflectanceSpec] = None,
    min_denominator: Optional[float] = None,
    provenance: Optional[Dict[str, Any]] = None,
    apply_reflectance: bool = True,
) -> AnalysisResult:
    """Compute ANY configured index from role-keyed bands.

    This is the ONE implementation every index uses. `definition` (from
    config/indices/<name>.yml) supplies the roles, the formula shape, the
    denominator guard, the value range, the colormap and the wording; the
    caller supplies the arrays keyed by ROLE ("red", "green", "nir", ...).

    Ordering is taken from `definition.roles`, so `bands_used` and the valid
    mask are reported in the same order for every index -- and for NDVI that
    order is exactly what Phase 3 produced.

    Raises:
        KeyError:  a required role is missing from `bands`.
        ValueError: the two bands of a normalised difference differ in shape.
                    Subtracting them would be meaningless; the caller must
                    resample onto one grid and record that it did.
    """
    roles = list(definition.roles)
    missing = [r for r in roles if r not in bands]
    if missing:
        raise KeyError(
            f"{definition.short_name} needs band role(s) {', '.join(missing)}; "
            f"got {', '.join(bands) or 'no bands'}."
        )

    arrays: Dict[str, Any] = {r: np.asarray(bands[r]) for r in roles}
    a_role = definition.numerator_role           # e.g. NIR for NDVI, GREEN for NDWI
    b_role = definition.denominator_role         # e.g. RED for NDVI, NIR for NDWI
    for role in (a_role, b_role):
        if role not in arrays:
            raise KeyError(f"{definition.short_name} needs band role {role!r}.")

    a = arrays[a_role]
    b = arrays[b_role]
    if a.shape != b.shape:
        raise ValueError(
            f"{definition.role_label(b_role)} and {definition.role_label(a_role)} "
            f"bands must be the same shape, got {b.shape} vs {a.shape}. "
            "Resample them onto a common grid first (and record the method used)."
        )

    nodatas = nodatas or {}
    labels = labels or {}
    guard = float(definition.min_denominator if min_denominator is None else min_denominator)

    spec = reflectance if (apply_reflectance and reflectance is not None) else None
    if spec is None or not spec.is_reflectance:
        spec = RAW_SPEC if spec is None else spec

    # ---- 1. mask BEFORE any arithmetic (nodata is defined on raw DN) ------- #
    mask = build_valid_mask(*[arrays[r] for r in roles],
                            nodatas=[nodatas.get(r) for r in roles])

    # ---- 2. to reflectance (float32), invalid positions NaN ---------------- #
    if spec.is_reflectance and not (spec.scale == 1.0 and spec.offset == 0.0):
        values = {r: spec.apply(arrays[r]) for r in roles}
    else:
        values = {r: arrays[r].astype(np.float32, copy=False) for r in roles}

    # ---- 3. the ratio, with a NaN-safe denominator ------------------------- #
    out = np.full(a.shape, np.nan, dtype=np.float32)
    if definition.is_normalized_difference:
        num, den = values[a_role], values[b_role]
        denom = num + den
    else:                                        # pragma: no cover -- future kinds
        raise ValueError(
            f"Index kind {definition.kind!r} is not implemented. Only "
            f"{definition.is_normalized_difference and 'normalized_difference'} "
            f"formulae are computed, so an unsupported index is never approximated."
        )

    finite_denom = np.isfinite(denom) & (np.abs(denom) >= guard)
    ok = mask & finite_denom

    if np.any(ok):
        with np.errstate(divide="ignore", invalid="ignore"):
            computed = (num[ok] - den[ok]) / denom[ok]
        computed = np.asarray(computed, dtype=np.float32)
        good = np.isfinite(computed)
        idx_ok = np.flatnonzero(ok.ravel())
        out.ravel()[idx_ok[good]] = computed[good]

    final_mask = np.isfinite(out)

    # ---- 4. quality signals (symptoms, not silent fixes) ------------------- #
    caveats: list[str] = []
    counts = count_summary(final_mask)
    stats = describe_valid(out, final_mask)
    lo, hi = definition.value_range

    if stats is not None:
        vals = out[final_mask]
        out_of_range = int(np.count_nonzero((vals < lo - 1e-6) | (vals > hi + 1e-6)))
        if out_of_range:
            caveats.append(
                f"{out_of_range:,} pixel(s) fall outside the theoretical "
                f"{definition.short_name} range [{lo:g}, {hi:g}]. That is a symptom -- "
                f"check band mapping, reflectance scaling and cloud masking. "
                f"Values are reported unclipped."
            )
    else:
        caveats.append(
            f"No valid pixels: no {definition.short_name} statistics are reported."
        )

    if not spec.is_reflectance:
        caveats.append(
            "Inputs were treated as raw digital numbers (no reflectance scaling known). "
            f"{definition.short_name} is a ratio, so it is unaffected by a purely "
            "multiplicative scaling, but an unknown additive offset WOULD bias it."
        )
    caveats.extend(spec.warnings)

    observed: Optional[Tuple[float, float]] = (
        (stats["min"], stats["max"]) if stats else None
    )
    method = f"{definition.short_name} = {definition.formula}"
    if definition.citation:
        method = f"{method}   [{definition.citation}]"

    return AnalysisResult(
        name=definition.name,
        label=definition.label,
        description=definition.description,
        method=method,
        array=out,
        mask=final_mask,
        crs=crs,
        transform=transform,
        nodata=float("nan"),
        value_range=(lo, hi),
        observed_range=observed,
        colormap=definition.colormap,
        stats=stats,
        counts=counts,
        reflectance=spec,
        bands_used={r: labels.get(r, r) for r in roles},
        provenance=dict(provenance or {}),
        caveats=tuple(caveats),
    )


def index_from_dataset(
    ds: Any,
    definition: IndexDefinition,
    band_indexes: Dict[str, int],
    *,
    window: Any = None,
    reflectance: Optional[ReflectanceSpec] = None,
    profile: Optional[str] = None,
    auto_detect_reflectance: bool = False,
    provenance: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """Read the configured roles from a rasterio dataset and compute the index.

    `window` makes this ROI-first: only the requested pixel window is read, so a
    5 km ROI never pulls the whole 20 km scene into memory.

    Band ROLES must already be resolved (error codes are not). Callers that
    cannot establish them must refuse before calling this.
    """
    from rasterio.windows import transform as window_transform

    missing = [r for r in definition.roles if r not in band_indexes]
    if missing:
        raise KeyError(
            f"{definition.short_name} needs band role(s) {', '.join(missing)}; "
            f"got {', '.join(band_indexes) or 'none'}."
        )
    for role, idx in band_indexes.items():
        if not 1 <= int(idx) <= ds.count:
            raise IndexError(
                f"Band {idx} (role {role}) does not exist "
                f"(raster has {ds.count} bands)."
            )
    if len(set(int(i) for i in band_indexes.values())) != len(band_indexes):
        raise ValueError(
            "Two roles point at the same band index; an index needs two "
            "different bands."
        )

    bands = {role: ds.read(int(idx), window=window)
             for role, idx in band_indexes.items()}
    nodatas = {role: ds.nodatavals[int(idx) - 1] for role, idx in band_indexes.items()}
    labels = {role: (ds.descriptions[int(idx) - 1] or f"band {idx}")
              for role, idx in band_indexes.items()}

    spec = reflectance
    report: Dict[str, Any] = {}
    if spec is None and auto_detect_reflectance:
        first_role = definition.roles[0]
        sample = _valid_sample(bands[first_role], nodatas[first_role])
        spec, report = detect_reflectance_spec(ds=ds, profile=profile, dn_sample=sample)
    if spec is None:
        spec = RAW_SPEC

    prov = dict(provenance or {})
    prov.setdefault("bands", {
        role: {
            "index": int(idx),
            "name": labels[role],
            "band_id": definition.band_id(role, profile),
            "wavelength_nm": definition.wavelengths_nm.get(role),
        }
        for role, idx in band_indexes.items()
    })
    prov.setdefault("raster", {
        "width": ds.width, "height": ds.height,
        "crs": str(ds.crs) if ds.crs else None,
        "transform": tuple(ds.transform)[:6], "driver": ds.driver,
        "window": (tuple(window.flatten()) if hasattr(window, "flatten")
                   else (tuple(window.toranges()) if window is not None else None)),
    })
    prov.setdefault("index", definition.to_dict())
    if report:
        prov.setdefault("reflectance_report", report)

    transform = window_transform(window, ds.transform) if window is not None else ds.transform
    return compute_index(
        definition, bands, transform=transform, crs=ds.crs,
        nodatas=nodatas, labels=labels, reflectance=spec, provenance=prov,
    )


# --------------------------------------------------------------------------- #
# NDVI
# --------------------------------------------------------------------------- #
def compute_ndvi(
    red: np.ndarray,
    nir: np.ndarray,
    transform: Optional[Affine] = None,
    crs: Any = None,
    red_nodata: Optional[float] = None,
    nir_nodata: Optional[float] = None,
    reflectance: Optional[ReflectanceSpec] = None,
    red_label: str = "red",
    nir_label: str = "nir",
    provenance: Optional[Dict[str, Any]] = None,
    min_denominator: float = DEFAULT_MIN_DENOMINATOR,
    apply_reflectance: bool = True,
) -> AnalysisResult:
    """Compute NDVI from two aligned bands. Pure function: arrays in, result out.

    Phase 11: this is now a thin, faithful delegate to `compute_index()` with the
    NDVI definition from config/indices/ndvi.yml. The signature, the returned
    `AnalysisResult` and every string in it are unchanged -- verified by
    tests/test_phase11_ndwi.py::test_ndvi_delegates_with_identical_output and by
    the whole Phase 1-10 suite. NDVI stays the reference index: nothing about
    its semantics moved into configuration.

    Args:
        red, nir:      aligned 2-D arrays (same shape, same grid).
        transform/crs: georeferencing of those arrays, carried into the result.
        red/nir_nodata: per-band nodata (None if the file declares none).
        reflectance:   ReflectanceSpec; when None or `apply_reflectance` is False
                       the inputs are assumed to be in comparable units already.
        min_denominator: |NIR+Red| below this is undefined, not zero.

    Returns:
        AnalysisResult with `array` (NaN where invalid), `mask`, stats, counts,
        provenance and caveats.
    """
    return compute_index(
        get_index(NDVI_NAME),
        {"red": red, "nir": nir},
        transform=transform,
        crs=crs,
        nodatas={"red": red_nodata, "nir": nir_nodata},
        labels={"red": red_label, "nir": nir_label},
        reflectance=reflectance,
        min_denominator=min_denominator,
        provenance=provenance,
        apply_reflectance=apply_reflectance,
    )


def ndvi_from_dataset(
    ds: rasterio.DatasetReader,
    red_index: int,
    nir_index: int,
    reflectance: Optional[ReflectanceSpec] = None,
    profile: Optional[str] = None,
    auto_detect_reflectance: bool = True,
    provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[AnalysisResult, ReflectanceSpec, Dict[str, Any]]:
    """Read two bands at NATIVE resolution and compute NDVI.

    This is the entry point the future query router will call. It returns
    (AnalysisResult, ReflectanceSpec, validation_report) so the caller can show
    how the scaling was chosen and whether it survived validation.
    """
    if red_index == nir_index:
        raise ValueError("Red and NIR must be different bands.")
    for idx in (red_index, nir_index):
        if not 1 <= idx <= ds.count:
            raise IndexError(f"Band {idx} does not exist (raster has {ds.count} bands).")

    red = ds.read(red_index)
    nir = ds.read(nir_index)

    spec = reflectance
    report: Dict[str, Any] = {}
    if spec is None and auto_detect_reflectance:
        sample = _valid_sample(red, ds.nodatavals[red_index - 1])
        spec, report = detect_reflectance_spec(ds=ds, profile=profile, dn_sample=sample)
    if spec is None:
        spec = RAW_SPEC

    prov = dict(provenance or {})
    prov.setdefault("bands", {
        "red": {"index": red_index, "name": ds.descriptions[red_index - 1] or f"band {red_index}"},
        "nir": {"index": nir_index, "name": ds.descriptions[nir_index - 1] or f"band {nir_index}"},
    })
    prov.setdefault("raster", {
        "width": ds.width, "height": ds.height, "crs": str(ds.crs) if ds.crs else None,
        "transform": tuple(ds.transform)[:6], "driver": ds.driver,
    })

    result = compute_ndvi(
        red, nir,
        transform=ds.transform,
        crs=ds.crs,
        red_nodata=ds.nodatavals[red_index - 1],
        nir_nodata=ds.nodatavals[nir_index - 1],
        reflectance=spec,
        red_label=prov["bands"]["red"]["name"],
        nir_label=prov["bands"]["nir"]["name"],
        provenance=prov,
    )
    return result, spec, report


def _valid_sample(band: np.ndarray, nodata: Optional[float], cap: int = 200_000) -> np.ndarray:
    b = np.asarray(band)
    m = np.isfinite(b)
    if nodata is not None:
        m = m & (~np.isnan(b) if (isinstance(nodata, float) and np.isnan(nodata)) else (b != nodata))
    vals = b[m]
    if vals.size > cap:
        rng = np.random.default_rng(0)
        vals = vals[rng.choice(vals.size, cap, replace=False)]
    return vals


# --------------------------------------------------------------------------- #
# optional, explicitly ILLUSTRATIVE classification
# --------------------------------------------------------------------------- #
# Default breakpoints are the ones most often quoted in remote-sensing
# teaching material. They are NOT validated for this sensor, region, season or
# crop, and the UI must say so. Anything that depends on them is display only.
ILLUSTRATIVE_BREAKS: Tuple[float, float, float] = (0.0, 0.2, 0.5)
ILLUSTRATIVE_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("water / non-vegetated surface", "NDVI < 0"),
    ("bare soil / built-up", "0 to 0.2"),
    ("sparse / stressed vegetation", "0.2 to 0.5"),
    ("dense vegetation", "> 0.5"),
)


def classify_ndvi(
    array: np.ndarray,
    mask: np.ndarray,
    breaks: Sequence[float] = ILLUSTRATIVE_BREAKS,
    stats: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Approximate, ILLUSTRATIVE class shares over valid pixels.

    Takes (array, mask) rather than an AnalysisResult so it stays usable on
    decimated copies and ROI subsets.

    Returns None when there are no valid pixels. The returned dict always
    carries the caveat string, so it cannot be quoted without it.
    """
    if stats is None:
        return None
    b = list(breaks)
    if len(b) != 3 or b != sorted(b):
        raise ValueError("breaks must be three ascending values")
    # Half-open bins [low, high) so a value landing exactly on a breakpoint is
    # counted once. The top bin keeps its upper edge (<= 1.0).
    shares = {
        "water_or_non_vegetated": fraction_within(array, mask, -1.0, b[0], include_high=False)
        if b[0] > -1.0
        else fraction_within(array, mask, -1.0, b[0]),
        "bare_or_built": fraction_within(array, mask, b[0], b[1], include_high=False),
        "sparse_vegetation": fraction_within(array, mask, b[1], b[2], include_high=False),
        "dense_vegetation": fraction_within(array, mask, b[2], 1.0),
    }
    return {
        "breaks": b,
        "shares": shares,
        "classes": [
            {"name": name, "range": rng, "share": shares[key]}
            for (name, rng), key in zip(
                ILLUSTRATIVE_CLASSES,
                ("water_or_non_vegetated", "bare_or_built", "sparse_vegetation", "dense_vegetation"),
            )
        ],
        "caveat": (
            "ILLUSTRATIVE ONLY. These breakpoints are generic teaching defaults "
            f"({b[0]}, {b[1]}, {b[2]}) and are NOT validated for this sensor, region, "
            "season or crop type. Do not use them for decisions without a documented, "
            "region-specific source."
        ),
    }
