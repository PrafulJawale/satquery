"""Phase 11 -- NDWI: the second index, on the first (and only) index engine.

The point of these tests is not "NDWI works". It is that:

  * NDWI is computed by the SAME code as NDVI, from configuration;
  * NDVI itself did not move (see the delegation test below);
  * the engine refuses rather than guesses when band roles are unknown;
  * nothing anywhere converts NDWI into a water/flood claim;
  * the queries Phase 11 must NOT answer stay unanswered.

Synthetic rasters are used for the unit tests. Real data is used ONLY for the
hand-check and the window tests, never as a substitute for them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from analyses.base import AnalysisContext, IndexContext, Status
from analyses.ndwi import NDWI_CAVEAT, OVERSIZED_MESSAGE, run_ndwi_roi_stats
from core.alignment import DEFAULT_MAX_CELLS
from core.bands import guess_band_roles
from core.index_definitions import (
    IndexDefinition,
    all_indices,
    available_indices,
    get_index,
    load_index_config,
)
from core.indices import compute_index, compute_ndvi, index_from_dataset
from core.raster import describe_path
from core.reflectance import RAW_SPEC, ReflectanceSpec
from core.router import Intent, PLANNED_INTENTS, parse_query
from core.roi import ROISelection
from core.statistics import ROINDVIStats

SCENE = os.path.join("data", "sample", "s2_s2b-36ruv-20230806-0-l2a_2048px.tif")

EPSG = 32636
RES = 10.0
WEST, NORTH = 600000.0, 3400000.0


# =========================================================================== #
# helpers -- synthetic scenes so the unit tests do not depend on real data
# =========================================================================== #
def _write_scene(path: str,
                 bands: Dict[str, np.ndarray],
                 descriptions: Any = None,
                 nodata: Any = 0,
                 dtype: str = "uint16") -> str:
    """Write a small 4-band Sentinel-2-like GeoTIFF (B02/B03/B04/B08)."""
    order = ["blue", "green", "red", "nir"]
    stack = np.stack([np.asarray(bands[r], dtype=dtype) for r in order])
    h, w = stack.shape[1:]
    transform = from_origin(WEST, NORTH, RES, RES)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=4, dtype=dtype,
        crs=f"EPSG:{EPSG}", transform=transform, nodata=nodata,
    ) as ds:
        ds.write(stack)
        if descriptions is None:
            # The real Sentinel-2 naming: B05 would be red-edge, not NIR, and a
            # wrong name here would make the role test meaningless.
            descriptions = ["B02_blue_490nm", "B03_green_560nm",
                            "B04_red_665nm", "B08_nir_842nm"]
        for i, d in enumerate(descriptions, start=1):
            ds.set_band_description(i, d)
    return path


def _scene(tmp_path, green, nir, **kw) -> str:
    blue = np.zeros_like(np.asarray(green))
    red = np.zeros_like(np.asarray(green))
    return _write_scene(str(tmp_path / "s.tif"),
                        {"blue": blue, "green": green, "red": red, "nir": nir}, **kw)


def _real_roi(offset_m: float, size_m: float) -> ROISelection:
    """A square ROI inside the REAL sample scene, offset from its top-left."""
    with rasterio.open(SCENE) as ds:
        minx, maxy = ds.bounds.left, ds.bounds.top
    return _roi(minx + offset_m, maxy - offset_m - size_m, size_m)


def _roi(minx: float, miny: float, size_m: float) -> ROISelection:
    geom = box(minx, miny, minx + size_m, miny + size_m)
    return ROISelection(is_valid=True, intersects_raster=True,
                        area_m2=size_m * size_m,
                        geometry_raster_crs=geom, raster_crs=f"EPSG:{EPSG}")


def _ctx(path: str, roles: Dict[str, int] = None, **kw) -> IndexContext:
    return IndexContext(
        path=path,
        roles=roles if roles is not None else {"green": 2, "nir": 4},
        scale=kw.pop("scale", 0.0001), offset=kw.pop("offset", 0.0),
        is_reflectance=kw.pop("is_reflectance", True),
        profile=kw.pop("profile", "sentinel-2-l2a"),
        source_label=kw.pop("source_label", path),
        reflectance_source=kw.pop("reflectance_source", "detected"),
        role_confidence=kw.pop("role_confidence", "high"),
        role_evidence=kw.pop("role_evidence", ("metadata",)),
    )


def _query(text: str = "What is the NDWI of this area?"):
    return parse_query(text)


# =========================================================================== #
# 1. configuration / abstraction
# =========================================================================== #
def test_ndwi_config_exists_and_is_complete():
    cfg = load_index_config("ndwi")
    for key in ("name", "formula", "required_roles", "min_denominator",
                "value_range", "description", "limitations", "version"):
        assert key in cfg, f"config/indices/ndwi.yml is missing {key}"


def test_ndwi_definition_matches_the_requested_formula():
    d = get_index("ndwi")
    assert d.numerator_role == "green"
    assert d.denominator_role == "nir"
    assert d.formula == "(GREEN - NIR) / (GREEN + NIR)"
    assert d.is_normalized_difference
    assert d.band_id("green", "sentinel-2-l2a") == "B03"
    assert d.band_id("nir", "sentinel-2-l2a") == "B08"


def test_denominator_guard_is_the_inherited_ndvi_convention():
    """Not invented for NDWI: the same 1e-6 NDVI has used since Phase 3."""
    assert get_index("ndwi").min_denominator == pytest.approx(1e-6)
    assert get_index("ndvi").min_denominator == pytest.approx(1e-6)


def test_reflectance_convention_is_the_documented_sentinel2_one():
    d = get_index("ndwi")
    assert d.reflectance_scale == pytest.approx(0.0001)
    assert d.reflectance_offset == pytest.approx(0.0)


def test_no_water_threshold_is_hidden_in_the_configuration():
    """Turning NDWI into 'water if > X' is a claim Phase 11 does not make."""
    cfg = load_index_config("ndwi")
    blob = str(cfg).lower()
    for banned in ("water_threshold", "is_water", "water_class", "flood_threshold",
                   "water_min", "water_cutoff"):
        assert banned not in blob, f"{banned} would make NDWI a water classifier"
    d = get_index("ndwi")
    assert not any("water if" in x.lower() for x in d.limitations)


def test_both_indices_are_served_by_one_registry():
    names = set(available_indices())
    assert {"ndvi", "ndwi"} <= names
    definitions = all_indices()
    assert isinstance(definitions["ndwi"], IndexDefinition)


# =========================================================================== #
# 2. THE DELEGATION GATE -- NDVI must not have moved
# =========================================================================== #
def _reference_ndvi(red, nir, transform=None, crs=None, red_nodata=None,
                    nir_nodata=None, reflectance=None, red_label="red",
                    nir_label="nir", provenance=None,
                    min_denominator=1e-6, apply_reflectance=True):
    """The Phase 3 implementation, verbatim, before Phase 11 touched it.

    Kept here as an executable record: `compute_ndvi` must remain
    indistinguishable from this, field by field, including every string.
    """
    from core.indices import build_valid_mask
    from core.statistics import count_summary, describe_valid

    red = np.asarray(red)
    nir = np.asarray(nir)
    if red.shape != nir.shape:
        raise ValueError(
            f"Red and NIR bands must be the same shape, got {red.shape} vs {nir.shape}. "
            "Resample them onto a common grid first (and record the method used)."
        )
    spec = reflectance if (apply_reflectance and reflectance is not None) else None
    if spec is None or not spec.is_reflectance:
        spec = RAW_SPEC if spec is None else spec

    mask = build_valid_mask(red, nir, nodatas=(red_nodata, nir_nodata))
    red_f = red.astype(np.float32, copy=False)
    nir_f = nir.astype(np.float32, copy=False)
    if spec.is_reflectance and not (spec.scale == 1.0 and spec.offset == 0.0):
        red_r = spec.apply(red_f)
        nir_r = spec.apply(nir_f)
    else:
        red_r = red_f.astype(np.float32, copy=False)
        nir_r = nir_f.astype(np.float32, copy=False)

    out = np.full(red.shape, np.nan, dtype=np.float32)
    denom = nir_r + red_r
    finite_denom = np.isfinite(denom) & (np.abs(denom) >= min_denominator)
    ok = mask & finite_denom
    if np.any(ok):
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (nir_r[ok] - red_r[ok]) / denom[ok]
        values = np.asarray(values, dtype=np.float32)
        good = np.isfinite(values)
        idx_ok = np.flatnonzero(ok.ravel())
        out.ravel()[idx_ok[good]] = values[good]

    final_mask = np.isfinite(out)
    caveats = []
    counts = count_summary(final_mask)
    stats = describe_valid(out, final_mask)
    if stats is not None:
        vals = out[final_mask]
        out_of_range = int(np.count_nonzero((vals < -1.0 - 1e-6) | (vals > 1.0 + 1e-6)))
        if out_of_range:
            caveats.append(
                f"{out_of_range:,} pixel(s) fall outside the theoretical NDVI range "
                f"[-1, 1]. That is a symptom -- check band mapping, reflectance scaling "
                f"and cloud masking. Values are reported unclipped."
            )
    else:
        caveats.append("No valid pixels: no NDVI statistics are reported.")
    if not spec.is_reflectance:
        caveats.append(
            "Inputs were treated as raw digital numbers (no reflectance scaling known). "
            "NDVI is a ratio, so it is unaffected by a purely multiplicative scaling, "
            "but an unknown additive offset WOULD bias it."
        )
    caveats.extend(spec.warnings)
    from core.models import AnalysisResult
    return AnalysisResult(
        name="ndvi",
        label="NDVI (Normalised Difference Vegetation Index)",
        description=get_index("ndvi").description,
        method="NDVI = (NIR - Red) / (NIR + Red)   [Rouse et al. 1974; Tucker 1979]",
        array=out, mask=final_mask, crs=crs, transform=transform,
        nodata=float("nan"), value_range=(-1.0, 1.0),
        observed_range=((stats["min"], stats["max"]) if stats else None),
        colormap="RdYlGn", stats=stats, counts=counts, reflectance=spec,
        bands_used={"red": red_label, "nir": nir_label},
        provenance=dict(provenance or {}), caveats=tuple(caveats),
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ndvi_delegates_with_identical_output(seed):
    """compute_ndvi() == the historical implementation, exactly.

    This is the gate on refactoring NDVI onto the generic engine: if this fails,
    NDVI moves back to being its own implementation.
    """
    rng = np.random.default_rng(seed)
    red = np.concatenate([
        rng.integers(1, 8000, size=(32, 32), dtype="uint16"),
        np.zeros((4, 32), dtype="uint16"),          # nodata row
    ])
    nir = np.concatenate([
        rng.integers(1, 8000, size=(32, 32), dtype="uint16"),
        np.zeros((4, 32), dtype="uint16"),
    ])
    spec = ReflectanceSpec(scale=0.0001, offset=0.0, profile="sentinel-2-l2a",
                           source="detected", is_reflectance=True)

    a = compute_ndvi(red, nir, red_nodata=0, nir_nodata=0, reflectance=spec,
                     red_label="B04_red_665nm", nir_label="B08_nir_842nm",
                     provenance={"raster": "x"})
    b = _reference_ndvi(red, nir, red_nodata=0, nir_nodata=0, reflectance=spec,
                        red_label="B04_red_665nm", nir_label="B08_nir_842nm",
                        provenance={"raster": "x"})

    assert a.name == b.name
    assert a.label == b.label
    assert a.method == b.method
    assert a.colormap == b.colormap
    assert a.value_range == b.value_range
    assert a.bands_used == b.bands_used
    assert list(a.caveats) == list(b.caveats)
    assert a.counts == b.counts
    assert a.stats == b.stats
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_array_equal(a.array, b.array)      # bit-for-bit


def test_ndvi_shape_mismatch_message_is_unchanged():
    with pytest.raises(ValueError, match="Red and NIR bands must be the same shape"):
        compute_ndvi(np.zeros((4, 4)), np.zeros((5, 5)))


def test_ndvi_out_of_range_caveat_wording_is_unchanged():
    # reflectance > 1 forces values outside [-1, 1] without any scaling
    red = np.full((2, 2), -1.0, dtype="float32")       # (5 - -1)/(5 + -1) = 1.5
    nir = np.full((2, 2), 5.0, dtype="float32")
    r = compute_ndvi(red, nir, reflectance=None, apply_reflectance=False)
    assert any("fall outside the theoretical NDVI range [-1, 1]" in c for c in r.caveats)


def test_compute_index_and_compute_ndvi_agree():
    rng = np.random.default_rng(7)
    red = rng.integers(1, 6000, size=(16, 16), dtype="uint16")
    nir = rng.integers(1, 6000, size=(16, 16), dtype="uint16")
    spec = ReflectanceSpec(scale=0.0001, offset=0.0, profile="sentinel-2-l2a",
                           source="detected", is_reflectance=True)
    a = compute_ndvi(red, nir, reflectance=spec)
    b = compute_index(get_index("ndvi"), {"red": red, "nir": nir}, reflectance=spec)
    np.testing.assert_array_equal(a.array, b.array)
    assert a.method == b.method


# =========================================================================== #
# 3. NDWI numerical behaviour
# =========================================================================== #
def _ndwi_of(green_dn, nir_dn, **kw):
    return compute_index(get_index("ndwi"), {"green": green_dn, "nir": nir_dn}, **kw)


def test_ndwi_formula_is_correct():
    out = _ndwi_of(np.array([[1000.0]]), np.array([[500.0]]),
                   reflectance=None, apply_reflectance=False)
    assert out.array[0, 0] == pytest.approx((1000 - 500) / (1000 + 500))


def test_ndwi_hand_computed_values_match():
    pairs = [((1000.0, 500.0), 1 / 3), ((500.0, 1000.0), -1 / 3),
             ((1000.0, 1000.0), 0.0), ((0.0, 1000.0), -1.0),
             ((3000.0, 1000.0), 0.5)]
    for (g, n), expected in pairs:
        out = _ndwi_of(np.array([[g]]), np.array([[n]]),
                       reflectance=None, apply_reflectance=False)
        assert out.array[0, 0] == pytest.approx(expected, abs=1e-7), (g, n)


def test_reflectance_scaling_is_applied_before_the_ratio():
    # DN 2000 / 10000 = 0.2 green, DN 1000 -> 0.1 nir (a ratio is scale-invariant
    # for a pure scale, so this proves the pipeline ran, not the value)
    spec = ReflectanceSpec(scale=0.0001, offset=0.0, profile="sentinel-2-l2a",
                           source="detected", is_reflectance=True)
    out = _ndwi_of(np.array([[2000]], dtype="uint16"),
                   np.array([[1000]], dtype="uint16"), reflectance=spec)
    assert out.array[0, 0] == pytest.approx(1 / 3, abs=1e-6)


def test_an_additive_offset_changes_the_result():
    """Proof the offset really is applied -- a scale alone would not move NDWI."""
    spec = ReflectanceSpec(scale=1.0, offset=0.05, profile=None,
                           source="user_supplied", is_reflectance=True)
    out = _ndwi_of(np.array([[0.1]]), np.array([[0.1]]), reflectance=spec)
    assert out.array[0, 0] == pytest.approx(0.0)          # (0.15-0.15)/(0.30)


def test_denominator_guard_is_not_a_zero_division():
    out = _ndwi_of(np.array([[1e-7]]), np.array([[-1e-7]]),
                   reflectance=None, apply_reflectance=False)
    assert np.isnan(out.array[0, 0])
    assert not out.mask[0, 0]


def test_zero_over_zero_is_invalid_not_zero():
    out = _ndwi_of(np.array([[0.0]]), np.array([[0.0]]),
                   reflectance=None, apply_reflectance=False)
    assert np.isnan(out.array[0, 0])
    assert not out.mask[0, 0]


def test_negative_values_are_allowed_when_they_are_the_answer():
    out = _ndwi_of(np.array([[100.0]]), np.array([[900.0]]),
                   reflectance=None, apply_reflectance=False)
    assert out.array[0, 0] == pytest.approx(-0.8)


def test_constant_values_produce_a_constant_index():
    g = np.full((4, 4), 2000, dtype="uint16")
    n = np.full((4, 4), 1000, dtype="uint16")
    out = _ndwi_of(g, n, reflectance=ReflectanceSpec(
        scale=0.0001, offset=0.0, is_reflectance=True, source="detected"))
    assert np.allclose(out.array[~np.isnan(out.array)], 1 / 3, atol=1e-6)


def test_nodata_makes_a_pixel_invalid_on_both_roles():
    g = np.array([[1000, 0, 1000]], dtype="uint16")      # 0 = nodata
    n = np.array([[1000, 1000, 0]], dtype="uint16")
    out = _ndwi_of(g, n, nodatas={"green": 0, "nir": 0},
                   reflectance=None, apply_reflectance=False)
    assert out.mask[0, 0]
    assert not out.mask[0, 1]          # green nodata
    assert not out.mask[0, 2]          # nir nodata


def test_nan_inputs_are_invalid():
    g = np.array([[np.nan, 1000.0]])
    n = np.array([[1000.0, 1000.0]])
    out = _ndwi_of(g, n, reflectance=None, apply_reflectance=False)
    assert not out.mask[0, 0]
    assert out.mask[0, 1]


def test_valid_pixels_are_the_intersection_of_both_bands():
    g = np.array([[1000, 1000, 0, 1000]], dtype="uint16")
    n = np.array([[1000, 0, 1000, 1000]], dtype="uint16")
    out = _ndwi_of(g, n, nodatas={"green": 0, "nir": 0},
                   reflectance=None, apply_reflectance=False)
    assert out.mask.tolist() == [[True, False, False, True]]


def test_statistics_describe_only_the_valid_pixels():
    g = np.array([[1000, 0, 3000, 2000]], dtype="uint16")
    n = np.array([[1000, 500, 1000, 1000]], dtype="uint16")
    out = _ndwi_of(g, n, nodatas={"green": 0, "nir": 0},
                   reflectance=None, apply_reflectance=False)
    valid = out.array[out.mask]
    assert out.stats["min"] == pytest.approx(valid.min())
    assert out.stats["max"] == pytest.approx(valid.max())
    assert out.stats["mean"] == pytest.approx(valid.mean())
    assert out.stats["median"] == pytest.approx(np.median(valid))
    assert out.stats["std"] == pytest.approx(float(valid.std()))
    assert out.counts["valid_pixels"] == int(valid.size)


def test_ndwi_and_ndvi_share_one_engine_but_not_one_answer():
    """Same code path, different roles: the two indices must not be confusable."""
    g = np.array([[1000]], dtype="uint16")
    n = np.array([[500]], dtype="uint16")
    spec = ReflectanceSpec(scale=0.0001, offset=0.0, is_reflectance=True,
                           source="detected")
    ndwi = compute_index(get_index("ndwi"), {"green": g, "nir": n}, reflectance=spec)
    ndvi = compute_index(get_index("ndvi"), {"nir": n, "red": g}, reflectance=spec)
    assert ndwi.array[0, 0] == pytest.approx(1 / 3)
    assert ndvi.array[0, 0] == pytest.approx(-1 / 3)      # (NIR-Red)/(NIR+Red)


def test_index_from_dataset_reads_only_the_requested_window(tmp_path):
    path = _scene(tmp_path, green=np.full((64, 64), 2000, dtype="uint16"),
                  nir=np.full((64, 64), 1000, dtype="uint16"))
    spec = ReflectanceSpec(scale=0.0001, offset=0.0, is_reflectance=True,
                           source="detected")
    with rasterio.open(path) as ds:
        full = index_from_dataset(ds, get_index("ndwi"), {"green": 2, "nir": 4},
                                  reflectance=spec)
        win = rasterio.windows.Window(10, 10, 8, 8)
        part = index_from_dataset(ds, get_index("ndwi"), {"green": 2, "nir": 4},
                                  window=win, reflectance=spec)
    assert full.array.shape == (64, 64)
    assert part.array.shape == (8, 8)
    assert part.provenance["raster"]["window"] is not None
    np.testing.assert_allclose(part.array, full.array[10:18, 10:18], atol=1e-6)


# =========================================================================== #
# 4. band handling
# =========================================================================== #
def test_b03_and_b08_are_selected_from_band_metadata(tmp_path):
    path = _scene(tmp_path, green=np.full((8, 8), 2000, dtype="uint16"),
                  nir=np.full((8, 8), 1000, dtype="uint16"))
    guess = guess_band_roles(describe_path(path))
    assert guess.roles.get("green") == 2
    assert guess.roles.get("nir") == 4
    assert get_index("ndwi").band_id("green", guess.profile) == "B03"
    assert get_index("ndwi").band_id("nir", guess.profile) == "B08"


def test_missing_green_role_is_refused_not_guessed(tmp_path):
    path = _scene(tmp_path, green=np.full((8, 8), 2000, dtype="uint16"),
                  nir=np.full((8, 8), 1000, dtype="uint16"))
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=_roi(WEST, NORTH - 80.0, 80.0),
                        index_context=_ctx(path, roles={"nir": 4})),
        _query())
    assert ex.status is Status.UNSUPPORTED
    assert "green" in ex.message.lower()
    assert ex.result is None


def test_missing_nir_role_is_refused_not_guessed(tmp_path):
    path = _scene(tmp_path, green=np.full((8, 8), 2000, dtype="uint16"),
                  nir=np.full((8, 8), 1000, dtype="uint16"))
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=_roi(WEST, NORTH - 80.0, 80.0),
                        index_context=_ctx(path, roles={"green": 2})),
        _query())
    assert ex.status is Status.UNSUPPORTED
    assert "nir" in ex.message.lower()


def test_unlabelled_bands_produce_no_roles_and_are_refused(tmp_path):
    """Band ORDER is never a role: an unlabelled 4-band file yields no NDWI."""
    path = _scene(tmp_path, green=np.full((8, 8), 2000, dtype="uint16"),
                  nir=np.full((8, 8), 1000, dtype="uint16"),
                  descriptions=["", "", "", ""])
    guess = guess_band_roles(describe_path(path))
    assert "green" not in guess.roles or guess.confidence in ("none", "low")
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=_roi(WEST, NORTH - 80.0, 80.0),
                        index_context=_ctx(path, roles={})),
        _query())
    assert ex.status is Status.UNSUPPORTED


def test_no_index_context_at_all_is_refused(tmp_path):
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=_roi(WEST, NORTH - 80.0, 80.0)), _query())
    assert ex.status is Status.UNSUPPORTED


# =========================================================================== #
# 5. ROI behaviour
# =========================================================================== #
def test_roi_missing_returns_needs_roi(tmp_path):
    path = _scene(tmp_path, green=np.full((8, 8), 2000, dtype="uint16"),
                  nir=np.full((8, 8), 1000, dtype="uint16"))
    ex = run_ndwi_roi_stats(AnalysisContext(index_context=_ctx(path)), _query())
    assert ex.status is Status.NEEDS_ROI
    assert "select an area" in ex.message.lower()


def test_statistics_cover_only_the_roi(tmp_path):
    """Pixel-CENTRE inclusion over a 10x10 m box: exactly 8x8 cells."""
    path = _scene(tmp_path, green=np.full((64, 64), 2000, dtype="uint16"),
                  nir=np.full((64, 64), 1000, dtype="uint16"))
    # aligned to the grid: pixel CENTRES (WEST+5, NORTH-5 ...) fall strictly
    # inside, so exactly 8x8 cells are counted -- no boundary ambiguity.
    roi = _roi(WEST, NORTH - 80.0, 80.0)
    ex = run_ndwi_roi_stats(AnalysisContext(roi=roi, index_context=_ctx(path)), _query())
    assert ex.status is Status.OK
    assert ex.result.pixels_inside_roi == 64
    assert ex.result.valid_pixels == 64
    assert ex.result.stats["mean"] == pytest.approx(1 / 3, abs=1e-6)


def test_roi_outside_the_raster_has_no_pixels(tmp_path):
    path = _scene(tmp_path, green=np.full((8, 8), 2000, dtype="uint16"),
                  nir=np.full((8, 8), 1000, dtype="uint16"))
    roi = _roi(WEST + 100000.0, NORTH + 100000.0, 80.0)
    ex = run_ndwi_roi_stats(AnalysisContext(roi=roi, index_context=_ctx(path)), _query())
    assert ex.status is Status.NO_VALID_PIXELS
    assert ex.result.pixels_inside_roi == 0


def test_small_roi_reads_a_window_not_the_whole_scene(tmp_path):
    path = _scene(tmp_path, green=np.full((256, 256), 2000, dtype="uint16"),
                  nir=np.full((256, 256), 1000, dtype="uint16"))
    roi = _roi(WEST + 1000.0, NORTH - 1100.0, 200.0)     # 20x20 cells
    ex = run_ndwi_roi_stats(AnalysisContext(roi=roi, index_context=_ctx(path)), _query())
    assert ex.status is Status.OK
    w = ex.provenance["roi_window"]
    assert w[2] < 256 and w[3] < 256          # a window, not the scene
    assert ex.provenance["roi_cells"] < 256 * 256


def test_large_roi_is_refused_not_coarsened():
    """The whole 2048x2048 scene is 4.19M cells -- over the 2M budget."""
    roi = _real_roi(0.0, 20480.0)
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=roi, index_context=_ctx(SCENE)), _query())
    assert ex.status is Status.INSUFFICIENT_DATA
    assert "budget" in ex.message.lower()
    assert ex.provenance["roi_cells"] > DEFAULT_MAX_CELLS
    assert ex.provenance["cell_budget"] == DEFAULT_MAX_CELLS
    assert ex.provenance["refused"] == "roi_exceeds_cell_budget"
    assert ex.result is None


def test_result_carries_the_index_name_and_runtime():
    roi = _real_roi(2000.0, 5120.0)
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=roi, index_context=_ctx(SCENE)), _query())
    assert ex.status is Status.OK
    assert ex.result.index_name == "ndwi"
    assert ex.result.runtime_ms > 0
    assert isinstance(ex.result, ROINDVIStats)


def test_ndvi_results_still_report_the_ndvi_index_name():
    """The new field defaults to 'ndvi', so Phase 6 saw no change."""
    assert ROINDVIStats().index_name == "ndvi"


# =========================================================================== #
# 6. provenance
# =========================================================================== #
def test_provenance_records_formula_bands_scale_offset_and_config():
    roi = _real_roi(2000.0, 5120.0)
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=roi, index_context=_ctx(SCENE)), _query())
    p = ex.provenance
    assert p["formula"] == "(GREEN - NIR) / (GREEN + NIR)"
    assert p["citation"] == "McFeeters 1996"
    assert p["bands"]["green"]["band_id"] == "B03"
    assert p["bands"]["nir"]["band_id"] == "B08"
    assert p["reflectance"]["scale"] == pytest.approx(0.0001)
    assert p["reflectance"]["offset"] == pytest.approx(0.0)
    assert p["min_denominator"] == pytest.approx(1e-6)
    assert p["index"]["version"] == "phase11"


# =========================================================================== #
# 7. wording: an index is not a water body
# =========================================================================== #
def test_every_result_carries_the_caveat():
    roi = _real_roi(2000.0, 5120.0)
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=roi, index_context=_ctx(SCENE)), _query())
    assert NDWI_CAVEAT in ex.warnings
    assert "flood" in NDWI_CAVEAT.lower()


def test_the_answer_never_calls_ndwi_water():
    roi = _real_roi(2000.0, 5120.0)
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=roi, index_context=_ctx(SCENE)), _query())
    low = ex.message.lower()
    for banned in ("flood extent", "water body", "water bodies", "is water",
                   "water quality", "water availability", "flooded"):
        assert banned not in low
    assert "ndwi" in low


def test_no_water_classification_is_offered_anywhere():
    d = get_index("ndwi")
    for limitation in d.limitations:
        assert "threshold" not in limitation.lower() or "no water" in limitation.lower()


# =========================================================================== #
# 8. router
# =========================================================================== #
@pytest.mark.parametrize("query", [
    "What is the NDWI of this area?",
    "Calculate NDWI here.",
    "Show the water index for this ROI.",
    "What is the water index of this area?",
    "Show NDWI for this region.",
])
def test_ndwi_queries_route_to_ndwi(query):
    assert parse_query(query).intent is Intent.NDWI_ROI_STATS


@pytest.mark.parametrize("query", [
    "What is the NDVI of this area?",
    "Show vegetation health here.",
    "Calculate the vegetation index here.",
])
def test_ndvi_queries_still_route_to_ndvi(query):
    assert parse_query(query).intent is Intent.NDVI_ROI_STATS


def test_flood_queries_stay_unsupported():
    for q in ("Did flooding happen?", "Show flood areas.",
              "Detect flood change here.", "Flood detection using NDWI"):
        assert parse_query(q).intent is Intent.FLOOD_CHANGE
    assert Intent.FLOOD_CHANGE in PLANNED_INTENTS


@pytest.mark.parametrize("query", [
    "Compare NDWI before and after",
    "Show the NDWI difference between two dates",
    "What is the NDWI change here?",
    "Show me water index change over time",
])
def test_temporal_ndwi_is_recognised_and_refused(query):
    assert parse_query(query).intent is Intent.TEMPORAL_NDWI


@pytest.mark.parametrize("query", [
    "Find cropland near water",
    "Find areas within 500 m of water",
])
def test_water_proximity_stays_a_phase9_spatial_query(query):
    assert parse_query(query).intent is Intent.SPATIAL_QUERY


def test_bare_ndwi_does_not_override_flood_vocabulary():
    assert parse_query("NDWI flood extent").intent is Intent.FLOOD_CHANGE


def test_ambiguous_water_question_is_not_answered_by_ndwi():
    """Generic 'water' is a Phase 9 spatial question, not an index question."""
    assert parse_query("Where is water?").intent is Intent.SPATIAL_QUERY


def test_temporal_ndwi_has_no_handler():
    from analyses.registry import get_spec
    spec = get_spec(Intent.TEMPORAL_NDWI)
    assert spec.handler is None
    assert not spec.available


def test_ndwi_is_registered_and_available():
    from analyses.registry import available_specs, get_spec
    assert Intent.NDWI_ROI_STATS in available_specs()
    spec = get_spec(Intent.NDWI_ROI_STATS)
    assert spec.handler is not None
    assert any("NDWI" in q for q in spec.example_queries)


def test_suggestions_still_include_the_phase8_cotton_example():
    """The NDWI spec is appended last so this cannot be pushed out."""
    from analyses.registry import suggestions
    assert "Can I grow cotton here?" in suggestions(limit=8)


# =========================================================================== #
# 9. REAL DATA: hand-check and a real ROI
# =========================================================================== #
@pytest.mark.skipif(not os.path.exists(SCENE), reason="sample scene not present")
def test_real_data_hand_check_8x8_block():
    """NDWI computed by hand from raw B03/B08 DN, without the engine."""
    ROW, COL, N = 1000, 1000, 8
    with rasterio.open(SCENE) as ds:
        green = ds.read(2, window=rasterio.windows.Window(COL, ROW, N, N)).astype("float64")
        nir = ds.read(4, window=rasterio.windows.Window(COL, ROW, N, N)).astype("float64")
    g, n = green / 10000.0, nir / 10000.0
    den = g + n
    hand = np.where(np.abs(den) > 1e-6, (g - n) / np.where(den == 0, 1.0, den), np.nan)

    # the engine, over an ROI that IS that block (plus the 2-cell read buffer)
    with rasterio.open(SCENE) as ds:
        _minx, _maxy = ds.bounds.left, ds.bounds.top
    x0 = _minx + COL * RES                       # left edge of the block
    y0 = _maxy - (ROW + N) * RES                 # bottom edge of the block
    roi = ROISelection(is_valid=True, intersects_raster=True,
                       area_m2=(N * RES) ** 2,
                       geometry_raster_crs=box(x0, y0, x0 + N * RES, y0 + N * RES),
                       raster_crs="EPSG:32636")
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=roi, index_context=_ctx(SCENE)), _query())
    assert ex.status is Status.OK, ex.message

    engine = np.asarray(ex.result.raster)
    # the ROI sits inside the buffered window: locate the exact block
    sub = engine[2:2 + N, 2:2 + N]
    diff = np.nanmax(np.abs(sub - hand))
    assert diff < 1e-6, f"engine vs hand-computed NDWI differ by {diff:.3e}"


@pytest.mark.skipif(not os.path.exists(SCENE), reason="sample scene not present")
def test_real_data_roi_statistics_are_reported_honestly():
    roi = _real_roi(2000.0, 5120.0)                        # 5.12 km, 512x512 cells
    ex = run_ndwi_roi_stats(
        AnalysisContext(roi=roi, index_context=_ctx(SCENE)), _query())
    assert ex.status is Status.OK
    r = ex.result
    assert r.pixels_inside_roi == 512 * 512
    assert r.valid_pixels == r.pixels_inside_roi
    assert r.valid_fraction == pytest.approx(1.0)
    assert -1.0 <= r.stats["min"] <= r.stats["max"] <= 1.0
    assert r.stats["min"] <= r.stats["mean"] <= r.stats["max"]
    assert r.area_m2 == pytest.approx(5120.0 * 5120.0, rel=1e-3)


@pytest.mark.skipif(not os.path.exists(SCENE), reason="sample scene not present")
def test_real_data_band_roles_come_from_the_scene_metadata():
    guess = guess_band_roles(describe_path(SCENE))
    assert guess.roles.get("green") == 2 and guess.roles.get("nir") == 4
    assert guess.profile == "sentinel-2-l2a"
