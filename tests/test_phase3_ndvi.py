"""Phase 3 tests: NDVI engine, reflectance handling and masking.

Run:  python -m pytest tests -q

These tests encode the failures that matter: invalid pixels silently becoming
zero, metadata offsets that produce impossible reflectance, statistics reported
for an empty scene, and georeferencing lost between bands and result.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio import Affine
from rasterio.io import MemoryFile

from core.bands import guess_band_roles
from core.indices import (
    ILLUSTRATIVE_BREAKS,
    build_valid_mask,
    classify_ndvi,
    compute_ndvi,
    ndvi_from_dataset,
)
from core.preview import analysis_to_geotiff_bytes, decimate_array, ndvi_to_rgba
from core.raster import describe_path, open_dataset
from core.reflectance import (
    RAW_SPEC,
    ReflectanceSpec,
    detect_reflectance_spec,
    spec_from_profile,
    validate_spec,
)
from core.samples import list_samples, sample_path

S2_SAMPLE = next((n for n in list_samples() if n.startswith("s2_")), None)
needs_s2 = pytest.mark.skipif(S2_SAMPLE is None, reason="Sentinel-2 sample not downloaded")

S2_SPEC = spec_from_profile("sentinel-2-l2a")
TRANSFORM = Affine(10.0, 0.0, 400_000.0, 0.0, -10.0, 4_000_000.0)
CRS = "EPSG:32643"


# --------------------------------------------------------------------------- #
# 1. normal calculation with known expected values
# --------------------------------------------------------------------------- #
def test_ndvi_known_values_reflectance_mode():
    red = np.array([[0.2, 0.1], [0.6, 0.0]], dtype=np.float32)
    nir = np.array([[0.5, 0.1], [0.1, 0.4]], dtype=np.float32)
    res = compute_ndvi(red, nir, transform=TRANSFORM, crs=CRS, apply_reflectance=False)

    # (0.5-0.2)/(0.5+0.2) = 0.428571...
    assert res.array[0, 0] == pytest.approx(0.3 / 0.7, abs=1e-6)
    # identical bands -> 0
    assert res.array[0, 1] == pytest.approx(0.0, abs=1e-6)
    # (0.1-0.6)/(0.1+0.6) = -0.714285...
    assert res.array[1, 0] == pytest.approx(-0.5 / 0.7, abs=1e-6)
    # red = 0 -> +1
    assert res.array[1, 1] == pytest.approx(1.0, abs=1e-6)


def test_ndvi_known_values_from_digital_numbers():
    """Sentinel-2 DN -> reflectance, then NDVI (hand-computed)."""
    red = np.array([[2000, 6000]], dtype=np.uint16)     # 0.20, 0.60
    nir = np.array([[5000, 1000]], dtype=np.uint16)     # 0.50, 0.10
    res = compute_ndvi(red, nir, reflectance=S2_SPEC)
    assert res.array[0, 0] == pytest.approx(0.3 / 0.7, abs=1e-6)
    assert res.array[0, 1] == pytest.approx(-0.5 / 0.7, abs=1e-6)


def test_ndvi_is_invariant_to_pure_multiplicative_scaling():
    """Documented design property: with offset 0, DN and reflectance give the same NDVI."""
    red_dn = np.array([[1000, 4000, 7000]], dtype=np.uint16)
    nir_dn = np.array([[8000, 3000, 1000]], dtype=np.uint16)
    from_dn = compute_ndvi(red_dn, nir_dn, reflectance=S2_SPEC)
    from_refl = compute_ndvi(
        red_dn.astype(np.float32) * 1e-4, nir_dn.astype(np.float32) * 1e-4,
        apply_reflectance=False,
    )
    assert np.allclose(from_dn.array, from_refl.array, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. invalid pixels: zero denominator, nodata, NaN, infinity
# --------------------------------------------------------------------------- #
def test_zero_denominator_is_invalid_not_zero():
    """NIR + Red == 0 is undefined. It must NOT become NDVI = 0 (which reads as bare soil)."""
    red = np.array([[0.1, 0.2]], dtype=np.float32)
    nir = np.array([[-0.1, 0.2]], dtype=np.float32)
    res = compute_ndvi(red, nir, apply_reflectance=False)
    assert np.isnan(res.array[0, 0]), "zero denominator must produce NaN, not 0"
    assert res.mask[0, 0] is np.False_
    assert res.array[0, 1] == pytest.approx(0.0)     # this one is genuinely zero
    assert res.mask[0, 1] is np.True_


def test_near_zero_denominator_is_guarded():
    red = np.array([[1e-9]], dtype=np.float32)
    nir = np.array([[-1e-9]], dtype=np.float32)
    res = compute_ndvi(red, nir, apply_reflectance=False, min_denominator=1e-6)
    assert np.isnan(res.array[0, 0])
    assert res.mask[0, 0] is np.False_


def test_nodata_pixels_are_excluded():
    red = np.array([[0, 500, 700]], dtype=np.uint16)      # 0 == nodata
    nir = np.array([[900, 0, 900]], dtype=np.uint16)
    res = compute_ndvi(red, nir, red_nodata=0, nir_nodata=0, reflectance=S2_SPEC)
    assert res.mask[0, 0] is np.False_     # red nodata
    assert res.mask[0, 1] is np.False_     # nir nodata
    assert res.mask[0, 2] is np.True_      # both valid
    assert np.isnan(res.array[0, 0]) and np.isnan(res.array[0, 1])


def test_nan_inputs_are_excluded():
    red = np.array([[np.nan, 0.2]], dtype=np.float32)
    nir = np.array([[0.5, np.nan]], dtype=np.float32)
    res = compute_ndvi(red, nir, apply_reflectance=False)
    assert res.mask.tolist() == [[False, False]]
    assert np.isnan(res.array).all()


def test_infinite_inputs_are_excluded():
    red = np.array([[np.inf, 0.2], [-np.inf, 0.3]], dtype=np.float32)
    nir = np.array([[0.5, np.inf], [0.4, 0.6]], dtype=np.float32)
    res = compute_ndvi(red, nir, apply_reflectance=False)
    assert res.mask[0, 0] is np.False_
    assert res.mask[0, 1] is np.False_
    assert res.mask[1, 0] is np.False_
    assert res.mask[1, 1] is np.True_


def test_nan_nodata_is_handled():
    red = np.array([[np.nan, 0.2]], dtype=np.float32)
    nir = np.array([[0.5, 0.6]], dtype=np.float32)
    res = compute_ndvi(red, nir, red_nodata=float("nan"), apply_reflectance=False)
    assert res.mask[0, 0] is np.False_
    assert res.mask[0, 1] is np.True_


# --------------------------------------------------------------------------- #
# 3. all-invalid raster
# --------------------------------------------------------------------------- #
def test_all_invalid_scene_reports_no_statistics():
    red = np.zeros((4, 4), dtype=np.uint16)      # every pixel is nodata
    nir = np.zeros((4, 4), dtype=np.uint16)
    res = compute_ndvi(red, nir, red_nodata=0, nir_nodata=0, reflectance=S2_SPEC)

    assert res.stats is None, "no statistics may be reported for an all-invalid scene"
    assert res.counts["valid_pixels"] == 0
    assert res.counts["invalid_pixels"] == 16
    assert res.counts["valid_percentage"] == 0.0
    assert np.isnan(res.array).all()
    assert not res.mask.any()
    assert res.has_valid_pixels is False
    assert any("No valid pixels" in c for c in res.caveats)


def test_all_invalid_statistics_never_silently_zero():
    """A mean of 0.0 over an empty scene is the classic fabricated statistic."""
    red = np.full((3, 3), 0, dtype=np.uint16)
    nir = np.full((3, 3), 0, dtype=np.uint16)
    res = compute_ndvi(red, nir, red_nodata=0, nir_nodata=0)
    assert res.stats is None
    assert res.to_dict()["stats"] is None
    assert "no statistics" in " ".join(res.summary_lines()).lower()


# --------------------------------------------------------------------------- #
# 4. reflectance / scaling
# --------------------------------------------------------------------------- #
def test_sentinel2_profile_scaling():
    spec = spec_from_profile("sentinel-2-l2a")
    assert spec.scale == pytest.approx(1e-4)
    assert spec.offset == 0.0
    assert spec.verified is True
    assert np.allclose(spec.apply(np.array([10000], dtype=np.uint16)), [1.0], atol=1e-6)


def test_landsat_profile_applies_real_offset():
    spec = spec_from_profile("landsat-c2-l2")
    assert spec.offset == pytest.approx(-0.2)
    # rho = DN * 2.75e-5 - 0.2
    assert spec.apply(np.array([10000], dtype=np.uint16))[0] == pytest.approx(0.075, abs=1e-6)
    # and NDVI changes accordingly (unlike Sentinel-2, the offset does NOT cancel)
    red = np.array([[10000]], dtype=np.uint16)
    nir = np.array([[30000]], dtype=np.uint16)
    res = compute_ndvi(red, nir, reflectance=spec)
    expected = (0.625 - 0.075) / (0.625 + 0.075)
    assert res.array[0, 0] == pytest.approx(expected, abs=1e-5)


def test_bad_offset_is_rejected_by_physical_sanity_check():
    """The STAC-metadata trap: offset -0.1 makes most pixels negatively reflective."""
    dn = np.array([100, 500, 900, 1500, 3000], dtype=np.uint16)
    bad = ReflectanceSpec(scale=1e-4, offset=-0.1, source="file_tags")
    fixed, report = validate_spec(bad, dn)
    assert report["offset_rejected"] is True
    assert fixed.offset == 0.0
    assert any("rejected" in w.lower() for w in fixed.warnings)


def test_valid_spec_is_left_alone():
    dn = np.array([1000, 3000, 5000, 8000], dtype=np.uint16)
    good = ReflectanceSpec(scale=1e-4, offset=0.0, source="sensor_profile")
    fixed, report = validate_spec(good, dn)
    assert report["offset_rejected"] is False
    assert fixed.offset == 0.0
    assert report["negative_fraction"] == 0.0


def test_positive_reflectance_sanity_check_warns_on_impossible_values():
    dn = np.array([100, 200, 300], dtype=np.uint16)   # tiny DNs
    spec = ReflectanceSpec(scale=10.0, offset=0.0)    # absurd scale
    _fixed, report = validate_spec(spec, dn)
    assert report["implausible_high_fraction"] > 0.0


def test_raw_spec_used_when_nothing_is_known():
    res = compute_ndvi(
        np.array([[0.2]], dtype=np.float32), np.array([[0.5]], dtype=np.float32),
        reflectance=None, apply_reflectance=False,
    )
    assert res.array[0, 0] == pytest.approx(0.3 / 0.7, abs=1e-6)
    assert any("digital numbers" in c for c in res.caveats)


def test_raw_spec_object_is_never_none_in_result():
    res = compute_ndvi(np.array([[0.2]]), np.array([[0.5]]), reflectance=None)
    assert res.reflectance is not None
    assert res.reflectance.is_reflectance is False


# --------------------------------------------------------------------------- #
# 5. mask and metadata preservation
# --------------------------------------------------------------------------- #
def test_mask_matches_nan_positions_exactly():
    red = np.array([[0, 500], [700, 900]], dtype=np.uint16)
    nir = np.array([[900, 0], [800, 700]], dtype=np.uint16)
    res = compute_ndvi(red, nir, red_nodata=0, nir_nodata=0, reflectance=S2_SPEC)
    assert np.array_equal(res.mask, np.isfinite(res.array))


def test_no_invalid_pixel_is_stored_as_zero():
    red = np.array([[0, 500, 700]], dtype=np.uint16)
    nir = np.array([[900, 0, 900]], dtype=np.uint16)
    res = compute_ndvi(red, nir, red_nodata=0, nir_nodata=0, reflectance=S2_SPEC)
    invalid_positions = ~res.mask
    assert not np.any(res.array[invalid_positions] == 0.0)
    assert np.isnan(res.array[invalid_positions]).all()


def test_transform_crs_and_shape_are_preserved():
    red = np.ones((5, 7), dtype=np.uint16) * 1000
    nir = np.ones((5, 7), dtype=np.uint16) * 4000
    res = compute_ndvi(red, nir, transform=TRANSFORM, crs=CRS, reflectance=S2_SPEC)
    assert res.transform == TRANSFORM
    assert str(res.crs) == CRS
    assert res.native_shape == (5, 7)
    d = res.to_dict()
    assert d["transform"] == tuple(TRANSFORM)[:6]
    assert d["crs"] == CRS
    assert d["native_shape"] == [5, 7]


def test_counts_are_reported_even_when_stats_are_not():
    red = np.zeros((2, 2), dtype=np.uint16)
    nir = np.zeros((2, 2), dtype=np.uint16)
    res = compute_ndvi(red, nir, red_nodata=0, nir_nodata=0)
    assert res.counts["total_pixels"] == 4
    assert res.counts["valid_pixels"] == 0
    assert res.counts["invalid_pixels"] == 4


def test_shape_mismatch_is_rejected_loudly():
    with pytest.raises(ValueError, match="same shape"):
        compute_ndvi(np.zeros((4, 4)), np.zeros((4, 5)))


def test_build_valid_mask_combines_bands():
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    b = np.array([[1.0, 2.0], [np.inf, 4.0]])
    mask = build_valid_mask(a, b, nodatas=(None, None))
    assert mask.tolist() == [[True, False], [False, True]]


# --------------------------------------------------------------------------- #
# 6. visualisation must not disguise invalid pixels as vegetation
# --------------------------------------------------------------------------- #
def test_ndvi_rgba_masks_invalid_pixels():
    arr = np.array([[0.8, np.nan], [-0.9, 0.2]], dtype=np.float32)
    mask = np.isfinite(arr)
    rgba = ndvi_to_rgba(arr, mask, colormap="RdYlGn", vmin=-1.0, vmax=1.0)
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 1, 3] == 0, "invalid pixel must be fully transparent"
    assert rgba[0, 0, 3] == 255
    # a high-NDVI pixel must be green-dominant, not transparent
    assert rgba[0, 0, 1] > rgba[0, 0, 0]


def test_decimate_array_preserves_transform_and_nan_blocks():
    arr = np.full((64, 64), 0.5, dtype=np.float32)
    arr[0:8, 0:8] = np.nan
    out, transform, factor = decimate_array(arr, TRANSFORM, max_pixels=256)
    assert factor == 4
    assert out.shape == (16, 16)
    assert transform.a == pytest.approx(TRANSFORM.a * factor)
    assert np.isnan(out[0, 0]), "a fully invalid block stays invalid"
    assert out[8, 8] == pytest.approx(0.5)


def test_geotiff_export_keeps_georeferencing_and_provenance():
    arr = np.array([[0.8, np.nan], [0.1, 0.9]], dtype=np.float32)
    mask = np.isfinite(arr)
    meta = {
        "name": "ndvi", "label": "NDVI", "method": "(NIR-Red)/(NIR+Red)",
        "reflectance": S2_SPEC.to_dict(), "bands_used": {"red": "B04", "nir": "B08"},
        "provenance": {"dataset": "unit-test", "datetime": "1970-01-01"},
    }
    data = analysis_to_geotiff_bytes(arr, mask, TRANSFORM, CRS, meta)
    assert data[:2] in (b"II", b"MM")
    with MemoryFile(data) as mem, mem.open() as ds:
        assert ds.count == 1 and ds.dtypes[0] == "float32"
        assert str(ds.crs) == CRS
        assert ds.transform == TRANSFORM
        assert np.isnan(ds.nodata)
        assert ds.tags()["satquery_analysis"] == "ndvi"
        assert ds.tags()["satquery_dataset"] == "unit-test"
        back = ds.read(1)
    assert np.isnan(back[0, 1])
    assert back[0, 0] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# 7. illustrative classification
# --------------------------------------------------------------------------- #
def test_classification_is_flagged_as_illustrative():
    arr = np.array([[-0.5, 0.1], [0.3, 0.8]], dtype=np.float32)
    mask = np.isfinite(arr)
    out = classify_ndvi(arr, mask, stats={"min": -0.5, "max": 0.8})
    assert out is not None
    assert "NOT validated" in out["caveat"]
    assert sum(out["shares"].values()) == pytest.approx(1.0, abs=1e-6)
    assert out["breaks"] == list(ILLUSTRATIVE_BREAKS)


def test_classification_refuses_empty_scene():
    arr = np.full((2, 2), np.nan, dtype=np.float32)
    assert classify_ndvi(arr, np.isfinite(arr), stats=None) is None


# --------------------------------------------------------------------------- #
# 8. real Sentinel-2 sample, end to end
# --------------------------------------------------------------------------- #
@needs_s2
def test_sentinel2_end_to_end_ndvi():
    path = str(sample_path(S2_SAMPLE))
    info = describe_path(path)
    g = guess_band_roles(info)
    assert g.band("red") == 3 and g.band("nir") == 4

    with open_dataset(path) as ds:
        res, spec, report = ndvi_from_dataset(ds, 3, 4, profile=g.profile)

    assert spec.source == "sensor_profile"
    assert spec.scale == pytest.approx(1e-4) and spec.offset == 0.0
    assert report["validated"] is True
    assert report["offset_rejected"] is False

    # The whole window is land/water with no nodata gaps except a handful of
    # pixels where one band is missing.
    assert res.counts["total_pixels"] == 2048 * 2048
    assert res.counts["valid_percentage"] > 99.9
    assert res.counts["invalid_pixels"] > 0          # the partial-nodata pixels exist
    assert res.stats is not None

    # Physically sensible values for an irrigated delta in August
    assert -1.0 <= res.stats["min"] <= 0.0           # water present
    assert 0.5 <= res.stats["median"] <= 1.0         # mostly vegetated
    assert res.stats["mean"] == pytest.approx(0.61, abs=0.06)
    assert res.observed_range is not None
    assert res.native_shape == (2048, 2048)

    # georeferencing survives the trip from dataset to result
    assert str(res.crs) == str(info.spatial.crs_name)
    assert tuple(res.transform)[:6] == tuple(info.spatial.transform)


@needs_s2
def test_sentinel2_partial_nodata_pixels_are_not_treated_as_water():
    """NIR = 0 (nodata) must not become NDVI = -1 ('water')."""
    path = str(sample_path(S2_SAMPLE))
    with open_dataset(path) as ds:
        res, _spec, _report = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        nir = ds.read(4)
    nir_nodata = nir == 0
    assert nir_nodata.any(), "sample should contain NIR nodata pixels"
    assert not res.mask[nir_nodata].any(), "NIR nodata pixels must be masked out"
    assert np.isnan(res.array[nir_nodata]).all()


@needs_s2
def test_reflectance_detection_on_real_file_rejects_nothing_but_validates():
    path = str(sample_path(S2_SAMPLE))
    with open_dataset(path) as ds:
        spec, report = detect_reflectance_spec(ds=ds, profile="sentinel-2-l2a")
    assert spec.scale == pytest.approx(1e-4)
    assert spec.offset == 0.0
    assert report["sample_pixels"] > 0
    assert report["negative_fraction"] < 0.01
    assert report["median_reflectance"] > 0.0


def test_three_band_file_cannot_produce_ndvi():
    """A 3-band RGB file has no NIR band: the engine must not invent one."""
    info = describe_path(str(sample_path("RGB.byte.tif")))
    g = guess_band_roles(info)
    assert g.band("nir") is None
    with open_dataset(str(sample_path("RGB.byte.tif"))) as ds:
        with pytest.raises(IndexError):
            ndvi_from_dataset(ds, 1, 99)
