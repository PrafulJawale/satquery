"""Phase 2 tests: band inspection + RGB / false-colour rendering.

Run:  python -m pytest tests -q

Two kinds of tests live here:
  * pure-function tests on synthetic arrays (deterministic, no I/O), and
  * real-file tests on the bundled samples (including the real Sentinel-2 window).

The synthetic arrays are TEST INPUTS ONLY -- they are never presented as, or
saved as, satellite imagery.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio import Affine
from rasterio.io import MemoryFile

from core.bands import describe_guess, guess_band_roles
from core.preview import (
    band_histograms,
    composite,
    decimated_read,
    encode_png,
    make_preview,
    percentile_bounds,
    stretch_to_uint8,
    valid_mask,
)
from core.raster import describe_path, open_dataset
from core.samples import list_samples, sample_path

S2_SAMPLE = next((n for n in list_samples() if n.startswith("s2_")), None)
needs_s2 = pytest.mark.skipif(S2_SAMPLE is None, reason="Sentinel-2 sample not downloaded")


# --------------------------------------------------------------------------- #
# synthetic fixture -- a test input, NOT satellite data
# --------------------------------------------------------------------------- #
def _memory_raster(stack: np.ndarray, nodata: int = 0, crs: str = "EPSG:32643", px: float = 10.0):
    """Write a small in-memory GeoTIFF and return an open dataset."""
    count, height, width = stack.shape
    mem = MemoryFile()
    ds = mem.open(
        driver="GTiff", count=count, height=height, width=width, dtype=stack.dtype,
        crs=crs, transform=Affine(px, 0.0, 400_000.0, 0.0, -px, 4_000_000.0), nodata=nodata,
    )
    ds.write(stack)
    return ds, mem


@pytest.fixture
def ramp_raster():
    """4-band ramp (blue, green, red, nir) with a nodata border."""
    h = w = 64
    ramp = np.tile(np.linspace(100, 5000, w, dtype=np.uint16), (h, 1))
    stack = np.stack([ramp * f for f in (1.0, 1.2, 0.8, 3.0)]).astype(np.uint16)
    stack[:, :4, :] = 0          # nodata border
    ds, mem = _memory_raster(stack)
    yield ds
    ds.close()
    mem.close()


# --------------------------------------------------------------------------- #
# pure functions
# --------------------------------------------------------------------------- #
def test_valid_mask_excludes_nodata():
    arr = np.array([[0, 5], [7, 0]], dtype=np.uint16)
    mask = valid_mask(arr, 0)
    assert mask.tolist() == [[False, True], [True, False]]


def test_valid_mask_handles_nan_nodata():
    arr = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype="float32")
    mask = valid_mask(arr, float("nan"))
    assert mask.tolist() == [[True, False], [False, True]]


def test_valid_mask_with_no_nodata_keeps_everything():
    arr = np.arange(9, dtype=np.uint16).reshape(3, 3)
    assert valid_mask(arr, None).all()


def test_percentile_bounds_matches_numpy():
    rng = np.random.default_rng(0)
    vals = rng.integers(1, 10_000, size=50_000).astype(np.uint16)
    lo, hi = percentile_bounds(vals, 2, 98)
    assert lo == pytest.approx(float(np.percentile(vals, 2)), abs=1.0)
    assert hi == pytest.approx(float(np.percentile(vals, 98)), abs=1.0)


def test_percentile_bounds_survives_constant_and_empty():
    lo, hi = percentile_bounds(np.full(100, 5000, dtype=np.uint16), 2, 98)
    assert hi > lo
    lo, hi = percentile_bounds(np.array([], dtype=np.uint16), 2, 98)
    assert hi > lo


def test_stretch_maps_percentiles_to_endpoints_and_nodata_to_zero():
    arr = np.tile(np.arange(0, 1000, dtype=np.uint16), (4, 1))
    arr[0] = 0                                   # nodata row
    mask = valid_mask(arr, 0)
    lo, hi = percentile_bounds(arr[mask], 2, 98)
    out = stretch_to_uint8(arr, mask, lo, hi)
    assert out.dtype == np.uint8
    assert out[0].max() == 0                     # nodata never painted as data
    assert out[mask].min() == 0
    assert out[mask].max() == 255


# --------------------------------------------------------------------------- #
# rendering on the synthetic fixture
# --------------------------------------------------------------------------- #
def test_decimated_read_respects_max_pixels_and_transform(ramp_raster):
    stack, transform = decimated_read(ramp_raster, (1, 2, 3), max_pixels=1024)
    assert stack.shape[0] == 3
    assert stack.shape[1] * stack.shape[2] <= 1024 * 1.05
    # display transform = native transform scaled by the decimation factor
    factor = ramp_raster.width / stack.shape[2]
    assert abs(transform.a - ramp_raster.transform.a * factor) < 1e-9
    assert abs(transform.e - ramp_raster.transform.e * factor) < 1e-9


def test_composite_excludes_nodata_from_alpha(ramp_raster):
    stack, _ = decimated_read(ramp_raster, (1, 2, 3), max_pixels=1_000_000)
    masks = [valid_mask(stack[k], 0) for k in range(3)]
    rgb, alpha, bounds = composite(stack, masks, (0, 1, 2))
    assert rgb.shape == (*stack.shape[1:], 3)
    assert rgb.dtype == np.uint8
    assert alpha.min() == 0                      # the nodata border
    assert alpha.max() == 255
    assert np.all(rgb[alpha == 0] == 0)          # nothing painted where data is missing
    assert len(bounds) == 3


def test_joint_stretch_uses_one_range_for_all_channels(ramp_raster):
    stack, _ = decimated_read(ramp_raster, (1, 2, 3), max_pixels=1_000_000)
    masks = [valid_mask(stack[k], 0) for k in range(3)]
    _rgb_a, _alpha_a, per_band = composite(stack, masks, (0, 1, 2), stretch_mode="per_band")
    _rgb_b, _alpha_b, joint = composite(stack, masks, (0, 1, 2), stretch_mode="joint")
    assert len({b for b in joint}) == 1          # one shared range
    assert len({b for b in per_band}) == 3       # independent ranges


def test_make_preview_end_to_end(ramp_raster):
    res = make_preview(ramp_raster, (3, 2, 1), ("red", "green", "blue"), max_pixels=4096)
    assert res.image.ndim == 3 and res.image.shape[2] == 3
    assert 0.0 <= res.nodata_fraction <= 1.0
    assert len(res.band_stats) == 3
    assert all(s["valid_pixels"] > 0 for s in res.band_stats)
    assert res.transform is not None


def test_encode_png_produces_valid_bytes(ramp_raster):
    res = make_preview(ramp_raster, (3, 2, 1), ("red", "green", "blue"), max_pixels=4096)
    png = encode_png(res.image, res.alpha)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 100


# --------------------------------------------------------------------------- #
# band-role detection on real files
# --------------------------------------------------------------------------- #
@needs_s2
def test_sentinel2_band_roles_detected_from_real_metadata():
    info = describe_path(str(sample_path(S2_SAMPLE)))
    g = guess_band_roles(info)
    assert g.roles == {"blue": 1, "green": 2, "red": 3, "nir": 4}
    assert g.confidence == "high"
    assert g.profile == "sentinel-2-l2a"
    assert g.reflectance_scale == pytest.approx(1e-4)
    assert not g.needs_confirmation


def test_rgb_byte_uses_colour_interpretation_and_invents_no_nir():
    info = describe_path(str(sample_path("RGB.byte.tif")))
    g = guess_band_roles(info)
    assert (g.band("red"), g.band("green"), g.band("blue")) == (1, 2, 3)
    assert g.band("nir") is None
    assert any("near-infrared" in w for w in g.warnings)


def test_all_nodata_band_names_are_read():
    info = describe_path(str(sample_path("all-nodata.tif")))
    g = guess_band_roles(info)
    assert g.band("nir") == 4
    assert g.band("blue") == 1


def test_single_band_file_gets_no_confident_guess():
    info = describe_path(str(sample_path("byte.tif")))
    g = guess_band_roles(info)
    assert g.confidence in ("none", "low")
    assert g.band("nir") is None


def test_describe_guess_is_human_readable():
    info = describe_path(str(sample_path("RGB.byte.tif")))
    text = describe_guess(guess_band_roles(info))
    assert "red=band 1" in text and "confidence" in text.lower()


# --------------------------------------------------------------------------- #
# rendering on real files
# --------------------------------------------------------------------------- #
@needs_s2
def test_sentinel2_preview_is_clean_and_georeferenced():
    with open_dataset(str(sample_path(S2_SAMPLE))) as ds:
        res = make_preview(ds, (3, 2, 1), ("red", "green", "blue"), max_pixels=250_000)
        assert res.nodata_fraction < 0.01          # the fetch script picked a full window
        assert res.alpha.max() == 255
        native_px = abs(ds.transform.a)
        display_px = abs(res.transform.a)
        assert display_px == pytest.approx(native_px * ds.width / res.display_shape[1], rel=1e-6)
        # reflectance sanity: NIR must be the brightest band over this farmland
        stats = {s["band"]: s["p50"] for s in res.band_stats}
        assert stats["red"] > 0


@needs_s2
def test_false_colour_differs_from_true_colour():
    with open_dataset(str(sample_path(S2_SAMPLE))) as ds:
        rgb = make_preview(ds, (3, 2, 1), ("red", "green", "blue"), max_pixels=250_000)
        fcc = make_preview(ds, (4, 3, 2), ("nir", "red", "green"), max_pixels=250_000)
        assert fcc.image.shape == rgb.image.shape
        assert not np.array_equal(fcc.image, rgb.image)


def test_eight_bit_file_renders_with_transparent_nodata():
    with open_dataset(str(sample_path("RGB.byte.tif"))) as ds:
        res = make_preview(ds, (1, 2, 3), ("red", "green", "blue"), max_pixels=400_000)
        assert res.nodata_fraction > 0.1           # the scene has black corners
        assert res.alpha.min() == 0
        assert np.all(res.image[res.alpha == 0] == 0)


def test_all_nodata_file_renders_nothing():
    with open_dataset(str(sample_path("all-nodata.tif"))) as ds:
        res = make_preview(ds, (1, 2, 3), ("a", "b", "c"), max_pixels=100_000)
        assert res.nodata_fraction > 0.99
        assert res.alpha.max() == 0
        assert res.image.max() == 0


def test_problem_files_do_not_crash_the_renderer():
    for name in ("rotated.tif", "float_nan.tif"):
        with open_dataset(str(sample_path(name))) as ds:
            res = make_preview(ds, (1, 1, 1), ("a", "b", "c"), max_pixels=10_000)
            assert res.image.ndim == 3


def test_histograms_are_measured_not_modelled():
    with open_dataset(str(sample_path("RGB.byte.tif"))) as ds:
        hists = band_histograms(ds, (1, 2), bins=16, max_pixels=200_000)
    assert len(hists) == 2
    h = hists[0]
    assert len(h["counts"]) == 16
    assert sum(h["counts"]) == h["valid_pixels"] > 0
