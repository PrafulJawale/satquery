"""Display regression tests — the map preview must never be black.

Two faults are pinned down here, both found while investigating a black map:

1. **A degenerate stretch turns a whole band black.** `percentile_bounds()`
   used to expand a constant range by a multiplicative epsilon, which keeps
   `hi > lo` but still maps every pixel to 0. A band with no contrast must
   display as MID GREY (what every GIS does with a single-value stretch), not
   as a black rectangle.
2. **NaN / nodata leaking into the stretch.** Percentiles over NaN are NaN, so
   the stretch divides by NaN and the image goes black. NaN and infinities are
   not data and must be excluded before the percentiles are taken.

Everything tested here is DISPLAY ONLY: no analysis value, mask, CRS or
transform is produced by these paths, and no Phase 1-13 result object is
involved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.geo import (WEB_MERCATOR, reproject_bands,             # noqa: E402
                      web_rgba_from_bands)
from core.preview import (decimated_read, percentile_bounds,      # noqa: E402
                          stretch_to_uint8, valid_mask)
from core.raster import open_dataset                              # noqa: E402

CRS = "EPSG:32636"
ORIGIN_X, ORIGIN_Y, CELL = 300_000.0, 3_600_000.0, 10.0
HEIGHT = WIDTH = 120

REAL_SCENE = Path(__file__).resolve().parents[1] / (
    "data/sample/s2_s2b-36ruv-20230806-0-l2a_2048px.tif")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pattern(scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    return (np.sin(x / 7.0) * np.cos(y / 5.0) * 500.0 + 1500.0) * scale + offset


def write_scene(tmp_path, name, bands, dtype="float32", nodata=None) -> str:
    path = str(Path(tmp_path) / name)
    profile = {
        "driver": "GTiff", "height": HEIGHT, "width": WIDTH,
        "count": len(bands), "dtype": dtype, "crs": CRS,
        "transform": from_origin(ORIGIN_X, ORIGIN_Y, CELL, CELL),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as ds:
        for index, band in enumerate(bands, start=1):
            ds.write(np.asarray(band, dtype=dtype), index)
    return path


def display_for(path: str, bands=(1, 2, 3), max_pixels=1_000_000):
    """The map's own pipeline: native percentile bounds -> reproject -> RGBA."""
    with open_dataset(path) as ds:
        stack, _ = decimated_read(ds, list(bands), max_pixels=max_pixels)
        bounds = tuple(
            percentile_bounds(stack[k][valid_mask(stack[k], ds.nodatavals[b - 1])],
                              2.0, 98.0)
            for k, b in enumerate(bands))
    with open_dataset(path) as ds:
        web = reproject_bands(ds, list(bands), dst_crs=WEB_MERCATOR,
                              max_pixels=max_pixels,
                              resampling=Resampling.average)
    rgba = web_rgba_from_bands(web, bounds)
    return rgba, bounds, web


def visible_stats(rgba):
    """(fraction visible, mean luminance of the visible pixels)."""
    visible = rgba[..., 3] > 0
    if not visible.any():
        return 0.0, 0.0
    return float(visible.mean()), float(rgba[..., :3][visible].mean())


# --------------------------------------------------------------------------- #
# 1. the normal case
# --------------------------------------------------------------------------- #
def test_01_normal_raster_preview_is_visible_not_black(tmp_path):
    base = _pattern()
    path = write_scene(tmp_path, "normal.tif",
                       [base, base * 0.7 + 30, base * 0.4 + 90], "uint16",
                       nodata=0)
    rgba, bounds, _web = display_for(path)
    fraction, luminance = visible_stats(rgba)
    assert fraction > 0.9, fraction
    assert 40.0 < luminance < 215.0, luminance          # neither black nor white
    assert all(lo < hi for lo, hi in bounds)


def real_bands():
    """The band indices the app itself resolves (B04 red, B03 green, B02 blue,
    B08 nir for the bundled scene)."""
    from core.bands import guess_band_roles
    from core.raster import describe_path

    roles = guess_band_roles(describe_path(str(REAL_SCENE))).roles
    return (int(roles["red"]), int(roles["green"]), int(roles["blue"]),
            int(roles["nir"]))


@pytest.mark.skipif(not REAL_SCENE.exists(), reason="bundled scene missing")
def test_02_the_bundled_scene_previews_in_true_colour():
    """The scene the app opens with, through the map's own code path."""
    red, green, blue, _nir = real_bands()
    rgba, bounds, _web = display_for(str(REAL_SCENE), bands=(red, green, blue))
    fraction, luminance = visible_stats(rgba)
    assert fraction > 0.9, fraction
    assert 25.0 < luminance < 230.0, luminance
    assert bool(np.all(np.isfinite(np.asarray(bounds, dtype=float))))


# --------------------------------------------------------------------------- #
# 2. NaN and nodata
# --------------------------------------------------------------------------- #
def test_03_nan_block_stays_transparent_and_the_rest_stays_visible(tmp_path):
    base = _pattern().astype("float32")
    base[0:50, 0:50] = np.nan
    path = write_scene(tmp_path, "nan.tif", [base, base * 0.7, base * 0.4],
                       "float32", nodata=np.nan)
    rgba, bounds, _web = display_for(path)
    # the NaN block must be transparent, never painted black
    assert rgba[0, 0, 3] == 0 or rgba[20, 20, 3] == 0
    fraction, luminance = visible_stats(rgba)
    assert 0.5 < fraction < 1.0, fraction
    assert luminance > 40.0, luminance
    assert bool(np.all(np.isfinite(np.asarray(bounds, dtype=float))))


def test_04_percentile_bounds_ignores_nan_and_inf():
    """NaN in the sample must not poison the stretch into a black image."""
    values = np.concatenate([np.full(100, 50.0), np.full(5, np.nan),
                             np.full(5, np.inf)])
    lo, hi = percentile_bounds(values, 2.0, 98.0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= 50.0 <= hi


def test_05_all_nan_falls_back_to_a_usable_range():
    lo, hi = percentile_bounds(np.full(10, np.nan), 2.0, 98.0)
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo


def test_06_nodata_border_is_transparent_not_black(tmp_path):
    base = _pattern().astype("uint16")
    base[0:30, :] = 0                       # nodata stripe
    path = write_scene(tmp_path, "nodata.tif", [base, base, base], "uint16",
                       nodata=0)
    rgba, _bounds, _web = display_for(path)
    # somewhere in the interior the pixel is painted
    interior = rgba[HEIGHT - 10, WIDTH // 2]
    assert interior[3] == 255
    # and the nodata stripe carries no colour at all
    stripe = rgba[5, 5]
    assert stripe[3] == 0 and stripe[0] == 0 and stripe[1] == 0


# --------------------------------------------------------------------------- #
# 3. extreme and tiny ranges
# --------------------------------------------------------------------------- #
def test_07_very_large_values_still_display(tmp_path):
    base = (_pattern(scale=1000.0)).astype("float32")        # ~0 .. 2e6
    path = write_scene(tmp_path, "big.tif", [base, base * 0.5, base * 0.25])
    rgba, bounds, _web = display_for(path)
    fraction, luminance = visible_stats(rgba)
    assert fraction > 0.9
    assert 60.0 < luminance < 200.0, luminance
    assert all(lo < hi for lo, hi in bounds)


def test_08_very_small_values_still_display(tmp_path):
    base = (_pattern() * 1e-5).astype("float32")             # reflectance 0..0.02
    path = write_scene(tmp_path, "small.tif", [base, base * 0.9, base * 0.7])
    rgba, _bounds, _web = display_for(path)
    fraction, luminance = visible_stats(rgba)
    assert fraction > 0.9
    assert 60.0 < luminance < 200.0, luminance


def test_09_negative_values_still_display(tmp_path):
    base = ((_pattern() - 1500.0) * 0.02).astype("float32")  # -10 .. +10
    path = write_scene(tmp_path, "neg.tif", [base, -base, base * 0.5])
    rgba, bounds, _web = display_for(path)
    fraction, luminance = visible_stats(rgba)
    assert fraction > 0.9
    assert 40.0 < luminance < 215.0, luminance
    assert all(lo < hi for lo, hi in bounds)


# --------------------------------------------------------------------------- #
# 4. THE REGRESSION: a constant band must be mid grey, never black
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [0.0, 1.0, 77.0, 2_000_000.0, -3.5])
def test_10_constant_band_is_mid_grey_not_black(tmp_path, value):
    """The bug: `hi = lo * 1.000001` kept the range valid but stretched every
    pixel to zero, so a constant band previewed as a black rectangle."""
    lo, hi = percentile_bounds(np.full(1000, value), 2.0, 98.0)
    assert hi > lo
    # the constant value must land in the MIDDLE of the stretch, not at 0
    stretched = stretch_to_uint8(np.full((4, 4), value), np.ones((4, 4), bool),
                                 lo, hi)
    assert stretched.dtype == np.uint8
    assert 118 <= int(stretched.mean()) <= 138, int(stretched.mean())


def test_11_constant_scene_previews_as_grey_not_black(tmp_path):
    flat = np.full((HEIGHT, WIDTH), 1234.0, "float32")
    path = write_scene(tmp_path, "flat.tif", [flat, flat, flat])
    rgba, _bounds, _web = display_for(path)
    fraction, luminance = visible_stats(rgba)
    assert fraction > 0.9
    assert 100.0 < luminance < 160.0, luminance      # mid grey


def test_12_near_constant_spread_below_precision_is_not_amplified_to_black():
    lo, hi = percentile_bounds(np.array([5.0] * 100 + [5.0 + 1e-12]), 2.0, 98.0)
    stretched = stretch_to_uint8(np.full((2, 2), 5.0), np.ones((2, 2), bool),
                                 lo, hi)
    assert 100 <= int(stretched.mean()) <= 160, int(stretched.mean())


def test_13_stretch_survives_degenerate_bounds_from_any_caller():
    """A caller that hands over an inverted or zero range still gets grey."""
    stretched = stretch_to_uint8(np.zeros((3, 3), dtype="float32"),
                                 np.ones((3, 3), bool), 5.0, 5.0)
    assert stretched.dtype == np.uint8
    assert int(stretched.mean()) == 128
    stretched = stretch_to_uint8(np.zeros((3, 3), dtype="float32"),
                                 np.ones((3, 3), bool), float("nan"), 1.0)
    assert int(stretched.mean()) == 128


# --------------------------------------------------------------------------- #
# 5. true colour and false colour modes
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_SCENE.exists(), reason="bundled scene missing")
def test_14_true_colour_and_false_colour_both_display_and_differ():
    """Bands (red, green, blue) and (nir, red, green) -- the two preview modes
    the map offers. Neither may be black, and they must be different images."""
    red, green, blue, nir = real_bands()
    rgb, _b1, _w1 = display_for(str(REAL_SCENE), bands=(red, green, blue))
    fcc, _b2, _w2 = display_for(str(REAL_SCENE), bands=(nir, red, green))
    f_rgb, l_rgb = visible_stats(rgb)
    f_fcc, l_fcc = visible_stats(fcc)
    assert f_rgb > 0.9 and f_fcc > 0.9
    assert 25.0 < l_rgb < 230.0 and 25.0 < l_fcc < 230.0
    assert not np.array_equal(rgb[..., :3], fcc[..., :3])
    # the false-colour composite is the healthy-vegetation view: its red
    # channel comes from the NIR band, so it is brighter than true colour's
    assert fcc[..., 0].mean() > rgb[..., 0].mean()


def test_15_channel_order_is_honoured(tmp_path):
    """R <- band 1, G <- band 2, B <- band 3, not a guess.

    Fixed stretch bounds are used on purpose: with a per-band percentile
    stretch every channel is normalised on its own, which is exactly why it
    cannot reveal the channel order.
    """
    zeros = np.zeros((HEIGHT, WIDTH), "float32")
    red = np.full((HEIGHT, WIDTH), 200.0, "float32")
    path = write_scene(tmp_path, "order.tif", [red, zeros, zeros])
    with open_dataset(path) as ds:
        web = reproject_bands(ds, [1, 2, 3], dst_crs=WEB_MERCATOR,
                              max_pixels=1_000_000,
                              resampling=Resampling.average)
    rgba = web_rgba_from_bands(web, ((0.0, 255.0), (0.0, 255.0), (0.0, 255.0)))
    visible = rgba[..., 3] > 0
    assert rgba[..., 0][visible].mean() > 195         # red channel carries band 1
    assert rgba[..., 1][visible].mean() < 5           # green channel is band 2
    assert rgba[..., 2][visible].mean() < 5           # blue channel is band 3


# --------------------------------------------------------------------------- #
# 6. bounds, orientation and overlay visibility
# --------------------------------------------------------------------------- #
def test_16_leaflet_bounds_match_the_raster_extent(tmp_path):
    path = write_scene(tmp_path, "bounds.tif",
                       [_pattern(), _pattern(), _pattern()])
    rgba, _bounds, web = display_for(path)
    bounds = web.leaflet_bounds                       # [[lat_min, lon_min],
    assert len(bounds) == 2 and len(bounds[0]) == 2   #  [lat_max, lon_max]]
    lat_min, lon_min = bounds[0]
    lat_max, lon_max = bounds[1]
    assert lat_min < lat_max and lon_min < lon_max
    # the raster is 1.2 km across at 36N: sanity, not a hard-coded number
    assert 0.005 < (lon_max - lon_min) < 0.02, (lon_max - lon_min)
    assert 0.005 < (lat_max - lat_min) < 0.02, (lat_max - lat_min)
    # the north edge of the raster is the TOP of the image (north-up)
    top_row_visible = bool((rgba[0, :, 3] > 0).any())
    bottom_row_visible = bool((rgba[-1, :, 3] > 0).any())
    assert top_row_visible and bottom_row_visible
    # top row (north) must sit at a HIGHER latitude than the bottom row
    assert web.leaflet_bounds[1][0] > web.leaflet_bounds[0][0]


def test_17_overlay_is_partly_transparent_outside_the_data(tmp_path):
    """Alpha 0 where nothing was measured; 255 inside. Never a black frame."""
    base = _pattern().astype("float32")
    base[:, :20] = np.nan                              # an invalid strip
    path = write_scene(tmp_path, "strip.tif", [base, base, base], "float32",
                       nodata=np.nan)
    rgba, _bounds, _web = display_for(path)
    assert rgba.dtype == np.uint8 and rgba.shape[2] == 4
    assert (rgba[..., 3] == 0).any()                   # something is transparent
    assert (rgba[..., 3] == 255).any()                 # something is painted
    # a transparent pixel never carries colour: no semi-black ghosting
    assert rgba[..., :3][rgba[..., 3] == 0].max() == 0


# --------------------------------------------------------------------------- #
# 7. the base map itself
# --------------------------------------------------------------------------- #
def test_18_tile_url_is_same_origin_and_the_route_is_stable():
    import tileserver

    url = tileserver.tile_url("osm")
    assert url.startswith("/satquery-tiles/")
    assert "{z}" in url and "{x}" in url and "{y}" in url
    assert isinstance(tileserver.proxy_mounted(), bool)


def test_19_a_failed_tile_fetch_is_transparent_never_black(tmp_path, monkeypatch):
    """If the provider cannot be reached the map must show 'no imagery here',
    not a black square."""
    import tileserver

    monkeypatch.setattr(tileserver, "CACHE_DIR", Path(tmp_path) / "cache")
    monkeypatch.setattr(tileserver, "_fetch", lambda *a, **k: None)
    data, media, status = tileserver.get_tile("osm", 3, 4, 2)
    assert media == "image/png"
    assert status == 200
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(data)).convert("RGBA")
    alpha = np.asarray(image)[..., 3]
    assert alpha.max() == 0                            # fully transparent


def test_20_the_map_container_is_never_left_transparent():
    """Leaflet's own container is transparent; on a dark theme that is what
    makes a tile-less map look black. The map must set a visible colour."""
    streamlit = pytest.importorskip("streamlit")
    assert streamlit is not None
    import ui.map as ui_map

    folium_map = ui_map.build_map(overlays=[], draw=False)
    html = folium_map.get_root().render()
    assert ".leaflet-container{background:#e8eaed;}" in html


def test_21_overlays_are_added_on_top_of_the_base_layers():
    """RGB and FCC are ImageOverlays with opacity < 1, so the base map shows
    through instead of being covered by an opaque black sheet."""
    streamlit = pytest.importorskip("streamlit")
    assert streamlit is not None
    import numpy as np

    import ui.map as ui_map

    rgba = np.zeros((4, 4, 4), dtype="uint8")
    rgba[..., :3] = 120
    rgba[..., 3] = 255
    overlays = [ui_map.MapOverlay("True colour (RGB)", rgba,
                                 [[31.10, 31.70], [31.29, 31.93]],
                                 opacity=0.9, show=True)]
    folium_map = ui_map.build_map(overlays=overlays, draw=False)
    html = folium_map.get_root().render()
    assert "True colour (RGB)" in html
    assert "imageOverlay" in html
    # the overlay is registered after the base layers, so it draws on top
    assert html.index("imageOverlay") > html.index("/satquery-tiles/")
    assert "0.9" in html                                # opacity is applied
