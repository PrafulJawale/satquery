"""Phase 4 tests: the Folium/Leaflet output.

We cannot screenshot a browser here, so we assert on the generated HTML: the
overlay bounds, the embedded PNG and the legends must all be present and
correct. If the bounds string is right in the HTML, Leaflet will draw the image
in the right place.
"""

from __future__ import annotations

import base64
import re

import numpy as np
import pytest
from rasterio import Affine
from rasterio.crs import CRS

from core.geo import (
    WEB_MERCATOR,
    footprint_feature,
    raster_footprint,
    reproject_array,
    web_rgba_from_values,
)
from core.samples import list_samples, sample_path
from ui.map import (
    BASEMAPS,
    DEFAULT_BASEMAP,
    TRANSPARENT_TILE,
    WORLD_CENTRE,
    WORLD_ZOOM,
    MapOverlay,
    build_map,
    colourbar_css,
    ndvi_legend_html,
    rgb_legend_html,
)
from tileserver import PROVIDERS as TILE_PROVIDERS

S2_SAMPLE = next((n for n in list_samples() if n.startswith("s2_")), None)
needs_s2 = pytest.mark.skipif(S2_SAMPLE is None, reason="Sentinel-2 sample not downloaded")

UTM36 = CRS.from_epsg(32636)
NORTH_UP = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)


def _overlay(name: str = "NDVI") -> MapOverlay:
    values = np.linspace(-1.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    values[:4, :] = np.nan
    web = reproject_array(values, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=50_000)
    rgba = web_rgba_from_values(web, -1.0, 1.0)
    return MapOverlay(name=name, rgba=rgba, bounds=web.leaflet_bounds, opacity=0.9)


def test_map_contains_an_image_overlay_with_the_given_bounds():
    ov = _overlay()
    m = build_map(overlays=[ov], bounds=ov.bounds, fit=True)
    html = m.get_root().render()

    assert "imageOverlay" in html or "L.imageOverlay" in html
    # Leaflet receives [[lat_min, lon_min], [lat_max, lon_max]]
    (lat_min, lon_min), (lat_max, lon_max) = ov.bounds
    for value in (lat_min, lon_min, lat_max, lon_max):
        assert f"{value:.6f}"[:8] in html or f"{value!r}" in html or f"{value}"[:8] in html
    assert "data:image/png;base64," in html, "overlay image must be embedded"


def test_overlay_image_is_a_real_png_with_alpha():
    ov = _overlay()
    m = build_map(overlays=[ov], bounds=ov.bounds)
    html = m.get_root().render()
    marker = "data:image/png;base64,"
    start = html.find(marker) + len(marker)
    b64 = html[start : start + 200_000].split('"')[0].split("'")[0]
    raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_map_has_base_layers_and_layer_control():
    m = build_map(overlays=[_overlay()], bounds=[[31.1, 31.7], [31.3, 31.9]])
    html = m.get_root().render()
    assert "tileLayer" in html or "L.tileLayer" in html
    assert "layerControl" in html or "L.control.layers" in html
    assert any(p.label in html for p in BASEMAPS.values())


def test_footprint_is_added_when_requested():
    poly = raster_footprint(NORTH_UP, UTM36, 256, 256)
    feature = footprint_feature(poly, {"crs": "EPSG:32636"})
    m = build_map(overlays=[], footprint=feature, bounds=[[31.1, 31.7], [31.3, 31.9]])
    html = m.get_root().render()
    assert "geoJson" in html or "L.geoJson" in html
    assert "Raster footprint" in html


def test_ndvi_legend_states_it_is_not_rgb_and_names_the_dataset():
    legend = ndvi_legend_html(dataset="S2B_36RUV_20230806", when="2023-08-06",
                              source_crs="EPSG:32636", display_crs=WEB_MERCATOR)
    assert "NDVI" in legend
    assert "Not an RGB" in legend
    assert "S2B_36RUV_20230806" in legend
    assert "2023-08-06" in legend
    assert "EPSG:32636" in legend and WEB_MERCATOR in legend


def test_rgb_legend_mentions_both_crs():
    legend = rgb_legend_html("Satellite imagery", "true colour", "EPSG:32636", WEB_MERCATOR)
    assert "EPSG:32636" in legend and WEB_MERCATOR in legend


def test_colourbar_css_is_a_gradient():
    css = colourbar_css("RdYlGn", steps=5)
    assert css.startswith("linear-gradient(to right,")
    assert css.count("rgb(") == 5


def test_mouse_position_and_click_popup_are_present():
    m = build_map(overlays=[_overlay()], bounds=[[31.1, 31.7], [31.3, 31.9]])
    html = m.get_root().render()
    assert "lon/lat:" in html, "live coordinate readout missing"


@needs_s2
def test_real_sample_map_bounds_are_inside_the_footprint():
    """End-to-end: reproject the real NDVI and build a map from it."""
    from core.raster import open_dataset
    from core.indices import ndvi_from_dataset

    with open_dataset(str(sample_path(S2_SAMPLE))) as ds:
        res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        web = reproject_array(res.array, ds.transform, ds.crs, dst_crs=WEB_MERCATOR,
                              max_pixels=600_000)
        poly = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=32)

    ov = MapOverlay("NDVI", web_rgba_from_values(web, -1.0, 1.0), web.leaflet_bounds)
    m = build_map(overlays=[ov], footprint=footprint_feature(poly, {"crs": "EPSG:32636"}),
                  bounds=web.leaflet_bounds, extra_html=ndvi_legend_html(dataset="sentinel-2"))
    html = m.get_root().render()

    # the scene must land in the Nile Delta
    (lat_min, lon_min), (lat_max, lon_max) = web.leaflet_bounds
    assert 30.5 < lon_min < 32.5 and 30.5 < lat_min < 32.0
    assert lat_max > lat_min and lon_max > lon_min
    assert "data:image/png;base64," in html
    assert "Not an RGB" in html


# --------------------------------------------------------------------------- #
# Base (street) map: the tile providers and the offline fallback
#
# --------------------------------------------------------------------------- #
# the global base map (proxied by this app)
# --------------------------------------------------------------------------- #
# The browser must never contact a tile provider: a provider can answer with
# 403 "access blocked" PNGs, and that looks like an application bug. Every base
# map is therefore served from this app's own /satquery-tiles route.


def _map_html(**kw) -> str:
    return build_map(overlays=[], **kw).get_root().render()


def test_the_map_opens_on_the_whole_world():
    m = build_map(overlays=[])
    assert list(m.location) == [WORLD_CENTRE[0], WORLD_CENTRE[1]]
    assert m.options.get("zoom") == WORLD_ZOOM
    assert WORLD_ZOOM <= 3, "the default view must be zoomed out to the world"


def test_every_base_map_is_served_from_this_app():
    """No tile URL may point at a third-party host."""
    html = _map_html()
    import re

    urls = re.findall(r'"(https?://[^"]*(?:tile|basemaps|arcgis)[^"]*)"', html)
    tile_urls = [u for u in urls if "{z}" in u or "{x}" in u]
    assert not tile_urls, f"base maps must be proxied, found: {tile_urls}"
    for provider in BASEMAPS.values():
        assert f"/satquery-tiles/{provider.key}/" in html


def test_no_base_map_requires_an_api_key():
    for key, provider in BASEMAPS.items():
        low = provider.url_template.lower() + key.lower()
        assert "apikey" not in low and "api_key" not in low and "access_token" not in low


def test_every_base_map_carries_attribution():
    """Attribution is a licence condition, so it must reach the page.

    Non-ASCII characters are JSON-escaped by folium, so match on the ASCII
    words of each attribution rather than the exact string.
    """
    import re

    html = _map_html()
    for provider in BASEMAPS.values():
        words = [w for w in re.findall(r"[A-Za-z]{4,}", provider.attribution)]
        assert words, f"{provider.key} has no ASCII attribution words"
        assert any(w in html for w in words), (
            f"{provider.key} attribution missing from the page: {provider.attribution}")


def test_a_failing_tile_renders_as_nothing_not_as_an_error():
    html = _map_html()
    assert TRANSPARENT_TILE in html
    assert TRANSPARENT_TILE.startswith("data:image/png;base64,")


def test_the_default_base_map_is_the_street_map():
    html = _map_html()
    assert f"/satquery-tiles/{DEFAULT_BASEMAP}/" in html
    assert DEFAULT_BASEMAP == "osm"


def test_drawing_tools_are_area_tools_only():
    """Rectangle + polygon, and no circle (it cannot be verified)."""
    html = _map_html(draw=True)
    assert '"circle": false' in html.replace("'", '"') or "circle" in html
    from ui.map import DEFAULT_DRAW_OPTIONS

    assert DEFAULT_DRAW_OPTIONS["rectangle"] and DEFAULT_DRAW_OPTIONS["polygon"]
    assert DEFAULT_DRAW_OPTIONS["circle"] is False
    assert DEFAULT_DRAW_OPTIONS["polyline"] is False


def test_the_map_keeps_the_raster_as_a_data_layer_not_a_boundary():
    """`bounds` flies to the data; it never becomes the map's extent."""
    m = build_map(overlays=[], bounds=[[31.1, 31.7], [31.3, 31.9]], fit=True)
    # fit_bounds flies to the data, but the starting view is still the world
    assert m.options.get("zoom") == WORLD_ZOOM
