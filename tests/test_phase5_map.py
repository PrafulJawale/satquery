"""Phase 5 tests: the drawing control in the rendered map.

These assert on the HTML/JS that Folium produces, because that is what the
browser will execute. They cannot prove a human can draw with a mouse -- but
they do prove the tools are wired up, the unwanted tools are off, and the
streamlit-folium integration shim is present.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio import Affine
from rasterio.crs import CRS

from core.geo import WEB_MERCATOR, footprint_feature, raster_footprint, reproject_array, web_rgba_from_values
from core.samples import list_samples, sample_path
from ui.map import MapOverlay, build_map

UTM36 = CRS.from_epsg(32636)
NORTH_UP = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
BOUNDS = [[31.1039213422, 31.7098281537], [31.2905365386, 31.9271489520]]

S2_SAMPLE = next((n for n in list_samples() if n.startswith("s2_")), None)
needs_s2 = pytest.mark.skipif(S2_SAMPLE is None, reason="Sentinel-2 sample not downloaded")


def _overlay():
    values = np.linspace(-1.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    web = reproject_array(values, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=50_000)
    return MapOverlay("NDVI", web_rgba_from_values(web, -1.0, 1.0), web.leaflet_bounds)


def _draw_html(**kw) -> str:
    return build_map(overlays=[_overlay()], bounds=BOUNDS, draw=True, **kw).get_root().render()


# --------------------------------------------------------------------------- #
def test_draw_control_is_added_when_requested():
    assert "L.Control.Draw" in _draw_html()


def test_draw_control_is_absent_by_default():
    html = build_map(overlays=[_overlay()], bounds=BOUNDS).get_root().render()
    assert "L.Control.Draw" not in html
    assert "window.drawnItems" not in html


def test_rectangle_and_polygon_tools_are_enabled():
    html = _draw_html()
    assert '"polygon": {' in html
    assert '"rectangle": {' in html


def test_non_area_tools_are_disabled():
    html = _draw_html()
    for tool in ("polyline", "marker", "circlemarker", "circle"):
        assert f'"{tool}": false' in html, f"{tool} must be disabled"


def test_edit_and_delete_are_enabled():
    html = _draw_html()
    assert '"edit": true' in html
    assert '"remove": true' in html


def test_streamlit_folium_drawn_items_shim_is_present():
    """Without `window.drawnItems`, streamlit-folium reports no drawings at all."""
    html = _draw_html()
    assert "window.drawnItems" in html
    assert "DOMContentLoaded" in html, "the alias must be deferred until the map exists"


def test_the_shim_points_at_the_real_feature_group_variable():
    html = _draw_html()
    marker = html.find("window.drawnItems =")
    line = html[marker:marker + 120].splitlines()[0]
    assert "feature_group_" in line, line


def test_defaults_can_be_overridden():
    html = build_map(
        overlays=[], bounds=BOUNDS, draw=True,
        draw_options={"polyline": True, "polygon": False, "rectangle": False,
                      "marker": False, "circle": False, "circlemarker": False},
    ).get_root().render()
    assert '"polyline": true' in html
    assert '"polygon": false' in html


def test_phase_4_features_survive_with_drawing_enabled():
    poly = raster_footprint(NORTH_UP, UTM36, 256, 256)
    html = build_map(
        overlays=[_overlay()],
        footprint=footprint_feature(poly, {"crs": "EPSG:32636"}),
        bounds=BOUNDS,
        extra_html="<div>legend</div>",
        draw=True,
    ).get_root().render()
    assert "Raster footprint" in html
    assert "data:image/png;base64," in html
    assert "legend" in html
    assert "lon/lat:" in html
    assert "L.Control.Draw" in html


@needs_s2
def test_real_scene_map_with_drawing_is_built_in_the_right_place():
    from core.indices import ndvi_from_dataset
    from core.raster import open_dataset

    with open_dataset(str(sample_path(S2_SAMPLE))) as ds:
        res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        web = reproject_array(res.array, ds.transform, ds.crs, dst_crs=WEB_MERCATOR,
                              max_pixels=600_000)
    m = build_map(
        overlays=[MapOverlay("NDVI", web_rgba_from_values(web, -1.0, 1.0), web.leaflet_bounds)],
        bounds=web.leaflet_bounds,
        draw=True,
    )
    html = m.get_root().render()
    (lat_min, lon_min), (lat_max, lon_max) = web.leaflet_bounds
    assert 30.5 < lon_min < 32.5 and 30.5 < lat_min < 32.0
    assert "L.Control.Draw" in html
    assert "window.drawnItems" in html


def test_delete_bridge_is_injected():
    """Regression guard for the leaflet-draw 1.0.2 delete trap.

    leaflet-draw only fires the map-level `draw:deleted` when the user clicks
    "Save"; streamlit-folium listens for that event, so without this bridge a
    deleted shape would stay in `all_drawings` forever (a stale ROI).
    """
    html = _draw_html()
    assert "satquery-draw-bridge" in html
    assert "draw:deleted" in html
    assert "__sq_bridge" in html


def test_drawn_items_alias_is_present_for_standalone_pages():
    """Folium-only HTML (e.g. artifacts/phase5_draw.html) needs the alias;
    inside Streamlit, streamlit-folium sets window.drawnItems itself."""
    html = _draw_html()
    assert "window.drawnItems" in html
