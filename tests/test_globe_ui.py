"""The primary map: the globe, its controls, and the base maps it offers.

WHAT IS UNDER TEST
------------------
* the map's arguments are JSON-safe, validated and honest -- a coordinate
  outside the Earth is dropped, never clamped into a plausible lie;
* raster overlays keep exactly the bounds the analysis produced, so a layer
  cannot drift between the globe and the flat detail mode;
* the frontend is the **one** map: it loads a pinned 3D globe engine, draws
  imagery through this app's own tile proxy (no provider contact, no API key),
  offers zoom / home / imagery / drawing controls, and reports events with an
  id so each is handled once;
* the "Light" base map is gone, and what remains needs no key.

Nothing here touches analysis: the map reports coordinates and geometry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tileserver import DEFAULT_PROVIDER, PROVIDERS
from ui.globe import FRONTEND_DIR, MAP_HEIGHT, map_args, overlay_payload

HTML = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
APP_PY = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")


class _Overlay:
    """Stand-in for `ui.map.MapOverlay` (same attribute contract)."""

    def __init__(self, name, rgba, bounds, opacity=0.9, show=True):
        self.name = name
        self.rgba = rgba
        self.bounds = bounds
        self.opacity = opacity
        self.show = show


def _rgba():
    import numpy as np

    return np.zeros((4, 4, 4), dtype="uint8")


# --------------------------------------------------------------- base maps
def test_the_light_base_map_is_gone():
    """'Light' needed a key and produced errors: removed, not hidden."""
    assert "light" not in PROVIDERS
    labels = {p.label for p in PROVIDERS.values()}
    assert "Light" not in labels
    assert labels == {"Streets", "Satellite"}


def test_the_working_base_maps_remain():
    assert {"Streets", "Satellite"} <= {p.label for p in PROVIDERS.values()}
    assert DEFAULT_PROVIDER == "osm"


def test_no_base_map_needs_an_api_key():
    for key, provider in PROVIDERS.items():
        haystack = (provider.url_template + key + provider.label).lower()
        for token in ("apikey", "api_key", "api-key", "access_token", "appid"):
            assert token not in haystack, f"{key} would need a key: {token}"


def test_every_base_map_carries_attribution():
    for key, provider in PROVIDERS.items():
        assert provider.attribution.strip(), f"{key} has no attribution"


# ------------------------------------------------------------- map args
def test_map_args_are_json_safe():
    args = map_args(
        overlays=[_Overlay("True colour (RGB)", _rgba(), [[18.0, 73.5], [19.0, 74.5]])],
        centre=[18.5204, 73.8567],
        zoom=11,
        base="satellite",
        attribution="Tiles © Esri",
        footprint=[[27.9, 86.8], [28.1, 87.0]],
        roi={"type": "Polygon", "coordinates": [[[86.9, 27.9], [87.0, 27.9],
                                                 [87.0, 28.0], [86.9, 27.9]]]},
        marker=[27.9881, 86.9250],
        marker_label="Mount Everest",
    )
    assert json.loads(json.dumps(args)) == args
    assert args["centre"] == [18.5204, 73.8567]
    assert args["marker"] == [27.9881, 86.9250]
    assert args["footprint"] == [[27.9, 86.8], [28.1, 87.0]]
    assert args["base"] == "satellite"
    assert args["attribution"] == "Tiles © Esri"
    assert args["height_px"] == MAP_HEIGHT
    assert args["overlays"][0]["name"] == "True colour (RGB)"
    assert args["overlays"][0]["image"].startswith("data:image/png;base64,")


def test_map_args_defaults_are_honest():
    args = map_args()
    assert args["centre"] is None          # no position invented
    assert args["marker"] is None
    assert args["overlays"] == []
    assert args["footprint"] is None
    assert args["base"] == "satellite"


def test_impossible_points_are_dropped_not_clamped():
    """A coordinate outside the Earth is a bug; rounding it would invent one."""
    assert map_args(centre=[95.0, 10.0])["centre"] is None
    assert map_args(centre=[10.0, 190.0])["centre"] is None
    assert map_args(centre=["not", "numbers"])["centre"] is None
    assert map_args(centre=None)["centre"] is None
    assert map_args(marker=[12.0, 34.0])["marker"] == [12.0, 34.0]
    assert map_args(footprint=[[95.0, 10.0], [96.0, 11.0]])["footprint"] is None


def test_overlays_keep_the_analysis_bounds_exactly():
    """The globe and the flat map must agree, so both use the same bounds."""
    got = overlay_payload([_Overlay("NDVI", _rgba(), [[-1.5, 30.25], [2.5, 33.75]],
                                    opacity=0.6, show=False)])
    assert len(got) == 1
    assert got[0]["south"] == -1.5 and got[0]["north"] == 2.5
    assert got[0]["west"] == 30.25 and got[0]["east"] == 33.75
    assert got[0]["opacity"] == 0.6
    assert got[0]["show"] is False


def test_unusable_overlays_are_dropped_not_drawn_wrong():
    import numpy as np

    payload = overlay_payload([
        _Overlay("no image", None, [[0, 0], [1, 1]]),
        _Overlay("bad bounds", np.zeros((2, 2, 4), "uint8"), [[95, 0], [96, 1]]),
        _Overlay("no bounds", np.zeros((2, 2, 4), "uint8"), None),
        _Overlay("good", np.zeros((2, 2, 4), "uint8"), [[0, 0], [1, 1]]),
    ])
    assert [p["name"] for p in payload] == ["good"]


def test_map_args_are_plain_floats_not_numpy():
    import numpy as np

    args = map_args(centre=[np.float64(18.5), np.float64(73.9)])
    assert isinstance(args["centre"][0], float)
    json.dumps(args)                       # raises if anything is unserialisable


# -------------------------------------------------------------- frontend
def test_the_frontend_exists_and_is_served_locally():
    assert (FRONTEND_DIR / "index.html").is_file()


def test_the_globe_engine_is_a_pinned_version():
    """A pinned CDN build: reproducible, and never "whatever is newest"."""
    refs = re.findall(r"https://unpkg\.com/cesium@([\d.]+)/", HTML)
    assert refs, "the globe engine must be loaded from a pinned version"
    assert len(set(refs)) == 1, f"one pinned version, found {set(refs)}"
    major, minor = (int(x) for x in refs[0].split(".")[:2])
    assert (major, minor) >= (1, 104)


def test_imagery_comes_from_this_apps_own_proxy():
    assert "/satquery-tiles/" in HTML
    # no tile provider is contacted by the browser
    for host in ("tile.openstreetmap.org", "arcgisonline.com", "cartocdn.com"):
        assert host not in HTML, f"the browser must not call {host}"


def test_the_globe_needs_no_api_key():
    lowered = HTML.lower()
    for token in ("apikey", "api_key", "api-key", "access_token", "ion.defaultaccesstoken"):
        assert token not in lowered, f"the globe must not need a key ({token})"


def test_the_globe_is_a_real_sphere():
    assert "Cesium.Viewer" in HTML
    assert "EllipsoidTerrainProvider" in HTML
    assert "SceneMode" in HTML or "scene3DOnly" in HTML
    assert "camera.flyTo" in HTML


def test_the_globe_has_the_required_controls():
    for control in ('id="zoomIn"', 'id="zoomOut"', 'id="home"', 'id="baseSeg"',
                    'id="drawSeg"', "credits", 'id="layers"'):
        assert control in HTML, f"missing control: {control}"


def test_imagery_switch_offers_satellite_and_streets_only():
    bases = re.findall(r'data-base="(\w+)"', HTML)
    assert set(bases) == {"satellite", "streets"}, bases
    assert "Light" not in HTML and "light_all" not in HTML


def test_drawn_areas_become_geojson_polygons():
    """The ROI a user draws is handed to core.roi as GeoJSON, like Leaflet's."""
    assert '"Feature"' in HTML and '"Polygon"' in HTML
    assert "coordinates: [ring]" in HTML
    assert "finishDrawing" in HTML
    assert "geometry" in HTML


def test_events_carry_an_id_so_each_is_handled_once():
    assert "eventSeq" in HTML
    assert "streamlit:setComponentValue" in HTML
    assert "streamlit:componentReady" in HTML


def test_there_is_a_fallback_when_the_3d_engine_cannot_load():
    """No empty box: a 2D-rendered globe takes over, and the app is told."""
    assert "initFallback" in HTML
    assert 'event("unavailable"' in HTML.replace(" ", "").replace("\n", "")
    assert "CESIUM_VERSION_OK" in HTML


def test_the_globe_offers_one_controlled_switch_to_the_flat_map():
    """One map at a time: the flat view is opened deliberately, not stacked."""
    assert 'event("mode", { value: "detail" })' in HTML
    assert HTML.count("streamlit_folium") == 0


# --------------------------------------------------------------- app wiring
def test_the_app_renders_one_map_and_switches_modes():
    assert "map_mode = st.session_state.get(\"map_mode\", \"globe\")" in APP_PY
    assert "render_globe(" in APP_PY
    # the flat map is only built inside the detail branch
    detail = APP_PY.split('if map_mode == "globe":')[1].split("else:")[0]
    assert "build_map(" not in detail, "the flat map must not render in globe mode"


def test_the_app_guards_against_a_map_rerun_loop():
    assert "map_event_id" in APP_PY


def test_search_moves_the_primary_map_not_a_second_one():
    assert "st.session_state[\"map_centre\"] = [place.lat, place.lon]" in APP_PY
    assert 'centre=st.session_state.get("map_centre") or list(WORLD_CENTRE)' in APP_PY
