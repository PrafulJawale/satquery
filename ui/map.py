"""ui/map.py -- Folium/Leaflet map construction (PHASE 4).

This module turns *already computed* geographic products into a web map. It
performs no analysis and, crucially, no coordinate guessing: every overlay
arrives with bounds that were derived from a raster CRS + affine transform by
`core.geo`.

Base map tiles are fetched by the server and re-served from this app's own
origin (`/satquery-tiles/...`), so the browser never talks to a tile provider.
The map is a global navigation surface; the raster is data drawn on top of it.
"""

from __future__ import annotations
from functools import lru_cache

import base64
import io
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import folium
from folium import plugins

# --------------------------------------------------------------------------- #
# base maps: served by THIS app, never fetched by the browser
# --------------------------------------------------------------------------- #
# Every tile is requested from `/satquery-tiles/...`, a route this very server
# exposes (see `tileserver.py`, mounted by `serve.py`). The browser therefore
# never contacts a tile provider, so it cannot be handed a 403 "access blocked"
# tile, an API-key warning or a broken provider image. If the server cannot get
# a tile, it serves a transparent placeholder: no imagery, never an error.
#
# The providers are public, no-API-key and OSM-compatible. Attribution is set
# per layer and shown in the map's attribution control.

from tileserver import (  # noqa: E402  (import kept next to its documentation)
    DEFAULT_PROVIDER,
    PROVIDERS as TILE_PROVIDERS,
    STATS as TILE_STATS,
    TileProvider,
    tile_url,
)

BASEMAPS: Dict[str, TileProvider] = TILE_PROVIDERS
DEFAULT_BASEMAP: str = DEFAULT_PROVIDER

# Where a world view starts. The map is a navigation surface for the whole
# planet; the raster is data that sits on top of it, not the map's boundary.
WORLD_CENTRE: Tuple[float, float] = (20.0, 10.0)
WORLD_ZOOM: int = 2

# Longest label the layer control has to accommodate ("Satellite").
BASEMAP_LABELS: Tuple[str, ...] = tuple(p.label for p in TILE_PROVIDERS.values())

# A 1x1 transparent PNG. Used as `errorTileUrl` so that a tile the server could
# not fetch renders as *nothing at all* -- never as provider error text.
TRANSPARENT_TILE = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlE"
    "QVR42mNkYGD4DwABBAEAX+n0dwAAAABJRU5ErkJggg=="
)


# --- drawing (Phase 5) ----------------------------------------------------- #
# Only area tools. Polyline / marker / circle-marker are not areas. Circle is
# off too: it would arrive as centre+radius rather than a polygon, and
# streamlit-folium converts it with an approximation we cannot verify.
DEFAULT_DRAW_OPTIONS: Dict[str, Any] = {
    "polyline": False,
    "marker": False,
    "circlemarker": False,
    "circle": False,
    "rectangle": {"shapeOptions": {"color": "#0066ff", "weight": 2, "fillOpacity": 0.08}},
    "polygon": {
        "shapeOptions": {"color": "#0066ff", "weight": 2, "fillOpacity": 0.08},
        "allowIntersection": False,
    },
}
DEFAULT_EDIT_OPTIONS: Dict[str, Any] = {"edit": True, "remove": True}


def _inject_onerror_script(m: "folium.Map", script: str, token: str) -> None:
    """Carry `script` into the page on the back of a deliberately missing image.

    streamlit-folium rebuilds the map body with `innerHTML`, which does not run
    <script> tags; the onerror handler of a missing <img> does. (A later
    streamlit-folium/Streamlit combination stopped delivering that event too --
    see the note in `build_map` -- but the carrier is kept because the HTML is
    part of the documented, tested integration contract.)
    """
    m.get_root().html.add_child(
        folium.Element(
            f"<img src='satquery-{token}.png' alt='' style='display:none' "
            f"onerror=\"try{{ {script} }}catch(e){{}}\" />"
        )
    )


@dataclass
class MapOverlay:
    """One image layer, ready to place on the map.

    `rgba` is a uint8 (H, W, 4) array and `bounds` is
    [[lat_min, lon_min], [lat_max, lon_max]] -- both produced by `core.geo`.
    """

    name: str
    rgba: Any
    bounds: List[List[float]]
    opacity: float = 1.0
    show: bool = True


# --------------------------------------------------------------------------- #
# legend
# --------------------------------------------------------------------------- #
def colourbar_css(colormap: str = "RdYlGn", steps: int = 12) -> str:
    """Sample a matplotlib colormap into a CSS linear-gradient."""
    import matplotlib

    cmap = matplotlib.colormaps.get(colormap) or matplotlib.colormaps["viridis"]
    stops = []
    for i in range(steps):
        frac = i / (steps - 1)
        r, g, b, _a = [int(round(v * 255)) for v in cmap(frac)]
        stops.append(f"rgb({r},{g},{b}) {frac * 100:.0f}%")
    return "linear-gradient(to right, " + ", ".join(stops) + ")"


def ndvi_legend_html(
    dataset: str = "",
    when: str = "",
    vmin: float = -1.0,
    vmax: float = 1.0,
    colormap: str = "RdYlGn",
    source_crs: str = "",
    display_crs: str = "",
) -> str:
    """Legend/caption box for the NDVI layer.

    Says plainly that this is NDVI and NOT an RGB image, names the dataset and
    date, and states that invalid pixels are transparent. A red/green NDVI map
    is easy to misread as a photograph otherwise.
    """
    gradient = colourbar_css(colormap)
    return f"""
<div style="
    position: fixed; bottom: 26px; left: 26px; z-index: 9999;
    background: rgba(255,255,255,0.94); padding: 10px 12px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35); font-family: sans-serif; font-size: 12px;
    max-width: 290px; color: #111;">
  <div style="font-weight: 700; font-size: 13px; margin-bottom: 2px;">
    NDVI — vegetation index
  </div>
  <div style="color:#b00020; font-weight:600; margin-bottom: 6px;">
    Not an RGB / satellite photo
  </div>
  <div style="background: {gradient}; height: 12px; border-radius: 3px;
              border: 1px solid #999;"></div>
  <div style="display:flex; justify-content:space-between; margin-top: 2px;">
    <span>{vmin:g}</span><span>NDVI</span><span>{vmax:g}</span>
  </div>
  <div style="margin-top: 6px; color:#333; line-height:1.35;">
    Continuous values only — <b>no validated vegetation threshold</b>.<br>
    Invalid pixels are <b>transparent</b>, never green.
  </div>
  <div style="margin-top: 6px; color:#555; font-size: 11px; line-height:1.35;">
    {dataset}<br>{when}
    {('<br>' + source_crs + ' → ' + display_crs) if source_crs else ''}
  </div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# Phase 8 -- suitability legend (deliberately NOT the NDVI palette)
# --------------------------------------------------------------------------- #
#: class code -> RGBA. Blues/teals for the suitable classes, greys for the rest:
#: the NDVI layer is a red-yellow-green ramp, so nothing here can be mistaken
#: for it. "Insufficient data" is a pale, translucent grey -- visibly different
#: from "Unsuitable", which is a solid dark grey.
SUITABILITY_COLOURS: Dict[int, Tuple[int, int, int, int]] = {
    4: (8, 64, 129, 215),        # Highly suitable
    3: (43, 140, 190, 205),      # Moderately suitable
    2: (123, 204, 196, 195),     # Marginal
    1: (80, 80, 80, 205),        # Unsuitable
    0: (210, 210, 210, 95),      # Insufficient data
}
SUITABILITY_CLASS_ORDER: Tuple[int, ...] = (4, 3, 2, 1, 0)
SUITABILITY_CLASS_NAMES: Dict[int, str] = {
    4: "Highly suitable",
    3: "Moderately suitable",
    2: "Marginal",
    1: "Unsuitable",
    0: "Insufficient data",
}


# --------------------------------------------------------------------------- #
# Phase 9 -- spatial-query result palette (categorical, three-valued)
# --------------------------------------------------------------------------- #
#: result state -> RGBA. Deliberately categorical: a query either matches, does
#: not match, or cannot be decided. No continuous ramp, because a "how much?"
#: colour scale would imply a quantity that does not exist.
SPATIAL_STATE_COLOURS: Dict[int, Tuple[int, int, int, int]] = {
    2: (35, 155, 86, 200),        # MATCH            - strong green
    1: (150, 150, 150, 70),       # NO MATCH         - faint grey, still drawn
    0: (230, 159, 0, 170),        # INSUFFICIENT     - amber, never grey
}
SPATIAL_STATE_ORDER: Tuple[int, ...] = (2, 1, 0)
SPATIAL_STATE_NAMES: Dict[int, str] = {
    2: "Match — satisfies all conditions",
    1: "No match — measured, does not satisfy",
    0: "Insufficient data — not established",
}


def spatial_rgba(state_codes: Any) -> Any:
    """Three-valued result mask -> uint8 RGBA, using the spatial palette."""
    import numpy as np

    codes = np.asarray(state_codes)
    out = np.zeros((*codes.shape, 4), dtype="uint8")
    for code, rgba in SPATIAL_STATE_COLOURS.items():
        out[codes == code] = rgba
    return out


def spatial_legend_html(expression: str = "",
                        analysis_resolution: Optional[float] = None,
                        native_resolutions: Optional[Dict[str, Any]] = None,
                        distance_m: Optional[float] = None) -> str:
    """Legend for the spatial-query overlay.

    Says what the map cannot: that NO MATCH is a measured result, not a failure,
    and that INSUFFICIENT DATA is not the same thing.
    """
    swatches = "".join(
        f'<div style="display:flex; align-items:center; gap:6px; margin-top:3px;">'
        f'<span style="display:inline-block; width:18px; height:12px; '
        f'border:1px solid #777; background:rgba({r},{g},{b},{a / 255:.2f});"></span>'
        f'<span>{SPATIAL_STATE_NAMES[c]}</span></div>'
        for c in SPATIAL_STATE_ORDER
        for (r, g, b, a) in (SPATIAL_STATE_COLOURS[c],)
    )
    query = (f"<div style='margin-top:4px; color:#333; font-size:11px; "
             f"line-height:1.35;'>{expression}</div>" if expression else "")
    native = ""
    if native_resolutions:
        native = ("<div style='margin-top:6px; color:#555; font-size:11px; "
                  "line-height:1.35;'>Native detail: "
                  + " · ".join(f"{k} {v}" for k, v in native_resolutions.items())
                  + (f"<br>Analysis grid: {analysis_resolution:g} m — resampling adds "
                     "no information." if analysis_resolution else "") + "</div>")
    distance = (f"<div style='margin-top:4px; color:#555; font-size:11px;'>"
                f"Proximity distance: {distance_m:g} m</div>"
                if distance_m else "")
    return f"""
<div style="
    position: fixed; bottom: 26px; left: 26px; z-index: 9999;
    background: rgba(255,255,255,0.94); padding: 10px 12px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35); font-family: sans-serif; font-size: 12px;
    max-width: 320px; color: #111;">
  <div style="font-weight: 700; font-size: 13px;">Spatial query result</div>
  <div style="color:#b00020; font-weight:600; margin: 2px 0 6px;">
    Categorical result — not a recommendation
  </div>
  {swatches}
  <div style="margin-top: 6px; color:#333; line-height:1.35;">
    <b>No match</b> means measured and not satisfying — a result, not a failure.
    <b>Insufficient data</b> means not established, and is never counted as a
    non-match.
  </div>
  {query}
  {distance}
  {native}
</div>
""".strip()


# =========================================================================== #
# Phase 10 -- temporal change layers
# =========================================================================== #
# Class names, colours and order come from config/temporal/ndvi_change.yml so
# the map can never disagree with the engine about what a colour means.
# =========================================================================== #
_FALLBACK_CHANGE_COLOURS = {
    0: (150, 150, 150, 90),      # insufficient data
    1: (215, 25, 28, 200),       # decrease
    2: (250, 247, 190, 190),     # stable
    3: (26, 150, 65, 200),       # increase
}
_FALLBACK_CHANGE_NAMES = {0: "Insufficient data", 1: "Decrease",
                          2: "Stable", 3: "Increase"}
_FALLBACK_CHANGE_ORDER = (3, 2, 1, 0)


@lru_cache(maxsize=1)
def temporal_change_style():
    """(names, colours, order) for the change classes, from the YAML config."""
    try:
        from core.temporal import load_temporal_config

        classes = (load_temporal_config().get("classes") or {})
        names = {int(k): str(v) for k, v in (classes.get("labels") or {}).items()}
        colours = {int(k): tuple(int(v) for v in rgba)
                   for k, rgba in (classes.get("colours") or {}).items()}
        order = tuple(int(c) for c in (classes.get("order") or ()))
        if names and colours and order:
            return names, colours, order
    except Exception:            # a display palette must never break the map
        pass
    return _FALLBACK_CHANGE_NAMES, _FALLBACK_CHANGE_COLOURS, _FALLBACK_CHANGE_ORDER


def ndvi_change_rgba(class_codes: Any) -> Any:
    """Change-class raster -> uint8 RGBA, using the configured palette."""
    import numpy as np

    _names, colours, _order = temporal_change_style()
    codes = np.asarray(class_codes)
    out = np.zeros((*codes.shape, 4), dtype="uint8")
    for code, rgba in colours.items():
        out[codes == code] = tuple(rgba)
    return out


def ndvi_change_legend_html(before_date: str = "",
                            after_date: str = "",
                            before_scene: str = "",
                            after_scene: str = "",
                            threshold: Optional[float] = None,
                            resolution: Optional[float] = None,
                            resampled: bool = False,
                            delta_min: Optional[float] = None,
                            delta_max: Optional[float] = None,
                            show_delta_ramp: bool = True) -> str:
    """Legend for the temporal layers.

    It carries the three statements the map cannot make for itself: that the
    classes come from a THRESHOLD (a display convention), that a decrease is a
    decrease in the INDEX (not a cause), and that INSUFFICIENT DATA is not the
    same thing as STABLE.
    """
    names, colours, order = temporal_change_style()

    swatches = "".join(
        f'<div style="display:flex; align-items:center; gap:6px; margin-top:3px;">'
        f'<span style="display:inline-block; width:18px; height:12px; '
        f'border:1px solid #777; background:rgba({r},{g},{b},{a / 255:.2f});"></span>'
        f'<span>{names.get(c, str(c))}</span></div>'
        for c in order
        for (r, g, b, a) in (colours.get(c, (0, 0, 0, 0)),)
    )

    ramp = ""
    if show_delta_ramp:
        lo = -1.0 if delta_min is None else min(-0.05, float(delta_min))
        hi = 1.0 if delta_max is None else max(0.05, float(delta_max))
        span = max(abs(lo), abs(hi))
        ramp = (
            f'<div style="margin-top:8px;">'
            f'<div style="font-weight:600;">&#916;NDVI (after &#8722; before)</div>'
            f'<div style="height:12px; border-radius:3px; margin-top:3px; '
            f'border:1px solid #777; background:{colourbar_css("RdYlGn", 12)};"></div>'
            f'<div style="display:flex; justify-content:space-between; font-size:10px; '
            f'color:#555;"><span>&#8722;{span:g}</span><span>0</span>'
            f'<span>+{span:g}</span></div></div>'
        )

    when = " &#8594; ".join(x for x in (before_date, after_date) if x)
    scenes = ""
    if before_scene or after_scene:
        scenes = (f"<div style='margin-top:4px; color:#555; font-size:11px; "
                  f"line-height:1.35;'>{before_scene or '?'} &#8594; "
                  f"{after_scene or '?'}</div>")
    thr = (f"<div style='margin-top:4px; color:#555; font-size:11px;'>"
           f"Display threshold: &#177;{threshold:g} &#916;NDVI</div>"
           if threshold is not None else "")
    res = (f"<div style='margin-top:4px; color:#555; font-size:11px;'>"
           f"Analysis grid: {resolution:g} m</div>" if resolution else "")
    warn = ("<div style='margin-top:4px; color:#8a6d00; font-size:11px;'>"
            "Scenes were resampled onto one common grid — resampling adds no "
            "information.</div>" if resampled else "")

    return f"""
<div style="
    position: fixed; bottom: 26px; left: 26px; z-index: 9999;
    background: rgba(255,255,255,0.94); padding: 10px 12px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35); font-family: sans-serif; font-size: 12px;
    max-width: 320px; color: #111;">
  <div style="font-weight: 700; font-size: 13px;">NDVI change {when}</div>
  <div style="color:#b00020; font-weight:600; margin: 2px 0 6px;">
    Change in the vegetation index — not a cause
  </div>
  {swatches}
  {ramp}
  <div style="margin-top: 6px; color:#333; line-height:1.35;">
    <b>Decrease</b> means the index fell. It does not establish deforestation,
    crop failure, drought or flooding. <b>Insufficient data</b> means not
    measured on both dates, and is never counted as stable.
  </div>
  {scenes}{thr}{res}{warn}
</div>
""".strip()


def ndwi_legend_html(vmin: float = -1.0,
                     vmax: float = 1.0,
                     dataset: str = "",
                     bands: str = "",
                     resolution: Optional[float] = None) -> str:
    """Legend for the NDWI layer.

    Two things it must say, because the map alone cannot:
      * this is a CONTINUOUS spectral index, not a water map -- there is no
        "water / not water" class here, and no NDWI value is called water;
      * an index is not a flood, an extent, an availability or a quality.
    """
    gradient = colourbar_css("RdYlBu")
    meta = ""
    if dataset or bands:
        meta = (f"<div style='margin-top:6px; color:#555; font-size:11px; "
                f"line-height:1.35;'>{dataset}"
                f"{(' · ' + bands) if bands else ''}"
                f"{(' · ' + f'{resolution:g} m') if resolution else ''}</div>")
    return f"""
<div style="
    position: fixed; bottom: 26px; left: 26px; z-index: 9999;
    background: rgba(255,255,255,0.94); padding: 10px 12px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35); font-family: sans-serif; font-size: 12px;
    max-width: 300px; color: #111;">
  <div style="font-weight: 700; font-size: 13px; margin-bottom: 2px;">
    NDWI — Water Index
  </div>
  <div style="color:#b00020; font-weight:600; margin-bottom: 6px;">
    Continuous index — no water threshold
  </div>
  <div style="background: {gradient}; height: 12px; border-radius: 3px;
              border: 1px solid #999;"></div>
  <div style="display:flex; justify-content:space-between; margin-top: 2px;">
    <span>{vmin:g}</span><span>NDWI</span><span>{vmax:g}</span>
  </div>
  <div style="margin-top: 6px; color:#333; line-height:1.35;">
    NDWI is a spectral index. This result does not by itself establish flood
    extent, water availability, or water quality.
  </div>
  <div style="margin-top: 6px; color:#333; line-height:1.35;">
    Invalid pixels are <b>transparent</b>, never painted as water.
  </div>
  {meta}
</div>
""".strip()


def suitability_rgba(class_codes: Any) -> Any:
    """Class-code raster -> uint8 RGBA, using the suitability palette."""
    import numpy as np

    codes = np.asarray(class_codes)
    out = np.zeros((*codes.shape, 4), dtype="uint8")
    for code, rgba in SUITABILITY_COLOURS.items():
        out[codes == code] = rgba
    return out


def suitability_legend_html(crop: str = "cotton",
                            analysis_resolution: Optional[float] = None,
                            native_resolutions: Optional[Dict[str, str]] = None,
                            scenario: str = "") -> str:
    """Legend for the suitability overlay.

    It must say two things the map alone cannot: that this is an EXPERIMENTAL
    screening, and that "Unsuitable" (measured) is not "Insufficient data"
    (not measured).
    """
    swatches = "".join(
        f'<div style="display:flex; align-items:center; gap:6px; margin-top:3px;">'
        f'<span style="display:inline-block; width:18px; height:12px; '
        f'border:1px solid #777; background:rgba({r},{g},{b},{a / 255:.2f});"></span>'
        f'<span>{SUITABILITY_CLASS_NAMES[c]}</span></div>'
        for c, (r, g, b, a) in ((c, SUITABILITY_COLOURS[c]) for c in SUITABILITY_CLASS_ORDER)
    )
    native = ""
    if native_resolutions:
        native = ("<div style='margin-top:6px; color:#555; font-size:11px; "
                  "line-height:1.35;'>Native detail: "
                  + " · ".join(f"{k} {v}" for k, v in native_resolutions.items())
                  + (f"<br>Analysis grid: {analysis_resolution:g} m — resampling adds "
                     "no information." if analysis_resolution else "")
                  + "</div>")
    return f"""
<div style="
    position: fixed; bottom: 26px; left: 26px; z-index: 9999;
    background: rgba(255,255,255,0.94); padding: 10px 12px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35); font-family: sans-serif; font-size: 12px;
    max-width: 300px; color: #111;">
  <div style="font-weight: 700; font-size: 13px;">
    Experimental screening — {crop}{(' · ' + scenario) if scenario else ''}
  </div>
  <div style="color:#b00020; font-weight:600; margin: 2px 0 6px;">
    Screening classes only — not a recommendation
  </div>
  {swatches}
  <div style="margin-top: 6px; color:#333; line-height:1.35;">
    <b>Unsuitable</b> = measured and limiting.
    <b>Insufficient data</b> = not measured — never treated as unsuitable.
  </div>
  {native}
</div>
""".strip()


def rgb_legend_html(title: str, bands: str, source_crs: str, display_crs: str) -> str:
    return f"""
<div style="
    position: fixed; bottom: 26px; left: 26px; z-index: 9999;
    background: rgba(255,255,255,0.94); padding: 8px 12px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35); font-family: sans-serif; font-size: 12px;
    max-width: 290px; color: #111;">
  <div style="font-weight: 700;">{title}</div>
  <div style="color:#444;">Display channels: {bands}</div>
  <div style="color:#555; font-size: 11px;">{source_crs} → {display_crs}</div>
</div>
""".strip()


# --------------------------------------------------------------------------- #
# the map
# --------------------------------------------------------------------------- #
def build_map(
    overlays: Sequence[MapOverlay],
    footprint: Optional[Dict[str, Any]] = None,
    bounds: Optional[List[List[float]]] = None,
    tiles: str = DEFAULT_BASEMAP,
    show_footprint: bool = True,
    extra_html: Optional[str] = None,
    fit: bool = False,
    draw: bool = False,
    draw_options: Optional[Dict[str, Any]] = None,
    edit_options: Optional[Dict[str, Any]] = None,
    centre: Optional[Sequence[float]] = None,
    zoom: Optional[int] = None,
    min_zoom: int = 2,
    max_zoom: int = 19,
    marker: Optional[Sequence[float]] = None,
    marker_label: str = "Searched place",
) -> folium.Map:
    """Assemble the Folium map: a world navigation surface with data on top.

    `bounds` = [[lat_min, lon_min], [lat_max, lon_max]] of the *raster* extent.
    It is used for `fit_bounds` when the caller wants to fly to the data; it is
    never the map's extent. Without `centre`/`zoom` and without `fit`, the map
    opens on the whole world.

    `marker` = [lat, lon] of a *navigation* pin (a place found by search, or a
    point picked on the globe). It is a display pin only: it carries no data,
    it is not an overlay, it is deliberately not in the layer control (so the
    analysis layers keep their names), and it never influences an analysis.

    `tiles` is a key of `BASEMAPS`. Every base map is served from this app's own
    `/satquery-tiles` route, so the browser never contacts a provider.

    `draw=True` adds the rectangle + polygon tools. Everything drawn in the
    browser is reported back as EPSG:4326 GeoJSON; `core.roi` does the CRS work.
    """
    if centre is not None:
        start_centre = [float(centre[0]), float(centre[1])]
    elif bounds:
        start_centre = [
            (bounds[0][0] + bounds[1][0]) / 2.0,
            (bounds[0][1] + bounds[1][1]) / 2.0,
        ]
    else:
        start_centre = [WORLD_CENTRE[0], WORLD_CENTRE[1]]

    start_zoom = int(zoom) if zoom is not None else WORLD_ZOOM

    m = folium.Map(
        location=start_centre,
        zoom_start=start_zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
        zoom_control=True,          # ordinary + / - zoom control
        world_copy_jump=True,       # panning across the antimeridian keeps working
        min_zoom=min_zoom,
        max_zoom=max_zoom,
    )

    # Leaflet's container is transparent, so with no tiles the *page* shows
    # through -- and on a dark theme that reads as "the map is broken/black"
    # rather than "no imagery here". A neutral background keeps the difference
    # between "no base map" and "a black raster" visible.
    m.get_root().header.add_child(folium.Element(
        "<style>"
        ".leaflet-container{background:#e8eaed;}"
        ".leaflet-container img.leaflet-tile{mix-blend-mode:normal;}"
        "</style>"))

    # Base layers: one per public provider, all proxied by this server. The
    # browser only ever requests /satquery-tiles/... from this origin, so no
    # layer here can show a 403, an "access blocked" tile or an API-key notice.
    for key, provider in BASEMAPS.items():
        folium.TileLayer(
            tiles=tile_url(key),
            name=provider.label,
            attr=provider.attribution,
            overlay=False,                 # a base layer: radio button in the control
            control=True,
            show=(key == tiles),
            max_zoom=provider.max_zoom,
            errorTileUrl=TRANSPARENT_TILE,  # a failed tile is blank, never an error
        ).add_to(m)

    if show_footprint and footprint is not None:
        folium.GeoJson(
            footprint,
            name="Raster footprint",
            style_function=lambda _f: {
                "color": "#ff7800",
                "weight": 2,
                "fillColor": "#ff7800",
                "fillOpacity": 0.03,
                "dashArray": "4",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[k for k in (footprint.get("properties") or {})],
                aliases=[f"{k}:" for k in (footprint.get("properties") or {})],
            )
            if footprint.get("properties")
            else None,
            control=True,
            show=True,
        ).add_to(m)

    # A searched place: a navigation pin, drawn as a vector marker so it needs
    # no icon image and cannot fail as a broken tile.
    if marker is not None and len(marker) >= 2:
        try:
            pin_lat, pin_lon = float(marker[0]), float(marker[1])
        except (TypeError, ValueError):
            pin_lat = pin_lon = float("nan")
        if -90.0 <= pin_lat <= 90.0 and -180.0 <= pin_lon <= 180.0:
            folium.CircleMarker(
                location=[pin_lat, pin_lon],
                radius=7,
                color="#ffffff",
                weight=2,
                fill=True,
                fill_color="#d7263d",
                fill_opacity=0.95,
                tooltip=str(marker_label),
                control=False,        # not a data layer: it stays out of the control
            ).add_to(m)

    for ov in overlays:
        folium.raster_layers.ImageOverlay(
            image=ov.rgba,                    # folium encodes ndarray -> PNG data URI (alpha preserved)
            bounds=ov.bounds,                 # already [[lat_min, lon_min], [lat_max, lon_max]]
            name=ov.name,
            opacity=ov.opacity,
            show=ov.show,
            mercator_project=False,           # already reprojected to EPSG:3857 in core.geo
            pixelated=False,
            control=True,
            interactive=True,
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Coordinate inspection: click for a popup, and a live readout while moving.
    folium.LatLngPopup().add_to(m)
    plugins.MousePosition(
        position="bottomright",
        separator=" , ",
        prefix="lon/lat:",
        lat_formatter="function(num) {return L.Util.formatNum(num, 6);}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 6);}",
    ).add_to(m)

    if draw:
        # A named FeatureGroup holds the drawn layers. It is what Leaflet.Draw
        # edits/deletes and what we alias below for streamlit-folium.
        drawn = folium.FeatureGroup(name="Selection (drawn)", show=True, control=False)
        drawn.add_to(m)
        plugins.Draw(
            export=False,                      # no file-export button: clutter
            feature_group=drawn,
            show_geometry_on_click=False,      # folium's default pops an alert() of the GeoJSON
            position="topleft",
            draw_options=draw_options or DEFAULT_DRAW_OPTIONS,
            edit_options=edit_options or DEFAULT_EDIT_OPTIONS,
        ).add_to(m)

        # ------------------------------------------------------------------ #
        # INTEGRATION SHIMS -- both verified against streamlit-folium 0.27.4
        #
        # 1. ALIAS (only needed for standalone folium HTML, e.g. the artefacts).
        #    streamlit-folium's frontend builds `all_drawings` from
        #    `window.drawnItems.toGeoJSON().features` and nothing else; folium's
        #    Draw control instead declares a private `var drawnItems_<hash>`.
        #    In Streamlit, streamlit-folium sets the alias itself, so this is a
        #    no-op there -- it matters for `m.save()` output.
        #
        # 2. DELETE BRIDGE (needed everywhere).
        #    leaflet-draw 1.0.2 fires `deleted` on the LAYER when a shape is
        #    removed, and only fires the map-level `draw:deleted` when the user
        #    clicks "Save". streamlit-folium listens for the map-level event, so
        #    without this bridge a deleted shape would stay in `all_drawings`
        #    forever -- i.e. a stale ROI surviving deletion. Verified:
        #    window.drawnItems had 0 layers while all_drawings still reported 1.
        #
        # The bridge is injected through an <img onerror> handler because
        # streamlit-folium injects the body with innerHTML, which does NOT
        # execute <script> tags (verified: a script tag never ran, an img
        # onerror handler did).
        # ------------------------------------------------------------------ #
        m.get_root().script.add_child(
            folium.Element(
                "document.addEventListener('DOMContentLoaded', function () {"
                f" if (typeof {drawn.get_name()} !== 'undefined' && !window.drawnItems) {{"
                f"   window.drawnItems = {drawn.get_name()};"
                " }"
                "});"
            )
        )
        bridge = (
            "var _t=0;"
            "function _sq_install(){"
            " var fg=window.drawnItems;"
            " if(!fg){ if(_t++<40){setTimeout(_sq_install,100);} return; }"
            " if(fg.__sq_bridge){return;}"
            " fg.__sq_bridge=1;"
            " fg.on('layeradd',function(e){"
            "   e.layer.on('deleted',function(){"
            "     setTimeout(function(){ if(fg._map){fg._map.fire('draw:deleted',{sq_bridge:1});} },0);"
            "   });"
            " });"
            "}"
            "_sq_install();"
        )
        _inject_onerror_script(m, bridge, "draw-bridge")

    # A tile provider can be unreachable (offline, proxy, provider outage).
    # Two things cover that, and neither is custom JavaScript: every provider
    # carries an `errorTileUrl`, so a failed tile renders as plain grey instead
    # of nothing (never as data), and `NO_BASEMAP` gives the user a clean,
    # explicitly empty basemap. The Streamlit UI carries the wording.

    if extra_html:
        m.get_root().html.add_child(folium.Element(extra_html))

    if fit and bounds:
        m.fit_bounds(bounds)

    return m


# =========================================================================== #
# Phase 12 -- composed multi-condition result
# =========================================================================== #
# A composition is a NEW layer, never a replacement for the Phase 9 spatial
# layer or the Phase 10 change layer: those stay in the layer control and keep
# their own legends. The palette is categorical for the same reason Phase 9's
# is: a match is not "more" or "less" than a non-match.
#
# MATCH is teal here rather than Phase 9's green so that two result layers on
# the same map cannot be confused with one another. NO MATCH and INSUFFICIENT
# are the same grey and amber used everywhere else -- grey and amber mean one
# thing in this application.
# =========================================================================== #
COMPOSITION_STATE_COLOURS: Dict[int, Tuple[int, int, int, int]] = {
    2: (13, 143, 168, 210),        # MATCH        - teal
    1: (150, 150, 150, 60),        # NO MATCH     - faint grey, still drawn
    0: (230, 159, 0, 170),         # INSUFFICIENT - amber, never grey
}
COMPOSITION_STATE_ORDER: Tuple[int, ...] = (2, 1, 0)
COMPOSITION_STATE_NAMES: Dict[int, str] = {
    2: "Match — satisfies every condition",
    1: "No match — measured, does not satisfy",
    0: "Unknown — insufficient data, not a match and not a non-match",
}


def composition_rgba(state_codes: Any) -> Any:
    """Three-valued composition mask -> uint8 RGBA."""
    import numpy as np

    codes = np.asarray(state_codes)
    out = np.zeros((*codes.shape, 4), dtype="uint8")
    for code, rgba in COMPOSITION_STATE_COLOURS.items():
        out[codes == code] = rgba
    return out


def composition_legend_html(expression: str = "",
                            analysis_resolution: Optional[float] = None,
                            thresholds: Sequence[str] = (),
                            dates: Optional[str] = None,
                            matched_cells: Optional[int] = None,
                            unknown_cells: Optional[int] = None) -> str:
    """Legend for the composed-condition overlay.

    Carries the scientific boundary onto the map: the layer shows where the
    conditions hold together, and nothing about why.
    """
    swatches = "".join(
        f'<div style="display:flex; align-items:center; gap:6px; margin-top:3px;">'
        f'<span style="display:inline-block; width:18px; height:12px; '
        f'border:1px solid #777; background:rgba({r},{g},{b},{a / 255:.2f});"></span>'
        f'<span>{COMPOSITION_STATE_NAMES[c]}</span></div>'
        for c in COMPOSITION_STATE_ORDER
        for (r, g, b, a) in (COMPOSITION_STATE_COLOURS[c],)
    )
    query = (f"<div style='margin-top:4px; color:#333; font-size:11px; "
             f"line-height:1.35;'><b>{expression}</b></div>" if expression else "")
    threshold_lines = "".join(
        f"<div style='color:#333; font-size:11px; line-height:1.35;'>{t}</div>"
        for t in thresholds)
    thresholds_html = (
        f"<div style='margin-top:6px;'><div style='font-weight:600; font-size:11px;'>"
        f"Thresholds</div>{threshold_lines}</div>" if threshold_lines else "")
    dates_html = (f"<div style='margin-top:5px; color:#333; font-size:11px;'>"
                  f"{dates}</div>" if dates else "")
    counts = ""
    if unknown_cells:
        counts = (f"<div style='margin-top:5px; color:#8a5a00; font-size:11px;'>"
                  f"{unknown_cells:,} cells could not be decided and are drawn "
                  f"amber — they are not counted as non-matches.</div>")
    elif matched_cells is not None:
        counts = (f"<div style='margin-top:5px; color:#333; font-size:11px;'>"
                  f"{matched_cells:,} cells match; every measured cell was "
                  f"decided.</div>")
    native = (f"<div style='margin-top:6px; color:#555; font-size:11px;'>"
              f"Analysis grid: {analysis_resolution:g} m — the composition is "
              f"computed on one verified grid, never resampled into "
              f"agreement.</div>" if analysis_resolution else "")
    return f"""
<div style="
    position: fixed; bottom: 26px; left: 26px; z-index: 9999;
    background: rgba(255,255,255,0.94); padding: 10px 12px; border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35); font-family: sans-serif; font-size: 12px;
    max-width: 340px; color: #111;">
  <div style="font-weight: 700; font-size: 13px;">Composed condition result</div>
  <div style="color:#b00020; font-weight:600; margin: 2px 0 6px;">
    Geographic evidence — not causal attribution
  </div>
  {swatches}
  {query}
  {thresholds_html}
  {dates_html}
  {counts}
  {native}
  <div style="margin-top: 6px; color:#333; line-height:1.35;">
    Several conditions holding together tells you where they coincide. It does
    not tell you what caused them.
  </div>
</div>
""".strip()
