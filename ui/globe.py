"""ui/globe.py -- the PRIMARY map: a 3D globe, with a flat detail mode.

This module owns the one and only map surface in SatQuery AI.

WHAT IT IS
----------
The map is a **3D spherical globe** (CesiumJS): drag to rotate, scroll or use
the zoom control to fly towards a location, search for any place on Earth and
the camera goes there. The analysis imagery -- true colour, false colour, NDVI
and the evidence layers -- is draped on the globe in its true geographic
position, from the same bounds the 2D pipeline uses, so what you see is where
the numbers came from.

The flat Leaflet map still exists, but it is **not** a second map: it is an
optional *detail mode* the user opens deliberately (the "2D map" control), for
pixel-precise drawing and inspection, and while it is open the globe is not
rendered at all. One map at a time, and the globe is the default.

WHY THE DETAIL MODE STILL EXISTS
--------------------------------
A globe is the right surface for orientation and navigation; a projected,
north-up raster canvas is the right surface for measuring. Rectangle and
polygon drawing, pixel inspection and overlay comparison are exact on the flat
projection, and the analysis code has always worked on that projection. Rather
than approximate them on a sphere, the app offers the flat view as an explicit
mode -- same data, same bounds, same CRS, switched by the user.

ENGINE, DEPENDENCIES AND DEGRADATION
------------------------------------
* CesiumJS is loaded from a **pinned CDN version**; the app's imagery still
  comes from this app's own `/satquery-tiles` proxy, so no provider is ever
  contacted by the browser and **no API key is needed** for anything.
* If the 3D engine cannot load (no network, blocked CDN, no WebGL), the
  component falls back to an orthographic globe drawn on a 2D canvas from the
  same proxied imagery, tells the app, and offers the flat detail mode for
  overlays and drawing. It never renders an empty box.
* The Streamlit component protocol is implemented by hand in
  `globe_frontend/index.html`: no npm, no build step.

WHAT IT NEVER DOES
------------------
It reports coordinates and geometry; it never computes, re-projects or decides
anything. Searching a place moves the camera -- it does not run an analysis.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit.components.v1 as components

from core.geo import rgba_to_png_bytes

#: Served by Streamlit itself (same origin) -- see `index.html`.
FRONTEND_DIR: Path = Path(__file__).resolve().parent / "globe_frontend"

_component = components.declare_component("satquery_map", path=str(FRONTEND_DIR))

MAP_HEIGHT: int = 600
MAP_MIN_HEIGHT: int = 320

#: Camera heights (metres) used when the app asks for a named view.
HOME_HEIGHT_M: float = 2.4e7


def _finite(value: Any) -> bool:
    try:
        return value is not None and float(value) == float(value)
    except (TypeError, ValueError):
        return False


def _png_data_uri(rgba: Any) -> Optional[str]:
    """Encode an RGBA array as a data URI the globe can drape on the sphere."""
    if rgba is None:
        return None
    try:
        png = rgba_to_png_bytes(rgba)
    except Exception:
        return None
    if not png:
        return None
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def overlay_payload(overlays: Sequence[Any]) -> List[Dict[str, Any]]:
    """Turn `ui.map.MapOverlay` objects into JSON the globe can draw.

    Each overlay keeps exactly the bounds the 2D map uses
    ([[south, west], [north, east]]), so a layer cannot drift between modes.
    """
    payload: List[Dict[str, Any]] = []
    for ov in overlays or []:
        name = str(getattr(ov, "name", "") or "layer")
        bounds = getattr(ov, "bounds", None)
        if not bounds or len(bounds) < 2:
            continue
        try:
            south, west = float(bounds[0][0]), float(bounds[0][1])
            north, east = float(bounds[1][0]), float(bounds[1][1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
            continue
        if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
            continue
        image = _png_data_uri(getattr(ov, "rgba", None))
        if not image:
            continue
        payload.append({
            "name": name,
            "image": image,
            "south": south, "west": west, "north": north, "east": east,
            "opacity": float(getattr(ov, "opacity", 1.0) or 1.0),
            "show": bool(getattr(ov, "show", True)),
        })
    return payload


def map_args(
    *,
    overlays: Optional[Sequence[Any]] = None,
    centre: Optional[Sequence[float]] = None,
    zoom: Optional[int] = None,
    height_m: Optional[float] = None,
    base: str = "satellite",
    attribution: str = "",
    footprint: Optional[Sequence[Sequence[float]]] = None,
    roi: Optional[Dict[str, Any]] = None,
    marker: Optional[Sequence[float]] = None,
    marker_label: str = "",
    height_px: int = MAP_HEIGHT,
) -> Dict[str, Any]:
    """Build the (JSON-safe) argument dict for the map frontend.

    Pure function: unit tests call this directly, without Streamlit.
    """
    box: Optional[list] = None
    if footprint is not None and len(footprint) >= 2:
        try:
            south, west = float(footprint[0][0]), float(footprint[0][1])
            north, east = float(footprint[1][0]), float(footprint[1][1])
        except (TypeError, ValueError, IndexError):
            south = west = north = east = float("nan")
        if all(-90.0 <= v <= 90.0 for v in (south, north)) and \
           all(-180.0 <= v <= 180.0 for v in (west, east)):
            box = [[south, west], [north, east]]

    point: Optional[list] = None
    if centre is not None and len(centre) >= 2 and _finite(centre[0]) and _finite(centre[1]):
        lat, lon = float(centre[0]), float(centre[1])
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            point = [lat, lon]

    pin: Optional[list] = None
    if marker is not None and len(marker) >= 2 and _finite(marker[0]) and _finite(marker[1]):
        mlat, mlon = float(marker[0]), float(marker[1])
        if -90.0 <= mlat <= 90.0 and -180.0 <= mlon <= 180.0:
            pin = [mlat, mlon]

    return {
        "overlays": overlay_payload(overlays or []),
        "centre": point,
        "zoom": None if zoom is None else int(zoom),
        "height_m": None if height_m is None else float(height_m),
        "base": str(base or "satellite"),
        "attribution": str(attribution or ""),
        "footprint": box,
        "roi": roi if isinstance(roi, dict) else None,
        "marker": pin,
        "marker_label": str(marker_label or ""),
        "height_px": int(max(MAP_MIN_HEIGHT, height_px)),
    }


def render_globe(*, key: str = "satquery_map", **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Render the globe map and return its latest event (or None).

    Events: {"type": "click"|"roi"|"base"|"layers"|"mode"|"unavailable", ...}
    with a monotonically increasing "id", so the app can act on each one
    exactly once -- a component value persists across reruns, like any widget.
    """
    return _component(key=key, default=None, **map_args(**kwargs))
