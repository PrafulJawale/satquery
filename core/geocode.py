"""core/geocode.py -- place search (Nominatim), run on the SERVER.

The search box must not depend on the browser reaching a geocoder: on a
restricted network that request is exactly as likely to be blocked as a tile
request. So the lookup happens here, in Python, and the app only ever receives
coordinates.

Nominatim's usage policy requires a descriptive User-Agent and at most one
request per second; both are honored here, and results are cached in the
session so a re-run does not re-query.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SatQuery-AI/1.0 (geospatial research app; low-volume use)"
TIMEOUT_S = 10.0
_MIN_INTERVAL_S = 1.0      # Nominatim policy: no more than 1 request / second
_last_call: Dict[str, float] = {"t": 0.0}

#: Shorter than this and Nominatim is mostly noise (and mostly load for it).
MIN_QUERY_CHARS = 3

#: Shown in the UI: Nominatim's licence requires attribution, and the app
#: credits the geocoder it actually uses rather than a generic "maps".
ATTRIBUTION = "Place search by OpenStreetMap via Nominatim"


@dataclass(frozen=True)
class Place:
    """One geocoding result."""

    name: str
    lat: float
    lon: float
    bbox: Optional[List[float]] = None     # [south, north, west, east]

    @property
    def zoom_for_bbox(self) -> Optional[int]:
        """A zoom level that frames the place's bounding box, if it has one."""
        if not self.bbox or len(self.bbox) != 4:
            return None
        south, north, west, east = self.bbox
        span = max(north - south, east - west)
        if span <= 0:
            return None
        # 360 degrees of longitude at zoom 0, halving each level.
        import math

        zoom = int(math.log2(360.0 / span)) if span > 0 else None
        return max(2, min(18, zoom)) if zoom is not None else None

    def label(self) -> str:
        return f"{self.name} ({self.lat:.4f}, {self.lon:.4f})"


def geocode(query: str, limit: int = 5) -> List[Place]:
    """Look up `query`. Returns [] when nothing is found or the lookup fails.

    A failure is never raised into the UI: the search box simply shows
    "no matches", which is honest, rather than an error about a geocoder.
    """
    query = (query or "").strip()
    if len(query) < MIN_QUERY_CHARS:
        return []

    wait = _MIN_INTERVAL_S - (time.time() - _last_call["t"])
    if wait > 0:
        time.sleep(wait)
    _last_call["t"] = time.time()

    params = urllib.parse.urlencode(
        {"q": query, "format": "jsonv2", "limit": str(limit), "addressdetails": "0"}
    )
    request = urllib.request.Request(
        f"{NOMINATIM_URL}?{params}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        import json

        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    places: List[Place] = []
    for item in raw or []:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            # An impossible coordinate is not a place; it is a bug (or a
            # mangled response). Skipping it is the only honest option -- there
            # is nothing to round it to.
            continue
        bbox = None
        raw_bbox = item.get("boundingbox")
        if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
            try:
                south, north, west, east = (float(v) for v in raw_bbox)
                bbox = [south, north, west, east]
            except (TypeError, ValueError):
                bbox = None
        places.append(
            Place(name=item.get("display_name", query)[:120], lat=lat, lon=lon, bbox=bbox)
        )
    return places
