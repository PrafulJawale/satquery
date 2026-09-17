"""A same-origin tile proxy for the SatQuery map.

Why this exists
---------------
The map used to let the *browser* fetch tiles straight from OpenStreetMap. On a
restricted network OSM answers with HTTP 403 PNGs that literally read "Access
blocked", so the map filled up with error tiles that look like an application
bug. The browser's network cannot be fixed from here; the server's can.

So the server fetches every tile itself -- the sandbox reaches OSM, Esri and
CARTO fine -- caches it on disk, and serves it from the app's *own* origin:

    /satquery-tiles/<provider>/<z>/<x>/<y>.png

The browser therefore only ever talks to the app. It cannot receive a 403, an
"access blocked" tile, or an API-key error, because it never contacts a tile
provider at all.

Providers are public, no-API-key and OSM-compatible. Nothing here fabricates
imagery: every byte served was fetched from the named provider, and the
attribution for each provider is exported alongside the URL.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Public, no-API-key, OSM-compatible raster tile providers.
USER_AGENT = "SatQuery-AI/1.0 (research prototype; low-volume tile use)"
FETCH_TIMEOUT_S = 10.0
MAX_ZOOM = 19
MIN_ZOOM = 0

# Where cached tiles live. Outside the workspace: regenerable, and it must not
# bloat the project snapshot.
CACHE_DIR = Path(os.environ.get("SATQUERY_TILE_CACHE", "/tmp/satquery_tile_cache"))

# The URL prefix the browser uses. Absolute path: the folium map is rendered in
# a srcdoc iframe, so a relative path would resolve against the app's mount
# point instead of the server root.
ROUTE = "/satquery-tiles"


@dataclass(frozen=True)
class TileProvider:
    """A public raster tile provider reached through this proxy."""

    key: str
    label: str
    url_template: str
    attribution: str
    max_zoom: int = MAX_ZOOM
    subdomains: str = "abc"
    fmt: str = "image/png"


PROVIDERS: Dict[str, TileProvider] = {
    "osm": TileProvider(
        key="osm",
        label="Streets",
        url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="© OpenStreetMap contributors",
    ),
    "satellite": TileProvider(
        key="satellite",
        label="Satellite",
        url_template=(
            "https://server.arcgisonline.com/ArcGIS/rest/services"
            "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attribution="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        max_zoom=18,
    ),
}

DEFAULT_PROVIDER = "osm"

# A single transparent PNG. Served when a tile cannot be fetched, so a transient
# failure shows as *no imagery* and never as an error message baked into a tile.
_TRANSPARENT_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6300010000050001aa8b1a3b00000000"
    "49454e44ae426082"
)

# Simple operational counters, surfaced in the UI so a dead provider is visible
# instead of silent.
STATS: Dict[str, int] = {"served": 0, "cached": 0, "fetched": 0, "failed": 0}


def _cache_path(key: str, z: int, x: int, y: int) -> Path:
    return CACHE_DIR / key / str(z) / str(x) / f"{y}.png"


def _fetch(provider: TileProvider, z: int, x: int, y: int) -> Optional[bytes]:
    """Fetch one tile from the provider, with a descriptive User-Agent."""
    host = provider.url_template
    if "{s}" in host:
        sub = provider.subdomains[(x + y) % len(provider.subdomains)]
        url = host.format(s=sub, z=z, x=x, y=y)
    else:
        url = host.format(z=z, x=x, y=y)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/png,image/jpeg,image/*",
        },
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
        data = response.read()
    # A provider error page is not a tile: refuse to serve something that is
    # obviously not an image, so error text can never reach the map.
    if not data:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"\xff\xd8":
        return data
    return None


def get_tile(key: str, z: int, x: int, y: int) -> Tuple[bytes, str, int]:
    """Return (bytes, media_type, http_status) for one tile.

    Never returns provider error content: on any failure the caller gets the
    transparent placeholder, which renders as "no imagery here".
    """
    provider = PROVIDERS.get(key)
    if provider is None:
        return _TRANSPARENT_PNG, "image/png", 404
    if not (MIN_ZOOM <= z <= provider.max_zoom):
        return _TRANSPARENT_PNG, "image/png", 404
    limit = 2 ** z
    if not (0 <= x < limit and 0 <= y < limit):
        return _TRANSPARENT_PNG, "image/png", 404

    path = _cache_path(key, z, x, y)
    if path.is_file():
        try:
            STATS["cached"] += 1
            STATS["served"] += 1
            return path.read_bytes(), provider.fmt, 200
        except OSError:
            pass

    try:
        data = _fetch(provider, z, x, y)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        data = None

    if data is None:
        STATS["failed"] += 1
        return _TRANSPARENT_PNG, "image/png", 200

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError:
        pass
    STATS["fetched"] += 1
    STATS["served"] += 1
    # Label the bytes by what they actually are: Esri serves JPEG, OSM serves
    # PNG, and a wrong content type is exactly the kind of detail that turns
    # into a broken tile later.
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        media = "image/png"
    elif data[:2] == b"\xff\xd8":
        media = "image/jpeg"
    else:
        media = provider.fmt
    return data, media, 200


# --------------------------------------------------------------------------- #
# ASGI endpoint (mounted into the running Streamlit app by serve.py)
# --------------------------------------------------------------------------- #
async def tile_endpoint(request) -> object:  # noqa: ANN001 - Starlette Request
    """`/satquery-tiles/<provider>/<z>/<x>/<y>.png`"""
    from starlette.responses import Response

    params = request.path_params
    try:
        z = int(params["z"])
        x = int(params["x"])
        y = int(str(params["y"]).split(".")[0])  # tolerate ".png"
    except (KeyError, ValueError):
        return Response(_TRANSPARENT_PNG, media_type="image/png", status_code=404)

    import asyncio

    body, media_type, status = await asyncio.get_running_loop().run_in_executor(
        None, get_tile, params.get("provider", ""), z, x, y
    )
    return Response(
        body,
        media_type=media_type,
        status_code=status,
        headers={
            "Cache-Control": "public, max-age=604800",
            "X-SatQuery-Tile": "proxied",
        },
    )


def install(app) -> None:  # noqa: ANN001 - Starlette application
    """Mount the tile route onto an existing Starlette app.

    The route is inserted at the *front* of the router: Streamlit registers a
    catch-all that serves the app shell for unknown paths, and `add_route`
    appends, so a tile request would otherwise be answered with HTML.
    """
    from starlette.routing import Route

    route = Route(
        f"{ROUTE}/{{provider}}/{{z}}/{{x}}/{{y}}", tile_endpoint, methods=["GET"]
    )
    app.router.routes.insert(0, route)


def proxy_mounted() -> bool:
    """True when THIS process was started through `serve.py`.

    `serve.py` patches `create_starlette_app` before Streamlit builds its
    server -- that is the only supported way to add a route to a Streamlit app.
    Started any other way (`streamlit run app.py`), the route does not exist and
    every `/satquery-tiles/...` request is answered with the app's own HTML, so
    the map silently loses its base map. The UI checks this and says so instead
    of showing an empty surface.
    """
    try:
        import streamlit.web.server.starlette.starlette_server as starlette_server
    except Exception:
        return False
    return bool(getattr(getattr(starlette_server, "create_starlette_app", None),
                        "_satquery_patched", False))


def tile_url(key: str) -> str:
    """The browser-facing URL template for a provider."""
    return f"{ROUTE}/{key}/{{z}}/{{x}}/{{y}}.png"


def warm_cache(keys=("osm", "satellite"), zooms=(0, 1, 2)) -> int:
    """Pre-fetch the low-zoom tiles so a world view paints instantly."""
    count = 0
    for key in keys:
        for z in zooms:
            for x in range(2 ** z):
                for y in range(2 ** z):
                    body, _, status = get_tile(key, z, x, y)
                    if status == 200 and len(body) > 100:
                        count += 1
    return count


if __name__ == "__main__":
    started = time.time()
    n = warm_cache()
    print(f"warmed {n} tiles in {time.time() - started:.1f}s")
    print("stats:", STATS)
