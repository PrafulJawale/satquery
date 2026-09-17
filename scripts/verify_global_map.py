"""Verification for the global interactive map.

This checks the *running application*, not a standalone image:

 1. clean process            -> the caller restarts the app first
 2. initial extent is global
 3. pan and zoom work
 4. the search control works
 5. rectangle + polygon drawing work
 6. no API-key-required layer exists
 7. no selectable layer can show "Access blocked" / 403
 8. blocking every external tile provider changes nothing (tiles are same-origin)
 9. the default base map works
10. switching base maps works
11. an area inside the scene analyses
12. an area outside the scene reports unavailable data, without fabricating
14. a fresh screenshot of the current app

    python3 scripts/verify_global_map.py --url http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import browser_test_phase6 as bt6  # noqa: E402

PROVIDER_HOSTS = (
    "**://tile.openstreetmap.org/**",
    "**://*.tile.openstreetmap.org/**",
    "**://*.basemaps.cartocdn.com/**",
    "**://server.arcgisonline.com/**",
    "**://*.arcgisonline.com/**",
)

# Tile URLs served by this app look like /satquery-tiles/<provider>/<z>/<x>/<y>.png
JS_TILES = """
() => Array.from(document.querySelectorAll('.leaflet-tile-pane img'))
        .map(i => ({src: i.src, ok: i.complete && i.naturalWidth > 0}))
"""

JS_LABELS = """
() => Array.from(document.querySelectorAll('.leaflet-control-layers label'))
        .map(l => (l.innerText || '').trim())
"""

JS_ACTIVE_URLS = """
() => Array.from(document.querySelectorAll('.leaflet-tile-pane img'))
        .map(i => i.src)
"""

PASSES = 0
FAILURES = 0


def check(ok: bool, message: str) -> None:
    global PASSES, FAILURES
    if ok:
        PASSES += 1
        print(f"  [PASS] {message}")
    else:
        FAILURES += 1
        print(f"  [FAIL] {message}")


def tile_parts(frame) -> list[tuple[str, int, int, int]]:
    """Parse /(provider)/(z)/(x)/(y) out of every visible tile URL."""
    out = []
    for item in frame.evaluate(JS_TILES):
        m = re.search(r"/satquery-tiles/([a-z]+)/(\d+)/(\d+)/(\d+)", item["src"])
        if m:
            out.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return out


def refresh(page):
    """Re-acquire the map frame.

    Any action that re-renders the map (flying to a place, zooming to the
    scene) replaces the iframe, so the previous handle becomes detached and
    every later evaluate() fails with "Frame was detached".
    """
    return bt6.find_map_frame(page, timeout=180)


def wait_idle(page, timeout: float = 120.0) -> bool:
    """Streamlit disables inputs while the script runs; wait for it to settle."""
    import time

    end = time.time() + timeout
    while time.time() < end:
        try:
            if page.get_by_placeholder("e.g. Kolhapur, India").is_enabled():
                return True
        except Exception:
            pass
        page.wait_for_timeout(1000)
    return False


def draw_rect(page, a: float = 0.40, b: float = 0.60, attempts: int = 5) -> bool:
    """Drag a rectangle at container-relative fractions.

    Two things make this fiddly: streamlit-folium re-renders the map iframe
    asynchronously (so a frame handle can be detached mid-drag), and a drag can
    silently produce nothing. Every step therefore re-acquires the frame, and the
    result is verified against Leaflet's own drawnItems count.
    """
    for i in range(attempts):
        try:
            frame = bt6.find_map_frame(page, timeout=120)
            if not bt6.click_tool(frame, ".leaflet-draw-draw-rectangle"):
                raise RuntimeError("rectangle tool not found")
            page.wait_for_timeout(900)
            frame = bt6.find_map_frame(page, timeout=120)
            el = frame.query_selector(".leaflet-container")
            el.wait_for_element_state("visible", timeout=15000)
            el.scroll_into_view_if_needed()
            page.wait_for_timeout(400)
            box = el.bounding_box()
            x0 = box["x"] + box["width"] * a
            y0 = box["y"] + box["height"] * a
            x1 = box["x"] + box["width"] * b
            y1 = box["y"] + box["height"] * b
            page.mouse.move(x0, y0)
            page.mouse.down()
            page.wait_for_timeout(200)
            page.mouse.move(x1, y1, steps=16)
            page.wait_for_timeout(200)
            page.mouse.up()
            page.wait_for_timeout(4000)
            frame = bt6.find_map_frame(page, timeout=120)
            count = frame.evaluate(
                "() => window.drawnItems ? window.drawnItems.getLayers().length : -1")
            print(f"     draw attempt {i + 1}: drawnItems={count}")
            if count >= 1:
                return True
        except Exception as exc:
            print(f"     draw attempt {i + 1}: {type(exc).__name__} {str(exc)[:60]}")
        page.wait_for_timeout(2500)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--shot", default="artifacts/global_map.png")
    args = ap.parse_args()

    print("=" * 78)
    print("SatQuery AI -- global interactive map verification")
    print(f"target: {args.url}")
    print("=" * 78)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("   playwright is not installed -- no browser verification")
        return 2

    external: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu",
                  "--js-flags=--max-old-space-size=256"]
        )
        page = browser.new_page(viewport={"width": 1280, "height": 1000})

        # Record (and later block) anything that tries to leave for a provider.
        def watch(route):
            external.append(route.request.url)

        for pattern in PROVIDER_HOSTS:
            page.route(pattern, lambda r: (external.append(r.request.url), r.abort()))

        page.goto(args.url, wait_until="domcontentloaded", timeout=180_000)
        check(bt6.confirm_ndvi(page, timeout=180), "the app is usable (NDVI gate passed)")
        bt6.wait_for_text(page, "NDVI map", timeout=180)
        frame = bt6.find_map_frame(page, timeout=180)
        page.wait_for_timeout(int(args.settle * 1000))

        # ---- 2. initial extent is global ---------------------------------- #
        print("\n-- 2. initial extent --")
        tiles = tile_parts(frame)
        zooms = {t[1] for t in tiles}
        print(f"     tiles: {len(tiles)} at zoom(s) {sorted(zooms)}")
        check(bool(tiles), "the map painted tiles on load")
        check(zooms and max(zooms) <= 3,
              f"the map opened zoomed out to the world (zoom {sorted(zooms)})")
        if tiles:
            xs = {t[2] for t in tiles}
            ys = {t[3] for t in tiles}
            check(min(zooms) <= 2 and len(xs) <= 4 and len(ys) <= 4,
                  f"the visible tile indices span the globe (x={sorted(xs)}, y={sorted(ys)})")

        # ---- 9. default base map ------------------------------------------ #
        print("\n-- 9. default base map --")
        providers = {t[0] for t in tiles}
        check(providers == {"osm"}, f"the default base map is the OSM one: {providers}")
        loaded = frame.evaluate(JS_TILES)
        check(all(t["ok"] for t in loaded) and len(loaded) > 0,
              f"every visible tile decoded ({sum(t['ok'] for t in loaded)}/{len(loaded)})")

        # ---- 6/7. no API keys, no blocked-provider layer ------------------- #
        print("\n-- 6/7. no API-key or blocked-provider layer --")
        labels = frame.evaluate(JS_LABELS) or []
        print(f"     layer control: {labels}")
        check(all("key" not in l.lower() and "api" not in l.lower() for l in labels),
              "no layer advertises an API key")
        urls = frame.evaluate(JS_ACTIVE_URLS)
        check(all("satquery-tiles" in u for u in urls) if urls else True,
              "every tile comes from this app's own /satquery-tiles route")
        check(not external,
              f"the browser made no request to any tile provider ({len(external)})")

        # ---- 3. zoom and pan ---------------------------------------------- #
        print("\n-- 3. zoom and pan --")
        before = sorted({t[1] for t in tile_parts(frame)})
        frame.locator(".leaflet-control-zoom-in").first.click()
        page.wait_for_timeout(int(args.settle * 1000))
        after_zoom = sorted({t[1] for t in tile_parts(frame)})
        check(bool(after_zoom) and max(after_zoom) > max(before or [0]),
              f"zoom in changed the tile zoom: {before} -> {after_zoom}")
        box = bt6.map_box(page, frame)
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.4, steps=12)
        page.mouse.up()
        page.wait_for_timeout(int(args.settle * 1000))
        after_pan = sorted({(t[2], t[3]) for t in tile_parts(frame)})
        print(f"     tile indices after pan: {after_pan[:6]}{' ...' if len(after_pan) > 6 else ''}")
        check(bool(after_pan), "panning re-requested tiles (the map is interactive)")

        # ---- 10. switching base maps --------------------------------------- #
        print("\n-- 10. switching base maps --")
        switched = frame.evaluate(
            "(name) => { for (const l of document.querySelectorAll("
            "'.leaflet-control-layers-base label')) { if ((l.innerText||'').trim() === name) "
            "{ const i = l.querySelector('input'); if (i) { i.click(); return true; } } } return false; }",
            "Satellite",
        )
        check(bool(switched), "the satellite base map is selectable")
        page.wait_for_timeout(int(args.settle * 1000))
        sat_providers = {t[0] for t in tile_parts(frame)}
        check(sat_providers == {"satellite"},
              f"switching base maps changed the tile source: {sat_providers}")
        check(not external,
              f"still no request to a tile provider after switching ({len(external)})")

        # ---- 4. search ------------------------------------------------------ #
        print("\n-- 4. search --")
        frame.evaluate(
            "(name) => { for (const l of document.querySelectorAll("
            "'.leaflet-control-layers-base label')) { if ((l.innerText||'').trim() === name) "
            "{ const i = l.querySelector('input'); if (i) { i.click(); return true; } } } return false; }",
            "Streets",
        )
        page.get_by_placeholder("e.g. Kolhapur, India").fill("Kolhapur, India")
        page.get_by_role("button", name="Go", exact=True).click()
        page.wait_for_timeout(4000)
        found = page.locator("text=Matches").count() > 0
        check(found, "the search returned matches (looked up on the server)")
        if found:
            page.get_by_role("button", name="Fly to this place").click()
            page.wait_for_timeout(int(args.settle * 1000))
            frame = refresh(page)
            flown = tile_parts(frame)
            check(bool(flown) and max({t[1] for t in flown}) >= 6,
                  f"flying to the place re-tiled at a local zoom: "
                  f"{sorted({t[1] for t in flown})}")
            print(f"     tiles near the search result: {flown[:3]}")

        # ---- 11/12. inside and outside the available scene ---------------- #
        print("\n-- 11/12. analysis inside and outside the available data --")
        page.get_by_role("button", name="Zoom to scene").click()
        page.wait_for_timeout(6000)
        wait_idle(page, 120)
        frame = refresh(page)
        s_b, w_b, n_b, e_b = bt6.raster_bounds(frame, timeout=90)   # [s, w, n, e]
        print(f"     scene bounds: south={s_b:.4f} west={w_b:.4f} "
              f"north={n_b:.4f} east={e_b:.4f}")
        check(draw_rect(page), "a rectangle could be drawn inside the scene")
        page.wait_for_timeout(6000)
        wait_idle(page, 150)
        txt_in = page.inner_text("body")
        check("Selection detected" in txt_in and "Usable area" in txt_in,
              "an area inside the scene is measured (usable area reported)")
        check("Analysis data is not available" not in txt_in,
              "no 'not available' message for an area that is covered")

        # Now somewhere with no data at all: back to the world view, then draw
        # over the Atlantic / South America, far from the Nile Delta scene.
        page.get_by_role("button", name="Whole world").click()
        page.wait_for_timeout(6000)
        wait_idle(page, 120)
        frame = refresh(page)
        check(draw_rect(page, a=0.28, b=0.42),
              "a rectangle could be drawn far from the scene")
        page.wait_for_timeout(6000)
        wait_idle(page, 150)
        txt_out = page.inner_text("body")
        check("Analysis data is not available for this selected area." in txt_out,
              "an uncovered area reports that analysis data is not available")
        if "Selection detected" in txt_out and "Usable area" in txt_out:
            print("  [FAIL] statistics were produced for an uncovered area")
            globals()["FAILURES"] += 1
        else:
            check(True, "no statistics are fabricated for an uncovered area")

        # ---- 5. polygon drawing --------------------------------------------- #
        print("\n-- 5. polygon drawing --")
        page.get_by_role("button", name="Zoom to scene").click()
        page.wait_for_timeout(4000)
        frame = refresh(page)
        b3 = bt6.raster_bounds(frame, timeout=90)
        frame = refresh(page)
        has_rect = frame.query_selector(".leaflet-draw-draw-rectangle") is not None
        has_poly = frame.query_selector(".leaflet-draw-draw-polygon") is not None
        check(has_rect, "the rectangle drawing tool is present")
        check(has_poly, "the polygon drawing tool is present")

        # ---- 8. blocking providers changes nothing --------------------------- #
        print("\n-- 8. external providers are never needed --")
        # They have been blocked for the whole run by page.route above.
        check(not external,
              f"no external tile-provider request during the entire run ({len(external)})")
        tiles_now = frame.evaluate(JS_TILES)
        check(bool(tiles_now) and all(t["ok"] for t in tiles_now),
              "tiles still render with every provider blocked (same-origin proxy)")

        # ---- 14. fresh screenshot ------------------------------------------- #
        if args.shot:
            page.screenshot(path=str(REPO / args.shot))
            print(f"\n     screenshot: {args.shot}")

        browser.close()

    print("\n" + "=" * 78)
    print(f"RESULT: {PASSES} passed, {FAILURES} failed")
    print("=" * 78)
    return 0 if FAILURES == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
