"""Phase 5 BROWSER test -- genuinely drives the map with a mouse.

This is the only test in the repo that proves a human can actually draw: it
launches Chromium (Playwright), clicks Leaflet.Draw's rectangle tool, DRAGS the
mouse across the map, and then checks what the Streamlit app reports back.

Coordinates are NOT guessed from fractions of the map: the raster overlay's real
lat/lon bounds are read out of Leaflet and converted to page coordinates with
`latLngToContainerPoint`, so "inside", "hanging off the edge" and "far outside"
mean exactly what they say.

Requires the app to be running:

    streamlit run app.py --server.address 0.0.0.0 --server.port 8501

Run:
    python scripts/browser_test_phase5.py [--url http://127.0.0.1:8501]

Screenshots are written to artifacts/phase5_browser_*.png.
Exits 0 only if every scenario passes; 2 means Playwright is unavailable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS = REPO_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

_results: list[tuple[bool, str]] = []


def check(ok: bool, message: str) -> None:
    _results.append((bool(ok), message))
    print(f"   [{'PASS' if ok else 'FAIL'}] {message}", flush=True)


JS_OVERLAY_BOUNDS = """() => {
  const m = window.drawnItems && window.drawnItems._map;
  if (!m) return null;
  let bounds = null;
  m.eachLayer(l => { if (l._image && l._bounds) bounds = l._bounds; });
  if (!bounds) return null;
  return [bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast()];
}"""

JS_LAST_SHAPE_CENTRE = """() => {
  const layers = window.drawnItems ? window.drawnItems.getLayers() : [];
  if (!layers.length) return null;
  const c = layers[layers.length - 1].getBounds().getCenter();
  const p = window.drawnItems._map.latLngToContainerPoint(c);
  return [p.x, p.y];
}"""

JS_LAYER_COUNT = "() => (window.drawnItems ? window.drawnItems.getLayers().length : -1)"


def find_map_frame(page, timeout: float = 120.0):
    """Wait for the streamlit-folium iframe that actually contains a Leaflet map."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for frame in page.frames:
            try:
                if frame.query_selector(".leaflet-container"):
                    return frame
            except Exception:
                continue
        page.wait_for_timeout(500)
    raise RuntimeError("no Leaflet map iframe appeared")


def map_box(page, frame):
    """Bounding box of the Leaflet container in PAGE coordinates.

    Playwright reports child-frame element boxes relative to the main viewport,
    so no iframe offset must be added -- but the map sits far down a long
    Streamlit page and must be scrolled into view before mouse events land.
    """
    el = frame.query_selector(".leaflet-container")
    el.wait_for_element_state("visible")
    el.scroll_into_view_if_needed()
    page.wait_for_timeout(250)
    return el.bounding_box()


def point_of(page, frame, lat: float, lon: float):
    """(lat, lon) -> page coordinates, via Leaflet's own projection."""
    box = map_box(page, frame)
    pt = frame.evaluate(
        "([lat, lon]) => { const p = window.drawnItems._map.latLngToContainerPoint([lat, lon]);"
        " return [p.x, p.y]; }",
        [lat, lon],
    )
    return box["x"] + pt[0], box["y"] + pt[1]


def click_tool(frame, selector: str) -> bool:
    el = frame.query_selector(selector)
    if el is None:
        return False
    el.click()
    return True


def draw_rectangle(page, frame, south, west, north, east) -> None:
    """Activate the rectangle tool and drag it out with real mouse events."""
    click_tool(frame, ".leaflet-draw-draw-rectangle")
    page.wait_for_timeout(200)
    x0, y0 = point_of(page, frame, south, west)
    x1, y1 = point_of(page, frame, north, east)
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=14)
    page.mouse.up()
    page.wait_for_timeout(500)


def wait_for_text(page, text: str, timeout: float = 45.0) -> bool:
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout * 1000)
        return True
    except Exception:
        return False


def wait_for_text_gone(page, text: str, timeout: float = 20.0) -> bool:
    try:
        page.wait_for_selector(f"text={text}", state="detached", timeout=timeout * 1000)
        return True
    except Exception:
        return False


def raster_bounds(frame, timeout: float = 60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        bounds = frame.evaluate(JS_OVERLAY_BOUNDS)
        if bounds:
            return bounds
        page_wait = 0.5
        time.sleep(page_wait)
    raise RuntimeError("the raster overlay never appeared on the map")


# --------------------------------------------------------------------------- #
def scenario_inside(page, frame, b) -> None:
    print("\n--- scenario 1: draw a rectangle inside the raster ---", flush=True)
    s, w, n, e = b
    h, wd = n - s, e - w
    draw_rectangle(page, frame, s + 0.35 * h, w + 0.35 * wd, s + 0.65 * h, w + 0.65 * wd)
    page.screenshot(path=str(ARTIFACTS / "phase5_browser_inside.png"))
    check(wait_for_text(page, "Selection detected"),
          "the app reports 'Selection detected' after a real mouse drag")
    check(wait_for_text(page, "100.0% of the drawn shape is inside the raster", timeout=10),
          "the selection is fully inside the raster")
    check(wait_for_text(page, "geometry stored in EPSG:32636", timeout=10),
          "the panel names the raster CRS the geometry was stored in")


def metric_values(page):
    return page.eval_on_selector_all(
        "[data-testid='stMetricValue']", "els => els.map(e => e.innerText)"
    )


def scenario_replace(page, frame, b) -> None:
    print("\n--- scenario 2: draw again -- the new shape replaces the old one ---", flush=True)
    before_text = page.inner_text("body")
    before_metrics = metric_values(page)
    s, w, n, e = b
    h, wd = n - s, e - w
    draw_rectangle(page, frame, s + 0.15 * h, w + 0.15 * wd, s + 0.30 * h, w + 0.30 * wd)
    page.screenshot(path=str(ARTIFACTS / "phase5_browser_replace.png"))
    # NOTE: waiting for "Selection detected" would pass instantly -- that text is
    # already on the page from scenario 1. Wait for something that can only
    # appear once Python has seen the SECOND shape.
    got_second = wait_for_text(page, "shapes are drawn")
    check(got_second, "the app received the second drawing (it now reports 2 shapes)")
    check(page.inner_text("body") != before_text, "the panel text changed")
    check(metric_values(page) != before_metrics,
          "the reported area changed -- the newest shape replaced the older one")
    check(frame.evaluate(JS_LAYER_COUNT) == 2, "the map holds both drawn layers")


def scenario_delete(page, frame) -> None:
    print("\n--- scenario 3: delete every shape -- no stale ROI ---", flush=True)
    check(click_tool(frame, ".leaflet-draw-edit-remove"), "the delete tool is present")
    deleted = 0
    # NOTE: leaflet-draw only fires map-level draw:deleted on "Save"; our bridge
    # (ui/map.py) makes each delete propagate immediately, so no Save click is
    # needed. This scenario deliberately does NOT click Save.
    for _ in range(6):
        if frame.evaluate(JS_LAYER_COUNT) <= 0:
            break
        click_tool(frame, ".leaflet-draw-edit-remove")     # (re)arm remove mode
        page.wait_for_timeout(250)
        centre = frame.evaluate(JS_LAST_SHAPE_CENTRE)
        if centre is None:
            break
        box = map_box(page, frame)
        page.mouse.click(box["x"] + centre[0], box["y"] + centre[1])
        page.wait_for_timeout(1200)
        deleted += 1
    print(f"   (clicked delete on {deleted} shape(s))")
    check(frame.evaluate(JS_LAYER_COUNT) == 0, "Leaflet reports no drawn layers left")
    ok = wait_for_text(page, "Draw a rectangle or polygon")
    page.screenshot(path=str(ARTIFACTS / "phase5_browser_deleted.png"))
    check(ok, "the panel returns to the empty state")
    check(not wait_for_text(page, "Selection detected", timeout=6),
          "the deleted selection is NOT still shown (no stale ROI)")


def scenario_partial(page, frame, b) -> None:
    print("\n--- scenario 4: a selection hanging off the raster edge ---", flush=True)
    s, w, n, e = b
    h, wd = n - s, e - w
    draw_rectangle(page, frame, s + 0.40 * h, e - 0.30 * wd, s + 0.60 * h, e + 0.30 * wd)
    page.screenshot(path=str(ARTIFACTS / "phase5_browser_partial.png"))
    check(wait_for_text(page, "Selection detected"), "a straddling selection is accepted")
    check(wait_for_text(page, "Only the portion inside the available raster", timeout=10),
          "the UI says only the inside portion will be used")


def scenario_outside(page, frame, b) -> None:
    print("\n--- scenario 5: a selection completely outside the raster ---", flush=True)
    s, w, n, e = b
    h, wd = n - s, e - w
    draw_rectangle(page, frame, s + 0.10 * h, w - 0.75 * wd, s + 0.25 * h, w - 0.55 * wd)
    page.screenshot(path=str(ARTIFACTS / "phase5_browser_outside.png"))
    check(wait_for_text(page, "Analysis data is not available for this selected area"),
          "an outside selection produces the explicit 'does not overlap' message")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    args = ap.parse_args()

    print("=" * 78)
    print(f"SatQuery AI -- Phase 5 BROWSER test against {args.url}")
    print("=" * 78)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("   Playwright is not installed -- no browser verification was possible.")
        return 2

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
            frame = find_map_frame(page)
            check(True, "the map iframe rendered a Leaflet map")
            check(frame.query_selector(".leaflet-draw-draw-rectangle") is not None,
                  "the rectangle tool is present")
            check(frame.query_selector(".leaflet-draw-draw-polygon") is not None,
                  "the polygon tool is present")
            check(frame.query_selector(".leaflet-draw-draw-marker") is None,
                  "the marker tool is absent")
            check(frame.query_selector(".leaflet-draw-draw-polyline") is None,
                  "the polyline tool is absent")
            check(frame.query_selector(".leaflet-draw-draw-circle") is None,
                  "the circle tool is absent")

            b = raster_bounds(frame)
            print(f"   raster overlay bounds read from Leaflet: "
                  f"lat {b[0]:.5f}..{b[2]:.5f}, lon {b[1]:.5f}..{b[3]:.5f}")

            scenario_inside(page, frame, b)
            scenario_replace(page, frame, b)
            scenario_delete(page, frame)
            scenario_partial(page, frame, b)
            scenario_outside(page, frame, b)
        finally:
            page.screenshot(path=str(ARTIFACTS / "phase5_browser_final.png"))
            browser.close()

    passed = sum(1 for ok, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 78)
    print(f"BROWSER RESULT: {passed} passed, {failed} failed")
    print("=" * 78)
    for ok, msg in _results:
        if not ok:
            print(f"  FAILED: {msg}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
