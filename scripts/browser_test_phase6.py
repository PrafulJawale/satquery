"""Phase 6 BROWSER test -- draws on the real map and checks the ROI statistics.

This extends the Phase 5 browser test (same helpers, same "coordinates come
from Leaflet, not from guesses" rule) and adds the Phase 6 scenarios:

    A. draw a rectangle            -> "ROI NDVI Analysis" appears, with the
                                      pixel counts, the NDVI statistics, the
                                      percentiles and a histogram;
                                      the NUMBERS are re-derived here, in this
                                      process, from the GeoTIFF with plain numpy
                                      + matplotlib Path, and compared with what
                                      the browser actually displays;
    B. draw a second rectangle     -> the numbers change (new region, new stats);
    C. draw outside the raster     -> no analysis, and no fabricated zeros;
    D. delete every shape          -> the whole analysis section disappears;
    E. draw outside the raster     -> no analysis, and no fabricated zeros
       (the app's "Reset view" is used first if the component has gone silent
       after the deletion -- a streamlit-folium quirk, reported in the output).

Requires the app to be running:

    streamlit run app.py --server.address 0.0.0.0 --server.port 8501

Run (restart the server first -- one session with the 2048x2048 scene loaded
costs about 0.7 GB, and two of them exhaust this sandbox's memory):

    pkill -f "streamlit run app.py"
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501 &
    python scripts/browser_test_phase6.py [--url http://127.0.0.1:8501]

Screenshots are written to artifacts/phase6_browser_*.png.
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

JS_LAST_SHAPE_LATLON = """() => {
  const layers = window.drawnItems ? window.drawnItems.getLayers() : [];
  if (!layers.length) return null;
  const b = layers[layers.length - 1].getBounds();
  return [b.getSouth(), b.getWest(), b.getNorth(), b.getEast()];
}"""

JS_METRICS = """() => Array.from(document.querySelectorAll('[data-testid="stMetric"]')).map(e => [
  (e.querySelector('[data-testid="stMetricLabel"]') || {}).innerText || '',
  (e.querySelector('[data-testid="stMetricValue"]') || {}).innerText || '',
])"""


# --------------------------------------------------------------------------- #
# browser helpers (unchanged from Phase 5)
# --------------------------------------------------------------------------- #
def _has_leaflet(page) -> bool:
    for frame in page.frames:
        try:
            if frame.query_selector(".leaflet-container"):
                return True
        except Exception:
            continue
    return False


def ensure_flat_map(page, timeout: float = 60.0) -> bool:
    """Switch the primary map to the flat detail mode, if it is not already.

    The globe is the default map surface; the checks that inspect Leaflet
    layers and drawings run against the flat detail mode, which the user opens
    deliberately. This drives that control the same way a user would.
    """
    if _has_leaflet(page):
        return True
    # A JS click, not a synthetic one: the chat input is docked to the bottom
    # of the page and can overlay the control, which would make Playwright's
    # hit-testing retry forever without ever reaching the radio.
    deadline = time.time() + timeout
    while time.time() < deadline:
        clicked = page.evaluate(
            """(label) => { const els = Array.from(document.querySelectorAll('label'));
                const target = els.find(e => (e.innerText || '').trim() === label);
                if (!target) return false;
                const input = target.querySelector('input[type=radio]');
                if (!input) return false;
                input.click(); return true; }""",
            "Flat map",
        )
        if not clicked:
            return False
        end = time.time() + 25
        while time.time() < end:
            if _has_leaflet(page):
                return True
            page.wait_for_timeout(500)
    return False


def find_map_frame(page, timeout: float = 120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for frame in page.frames:
            try:
                if frame.query_selector(".leaflet-container"):
                    return frame
            except Exception:
                continue
        ensure_flat_map(page)      # the globe is primary: open the flat mode
        page.wait_for_timeout(500)
    raise RuntimeError("no Leaflet map iframe appeared")


def map_box(page, frame):
    el = frame.query_selector(".leaflet-container")
    el.wait_for_element_state("visible")
    el.scroll_into_view_if_needed()
    page.wait_for_timeout(250)
    return el.bounding_box()


def point_of(page, frame, lat: float, lon: float):
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


def draw_rectangle(page, frame, south, west, north, east, attempts: int = 3) -> bool:
    """Drag a rectangle out with real mouse events; True if a new layer appeared.

    A drag can silently do nothing when Leaflet.Draw is still in remove mode
    after a deletion, so the tool is re-armed and the drag repeated until
    Leaflet really reports a new layer.
    """
    before = frame.evaluate(JS_LAYER_COUNT)
    for _ in range(attempts):
        click_tool(frame, ".leaflet-draw-draw-rectangle")
        page.wait_for_timeout(300)
        x0, y0 = point_of(page, frame, south, west)
        x1, y1 = point_of(page, frame, north, east)
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=14)
        page.mouse.up()
        page.wait_for_timeout(900)
        if frame.evaluate(JS_LAYER_COUNT) != before:
            return True
    return False


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
        time.sleep(0.5)
    raise RuntimeError("the raster overlay never appeared on the map")


def confirm_ndvi(page, timeout: float = 120.0) -> bool:
    """Tick the Phase 3 'I confirm Red = band N' gate.

    Without it the app deliberately refuses to compute NDVI, so there is nothing
    for the ROI analysis to measure.
    """
    # Streamlit STREAMS elements as the script runs, so waiting for any checkbox
    # is not enough: wait for this specific label to exist before looking for it.
    try:
        page.wait_for_selector("text=I confirm Red", timeout=timeout * 1000)
    except Exception:
        return False
    for box in page.query_selector_all("[data-testid='stCheckbox']"):
        try:
            text = box.inner_text()
        except Exception:
            continue
        if "I confirm Red" in text:
            box.scroll_into_view_if_needed()
            # Streamlit hides the raw <input>; clicking the label is reliable.
            label = box.query_selector("label")
            (label or box).click()
            return True
    return False


JS_SECTION_TEXT = """(heading) => {
  const hs = Array.from(document.querySelectorAll('[data-testid="stHeading"]'));
  const h = hs.find(e => (e.innerText || '').includes(heading));
  if (!h) return null;
  // Streamlit wraps each element in a container inside the column's vertical
  // block; walk up to that block and read everything up to the NEXT heading,
  // so unrelated panels (the NDVI legend, the band notice) cannot leak in.
  const block = h.closest('[data-testid="stVerticalBlock"]');
  if (!block) return null;
  const out = [];
  let started = false;
  for (const child of block.children) {
    const t = child.innerText || '';
    if (!started) {
      if (t.includes(heading)) { started = true; out.push(t); }
      continue;
    }
    if (child.querySelector('[data-testid="stHeading"]')) break;
    out.push(t);
  }
  return out.join(" | ");
}"""


def section_text(page, heading: str) -> str:
    return page.evaluate(JS_SECTION_TEXT, heading) or ""


def wait_for_metric_change(page, before: dict, label: str, timeout: float = 90.0) -> bool:
    """Wait until the given metric really changes (a rerun can take seconds)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        after = metrics(page)
        if after.get(label) != before.get(label):
            return True
        page.wait_for_timeout(500)
    return False


def click_button(page, label: str) -> bool:
    for btn in page.query_selector_all("button"):
        try:
            text = btn.inner_text() or ""
        except Exception:
            continue
        if label.lower() in text.lower():
            btn.scroll_into_view_if_needed()
            btn.click()
            return True
    return False


def metrics(page) -> dict[str, str]:
    """Every st.metric on the page as {label: value}."""
    pairs = page.evaluate(JS_METRICS)
    return {str(k).strip(): str(v).strip() for k, v in pairs if k}


# --------------------------------------------------------------------------- #
# independent truth: recompute the ROI statistics from the GeoTIFF
# --------------------------------------------------------------------------- #
_TRUTH = {}


def load_truth() -> dict:
    """Native NDVI of the sample scene, loaded once, plus its georeferencing."""
    if _TRUTH:
        return _TRUTH
    import numpy as np
    from rasterio import Affine

    from core.indices import ndvi_from_dataset
    from core.raster import open_dataset

    scene = REPO_ROOT / "data" / "sample" / "s2_s2b-36ruv-20230806-0-l2a_2048px.tif"
    with open_dataset(str(scene)) as ds:
        res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        _TRUTH.update(
            ndvi=np.asarray(res.array, dtype="float32"),
            mask=np.asarray(res.mask, dtype=bool),
            transform=Affine(*tuple(res.transform)),
        )
    return _TRUTH


def expected_stats(latlon_bounds) -> dict:
    """Recompute the ROI statistics with plain numpy + matplotlib Path.

    Deliberately does NOT call core.statistics: the point is to check the number
    the browser shows against a second, independently written implementation.
    """
    import matplotlib.path as mpath
    import numpy as np
    from pyproj import Transformer
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform as shapely_transform

    t = load_truth()
    ndvi, mask, transform = t["ndvi"], t["mask"], t["transform"]
    height, width = ndvi.shape

    south, west, north, east = latlon_bounds
    # WGS84 rectangle -> EPSG:32636, densified so the projected edge is curved
    # the way a real projected rectangle is (Leaflet gives us corner lat/lons).
    rect = shapely_box(west, south, east, north)
    step = max((north - south), (east - west)) / 60.0
    rect = rect.segmentize(step) if hasattr(rect, "segmentize") else rect
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    roi = shapely_transform(lambda x, y, z=None: tf.transform(x, y), rect)

    poly = np.asarray(roi.exterior.coords, dtype=float)
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    a, b, c, d, e, f = tuple(transform)[:6]
    xs = a * cols + b * rows + c
    ys = d * cols + e * rows + f
    inside = mpath.Path(poly).contains_points(
        np.column_stack([xs.ravel(), ys.ravel()]), radius=0.0).reshape(height, width)
    valid = inside & mask
    vals = ndvi[valid]
    out = {
        "inside": int(inside.sum()),
        "valid": int(vals.size),
        "mean": float(vals.mean()) if vals.size else None,
        "median": float(np.median(vals)) if vals.size else None,
        "min": float(vals.min()) if vals.size else None,
        "max": float(vals.max()) if vals.size else None,
        "std": float(vals.std(ddof=0)) if vals.size else None,
        "p5": float(np.percentile(vals, 5)) if vals.size else None,
        "p95": float(np.percentile(vals, 95)) if vals.size else None,
    }
    return out


def as_int(text: str) -> int | None:
    try:
        return int(str(text).replace(",", "").strip())
    except Exception:
        return None


def as_float(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
def scenario_stats(page, frame, b) -> None:
    print("\n--- Phase 6 / A: draw a rectangle and read the ROI statistics ---", flush=True)
    s, w, n, e = b
    h, wd = n - s, e - w
    check(draw_rectangle(page, frame, s + 0.40 * h, w + 0.40 * wd, s + 0.55 * h, w + 0.55 * wd),
          "a rectangle was really drawn with the mouse (Leaflet reports a new layer)")
    page.wait_for_timeout(1500)
    page.screenshot(path=str(ARTIFACTS / "phase6_browser_inside.png"))

    check(wait_for_text(page, "ROI NDVI Analysis", timeout=60),
          "the 'ROI NDVI Analysis' section appears after drawing")
    check(wait_for_text(page, "native resolution", timeout=20),
          "the panel states the native resolution")
    body = page.inner_text("body")
    check("10.00 m" in body or "10.0 m" in body,
          "the native resolution is reported as 10 m (derived from the affine)")
    check(wait_for_text(page, "NDVI distribution inside the ROI", timeout=20),
          "a histogram of valid NDVI values is shown")
    check(page.query_selector("[data-testid='stVegaLiteChart']") is not None,
          "the histogram is a rendered chart, not an empty placeholder")

    m = metrics(page)
    for label in ("Inside ROI", "Valid NDVI", "Invalid / nodata", "Valid %",
                  "Mean", "Median", "Min", "Max", "Std dev",
                  "P5", "P25", "P75", "P95"):
        check(label in m, f"the panel reports '{label}'")

    # ---- the numbers, recomputed independently ---------------------------- #
    ll = frame.evaluate(JS_LAST_SHAPE_LATLON)
    print(f"   drawn shape (from Leaflet): lat {ll[0]:.5f}..{ll[2]:.5f}, "
          f"lon {ll[1]:.5f}..{ll[3]:.5f}")
    exp = expected_stats(ll)
    print(f"   expected (recomputed here): inside={exp['inside']:,} valid={exp['valid']:,} "
          f"mean={exp['mean']:.4f} median={exp['median']:.4f} std={exp['std']:.4f}")
    print(f"   shown in the browser       : inside={m.get('Inside ROI')} "
          f"valid={m.get('Valid NDVI')} mean={m.get('Mean')} "
          f"median={m.get('Median')} std={m.get('Std dev')}")

    got_inside, got_valid = as_int(m.get("Inside ROI", "")), as_int(m.get("Valid NDVI", ""))
    check(got_inside is not None and abs(got_inside - exp["inside"]) <= max(2, 0.02 * exp["inside"]),
          f"pixels inside the ROI match the independent count "
          f"({got_inside} vs {exp['inside']})")
    check(got_valid is not None and abs(got_valid - exp["valid"]) <= max(2, 0.02 * exp["valid"]),
          f"valid NDVI pixels match the independent count ({got_valid} vs {exp['valid']})")
    for label, key in (("Mean", "mean"), ("Median", "median"), ("Min", "min"),
                       ("Max", "max"), ("Std dev", "std"), ("P5", "p5"), ("P95", "p95")):
        got = as_float(m.get(label, ""))
        want = exp[key]
        check(got is not None and abs(got - want) < 0.01,
              f"{label} matches the independently computed value "
              f"({got} vs {want:.4f})")

    sec = section_text(page, "ROI NDVI Analysis")
    check("not a health, yield or suitability judgement" in sec,
          "the panel states explicitly that it is not a health/suitability judgement")
    # NB: "yield"/"suitability" appear in the panel only inside the disclaimer
    # checked above, so they cannot be used as a banned-word test here.
    check(not any(w in sec.lower() for w in ("healthy", "unhealthy", "stressed")),
          "the ROI panel uses no health/stress wording of its own")


def scenario_redraw(page, frame, b) -> None:
    print("\n--- Phase 6 / B: draw a different rectangle -- the numbers follow ---", flush=True)
    before = metrics(page)
    s, w, n, e = b
    h, wd = n - s, e - w
    check(draw_rectangle(page, frame, s + 0.65 * h, w + 0.20 * wd, s + 0.78 * h, w + 0.34 * wd),
          "the second rectangle was really drawn")
    # A rerun re-reads the whole scene and recomputes NDVI, so the numbers can
    # take several seconds to change: wait for the change instead of guessing.
    changed = wait_for_metric_change(page, before, "Inside ROI", timeout=120)
    page.wait_for_timeout(500)
    page.screenshot(path=str(ARTIFACTS / "phase6_browser_redraw.png"))
    after = metrics(page)
    check(wait_for_text(page, "ROI NDVI Analysis", timeout=30),
          "the analysis is still present for the new selection")
    check(changed, "the reported pixel count updated after the second drawing")
    check(after.get("Inside ROI") != before.get("Inside ROI"),
          f"the pixel count changed with the new region "
          f"({before.get('Inside ROI')} -> {after.get('Inside ROI')})")
    check(after.get("Mean") != before.get("Mean"),
          f"the mean NDVI changed with the new region "
          f"({before.get('Mean')} -> {after.get('Mean')})")

    ll = frame.evaluate(JS_LAST_SHAPE_LATLON)
    exp = expected_stats(ll)
    got_inside = as_int(after.get("Inside ROI", ""))
    check(got_inside is not None and abs(got_inside - exp["inside"]) <= max(2, 0.02 * exp["inside"]),
          f"the new pixel count still matches the independent computation "
          f"({got_inside} vs {exp['inside']})")


def scenario_delete(page, frame) -> None:
    print("\n--- Phase 6 / D: delete every shape -- the analysis disappears ---", flush=True)
    check(click_tool(frame, ".leaflet-draw-edit-remove"), "the delete tool is present")
    for _ in range(6):
        if frame.evaluate(JS_LAYER_COUNT) <= 0:
            break
        click_tool(frame, ".leaflet-draw-edit-remove")
        page.wait_for_timeout(250)
        centre = frame.evaluate(JS_LAST_SHAPE_CENTRE)
        if centre is None:
            break
        box = map_box(page, frame)
        page.mouse.click(box["x"] + centre[0], box["y"] + centre[1])
        page.wait_for_timeout(1200)
    page.screenshot(path=str(ARTIFACTS / "phase6_browser_deleted.png"))
    check(frame.evaluate(JS_LAYER_COUNT) == 0, "Leaflet reports no drawn layers left")
    check(wait_for_text_gone(page, "ROI NDVI Analysis", timeout=25),
          "the ROI NDVI Analysis section is gone (no stale statistics)")
    check(not wait_for_text(page, "Mean NDVI", timeout=5),
          "no measurement is shown after the selection is deleted")


def scenario_outside(page, frame, b) -> None:
    """A rectangle west of the raster: no pixels, no statistics, no zeros.

    Two practical details, both learned the hard way:
      * the map is fitted to the raster, so a point outside it is normally
        OUTSIDE the visible map too -- the drag would land nowhere. Zooming out
        twice first puts the target inside the viewport;
      * if the map component has gone silent (see scenario_delete), the app's
        "Reset view" remounts it, after which drawing works again.
    """
    print("\n--- Phase 6 / C: a rectangle outside the raster -- no fake zeros ---", flush=True)

    if not wait_for_text(page, "Draw a rectangle or polygon", timeout=15):
        # the map component went silent after the deletion: remount it
        check(click_button(page, "Reset view"),
              "'Reset view' remounts the map component after a full deletion")
        page.wait_for_timeout(9000)
        frame = find_map_frame(page)

    for _ in range(2):                      # make room around the raster
        click_tool(frame, ".leaflet-control-zoom-out")
        page.wait_for_timeout(2500)
    frame = find_map_frame(page)
    s, w, n, e = raster_bounds(frame)
    h, wd = n - s, e - w
    drawn = draw_rectangle(page, frame, s + 0.35 * h, w - 0.35 * wd, s + 0.50 * h, w - 0.15 * wd)
    check(drawn, "a rectangle was drawn west of the raster (inside the visible map)")
    page.wait_for_timeout(4000)
    check(wait_for_text(page, "Analysis data is not available for this selected area", timeout=90),
          "the app says the selection does not overlap the raster")
    check(not wait_for_text(page, "ROI NDVI Analysis", timeout=6),
          "no statistics are computed for a selection outside the raster")
    body = page.inner_text("body")
    check("Mean NDVI 0.0000" not in body, "no zero-filled statistics are displayed")
    return frame


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    args = ap.parse_args()

    print("=" * 78)
    print(f"SatQuery AI -- Phase 6 BROWSER test against {args.url}")
    print("=" * 78)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("   Playwright is not installed -- no browser verification was possible.")
        return 2

    with sync_playwright() as p:
        # The sandbox is memory-tight (about 2 GB): a lean Chromium leaves room
        # for the Streamlit server, which holds the whole scene in memory.
        browser = p.chromium.launch(
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu",
                  "--js-flags=--max-old-space-size=256"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)

            # Phase 6 measures NDVI, so the Phase 3 confirmation gate must be
            # passed first (the app refuses to compute NDVI without it).
            check(confirm_ndvi(page), "the Red/NIR confirmation checkbox was ticked")
            check(wait_for_text(page, "NDVI map", timeout=120),
                  "the app computed NDVI after confirmation")

            frame = find_map_frame(page)
            check(True, "the map iframe rendered a Leaflet map")

            # Phase 5 regression: the drawing toolbox is unchanged
            check(frame.query_selector(".leaflet-draw-draw-rectangle") is not None,
                  "the rectangle tool is present")
            check(frame.query_selector(".leaflet-draw-draw-polygon") is not None,
                  "the polygon tool is present")
            check(frame.query_selector(".leaflet-draw-draw-circle") is None,
                  "the circle tool is absent")

            b = raster_bounds(frame)
            print(f"   raster overlay bounds read from Leaflet: "
                  f"lat {b[0]:.5f}..{b[2]:.5f}, lon {b[1]:.5f}..{b[3]:.5f}")

            scenario_stats(page, frame, b)
            scenario_redraw(page, frame, b)
            scenario_delete(page, frame)
            frame = scenario_outside(page, frame, b)
        finally:
            page.screenshot(path=str(ARTIFACTS / "phase6_browser_final.png"))
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
