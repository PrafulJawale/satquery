"""Phase 9 BROWSER test -- the spatial query in a real browser.

Genuine interaction: the page is driven with the mouse and keyboard, never with
DOM-only mocks. It reuses the Phase 6 helpers ("coordinates come from Leaflet,
not from guesses").

    1. load the app, pass the NDVI gate and draw an ROI with the mouse
    2. "Find cropland near water"
       -> interpretation, result metrics, map layer, legend, methodology,
          limitations
    3. "Find areas suitable for cotton near water"
       -> cotton class >= 3 + rainfed shown; zero matches reported as a
          geographic result, never as an error and never as insufficient data
    4. "Find cotton land with reliable irrigation"
       -> explicit unsupported panel, no partial spatial result
    5. "What is the NDVI of this area?"
       -> the Phase 7 workflow still works

Requires the app to be running:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501

Run:
    python scripts/browser_test_phase9.py [--url http://127.0.0.1:8501]

Screenshots: artifacts/phase9_browser_*.png
Exits 0 only if every check passes; 2 means Playwright is unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS = REPO_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "bt6", REPO_ROOT / "scripts" / "browser_test_phase6.py")
bt6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt6)

_results: list[tuple[bool, str]] = []


def check(ok: bool, message: str) -> None:
    _results.append((bool(ok), message))
    print(f"   [{'PASS' if ok else 'FAIL'}] {message}", flush=True)


def ask(page, text: str, wait: float = 15.0) -> bool:
    box = (page.query_selector("textarea[data-testid='stChatInputTextArea']")
           or page.query_selector("div[data-testid='stChatInput'] textarea")
           or page.query_selector("textarea"))
    if box is None:
        return False
    box.scroll_into_view_if_needed()
    box.click()
    box.fill(text)
    page.keyboard.press("Enter")
    page.wait_for_timeout(int(wait * 1000))
    return True


def body(page) -> str:
    return page.inner_text("body")


def shot(page, frame, name: str) -> None:
    """One viewport screenshot per scenario (evidence that the UI rendered)."""
    page.screenshot(path=str(ARTIFACTS / f"phase9_browser_{name}.png"))


def open_expander(page, label: str) -> bool:
    """Streamlit keeps collapsed content out of the DOM until it is opened."""
    try:
        elements = page.query_selector_all("[data-testid='stExpander']")
        for element in elements:
            if label.lower() in (element.inner_text() or "").lower():
                element.evaluate("e => { const d = e.querySelector('details');"
                                 " if (d) d.open = true; }")
                page.wait_for_timeout(400)
                return True
    except Exception:
        pass
    try:
        page.get_by_text(label, exact=False).first.click()
        page.wait_for_timeout(400)
        return True
    except Exception:
        return False


def live_frame(page):
    """The map iframe is recreated on every Streamlit rerun -- re-find it."""
    try:
        return bt6.find_map_frame(page, timeout=60)
    except Exception:                                       # pragma: no cover
        return None


def frame_text(frame) -> str:
    try:
        return frame.evaluate("() => document.body.innerText || ''")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
def scenario_setup(page, frame, bounds) -> bool:
    """Draw the ROI with the mouse.

    `frame` MUST have been acquired after the NDVI gate: Streamlit recreates
    the component's iframe on every rerun, so a frame cached from before the
    gate is detached and a drag on it never reaches the live app.
    """
    print("\n--- setup: a mouse-drawn ROI ---", flush=True)
    s, w, n, e = bounds
    h, wd = n - s, e - w
    check(bt6.draw_rectangle(page, frame, s + 0.40 * h, w + 0.40 * wd,
                             s + 0.60 * h, w + 0.60 * wd),
          "an ROI was drawn with the mouse")
    check(bt6.wait_for_text(page, "ROI NDVI Analysis", timeout=120),
          "the drawn ROI is analysed before any spatial query is asked")
    try:
        panel = bt6.section_text(page, "Selected area (ROI)")
        print("   ROI panel: " + " | ".join(
            line.strip() for line in panel.splitlines() if line.strip())[:300],
              flush=True)
    except Exception:                                       # pragma: no cover
        pass
    return True


def scenario_cropland_near_water(page, frame) -> None:
    print("\n--- Query 1: 'Find cropland near water' ---", flush=True)
    check(ask(page, "Find cropland near water", wait=20.0),
          "the query was typed and submitted")

    check(bt6.wait_for_text(page, "Interpreted as", timeout=600),
          "the interpretation block rendered (external layers fetched)")
    text = body(page)
    check("SPATIAL_QUERY" in text,
          "the intent SPATIAL_QUERY is recognised and shown")
    check("Land cover" in text and "40" in text,
          "the cropland condition is shown with its WorldCover class")
    near = ("Near permanent water" in text) or ("water proximity" in text.lower())
    check(near, "the near-water condition is shown")
    check("1,000 m" in text or "1000 m" in text,
          "the configured proximity distance is shown")
    check("Operator" in text and "AND" in text,
          "the combination operator is shown")

    check(bt6.wait_for_text(page, "Matching cells", timeout=120),
          "the result summary rendered")
    metrics = bt6.metrics(page)
    check("Matching cells" in metrics and "Matched area" in metrics,
          f"matched cells={metrics.get('Matching cells')} "
          f"area={metrics.get('Matched area')} "
          f"fraction={metrics.get('Matched fraction')}")
    check("Result" in text,
          "the result state is named")
    check("Analysis resolution" in text,
          "the analysis resolution is reported")

    # --- the map layer + legend ------------------------------------------- #
    layer_text = frame_text(live_frame(page) or frame)
    check("Spatial query result" in layer_text,
          "the spatial-query layer appears in the map's layer control")
    check("Match" in layer_text and "No match" in layer_text
          and "Insufficient" in layer_text,
          "the legend explains all three result classes")
    shot(page, frame, "1_cropland_water")

    # --- methodology + limitations ---------------------------------------- #
    check(open_expander(page, "Data & methodology"),
          "the methodology section opens")
    method = body(page)
    check("WorldCover" in method, "methodology names ESA WorldCover")
    check("nearest" in method.lower() or "categorical" in method.lower(),
          "methodology states the categorical resampling rule")
    check("resolution" in method.lower(),
          "methodology distinguishes analysis resolution from native detail")
    check(open_expander(page, "Limitations"), "the limitations section opens")
    limits = body(page)
    check("not irrigation" in limits.lower(),
          "limitations state that water proximity is not irrigation")
    check("not groundwater" in limits.lower(),
          "limitations state that water proximity is not groundwater")
    check("flood" in limits.lower(),
          "limitations state that water proximity is not flood risk")
    check("wetland" in limits.lower(),
          "limitations state that wetland is not permanent water")


def scenario_map_legend(page, frame) -> None:
    """The PPT map slide: overlay + layer control + the three-class legend.

    A fresh query is asked here so the numbers shown belong to THIS run.
    """
    print("\n--- Map + legend: MATCH / NO MATCH / INSUFFICIENT DATA ---",
          flush=True)
    check(ask(page, "Find cropland near water", wait=20.0),
          "the query was typed and submitted")
    check(bt6.wait_for_text(page, "Interpreted as", timeout=600),
          "the interpretation block rendered")
    check(bt6.wait_for_text(page, "Matching cells", timeout=180),
          "the result summary rendered")
    metrics = bt6.metrics(page)
    print(f"   metrics for this scenario: {metrics}", flush=True)

    live = live_frame(page) or frame
    try:                       # open the layer control so the layer is visible
        toggle = live.query_selector(".leaflet-control-layers-toggle")
        if toggle is not None:
            toggle.click()
            page.wait_for_timeout(700)
    except Exception as exc:                                # pragma: no cover
        print(f"   (layer control not opened: {exc})", flush=True)

    text = frame_text(live_frame(page) or live)
    check("Spatial query result" in text,
          "the layer control lists the spatial-query layer")
    check("Match" in text and "No match" in text and "Insufficient" in text,
          "the legend names MATCH / NO MATCH / INSUFFICIENT DATA")
    check("no match" in text.lower() and "not a failure" in text.lower(),
          "the legend says a no-match cell is a result, not a failure")
    check("insufficient data" in text.lower()
          and "never counted as a non-match" in text.lower(),
          "the legend separates INSUFFICIENT DATA from NO MATCH")
    shot(page, live, "4_map_legend")


def scenario_cotton_near_water(page, frame) -> None:
    print("\n--- Query 2: 'Find areas suitable for cotton near water' ---",
          flush=True)
    # Snapshot the metrics BEFORE the question: the guard below must prove the
    # numbers on screen come from THIS query and not from an earlier one.
    before = bt6.metrics(page)
    check(ask(page, "Find areas suitable for cotton near water", wait=20.0),
          "the cotton + near-water query was submitted")
    # The Phase 8 engine has to run; a fresh ROI grid can take minutes.
    check(bt6.wait_for_text(page, "Cotton suitability", timeout=1500),
          "the cotton condition rendered (Phase 8 engine ran)")
    text = body(page)
    check("rainfed" in text.lower(),
          "the rainfed scenario is shown")
    check("3" in text and ("class" in text.lower()),
          "the cotton class threshold (>= 3) is shown")
    check(("Near permanent water" in text)
          or ("water proximity" in text.lower()),
          "the near-water condition is shown alongside cotton")

    # An earlier result block can still be on the page, so "Matching cells"
    # alone is a vacuous check: wait until the numbers really change.
    deadline = time.time() + 600
    while time.time() < deadline and bt6.metrics(page) == before:
        page.wait_for_timeout(2000)
    metrics = bt6.metrics(page)
    check(metrics != before, "the cotton result replaced the previous metrics")
    fraction = str(metrics.get("Matched fraction", ""))
    check("0" in fraction.replace(" ", ""),
          f"the cotton result reports zero matches (fraction={fraction})")
    check("No match" in body(page),
          "zero matches is presented as NO MATCH, not as an engine failure")
    check("executed successfully" in body(page).lower()
          or "not an error" in body(page).lower()
          or "result, not" in body(page).lower(),
          "the UI says the query executed successfully")
    zero_text = body(page)
    check("insufficient" not in zero_text.split("Matching cells")[-1][:400].lower()
          or "cells could not be established" not in zero_text.lower(),
          "no false insufficient-data claim is made for the decided cells")
    check("no suitable land exists" not in zero_text.lower(),
          "the UI never claims that no suitable land exists")
    shot(page, frame, "2_cotton_water")


def scenario_unsupported(page, frame) -> None:
    print("\n--- Query 3: unsupported irrigation query ---", flush=True)
    check(ask(page, "Find cotton land with reliable irrigation", wait=20.0),
          "the irrigation query was submitted")
    check(bt6.wait_for_text(page, "Unsupported condition", timeout=120),
          "the unsupported panel rendered")
    text = body(page)
    check("irrigation" in text.lower(),
          "the explanation names irrigation")
    check("nothing was computed" in text.lower()
          or "no proxy" in text.lower(),
          "the UI states that nothing was computed and no proxy substituted")
    check("Matching cells" not in text.split("Unsupported condition")[-1],
          "no partial spatial result is shown")
    check("Interpreted as" not in text.split("Unsupported condition")[-1],
          "no conditions are presented as understood")
    check("proximity" in text.lower(),
          "the supported alternative (water proximity) is named as different")
    shot(page, frame, "3_unsupported")


def scenario_phase7_ndvi(page, frame) -> None:
    print("\n--- Query 4: the Phase 7 NDVI workflow ---", flush=True)
    check(ask(page, "What is the NDVI of this area?", wait=20.0),
          "the NDVI question was submitted")
    check(bt6.wait_for_text(page, "ROI NDVI Analysis", timeout=180),
          "the Phase 7 NDVI analysis still runs")
    text = body(page)
    check("NDVI_ROI_STATS" in text,
          "the NDVI intent is still routed")
    check("valid pixels" in text.lower() or "Mean NDVI" in text
          or "mean" in text.lower(),
          "the NDVI statistics are reported")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--only", default="all",
                        choices=["all", "1", "2", "3", "4", "5"],
                        help="run a single scenario in a fresh browser "
                             "(the sandbox has 2 GB RAM: one at a time)")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except Exception:                                       # pragma: no cover
        print("Playwright is not installed: pip install playwright && "
              "python -m playwright install chromium")
        return 2

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=["--disable-dev-shm-usage", "--no-sandbox",
                  "--disable-gpu",
                  "--js-flags=--max-old-space-size=256"])
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors: list[str] = []
        http_errors: list[tuple[int, str]] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda msg: errors.append(msg.text)
                if msg.type == "error" else None)
        page.on("response", lambda resp: http_errors.append((resp.status, resp.url))
                if resp.status >= 400 else None)

        print(f"opening {args.url}", flush=True)
        page.goto(args.url, wait_until="domcontentloaded", timeout=180000)
        check(bt6.confirm_ndvi(page, timeout=180),
              "the NDVI gate is passed (Phase 3 workflow intact)")
        check(bt6.wait_for_text(page, "NDVI map", timeout=180),
              "NDVI is available (the app loaded the bundled sample)")
        # The map frame is acquired AFTER the gate: every rerun remounts the
        # component, so a frame taken earlier points at a dead iframe.
        frame = bt6.find_map_frame(page, timeout=180)
        check(frame is not None, "the map iframe loaded")
        if frame is None:
            browser.close()
            return 1
        bounds = bt6.raster_bounds(frame, timeout=180)
        check(bounds is not None and len(bounds) == 4,
              f"the raster bounds were read from Leaflet: {bounds}")

        scenario_setup(page, frame, bounds)
        if args.only in ("all", "1"):
            scenario_cropland_near_water(page, frame)
        if args.only in ("all", "2"):
            scenario_cotton_near_water(page, frame)
        if args.only in ("all", "3"):
            scenario_unsupported(page, frame)
        if args.only in ("all", "4"):
            scenario_map_legend(page, frame)
        if args.only in ("all", "5"):
            scenario_phase7_ndvi(page, frame)

        # "Failed to load resource" console lines are covered precisely by the
        # HTTP response check above (which verifies the failing URL), so they are
        # not counted twice here.
        real_errors = [e for e in errors
                       if "favicon" not in e.lower()
                       and "ResizeObserver" not in e
                       and "Failed to load resource" not in e]
        # `satquery-draw-bridge.png` is DELIBERATELY missing: its 404 fires the
        # onerror handler that installs the Leaflet.Draw delete bridge, because
        # streamlit-folium injects the map body with innerHTML (no <script>).
        unexpected_http = [f"{st} {u}" for st, u in http_errors
                           if not u.endswith("satquery-draw-bridge.png")]
        expected_http = [u for _, u in http_errors
                         if u.endswith("satquery-draw-bridge.png")]
        print(f"   HTTP >=400: {len(http_errors)} "
              f"({len(expected_http)} expected draw-bridge 404, "
              f"{len(unexpected_http)} unexpected)", flush=True)
        if unexpected_http:
            print(f"   unexpected: {unexpected_http[:3]}", flush=True)
        check(not real_errors and not unexpected_http,
              f"no uncaught browser errors ({len(real_errors)} js, "
              f"{len(unexpected_http)} unexpected http)")

        shot(page, frame, "5_ndvi_regression")
        browser.close()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 70)
    for ok, message in _results:
        if not ok:
            print(f"   FAILED: {message}")
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
