"""Phase 7 BROWSER test -- asks the app a question in a real browser.

It reuses the Phase 6 browser helpers (same "coordinates come from Leaflet, not
from guesses" rule) and drives the new "Ask SatQuery" chat input:

    1. load the app and confirm the Red/NIR mapping
    2. draw an ROI with the mouse
    3. ask "What is the NDVI of this area?"
       -> assert the routed INTENT, the ANSWER and the STATISTICS all appear
    4. ask "Can I grow cotton here?"
       -> assert the app says it is not available and shows no numbers
    5. delete every shape and ask the NDVI question again
       -> assert the user is told to select an area first

Requires the app to be running (restart it first -- one session with the 2048x2048
scene costs about 0.7 GB and this sandbox has about 2 GB):

    streamlit run app.py --server.address 0.0.0.0 --server.port 8501

Run:
    python scripts/browser_test_phase7.py [--url http://127.0.0.1:8501]

Screenshots: artifacts/phase7_browser_*.png
Exits 0 only if every scenario passes; 2 means Playwright is unavailable.
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

# Reuse the Phase 6 helpers instead of copy-pasting them.
_spec = importlib.util.spec_from_file_location(
    "bt6", REPO_ROOT / "scripts" / "browser_test_phase6.py")
bt6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt6)

_results: list[tuple[bool, str]] = []


def check(ok: bool, message: str) -> None:
    _results.append((bool(ok), message))
    print(f"   [{'PASS' if ok else 'FAIL'}] {message}", flush=True)


def ask(page, text: str, wait: float = 12.0) -> bool:
    """Type into the SatQuery chat input and submit with Enter."""
    box = page.query_selector("textarea[data-testid='stChatInputTextArea']")
    if box is None:
        box = page.query_selector("div[data-testid='stChatInput'] textarea")
    if box is None:
        box = page.query_selector("textarea")
    if box is None:
        return False
    box.scroll_into_view_if_needed()
    box.click()
    box.fill(text)
    page.keyboard.press("Enter")
    page.wait_for_timeout(int(wait * 1000))
    return True


def delete_all_shapes(page, frame) -> None:
    """Delete every drawn shape (Phase 6's bridge makes each delete propagate)."""
    bt6.click_tool(frame, ".leaflet-draw-edit-remove")
    for _ in range(6):
        if frame.evaluate(bt6.JS_LAYER_COUNT) <= 0:
            break
        bt6.click_tool(frame, ".leaflet-draw-edit-remove")
        page.wait_for_timeout(300)
        centre = frame.evaluate(bt6.JS_LAST_SHAPE_CENTRE)
        if centre is None:
            break
        box = bt6.map_box(page, frame)
        page.mouse.click(box["x"] + centre[0], box["y"] + centre[1])
        page.wait_for_timeout(1200)


def body(page) -> str:
    return page.inner_text("body")


# --------------------------------------------------------------------------- #
def scenario_supported_query(page, frame, bounds) -> None:
    print("\n--- Phase 7 / A: a supported NDVI question ---", flush=True)
    s, w, n, e = bounds
    h, wd = n - s, e - w
    check(bt6.draw_rectangle(page, frame, s + 0.40 * h, w + 0.40 * wd, s + 0.55 * h, w + 0.55 * wd),
          "an ROI was drawn with the mouse")
    check(bt6.wait_for_text(page, "ROI NDVI Analysis", timeout=90),
          "the Phase 6 ROI statistics are present (context for the question)")

    check(ask(page, "What is the NDVI of this area?"),
          "the SatQuery chat input accepted the question")
    check(bt6.wait_for_text(page, "NDVI_ROI_STATS", timeout=60),
          "the routed intent NDVI_ROI_STATS is shown")
    check(bt6.wait_for_text(page, "Intent:", timeout=20),
          "the answer is labelled with its intent")
    check(bt6.wait_for_text(page, "mean NDVI", timeout=30),
          "the answer states the observed mean NDVI")
    check(bt6.wait_for_text(page, "valid pixels", timeout=20),
          "the answer states the valid-pixel count")
    text = body(page)
    check("not a crop-health" in text,
          "the answer says it is not a crop-health diagnosis")
    # the detailed statistics are rendered by the SAME panel as section 4
    check(text.count("ROI NDVI Analysis") >= 2,
          "the detailed ROI statistics are rendered inside the answer")
    check(bt6.wait_for_text(page, "NDVI distribution inside the ROI", timeout=20),
          "the histogram is part of the routed answer")
    page.screenshot(path=str(ARTIFACTS / "phase7_browser_supported.png"))

    metrics = bt6.metrics(page)
    check("Mean" in metrics and "Valid NDVI" in metrics,
          f"the routed answer shows statistics (mean={metrics.get('Mean')}, "
          f"valid={metrics.get('Valid NDVI')})")


def scenario_unsupported_query(page) -> None:
    print("\n--- Phase 7 / B: an unsupported question fabricates nothing ---", flush=True)
    # Phase 8 note: cotton is now supported, so the "no analysis" path is
    # exercised with a crop the engine does not implement (rice). Cotton itself
    # is covered by scripts/browser_test_phase8.py.
    check(ask(page, "Can I grow rice here?"), "the crop question was submitted")
    check(bt6.wait_for_text(page, "CROP_SUITABILITY", timeout=60),
          "the intent CROP_SUITABILITY is recognised and shown")
    check(bt6.wait_for_text(page, "Only cotton suitability is currently supported.",
                            timeout=30),
          "the app answers that only cotton is supported")
    text = body(page)
    after = text.split("Only cotton suitability is currently supported.")[-1].lower()
    check("growing-season precipitation" not in after and "screening class" not in after,
          "no suitability judgement was produced for rice")
    check(ask(page, "Show flood areas."), "the flood question was submitted")
    check(bt6.wait_for_text(page, "FLOOD_CHANGE", timeout=60),
          "the intent FLOOD_CHANGE is recognised and shown")
    check(bt6.wait_for_text(page, "Flood analysis is not available yet.", timeout=30),
          "the app says flood analysis is not available yet")
    check(ask(page, "Tell me about this area."), "an ambiguous question was submitted")
    check(bt6.wait_for_text(page, "I could not match that to an available analysis.", timeout=30),
          "an ambiguous question is not guessed as NDVI")
    page.screenshot(path=str(ARTIFACTS / "phase7_browser_unsupported.png"))


def scenario_no_roi(page, frame) -> None:
    print("\n--- Phase 7 / C: after deleting the ROI the router asks for one ---", flush=True)
    delete_all_shapes(page, frame)
    check(frame.evaluate(bt6.JS_LAYER_COUNT) == 0, "Leaflet reports no drawn layers")
    check(bt6.wait_for_text(page, "Draw a rectangle or polygon", timeout=60),
          "the app returned to the empty-selection state")
    check(ask(page, "What is the NDVI of this area?"), "the question was submitted again")
    check(bt6.wait_for_text(page, "Please select an area on the map first.", timeout=60),
          "the user is told to select an area first")
    text = body(page)
    check("mean NDVI" not in text.split("Please select an area on the map first.")[-1],
          "no statistics were produced without a selection")
    page.screenshot(path=str(ARTIFACTS / "phase7_browser_no_roi.png"))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    args = ap.parse_args()

    print("=" * 78)
    print(f"SatQuery AI -- Phase 7 BROWSER test against {args.url}")
    print("=" * 78)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("   Playwright is not installed -- no browser verification was possible.")
        return 2

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu",
                  "--js-flags=--max-old-space-size=256"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
            check(bt6.confirm_ndvi(page), "the Red/NIR confirmation checkbox was ticked")
            check(bt6.wait_for_text(page, "NDVI map", timeout=120),
                  "NDVI is available for the router to use")
            frame = bt6.find_map_frame(page)
            check(bt6.wait_for_text(page, "Ask SatQuery", timeout=60),
                  "the 'Ask SatQuery' section is present")
            bounds = bt6.raster_bounds(frame)

            scenario_supported_query(page, frame, bounds)
            scenario_unsupported_query(page)
            scenario_no_roi(page, frame)
        finally:
            page.screenshot(path=str(ARTIFACTS / "phase7_browser_final.png"))
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
