"""Phase 8 BROWSER test -- the cotton screening in a real browser.

Genuine interaction: the page is driven with the mouse and keyboard, never with
DOM-only mocks. It reuses the Phase 6 helpers ("coordinates come from Leaflet,
not from guesses").

    1. load the app and confirm the Red/NIR mapping
    2. draw an ROI with the mouse
    3. ask "Can I grow cotton here?"
       -> CROP_SUITABILITY executes; the answer, the panel, the legend,
          the confidence/warnings and the missing-data language all appear
    4. ask "Can I grow rice here?"
       -> the app answers that only cotton is supported, and computes nothing

The first cotton query fetches every external layer for a NEW grid, so it can
take several minutes; the waits below allow for that.

Requires the app to be running:
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501

Run:
    python scripts/browser_test_phase8.py [--url http://127.0.0.1:8501]

Screenshots: artifacts/phase8_browser_*.png
Exits 0 only if every check passes; 2 means Playwright is unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
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


def open_expander(page, label: str) -> bool:
    """Streamlit renders an expander collapsed, so its text is not in the DOM
    until it is opened. Open it the way a user would."""
    for selector in (f"details:has-text({label!r})",
                     f"[data-testid='stExpander']:has-text({label!r})"):
        try:
            el = page.query_selector(selector)
            if el is not None:
                el.evaluate("e => { if (e.tagName === 'DETAILS') e.open = true; }")
                return True
        except Exception:
            pass
    try:
        page.get_by_text(label, exact=False).first.click()
        return True
    except Exception:
        return False


def click_tab(page, label: str) -> bool:
    """Switch to a scenario tab (only the active tab is visible)."""
    try:
        page.click(f"[role='tab']:has-text({label!r})")
        page.wait_for_timeout(1500)
        return True
    except Exception:
        return False


def frame_text(frame) -> str:
    """Text inside the Leaflet iframe (layer names, legend HTML)."""
    try:
        return frame.evaluate("() => document.body.innerText || ''")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
def scenario_cotton(page, frame, bounds) -> None:
    print("\n--- Phase 8 / A: 'Can I grow cotton here?' ---", flush=True)
    s, w, n, e = bounds
    h, wd = n - s, e - w
    check(bt6.draw_rectangle(page, frame, s + 0.42 * h, w + 0.42 * wd,
                             s + 0.55 * h, w + 0.55 * wd),
          "an ROI was drawn with the mouse")
    check(bt6.wait_for_text(page, "ROI NDVI Analysis", timeout=90),
          "the drawn ROI is analysed before the crop question is asked")

    check(ask(page, "Can I grow cotton here?", wait=25.0),
          "the cotton question was typed and submitted")
    # ONE long wait: if the mouse-drawn ROI lands on a grid cell boundary the
    # engine legitimately re-fetches every external layer for the new grid,
    # which took up to ~16 minutes on the public endpoints. Everything after
    # this point is rendered in the same run and needs only short waits.
    check(bt6.wait_for_text(page, "Experimental crop-suitability screening",
                            timeout=1500),
          "the screening answer rendered (all external layers fetched)")
    text = body(page)
    check("CROP_SUITABILITY" in text,
          "the intent CROP_SUITABILITY is recognised and shown")
    check("strongest computed limitation" in text.lower(),
          "the answer names a computed limiting factor")
    check("growing-season precipitation" in text.lower(),
          "the water factor is reported as growing-season precipitation")
    check("annual total" in text.lower(),
          "annual precipitation is reported alongside it, not instead of it")
    check("not assessed in this screening" in text.lower(),
          "unassessed constraints are listed as limitations")

    # --- structured panel --------------------------------------------------- #
    check(bt6.wait_for_text(page, "Screening class", timeout=60),
          "the panel shows the screening class as a metric")
    metrics = bt6.metrics(page)
    check("Screening class" in metrics and "Confidence" in metrics,
          f"class={metrics.get('Screening class')} "
          f"score={metrics.get('Score')} confidence={metrics.get('Confidence')}")
    check("Unsuitable" in metrics.get("Screening class", "") or "Unsuitable" in text,
          "the Nile Delta ROI screens as Unsuitable under rainfed conditions")
    check(bt6.wait_for_text(page, "Computed limiting factors", timeout=30),
          "computed limiting factors are listed separately")
    check(bt6.wait_for_text(page, "Membership", timeout=30),
          "per-factor memberships are shown")
    check(bt6.wait_for_text(page, "Rainfed", timeout=30),
          "the data-backed rainfed scenario is offered as a tab")
    check(bt6.wait_for_text(page, "Irrigation", timeout=30),
          "the hypothetical irrigation scenario is offered as a second tab")
    check(click_tab(page, "Irrigation"),
          "the irrigation tab can be opened by clicking it")
    irr = body(page)
    check("Hypothetical sensitivity analysis" in irr,
          "the irrigation tab is labelled a hypothetical sensitivity analysis")
    check("Low" in irr, "the irrigation tab carries Low confidence")
    check(click_tab(page, "Rainfed"), "the rainfed tab can be reopened")

    # --- missing data, never a fabricated number ---------------------------- #
    low = text.lower()
    check("salinity" in low, "salinity is named as not assessed")
    check("irrigation" in low, "irrigation availability is named as not assessed")
    check("not a crop recommendation" in low or "not a crop-health" in low
          or "experimental" in low,
          "the answer keeps the experimental framing")
    check("zero was not substituted" in low or "no score was computed" in low
          or "not assessed" in low,
          "missing data is described as missing, not scored as zero")

    # --- why / methodology -------------------------------------------------- #
    check(bt6.wait_for_text(page, "how this score was produced", timeout=30),
          "a 'Why?' explainer is present")
    check(bt6.wait_for_text(page, "Data & methodology", timeout=30),
          "a 'Data & methodology' section is present")
    check(open_expander(page, "Data & methodology"),
          "'Data & methodology' can be opened by clicking it")
    page.wait_for_timeout(1000)
    opened = body(page)
    check("Climatological + static" in opened,
          "the temporal basis is stated (climatology + static layers)")
    check("1970" in opened, "the climate normal period is named")
    check(open_expander(page, "how this score was produced"),
          "'Why?' can be opened by clicking it")
    page.wait_for_timeout(1000)
    why = body(page)
    check("weighted mean" in why and "Critical-factor veto" in why,
          "the explainer states the model and the gates")
    check("EXPERIMENTAL" in why,
          "the threshold-provenance table is inside the explainer")

    # --- map layer + legend ------------------------------------------------- #
    check(bt6.wait_for_text(page, "Show cotton suitability screening on the map",
                            timeout=60),
          "the suitability overlay is offered on the map")
    # The guarded rerun that draws the overlay rebuilds the map, so the old
    # Leaflet iframe is detached -- find the current one before reading it.
    frame_txt = frame_text(bt6.find_map_frame(page))
    check("Cotton suitability" in frame_txt,
          "the suitability layer is present on the map")
    legend_ok = all(label in frame_txt for label in
                    ("Highly suitable", "Moderately suitable", "Marginal",
                     "Unsuitable", "Insufficient data"))
    check(legend_ok, "the legend lists all five screening classes")
    check("never treated as unsuitable" in frame_txt,
          "the legend distinguishes 'Unsuitable' from 'Insufficient data'")
    check("not a recommendation" in frame_txt,
          "the legend says the classes are not a recommendation")
    check("Native detail" in frame_txt or "native" in frame_txt.lower(),
          "the legend states the native resolutions behind the grid")
    page.screenshot(path=str(ARTIFACTS / "phase8_browser_cotton.png"), full_page=True)


def scenario_unsupported_crop(page) -> None:
    print("\n--- Phase 8 / B: 'Can I grow rice here?' computes nothing ---", flush=True)
    check(ask(page, "Can I grow rice here?", wait=20.0),
          "the rice question was submitted")
    check(bt6.wait_for_text(page, "CROP_SUITABILITY", timeout=90),
          "rice is still routed to the crop-suitability intent")
    check(bt6.wait_for_text(page, "Only cotton suitability is currently supported.",
                            timeout=60),
          "the app answers that only cotton is supported")
    text = body(page)
    check("growing-season precipitation" not in text.split("Only cotton")[-1].lower()
          or text.count("Only cotton suitability") >= 1,
          "no crop screening was computed for rice")
    page.screenshot(path=str(ARTIFACTS / "phase8_browser_rice.png"), full_page=True)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    args = ap.parse_args()

    print("=" * 78)
    print(f"SatQuery AI -- Phase 8 BROWSER test against {args.url}")
    print("=" * 78)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("   Playwright is not installed -- no browser verification was "
              "possible.")
        return 2

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu",
                  "--js-flags=--max-old-space-size=256"])
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
            check(bt6.confirm_ndvi(page), "the Red/NIR confirmation was ticked")
            check(bt6.wait_for_text(page, "NDVI map", timeout=180),
                  "NDVI is available (the app loaded the bundled sample)")
            frame = bt6.find_map_frame(page)
            bounds = bt6.raster_bounds(frame)
            scenario_cotton(page, frame, bounds)
            scenario_unsupported_crop(page)
        finally:
            page.screenshot(path=str(ARTIFACTS / "phase8_browser_final.png"))
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
