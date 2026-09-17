"""Phase 10 -- one fresh browser smoke test of the temporal NDVI comparison.

This is deliberately ONE scenario, run against a freshly started server:

    draw a ROI inside the Sentinel-2 scene
      -> ask "Compare NDVI before and after."
      -> the change panel, the numbers and the map layers must all appear

It reuses the drawing helpers that were proven against this app (the streamlit-
folium iframe re-renders asynchronously, so every step re-acquires the frame).

Usage:
    python scripts/verify_phase10_browser.py --url http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import browser_test_phase6 as bt6                       # noqa: E402
from verify_global_map import draw_rect, wait_idle      # noqa: E402

CHAT_CSS = "textarea[placeholder*='Ask SatQuery']"
CAUSAL_CLAIMS = ("deforestation", "crop failure", "flooding", "drought")


def check(results, ok, label, detail=""):
    results.append((bool(ok), label, str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--shot", default="artifacts/phase10_change.png")
    args = ap.parse_args()

    print("=" * 74)
    print("SatQuery AI -- Phase 10 temporal NDVI change: browser smoke test")
    print(f"target: {args.url}")
    print("=" * 74)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed -- no browser verification")
        return 2

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu",
            "--js-flags=--max-old-space-size=256",
        ])
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=240000)
            page.wait_for_timeout(args.settle * 1000)
            wait_idle(page, timeout=180)

            # -- 1. the app and its map ------------------------------------- #
            frame = bt6.find_map_frame(page, timeout=180)
            check(results, frame is not None, "the interactive map is present")
            body = page.inner_text("body")

            # -- 2. the two acquisitions are offered ------------------------ #
            check(results, "2023-01-18" in body and "2023-08-06" in body,
                  "both acquisition dates are offered in the selectors")
            check(results, "Before (earlier acquisition)" in body
                  and "After (later acquisition)" in body,
                  "the before/after selectors are labelled")

            # -- 3. zoom to the scene and draw a ROI inside it --------------- #
            for label in ("Zoom to scene",):
                try:
                    page.get_by_role("button", name=label).click(timeout=20000)
                except Exception as exc:
                    check(results, False, f"clicked '{label}'", str(exc)[:80])
                page.wait_for_timeout(5000)
                wait_idle(page, timeout=180)

            drawn = draw_rect(page, 0.44, 0.58)
            check(results, drawn, "a rectangle was drawn on the map")
            page.wait_for_timeout(6000)
            wait_idle(page, timeout=240)
            body = page.inner_text("body")
            check(results, "Usable area" in body,
                  "the ROI was registered for analysis",
                  next((l for l in body.splitlines() if "Usable area" in l), ""))
            check(results, "Analysis data is not available" not in body,
                  "the ROI is inside the analysed scene")

            # -- 4. ask the temporal question -------------------------------- #
            box = page.locator(CHAT_CSS)
            box.wait_for(state="visible", timeout=120000)
            box.click()
            box.fill("Compare NDVI before and after.")
            box.press("Enter")

            # The comparison reads two 10 m scenes over the ROI and then builds
            # four display layers, so allow it real time.
            deadline = time.time() + 420
            found = False
            while time.time() < deadline:
                page.wait_for_timeout(4000)
                body = page.inner_text("body")
                if "NDVI change" in body and "NDVI after (mean)" in body:
                    found = True
                    break
            check(results, found, "the change panel rendered")

            # -- 5. the numbers and the wording ------------------------------- #
            check(results, "NDVI before (mean)" in body, "before NDVI is reported")
            check(results, "NDVI after (mean)" in body, "after NDVI is reported")
            check(results, "ΔNDVI (mean)" in body, "ΔNDVI is reported")
            check(results, "Cells compared" in body, "the compared-cell count is reported")
            check(results, "Increase" in body and "Decrease" in body
                  and "Stable" in body, "all three change classes are shown")
            # The class table is a dataframe (canvas), so assert on the app's
            # own wording instead: it must say that unusable cells are reported
            # as insufficient data and never counted as unchanged.
            check(results,
                  "reported as insufficient data, never as unchanged" in body,
                  "insufficient data is separated from stable")
            check(results, "Additional data is required to identify the cause" in body,
                  "the causal caveat is shown")

            # the caveat NAMES what it rules out; nothing else may claim a cause
            caveat_at = body.find("Additional data is required")
            rest = body[:caveat_at] if caveat_at > 0 else body
            claims = [w for w in CAUSAL_CLAIMS if w in rest.lower()]
            check(results, not claims, "no causal claim outside the caveat",
                  ", ".join(claims))

            # -- 6. the map layers ------------------------------------------- #
            frame = bt6.find_map_frame(page, timeout=180)
            names = frame.evaluate(
                "() => Array.from(document.querySelectorAll("
                "'.leaflet-control-layers-overlays label span')).map(e => "
                "e.textContent.trim())")
            joined = " | ".join(names)
            check(results, any("ΔNDVI" in n for n in names),
                  "the ΔNDVI layer is on the map", joined)
            check(results, any("change class" in n.lower() for n in names),
                  "the change-class layer is on the map")
            check(results, any("NDVI before" in n for n in names)
                  and any("NDVI after" in n for n in names),
                  "the before/after NDVI layers are on the map")
            check(results, not any(n.strip() == "NDVI" for n in names),
                  "no temporal layer is squatting on the 'NDVI' name", joined)

            # -- 6b. nothing that was on the map before was removed ---------- #
            # "Do not overwrite existing analytical layers" is verified here as
            # COEXISTENCE: every pre-existing layer is still in the control,
            # alongside the four new ones. (The single-date "NDVI" layer itself
            # needs the Phase 3 band-confirmation gate, which this scenario does
            # not perform -- that gate is unchanged and is covered by the
            # Phase 1-9 regression suite.)
            for pre in ("Raster footprint", "True colour (RGB)",
                        "False colour (NIR-R-G)"):
                check(results, any(pre.lower() in n.lower() for n in names),
                      f"pre-existing layer still present: {pre}")
            check(results, len(names) >= 8,
                  "the temporal layers were added, not substituted",
                  f"{len(names)} overlay entries")

            # -- 7. a fresh screenshot ---------------------------------------- #
            page.wait_for_timeout(3000)
            out = Path(args.shot)
            out.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(out), full_page=False)
            print(f"  screenshot: {out}")
        finally:
            try:
                page.screenshot(path=str(Path(args.shot).with_suffix(".final.png")))
            except Exception:
                pass
            browser.close()

    failed = [r for r in results if not r[0]]
    print("-" * 74)
    print(f"RESULT: {len(results) - len(failed)} passed, {len(failed)} failed")
    for ok, label, detail in failed:
        print(f"  FAILED: {label} -- {detail}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
