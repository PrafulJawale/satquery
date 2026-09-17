#!/usr/bin/env python
"""Phase 11 -- one fresh browser smoke test of NDWI.

One scenario, against a freshly started server:

    draw a ROI inside the Sentinel-2 scene
      -> ask "What is the NDWI of this area?"
      -> the NDWI panel, the caveat and the NDWI map layer must appear,
         with every pre-existing layer still on the map
      -> then ask "Did flooding happen?" and "Compare NDWI before and after."
         -- both must be refused, by name

Usage:
    python scripts/verify_phase11_browser.py --url http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import browser_test_phase6 as bt6                        # noqa: E402
from verify_global_map import draw_rect, wait_idle       # noqa: E402

sys.path.insert(0, str(SCRIPTS.parent))
from core.index_definitions import get_index              # noqa: E402

LIMITATIONS = tuple(get_index("ndwi").limitations)

CHAT_CSS = "textarea[placeholder*='Ask SatQuery']"
CAVEAT = ("NDWI is a spectral index. This result does not by itself establish "
          "flood extent, water availability, or water quality.")
# Claims that may only ever appear inside that caveat.
BANNED_OUTSIDE_CAVEAT = ("water body", "water bodies", "is water", "flooded",
                         "water quality", "water availability", "flood extent")


def check(results, ok, label, detail=""):
    results.append((bool(ok), label, str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def ask(page, text, results, wait_for, deadline_s=420):
    box = page.locator(CHAT_CSS)
    box.wait_for(state="visible", timeout=120000)
    box.click()
    box.fill(text)
    box.press("Enter")
    deadline = time.time() + deadline_s
    body = ""
    while time.time() < deadline:
        page.wait_for_timeout(4000)
        body = page.inner_text("body")
        if wait_for(body):
            break
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--shot", default="artifacts/phase11_ndwi.png")
    args = ap.parse_args()

    print("=" * 74)
    print("SatQuery AI -- Phase 11 NDWI: browser smoke test")
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

            # -- 2. zoom to the scene and draw a ROI inside it --------------- #
            try:
                page.get_by_role("button", name="Zoom to scene").click(timeout=20000)
            except Exception as exc:
                check(results, False, "clicked 'Zoom to scene'", str(exc)[:80])
            page.wait_for_timeout(5000)
            wait_idle(page, timeout=180)

            drawn = draw_rect(page, 0.44, 0.58)
            check(results, drawn, "a rectangle was drawn on the map")
            page.wait_for_timeout(6000)
            wait_idle(page, timeout=240)
            body = page.inner_text("body")
            check(results, "Usable area" in body, "the ROI was registered for analysis")
            check(results, "Analysis data is not available" not in body,
                  "the ROI is inside the analysed scene")

            # -- 3. ask for NDWI --------------------------------------------- #
            body = ask(page, "What is the NDWI of this area?", results,
                       wait_for=lambda b: "NDWI mean" in b and "NDWI — water index" in b)
            check(results, "NDWI — water index" in body, "the NDWI panel rendered")
            check(results, "NDWI mean" in body, "NDWI mean is reported")
            check(results, "NDWI median" in body, "NDWI median is reported")
            check(results, "NDWI std dev" in body, "NDWI standard deviation is reported")
            check(results, "Cells in selection" in body, "the ROI cell count is reported")
            check(results, "Valid cells" in body, "the valid-pixel count is reported")

            # -- 4. the caveat, and nothing that contradicts it --------------- #
            check(results, CAVEAT in body, "the NDWI caveat is shown")
            # The caveat is deliberately repeated (result panel AND map legend),
            # and the limitations are explicit NEGATIONS ("not a water body").
            # Both are known-good text from config/indices/ndwi.yml, so strip
            # every copy of them and then look for any real claim.
            rest = body
            for _ in range(10):
                if CAVEAT not in rest:
                    break
                rest = rest.replace(CAVEAT, " ")
            for limitation in LIMITATIONS:
                rest = rest.replace(limitation, " ")
            leaks = [w for w in BANNED_OUTSIDE_CAVEAT if w in rest.lower()]
            check(results, not leaks,
                  "no water/flood claim outside the caveat and limitations",
                  ", ".join(leaks))

            # -- 5. the map layer --------------------------------------------- #
            frame = bt6.find_map_frame(page, timeout=180)
            names = frame.evaluate(
                "() => Array.from(document.querySelectorAll("
                "'.leaflet-control-layers-overlays label span')).map(e => "
                "e.textContent.trim())")
            joined = " | ".join(names)
            check(results, any("NDWI" in n for n in names),
                  "the NDWI layer is on the map", joined)

            # -- 5b. it was ADDED, not substituted ---------------------------- #
            for pre in ("Raster footprint", "True colour (RGB)",
                        "False colour (NIR-R-G)"):
                check(results, any(pre.lower() in n.lower() for n in names),
                      f"pre-existing layer still present: {pre}")
            check(results, len(names) >= 4,
                  "the NDWI layer was added, not substituted",
                  f"{len(names)} overlay entries")

            # -- 6. a flood question is still refused -------------------------- #
            body = ask(page, "Did flooding happen?", results,
                       wait_for=lambda b: "not available yet" in b, deadline_s=180)
            check(results, "not available yet" in body,
                  "a flood question is refused")
            check(results, "FLOOD" in body.upper(),
                  "the refusal names the flood intent")

            # -- 7. temporal NDWI is still refused ----------------------------- #
            body = ask(page, "Compare NDWI before and after.", results,
                       wait_for=lambda b: "NDWI change between two dates" in b,
                       deadline_s=180)
            check(results, "not available yet" in body,
                  "temporal NDWI is refused")
            check(results, "single date" in body.lower(),
                  "the refusal says why (single-date NDWI only)")

            # -- 8. a fresh screenshot ----------------------------------------- #
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
