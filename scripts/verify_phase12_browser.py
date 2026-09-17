#!/usr/bin/env python
"""Phase 12 -- one fresh browser smoke test of composed multi-condition queries.

One scenario, against a freshly started server:

    draw a ROI inside the Sentinel-2 scene
      -> ask "Find cropland with NDVI greater than 0.6 and NDWI less than -0.4"
         -- the composed-condition panel, the threshold provenance, the
            undecided-cell count and the composed map layer must appear,
            with every pre-existing layer still on the map
      -> ask "Find cropland with high NDVI"
         -- refused: no threshold is invented, and an opt-in is offered
      -> ask "Show areas with vegetation decrease near permanent water."
         -- composed over two dates, with the boundary statement on screen
      -> ask "Did flooding happen?" and "Compare NDWI before and after."
         -- both still refused, by name

Usage:
    python scripts/verify_phase12_browser.py --url http://127.0.0.1:8501
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

CHAT_CSS = "textarea[placeholder*='Ask SatQuery']"

CAVEAT = "A combined condition is geographic evidence, not causal attribution."
# Words that may appear ONLY inside the boundary statement or the limitations.
BANNED_OUTSIDE_CAVEAT = ("flooded", "flooding", "caused by", "crop failure",
                         "drought", "damage", "deforestation")
LIMITATIONS = (
    "NDVI decrease together with water proximity does not establish flooding.",
    "NDVI decrease together with a high NDWI does not establish flooding.",
    "A low NDVI together with a high NDWI does not classify an area as water.",
    "NDVI decrease does not establish crop failure, drought, deforestation or damage.",
    "NDWI does not establish water availability, water quality or flood extent.",
    "Thresholds are user-specified or explicitly labelled conventions; none of them is a validated scientific classification.",
    "Unknown cells are never counted as matches or as non-matches.",
    "Conditions are combined only after CRS, transform, shape and cell size have been verified identical; grids are never silently resampled.",
    "Land cover is the ESA WorldCover 2021 epoch; it is not a statement about the current season or year.",
)


def check(results, ok, label, detail=""):
    results.append((bool(ok), label, str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def ask(page, text, results, wait_for, deadline_s=420):
    """Type `text`, make sure the app REGISTERED it, then wait for the answer.

    Streamlit discards keystrokes that land while a rerun is in flight, so the
    submission is verified (the question must appear in the transcript) and
    retried. Without this a slow scenario silently reports "no answer" when in
    fact the question was never asked.
    """
    marker = text.split(".")[0][:30]
    deadline = time.time() + deadline_s
    body = ""
    while time.time() < deadline:
        try:
            box = page.locator(CHAT_CSS)
            box.wait_for(state="visible", timeout=120000)
            box.click()
            box.fill(text)
            box.press("Enter")
        except Exception:
            page.wait_for_timeout(4000)
            continue
        for _ in range(10):                      # ~30 s to register the message
            page.wait_for_timeout(3000)
            body = page.inner_text("body")
            if marker in body:
                break
        else:
            continue                             # never registered -- ask again
        while time.time() < deadline:            # now wait for the answer
            page.wait_for_timeout(6000)
            body = page.inner_text("body")
            if wait_for(body):
                return body
        return body
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--shot", default="artifacts/phase12_composition.png")
    args = ap.parse_args()

    print("=" * 74)
    print("SatQuery AI -- Phase 12 composition: browser smoke test")
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

            drawn = draw_rect(page, 0.46, 0.54)
            check(results, drawn, "a rectangle was drawn on the map")
            page.wait_for_timeout(6000)
            wait_idle(page, timeout=240)
            body = page.inner_text("body")
            check(results, "Usable area" in body, "the ROI was registered for analysis")
            check(results, "Analysis data is not available" not in body,
                  "the ROI is inside the analysed scene")

            # -- 3. a composed query with two explicit thresholds ------------ #
            body = ask(
                page,
                "Find cropland with NDVI greater than 0.6 and NDWI less than -0.4",
                results,
                wait_for=lambda b: "Composed condition" in b
                and "Matching cells" in b)
            check(results, "Composed condition" in body,
                  "the composed-condition panel rendered")
            # The answer names the analysis in the user's words; the intent code
            # is internal and is deliberately not printed any more.
            check(results, "Combined geographic conditions" in body,
                  "the query was routed to the composition intent")
            check(results, "Matching cells" in body, "matching cells are reported")
            check(results, "Undecided cells" in body,
                  "undecided cells are reported as their own number")
            check(results, "Measured, not matching" in body,
                  "measured non-matches are reported separately")
            check(results, "Matched area" in body, "the matched area is reported")
            check(results, "your query" in body,
                  "the threshold provenance is shown (your query)")
            check(results, "NDVI > 0.6" in body or "NDVI > 0.6" in body,
                  "the NDVI threshold is shown with its operator")
            check(results, "NDWI < -0.4" in body or "NDWI < -0.4" in body,
                  "the NDWI threshold is shown with its operator")
            check(results, "Interpreted as" in body,
                  "the normalised query is shown")

            # -- 4. the boundary statement, and nothing against it ----------- #
            check(results, CAVEAT in body,
                  "the boundary statement is on screen")
            rest = body
            for _ in range(12):
                if CAVEAT not in rest:
                    break
                rest = rest.replace(CAVEAT, " ")
            for limitation in LIMITATIONS:
                rest = rest.replace(limitation, " ")
            leaks = [w for w in BANNED_OUTSIDE_CAVEAT if w in rest.lower()]
            check(results, not leaks,
                  "no causal claim outside the boundary statement",
                  ", ".join(leaks))

            # -- 5. the map layer -------------------------------------------- #
            frame = bt6.find_map_frame(page, timeout=180)
            names = frame.evaluate(
                "() => Array.from(document.querySelectorAll("
                "'.leaflet-control-layers-overlays label span')).map(e => "
                "e.textContent.trim())")
            joined = " | ".join(names)
            check(results, any("Composed conditions" in n for n in names),
                  "the composed-condition layer is on the map", joined)

            # -- 5b. it was ADDED, not substituted --------------------------- #
            for pre in ("Raster footprint", "True colour (RGB)",
                        "False colour (NIR-R-G)"):
                check(results, any(pre.lower() in n.lower() for n in names),
                      f"pre-existing layer still present: {pre}")
            check(results, len(names) >= 4,
                  "the composed layer was added, not substituted",
                  f"{len(names)} overlay entries")

            # -- 6. a bare "high NDVI" is refused, with a fix ---------------- #
            body = ask(page, "Find cropland with high NDVI", results,
                       wait_for=lambda b: "no threshold will be invented" in b,
                       deadline_s=240)
            check(results, "no threshold will be invented" in body,
                  "a missing threshold is refused, not invented")
            check(results, "0.6" in body,
                  "the refusal shows how to supply one")
            check(results, "labelled convention" in body.lower(),
                  "the opt-in convention is offered, not applied")

            # -- 6b. opt-in really is opt-in: nothing happens until clicked -- #
            check(results, "enabled by you" not in body.lower(),
                  "no result was produced before the opt-in was clicked")
            clicked = False
            for locator in page.get_by_text(
                    "Use a labelled convention instead (opt-in)").all():
                try:
                    locator.click(timeout=8000)
                    clicked = True
                except Exception:
                    continue
            check(results, clicked, "the opt-in expander opened")
            page.wait_for_timeout(3000)
            for locator in page.get_by_role(
                    "button", name="high NDVI").all():
                try:
                    locator.click(timeout=8000)
                except Exception:
                    continue
            deadline = time.time() + 300
            body = ""
            while time.time() < deadline:
                page.wait_for_timeout(6000)
                body = page.inner_text("body")
                # the phrase below appears ONLY in the per-condition provenance
                # line of a result that actually used the convention -- the
                # convention's own note text is worded differently.
                if "enabled by you" in body.lower():
                    break
            check(results, "display/query convention" in body.lower(),
                  "the chosen convention is used and labelled as a convention")
            check(results, "not a scientific classification" in body.lower(),
                  "the convention is not presented as science")

            # -- 7. a temporal + spatial composition ------------------------- #
            body = ask(page, "Show areas with vegetation decrease near permanent water.",
                       results,
                       wait_for=lambda b: "change class = decrease" in b.lower()
                       and "mapped permanent water" in b.lower(),
                       deadline_s=600)
            check(results, "mapped permanent water" in body.lower(),
                  "the spatial condition is named")
            check(results, "change class = decrease" in body.lower(),
                  "the change condition is named")
            check(results, "Sources, dates and grid" in body,
                  "sources and dates are offered")
            for locator in page.get_by_text("Sources, dates and grid").all():
                try:
                    locator.click(timeout=8000)
                except Exception:
                    continue
            page.wait_for_timeout(5000)
            body = page.inner_text("body")
            check(results, "Dates: 2023-01-18" in body
                  and "2023-08-06" in body,
                  "the answer carries both acquisition dates")
            check(results, "Sources, dates and grid" in body,
                  "sources and dates are available")

            # -- 8. the Phase 11 refusals still hold -------------------------- #
            body = ask(page, "Did flooding happen?", results,
                       wait_for=lambda b: "not available yet" in b,
                       deadline_s=180)
            check(results, "not available yet" in body,
                  "a flood question is still refused")
            body = ask(page, "Compare NDWI before and after.", results,
                       wait_for=lambda b: "NDWI change between two dates" in b,
                       deadline_s=180)
            check(results, "not available yet" in body,
                  "temporal NDWI is still refused")

            # -- 9. a fresh screenshot ---------------------------------------- #
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
