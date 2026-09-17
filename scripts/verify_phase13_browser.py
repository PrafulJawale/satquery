#!/usr/bin/env python
"""Phase 13 -- one fresh browser smoke test of the evidence panel.

One scenario, against a freshly started server:

    draw a ROI inside the Sentinel-2 scene
      -> ask "Find cropland with NDVI greater than 0.6 and NDWI less than -0.4"
         -- the evidence panel must render next to the UNCHANGED Phase 12
            panel, with the counts, the threshold provenance, the undecided
            cells, the boundary statement and the export control
      -> switch on the per-condition evidence layers
         -- one extra layer per condition, named with its provenance, and the
            combined layer still on the map
      -> ask "Show areas with vegetation decrease near permanent water."
         -- the two analysis dates must be on screen
      -> ask the pre-existing questions again
         -- NDVI statistics still answer, flooding and temporal NDWI are still
            unsupported, and a bare "high NDVI" is still refused

Submission is VERIFIED: Streamlit swallows keystrokes that land during a
rerun, so every question is checked against the transcript and re-asked until
it has actually been registered.

Usage:
    python scripts/verify_phase13_browser.py --url http://127.0.0.1:8501
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

BOUNDARY = "A combined condition is geographic evidence, not causal attribution."
# Words that may appear ONLY inside the boundary statement or the limitations.
BANNED_OUTSIDE_BOUNDARY = ("flooded", "flooding", "caused by", "crop failure",
                           "drought", "damage", "deforestation")

CASE_A = "Find cropland with NDVI greater than 0.6 and NDWI less than -0.4"
CASE_B = "Show areas with vegetation decrease near permanent water."

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    results.append((bool(ok), label, str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def ask(page, text, wait_for, deadline_s=420, marker=None):
    """Type `text`, verify the app REGISTERED it, then wait for the answer."""
    marker = marker or text.split(".")[0][:30]
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
        registered = False
        for _ in range(10):                      # ~30 s to register the message
            page.wait_for_timeout(3000)
            body = body_text(page)
            if marker in body:
                registered = True
                break
        if not registered:
            continue                             # never registered -- ask again
        while time.time() < deadline:            # now wait for the answer
            page.wait_for_timeout(6000)
            body = body_text(page)
            if wait_for(body):
                return body
        return body
    return body


def body_text(page, timeout: float = 120000.0) -> str:
    """The whole page as text.

    Five answers deep, the page carries every evidence panel at once, so a
    single read can exceed Playwright's 30 s default. A read that fails is
    retried rather than aborting the whole verification.
    """
    for _ in range(3):
        try:
            return page.inner_text("body", timeout=timeout)
        except Exception:
            page.wait_for_timeout(3000)
    return ""


def metric_after(body: str, label: str) -> str:
    """The number Streamlit printed under `label` (metrics render as
    'label' then the value, each in its own element)."""
    import re

    # The LAST occurrence: the newest answer is at the bottom of the chat.
    index = body.rfind(label)
    if index < 0:
        return ""
    tail = body[index + len(label):index + len(label) + 120]
    match = re.search(r"[\d][\d,]*(\.\d+)?", tail)
    return match.group(0) if match else ""


def overlay_names(page) -> list[str]:
    """The names currently in the Leaflet layers control."""
    frame = bt6.find_map_frame(page, timeout=180)
    if frame is None:
        return []
    return frame.evaluate(
        "() => Array.from(document.querySelectorAll("
        "'.leaflet-control-layers-overlays label span')).map(e => "
        "e.textContent.trim())")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--shot", default="artifacts/phase13_evidence.png")
    args = ap.parse_args()

    print("=" * 74)
    print("SatQuery AI -- Phase 13 evidence: browser smoke test")
    print(f"target: {args.url}")
    print("=" * 74)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed -- no browser verification")
        return 2

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu",
            "--js-flags=--max-old-space-size=256",
        ])
        page = browser.new_page(viewport={"width": 1360, "height": 1000})
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=240000)
            page.wait_for_timeout(args.settle * 1000)
            wait_idle(page, timeout=180)

            # -- 1. the app and its map ------------------------------------- #
            frame = bt6.find_map_frame(page, timeout=180)
            check(frame is not None, "the interactive map is present")

            # -- 2. zoom to the scene and draw a ROI inside it --------------- #
            try:
                page.get_by_role("button", name="Zoom to scene").click(timeout=20000)
            except Exception as exc:
                check(False, "clicked 'Zoom to scene'", str(exc)[:80])
            page.wait_for_timeout(5000)
            wait_idle(page, timeout=180)

            check(draw_rect(page, 0.46, 0.54), "a rectangle was drawn on the map")
            page.wait_for_timeout(6000)
            wait_idle(page, timeout=240)
            body = body_text(page)
            check("Usable area" in body, "the ROI was registered for analysis")

            # -- 3. the composed query, and the evidence panel beside it ----- #
            body = ask(
                page, CASE_A,
                wait_for=lambda b: "Why this result?" in b
                and "Matched / valid cells" in b,
                marker="Find cropland with NDVI greater than 0")
            check("Why this result?" in body, "the evidence panel rendered")
            check("Composed condition" in body,
                  "the Phase 12 panel still renders, unchanged")
            check("Matching cells" in body,
                  "the Phase 12 counts are still reported")
            # The ROI is drawn by a drag, so the counts cannot be hard-coded:
            # the check is that the evidence panel and the Phase 12 panel
            # report the SAME numbers, which is the whole claim of Phase 13.
            phase12_matched = metric_after(body, "Matching cells")
            evidence_matched = metric_after(body, "Matched / valid cells")
            check(bool(phase12_matched) and phase12_matched == evidence_matched,
                  "the evidence panel reports the engine's matched count",
                  f"phase 12 {phase12_matched} vs evidence {evidence_matched}")
            phase12_unknown = metric_after(body, "Undecided cells")
            check(bool(phase12_unknown) and "Undecided cells" in body,
                  "the undecided cells are reported as their own number",
                  phase12_unknown)
            check("km²" in body, "the matched area is reported in km²",
                  metric_after(body, "Matched area"))
            check("from your query" in body,
                  "the threshold provenance is shown (your query)")
            check("Export evidence (JSON)" in body,
                  "the export control is present")
            check(BOUNDARY in body, "the boundary statement is on screen")

            # no causal wording anywhere except the boundary/limitations
            rest = body
            for _ in range(12):
                if BOUNDARY not in rest:
                    break
                rest = rest.replace(BOUNDARY, " ")
            for limitation in ("does not establish flooding",
                               "does not classify an area as water",
                               "does not establish crop failure",
                               "does not establish water availability"):
                rest = rest.replace(limitation, " ")
            leaks = [w for w in BANNED_OUTSIDE_BOUNDARY if w in rest.lower()]
            # 'flooding'/'flooded' may also appear in the refusal text for a
            # flood question, which is asked later; this body predates that.
            check(not leaks, "no causal claim outside the boundary statement",
                  ", ".join(leaks))

            # -- 4. the map layers: combined still there, evidence opt-in ---- #
            names = overlay_names(page)
            joined = " | ".join(names)
            check(any("Composed conditions" in n for n in names),
                  "the combined layer is still on the map", joined)
            for pre in ("Raster footprint", "True colour (RGB)",
                        "False colour (NIR-R-G)"):
                check(any(pre.lower() in n.lower() for n in names),
                      f"pre-existing layer still present: {pre}")
            check(not any(n.startswith("Evidence") for n in names),
                  "no evidence layer was added automatically (opt-in)")

            toggled = False
            for locator in page.get_by_text(
                    "Show individual condition layers on the map").all():
                try:
                    locator.click(timeout=10000)
                    toggled = True
                    break
                except Exception:
                    continue
            check(toggled, "the evidence-layer switch was clicked")
            deadline = time.time() + 240
            evidence_names: list[str] = []
            while time.time() < deadline:
                page.wait_for_timeout(6000)
                names = overlay_names(page)
                evidence_names = [n for n in names if n.startswith("Evidence")]
                if evidence_names:
                    break
            check(len(evidence_names) >= 2,
                  "one evidence layer per condition appeared",
                  " | ".join(evidence_names))
            check(any("NDVI" in n for n in evidence_names)
                  and any("NDWI" in n or "cropland" in n.lower()
                          for n in evidence_names),
                  "the layer names carry the condition's provenance",
                  " | ".join(evidence_names))
            check(any("Composed conditions" in n for n in overlay_names(page)),
                  "the combined layer stayed after the evidence layers were added")

            # -- 5. case B: two dates on screen ----------------------------- #
            body = ask(
                page, CASE_B,
                wait_for=lambda b: "2023-01-18" in b and "2023-08-06" in b,
                marker="Show areas with vegetation decrease near permanent water")
            check("2023-01-18" in body and "2023-08-06" in body,
                  "both analysis dates are displayed")
            matched_b = metric_after(body, "Matched / valid cells")
            phase12_b = metric_after(body, "Matching cells")
            check(bool(matched_b) and matched_b == phase12_b,
                  "case B reports the engine's matched count",
                  f"phase 12 {phase12_b} vs evidence {matched_b}")

            # -- 6. the export control really is a download ------------------ #
            download = page.get_by_role("button", name="Export evidence (JSON)")
            check(download.count() > 0, "the JSON export control exists")
            # A real Streamlit download: the app must be idle when the button
            # is pressed. Pressing it while the app is re-running (the panel is
            # rebuilt on every rerun, and the two answers above keep adding map
            # layers) can leave Chromium saving the file under its media URL
            # instead of the file name. Wait for idle, and allow a second
            # press -- the assertion on the name itself is unchanged.
            suffix = ""
            problem = ""
            for _attempt in range(3):
                wait_idle(page, timeout=180)
                page.wait_for_timeout(2000)
                try:
                    with page.expect_download(timeout=60000) as info:
                        page.get_by_role(
                            "button", name="Export evidence (JSON)").first.click()
                    suffix = str(info.value.suggested_filename)
                except Exception as exc:
                    problem = str(exc)[:120]
                    continue
                if suffix.startswith("satquery_evidence_") \
                        and suffix.endswith(".json"):
                    break
            check(suffix.startswith("satquery_evidence_")
                  and suffix.endswith(".json"),
                  "the export produces a JSON file", suffix or problem)

            # -- 7. the pre-existing behaviour is unchanged ------------------ #
            bt6.confirm_ndvi(page, timeout=180)     # the Phase 3 band gate
            page.wait_for_timeout(4000)
            wait_idle(page, timeout=180)
            body = ask(page, "What is the NDVI of this area?",
                       wait_for=lambda b: "mean NDVI" in b
                       or "NDVI of" in b or "confirm" in b.lower(),
                       deadline_s=300)
            check("mean NDVI" in body, "NDVI ROI statistics still answer",
                  "" if "mean NDVI" in body else body[-160:])

            body = ask(page, "Did flooding happen?",
                       wait_for=lambda b: "not available yet." in b,
                       deadline_s=300)
            check("not available yet." in body,
                  "flood detection is still unsupported, by name")

            body = ask(page, "Compare NDWI before and after.",
                       wait_for=lambda b: "not available yet." in b,
                       deadline_s=300)
            check("not available yet." in body,
                  "temporal NDWI is still unsupported, by name")

            body = ask(page, "Find cropland with high NDVI",
                       wait_for=lambda b: "no threshold will be invented" in b,
                       deadline_s=300)
            check("no threshold will be invented" in body,
                  "a bare 'high NDVI' is still refused, not invented")

            try:
                page.screenshot(path=args.shot, full_page=False)
            except Exception:
                pass
        finally:
            passed = sum(1 for ok_, _, _ in results if ok_)
            print("\n" + "=" * 74)
            for ok_, label, detail in results:
                if not ok_:
                    print(f"  FAILED: {label}" + (f" -- {detail}" if detail else ""))
            print(f"RESULT: {passed}/{len(results)} browser checks passed")
            print("=" * 74)
            browser.close()
    return 0 if all(ok_ for ok_, _, _ in results) and results else 1


if __name__ == "__main__":
    sys.exit(main())
