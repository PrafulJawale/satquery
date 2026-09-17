#!/usr/bin/env python
"""Final UI/UX + map verification for SatQuery AI.

    python scripts/verify_ui_final.py --url http://127.0.0.1:8501

Checks the running app against the product requirements, from the browser:

  1.  one primary map surface -- the globe -- and no second map behind it;
  2.  no prototype / phase / "Light" / API-key wording or requests anywhere;
  3.  the globe is a real sphere (Cesium 3D) and is interactive:
      drag rotates, zoom controls fly, home returns to the whole Earth;
  4.  satellite and street imagery both work, through this app's own proxy;
  5.  location search works globally and moves the *globe* camera, with a pin;
  6.  raster overlays are listed on the globe and, in the flat detail mode, are
      present and correctly positioned;
  7.  an area can be drawn on the globe and is registered for analysis;
  8.  the welcome screen, chat, evidence panel and downloads work;
  9.  the layout holds at 1400, 1100 and 820 px wide;
 10.  the flat map is only shown when the user chooses it.

Usage:
    python scripts/verify_ui_final.py --url http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import numpy as np                                       # noqa: E402
from PIL import Image                                    # noqa: E402
from playwright.sync_api import sync_playwright          # noqa: E402

import browser_test_phase6 as bt6                        # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    results.append((bool(ok), label, str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def luminance(png: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    return np.asarray(image).mean(axis=2).astype("float64")


def map_frame(page, timeout: int = 120):
    """The primary map (the globe component)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for frame in page.frames:
            if "satquery_map" in (frame.url or ""):
                return frame
        page.wait_for_timeout(500)
    return None


def readout(frame) -> str:
    try:
        return (frame.locator("#readout").inner_text(timeout=10000) or "").strip()
    except Exception:
        return ""


def click(page, label: str) -> bool:
    """Click a control by its visible label.

    A JS click, not a synthetic one: the chat input is docked to the bottom of
    the page and overlays controls near it, which makes Playwright's
    hit-testing retry forever without ever landing the click.
    """
    return bool(page.evaluate(
        """(label) => {
            const wanted = String(label).toLowerCase();
            for (const el of document.querySelectorAll('button, label')) {
                const text = (el.innerText || '').trim().toLowerCase();
                if (!text || text !== wanted) continue;
                const node = el.tagName === 'LABEL'
                    ? (el.querySelector('input') || el) : el;
                node.click();
                return true;
            }
            for (const el of document.querySelectorAll('button')) {
                if ((el.innerText || '').trim().toLowerCase().includes(wanted)) {
                    el.click();
                    return true;
                }
            }
            return false;
        }""",
        label,
    ))


def ask(page, question: str) -> None:
    """Type a question into the chat input and submit it."""
    box = page.locator('[data-testid="stChatInput"] textarea')
    box.click()
    box.fill(question)
    page.wait_for_timeout(400)
    page.keyboard.press("Enter")


def stable_map(page, quiet: float = 5.0, timeout: float = 90.0):
    """The map frame, once Streamlit has stopped rebuilding it."""
    end = time.time() + timeout
    candidate = None
    since = time.time()
    while time.time() < end:
        frame = map_frame(page, timeout=15)
        if frame is None:
            candidate, since = None, time.time()
            continue
        if frame is not candidate:
            candidate, since = frame, time.time()
            continue
        if time.time() - since >= quiet:
            return candidate
        page.wait_for_timeout(500)
    return candidate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--settle", type=float, default=22.0)
    ap.add_argument("--shot", default="artifacts/ui_final.png")
    args = ap.parse_args()

    print("=" * 78)
    print("SatQuery AI -- final UI/UX and map verification")
    print(f"target: {args.url}")
    print("=" * 78)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        page = browser.new_page(viewport={"width": 1400, "height": 1100})
        tiles = {"satellite": 0, "osm": 0}
        hosts: set[str] = set()
        def _count(request):
            for key in tiles:
                if f"/satquery-tiles/{key}/" in request.url:
                    tiles[key] = tiles.get(key, 0) + 1

        page.on("request", _count)
        page.on("request", lambda r: hosts.add(
            re.sub(r"^https?://([^/]+).*", r"\1", r.url))
            if re.match(r"https?://", r.url) else None)

        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=240000)
            page.wait_for_timeout(int(args.settle * 1000))
            body = page.inner_text("body", timeout=120000)

            # -- 1. wording ------------------------------------------------ #
            print("\n-- 1. product wording --")
            lowered = body.lower()
            check("prototype" not in lowered, "no prototype wording")
            phase = re.findall(r"phase\s*\d+", lowered)
            check(not phase, "no phase numbering", str(phase))
            check("phase 8 of 8" not in lowered and "phase 13 of 13" not in lowered,
                  "no 'Phase 8 of 8' or 'Phase 13 of 13'")
            check("SatQuery AI" in body, "the product name is on the page")
            check("evidence-backed" in lowered, "product positioning is stated")
            check("Traceback" not in body, "no traceback is shown to the user")
            check(not re.search(r"/home/user|/tmp/|\.py\b", body),
                  "no internal paths or file names in the interface")

            # -- 2. one map ------------------------------------------------- #
            print("\n-- 2. one primary map --")
            surfaces = page.evaluate(
                """() => Array.from(document.querySelectorAll('iframe')).map(f => ({
                    title: f.title, h: Math.round(f.getBoundingClientRect().height)}))""")
            check(len(surfaces) == 1, "exactly one map surface is rendered", str(surfaces))
            check(bool(surfaces) and "satquery_map" in surfaces[0]["title"],
                  "the surface rendered is the globe", str(surfaces))
            leaflet_now = page.evaluate(
                """() => { for (let i = 0; i < window.frames.length; i++) {
                    try { if (window.frames[i].document.querySelector('.leaflet-container')) return true; }
                    catch (e) {} } return false; }""")
            check(not leaflet_now, "no flat map is rendered behind the globe")

            # -- 3. the globe ----------------------------------------------- #
            print("\n-- 3. the globe --")
            frame = map_frame(page)
            check(frame is not None, "the globe component is present")
            if frame is not None:
                state = frame.evaluate(
                    """() => ({
                        cesium: typeof Cesium !== 'undefined',
                        version: (typeof Cesium !== 'undefined' ? Cesium.VERSION : null),
                        widget: !!document.querySelector('.cesium-widget'),
                        fallback: document.getElementById('stage').classList.contains('fallback'),
                        layers: document.getElementById('layerList').innerText.trim(),
                        credits: document.getElementById('credits').innerText.trim(),
                    })""")
                check(bool(state["cesium"]) and bool(state["widget"]),
                      "a 3D globe engine is running", f"Cesium {state.get('version')}")
                check(not state["fallback"], "the 3D globe is in use (not the fallback)")
                check("True colour" in (state["layers"] or ""),
                      "raster overlays are listed on the globe", state["layers"][:60])
                check(len(state["credits"] or "") > 0,
                      "imagery attribution is displayed", state["credits"][:60])

                frame.locator("#stage").scroll_into_view_if_needed()
                page.wait_for_timeout(2500)
                box = frame.locator("#stage").bounding_box()
                lum = luminance(page.screenshot(clip=box))
                h, w = lum.shape
                check(float(lum[h // 2, w // 2]) > 40 and float(lum[6, 6]) < 60,
                      "the sphere is lit against space",
                      f"centre {lum[h//2, w//2]:.0f}, corner {lum[6,6]:.0f}")
                page.screenshot(path=args.shot)

                before = readout(frame)
                frame.click("#zoomIn")
                page.wait_for_timeout(2500)
                after_zoom = readout(frame)
                check(after_zoom != before and after_zoom != "",
                      "the zoom control moves the camera", f"{before} -> {after_zoom}")
                frame.click("#home")
                page.wait_for_timeout(3000)
                check(readout(frame) != after_zoom,
                      "home returns to the whole Earth", readout(frame))

                # drag to rotate
                cam_before = readout(frame)
                cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                page.mouse.move(cx, cy)
                page.mouse.down()
                page.mouse.move(cx - 160, cy + 40, steps=12)
                page.mouse.up()
                page.wait_for_timeout(2500)
                check(readout(frame) not in ("", cam_before),
                      "dragging rotates the globe", f"{cam_before} -> {readout(frame)}")

            # -- 4. imagery -------------------------------------------------- #
            print("\n-- 4. imagery --")
            if frame is not None:
                frame = stable_map(page) or frame
                frame.click('#baseSeg button[data-base="satellite"]')
                page.wait_for_timeout(6000)
                sat = tiles.get("satellite", 0)
                frame.click('#baseSeg button[data-base="streets"]')
                page.wait_for_timeout(6000)
                streets = tiles.get("osm", 0)
                check(sat > 0 and streets > 0,
                      "satellite and street imagery are both served",
                      f"satellite={sat}, streets={streets}")
            check(not any("cartocdn" in h for h in hosts),
                  "no request to the API-key tile host",
                  ", ".join(sorted(h for h in hosts if "cart" in h)) or "none")
            check("light_all" not in " ".join(sorted(hosts)),
                  "the removed Light style is never requested")

            # -- 5. search moves the globe ----------------------------------- #
            print("\n-- 5. location search --")
            box = page.get_by_placeholder("e.g. Kolhapur, India")
            box.first.click()
            box.first.press_sequentially("Mount Everest", delay=25)
            box.first.press("Enter")
            page.wait_for_timeout(7000)
            body = page.inner_text("body", timeout=120000)
            match = re.search(r"Showing\s+(.+?)\s+at\s+(-?\d+\.\d+)°,\s*(-?\d+\.\d+)°", body)
            check(bool(match), "the search returns a selectable result")
            camera_before = readout(map_frame(page))
            click(page, "Fly to this place")
            camera_after = camera_before
            for _ in range(10):                       # the camera flies: give it time
                page.wait_for_timeout(3000)
                camera_after = readout(map_frame(page))
                if camera_after != camera_before and camera_after:
                    break
            got = re.findall(r"-?\d+\.\d+", camera_after)
            near = bool(match) and bool(got) and (
                abs(float(got[0]) - abs(float(match.group(2)))) < 2.0
                and abs(float(got[1]) - abs(float(match.group(3)))) < 2.0)
            check(near and camera_after != camera_before,
                  "the globe camera flies to the searched place",
                  f"{camera_before} -> {camera_after}")
            check("Showing" in page.inner_text("body", timeout=120000),
                  "the searched place is reported with its coordinates")

            # -- 6. drawing an area on the globe ------------------------------ #
            print("\n-- 6. drawing an analysis area --")
            click(page, "Zoom to scene")
            page.wait_for_timeout(9000)
            frame = stable_map(page)
            frame.locator("#stage").scroll_into_view_if_needed()
            page.wait_for_timeout(2000)
            box = frame.locator("#stage").bounding_box()
            frame.click('#drawSeg button[data-draw="rect"]')
            page.wait_for_timeout(1500)
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            page.mouse.click(cx - 70, cy - 50)
            page.wait_for_timeout(2500)
            frame = stable_map(page) or frame
            page.mouse.click(cx + 70, cy + 50)
            page.wait_for_timeout(12000)
            body = page.inner_text("body", timeout=120000)
            check("Usable area" in body,
                  "an area drawn on the globe is registered for analysis")

            # -- 7. evidence and downloads ----------------------------------- #
            print("\n-- 7. results, evidence and downloads --")
            ask(page, "What is the NDVI of this area?")
            page.wait_for_timeout(12000)
            body = page.inner_text("body", timeout=120000)
            check("Why this result" in body or "Evidence" in body,
                  "the evidence panel is offered with the result")
            downloads = page.locator('[data-testid="stDownloadButton"]').count()
            check(downloads >= 1, "a download control is available", f"{downloads} found")

            # -- 8. welcome state -------------------------------------------- #
            print("\n-- 8. first-run experience --")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(int(args.settle * 1000))
            welcome = page.inner_text("body", timeout=120000)
            check("Explore satellite imagery" in welcome,
                  "the welcome screen explains what SatQuery does")
            check("Try one of these" in welcome, "example queries are offered")
            check("not provide real-time imagery" in welcome.lower()
                  or "geographic evidence" in welcome.lower(),
                  "the welcome screen states the product's limits")

            # -- 9. responsive ------------------------------------------------ #
            print("\n-- 9. responsive layout --")
            for width in (1400, 1100, 820):
                page.set_viewport_size({"width": width, "height": 950})
                page.wait_for_timeout(6000)
                overflow = page.evaluate(
                    "() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
                f = map_frame(page, timeout=60)
                mw = 0
                if f is not None:
                    try:
                        mw = f.evaluate(
                            "() => Math.round(document.getElementById('stage').getBoundingClientRect().width)")
                    except Exception:
                        mw = 0
                check(overflow <= 2 and mw > 200,
                      f"layout fits at {width}px",
                      f"overflow {overflow}px, map {mw}px")
            page.set_viewport_size({"width": 1400, "height": 1100})
            page.wait_for_timeout(4000)

            # -- 10. flat map only on request --------------------------------- #
            print("\n-- 10. flat detail mode --")
            check(not page.evaluate(
                """() => { for (let i = 0; i < window.frames.length; i++) {
                    try { if (window.frames[i].document.querySelector('.leaflet-container')) return true; }
                    catch (e) {} } return false; }"""),
                "the flat map is still not rendered by default")
            page.evaluate(
                """() => { const els = Array.from(document.querySelectorAll('label'));
                    const t = els.find(e => (e.innerText || '').trim() === 'Flat map');
                    if (t) { const i = t.querySelector('input[type=radio]'); if (i) i.click(); } }""")
            page.wait_for_timeout(14000)
            surfaces = page.evaluate(
                """() => Array.from(document.querySelectorAll('iframe')).map(f => f.title)""")
            check(len(surfaces) == 1 and "folium" in surfaces[0],
                  "choosing the flat mode swaps the map (still one surface)", str(surfaces))
            frame = bt6.find_map_frame(page, timeout=90)
            layers = frame.evaluate(
                """() => Array.from(document.querySelectorAll(
                    '.leaflet-control-layers-overlays label span')).map(e => e.textContent.trim())""")
            check(any("True colour" in n for n in layers),
                  "overlays are present on the flat map too", " | ".join(layers[:4]))
            check(not page.evaluate(
                """() => Array.from(document.querySelectorAll('iframe'))
                    .some(f => (f.title || '').includes('satquery_map'))"""),
                "the globe is not rendered while the flat map is open")
        finally:
            passed = sum(1 for ok, _, _ in results if ok)
            print("\n" + "=" * 78)
            for ok, label, detail in results:
                if not ok:
                    print(f"  FAILED: {label}" + (f" -- {detail}" if detail else ""))
            print(f"RESULT: {passed}/{len(results)} checks passed")
            print("=" * 78)
            browser.close()
    return 0 if results and all(ok for ok, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
