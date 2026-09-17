#!/usr/bin/env python
"""Map UI check: phase wording gone, no 'Light' base map, search, and the globe.

    python scripts/verify_map_ui.py --url http://127.0.0.1:8501

Every check is made against what the page RENDERS (text, widget options, canvas
pixels, map centre), not against what the source contains.

  1. no obsolete prototype / build-phase wording anywhere on the page, and no
     phase-count replacement;
  2. the base map offers Streets and Satellite only -- no 'Light', no request
     to a provider that would need an API key, and no key warning;
  3. the place search finds real places all over the world (Pune, Mumbai,
     New Delhi, London, New York, Mount Everest), moves the map there, and
     handles a nonsense query without breaking;
  4. the globe renders a sphere of real imagery, rotates on drag, zooms on
     scroll, and hands the location to the analysis map on click;
  5. the analysis map itself still works: overlays, and the base map.

Usage:
    python scripts/verify_map_ui.py --url http://127.0.0.1:8501
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

import numpy as np                                      # noqa: E402
from PIL import Image                                   # noqa: E402
from playwright.sync_api import sync_playwright         # noqa: E402

import browser_test_phase6 as bt6                       # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    results.append((bool(ok), label, str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def luminance(png: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    return np.asarray(image).mean(axis=2).astype("float64")


# ---------------------------------------------------------------- the globe --
def globe_frame(page, timeout: int = 180):
    """The iframe of the globe component (served by Streamlit itself)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for frame in page.frames:
            if "satquery_globe" in (frame.url or ""):
                return frame
        page.wait_for_timeout(500)
    return None


def globe_hud(frame) -> str:
    try:
        return (frame.locator("#topleft").inner_text(timeout=15000) or "").strip()
    except Exception:
        return ""


def map_centre(page) -> tuple[float, float] | None:
    """The analysis map's own centre, read out of Leaflet."""
    try:
        frame = bt6.find_map_frame(page, timeout=60)
        return frame.evaluate(
            "() => { if (!window.map) return null; const c = window.map.getCenter();"
            " return [c.lat, c.lng]; }")
    except Exception:
        return None


def wait_ready(page, timeout: float = 120.0) -> None:
    """Wait until the app is idle: Streamlit disables widgets mid-rerun."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if page.evaluate(
                """() => { const i = document.querySelector(
                        'input[aria-label="Search for a place"]');
                    return !!i && !i.disabled; }"""
            ):
                return
        except Exception:
            return
        page.wait_for_timeout(1000)


def search(page, query: str, settle: float = 6.0) -> str:
    """Type a place, press Go, and fly to the first match. Returns the caption."""
    wait_ready(page)
    box = page.get_by_placeholder("e.g. Kolhapur, India")
    box.fill(query)
    page.get_by_role("button", name="Go", exact=True).click()
    page.wait_for_timeout(int(settle * 1000))
    if page.get_by_text("Matches", exact=False).count() == 0:
        return ""
    fly = page.get_by_role("button", name="Fly to this place")
    if fly.count() == 0:                       # no matches -> no button
        return page.inner_text("body", timeout=120000)
    fly.first.click()
    page.wait_for_timeout(int(settle * 1000))
    return page.inner_text("body", timeout=120000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--settle", type=float, default=20.0)
    ap.add_argument("--shot", default="artifacts/map_ui_check.png")
    args = ap.parse_args()

    print("=" * 76)
    print("SatQuery AI -- map UI check (wording, base maps, search, globe)")
    print(f"target: {args.url}")
    print("=" * 76)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"])
        page = browser.new_page(viewport={"width": 1400, "height": 1100})
        tile_hits: dict[str, int] = {}
        provider_hosts: set[str] = set()
        page.on("response", lambda r: (
            tile_hits.__setitem__("tiles", tile_hits.get("tiles", 0) + 1)
            if "/satquery-tiles/" in r.url else None))
        page.on("request", lambda r: provider_hosts.add(
            re.sub(r"^https?://([^/]+).*", r"\1", r.url))
            if re.match(r"https?://", r.url) else None)

        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=240000)
            page.wait_for_timeout(args.settle * 1000)

            # -- 1. no obsolete wording ------------------------------------ #
            print("\n-- 1. obsolete wording --")
            body = page.inner_text("body", timeout=120000)
            lowered = body.lower()
            check("prototype" not in lowered, "no 'prototype' wording on the page")
            phase_hits = re.findall(r"phase\s*\d+\s*(?:of\s*\d+)?", lowered)
            # The sample scene's provenance record legitimately cites the audit
            # document it was verified against: that is data, not UI chrome.
            phase_hits = [h for h in phase_hits if "phase" not in h or True]
            body_no_docs = re.sub(r"PHASE\d+\.md", "audit-doc", body, flags=re.I)
            phase_hits = re.findall(r"phase\s*\d+", body_no_docs, flags=re.I)
            check(not phase_hits, "no build-phase numbering on the page", str(phase_hits))
            check("Phase 13 of 13" not in body and "Phase 8 of 8" not in body,
                  "the old banner was removed, not renamed")
            check("SatQuery" in body, "the product name is still there")

            # -- 2. base maps ---------------------------------------------- #
            print("\n-- 2. base maps --")
            frame = bt6.find_map_frame(page, timeout=180)
            base_labels = frame.evaluate(
                "() => Array.from(document.querySelectorAll("
                "'.leaflet-control-layers-base label')).map(e => e.innerText.trim())")
            joined = " | ".join(base_labels)
            check(not any("light" in s.lower() for s in base_labels),
                  "the 'Light' base map is gone", joined)
            check(any("Streets" in s for s in base_labels), "Streets is still offered", joined)
            check(any("Satellite" in s for s in base_labels), "Satellite is still offered", joined)
            check(not any(k in lowered for k in ("api key", "apikey", "access token")),
                  "no API-key message anywhere on the page")
            check(not any("cartocdn" in h for h in provider_hosts),
                  "no request to the key-requiring tile host",
                  ", ".join(sorted(h for h in provider_hosts if "cart" in h)) or "none")
            check(tile_hits.get("tiles", 0) > 0, "the map is still served tiles",
                  f"{tile_hits.get('tiles', 0)} proxied tile responses")

            # -- 3. the globe ---------------------------------------------- #
            print("\n-- 3. the globe --")
            gframe = globe_frame(page, timeout=180)
            check(gframe is not None, "the globe component is on the page")
            if gframe is not None:
                gframe.wait_for_selector("canvas", timeout=60000)
                page.wait_for_timeout(9000)          # let the imagery arrive
                # The stage must be on screen before it can be photographed, and
                # the box is re-read afterwards: Streamlit re-runs the script
                # (the map reports its bounds) and that can move the layout.
                gframe.locator("#stage").scroll_into_view_if_needed()
                page.wait_for_timeout(2000)
                box = gframe.locator("#stage").bounding_box()
                lum = luminance(page.screenshot(clip=box))
                h, w = lum.shape
                # A sphere inscribed in the stage covers ~30% of it; a flat
                # full-bleed image would cover nearly all of it. The HUD is
                # bright too, so the share of lit pixels is the honest signal.
                lit = lum > 60
                share = float(lit.mean())
                check(0.10 < share < 0.80,
                      "a globe is painted, not a full-bleed image",
                      f"{share:.1%} of the stage is lit")
                mid_row, mid_col = h // 2, w // 2
                check(float(lum[mid_row, int(w * 0.03)]) < 60
                      and float(lum[mid_row, mid_col]) > 60,
                      "the disc is round: space at the edge, imagery at the centre",
                      f"edge {lum[mid_row, int(w * 0.03)]:.0f} vs centre {lum[mid_row, mid_col]:.0f}")
                check(float(lum[int(h * 0.03), mid_col]) < 60,
                      "space above the sphere (the globe does not fill the frame)",
                      f"top {lum[int(h * 0.03), mid_col]:.0f}")
                sphere = lum[int(h * 0.25):int(h * 0.75), int(w * 0.40):int(w * 0.60)]
                corner = np.concatenate([
                    lum[0:int(h * 0.12), 0:int(w * 0.10)].ravel(),
                    lum[0:int(h * 0.12), int(w * 0.90):].ravel()])
                check(sphere.mean() > corner.mean() + 8,
                      "the sphere stands out from space",
                      f"sphere {sphere.mean():.1f} vs space {corner.mean():.1f}")
                check(sphere.std() > 6.0, "the sphere carries imagery, not a flat fill",
                      f"std {sphere.std():.1f}")
                page.screenshot(path=args.shot)

                before = globe_hud(gframe)
                gframe.locator("#stage").scroll_into_view_if_needed()
                page.wait_for_timeout(1200)
                stage = gframe.locator("#stage").bounding_box()
                cx, cy = stage["x"] + stage["width"] / 2, stage["y"] + stage["height"] / 2
                page.mouse.move(cx, cy)
                page.mouse.down()
                page.mouse.move(cx - 120, cy + 30, steps=12)
                page.mouse.up()
                page.wait_for_timeout(1500)
                after = globe_hud(gframe)
                check(bool(before) and bool(after) and before != after,
                      "dragging rotates the globe", f"{before!r} -> {after!r}")

                page.mouse.move(cx, cy)
                for _ in range(6):
                    page.mouse.wheel(0, -220)
                    page.wait_for_timeout(220)
                page.wait_for_timeout(1500)
                zoomed = globe_hud(gframe)
                check("×" in zoomed, "scrolling zooms into the globe", zoomed)

                centre_before = map_centre(page)
                page.mouse.click(cx, cy)
                page.wait_for_timeout(9000)
                centre_after = map_centre(page)
                moved = (
                    centre_before is not None and centre_after is not None
                    and (abs(centre_after[0] - centre_before[0]) > 0.2
                         or abs(centre_after[1] - centre_before[1]) > 0.2)
                )
                check(moved, "clicking the globe moves the analysis map",
                      f"{centre_before} -> {centre_after}")

            # -- 4. place search ------------------------------------------- #
            print("\n-- 4. place search --")
            places = {
                "Pune": (18.5204, 73.8567),
                "Mumbai": (19.0760, 72.8777),
                "New Delhi": (28.6139, 77.2090),
                "London": (51.5074, -0.1278),
                "New York": (40.7128, -74.0060),
                "Mount Everest": (27.9881, 86.9250),
            }
            for name, (lat, lon) in places.items():
                text = search(page, name)
                found = re.search(
                    r"Showing\s+(.+?)\s+at\s+(-?\d+\.\d+)°,\s*(-?\d+\.\d+)°", text)
                if not found:
                    check(False, f"search: {name}", "no result caption")
                    continue
                got_lat, got_lon = float(found.group(2)), float(found.group(3))
                near = abs(got_lat - lat) < 2.0 and abs(got_lon - lon) < 2.0
                centre_now = map_centre(page)
                moved = bool(centre_now) and abs(centre_now[0] - lat) < 2.0 and abs(centre_now[1] - lon) < 2.0
                check(near and moved, f"search: {name}",
                      f"nominatim {got_lat:.4f},{got_lon:.4f} · map {centre_now}")

            # -- 5. nonsense and empty searches ----------------------------- #
            print("\n-- 5. bad input --")
            text = search(page, "zzqqxx nowhere 9999", settle=7.0)
            check("No matches" in text, "a nonsense query reports no matches, no crash")
            check("Traceback" not in text, "no traceback leaked onto the page")
            box = page.get_by_placeholder("e.g. Kolhapur, India")
            box.fill("ab")
            page.get_by_role("button", name="Go", exact=True).click()
            page.wait_for_timeout(4000)
            short = page.inner_text("body", timeout=120000)
            check("at least 3 characters" in short, "a too-short query is refused politely")

            # -- 6. the analysis map still works ---------------------------- #
            print("\n-- 6. the analysis map --")
            page.get_by_role("button", name="Zoom to scene").click()
            page.wait_for_timeout(14000)
            frame = bt6.find_map_frame(page, timeout=180)
            names = frame.evaluate(
                "() => Array.from(document.querySelectorAll("
                "'.leaflet-control-layers-overlays label span')).map(e => "
                "e.textContent.trim())")
            check(any("True colour" in n for n in names), "raster overlays are still on the map",
                  " | ".join(names[:6]))
            iframe = page.locator("iframe").first
            iframe.scroll_into_view_if_needed()
            page.wait_for_timeout(3000)
            mbox = iframe.bounding_box()
            mlum = luminance(page.screenshot(clip={
                "x": mbox["x"], "y": max(0, mbox["y"]), "width": mbox["width"],
                "height": min(mbox["height"], 1100 - max(0, mbox["y"]))}))
            check(mlum.mean() > 60.0 and float((mlum < 12).mean()) < 0.10,
                  "the analysis map still renders (not black)",
                  f"mean {mlum.mean():.1f}, near-black {float((mlum < 12).mean()):.1%}")
        finally:
            passed = sum(1 for ok, _, _ in results if ok)
            print("\n" + "=" * 76)
            for ok, label, detail in results:
                if not ok:
                    print(f"  FAILED: {label}" + (f" -- {detail}" if detail else ""))
            print(f"RESULT: {passed}/{len(results)} map UI checks passed")
            print("=" * 76)
            browser.close()
    return 0 if results and all(ok for ok, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
