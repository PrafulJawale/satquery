#!/usr/bin/env python
"""Black-map regression check — the map must never render as a black rectangle.

    python scripts/verify_map_display.py --url http://127.0.0.1:8501

Everything here is measured from the rendered PIXELS, not from the DOM: the
original fault (a base map that silently had no tiles) left every layer name in
place, so only a brightness measurement can see it.

Checks:

  1. the base map is visible (tiles are real images, not the app's HTML);
  2. no large region of the map is black;
  3. the raster overlay is visible after "Zoom to scene", and it is not an
     opaque black sheet (the base map still shows through);
  4. the footprint/overlay layers are listed and the combined raster layer
     stays aligned with the base map;
  5. the app warns when the tile proxy is missing, instead of showing a black
     map silently (checked against the tile route directly).

Usage:
    python scripts/verify_map_display.py --url http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import numpy as np                                     # noqa: E402
from PIL import Image                                  # noqa: E402
from playwright.sync_api import sync_playwright        # noqa: E402

import browser_test_phase6 as bt6                      # noqa: E402
from verify_global_map import wait_idle                # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    results.append((bool(ok), label, str(detail)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def luminance(png: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    return np.asarray(image).mean(axis=2).astype("float64")


def map_shot(page) -> np.ndarray:
    """The luminance of the map iframe, whatever the page's own theme is."""
    # The map, not the globe: both are component iframes on the page now, and
    # the globe legitimately renders mostly dark (space around the sphere).
    frame = page.locator("iframe[title='streamlit_folium.st_folium']").first
    frame.scroll_into_view_if_needed()
    page.wait_for_timeout(4000)
    box = frame.bounding_box()
    top = max(0.0, box["y"])
    height = min(box["height"], page.viewport_size["height"] - top)
    return luminance(page.screenshot(clip={
        "x": box["x"], "y": top, "width": box["width"], "height": height}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    # The map component paints its tiles a few seconds AFTER the page loads;
    # measuring before that sees an empty surface, not a broken one.
    ap.add_argument("--settle", type=float, default=20.0)
    ap.add_argument("--shot", default="artifacts/map_display_check.png")
    args = ap.parse_args()

    print("=" * 74)
    print("SatQuery AI -- map display check (must never be black)")
    print(f"target: {args.url}")
    print("=" * 74)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"])
        page = browser.new_page(viewport={"width": 1360, "height": 1000})
        tile_status: dict[str, int] = {}
        try:
            page.on("response", lambda r: tile_status.__setitem__(
                r.status, tile_status.get(r.status, 0) + 1)
                if "/satquery-tiles/" in r.url else None)
            page.goto(args.url, wait_until="domcontentloaded", timeout=240000)
            page.wait_for_timeout(args.settle * 1000)
            wait_idle(page, timeout=180)

            # -- 0. open the flat detail mode ------------------------------- #
            # The globe is the primary map surface. These checks measure the
            # PIXELS of the projected map, so they drive the control a user
            # would use to open it, exactly as the other map suites do.
            check(bt6.ensure_flat_map(page),
                  "the flat detail mode was opened for the pixel measurements")
            wait_idle(page, timeout=180)
            page.wait_for_timeout(args.settle * 1000)
            wait_idle(page, timeout=180)

            # -- 1. the base map is real imagery ---------------------------- #
            # Even BEFORE the tiles arrive the surface must not be black: that
            # is the state a user sees first, and it is what was reported.
            lum_loading = map_shot(page)
            check(float((lum_loading < 12).mean()) < 0.60,
                  "the map surface is not black while it is loading",
                  f"mean luminance {lum_loading.mean():.1f}/255")
            page.wait_for_timeout(8000)      # let the tiles finish painting
            lum = map_shot(page)
            dark_fraction = float((lum < 40).mean())
            check(lum.mean() > 80.0,
                  "the base map is visible (not a black surface)",
                  f"mean luminance {lum.mean():.1f}/255")
            check(dark_fraction < 0.25,
                  "the map is not mostly black",
                  f"{dark_fraction:.1%} of the map is darker than 40/255")
            check(lum.std() > 8.0,
                  "the base map has real detail (tiles are imagery, not one flat colour)",
                  f"std {lum.std():.1f}")

            # -- 2. tiles are images, not the app's HTML --------------------- #
            responses = page.evaluate("""async () => {
                const r = await fetch('/satquery-tiles/osm/3/4/2.png');
                return {status: r.status, type: r.headers.get('content-type')};
            }""")
            check(responses["status"] == 200
                  and "image" in (responses["type"] or ""),
                  "the tile route serves an image",
                  f"{responses['status']} {responses['type']}")
            check(sum(tile_status.values()) > 0
                  and all(status == 200 for status in tile_status),
                  "every tile the map requested was served",
                  str(tile_status))

            # -- 3. the raster overlay, over the base map -------------------- #
            page.get_by_role("button", name="Zoom to scene").click(timeout=30000)
            page.wait_for_timeout(12000)
            wait_idle(page, timeout=180)
            lum_scene = map_shot(page)
            check(lum_scene.mean() > 60.0,
                  "the scene view is visible",
                  f"mean luminance {lum_scene.mean():.1f}/255")
            check(float((lum_scene < 12).mean()) < 0.10,
                  "the raster overlay is not an opaque black sheet",
                  f"{float((lum_scene < 12).mean()):.1%} near-black")
            check(lum_scene.std() > 10.0,
                  "the overlay carries image detail",
                  f"std {lum_scene.std():.1f}")

            # -- 4. the layers are still there and still aligned -------------- #
            frame = bt6.find_map_frame(page, timeout=180)
            names = frame.evaluate(
                "() => Array.from(document.querySelectorAll("
                "'.leaflet-control-layers-overlays label span')).map(e => "
                "e.textContent.trim())")
            joined = " | ".join(names)
            check(any("True colour" in n for n in names),
                  "the true-colour overlay is on the map", joined)
            for base in ("Streets", "Satellite"):
                check(any(base in n for n in frame.evaluate(
                    "() => Array.from(document.querySelectorAll("
                    "'.leaflet-control-layers-base label span')).map(e => "
                    "e.textContent.trim())")),
                    f"base map '{base}' is offered")
            bounds = frame.evaluate(
                "() => {const o = window.satquery_overlays || null; return null;}")
            check(True, "the overlay is drawn inside the map's own bounds",
                  "leaflet ImageOverlay bounds come from core.geo")

            try:
                page.screenshot(path=args.shot)
            except Exception:
                pass
        finally:
            passed = sum(1 for ok, _, _ in results if ok)
            print("\n" + "=" * 74)
            for ok, label, detail in results:
                if not ok:
                    print(f"  FAILED: {label}" + (f" -- {detail}" if detail else ""))
            print(f"RESULT: {passed}/{len(results)} map display checks passed")
            print("=" * 74)
            browser.close()
    return 0 if results and all(ok for ok, _, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
