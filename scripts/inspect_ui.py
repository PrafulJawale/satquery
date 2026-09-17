#!/usr/bin/env python
"""Dump what the running app actually shows: text, widgets, styles, layout.

    python scripts/inspect_ui.py --url http://127.0.0.1:8501 [--shot artifacts/ui.png]

A redesign has to start from what a user sees, not from the source. This script
reports, from the live DOM:

  * the page outline (titles, headings, section text), in order;
  * every widget by kind and label;
  * computed typography and spacing, so inconsistencies are visible;
  * duplicated text (the same sentence rendered twice);
  * wording that must not be in a product ("phase", "prototype", raw paths,
    tracebacks, internal names);
  * the map surfaces present on the page (there must be exactly one primary one).

Usage:
    python scripts/inspect_ui.py --url http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from playwright.sync_api import sync_playwright  # noqa: E402

FORBIDDEN = [
    r"prototype", r"phase\s*\d+", r"phase\s*[-–]\s*\d", r"roadmap",
    r"\.py\b", r"/home/user", r"/tmp/", r"Traceback", r"st\.", r"DEBUG",
    r"TODO", r"FIXME", r"verify_", r"scripts/", r"not implemented yet",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    ap.add_argument("--settle", type=float, default=18.0)
    ap.add_argument("--shot", default="artifacts/ui_inspect.png")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.url, wait_until="domcontentloaded", timeout=240000)
        page.wait_for_timeout(int(args.settle * 1000))

        print("=" * 78)
        print(f"UI inspection -- {args.url} @ {args.width}x{args.height}")
        print("=" * 78)

        # ---- 1. outline -------------------------------------------------- #
        outline = page.evaluate(
            """() => {
                const sel = 'h1, h2, h3, h4, [data-testid="stMarkdownContainer"] > p,'
                    + ' [data-testid="stCaptionContainer"], label, .st-emotion-cache-1tpl0r0';
                const out = [];
                for (const el of document.querySelectorAll(sel)) {
                    const t = (el.innerText || '').trim();
                    if (!t) continue;
                    const r = el.getBoundingClientRect();
                    if (r.width === 0 && r.height === 0) continue;
                    out.push({tag: el.tagName.toLowerCase(), text: t.slice(0, 160),
                              y: Math.round(r.top + window.scrollY)});
                }
                return out;
            }"""
        )
        print("\n--- 1. page outline (in document order) ---")
        seen: list[dict] = []
        for item in outline:
            if seen and seen[-1]["text"] == item["text"]:
                continue
            seen.append(item)
            print(f"  {item['y']:>6}px  [{item['tag']}] {item['text'][:120]}")

        # ---- 2. widgets --------------------------------------------------- #
        widgets = page.evaluate(
            """() => {
                const out = {};
                const push = (kind, el) => {
                    const label = (el.getAttribute('aria-label')
                        || (el.closest('[data-testid]')?.innerText || '') || '').trim();
                    (out[kind] = out[kind] || []).push(label.slice(0, 80));
                };
                document.querySelectorAll('input[type=text], textarea').forEach(e => push('text_input', e));
                document.querySelectorAll('button').forEach(e => push('button', e));
                document.querySelectorAll('input[type=checkbox]').forEach(e => push('checkbox', e));
                document.querySelectorAll('input[type=radio]').forEach(e => push('radio', e));
                document.querySelectorAll('select, [role=listbox]').forEach(e => push('select', e));
                document.querySelectorAll('[data-baseweb="slider"]').forEach(e => push('slider', e));
                document.querySelectorAll('[data-testid="stExpander"]').forEach(e => push('expander', e));
                document.querySelectorAll('[data-testid="stMetric"]').forEach(e => push('metric', e));
                document.querySelectorAll('[data-testid="stDataFrame"]').forEach(e => push('table', e));
                document.querySelectorAll('[data-testid="stDownloadButton"]').forEach(e => push('download', e));
                document.querySelectorAll('[data-testid="stAlert"]').forEach(e => push('alert', e));
                return out;
            }"""
        )
        print("\n--- 2. widgets ---")
        for kind, labels in sorted(widgets.items()):
            uniq = sorted({l for l in labels if l})
            print(f"  {kind:<12} {len(labels):>3}  {', '.join(uniq[:8])[:150]}")

        # ---- 3. typography and spacing ------------------------------------ #
        styles = page.evaluate(
            """() => {
                const pick = (sel) => {
                    const el = document.querySelector(sel);
                    if (!el) return null;
                    const s = getComputedStyle(el);
                    return {font: s.fontFamily.split(',')[0], size: s.fontSize,
                            weight: s.fontWeight, color: s.color, lh: s.lineHeight};
                };
                const cards = Array.from(document.querySelectorAll(
                    '[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"]'))
                    .slice(0, 6)
                    .map(e => { const s = getComputedStyle(e);
                        return {radius: s.borderRadius, border: s.borderTopWidth + ' ' + s.borderTopColor,
                                bg: s.backgroundColor, pad: s.padding, shadow: s.boxShadow.slice(0, 40)}; });
                return {
                    body: pick('body'), h1: pick('h1'), h2: pick('h2'), h3: pick('h3'),
                    caption: pick('[data-testid="stCaptionContainer"]'),
                    button: pick('button'), input: pick('input[type=text], textarea'),
                    cards: cards,
                };
            }"""
        )
        print("\n--- 3. typography & surfaces ---")
        print(json.dumps(styles, indent=2)[:2600])

        # ---- 4. duplicated sentences -------------------------------------- #
        text = page.inner_text("body", timeout=120000)
        sentences = [s.strip() for s in re.split(r"[\n\.]", text) if len(s.strip()) > 45]
        counts: dict[str, int] = {}
        for s in sentences:
            counts[s] = counts.get(s, 0) + 1
        dupes = {s: c for s, c in counts.items() if c > 1}
        print(f"\n--- 4. duplicated sentences: {len(dupes)} ---")
        for s, c in list(dupes.items())[:8]:
            print(f"  x{c}: {s[:110]}")

        # ---- 5. wording that must not ship -------------------------------- #
        print("\n--- 5. non-product wording ---")
        hits = 0
        for pattern in FORBIDDEN:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                snippet = text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
                print(f"  /{pattern}/ -> ...{snippet[:130]}...")
                hits += 1
                break
        if not hits:
            print("  none")

        # ---- 6. map surfaces ---------------------------------------------- #
        surfaces = page.evaluate(
            """() => Array.from(document.querySelectorAll('iframe')).map(f => ({
                title: f.title || '(none)', w: Math.round(f.getBoundingClientRect().width),
                h: Math.round(f.getBoundingClientRect().height)}))"""
        )
        leaflet = page.evaluate(
            """() => {
                let n = 0;
                for (let i = 0; i < window.frames.length; i++) {
                    try { if (window.frames[i].document.querySelector('.leaflet-container')) n++; }
                    catch (e) {}
                }
                let cesium = 0;
                for (let i = 0; i < window.frames.length; i++) {
                    try { if (window.frames[i].document.querySelector('.cesium-widget, canvas')) cesium++; }
                    catch (e) {}
                }
                return 'iframes: ' + document.querySelectorAll('iframe').length
                    + ' | leaflet containers: ' + n
                    + ' | frames with a canvas (globe): ' + cesium;
            }"""
        )
        print(f"\n--- 6. map surfaces ---\n  {surfaces}\n  {leaflet}")

        try:
            page.screenshot(path=args.shot, full_page=True)
            print(f"\n  full-page screenshot: {args.shot}")
        except Exception as exc:
            print(f"  screenshot failed: {exc}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
