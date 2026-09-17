"""The rendered app must be free of obsolete build-phase language.

`streamlit.testing.v1.AppTest` runs the real `app.py`, so these assertions are
about what a user actually sees -- not about what the source happens to
contain. Two rules are enforced:

1. No phase numbering anywhere in the UI. The old banner ("Prototype · Phase 8
   of 8 ...") is GONE, not renamed: there is no "Phase 13 of 13" either. A build
   phase is information for the people building the app, not for the people
   using it.
2. The base-map picker offers only base maps that work without an API key.
   "Light" is removed; "Streets" and "Satellite" remain.

The search box is asserted here too, so the navigation surface cannot disappear
in a later cleanup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

pytest.importorskip("streamlit", reason="streamlit is required for app tests")


def _run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"
    return at


def _all_text(at) -> str:
    """Every piece of text the app puts on the page."""
    parts: list[str] = []
    for name in ("title", "header", "subheader", "markdown", "caption", "text",
                 "warning", "error", "info", "success", "help", "tooltip"):
        for element in getattr(at, name, []):
            value = getattr(element, "value", None)
            if isinstance(value, str):
                parts.append(value)
    for select in getattr(at, "selectbox", []):
        parts.append(str(getattr(select, "label", "")))
        parts.extend(str(o) for o in getattr(select, "options", []))
    for slider in list(getattr(at, "slider", [])) + list(getattr(at, "select_slider", [])):
        parts.append(str(getattr(slider, "label", "")))
        help_text = getattr(slider, "help", None)
        if isinstance(help_text, str):
            parts.append(help_text)
    for inp in getattr(at, "text_input", []):
        parts.append(str(getattr(inp, "label", "")))
        help_text = getattr(inp, "help", None)
        if isinstance(help_text, str):
            parts.append(help_text)
    for button in getattr(at, "button", []):
        parts.append(str(getattr(button, "label", "")))
        help_text = getattr(button, "help", None)
        if isinstance(help_text, str):
            parts.append(help_text)
    for checkbox in getattr(at, "checkbox", []):
        parts.append(str(getattr(checkbox, "label", "")))
        help_text = getattr(checkbox, "help", None)
        if isinstance(help_text, str):
            parts.append(help_text)
    return " \n ".join(parts)


def test_no_phase_numbering_is_rendered():
    at = _run_app()
    text = _all_text(at)
    # Provenance is DATA, not UI chrome: the sample scene's provenance record
    # cites the audit document it came from (`docs/PHASE2.md`). That citation is
    # part of the evidence trail and must not be rewritten, so filename
    # references are set aside before scanning for build-phase wording.
    text = re.sub(r"\bPHASE\d+\.md\b", "audit-document", text, flags=re.IGNORECASE)
    hits = re.findall(r"phase\s*\d+", text, flags=re.IGNORECASE)
    assert not hits, f"the UI still numbers its build phases: {hits}"
    assert "prototype" not in text.lower(), "obsolete prototype wording is back"
    assert "of 8 —" not in text and "of 13" not in text


def test_the_prototype_banner_is_gone_not_renamed():
    """The removal is a deletion: no phase-count replacement may appear."""
    at = _run_app()
    text = _all_text(at)
    for forbidden in ("Prototype ·", "Phase 8 of 8", "Phase 13 of 13",
                      "of 8 —", "of 13 —"):
        assert forbidden not in text, f"found the old wording: {forbidden!r}"


def test_the_app_still_says_what_it_is():
    """Removal, not vandalism: the product name and purpose remain."""
    at = _run_app()
    text = _all_text(at)
    assert "SatQuery" in text
    assert "geospatial" in text.lower()


def test_the_light_base_map_is_not_offered():
    at = _run_app()
    pickers = [s for s in at.selectbox if "imagery" in str(getattr(s, "label", "")).lower()]
    assert pickers, "the base-map picker is missing"
    options = [str(o) for o in pickers[0].options]
    assert "Light" not in options, f"'Light' is back: {options}"
    assert "Streets" in options and "Satellite" in options


def test_the_default_base_map_is_streets():
    """The default still works with no key, and still is the street map."""
    at = _run_app()
    pickers = [s for s in at.selectbox if "imagery" in str(getattr(s, "label", "")).lower()]
    assert str(pickers[0].value) == "osm", f"unexpected default: {pickers[0].value!r}"


def test_no_api_key_warning_is_rendered():
    at = _run_app()
    text = _all_text(at).lower()
    for token in ("api key", "apikey", "access token", "sign up for a"):
        assert token not in text, f"a map style still asks for a key: {token}"


def test_the_place_search_box_is_still_there():
    """Global navigation must survive the cleanup."""
    at = _run_app()
    labels = [str(t.label) for t in at.text_input]
    assert any("search" in label.lower() for label in labels), labels
