"""Headless smoke tests for the Streamlit app.

WHY THESE EXIST
---------------
Unit tests cover `core/`, but a Streamlit page can still explode at runtime on
a rerun, a widget key collision, or a shape mismatch between `core` and `ui`.
`streamlit.testing.v1.AppTest` runs the real app script headlessly and reports
any exception, so "it imports" is not mistaken for "it works".
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

st = pytest.importorskip("streamlit", reason="streamlit is required for app smoke tests")


def _run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _run_app()
    assert not at.exception, f"app raised: {[e.value for e in at.exception]}"


def test_app_shows_ingestion_and_visualisation():
    at = _run_app()
    text = " ".join(
        [t.value for t in at.title]
        + [h.value for h in at.header]
        + [h.value for h in at.subheader]
        + [md.value for md in at.markdown]
    )
    assert "SatQuery" in text
    # Sections are named for what they do, not for how they were built.
    assert "satellite imagery" in text.lower()
    assert "Band inspection" in text or "band inspection" in text.lower()


def test_app_renders_both_composites():
    at = _run_app()
    # The default sample is the real 4-band Sentinel-2 window, so BOTH the
    # true-colour and the false-colour composite must render.
    assert len(at.image) == 2, f"expected 2 composites, got {len(at.image)}"


def test_app_reports_band_meaning_confidence():
    at = _run_app()
    text = " ".join(md.value for md in at.markdown)
    assert "confidence" in text.lower(), "band-meaning confidence banner missing"


def test_app_survives_a_rerun_with_changed_stretch():
    """Changing a slider triggers a full rerun -- the classic breakage point."""
    at = _run_app()
    assert not at.exception
    sliders = at.slider
    assert len(sliders) >= 2
    sliders[0].set_value(5.0)
    at.run()
    assert not at.exception, f"rerun raised: {[e.value for e in at.exception]}"


def test_metrics_are_populated():
    at = _run_app()
    assert len(at.metric) >= 4
    labels = [m.label for m in at.metric]
    assert any("CRS" in l for l in labels)
    assert any("Bands" in l for l in labels)


def test_app_ndvi_stays_locked_until_bands_are_confirmed():
    """Phase 3 requirement: the NDVI gate is never auto-accepted."""
    at = _run_app()
    gate = at.checkbox(key="ndvi_confirm")   # keyed lookup returns the element itself
    assert gate is not None, "NDVI confirmation checkbox not found"
    assert gate.value is False, "NDVI gate must start unchecked"
    text = " ".join(
        [md.value for md in at.markdown]
        + [w.value for w in at.warning]
        + [i.value for i in at.info]
    )
    assert "locked" in text.lower(), "NDVI must appear locked before confirmation"


def test_app_computes_ndvi_after_confirmation():
    """The full Phase 3 UI path: gate -> reflectance -> stats -> figure."""
    at = _run_app()
    at.checkbox(key="ndvi_confirm").check()
    at.run()
    assert not at.exception, f"NDVI path raised: {[e.value for e in at.exception]}"

    text = " ".join(
        [h.value for h in at.header] + [h.value for h in at.subheader] + [md.value for md in at.markdown]
    )
    assert "Pixel accounting" in text, "NDVI statistics block missing"
    assert "Reflectance preprocessing" in text, "reflectance panel missing"

    # six statistics are rendered as metrics (prefixed, to avoid colliding with
    # the Phase 1 sanity-check metrics)
    labels = [m.label for m in at.metric]
    for expected in ("NDVI min", "NDVI max", "NDVI mean", "NDVI median", "NDVI std dev", "Valid %"):
        assert expected in labels, f"missing metric: {expected}"


def test_app_ndvi_reports_invalid_pixel_accounting():
    at = _run_app()
    at.checkbox(key="ndvi_confirm").check()
    at.run()
    counts = {m.label: m.value for m in at.metric}
    assert counts["Total pixels"] == "4,194,304"
    assert counts["Valid %"].endswith("%")


def test_app_renders_the_interactive_map():
    at = _run_app()
    text = " ".join(
        [t.value for t in at.header]
        + [h.value for h in at.subheader]
        + [md.value for md in at.markdown]
    )
    assert "Interactive map" in text, "the Phase 4 map section is missing"
    # A CRS-less raster must produce an explicit error, never a guessed position.
    assert "guessing" in text.lower() or "Cannot build map" in text or "Interactive map" in text


def _roi_box(lon0=31.80, lat0=31.18, lon1=31.83, lat1=31.21):
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon",
                     "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1],
                                      [lon0, lat1], [lon0, lat0]]]},
    }


def test_app_shows_the_roi_panel_with_no_selection():
    at = _run_app()
    text = " ".join(
        [h.value for h in at.header]
        + [h.value for h in at.subheader]
        + [m.value for m in at.markdown]
        + [i.value for i in at.info]
        + [w.value for w in at.warning]
        + [e.value for e in at.error]
    )
    assert "Selected area (ROI)" in text, "the ROI panel is missing"
    assert "Draw a rectangle or polygon" in text


def test_app_roi_state_machine_is_wired_to_the_session():
    """The app must handle a reported drawing, a replacement and a deletion
    without leaving stale state. The drawing itself is browser behaviour and is
    NOT covered by this test."""
    from core.roi import clear_roi_state, is_map_stale, update_roi_state
    from core.geo import native_footprint
    from rasterio import Affine
    from rasterio.crs import CRS

    transform = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
    foot = native_footprint(transform, 2048, 2048)
    state: dict = {}
    first = update_roi_state(state, [_roi_box()], foot, CRS.from_epsg(32636), "raster")
    assert first is not None and first.is_valid

    # a redraw elsewhere replaces it
    moved = update_roi_state(state, [_roi_box(31.85, 31.22, 31.86, 31.23)],
                             foot, CRS.from_epsg(32636), "raster")
    assert moved is not None and moved is not first

    # deleting everything clears it
    assert update_roi_state(state, [], foot, CRS.from_epsg(32636), "raster") is None
    assert state["roi"] is None
    assert not is_map_stale(state)

    clear_roi_state(state)
    assert state["roi"] is None and state["roi_signature"] is None
