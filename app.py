"""SatQuery AI -- Streamlit entry point.

An interactive geospatial intelligence assistant: ingest a GeoTIFF/COG, preview
it (true colour and false colour), compute continuous spectral indicators, ask
questions in natural language, draw an area of interest, and get evidence-backed
results with provenance.

The heavy lifting is Streamlit-free on purpose -- `core/` (raster IO,
reprojection, indices, ROI geometry) and `analyses/` (the query router and
engines) can be called from tests or a script; this file is only the UI.

Two rules the UI keeps visible at all times: the map shows **display copies**
(every number comes from the native raster), and a result that cannot be
supported is reported as unsupported rather than guessed.

Run with:
    python serve.py

`serve.py` mounts the tile proxy (`/satquery-tiles`) before Streamlit builds its
server; starting this file directly leaves the map without a base map.
"""

from __future__ import annotations

import base64
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import streamlit as st
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError

from core.bands import describe_guess, guess_band_roles
from core.geo import (
    WEB_MERCATOR,
    MissingCRSError,
    footprint_feature,
    grid_bounds_wgs84,
    lonlat_to_pixel,
    native_footprint,
    pixel_to_lonlat,
    raster_footprint,
    reproject_array,
    reproject_bands,
    rgba_to_png_bytes,
    web_rgba_from_bands,
    web_rgba_from_values,
)
from core.roi import clear_roi_state, is_map_stale, update_roi_state
from ui.globe import MAP_HEIGHT, render_globe
from core.geocode import ATTRIBUTION as GEOCODER_ATTRIBUTION, MIN_QUERY_CHARS, geocode
from analyses import (
    AnalysisContext,
    IndexContext,
    NdviContext,
    route,
    suggestions,
)
from analyses.evidence import evidence_from_entry
from core.planner import (
    MockLLMProvider,
    LLMPlanner,
    execute_with_fallback,
    ToolResult,
    ConversationState,
)
from core.session import (
    build_session_from_app_state,
    load_session_from_file,
    save_session_to_file,
    apply_session_to_conversation_state,
    get_session_summary,
    Session,
    # Step 11: Session Organization & Discovery
    SessionListEntry,
    SessionFilter,
    list_sessions,
    filter_sessions,
    search_sessions,
    archive_session,
    delete_session,
    list_archived_sessions,
    restore_archived_session,
    create_checkpoint,
    list_checkpoints,
    load_checkpoint,
    delete_checkpoint,
)
from core.llm_provider import create_provider_from_env
from core.geometry import GeometryError
from core.statistics import calculate_roi_ndvi_stats, roi_pixel_mask
from core.indices import classify_ndvi, ndvi_from_dataset
from core.preview import (
    DEFAULT_MAX_PIXELS,
    all_band_stats,
    analysis_to_geotiff_bytes,
    band_histograms,
    decimate_array,
    decimated_read,
    encode_png,
    figure_to_png,
    make_preview,
    ndvi_histogram,
    percentile_bounds,
    render_ndvi_figure as build_ndvi_figure,   # core builds the figure; ui displays it
    valid_mask,
)
from core.raster import describe_bytes, describe_path, open_dataset, open_upload, window_stats
from core.reflectance import ReflectanceSpec, detect_reflectance_spec
from core.samples import (SAMPLE_DIR, list_samples, sample_path,
                          sample_provenance)
# Phase 10: the temporal scene-pair model. Imported here (not inside a function)
# because the scene selectors are part of the page, not of one callback.
from core.temporal import ScenePair, discover_scenes, load_temporal_config
from streamlit_folium import st_folium

from typing import Any, Dict, List, Optional, Tuple

from tileserver import proxy_mounted
from ui.map import (BASEMAPS, DEFAULT_BASEMAP, WORLD_CENTRE, WORLD_ZOOM,
                    MapOverlay, build_map, composition_legend_html,
                    composition_rgba, ndvi_change_legend_html,
                    ndvi_change_rgba, ndvi_legend_html, ndwi_legend_html,
                    rgb_legend_html,
                    spatial_legend_html, spatial_rgba, suitability_legend_html,
                    suitability_rgba)
from ui.evidence_panel import (evidence_layer_choice,
                               evidence_layer_colour, render_evidence,
                               web_evidence_mask,
                               render_evidence_explorer,
                               render_provenance_timeline,
                               render_evidence_comparison,
                               export_evidence_report)
from ui.components import (
    band_options,
    render_band_inspector,
    render_band_mapping,
    render_bands_table,
    render_composite,
    render_guess_banner,
    render_histograms,
    render_metrics,
    render_ndvi_classes,
    render_ndvi_figure,
    render_ndwi_stats,
    render_ndvi_change,
    render_ndvi_provenance,
    render_ndvi_stats,
    render_provenance,
    render_ask_satquery,
    render_answer,
    render_welcome,
    render_crop_suitability,
    render_multi_condition,
    render_spatial_query,
    render_unsupported_condition,
    render_roi_analysis,
    render_raw_metadata,
    render_reflectance_panel,
    render_roadmap,
    render_roi_panel,
    render_spatial_table,
    render_warnings,
    render_conversation_context,
    render_clarification,
)

_MARK_PATH = str(Path(__file__).resolve().parent / "assets" / "satquery_mark.svg")
st.set_page_config(page_title="SatQuery AI", page_icon=_MARK_PATH, layout="wide")

from ui.theme import inject as _inject_theme   # noqa: E402  (after page config)

_inject_theme()

st.logo(_MARK_PATH, size="large")

st.markdown(
    """
    <div class="sq-header">
      <img src="data:image/svg+xml;base64,{mark}" alt="" />
      <div>
        <div class="sq-title">SatQuery AI</div>
        <div class="sq-sub">Explore satellite imagery and obtain evidence-backed
        geospatial insights through natural-language queries.</div>
      </div>
      <span class="sq-badge">Geospatial intelligence</span>
    </div>
    """.format(
        mark=base64.b64encode(Path(_MARK_PATH).read_bytes()).decode("ascii")
        if Path(_MARK_PATH).exists() else ""
    ),
    unsafe_allow_html=True,
)

def _source_title(name: str, provenance: Optional[dict]) -> str:
    """A human name for the loaded scene.

    The file name is an implementation detail; what matters to the user is
    which acquisition this is. Falls back to the file stem only when the
    provenance record has no name (e.g. an uploaded file).
    """
    pv = provenance or {}
    for key in ("name", "title"):
        value = str(pv.get(key) or "").strip()
        if value:
            return value
    platform = str(pv.get("platform") or "").strip()
    date = str(pv.get("datetime") or pv.get("date_label") or "").strip()[:10]
    if platform or date:
        return " · ".join(x for x in (platform, date) if x)
    return Path(name).stem


def _section(title: str, note: str = "", number: str = "") -> str:
    """A product section header (see ui.theme)."""
    from ui.theme import section as _sec

    return _sec(title, note=note, number=number)


# --------------------------------------------------------------------------- #
# place search (global navigation)
#
# The geocoder runs on the SERVER (core.geocode -> Nominatim) for the same
# reason tiles do: the browser may sit on a network that blocks third-party
# requests. Search is NAVIGATION ONLY -- it moves the camera and drops a pin.
# It never reads the raster, changes the ROI, or re-runs an analysis; only an
# explicit analysis does that.
# --------------------------------------------------------------------------- #
def run_place_search(query: str) -> None:
    """Look up `query` and store the outcome in the session. Never raises."""
    text = (query or "").strip()
    st.session_state["search_query"] = text
    st.session_state["search_results"] = []
    st.session_state["search_note"] = ""
    st.session_state["map_fit_scene"] = False
    if len(text) < MIN_QUERY_CHARS:
        st.session_state["search_note"] = (
            f"Type at least {MIN_QUERY_CHARS} characters to search."
        )
        return
    try:
        places = geocode(text, limit=5)
    except Exception:                       # a geocoder outage is not an error
        places = []                         # the user must read: it is "no match"
    st.session_state["search_results"] = places
    if not places:
        st.session_state["search_note"] = (
            "No matches for that search (looked up on the server). Try a bigger "
            "place nearby, or a fuller address."
        )


def fly_to_place(index: object = None) -> None:
    """Move the camera to match number `index` and pin it.

    Navigation only: the raster, the ROI and every result are untouched.
    """
    results = st.session_state.get("search_results") or []
    try:
        place = results[int(index)] if index is not None else results[0]
    except (TypeError, ValueError, IndexError):
        return
    st.session_state["map_centre"] = [place.lat, place.lon]
    st.session_state["map_zoom"] = place.zoom_for_bbox or 12
    st.session_state["map_fit_scene"] = False
    st.session_state["map_marker"] = [place.lat, place.lon]
    st.session_state["map_marker_label"] = f"Searched place: {place.name}"
    st.session_state["globe_centre"] = [place.lat, place.lon]
    st.session_state["globe_scale"] = None


def search_from_widget() -> None:
    run_place_search(st.session_state.get("search_input", ""))


def fly_from_widget() -> None:
    fly_to_place(st.session_state.get("search_pick", 0))


# --------------------------------------------------------------------------- #
# cached readers
#
# Streamlit re-runs this script on every widget interaction, so anything
# expensive must be cached. We cache *results* (dicts / small arrays), never an
# open GDAL dataset, which is not safe to share across reruns.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading GeoTIFF metadata…")
def metadata_from_path(path: str, label: str) -> dict:
    return describe_path(path, label=label).to_dict()


@st.cache_data(show_spinner="Reading GeoTIFF metadata…")
def metadata_from_bytes(file_bytes: bytes, label: str) -> dict:
    return describe_bytes(file_bytes, label=label).to_dict()


@st.cache_data(show_spinner="Sampling pixels…")
def sanity_stats_from_path(path: str, band: int = 1) -> dict:
    with open_dataset(path) as ds:
        return window_stats(ds, band=band)


@st.cache_data(show_spinner="Sampling pixels…")
def sanity_stats_from_bytes(file_bytes: bytes, band: int = 1) -> dict:
    with open_upload(file_bytes) as ds:
        return window_stats(ds, band=band)


@st.cache_data(max_entries=6, show_spinner="Inspecting bands…")
def band_stats_path(path: str, max_pixels: int) -> list:
    with open_dataset(path) as ds:
        return all_band_stats(ds, max_pixels=max_pixels)


@st.cache_data(max_entries=6, show_spinner="Inspecting bands…")
def band_stats_bytes(file_bytes: bytes, max_pixels: int) -> list:
    with open_upload(file_bytes) as ds:
        return all_band_stats(ds, max_pixels=max_pixels)


@st.cache_data(max_entries=8, show_spinner="Rendering composite…")
def preview_path(path, bands, names, max_pixels, lo, hi, mode):
    with open_dataset(path) as ds:
        res = make_preview(ds, bands, names, max_pixels=max_pixels,
                           low_pct=lo, high_pct=hi, stretch_mode=mode)
    return res.image, res.alpha, res.to_dict()


@st.cache_data(max_entries=8, show_spinner="Rendering composite…")
def preview_bytes(file_bytes, bands, names, max_pixels, lo, hi, mode):
    with open_upload(file_bytes) as ds:
        res = make_preview(ds, bands, names, max_pixels=max_pixels,
                           low_pct=lo, high_pct=hi, stretch_mode=mode)
    return res.image, res.alpha, res.to_dict()


@st.cache_data(max_entries=4, show_spinner="Computing histograms…")
def hist_path(path, bands, bins, max_pixels) -> list:
    with open_dataset(path) as ds:
        return band_histograms(ds, bands, bins=bins, max_pixels=max_pixels)


@st.cache_data(max_entries=4, show_spinner="Computing histograms…")
def hist_bytes(file_bytes, bands, bins, max_pixels) -> list:
    with open_upload(file_bytes) as ds:
        return band_histograms(ds, bands, bins=bins, max_pixels=max_pixels)


# --- Phase 3: reflectance detection + NDVI ---------------------------------- #
@st.cache_data(max_entries=4, show_spinner="Detecting reflectance scaling…")
def spec_from_path(path: str, profile):
    with open_dataset(path) as ds:
        spec, report = detect_reflectance_spec(ds=ds, profile=profile)
    return spec.to_dict(), report


@st.cache_data(max_entries=4, show_spinner="Detecting reflectance scaling…")
def spec_from_bytes(file_bytes: bytes, profile):
    with open_upload(file_bytes) as ds:
        spec, report = detect_reflectance_spec(ds=ds, profile=profile)
    return spec.to_dict(), report


@st.cache_data(max_entries=2, show_spinner="Computing NDVI…")
def ndvi_from_path(path, red_i, nir_i, scale, offset, profile, source, is_refl):
    with open_dataset(path) as ds:
        spec = ReflectanceSpec(scale=scale, offset=offset, profile=profile,
                               source=source, is_reflectance=is_refl)
        result, _used, _rep = ndvi_from_dataset(ds, red_i, nir_i, reflectance=spec,
                                                auto_detect_reflectance=False)
        return result.array, result.mask, result.to_dict()


# --- Phase 4: reprojection for the web map (DISPLAY ONLY) ------------------- #
@st.cache_data(max_entries=4, show_spinner="Computing stretch…")
def stretch_bounds_from_path(path, bands, max_pixels, low, high):
    """Percentile bounds from the NATIVE decimated read, so the map shows the
    same stretch as the Phase 2 panel."""
    with open_dataset(path) as ds:
        stack, _ = decimated_read(ds, list(bands), max_pixels=max_pixels)
        out = []
        for k, band_index in enumerate(bands):
            m = valid_mask(stack[k], ds.nodatavals[band_index - 1])
            out.append(percentile_bounds(stack[k][m], low, high))
    return tuple(out)


@st.cache_data(max_entries=4, show_spinner="Computing stretch…")
def stretch_bounds_from_bytes(file_bytes, bands, max_pixels, low, high):
    with open_upload(file_bytes) as ds:
        stack, _ = decimated_read(ds, list(bands), max_pixels=max_pixels)
        out = []
        for k, band_index in enumerate(bands):
            m = valid_mask(stack[k], ds.nodatavals[band_index - 1])
            out.append(percentile_bounds(stack[k][m], low, high))
    return tuple(out)


@st.cache_data(max_entries=3, show_spinner="Reprojecting imagery for the map…")
def web_bands_from_path(path, bands, max_pixels, stretch_bounds, resampling_name="average"):
    from rasterio.enums import Resampling

    with open_dataset(path) as ds:
        web = reproject_bands(ds, list(bands), dst_crs=WEB_MERCATOR,
                              resampling=Resampling[resampling_name], max_pixels=max_pixels)
    rgba = web_rgba_from_bands(web, stretch_bounds)
    return rgba, web.to_dict()

def _composition_threshold_lines(result: Any) -> List[str]:
    """One line per threshold, naming the index and WHERE the value came from.

    The map legend has to carry provenance: a number with no origin is exactly
    what Phase 12 refuses to produce.
    """
    lines: List[str] = []
    for entry in getattr(result, "condition_results", None) or []:
        if entry.get("kind") != "spectral":
            continue
        spec = entry.get("threshold_provenance") or {}
        origin = str(spec.get("provenance") or "")
        words = {"user_specified": "your query",
                 "config_convention": "labelled convention (opt-in)",
                 "relative": "relative to this area"}.get(origin, origin)
        lines.append(
            f"{str(entry.get('operator', ''))} {entry.get('threshold')} "
            f"({entry.get('source', '')}) — {words}")
    return lines


def _composition_date_line(result: Any) -> str:
    """Before → after when the composition contains a change condition."""
    dates = getattr(result, "source_dates", None) or {}
    before, after = dates.get("before"), dates.get("after")
    if before or after:
        return f"Dates: {before or '—'} → {after or '—'}"
    if dates.get("scene_date"):
        return f"Scene date: {dates['scene_date']}"
    return ""


def _render_threshold_opt_in(entry: Dict[str, Any], context: Any) -> None:
    """The only way a convention threshold is ever used: the user asks for it.

    Nothing here is automatic. Each button names the convention and repeats
    that it is a display/query convention, not a scientific classification.
    """
    from core.multi_condition import (load_threshold_config,
                                      parse_composed_query)

    query_text = str(entry.get("query") or "")
    try:
        composed = parse_composed_query(query_text)
        wanted = {c.threshold.index for c in composed.missing_thresholds
                  if c.threshold is not None}
    except Exception:
        return
    if not wanted:
        return

    conventions = (load_threshold_config().get("conventions") or {})
    with st.expander("Use a labelled convention instead (opt-in)",
                     expanded=False):
        st.caption(
            "The system will not invent a threshold. If you want one of the "
            "labelled conventions below applied to THIS question only, choose "
            "it here. It will be recorded as a convention, never as a "
            "scientific classification.")
        for key, spec in conventions.items():
            if str(spec.get("index")) not in wanted:
                continue
            label = (f"{spec.get('label', key)} — "
                     f"{spec.get('operator', '>')} {spec.get('value')}")
            if st.button(label, key=f"conv_{entry.get('_id', 0)}_{key}",
                         help=str(spec.get("note", ""))):
                execution = route(query_text, context, convention=key)
                history = st.session_state.get("chat_history", [])
                if history and history[-1] is entry:
                    new_entry = execution.to_dict()
                    new_entry["result"] = execution.result
                    # Carry the entry's identity across the re-route. Without it
                    # the entry loses its id, and two entries then claim the
                    # same widget keys (`evidence_0_layers_0`), which Streamlit
                    # rejects -- a crash in the evidence panel for a perfectly
                    # ordinary question. Nothing about the result changes.
                    new_entry["_id"] = entry.get("_id", 0)
                    if execution.ok and execution.result is not None:
                        st.session_state["last_composition"] = execution.result
                    history[-1] = new_entry
                    st.session_state["chat_history"] = history
                st.rerun()
            st.caption(spec.get("note", ""))



@st.cache_data(max_entries=3, show_spinner="Reprojecting imagery for the map…")
def web_bands_from_bytes(file_bytes, bands, max_pixels, stretch_bounds, resampling_name="average"):
    from rasterio.enums import Resampling

    with open_upload(file_bytes) as ds:
        web = reproject_bands(ds, list(bands), dst_crs=WEB_MERCATOR,
                              resampling=Resampling[resampling_name], max_pixels=max_pixels)
    rgba = web_rgba_from_bands(web, stretch_bounds)
    return rgba, web.to_dict()


@st.cache_data(max_entries=3, show_spinner="Reprojecting NDVI for the map…")
def web_ndvi_from_arrays(ndvi_arr, ndvi_mask, transform_tuple, crs_wkt, max_pixels, vmin, vmax,
                         colormap="RdYlGn"):
    """Reproject an in-memory NDVI result (the native array stays as it is).

    Phase 11: `colormap` was added so the same code serves any continuous index
    (NDVI keeps RdYlGn; NDWI uses a blue diverging ramp). Existing call sites
    are unchanged because the default reproduces the old behaviour.
    """
    from rasterio.crs import CRS

    web = reproject_array(
        np.where(np.asarray(ndvi_mask), np.asarray(ndvi_arr), np.nan).astype(np.float32),
        src_transform=Affine(*transform_tuple),
        src_crs=CRS.from_user_input(crs_wkt),
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.average,
        max_pixels=max_pixels,
    )
    rgba = web_rgba_from_values(web, vmin, vmax, colormap=colormap)
    return rgba, web.to_dict()


@st.cache_data(max_entries=3, show_spinner="Reprojecting suitability classes…")
def web_suitability_from_codes(codes, transform_tuple, crs_wkt, max_pixels):
    """Display copy of the suitability raster.

    CATEGORICAL -> nearest neighbour. The analysis ran on the common analysis
    grid in a metric CRS; this web-mercator copy exists only so Leaflet can
    draw it, and is never used for any number shown to the user.
    """
    from rasterio.crs import CRS

    web = reproject_array(
        np.asarray(codes).astype("float32"),
        src_transform=Affine(*transform_tuple),
        src_crs=CRS.from_user_input(crs_wkt),
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.nearest,
        max_pixels=max_pixels,
    )
    arr = np.asarray(web.array)
    # NaN means "no cell was measured here"; give it the 0 code explicitly so
    # the int cast is exact and no warning is produced.
    codes = np.where(np.isfinite(arr), np.rint(arr), 0).astype("int16")
    rgba = suitability_rgba(codes)
    rgba[~np.asarray(web.mask)] = (0, 0, 0, 0)      # nowhere we did not measure
    return rgba, web.to_dict()


@st.cache_data(max_entries=3, show_spinner="Reprojecting spatial-query result…")
def web_spatial_from_mask(state, inside, transform_tuple, crs_wkt, max_pixels):
    """Display copy of the three-valued spatial-query mask.

    CATEGORICAL -> nearest neighbour, exactly like the Phase 8 suitability
    layer. Cells OUTSIDE the ROI are made fully transparent: they were never
    analysed, so they must not be painted as "insufficient data".
    """
    from rasterio.crs import CRS

    src_crs = CRS.from_user_input(crs_wkt)
    web = reproject_array(
        np.asarray(state).astype("float32"),
        src_transform=Affine(*transform_tuple),
        src_crs=src_crs,
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.nearest,
        max_pixels=max_pixels,
    )
    web_inside = reproject_array(
        np.asarray(inside).astype("float32"),
        src_transform=Affine(*transform_tuple),
        src_crs=src_crs,
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.nearest,
        max_pixels=max_pixels,
    )
    arr = np.asarray(web.array)
    codes = np.where(np.isfinite(arr), np.rint(arr), 0).astype("int16")
    rgba = spatial_rgba(codes)
    valid = np.asarray(web.mask) & (np.nan_to_num(np.asarray(web_inside.array),
                                                  nan=0.0) > 0.5)
    rgba[~valid] = (0, 0, 0, 0)
    return rgba, web.to_dict()


@st.cache_data(max_entries=3, show_spinner="Reprojecting NDVI change…")
def web_delta_from_array(values, inside, transform_tuple, crs_wkt, max_pixels,
                         symmetric_range):
    """Display copy of the continuous dNDVI raster.

    Continuous -> average resampling, like every other continuous layer. The
    colour ramp is DIVERGING AND SYMMETRIC about zero, so zero change is the
    neutral middle colour: an asymmetric ramp would make "no change" look like
    a change. Cells outside the selection are transparent -- they were never
    compared.
    """
    from rasterio.crs import CRS

    arr = np.asarray(values).astype("float32")
    web = reproject_array(
        arr,
        src_transform=Affine(*transform_tuple),
        src_crs=CRS.from_user_input(crs_wkt),
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.average,
        max_pixels=max_pixels,
    )
    web_inside = reproject_array(
        np.asarray(inside).astype("float32"),
        src_transform=Affine(*transform_tuple),
        src_crs=CRS.from_user_input(crs_wkt),
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.nearest,
        max_pixels=max_pixels,
    )
    span = max(float(symmetric_range), 0.05)
    rgba = web_rgba_from_values(web, -span, span, colormap="RdYlGn")
    keep = np.asarray(web.mask) & (np.nan_to_num(np.asarray(web_inside.array),
                                                 nan=0.0) > 0.5)
    rgba[~keep] = (0, 0, 0, 0)
    return rgba, web.to_dict()


@st.cache_data(max_entries=3, show_spinner="Reprojecting change classes…")
def web_change_from_codes(codes, inside, transform_tuple, crs_wkt, max_pixels):
    """Display copy of the change-class raster.

    CATEGORICAL -> nearest neighbour, exactly like the Phase 8 and Phase 9
    layers. Cells outside the selection are transparent rather than painted as
    "insufficient data": they were not measured, which is a different statement
    from "measured and unusable".
    """
    from rasterio.crs import CRS

    web = reproject_array(
        np.asarray(codes).astype("float32"),
        src_transform=Affine(*transform_tuple),
        src_crs=CRS.from_user_input(crs_wkt),
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.nearest,
        max_pixels=max_pixels,
    )
    web_inside = reproject_array(
        np.asarray(inside).astype("float32"),
        src_transform=Affine(*transform_tuple),
        src_crs=CRS.from_user_input(crs_wkt),
        dst_crs=WEB_MERCATOR,
        resampling=Resampling.nearest,
        max_pixels=max_pixels,
    )
    arr = np.asarray(web.array)
    classed = np.where(np.isfinite(arr), np.rint(arr), 0).astype("int16")
    rgba = ndvi_change_rgba(classed)
    keep = np.asarray(web.mask) & (np.nan_to_num(np.asarray(web_inside.array),
                                                 nan=0.0) > 0.5)
    rgba[~keep] = (0, 0, 0, 0)
    return rgba, web.to_dict()


@st.cache_data(show_spinner="Computing footprint…")
def footprint_from_path(path, segments: int = 32):
    """Densified WGS84 footprint of a raster on disk, or None if it has no CRS."""
    with open_dataset(path) as ds:
        crs_str = str(ds.crs) if ds.crs is not None else None
        if crs_str is None:
            return None
        poly = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=segments)
        bounds = grid_bounds_wgs84(ds.transform, ds.crs, ds.width, ds.height)
        centre = pixel_to_lonlat(ds.transform, ds.crs, ds.width / 2.0, ds.height / 2.0)
    return {
        "feature": footprint_feature(poly, {"crs": crs_str, "scene": Path(path).name}),
        "bounds": list(bounds),
        "centre": list(centre),
        "vertices": len(poly.exterior.coords),
        "crs": crs_str,
    }


@st.cache_data(show_spinner="Computing footprint…")
def footprint_from_bytes(file_bytes, segments: int = 32):
    with open_upload(file_bytes) as ds:
        crs_str = str(ds.crs) if ds.crs is not None else None
        if crs_str is None:
            return None
        poly = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=segments)
        bounds = grid_bounds_wgs84(ds.transform, ds.crs, ds.width, ds.height)
        centre = pixel_to_lonlat(ds.transform, ds.crs, ds.width / 2.0, ds.height / 2.0)
    return {
        "feature": footprint_feature(poly, {"crs": crs_str, "scene": "uploaded GeoTIFF"}),
        "bounds": list(bounds),
        "centre": list(centre),
        "vertices": len(poly.exterior.coords),
        "crs": crs_str,
    }


@st.cache_data(max_entries=2, show_spinner="Computing NDVI…")
def ndvi_from_bytes(file_bytes, red_i, nir_i, scale, offset, profile, source, is_refl):
    with open_upload(file_bytes) as ds:
        spec = ReflectanceSpec(scale=scale, offset=offset, profile=profile,
                               source=source, is_reflectance=is_refl)
        result, _used, _rep = ndvi_from_dataset(ds, red_i, nir_i, reflectance=spec,
                                                auto_detect_reflectance=False)
        return result.array, result.mask, result.to_dict()


# --------------------------------------------------------------------------- #
# sidebar: data source + render controls
# --------------------------------------------------------------------------- #
st.sidebar.title("Data source")
mode = st.sidebar.radio(
    "Choose input",
    ("Bundled sample", "Upload GeoTIFF"),
    help="SatQuery reads local GeoTIFF/COG files. Searching a remote catalogue "
         "from inside the app is not part of this release.",
)

info: dict | None = None
prov: dict | None = None
source_kind = "none"

if mode == "Bundled sample":
    names = list_samples()
    if not names:
        st.sidebar.error("No sample GeoTIFFs are bundled with this installation.")
    else:
        # Put the real multispectral Sentinel-2 sample first: it is the only one
        # with a genuine NIR band, which every index needs.
        ordered = [n for n in names if n.startswith("s2_")] + [n for n in names if not n.startswith("s2_")]
        chosen = st.sidebar.selectbox("Sample file", ordered, index=0)
        pv = sample_provenance(chosen)
        prov = pv.to_dict() if pv else None
        if prov:
            st.sidebar.caption(
                ("Real satellite data" if prov["is_real_satellite_data"] else "Synthetic fixture")
                + (" · reflectance-valid" if prov["bands_are_physically_valid"] else " · NOT index-valid")
            )
        try:
            path = sample_path(chosen)
            info = metadata_from_path(str(path), chosen)
            source_kind = "path"
            st.sidebar.caption(
                f"Source: **{_source_title(chosen, prov)}** · bundled sample"
            )
        except RasterioIOError as exc:
            st.error(f"Could not open sample `{chosen}`: {exc}")
else:
    uploaded = st.sidebar.file_uploader(
        "Upload a GeoTIFF",
        type=["tif", "tiff", "geotiff"],
        help="Files are read in memory. Very large scenes may exceed the upload "
             "limit configured for this deployment.",
    )
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            info = metadata_from_bytes(raw, uploaded.name)
            source_kind = "bytes"
            st.session_state["__upload_bytes__"] = raw
            st.session_state["__upload_name__"] = uploaded.name
            st.sidebar.caption(
                f"Source: **{Path(uploaded.name).stem}** · "
                f"{len(raw) / 1e6:.1f} MB · held in memory for this session"
            )
        except RasterioIOError as exc:
            st.error(
                "**That file could not be read as a GeoTIFF.**  \n"
                "SatQuery reads GeoTIFF/COG through GDAL. JPEG2000 (.jp2, common "
                "in Sentinel-2 SAFE) needs a GDAL build with the JP2OpenJPEG "
                "driver."
            )
    else:
        st.sidebar.info(
            "No raster loaded. Choose a bundled sample, or upload a GeoTIFF/COG "
            "to begin."
        )

st.sidebar.divider()
st.sidebar.subheader("Render controls")
low_pct = st.sidebar.slider("Low percentile (black point)", 0.0, 10.0, 2.0, 0.5)
high_pct = st.sidebar.slider("High percentile (white point)", 90.0, 100.0, 98.0, 0.5)
stretch_mode = st.sidebar.radio(
    "Stretch mode",
    ("per_band", "joint"),
    format_func=lambda m: "per-band (natural look)" if m == "per_band" else "joint (comparable brightness)",
    help="Per-band stretches each channel independently. Joint uses one range for all "
         "three channels so two composites can be compared directly.",
)
max_pixels = st.sidebar.select_slider(
    "Render resolution (max pixels)",
    options=[250_000, 600_000, DEFAULT_MAX_PIXELS, 4_000_000, 12_000_000],
    value=DEFAULT_MAX_PIXELS,
    format_func=lambda v: f"{v:,} px",
    help="Display-only decimation. Analysis always happens at native resolution.",
)
st.sidebar.caption(
    "These controls change **display only**. They never alter the data and are never "
    "fed into any computation."
)

render_roadmap()


# --------------------------------------------------------------------------- #
# main panel
# --------------------------------------------------------------------------- #
if info is None:
    st.info("Select a bundled sample or upload a GeoTIFF to begin.", icon=None)
    st.stop()

st.markdown(
    _section("Satellite imagery", number="1",
             note="True-colour and false-colour composites of the loaded raster"),
    unsafe_allow_html=True,
)
st.caption(
    f"Scene: **{_source_title(str(info.get('source_label') or ''), prov)}**"
)
render_metrics(info)

# ---------------------------- band inspection ------------------------------ #
from core.models import RasterInfo, BandInfo, SpatialInfo  # noqa: E402  (rebuild the typed object)


def rebuild_raster_info(d: dict) -> RasterInfo:
    """Rebuild the typed object from its dict form (UI speaks dicts, core speaks dataclasses)."""
    sp = d["spatial"]
    return RasterInfo(
        source_label=d["source_label"], source_path=d["source_path"], file_size_bytes=d["file_size_bytes"],
        driver=d["driver"], width=d["width"], height=d["height"], count=d["count"],
        dtypes=tuple(d["dtypes"]),
        bands=tuple(
            BandInfo(index=b["index"], name=b["name"], dtype=b["dtype"], nodata=b["nodata"],
                     block_shape=tuple(b["block_shape"]), color_interp=b.get("color_interp"))
            for b in d["bands"]
        ),
        spatial=SpatialInfo(
            has_crs=sp["has_crs"], crs_epsg=sp["crs_epsg"], crs_name=sp["crs_name"], crs_wkt=sp.get("crs_wkt"),
            is_geographic=sp["is_geographic"], is_projected=sp["is_projected"], linear_units=sp["linear_units"],
            transform=tuple(sp["transform"]), pixel_size_x=sp["pixel_size_x"], pixel_size_y=sp["pixel_size_y"],
            is_north_up=sp["is_north_up"], is_rotated=sp["is_rotated"],
            bounds_native=sp["bounds_native"], bounds_wgs84=sp["bounds_wgs84"],
            footprint_wgs84=sp["footprint_wgs84"], approx_area_km2=sp["approx_area_km2"],
        ),
        nodata_per_band=tuple(d["nodata_per_band"]), tiled=d["tiled"],
        overview_levels=tuple(d["overview_levels"]), block_shape=tuple(d["block_shape"]),
        compress=d["compress"], interleave=d["interleave"], looks_like_cog=d["looks_like_cog"],
        tags=d["tags"], warnings=tuple(d["warnings"]), estimated_full_read_mb=d["estimated_full_read_mb"],
    )


typed_info = rebuild_raster_info(info)
guess = guess_band_roles(typed_info).to_dict()

# Dataset identity, used by the NDVI figure, the map legend and the provenance table.
_tags = info.get("tags", {}) or {}
prov_extra = {
    "dataset": _tags.get("stac_item_id") or info["source_label"],
    "datetime": _tags.get("datetime") or _tags.get("TIFFTAG_DATETIME") or "unknown",
    "sensor": _tags.get("platform") or "unknown",
    "mgrs_tile": _tags.get("mgrs_tile") or "",
    "cloud_cover": _tags.get("eo_cloud_cover") or "",
}

stats_rows = (
    band_stats_path(info["source_path"], 1_000_000)
    if source_kind == "path" and info.get("source_path")
    else band_stats_bytes(st.session_state.get("__upload_bytes__", b""), 1_000_000)
)

render_band_inspector(stats_rows, guess["roles"])
render_guess_banner(guess)
st.caption(f"Summary: {describe_guess(guess_band_roles(typed_info))}")

# ---------------------------- band mapping --------------------------------- #
mapping = render_band_mapping(info, guess, key_prefix="bm")

red_i, green_i, blue_i = mapping["red"], mapping["green"], mapping["blue"]
nir_i = mapping["nir"]

# ---------------------------- composites ----------------------------------- #
st.divider()
st.subheader("Composites", divider="gray")

col_rgb, col_fcc = st.columns(2, gap="large")

with col_rgb:
    try:
        img, alpha, meta = (
            preview_path(info["source_path"], (red_i, green_i, blue_i), ("red", "green", "blue"),
                         max_pixels, low_pct, high_pct, stretch_mode)
            if source_kind == "path" and info.get("source_path")
            else preview_bytes(st.session_state.get("__upload_bytes__", b""), (red_i, green_i, blue_i),
                               ("red", "green", "blue"), max_pixels, low_pct, high_pct, stretch_mode)
        )
        render_composite(
            img, meta,
            title="True colour (RGB)",
            help_text=(
                "**What it is:** the red, green and blue bands shown through their own "
                "colours — approximately what a camera in space would see.\n\n"
                "**What it is for:** recognising context (fields, roads, water, built-up areas).\n\n"
                "**Limits:** brightness is stretched for display, so colours are NOT physical "
                "reflectance. Healthy vegetation looks green here, but you cannot measure "
                "vegetation condition from it."
            ),
            download_name="satquery_true_colour",
            png_bytes=encode_png(img, alpha),
        )
    except Exception as exc:
        st.error(f"Could not render true-colour composite: {exc}")

with col_fcc:
    if nir_i is None:
        st.markdown("**False colour (NIR–Red–Green)**")
        st.info(
            "A false-colour composite needs a near-infrared band, and none is mapped for "
            "this file. Assign one in **Band mapping** above (if the file actually has one) — "
            "note that a 3-band RGB file has no NIR band at all.",
            icon=None,
        )
    else:
        try:
            img2, alpha2, meta2 = (
                preview_path(info["source_path"], (nir_i, red_i, green_i), ("nir", "red", "green"),
                             max_pixels, low_pct, high_pct, stretch_mode)
                if source_kind == "path" and info.get("source_path")
                else preview_bytes(st.session_state.get("__upload_bytes__", b""), (nir_i, red_i, green_i),
                                   ("nir", "red", "green"), max_pixels, low_pct, high_pct, stretch_mode)
            )
            render_composite(
                img2, meta2,
                title="False colour (NIR–Red–Green)",
                help_text=(
                    "**What it is:** the near-infrared band is displayed in the RED channel, "
                    "red in GREEN, green in BLUE. It is called *false* colour because the colours "
                    "are assigned, not natural.\n\n"
                    "**How to read it:** healthy vegetation reflects NIR strongly, so vegetated "
                    "land appears bright red/magenta — the brighter the red, the more vigorous the "
                    "canopy. Water absorbs NIR and appears near-black. Bare soil and built-up areas "
                    "appear grey/cyan/brown.\n\n"
                    "**Limits:** this is a qualitative view. It suggests where vegetation is "
                    "vigorous; it does not measure it. NDVI does the measurement, and "
                    "threshold-free claims wait until it has been computed."
                ),
                download_name="satquery_false_colour",
                png_bytes=encode_png(img2, alpha2),
            )
        except Exception as exc:
            st.error(f"Could not render false-colour composite: {exc}")

# ---------------------------- histograms ----------------------------------- #
with st.expander("Band histograms and stretch rationale", expanded=False):
    bands_for_hist = [b for b in (blue_i, green_i, red_i, nir_i) if b]
    hists = (
        hist_path(info["source_path"], tuple(bands_for_hist), 48, 1_000_000)
        if source_kind == "path" and info.get("source_path")
        else hist_bytes(st.session_state.get("__upload_bytes__", b""), tuple(bands_for_hist), 48, 1_000_000)
    )
    render_histograms(hists)

# ---------------------------- Phase 3: NDVI ------------------------------- #
st.divider()
st.markdown(_section("Vegetation index (NDVI)", number="2",
                      note="Computed on the native raster, never on a preview"),
            unsafe_allow_html=True)

ndvi_arr = ndvi_mask = ndvi_meta = None       # filled in below when unlocked

# Requirement: the Red and NIR mapping must be CONFIRMED BY THE USER. Even a
# high-confidence automatic detection is only a suggestion until it is ticked.
with st.container(border=True):
    st.markdown("**Step 1 — confirm the Red and NIR bands**")
    if nir_i is None:
        st.error(
            "NDVI needs a near-infrared band and none is mapped for this file. Assign "
            "one in **Band mapping** above — if the file genuinely has no NIR band "
            "(a 3-band RGB file, for example) NDVI is impossible, and that is a data "
            "limitation, not something to approximate.",
            icon=None,
        )
    else:
        st.markdown(
            f"- Red = **band {red_i}**  \n"
            f"- NIR = **band {nir_i}**"
        )
    with st.expander("Evidence used for this mapping", expanded=True):
        st.caption(f"Automatic confidence: **{guess['confidence'].upper()}**"
                   + (f" · profile `{guess['profile']}`" if guess.get("profile") else ""))
        for e in guess.get("evidence", []):
            st.markdown(f"- {e}")
        for w in guess.get("warnings", []):
            st.markdown(f"- **Note:** {w}")
        st.caption(
            "Band meaning is metadata, not physics. Confirmation is required even when "
            "detection confidence is high — an undetected mapping error produces NDVI "
            "values that look entirely plausible and are wrong."
        )

    ndvi_confirmed = st.checkbox(
        f"I confirm Red = band {red_i} and NIR = band {nir_i} for this dataset",
        value=False,
        key="ndvi_confirm",
        disabled=(nir_i is None),
    )

if not (nir_i and ndvi_confirmed):
    st.info(
        "NDVI is locked until the Red and NIR mapping is confirmed above. This gate is "
        "deliberate and is not bypassed for high-confidence detections.",
        icon=None,
    )
else:
    # ---- Step 2: reflectance ------------------------------------------------ #
    st.markdown("**Step 2 — reflectance preprocessing**")
    spec_dict, report = (
        spec_from_path(info["source_path"], guess.get("profile"))
        if source_kind == "path" and info.get("source_path")
        else spec_from_bytes(st.session_state.get("__upload_bytes__", b""), guess.get("profile"))
    )
    render_reflectance_panel(spec_dict, report)

    override = st.checkbox("Override the detected scaling", value=False, key="refl_override")
    if override:
        c1, c2 = st.columns(2)
        scale_in = c1.number_input("Scale (DN → reflectance)", value=float(spec_dict["scale"]),
                                   format="%.8f", key="refl_scale")
        offset_in = c2.number_input("Offset", value=float(spec_dict["offset"]),
                                    format="%.6f", key="refl_offset")
        st.warning(
            "User-supplied scaling is **not validated** against pixel values. Use it only "
            "when you know the product's conversion (e.g. Landsat C2 L2: 0.0000275 / -0.2).",
            icon=None,
        )
        use_scale, use_offset, use_source = float(scale_in), float(offset_in), "user_supplied"
    else:
        use_scale, use_offset, use_source = spec_dict["scale"], spec_dict["offset"], spec_dict["source"]
    # Phase 11: remembered so the NDWI analysis uses the SAME reflectance
    # convention the user settled on here, instead of silently re-deriving one.
    st.session_state["refl_scale_used"] = use_scale
    st.session_state["refl_offset_used"] = use_offset
    st.session_state["refl_source_used"] = use_source

    # ---- Step 3: compute ---------------------------------------------------- #
    st.markdown("**Step 3 — NDVI**")
    try:
        ndvi_arr, ndvi_mask, ndvi_meta = (
            ndvi_from_path(info["source_path"], red_i, nir_i, use_scale, use_offset,
                           guess.get("profile"), use_source, spec_dict["is_reflectance"])
            if source_kind == "path" and info.get("source_path")
            else ndvi_from_bytes(st.session_state.get("__upload_bytes__", b""), red_i, nir_i,
                                 use_scale, use_offset, guess.get("profile"), use_source,
                                 spec_dict["is_reflectance"])
        )
        ndvi_meta = dict(ndvi_meta)
        ndvi_meta["provenance"] = {**(ndvi_meta.get("provenance") or {}), **prov_extra}

        render_ndvi_stats(ndvi_meta)

        # ---- Step 4: visualise --------------------------------------------- #
        if ndvi_meta.get("stats"):
            st.subheader("NDVI map", divider="gray")
            scale_mode = st.radio(
                "Colour scale",
                ("fixed -1 to +1", "adaptive p2–p98"),
                horizontal=True,
                key="ndvi_scale_mode",
                help="Fixed is comparable between scenes. Adaptive is display-only and "
                     "changes the meaning of a colour between datasets.",
            )
            if scale_mode == "fixed -1 to +1":
                vmin, vmax = -1.0, 1.0
            else:
                pct = ndvi_meta["stats"]["percentiles"]
                vmin, vmax = float(pct["p2"]), float(pct["p98"])
                if vmax <= vmin:
                    vmin, vmax = -1.0, 1.0

            disp_arr, disp_transform, factor = decimate_array(
                ndvi_arr, Affine(*ndvi_meta["transform"]), max_pixels=max_pixels
            )
            disp_mask = np.isfinite(disp_arr)

            import pandas as pd

            fig = build_ndvi_figure(
                disp_arr,
                disp_mask,
                title="NDVI — Normalised Difference Vegetation Index",
                subtitle=(
                    f"NOT an RGB or false-colour image · invalid pixels transparent · "
                    f"{prov_extra['dataset']} · {prov_extra['datetime']}"
                ),
                footer=(
                    f"{ndvi_meta['method']}  |  red = band {red_i}, nir = band {nir_i}  |  "
                    f"{ndvi_meta.get('reflectance', {}).get('label', 'raw DN')}  |  "
                    f"rendered 1/{factor} block average of a "
                    f"{ndvi_meta['native_shape'][0]}×{ndvi_meta['native_shape'][1]} native result  |  "
                    f"continuous values — no validated vegetation threshold applied"
                ),
                vmin=vmin,
                vmax=vmax,
            )
            render_ndvi_figure(fig, figure_to_png(fig), tuple(ndvi_meta["native_shape"]),
                               disp_arr.shape, factor)

            with st.expander("NDVI value distribution (measured)", expanded=False):
                hist = ndvi_histogram(ndvi_arr, ndvi_mask)
                if hist["counts"]:
                    df = pd.DataFrame({"NDVI": hist["centers"], "count": hist["counts"]}).set_index("NDVI")
                    st.bar_chart(df, width="stretch")
                    st.caption(
                        "Measured counts of valid NDVI pixels across the full theoretical "
                        "range (-1 to +1). Invalid pixels are excluded, not bucketed at zero."
                    )

            st.subheader("Illustrative classes (optional, NOT validated)", divider="gray")
            if st.checkbox("Show approximate class shares", value=False, key="ndvi_classes"):
                classification = classify_ndvi(ndvi_arr, ndvi_mask, stats=ndvi_meta["stats"])
                render_ndvi_classes(classification)

            st.subheader("Export", divider="gray")
            tif_bytes = analysis_to_geotiff_bytes(
                ndvi_arr, ndvi_mask, Affine(*ndvi_meta["transform"]),
                ndvi_meta["crs"], ndvi_meta
            )
            st.download_button(
                "Download NDVI raster (GeoTIFF, float32, georeferenced)",
                data=tif_bytes,
                file_name="satquery_ndvi.tif",
                mime="image/tiff",
                width="stretch",
            )
            st.caption(
                "The exported raster keeps the source CRS and transform, stores NaN for "
                "invalid pixels, and carries provenance tags. It is a real geospatial "
                "product, not a screenshot."
            )

        render_ndvi_provenance(ndvi_meta)
    except Exception as exc:
        st.error(f"NDVI computation failed: {exc}")

# ---------------------------- Phase 4: interactive map -------------------- #
st.divider()
st.markdown(_section("Map", number="3",
                      note="Globe navigation, imagery and analysis overlays"),
            unsafe_allow_html=True)

spatial = info["spatial"]
if not spatial["has_crs"]:
    st.error(
        "**This raster has no CRS, so it cannot be placed on a map.**  \n"
        "A web map needs to know which coordinate system the numbers are in. Without a "
        "CRS we would have to guess coordinates -- and a guessed position that looks "
        "plausible is worse than no map at all. (Try `RGB.byte.tif`, `byte.tif` or the "
        "Sentinel-2 sample, all of which are georeferenced.)",
        icon=None,
    )
else:
    _mode_cols = st.columns([1.1, 1.5, 1, 1, 1])
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    keys = list(BASEMAPS)

    # ---- map events ------------------------------------------------------ #
    # Handled here, BEFORE the controls are built: the map reports through its
    # key in session state, so a change made on the globe (imagery, mode, a
    # drawn area, a clicked point) is applied in the same run -- and a keyed
    # widget can be updated here, which it cannot once it exists.
    _event = st.session_state.get("satquery_map")
    if isinstance(_event, dict) and _event.get("id") is not None:
        if _event["id"] != st.session_state.get("map_event_id"):
            st.session_state["map_event_id"] = _event["id"]
            _kind = _event.get("type")
            if _kind == "click":
                _lat, _lon = float(_event["lat"]), float(_event["lon"])
                if -90.0 <= _lat <= 90.0 and -180.0 <= _lon <= 180.0:
                    st.session_state["map_centre"] = [_lat, _lon]
                    st.session_state["map_marker"] = [_lat, _lon]
                    st.session_state["map_marker_label"] = (
                        f"Selected point · {_lat:.5f}°, {_lon:.5f}°")
            elif _kind == "roi":
                # Same shape Leaflet Draw produces, so core.roi and every
                # analysis downstream are untouched.
                st.session_state["globe_roi_feature"] = _event.get("feature")
            elif _kind == "base" and _event.get("value") in keys:
                st.session_state["basemap_key"] = _event["value"]
            elif _kind == "layers":
                st.session_state["globe_layer_state"] = _event.get("value") or []
            elif _kind == "mode" and _event.get("value") == "detail":
                st.session_state["map_mode"] = "detail"
                st.session_state["map_view_mode"] = "Flat map"
            elif _kind == "unavailable":
                st.session_state["globe_unavailable"] = str(
                    _event.get("reason") or "")

    if st.session_state.get("basemap_key") not in keys:
        st.session_state["basemap_key"] = DEFAULT_BASEMAP
    base_tile = c1.selectbox(
        "Reference imagery",
        keys,
        format_func=lambda k: BASEMAPS[k].label,
        key="basemap_key",
    )

    # Where this scene sits on Earth, as [[south, west], [north, east]]. Used by
    # the globe (to draw the scene outline) and by "Zoom to scene".
    _b = spatial.get("bounds_wgs84") or ()
    scene_bounds = (
        [[float(_b[1]), float(_b[0])], [float(_b[3]), float(_b[2])]]
        if len(_b) == 4 else None
    )

    # Search runs on the SERVER (core.geocode -> Nominatim). If the browser did
    # it, the same network that blocks tiles could block the search too.
    s1, s2, s3, s4 = st.columns([2.2, 1, 1, 1])
    # No `value=`: the widget owns what the user typed. Feeding it back from
    # session state made Enter overwrite the query with whatever the previous
    # rerun had stored, which quietly turned a real search into "too short".
    query = s1.text_input(
        "Search for a place",
        placeholder="e.g. Kolhapur, India  ·  Nile Delta  ·  Cairo",
        key="search_input",
        on_change=search_from_widget,        # Enter works too, not just "Go"
        help="Any place on Earth: a city, landmark, address, region or country. "
             "Moving the map never changes your data -- only an analysis does.",
    )
    if s2.button("Go", key="search_go", use_container_width=True):
        run_place_search(query)
    if s3.button("Whole world", key="world_view", use_container_width=True):
        st.session_state["map_centre"] = list(WORLD_CENTRE)
        st.session_state["map_zoom"] = WORLD_ZOOM
        st.session_state["map_fit_scene"] = False
        st.session_state["search_results"] = []
        st.session_state["search_note"] = ""
        st.session_state["map_marker"] = None
        st.session_state["globe_centre"] = list(WORLD_CENTRE)
        st.session_state["globe_scale"] = 1.0
    if s4.button("Zoom to scene", key="zoom_scene", use_container_width=True):
        # `map_fit_scene` frames the scene on the flat detail map; the globe
        # needs an explicit centre and zoom, so it gets the same place.
        st.session_state["map_fit_scene"] = True
        st.session_state["map_centre"] = None
        if scene_bounds:
            centre_lat = (scene_bounds[0][0] + scene_bounds[1][0]) / 2.0
            centre_lon = (scene_bounds[0][1] + scene_bounds[1][1]) / 2.0
            span = max(scene_bounds[1][0] - scene_bounds[0][0],
                       scene_bounds[1][1] - scene_bounds[0][1], 1e-6)
            import math as _math

            zoom = int(_math.log2(360.0 / span)) - 1
            st.session_state["map_zoom"] = max(6, min(14, zoom))
            st.session_state["map_centre"] = [centre_lat, centre_lon]
            st.session_state["globe_centre"] = [centre_lat, centre_lon]
            st.session_state["map_marker"] = None



    if not proxy_mounted():
        # The operator needs the command, the user needs the fact: the fix is
        # logged for whoever runs the app, the interface stays professional.
        print("[satquery] tile proxy not mounted: start the app with "
              "`python serve.py` so /satquery-tiles exists.")
        st.info(
            "Reference imagery is unavailable in this session, so the globe is "
            "showing the raster overlay without a satellite or street backdrop. "
            "Analyses, measurements and exports are unaffected.",
            icon=None,
        )

    st.caption(
        "Navigate anywhere, search for a place, then draw an area to analyse it. "
        "Analysis runs only where georeferenced raster data exists; elsewhere "
        "SatQuery says so instead of inventing numbers."
    )
    show_rgb = c2.checkbox("True colour", value=True)
    show_fcc = c3.checkbox("False colour", value=nir_i is not None)
    show_ndvi = c4.checkbox("NDVI", value=ndvi_meta is not None)
    c5, c6, c7, c8 = st.columns([1.2, 1, 1, 1])
    overlay_px = c5.select_slider(
        "Overlay resolution", options=[300_000, 600_000, 1_000_000, 2_000_000],
        value=600_000, format_func=lambda v: f"{v:,} px",
        help="Display-only reprojection resolution. Analysis is unaffected.",
    )
    opacity = c6.slider("Opacity", 0.3, 1.0, 0.9, 0.05)
    show_foot = c7.checkbox("Show footprint", value=True)
    if c8.button(
        "Reset view",
        width="stretch",
        help="Recentres the map on the raster extent. This redraws the map, so the drawn "
             "outline is not preserved; the recorded selection is kept and flagged.",
    ):
        st.session_state["map_reset"] = st.session_state.get("map_reset", 0) + 1

    overlays: list[MapOverlay] = []
    web_meta_rgb = web_meta_ndvi = None
    src_crs_label = spatial["crs_name"] or "native CRS"

    try:
        if show_rgb:
            bands = (red_i, green_i, blue_i)
            if source_kind == "path" and info.get("source_path"):
                bounds_rgb = stretch_bounds_from_path(info["source_path"], bands, 1_000_000, low_pct, high_pct)
                rgba_rgb, web_meta_rgb = web_bands_from_path(info["source_path"], bands, overlay_px, bounds_rgb)
            else:
                bounds_rgb = stretch_bounds_from_bytes(st.session_state.get("__upload_bytes__", b""), bands, 1_000_000, low_pct, high_pct)
                rgba_rgb, web_meta_rgb = web_bands_from_bytes(st.session_state.get("__upload_bytes__", b""), bands, overlay_px, bounds_rgb)
            overlays.append(MapOverlay("True colour (RGB)", rgba_rgb, web_meta_rgb["leaflet_bounds"],
                                       opacity=opacity, show=True))

        if show_fcc and nir_i is not None:
            bands = (nir_i, red_i, green_i)
            if source_kind == "path" and info.get("source_path"):
                bounds_fcc = stretch_bounds_from_path(info["source_path"], bands, 1_000_000, low_pct, high_pct)
                rgba_fcc, web_meta_fcc = web_bands_from_path(info["source_path"], bands, overlay_px, bounds_fcc)
            else:
                bounds_fcc = stretch_bounds_from_bytes(st.session_state.get("__upload_bytes__", b""), bands, 1_000_000, low_pct, high_pct)
                rgba_fcc, web_meta_fcc = web_bands_from_bytes(st.session_state.get("__upload_bytes__", b""), bands, overlay_px, bounds_fcc)
            overlays.append(MapOverlay("False colour (NIR-R-G)", rgba_fcc, web_meta_fcc["leaflet_bounds"],
                                       opacity=opacity, show=False))

        ndvi_scale_mode = st.session_state.get("ndvi_scale_mode", "fixed -1 to +1")
        if show_ndvi and ndvi_arr is not None:
            if ndvi_scale_mode == "fixed -1 to +1":
                vmin, vmax = -1.0, 1.0
            else:
                pct = ndvi_meta["stats"]["percentiles"]
                vmin, vmax = float(pct["p2"]), float(pct["p98"])
            rgba_ndvi, web_meta_ndvi = web_ndvi_from_arrays(
                ndvi_arr, ndvi_mask, tuple(ndvi_meta["transform"]), str(ndvi_meta["crs"]),
                overlay_px, vmin, vmax,
            )
            overlays.append(MapOverlay("NDVI", rgba_ndvi, web_meta_ndvi["leaflet_bounds"],
                                       opacity=opacity, show=True))
        elif show_ndvi:
            st.info("NDVI is not computed yet — confirm the Red/NIR mapping in section 3 first.", icon=None)
    except MissingCRSError as exc:
        st.error(f"Cannot build map layers: {exc}")
    except Exception as exc:
        st.error(f"Map layer preparation failed: {exc}")

    fp = (
        footprint_from_path(info["source_path"], 32)
        if (source_kind == "path" and info.get("source_path"))
        else footprint_from_bytes(st.session_state.get("__upload_bytes__", b""), 32)
    )
    map_bounds = web_meta_ndvi["leaflet_bounds"] if web_meta_ndvi else (
        web_meta_rgb["leaflet_bounds"] if web_meta_rgb else None
    )

    if overlays and fp:
        # Phase 8: declared here (the overlay itself is built a few lines below,
        # and the legend is chosen after that) so nothing is used before it exists.
        suit = st.session_state.get("last_suitability")
        show_suit = False
        if web_meta_ndvi and show_ndvi:
            legend = ndvi_legend_html(
                dataset=prov_extra.get("dataset", info["source_label"]),
                when=prov_extra.get("datetime", ""),
                vmin=vmin, vmax=vmax, source_crs=src_crs_label, display_crs=WEB_MERCATOR,
            )
        else:
            legend = rgb_legend_html("Satellite imagery",
                                     "true colour / false colour",
                                     src_crs_label, WEB_MERCATOR)

        # ---- optional display overlay: the pixels actually analysed ------- #
        # OFF by default: switching it on changes the map HTML, which remounts
        # the component and clears the drawn layer (documented limitation).
        show_analysed = st.checkbox(
            "Show analysed pixels on the map",
            value=False,
            key="roi_show_analysed",
            help="Reprojected copy of the native pixels measured inside the "
                 "selection - display only, never used for the numbers below.",
        )
        # ---- Phase 8: the suitability overlay (its own palette + legend) --- #
        if suit is not None and getattr(suit, "scenarios", None):
            show_suit = st.checkbox(
                "Show cotton suitability screening on the map", value=True,
                key="roi_show_suitability",
                help="Experimental screening classes on the analysis grid. "
                     "Display copy only: the numbers were computed on the "
                     "common grid, never on this reprojected image.",
            )
            if show_suit:
                try:
                    primary = (suit.scenarios.get("rainfed")
                               or next(iter(suit.scenarios.values())))
                    rgba_s, web_meta_s = web_suitability_from_codes(
                        primary.suitability_raster,
                        tuple(primary.raster_transform)[:6],
                        str(primary.raster_crs), overlay_px)
                    overlays.append(
                        MapOverlay("Cotton suitability (experimental screening)",
                                   rgba_s, web_meta_s["leaflet_bounds"],
                                   opacity=0.85, show=True))
                except Exception as exc:  # a display layer must never break the app
                    st.caption(f"Suitability overlay unavailable: {exc}")
                    show_suit = False

        # ---- Phase 11: the NDWI overlay (+ its own legend) ---------------- #
        # ADDED, never substituted: the NDVI, true-colour, false-colour,
        # suitability and temporal layers below/above are untouched.
        ndwi_result = st.session_state.get("last_ndwi")
        show_ndwi_layer = False
        if (ndwi_result is not None
                and getattr(ndwi_result, "raster", None) is not None
                and getattr(ndwi_result, "transform", None)):
            show_ndwi_layer = st.checkbox(
                "Show NDWI layer on the map", value=True, key="roi_show_ndwi")
        if show_ndwi_layer and ndwi_result is not None:
            try:
                _arr = np.asarray(ndwi_result.raster)
                _mask = (np.asarray(ndwi_result.mask)
                         if getattr(ndwi_result, "mask", None) is not None
                         else np.isfinite(_arr))
                rgba_ndwi_i, web_meta_ndwi_i = web_ndvi_from_arrays(
                    _arr, _mask, tuple(ndwi_result.transform),
                    str(ndwi_result.crs), overlay_px, -1.0, 1.0,
                    colormap="RdYlBu",
                )
                overlays.append(MapOverlay(
                    "NDWI — Water Index", rgba_ndwi_i,
                    web_meta_ndwi_i["leaflet_bounds"],
                    opacity=opacity, show=True))
            except Exception as exc:  # a display layer must never break the app
                st.caption(f"NDWI overlay unavailable: {exc}")
                show_ndwi_layer = False

        # ---- Phase 10: the temporal overlays (+ their own legend) --------- #
        temporal_result = st.session_state.get("last_temporal")
        show_temporal = False
        if (temporal_result is not None
                and getattr(temporal_result, "class_raster", None) is not None):
            show_temporal = st.checkbox(
                "Show NDVI change layers on the map", value=True,
                key="roi_show_temporal",
                help="ΔNDVI and the change classes for the selected area. "
                     "Display copies only: the numbers were computed on the "
                     "analysis grid, never on this reprojected image.")
            if show_temporal:
                try:
                    _trs = tuple(temporal_result.transform)[:6]
                    _crs = str(temporal_result.crs)
                    _inside = np.asarray(temporal_result.roi_mask)
                    if _inside is None or _inside.size == 0:
                        _inside = np.isfinite(np.asarray(temporal_result.change_raster))
                    _delta = np.asarray(temporal_result.change_raster)
                    _span = float(np.nanmax(np.abs(_delta))) if np.isfinite(_delta).any() else 1.0
                    _span = float(min(max(_span, 0.05), 1.0))

                    # 1. before NDVI and after NDVI (the same ramp as section 3)
                    for _role, _arr, _date in (
                        ("before", temporal_result.before_raster,
                         temporal_result.before_date),
                        ("after", temporal_result.after_raster,
                         temporal_result.after_date),
                    ):
                        _a = np.asarray(_arr)
                        _rgba_t, _web_t = web_ndvi_from_arrays(
                            _a, np.isfinite(_a), _trs, _crs, overlay_px, -1.0, 1.0)
                        overlays.append(MapOverlay(
                            f"NDVI {_role} ({_date})", _rgba_t,
                            _web_t["leaflet_bounds"], opacity=0.8, show=False))

                    # 2. ΔNDVI -- diverging, symmetric about zero
                    _rgba_d, _web_d = web_delta_from_array(
                        _delta, _inside, _trs, _crs, overlay_px, _span)
                    overlays.append(MapOverlay(
                        "ΔNDVI (after − before)", _rgba_d,
                        _web_d["leaflet_bounds"], opacity=0.85, show=True))

                    # 3. the change classes
                    _rgba_c, _web_c = web_change_from_codes(
                        np.asarray(temporal_result.class_raster), _inside,
                        _trs, _crs, overlay_px)
                    overlays.append(MapOverlay(
                        "NDVI change class (increase / stable / decrease / "
                        "insufficient)", _rgba_c, _web_c["leaflet_bounds"],
                        opacity=0.9, show=False))
                except Exception as exc:  # a display layer must never break the app
                    st.caption(f"Change overlay unavailable: {exc}")
                    show_temporal = False

        # ---- Phase 9: the spatial-query overlay (+ its own legend) -------- #
        spatial_result = st.session_state.get("last_spatial_query")
        show_spatial = False
        if (spatial_result is not None
                and getattr(spatial_result, "result_mask", None) is not None):
            show_spatial = st.checkbox(
                "Show spatial-query result on the map", value=True,
                key="roi_show_spatial",
                help="Three-valued result: MATCH / NO MATCH / INSUFFICIENT DATA. "
                     "Display copy only -- the numbers were computed on the "
                     "analysis grid, never on this reprojected image.",
            )
            if show_spatial:
                try:
                    grid = spatial_result.grid or {}
                    mask = np.asarray(spatial_result.result_mask)
                    roi_array = getattr(spatial_result, "roi_mask", None)
                    if roi_array is None:
                        roi_array = np.ones(mask.shape, dtype=bool)
                    rgba_q, web_meta_q = web_spatial_from_mask(
                        mask, np.asarray(roi_array), tuple(grid["transform"]),
                        str(grid["crs"]), overlay_px)
                    overlays.append(
                        MapOverlay("Spatial query result (MATCH / NO MATCH / "
                                   "INSUFFICIENT DATA)",
                                   rgba_q, web_meta_q["leaflet_bounds"],
                                   opacity=0.9, show=True))
                except Exception as exc:  # a display layer must never break the app
                    st.caption(f"Spatial overlay unavailable: {exc}")
                    show_spatial = False

        # ---- Phase 12: the composed-condition overlay --------------------- #
        # ADDED, never replacing: the Phase 9 spatial layer, the Phase 10
        # change layer and the Phase 11 index layer all stay in the control.
        composed_result = st.session_state.get("last_composition")
        show_composed = False
        if (composed_result is not None
                and getattr(composed_result, "combined_mask", None) is not None):
            show_composed = st.checkbox(
                "Show composed-condition result on the map", value=True,
                key="roi_show_composed",
                help="MATCH / NO MATCH / UNKNOWN over the selected area. "
                     "Display copy only -- the counts were computed on the "
                     "analysis grid, never on this reprojected image.",
            )
            if show_composed:
                try:
                    grid_c = composed_result.grid or {}
                    mask_c = np.asarray(composed_result.combined_mask)
                    roi_c = getattr(composed_result, "roi_mask", None)
                    if roi_c is None:
                        roi_c = np.ones(mask_c.shape, dtype=bool)
                    rgba_c, web_meta_c = web_spatial_from_mask(
                        mask_c, np.asarray(roi_c), tuple(grid_c["transform"]),
                        str(grid_c["crs"]), overlay_px)
                    overlays.append(
                        MapOverlay("Composed conditions (MATCH / NO MATCH / "
                                   "UNKNOWN)",
                                   rgba_c, web_meta_c["leaflet_bounds"],
                                   opacity=0.9, show=True))
                except Exception as exc:  # a display layer must never break the app
                    st.caption(f"Composed-condition overlay unavailable: {exc}")
                    show_composed = False

        # ---- Phase 13: per-condition evidence layers (OPT-IN) ------------- #
        # Added after the composed layer, never instead of it, and only while
        # the user has switched them on for the answer currently on the map.
        for _owner in (composed_result, spatial_result):
            if _owner is None:
                continue
            _roi = getattr(_owner, "roi_mask", None)
            for _name, _cond, _mask in evidence_layer_choice(_owner):
                try:
                    _grid = _mask.grid
                    _state = np.asarray(_mask.state)
                    _inside = (np.ones(_state.shape, dtype=bool) if _roi is None
                               else np.asarray(_roi, dtype=bool))
                    if _inside.shape != _state.shape:
                        _inside = np.ones(_state.shape, dtype=bool)
                    _colour = evidence_layer_colour(_owner, _cond)
                    _rgba_e, _bounds_e = web_evidence_mask(
                        _state, _inside, tuple(_grid.transform),
                        str(_grid.crs), overlay_px, _colour)
                    overlays.append(MapOverlay(_name, _rgba_e, _bounds_e,
                                               opacity=0.6, show=True))
                except Exception as exc:  # a display layer must never break the app
                    st.caption(f"Evidence layer '{_name}' unavailable: {exc}")

        prev_roi = st.session_state.get("roi")
        if show_analysed and prev_roi is not None and prev_roi.usable and ndvi_arr is not None:
            try:
                _inside, _ = roi_pixel_mask(
                    prev_roi.geometry_raster_crs,
                    Affine(*spatial["transform"]),
                    info["height"],
                    info["width"],
                )
                _native = np.where(_inside & np.asarray(ndvi_mask), 1.0, np.nan).astype("float32")
                _web = reproject_array(
                    _native, Affine(*spatial["transform"]),
                    spatial.get("crs_wkt") or spatial["crs_epsg"],
                    dst_crs=WEB_MERCATOR, max_pixels=overlay_px,
                )
                _a = (np.nan_to_num(_web.array, nan=0.0) > 0.5) & _web.mask
                _rgba = np.zeros((*_a.shape, 4), "uint8")
                _rgba[_a] = (255, 0, 210, 190)
                overlays.append(
                    MapOverlay("Analysed pixels (ROI)", _rgba, _web.leaflet_bounds,
                               opacity=0.8, show=True)
                )
            except Exception as exc:  # never let a display-only layer break the app
                st.caption(f"Analysed-pixel overlay unavailable: {exc}")

        # Phase 11: the NDWI legend sits lowest in the analysis-legend order --
        # spatial, suitability and temporal (each of which changes what the map
        # MEANS more than one more continuous index) still win.
        if show_ndwi_layer and ndwi_result is not None:
            _ndwi_prov = st.session_state.get("last_ndwi_prov") or {}
            _nb = _ndwi_prov.get("bands") or {}
            legend = ndwi_legend_html(
                vmin=-1.0, vmax=1.0,
                dataset=str(info.get("source_label", "")),
                bands=" + ".join(
                    f"{_r} = band {_v.get('index')}"
                    f"{(' (' + _v.get('band_id') + ')') if _v.get('band_id') else ''}"
                    for _r, _v in _nb.items()),
                resolution=(_ndwi_prov.get("native_resolution_m") or [None])[0],
            )

        # Phase 9: the spatial result has its own legend too, and wins when shown
        if show_spatial and spatial_result is not None:
            _distance = None
            for _condition in getattr(spatial_result, "conditions", ()) or ():
                _params = dict(getattr(_condition, "parameters", {}) or {})
                if "distance_m" in _params:
                    _distance = float(_params["distance_m"])
            legend = spatial_legend_html(
                expression=getattr(spatial_result, "expression", "") or "",
                analysis_resolution=getattr(spatial_result, "analysis_resolution", None),
                native_resolutions=getattr(spatial_result, "source_resolutions", {}),
                distance_m=_distance)

        # Phase 8: the screening overlay gets its own legend -- never the NDVI ramp
        if (not show_spatial) and show_suit and suit is not None:
            _p = (suit.scenarios.get("rainfed")
                  or next(iter(suit.scenarios.values())))
            legend = suitability_legend_html(
                crop="cotton",
                analysis_resolution=(suit.grid.resolution if suit.grid else None),
                native_resolutions=getattr(_p, "native_resolutions", {}),
                scenario="rainfed (data-backed)")

        # Phase 12: the composed-condition legend wins when that layer is shown
        # -- it names the thresholds and the dates that produced THIS answer.
        if show_composed and composed_result is not None:
            legend = composition_legend_html(
                expression=getattr(composed_result, "expression", "") or "",
                analysis_resolution=getattr(composed_result,
                                            "analysis_resolution", None),
                thresholds=_composition_threshold_lines(composed_result),
                dates=_composition_date_line(composed_result),
                matched_cells=int(getattr(composed_result,
                                          "matched_cell_count", 0) or 0),
                unknown_cells=int(getattr(composed_result,
                                          "insufficient_cell_count", 0) or 0))

        # Phase 10: the temporal legend wins when the change layers are shown --
        # a diverging ramp and a class palette must never be read with the NDVI
        # legend, and "decrease" must be labelled as an index change.
        if show_temporal and temporal_result is not None:
            legend = ndvi_change_legend_html(
                before_date=getattr(temporal_result, "before_date", "") or "",
                after_date=getattr(temporal_result, "after_date", "") or "",
                before_scene=getattr(temporal_result, "before_scene", "") or "",
                after_scene=getattr(temporal_result, "after_scene", "") or "",
                threshold=(temporal_result.thresholds or {}).get("increase"),
                resolution=getattr(temporal_result, "resolution", None),
                resampled=bool((temporal_result.alignment or {}).get("resampled")),
                delta_min=getattr(temporal_result, "delta_min", None),
                delta_max=getattr(temporal_result, "delta_max", None),
            )

        # ------------------------------------------------------------------ #
        # ONE map surface.
        #
        # The globe is the primary map: navigation, imagery and the analysis
        # overlays all live on it. The flat, projected map is an explicit
        # *detail mode* the user opens for pixel-precise drawing and
        # inspection. Only one of the two is ever rendered, and the globe is
        # what opens by default.
        # ------------------------------------------------------------------ #
        # One map, chosen explicitly: the globe is the primary view, the flat
        # projected map is a detail mode for precise drawing and inspection.
        map_mode = st.session_state.get("map_mode", "globe")
        if map_mode not in ("globe", "detail"):
            map_mode = "globe"
        st.session_state.setdefault(
            "map_view_mode", "Globe" if map_mode == "globe" else "Flat map")
        _view = _mode_cols[0].radio(
            "Map view",
            ("Globe", "Flat map"),
            horizontal=True,
            key="map_view_mode",
            help="Globe: a 3D Earth for navigation and context. Flat map: the "
                 "projected raster view, for precise area drawing and pixel "
                 "inspection. Only one is shown at a time.",
        )
        _mode_cols[1].caption(
            "3D Earth — drag to rotate, scroll to zoom"
            if _view == "Globe" else
            "Projected view — draw or inspect cells precisely"
        )
        map_mode = "globe" if _view == "Globe" else "detail"
        st.session_state["map_mode"] = map_mode

        # Layer visibility chosen on the globe survives a rerun (display only:
        # it never reaches an analysis).
        saved_layers = st.session_state.get("globe_layer_state") or []
        if saved_layers:
            _show = {str(x.get("name")): bool(x.get("show")) for x in saved_layers}
            for _ov in overlays:
                if _ov.name in _show:
                    _ov.show = _show[_ov.name]

        roi_feature = st.session_state.get("globe_roi_feature")
        roi_geometry = roi_feature.get("geometry") if isinstance(roi_feature, dict) else None

        if map_mode == "globe":
            map_event = render_globe(
                overlays=overlays,
                centre=st.session_state.get("map_centre") or list(WORLD_CENTRE),
                zoom=st.session_state.get("map_zoom"),
                base=base_tile,
                attribution=BASEMAPS[base_tile].attribution,
                footprint=scene_bounds,
                roi=roi_geometry,
                marker=st.session_state.get("map_marker"),
                marker_label=st.session_state.get("map_marker_label", ""),
                height_px=MAP_HEIGHT,
                key="satquery_map",
            )
            # Events are handled at the top of this section (see "map events"),
            # before the controls are built, so they land in the same run.

            if st.session_state.get("globe_unavailable"):
                st.caption(
                    "The 3D globe engine could not be loaded here, so the map is "
                    "showing a 2D-rendered globe. Open the **2D map** from the map "
                    "controls for overlays and area drawing — the analysis is "
                    "identical in both."
                )

            # `None` = the map has not reported a drawing yet (keeps the stored
            # selection and flags it); `[]` = the user cleared it.
            if st.session_state.get("globe_roi_feature"):
                drawings = [st.session_state["globe_roi_feature"]]
            elif "globe_roi_feature" in st.session_state:
                drawings = []
            else:
                drawings = None
            out = None
        else:
            # ---- detail mode: the flat, projected analysis map ------------- #
            # Chosen deliberately by the user; the globe is not rendered.
            fit_scene = bool(st.session_state.get("map_fit_scene"))
            fmap = build_map(
                overlays=overlays,
                footprint=fp["feature"] if show_foot else None,
                bounds=map_bounds,
                tiles=base_tile,
                show_footprint=show_foot,
                extra_html=legend,
                fit=fit_scene,
                centre=st.session_state.get("map_centre"),
                zoom=st.session_state.get("map_zoom"),
                draw=True,
                marker=st.session_state.get("map_marker"),
                marker_label=st.session_state.get("map_marker_label", "Searched place"),
            )
            # While the map component is loading, its iframe is an empty
            # rectangle that shows the page through it. A neutral background
            # makes "still loading" look like "loading", not "broken".
            st.markdown(
                "<style>"
                "iframe[title='streamlit_folium.st_folium']{background:#e8eaed;}"
                "</style>",
                unsafe_allow_html=True,
            )
            if st.button("← Back to the globe", key="map_back_to_globe"):
                st.session_state["map_mode"] = "globe"
                st.session_state["map_view_mode"] = "Globe"
                st.rerun()
            out = st_folium(
                fmap,
                height=560,
                use_container_width=True,
                returned_objects=["last_clicked", "bounds", "zoom", "all_drawings"],
                key="satquery_map_"
                f"{st.session_state.get('map_reset', 0)}_"
                f"{st.session_state.get('roi_clear', 0)}",
            )
            drawings = (out or {}).get("all_drawings")

    results = st.session_state.get("search_results") or []
    if results:
        st.selectbox(
            "Matches",
            range(len(results)),
            format_func=lambda i: results[i].label(),
            key="search_pick",
            on_change=fly_from_widget,       # picking a match flies straight there
        )
        place = results[min(int(st.session_state.get("search_pick") or 0), len(results) - 1)]
        if st.button("Fly to this place", key="search_fly"):
            fly_to_place(int(st.session_state.get("search_pick") or 0))
            # The results sit below the map, so the camera is set after the map
            # has already been rendered: rerun to hand the globe its new target.
            st.rerun()
        st.caption(
            f"Showing **{place.name}** at `{place.lat:.5f}°, {place.lon:.5f}°` "
            f"(EPSG:4326). {GEOCODER_ATTRIBUTION}. Navigation only: no raster, "
            f"ROI or result is changed."
        )
    elif st.session_state.get("search_note"):
        st.caption(st.session_state["search_note"])
    elif st.session_state.get("search_query"):
        st.caption("No matches for that search (looked up on the server).")
    # ---- Phase 5: the drawn area -> a validated ROI -------------------- #
    # `all_drawings` is the authoritative list (see core.roi): None means
    # the map has not reported yet, [] means everything was deleted.
    # The flat map reports it directly; the globe already set `drawings`
    # above from its own drawing event -- do not overwrite it here.
    if map_mode == "detail":
        drawings = (out or {}).get("all_drawings")
    upload_bytes = st.session_state.get("__upload_bytes__", b"")
    raster_key = "|".join(
        [
            info.get("source_path") or f"upload:{st.session_state.get('__upload_name__', '')}:{len(upload_bytes)}",
            str(info["width"]),
            str(info["height"]),
            str(tuple(spatial["transform"])),
            str(spatial.get("crs_wkt") or spatial["crs_epsg"]),
        ]
    )
    try:
        roi = update_roi_state(
            st.session_state,
            drawings,
            native_footprint(Affine(*spatial["transform"]), info["width"], info["height"]),
            spatial.get("crs_wkt") or spatial["crs_epsg"],
            raster_key=raster_key,
        )
        st.subheader("Selected area (ROI)", divider="gray")
        render_roi_panel(roi, stale_map=is_map_stale(st.session_state))
        if st.button("Clear selection", disabled=roi is None, key="roi_clear_btn"):
            clear_roi_state(st.session_state)
            # Both surfaces keep their own drawing: drop the globe's copy as
            # well, and bump the flat map's key so its drawn layer really
            # disappears, then rerun so the change takes effect now.
            st.session_state["globe_roi_feature"] = None
            st.session_state["roi_clear"] = st.session_state.get("roi_clear", 0) + 1
            st.rerun()

        # ---- Phase 6: measure the selection against the NATIVE NDVI --- #
        if roi is not None and roi.usable:
            if ndvi_arr is None:
                st.info(
                    "NDVI is not computed yet. Confirm the Red / NIR mapping in "
                    "section 3, then the selected area can be measured.",
                    icon=None,
                )
            else:
                try:
                    nat_t = Affine(*ndvi_meta["transform"])
                    roi_stats = calculate_roi_ndvi_stats(
                        ndvi_arr,
                        roi.geometry_raster_crs,
                        nat_t,
                        ndvi_mask,
                        crs=ndvi_meta["crs"],
                        roi_crs=roi.raster_crs,
                    )
                    hist = None
                    if roi_stats.valid_pixels:
                        inside_mask, _ = roi_pixel_mask(
                            roi.geometry_raster_crs, nat_t, *ndvi_arr.shape
                        )
                        hist = ndvi_histogram(ndvi_arr, inside_mask & np.asarray(ndvi_mask))
                    render_roi_analysis(roi_stats.to_dict(), hist)
                except GeometryError as exc:
                    st.error(f"Could not analyse the selection: {exc}")
    except MissingCRSError as exc:
        st.error(f"Area selection unavailable: {exc}")

    # ---- coordinate inspection ---------------------------------------- #
    st.caption("Click the map to inspect a coordinate; the mouse position is shown bottom-right.")
    if out:
        if out.get("last_clicked"):
            lat = out["last_clicked"]["lat"]
            lon = out["last_clicked"]["lng"]
            st.markdown(f"**Clicked:** `{lat:.6f}° , {lon:.6f}°` (EPSG:4326)")
            row, col = lonlat_to_pixel(
                Affine(*spatial["transform"]),
                spatial.get("crs_wkt") or spatial["crs_epsg"],
                lon, lat,
            )
            if row is not None and 0 <= row < info["height"] and 0 <= col < info["width"]:
                msg = f"maps to native pixel (row {row}, col {col}) in {src_crs_label}"
                if ndvi_arr is not None and np.isfinite(ndvi_arr[row, col]):
                    msg += f" — NDVI there: **{float(ndvi_arr[row, col]):.4f}**"
                else:
                    msg += " — (no valid NDVI value at that pixel)"
                st.caption(msg)
            else:
                st.caption("that point lies outside the raster extent")
        if out.get("zoom") is not None:
            st.caption(f"Zoom level: {out['zoom']}")
    else:
        st.warning("No map layers available for this raster.", icon=None)

    # ---- reprojection transparency ---------------------------------------- #
    with st.expander("Display projection and reprojection", expanded=False):
        st.markdown(
            f"- **Source (analysis) CRS:** `{src_crs_label}` — grid "
            f"{info['width']} × {info['height']} px\n"
            f"- **Destination (display) CRS:** `{WEB_MERCATOR}` (Web Mercator)\n"
            f"- Bounds are computed from the *destination* grid corners, then converted "
            f"to EPSG:4326 for Leaflet."
        )
        for meta in (web_meta_rgb, web_meta_ndvi):
            if meta:
                st.json(meta, expanded=False)
        st.caption(
            "The reprojected rasters are **display copies**. The native NDVI array, its "
            "statistics and its transform are untouched; reprojection resamples pixels and "
            "would slightly change values if it were used for analysis."
        )

# ---------------------------- Phase 1 report ------------------------------- #
with st.expander("File and ingest details", expanded=False):
    left, right = st.columns([1, 1], gap="large")
    with left:
        render_spatial_table(info)
    with right:
        render_bands_table(info)
    render_warnings(list(info["warnings"]))

    st.markdown("**Pixel sanity check** (reads one small window — not a preview)")
    try:
        stats = (
            sanity_stats_from_path(info["source_path"], 1)
            if source_kind == "path" and info.get("source_path")
            else sanity_stats_from_bytes(st.session_state.get("__upload_bytes__", b""), 1)
        )
        if stats["all_nodata"]:
            st.warning("Every sampled pixel is nodata (expected for the all-nodata fixture).", icon=None)
        # Prefixed labels: these are raw-DN window samples, not analysis results.
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Window valid px", f"{stats['valid_pixels']:,}")
        s2.metric("Window min (DN)", "—" if stats["min"] is None else f"{stats['min']:.4g}")
        s3.metric("Window mean (DN)", "—" if stats["mean"] is None else f"{stats['mean']:.4g}")
        s4.metric("Window max (DN)", "—" if stats["max"] is None else f"{stats['max']:.4g}")
    except Exception as exc:
        st.error(f"Sanity check failed: {exc}")

# ----------------------- Phase 7: Ask SatQuery --------------------------- #
st.divider()
st.markdown(_section("Ask SatQuery", number="4",
                      note="Natural-language questions, evidence-backed answers"),
            unsafe_allow_html=True)

st.caption(
    "The router decides **what** is being asked; the analysis engine does the "
    "mathematics. Nothing below is computed from the text — the text only "
    "selects a registered analysis."
)

# The context the engines are allowed to use. Built here (the only place that
# knows about session state) and handed to the router as plain data.
current_roi = st.session_state.get("roi")
ndvi_ctx = None
if ndvi_arr is not None and ndvi_meta is not None:
    ndvi_ctx = NdviContext(
        array=ndvi_arr,
        mask=ndvi_mask,
        crs=ndvi_meta.get("crs"),
        transform=Affine(*ndvi_meta["transform"]),
        bands=(ndvi_meta.get("provenance") or {}).get("bands") or {},
        source_label=str(info.get("source_path") or st.session_state.get("__upload_name__", "")),
    )
# -------------------- Phase 11: the index input context ------------------- #
# An index analysis is ROI-first: the engine needs the SOURCE plus the band
# ROLES, and reads only the window the selection covers. The roles come from
# band metadata (descriptions / wavelengths), never from band position -- if
# they cannot be established, the engine refuses with UNSUPPORTED instead of
# guessing a mapping and returning a plausible, wrong index.
index_ctx = None
# `guess` is the DICT form of BandRoleGuess (built in section 2); read it as a
# dict, with getattr fallbacks so this block cannot silently produce an empty
# context for either representation.
_gval = (lambda k, default=None: guess.get(k, default)
         if isinstance(guess, dict) else getattr(guess, k, default))
_roles = dict(_gval("roles") or {})
_green, _nir = _roles.get("green"), _roles.get("nir")
if (source_kind == "path" and info.get("source_path")
        and "green" in _roles and "nir" in _roles):
    index_ctx = IndexContext(
        path=str(info["source_path"]),
        # ALL resolved roles, not just green+nir: Phase 12 composes NDVI as
        # well as NDWI, and each index picks the roles it needs. Extra roles
        # are ignored by every engine, so Phase 11 is unaffected.
        roles={str(k): int(v) for k, v in _roles.items()},
        scale=float(st.session_state.get("refl_scale_used",
                                         _gval("reflectance_scale") or 1.0)),
        offset=float(st.session_state.get("refl_offset_used",
                                          _gval("reflectance_offset") or 0.0)),
        is_reflectance=True,
        profile=_gval("profile"),
        source_label=str(info["source_path"]),
        reflectance_source=str(st.session_state.get("refl_source_used", "detected")),
        role_confidence=str(_gval("confidence", "none")),
        role_evidence=tuple(_gval("evidence") or ()),
    )

# -------------------- Phase 10: the two acquisitions ---------------------- #
# A change question is only meaningful for two KNOWN dates, and this app never
# chooses them on the user's behalf (a plausible-looking default date is a
# fabricated result). The pair is selected here, explicitly, and handed to the
# engine as data.
TEMPORAL_INTENTS = ("NDVI_CHANGE_ROI", "TEMPORAL_COMPARISON", "VEGETATION_CHANGE")

temporal_pair = None
temporal_scenes = discover_scenes(str(SAMPLE_DIR))
if len(temporal_scenes) >= 2:
    st.subheader("Temporal comparison — choose two acquisitions", divider="gray")
    st.caption(
        "Both dates must be selected. SatQuery never picks a date for you, and a "
        "comparison is computed only where BOTH acquisitions have a valid value."
    )
    _labels = [
        f"{sc.date_label} · {sc.source.get('platform') or 'unknown platform'} · "
        f"{sc.source.get('mgrs_tile') or sc.label}"
        for sc in temporal_scenes
    ]
    _b_col, _a_col = st.columns(2)
    _b_idx = _b_col.selectbox(
        "Before (earlier acquisition)", range(len(_labels)), index=0,
        format_func=lambda i: _labels[i], key="temporal_before_idx",
        help="The earlier scene. Its date and the later date define the interval.")
    _a_idx = _a_col.selectbox(
        "After (later acquisition)", range(len(_labels)), index=len(_labels) - 1,
        format_func=lambda i: _labels[i], key="temporal_after_idx",
        help="The later scene. NDVI change = after − before.")
    if _b_idx == _a_idx:
        st.caption("Pick two different acquisitions — a scene cannot be compared "
                   "with itself.")
    else:
        temporal_pair = ScenePair(before=temporal_scenes[_b_idx],
                                  after=temporal_scenes[_a_idx])
        st.caption(
            f"Comparing **{temporal_pair.before.date_label} → "
            f"{temporal_pair.after.date_label}** "
            f"({temporal_pair.before.label} → {temporal_pair.after.label})."
        )
else:
    st.caption(
        "Temporal comparison needs at least two dated acquisitions with both a "
        "red and a near-infrared band; fewer than two were found in the sample "
        "directory, so change questions will ask for dates instead of answering."
    )

analysis_context = AnalysisContext(
    roi=current_roi if (current_roi is not None and current_roi.usable) else None,
    ndvi=ndvi_ctx,
    ndvi_confirmed=ndvi_arr is not None,       # the Phase 3 gate has been passed
    raster_label=str(info.get("source_path") or "uploaded file"),
    temporal_pair=temporal_pair,
    index_context=index_ctx,          # Phase 11: source + resolved band roles
)

# Step 3: LLM Planner boundary with real provider when configured
# Try to create real provider from environment; fall back to MockLLMProvider
_planner_provider = create_provider_from_env() or MockLLMProvider()

# Step 5: Conversation state for multi-turn context
if "conversation_state" not in st.session_state:
    st.session_state["conversation_state"] = ConversationState()
conversation_state = st.session_state["conversation_state"]

_planner = LLMPlanner(
    _planner_provider,
    analysis_context,
    fallback_to_deterministic=True,
    conversation_state=conversation_state
)

_history = st.session_state.get("chat_history", [])
if not _history:
    # First screen: what this product does, and how to begin. The examples come
    # from the analysis registry, so nothing unsupported is ever advertised.
    render_welcome(suggestions(limit=6))
else:
    st.caption("Supported questions: " + " · ".join(f"_{q}_" for q in suggestions()))

query = render_ask_satquery()

# Step 7: Show conversation context indicator (if any)
# Determine what context was inherited for this query
inherited_context = None
if conversation_state and query:
    # Resolve references to see what would be inherited
    _, inherited_args = conversation_state.resolve_references(query)
    if inherited_args:
        inherited_context = inherited_args

# Render conversation context indicator
render_conversation_context(conversation_state, analysis_context, inherited_context)

# Step 9: Session Persistence UI
st.markdown("---")
col_save, col_load, col_clear = st.columns([1, 1, 1])

with col_save:
    if st.button("Save Session", key="save_session_btn", use_container_width=True):
        session = build_session_from_app_state(
            conversation_state=conversation_state,
            chat_history=st.session_state.get("chat_history", []),
            current_raster=info,
            current_roi=current_roi if (current_roi is not None and current_roi.usable) else None,
            current_arguments=entry.get("current_arguments") if 'entry' in locals() else None,
            evidence_package=evidence_from_entry(st.session_state.get("chat_history", [])[-1]).to_dict() if st.session_state.get("chat_history") else None,
        )
        save_session_to_file(session, f"satquery_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        st.success("Session saved!")

with col_load:
    uploaded_file = st.file_uploader("Load Session", type=["json"], key="load_session_uploader")
    if uploaded_file is not None:
        try:
            session = load_session_from_file(uploaded_file)
            is_valid, errors = session.validate()
            if not is_valid:
                st.error(f"Invalid session: {', '.join(errors)}")
            else:
                # Apply session to conversation state
                apply_session_to_conversation_state(session, conversation_state)
                # Note: The authoritative raster/ROI must be reloaded by the user
                st.session_state["_session_restored"] = True
                st.success("Session loaded! Raster/ROI must be reloaded for new analysis.")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to load session: {e}")

with col_clear:
    if st.button("Clear Session", key="clear_session_btn", use_container_width=True):
        conversation_state.clear()
        st.session_state["chat_history"] = []
        st.session_state["conversation_state"] = ConversationState()
        st.success("Session cleared!")
        st.rerun()

# Show session restore notice
if st.session_state.get("_session_restored"):
    st.info("Session context restored. The authoritative raster and ROI must be re-selected before running new analyses.")
    st.session_state["_session_restored"] = False

# --- Step 10: Session Management Helper Functions ---

def _render_session_comparison() -> None:
    """Render session comparison UI."""
    st.markdown("**Compare Two Sessions**")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Session A**")
        file_a = st.file_uploader("Session A file", type=["json"], key="compare_session_a")

    with col_b:
        st.markdown("**Session B**")
        file_b = st.file_uploader("Session B file", type=["json"], key="compare_session_b")

    if file_a and file_b:
        try:
            session_a = load_session_from_file(file_a)
            session_b = load_session_from_file(file_b)

            is_valid_a, errors_a = session_a.validate()
            is_valid_b, errors_b = session_b.validate()

            if not is_valid_a:
                st.error(f"Session A invalid: {', '.join(errors_a)}")
            if not is_valid_b:
                st.error(f"Session B invalid: {', '.join(errors_b)}")

            if is_valid_a and is_valid_b:
                diff = session_a.diff(session_b)
                summary = session_a.get_comparison_summary(session_b)

                st.markdown("**Comparison Summary**")

                # Show summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Added", summary["summary"]["added"])
                with col2:
                    st.metric("Removed", summary["summary"]["removed"])
                with col3:
                    st.metric("Changed", summary["summary"]["changed"])
                with col4:
                    st.metric("Unchanged", summary["summary"]["unchanged"])

                # Show detailed changes
                if summary["metadata_changed"]:
                    st.warning("Metadata changed")
                if summary["conversation_changed"]:
                    st.warning("Conversation changed")
                if summary["evidence_changed"]:
                    st.warning("Evidence changed")
                if summary["raster_changed"]:
                    st.warning("Raster context changed")
                if summary["roi_changed"]:
                    st.warning("ROI changed")
                if summary["title_changed"]:
                    st.warning("Title changed")
                if summary["tags_changed"]:
                    st.warning("Tags changed")

                # Show detailed diff
                with st.expander("Detailed Diff", expanded=False):
                    if diff["unchanged"]:
                        st.markdown("**Unchanged**")
                        for k, v in diff["unchanged"].items():
                            st.caption(f"  {k}: {v}")

                    if diff["added"]:
                        st.markdown("**Added**")
                        for k, v in diff["added"].items():
                            st.caption(f"  + {k}: {v}")

                    if diff["removed"]:
                        st.markdown("**Removed**")
                        for k, v in diff["removed"].items():
                            st.caption(f"  - {k}: {v}")

                    if diff["changed"]:
                        st.markdown("**Changed**")
                        for k, v in diff["changed"].items():
                            st.caption(f"  ~ {k}: {v['from']} → {v['to']}")

                    if diff["unavailable"]:
                        st.markdown("**Unavailable in Both**")
                        for k in diff["unavailable"].keys():
                            st.caption(f"  ? {k}")

        except Exception as e:
            st.error(f"Comparison failed: {e}")


def _render_session_forking(conversation_state: ConversationState) -> None:
    """Render session forking UI."""
    st.markdown("**Fork Current Session**")

    # Build current session
    session = build_session_from_app_state(
        conversation_state=conversation_state,
        chat_history=st.session_state.get("chat_history", []),
        current_raster=info,
        current_roi=current_roi if (current_roi is not None and current_roi.usable) else None,
    )

    new_title = st.text_input("Fork title (optional)", placeholder="e.g., 'NDVI Analysis - Variant B'")

    if st.button("Fork Session", use_container_width=True):
        forked = session.fork(new_title or "")

        # Save the fork
        save_session_to_file(forked, f"satquery_session_{forked.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        st.success(f"Session forked! New session ID: {forked.session_id}")
        st.info("Forked session saved. Note: Authoritative raster/ROI must be re-selected for new analysis.")

        # Offer to load the fork
        if st.button("Load Forked Session", use_container_width=True):
            apply_session_to_conversation_state(forked, conversation_state)
            st.session_state["_session_restored"] = True
            st.rerun()


def _render_session_metadata(conversation_state: ConversationState) -> None:
    """Render session metadata editing UI."""
    st.markdown("**Session Title, Description & Tags**")

    # Build current session
    session = build_session_from_app_state(
        conversation_state=conversation_state,
        chat_history=st.session_state.get("chat_history", []),
        current_raster=info,
        current_roi=current_roi if (current_roi is not None and current_roi.usable) else None,
    )

    # Title
    new_title = st.text_input("Title", value=session.metadata.title or "", placeholder="e.g., 'Pune Flood Analysis'")

    # Description
    new_description = st.text_area("Description", value=session.metadata.description or "", placeholder="Describe this analysis session...")

    # Tags
    tags_input = st.text_input("Tags (comma-separated)", value=", ".join(session.metadata.tags) if session.metadata.tags else "", placeholder="e.g., flood, temporal, sentinel-2")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Update Metadata", use_container_width=True):
            session.update_metadata(
                title=new_title or None,
                description=new_description or None,
                tags=[t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else None,
            )
            # Save updated session
            save_session_to_file(session, f"satquery_session_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            st.success("Metadata updated and session saved!")

    with col2:
        if st.button("Add Annotation", use_container_width=True):
            annotation_text = st.text_input("Annotation", key="new_annotation")
            if annotation_text:
                session.add_annotation(annotation_text, author="User")
                save_session_to_file(session, f"satquery_session_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                st.success("Annotation added!")
                st.rerun()

    # Show existing annotations
    if session.metadata.annotations:
        st.markdown("**Annotations**")
        for ann in session.metadata.annotations:
            st.caption(f"📝 {ann.text} — *{ann.author}* ({ann.timestamp})")

    # Show session info
    st.markdown("---")
    st.markdown("**Session Info**")
    st.caption(f"Session ID: {session.session_id}")
    st.caption(f"Created: {session.metadata.created_at}")
    st.caption(f"Updated: {session.metadata.updated_at}")
    if session.metadata.parent_session_id:
        st.caption(f"Forked from: {session.metadata.parent_session_id}")
    if session.metadata.forked_at:
        st.caption(f"Forked at: {session.metadata.forked_at}")


def _render_session_templates() -> None:
    """Render session templates UI."""
    st.markdown("**Start New Analysis from Template**")

    templates = Session.get_templates()

    # Display templates in a grid
    cols = st.columns(3)
    for i, (key, template) in enumerate(templates.items()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{template['title']}**")
                st.caption(template['description'])
                st.caption(f"Tags: {', '.join(template['tags'])}")
                st.caption(f"Expected: {template['expected_intent']}")
                st.caption(f"Required: {', '.join(template['required_inputs'])}")

                if st.button(f"Start {template['title']}", key=f"template_{key}", use_container_width=True):
                    # Create new session from template
                    new_session = Session.from_template(key)

                    # Apply to conversation state
                    apply_session_to_conversation_state(new_session, conversation_state)

                    st.success(f"Template '{template['title']}' loaded!")
                    st.info(f"Expected analysis: {template['expected_intent']}")
                    st.info(f"Required inputs: {', '.join(template['required_inputs'])}")
                    st.rerun()


# Step 10: Session Management (Comparison, Forking, Tagging, Templates)
st.markdown("---")
st.subheader("Session Management")

# Session comparison
with st.expander("Compare Sessions", expanded=False):
    _render_session_comparison()

# Session forking
with st.expander("Fork Session", expanded=False):
    _render_session_forking(conversation_state)

# Session tagging and metadata
with st.expander("Session Metadata", expanded=False):
    _render_session_metadata(conversation_state)

# Session templates
with st.expander("Start from Template", expanded=False):
    _render_session_templates()


# Step 11: Session Browser & Organization
st.markdown("---")
st.subheader("Session Browser & Organization")

def _render_session_browser(conversation_state: ConversationState) -> None:
    """Render the session browser with discovery, filtering, search, and management."""
    # Load all sessions
    all_entries = list_sessions()

    # Track currently loaded session
    current_session_id = None
    if conversation_state and hasattr(conversation_state, 'current_session_id'):
        current_session_id = conversation_state.current_session_id
    # Try to get from session metadata if available
    try:
        current_session = build_session_from_app_state(
            conversation_state=conversation_state,
            chat_history=st.session_state.get("chat_history", []),
        )
        if current_session.metadata.session_id:
            current_session_id = current_session.metadata.session_id
    except Exception:
        pass

    # --- Filter Controls ---
    st.markdown("**Filters**")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        # Collect all unique tags from sessions
        all_tags = sorted(set(tag for entry in all_entries for tag in entry.tags))
        selected_tags = st.multiselect("Tags (ALL must match)", all_tags, key="session_filter_tags")

    with filter_col2:
        selected_tags_any = st.multiselect("Tags (ANY can match)", all_tags, key="session_filter_tags_any")

    with filter_col3:
        # Collect all unique intents
        all_intents = sorted(set(entry.intent for entry in all_entries if entry.intent))
        selected_intent = st.selectbox("Intent", ["All"] + all_intents, key="session_filter_intent")

    with filter_col4:
        # Date range
        date_from = st.date_input("Updated after", value=None, key="session_filter_date_from")
        date_to = st.date_input("Updated before", value=None, key="session_filter_date_to")

    # Search box
    search_query = st.text_input("Search sessions", placeholder="Search title, description, tags, session ID...", key="session_search")

    # Build filter criteria
    filter_criteria = SessionFilter(
        tags=selected_tags if selected_tags else None,
        tag_any=selected_tags_any if selected_tags_any else None,
        intent=selected_intent if selected_intent != "All" else None,
        date_from=date_from.isoformat() if date_from else None,
        date_to=date_to.isoformat() if date_to else None,
    )

    # Apply filters
    filtered_entries = filter_sessions(all_entries, filter_criteria)

    # Apply search
    if search_query:
        filtered_entries = search_sessions(filtered_entries, search_query)

    # --- Session List ---
    st.markdown(f"**Sessions ({len(filtered_entries)} of {len(all_entries)})**")

    if not filtered_entries:
        st.info("No sessions match the current filters.")
    else:
        for entry in filtered_entries:
            is_current = entry.session_id == current_session_id
            prefix = "▶ " if is_current else "  "

            with st.container(border=True):
                col_main, col_actions = st.columns([4, 1])

                with col_main:
                    title_display = f"{prefix}**{entry.title or '(Untitled)'}**"
                    if is_current:
                        title_display += " `← CURRENT`"
                    st.markdown(title_display)

                    if entry.description:
                        st.caption(entry.description)

                    # Metadata row
                    meta_parts = []
                    if entry.tags:
                        meta_parts.append(f"Tags: {', '.join(entry.tags)}")
                    if entry.intent:
                        meta_parts.append(f"Intent: {entry.intent}")
                    meta_parts.append(f"Updated: {entry.updated_at[:19].replace('T', ' ')}")
                    if entry.parent_session_id:
                        meta_parts.append(f"Forked from: {entry.parent_session_id[:8]}")
                    st.caption(" · ".join(meta_parts))

                    # Status indicators
                    status_parts = []
                    if entry.has_conversation:
                        status_parts.append("💬 Conversation")
                    if entry.has_evidence:
                        status_parts.append("📊 Evidence")
                    if entry.has_raster:
                        status_parts.append("📍 Raster")
                    if entry.has_roi:
                        status_parts.append("🔲 ROI")
                    if entry.num_annotations > 0:
                        status_parts.append(f"📝 {entry.num_annotations} annotations")
                    if entry.num_chat_entries > 0:
                        status_parts.append(f"💭 {entry.num_chat_entries} chats")
                    if status_parts:
                        st.caption(" | ".join(status_parts))

                with col_actions:
                    # Load button
                    if st.button("Load", key=f"load_session_{entry.session_id}", use_container_width=True):
                        try:
                            session = load_session_from_file(entry.filepath)
                            is_valid, errors = session.validate()
                            if not is_valid:
                                st.error(f"Invalid session: {', '.join(errors)}")
                            else:
                                apply_session_to_conversation_state(session, conversation_state)
                                st.session_state["_session_restored"] = True
                                st.success("Session loaded! Raster/ROI must be reloaded for new analysis.")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load: {e}")

                    # Archive button
                    if st.button("Archive", key=f"archive_session_{entry.session_id}", use_container_width=True):
                        success, msg = archive_session(entry.session_id)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                    # Checkpoints button (shows count)
                    checkpoints = list_checkpoints(entry.session_id)
                    if checkpoints:
                        if st.button(f"Checkpoints ({len(checkpoints)})", key=f"checkpoints_session_{entry.session_id}", use_container_width=True):
                            st.session_state[f"show_checkpoints_{entry.session_id}"] = True
                            st.rerun()

                    # Show checkpoints if requested
                    if st.session_state.get(f"show_checkpoints_{entry.session_id}"):
                        with st.expander(f"Checkpoints for {entry.title}", expanded=True):
                            for cp in checkpoints:
                                cp_col1, cp_col2 = st.columns([3, 1])
                                with cp_col1:
                                    label_display = f"**{cp['label']}**" if cp['label'] else "*(no label)*"
                                    st.caption(f"{label_display} — {cp['created_at'][:19].replace('T', ' ')}")
                                with cp_col2:
                                    if st.button("Load", key=f"load_cp_{entry.session_id}_{cp['filename']}", use_container_width=True):
                                        checkpoint_session = load_checkpoint(entry.session_id, cp['filename'])
                                        if checkpoint_session:
                                            apply_session_to_conversation_state(checkpoint_session, conversation_state)
                                            st.session_state["_session_restored"] = True
                                            st.success("Checkpoint loaded!")
                                            st.session_state[f"show_checkpoints_{entry.session_id}"] = False
                                            st.rerun()
                                        else:
                                            st.error("Failed to load checkpoint")
                                    if st.button("Delete", key=f"del_cp_{entry.session_id}_{cp['filename']}", use_container_width=True):
                                        success, msg = delete_checkpoint(entry.session_id, cp['filename'])
                                        if success:
                                            st.success(msg)
                                            st.rerun()
                                        else:
                                            st.error(msg)
                            if st.button("Close", key=f"close_cp_{entry.session_id}", use_container_width=True):
                                st.session_state[f"show_checkpoints_{entry.session_id}"] = False
                                st.rerun()

                    # Create checkpoint button
                    if st.button("✓ Checkpoint", key=f"create_cp_{entry.session_id}", use_container_width=True):
                        # Need to load the session first to create checkpoint
                        try:
                            session = load_session_from_file(entry.filepath)
                            success, msg = create_checkpoint(session, label=f"manual_{datetime.now().strftime('%H%M%S')}")
                            if success:
                                st.success(f"Checkpoint created: {msg}")
                            else:
                                st.error(msg)
                        except Exception as e:
                            st.error(f"Failed to create checkpoint: {e}")

                    # Delete button (with confirmation)
                    if st.button("🗑 Delete", key=f"delete_session_{entry.session_id}", use_container_width=True):
                        st.session_state[f"confirm_delete_{entry.session_id}"] = True
                        st.rerun()

                    if st.session_state.get(f"confirm_delete_{entry.session_id}"):
                        st.warning("Delete permanently? This cannot be undone.")
                        del_col1, del_col2 = st.columns(2)
                        with del_col1:
                            if st.button("Yes, delete", key=f"confirm_yes_{entry.session_id}", use_container_width=True):
                                success, msg = delete_session(entry.session_id, confirm=True)
                                if success:
                                    st.success(msg)
                                    st.session_state[f"confirm_delete_{entry.session_id}"] = False
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with del_col2:
                            if st.button("Cancel", key=f"confirm_no_{entry.session_id}", use_container_width=True):
                                st.session_state[f"confirm_delete_{entry.session_id}"] = False
                                st.rerun()

    # --- Archived Sessions ---
    with st.expander("Archived Sessions", expanded=False):
        archived_entries = list_archived_sessions()
        if not archived_entries:
            st.caption("No archived sessions.")
        else:
            for entry in archived_entries:
                with st.container(border=True):
                    col_arch, col_restore = st.columns([4, 1])
                    with col_arch:
                        st.markdown(f"**{entry.title or '(Untitled)'}**")
                        st.caption(f"Archived: {entry.updated_at[:19].replace('T', ' ')} · ID: {entry.session_id}")
                    with col_restore:
                        if st.button("Restore", key=f"restore_{entry.session_id}", use_container_width=True):
                            success, msg = restore_archived_session(Path(entry.filepath).name)
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)


_render_session_browser(conversation_state)

if query is None and st.session_state.get("sq_pending_query"):
    # An example query was clicked: run it exactly as if it had been typed.
    query = st.session_state.pop("sq_pending_query")

if query:
    # Step 2: Execute via planner with deterministic fallback
    tool_result, used_fallback = execute_with_fallback(
        query, analysis_context, _planner, conversation_state
    )

    # Convert ToolResult to the format expected by the rest of the UI (similar to AnalysisExecution)
    # We need to reconstruct an execution-like object for compatibility
    from analyses.base import AnalysisExecution, Status, Intent
    from core.tools import intent_to_tool_name

    # Determine intent from tool_name
    intent = Intent.UNKNOWN
    if tool_result.tool_name != "clarification":
        intent = intent_to_tool_name(tool_result.tool_name) or Intent.UNKNOWN

    # Build a compatible execution object
    execution = AnalysisExecution(
        intent=intent,
        status=Status(tool_result.status) if tool_result.status in [s.value for s in Status] else Status.ERROR,
        query=query,
        normalized_query=query,
        confidence=1.0 if not used_fallback else 0.0,
        explanation="LLM planner" if not used_fallback else "Deterministic fallback",
        matched=(),
        result=tool_result.result,
        message=tool_result.message,
        warnings=tuple(tool_result.warnings),
        provenance=tool_result.evidence or {"engine": "planner" if not used_fallback else "fallback"},
    )

    # Handle clarification specially
    if tool_result.status == "NEEDS_CLARIFICATION":
        # Step 7: Use improved clarification rendering
        # Note: Clarification turns do NOT overwrite authoritative context
        # (ConversationState only updates on successful tool calls, not clarifications)
        render_clarification(tool_result)
        # Don't add to history, just show the clarification
        st.stop()

    entry = execution.to_dict()
    entry["result"] = execution.result          # keep the object for the details
    entry["_planner_fallback"] = used_fallback

    # The histogram is display evidence for the SAME pixels the engine measured;
    # it is derived from the native array + the same mask, never from the map.
    # Phase 8: the histogram belongs to the NDVI intent only -- a crop screening
    # result carries no valid_pixels and must not be asked for one.
    hist = None
    if (execution.ok
            and execution.intent.value == "NDVI_ROI_STATS"
            and ndvi_arr is not None
            and getattr(execution.result, "valid_pixels", 0)):
        try:
            inside_mask, _ = roi_pixel_mask(
                current_roi.geometry_raster_crs,
                Affine(*ndvi_meta["transform"]),
                *ndvi_arr.shape,
            )
            hist = ndvi_histogram(ndvi_arr, inside_mask & np.asarray(ndvi_mask))
        except Exception as exc:  # display-only evidence must never break the run
            st.caption(f"(histogram unavailable: {exc})")
    entry["histogram"] = hist
    entry["_id"] = len(st.session_state.get("chat_history", []))

    if execution.intent.value in TEMPORAL_INTENTS:
        # Phase 10: the map section draws the change layers from this. A refused
        # comparison produced no result, so any previous layer is cleared.
        st.session_state["last_temporal"] = execution.result if execution.ok else None
    if execution.intent.value == "NDWI_ROI_STATS":
        # Phase 11: the map section draws the NDWI layer from this. A refused
        # analysis produced no raster, so any previous layer is cleared.
        st.session_state["last_ndwi"] = execution.result if execution.ok else None
        st.session_state["last_ndwi_prov"] = execution.provenance if execution.ok else None
    if execution.intent.value == "CROP_SUITABILITY" and execution.ok:
        # kept so the MAP SECTION (rendered above this block) can draw the
        # screening overlay
        st.session_state["last_suitability"] = execution.result
    if execution.intent.value == "SPATIAL_QUERY":
        # Phase 9: the map section draws the result layer from this. An
        # unsupported query produced no mask, so any previous layer is cleared.
        st.session_state["last_spatial_query"] = (
            execution.result if execution.ok
            and getattr(execution.result, "result_mask", None) is not None else None)
    if execution.intent.value == "MULTI_CONDITION":
        # Phase 12: the map section draws the composed mask from this. A
        # refused composition produced no result, so any previous layer is
        # cleared rather than left on the map answering a different question.
        st.session_state["last_composition"] = (
            execution.result if execution.ok
            and getattr(execution.result, "combined_mask", None) is not None
            else None)

    history = st.session_state.get("chat_history", [])
    history.append(entry)
    st.session_state["chat_history"] = history[-5:]

    if (execution.intent.value == "SPATIAL_QUERY" and execution.ok
            and getattr(execution.result, "result_mask", None) is not None
            and st.session_state.get("_spatial_drawn") is not execution.result):
        # One guarded rerun so the answer and its map layer appear together.
        st.session_state["_spatial_drawn"] = execution.result
        st.rerun()

    if (execution.intent.value in TEMPORAL_INTENTS and execution.ok
            and st.session_state.get("_temporal_drawn") is not execution.result):
        # One guarded rerun so the answer and its map layers appear together.
        # It cannot loop: on the rerun the chat input is empty, so `query` is
        # None and this branch is never reached again.
        st.session_state["_temporal_drawn"] = execution.result
        st.rerun()

    if (execution.intent.value == "NDWI_ROI_STATS" and execution.ok
            and st.session_state.get("_ndwi_drawn") is not execution.result):
        # One guarded rerun so the answer and its map layer appear together.
        # It cannot loop: on the rerun the chat input is empty, so `query` is
        # None and this branch is never reached again.
        st.session_state["_ndwi_drawn"] = execution.result
        st.rerun()

    if (execution.intent.value == "MULTI_CONDITION" and execution.ok
            and st.session_state.get("_composed_drawn") is not execution.result):
        # One guarded rerun so the answer and its map layer appear together.
        # It cannot loop: on the rerun the chat input is empty, so `query` is
        # None and this branch is never reached again.
        st.session_state["_composed_drawn"] = execution.result
        st.rerun()

    if (execution.intent.value == "CROP_SUITABILITY" and execution.ok
            and st.session_state.get("_suit_drawn") is not execution.result):
        # One guarded rerun so the answer and its map layer appear together.
        # It cannot loop: on the rerun the chat input is empty, so `query` is
        # None and this branch is never reached again.
        st.session_state["_suit_drawn"] = execution.result
        st.rerun()

for entry in st.session_state.get("chat_history", []):
    render_answer(entry)
    if entry.get("status") == "NEEDS_THRESHOLD" and \
            entry.get("intent") == "MULTI_CONDITION":
        _render_threshold_opt_in(entry, analysis_context)
    if entry.get("status") == "UNSUPPORTED_CONDITION":
        render_unsupported_condition(entry)
    elif entry.get("status") == "OK" and entry.get("result") is not None:
        if entry.get("intent") == "SPATIAL_QUERY":
            render_spatial_query(entry["result"])
        elif entry.get("intent") == "MULTI_CONDITION":
            # Phase 12: conditions, threshold provenance, undecided cells and
            # the "geographic evidence, not causal attribution" boundary.
            render_multi_condition(entry["result"])
        elif entry.get("intent") == "NDWI_ROI_STATS":
            # Phase 11: counts, statistics and the mandatory "an index is not a
            # water body" caveat.
            render_ndwi_stats(entry["result"], entry.get("provenance"))
        elif entry.get("intent") in TEMPORAL_INTENTS:
            # Phase 10: the numbers AND the caveat that they do not identify a cause
            render_ndvi_change(entry["result"], entry.get("warnings"))
        elif entry.get("intent") == "CROP_SUITABILITY":
            render_crop_suitability(entry["result"])
        else:
            # The SAME panel used in section 4: the chat cannot drift from the
            # analysis it is describing, because it does not re-implement it.
            render_roi_analysis(entry["result"].to_dict(), entry.get("histogram"))
        # Phase 13: the evidence behind the answer, for EVERY intent. The
        # panels above are untouched; this is added below them and reads the
        # same result object they do.
        render_evidence(entry, key=f"evidence_{entry.get('_id', 0)}")

        # Step 8: Enhanced Evidence & Provenance UX
        # Get conversation state for context-aware features
        conv_state = st.session_state.get("conversation_state")

        # Evidence Explorer with filtering
        with st.expander("Explore Evidence", expanded=False):
            package = evidence_from_entry(entry)
            if package:
                render_evidence_explorer(package, key=f"explorer_{entry.get('_id', 0)}")

        # Provenance Timeline
        with st.expander("Provenance Timeline", expanded=False):
            package = evidence_from_entry(entry)
            if package:
                render_provenance_timeline(package, key=f"timeline_{entry.get('_id', 0)}")

        # Evidence Comparison (for multi-turn chains)
        with st.expander("Compare Evidence", expanded=False):
            package = evidence_from_entry(entry)
            if package:
                render_evidence_comparison(entry, conversation_state=conv_state,
                                           key=f"compare_{entry.get('_id', 0)}")

        # Export Reproducible Report
        with st.expander("Export Report", expanded=False):
            package = evidence_from_entry(entry)
            if package:
                export_evidence_report(entry, conversation_state=conv_state,
                                       key=f"export_{entry.get('_id', 0)}")

st.divider()
render_provenance(prov)
render_raw_metadata(info)

