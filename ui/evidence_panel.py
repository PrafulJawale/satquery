"""Phase 13 -- "Why this result?": evidence, provenance and export.

Everything here is a RENDERING of an `EvidencePackage`. No number is computed
in this module: the counts, dates, thresholds, provenance and limitations all
come from the frozen result objects that Phase 9-12 produced, through
`analyses.evidence`. That is the whole point -- the panel cannot disagree with
the engine, because it never re-implements it.

Layout rules that come straight from the brief:
  * the primary answer stays short;
  * detail lives in expanders;
  * the causal/evidence boundary is printed, never buried;
  * the individual evidence layers are OPT-IN and off by default, and the
    combined result layer is left exactly as it was.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from analyses.evidence import (UNAVAILABLE, condition_rows,
                               evidence_from_entry, grid_rows, source_rows)
from analyses.evidence_masks import evidence_masks

__all__ = ["render_evidence", "evidence_layer_choice",
           "evidence_layer_colour", "web_evidence_mask"]

#: Intents whose answer is a COMPOSED geographic condition. For these the
#: causal/evidence boundary is printed in the open, not inside an expander.
COMPOSED_INTENTS = ("MULTI_CONDITION", "SPATIAL_QUERY")

_EVIDENCE_COLOURS = {
    "spatial": (72, 133, 237),     # blue   -- a mapped land-cover class
    "spectral": (46, 160, 67),     # green  -- a spectral index threshold
    "temporal": (214, 138, 26),    # orange -- a change between two dates
    "statistics": (140, 92, 214),  # violet -- a measurement, not a filter
}


# =========================================================================== #
# the panel
# =========================================================================== #
def render_evidence(entry: Dict[str, Any], key: str = "evidence") -> None:
    """Render the evidence panel for one answered chat entry."""
    package = evidence_from_entry(entry)
    if package is None:
        return  # a refusal has no evidence to show, by construction

    explanation = package.explanation
    intent = str(package.intent or "").upper()
    with st.container():
        st.markdown("**Why this result?**")
        st.caption(
            "Every line below is taken from the result object the engine "
            "produced. Nothing here is re-computed or inferred.")

        # ---- 1. the answer, in one sentence -------------------------------- #
        st.markdown(f"**Answer** — {explanation['what_was_found']}")

        # ---- 2. what the engine looked at, in one line each ---------------- #
        evaluated = explanation["what_was_evaluated"]
        if evaluated:
            st.markdown("**What was measured**")
            for line in evaluated:
                st.caption(f"· {line}")

        # The boundary is a printed line for a COMPOSED condition: the one
        # sentence that must never be scrolled past.
        if intent in COMPOSED_INTENTS and explanation["boundary"]:
            st.warning(explanation["boundary"], icon=None)

        # ---- 3. the numbers ------------------------------------------------ #
        counts = explanation["counts"]
        unknown = int(counts.get("insufficient", 0) or 0)
        columns = st.columns(4)
        for column, (label, value) in zip(columns, [
                ("Matched / valid cells", f"{int(counts.get('matched', 0)):,}"),
                ("Measured, not matching", f"{int(counts.get('non_matching', 0)):,}"),
                ("Undecided cells", f"{unknown:,}"),
                ("Matched area", f"{package.matched_area_km2:,.3f} km²"),
        ]):
            column.metric(label, value)
        if explanation["unknown_note"] and unknown:
            st.caption(f"Unknown — {explanation['unknown_note']}")

        # ---- 4. why these cells matched ------------------------------------ #
        rows = condition_rows(package)
        if rows:
            with st.expander("Why these cells matched", expanded=False):
                st.caption(explanation["how_it_was_evaluated"])
                st.markdown(_table(
                    ("Condition", "Source", "Threshold / parameter",
                     "Provenance", "Matched", "Not matching", "Undecided"),
                    [[row["condition"], row["source"], row["parameter"],
                      row["origin"], row["matched"], row["non_matching"],
                      row["unknown"]] for row in rows],
                ))

        # ---- 5. evidence sources ------------------------------------------- #
        sources = list(explanation["evidence_sources"])
        if sources:
            with st.expander("Evidence sources", expanded=False):
                for line in sources:
                    st.caption(f"· {line}")

        # ---- 6. analysis / grid details ------------------------------------ #
        with st.expander("Analysis details", expanded=False):
            st.markdown(_table(("Field", "Value"),
                               [[k, v] for k, v in grid_rows(package)]))
            # `grid_rows` ends with one "Threshold — <name>" row per
            # threshold, each showing where the value came from.
            thresholds = [(name, text) for name, text in grid_rows(package)
                          if name.startswith("Threshold")]
            if thresholds:
                st.markdown("**Threshold provenance**")
                for name, text in thresholds:
                    st.caption(f"· {name} — {text}")

        # ---- 7. limitations ------------------------------------------------ #
        limitations = list(explanation["limitations"])
        if limitations:
            with st.expander("Limitations", expanded=False):
                for line in limitations:
                    st.caption(f"· {line}")

        # ---- 8. per-condition layers, OPT-IN ------------------------------- #
        _render_layer_opt_in(package, entry, key)

        # ---- 9. export ----------------------------------------------------- #
        st.download_button(
            "Export evidence (JSON)",
            data=package.to_json().encode("utf-8"),
            file_name=f"satquery_evidence_{str(package.intent).lower()}_"
                      f"{entry.get('_id', 0)}.json",
            mime="application/json",
            width="stretch",
            key=f"{key}_export_{entry.get('_id', 0)}",
            help="The facts that produced this answer: the counts, the "
                 "conditions, their provenance, the grid and the explanation "
                 "text. Nothing is added that the engine did not report.",
        )


def _table(header: Tuple[str, ...], rows: List[List[Any]]) -> str:
    """A markdown table, so the values are real text in the page."""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = [str(cell).replace("|", "\\|") for cell in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _render_layer_opt_in(package: Any, entry: Dict[str, Any], key: str) -> None:
    """Opt-in switch for the per-condition evidence layers.

    Off by default, as agreed: the map is not overloaded automatically and the
    combined layer keeps answering the question that was asked. Toggling it
    triggers exactly one rerun so the map section above can add the layers.
    """
    result = entry.get("result")
    masks = evidence_masks(result)
    if not masks:
        return
    state_key = "evidence_layers_on"
    owner_key = "evidence_layers_owner"
    previous = bool(st.session_state.get(state_key, False))
    # A layer set belongs to ONE answer: a new query drops it again.
    if st.session_state.get(owner_key) != id(result):
        st.session_state[state_key] = False
        st.session_state[owner_key] = id(result)
        previous = False
    show = st.checkbox(
        "Show individual condition layers on the map",
        value=previous,
        key=f"{key}_layers_{entry.get('_id', 0)}",
        help="Adds one display layer per condition (for example "
             "'Evidence — NDVI > 0.6'). The combined result layer stays. "
             "Display copies only: the counts were computed on the analysis "
             "grid, never on the reprojected image.",
    )
    if show != previous:
        st.session_state[state_key] = show
        st.rerun()


# =========================================================================== #
# the layers themselves
# =========================================================================== #
def evidence_layer_choice(result: Any) -> List[Tuple[str, str, Any]]:
    """(layer name, condition name, GridMask) for the currently enabled set."""
    if not st.session_state.get("evidence_layers_on", False):
        return []
    if st.session_state.get("evidence_layers_owner") != id(result):
        return []
    labels: Dict[str, str] = {}
    for entry in getattr(result, "condition_results", ()) or ():
        labels[str(entry.get("name", ""))] = str(
            entry.get("label") or entry.get("name") or "condition")
    masks = evidence_masks(result)
    return [(f"Evidence — {labels.get(name, name)}", name, mask)
            for name, mask in masks.items()]


def evidence_layer_colour(result: Any, condition_name: str) -> Tuple[int, int, int]:
    """Colour by the KIND of evidence, so name, colour and legend agree."""
    for entry in getattr(result, "condition_results", ()) or ():
        if str(entry.get("name", "")) == str(condition_name):
            return _EVIDENCE_COLOURS.get(str(entry.get("kind", "spatial")),
                                         _EVIDENCE_COLOURS["spatial"])
    if str(condition_name) in ("increase", "stable", "decrease"):
        return _EVIDENCE_COLOURS["temporal"]
    return _EVIDENCE_COLOURS["spatial"]


@st.cache_data(max_entries=6, show_spinner=False)
def web_evidence_mask(state: Any, inside: Any, transform_tuple: Tuple[float, ...],
                      crs_wkt: str, max_pixels: int, colour: Tuple[int, int, int],
                      ) -> Tuple[Any, Any]:
    """Display copy of one evidence mask.

    Separate from `web_spatial_from_mask` on purpose: its own cache cannot
    evict the combined result layer, and its own colour identifies WHICH kind
    of evidence the layer is. Categorical -> nearest neighbour, and cells
    outside the ROI are fully transparent because they were never analysed.
    """
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling

    from core.geo import WEB_MERCATOR, reproject_array

    src_crs = CRS.from_user_input(crs_wkt)
    web = reproject_array(np.asarray(state).astype("float32"),
                          src_transform=Affine(*transform_tuple),
                          src_crs=src_crs, dst_crs=WEB_MERCATOR,
                          resampling=Resampling.nearest,
                          max_pixels=max_pixels)
    web_inside = reproject_array(np.asarray(inside).astype("float32"),
                                 src_transform=Affine(*transform_tuple),
                                 src_crs=src_crs, dst_crs=WEB_MERCATOR,
                                 resampling=Resampling.nearest,
                                 max_pixels=max_pixels)
    arr = np.asarray(web.array)
    codes = np.where(np.isfinite(arr), np.rint(arr), 0).astype("int16")
    valid = np.asarray(web.mask) & (np.nan_to_num(
        np.asarray(web_inside.array), nan=0.0) > 0.5)
    rgba = np.zeros((*codes.shape, 4), "uint8")
    rgba[codes == 2] = (*colour, 190)          # TRUE
    rgba[codes == 1] = (120, 120, 120, 70)     # measured, not matching
    rgba[~valid] = (0, 0, 0, 0)                # never analysed
    return rgba, web.leaflet_bounds
