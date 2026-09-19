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
           "evidence_layer_colour", "web_evidence_mask",
           "render_evidence_explorer", "render_provenance_timeline",
           "render_evidence_comparison", "export_evidence_report"]

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


# =========================================================================== #
# Step 8: Enhanced Evidence & Provenance UX
# =========================================================================== #

def render_evidence_explorer(package: Any, key: str = "evidence_explorer") -> None:
    """Interactive evidence exploration with filtering/grouping.

    Allows filtering evidence records by condition type, date, layer, source.
    Only shows filters for metadata that is actually present in the package.
    Filtering affects presentation only, not the underlying analysis result.
    """
    if not package.records and not package.statistics:
        st.caption("No evidence records to explore.")
        return

    st.markdown("**Evidence Explorer**")
    st.caption("Filter and explore the evidence records that produced this answer.")

    # Collect all records for filtering
    all_records = list(package.records) + list(package.statistics)
    if package.combined:
        all_records = [package.combined] + all_records

    # Determine available filter dimensions from actual data
    kinds = sorted(set(r.kind for r in all_records))
    has_dates = any(r.source_dates for r in all_records)
    has_bands = any(r.band_or_index for r in all_records)
    has_sources = any(r.source_dataset and r.source_dataset != UNAVAILABLE for r in all_records)
    has_conditions = any(r.condition for r in all_records)

    # Filter controls
    filter_cols = st.columns(4)
    filters = {}

    with filter_cols[0]:
        if kinds:
            selected_kinds = st.multiselect(
                "Kind",
                options=kinds,
                default=kinds,
                key=f"{key}_kind_filter"
            )
            filters["kind"] = selected_kinds

    with filter_cols[1]:
        if has_conditions:
            conditions = sorted(set(r.condition for r in all_records if r.condition))
            selected_conditions = st.multiselect(
                "Condition",
                options=conditions,
                default=conditions,
                key=f"{key}_condition_filter"
            )
            filters["condition"] = selected_conditions

    with filter_cols[2]:
        if has_bands:
            bands = sorted(set(r.band_or_index for r in all_records if r.band_or_index))
            selected_bands = st.multiselect(
                "Band / Index",
                options=bands,
                default=bands,
                key=f"{key}_band_filter"
            )
            filters["band"] = selected_bands

    with filter_cols[3]:
        if has_sources:
            sources = sorted(set(r.source_dataset for r in all_records
                               if r.source_dataset and r.source_dataset != UNAVAILABLE))
            selected_sources = st.multiselect(
                "Source",
                options=sources,
                default=sources,
                key=f"{key}_source_filter"
            )
            filters["source"] = selected_sources

    # Apply filters
    filtered_records = all_records
    if filters.get("kind"):
        filtered_records = [r for r in filtered_records if r.kind in filters["kind"]]
    if filters.get("condition"):
        filtered_records = [r for r in filtered_records if r.condition in filters["condition"]]
    if filters.get("band"):
        filtered_records = [r for r in filtered_records if r.band_or_index in filters["band"]]
    if filters.get("source"):
        filtered_records = [r for r in filtered_records
                          if r.source_dataset in filters["source"]]

    st.caption(f"Showing {len(filtered_records)} of {len(all_records)} records")

    # Grouping selector
    group_by = st.selectbox(
        "Group by",
        options=["None", "Kind", "Condition", "Band/Index", "Source"],
        index=0,
        key=f"{key}_group_by"
    )

    # Display records
    if group_by == "None":
        _render_record_list(filtered_records, key)
    else:
        _render_grouped_records(filtered_records, group_by.lower().replace("/", "_"), key)


def _render_record_list(records: List[Any], key: str) -> None:
    """Render a flat list of evidence records."""
    for i, record in enumerate(records):
        with st.expander(f"{record.kind.title()}: {record.label}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"**Kind:** {record.kind}")
                st.caption(f"**Label:** {record.label}")
                if record.condition:
                    st.caption(f"**Condition:** {record.condition}")
                if record.band_or_index:
                    st.caption(f"**Band/Index:** {record.band_or_index}")
            with col2:
                if record.source_analysis:
                    st.caption(f"**Analysis:** {record.source_analysis}")
                if record.source_dataset and record.source_dataset != UNAVAILABLE:
                    st.caption(f"**Dataset:** {record.source_dataset}")
                if record.source_dates:
                    st.caption(f"**Dates:** {', '.join(record.source_dates)}")

            # Counts
            counts = record.counts
            if counts:
                st.caption(f"**Matched:** {counts.get('matched', 0):,}")
                st.caption(f"**Not matching:** {counts.get('non_matching', 0):,}")
                st.caption(f"**Undecided:** {counts.get('insufficient', 0):,}")
                st.caption(f"**Total:** {counts.get('total', 0):,}")

            # Provenance
            if record.provenance:
                with st.expander("Provenance", expanded=False):
                    st.json(record.provenance)


def _render_grouped_records(records: List[Any], group_by: str, key: str) -> None:
    """Render records grouped by the specified dimension."""
    from collections import defaultdict

    groups = defaultdict(list)
    for record in records:
        if group_by == "kind":
            groups[record.kind].append(record)
        elif group_by == "condition":
            groups[record.condition or "unavailable"].append(record)
        elif group_by == "band":
            groups[record.band_or_index or "unavailable"].append(record)
        elif group_by == "source":
            groups[record.source_dataset or UNAVAILABLE].append(record)

    for group_name, group_records in sorted(groups.items()):
        with st.expander(f"{group_by.title()}: {group_name} ({len(group_records)} records)", expanded=False):
            _render_record_list(group_records, f"{key}_{group_by}_{group_name}")


def render_provenance_timeline(package: Any, key: str = "provenance_timeline") -> None:
    """Render a provenance timeline showing the logical chain of evidence.

    Shows analysis components, source datasets, dates, and processing steps
    in a logical order. When timestamps are unavailable, uses deterministic
    logical ordering.
    """
    st.markdown("**Provenance Timeline**")
    st.caption("The logical chain of evidence used to produce this answer.")

    # Build timeline steps from the package
    steps = []

    # 1. Query
    steps.append({
        "step": 1,
        "title": "User Query",
        "description": package.query or UNAVAILABLE,
        "type": "query",
        "details": {
            "Normalized": package.normalized_query or UNAVAILABLE,
            "Intent": package.intent or UNAVAILABLE,
        }
    })

    # 2. Conditions evaluated
    if package.records:
        for i, record in enumerate(package.records):
            steps.append({
                "step": len(steps) + 1,
                "title": f"Condition: {record.label}",
                "description": f"Kind: {record.kind}, Type: {record.condition or 'N/A'}",
                "type": "condition",
                "details": {
                    "Source Analysis": record.source_analysis or UNAVAILABLE,
                    "Source Dataset": record.source_dataset or UNAVAILABLE,
                    "Band/Index": record.band_or_index or UNAVAILABLE,
                    "Dates": ", ".join(record.source_dates) if record.source_dates else UNAVAILABLE,
                    "Operator": record.operator or UNAVAILABLE,
                    "Threshold": str(record.threshold) if record.threshold is not None else UNAVAILABLE,
                    "Threshold Provenance": record.threshold_provenance or UNAVAILABLE,
                    "Grid": record.grid or UNAVAILABLE,
                    "Counts": record.counts,
                    "Area (m²)": record.area_m2,
                    "Fraction": record.fraction,
                }
            })

    # 3. Combined result
    if package.combined:
        steps.append({
            "step": len(steps) + 1,
            "title": "Combined Result",
            "description": f"Operator: {package.combined.source_analysis}",
            "type": "combined",
            "details": {
                "Label": package.combined.label,
                "Counts": package.combined.counts,
                "Area (m²)": package.combined.area_m2,
                "Fraction": package.combined.fraction,
                "Expression": package.expression or UNAVAILABLE,
            }
        })

    # 4. Statistics
    if package.statistics:
        for stat in package.statistics:
            steps.append({
                "step": len(steps) + 1,
                "title": f"Statistics: {stat.label}",
                "description": f"Kind: statistics",
                "type": "statistics",
                "details": {
                    "Band/Index": stat.band_or_index or UNAVAILABLE,
                    "Source Analysis": stat.source_analysis or UNAVAILABLE,
                    "Source Dataset": stat.source_dataset or UNAVAILABLE,
                    "Parameters": stat.parameters,
                    "Counts": stat.counts,
                }
            })

    # 5. Sources summary
    if package.sources:
        steps.append({
            "step": len(steps) + 1,
            "title": "Evidence Sources",
            "description": f"{len(package.sources)} source(s) used",
            "type": "sources",
            "details": {
                "Sources": [f"{s.get('analysis', '')} — {s.get('dataset', '')}"
                          for s in package.sources]
            }
        })

    # 6. Grid/Alignment
    if package.grid or package.alignment:
        steps.append({
            "step": len(steps) + 1,
            "title": "Grid & Alignment",
            "description": "Analysis grid parameters",
            "type": "grid",
            "details": {
                "Grid": package.grid or UNAVAILABLE,
                "Alignment": package.alignment or UNAVAILABLE,
            }
        })

    # 7. Limitations
    if package.limitations:
        steps.append({
            "step": len(steps) + 1,
            "title": "Limitations",
            "description": f"{len(package.limitations)} limitation(s) noted",
            "type": "limitations",
            "details": {
                "Limitations": list(package.limitations)
            }
        })

    # Render timeline
    for step in steps:
        with st.expander(f"Step {step['step']}: {step['title']}", expanded=(step['step'] <= 2)):
            st.caption(step['description'])
            if step.get('details'):
                for k, v in step['details'].items():
                    if isinstance(v, (list, tuple)):
                        st.caption(f"**{k}:**")
                        for item in v:
                            st.caption(f"  · {item}")
                    elif isinstance(v, dict):
                        st.caption(f"**{k}:**")
                        for k2, v2 in v.items():
                            st.caption(f"  · {k2}: {v2}")
                    else:
                        st.caption(f"**{k}:** {v}")


def render_evidence_comparison(entry: Dict[str, Any],
                               conversation_state: Optional[Any] = None,
                               key: str = "evidence_comparison") -> None:
    """Render visual diff/comparison for multi-turn chains.

    Only appears when comparable evidence actually exists in the conversation
    (e.g., two temporal analyses, or before/after analyses). Reuses existing
    multi-condition/change-detection outputs.
    """
    package = evidence_from_entry(entry)
    if package is None:
        return

    # Check if this result has comparable evidence
    # Temporal analyses have before/after, spatial has before/after in multi-condition
    has_temporal = any(r.kind == "temporal" for r in package.records)
    has_combined = package.combined is not None
    has_statistics = bool(package.statistics)

    # Check conversation history for previous comparable results
    comparable_entries = []
    if conversation_state and hasattr(conversation_state, 'get_recent_turns'):
        for turn in conversation_state.get_recent_turns(5):
            if turn.get('tool_name') in ('temporal_compare', 'compute_ndvi', 'compute_ndwi',
                                          'crop_suitability', 'multi_condition_query'):
                comparable_entries.append(turn)

    if not (has_temporal or has_combined or has_statistics or comparable_entries):
        return

    st.markdown("**Evidence Comparison**")
    st.caption("Compare evidence across related analyses (when available).")

    comparison_type = None
    if has_temporal:
        comparison_type = "temporal"
    elif has_combined and package.records:
        comparison_type = "conditions"
    elif has_statistics:
        comparison_type = "statistics"

    if comparison_type == "temporal":
        _render_temporal_comparison(package, key)
    elif comparison_type == "conditions":
        _render_conditions_comparison(package, key)
    elif comparison_type == "statistics":
        _render_statistics_comparison(package, key)

    # If we have conversation history, offer to compare with previous turn
    if comparable_entries:
        with st.expander("Compare with previous analysis", expanded=False):
            st.caption("Select a previous analysis to compare evidence:")
            for i, turn in enumerate(comparable_entries):
                if st.button(
                    f"Compare: {turn.get('user_query', 'Previous analysis')} "
                    f"({turn.get('tool_name', 'unknown')})",
                    key=f"{key}_compare_{i}"
                ):
                    # Store comparison target in session state
                    st.session_state[f"{key}_compare_target"] = turn
                    st.rerun()


def _render_temporal_comparison(package: Any, key: str) -> None:
    """Render before/after comparison for temporal analyses."""
    temporal_records = [r for r in package.records if r.kind == "temporal"]
    if not temporal_records:
        return

    st.markdown("**Temporal Comparison (Before → After)**")

    # Build comparison data
    classes = ["increase", "stable", "decrease", "insufficient"]
    for cls in classes:
        record = next((r for r in temporal_records if r.condition == cls), None)
        if record:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**{cls.title()}**")
            with col2:
                st.caption(f"Matched: {record.matched:,}")
            with col3:
                st.caption(f"Fraction: {record.fraction:.1%}" if record.fraction else "N/A")


def _render_conditions_comparison(package: Any, key: str) -> None:
    """Render comparison across multiple conditions."""
    st.markdown("**Condition-by-Condition Comparison**")

    rows = []
    for record in package.records:
        rows.append({
            "Condition": record.label,
            "Kind": record.kind,
            "Matched": f"{record.matched:,}",
            "Not Matching": f"{record.non_matching:,}",
            "Undecided": f"{record.unknown:,}",
            "Fraction": f"{record.fraction:.1%}" if record.fraction else "N/A",
        })

    if rows:
        st.markdown(_table(
            ("Condition", "Kind", "Matched", "Not Matching", "Undecided", "Fraction"),
            [[r["Condition"], r["Kind"], r["Matched"], r["Not Matching"],
              r["Undecided"], r["Fraction"]] for r in rows],
        ))


def _render_statistics_comparison(package: Any, key: str) -> None:
    """Render statistics comparison."""
    if not package.statistics:
        return

    st.markdown("**Statistics Comparison**")

    for stat in package.statistics:
        params = stat.parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"**{stat.band_or_index.upper()}**")
        with col2:
            if params.get("mean") is not None:
                st.caption(f"Mean: {params['mean']:.4f}")
        with col3:
            if params.get("valid_pixels"):
                st.caption(f"Valid cells: {params['valid_pixels']:,}")


def export_evidence_report(entry: Dict[str, Any],
                          conversation_state: Optional[Any] = None,
                          key: str = "evidence_export") -> None:
    """Export the current result with evidence and conversation context as a reproducible report.

    Includes: user query, analysis/tool, result/status, structured arguments,
    ROI availability, dates, crop, evidence items, data sources, provenance,
    and conversation context. Never includes secrets, credentials, or large
    raster arrays.
    """
    package = evidence_from_entry(entry)
    if package is None:
        st.caption("No evidence to export for this entry.")
        return

    # Build the report
    report = {
        "schema": "satquery-report/1",
        "query": package.query,
        "normalized_query": package.normalized_query,
        "intent": package.intent,
        "status": package.status,
        "expression": package.expression,
        "result": {
            "status": package.status,
            "matched_cells": package.matched_cells,
            "non_matching_cells": package.non_matching_cells,
            "unknown_cells": package.unknown_cells,
            "analysed_cells": package.analysed_cells,
            "matched_area_m2": package.matched_area_m2,
            "unknown_handling": package.unknown_handling,
        },
        "evidence": {
            "conditions": [r.to_dict() for r in package.records],
            "combined": package.combined.to_dict() if package.combined else None,
            "statistics": [r.to_dict() for r in package.statistics],
            "sources": [dict(s) for s in package.sources],
            "grid": dict(package.grid),
            "alignment": dict(package.alignment),
            "threshold_provenance": dict(package.threshold_provenance),
            "limitations": list(package.limitations),
            "boundary": package.boundary,
            "runtime_ms": package.runtime_ms,
        },
        "explanation": package.explanation,
    }

    # Add conversation context if available
    if conversation_state:
        summary = conversation_state.get_context_summary() if hasattr(conversation_state, 'get_context_summary') else {}
        if summary:
            report["conversation_context"] = {
                "recent_turns": summary.get("recent_turns", []),
                "current_roi_available": summary.get("current_roi_available", False),
                "current_dates": summary.get("current_dates", [None, None]),
                "current_crop": summary.get("current_crop"),
                "current_intent": summary.get("current_intent"),
            }

    # Remove unavailable/empty fields for cleaner export
    report = _clean_report(report)

    # Generate filename
    intent = str(package.intent or "unknown").lower()
    entry_id = entry.get('_id', 0)
    filename = f"satquery_report_{intent}_{entry_id}.json"

    st.download_button(
        "Export Reproducible Report (JSON)",
        data=json.dumps(report, indent=2, sort_keys=True).encode("utf-8"),
        file_name=filename,
        mime="application/json",
        width="stretch",
        key=f"{key}_report_{entry.get('_id', 0)}",
        help="Download a reproducible report including the query, analysis, evidence, "
             "provenance, and conversation context. No secrets or large arrays included.",
    )


def _clean_report(obj: Any) -> Any:
    """Recursively remove UNAVAILABLE and empty fields from report."""
    if obj is None or obj == UNAVAILABLE:
        return None
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            cv = _clean_report(v)
            if cv is not None and cv != {} and cv != [] and cv != "":
                cleaned[k] = cv
        return cleaned
    if isinstance(obj, (list, tuple)):
        cleaned = [_clean_report(v) for v in obj]
        return [v for v in cleaned if v is not None and v != {} and v != [] and v != ""]
    return obj


# Import json at module level for export function
import json
