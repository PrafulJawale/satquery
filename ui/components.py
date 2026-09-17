"""ui/components.py -- reusable Streamlit rendering blocks.

CONTRACT: every function here takes plain dicts/JSON, never a rasterio object.
That keeps the UI dumb and swappable: we can replace Streamlit with FastAPI +
React later by re-implementing only this package.
"""

from __future__ import annotations

import html
import re

from typing import Any, Dict, List, Optional

import streamlit as st

from core.roi import OUTSIDE_MESSAGE

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    step = 1024.0
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < step or unit == "GB":
            return f"{val:.1f} {unit}" if unit != "B" else f"{int(val)} B"
        val /= step
    return f"{val:.1f} GB"


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (v != v):  # NaN
        return "NaN"
    return f"{v:,.{nd}f}"


# --------------------------------------------------------------------------- #
# metadata rendering
# --------------------------------------------------------------------------- #
def render_metrics(info: Dict[str, Any]) -> None:
    sp = info["spatial"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raster size", f"{info['width']} × {info['height']} px")
    c2.metric("Bands", str(info["count"]))
    c3.metric("Data type", ", ".join(sorted(set(info["dtypes"]))))
    c4.metric("Pixel size", sp["resolution_label"])

    c5, c6, c7, c8 = st.columns(4)
    crs_label = f"EPSG:{sp['crs_epsg']}" if sp.get("crs_epsg") else (sp.get("crs_name") or "NONE")
    c5.metric("CRS", crs_label)
    c6.metric("North-up", "yes" if sp["is_north_up"] else "NO (rotated)")
    area = sp.get("approx_area_km2")
    c7.metric("Coverage", f"{area:,.1f} km²" if area else "—")
    c8.metric("File size", _human_bytes(info.get("file_size_bytes")))


def render_spatial_table(info: Dict[str, Any]) -> None:
    sp = info["spatial"]
    st.subheader("Georeferencing", divider="gray")

    rows = [
        ("Has CRS", "yes" if sp["has_crs"] else "NO"),
        ("CRS EPSG code", sp.get("crs_epsg") or "—"),
        ("CRS name", sp.get("crs_name") or "—"),
        ("Kind", "geographic (degrees)" if sp["is_geographic"] else ("projected" if sp["is_projected"] else "unknown")),
        ("Linear unit", sp.get("linear_units") or "—"),
        ("Pixel size X", _fmt(sp["pixel_size_x"], 6)),
        ("Pixel size Y", _fmt(sp["pixel_size_y"], 6)),
        ("Square pixels", "yes" if abs(abs(sp["pixel_size_x"]) - abs(sp["pixel_size_y"])) < 1e-9 else "no"),
        ("North-up (unrotated)", "yes" if sp["is_north_up"] else "no — rotated / skewed"),
    ]
    st.dataframe(
        [{"Property": k, "Value": str(v)} for k, v in rows],
        hide_index=True,
        width="stretch",
    )

    st.markdown("**Affine transform** (`x = a·col + b·row + c`, `y = d·col + e·row + f`)")
    a, b, c, d, e, f = sp["transform"]
    t1, t2 = st.columns(2)
    t1.code(f"a (x scale)      = {a:.10g}\nb (x rotation)   = {b:.10g}\nc (x origin)     = {c:.10g}")
    t2.code(f"d (y rotation)   = {d:.10g}\ne (y scale)      = {e:.10g}\nf (y origin)     = {f:.10g}")

    st.markdown("**Bounds**")
    bn = sp.get("bounds_native")
    bw = sp.get("bounds_wgs84")
    b1, b2 = st.columns(2)
    with b1:
        st.caption("In the file's own CRS")
        if bn:
            st.code(f"left   {bn[0]:,.4f}\nbottom {bn[1]:,.4f}\nright  {bn[2]:,.4f}\ntop    {bn[3]:,.4f}")
        else:
            st.write("—")
    with b2:
        st.caption("WGS84 (lon / lat)")
        if bw:
            st.code(f"min lon {bw[0]:,.6f}\nmin lat {bw[1]:,.6f}\nmax lon {bw[2]:,.6f}\nmax lat {bw[3]:,.6f}")
        else:
            st.write("not available (no CRS)")

    foot = sp.get("footprint_wgs84")
    st.caption(
        f"Footprint outline: {len(foot)} densified vertices in WGS84"
        if foot
        else "Footprint: unavailable (no CRS)"
    )
    st.caption(
        "The footprint is densified (extra points inserted along each edge) because "
        "reprojecting a projected raster into lon/lat bends straight edges into curves. "
        "Using only 4 corners would understate the real coverage."
    )


def render_bands_table(info: Dict[str, Any]) -> None:
    st.subheader("Bands", divider="gray")
    rows = []
    for b in info["bands"]:
        rows.append(
            {
                "Band": b["index"],
                "Name (from file)": b["name"] or "— (unnamed)",
                "Data type": b["dtype"],
                "Nodata": "—" if b["nodata"] is None else _fmt(b["nodata"], 3),
                "Block (H×W)": f"{b['block_shape'][0]} × {b['block_shape'][1]}",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")

    st.markdown("**Storage**")
    st.dataframe(
        [
            {"Property": "Driver", "Value": info["driver"]},
            {"Property": "Tiled", "Value": str(info["tiled"])},
            {"Property": "Block shape", "Value": f"{info['block_shape'][0]} × {info['block_shape'][1]}"},
            {"Property": "Overview levels", "Value": str(list(info["overview_levels"])) or "—"},
            {"Property": "Compression", "Value": str(info["compress"]) or "—"},
            {"Property": "Interleave", "Value": str(info["interleave"]) or "—"},
            {"Property": "Looks like COG (heuristic)", "Value": str(info["looks_like_cog"])},
            {"Property": "Est. full read memory", "Value": f"{info['estimated_full_read_mb']:,.1f} MB"},
        ],
        hide_index=True,
        width="stretch",
    )

    if info["count"] >= 4:
        st.info(
            f"This file has {info['count']} bands, so it *may* contain a NIR band — but a band "
            "count is NOT evidence of band meaning. Red and NIR must be mapped "
            "explicitly (from band names/metadata, or by telling the app). Nothing is guessed now.",
            icon=None,
        )
    elif info["count"] == 3:
        st.info(
            "3 bands: suitable for an RGB preview. NDVI needs a NIR band, so a 3-band file "
            "cannot support vegetation indices unless one of the bands is actually NIR.",
            icon=None,
        )


def render_warnings(warnings: List[str]) -> None:
    st.subheader("Caveats detected", divider="gray")
    if not warnings:
        st.success("No structural problems detected in the file's georeferencing or nodata conventions.")
        return
    for w in warnings:
        (st.error if w.startswith(("NO CRS", "ROTATED", "8-BIT")) else st.warning)(w, icon=None)


#: Internal references ("see docs/PHASE2.md") belong in the repository, not in
#: a product interface. Provenance DATA is never rewritten -- only the line the
#: user reads is cleaned.
_DOC_REFERENCE = re.compile(
    r"(?:--\s*|,\s*)?see\s+docs/[A-Za-z0-9_.\- ]+\.md", re.IGNORECASE
)


def clean_note(text: Optional[str]) -> str:
    """Strip internal document references from text shown to a user."""
    if not text:
        return ""
    out = _DOC_REFERENCE.sub("", str(text))
    out = re.sub(r"--\s*\)", ")", out)
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s+\)", ")", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def render_provenance(prov: Optional[Dict[str, Any]]) -> None:
    if not prov:
        return
    st.subheader("Scene provenance", divider="gray")
    badge = "Real satellite data" if prov["is_real_satellite_data"] else "Synthetic test fixture"
    valid = (
        "pixel values are analysis-ready reflectance"
        if prov["bands_are_physically_valid"]
        else "pixel values are NOT valid for spectral indices"
    )
    st.markdown(
        f'<div class="sq-chips">'
        f'<span class="sq-chip">{html.escape(badge)}</span>'
        f'<span class="sq-chip">{html.escape(valid)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{prov['name']}**")
    st.write(prov["what_it_is"])
    c1, c2 = st.columns(2)
    c1.markdown("**Good for:** " + (", ".join(prov["good_for"]) or "—"))
    c2.markdown("**Not valid for:** " + (", ".join(prov["not_good_for"]) or "—"))
    if prov["source_url"]:
        st.caption(f"Source: {prov['source_url']}")
    notes = clean_note(prov.get("notes"))
    if notes:
        st.caption(notes)


def render_raw_metadata(info: Dict[str, Any]) -> None:
    """Technical detail, kept out of the way: useful to a specialist, never
    part of the story the interface tells."""
    with st.expander("File metadata (technical)", expanded=False):
        st.json(info, expanded=False)
    if info.get("tags"):
        with st.expander("GDAL tags (technical)", expanded=False):
            st.json(info["tags"], expanded=False)


def render_roadmap(current_phase: Optional[int] = None) -> None:
    """Shows what this build does -- and what it deliberately does not.

    `current_phase` is accepted and ignored: it is kept so call sites that used
    to pass a phase number keep working. The UI no longer numbers its
    capabilities, because the number described the build order, not the user.
    """
    built = [
        "GeoTIFF / COG ingestion, metadata and quality checks",
        "True-colour and false-colour raster previews",
        "NDVI and NDWI -- continuous, un-thresholded measurements",
        "Interactive map: raster overlays, place search, globe navigation",
        "Area-of-interest drawing and zonal statistics",
        "Spatial, temporal and multi-condition analysis",
        "Evidence, provenance and export for every result",
    ]
    limits = [
        "No real-time imagery: you bring the raster, the app analyses it",
        "Previews are display copies -- every number comes from the native raster",
        "No land-cover classification, flood model, anomaly claim or prediction",
    ]
    with st.expander("What SatQuery AI does — and what it does not", expanded=False):
        st.markdown("**Included**")
        for item in built:
            st.markdown(f"- {item}")
        st.markdown("**Not included**")
        for item in limits:
            st.markdown(f"- {item}")


# =========================================================================== #
# PHASE 2 -- band inspection and raster visualisation
# =========================================================================== #
def render_band_inspector(stats_rows: List[Dict[str, Any]], roles: Dict[str, int]) -> None:
    """Per-band table: what the file says, what we guessed, what the pixels look like.

    Statistics come from a DECIMATED read (they are indicative) -- Phase 6 adds
    exact statistics. Saying so in the caption is part of being honest.
    """
    st.subheader("Band inspection", divider="gray")

    role_of = {idx: role for role, idx in roles.items()}
    rows = []
    for i, s in enumerate(stats_rows, start=1):
        rows.append(
            {
                "Band": i,
                "Name": s.get("band_name") or "— (unnamed)",
                "Guessed role": role_of.get(i, "—"),
                "Type": s["dtype"],
                "Nodata": "—" if s["nodata"] is None else _fmt(s["nodata"], 3),
                "Valid %": f"{100 * s['valid_fraction']:.1f}",
                "Min": _fmt(s["min"], 0),
                "p2": _fmt(s["p2"], 0),
                "Median": _fmt(s["p50"], 0),
                "p98": _fmt(s["p98"], 0),
                "Max": _fmt(s["max"], 0),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Statistics are computed on the **decimated preview read**, not the full scene, "
        "so treat them as indicative -- exact figures come from the analysis "
        "itself. Values are raw digital numbers (DN), not reflectance."
    )

    empty = [r["Band"] for r in rows if r["Valid %"] == "0.0"]
    if empty:
        st.warning(
            f"Band(s) {empty} contain no valid pixels at all (all nodata). "
            "Any statistic computed without masking would look like a real number.",
            icon=None,
        )


def render_guess_banner(guess: Dict[str, Any]) -> None:
    """Shows the band-meaning guess together with the evidence behind it."""
    with st.container(border=True):
        st.markdown(
            f"**Band meaning: {guess.get('confidence', 'none').upper()} confidence**"
            + (f" — profile `{guess.get('profile')}`" if guess.get("profile") else "")
        )
        if guess.get("roles"):
            st.caption(
                " · ".join(
                    f"{role} = band {idx}" for role, idx in sorted(guess["roles"].items(), key=lambda kv: kv[1])
                )
            )
        if guess.get("evidence"):
            with st.expander("Why? (evidence used)", expanded=False):
                for e in guess["evidence"]:
                    st.markdown(f"- {e}")
        for w in guess.get("warnings", []):
            st.caption("Note — " + w)


def band_options(info: Dict[str, Any]) -> List[str]:
    """Human labels for the band selector."""
    out = []
    for b in info["bands"]:
        name = b["name"] or ("band " + str(b["index"]))
        out.append(f"{b['index']} · {name}")
    return out


def render_band_mapping(
    info: Dict[str, Any], guess: Dict[str, Any], key_prefix: str = "bm"
) -> Dict[str, Any]:
    """Let the user confirm or override which band plays which role.

    Returns {"red": int, "green": int, "blue": int, "nir": int|None, "confirmed": bool}.
    NOTE: we never silently trust the guess -- the user must confirm it.
    """
    st.subheader("Band mapping", divider="gray")
    options = band_options(info)
    n = info["count"]

    def idx_label(role: Optional[int]) -> Optional[str]:
        return options[role - 1] if role and 1 <= role <= n else None

    roles = guess.get("roles", {})
    defaults = {
        "red": idx_label(roles.get("red")) or (options[2] if n >= 3 else options[0]),
        "green": idx_label(roles.get("green")) or (options[1] if n >= 3 else options[0]),
        "blue": idx_label(roles.get("blue")) or (options[0] if n >= 3 else options[0]),
    }
    nir_default = idx_label(roles.get("nir"))

    c1, c2, c3, c4 = st.columns(4)
    red = c1.selectbox("Red band", options, index=options.index(defaults["red"]), key=f"{key_prefix}_r")
    green = c2.selectbox("Green band", options, index=options.index(defaults["green"]), key=f"{key_prefix}_g")
    blue = c3.selectbox("Blue band", options, index=options.index(defaults["blue"]), key=f"{key_prefix}_b")

    nir_opts = ["— none (no NIR band) —"] + options
    nir_index = nir_opts.index(nir_default) if nir_default else 0
    nir_sel = c4.selectbox("NIR band", nir_opts, index=nir_index, key=f"{key_prefix}_nir")

    def to_index(label: Optional[str]) -> Optional[int]:
        if not label or label.startswith("—"):
            return None
        return int(str(label).split("·")[0].strip())

    mapping = {
        "red": to_index(red),
        "green": to_index(green),
        "blue": to_index(blue),
        "nir": to_index(nir_sel),
    }

    st.caption(
        "Band meaning is **metadata, not physics**: a 4-band file is not automatically "
        "blue/green/red/NIR. Selectors above are pre-filled from band names and GDAL "
        "colour interpretation, but they are only a suggestion until you confirm."
    )
    mapping["confirmed"] = st.checkbox(
        "I confirm this band mapping is correct for this file",
        value=False,
        key=f"{key_prefix}_confirm",
        help="NDVI refuses to run on an unconfirmed band mapping.",
    )
    if not mapping["confirmed"]:
        st.info("Unconfirmed: composites below are for visualisation only and no index will be computed yet.", icon=None)
    return mapping


def render_composite(
    image: Any,
    meta: Dict[str, Any],
    title: str,
    help_text: str,
    download_name: str,
    png_bytes: Optional[bytes] = None,
) -> None:
    """One rendered composite: image, then the numbers that explain it."""
    st.markdown(f"**{title}**")
    st.image(
        image,
        caption=f"{' + '.join(meta['channels'])}  ·  rendered {meta['display_shape'][0]}×{meta['display_shape'][1]} px "
                f"from {meta['source_shape'][0]}×{meta['source_shape'][1]} px "
                f"(1/{meta['decimation_factor']:.1f} decimation)",
        width="stretch",
    )
    with st.expander(f"{title} — details", expanded=False):
        st.markdown(help_text)
        st.json(meta, expanded=False)
        st.caption(
            f"Rendered transform (Affine): {[round(v, 4) for v in meta['transform']]}  ·  "
            f"nodata in view: {100 * meta['nodata_fraction']:.2f}%"
        )
        st.markdown("**Stretch applied (DN → 0–255)**")
        for ch, (lo, hi) in zip(meta["channels"], meta["stretch_bounds"]):
            st.caption(f"- {ch}: {lo:,.1f} → {hi:,.1f} (percentiles {meta['stretch_percentiles'][0]}–{meta['stretch_percentiles'][1]})")
        if png_bytes:
            st.download_button(
                f"Download {download_name}.png",
                data=png_bytes,
                file_name=f"{download_name}.png",
                mime="image/png",
                key=f"dl_{download_name}",
            )


def render_histograms(histograms: List[Dict[str, Any]]) -> None:
    """Per-band value distribution, computed from REAL decimated pixel values.

    (`histograms` comes from `core.preview.band_histograms` -- the bars below are
    measured counts, never a fitted curve.)
    """
    st.subheader("Value distribution (decimated)", divider="gray")
    st.caption(
        "Reflectance values cluster in a narrow low range with a long bright tail. "
        "That is why we stretch between percentiles instead of min and max: a min/max "
        "stretch is dominated by a handful of bright pixels and renders almost black."
    )
    for h in histograms:
        if not h.get("counts"):
            continue
        st.markdown(
            f"`{h['band_name']}` — median {h['p50']:.0f}, "
            f"p2–p98 = {h['p2']:.0f}–{h['p98']:.0f} DN  "
            f"({h['valid_pixels']:,} valid px sampled)"
        )
        st.bar_chart({"count": h["counts"]}, height=90)


# =========================================================================== #
# PHASE 3 -- NDVI result rendering
# =========================================================================== #
def render_reflectance_panel(spec: Dict[str, Any], report: Dict[str, Any]) -> None:
    """Show exactly how DN became reflectance, and how that was checked."""
    st.subheader("Reflectance preprocessing", divider="gray")

    if not spec.get("is_reflectance"):
        st.warning(
            "No reflectance scaling is known for this file, so values are treated as raw "
            "digital numbers. NDVI is a ratio, so a purely multiplicative scale cancels out "
            "— but an unknown additive offset would bias the result.",
            icon=None,
        )
    else:
        st.code(spec["label"], language="text")
        c1, c2, c3 = st.columns(3)
        c1.metric("Scale", f"{spec['scale']:g}")
        c2.metric("Offset", f"{spec['offset']:+g}")
        c3.metric("Source", spec["source"])

    if report.get("validated"):
        neg = report.get("negative_fraction")
        med = report.get("median_reflectance")
        bits = [f"{report.get('sample_pixels', 0):,} pixels sampled"]
        if neg is not None:
            bits.append(f"{100 * neg:.2f}% below {'-0.01'}")
        if med is not None:
            bits.append(f"median reflectance {med:.4f}")
        st.caption("Empirical validation — " + " · ".join(bits))
        if report.get("offset_rejected"):
            st.error(
                "A metadata offset was REJECTED by the physical sanity check: applying it "
                "made a large share of pixels negatively reflective, which is impossible. "
                "Offset forced to 0 after checking real pixel values.",
                icon=None,
            )
    else:
        st.caption("Reflectance scaling was not validated against pixel values.")

    with st.expander("How this scaling was chosen (evidence)", expanded=False):
        for e in spec.get("evidence", []):
            st.markdown(f"- {e}")
        for w in spec.get("warnings", []):
            st.markdown(f"- **Note:** {w}")
        st.json({"spec": spec, "validation": report}, expanded=False)


def render_ndvi_stats(result: Dict[str, Any]) -> None:
    """Counts first (always available), then statistics (only if valid pixels exist)."""
    c = result.get("counts", {})
    st.subheader("Pixel accounting", divider="gray")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total pixels", f"{c.get('total_pixels', 0):,}")
    a2.metric("Valid", f"{c.get('valid_pixels', 0):,}")
    a3.metric("Invalid / masked", f"{c.get('invalid_pixels', 0):,}")
    # 4 decimals: rounding 99.9996% to "100.00%" would contradict the invalid count.
    a4.metric("Valid %", f"{c.get('valid_percentage', 0):.4f}%")
    st.caption(
        "Invalid = nodata, NaN, ±infinity, or an undefined ratio (|NIR + Red| below the "
        "denominator guard). Invalid pixels are stored as NaN and are **never** counted as "
        "zero NDVI, which would read as bare rock."
    )

    st.subheader("NDVI statistics", divider="gray")
    s = result.get("stats")
    if not s:
        st.error(
            "No valid pixels — **no statistics are reported**. Reporting a mean of 0.0 here "
            "would be fabricated data.",
            icon=None,
        )
        return

    # Labels are prefixed: the Phase 1 sanity check also shows min/mean/max, and
    # duplicated metric labels make the two blocks impossible to tell apart.
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("NDVI min", f"{s['min']:.4f}")
    b2.metric("NDVI max", f"{s['max']:.4f}")
    b3.metric("NDVI mean", f"{s['mean']:.4f}")
    b4.metric("NDVI median", f"{s['median']:.4f}")
    b5.metric("NDVI std dev", f"{s['std']:.4f}")
    b6.metric("NDVI valid px", f"{s['valid_pixels']:,}")

    pct = s.get("percentiles", {})
    if pct:
        st.caption(
            "Percentiles (valid pixels only): "
            + " · ".join(f"{k} = {v:.3f}" for k, v in pct.items())
        )
    st.caption(
        "Statistics are computed at **native resolution** over valid pixels only. "
        "These are continuous NDVI values; no vegetation-health threshold is implied."
    )


def render_ndwi_stats(result: Any, provenance: Optional[Dict[str, Any]] = None) -> None:
    """Phase 11: NDWI statistics for the selected area.

    Mirrors `render_ndvi_stats` in layout, because the two are the same kind of
    answer: counts first, then statistics, then the caveat. The caveat is
    mandatory -- an NDWI number invites exactly the claims Phase 11 does not
    make.
    """
    d = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
    prov = provenance or {}
    index = prov.get("index") or {}
    st.subheader("NDWI — water index", divider="gray")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Cells in selection", f"{d.get('pixels_inside_roi', 0):,}")
    a2.metric("Valid cells", f"{d.get('valid_pixels', 0):,}")
    a3.metric("Invalid / masked", f"{d.get('invalid_pixels', 0):,}")
    a4.metric("Valid %", f"{100 * d.get('valid_fraction', 0):.4f}%")
    st.caption(
        "Invalid = nodata, NaN, ±infinity, or an undefined ratio (|Green + NIR| "
        "below the denominator guard). Invalid pixels are never counted as a "
        "value, which would read as a surface measurement."
    )

    stats = d.get("stats")
    if not stats:
        st.error("No valid pixels — **no statistics are reported**. Reporting a "
                 "mean of 0.0 here would be fabricated data.", icon=None)
        return

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("NDWI min", f"{stats['min']:.4f}")
    b2.metric("NDWI max", f"{stats['max']:.4f}")
    b3.metric("NDWI mean", f"{stats['mean']:.4f}")
    b4.metric("NDWI median", f"{stats['median']:.4f}")
    b5.metric("NDWI std dev", f"{stats['std']:.4f}")

    pct = stats.get("percentiles", {})
    if pct:
        st.caption("Percentiles (valid pixels only): "
                   + " · ".join(f"{k} = {v:.3f}" for k, v in pct.items()))
    st.caption(
        f"Computed at **native resolution** ({prov.get('native_resolution_m', ['?'])[0]} m) "
        f"over the valid pixels inside the selection. Continuous index values — "
        f"**no water / non-water threshold is applied**."
    )

    # The caveat is not optional and is not a footnote.
    st.warning(
        "NDWI is a spectral index. This result does not by itself establish "
        "flood extent, water availability, or water quality.", icon=None)

    with st.expander("How this index was computed", expanded=False):
        if index.get("formula"):
            st.markdown(f"**Formula.** NDWI = {index['formula']}"
                        + (f"  — {index['citation']}" if index.get("citation") else ""))
        bands = prov.get("bands") or {}
        if bands:
            st.markdown("**Bands.** "
                        + ", ".join(f"{role} = band {b.get('index')}"
                                    f" ({b.get('band_id') or b.get('name')})"
                                    for role, b in bands.items()))
        refl = prov.get("reflectance") or {}
        if refl:
            st.markdown(f"**Reflectance.** DN × {refl.get('scale'):g}"
                        f"{(' + ' + str(refl.get('offset'))) if refl.get('offset') else ''}"
                        f" (source: {refl.get('source')}).")
        st.markdown(f"**Denominator guard.** |Green + NIR| "
                    f"{index.get('min_denominator', 1e-6):g} → undefined, not zero.")
        if prov.get("roi_window"):
            st.markdown(f"**ROI window** (col, row, width, height): "
                        f"{prov['roi_window']} — only this window was read.")
        if prov.get("role_resolution"):
            st.markdown(f"**Band-role confidence:** "
                        f"{prov['role_resolution'].get('confidence')}.")
        for item in (index.get("limitations") or []):
            st.markdown(f"- {item}")


def render_ndvi_change(result: Any, warnings: Any = ()) -> None:
    """Phase 10: the textual summary that MUST accompany the change map.

    A map of red and green areas invites the reader to invent a cause. This
    panel is the counterweight: it reports the index, states the threshold that
    produced the classes, and says plainly that the cause is not established.
    """
    d = result.to_dict()
    st.subheader(f"NDVI change — {d['before_date']} → {d['after_date']}",
                 divider="gray")

    # -- the numbers -------------------------------------------------------- #
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("NDVI before (mean)", f"{d['before_mean']:.4f}")
    a2.metric("NDVI after (mean)", f"{d['after_mean']:.4f}")
    a3.metric("ΔNDVI (mean)", f"{d['delta_mean']:+.4f}")
    a4.metric("ΔNDVI (median)", f"{d['delta_median']:+.4f}")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Cells compared", f"{d['valid_pixel_count']:,}")
    b2.metric("Increase", f"{d['increased_pct']:.1f}%")
    b3.metric("Stable", f"{d['stable_pct']:.1f}%")
    b4.metric("Decrease", f"{d['decreased_pct']:.1f}%")

    st.caption(
        f"Compared on {d['valid_pixel_count']:,} of {d['roi_cell_count']:,} cells in "
        f"the selected area ({d['valid_fraction_of_roi'] * 100:.1f}%); "
        f"{d['insufficient_count']:,} cells did not have a valid NDVI value on both "
        f"dates and are reported as insufficient data, never as unchanged."
    )

    # -- what changed, in words --------------------------------------------- #
    direction = d.get("net_direction")
    if direction == "increase":
        st.success("Vegetation index increased.", icon=None)
    elif direction == "decrease":
        st.warning("Vegetation index decreased. The observed change is consistent "
                   "with reduced vegetation signal.", icon=None)
    elif direction == "stable":
        st.info("Vegetation index showed no change above the display threshold.",
                icon=None)
    else:
        st.info("Change is mixed: increases and decreases are both present.",
                icon=None)

    # -- classification, with its threshold --------------------------------- #
    thr = d.get("thresholds", {})
    rows = [
        {"Class": name, "Cells": f"{d[key + '_count']:,}",
         "Share of compared": f"{d[key + '_pct']:.1f}%"}
        for key, name in (("increased", "Increase"), ("stable", "Stable"),
                          ("decreased", "Decrease"))
    ]
    rows.append({"Class": "Insufficient data", "Cells": f"{d['insufficient_count']:,}",
                 "Share of compared": "— (excluded)"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        f"Classification rule: ΔNDVI ≥ +{thr.get('increase', 0.1):g} → Increase; "
        f"ΔNDVI ≤ −{thr.get('decrease', 0.1):g} → Decrease; otherwise Stable. "
        f"{thr.get('rationale', '')} ΔNDVI = NDVI_after − NDVI_before."
    )

    # -- the causal caveat is not optional ---------------------------------- #
    st.error(
        "Additional data is required to identify the cause. An NDVI difference does "
        "not by itself establish deforestation, crop failure, flooding, drought or "
        "any other specific event — only that the measured index changed.",
        icon=None,
    )

    # -- how the two scenes were compared ----------------------------------- #
    alignment = d.get("alignment", {})
    prov = d.get("provenance", {})
    with st.expander("How this comparison was computed", expanded=False):
        st.markdown(
            f"**Acquisitions.** Before: {d['before_scene']} "
            f"({prov.get('scenes', {}).get('before', {}).get('source', {}).get('platform', 'unknown')}, "
            f"{d['before_date']}). After: {d['after_scene']} "
            f"({prov.get('scenes', {}).get('after', {}).get('source', {}).get('platform', 'unknown')}, "
            f"{d['after_date']})."
        )
        st.markdown(
            f"**Grid.** {alignment.get('method', 'unknown')} — "
            f"{alignment.get('resolution', 0):g} m cells, "
            f"resampling: {alignment.get('resampling') or 'none (native grid)'}." + "\n\n" +
            f"{alignment.get('note', '')}"
        )
        st.markdown(
            f"**Formulas.** NDVI = {prov.get('formula_ndvi', '(NIR − RED) / (NIR + RED)')}; "
            f"ΔNDVI = {prov.get('formula_delta', 'NDVI_after − NDVI_before')}. "
            f"Reflectance = DN × {prov.get('reflectance', {}).get('before', {}).get('scale', 0.0001):g}."
        )
        st.markdown(
            f"**Runtime.** {prov.get('runtime_ms', 0):g} ms over "
            f"{prov.get('roi_cells', 0):,} cells."
        )
        for line in d.get("limitations", []):
            st.markdown(f"- {line}")

    for warning in (warnings or ()):
        if warning:
            st.warning(str(warning), icon=None)


def render_ndvi_classes(classification: Optional[Dict[str, Any]]) -> None:
    if not classification:
        return
    st.warning(classification["caveat"], icon=None)
    rows = [
        {
            "Illustrative class": c["name"],
            "NDVI range": c["range"],
            "Share of valid pixels": f"{100 * c['share']:.1f}%",
        }
        for c in classification["classes"]
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def render_ndvi_figure(
    fig: Any,
    png_bytes: bytes,
    native_shape: Tuple[int, int],
    display_shape: Tuple[int, int],
    decimation: int,
) -> None:
    st.pyplot(fig, width="stretch")
    st.caption(
        f"Analysis array: **{native_shape[0]}×{native_shape[1]} px at native resolution**. "
        f"Rendered here at {display_shape[0]}×{display_shape[1]} px (1/{decimation} block "
        f"average, display only) with its own transform. Invalid pixels are transparent."
    )
    st.download_button(
        "Download NDVI map (PNG)",
        data=png_bytes,
        file_name="satquery_ndvi.png",
        mime="image/png",
        width="stretch",
    )


def render_ndvi_provenance(result: Dict[str, Any]) -> None:
    prov = result.get("provenance") or {}
    spec = result.get("reflectance") or {}
    st.subheader("Provenance", divider="gray")
    rows = [
        {"Field": "Analysis", "Value": result.get("label") or result.get("name", "")},
        {"Field": "Method", "Value": result.get("method", "")},
        {"Field": "Red band", "Value": str(result.get("bands_used", {}).get("red", "—"))},
        {"Field": "NIR band", "Value": str(result.get("bands_used", {}).get("nir", "—"))},
        {"Field": "Reflectance", "Value": spec.get("label", "raw DN")},
        {"Field": "Native CRS", "Value": str(prov.get("raster", {}).get("crs", "—"))},
        {"Field": "Dimensions", "Value": str(result.get("native_shape", "—"))},
        {"Field": "Dataset", "Value": str(prov.get("dataset", "—"))},
        {"Field": "Acquisition", "Value": str(prov.get("datetime", "—"))},
    ]
    st.dataframe(rows, hide_index=True, width="stretch")

    for cav in result.get("caveats", []):
        st.caption("Caveat — " + cav)


# --------------------------------------------------------------------------- #
# Phase 5 -- ROI panel
# --------------------------------------------------------------------------- #
def render_roi_panel(selection: Optional[Any], stale_map: bool = False) -> None:
    """Show the state of the user's area selection.

    Takes the `core.roi.ROISelection` object (or None). Deliberately says
    nothing about crop suitability, vegetation health or any other analysis:
    Phase 5 only captures geometry.
    """
    from core.geometry import format_area

    if selection is None:
        st.info("Draw a rectangle or polygon on the map to select an area for analysis.",
                icon=None)
        st.caption(
            "Use **Area** on the globe and click the corners of the region you "
            "want to analyse. The selection is converted from map coordinates "
            "(EPSG:4326) into the raster's own CRS before anything is measured."
        )
        return

    data = selection.to_dict()

    if not data["is_valid"]:
        st.error("Selection could not be used. Please draw a valid polygon or rectangle.",
                 icon=None)
        if data["message"]:
            st.caption(data["message"])
        for w in data["warnings"]:
            st.caption(f"- {w}")
        return

    if not data["intersects_raster"]:
        st.error(OUTSIDE_MESSAGE, icon=None)
        for w in data["warnings"]:
            st.caption(f"- {w}")
        return

    st.success("Selection detected — this region is recorded for later analysis.", icon=None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Geometry", data["geometry_type"])
    c2.metric("Usable area", data["area_display"])
    hec = data["area_hectares"]
    km2 = data["area_km2"]
    c3.metric("Hectares", f"{hec:,.3f}" if hec < 1000 else f"{hec:,.1f}")
    c4.metric("km²", f"{km2:,.4f}" if km2 < 10 else f"{km2:,.2f}")

    st.caption(
        f"Approximate area ({data['area_method']}) · parts: {data['num_parts']} · "
        f"{data['overlap_fraction'] * 100:.1f}% of the drawn shape is inside the raster · "
        f"geometry stored in {data['raster_crs']} (the raster's own CRS), ready for analysis."
    )

    if data["was_clipped"]:
        st.warning(
            "Only the portion inside the available raster will be used for later analysis.",
            icon=None,
        )
    for w in data["warnings"]:
        st.caption(f"- {w}")

    if stale_map:
        st.warning(
            "The map was redrawn, so the outline may no longer be shown on it. The "
            "selection above is still recorded — redraw it, or press **Clear selection**.",
            icon=None,
        )

    with st.expander("Selection geometry (GeoJSON)", expanded=False):
        st.caption("As drawn (EPSG:4326, for display and provenance):")
        st.json(data["original_geometry"], expanded=False)
        st.caption(f"Usable part, in {data['raster_crs']} (what the analysis will read):")
        st.json(data["geometry_raster_crs"], expanded=False)

    st.caption(
        "No statistics are computed here. The analysis rasterises this geometry "
        "against the native pixel grid and summarises the pixels inside it."
    )


# --------------------------------------------------------------------------- #
# Phase 6 -- ROI NDVI analysis
# --------------------------------------------------------------------------- #
def render_roi_analysis(result: Optional[Dict[str, Any]],
                        histogram: Optional[Dict[str, Any]] = None) -> None:
    """Statistics of the NATIVE NDVI inside the drawn region.

    Deliberately says nothing about crop health or suitability: Phase 6 proves
    that a selected region can be measured, not that it can be judged.
    """
    import pandas as pd

    if result is None:
        return
    if not result.get("stats"):
        st.warning(result.get("message") or "No valid NDVI pixels were found inside the selected area.",
                   icon=None)
        st.caption(
            f"Pixels geometrically inside the selection: "
            f"{result.get('pixels_inside_roi', 0):,}. All of them are nodata or "
            "otherwise invalid, so no value is reported — a zero here would be "
            "fabricated, not measured."
        )
        for w in result.get("warnings", []):
            st.caption(f"- {w}")
        return

    stats = result["stats"]
    pct = stats["percentiles"]

    st.subheader("ROI NDVI Analysis", divider="gray")
    st.caption(
        f"Area: **{result['area_m2'] / 10_000:,.3f} ha** "
        f"({result['area_km2']:,.4f} km²) · "
        f"native resolution: **{result['pixel_width']:,.2f} m × {result['pixel_height']:,.2f} m** "
        f"({result['pixel_area_m2']:,.1f} m² per pixel) · "
        f"{result['crs']}"
    )

    st.markdown("**Pixels**")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Inside ROI", f"{result['pixels_inside_roi']:,}")
    p2.metric("Valid NDVI", f"{result['valid_pixels']:,}")
    p3.metric("Invalid / nodata", f"{result['invalid_pixels']:,}")
    p4.metric("Valid %", f"{100 * result['valid_fraction']:.2f}%")
    st.caption(
        "“Inside ROI” counts pixel centres geometrically inside the drawn polygon. "
        "“Valid NDVI” is the subset that also has a usable value — the two are "
        "different numbers on purpose."
    )

    st.markdown("**NDVI (valid pixels only)**")
    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("Mean", f"{stats['mean']:.4f}")
    n2.metric("Median", f"{stats['median']:.4f}")
    n3.metric("Min", f"{stats['min']:.4f}")
    n4.metric("Max", f"{stats['max']:.4f}")
    n5.metric("Std dev", f"{stats['std']:.4f}")

    st.markdown("**Percentiles**")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("P5", f"{pct['p5']:.4f}")
    q2.metric("P25", f"{pct['p25']:.4f}")
    q3.metric("P75", f"{pct['p75']:.4f}")
    q4.metric("P95", f"{pct['p95']:.4f}")

    st.info(
        f"Selected area has a mean NDVI of **{stats['mean']:.2f}** "
        f"(median {stats['median']:.2f}) over {result['valid_pixels']:,} valid pixels. "
        f"Values range from {stats['min']:.2f} to {stats['max']:.2f}. "
        "This is a measurement of the selected pixels only — it is not a health, "
        "yield or suitability judgement.",
        icon=None,
    )

    if histogram and histogram.get("counts"):
        st.markdown("**NDVI distribution inside the ROI**")
        df = pd.DataFrame({"NDVI": histogram["centers"], "pixels": histogram["counts"]}).set_index("NDVI")
        st.bar_chart(df, width="stretch")
        st.caption(
            "Counts of VALID NDVI pixels inside the selection, over the full "
            "theoretical range (-1 to +1). Invalid pixels are excluded, not "
            "bucketed at zero. No vegetation-health thresholds are applied."
        )

    for w in result.get("warnings", []):
        st.caption(f"- {w}")
    st.caption(
        "Computed from the native analysis raster at full resolution. The map "
        "layer below is a resampled copy for display only and is never used here."
    )


# --------------------------------------------------------------------------- #
# Phase 7 -- the SatQuery query interface
# --------------------------------------------------------------------------- #
#: Intent codes are internal; users read what the analysis actually did.
INTENT_LABELS: Dict[str, str] = {
    "NDVI_ROI_STATS": "NDVI statistics for the selected area",
    "NDWI_ROI_STATS": "NDWI statistics for the selected area",
    "NDVI_CHANGE_ROI": "NDVI change between two acquisitions",
    "TEMPORAL_COMPARISON": "Comparison between two acquisitions",
    "VEGETATION_CHANGE": "Vegetation change between two acquisitions",
    "CROP_SUITABILITY": "Crop-suitability screening",
    "SPATIAL_QUERY": "Spatial conditions",
    "MULTI_CONDITION": "Combined geographic conditions",
    "FLOOD_CHANGE": "Not available",
    "TEMPORAL_NDWI": "Not available",
    "UNKNOWN": "Question not recognised",
    "UNSUPPORTED": "Not available",
}


def intent_label(intent: str) -> str:
    """The intent, in the user's words. Unknown codes are shown plain."""
    return INTENT_LABELS.get(str(intent), str(intent).replace("_", " ").title())


def render_ask_satquery(hint: str = "") -> Optional[str]:
    """Render the query box; return the submitted query (or None).

    The wording is generated from the analysis registry, so a future engine
    advertises itself here without any change to this function.
    """
    if hint:
        st.caption(hint)
    return st.chat_input("Ask SatQuery — e.g. “What is the NDVI of this area?”")


def render_welcome(examples: Sequence[str]) -> None:
    """The first screen: what this product does, and how to begin.

    Deliberately narrow: it advertises the analyses that are actually
    registered, and it says what SatQuery does not do -- no real-time imagery,
    no prediction, no causal claims.
    """
    import streamlit as _st

    _st.markdown(
        """
        <div class="sq-cards">
          <div class="sq-card">
            <div class="sq-card-title">Explore satellite imagery</div>
            <div class="sq-card-body">Load a GeoTIFF or COG and inspect
            true-colour and false-colour composites, with reflectance-aware
            stretching.</div>
          </div>
          <div class="sq-card">
            <div class="sq-card-title">Inspect spectral indicators</div>
            <div class="sq-card-body">NDVI and NDWI are computed on the native
            raster as continuous measurements — no hidden thresholds.</div>
          </div>
          <div class="sq-card">
            <div class="sq-card-title">Compare temporal changes</div>
            <div class="sq-card-body">Choose two dated acquisitions and quantify
            change cell by cell over the interval you select.</div>
          </div>
          <div class="sq-card">
            <div class="sq-card-title">Evaluate spatial conditions</div>
            <div class="sq-card-body">Ask about distance, land cover and index
            ranges, and see exactly which cells qualified.</div>
          </div>
          <div class="sq-card">
            <div class="sq-card-title">Combine multiple conditions</div>
            <div class="sq-card-body">Compose several conditions into one answer,
            with each condition reported on its own.</div>
          </div>
          <div class="sq-card">
            <div class="sq-card-title">Understand evidence and limits</div>
            <div class="sq-card-body">Every result states what was measured, what
            was assumed, what is unknown, and what it does not claim.</div>
          </div>
        </div>
        <div class="sq-note"><strong>Scope.</strong> SatQuery analyses the
        satellite raster you load into it. It does not provide real-time imagery,
        and its results are geographic evidence — not predictions, yield
        estimates or causal explanations.</div>
        """,
        unsafe_allow_html=True,
    )

    if examples:
        _st.caption("Try one of these:")
        cols = _st.columns(min(3, max(1, len(examples))))
        for i, example in enumerate(examples):
            with cols[i % len(cols)]:
                if _st.button(example, key=f"sq_example_{i}", width="stretch"):
                    _st.session_state["sq_pending_query"] = example
                    _st.rerun()


def render_answer(entry: Dict[str, Any]) -> None:
    """Render one turn: the question, the routed analysis, and the answer.

    Everything shown here comes from the structured `AnalysisExecution`; nothing
    is scraped from another widget, and no number is computed in this function.
    """
    with st.chat_message("user"):
        st.markdown(entry.get("query", ""))

    with st.chat_message("assistant"):
        intent = entry.get("intent", "UNKNOWN")
        confidence = entry.get("confidence", 0.0)
        st.caption(f"**{intent_label(intent)}** · routing confidence {confidence:.2f}")
        if entry.get("explanation"):
            st.caption(entry["explanation"])

        status = entry.get("status", "")
        message = entry.get("message", "")
        if status == "OK":
            st.success(message, icon=None)
        elif status in ("NEEDS_ROI", "NEEDS_NDVI_CONFIRMATION",
                        "NEEDS_THRESHOLD", "NEEDS_TWO_DATES"):
            # A refusal that names what to do next is a request, not a failure
            st.warning(message, icon=None)
        elif status in ("UNSUPPORTED", "UNKNOWN"):
            st.info(message, icon=None)
        else:
            st.error(message or "The analysis did not complete.", icon=None)

        for warning in entry.get("warnings", []) or []:
            st.caption(f"- {warning}")


# --------------------------------------------------------------------------- #
# Phase 8 -- crop-suitability screening result
# --------------------------------------------------------------------------- #
def render_crop_suitability(screening: Any) -> None:
    """Render an experimental crop-suitability screening.

    Everything shown is read from the structured result object -- nothing is
    recomputed here, and no number is invented. The panel is deliberately
    shaped as an explanation with caveats, not as a certificate.
    """
    import pandas as pd

    scenarios = getattr(screening, "scenarios", {}) or {}
    if not scenarios:
        return

    st.caption(
        "Experimental crop-suitability screening. It is not a crop "
        "recommendation, a yield prediction, a soil diagnosis or an irrigation "
        "statement, and it must be validated with local agronomic and field "
        "information."
    )

    tabs = st.tabs([_scenario_tab_label(name, res) for name, res in scenarios.items()])
    for tab, (name, res) in zip(tabs, scenarios.items()):
        with tab:
            _render_one_scenario(name, res, pd)

    _render_explanation(screening)


def _scenario_tab_label(name: str, res: Any) -> str:
    label = str(getattr(res, "scenario", name)).replace("_", " ")
    return f"{label.title()} · {getattr(res, 'classification', '')}"


def _render_one_scenario(name: str, res: Any, pd) -> None:
    if getattr(res, "assumed_factors", None):
        st.warning(
            "Hypothetical sensitivity analysis. Under the hypothetical "
            "assumption that adequate irrigation water is continuously "
            "available, the water factor is removed and the remaining factors "
            "are re-weighted. This is NOT a suitability assessment of the land: "
            "the dominant real constraint (water) is assumed away and salinity "
            "was not assessed.",
            icon=None,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Screening class", str(getattr(res, "classification", "—")))
    c2.metric("Score", "—" if res.score is None else f"{res.score:.2f}")
    c3.metric("Confidence", str(getattr(res, "confidence", "—")))

    # ---- computed limiting factors ---------------------------------------- #
    limiting = list(getattr(res, "computed_limiting_factors", []) or [])
    if limiting:
        st.markdown("**Computed limiting factors**")
        for item in limiting:
            value = item.get("value")
            unit = item.get("unit") or ""
            optimum = item.get("optimum_range") or [None, None]
            st.write(
                f"- **{item.get('label', item.get('factor'))}** — "
                f"{value if value is None else f'{value:,.2f}'} {unit} "
                f"(optimum {optimum[0]}–{optimum[1]} {unit}, "
                f"membership {item.get('membership')})."
            )
    elif res.score is not None:
        st.markdown("**Computed limiting factors**")
        st.caption("No computed factor was limiting at this location.")

    # ---- water: seasonal and annual, never swapped ------------------------ #
    if res.growing_season_precipitation is not None:
        months = res.growing_season_months or []
        st.markdown("**Water**")
        st.caption(
            f"Growing-season precipitation (months {months[0]}–{months[-1]}) is "
            f"**{res.growing_season_precipitation:,.0f} mm**; the annual total is "
            f"**{res.annual_precipitation:,.0f} mm**. The seasonal figure is the "
            "one that is scored — cotton cannot use rain that falls outside its "
            "growing season."
        )

    # ---- factor table ------------------------------------------------------ #
    rows = []
    for f in getattr(res, "factors", []) or []:
        rows.append({
            "Factor": f.label,
            "Category": f.category.replace("_", " "),
            "Value": "—" if f.value is None else f"{f.value:,.2f}",
            "Unit": f.unit or "—",
            "Membership": "—" if f.membership is None else f"{f.membership:.2f}",
            "Status": f.status,
        })
    if rows:
        st.markdown("**Factors**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Membership 1.00 = inside the configured optimum, 0.00 = outside the "
            "absolute tolerance. “assumed”/“missing” factors carry no score."
        )

    # ---- limitations that are NOT computed limitations --------------------- #
    assumptions = [a for a in getattr(res, "assumptions", []) or []]
    if assumptions:
        st.markdown("**Not assessed / assumed**")
        for a in assumptions:
            st.caption(f"- {a}")

    fractions = getattr(res, "class_fractions", {}) or {}
    if fractions:
        st.markdown("**Area by class**")
        st.dataframe(
            pd.DataFrame([{"Class": k, "Share of ROI": f"{v:.1%}"}
                          for k, v in fractions.items() if v > 0]),
            hide_index=True, width="stretch")


def _render_explanation(screening: Any) -> None:
    import pandas as pd

    with st.expander("Why? — how this score was produced", expanded=False):
        st.markdown(
            "**Model.** Each factor is converted to a membership between 0 and 1 "
            "with a trapezoidal function (0 outside the absolute range, 1 inside "
            "the optimum, linear ramps between). The score is the weighted mean "
            "of the available memberships. Four separate gates then apply, in "
            "this order:"
        )
        st.markdown(
            "1. **Hard land-cover constraint** — built-up, permanent water, "
            "mangroves, snow/ice, moss and herbaceous wetland are exclusions, "
            "not scores.\n"
            "2. **Critical-factor veto** — a critical factor (temperature, water, "
            "pH) outside its absolute tolerance forces *Unsuitable* regardless of "
            "the score.\n"
            "3. **Missing critical factor** — no score at all; a zero is never "
            "substituted.\n"
            "4. **Assumed mandatory factor** — the hypothetical irrigation "
            "scenario is capped at *Moderately suitable* with Low confidence.\n"
        )
        st.caption(
            "There is no automatic “weakest factor demotes the class” rule: the "
            "score is the model, and the limiting factors are reported so you can "
            "see what drove it."
        )
        thresholds = list(getattr(next(iter(screening.scenarios.values())),
                                  "threshold_provenance", []) or [])
        if thresholds:
            st.markdown("**Thresholds and weights**")
            st.dataframe(
                pd.DataFrame([{
                    "Item": t.get("item"),
                    "Status": str(t.get("threshold_status", "")).upper(),
                    "Values": str(t.get("threshold_values", {})),
                    "Source": str(t.get("threshold_source", ""))[:220],
                } for t in thresholds]),
                hide_index=True, width="stretch")
            st.caption(
                "“EXPERIMENTAL” means the value was constructed for this "
                "screening and is not literature-derived. “LITERATURE-BACKED” "
                "means it comes from the cited source for cotton."
            )

    with st.expander("Data & methodology", expanded=False):
        records = list(getattr(next(iter(screening.scenarios.values())),
                               "provenance", []) or [])
        if records:
            st.markdown("**Datasets**")
            st.dataframe(
                pd.DataFrame([{
                    "Dataset": r.get("dataset"),
                    "Variable": r.get("variable"),
                    "Native res.": r.get("native_resolution"),
                    "Native CRS": r.get("native_crs"),
                    "Units": r.get("units"),
                    "Period": r.get("temporal_period"),
                    "Resampling": r.get("resampling"),
                    "Licence": r.get("license"),
                    "Accessed": r.get("access_date"),
                } for r in records]),
                hide_index=True, width="stretch")
        grid = getattr(screening, "grid", None)
        if grid is not None:
            st.markdown("**Grid**")
            st.caption(
                f"{grid.width} × {grid.height} cells at {grid.resolution:g} m in "
                f"{grid.crs}"
                + (f" — {grid.note}" if grid.note else "")
            )
        st.markdown("**Temporal basis**")
        st.caption(
            "Climatological + static (mixed): 1970–2000 climate normals, a static "
            "soil model, a static DEM and a 2021 land-cover epoch. Not a forecast "
            "for any specific season or year, and not real-time."
        )
        perf = getattr(screening, "performance", {}) or {}
        if perf:
            st.markdown("**Timings (seconds)**")
            st.dataframe(
                pd.DataFrame([{"Stage": k, "Seconds": v}
                              for k, v in perf.items()]),
                hide_index=True, width="stretch")

    st.error(
        "Experimental screening — not a crop recommendation, yield prediction, "
        "soil diagnosis or irrigation-availability statement. Thresholds and "
        "weights are experimental. Soil salinity, irrigation availability, soil "
        "depth and drainage were NOT assessed. Validate with local agronomic and "
        "field information.",
        icon=None,
    )


# =========================================================================== #
# Phase 9 -- the spatial-query result panel
# =========================================================================== #
#: What each result state means, in the user's language.
SPATIAL_STATE_HEADLINES: Dict[str, Tuple[str, str]] = {
    "ok": ("Match", "The requested conditions are satisfied somewhere in "
                    "the selected area."),
    "zero_matches": ("No match", "The query ran on every analysed cell and "
                                 "none satisfied all conditions. That is a "
                                 "result, not an error."),
    "insufficient_data": ("Insufficient data", "The required source data was "
                                               "missing or undecidable, so no "
                                               "conclusion is reported."),
}

SPATIAL_LIMITATIONS: Tuple[str, ...] = (
    "Permanent-water proximity is **not** irrigation availability.",
    "Permanent-water proximity is **not** groundwater access.",
    "Permanent-water proximity is **not** a flood-risk assessment.",
    "Wetland (WorldCover class 90) is **not** treated as permanent water.",
    "Cotton suitability is the **rainfed** scenario; irrigation is not "
    "modelled.",
    "The current sample AOIs contain no cotton cells reaching class ≥ 3, so "
    "cotton queries here return zero matches.",
    "Zero matches do **not** imply that no such land exists outside the "
    "selected area.",
    "External raster coverage and nodata can produce insufficient-data regions.",
    "Water semantics come from WorldCover class 80 (permanent water, 2021).",
    "JRC Global Surface Water is an independent cross-check, not the production "
    "water engine.",
)


def render_spatial_query(result: Any) -> None:
    """Render a multi-condition spatial-query result.

    Everything comes from the structured result object: nothing is recomputed
    and no number is invented. The panel must make four outcomes
    distinguishable at a glance: MATCH, NO MATCH, INSUFFICIENT DATA and
    UNSUPPORTED.
    """
    icon, headline = SPATIAL_STATE_HEADLINES.get(
        str(getattr(result, "status", "")),
        ("ℹ️", "Result available."))

    st.caption(
        "Experimental multi-condition spatial query. It is a screening of "
        "mapped datasets, not a recommendation, not irrigation information, "
        "and not a flood or groundwater assessment.")

    # ---- 1. interpretation: what was actually understood ------------------ #
    st.markdown("**Interpreted as**")
    for condition in getattr(result, "conditions", ()) or ():
        _render_condition_line(condition)
    operator = str(getattr(getattr(result, "operator", "and"), "value", "and"))
    st.caption(f"Operator: **{operator.upper()}** — every condition must hold."
               if operator == "and" else
               f"Operator: **{operator.upper()}** — any condition may hold.")

    # ---- 2. the result state ---------------------------------------------- #
    st.markdown(f"**Result** — {icon} {headline}")

    # ---- 3. result summary ------------------------------------------------ #
    counts = [
        ("Matching cells", f"{int(getattr(result, 'matched_cell_count', 0)):,}"),
        ("Matched area", f"{float(getattr(result, 'matched_area_m2', 0.0)) / 1e6:,.3f} km²"),
        ("Matched fraction", f"{float(getattr(result, 'matched_fraction', 0.0)) * 100:.1f}%"),
    ]
    columns = st.columns(3)
    for column, (label, value) in zip(columns, counts):
        column.metric(label, value)

    insufficient = int(getattr(result, "insufficient_cell_count", 0) or 0)
    if insufficient:
        st.caption(
            f"{insufficient:,} cells "
            f"({float(getattr(result, 'insufficient_fraction', 0.0)) * 100:.1f}%) "
            "could not be established from the available data; they are "
            "reported as insufficient and are **not** counted as non-matching.")

    resolution = getattr(result, "analysis_resolution", None)
    st.caption(
        f"Analysis resolution: **{resolution:g} m** per cell"
        + (f" · {result.effective_resolution_note}"
           if getattr(result, "effective_resolution_note", "") else "")
        if resolution else "")

    # ---- 4. per-condition evidence ---------------------------------------- #
    condition_results = getattr(result, "condition_results", None) or []
    if condition_results:
        with st.expander("Why this result — per-condition evidence", expanded=False):
            for entry in condition_results:
                cells = (entry.get("counts") or {}).get("matching_cells", 0)
                st.markdown(
                    f"**{entry.get('name', 'condition')}** — "
                    f"{int(cells):,} matching cells"
                    + (" *(negated)*" if entry.get("negated") else ""))
                if entry.get("interpretation"):
                    st.caption(str(entry["interpretation"]))
            _render_why(result)

    # ---- 5. methodology + limitations ------------------------------------- #
    with st.expander("Data & methodology", expanded=False):
        _render_methodology(result)
    with st.expander("Limitations", expanded=False):
        for item in SPATIAL_LIMITATIONS:
            st.markdown(f"- {item}")


def _render_condition_line(condition: Any) -> None:
    """One understood condition, with the numbers that make it checkable."""
    from core.spatial_query import ConditionType

    kind = getattr(condition, "condition_type", None)
    parameters = dict(getattr(condition, "parameters", {}) or {})
    negate = bool(getattr(condition, "negate", False))
    prefix = "NOT " if negate else ""

    if kind is ConditionType.CROP_SUITABILITY:
        st.markdown(
            f"{prefix} **Cotton suitability** — rainfed screening class ≥ "
            f"{parameters.get('min_class', 3)}")
        st.caption(f"Scenario: **{parameters.get('scenario', 'rainfed')}** "
                   f"(data-backed), crop: {parameters.get('crop', 'cotton')}.")
    elif kind is ConditionType.LAND_COVER_CLASS:
        classes = parameters.get("classes") or []
        st.markdown(
            f"{prefix} **Land cover** — WorldCover class "
            f"{', '.join(str(c) for c in classes)}"
            + (f" ({parameters.get('class_name', '')})"
               if parameters.get("class_name") else ""))
    elif kind is ConditionType.WATER:
        st.markdown(
            f"{prefix} **Permanent water** — WorldCover class "
            f"{', '.join(str(c) for c in (parameters.get('classes') or [80]))}")
        st.caption("The cell itself is mapped as permanent water. Wetland "
                   "(class 90) is not water.")
    elif kind is ConditionType.WATER_PROXIMITY:
        st.markdown(
            f"{prefix} **Near permanent water** — within "
            f"{float(parameters.get('distance_m', 0)):,.0f} m")
        st.caption("Distance from the cell centre to the nearest WorldCover "
                   "class-80 cell, measured in the projected CRS.")
    else:
        st.markdown(f"{prefix} {getattr(condition, 'label', str(condition))}")


def _render_why(result: Any) -> None:
    """Why the result came out this way, in geographically precise words."""
    from core.spatial_query import ConditionType

    st.markdown("**Why**")
    conditions = list(getattr(result, "conditions", ()) or ())
    cotton = [c for c in conditions
              if getattr(c, "condition_type", None) is ConditionType.CROP_SUITABILITY]
    if cotton and int(getattr(result, "matched_cell_count", 0)) == 0:
        st.caption(
            "No analysed cells in the selected ROI reached the cotton "
            f"suitability threshold of class "
            f"{dict(cotton[0].parameters).get('min_class', 3)} or higher under "
            "the rainfed scenario, so the combined result has no matching "
            "cells. The query itself executed successfully.")
    for condition in conditions:
        if getattr(condition, "condition_type", None) is ConditionType.WATER_PROXIMITY:
            distance = float(dict(condition.parameters).get("distance_m", 0))
            st.caption(
                f"Cells are classified as matching when their centres are "
                f"within {distance:,.0f} m of WorldCover class-80 permanent "
                "water. This is not irrigation, not groundwater and not flood "
                "risk.")


def _render_methodology(result: Any) -> None:
    from core.spatial_query import ConditionType

    conditions = list(getattr(result, "conditions", ()) or ())
    kinds = {getattr(c, "condition_type", None) for c in conditions}

    if ConditionType.CROP_SUITABILITY in kinds:
        for condition in conditions:
            if getattr(condition, "condition_type", None) is ConditionType.CROP_SUITABILITY:
                parameters = dict(condition.parameters)
                st.markdown(
                    f"**Cotton** — rainfed screening; minimum class "
                    f"**{parameters.get('min_class', 3)}** "
                    f"(scenario: {parameters.get('scenario', 'rainfed')}). "
                    "Thresholds, weights and sources are unchanged.")
    if ConditionType.WATER in kinds or ConditionType.LAND_COVER_CLASS in kinds:
        st.markdown("**Land cover / water** — ESA WorldCover 10 m 2021 v200, "
                    "nearest-neighbour resampling (categorical); class 80 = "
                    "permanent water, class 40 = cropland, class 90 = wetland "
                    "(not water).")
    if ConditionType.WATER_PROXIMITY in kinds:
        for condition in conditions:
            if getattr(condition, "condition_type", None) is ConditionType.WATER_PROXIMITY:
                distance = float(dict(condition.parameters).get("distance_m", 0))
                st.markdown(
                    f"**Water proximity** — {distance:,.0f} m, measured in the "
                    "projected analysis CRS (metres, never degrees) with "
                    "`scipy.ndimage.distance_transform_edt`, on a buffered "
                    "WorldCover window at least as wide as the requested "
                    "distance, then cropped back to the analysis grid. Cells "
                    "whose answer depends on area outside the window are "
                    "reported as insufficient data, never as 'far from water'.")
    st.markdown(
        f"**Resolution** — analysis grid "
        f"{getattr(result, 'analysis_resolution', 30):g} m; sources are native "
        f"{_native_summary(result)}. Resampling adds no information.")
    window = (getattr(result, "performance", {}) or {}).get("window") or {}
    if window.get("buffer_cells"):
        st.caption(
            f"Buffered read: +{window['buffer_cells']} cells per side "
            f"({window.get('buffer_metres', 0):,.0f} m); source window "
            f"{window.get('source_window_cells', ['?', '?'])[0]}×"
            f"{window.get('source_window_cells', ['?', '?'])[1]} cells, cropped "
            f"to {window.get('analysis_window_cells', ['?', '?'])[0]}×"
            f"{window.get('analysis_window_cells', ['?', '?'])[1]}.")


def _native_summary(result: Any) -> str:
    resolutions = dict(getattr(result, "source_resolutions", {}) or {})
    if not resolutions:
        return "10–1000 m"
    return "–".join(f"{v}" for v in list(resolutions.values())[:3])


def render_unsupported_condition(entry: Dict[str, Any]) -> None:
    """The 'we do not measure this' panel. No partial answer is offered."""
    blocked = ((entry.get("provenance") or {}).get("blocked_by") or [])
    st.error(
        "**Unsupported condition — nothing was computed.** The system does not "
        "currently measure what this query asks for, and no proxy was "
        "substituted.", icon=None)
    for item in blocked:
        label = item.get("label") or item.get("parameters", {}).get("topic", "")
        note = item.get("note") or ""
        st.markdown(f"- **{label}**")
        if note:
            st.caption(note)
    st.caption(
        "Measured instead: permanent water (WorldCover class 80) and distance "
        "to it. Those are different quantities — water proximity is not "
        "irrigation, not groundwater and not flood risk.")


# --------------------------------------------------------------------------- #
# Phase 12 -- composed multi-condition result
# --------------------------------------------------------------------------- #
def render_multi_condition(result: Any) -> None:
    """Render a composed multi-condition result.

    The panel has to keep five outcomes apart at a glance -- matches found, no
    matches, insufficient data, unsupported condition, missing threshold -- and
    it has to show where every threshold came from. Nothing here is recomputed
    and nothing is inferred: every line is a field of the result object.
    """
    status = str(getattr(result, "status", ""))
    headline = {
        "ok": ("", "Cells found"),
        "zero_matches": ("", "No cells satisfy every condition"),
        "insufficient_data": ("", "Not enough data to decide"),
    }.get(status, ("", "Result available"))

    st.caption(
        "Composed condition: several already-supported analyses combined on "
        "one verified grid.")
    # The boundary is printed, not hidden in a collapsed expander: it is the
    # one sentence that must never be scrolled past. It is read from the
    # result so the panel, the legend and the docs cannot drift apart.
    boundary = next((item for item in (getattr(result, "limitations", ()) or ())
                     if "not causal attribution" in item),
                    "A combined condition is geographic evidence, not causal "
                    "attribution.")
    st.info(boundary, icon=None)

    # ---- 1. what was understood ------------------------------------------ #
    st.markdown("**Interpreted as**")
    operator = str(getattr(result, "operator", "and") or "and")
    st.caption(f"Normalised query: _{getattr(result, 'normalized_query', '')}_")
    st.caption(f"Combined with **{operator.upper()}** — "
               + ("every condition must hold."
                  if operator == "and" else "any condition may hold."))

    # ---- 2. the outcome --------------------------------------------------- #
    st.markdown(f"**Result** — {headline[0]} {headline[1]}")

    # ---- 3. the numbers, with the unknowns beside them -------------------- #
    matched = int(getattr(result, "matched_cell_count", 0) or 0)
    non_matching = int(getattr(result, "non_matching_cell_count", 0) or 0)
    unknown = int(getattr(result, "insufficient_cell_count", 0) or 0)
    columns = st.columns(4)
    for column, (label, value) in zip(columns, [
            ("Matching cells", f"{matched:,}"),
            ("Measured, not matching", f"{non_matching:,}"),
            ("Undecided cells", f"{unknown:,}"),
            ("Matched area", f"{float(getattr(result, 'matched_area_km2', 0.0)):,.3f} km²"),
    ]):
        column.metric(label, value)

    if unknown:
        st.caption(
            f"{unknown:,} cells could not be established from the available "
            f"data. They are reported as **undecided** — never as matches and "
            f"never as non-matches.")
    if status == "insufficient_data":
        st.warning(
            "Every cell in the selected area is undecided, so this is not a "
            "result of \"nothing found\" — the conditions could not be "
            "evaluated here at all.", icon=None)

    # ---- 4. per-condition evidence, threshold provenance included --------- #
    condition_results = getattr(result, "condition_results", None) or []
    if condition_results:
        with st.expander("Conditions and thresholds", expanded=True):
            for entry in condition_results:
                name = entry.get("label") or entry.get("name") or "condition"
                st.markdown(
                    f"**{name}**"
                    + (" *(negated)*" if entry.get("negated") else "")
                    + f" — {int(entry.get('matched', 0)):,} matching, "
                    f"{int(entry.get('insufficient', 0)):,} undecided")
                st.caption(
                    f"source: {entry.get('source', '')} · "
                    f"kind: {entry.get('kind', '')}")
                threshold = entry.get("threshold")
                if threshold is not None:
                    spec = entry.get("threshold_provenance") or {}
                    st.caption(
                        f"threshold: **{entry.get('operator', '')} "
                        f"{threshold}** — {_threshold_origin(spec)}")
                if entry.get("class"):
                    st.caption(f"change class: **{entry['class']}**")
                for limitation in entry.get("limitations", ()) or ():
                    st.caption(f"· {limitation}")

    # ---- 5. attached evidence (measured, never a filter) ------------------ #
    summaries = getattr(result, "index_summaries", None) or {}
    if summaries:
        with st.expander("Attached evidence (measured, not filtered)",
                         expanded=True):
            st.caption(
                "These statistics were measured over the cells that matched "
                "the conditions above. They did not decide anything — no "
                "threshold was applied to them.")
            for index, summary in summaries.items():
                if not summary.get("defined"):
                    st.markdown(f"**{index.upper()}** — not available "
                                f"({summary.get('reason', 'no valid cells')})")
                    continue
                st.markdown(
                    f"**{index.upper()}** over the matched area — "
                    f"mean {summary.get('mean'):.3f}, "
                    f"median {summary.get('median'):.3f}, "
                    f"range {summary.get('min'):.3f} … {summary.get('max'):.3f}")
                st.caption(
                    f"{int(summary.get('valid_pixels', 0)):,} valid cells · "
                    f"{summary.get('computed_over', '')}")

    # ---- 6. sources, dates, grid ----------------------------------------- #
    with st.expander("Sources, dates and grid", expanded=False):
        st.markdown("**Source analyses**")
        for source in getattr(result, "source_analyses", ()) or ():
            st.markdown(f"- {source}")
        dates = getattr(result, "source_dates", None) or {}
        if dates.get("before") or dates.get("after"):
            st.caption(
                f"Dates: {dates.get('before', '—')} → {dates.get('after', '—')}")
        elif dates.get("scene_date"):
            st.caption(f"Scene date: {dates['scene_date']}")
        grid = getattr(result, "grid", None) or {}
        if grid:
            st.caption(
                f"Grid: {int(grid.get('width', 0))} × {int(grid.get('height', 0))} "
                f"cells at {float(grid.get('resolution_m', 0.0)) or 0:g} m · "
                f"CRS {grid.get('crs', '')}")
        alignment = getattr(result, "alignment", None) or {}
        if alignment:
            st.caption("Alignment: " + ", ".join(
                f"{k} = {v}" for k, v in alignment.items()))

    # ---- 7. limitations --------------------------------------------------- #
    with st.expander("Limitations", expanded=False):
        for item in getattr(result, "limitations", ()) or ():
            st.markdown(f"- {item}")


def _threshold_origin(spec: Dict[str, Any]) -> str:
    """One honest sentence about where a threshold came from."""
    provenance = str(spec.get("provenance") or "")
    detail = str(spec.get("detail") or "")
    if provenance == "user_specified":
        return "from your query."
    if provenance == "config_convention":
        return (f"a display/query convention ({detail or spec.get('note', '')}) "
                f"— enabled by you, and not a scientific classification.")
    if provenance == "relative":
        return (f"relative to this area ({detail or 'median'}) — a comparison "
                f"within the selected area, not an absolute class.")
    return detail or "origin not recorded."


def render_multi_condition_refusal(entry: Dict[str, Any]) -> None:
    """A refused composition, with the fix -- never a defaulted threshold."""
    st.caption(
        "No answer was produced: a composed query is answered only when every "
        "condition is fully specified.")
    for hint in (entry.get("hints") or ()):
        st.markdown(f"- {hint}")
