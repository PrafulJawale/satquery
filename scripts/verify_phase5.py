"""Phase 5 verification harness: ROI selection.

Run:
    python scripts/verify_phase5.py

Also writes:
    artifacts/phase5_draw.html   standalone Leaflet page WITH the draw tools
    artifacts/phase5_roi.png     static picture of inside / partial / outside

Exits 0 only if every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from pyproj import Transformer
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping

from core.geo import (
    WEB_MERCATOR,
    MissingCRSError,
    footprint_feature,
    native_footprint,
    raster_footprint,
    reproject_array,
    web_rgba_from_values,
)
from core.geometry import (
    CRS84,
    area_m2,
    geodesic_area_m2,
    parse_geometry,
    transform_geometry,
)
from core.indices import ndvi_from_dataset
from core.raster import open_dataset
from core.roi import (
    CLIPPED_MESSAGE,
    OUTSIDE_MESSAGE,
    clear_roi_state,
    drawing_signature,
    is_map_stale,
    select_roi,
    update_roi_state,
)
from core.samples import list_samples, sample_path
from ui.map import MapOverlay, build_map

_results: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> bool:
    _results.append((bool(condition), message))
    print(f"   [{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)


S2_FILE = next((n for n in list_samples() if n.startswith("s2_")), None)
UTM36 = CRS.from_epsg(32636)
WGS84 = CRS.from_epsg(4326)
NORTH_UP = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
ROTATED = Affine(10.0, 3.0, 377200.0, 2.0, -10.0, 3441820.0)
FOOT = native_footprint(NORTH_UP, 2048, 2048)
FOOT_ROT = native_footprint(ROTATED, 512, 512)


def to_ll(x, y, crs=UTM36):
    return Transformer.from_crs(crs, CRS84, always_xy=True).transform(x, y)


def utm_feature(x0, y0, x1, y1, crs=UTM36):
    ring = [to_ll(x, y, crs) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


def ll_feature(lon0, lat0, lon1, lat1):
    ring = [(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}


INSIDE = utm_feature(387000, 3431000, 388000, 3432000)
OUTSIDE = ll_feature(32.4, 32.4, 32.5, 32.5)
PARTIAL = utm_feature(397000, 3431000, 398000, 3432000)


# --------------------------------------------------------------------------- #
def verify_capture_contract() -> None:
    print("\n=== A. drawing capture (what the libraries actually do) ===")
    import inspect

    import streamlit_folium

    src = inspect.getsource(streamlit_folium)
    check("all_drawings" in src and "last_active_drawing" in src,
          "streamlit-folium exposes all_drawings / last_active_drawing")
    check('"all_drawings": None' in src,
          "all_drawings is None until the map reports drawing activity")

    html = build_map(overlays=[], bounds=[[31.1, 31.7], [31.3, 31.9]], draw=True).get_root().render()
    check("L.Control.Draw" in html, "the Draw control is in the rendered map")
    check("window.drawnItems" in html, "the streamlit-folium drawn-items shim is present")
    check('"polyline": false' in html and '"marker": false' in html
          and '"circle": false' in html and '"circlemarker": false' in html,
          "polyline / marker / circle / circle-marker are disabled")
    check('"polygon": {' in html and '"rectangle": {' in html,
          "polygon and rectangle are enabled")
    print("   note: browser drawing itself cannot be verified here -- see the "
          "Playwright attempt and the manual checklist in docs/PHASE5.md")


# --------------------------------------------------------------------------- #
def verify_parsing_and_validation() -> None:
    print("\n=== B. parsing and validation ===")
    check(parse_geometry(INSIDE).geom_type == "Polygon", "a rectangle arrives as a Polygon")

    bow = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    lon0, lat0 = to_ll(387440, 3431480)
    ring = [(lon0 + dx / 100, lat0 + dy / 100) for dx, dy in bow.exterior.coords]
    sel = select_roi([{"type": "Feature", "properties": {},
                       "geometry": {"type": "Polygon", "coordinates": [ring]}}], FOOT, UTM36)
    check(sel is not None and sel.is_valid and any("repaired" in w for w in sel.warnings),
          "a self-intersecting polygon is repaired AND the repair is reported")

    for name, feat, kind in [
        ("line", {"type": "Feature", "properties": {},
                  "geometry": {"type": "LineString",
                               "coordinates": [to_ll(387000, 3431000), to_ll(388000, 3432000)]}}, "LineString"),
        ("point", {"type": "Feature", "properties": {},
                   "geometry": {"type": "Point", "coordinates": list(to_ll(387440, 3431480))}}, "Point"),
    ]:
        got = select_roi([feat], FOOT, UTM36)
        check(got is not None and not got.is_valid and "not an area" in got.message,
              f"a {kind} is rejected as not an area")

    degenerate = {"type": "Feature", "properties": {},
                  "geometry": {"type": "Polygon", "coordinates": [[list(to_ll(387440, 3431480))] * 3]}}
    got = select_roi([degenerate], FOOT, UTM36)
    check(got is not None and not got.is_valid, "a zero-area shape is rejected")

    check(select_roi([], FOOT, UTM36) is None and select_roi(None, FOOT, UTM36) is None,
          "an empty selection yields no ROI")

    a = Polygon([to_ll(387000, 3431000), to_ll(387500, 3431000),
                 to_ll(387500, 3431500), to_ll(387000, 3431500)])
    b = Polygon([to_ll(388000, 3432000), to_ll(388500, 3432000),
                 to_ll(388500, 3432500), to_ll(388000, 3432500)])
    multi = select_roi([{"type": "Feature", "properties": {},
                         "geometry": mapping(MultiPolygon([a, b]))}], FOOT, UTM36)
    check(multi.num_parts == 2 and abs(multi.area_m2 - 500_000) < 100,
          f"a MultiPolygon reports {multi.num_parts} parts and the summed area")


# --------------------------------------------------------------------------- #
def verify_transformation() -> None:
    print("\n=== C. EPSG:4326 -> raster CRS ===")
    sel = select_roi([INSIDE], FOOT, UTM36)
    xs = [c[0] for c in sel.geometry_raster_crs.exterior.coords]
    check(all(300_000 < x < 500_000 for x in xs),
          "the stored geometry is in UTM metres, not degrees")

    back = Transformer.from_crs(UTM36, CRS84, always_xy=True)
    ring_ll = sel.original_geometry.exterior
    worst = max(ring_ll.distance(Point(*back.transform(x, y)))
                for x, y in sel.geometry_raster_crs.exterior.coords)
    check(worst < 1e-9, f"every stored vertex maps back onto the drawn ring (worst {worst:.2e} deg)")
    check(sel.original_geometry is not None and 30 < sel.original_geometry.centroid.x < 33,
          "the original drawn geometry is kept in lon/lat for provenance")

    try:
        select_roi([INSIDE], FOOT, None)
        check(False, "a CRS-less raster should refuse selection")
    except MissingCRSError:
        check(True, "a CRS-less raster refuses selection instead of guessing")


# --------------------------------------------------------------------------- #
def verify_intersection_and_clipping() -> None:
    print("\n=== D. intersection with the real footprint ===")
    inside = select_roi([INSIDE], FOOT, UTM36)
    check(inside.intersects_raster and not inside.was_clipped
          and abs(inside.overlap_fraction - 1.0) < 1e-6,
          "a fully-inside selection is accepted unclipped")
    check(abs(inside.area_m2 - 1_000_000) < 50,
          f"its area is 1 km² (got {inside.area_m2:,.0f} m²)")

    outside = select_roi([OUTSIDE], FOOT, UTM36)
    check(not outside.intersects_raster and outside.message == OUTSIDE_MESSAGE
          and not outside.usable,
          "a selection outside the raster is refused with a clear message")

    partial = select_roi([PARTIAL], FOOT, UTM36)
    check(partial.was_clipped and partial.message == CLIPPED_MESSAGE
          and 0 < partial.overlap_fraction < 1,
          f"a hanging-off selection is clipped (overlap {partial.overlap_fraction:.3f})")
    check(FOOT.contains(partial.geometry_raster_crs),
          "the clipped geometry lies inside the footprint")
    check(partial.area_m2 < partial.original_area_m2,
          "the usable area is smaller than the drawn area")

    # --- the bounding-box trap, on a rotated grid -------------------------- #
    x0, y0, x1, y1 = 377800.0, 3442400.0, 378200.0, 3442800.0
    trap = box(x0, y0, x1, y1)
    check(box(*FOOT_ROT.bounds).contains(trap), "precondition: inside the bounding box")
    check(not FOOT_ROT.contains(trap), "precondition: outside the real footprint")
    got = select_roi([utm_feature(x0, y0, x1, y1)], FOOT_ROT, UTM36)
    check(not got.intersects_raster,
          "bbox-inside / footprint-outside is refused (NOT a bounding-box test)")

    straddle = box(379400.0, 3442100.0, 380000.0, 3442500.0)
    got2 = select_roi([utm_feature(*straddle.bounds)], FOOT_ROT, UTM36)
    check(got2.was_clipped and got2.area_m2 < got2.original_area_m2 * 0.999,
          "a selection crossing a rotated edge is clipped to the quadrilateral")
    check(FOOT_ROT.buffer(1e-6).contains(got2.geometry_raster_crs),
          "the clipped part stays inside the rotated footprint")


# --------------------------------------------------------------------------- #
def verify_area() -> None:
    print("\n=== E. area ===")
    inside = select_roi([INSIDE], FOOT, UTM36)
    expected = geodesic_area_m2(inside.geometry_raster_crs, UTM36)
    check(abs(inside.area_m2 - expected) / expected < 0.01,
          f"projected area agrees with an independent geodesic area ({inside.area_m2:,.0f} m²)")
    check(abs(inside.area_hectares - 100.0) < 0.01, "1 km² = 100.00 ha")
    check("planar" in inside.area_method and "32636" in inside.area_method,
          "the method is recorded: " + inside.area_method)

    foot_ll = box(31.0, 31.0, 31.5, 31.5)
    geo = select_roi([ll_feature(31.1, 31.1, 31.3, 31.3)], foot_ll, WGS84)
    check("geodesic" in geo.area_method, "a geographic raster CRS uses geodesic area")
    check(geo.area_m2 > 4e8 and geo.area_m2 != 0.04,
          f"area is in m², not degrees² (got {geo.area_m2:,.0f} m²)")
    check(abs(geo.area_m2 - geodesic_area_m2(geo.geometry_raster_crs, WGS84)) < 1.0,
          "geodesic area matches pyproj on the same geometry")


# --------------------------------------------------------------------------- #
def verify_state_machine() -> None:
    print("\n=== F. draw / replace / delete -- no stale ROI ===")
    state: dict = {}
    first = update_roi_state(state, [INSIDE], FOOT, UTM36, "raster-A")
    check(first is not None and first.is_valid, "drawing creates a selection")

    same = update_roi_state(state, [INSIDE], FOOT, UTM36, "raster-A")
    check(same is first, "an unrelated rerun keeps the selection (no recompute)")

    second = update_roi_state(state, [PARTIAL], FOOT, UTM36, "raster-A")
    check(second is not None and second is not first, "a new drawing replaces the old one")

    check(update_roi_state(state, [], FOOT, UTM36, "raster-A") is None
          and state["roi"] is None,
          "deleting everything clears the selection")
    check(not is_map_stale(state), "and it is not left flagged as stale")

    update_roi_state(state, [INSIDE], FOOT, UTM36, "raster-A")
    kept = update_roi_state(state, None, FOOT, UTM36, "raster-A")
    check(kept is not None and is_map_stale(state),
          "a map remount with no report keeps the ROI but flags it")
    check(update_roi_state(state, [], FOOT, UTM36, "raster-A") is None,
          "the first report after a remount (empty) clears it")

    update_roi_state(state, [INSIDE], FOOT, UTM36, "raster-A")
    moved = update_roi_state(state, [INSIDE], FOOT, UTM36, "raster-B")
    check(moved is not None and state["roi_signature"] == drawing_signature([INSIDE], "raster-B"),
          "switching raster invalidates the recorded selection")

    clear_roi_state(state)
    check(state["roi"] is None and state["roi_signature"] is None,
          "the Clear-selection action forgets everything")


# --------------------------------------------------------------------------- #
def write_artifacts() -> None:
    print("\n=== G. artefacts ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inside = select_roi([INSIDE], FOOT, UTM36)
    partial = select_roi([PARTIAL], FOOT, UTM36)
    outside = select_roi([OUTSIDE], FOOT, UTM36)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    fig.suptitle("Phase 5 — ROI selection: inside / partial / outside, and the rotated-grid trap",
                 fontsize=13, fontweight="bold")

    # --- panel 1: what the user sees (WGS84) ------------------------------ #
    ax = axes[0]
    fx, fy = raster_footprint(NORTH_UP, UTM36, 2048, 2048).exterior.xy
    ax.plot(fx, fy, color="#444", lw=1.6, label="raster footprint (WGS84)")
    for sel, colour, name in [
        (inside, "#2ca02c", f"inside · {inside.area_m2/1e6:.3f} km²"),
        (partial, "#ff7f0e", f"partial · {partial.overlap_fraction*100:.0f}% inside"),
        (outside, "#d62728", "outside · refused"),
    ]:
        if sel.original_geometry is None:
            continue
        gx, gy = sel.original_geometry.exterior.xy
        ax.plot(gx, gy, color=colour, lw=2, label=name)
    ax.set_title("1 — the drawn shapes (EPSG:4326, as drawn)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    ax.ticklabel_format(useOffset=False)

    # --- panel 2: after transformation + clipping (raster CRS) ------------ #
    ax = axes[1]
    fx, fy = FOOT.exterior.xy
    ax.plot(fx, fy, color="#444", lw=1.6, label="footprint (EPSG:32636, metres)")
    dgx, dgy = partial.original_geometry_raster_crs_bounds = (
        transform_geometry(partial.original_geometry, CRS84, UTM36).exterior.xy
    )
    ax.plot(dgx, dgy, color="#ff7f0e", lw=1.6, ls="--", label="drawn, transformed to UTM")
    if partial.geometry_raster_crs is not None:
        cx, cy = partial.geometry_raster_crs.exterior.xy
        ax.fill(cx, cy, color="#2ca02c", alpha=0.45, label="usable after clipping")
    ax.set_title("2 — clipping happens in the raster CRS")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    ax.ticklabel_format(useOffset=False, style="sci", scilimits=(0, 0))

    # --- panel 3: the rotated trap ---------------------------------------- #
    ax = axes[2]
    qx, qy = FOOT_ROT.exterior.xy
    ax.plot(qx, qy, color="#444", lw=2, label="rotated footprint (true quad)")
    bx, by = box(*FOOT_ROT.bounds).exterior.xy
    ax.plot(bx, by, color="#1f77b4", lw=1.2, ls=":", label="its bounding box (NOT the extent)")
    trap = box(377800.0, 3442400.0, 378200.0, 3442800.0)
    tx, ty = trap.exterior.xy
    ax.plot(tx, ty, color="#d62728", lw=2, label="inside bbox, outside raster → refused")
    straddle = box(379400.0, 3442100.0, 380000.0, 3442500.0)
    sx, sy = straddle.exterior.xy
    ax.plot(sx, sy, color="#ff7f0e", lw=2, label="crosses the edge → clipped")
    ax.set_title("3 — the bounding-box trap on a rotated grid")
    ax.set_xlabel("easting (m)"); ax.set_ylabel("northing (m)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    ax.ticklabel_format(useOffset=False, style="sci", scilimits=(0, 0))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = REPO_ROOT / "artifacts" / "phase5_roi.png"
    fig.savefig(str(out), dpi=110)
    plt.close(fig)
    print(f"   figure -> {out.relative_to(REPO_ROOT)}")

    # --- standalone draw page --------------------------------------------- #
    if S2_FILE:
        with open_dataset(str(sample_path(S2_FILE))) as ds:
            res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
            web = reproject_array(res.array, ds.transform, ds.crs, dst_crs=WEB_MERCATOR,
                                  max_pixels=600_000)
            foot = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=32)
        m = build_map(
            overlays=[MapOverlay("NDVI", web_rgba_from_values(web, -1.0, 1.0), web.leaflet_bounds)],
            footprint=footprint_feature(foot, {"crs": "EPSG:32636"}),
            bounds=web.leaflet_bounds,
            draw=True,
        )
        page = REPO_ROOT / "artifacts" / "phase5_draw.html"
        m.save(str(page))
        print(f"   standalone drawing page -> {page.relative_to(REPO_ROOT)} "
              f"({page.stat().st_size / 1e6:.2f} MB)")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print("SatQuery AI -- Phase 5 verification: ROI selection")
    print("=" * 78)
    verify_capture_contract()
    verify_parsing_and_validation()
    verify_transformation()
    verify_intersection_and_clipping()
    verify_area()
    verify_state_machine()
    write_artifacts()

    passed = sum(1 for ok, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 78)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 78)
    for ok, msg in _results:
        if not ok:
            print(f"  FAILED: {msg}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
