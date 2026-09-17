"""Phase 5 tests: core/roi.py -- validation, CRS transform, clipping, area, state.

The "bbox trap" tests are the important ones: a selection that overlaps the
footprint's *bounding box* but not the footprint itself must never be accepted
as inside.
"""

from __future__ import annotations

import pytest
from pyproj import Transformer
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import LineString, Point, Polygon, mapping, box

from core.geo import MissingCRSError, native_footprint, raster_footprint
from core.roi import (
    CLIPPED_MESSAGE,
    EMPTY_MESSAGE,
    OUTSIDE_MESSAGE,
    clear_roi_state,
    drawing_signature,
    is_map_stale,
    pick_active_drawing,
    select_roi,
    update_roi_state,
)
from core.geometry import CRS84, geodesic_area_m2

UTM36 = CRS.from_epsg(32636)
WGS84 = CRS.from_epsg(4326)
NORTH_UP = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
ROTATED = Affine(10.0, 3.0, 377200.0, 2.0, -10.0, 3441820.0)

FOOT_NORTH = native_footprint(NORTH_UP, 2048, 2048)
FOOT_ROT = native_footprint(ROTATED, 512, 512)

CENTRE = (387440.0, 3431480.0)


def to_ll(x, y, crs=UTM36):
    return Transformer.from_crs(crs, CRS84, always_xy=True).transform(x, y)


def utm_box_feature(x0, y0, x1, y1, crs=UTM36):
    """Axis-aligned UTM rectangle -> GeoJSON Feature in EPSG:4326."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    ring = [to_ll(x, y, crs) for x, y in corners]
    return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def ll_box_feature(lon0, lat0, lon1, lat1):
    ring = [(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]
    return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def line_feature():
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "LineString", "coordinates": [to_ll(387000, 3431000), to_ll(388000, 3432000)]}}


def point_feature():
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "Point", "coordinates": list(to_ll(*CENTRE))}}


# --------------------------------------------------------------------------- #
# nothing / empty
# --------------------------------------------------------------------------- #
def test_no_drawings_returns_none():
    assert select_roi(None, FOOT_NORTH, UTM36) is None
    assert select_roi([], FOOT_NORTH, UTM36) is None


def test_empty_geometry_is_rejected():
    empty = {"type": "Feature", "properties": {},
             "geometry": {"type": "Polygon", "coordinates": [[]]}}
    sel = select_roi([empty], FOOT_NORTH, UTM36)
    assert sel is None or not sel.is_valid


def test_pick_active_drawing_ignores_non_areas():
    drawings = [line_feature(), point_feature(), utm_box_feature(387000, 3431000, 388000, 3432000)]
    feature, count = pick_active_drawing(drawings)
    assert count == 1 and feature is drawings[-1]


# --------------------------------------------------------------------------- #
# inside / outside / partial
# --------------------------------------------------------------------------- #
def test_rectangle_completely_inside_is_accepted_unclipped():
    sel = select_roi([utm_box_feature(387000, 3431000, 388000, 3432000)], FOOT_NORTH, UTM36)
    assert sel.is_valid and sel.intersects_raster
    assert sel.was_clipped is False
    assert abs(sel.overlap_fraction - 1.0) < 1e-6
    assert abs(sel.area_m2 - 1_000_000.0) < 50.0       # 1 km x 1 km
    assert sel.geometry_type == "Polygon"
    assert sel.raster_crs == "EPSG:32636"


def test_rectangle_completely_outside_is_refused():
    far = ll_box_feature(32.4, 32.4, 32.5, 32.5)        # well outside the Nile Delta scene
    sel = select_roi([far], FOOT_NORTH, UTM36)
    assert sel.is_valid is True                          # the geometry itself is fine
    assert sel.intersects_raster is False
    assert sel.message == OUTSIDE_MESSAGE
    assert sel.usable is False


def test_partially_overlapping_rectangle_is_clipped_and_reported():
    # raster ends at easting 397680 -- this box runs 320 m past it
    sel = select_roi([utm_box_feature(397000, 3431000, 398000, 3432000)], FOOT_NORTH, UTM36)
    assert sel.intersects_raster and sel.was_clipped
    assert 0.0 < sel.overlap_fraction < 1.0
    assert sel.message == CLIPPED_MESSAGE
    assert sel.area_m2 < sel.original_area_m2
    assert abs(sel.overlap_fraction - 0.68) < 0.02       # 680 m of 1000 m inside
    assert any("inside the raster" in w for w in sel.warnings)


def test_clipped_geometry_stays_inside_the_footprint():
    sel = select_roi([utm_box_feature(397000, 3431000, 398000, 3432000)], FOOT_NORTH, UTM36)
    assert FOOT_NORTH.contains(sel.geometry_raster_crs)


def test_original_geometry_is_kept_in_wgs84():
    box_f = utm_box_feature(387000, 3431000, 388000, 3432000)
    sel = select_roi([box_f], FOOT_NORTH, UTM36)
    lons = [c[0] for c in sel.original_geometry.exterior.coords]
    assert all(30 < lon < 33 for lon in lons), "provenance copy stays lon/lat"


# --------------------------------------------------------------------------- #
# the bounding-box trap, on a ROTATED grid
# --------------------------------------------------------------------------- #
def test_selection_outside_a_rotated_footprint_but_inside_its_bbox():
    """The whole point of true polygon intersection.

    This square sits inside the footprint's bounding box and outside the
    footprint itself. Bounding-box logic would accept it; we must not.
    """
    x0, y0, x1, y1 = 377800.0, 3442400.0, 378200.0, 3442800.0
    bbox = box(*FOOT_ROT.bounds)
    candidate = box(x0, y0, x1, y1)
    assert bbox.contains(candidate), "precondition: inside the bbox"
    assert not FOOT_ROT.contains(candidate), "precondition: outside the real footprint"

    sel = select_roi([utm_box_feature(x0, y0, x1, y1)], FOOT_ROT, UTM36)
    assert sel.intersects_raster is False
    assert sel.usable is False


def test_selection_partly_over_a_rotated_edge_is_clipped_not_rounded_up():
    # The quad's top edge runs from (377200, 3441820) to (382320, 3442844):
    # slope 0.2, so it sits at y ~= 3442260..3442380 across this box.
    x0, y0, x1, y1 = 379400.0, 3442100.0, 380000.0, 3442500.0
    candidate = box(x0, y0, x1, y1)
    assert box(*FOOT_ROT.bounds).contains(candidate)
    assert not FOOT_ROT.contains(candidate), "precondition: it does hang off the edge"
    sel = select_roi([utm_box_feature(x0, y0, x1, y1)], FOOT_ROT, UTM36)
    assert sel.intersects_raster is True
    assert sel.was_clipped is True
    assert sel.area_m2 < sel.original_area_m2
    # the usable part must be inside the real quadrilateral
    assert FOOT_ROT.buffer(1e-6).contains(sel.geometry_raster_crs)


def test_rotated_raster_selection_area_is_less_than_the_drawn_area():
    sel = select_roi([utm_box_feature(379400.0, 3442100.0, 380000.0, 3442500.0)], FOOT_ROT, UTM36)
    assert sel.was_clipped
    assert sel.area_m2 < sel.original_area_m2 * 0.999
    assert sel.area_m2 > 0


# --------------------------------------------------------------------------- #
# invalid / unsupported geometry
# --------------------------------------------------------------------------- #
def test_self_intersecting_polygon_is_repaired_with_a_warning():
    bow = [(0.0, 0.0), (0.02, 0.02), (0.02, 0.0), (0.0, 0.02)]
    lon0, lat0 = to_ll(*CENTRE)
    ring = [(lon0 + dx, lat0 + dy) for dx, dy in bow]
    feature = {"type": "Feature", "properties": {},
               "geometry": {"type": "Polygon", "coordinates": [ring]}}
    sel = select_roi([feature], FOOT_NORTH, UTM36)
    assert sel.is_valid
    assert any("repaired" in w for w in sel.warnings)
    assert sel.area_m2 > 0


def test_zero_area_selection_is_rejected():
    lon0, lat0 = to_ll(*CENTRE)
    ring = [(lon0, lat0), (lon0, lat0), (lon0, lat0)]
    feature = {"type": "Feature", "properties": {},
               "geometry": {"type": "Polygon", "coordinates": [ring]}}
    sel = select_roi([feature], FOOT_NORTH, UTM36)
    assert sel.is_valid is False


def test_unsupported_geometry_types_are_rejected():
    for feature in (line_feature(), point_feature()):
        sel = select_roi([feature], FOOT_NORTH, UTM36)
        assert sel.is_valid is False
        assert "not an area" in sel.message


# --------------------------------------------------------------------------- #
# multipolygon
# --------------------------------------------------------------------------- #
def test_multipolygon_selection_reports_parts_and_total_area():
    geom_a = Polygon([to_ll(387000, 3431000), to_ll(387500, 3431000),
                      to_ll(387500, 3431500), to_ll(387000, 3431500)])
    geom_b = Polygon([to_ll(388000, 3432000), to_ll(388500, 3432000),
                      to_ll(388500, 3432500), to_ll(388000, 3432500)])
    from shapely.geometry import MultiPolygon
    multi = MultiPolygon([geom_a, geom_b])
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(multi)}
    sel = select_roi([feature], FOOT_NORTH, UTM36)
    assert sel.is_valid and sel.intersects_raster
    assert sel.num_parts == 2
    assert abs(sel.area_m2 - 500_000.0) < 100.0      # two 500 m x 500 m squares


# --------------------------------------------------------------------------- #
# transformation correctness
# --------------------------------------------------------------------------- #
def test_selection_is_transformed_into_the_raster_crs():
    feature = utm_box_feature(387000, 3431000, 388000, 3432000)
    sel = select_roi([feature], FOOT_NORTH, UTM36)
    xs = [c[0] for c in sel.geometry_raster_crs.exterior.coords]
    assert all(300_000 < x < 500_000 for x in xs), "metres of easting, not degrees"
    # Densification adds vertices, so compare geometrically instead of pairwise:
    # every vertex of the raster-CRS ring, mapped back to lon/lat, must lie
    # exactly on the ring the user drew.
    back = Transformer.from_crs(UTM36, CRS84, always_xy=True)
    ring_ll = sel.original_geometry.exterior
    for gx, gy in sel.geometry_raster_crs.exterior.coords:
        lon, lat = back.transform(gx, gy)
        assert ring_ll.distance(Point(lon, lat)) < 1e-9
    # ...and every drawn corner must survive (within a millimetre)
    forward = Transformer.from_crs(CRS84, UTM36, always_xy=True)
    raster_ring = sel.geometry_raster_crs.exterior
    for lon, lat in ring_ll.coords:
        ex, ey = forward.transform(lon, lat)
        assert raster_ring.distance(Point(ex, ey)) < 1e-3


def test_area_matches_an_independent_geodesic_calculation():
    feature = utm_box_feature(387000, 3431000, 388000, 3432000)
    sel = select_roi([feature], FOOT_NORTH, UTM36)
    expected = geodesic_area_m2(sel.geometry_raster_crs, UTM36)
    assert abs(sel.area_m2 - expected) / expected < 0.01


# --------------------------------------------------------------------------- #
# geographic raster CRS
# --------------------------------------------------------------------------- #
def test_geographic_raster_crs_uses_geodesic_area_not_degrees_squared():
    foot = box(31.0, 31.0, 31.5, 31.5)                  # a lon/lat "raster"
    sel = select_roi([ll_box_feature(31.1, 31.1, 31.3, 31.3)], foot, WGS84)
    assert sel.is_valid and sel.intersects_raster
    assert "geodesic" in sel.area_method
    assert sel.area_m2 > 4e8                            # ~0.2 deg square near 31 N
    assert sel.area_m2 != 0.04                          # definitely not degrees squared
    expected = geodesic_area_m2(sel.geometry_raster_crs, WGS84)
    assert abs(sel.area_m2 - expected) < 1.0


def test_geographic_raster_clipping_still_works():
    foot = box(31.0, 31.0, 31.5, 31.5)
    sel = select_roi([ll_box_feature(31.4, 31.4, 31.8, 31.8)], foot, WGS84)
    assert sel.was_clipped is True
    assert sel.area_m2 < sel.original_area_m2


# --------------------------------------------------------------------------- #
# CRS-less raster
# --------------------------------------------------------------------------- #
def test_crs_less_raster_refuses_selection():
    with pytest.raises(MissingCRSError):
        select_roi([utm_box_feature(387000, 3431000, 388000, 3432000)], FOOT_NORTH, None)


# --------------------------------------------------------------------------- #
# state machine: draw / replace / delete / no stale state
# --------------------------------------------------------------------------- #
def _sel(state, drawings, foot=FOOT_NORTH, key="raster-A"):
    return update_roi_state(state, drawings, foot, UTM36, raster_key=key)


A = utm_box_feature(387000, 3431000, 388000, 3432000)
B = utm_box_feature(389000, 3433000, 390000, 3434000)


def test_drawing_creates_a_selection():
    state: dict = {}
    sel = _sel(state, [A])
    assert sel is not None and sel.is_valid
    assert abs(sel.area_m2 - 1e6) < 50


def test_a_new_drawing_replaces_the_old_one():
    state: dict = {}
    first = _sel(state, [A])
    second = _sel(state, [B])
    assert second is not None
    assert abs(second.area_m2 - 1e6) < 50
    assert second is not first
    # the geometry really moved (different corner)
    assert first.geometry_raster_crs.bounds != second.geometry_raster_crs.bounds


def test_deleting_everything_clears_the_selection():
    state: dict = {}
    _sel(state, [A])
    assert _sel(state, []) is None
    assert state["roi"] is None
    assert is_map_stale(state) is False


def test_no_stale_roi_after_a_remount_with_nothing_drawn():
    state: dict = {}
    _sel(state, [A])
    staled = _sel(state, None)          # map remounted, has not reported yet
    assert staled is state["roi"] and staled is not None
    assert is_map_stale(state) is True  # the UI is told the outline may be gone
    assert _sel(state, []) is None      # ...and the first report (empty) clears it


def test_unrelated_rerun_keeps_the_selection_without_recomputing():
    state: dict = {}
    first = _sel(state, [A])
    again = _sel(state, [A])
    assert again is first, "same drawings -> same object, no recompute"


def test_changing_raster_invalidates_the_selection():
    state: dict = {}
    _sel(state, [A], key="raster-A")
    moved = _sel(state, [A], key="raster-B")
    assert moved is not None
    assert state["roi_signature"] == drawing_signature([A], "raster-B")


def test_clear_button_forgets_everything():
    state: dict = {}
    _sel(state, [A])
    clear_roi_state(state)
    assert state["roi"] is None and state["roi_signature"] is None
    assert is_map_stale(state) is False


def test_signature_is_stable_and_order_sensitive():
    assert drawing_signature([A]) == drawing_signature([A])
    assert drawing_signature([A]) != drawing_signature([B])
    assert drawing_signature([A], "k") != drawing_signature([A], "other")


def test_more_than_one_shape_warns_and_uses_the_most_recent():
    state: dict = {}
    sel = _sel(state, [A, B])
    assert sel.drawn_count == 2
    assert any("most recent" in w for w in sel.warnings)
    assert abs(sel.geometry_raster_crs.bounds[0] - 389000.0) < 1.0
