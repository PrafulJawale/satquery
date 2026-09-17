"""Phase 9, Checkpoint C -- synthetic tests for `core.spatial`.

Everything here is a hand-built array with a hand-computed expected answer.
No network, no GeoTIFF, no Streamlit: the geometry is known by construction,
so a wrong distance or a collapsed third value fails loudly here instead of
quietly producing a plausible-looking map later.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.spatial import (
    FALSE, INSUFFICIENT, TRUE, DistanceGridError, Grid, GridMask,
    GridMismatchError, apply_roi, combine_all, crop_mask, land_cover_mask,
    mask_from_class_raster, negate, proximity_mask, require_compatible,
    water_mask_from_land_cover,
)
from core.spatial_query import Operator


# --------------------------------------------------------------------------- #
# helpers -- grids whose geometry is known exactly
# --------------------------------------------------------------------------- #
def grid(crs: str = "EPSG:32636", px: float = 30.0, w: int = 10, h: int = 8,
         x0: float = 377200.0, y0: float = 3462300.0) -> Grid:
    """A projected, square-cell grid: (px, 0, x0, 0, -px, y0)."""
    return Grid(crs=crs, transform=(px, 0.0, x0, 0.0, -px, y0), width=w, height=h,
                resolution_m=px)


def mask(match, valid=None, gr=None, name="c", source="test"):
    if not isinstance(match, np.ndarray):
        match = np.asarray(match, dtype=bool)
    if match.ndim == 1:
        match = match.reshape(1, -1)
    if valid is None:
        valid = np.ones(match.shape, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    return GridMask(grid=gr or grid(w=match.shape[1], h=match.shape[0]),
                    name=name, source=source, match=match, valid=valid)


def from_state(states, gr=None, name="c"):
    """Build a mask from a list of 2 (TRUE) / 1 (FALSE) / 0 (NA)."""
    arr = np.asarray(states, dtype=np.uint8).reshape(1, -1)
    gr = gr or grid(w=arr.shape[1], h=1)
    return GridMask(grid=gr, name=name, source="test",
                    match=(arr == TRUE), valid=(arr > INSUFFICIENT))


def states_of(m: GridMask):
    return list(m.state.ravel())


# =========================================================================== #
# C2. grid compatibility
# =========================================================================== #
def test_identical_grids_are_compatible():
    g1, g2 = grid(), grid()
    assert g1.is_compatible_with(g2)
    assert g1.mismatch_reason(g2) is None


def test_same_crs_written_differently_is_still_the_same_grid():
    a = grid(crs="EPSG:32636")
    b = grid(crs="WGS 84 / UTM zone 36N")
    assert a.is_compatible_with(b)


@pytest.mark.parametrize("change,kwargs", [
    ("crs", {"crs": "EPSG:32635"}),
    ("transform", {"y0": 3462330.0}),
    ("width", {"w": 11}),
    ("height", {"h": 9}),
    ("pixelsize", {"px": 10.0}),
])
def test_any_difference_is_refused(change, kwargs):
    g1, g2 = grid(), grid(**kwargs)
    reason = g1.mismatch_reason(g2)
    assert reason, f"{change} must be detected as a mismatch"
    with pytest.raises(GridMismatchError) as err:
        g1.require_compatible(g2)
    assert "different analysis grid" in str(err.value)


def test_incompatible_masks_are_never_combined():
    a = mask(np.zeros((8, 10), dtype=bool), gr=grid())
    b = mask(np.zeros((8, 11), dtype=bool), gr=grid(w=11))
    with pytest.raises(GridMismatchError):
        require_compatible([a, b])
    with pytest.raises(GridMismatchError):
        combine_all([a, b], Operator.AND)


def test_grid_geometry_comes_from_the_transform_not_an_assumption():
    g = grid(px=10.0)
    assert g.pixel_area_m2 == pytest.approx(100.0)
    g30 = grid(px=30.0)
    assert g30.pixel_area_m2 == pytest.approx(900.0)
    assert g30.pixel_area_m2 != 100.0        # 10x10 is never assumed


def test_mask_shape_must_match_its_grid():
    with pytest.raises(ValueError, match="shape"):
        GridMask(grid=grid(w=10, h=8), name="x", source="t",
                 match=np.zeros((8, 9), dtype=bool),
                 valid=np.ones((8, 9), dtype=bool))


# =========================================================================== #
# C5. three-valued logic
# =========================================================================== #
@pytest.mark.parametrize("left,right,expected", [
    (TRUE, TRUE, TRUE),
    (TRUE, FALSE, FALSE),
    (FALSE, TRUE, FALSE),
    (FALSE, FALSE, FALSE),
    (TRUE, INSUFFICIENT, INSUFFICIENT),
    (INSUFFICIENT, TRUE, INSUFFICIENT),
    (FALSE, INSUFFICIENT, FALSE),      # one failure is decisive
    (INSUFFICIENT, FALSE, FALSE),
    (INSUFFICIENT, INSUFFICIENT, INSUFFICIENT),
])
def test_and_truth_table(left, right, expected):
    out = combine_all([from_state([left]), from_state([right])], Operator.AND)
    assert states_of(out) == [expected]


@pytest.mark.parametrize("left,right,expected", [
    (FALSE, FALSE, FALSE),
    (TRUE, FALSE, TRUE),
    (FALSE, TRUE, TRUE),
    (TRUE, TRUE, TRUE),
    (FALSE, INSUFFICIENT, INSUFFICIENT),
    (INSUFFICIENT, FALSE, INSUFFICIENT),
    (TRUE, INSUFFICIENT, TRUE),        # one match is decisive
    (INSUFFICIENT, TRUE, TRUE),
    (INSUFFICIENT, INSUFFICIENT, INSUFFICIENT),
])
def test_or_truth_table(left, right, expected):
    out = combine_all([from_state([left]), from_state([right])], Operator.OR)
    assert states_of(out) == [expected]


@pytest.mark.parametrize("value,expected", [
    (TRUE, FALSE), (FALSE, TRUE), (INSUFFICIENT, INSUFFICIENT),
])
def test_not_truth_table(value, expected):
    assert states_of(negate(from_state([value]))) == [expected]


def test_missing_data_is_never_collapsed_to_false():
    m = from_state([INSUFFICIENT, TRUE, FALSE])
    assert m.n_insufficient == 1 and m.n_true == 1 and m.n_false == 1
    # AND with a false stays false; AND with a true stays unknown -- NOT false
    assert states_of(combine_all([m, from_state([FALSE] * 3)], Operator.AND))[0] == FALSE
    assert states_of(combine_all([m, from_state([TRUE] * 3)], Operator.AND))[0] == INSUFFICIENT


def test_three_way_combination_and_name():
    out = combine_all([from_state([TRUE], name="a"),
                       from_state([TRUE], name="b"),
                       from_state([FALSE], name="c")], Operator.AND)
    assert states_of(out) == [FALSE]
    assert out.name == "a AND b AND c"


def test_single_mask_combination_is_identity():
    m = from_state([TRUE, FALSE, INSUFFICIENT])
    for op in (Operator.AND, Operator.OR):
        assert states_of(combine_all([m], op)) == states_of(m)


# =========================================================================== #
# C3. the WATER condition
# =========================================================================== #
def test_water_all_water():
    lc = np.full((4, 5), 80, dtype=np.uint8)
    m = water_mask_from_land_cover(lc, grid(w=5, h=4))
    assert m.n_true == 20 and m.n_false == 0 and m.n_insufficient == 0


def test_water_no_water():
    lc = np.full((4, 5), 40, dtype=np.uint8)
    m = water_mask_from_land_cover(lc, grid(w=5, h=4))
    assert m.n_true == 0 and m.n_false == 20


def test_water_mixed():
    lc = np.array([[80, 40, 40, 40, 10],
                   [40, 80, 40, 40, 40]], dtype=np.uint8)
    m = water_mask_from_land_cover(lc, grid(w=5, h=2))
    assert m.n_true == 2 and m.n_false == 8
    assert m.match[0, 0] and m.match[1, 1]


def test_wetland_is_not_water():
    lc = np.array([[90, 80]], dtype=np.uint8)
    m = water_mask_from_land_cover(lc, grid(w=2, h=1))
    assert not m.match[0, 0]        # class 90
    assert m.valid[0, 0]            # but it IS valid data
    assert m.match[0, 1]            # class 80


def test_water_nodata_is_insufficient():
    lc = np.array([[80, 0, 40]], dtype=np.uint8)
    m = water_mask_from_land_cover(lc, grid(w=3, h=1))
    assert m.n_true == 1 and m.n_insufficient == 1 and m.n_false == 1
    float_lc = np.array([[80.0, np.nan, 40.0]])
    m2 = water_mask_from_land_cover(float_lc, grid(w=3, h=1))
    assert m2.n_insufficient == 1


def test_land_cover_mask_matches_any_listed_class():
    lc = np.array([[40, 50, 90, 80]], dtype=np.uint8)
    m = land_cover_mask(lc, grid(w=4, h=1), classes_wanted=[40, 50])
    assert list(m.match[0]) == [True, True, False, False]


# =========================================================================== #
# C4. water proximity
# =========================================================================== #
def test_proximity_distances_are_metres_not_pixels():
    lc = np.full((1, 10), 40, dtype=np.uint8)
    lc[0, 0] = 80                      # water only in the first column
    g = grid(px=30.0, w=10, h=1)
    m = proximity_mask(water_mask_from_land_cover(lc, g), 90.0)
    # 30 m pixels: columns at 0, 30, 60, 90 m -> within 90 m = first four
    assert list(m.match[0]) == [True] * 4 + [False] * 6


def test_proximity_zero_distance_water_cells_match():
    lc = np.array([[80, 40, 40]], dtype=np.uint8)
    m = proximity_mask(water_mask_from_land_cover(lc, grid(w=3, h=1)), 0.0)
    assert list(m.match[0]) == [True, False, False]


def test_proximity_threshold_boundary_is_inclusive():
    """'within 60 m' includes a cell exactly 60 m away."""
    lc = np.full((1, 5), 40, dtype=np.uint8)
    lc[0, 0] = 80
    m = proximity_mask(water_mask_from_land_cover(lc, grid(px=30.0, w=5, h=1)),
                       60.0)
    assert list(m.match[0]) == [True, True, True, False, False]     # 0/30/60 m
    m_less = proximity_mask(
        water_mask_from_land_cover(lc, grid(px=30.0, w=5, h=1)), 59.9)
    assert list(m_less.match[0]) == [True, True, False, False, False]


def test_proximity_scales_with_the_real_pixel_size():
    """The same array means different distances at 10 m and at 30 m."""
    lc = np.full((1, 10), 40, dtype=np.uint8)
    lc[0, 0] = 80
    at10 = proximity_mask(
        water_mask_from_land_cover(lc, grid(px=10.0, w=10, h=1)), 30.0)
    at30 = proximity_mask(
        water_mask_from_land_cover(lc, grid(px=30.0, w=10, h=1)), 30.0)
    assert list(at10.match[0]) == [True] * 4 + [False] * 6    # 0,10,20,30 m
    assert list(at30.match[0]) == [True] * 2 + [False] * 8    # 0,30 m


def test_proximity_two_dimensional_geometry():
    """A block of water in the middle of a 7x7 grid at 100 m resolution."""
    lc = np.full((7, 7), 40, dtype=np.uint8)
    lc[3, 3] = 80
    m = proximity_mask(water_mask_from_land_cover(lc, grid(px=100.0, w=7, h=7)),
                       150.0)
    assert m.match[3, 3]                       # centre: 0 m
    assert m.match[2, 3] and m.match[3, 4]     # orthogonal: 100 m
    assert m.match[2, 2]                       # diagonal: 141.4 m <= 150
    assert not m.match[1, 3]                   # 200 m
    assert not m.match[0, 0]


def test_proximity_non_square_array():
    lc = np.full((3, 12), 40, dtype=np.uint8)
    lc[:, 0] = 80
    m = proximity_mask(water_mask_from_land_cover(lc, grid(px=30.0, w=12, h=3)),
                       60.0)
    assert m.match.shape == (3, 12)
    assert m.match.all(axis=0)[:3].all() and not m.match[:, 3].any()


def test_proximity_with_no_water_is_unknown_not_far():
    """No water inside the window: the nearest water may lie just outside it,
    so nothing may be claimed as 'farther than 1 km from water'."""
    lc = np.full((4, 4), 40, dtype=np.uint8)
    m = proximity_mask(water_mask_from_land_cover(lc, grid(w=4, h=4)), 1000.0)
    assert m.n_true == 0
    assert m.n_false == 0                # never asserted from an unseeable edge
    assert m.n_insufficient == 16
    assert m.provenance["water_cells_found"] == 0


def test_proximity_far_from_water_far_from_the_edge_is_false():
    """Interior cells really are 'far'; cells near the window edge are not
    claimed either way, because water just outside cannot be ruled out."""
    lc = np.full((41, 80), 40, dtype=np.uint8)
    lc[20, 0] = 80                       # water at the left edge, 30 m cells
    m = proximity_mask(
        water_mask_from_land_cover(lc, grid(px=30.0, w=80, h=41)), 300.0)
    assert m.match[20, 0]                                 # the water cell
    assert m.state[20, 10] == TRUE                        # 300 m
    assert m.state[20, 40] == FALSE                       # 1200 m from water,
    assert not m.valid[20, 79]                            #  630 m from the edge
    assert not m.valid[20, 79] and m.state[20, 79] == INSUFFICIENT



def test_proximity_nodata_cell_is_insufficient():
    lc = np.array([[80, 0, 40]], dtype=np.uint8)
    water = water_mask_from_land_cover(lc, grid(w=3, h=1))
    m = proximity_mask(water, 1000.0)
    assert m.match[0, 0] and not m.valid[0, 1] and m.match[0, 2]
    assert m.n_insufficient == 1


def test_proximity_refuses_geographic_grids():
    lc = np.array([[80, 40]], dtype=np.uint8)
    water = water_mask_from_land_cover(lc, grid(crs="EPSG:4326", w=2, h=1))
    with pytest.raises(DistanceGridError, match="degrees"):
        proximity_mask(water, 1000.0)


def test_proximity_refuses_non_square_cells():
    lc = np.array([[80, 40]], dtype=np.uint8)
    g = Grid(crs="EPSG:32636", transform=(30.0, 0.0, 0.0, 0.0, -10.0, 0.0),
             width=2, height=1, resolution_m=30.0)
    with pytest.raises(DistanceGridError, match="square"):
        proximity_mask(water_mask_from_land_cover(lc, g), 1000.0)


def test_proximity_refuses_negative_distance():
    with pytest.raises(DistanceGridError):
        proximity_mask(water_mask_from_land_cover(
            np.array([[80]], dtype=np.uint8), grid(w=1, h=1)), -5.0)


# =========================================================================== #
# the regression the correction was about: NOT WATER != NOT WATER_PROXIMITY
# =========================================================================== #
def test_not_water_is_not_not_water_proximity():
    lc = np.full((21, 20), 40, dtype=np.uint8)
    lc[10, 0] = 80
    g = grid(px=30.0, w=20, h=21)
    water = water_mask_from_land_cover(lc, g)

    not_water = negate(water)                       # NOT class 80
    not_near = negate(proximity_mask(water, 60.0))  # farther than 60 m

    # 30 m cells: 0 m, 30 m and 60 m are "near"; 90 m onwards are not.
    assert list(not_water.match[10, :4]) == [False, True, True, True]
    assert list(not_near.match[10, :4]) == [False, False, False, True]
    assert not np.array_equal(not_water.match, not_near.match)
    assert not_water.name != not_near.name


# =========================================================================== #
# combinations and edge cases
# =========================================================================== #
def test_cotton_and_water():
    g = grid(px=30.0, w=3, h=1)
    cotton = mask_from_class_raster(np.array([[4, 3, 2]]), g, min_class=3,
                                    name="cotton", source="phase8")
    water = water_mask_from_land_cover(np.array([[80, 80, 40]]), g)
    out = combine_all([cotton, water], Operator.AND)
    assert states_of(out) == [TRUE, TRUE, FALSE]      # class 2 fails cotton


def test_cotton_and_near_water():
    g = grid(px=30.0, w=8, h=21)
    cotton = mask_from_class_raster(np.full((21, 8), 4), g, min_class=3,
                                    name="cotton", source="phase8")
    lc = np.full((21, 8), 40, dtype=np.uint8)
    lc[10, 0] = 80
    water = water_mask_from_land_cover(lc, g)
    out = combine_all([cotton, proximity_mask(water, 30.0)], Operator.AND)
    assert list(out.state[10, :6]) == [TRUE, TRUE, FALSE, FALSE, FALSE, FALSE]


def test_cropland_and_near_water():
    g = grid(px=30.0, w=8, h=21)
    lc = np.full((21, 8), 40, dtype=np.uint8)
    lc[10, 0] = 80
    crop = land_cover_mask(lc, g, classes_wanted=[40])
    water = water_mask_from_land_cover(lc, g)
    out = combine_all([crop, proximity_mask(water, 60.0)], Operator.AND)
    # The water cell itself is class 80, so it is not cropland and cannot match
    # (one class per cell). 30 m and 60 m are cropland AND near water;
    # 90 m onwards are cropland but far from water.
    assert list(out.state[10, :6]) == [FALSE, TRUE, TRUE, FALSE, FALSE, FALSE]


def test_cropland_and_not_water():
    g = grid(px=30.0, w=3, h=1)
    crop = land_cover_mask(np.array([[40, 40, 80]]), g, classes_wanted=[40])
    water = water_mask_from_land_cover(np.array([[40, 40, 80]]), g)
    out = combine_all([crop, negate(water)], Operator.AND)
    assert states_of(out) == [TRUE, TRUE, FALSE]


def test_cotton_or_cropland():
    g = grid(px=30.0, w=3, h=1)
    cotton = mask_from_class_raster(np.array([[1, 4, 1]]), g, min_class=3,
                                    name="cotton", source="phase8")
    crop = land_cover_mask(np.array([[40, 10, 10]]), g, classes_wanted=[40])
    out = combine_all([cotton, crop], Operator.OR)
    assert states_of(out) == [TRUE, TRUE, FALSE]


def test_cotton_and_not_water():
    g = grid(px=30.0, w=2, h=1)
    cotton = mask_from_class_raster(np.array([[4, 4]]), g, min_class=3,
                                    name="cotton", source="phase8")
    water = water_mask_from_land_cover(np.array([[80, 40]]), g)
    out = combine_all([cotton, negate(water)], Operator.AND)
    assert states_of(out) == [FALSE, TRUE]


def test_zero_matches_is_false_not_insufficient():
    out = combine_all([from_state([TRUE]), from_state([FALSE])], Operator.AND)
    assert out.n_true == 0 and out.n_false == 1 and out.n_insufficient == 0
    assert "insufficient" not in out.name


def test_all_matches():
    out = combine_all([from_state([TRUE] * 4), from_state([TRUE] * 4)],
                      Operator.AND)
    assert out.n_true == 4 and out.counts()["matched_fraction"] == 1.0


def test_partial_matches_and_area_use_the_real_cell_size():
    g = grid(px=30.0, w=4, h=1)
    m = mask([True, True, False, False], gr=g)
    counts = m.counts()
    assert counts["matching_cells"] == 2
    assert counts["matched_fraction"] == pytest.approx(0.5)
    assert counts["matched_area_m2"] == pytest.approx(2 * 900.0)
    assert counts["cell_area_m2"] == pytest.approx(900.0)


def test_insufficient_data_result_is_not_reported_as_zero_matches():
    combined = combine_all([from_state([TRUE]), from_state([INSUFFICIENT])],
                           Operator.AND)
    assert combined.n_true == 0
    assert combined.n_insufficient == 1        # distinguishable from FALSE
    assert combined.n_false == 0


def test_roi_clipping_makes_outside_cells_insufficient():
    g = grid(px=30.0, w=4, h=1)
    m = mask([True, True, True, True], gr=g)
    clipped = apply_roi(m, np.array([[True, True, False, False]]))
    assert clipped.n_true == 2 and clipped.n_insufficient == 2


def test_roi_mask_shape_is_validated():
    with pytest.raises(ValueError, match="ROI mask shape"):
        apply_roi(mask(np.array([[True, True]]), gr=grid(w=2, h=1)),
                  np.array([[True, True, True]]))


# =========================================================================== #
# C/D1. cropping a buffered window back to the analysis grid
# =========================================================================== #
def test_crop_returns_the_aligned_subwindow():
    big = grid(px=30.0, w=10, h=10, x0=0.0, y0=300.0)
    small = grid(px=30.0, w=6, h=6, x0=60.0, y0=240.0)   # 2 cells in, 2 down
    inner = np.zeros((10, 10), dtype=bool)
    inner[2:8, 2:8] = True
    mask = GridMask(grid=big, name="water", source="t", match=inner,
                    valid=np.ones((10, 10), dtype=bool))
    cropped = crop_mask(mask, small)
    assert cropped.grid is small
    assert cropped.match.shape == (6, 6)
    assert cropped.match.all()
    assert cropped.provenance["crop_offset_cells"] == [2, 2]


def test_crop_refuses_crs_and_pixel_size_changes():
    mask = GridMask(grid=grid(w=10, h=10), name="w", source="t",
                    match=np.zeros((10, 10), dtype=bool),
                    valid=np.ones((10, 10), dtype=bool))
    with pytest.raises(GridMismatchError, match="CRS"):
        crop_mask(mask, grid(crs="EPSG:32635", w=6, h=6))
    with pytest.raises(GridMismatchError, match="cell width"):
        crop_mask(mask, grid(px=10.0, w=6, h=6))


def test_crop_refuses_a_half_cell_offset():
    mask = GridMask(grid=grid(w=10, h=10), name="w", source="t",
                    match=np.zeros((10, 10), dtype=bool),
                    valid=np.ones((10, 10), dtype=bool))
    shifted = grid(w=6, h=6, x0=377215.0)
    with pytest.raises(GridMismatchError, match="whole cells"):
        crop_mask(mask, shifted)


def test_crop_refuses_a_target_that_is_not_contained():
    mask = GridMask(grid=grid(w=10, h=10), name="w", source="t",
                    match=np.zeros((10, 10), dtype=bool),
                    valid=np.ones((10, 10), dtype=bool))
    with pytest.raises(GridMismatchError, match="not contained"):
        crop_mask(mask, grid(w=20, h=20))
