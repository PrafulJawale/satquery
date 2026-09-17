"""Phase 6 tests: ROI zonal statistics from the NATIVE raster.

The centre-piece is a hand-checkable synthetic grid: a 10x10 array holding
0.00, 0.01, ... 0.99 so that every statistic can be verified with a calculator.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Polygon, box

from core.statistics import (
    NO_PIXELS_MESSAGE,
    NO_VALID_MESSAGE,
    ROINDVIStats,
    calculate_roi_ndvi_stats,
    pixel_geometry,
    roi_pixel_mask,
)

UTM36 = CRS.from_epsg(32636)
WGS84 = CRS.from_epsg(4326)

# 10 m pixels, origin at (377200, 3441820).  Pixel (row r, col c) centre:
#   x = 377200 + 10*(c + 0.5)
#   y = 3441820 - 10*(r + 0.5)
TEN_M = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
ROTATED = Affine(10.0, 3.0, 377200.0, 2.0, -10.0, 3441820.0)
NONSQUARE = Affine(20.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)   # 20 m x 10 m

H = W = 10
# row 0 -> 0.00..0.09, row 1 -> 0.10..0.19, ...
GRID = np.arange(100, dtype=np.float64).reshape(H, W) / 100.0
FULL_MASK = np.ones((H, W), dtype=bool)


def box_roi(col0: float, row0: float, col1: float, row1: float,
            transform: Affine = TEN_M) -> Polygon:
    """Pixel-corner box -> polygon in the raster CRS."""
    x0, y0 = transform * (col0, row0)
    x1, y1 = transform * (col1, row1)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# --------------------------------------------------------------------------- #
# pixel geometry derived from the affine
# --------------------------------------------------------------------------- #
def test_pixel_geometry_for_a_10m_grid():
    g = pixel_geometry(TEN_M, UTM36)
    assert (g["pixel_width"], g["pixel_height"], g["pixel_area_m2"]) == (10.0, 10.0, 100.0)


def test_pixel_geometry_is_not_assumed_to_be_10m():
    g = pixel_geometry(NONSQUARE, UTM36)
    assert (g["pixel_width"], g["pixel_height"]) == (20.0, 10.0)
    assert g["pixel_area_m2"] == 200.0


def test_pixel_geometry_for_a_rotated_affine():
    a, b, _c, d, e, _f = ROTATED[:6]
    g = pixel_geometry(ROTATED, UTM36)
    assert abs(g["pixel_area_m2"] - abs(a * e - b * d)) < 1e-9      # |det|
    assert g["pixel_area_m2"] == pytest.approx(106.0)               # |10*-10 - 3*2|
    assert g["pixel_width"] == pytest.approx(np.hypot(a, d))
    assert g["pixel_height"] == pytest.approx(np.hypot(b, e))


def test_pixel_geometry_converts_non_metre_units():
    # EPSG:2225 is in US survey feet; a "10 unit" pixel is not 10 m
    g = pixel_geometry(Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0), CRS.from_epsg(2225))
    assert 3.04 < g["pixel_width"] < 3.05          # 10 US survey feet in metres
    assert g["pixel_area_m2"] == pytest.approx(g["pixel_width"] ** 2)


# --------------------------------------------------------------------------- #
# hand-checkable statistics
# --------------------------------------------------------------------------- #
def test_hand_calculable_statistics_over_the_whole_grid():
    roi = box_roi(0, 0, W, H)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    v = np.arange(100) / 100.0

    assert res.pixels_inside_roi == 100
    assert res.valid_pixels == 100
    assert res.invalid_pixels == 0
    assert res.stats["min"] == pytest.approx(0.0)
    assert res.stats["max"] == pytest.approx(0.99)
    assert res.stats["mean"] == pytest.approx(v.mean())          # 0.495
    assert res.stats["median"] == pytest.approx(np.median(v))    # 0.495
    assert res.stats["std"] == pytest.approx(v.std())            # population std
    assert res.stats["percentiles"]["p5"] == pytest.approx(np.percentile(v, 5))
    assert res.stats["percentiles"]["p25"] == pytest.approx(np.percentile(v, 25))
    assert res.stats["percentiles"]["p75"] == pytest.approx(np.percentile(v, 75))
    assert res.stats["percentiles"]["p95"] == pytest.approx(np.percentile(v, 95))
    assert res.valid_fraction == 1.0
    # area: 100 pixels x 100 m2 = 10,000 m2 = 1 ha
    assert res.pixel_area_m2 == pytest.approx(100.0)
    assert res.valid_area_m2 == pytest.approx(10_000.0)
    assert res.area_m2 == pytest.approx(10_000.0)


def test_hand_calculable_statistics_over_a_known_sub_block():
    """Rows 2..5 (inclusive), cols 3..6 (inclusive) -> 16 known values."""
    roi = box_roi(3, 2, 7, 6)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    expected = GRID[2:6, 3:7]
    assert res.pixels_inside_roi == 16
    assert res.stats["mean"] == pytest.approx(expected.mean())
    assert res.stats["sum"] == pytest.approx(expected.sum())
    assert res.stats["min"] == pytest.approx(expected.min())
    assert res.stats["max"] == pytest.approx(expected.max())
    assert res.stats["median"] == pytest.approx(np.median(expected))
    assert res.stats["std"] == pytest.approx(expected.std())


def test_known_median_of_an_even_count():
    roi = box_roi(0, 0, 2, 2)                    # values 0.00, 0.01, 0.10, 0.11
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    assert res.valid_pixels == 4
    assert res.stats["median"] == pytest.approx((0.01 + 0.10) / 2)


# --------------------------------------------------------------------------- #
# inside / partial / outside / empty / all-invalid
# --------------------------------------------------------------------------- #
def test_roi_fully_inside():
    roi = box_roi(2, 2, 6, 6)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi == 16
    assert res.has_stats


def test_roi_partially_overlapping_the_raster():
    # hangs off the top-left corner of the grid
    roi = box_roi(-4, -4, 3, 3)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi == 9            # only cols/rows 0..2 exist
    assert res.valid_pixels == 9
    assert any("beyond the raster extent" in w for w in res.warnings)


def test_roi_completely_outside_returns_no_pixels_and_no_stats():
    roi = box(377200.0 - 5000, 3441820.0 + 5000, 377200.0 - 4000, 3441820.0 + 6000)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi == 0
    assert res.stats is None
    assert res.message == NO_PIXELS_MESSAGE
    assert not res.has_stats


def test_empty_geometry_returns_no_pixels_and_no_stats():
    res = calculate_roi_ndvi_stats(GRID, Polygon(), TEN_M, FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi == 0
    assert res.stats is None
    assert res.message == NO_PIXELS_MESSAGE


def test_none_geometry_is_handled():
    res = calculate_roi_ndvi_stats(GRID, None, TEN_M, FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi == 0 and res.stats is None


def test_roi_with_only_invalid_pixels_reports_no_stats_not_zero():
    roi = box_roi(0, 0, 4, 4)
    mask = np.zeros((H, W), dtype=bool)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, mask, crs=UTM36)
    assert res.pixels_inside_roi == 16
    assert res.valid_pixels == 0
    assert res.invalid_pixels == 16
    assert res.stats is None
    assert res.message == NO_VALID_MESSAGE


def test_mixed_valid_and_invalid_pixels():
    roi = box_roi(0, 0, 4, 2)                    # rows 0..1, cols 0..3 -> 8 px
    data = GRID.copy()
    data[0, 0] = np.nan
    data[0, 1] = np.inf
    res = calculate_roi_ndvi_stats(data, roi, TEN_M, np.isfinite(data), crs=UTM36)
    assert res.pixels_inside_roi == 8
    assert res.valid_pixels == 6
    assert res.invalid_pixels == 2
    assert res.valid_fraction == pytest.approx(6 / 8)
    assert res.stats["mean"] == pytest.approx(data[0:2, 0:4][np.isfinite(data[0:2, 0:4])].mean())
    assert any("no valid NDVI value" in w for w in res.warnings)


def test_nodata_nan_and_inf_are_all_excluded():
    data = np.array([[0.5, np.nan], [np.inf, -np.inf]], dtype=np.float64)
    roi = box_roi(0, 0, 2, 2)
    res = calculate_roi_ndvi_stats(data, roi, TEN_M, np.isfinite(data), crs=UTM36)
    assert res.pixels_inside_roi == 4
    assert res.valid_pixels == 1
    assert res.stats["mean"] == pytest.approx(0.5)


def test_valid_mask_is_respected_even_where_data_is_finite():
    """A pixel can be finite yet invalid (e.g. nodata == 0 or a failed ratio)."""
    roi = box_roi(0, 0, 4, 1)
    mask = np.ones((H, W), dtype=bool)
    mask[0, 0] = False                            # invalid although 0.00 is finite
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, mask, crs=UTM36)
    assert res.pixels_inside_roi == 4
    assert res.valid_pixels == 3
    assert res.stats["min"] == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# pixel-centre convention
# --------------------------------------------------------------------------- #
def test_pixel_centre_convention_excludes_pixels_whose_centre_is_outside():
    """Boundary at x = 377200 + 10*3.2 cuts column 3 (centre 377235)."""
    x_cut = 377200.0 + 32.0                       # column 3 spans 377230..377240
    roi = box(377200.0, 3441820.0 - 100.0, x_cut, 3441820.0)
    inside_centre, _ = roi_pixel_mask(roi, TEN_M, H, W, all_touched=False)
    inside_touch, _ = roi_pixel_mask(roi, TEN_M, H, W, all_touched=True)
    cols_centre = sorted(set(np.nonzero(inside_centre)[1]))
    cols_touch = sorted(set(np.nonzero(inside_touch)[1]))
    assert cols_centre == [0, 1, 2]               # centre of col 3 is 377235 > cut
    assert 3 in cols_touch                        # all_touched keeps the touched pixel
    assert inside_touch.sum() > inside_centre.sum()


def test_all_touched_defaults_to_false():
    roi = box_roi(0, 0, 5, 5)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    assert res.all_touched is False
    assert res.pixels_inside_roi == 25


def test_all_touched_true_increases_the_pixel_count():
    roi = box_roi(0, 0, 5.2, 5.2)                # cuts column 5 (centre 5.5) in half
    strict = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    loose = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36,
                                     all_touched=True)
    assert strict.pixels_inside_roi == 25        # cols/rows 0..4 by centre
    assert loose.pixels_inside_roi > strict.pixels_inside_roi


# --------------------------------------------------------------------------- #
# transforms: rotated / sheared, non-square
# --------------------------------------------------------------------------- #
def test_rotated_affine_is_supported():
    roi = box(377200.0, 3430000.0, 390000.0, 3445000.0)
    res = calculate_roi_ndvi_stats(GRID, roi, ROTATED, FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi > 0
    assert res.pixel_area_m2 == pytest.approx(106.0)
    # independent check: mask every pixel centre with shapely
    rows, cols = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    xs, ys = ROTATED * (cols, rows)
    expected = sum(roi.contains(__import__("shapely").geometry.Point(x, y))
                   for x, y in zip(xs.ravel(), ys.ravel()))
    assert res.pixels_inside_roi == expected


def test_rotated_affine_area_matches_the_determinant():
    roi = box(377200.0, 3430000.0, 390000.0, 3445000.0)
    res = calculate_roi_ndvi_stats(GRID, roi, ROTATED, FULL_MASK, crs=UTM36)
    assert res.valid_area_m2 == pytest.approx(res.valid_pixels * 106.0)


def test_non_square_pixels():
    roi = box_roi(0, 0, 5, 5, transform=NONSQUARE)
    res = calculate_roi_ndvi_stats(GRID, roi, NONSQUARE, FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi == 25
    assert res.pixel_width == 20.0 and res.pixel_height == 10.0
    assert res.valid_area_m2 == pytest.approx(25 * 200.0)


# --------------------------------------------------------------------------- #
# windowed workflow
# --------------------------------------------------------------------------- #
def test_windowed_result_equals_full_array_result():
    """The window optimisation must not change ANY number."""
    roi = box_roi(2, 3, 8, 9)
    windowed = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    assert windowed.window is not None
    rows, cols, h, w = windowed.window
    assert (h, w) == (6, 6)                        # only the ROI's window

    # reference: mask the whole raster without windowing
    from rasterio.features import geometry_mask

    full_inside = geometry_mask([roi], out_shape=(H, W), transform=TEN_M,
                                all_touched=False, invert=True)
    expected_valid = int(np.count_nonzero(full_inside & FULL_MASK))
    assert windowed.pixels_inside_roi == int(np.count_nonzero(full_inside))
    assert windowed.valid_pixels == expected_valid
    assert windowed.stats["mean"] == pytest.approx(GRID[full_inside].mean())


def test_windowed_result_equals_full_array_result_on_a_rotated_grid():
    roi = box(377200.0, 3430000.0, 390000.0, 3445000.0)
    windowed = calculate_roi_ndvi_stats(GRID, roi, ROTATED, FULL_MASK, crs=UTM36)
    from rasterio.features import geometry_mask

    full_inside = geometry_mask([roi], out_shape=(H, W), transform=ROTATED,
                                all_touched=False, invert=True)
    assert windowed.pixels_inside_roi == int(np.count_nonzero(full_inside))
    assert windowed.stats["mean"] == pytest.approx(GRID[full_inside].mean())


def test_window_is_clipped_to_the_raster():
    roi = box_roi(-100, -100, 5, 5)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    rows, cols, h, w = res.window
    assert rows == 0 and cols == 0
    assert h <= H and w <= W


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_crs_mismatch_raises_a_clear_error():
    roi = box_roi(0, 0, 4, 4)
    with pytest.raises(Exception) as exc:
        calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36, roi_crs=WGS84)
    assert "EPSG:4326" in str(exc.value) and "EPSG:32636" in str(exc.value)


def test_matching_crs_is_accepted():
    roi = box_roi(0, 0, 4, 4)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36, roi_crs=UTM36)
    assert res.has_stats


def test_shape_mismatch_between_data_and_mask_raises():
    with pytest.raises(ValueError):
        calculate_roi_ndvi_stats(GRID, box_roi(0, 0, 2, 2), TEN_M,
                                 np.ones((5, 5), dtype=bool), crs=UTM36)


def test_non_2d_input_raises():
    with pytest.raises(ValueError):
        calculate_roi_ndvi_stats(np.zeros((3, 3, 3)), box_roi(0, 0, 2, 2), TEN_M,
                                 crs=UTM36)


def test_geographic_raster_crs_uses_geodesic_area():
    """A lon/lat 'raster': area must not be degrees squared."""
    geo_t = Affine(0.01, 0.0, 31.0, 0.0, -0.01, 31.5)
    data = np.full((20, 20), 0.5)
    roi = box(31.05, 31.30, 31.13, 31.42)
    res = calculate_roi_ndvi_stats(data, roi, geo_t, np.ones((20, 20), bool), crs=WGS84)
    # cols 5..12 (8) x rows 8..19 (12) = 96 pixel centres inside
    assert res.valid_pixels == 96
    assert res.area_m2 > 1e6                       # ~0.08 deg box near 31 N
    assert res.area_m2 != 0.0064                   # definitely not degrees squared


# --------------------------------------------------------------------------- #
# multi-part ROI + serialisation
# --------------------------------------------------------------------------- #
def test_multipolygon_roi():
    part_a = box_roi(0, 0, 3, 3)
    part_b = box_roi(6, 6, 9, 9)
    res = calculate_roi_ndvi_stats(GRID, MultiPolygon([part_a, part_b]), TEN_M,
                                   FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi == 9 + 9
    assert res.stats["mean"] == pytest.approx(
        np.concatenate([GRID[0:3, 0:3].ravel(), GRID[6:9, 6:9].ravel()]).mean()
    )


def test_to_dict_is_json_safe():
    roi = box_roi(1, 1, 4, 4)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    data = res.to_dict()
    import json

    json.dumps(data)                                # must not raise
    assert data["pixels_inside_roi"] == 9
    assert data["area_hectares"] == pytest.approx(0.09)
    assert data["all_touched"] is False


def test_message_describes_the_measurement_without_judgement():
    roi = box_roi(0, 0, 5, 5)
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    text = (res.message + " " + " ".join(res.warnings)).lower()
    assert "mean ndvi" in text
    for forbidden in ("healthy", "unhealthy", "suitable", "cotton", "yield", "excellent"):
        assert forbidden not in text


def test_pixel_centre_exactly_on_the_boundary_is_included():
    """Documents the tie-breaking rule: `geometry_mask` includes a pixel whose
    centre lies exactly ON the polygon boundary (closed test)."""
    roi = box_roi(0, 0, 5.5, 5.5)          # boundary passes through centre of col/row 5
    res = calculate_roi_ndvi_stats(GRID, roi, TEN_M, FULL_MASK, crs=UTM36)
    assert res.pixels_inside_roi == 36     # cols/rows 0..5 -> 6 x 6


def test_the_native_array_is_never_mutated():
    snapshot = GRID.copy()
    mask_snapshot = FULL_MASK.copy()
    calculate_roi_ndvi_stats(GRID, box_roi(1, 1, 6, 6), TEN_M, FULL_MASK, crs=UTM36)
    assert np.array_equal(GRID, snapshot)
    assert np.array_equal(FULL_MASK, mask_snapshot)


def test_roi_inside_a_region_of_all_invalid_pixels_near_valid_ones():
    """The ROI and the validity mask are independent: geometry alone is not data."""
    data = GRID.copy()
    data[:, :] = 0.9
    data[0:5, 0:5] = np.nan                # north-west quadrant is nodata
    mask = np.isfinite(data)
    nw = box_roi(0, 0, 5, 5)
    se = box_roi(5, 5, 10, 10)
    res_nw = calculate_roi_ndvi_stats(data, nw, TEN_M, mask, crs=UTM36)
    res_se = calculate_roi_ndvi_stats(data, se, TEN_M, mask, crs=UTM36)
    assert res_nw.pixels_inside_roi == 25 and res_nw.valid_pixels == 0
    assert res_nw.stats is None and res_nw.message == NO_VALID_MESSAGE
    assert res_se.valid_pixels == 25 and res_se.stats["mean"] == pytest.approx(0.9)
