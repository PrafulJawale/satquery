"""Phase 8 -- SYNTHETIC tests for the grid, resampling and slope (no network).

Resampling is the quietest way to corrupt a geospatial result, so both paths are
pinned down with hand-calculable rasters:
    continuous -> bilinear  (a linear field must survive exactly)
    categorical -> nearest  (class codes must never be averaged)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import box

from core.alignment import AnalysisGrid, compute_slope, grid_signature, make_grid, read_into_grid, roi_mask

UTM36N = "EPSG:32636"


def _write(path, array, transform, crs=UTM36N, nodata=None, dtype=None):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0],
                       width=array.shape[1], count=1,
                       dtype=dtype or array.dtype, crs=crs, transform=transform,
                       nodata=nodata) as ds:
        ds.write(array, 1)
    return str(path)


# --------------------------------------------------------------------------- #
# 15. the common analysis grid
# --------------------------------------------------------------------------- #
def test_15_grid_snaps_outward_and_buffers_for_the_slope_kernel():
    grid = make_grid(box(0, 0, 1000, 1000), UTM36N, requested_resolution=30.0)
    assert grid.resolution == 30.0
    # 2 buffer cells (60 m) on each side, snapped outward to whole 30 m cells
    # left  = floor((0 - 60)/30)*30 = -60      right = ceil((1000+60)/30)*30 = 1080
    assert grid.bounds == (-60.0, -60.0, 1080.0, 1080.0)
    assert (grid.width, grid.height) == (38, 38)
    assert grid.transform.a == 30.0 and grid.transform.e == -30.0
    assert grid.transform.c == -60.0 and grid.transform.f == 1080.0
    assert grid.note == ""
    assert not grid.coarsened


def test_15b_cell_cap_coarsens_and_says_so():
    grid = make_grid(box(0, 0, 100_000, 100_000), UTM36N,
                     requested_resolution=30.0, max_cells=2_000_000)
    assert grid.requested_resolution == 30.0
    assert grid.resolution == 120.0           # 30 -> 60 -> 120
    assert grid.coarsened
    assert grid.cells <= 2_000_000
    assert "Requested analysis resolution 30 m" in grid.note
    assert "effective resolution is 120 m" in grid.note


def test_15c_grid_identity_is_stable_and_sensitive():
    a = make_grid(box(0, 0, 1000, 1000), UTM36N, 30.0)
    b = make_grid(box(0, 0, 1000, 1000), UTM36N, 30.0)
    c = make_grid(box(0, 0, 2000, 2000), UTM36N, 30.0)
    assert grid_signature(a) == grid_signature(b)
    assert grid_signature(a) != grid_signature(c)
    assert grid_signature(a, "clay") != grid_signature(a, "sand")


def test_15d_roi_mask_clips_to_the_geometry():
    grid = make_grid(box(0, 0, 1000, 1000), UTM36N, 30.0)
    inside = roi_mask(box(0, 0, 1000, 1000), grid)
    # the ROI covers 1000x1000 m; at 30 m that is 33x33 cell centres inside
    assert inside.shape == (grid.height, grid.width)
    assert int(inside.sum()) == 33 * 33
    small = roi_mask(box(0, 0, 150, 150), grid)
    assert int(small.sum()) == 5 * 5


# --------------------------------------------------------------------------- #
# 16. categorical data is resampled with nearest neighbour
# --------------------------------------------------------------------------- #
def test_16_categorical_resampling_never_invents_a_class(tmp_path):
    classes = np.array([[10, 20], [30, 80]], dtype="uint8")
    src = _write(tmp_path / "lc.tif", classes, from_origin(0, 200, 100, 100),
                 nodata=0)
    grid = AnalysisGrid(crs=UTM36N, transform=from_origin(0, 200, 25, 25),
                        width=8, height=8, resolution=25.0,
                        requested_resolution=25.0,
                        bounds=(0, 0, 200, 200))

    nearest = read_into_grid(src, grid, Resampling.nearest, src_nodata=0)
    assert set(np.unique(nearest[np.isfinite(nearest)])) <= {10.0, 20.0, 30.0, 80.0}

    # ... and the test is meaningful: bilinear WOULD invent intermediate classes
    bilinear = read_into_grid(src, grid, Resampling.bilinear, src_nodata=0)
    invented = set(np.unique(bilinear[np.isfinite(bilinear)])) - {10.0, 20.0, 30.0, 80.0}
    assert invented, "bilinear on class codes must be shown to fabricate values"


# --------------------------------------------------------------------------- #
# 17. continuous data is resampled bilinearly and survives exactly
# --------------------------------------------------------------------------- #
def test_17_bilinear_resampling_of_a_linear_field_is_exact(tmp_path):
    # z(x, y) = 2*(x/100) + 3*((400 - y)/100) is linear, so bilinear resampling
    # must reproduce it at every destination cell centre.
    rows, cols = np.mgrid[0:4, 0:4]
    values = (2.0 * (cols + 0.5) + 3.0 * (rows + 0.5)).astype("float32")
    src = _write(tmp_path / "ramp.tif", values, from_origin(0, 400, 100, 100))

    grid = AnalysisGrid(crs=UTM36N, transform=from_origin(100, 300, 50, 50),
                        width=4, height=4, resolution=50.0,
                        requested_resolution=50.0,
                        bounds=(100, 100, 300, 300))
    out = read_into_grid(src, grid, Resampling.bilinear)

    r, c = np.mgrid[0:4, 0:4]
    x = 100 + 50 * (c + 0.5)
    y = 300 - 50 * (r + 0.5)
    expected = 2.0 * (x / 100.0) + 3.0 * ((400.0 - y) / 100.0)
    assert np.allclose(out, expected, atol=1e-6)
    # hand check of one cell (row 0, col 0): 2*1.25 + 3*1.25 = 6.25
    assert out[0, 0] == pytest.approx(6.25, abs=1e-6)


# --------------------------------------------------------------------------- #
# 18. slope
# --------------------------------------------------------------------------- #
def test_18a_flat_and_constant_surfaces_have_zero_slope():
    flat = np.full((5, 5), 7.0, dtype="float32")
    t = from_origin(0, 150, 30, 30)
    assert np.allclose(compute_slope(flat, t), 0.0)
    assert np.allclose(compute_slope(np.zeros((5, 5), dtype="float32"), t), 0.0)


def test_18b_inclined_plane_slope_is_hand_calculable():
    # 2 m of rise per 30 m cell -> gradient 2/30 -> 6.667 %
    rows, _ = np.mgrid[0:10, 0:10]
    dem = (2.0 * rows).astype("float32")
    slope = compute_slope(dem, from_origin(0, 300, 30, 30))
    expected = (2.0 / 30.0) * 100.0
    # the interior is exact; the outermost row/column returns about half the
    # gradient because the border is replicated (documented edge handling)
    assert np.allclose(slope[1:-1, 1:-1], expected, rtol=1e-3)
    assert slope[5, 5] == pytest.approx(expected, rel=1e-3)
    assert np.allclose(slope[0, :], expected / 2, rtol=1e-3)


def test_18c_a_sloping_plane_in_two_directions_uses_the_hypotenuse():
    rows, cols = np.mgrid[0:10, 0:10]
    dem = (2.0 * rows + 1.0 * cols).astype("float32")   # dz/dy = 2/30, dz/dx = 1/30
    slope = compute_slope(dem, from_origin(0, 300, 30, 30))
    expected = math.hypot(2.0 / 30.0, 1.0 / 30.0) * 100.0
    assert slope[5, 5] == pytest.approx(expected, rel=1e-3)


def test_18d_nodata_propagates_into_the_slope():
    dem = np.full((6, 6), 10.0, dtype="float32")
    dem[3, 3] = np.nan
    slope = compute_slope(dem, from_origin(0, 180, 30, 30))
    assert np.isnan(slope[3, 3])
    # the eight neighbours of a nodata cell cannot be graded honestly either
    assert np.isnan(slope[2, 2]) and np.isnan(slope[4, 4])
    # cells far away are still fine
    assert slope[0, 0] == 0.0


def test_18e_slope_refuses_degree_grids_and_uneven_cells():
    dem = np.zeros((4, 4), dtype="float32")
    geographic = from_origin(31.0, 32.0, 0.0002777, 0.0002777)
    with pytest.raises(ValueError, match="north-up"):
        compute_slope(dem, from_origin(0, 120, 30, 20))
    with pytest.raises(ValueError, match="geographic"):
        compute_slope(dem, geographic, crs="EPSG:4326")


def test_18f_edges_keep_the_array_shape():
    dem = (np.arange(25, dtype="float32") * 2.0).reshape(5, 5)
    slope = compute_slope(dem, from_origin(0, 150, 30, 30))
    assert slope.shape == dem.shape
    assert np.isfinite(slope).all()
