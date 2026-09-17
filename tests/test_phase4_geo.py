"""Phase 4 tests: CRS handling, reprojection, footprints, map-layer encoding.

Run:  python -m pytest tests -q

The theme of these tests: a raster north-up in UTM is NOT north-up in Web
Mercator, so every overlay must be derived from CRS + affine transform, never
from assumed corners.
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.io import MemoryFile

from core.geo import (
    WEB_MERCATOR,
    MissingCRSError,
    footprint_feature,
    grid_bounds_wgs84,
    lonlat_to_pixel,
    pixel_to_crs,
    pixel_to_lonlat,
    raster_footprint,
    reproject_array,
    reproject_bands,
    rgba_to_png_bytes,
    web_rgba_from_bands,
    web_rgba_from_values,
)
from core.indices import ndvi_from_dataset
from core.raster import open_dataset
from core.samples import list_samples, sample_path

S2_SAMPLE = next((n for n in list_samples() if n.startswith("s2_")), None)
needs_s2 = pytest.mark.skipif(S2_SAMPLE is None, reason="Sentinel-2 sample not downloaded")

UTM36 = CRS.from_epsg(32636)
NORTH_UP = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
ROTATED = Affine(10.0, 3.0, 377200.0, 2.0, -10.0, 3441820.0)


def _memory_raster(data: np.ndarray, transform: Affine, crs, nodata=None):
    """Write a single-band in-memory GeoTIFF."""
    mem = MemoryFile()
    ds = mem.open(driver="GTiff", count=1, height=data.shape[0], width=data.shape[1],
                  dtype=data.dtype, crs=crs, transform=transform, nodata=nodata)
    ds.write(data, 1)
    return ds, mem


# --------------------------------------------------------------------------- #
# 1. CRS handling
# --------------------------------------------------------------------------- #
def test_pixel_to_crs_uses_the_affine():
    x, y = pixel_to_crs(NORTH_UP, 5, 7)
    assert x == NORTH_UP.c + 5 * NORTH_UP.a
    assert y == NORTH_UP.f + 7 * NORTH_UP.e


def test_utm_to_wgs84_matches_an_independent_pyproj_transform():
    """Our chain must agree with pyproj used directly on the same point."""
    from pyproj import Transformer

    x, y = pixel_to_crs(NORTH_UP, 1024, 1024)
    expected = Transformer.from_crs(UTM36, "EPSG:4326", always_xy=True).transform(x, y)
    got = pixel_to_lonlat(NORTH_UP, UTM36, 1024, 1024)
    assert got[0] == pytest.approx(expected[0], abs=1e-9)
    assert got[1] == pytest.approx(expected[1], abs=1e-9)


def test_utm_coordinates_are_not_latlon():
    """EPSG:32636 values are metres from a false origin -- not degrees."""
    lon, lat = pixel_to_lonlat(NORTH_UP, UTM36, 0, 0)
    assert -180 <= lon <= 180 and -90 <= lat <= 90
    x, y = pixel_to_crs(NORTH_UP, 0, 0)
    assert x > 300_000 and y > 3_000_000        # what Leaflet must never receive


def test_missing_crs_raises_instead_of_guessing():
    with pytest.raises(MissingCRSError):
        raster_footprint(NORTH_UP, None, 10, 10)
    with pytest.raises(MissingCRSError):
        reproject_array(np.zeros((4, 4), np.float32), NORTH_UP, None)


@needs_s2
def test_sentinel2_sample_is_in_the_expected_place():
    """Sanity: the real scene sits in the Nile Delta, not somewhere random."""
    with open_dataset(str(sample_path(S2_SAMPLE))) as ds:
        lon, lat = pixel_to_lonlat(ds.transform, ds.crs, ds.width / 2, ds.height / 2)
    assert 30.5 < lon < 32.5, f"longitude {lon} outside the expected region"
    assert 30.5 < lat < 32.0, f"latitude {lat} outside the expected region"


# --------------------------------------------------------------------------- #
# 2. footprints
# --------------------------------------------------------------------------- #
def test_footprint_is_densified_and_closed():
    poly = raster_footprint(NORTH_UP, UTM36, 2048, 2048)
    assert len(poly.exterior.coords) > 4, "footprint must be densified, not 4 corners"
    assert poly.exterior.coords[0] == poly.exterior.coords[-1]
    minx, miny, maxx, maxy = poly.bounds
    assert -180 <= minx <= 180 and -90 <= miny <= 90


def test_footprint_area_matches_the_known_ground_extent():
    """2048 px * 10 m = 20.48 km per side -> ~419 km2 (small UTM/Mercator skew)."""
    poly = raster_footprint(NORTH_UP, UTM36, 2048, 2048)
    from pyproj import Geod

    area = abs(Geod(ellps="WGS84").geometry_area_perimeter(poly)[0]) / 1e6
    assert 400 < area < 440, f"footprint area {area:.1f} km2 is implausible"


def test_rotated_transform_gives_a_quadrilateral_not_a_rectangle():
    """A rotated grid must not be described by its bounding box alone."""
    poly = raster_footprint(ROTATED, UTM36, 512, 512)
    minx, miny, maxx, maxy = poly.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    assert poly.area < bbox_area * 0.999, "rotated footprint must be smaller than its bbox"
    assert len(poly.exterior.coords) > 4


def test_footprint_bounds_agree_with_the_grid_bounds_for_north_up():
    poly = raster_footprint(NORTH_UP, UTM36, 1024, 1024)
    grid = grid_bounds_wgs84(NORTH_UP, UTM36, 1024, 1024)
    assert poly.bounds[0] == pytest.approx(grid[0], abs=1e-6)
    assert poly.bounds[2] == pytest.approx(grid[2], abs=1e-6)


def test_footprint_geojson_has_expected_structure():
    poly = raster_footprint(NORTH_UP, UTM36, 256, 256)
    feat = footprint_feature(poly, {"crs": "EPSG:32636"})
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    assert feat["properties"]["crs"] == "EPSG:32636"


# --------------------------------------------------------------------------- #
# 3. reprojection
# --------------------------------------------------------------------------- #
def test_reproject_changes_crs_grid_and_bounds():
    data = np.linspace(100, 5000, 64 * 64, dtype=np.uint16).reshape(64, 64)
    ds, mem = _memory_raster(data, NORTH_UP, UTM36, nodata=0)
    try:
        web = reproject_bands(ds, (1,), dst_crs=WEB_MERCATOR, max_pixels=100_000)
    finally:
        ds.close()
        mem.close()

    assert str(web.crs) == WEB_MERCATOR
    assert str(web.source_crs) == str(UTM36)
    assert web.source_shape == (64, 64)
    assert web.source_transform == NORTH_UP
    assert web.shape[0] * web.shape[1] <= 100_000 * 1.2
    lon_min, lat_min, lon_max, lat_max = web.bounds_wgs84
    assert lon_min < lon_max and lat_min < lat_max
    assert web.leaflet_bounds == [[lat_min, lon_min], [lat_max, lon_max]]


def test_reproject_keeps_source_geometry_metadata_intact():
    """The display copy must remember, and not replace, the analytical geometry."""
    data = np.full((32, 32), 3000, dtype=np.uint16)
    ds, mem = _memory_raster(data, NORTH_UP, UTM36, nodata=0)
    try:
        web = reproject_bands(ds, (1,), dst_crs=WEB_MERCATOR)
    finally:
        ds.close()
        mem.close()
    assert web.source_transform == NORTH_UP
    assert web.source_shape == (32, 32)
    assert str(web.crs) == WEB_MERCATOR and str(web.source_crs) == str(UTM36)


def test_nodata_does_not_bleed_or_become_zero():
    """Half valid, half NaN: the weight-grid trick must keep the valid half clean.

    If NaN were averaged in, values would collapse to NaN or to 0.
    """
    data = np.full((128, 128), 0.8, dtype=np.float32)
    data[:, :64] = np.nan                    # left half invalid
    web = reproject_array(data, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=200_000)

    assert web.mask.mean() == pytest.approx(0.5, abs=0.06)
    valid = web.array[web.mask]
    assert np.nanmin(valid) == pytest.approx(0.8, abs=1e-3), "valid values must survive resampling"
    assert np.nanmax(valid) == pytest.approx(0.8, abs=1e-3)
    # and nothing was silently turned into 0
    assert not np.any(np.nan_to_num(web.array, nan=9.9) == 0.0)


def test_mask_matches_nan_after_reprojection():
    data = np.full((64, 64), 0.3, dtype=np.float32)
    data[10:20, 10:20] = np.nan
    web = reproject_array(data, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR)
    assert np.array_equal(web.mask, np.isfinite(web.array))
    assert np.isnan(web.array[~web.mask]).all()


def test_reprojection_does_not_modify_the_input_array():
    """Analysis first, display second -- the native result must be untouched."""
    data = np.linspace(0.1, 0.9, 32 * 32, dtype=np.float32).reshape(32, 32)
    before = data.copy()
    reproject_array(data, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR)
    assert np.array_equal(data, before), "reprojection must not mutate its input"


@needs_s2
def test_reprojected_ndvi_values_match_the_native_result_at_same_locations():
    """Geographic placement check: same lon/lat -> same value, before and after."""
    with open_dataset(str(sample_path(S2_SAMPLE))) as ds:
        res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        web = reproject_array(res.array, ds.transform, ds.crs, dst_crs=WEB_MERCATOR,
                              max_pixels=1_000_000)

        rng = np.random.default_rng(3)
        rows = rng.integers(50, ds.height - 50, size=8)
        cols = rng.integers(50, ds.width - 50, size=8)
        diffs = []
        for r, c in zip(rows, cols):
            native = float(res.array[r, c])
            if not np.isfinite(native):
                continue
            lon, lat = pixel_to_lonlat(ds.transform, ds.crs, c + 0.5, r + 0.5)
            # locate the same point in the reprojected grid
            x, y = _wgs84_to_mercator(lon, lat)
            from rasterio.transform import rowcol

            row_d, col_d = rowcol(web.transform, x, y)
            if not (0 <= row_d < web.shape[0] and 0 <= col_d < web.shape[1]):
                continue
            reprojected = float(web.array[row_d, col_d])
            if np.isfinite(reprojected):
                diffs.append(abs(reprojected - native))
    assert diffs, "no comparable pixels found"
    # resampling averages neighbours, but over homogeneous farmland the value
    # at the same geographic point must stay close
    assert float(np.median(diffs)) < 0.05, f"median difference {np.median(diffs):.4f} too large"


def _wgs84_to_mercator(lon: float, lat: float):
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True).transform(lon, lat)


@needs_s2
def test_reprojected_bounds_contain_the_raster_footprint():
    with open_dataset(str(sample_path(S2_SAMPLE))) as ds:
        poly = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=64)
        web = reproject_bands(ds, (3, 2, 1), dst_crs=WEB_MERCATOR, max_pixels=600_000)
    lon_min, lat_min, lon_max, lat_max = web.bounds_wgs84
    fx_min, fy_min, fx_max, fy_max = poly.bounds
    assert lon_min <= fx_min + 1e-6 and lon_max >= fx_max - 1e-6
    assert lat_min <= fy_min + 1e-6 and lat_max >= fy_max - 1e-6


@needs_s2
def test_web_mercator_grid_is_larger_than_the_utm_grid_pixels():
    """At 31 deg N, 10 m ground = 10/cos(lat) Mercator units."""
    with open_dataset(str(sample_path(S2_SAMPLE))) as ds:
        web = reproject_bands(ds, (3, 2, 1), dst_crs=WEB_MERCATOR, max_pixels=600_000)
        native_px = abs(ds.transform.a)
    display_px = abs(web.transform.a) * (web.shape[1] / ds.width)
    assert display_px > native_px, "Mercator pixels are stretched relative to ground metres"


# --------------------------------------------------------------------------- #
# 4. map layer encoding
# --------------------------------------------------------------------------- #
def _web_from_values(values: np.ndarray) -> "object":
    return reproject_array(values, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=100_000)


def test_rgb_overlay_alpha_and_colour():
    data = np.stack([np.full((64, 64), 3000, dtype=np.float32)] * 3)
    data[0, 0, 0] = np.nan
    web = reproject_array(data, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=50_000)
    rgba = web_rgba_from_bands(web, ((0.0, 6000.0), (0.0, 6000.0), (0.0, 6000.0)))
    assert rgba.shape == (*web.shape, 4)
    assert rgba.dtype == np.uint8
    assert rgba[..., 3].max() == 255
    if (~web.mask).any():
        assert rgba[~web.mask][:, 3].max() == 0, "invalid pixels must be transparent"
        assert rgba[~web.mask].max() == 0, "invalid pixels must not be painted"


def test_rgb_overlay_preserves_a_given_stretch():
    """The same stretch bounds must give the same 8-bit value as Phase 2."""
    data = np.stack([np.full((64, 64), 3000, dtype=np.float32)] * 3)
    web = reproject_array(data, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=50_000)
    rgba = web_rgba_from_bands(web, ((1000.0, 5000.0), (1000.0, 5000.0), (1000.0, 5000.0)))
    expected = int(round((3000 - 1000) / (5000 - 1000) * 255))
    assert abs(int(rgba[web.mask][:, 0].mean()) - expected) <= 2


def test_ndvi_overlay_transparency_and_colours():
    values = np.full((64, 64), 0.8, dtype=np.float32)
    values[:8, :] = np.nan
    web = reproject_array(values, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=50_000)
    rgba = web_rgba_from_values(web, -1.0, 1.0, colormap="RdYlGn")
    assert rgba.shape == (*web.shape, 4)
    invalid = ~web.mask
    if invalid.any():
        assert rgba[invalid][:, 3].max() == 0
        assert rgba[invalid].max() == 0, "invalid NDVI must never be painted green"
    valid_px = rgba[web.mask]
    assert valid_px[:, 1].mean() > valid_px[:, 0].mean(), "high NDVI should be green-dominant"


def test_rgba_png_roundtrip():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    data = rgba_to_png_bytes(rgba)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_max_pixels_caps_the_display_grid():
    data = np.full((512, 512), 0.5, dtype=np.float32)
    web = reproject_array(data, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=10_000)
    assert web.shape[0] * web.shape[1] <= 10_000 * 1.5
