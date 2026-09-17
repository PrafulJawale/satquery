"""Phase 1 unit tests.

Run:  python -m pytest tests -q

These tests are the reason we can trust the app. They encode the mistakes we
are most likely to make later: NAD27 vs WGS84 confusion, silent nodata in
statistics, undensified footprints, non-JSON-serialisable metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.raster import (
    approx_area_km2,
    densify_ring,
    describe_path,
    footprint_wgs84,
    is_north_up,
    open_dataset,
    pixel_size,
    window_stats,
)
from core.samples import sample_path

# --------------------------------------------------------------------------- #
# pure helpers (no file needed)
# --------------------------------------------------------------------------- #
def test_densify_ring_inserts_points():
    ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    dense = densify_ring(ring, points_per_edge=8)
    assert len(dense) == 32                       # 4 edges * 8 points
    assert dense[0] == (0.0, 0.0)
    assert (10.0, 0.0) in dense                   # corners are preserved
    # halfway along the first edge
    assert any(abs(x - 5.0) < 1e-9 and abs(y - 0.0) < 1e-9 for x, y in dense)


def test_area_of_known_square_is_sane():
    # ~1 degree square at the equator -> roughly 12,000 km2
    ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    area = approx_area_km2(ring)
    assert area is not None
    assert 11_000 < area < 13_000


def test_area_needs_three_points():
    assert approx_area_km2(None) is None
    assert approx_area_km2([(0.0, 0.0), (1.0, 1.0)]) is None


# --------------------------------------------------------------------------- #
# RGB.byte.tif -- known values from gdalinfo
# --------------------------------------------------------------------------- #
def test_rgb_byte_known_values():
    info = describe_path(str(sample_path("RGB.byte.tif")))
    assert (info.width, info.height, info.count) == (791, 718, 3)
    assert info.dtypes == ("uint8", "uint8", "uint8")
    assert info.spatial.crs_epsg == 32618
    assert info.spatial.is_projected is True
    assert info.spatial.is_geographic is False
    assert tuple(round(v, 3) for v in info.spatial.bounds_native) == (
        101985.0,
        2611485.0,
        339315.0,
        2826915.0,
    )
    assert info.nodata_per_band == (0.0, 0.0, 0.0)
    assert info.spatial.is_north_up is True
    assert info.spatial.is_rotated is False


def test_rgb_byte_flagged_as_not_index_valid():
    """8-bit display products must be flagged: NDVI from them would be wrong."""
    info = describe_path(str(sample_path("RGB.byte.tif")))
    assert any("8-BIT" in w for w in info.warnings)
    assert info.is_analysis_ready_int is False


def test_rgb_byte_wgs84_bounds_are_geographic():
    info = describe_path(str(sample_path("RGB.byte.tif")))
    lon_min, lat_min, lon_max, lat_max = info.spatial.bounds_wgs84
    assert -180 <= lon_min < lon_max <= 180
    assert -90 <= lat_min < lat_max <= 90
    # EPSG:32618 is UTM zone 18N -> somewhere in the Americas, northern hemisphere
    assert -90 < lon_max < 0
    assert lat_min > 0


def test_footprint_is_densified_not_just_corners():
    with open_dataset(str(sample_path("RGB.byte.tif"))) as ds:
        foot = footprint_wgs84(ds)
    assert foot is not None
    assert len(foot) == 64
    # densification must actually change the outline vs 4 corners
    assert len({(round(x, 9), round(y, 9)) for x, y in foot}) > 4


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_missing_crs_is_reported_not_guessed():
    info = describe_path(str(sample_path("float_nan.tif")))
    assert info.spatial.has_crs is False
    assert info.spatial.crs_epsg is None
    assert info.spatial.footprint_wgs84 is None
    assert info.spatial.bounds_wgs84 is None
    assert any(w.startswith("NO CRS") for w in info.warnings)


def test_rotated_transform_is_detected():
    info = describe_path(str(sample_path("rotated.tif")))
    assert info.spatial.is_rotated is True
    assert info.spatial.is_north_up is False
    assert any(w.startswith("ROTATED") for w in info.warnings)


def test_pixel_size_is_reported_positive():
    with open_dataset(str(sample_path("RGB.byte.tif"))) as ds:
        sx, sy = pixel_size(ds.transform)
        assert is_north_up(ds.transform)
    assert sx == pytest.approx(300.0379, abs=1e-3)
    assert sy == pytest.approx(300.0418, abs=1e-3)
    assert sx > 0 and sy > 0


def test_all_nodata_file_reports_zero_valid_pixels():
    """The whole point: a naive mean() would return 0.0 and look plausible."""
    with open_dataset(str(sample_path("all-nodata.tif"))) as ds:
        stats = window_stats(ds, band=1)
    assert stats["all_nodata"] is True
    assert stats["valid_pixels"] == 0
    assert stats["mean"] is None
    assert stats["min"] is None and stats["max"] is None


def test_all_nodata_file_has_named_bands():
    info = describe_path(str(sample_path("all-nodata.tif")))
    assert info.count == 4
    assert [b.name for b in info.bands] == ["blue", "green", "red", "nir"]
    assert info.is_multispectral_candidate is True
    assert info.is_analysis_ready_int is True  # uint16 reflectance


def test_nad27_is_actually_transformed():
    """EPSG:26711 (NAD27) -> WGS84 must shift by tens of metres, not zero."""
    info = describe_path(str(sample_path("byte.tif")))
    assert info.spatial.crs_epsg == 26711
    native = info.spatial.bounds_native
    wgs = info.spatial.bounds_wgs84
    assert native is not None and wgs is not None
    # a real datum shift happened: the numbers are not merely copied through
    assert abs(wgs[0] - native[0]) > 1.0
    assert -125 < wgs[0] < -116   # UTM zone 11N, western USA
    assert 33 < wgs[1] < 35


def test_no_nodata_declared_is_warned():
    info = describe_path(str(sample_path("byte.tif")))
    assert info.nodata_per_band == (None,)
    assert any("NO NODATA" in w for w in info.warnings)


def test_single_band_is_not_multispectral_candidate():
    info = describe_path(str(sample_path("byte.tif")))
    assert info.is_multispectral_candidate is False


# --------------------------------------------------------------------------- #
# serialisation contract
# --------------------------------------------------------------------------- #
def test_raster_info_is_json_serialisable():
    for name in ("RGB.byte.tif", "byte.tif", "float_nan.tif", "rotated.tif", "all-nodata.tif"):
        info = describe_path(str(sample_path(name)))
        payload = json.dumps(info.to_dict())      # raises if not serialisable
        assert isinstance(payload, str)
        d = json.loads(payload)
        assert d["width"] == info.width
        assert d["spatial"]["transform"] == list(info.spatial.transform)
        assert len(d["bands"]) == info.count
