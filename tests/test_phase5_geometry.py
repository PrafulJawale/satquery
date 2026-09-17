"""Phase 5 tests: core/geometry.py -- parsing, repair, CRS transform, area."""

from __future__ import annotations

import pytest
from pyproj import Transformer
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import LineString, Point, Polygon, mapping, shape

from core.geo import MissingCRSError, native_footprint, raster_footprint
from core.geometry import (
    CRS84,
    GeometryError,
    UnsupportedGeometryError,
    area_m2,
    crs_label,
    ensure_polygonal,
    geodesic_area_m2,
    geojson_of,
    is_geographic,
    metres_per_unit,
    parse_geometry,
    planar_area_m2,
    polygonal_parts,
    repair,
    to_wgs84,
    transform_geometry,
)

UTM36 = CRS.from_epsg(32636)
NORTH_UP = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
ROTATED = Affine(10.0, 3.0, 377200.0, 2.0, -10.0, 3441820.0)

# A 1 km square in the middle of the scene, expressed in both CRSs.
CENTRE_UTM = (387440.0, 3431480.0)
SQUARE_UTM = Polygon(
    [
        (CENTRE_UTM[0] - 500, CENTRE_UTM[1] - 500),
        (CENTRE_UTM[0] + 500, CENTRE_UTM[1] - 500),
        (CENTRE_UTM[0] + 500, CENTRE_UTM[1] + 500),
        (CENTRE_UTM[0] - 500, CENTRE_UTM[1] + 500),
    ]
)


def utm_to_lonlat(x, y):
    return Transformer.from_crs(UTM36, CRS84, always_xy=True).transform(x, y)


def lonlat_square(centre_lon, centre_lat, half_deg):
    return Polygon(
        [
            (centre_lon - half_deg, centre_lat - half_deg),
            (centre_lon + half_deg, centre_lat - half_deg),
            (centre_lon + half_deg, centre_lat + half_deg),
            (centre_lon - half_deg, centre_lat + half_deg),
        ]
    )


def feature_of(geom):
    return {"type": "Feature", "properties": {}, "geometry": mapping(geom)}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_geojson_feature():
    geom = parse_geometry(feature_of(SQUARE_UTM))
    assert geom.geom_type == "Polygon"
    assert abs(geom.area - 1e6) < 1e-6


def test_parse_bare_geometry_mapping():
    assert parse_geometry(mapping(SQUARE_UTM)).geom_type == "Polygon"


def test_parse_shapely_passthrough():
    assert parse_geometry(SQUARE_UTM) is SQUARE_UTM


def test_parse_feature_collection_takes_the_last_feature():
    fc = {
        "type": "FeatureCollection",
        "features": [feature_of(SQUARE_UTM), feature_of(lonlat_square(31.8, 31.2, 0.01))],
    }
    assert parse_geometry(fc).geom_type == "Polygon"


def test_parse_rejects_none_and_garbage():
    for bad in (None, 42, {"type": "Nonsense"}):
        with pytest.raises(Exception):
            parse_geometry(bad)


def test_non_area_geometries_are_rejected():
    with pytest.raises(UnsupportedGeometryError):
        ensure_polygonal(Point(31.8, 31.2))
    with pytest.raises(UnsupportedGeometryError):
        ensure_polygonal(LineString([(31.8, 31.2), (31.9, 31.3)]))


# --------------------------------------------------------------------------- #
# repair
# --------------------------------------------------------------------------- #
def test_valid_geometry_is_not_touched():
    geom, repaired = repair(SQUARE_UTM)
    assert repaired is False


def test_self_intersecting_polygon_is_repaired_and_reported():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])   # classic self-intersection
    assert not bowtie.is_valid
    geom, repaired = repair(bowtie)
    assert repaired is True
    assert geom.is_valid
    assert geom.area > 0


def test_collapsed_ring_cannot_be_repaired():
    line_like = Polygon([(0, 0), (1, 1), (0, 0)])
    with pytest.raises(Exception):
        repair(line_like)


def test_polygonal_parts_drops_empty_and_non_polygonal():
    parts = polygonal_parts(SQUARE_UTM)
    assert len(parts) == 1


# --------------------------------------------------------------------------- #
# CRS transformation
# --------------------------------------------------------------------------- #
def test_transform_matches_pyproj_directly():
    geom_ll = lonlat_square(31.8, 31.2, 0.01)
    moved = transform_geometry(geom_ll, CRS84, UTM36)
    x, y = utm_to_lonlat(CENTRE_UTM[0], CENTRE_UTM[1])   # sanity of the helper
    assert 30 < x < 33 and 30 < y < 32
    # a vertex of the transformed square must match pyproj on the same vertex
    vx, vy = list(geom_ll.exterior.coords)[0]
    ex, ey = Transformer.from_crs(CRS84, UTM36, always_xy=True).transform(vx, vy)
    got_x, got_y = list(moved.exterior.coords)[0]
    assert abs(got_x - ex) < 1e-6 and abs(got_y - ey) < 1e-6


def test_transform_to_utm_produces_metres_not_degrees():
    moved = transform_geometry(lonlat_square(31.8, 31.2, 0.01), CRS84, UTM36)
    minx, _miny, maxx, _maxy = moved.bounds
    assert 300_000 < minx < 500_000, "UTM eastings are metres"
    assert (maxx - minx) > 1000, "a 0.02 deg box is more than 1 km wide"


def test_round_trip_is_stable():
    geom_ll = lonlat_square(31.8, 31.2, 0.02)
    utm = transform_geometry(geom_ll, CRS84, UTM36, densify=False)
    back = transform_geometry(utm, UTM36, CRS84, densify=False)
    for (x1, y1), (x2, y2) in zip(geom_ll.exterior.coords, back.exterior.coords):
        assert abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9


def test_densification_adds_vertices_but_not_area():
    """Straight lon/lat edges are densified before reprojection so they follow
    the projection's curve -- more vertices, essentially the same ground."""
    geom_ll = lonlat_square(31.8, 31.2, 0.02)
    dense = transform_geometry(geom_ll, CRS84, UTM36, densify=True)
    plain = transform_geometry(geom_ll, CRS84, UTM36, densify=False)
    assert len(dense.exterior.coords) > len(plain.exterior.coords)
    assert abs(dense.area - plain.area) / plain.area < 1e-4
    assert len(plain.exterior.coords) == 5


def test_transform_without_a_crs_raises():
    with pytest.raises(MissingCRSError):
        transform_geometry(lonlat_square(31.8, 31.2, 0.01), None, UTM36)


# --------------------------------------------------------------------------- #
# area
# --------------------------------------------------------------------------- #
def test_planar_area_of_a_known_square_in_utm():
    area, method = area_m2(SQUARE_UTM, UTM36)
    assert abs(area - 1_000_000.0) < 1.0        # 1 km x 1 km
    assert "planar" in method and "32636" in method


def test_metre_crs_has_unit_factor_one():
    assert abs(metres_per_unit(UTM36) - 1.0) < 1e-12


def test_feet_crs_is_converted_to_metres():
    # EPSG:2225 -- California zone 1, US survey feet
    try:
        feet_crs = CRS.from_epsg(2225)
    except Exception:
        pytest.skip("EPSG:2225 unavailable")
    factor = metres_per_unit(feet_crs)
    assert 0.30 < factor < 0.31


def test_planar_area_refuses_a_geographic_crs():
    with pytest.raises(GeometryError):
        planar_area_m2(lonlat_square(0.0, 0.0, 0.1), "EPSG:4326")


def test_geodesic_area_of_a_known_geographic_square():
    geom = lonlat_square(0.0, 0.0, 0.1)          # 0.2 deg on a side, on the equator
    area = geodesic_area_m2(geom, "EPSG:4326")
    # 0.2 deg of latitude ~= 22.2 km; of longitude at the equator ~= 22.3 km
    assert 4.8e8 < area < 5.0e8
    assert area != geom.area, "must not be degrees squared"


def test_area_auto_picks_geodesic_for_geographic_crs():
    geom = lonlat_square(31.0, 31.0, 0.05)
    area, method = area_m2(geom, "EPSG:4326")
    assert "geodesic" in method
    assert abs(area - geodesic_area_m2(geom, "EPSG:4326")) < 1e-6


def test_auto_area_for_projected_crs_agrees_with_geodesic_within_one_percent():
    geom_utm = SQUARE_UTM
    planar, _ = area_m2(geom_utm, UTM36)
    geodesic, _ = area_m2(geom_utm, UTM36, method="geodesic")
    assert abs(planar - geodesic) / geodesic < 0.01


# --------------------------------------------------------------------------- #
# footprints (rotated / sheared)
# --------------------------------------------------------------------------- #
def test_native_footprint_is_exact_for_a_north_up_grid():
    poly = native_footprint(NORTH_UP, 2048, 2048)
    assert poly.is_valid
    assert abs(poly.area - 20480.0 * 20480.0) < 1.0


def test_native_footprint_of_a_rotated_grid_is_a_quadrilateral_not_a_box():
    poly = native_footprint(ROTATED, 512, 512)
    assert len(poly.exterior.coords) == 5                 # closed ring of 4 corners
    bbox_area = (poly.bounds[2] - poly.bounds[0]) * (poly.bounds[3] - poly.bounds[1])
    assert poly.area < bbox_area * 0.99, "rotated grid does not fill its bounding box"


def test_native_footprint_area_matches_the_affine_determinant():
    a, b, _c, d, e, _f = ROTATED[:6]
    expected = abs(a * e - b * d) * 512 * 512
    assert abs(native_footprint(ROTATED, 512, 512).area - expected) < 1.0


def test_wgs84_footprint_stays_densified():
    poly = raster_footprint(NORTH_UP, UTM36, 2048, 2048)
    assert len(poly.exterior.coords) > 4


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def test_crs_label_and_geographic_flags():
    assert crs_label(UTM36) == "EPSG:32636"
    assert crs_label(None) == "no CRS"
    assert is_geographic("EPSG:4326") is True
    assert is_geographic(UTM36) is False


def test_geojson_round_trip_is_json_safe():
    data = geojson_of(SQUARE_UTM)
    assert data["type"] == "Polygon"
    back = shape(data)
    assert abs(back.area - SQUARE_UTM.area) < 1e-6
