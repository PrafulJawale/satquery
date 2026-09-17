"""Phase 10 -- temporal NDVI change: unit tests.

Two kinds of test live here, on purpose:

1. **Synthetic** (the majority). Tiny 2-band rasters with hand-chosen numbers
   exercise the edge cases -- nodata, grid mismatch, date ordering, constant
   values -- because those are almost impossible to find in real data and must
   be provable rather than observed.

2. **Real data** (clearly marked). The two bundled Sentinel-2 acquisitions are
   compared over an 8x8 pixel block whose expected values were computed
   independently from raw digital numbers. That is the test that says the
   engine works on the actual product, not only on fixtures.

Every test asserts on the STATUS as well as the numbers: a phase-10 result that
reports statistics when it should have refused is a worse bug than one that
refuses when it should have reported.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box

from analyses.base import AnalysisContext, Status
from analyses.ndvi_change import (
    CHANGE_DECREASE,
    CHANGE_INCREASE,
    CHANGE_INSUFFICIENT,
    CHANGE_STABLE,
    NDVIChangeResult,
    compare_ndvi,
    compose_message,
    run_ndvi_change,
)
from analyses.registry import route
from core.router import Intent, parse_query
from core.roi import ROISelection
from core.temporal import (
    ScenePair,
    SceneRef,
    discover_scenes,
    grids_identical,
    load_temporal_config,
    parse_date,
    plan_alignment,
    validate_pair,
)

SAMPLE_DIR = "data/sample"
REAL_AFTER = f"{SAMPLE_DIR}/s2_s2b-36ruv-20230806-0-l2a_2048px.tif"
REAL_BEFORE = f"{SAMPLE_DIR}/s2_s2b-36ruv-20230118-0-l2a_2048px.tif"

CRS_UTM = "EPSG:32636"


# =========================================================================== #
# helpers -- build tiny scenes on disk
# =========================================================================== #
def write_scene(path: str,
                red: np.ndarray,
                nir: np.ndarray,
                transform: Affine,
                crs: str = CRS_UTM,
                nodata: Optional[float] = None) -> str:
    """Write a 2-band (1=red, 2=nir) float32 GeoTIFF."""
    arr = np.stack([red.astype("float32"), nir.astype("float32")], axis=0)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": 2,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        dst.set_band_description(1, "B04_red_665nm")
        dst.set_band_description(2, "B08_nir_842nm")
    return path


def scene(path: str,
          when: Any = "2023-01-18",
          *,
          scale: float = 1.0,
          offset: float = 0.0,
          red_index: int = 1,
          nir_index: int = 2) -> SceneRef:
    """A SceneRef over a tiny synthetic raster. Values are already reflectance
    (scale=1, offset=0) so the expected NDVI is arithmetic, not a DN guess."""
    return SceneRef.from_path(path, date=when, red_index=red_index,
                              nir_index=nir_index, scale=scale, offset=offset)


def ndvi_of(red: Any, nir: Any) -> Any:
    """The textbook formula, computed independently of the engine."""
    red = np.asarray(red, dtype="float64")
    nir = np.asarray(nir, dtype="float64")
    den = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(den) > 1e-6, (nir - red) / np.where(den == 0, 1, den), np.nan)
    return out


def nir_for(red: Any, target_ndvi: Any) -> Any:
    """The NIR value that makes (NIR-RED)/(NIR+RED) equal `target_ndvi`.

    Inverts NDVI exactly: nir = red * (1 + ndvi) / (1 - ndvi). Used so the
    expected NDVI of a fixture is arithmetic instead of a guess.
    """
    red = np.asarray(red, dtype="float64")
    target = np.asarray(target_ndvi, dtype="float64")
    return (red * (1.0 + target) / (1.0 - target)).astype("float32")


def config_with(**overrides: Any) -> Dict[str, Any]:
    """The shipped configuration, with selected values replaced."""
    cfg = load_temporal_config()
    for key, value in overrides.items():
        section, name = key.split("__", 1)
        cfg.setdefault(section, {})[name] = value
    return cfg


def pair(before: SceneRef, after: SceneRef) -> ScenePair:
    return ScenePair(before=before, after=after)


def run(before: SceneRef, after: SceneRef, geometry: Any, **cfg: Any):
    """compare_ndvi with an explicit config when one is given."""
    return compare_ndvi(pair(before, after), geometry,
                        config=config_with(**cfg) if cfg else None)


# =========================================================================== #
# fixtures
# =========================================================================== #
@pytest.fixture
def square_scenes(tmp_path) -> Tuple[SceneRef, SceneRef, Any]:
    """A 10x10 scene pair on one identical grid, plus an ROI covering it.

    NDVI is constructed to be exactly known:
        before: red=0.2, nir=0.4  -> NDVI = 0.3333...
        after : red=0.2, nir=0.6  -> NDVI = 0.5
    """
    transform = from_origin(400000.0, 3450000.0, 10.0, 10.0)
    red_b = np.full((10, 10), 0.2, dtype="float32")
    nir_b = np.full((10, 10), 0.4, dtype="float32")
    red_a = np.full((10, 10), 0.2, dtype="float32")
    nir_a = np.full((10, 10), 0.6, dtype="float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform), "2023-08-06")
    return b, a, box(400000.0, 3450000.0 - 100.0, 400100.0, 3450000.0)


# =========================================================================== #
# 1. NDVI before / after / delta
# =========================================================================== #
def test_ndvi_before_is_correct(square_scenes):
    before, after, geom = square_scenes
    result, status, message, _ = run(before, after, geom)
    assert status is Status.OK, message
    # red 0.2 / nir 0.4 -> (0.4-0.2)/(0.4+0.2) = 1/3
    assert result.before_mean == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert result.before_median == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert result.before_std == pytest.approx(0.0, abs=1e-6)


def test_ndvi_after_is_correct(square_scenes):
    before, after, geom = square_scenes
    result, status, message, _ = run(before, after, geom)
    assert status is Status.OK, message
    # red 0.2 / nir 0.6 -> (0.6-0.2)/(0.6+0.2) = 0.5
    assert result.after_mean == pytest.approx(0.5, abs=1e-6)
    assert result.after_median == pytest.approx(0.5, abs=1e-6)


def test_delta_is_after_minus_before(square_scenes):
    before, after, geom = square_scenes
    result, status, message, _ = run(before, after, geom)
    assert status is Status.OK, message
    assert result.delta_mean == pytest.approx(0.5 - 1.0 / 3.0, abs=1e-6)
    assert result.delta_median == pytest.approx(0.5 - 1.0 / 3.0, abs=1e-6)
    # every comparable cell is an increase of the same size
    assert result.increased_count == 100
    assert result.decreased_count == 0
    assert result.stable_count == 0


def test_delta_raster_matches_hand_computation(tmp_path):
    """Pixel-by-pixel: the engine's delta equals (after NDVI - before NDVI)."""
    transform = from_origin(500000.0, 3450000.0, 10.0, 10.0)
    rng = np.random.default_rng(20240916)
    red_b = rng.uniform(0.05, 0.3, (8, 8)).astype("float32")
    nir_b = rng.uniform(0.3, 0.8, (8, 8)).astype("float32")
    red_a = rng.uniform(0.05, 0.3, (8, 8)).astype("float32")
    nir_a = rng.uniform(0.3, 0.8, (8, 8)).astype("float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform), "2023-08-06")
    geom = box(500000.0, 3450000.0 - 80.0, 500080.0, 3450000.0)

    result, status, message, _ = run(b, a, geom)
    assert status is Status.OK, message
    expected = ndvi_of(red_a, nir_a) - ndvi_of(red_b, nir_b)

    delta = np.asarray(result.change_raster)
    got = delta[np.isfinite(delta)]
    assert got.size == 64
    np.testing.assert_allclose(got, expected.ravel(), rtol=1e-5, atol=1e-6)


# =========================================================================== #
# 2. valid-pixel intersection and nodata
# =========================================================================== #
def test_valid_pixels_are_the_intersection_of_both_dates(tmp_path):
    """A pixel valid on ONE date only must not be compared, and must not be
    counted as a change of zero."""
    transform = from_origin(600000.0, 3450000.0, 10.0, 10.0)
    red_b = np.full((6, 6), 0.2, dtype="float32")
    nir_b = np.full((6, 6), 0.4, dtype="float32")
    nir_b[0, 0] = np.nan                    # before invalid here
    red_a = np.full((6, 6), 0.2, dtype="float32")
    nir_a = np.full((6, 6), 0.6, dtype="float32")
    nir_a[0, 1] = np.nan                    # after invalid here
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform), "2023-08-06")
    geom = box(600000.0, 3450000.0 - 60.0, 600060.0, 3450000.0)

    result, status, message, _ = run(b, a, geom)
    assert status is Status.OK, message
    assert result.roi_cell_count == 36
    assert result.valid_pixel_count == 34          # 2 of 36 dropped
    assert result.insufficient_count == 2          # reported, not hidden as stable
    assert (result.increased_count + result.decreased_count
            + result.stable_count) == 34           # only comparable cells are classified


def test_nodata_value_is_honoured(tmp_path):
    """A declared nodata value is not a real measurement."""
    transform = from_origin(610000.0, 3450000.0, 10.0, 10.0)
    red_b = np.full((5, 5), 0.2, dtype="float32")
    nir_b = np.full((5, 5), 0.4, dtype="float32")
    nir_b[2, 2] = -9999.0
    red_a = np.full((5, 5), 0.2, dtype="float32")
    nir_a = np.full((5, 5), 0.4, dtype="float32")
    b_path = write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform, nodata=-9999.0)
    a_path = write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform, nodata=-9999.0)
    b = scene(b_path, "2023-01-18")
    a = scene(a_path, "2023-08-06")
    geom = box(610000.0, 3450000.0 - 50.0, 610050.0, 3450000.0)

    result, status, message, _ = run(b, a, geom)
    assert status is Status.OK, message
    assert result.valid_pixel_count == 24          # the nodata cell is excluded
    assert result.insufficient_count == 1


# =========================================================================== #
# 3. dates
# =========================================================================== #
def test_missing_date_returns_needs_two_dates(square_scenes):
    before, after, geom = square_scenes
    before.date = None
    result, status, message, _ = run(before, after, geom)
    assert result is None
    assert status is Status.NEEDS_TWO_DATES
    assert "date" in message.lower()


def test_equal_dates_returns_needs_two_dates(square_scenes):
    before, after, geom = square_scenes
    after.date = before.date
    result, status, message, _ = run(before, after, geom)
    assert result is None
    assert status is Status.NEEDS_TWO_DATES
    assert "different" in message.lower()


def test_inverted_dates_are_refused(square_scenes):
    """before later than after is a contradiction, not a negative change."""
    before, after, geom = square_scenes
    before.date, after.date = date(2023, 8, 6), date(2023, 1, 18)
    result, status, message, _ = run(before, after, geom)
    assert result is None
    assert status is Status.ERROR
    assert "later than" in message


def test_missing_scene_returns_needs_two_dates(square_scenes):
    before, _after, geom = square_scenes
    result, status, message, _ = run(before, None, geom)
    assert result is None
    assert status is Status.NEEDS_TWO_DATES
    assert "two acquisitions" in message.lower()


def test_parse_date_rejects_garbage():
    assert parse_date("2023-01-18") == date(2023, 1, 18)
    assert parse_date("2023-08-06T08:41:42.997000Z") == date(2023, 8, 6)
    assert parse_date("not a date") is None
    assert parse_date(None) is None
    assert parse_date("") is None


# =========================================================================== #
# 4. required bands
# =========================================================================== #
def test_scene_without_nir_returns_unsupported(tmp_path):
    """A scene with only a red band cannot produce NDVI -- say so, do not guess."""
    transform = from_origin(700000.0, 3450000.0, 10.0, 10.0)
    red = np.full((4, 4), 0.2, dtype="float32")
    nir = np.full((4, 4), 0.4, dtype="float32")
    write_scene(str(tmp_path / "b.tif"), red, nir, transform)
    write_scene(str(tmp_path / "a.tif"), red, nir, transform)
    b = scene(str(tmp_path / "b.tif"), "2023-01-18", nir_index=5)   # no such band
    a = scene(str(tmp_path / "a.tif"), "2023-08-06")
    assert not b.has_required_bands
    result, status, message, _ = run(b, a, box(700000.0, 3450000.0 - 40.0, 700040.0, 3450000.0))
    assert result is None
    assert status is Status.UNSUPPORTED
    assert "near-infrared" in message.lower()


# =========================================================================== #
# 5. spatial overlap and coverage
# =========================================================================== #
def test_non_overlapping_scenes_are_refused(tmp_path):
    """Two scenes that share no ground: there is nothing to difference."""
    t1 = from_origin(100000.0, 3450000.0, 10.0, 10.0)
    t2 = from_origin(900000.0, 3450000.0, 10.0, 10.0)     # 800 km away
    red = np.full((4, 4), 0.2, dtype="float32")
    nir = np.full((4, 4), 0.4, dtype="float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red, nir, t1), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red, nir, t2), "2023-08-06")
    result, status, message, _ = compare_ndvi(pair(b, a), None)
    assert result is None
    assert status is Status.NO_TEMPORAL_OVERLAP
    assert "common area" in message.lower()


def test_roi_outside_one_scene_is_refused(tmp_path):
    """The ROI must be inside BOTH scenes. A partial coverage is not enough."""
    transform = from_origin(800000.0, 3450000.0, 10.0, 10.0)
    red = np.full((4, 4), 0.2, dtype="float32")
    nir = np.full((4, 4), 0.4, dtype="float32")
    write_scene(str(tmp_path / "b.tif"), red, nir, transform)
    # the after scene stops short: 2x4 instead of 4x4
    write_scene(str(tmp_path / "a.tif"), red[:, :2], nir[:, :2], transform)
    b = scene(str(tmp_path / "b.tif"), "2023-01-18")
    a = scene(str(tmp_path / "a.tif"), "2023-08-06")
    geom = box(800000.0, 3450000.0 - 40.0, 800040.0, 3450000.0)   # full 4x4
    result, status, message, _ = run(b, a, geom)
    assert result is None
    assert status is Status.NO_TEMPORAL_OVERLAP
    assert "both acquisitions" in message.lower()


# =========================================================================== #
# 6. grids and alignment
# =========================================================================== #
def test_identical_grids_are_detected(square_scenes):
    before, after, _ = square_scenes
    assert grids_identical(before, after) is True
    alignment = plan_alignment(before, after, box(400000.0, 3449900.0, 400100.0, 3450000.0))
    assert alignment.method == "identical_grid"
    assert alignment.resampling is None
    assert alignment.resampled is False


def test_identical_grids_are_not_resampled(square_scenes):
    before, after, geom = square_scenes
    result, status, message, _ = run(before, after, geom)
    assert status is Status.OK, message
    assert result.alignment["method"] == "identical_grid"
    assert result.alignment["resampled"] is False
    assert result.alignment["resolution"] == pytest.approx(10.0)
    assert result.resolution == pytest.approx(10.0)


def test_mismatched_grids_are_aligned_on_one_common_grid(tmp_path):
    """A 10 m scene and a 20 m scene covering the same ground: the comparison
    must be made on ONE explicit grid, at the COARSER resolution."""
    fine = from_origin(200000.0, 3450000.0, 10.0, 10.0)
    coarse = from_origin(200000.0, 3450000.0, 20.0, 20.0)
    red_b = np.full((20, 20), 0.2, dtype="float32")      # 200 m at 10 m
    nir_b = np.full((20, 20), 0.4, dtype="float32")
    red_a = np.full((10, 10), 0.2, dtype="float32")      # 200 m at 20 m
    nir_a = np.full((10, 10), 0.6, dtype="float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, fine), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, coarse), "2023-08-06")

    assert grids_identical(b, a) is False
    geom = box(200000.0, 3450000.0 - 200.0, 200200.0, 3450000.0)
    alignment = plan_alignment(b, a, geom)
    assert alignment.method == "common_grid_resampled"
    assert alignment.resolution == pytest.approx(20.0)   # the coarser of 10 and 20
    assert alignment.resampling == "nearest"

    result, status, message, _ = run(b, a, geom)
    assert status is Status.OK, message
    assert result.alignment["resampled"] is True
    assert result.resolution == pytest.approx(20.0)
    # resampling adds no information: it must be stated in the provenance
    assert result.provenance["resampling_adds_no_information"] is True
    # both scenes hold the same NDVI values, so the aligned difference is 0.5 - 1/3
    assert result.delta_mean == pytest.approx(0.5 - 1.0 / 3.0, abs=1e-5)


def test_alignment_keeps_the_same_ground_together(tmp_path):
    """Two scenes at different resolutions with a step edge: after alignment the
    left and right halves must still be the left and right halves."""
    fine = from_origin(300000.0, 3450000.0, 10.0, 10.0)
    coarse = from_origin(300000.0, 3450000.0, 20.0, 20.0)
    # NDVI 0.2 on the left half, 0.8 on the right half, in both scenes:
    #   red=0.4 nir=0.6 -> 0.2 ;  red=0.1 nir=0.9 -> 0.8
    red_b = np.where(np.arange(20)[None, :] < 10, 0.4, 0.1).astype("float32") * np.ones((20, 1))
    nir_b = np.where(np.arange(20)[None, :] < 10, 0.6, 0.9).astype("float32") * np.ones((20, 1))
    red_a = np.where(np.arange(10)[None, :] < 5, 0.4, 0.1).astype("float32") * np.ones((10, 1))
    nir_a = np.where(np.arange(10)[None, :] < 5, 0.6, 0.9).astype("float32") * np.ones((10, 1))
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, fine), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, coarse), "2023-08-06")
    geom = box(300000.0, 3450000.0 - 200.0, 300200.0, 3450000.0)

    result, status, message, _ = run(b, a, geom)
    assert status is Status.OK, message
    # identical content on both dates -> no change anywhere, including at the edge
    assert result.delta_mean == pytest.approx(0.0, abs=1e-6)
    assert result.delta_min == pytest.approx(0.0, abs=1e-6)
    assert result.delta_max == pytest.approx(0.0, abs=1e-6)
    assert result.stable_count == result.valid_pixel_count


# =========================================================================== #
# 7. classification thresholds
# =========================================================================== #
def test_threshold_classification(tmp_path):
    """delta >= +t -> Increase ; delta <= -t -> Decrease ; else Stable.

    The before scene is NDVI 0 everywhere (red == nir), and the after scene is
    built to have NDVI +0.15 / 0.00 / -0.15 by inverting the formula, so the
    deltas sit clearly either side of the 0.10 threshold and cannot be decided
    by a rounding error.
    """
    transform = from_origin(410000.0, 3450000.0, 10.0, 10.0)
    red_b = np.full((3, 3), 0.25, dtype="float32")
    nir_b = np.full((3, 3), 0.25, dtype="float32")            # NDVI = 0
    targets = np.array([[0.15, 0.0, -0.15]] * 3, dtype="float64")
    red_a = np.full((3, 3), 0.25, dtype="float32")
    nir_a = nir_for(0.25, targets)
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform), "2023-08-06")
    geom = box(410000.0, 3450000.0 - 30.0, 410030.0, 3450000.0)

    result, status, message, _ = run(b, a, geom)
    assert status is Status.OK, message
    assert result.increased_count == 3
    assert result.stable_count == 3
    assert result.decreased_count == 3
    assert result.delta_mean == pytest.approx(0.0, abs=1e-6)   # +0.15, 0, -0.15
    # and the classes are where they should be spatially
    classes = np.asarray(result.class_raster)
    assert (classes[:, 0] == CHANGE_INCREASE).all()
    assert (classes[:, 1] == CHANGE_STABLE).all()
    assert (classes[:, 2] == CHANGE_DECREASE).all()


def test_threshold_is_configurable(tmp_path):
    """The same data with a different threshold classifies differently."""
    transform = from_origin(420000.0, 3450000.0, 10.0, 10.0)
    red_b = np.full((1, 3), 0.25, dtype="float32")
    nir_b = np.full((1, 3), 0.25, dtype="float32")
    targets = np.array([[0.05, 0.0, -0.05]], dtype="float64")
    red_a = np.full((1, 3), 0.25, dtype="float32")
    nir_a = nir_for(0.25, targets)
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform), "2023-08-06")
    geom = box(420000.0, 3450000.0 - 10.0, 420030.0, 3450000.0)

    loose, s1, m1, _ = run(b, a, geom, change__increase_threshold=0.02,
                           change__decrease_threshold=0.02,
                           change__min_valid_pixels=3)
    assert s1 is Status.OK, m1
    assert (loose.increased_count, loose.stable_count, loose.decreased_count) == (1, 1, 1)

    strict, s2, m2, _ = run(b, a, geom, change__increase_threshold=0.20,
                            change__decrease_threshold=0.20,
                            change__min_valid_pixels=3)
    assert s2 is Status.OK, m2
    assert (strict.increased_count, strict.stable_count, strict.decreased_count) == (0, 3, 0)


def test_boundary_is_inclusive_on_both_sides(tmp_path):
    """delta == +t is Increase and delta == -t is Decrease (the documented rule).

    The values are chosen to be EXACT in binary floating point (0.25, 0.75 and
    a threshold of 0.5), so this tests the comparison operator and not the
    rounding of the fixture.
    """
    transform = from_origin(430000.0, 3450000.0, 10.0, 10.0)
    red_b = np.full((1, 2), 0.25, dtype="float32")
    nir_b = np.full((1, 2), 0.25, dtype="float32")            # NDVI = 0
    red_a = np.array([[0.25, 0.75]], dtype="float32")
    nir_a = np.array([[0.75, 0.25]], dtype="float32")         # NDVI = +0.5, -0.5
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform), "2023-08-06")
    geom = box(430000.0, 3450000.0 - 10.0, 430020.0, 3450000.0)
    result, status, message, _ = run(b, a, geom, change__increase_threshold=0.5,
                                     change__decrease_threshold=0.5,
                                     change__min_valid_pixels=2)
    assert status is Status.OK, message
    classes = np.asarray(result.class_raster)
    assert classes[0, 0] == CHANGE_INCREASE      # exactly +0.5 with t = 0.5
    assert classes[0, 1] == CHANGE_DECREASE      # exactly -0.5 with t = 0.5


# =========================================================================== #
# 8. degenerate cases
# =========================================================================== #
def test_all_zero_bands_have_no_valid_pixels(tmp_path):
    """red = nir = 0 everywhere: the NDVI denominator is zero, so nothing is
    defined. Zero change would be a fabrication."""
    transform = from_origin(440000.0, 3450000.0, 10.0, 10.0)
    red = np.zeros((4, 4), dtype="float32")
    nir = np.zeros((4, 4), dtype="float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red, nir, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red, nir, transform), "2023-08-06")
    result, status, message, _ = run(b, a, box(440000.0, 3450000.0 - 40.0, 440040.0, 3450000.0))
    assert result is None
    assert status is Status.NO_VALID_PIXELS
    assert "no pixel" in message.lower()


def test_all_invalid_returns_no_valid_pixels(tmp_path):
    transform = from_origin(450000.0, 3450000.0, 10.0, 10.0)
    red = np.full((4, 4), np.nan, dtype="float32")
    nir = np.full((4, 4), np.nan, dtype="float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red, nir, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red, nir, transform), "2023-08-06")
    result, status, _, _ = run(b, a, box(450000.0, 3450000.0 - 40.0, 450040.0, 3450000.0))
    assert result is None
    assert status is Status.NO_VALID_PIXELS


def test_constant_values_give_zero_change(tmp_path):
    """Nothing changed: the result must say so, not invent a trend."""
    transform = from_origin(460000.0, 3450000.0, 10.0, 10.0)
    red = np.full((5, 5), 0.2, dtype="float32")
    nir = np.full((5, 5), 0.5, dtype="float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red, nir, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red, nir, transform), "2023-08-06")
    result, status, message, _ = run(b, a, box(460000.0, 3450000.0 - 50.0, 460050.0, 3450000.0))
    assert status is Status.OK, message
    assert result.delta_mean == pytest.approx(0.0, abs=1e-7)
    assert result.delta_median == pytest.approx(0.0, abs=1e-7)
    assert result.delta_min == pytest.approx(0.0, abs=1e-7)
    assert result.delta_max == pytest.approx(0.0, abs=1e-7)
    assert result.stable_count == 25
    assert result.increased_count == 0
    assert result.decreased_count == 0
    assert result.net_direction == "stable"


def test_partial_coverage_returns_insufficient_data(tmp_path):
    """Most of the ROI unusable on one date -> refuse, do not report a subset."""
    transform = from_origin(470000.0, 3450000.0, 10.0, 10.0)
    red_b = np.full((10, 10), 0.2, dtype="float32")
    nir_b = np.full((10, 10), 0.4, dtype="float32")
    nir_b[:, :8] = np.nan                    # 80% of the before scene unusable
    red_a = np.full((10, 10), 0.2, dtype="float32")
    nir_a = np.full((10, 10), 0.6, dtype="float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform), "2023-08-06")
    result, status, message, _ = run(b, a, box(470000.0, 3450000.0 - 100.0, 470100.0, 3450000.0))
    assert result is None
    assert status is Status.INSUFFICIENT_DATA
    assert "20.0%" in message                 # the fraction is reported, not hidden


# =========================================================================== #
# 9. wording -- the engine must not claim a cause
# =========================================================================== #
CAUSAL_WORDS = ("deforestation", "crop failure", "flooding", "flood", "drought",
                "harvest", "yield loss")


@pytest.mark.parametrize("direction", ["increase", "decrease", "mixed"])
def test_message_never_claims_a_cause(tmp_path, direction):
    transform = from_origin(480000.0, 3450000.0, 10.0, 10.0)
    red_b = np.full((4, 4), 0.2, dtype="float32")
    nir_b = np.full((4, 4), 0.4, dtype="float32")
    if direction == "increase":
        nir_a = np.full((4, 4), 0.9, dtype="float32")
    elif direction == "decrease":
        nir_a = np.full((4, 4), 0.21, dtype="float32")
    else:
        nir_a = np.full((4, 4), 0.4, dtype="float32")   # no change -> "mixed"
    red_a = np.full((4, 4), 0.2, dtype="float32")
    b = scene(write_scene(str(tmp_path / "b.tif"), red_b, nir_b, transform), "2023-01-18")
    a = scene(write_scene(str(tmp_path / "a.tif"), red_a, nir_a, transform), "2023-08-06")
    result, status, message, _ = run(b, a, box(480000.0, 3450000.0 - 40.0, 480040.0, 3450000.0))
    assert status is Status.OK, message
    full = compose_message(result).lower()
    assert "additional data is required to identify the cause" in full
    # The caveat sentence NAMES the causes it rules out ("...does not by itself
    # establish deforestation, crop failure..."). That is the point of it, so
    # it is excluded before checking that nothing else asserts a cause.
    caveat_start = full.index("additional data is required")
    text = full[:caveat_start]
    for word in CAUSAL_WORDS:
        assert word not in text, f"the message claims '{word}'"


def test_result_carries_provenance_and_limitations(square_scenes):
    before, after, geom = square_scenes
    result, status, message, _ = run(before, after, geom)
    assert status is Status.OK, message
    d = result.to_dict()
    assert "NDVI_after - NDVI_before" in d["provenance"]["formula_delta"]
    assert d["provenance"]["scenes"]["before"]["date"] == "2023-01-18"
    assert d["provenance"]["scenes"]["after"]["date"] == "2023-08-06"
    assert d["thresholds"]["increase"] == 0.10
    assert len(d["limitations"]) >= 3
    assert set(d) >= {"before_date", "after_date", "valid_pixel_count",
                      "before_mean", "after_mean", "delta_mean", "before_median",
                      "after_median", "delta_median", "increased_count",
                      "decreased_count", "stable_count", "insufficient_count",
                      "thresholds", "alignment", "provenance"}
    # the rasters are deliberately NOT part of the JSON view (they are arrays)
    assert result.change_raster is not None
    assert result.class_raster is not None
    assert np.asarray(result.change_raster).shape == np.asarray(result.class_raster).shape


# =========================================================================== #
# 10. engine entry point and routing
# =========================================================================== #
def test_engine_needs_roi_before_anything_else(square_scenes):
    before, after, _ = square_scenes
    ctx = AnalysisContext(roi=None, temporal_pair=pair(before, after))
    execution = run_ndvi_change(ctx, parse_query("Compare NDVI before and after."))
    assert execution.status is Status.NEEDS_ROI
    assert execution.result is None


def test_engine_needs_two_dates(square_scenes):
    before, _after, geom = square_scenes
    roi = ROISelection(is_valid=True, intersects_raster=True, area_m2=10000.0,
                       geometry_raster_crs=geom, raster_crs=CRS_UTM)
    ctx = AnalysisContext(roi=roi, temporal_pair=None)
    execution = run_ndvi_change(ctx, parse_query("Compare NDVI before and after."))
    assert execution.status is Status.NEEDS_TWO_DATES
    assert execution.result is None


def test_engine_runs_end_to_end(square_scenes):
    before, after, geom = square_scenes
    roi = ROISelection(is_valid=True, intersects_raster=True, area_m2=10000.0,
                       geometry_raster_crs=geom, raster_crs=CRS_UTM)
    ctx = AnalysisContext(roi=roi, temporal_pair=pair(before, after))
    execution = run_ndvi_change(ctx, parse_query("Compare NDVI before and after."))
    assert execution.status is Status.OK, execution.message
    assert isinstance(execution.result, NDVIChangeResult)
    assert execution.result.delta_mean == pytest.approx(0.5 - 1.0 / 3.0, abs=1e-6)


def test_routing_returns_needs_two_dates_without_a_pair():
    """The router must ask for the dates, never silently pick two."""
    roi = ROISelection(is_valid=True, intersects_raster=True, area_m2=10000.0,
                       geometry_raster_crs=box(400000.0, 3449900.0, 400100.0, 3450000.0),
                       raster_crs=CRS_UTM)
    ctx = AnalysisContext(roi=roi)
    for text in ("Compare NDVI between these two dates.",
                 "How has vegetation changed in this area?",
                 "Compare before and after."):
        execution = route(text, ctx)
        assert execution.intent in (Intent.NDVI_CHANGE_ROI,
                                    Intent.VEGETATION_CHANGE,
                                    Intent.TEMPORAL_COMPARISON)
        assert execution.status is Status.NEEDS_TWO_DATES, text
        assert execution.result is None


@pytest.mark.parametrize("query", ["Show flood areas.",
                                   "Where did the water extent change?",
                                   "Detect flooding in this area.",
                                   "Use SAR to find flooded fields."])
def test_flood_and_sar_queries_stay_unsupported(query):
    """Flood / SAR change detection is NOT implemented. NDVI cannot answer it."""
    parsed = parse_query(query)
    if parsed.intent is Intent.FLOOD_CHANGE:
        roi = ROISelection(is_valid=True, intersects_raster=True, area_m2=10000.0,
                           geometry_raster_crs=box(400000.0, 3449900.0, 400100.0, 3450000.0),
                           raster_crs=CRS_UTM)
        execution = route(query, AnalysisContext(roi=roi))
        assert execution.status is Status.UNSUPPORTED, query
        assert execution.result is None


def test_temporal_vocabulary_routes_to_the_temporal_engine():
    for text, expected in (
        ("Compare NDVI between these two dates.", Intent.NDVI_CHANGE_ROI),
        ("Show the NDVI difference for this area.", Intent.NDVI_CHANGE_ROI),
        ("Compare before and after.", Intent.TEMPORAL_COMPARISON),
        ("How has vegetation changed in this area?", Intent.VEGETATION_CHANGE),
        ("Find vegetation loss.", Intent.VEGETATION_CHANGE),
    ):
        assert parse_query(text).intent is expected, text


# =========================================================================== #
# 11. REAL DATA -- the comparison that has to work
# =========================================================================== #
real = pytest.mark.skipif(
    not (rasterio and __import__("pathlib").Path(REAL_BEFORE).exists()
         and __import__("pathlib").Path(REAL_AFTER).exists()),
    reason="both Sentinel-2 sample scenes must be present",
)


@real
def test_bundled_scenes_are_discovered_and_dated():
    scenes = discover_scenes(SAMPLE_DIR)
    dates = {s.date.isoformat() for s in scenes}
    assert {"2023-01-18", "2023-08-06"} <= dates, dates
    before = next(s for s in scenes if s.date == date(2023, 1, 18))
    after = next(s for s in scenes if s.date == date(2023, 8, 6))
    assert before.source["mgrs_tile"] == after.source["mgrs_tile"] == "MGRS-36RUV"
    assert before.source["platform"] == after.source["platform"] == "sentinel-2b"
    assert before.scale == after.scale == 0.0001


@real
def test_bundled_scenes_share_one_grid():
    """The two bundled acquisitions were cut from the same pixel window, so the
    real-data comparison is a native-grid difference -- no resampling."""
    before = SceneRef.from_path(REAL_BEFORE)
    after = SceneRef.from_path(REAL_AFTER)
    assert grids_identical(before, after) is True
    assert before.transform == after.transform
    assert before.crs == after.crs


@real
def test_real_data_hand_check_8x8_block():
    """INDEPENDENT HAND-CHECK on real Sentinel-2 data.

    An 8x8 block at row 1000 / column 1000 is read straight from the two
    GeoTIFFs and reduced to NDVI with the textbook formula
    NDVI = (NIR - RED) / (NIR + RED), reflectance = DN / 10000 -- no engine
    code involved. The engine's delta raster must reproduce it exactly.

    Hand-computed reference for the first pixel:
        before: red 265, nir 4720 -> (0.4720-0.0265)/(0.4720+0.0265) = +0.893681
        after : red 242, nir 3480 -> (0.3480-0.0242)/(0.3480+0.0242) = +0.869962
        delta = -0.023719
    """
    from pathlib import Path

    if not (Path(REAL_BEFORE).exists() and Path(REAL_AFTER).exists()):
        pytest.skip("sample scenes not present")

    row_off, col_off, size = 1000, 1000, 8
    with rasterio.open(REAL_BEFORE) as ds:
        transform = ds.transform
        window = rasterio.windows.Window(col_off, row_off, size, size)
        red_b = ds.read(3, window=window).astype("float64")
        nir_b = ds.read(4, window=window).astype("float64")
    with rasterio.open(REAL_AFTER) as ds:
        red_a = ds.read(3, window=window).astype("float64")
        nir_a = ds.read(4, window=window).astype("float64")

    nd_before = ndvi_of(red_b / 10000.0, nir_b / 10000.0)
    nd_after = ndvi_of(red_a / 10000.0, nir_a / 10000.0)
    expected = nd_after - nd_before

    # the hand-computed first pixel
    assert nd_before[0, 0] == pytest.approx(0.893681, abs=1e-5)
    assert nd_after[0, 0] == pytest.approx(0.869962, abs=1e-5)
    assert expected[0, 0] == pytest.approx(-0.023719, abs=1e-5)

    # a geometry EXACTLY covering those 64 pixels
    left = transform.c + col_off * transform.a
    top = transform.f + row_off * transform.e
    geom = box(left, top - size * 10.0, left + size * 10.0, top)

    before = SceneRef.from_path(REAL_BEFORE)
    after = SceneRef.from_path(REAL_AFTER)
    result, status, message, _ = compare_ndvi(pair(before, after), geom)
    assert status is Status.OK, message
    assert result.roi_cell_count == 64
    assert result.valid_pixel_count == 64

    delta = np.asarray(result.change_raster)
    got = delta[np.isfinite(delta)]
    assert got.size == 64
    np.testing.assert_allclose(got, expected.ravel(), rtol=1e-5, atol=1e-6)
    assert result.before_mean == pytest.approx(float(np.mean(nd_before)), abs=1e-6)
    assert result.after_mean == pytest.approx(float(np.mean(nd_after)), abs=1e-6)
    assert result.delta_mean == pytest.approx(float(np.mean(expected)), abs=1e-6)


@real
def test_real_data_comparison_over_a_5km_window():
    """A real, seasonal before/after comparison on the bundled Sentinel-2 pair.

    512 x 512 pixels of the actual Nile Delta window at the native 10 m
    resolution -- 262,144 real cells, comfortably inside the 2,000,000-cell
    budget the rest of the app also uses.
    """
    from pathlib import Path

    if not (Path(REAL_BEFORE).exists() and Path(REAL_AFTER).exists()):
        pytest.skip("sample scenes not present")

    before = SceneRef.from_path(REAL_BEFORE)
    after = SceneRef.from_path(REAL_AFTER)
    minx, miny, maxx, maxy = before.bounds
    geom = box(minx + 2000.0, miny + 2000.0, minx + 2000.0 + 5120.0, miny + 2000.0 + 5120.0)
    assert before.footprint.covers(geom) and after.footprint.covers(geom)

    result, status, message, _ = compare_ndvi(pair(before, after), geom)
    assert status is Status.OK, message

    assert result.before_date == "2023-01-18"
    assert result.after_date == "2023-08-06"
    assert result.roi_cell_count == 512 * 512
    assert result.valid_pixel_count == 512 * 512
    assert result.valid_fraction_of_roi == pytest.approx(1.0, abs=1e-6)
    assert -1.0 <= result.before_mean <= 1.0
    assert -1.0 <= result.after_mean <= 1.0
    assert -1.0 <= result.delta_mean <= 1.0
    assert result.before_median > 0.0                      # a vegetated delta
    assert (result.increased_count + result.decreased_count
            + result.stable_count) == result.valid_pixel_count
    # a real seasonal pair must show change in BOTH directions
    assert result.increased_count > 0
    assert result.decreased_count > 0
    assert result.stable_count > 0
    assert result.alignment["method"] == "identical_grid"
    assert result.alignment["resampled"] is False
    assert result.resolution == pytest.approx(10.0)


@real
def test_an_oversized_selection_is_refused_not_downsampled():
    """A selection larger than the cell budget is refused with an explanation.

    Silently coarsening it would change the question without telling the user,
    which is the same class of error as inventing a date.
    """
    from pathlib import Path

    if not (Path(REAL_BEFORE).exists() and Path(REAL_AFTER).exists()):
        pytest.skip("sample scenes not present")

    before = SceneRef.from_path(REAL_BEFORE)
    after = SceneRef.from_path(REAL_AFTER)
    result, status, message, _ = compare_ndvi(pair(before, after), before.footprint)
    assert result is None
    assert status is Status.INSUFFICIENT_DATA
    assert "too large" in message
    assert "smaller" in message


@real
def test_real_data_roi_outside_coverage_is_refused():
    """A selected area outside BOTH acquisitions gets no numbers at all."""
    before = SceneRef.from_path(REAL_BEFORE)
    after = SceneRef.from_path(REAL_AFTER)
    status, message, _ = validate_pair(pair(before, after),
                                       box(0.0, 0.0, 1000.0, 1000.0))
    assert status.value == "NO_TEMPORAL_OVERLAP"
    assert "not covered by both" in message
