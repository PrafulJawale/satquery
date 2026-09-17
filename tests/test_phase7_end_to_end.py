"""Phase 7 -- ONE deterministic end-to-end test on the real Sentinel-2 sample.

    REAL SAMPLE -> NDVI confirmation -> ROI context -> query -> router
               -> registry -> Phase 6 engine -> structured result

The claim being proved: **routing does not change the analysis**. The statistics
returned through the router must be identical to calling the Phase 6 engine
directly, on real satellite data.

No network. The sample is bundled in `data/sample/`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from rasterio import Affine
from shapely.geometry import box

from analyses import (
    AnalysisContext,
    Intent,
    NdviContext,
    Status,
    route,
)
from core.indices import ndvi_from_dataset
from core.raster import open_dataset
from core.roi import ROISelection
from core.statistics import calculate_roi_ndvi_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE = REPO_ROOT / "data" / "sample" / "s2_s2b-36ruv-20230806-0-l2a_2048px.tif"

QUERY = "What is the NDVI of this area?"


@lru_cache(maxsize=1)
def load_native_ndvi():
    """Read the sample once per session: native NDVI + mask + georeferencing."""
    if not SCENE.exists():
        pytest.skip(f"sample scene missing: {SCENE}")
    with open_dataset(str(SCENE)) as ds:
        result, _spec, _report = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        return {
            "array": np.asarray(result.array, dtype="float32"),
            "mask": np.asarray(result.mask, dtype=bool),
            "crs": result.crs,
            "transform": Affine(*tuple(result.transform)),
            "bands": dict(result.bands_used or {}),
        }


def roi_over_the_centre(side_m: float = 1000.0) -> ROISelection:
    native = load_native_ndvi()
    transform, height, width = native["transform"], *native["array"].shape
    minx, miny = transform * (0, height)
    maxx, maxy = transform * (width, 0)
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half = side_m / 2.0
    geom = box(cx - half, cy - half, cx + half, cy + half)
    return ROISelection(
        is_valid=True, intersects_raster=True, area_m2=float(geom.area),
        raster_crs="EPSG:32636", geometry_raster_crs=geom,
        geometry_type="Polygon", num_parts=1,
    )


def confirmed_context(roi):
    native = load_native_ndvi()
    return AnalysisContext(
        roi=roi,
        ndvi=NdviContext(array=native["array"], mask=native["mask"], crs=native["crs"],
                         transform=native["transform"], bands=native["bands"],
                         source_label=SCENE.name),
        ndvi_confirmed=True,
        raster_label=SCENE.name,
    )


# --------------------------------------------------------------------------- #
def test_end_to_end_matches_direct_engine_call():
    native = load_native_ndvi()
    roi = roi_over_the_centre()
    context = confirmed_context(roi)

    # --- through the router ------------------------------------------------ #
    execution = route(QUERY, context)

    # --- the same analysis, called directly -------------------------------- #
    direct = calculate_roi_ndvi_stats(
        native["array"], roi.geometry_raster_crs, native["transform"], native["mask"],
        crs=native["crs"], roi_crs=roi.raster_crs,
    )

    assert execution.intent is Intent.NDVI_ROI_STATS
    assert execution.status is Status.OK
    assert execution.ok

    # identity of every reported number
    assert execution.result.pixels_inside_roi == direct.pixels_inside_roi == 10_000
    assert execution.result.valid_pixels == direct.valid_pixels
    assert execution.result.invalid_pixels == direct.invalid_pixels
    assert execution.result.valid_fraction == pytest.approx(direct.valid_fraction)
    assert execution.result.area_m2 == pytest.approx(direct.area_m2)
    assert execution.result.valid_area_m2 == pytest.approx(direct.valid_area_m2)
    for key, value in direct.stats.items():
        if key == "percentiles":
            for q, pct in value.items():
                assert execution.result.stats["percentiles"][q] == pytest.approx(pct)
        else:
            assert execution.result.stats[key] == pytest.approx(value)

    # the structured payloads are identical too -- routing adds provenance only
    routed = execution.to_dict()["result"]
    expected = direct.to_dict()
    for key, value in expected.items():
        assert routed[key] == value, key
    assert routed["message"] == expected["message"]

    # provenance still says WHERE the numbers came from
    assert execution.provenance["engine"].startswith("core.statistics")
    assert execution.provenance["crs"] == "EPSG:32636"
    assert execution.provenance["native_resolution_m"] == [10.0, 10.0]
    assert execution.provenance["pixels_inside_roi"] == 10_000


def test_end_to_end_answer_wording():
    execution = route(QUERY, confirmed_context(roi_over_the_centre()))
    assert execution.status is Status.OK
    assert "mean NDVI" in execution.message
    assert "valid pixels" in execution.message
    # the ANSWER itself makes no claim (the caveat mentions these words only to
    # deny them, so it is checked separately rather than banned as a substring)
    low = execution.message.lower()
    assert not any(w in low for w in ("healthy", "suitab", "yield", "disease", "flood"))
    assert any("not a crop-health" in w for w in execution.warnings)


def test_end_to_end_without_confirmation():
    native = load_native_ndvi()
    roi = roi_over_the_centre()
    context = AnalysisContext(
        roi=roi,
        ndvi=NdviContext(array=native["array"], mask=native["mask"], crs=native["crs"],
                         transform=native["transform"], bands=native["bands"]),
        ndvi_confirmed=False,
    )
    execution = route(QUERY, context)
    assert execution.status is Status.NEEDS_NDVI_CONFIRMATION
    assert execution.result is None


def test_end_to_end_without_roi():
    execution = route(QUERY, confirmed_context(None))
    assert execution.status is Status.NEEDS_ROI
    assert execution.result is None


@pytest.mark.parametrize("query,intent", [
    # "Can I grow cotton here?" used to live here. Phase 8 gave CROP_SUITABILITY
    # a real engine, so it is no longer an unsupported request; its behaviour is
    # tested in tests/test_phase8_router.py (synthetic ROI, no network).
    # "How has the vegetation changed since 2023?" left this list in Phase 10:
    # it is answered by the temporal NDVI engine, and is now tested in
    # tests/test_phase10_ndvi_change.py.
    ("Show flood areas.", Intent.FLOOD_CHANGE),
    ("Detect flood change in this area.", Intent.FLOOD_CHANGE),
])
def test_end_to_end_unsupported_requests_compute_nothing(query, intent):
    execution = route(query, confirmed_context(roi_over_the_centre()))
    assert execution.intent is intent
    assert execution.status is Status.UNSUPPORTED
    assert execution.result is None
    assert execution.provenance.get("engine") is None
    assert "not available yet" in execution.message
