"""Phase 8 -- shared plumbing for external data sources.

Responsibilities
    * read ONLY the window that covers the analysis grid (never a global raster)
    * reproject + resample with the correct method for the variable type
    * cache the subset under data/external/<source>/ with a JSON provenance
      sidecar whose identity includes dataset, version, variable, source URL,
      AOI grid, resolution and processing -- so stale data can never be reused
    * measure how long each stage took (brief item #27)

This module never decides what a value MEANS. Units and scaling belong to the
per-source modules and to config/crops/<crop>.yml.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds

from ..alignment import AnalysisGrid, grid_signature

__all__ = [
    "CACHE_ROOT",
    "stage_totals",
    "SourceRecord",
    "LayerResult",
    "load_layer",
    "read_sources_into_grid",
    "CACHE_KEY_FIELDS",
]

CACHE_ROOT = os.environ.get("SATQUERY_EXTERNAL_DIR", "data/external")

#: Run totals for brief item #27: "source access", "windowed read +
#: reprojection" and "read back from the local cache". `load_layer` adds to
#: these; a run resets them at the start. Reporting only, never logic.
_STAGE_TOTALS: Dict[str, float] = {"open": 0.0, "read": 0.0, "cache": 0.0}


def stage_totals(reset: bool = False) -> Dict[str, float]:
    """Cumulative per-stage seconds since the last reset."""
    global _STAGE_TOTALS
    if reset:
        _STAGE_TOTALS = {"open": 0.0, "read": 0.0, "cache": 0.0}
    return _STAGE_TOTALS

#: Everything that changes the bytes on disk must be part of the cache key.
CACHE_KEY_FIELDS = ("dataset", "version", "variable", "source_url",
                    "aoi_signature", "resolution", "resampling", "band")


@dataclass
class SourceRecord:
    """Machine-readable provenance for one factor layer (brief item #17)."""

    dataset: str
    version: str
    variable: str
    native_resolution: str
    native_crs: str
    units: str
    temporal_period: str
    source_url: str
    license: str
    access_date: str = ""
    source_nodata: Optional[float] = None
    resampling: str = ""
    processing: str = ""
    limitations: str = ""
    dataset_status: str = "literature-backed"     # frozen for datasets, flipped for thresholds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerResult:
    array: Any
    record: SourceRecord
    from_cache: bool = False
    seconds: float = 0.0
    #: how much of `seconds` was spent opening the source (access), and how much
    #: on the windowed read + reprojection. Reported separately -- they are
    #: different costs and only one of them is network-bound.
    open_seconds: float = 0.0
    read_seconds: float = 0.0
    cache_path: str = ""


def _ensure(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def read_sources_into_grid(urls: Sequence[str],
                           grid: AnalysisGrid,
                           resampling: Resampling,
                           band: int = 1,
                           src_nodata: Optional[float] = None,
                           stage: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Windowed read of one or more remote tiles -> the analysis grid.

    Single source (the normal case) goes through WarpedVRT: GDAL pulls exactly
    the source window that projects onto the grid and nothing else.

    Multiple tiles are mosaicked first. `merge()` snaps to whole source pixels,
    which DROPS the partial pixel at the edge of the requested bounds and would
    leave a NaN band along the border of the output -- so the merge bounds are
    inflated by one source pixel on every side before reprojecting.

    Steps (brief item #15):
        1. open the remote source(s)      (header only, no bulk read)
        2. read only the AOI window
        3. reproject that small array onto the grid
        4. close the sources
    """
    t_open = time.perf_counter()
    srcs = [rasterio.open(u) for u in urls]
    if stage is not None:
        stage["open_seconds"] = stage.get("open_seconds", 0.0) + (
            time.perf_counter() - t_open)
    try:
        nodata = src_nodata if src_nodata is not None else srcs[0].nodatavals[band - 1]
        dst = np.full((grid.height, grid.width), np.nan, dtype="float32")

        t_read = time.perf_counter()
        if len(srcs) == 1:
            with WarpedVRT(srcs[0],
                           crs=grid.crs, transform=grid.transform,
                           width=grid.width, height=grid.height,
                           resampling=resampling, src_nodata=nodata,
                           nodata=np.nan, dtype="float32") as vrt:
                dst = vrt.read(band)
        else:  # pragma: no cover -- only ROIs straddling a tile boundary
            left, bottom, right, top = transform_bounds(
                grid.crs, srcs[0].crs, *grid.bounds, densify_pts=21)
            sx = abs(float(srcs[0].transform.a))
            sy = abs(float(srcs[0].transform.e))
            mosaic, mosaic_transform = merge(
                srcs, bounds=(left - sx, bottom - sy, right + sx, top + sy),
                nodata=nodata, resampling=resampling, indexes=[band])
            reproject(
                source=mosaic, destination=dst,
                src_transform=mosaic_transform, src_crs=srcs[0].crs,
                dst_transform=grid.transform, dst_crs=grid.crs,
                resampling=resampling, src_nodata=nodata, dst_nodata=np.nan)

        if stage is not None:
            stage["read_seconds"] = stage.get("read_seconds", 0.0) + (
                time.perf_counter() - t_read)

        if nodata is not None and np.isfinite(nodata):
            dst[dst == nodata] = np.nan
        return dst.astype("float32", copy=False)
    finally:
        for s in srcs:
            try:
                s.close()
            except Exception:  # pragma: no cover
                pass


def load_layer(*,
               dataset: str,
               version: str,
               variable: str,
               urls: Sequence[str],
               grid: AnalysisGrid,
               resampling: Resampling,
               native_resolution: str,
               native_crs: str,
               units: str,
               temporal_period: str,
               license: str,
               limitations: str,
               processing: str = "",
               band: int = 1,
               src_nodata: Optional[float] = None,
               source_url: str = "",
               use_cache: bool = True,
               extra: str = "") -> LayerResult:
    """Fetch (or reuse) one factor layer and return it with its provenance."""
    import datetime

    aoi_sig = grid_signature(grid, extra=extra or variable)
    url_key = source_url or (urls[0] if urls else "")
    identity = {
        "dataset": dataset, "version": version, "variable": variable,
        "source_url": url_key, "aoi_signature": aoi_sig,
        "resolution": grid.resolution, "resampling": resampling.name, "band": band,
    }
    safe_var = variable.replace("/", "-").replace(" ", "_")
    stem = os.path.join(CACHE_ROOT, dataset, f"{dataset}_{version}_{safe_var}_{aoi_sig}")
    tif_path, json_path = f"{stem}.tif", f"{stem}.json"

    if use_cache and os.path.exists(tif_path) and os.path.exists(json_path):
        t_cache = time.perf_counter()
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            if all(saved.get("identity", {}).get(k) == v for k, v in identity.items()):
                with rasterio.open(tif_path) as ds:
                    arr = ds.read(1)
                rec = SourceRecord(**{k: v for k, v in saved["record"].items()
                                      if k in SourceRecord.__dataclass_fields__})
                _cache_s = time.perf_counter() - t_cache
                _STAGE_TOTALS["cache"] += _cache_s
                return LayerResult(arr, rec, from_cache=True, seconds=_cache_s,
                                   cache_path=tif_path)
        except Exception:  # pragma: no cover -- a corrupt cache is simply refetched
            pass

    t0 = time.perf_counter()
    stage: Dict[str, float] = {}
    arr = read_sources_into_grid(list(urls), grid, resampling, band=band,
                                 src_nodata=src_nodata, stage=stage)
    seconds = time.perf_counter() - t0
    _open_s = float(stage.get("open_seconds", 0.0))
    _read_s = float(stage.get("read_seconds", 0.0))
    _STAGE_TOTALS["open"] += _open_s
    _STAGE_TOTALS["read"] += _read_s

    record = SourceRecord(
        dataset=dataset, version=version, variable=variable,
        native_resolution=native_resolution, native_crs=native_crs, units=units,
        temporal_period=temporal_period, source_url=url_key, license=license,
        access_date=datetime.date.today().isoformat(),
        source_nodata=src_nodata,
        resampling=resampling.name,
        processing=processing or (f"windowed read; reprojected to {grid.crs} "
                                  f"at {grid.resolution:g} m; {resampling.name}"),
        limitations=limitations,
    )

    if use_cache:
        try:
            _ensure(tif_path)
            profile = {
                "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
                "count": 1, "dtype": "float32", "crs": grid.crs,
                "transform": grid.transform, "nodata": float("nan"),
                "compress": "deflate",
            }
            with rasterio.open(tif_path, "w", **profile) as ds:
                ds.write(arr.astype("float32"), 1)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump({"identity": identity, "record": record.to_dict()},
                          fh, indent=2, default=str)
        except Exception:  # pragma: no cover -- caching is best effort
            tif_path = ""

    return LayerResult(arr, record, from_cache=False, seconds=seconds,
                       cache_path=tif_path, open_seconds=_open_s,
                       read_seconds=_read_s)


def value_at_centre(array: Any) -> Optional[float]:
    """Central pixel value -- used only for probes and logging, never for scoring."""
    a = np.asarray(array, dtype="float64")
    if a.size == 0 or not np.isfinite(a).any():
        return None
    h, w = a.shape
    return float(a[h // 2, w // 2])
