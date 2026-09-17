"""Phase 8 -- WorldClim 2.1 monthly climatologies (1970-2000 normals, 30 arc-sec).

WHAT THIS PROVIDES
    * growing-season precipitation   (sum of monthly means over the season)
    * annual precipitation           (sum of all 12 monthly means) -- reported,
                                     never silently substituted for the seasonal value
    * growing-season temperature     (mean of PER-MONTH trapezoid memberships)
    * an explicitly approximate seasonal temperature sum (GDD context only)

HOW THE FILES ARE READ
    The monthly GeoTIFFs live inside 4-5 GB zip archives on a remote server.
    They are opened through /vsizip//vsicurl and read WINDOW-WISE, so only a few
    MB cross the network per month. Nothing is ever downloaded whole.

    Values come out already scaled (rasterio applies the GeoTIFF scale/offset),
    i.e. degrees Celsius and millimetres -- verified against known sites.

NOT PROVIDED: daily temperatures. Everything derived here is monthly.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from rasterio.enums import Resampling

from ..alignment import AnalysisGrid
from ..suitability import approximate_gdd, trapezoid_membership
from .base import LayerResult, SourceRecord, load_layer

__all__ = ["DATASET", "VERSION", "monthly_url", "fetch_climate", "LIMITATIONS"]

DATASET = "worldclim"
VERSION = "2.1_30s"
BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"

LIMITATIONS = ("1970-2000 climatological normals: they describe the long-term "
               "average, not any specific season or year. Station interpolation "
               "is weak where station density is low. ~1 km pixels are effectively "
               "constant across a 20 km AOI. Monthly means contain no daily "
               "extremes.")


def monthly_url(variable: str, month: int) -> str:
    """e.g. /vsizip//vsicurl/.../wc2.1_30s_tmin.zip/wc2.1_30s_tmin_04.tif"""
    return (f"/vsizip//vsicurl/{BASE}/wc2.1_30s_{variable}.zip/"
            f"wc2.1_30s_{variable}_{month:02d}.tif")


def _fetch(variable: str, month: int, grid: AnalysisGrid,
           use_cache: bool) -> LayerResult:
    return load_layer(
        dataset=DATASET, version=VERSION, variable=f"{variable}_{month:02d}",
        urls=[monthly_url(variable, month)], grid=grid,
        resampling=Resampling.bilinear,
        native_resolution="30 arc-sec (~1 km)", native_crs="EPSG:4326",
        units="degC (tmin/tmax), mm (prec)",
        temporal_period="1970-2000 monthly normals",
        license="CC BY 4.0 (Fick & Hijmans 2017)",
        limitations=LIMITATIONS,
        processing=(f"windowed read from the remote zip; bilinear reprojection to "
                    f"{grid.crs} at {grid.resolution:g} m"),
        source_url=monthly_url(variable, month), use_cache=use_cache)


def fetch_climate(grid: AnalysisGrid,
                  growing_season_months: Sequence[int],
                  temperature_spec: Dict[str, Any],
                  inside: Optional[np.ndarray] = None,
                  use_cache: bool = True,
                  progress: Optional[Callable[[str], None]] = None,
                  ) -> Dict[str, Any]:
    """Accumulate the climate factors month by month (memory-safe).

    Only a few arrays are alive at any moment: the running sums plus the current
    month. The 12 monthly layers are never all held in memory.
    """
    season = [int(m) for m in growing_season_months]
    opt = [float(v) for v in temperature_spec.get("optimum", [0, 1])]
    abso = [float(v) for v in temperature_spec.get("absolute", [0, 1])]

    annual_prec = np.zeros((grid.height, grid.width), dtype="float64")
    season_prec = np.zeros_like(annual_prec)
    temp_sum = np.zeros_like(annual_prec)
    temp_n = np.zeros_like(annual_prec)
    monthly_stats: Dict[int, Dict[str, Optional[float]]] = {}
    records: List[SourceRecord] = []
    warnings: List[str] = []
    timings: Dict[str, float] = {"precipitation": 0.0, "temperature": 0.0}

    def _mean(arr: np.ndarray) -> Optional[float]:
        if inside is not None and inside.any():
            vals = arr[inside & np.isfinite(arr)]
        else:
            vals = arr[np.isfinite(arr)]
        return float(vals.mean()) if vals.size else None

    for month in range(1, 13):
        if progress:
            progress(f"WorldClim precipitation, month {month:02d}/12")
        t0 = time.perf_counter()
        prec = _fetch("prec", month, grid, use_cache)
        timings["precipitation"] += time.perf_counter() - t0
        records.append(prec.record)
        if not np.isfinite(prec.array).any():
            warnings.append(f"WorldClim precipitation for month {month:02d} had no "
                            "data over the area; it was excluded, not zero-filled.")
            continue
        annual_prec += np.nan_to_num(prec.array, nan=0.0)
        if month in season:
            season_prec += np.nan_to_num(prec.array, nan=0.0)
        monthly_stats.setdefault(month, {})["prec_mm"] = _mean(prec.array)
        del prec

    for month in season:
        if progress:
            progress(f"WorldClim temperature, month {month:02d}")
        t0 = time.perf_counter()
        tmin = _fetch("tmin", month, grid, use_cache)
        tmax = _fetch("tmax", month, grid, use_cache)
        timings["temperature"] += time.perf_counter() - t0
        records.extend([tmin.record, tmax.record])
        if not np.isfinite(tmin.array).any() or not np.isfinite(tmax.array).any():
            warnings.append(f"WorldClim temperature for month {month:02d} had no "
                            "data over the area; that month was not scored.")
            continue
        tmean = (tmin.array + tmax.array) / 2.0
        mem = trapezoid_membership(tmean, abso[0], opt[0], opt[1], abso[1])
        temp_sum += np.nan_to_num(mem, nan=0.0)
        temp_n += np.isfinite(mem).astype("float64")
        stats = monthly_stats.setdefault(month, {})
        stats["tmin_c"] = _mean(tmin.array)
        stats["tmax_c"] = _mean(tmax.array)
        stats["tmean_c"] = _mean(tmean)
        del tmin, tmax, tmean, mem

    with np.errstate(invalid="ignore", divide="ignore"):
        temperature_membership = np.where(temp_n > 0, temp_sum / np.maximum(temp_n, 1), np.nan)

    # context only: an explicitly approximate seasonal temperature sum
    months_with_stats = [m for m in season if monthly_stats.get(m, {}).get("tmin_c") is not None]
    gdd = None
    if months_with_stats:
        gdd = approximate_gdd(
            [monthly_stats[m]["tmin_c"] for m in months_with_stats],
            [monthly_stats[m]["tmax_c"] for m in months_with_stats],
            months_with_stats,
            base_temp=float(temperature_spec.get("gdd_base_temp_c", 15.6)))

    return {
        "growing_season_precipitation": season_prec.astype("float32"),
        "annual_precipitation": annual_prec.astype("float32"),
        "growing_season_temperature": temperature_membership.astype("float32"),
        "monthly_stats": monthly_stats,
        "approx_gdd": gdd,
        "records": records,
        "warnings": warnings,
        "timings": timings,
        "months_used": season,
    }
