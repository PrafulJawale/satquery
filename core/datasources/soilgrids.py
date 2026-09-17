"""Phase 8 -- ISRIC SoilGrids 2.0 (250 m soil properties).

SoilGrids is a MACHINE-LEARNING PREDICTION, not a laboratory measurement, and
its VRTs carry no GDAL scale metadata: the documented conversion factors are
applied here, from a table, per property.

Depths: 0-5 cm and 5-15 cm are combined as a thickness-weighted mean
(5/15 and 10/15) and reported as "0-15 cm".

Deliberately NOT used for scoring:
    soc      -- implausible values at the Nile Delta test site and a 750 m grid
                offset relative to the other property VRTs; context only.
No salinity layer exists in SoilGrids -- see the unassessed constraints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rasterio.enums import Resampling

from ..alignment import AnalysisGrid
from .base import LayerResult, SourceRecord, load_layer

__all__ = ["DATASET", "VERSION", "PROPERTIES", "DEPTHS", "WEIGHTS",
           "fetch_property", "fetch_texture_and_ph", "LIMITATIONS"]

DATASET = "soilgrids"
VERSION = "2.0"

#: ISRIC documented mapped units -> conversion to the unit we score.
PROPERTIES: Dict[str, Dict[str, Any]] = {
    "phh2o": {"scale": 0.1, "unit": "pH", "mapped": "pH x 10"},
    "clay": {"scale": 0.1, "unit": "%", "mapped": "g/kg"},
    "sand": {"scale": 0.1, "unit": "%", "mapped": "g/kg"},
    "silt": {"scale": 0.1, "unit": "%", "mapped": "g/kg"},
    "cec": {"scale": 0.1, "unit": "cmol(c)/kg", "mapped": "mmol(c)/kg"},
    "bdod": {"scale": 0.01, "unit": "kg/dm3", "mapped": "cg/cm3"},
    "nitrogen": {"scale": 0.01, "unit": "g/kg", "mapped": "cg/kg"},
    "soc": {"scale": 0.1, "unit": "g/kg", "mapped": "dg/kg", "context_only": True},
}

DEPTHS: Tuple[str, ...] = ("0-5cm", "5-15cm")
WEIGHTS: Tuple[float, ...] = (5.0, 10.0)          # layer thicknesses, mm of a 15 cm slab

LIMITATIONS = (
    "Machine-learning prediction at 250 m, not a measurement; uncertainty is "
    "large at field scale; property VRTs are not on an identical grid (the soc "
    "grid is offset ~750 m); no salinity, depth or drainage layer.")


def _url(prop: str, depth: str, stat: str = "mean") -> str:
    return f"https://files.isric.org/soilgrids/latest/data/{prop}/{prop}_{depth}_{stat}.vrt"


def fetch_property(prop: str,
                   grid: AnalysisGrid,
                   depths: Tuple[str, ...] = DEPTHS,
                   weights: Tuple[float, ...] = WEIGHTS,
                   use_cache: bool = True) -> Tuple[np.ndarray, List[SourceRecord]]:
    """Thickness-weighted mean of one property over the requested depths."""
    spec = PROPERTIES[prop]
    total = float(sum(weights[: len(depths)]))
    acc: Optional[np.ndarray] = None
    records: List[SourceRecord] = []
    for depth, w in zip(depths, weights):
        res = load_layer(
            dataset=DATASET, version=VERSION, variable=f"{prop}_{depth}",
            urls=[_url(prop, depth)], grid=grid, resampling=Resampling.bilinear,
            native_resolution="250 m", native_crs="ESRI:54052 (Interrupted Goode Homolosine)",
            units=f"{spec['mapped']} (stored) -> {spec['unit']} (x{spec['scale']})",
            temporal_period="static model output (2019 release)",
            license="CC BY 4.0 (Poggio et al. 2021, SOIL 7:217-240)",
            limitations=LIMITATIONS,
            processing=(f"windowed read; bilinear reprojection to {grid.crs} at "
                        f"{grid.resolution:g} m; stored value x {spec['scale']} -> {spec['unit']}"),
            source_url=_url(prop, depth), use_cache=use_cache)
        arr = res.array * float(spec["scale"])
        arr = np.where(np.isfinite(arr), arr, np.nan)
        acc = arr * (w / total) if acc is None else acc + arr * (w / total)
        records.append(res.record)
    return acc, records


def fetch_texture_and_ph(grid: AnalysisGrid,
                         use_cache: bool = True) -> Dict[str, Any]:
    """Everything the cotton model needs from soil: pH, sand, silt, clay (+ context)."""
    out: Dict[str, Any] = {"records": [], "values": {}}
    for prop in ("phh2o", "clay", "sand", "silt"):
        arr, recs = fetch_property(prop, grid, use_cache=use_cache)
        key = {"phh2o": "phh2o", "clay": "clay_pct", "sand": "sand_pct",
               "silt": "silt_pct"}[prop]
        out[key] = arr
        out["records"].extend(recs)

    # context only -- never scored (see module docstring)
    try:
        soc_arr, soc_recs = fetch_property("soc", grid, use_cache=use_cache)
        out["soc_g_per_kg"] = soc_arr
        out["soc_context_only"] = True
    except Exception:  # pragma: no cover -- context must never break the analysis
        out["soc_g_per_kg"] = None
    return out
