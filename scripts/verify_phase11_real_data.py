#!/usr/bin/env python
"""Phase 11 -- REAL-DATA evidence for NDWI.

Run from the repository root:

    PYTHONPATH=. python scripts/verify_phase11_real_data.py

It does four things, in order, and prints everything it concludes:

  1. the provenance of the scene it is measuring;
  2. an INDEPENDENT hand calculation of NDWI from raw B03/B08 DN -- no engine
     code is involved in producing the expected value;
  3. one real ROI analysis (5.12 km window) with the full statistics;
  4. the engine's own NDWI against the hand calculation, with the largest
     absolute difference stated explicitly.

Nothing here is a scientific interpretation of the resulting values.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import rasterio
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyses.base import AnalysisContext, IndexContext          # noqa: E402
from analyses.ndwi import run_ndwi_roi_stats                     # noqa: E402
from core.bands import guess_band_roles                          # noqa: E402
from core.raster import describe_path                            # noqa: E402
from core.router import parse_query                              # noqa: E402
from core.roi import ROISelection                                # noqa: E402

SCENE = os.path.join("data", "sample", "s2_s2b-36ruv-20230806-0-l2a_2048px.tif")
BLOCK_ROW, BLOCK_COL, N = 1000, 1000, 8
ROI_OFFSET_M = 2000.0
ROI_SIZE_M = 5120.0

hr = "=" * 74


def section(title: str) -> None:
    print(f"\n{hr}\n{title}\n{hr}")


if not os.path.exists(SCENE):
    sys.exit(f"sample scene not found: {SCENE}")

# --------------------------------------------------------------------------- #
section("1. WHAT IS BEING MEASURED")
# --------------------------------------------------------------------------- #
with rasterio.open(SCENE) as ds:
    minx, miny, maxx, maxy = ds.bounds
    print(f"  file          {SCENE}")
    print(f"  size          {ds.width} x {ds.height} @ {ds.res[0]:g} m")
    print(f"  crs           {ds.crs}")
    print(f"  bands         {list(ds.descriptions)}")
    print(f"  nodata        {ds.nodatavals}")
    print(f"  bounds        {minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}")

guess = guess_band_roles(describe_path(SCENE))
print(f"  band roles    {guess.roles}  (confidence: {guess.confidence})")
print(f"  profile       {guess.profile}")
green_idx, nir_idx = guess.roles["green"], guess.roles["nir"]
from core.index_definitions import get_index                     # noqa: E402
definition = get_index("ndwi")
print(f"  ndwi roles    green = band {green_idx} "
      f"({definition.band_id('green', guess.profile)}), "
      f"nir = band {nir_idx} ({definition.band_id('nir', guess.profile)})")
print(f"  formula       NDWI = {definition.formula}   [{definition.citation}]")
print(f"  reflectance   DN x {guess.reflectance_scale:g} + {guess.reflectance_offset:g}")

# --------------------------------------------------------------------------- #
section(f"2. INDEPENDENT HAND CHECK -- {N}x{N} BLOCK AT ROW {BLOCK_ROW}, COL {BLOCK_COL}")
# --------------------------------------------------------------------------- #
with rasterio.open(SCENE) as ds:
    g_dn = ds.read(green_idx, window=rasterio.windows.Window(
        BLOCK_COL, BLOCK_ROW, N, N)).astype("float64")
    n_dn = ds.read(nir_idx, window=rasterio.windows.Window(
        BLOCK_COL, BLOCK_ROW, N, N)).astype("float64")

g_r, n_r = g_dn / 10000.0, n_dn / 10000.0
den = g_r + n_r
hand = np.where(np.abs(den) > 1e-6, (g_r - n_r) / np.where(den == 0, 1.0, den), np.nan)

print(f"  DN   green(0,0) = {g_dn[0,0]:>6.0f}   nir(0,0) = {n_dn[0,0]:>6.0f}")
print(f"  refl green(0,0) = {g_r[0,0]:>6.4f}   nir(0,0) = {n_r[0,0]:>6.4f}")
print(f"  hand-computed NDWI(0,0) = {hand[0,0]:+.6f}")
print(f"  hand-computed NDWI mean over the block = {np.nanmean(hand):+.6f}"
      f"  (min {np.nanmin(hand):+.6f}, max {np.nanmax(hand):+.6f})")

# --------------------------------------------------------------------------- #
section("3. THE ENGINE OVER A REAL 5.12 km WINDOW")
# --------------------------------------------------------------------------- #
x0 = minx + ROI_OFFSET_M
y0 = maxy - ROI_OFFSET_M - ROI_SIZE_M
roi = ROISelection(is_valid=True, intersects_raster=True,
                   area_m2=ROI_SIZE_M ** 2,
                   geometry_raster_crs=box(x0, y0, x0 + ROI_SIZE_M, y0 + ROI_SIZE_M),
                   raster_crs=str(rasterio.open(SCENE).crs))

ctx = IndexContext(path=SCENE,
                   roles={"green": int(green_idx), "nir": int(nir_idx)},
                   scale=float(guess.reflectance_scale or 0.0001),
                   offset=float(guess.reflectance_offset or 0.0),
                   profile=guess.profile, source_label=SCENE,
                   role_confidence=guess.confidence,
                   role_evidence=tuple(guess.evidence))

execution = run_ndwi_roi_stats(
    AnalysisContext(roi=roi, index_context=ctx),
    parse_query("What is the NDWI of this area?"))

print(f"  status        {execution.status.value}")
if execution.result is None:
    sys.exit(f"  the engine refused: {execution.message}")

r = execution.result
s = r.stats
print(f"  ROI cells     {r.pixels_inside_roi:,}")
print(f"  valid cells   {r.valid_pixels:,}  ({100*r.valid_fraction:.2f}% of the ROI)")
print(f"  NDWI min/max  {s['min']:+.4f} / {s['max']:+.4f}")
print(f"  NDWI mean     {s['mean']:+.4f}")
print(f"  NDWI median   {s['median']:+.4f}")
print(f"  NDWI std dev  {s['std']:+.4f}")
print(f"  window read   {execution.provenance['roi_window']} (col, row, w, h)")
print(f"  runtime       {r.runtime_ms:.0f} ms")

# --------------------------------------------------------------------------- #
section("4. DOES THE ENGINE REPRODUCE THE HAND CHECK?")
# --------------------------------------------------------------------------- #
# Re-run the engine over an ROI that IS the 8x8 block, so the comparison is
# pixel-for-pixel rather than "somewhere inside a bigger window".
bx0 = minx + BLOCK_COL * 10.0
by0 = maxy - (BLOCK_ROW + N) * 10.0
block_roi = ROISelection(
    is_valid=True, intersects_raster=True, area_m2=(N * 10.0) ** 2,
    geometry_raster_crs=box(bx0, by0, bx0 + N * 10.0, by0 + N * 10.0),
    raster_crs=roi.raster_crs)
block_ex = run_ndwi_roi_stats(
    AnalysisContext(roi=block_roi, index_context=ctx),
    parse_query("What is the NDWI of this area?"))
engine_block = np.asarray(block_ex.result.raster)

# the block sits inside the 2-cell read buffer
sub = engine_block[2:2 + N, 2:2 + N]
diff = float(np.nanmax(np.abs(sub - hand)))
print(f"  engine cells: {sub.size} (expected {N*N})")
print(f"  max |engine - hand-computed| = {diff:.3e}")
print(f"  RESULT: {'MATCH' if diff < 1e-6 else 'MISMATCH'}")
assert diff < 1e-6, f"hand check failed: {diff}"

# --------------------------------------------------------------------------- #
section("5. THE WORDING SHOWN TO THE USER")
# --------------------------------------------------------------------------- #
print(f"\n  {execution.message}\n")
for w in execution.warnings:
    print(f"  ! {w}")

banned = ("flood extent", "water body", "water quality", "water availability", "flooded")
low = execution.message.lower()
leaks = [b for b in banned if b in low]
print(f"\n  causal / water claims in the answer: {leaks or 'none'}")

# --------------------------------------------------------------------------- #
section("6. PROVENANCE")
# --------------------------------------------------------------------------- #
prov = dict(execution.provenance)
prov["index"]["limitations"] = f"{len(prov['index']['limitations'])} limitation(s)"
print(json.dumps(prov, indent=2, default=str))

print(f"\n{hr}\nPhase 11 real-data verification: OK\n{hr}")
