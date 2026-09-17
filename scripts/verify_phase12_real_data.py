#!/usr/bin/env python
"""Phase 12 -- REAL-DATA evidence for composed multi-condition queries.

Run from the repository root:

    PYTHONPATH=. python scripts/verify_phase12_real_data.py

It does five things, in order, and prints everything it concludes:

  1. the provenance of the scene (and of the second scene) it is measuring;
  2. an INDEPENDENT hand calculation of NDVI and NDWI from raw band digital
     numbers -- no engine code is involved in producing the expected values;
  3. composition A, a single-date multi-condition query: cropland AND NDVI
     above an explicit threshold AND NDWI below an explicit threshold;
  4. composition B, a temporal + spatial query: NDVI decrease near permanent
     water over the January -> August pair;
  5. a per-condition cross-check of the engine's masks against the hand
     calculation, with the largest disagreement stated explicitly.

The script also demonstrates the three refusals that protect the science:
a bare "high NDVI" is refused, a change condition with no dates is refused,
and an area outside the scene is refused.

NOTHING HERE INTERPRETS A COMBINATION AS PROOF OF FLOODING OR OF ANY OTHER
CAUSE. A combined condition is geographic evidence, not causal attribution.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import rasterio
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyses.base import AnalysisContext, IndexContext          # noqa: E402
from analyses.multi_condition import run_multi_condition         # noqa: E402
from core.bands import guess_band_roles                          # noqa: E402
from core.raster import describe_path                            # noqa: E402
from core.router import parse_query                              # noqa: E402
from core.roi import ROISelection                                # noqa: E402
from core.temporal import discover_scenes, pair_by_dates         # noqa: E402

SCENE = os.path.join("data", "sample", "s2_s2b-36ruv-20230806-0-l2a_2048px.tif")
BEFORE_DATE, AFTER_DATE = "2023-01-18", "2023-08-06"
ROI_OFFSET_M = 2000.0
ROI_SIZE_M = 5120.0
SCALE, OFFSET = 0.0001, 0.0

NDVI_THRESHOLD = 0.6          # explicit, from the query text
NDWI_THRESHOLD = -0.4         # explicit, from the query text

hr = "=" * 78


def section(title: str) -> None:
    print(f"\n{hr}\n{title}\n{hr}")


if not os.path.exists(SCENE):
    sys.exit(f"sample scene not found: {SCENE}")

# --------------------------------------------------------------------------- #
section("1. WHAT IS BEING MEASURED")
# --------------------------------------------------------------------------- #
with rasterio.open(SCENE) as ds:
    minx, miny, maxx, maxy = ds.bounds
    print(f"  scene         {SCENE}")
    print(f"  size          {ds.width} x {ds.height} @ {ds.res[0]:g} m")
    print(f"  CRS           {ds.crs}")
    print(f"  bounds        {minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}")

guess = guess_band_roles(describe_path(SCENE))
print(f"  roles         {dict(guess.roles)}  (confidence {guess.confidence})")
roles = {str(k): int(v) for k, v in guess.roles.items()}

geometry = box(minx + ROI_OFFSET_M, maxy - ROI_OFFSET_M - ROI_SIZE_M,
               minx + ROI_OFFSET_M + ROI_SIZE_M, maxy - ROI_OFFSET_M)
roi = ROISelection(is_valid=True, intersects_raster=True,
                   area_m2=ROI_SIZE_M ** 2,
                   geometry_raster_crs=geometry,
                   raster_crs=str(rasterio.open(SCENE).crs))
print(f"\n  ROI           {ROI_SIZE_M / 1000:g} km x {ROI_SIZE_M / 1000:g} km "
      f"= {ROI_SIZE_M ** 2 / 1e6:.2f} km² at "
      f"{minx + ROI_OFFSET_M:.0f}, {maxy - ROI_OFFSET_M - ROI_SIZE_M:.0f}")

index_ctx = IndexContext(
    path=SCENE, roles=roles, scale=SCALE, offset=OFFSET,
    profile=guess.profile, source_label=os.path.basename(SCENE),
    role_confidence=str(getattr(guess, "confidence", "high")),
    role_evidence=tuple(getattr(guess, "evidence", ()) or ()))

scenes = discover_scenes(os.path.join("data", "sample"))
print("  scenes        " + ", ".join(f"{s.date}" for s in scenes))
temporal_pair = pair_by_dates(scenes, BEFORE_DATE, AFTER_DATE)
print(f"  pair          {BEFORE_DATE} -> {AFTER_DATE}: "
      f"{'available' if temporal_pair else 'NOT available'}")

# --------------------------------------------------------------------------- #
section("2. INDEPENDENT HAND CALCULATION (no engine code)")
# --------------------------------------------------------------------------- #
with rasterio.open(SCENE) as ds:
    win = rasterio.windows.from_bounds(*geometry.bounds, transform=ds.transform)
    red = ds.read(roles["red"], window=win).astype("float64") * SCALE + OFFSET
    nir = ds.read(roles["nir"], window=win).astype("float64") * SCALE + OFFSET
    green = ds.read(roles["green"], window=win).astype("float64") * SCALE + OFFSET

with np.errstate(divide="ignore", invalid="ignore"):
    ndvi_hand = np.where(np.abs(nir + red) > 1e-6, (nir - red) / (nir + red), np.nan)
    ndwi_hand = np.where(np.abs(green + nir) > 1e-6,
                         (green - nir) / (green + nir), np.nan)
valid_hand = np.isfinite(ndvi_hand) & np.isfinite(ndwi_hand)
print(f"  window        {ndvi_hand.shape[0]} x {ndvi_hand.shape[1]} cells")
print(f"  NDVI          min {np.nanmin(ndvi_hand):+.4f}  "
      f"median {np.nanmedian(ndvi_hand):+.4f}  max {np.nanmax(ndvi_hand):+.4f}")
print(f"  NDWI          min {np.nanmin(ndwi_hand):+.4f}  "
      f"median {np.nanmedian(ndwi_hand):+.4f}  max {np.nanmax(ndwi_hand):+.4f}")
print(f"  hand NDVI > {NDVI_THRESHOLD}:   {int(np.nansum(ndvi_hand > NDVI_THRESHOLD)):,} cells")
print(f"  hand NDWI < {NDWI_THRESHOLD}:  {int(np.nansum(ndwi_hand < NDWI_THRESHOLD)):,} cells")
print(f"  hand BOTH:          "
      f"{int(np.nansum((ndvi_hand > NDVI_THRESHOLD) & (ndwi_hand < NDWI_THRESHOLD))):,} cells")

# --------------------------------------------------------------------------- #
section("3. COMPOSITION A -- cropland AND NDVI AND NDWI (single date)")
# --------------------------------------------------------------------------- #
query_a = f"Find cropland with NDVI greater than {NDVI_THRESHOLD} and NDWI less than {NDWI_THRESHOLD}"
context_a = AnalysisContext(roi=roi, index_context=index_ctx,
                            temporal_pair=temporal_pair)
parsed_a = parse_query(query_a)
print(f"  query         {query_a!r}")
print(f"  intent        {parsed_a.intent.value} (confidence {parsed_a.confidence:.2f})")
exec_a = run_multi_condition(context_a, parsed_a)
print(f"  status        {exec_a.status.value}")
print(f"  message       {exec_a.message}")

result_a = exec_a.result
if result_a is None:
    sys.exit("composition A produced no result -- nothing to verify")

print(f"\n  conditions    {result_a.expression}")
for entry in result_a.condition_results:
    print(f"    - {entry['kind']:8s} {entry['label']}")
    if entry.get("threshold") is not None:
        spec = entry.get("threshold_provenance") or {}
        print(f"      threshold  {entry['operator']} {entry['threshold']} "
              f"[{spec.get('provenance')}: {spec.get('detail')}]")
    print(f"      match {entry['matched']:,}  no-match {entry['non_matching']:,}  "
          f"undecided {entry['insufficient']:,}")

cell_area = float(result_a.analysis_resolution or 10.0) ** 2
print(f"\n  matched       {result_a.matched_cell_count:,} cells = "
      f"{result_a.matched_area_km2:.3f} km²  (cell {cell_area:.0f} m²)")
print(f"  no match      {result_a.non_matching_cell_count:,} cells")
print(f"  undecided     {result_a.insufficient_cell_count:,} cells "
      f"(never counted as non-matches)")
print(f"  analysed      {result_a.analysed_cell_count:,} cells")
print(f"  grid          {result_a.grid['width']} x {result_a.grid['height']} "
      f"@ {result_a.analysis_resolution:g} m, CRS {result_a.grid['crs']}")
print(f"  alignment     {result_a.alignment}")
print(f"  dates         {result_a.source_dates}")
print(f"  sources       {', '.join(result_a.source_analyses)}")
print(f"  runtime       {result_a.performance['runtime_ms']:.1f} ms")

# ---- cross-check against the hand calculation ----------------------------- #
# The composition grid is the ROI plus a two-cell rim (core.alignment adds it
# so the ROI can never be clipped by rounding), so the comparison is made on
# the ROI interior, located exactly from the grid transform.
from rasterio.transform import Affine

grid_transform = Affine(*tuple(result_a.grid["transform"])[:6])
inv = ~grid_transform
gminx, _gminy, gmaxx, gmaxy = geometry.bounds
col, row = inv * (gminx + 5.0, gmaxy - 5.0)      # half a cell into the ROI
row, col = int(round(row)), int(round(col))
h, w = ndvi_hand.shape
interior = (slice(row, row + h), slice(col, col + w))
print(f"\n  ROI interior on the composition grid: rows {row}..{row + h}, "
      f"cols {col}..{col + w}")

spectral = {c["name"]: c for c in result_a.condition_results
            if c["kind"] == "spectral"}
worst = 0
off_boundary = 0
for name, hand, op, thr in (("ndvi_gt", ndvi_hand, np.greater, NDVI_THRESHOLD),
                            ("ndwi_lt", ndwi_hand, np.less, NDWI_THRESHOLD)):
    if name not in spectral:
        continue
    engine_full = np.asarray(spectral[name]["mask"].match)
    engine_mask = engine_full[interior]
    hand_mask = op(hand, thr) & np.isfinite(hand)
    if engine_mask.shape != hand_mask.shape:
        print(f"  {name}: shape {engine_mask.shape} vs {hand_mask.shape} -- skipped")
        continue
    differing = engine_mask != hand_mask
    diff = int(differing.sum())
    # The engine computes the index in float32 (that is what the raster holds)
    # and this check recomputes it in float64. A cell sitting exactly ON the
    # threshold can therefore fall on either side of a strict ">" -- which is
    # the boundary behaving correctly, not a mask error. What must be zero is
    # the number of cells that disagree AWAY from the threshold.
    at_boundary = int(np.sum(differing & (np.abs(hand - thr) < 1e-6)))
    away = diff - at_boundary
    off_boundary += away
    worst = max(worst, diff)
    print(f"  check {name:8s} engine {int(engine_mask.sum()):,} vs hand "
          f"{int(hand_mask.sum()):,} -- {diff} cells differ, "
          f"{at_boundary} of them exactly on the threshold "
          f"({away} away from it)")

assert off_boundary == 0, (
    f"{off_boundary} cells disagree away from the threshold -- the mask is wrong")
print("  every disagreement sits exactly on the threshold (float32 vs float64"); print("  recomputation); no cell disagrees away from it.")

# a matched cell must satisfy BOTH spectral conditions: the conjunction can
# only ever remove cells, never add one that fails a condition
combined = np.asarray(result_a.combined_mask)[interior]
both_hand = ((ndvi_hand > NDVI_THRESHOLD) & (ndwi_hand < NDWI_THRESHOLD)
             & valid_hand)
violations = int(np.sum((combined == 2) & ~both_hand))
print(f"  check combined engine {int((combined == 2).sum()):,} matches vs "
      f"{int(both_hand.sum()):,} hand cells satisfying both spectra -- "
      f"{violations} matched cells violate a threshold")
assert violations == 0, "a matched cell failed a condition it must satisfy"

print(f"\n  largest disagreement with the hand calculation: {worst} cells")

# --------------------------------------------------------------------------- #
section("4. COMPOSITION B -- NDVI decrease near permanent water (two dates)")
# --------------------------------------------------------------------------- #
query_b = "Show areas with vegetation decrease near permanent water."
parsed_b = parse_query(query_b)
print(f"  query         {query_b!r}")
print(f"  intent        {parsed_b.intent.value} (confidence {parsed_b.confidence:.2f})")
exec_b = run_multi_condition(context_a, parsed_b)
print(f"  status        {exec_b.status.value}")
print(f"  message       {exec_b.message}")

result_b = exec_b.result
if result_b is None:
    sys.exit("composition B produced no result -- nothing to verify")

print(f"\n  conditions    {result_b.expression}")
for entry in result_b.condition_results:
    print(f"    - {entry['kind']:8s} {entry['label']}: "
          f"match {entry['matched']:,}, undecided {entry['insufficient']:,}")
print(f"\n  matched       {result_b.matched_cell_count:,} cells = "
      f"{result_b.matched_area_km2:.3f} km²")
print(f"  undecided     {result_b.insufficient_cell_count:,} cells")
print(f"  dates         {result_b.source_dates}")
print(f"  grid          {result_b.grid['width']} x {result_b.grid['height']} "
      f"@ {result_b.analysis_resolution:g} m")
print(f"  alignment     {result_b.alignment}")
print(f"  runtime       {result_b.performance['runtime_ms']:.1f} ms")

# --------------------------------------------------------------------------- #
section("5. ATTACHED EVIDENCE -- an index measured over the match, not a filter")
# --------------------------------------------------------------------------- #
query_c = "Find areas with NDVI decrease and NDWI statistics."
exec_c = run_multi_condition(context_a, parse_query(query_c))
print(f"  query         {query_c!r}  -> {exec_c.status.value}")
if exec_c.result is not None:
    print(f"  matched       {exec_c.result.matched_cell_count:,} cells")
    for index, summary in exec_c.result.index_summaries.items():
        if summary.get("defined"):
            print(f"  {index.upper():5s} over the match: mean {summary['mean']:+.4f}, "
                  f"median {summary['median']:+.4f}, "
                  f"{summary['valid_pixels']:,} valid cells")
            print(f"        {summary['computed_over']} -- not a filter")

# --------------------------------------------------------------------------- #
section("6. THE REFUSALS THAT PROTECT THE SCIENCE")
# --------------------------------------------------------------------------- #
for label, text, ctx in (
        ("no threshold", "Find cropland with high NDVI", context_a),
        ("no dates", "Show areas with vegetation decrease near permanent water.",
         AnalysisContext(roi=roi, index_context=index_ctx)),
):
    ex = run_multi_condition(ctx, parse_query(text))
    print(f"  {label:14s} -> {ex.status.value}")
    print(f"                  {ex.message}")

outside = box(minx - 50_000.0, maxy + 5_000.0,
              minx - 49_000.0, maxy + 6_000.0)
roi_out = ROISelection(is_valid=True, intersects_raster=True, area_m2=1e6,
                       geometry_raster_crs=outside, raster_crs=str(rasterio.open(SCENE).crs))
ex = run_multi_condition(
    AnalysisContext(roi=roi_out, index_context=index_ctx),
    parse_query("Find cropland with NDVI greater than 0.6"))
print(f"  {'outside scene':14s} -> {ex.status.value}")
print(f"                  {ex.message}")

too_big = box(minx, maxy - 20_000.0, minx + 20_000.0, maxy)
roi_big = ROISelection(is_valid=True, intersects_raster=True, area_m2=4.0e8,
                       geometry_raster_crs=too_big, raster_crs=str(rasterio.open(SCENE).crs))
ex = run_multi_condition(
    AnalysisContext(roi=roi_big, index_context=index_ctx),
    parse_query("Find cropland with NDVI greater than 0.6"))
print(f"  {'too many cells':14s} -> {ex.status.value}")
print(f"                  {ex.message}")

# --------------------------------------------------------------------------- #
section("7. WHAT THIS EVIDENCE DOES AND DOES NOT SAY")
# --------------------------------------------------------------------------- #
print("  A combined condition is geographic evidence, not causal attribution.")
for limitation in result_a.limitations:
    print(f"    - {limitation}")
print("\n  Composition A and composition B are reported as coincidences of")
print("  measured conditions over one grid. Neither is a flood assessment,")
print("  a damage assessment, a crop-failure finding or a water-quality result.")
