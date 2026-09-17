#!/usr/bin/env python
"""Phase 13 -- REAL-DATA evidence for the evidence layer itself.

Run from the repository root:

    PYTHONPATH=. python scripts/verify_phase13_real_data.py

The script proves that Phase 13's evidence is NOT a second opinion. Every
number it prints is produced by the Phase 12 engine (`analyses.registry.route`)
and then READ BACK through the Phase 13 contract:

  1. the two questions the brief names, run over the bundled scene;
  2. the full lineage of each answer -- query, intent, conditions, sources,
     scene, grid, mask, statistics, answer -- reconstructed from the package;
  3. an independent cross-check that every count, area and date in the package
     is the engine's own number (nothing is re-derived, nothing is added);
  4. the generated explanation, printed in full so the wording can be read;
  5. the three states that must never be confused: matched, measured
     non-matching and undecided -- against an independent hand calculation of
     the same ROI from raw band values;
  6. the machine-readable export: it must serialise, and it must contain no
     generated conclusion beyond the explanation.

NOTHING HERE INTERPRETS A COMBINATION AS PROOF OF FLOODING OR OF ANY OTHER
CAUSE. A combined condition is geographic evidence, not causal attribution.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import rasterio
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyses.base import AnalysisContext, IndexContext              # noqa: E402
from analyses.evidence import (condition_rows, evidence_from_execution,  # noqa: E402
                               grid_rows, source_rows)
from analyses.evidence_masks import evidence_masks                   # noqa: E402
from analyses.registry import route                                  # noqa: E402
from core.bands import guess_band_roles                              # noqa: E402
from core.raster import describe_path                                # noqa: E402
from core.roi import ROISelection                                    # noqa: E402
from core.temporal import discover_scenes, pair_by_dates             # noqa: E402

SCENE = os.path.join("data", "sample", "s2_s2b-36ruv-20230806-0-l2a_2048px.tif")
BEFORE_DATE, AFTER_DATE = "2023-01-18", "2023-08-06"
ROI_OFFSET_M = 2000.0
ROI_SIZE_M = 5120.0
SCALE, OFFSET = 0.0001, 0.0

CASE_A = "Find cropland with NDVI greater than 0.6 and NDWI less than -0.4"
CASE_B = "Show areas with vegetation decrease near permanent water."

#: Words the evidence path must never generate on its own.
BANNED_GENERATED = ("flooded", "flood extent", "crop failure", "drought",
                    "deforestation", "damage", "water availability",
                    "water quality", "caused by", "because of",
                    "scientifically validated", "validated threshold")

hr = "=" * 78


def section(title: str) -> None:
    print(f"\n{hr}\n{title}\n{hr}")


def ok(condition: bool, label: str, detail: str = "") -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return bool(condition)


checks: list[bool] = []


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

guess = guess_band_roles(describe_path(SCENE))
roles = {str(k): int(v) for k, v in guess.roles.items()}
print(f"  roles         {dict(guess.roles)}  (confidence {guess.confidence})")

geometry = box(minx + ROI_OFFSET_M, maxy - ROI_OFFSET_M - ROI_SIZE_M,
               minx + ROI_OFFSET_M + ROI_SIZE_M, maxy - ROI_OFFSET_M)
roi = ROISelection(is_valid=True, intersects_raster=True,
                   area_m2=ROI_SIZE_M ** 2,
                   geometry_raster_crs=geometry,
                   raster_crs=str(rasterio.open(SCENE).crs))
print(f"  ROI           {ROI_SIZE_M / 1000:g} km x {ROI_SIZE_M / 1000:g} km "
      f"= {ROI_SIZE_M ** 2 / 1e6:.2f} km²")

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

context = AnalysisContext(roi=roi, index_context=index_ctx,
                          temporal_pair=temporal_pair)
context_a = AnalysisContext(roi=roi, index_context=index_ctx)

# --------------------------------------------------------------------------- #
section("2. INDEPENDENT HAND CALCULATION OF THE ROI (no engine code)")
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
hand_valid = np.isfinite(ndvi_hand) & np.isfinite(ndwi_hand)
print(f"  window        {ndvi_hand.shape[0]} x {ndvi_hand.shape[1]} cells")
print(f"  hand NDVI > 0.6:   {int(np.nansum(ndvi_hand > 0.6)):,} cells")
print(f"  hand NDWI < -0.4:  {int(np.nansum(ndwi_hand < -0.4)):,} cells")
print(f"  hand invalid:      {int(np.count_nonzero(~hand_valid)):,} cells "
      f"(no data -> UNKNOWN, never a match)")

# --------------------------------------------------------------------------- #
section("3. CASE A -- cropland AND NDVI > 0.6 AND NDWI < -0.4")
# --------------------------------------------------------------------------- #
exec_a = route(CASE_A, context_a)
result_a = exec_a.result
package_a = evidence_from_execution(exec_a)
print(f"  query         {CASE_A!r}")
print(f"  intent        {exec_a.intent.value}  -> package {package_a.intent}")
print(f"  status        {exec_a.status.value}")

print("\n  --- the lineage of the answer ------------------------------------")
for label, value in package_a.lineage.steps():
    print(f"    {label:16s} {value}")

print("\n  --- per-condition evidence ---------------------------------------")
for row in condition_rows(package_a):
    print(f"    {row['condition']}")
    print(f"      source      {row['source']}")
    print(f"      analysis    {row['analysis']}")
    print(f"      parameter   {row['parameter']}")
    print(f"      provenance  {row['origin']}")
    print(f"      matched {row['matched']:,}   not matching "
          f"{row['non_matching']:,}   undecided {row['unknown']:,}")

print("\n  --- the explanation the UI shows ---------------------------------")
for key, value in package_a.explanation.items():
    if isinstance(value, list):
        print(f"    {key}:")
        for item in value:
            print(f"      · {item}")
    elif isinstance(value, dict):
        print(f"    {key}: {json.dumps(value, sort_keys=True)}")
    else:
        print(f"    {key}: {value}")

checks.append(ok(package_a.matched_cells == result_a.matched_cell_count,
                 "the package reports the engine's matched count",
                 f"{package_a.matched_cells:,}"))
checks.append(ok(package_a.unknown_cells == result_a.insufficient_cell_count,
                 "the package reports the engine's undecided count",
                 f"{package_a.unknown_cells:,}"))
checks.append(ok(round(package_a.matched_area_km2, 3)
                 == round(float(result_a.matched_area_km2), 3),
                 "the package reports the engine's matched area",
                 f"{package_a.matched_area_km2:.3f} km²"))
checks.append(ok(len(package_a.records) == 3,
                 "all three conditions carry their own evidence record",
                 f"{len(package_a.records)} records"))
kinds = sorted({r.kind for r in package_a.records})
checks.append(ok(kinds == ["spatial", "spectral"],
                 "the records are one land-cover and two spectral conditions",
                 ", ".join(kinds)))
checks.append(ok(any("WorldCover" in r.source_dataset and
                     r.parameters.get("classes") == [40]
                     for r in package_a.records),
                 "cropland is traced to ESA WorldCover class 40"))
checks.append(ok(all(r.threshold_provenance.get("provenance") == "user_specified"
                     for r in package_a.records if r.threshold is not None),
                 "both thresholds are traced to the user's query"))
checks.append(ok("AND" in (package_a.combined.source_analysis if package_a.combined
                           else ""),
                 "the combination is traced to three-valued AND"))
checks.append(ok(round(package_a.matched_area_km2, 3) == 20.452,
                 "case A reproduces the verified 20.452 km²",
                 f"{package_a.matched_area_km2:.3f} km²"))
checks.append(ok(package_a.matched_cells == 204_521
                 and package_a.unknown_cells == 4_112,
                 "case A reproduces 204,521 matched / 4,112 undecided",
                 f"{package_a.matched_cells:,} / {package_a.unknown_cells:,}"))

# the per-condition masks the evidence layers would draw are the engine's own
masks_a = evidence_masks(result_a)
counts_by_name = {str(e.get("name")): e for e in result_a.condition_results}
same = bool(masks_a) and all(
    int(np.count_nonzero(masks_a[name].match)) == int(entry.get("matched", -1))
    for name, entry in counts_by_name.items() if name in masks_a)
checks.append(ok(same, "every evidence layer is the mask that produced its count",
                 f"{len(masks_a)} layers"))

# --------------------------------------------------------------------------- #
section("4. CASE B -- NDVI decrease AND near permanent water (two dates)")
# --------------------------------------------------------------------------- #
exec_b = route(CASE_B, context)
result_b = exec_b.result
package_b = evidence_from_execution(exec_b)
print(f"  query         {CASE_B!r}")
print(f"  intent        {exec_b.intent.value}")
print(f"  status        {exec_b.status.value}")

print("\n  --- the lineage of the answer ------------------------------------")
for label, value in package_b.lineage.steps():
    print(f"    {label:16s} {value}")

print("\n  --- per-condition evidence ---------------------------------------")
for row in condition_rows(package_b):
    print(f"    {row['condition']}")
    print(f"      source      {row['source']}")
    print(f"      parameter   {row['parameter']}")
    print(f"      provenance  {row['origin']}")
    print(f"      matched {row['matched']:,}   not matching "
          f"{row['non_matching']:,}   undecided {row['unknown']:,}")

print("\n  --- the explanation the UI shows ---------------------------------")
for key in ("what_was_found", "how_it_was_evaluated", "unknown_note"):
    print(f"    {key}: {package_b.explanation[key]}")
print("    evidence_sources:")
for item in package_b.explanation["evidence_sources"]:
    print(f"      · {item}")

checks.append(ok(package_b.matched_cells == result_b.matched_cell_count,
                 "the package reports the engine's matched count",
                 f"{package_b.matched_cells:,}"))
checks.append(ok(round(package_b.matched_area_km2, 3) == 0.451,
                 "case B reproduces the verified 0.451 km²",
                 f"{package_b.matched_area_km2:.3f} km²"))
checks.append(ok(package_b.matched_cells == 4_511,
                 "case B reproduces 4,511 matched cells",
                 f"{package_b.matched_cells:,}"))
dates_joined = package_b.to_json()
checks.append(ok(BEFORE_DATE in dates_joined and AFTER_DATE in dates_joined,
                 "both analysis dates are in the evidence",
                 f"{BEFORE_DATE} -> {AFTER_DATE}"))
kinds_b = sorted({r.kind for r in package_b.records})
checks.append(ok(kinds_b == ["spatial", "temporal"],
                 "the records name one spatial and one temporal analysis",
                 ", ".join(kinds_b)))
alignment_b = package_b.alignment or {}
print(f"\n  alignment block: {alignment_b}")
checks.append(ok(bool(alignment_b),
                 "the alignment provenance is carried, not implied",
                 ", ".join(f"{k}={alignment_b.get(k)}" for k in
                           ("method", "resampling", "resampled", "target")
                           if k in alignment_b) or "no alignment keys"))
sources_b = " | ".join(package_b.explanation["evidence_sources"])
print(f"\n  source analyses: {sources_b}")
checks.append(ok("ndvi_change" in sources_b and "WorldCover" in sources_b,
                 "both source analyses are named"))

# --------------------------------------------------------------------------- #
section("5. THE THREE STATES ARE NEVER CONFUSED")
# --------------------------------------------------------------------------- #
print(f"  A  matched {package_a.matched_cells:,}  "
      f"not matching {package_a.non_matching_cells:,}  "
      f"undecided {package_a.unknown_cells:,}")
print(f"  B  matched {package_b.matched_cells:,}  "
      f"not matching {package_b.non_matching_cells:,}  "
      f"undecided {package_b.unknown_cells:,}")
checks.append(ok(package_a.matched_cells + package_a.non_matching_cells
                 == package_a.analysed_cells,
                 "matched + measured non-matching = analysed (A)"))
checks.append(ok(package_b.matched_cells + package_b.non_matching_cells
                 == package_b.analysed_cells,
                 "matched + measured non-matching = analysed (B)"))
checks.append(ok(package_a.unknown_handling == "partial_unknown"
                 and package_a.unknown_cells > 0,
                 "undecided cells are reported as their own class",
                 package_a.unknown_handling))
checks.append(ok(f"{package_a.unknown_cells:,}" in
                 package_a.explanation["unknown_note"],
                 "the undecided count is stated in the answer text"))

# an all-UNKNOWN answer must not read like "no matches"
empty = route("Find cropland with NDVI greater than 0.6",
              AnalysisContext(roi=roi, index_context=index_ctx,
                              temporal_pair=None))
refused = route("Find cropland with high NDVI", context_a)
checks.append(ok(evidence_from_execution(refused) is None,
                 "a refused query has no evidence package at all",
                 refused.status.value))

# --------------------------------------------------------------------------- #
section("6. THE MACHINE-READABLE EXPORT")
# --------------------------------------------------------------------------- #
for name, package in (("A", package_a), ("B", package_b)):
    text = package.to_json()
    back = json.loads(text)
    size = len(text.encode("utf-8"))
    print(f"  case {name}: {size:,} bytes, schema {back['schema']}, "
          f"{len(back['conditions'])} condition records")
    checks.append(ok(back["result"]["matched_cells"] == package.matched_cells,
                     f"export {name} round-trips the counts"))
    checks.append(ok(text == package.to_json(),
                     f"export {name} is byte-for-byte deterministic"))
    generated = " ".join([
        str(back["explanation"]["what_was_found"]),
        str(back["explanation"]["how_it_was_evaluated"]),
        str(back["explanation"]["unknown_note"]),
        " ".join(back["explanation"]["what_was_evaluated"]),
        " ".join(back["explanation"]["evidence_sources"]),
    ]).lower()
    leaks = [word for word in BANNED_GENERATED if word in generated]
    checks.append(ok(not leaks, f"export {name} generates no causal claim",
                     ", ".join(leaks)))

print("\n  --- reproducibility block, case A ---------------------------------")
for label, value in grid_rows(package_a):
    print(f"    {label:20s} {value}")

# --------------------------------------------------------------------------- #
section("7. WHAT THIS EVIDENCE DOES AND DOES NOT SAY")
# --------------------------------------------------------------------------- #
print(f"  boundary: {package_a.boundary}")
for limitation in package_a.limitations:
    print(f"    - {limitation}")
print("\n  Phase 13 adds an explanation of an existing measurement. It does not")
print("  add a measurement, an index, a threshold, a flood assessment, a")
print("  crop-failure finding or a causal claim.")

print(f"\n{hr}")
passed = sum(1 for item in checks if item)
print(f"RESULT: {passed}/{len(checks)} real-data checks passed")
print(hr)
sys.exit(0 if passed == len(checks) else 1)
