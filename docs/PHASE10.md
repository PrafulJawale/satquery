# Phase 10 — Temporal NDVI Change Analysis and Before/After Comparison

**Status:** implemented and verified on real data.
**Scope:** compare NDVI for the *same* selected area across **two dated
acquisitions**. Nothing else is claimed.

---

## 1. Objective and scope

The first — and so far only — temporal capability is:

> Compare NDVI for the same ROI across two valid satellite scenes or acquisition
> dates.

Everything the phase does is in service of that one sentence. Flood detection,
SAR, NDWI change, time series and trend fitting are **not** implemented and are
refused (§9).

The governing rule is scientific caution: a ΔNDVI number is evidence that the
*index* changed, never of *why* it changed. That rule is enforced in code (the
wording comes from configuration), in tests (a test fails if a causal word
appears outside the caveat), and in the UI (the map legend restates it).

---

## 2. Files added and changed

### Added

| File | Purpose |
|---|---|
| `core/temporal.py` | Scene-pair model (`SceneRef`, `ScenePair`), validation, alignment planning, date parsing, scene discovery |
| `analyses/ndvi_change.py` | The engine: `NDVIChangeResult`, `compare_ndvi()`, `compose_message()`, `run_ndvi_change()` |
| `config/temporal/ndvi_change.yml` | Every threshold, class code, colour, formula string, wording string and limitation |
| `data/sample/s2_s2b-36ruv-20230118-0-l2a_2048px.tif` | The second real acquisition (23.8 MB) + `.provenance.json` |
| `scripts/fetch_sentinel2_second_scene.py` | Reproducible fetch of a second scene, pinned to the same pixel window |
| `scripts/verify_phase10_real_data.py` | Reproducible real-data verification (prints the numbers in §7) |
| `scripts/verify_phase10_browser.py` | One fresh browser smoke test |
| `tests/test_phase10_ndvi_change.py` | 44 unit tests + the real-data hand-check |
| `docs/PHASE10.md` | This report |

### Changed

| File | Change | Why |
|---|---|---|
| `analyses/base.py` | `Status.NEEDS_TWO_DATES`, `Status.NO_TEMPORAL_OVERLAP`; `AnalysisContext.temporal_pair` (optional) + `has_temporal_pair` | New outcomes; the engine can name the missing date, which a generic context message cannot. All fields are optional, so every Phase 1–9 caller is unchanged. |
| `core/router.py` | `Intent.NDVI_CHANGE_ROI`, `Intent.TEMPORAL_COMPARISON`; `VEGETATION_CHANGE` moved from *planned* to *implemented*; `REQUIRED_CONTEXT` entries; pattern vocabulary | Deterministic routing, no free-form parsing |
| `analyses/registry.py` | Three intents registered against one engine; `FLOOD_CHANGE` left planned | One place binds intent → code |
| `ui/map.py` | `ndvi_change_rgba()`, `ndvi_change_legend_html()`, `temporal_change_style()` (reads the palette from config) | Dedicated layers + a legend that says what the map cannot |
| `ui/components.py` | `render_ndvi_change()` | Numbers, classes, threshold, caveat, provenance, limitations |
| `app.py` | Two-acquisition selectors, context wiring, four display layers, legend priority, chat rendering | Integration |

### Not changed

`core/indices.py` (NDVI maths), `core/alignment.py`, `core/spatial_query.py`,
`core/suitability.py`, `analyses/ndvi.py`, `analyses/spatial_query.py`,
`analyses/crop_suitability.py`, `core/roi.py`, `core/router.py`'s Phase 1–9
vocabulary, `tileserver.py`, `ui/map.py`'s basemaps, `docs/PHASE9_REPORT.md`.

---

## 3. Supported user queries

Routed by the deterministic router (phrase + token scoring, no free-form NL
parsing). Verified by `tests/test_phase10_ndvi_change.py`.

| Query | Intent |
|---|---|
| *"Compare NDVI between these two dates."* | `NDVI_CHANGE_ROI` |
| *"Show the NDVI difference for this area."* | `NDVI_CHANGE_ROI` |
| *"Show NDVI change in this area."* | `NDVI_CHANGE_ROI` |
| *"What is the NDVI change here?"* | `NDVI_CHANGE_ROI` |
| *"Compare before and after."* | `TEMPORAL_COMPARISON` |
| *"How has this area changed between these two dates?"* | `TEMPORAL_COMPARISON` |
| *"How has vegetation changed in this area?"* | `VEGETATION_CHANGE` |
| *"Find vegetation loss."* | `VEGETATION_CHANGE` |
| *"Which parts of this ROI experienced vegetation decrease?"* | `VEGETATION_CHANGE` |

All three intents run the same engine. They differ only in the vocabulary that
reaches it, because "compare NDVI between two dates", "compare before and after"
and "how has the vegetation changed" are three different ways of asking the same
question.

If dates or scenes are missing the engine returns **`NEEDS_TWO_DATES`**.
The system never selects a date on the user's behalf: a plausible-looking
default date is a fabricated result.

---

## 4. Data sources and acquisition dates

Two genuine Copernicus Sentinel-2 MSI Level-2A acquisitions of the **same
20.48 km window** (MGRS tile `36RUV`, Nile Delta, Egypt):

| Role | STAC item | Date | Platform | Cloud | EPSG | Grid |
|---|---|---|---|---|---|---|
| before | `S2B_36RUV_20230118_0_L2A` | 2023-01-18 | sentinel-2b | 0.061 % | 32636 | 2048×2048 @ 10 m |
| after | `S2B_36RUV_20230806_0_L2A` | 2023-08-06 | sentinel-2b | 0.002 % | 32636 | 2048×2048 @ 10 m |

* **Source:** AWS Open Data Sentinel-2 L2A COG archive via the Earth Search STAC
  API (`https://earth-search.aws.element84.com/v1`).
* **Licence:** Copernicus Open Data Licence (free, attribution requested).
* **Bands:** B02 blue 490 nm, B03 green 560 nm, B04 red 665 nm, B08 nir 842 nm
  (10 m, uint16 DN, nodata 0, reflectance = DN × 0.0001, offset 0 — the same
  convention verified in Phase 2).
* **Why this pair:** same sensor, same processing level, same MGRS tile, same
  CRS, and both nearly cloud-free — so the comparison is like-for-like. 60
  acquisitions of this window exist for 2023; this pair is the cloud-free
  January/August contrast.

The second scene is fetched by `scripts/fetch_sentinel2_second_scene.py`, which
reuses the **same pixel window** as the first (`col_off=7720, row_off=3774,
2048×2048`). That pin is the reason the real-data path needs no resampling
(§5). The script asserts CRS, shape and transform equality before writing, and
records `grid_matches_reference: true` in the sidecar provenance.

---

## 5. Alignment method

Alignment is decided by `core/temporal.plan_alignment()` and **recorded** in
every result (`result.alignment`, and `result.provenance`).

**Case A — identical grids (the real-data path).**
Same CRS, same affine transform (tolerance 1e-9), same shape → method
`identical_grid`, `resampling: null`. The difference is computed on the native
10 m grid: no interpolation, nothing invented. This is what the two bundled
scenes produce, and the script verifies it (`identical grid … : True`).

**Case B — different grids.**
Method `common_grid_resampled`:

* one explicit common grid covering the ROI, in the CRS of the **coarser** scene
  (ties go to `before`, so the choice is deterministic);
* cell size = the coarser of the two native cell sizes, never finer — resampling
  cannot create detail that was not measured;
* `nearest` resampling (continuous reflectance: never bilinear artefacts;
  categorical: mandatory);
* both scenes are read onto that grid — never one into the other, and never
  implicitly;
* nodata is tracked per scene; the comparison uses the **intersection** of the
  two valid masks;
* the record states `resampled: true`, the resolution, the target and the
  method, and the result carries the sentence *"Resampling adds no
  information."*

**The guard that makes this honest:** arrays are only ever subtracted after the
grid has been established. When the grids differ, both arrays are read onto the
common grid *before* any arithmetic. There is no code path that subtracts two
arrays of different shape or transform.

---

## 6. Formulas, classification and provenance

**NDVI** (identical to the single-date engine — `core.indices.compute_ndvi` is
called, not reimplemented, so the two cannot drift):

```
reflectance = DN × 0.0001 + 0.0
NDVI        = (NIR − RED) / (NIR + RED)      undefined where |NIR + RED| < 1e-6
```

**ΔNDVI:**

```
ΔNDVI = NDVI_after − NDVI_before        computed only where BOTH are valid
```

**Classification** (`config/temporal/ndvi_change.yml`):

```
ΔNDVI ≥  +threshold  →  Increase        (code 3, green)
ΔNDVI ≤  −threshold  →  Decrease        (code 1, red)
otherwise            →  Stable          (code 2, pale)
not valid on both    →  Insufficient    (code 0, grey) — never counted as stable
```

* Threshold **0.10** for both directions, configurable in the YAML. Its
  rationale is recorded verbatim in the config and shown in the UI:
  *"Display convention, not a validated threshold… It carries no causal
  meaning."* The boundary is inclusive on both sides (tested).
* Insufficient data is 5–15 % of a typical ROI and is always reported as its own
  class — an unknown is not a zero.
* A comparison is refused (no statistics at all) when fewer than **60 %** of the
  ROI's cells are valid on both dates, or fewer than **4** cells are comparable.

**Provenance** returned with every result: engine, config version, both formula
strings, reflectance scale/offset per scene, band indices and names, the full
`SceneRef` records (item id, datetime, platform, cloud cover, MGRS tile, EPSG,
licence), ROI cell count, valid fraction, runtime, the alignment record, the
thresholds, and the limitations list.

---

## 7. Real-data numerical verification

Reproduce with `python scripts/verify_phase10_real_data.py` (PYTHONPATH=repo
root).

### 7.1 Independent hand-check (8×8 block, raw DN, no engine code)

Block at row 1000, column 1000. Reflectance = DN / 10000.

```
before: red 265,  nir 4720  →  (0.4720 − 0.0265) / (0.4720 + 0.0265) = +0.893681
after : red 242,  nir 3480  →  (0.3480 − 0.0242) / (0.3480 + 0.0242) = +0.869962
delta                       →  −0.023719
```

The engine's ΔNDVI raster over the same block: **64/64 cells match**, maximum
absolute difference **1.48 × 10⁻⁷** (float32 rounding). `RESULT: MATCH`.

### 7.2 A real 5.12 km window (512 × 512 px at 10 m, native grid)

```
ROI cells            262,144
comparable cells     262,144  (100.0 % of the ROI)
NDVI before          mean +0.6316   median +0.7578   std 0.2840
NDVI after           mean +0.7237   median +0.8770   std 0.2912
ΔNDVI                mean +0.0922   median +0.0202   std 0.2243
ΔNDVI range          −0.800 … +0.783
classes (t = ±0.10)  increase 28.1 %   stable 64.6 %   decrease 7.3 %
alignment            identical_grid, 10 m, resampling = none
runtime              ~105 ms
```

The direction (a greener August than January over this part of the Nile Delta),
the mixed field-level pattern and the wide ΔNDVI range are all consistent with
real crop rotation rather than a uniform artefact.

### 7.3 Whole-scene context (overview statistics, not the engine)

Mean NDVI 0.6176 (January) → 0.6144 (August); Δ median −0.0091. The 5.12 km
window in §7.2 is not the whole scene, and the two figures are not in conflict:
they are different areas. This is exactly why the tool reports the selected area
only and never extrapolates.

---

## 8. Tests

```
tests/test_phase10_ndvi_change.py   44 passed
full suite (pytest tests -q)       553 passed, 0 failed   (509 before + 44 new)
```

Coverage against the required list:

| Requirement | Test |
|---|---|
| NDVI before / after correct | `test_ndvi_before_is_correct`, `test_ndvi_after_is_correct` |
| ΔNDVI correct | `test_delta_is_after_minus_before`, `test_delta_raster_matches_hand_computation` |
| valid-pixel intersection | `test_valid_pixels_are_the_intersection_of_both_dates` |
| nodata handling | `test_nodata_value_is_honoured` |
| date ordering | `test_inverted_dates_are_refused`, `test_equal_dates_returns_needs_two_dates`, `test_missing_date_returns_needs_two_dates`, `test_parse_date_rejects_garbage` |
| missing date / scene | `test_missing_scene_returns_needs_two_dates`, `test_engine_needs_two_dates`, `test_routing_returns_needs_two_dates_without_a_pair` |
| missing band | `test_scene_without_nir_returns_unsupported` |
| non-overlapping scenes | `test_non_overlapping_scenes_are_refused` |
| CRS/grid mismatch | `test_mismatched_grids_are_aligned_on_one_common_grid` |
| alignment correctness | `test_alignment_keeps_the_same_ground_together`, `test_identical_grids_are_detected`, `test_identical_grids_are_not_resampled` |
| threshold classification | `test_threshold_classification`, `test_threshold_is_configurable`, `test_boundary_is_inclusive_on_both_sides` |
| all-zero / all-invalid / constant | `test_all_zero_bands_have_no_valid_pixels`, `test_all_invalid_returns_no_valid_pixels`, `test_constant_values_give_zero_change` |
| ROI outside temporal coverage | `test_roi_outside_one_scene_is_refused`, `test_real_data_roi_outside_coverage_is_refused` |
| unsupported flood query | `test_flood_and_sar_queries_stay_unsupported` |
| Phase 1–9 regression | the full suite |
| independent hand-check | `test_real_data_hand_check_8x8_block` |
| real data end to end | `test_real_data_comparison_over_a_5km_window`, `test_bundled_scenes_are_discovered_and_dated`, `test_bundled_scenes_share_one_grid` |

Four Phase 7/8 tests were **updated** rather than the implementation reverted:
they asserted the old fact that `VEGETATION_CHANGE` had no engine. Each was
rewritten to keep its original invariant — notably
`test_vegetation_change_is_not_silently_ndvi`, now sharper than before: a change
question must reach a temporal engine and ask for dates, and must *never* return
a single-date NDVI mean.

---

## 9. Browser verification

One fresh smoke test against a freshly started server
(`scripts/verify_phase10_browser.py`): **23 passed, 0 failed**.

It draws a ROI on the live map, asks *"Compare NDVI before and after."* and
verifies in the DOM: both acquisition dates offered; the ROI registered; the
change panel with before/after/ΔNDVI metrics; all three classes listed;
insufficient data separated from stable; the causal caveat present; **no causal
claim** anywhere outside that caveat; the four new map layers
(`NDVI before (2023-01-18)`, `NDVI after (2023-08-06)`,
`ΔNDVI (after − before)`, `NDVI change class (…)`); and every pre-existing layer
(`Raster footprint`, `True colour (RGB)`, `False colour (NIR-R-G)`) still
present — the temporal layers are **added, not substituted**.

OCR of the resulting screenshot confirms the legend on screen:

> **NDVI change 2023-01-18 → 2023-08-06** — *Change in the vegetation index —
> not a cause* … *Decrease means the index fell. It does not establish
> deforestation, crop failure, drought or flooding. Insufficient data means not
> measured on both dates, and is never counted as stable.*

---

## 10. Known limitations

1. **Two acquisitions only.** This is a before/after comparison, not a time
   series; no trend, seasonality or rate of change is computed.
2. **No causal attribution.** A decrease is a decrease in the index. Identifying
   deforestation, crop failure, drought or flood requires data this app does not
   have.
3. **Cloud and cloud shadow are not masked per pixel.** Scene-wide cloud cover is
   reported in the provenance instead (0.061 % and 0.002 % for the bundled
   pair). A partly cloudy pair would produce partly meaningless differences; the
   engine does not detect this.
4. **No atmospheric/BRDF normalisation** beyond Sentinel-2 L2A processing.
   Illumination and view-angle differences between dates contribute to ΔNDVI.
5. **The 0.10 threshold is a display convention**, not a validated agronomic or
   ecological threshold.
6. **Cell budget: 2,000,000 cells** (the same limit the Phase 8/9 analyses use).
   The full 20.5 km scene is 4.19 M cells, so a selection covering the whole
   scene is *refused with an explanation* rather than silently coarsened.
7. **Resampling adds no information.** Where grids differ, results are limited to
   the coarser native resolution; the record says so.
8. **ROI-first, like the rest of the app.** Only the selected area is read and
   compared; results are never extrapolated beyond it.
9. **Band roles come from band descriptions** (B04/B08), falling back to the
   documented sample order. A raster with neither is rejected rather than
   guessed.
10. **No date is ever inferred.** Missing or ambiguous dates return
    `NEEDS_TWO_DATES`.

---

## 11. Explicitly unsupported

Refused, not approximated:

* **Flood / water-extent change** (`FLOOD_CHANGE`) — needs a water index and a
  second date; NDVI cannot answer it. Returns `UNSUPPORTED` with the message
  *"Flood analysis is not available yet."*
* **SAR / radar change detection** — no SAR data, no speckle-aware processing.
* **NDWI or other index differences** — only NDVI is implemented.
* **Time series, trends, anomaly detection** — two dates is not a series.
* **Change in crop suitability, land cover or yield** — different analyses, none
  of which have a temporal engine.

---

## 12. Confirmation that Phase 9 is unchanged

* `docs/PHASE9_REPORT.md` was not modified.
* `core/spatial_query.py`, `analyses/spatial_query.py`, `core/suitability.py`,
  `analyses/crop_suitability.py` and `config/spatial/*` were not modified.
* The Phase 9 status semantics (`UNSUPPORTED_CONDITION`, `INSUFFICIENT_DATA`,
  three-valued masks, distance rule, edge rule) are untouched; `Status` gained
  two members and nothing else.
* `FLOOD_CHANGE` remains a planned intent with no handler, so flood and SAR
  queries still return `UNSUPPORTED`.
* The 509 pre-existing tests still pass; the four that changed were the ones
  whose assertions encoded "vegetation change has no engine yet".
* The basemap/global-map work (server-side tile proxy, world extent, search,
  ROI drawing) is untouched.
