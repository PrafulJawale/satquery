# Phase 11 — Extensible Multispectral Index Analysis (NDWI)

**NDWI is an index calculation, not flood detection.**

**Status:** implemented and verified on real data.

---

## 1. Objective

Add a second spectral index — **NDWI** — without creating a second index
engine. NDVI has been the only index since Phase 3; a fourth phase would
otherwise have produced a fourth copy of the same denominator guard, the same
nodata handling, the same NaN bookkeeping and the same caveat wording. So an
index is now **configuration plus one shared computation**, and NDWI is the
first index added that way.

## 2. Scope

Exactly one new capability: **NDWI statistics for a user-selected area**,
computed from native B03 (green) and B08 (NIR) reflectance.

Not in scope, and refused rather than approximated: flood detection, flood
change, water extent change between dates, SAR, water-body segmentation,
automatic water extraction, water proximity, temporal NDWI, time series,
water-quality / irrigation / groundwater inference. See §14.

## 3. Architecture

```
config/indices/ndvi.yml ─┐
                         ├─► core/index_definitions.py   IndexDefinition
config/indices/ndwi.yml ─┘        (roles, formula, guard, range, wording)
                                        │
                                        ▼
                     core/indices.py   compute_index(definition, bands)
                                        │  (ONE implementation)
                    ┌───────────────────┴───────────────────┐
            compute_ndvi(...)  delegates              index_from_dataset(...)
                    │                                        │
            analyses/ndvi.py  UNCHANGED              analyses/ndwi.py
                                                            │
                                              ROINDVIStats (+ index_name)
```

* `core/index_definitions.py` — the `IndexDefinition` dataclass and a small
  YAML-backed registry (`get_index("ndwi")`, `available_indices()`).
* `core/indices.py` — `compute_index()` (the only place an index is computed)
  and `index_from_dataset()` (ROI-windowed reader).
* `analyses/ndwi.py` — an adapter in the same shape as `analyses/ndvi.py`:
  it validates the context, delegates the arithmetic, and turns the result into
  an `AnalysisExecution`. It contains no index arithmetic of its own.

## 4. Index abstraction

An `IndexDefinition` carries `name`, `short_name`, `display_name`, `label`,
`kind`, `numerator_role`, `denominator_role`, `required_roles`, `role_labels`,
`sensor_bands`, `wavelengths_nm`, `formula`, `citation`, `min_denominator`,
`value_range`, `colormap`, `reflectance_scale/offset`, `description`, `caveat`,
`limitations` and `version`.

`kind` is `normalized_difference` for NDVI and NDWI:

```
value = (numerator − denominator) / (numerator + denominator)
```

`kind: custom` is reserved for SAVI/EVI-style formulae, which are refused today
rather than approximated (`compute_index` raises for an unimplemented kind).

**Adding NDBI or MNDWI later means adding one YAML file.** No new Python.

**NDVI was not left behind.** `compute_ndvi()` keeps its exact signature and
now delegates to `compute_index()` with the NDVI definition. The gate on that
decision is
`tests/test_phase11_ndwi.py::test_ndvi_delegates_with_identical_output`, which
compares `compute_ndvi` against **the Phase 3 implementation, reproduced
verbatim in the test**, over five random seeds: name, label, method, colormap,
value range, `bands_used`, every caveat string, counts, statistics and the raw
array are all bit-for-bit identical. Two further tests pin the shape-mismatch
message and the out-of-range caveat wording.

## 5. NDWI formula

```
reflectance  = DN × 0.0001 + 0.0        (Sentinel-2 L2A convention, Phase 2/10)
NDWI         = (GREEN − NIR) / (GREEN + NIR)                 [McFeeters 1996]
|GREEN + NIR| < 1e-6  →  undefined (NaN), never 0.0
```

The denominator guard is **inherited from the NDVI convention**, not invented
for NDWI: `min_denominator = 1.0e-6` in `config/indices/ndwi.yml`, asserted
equal to NDVI's by a test. Values are computed from native analytical raster
band values — never from RGB/FCC display images and never from
display-resampled imagery.

There is **no water threshold anywhere in Phase 11**. A rule such as "water if
NDWI > 0" would convert a continuous index into an unsupported water-extraction
claim, and NDWI is known to over-detect water over built-up surfaces
(`docs/ARCHITECTURE_REVIEW.md` recommends MNDWI for water mapping — and MNDWI
needs a SWIR band this sample does not carry).

## 6. Band mapping

Roles come from **band metadata**, never from band position:
`core.bands.guess_band_roles()` matches Sentinel-2 band ids (`B03_green_560nm`),
then Landsat ids, then central wavelengths, then plain words.

```
green → B03 (band 2)     nir → B08 (band 4)     confidence: high
```

If either role cannot be established from evidence, the analysis returns
**`UNSUPPORTED`** and names the missing role. Band order is never used as a
fallback — a test writes an unlabelled 4-band raster and asserts no NDWI is
produced.

## 7. ROI processing

ROI-first, like Phase 10:

1. `core.alignment.native_roi_grid()` snaps the ROI to the raster's native grid
   and clamps it to the raster (2-cell read buffer, floor/ceil so the ROI can
   never be clipped by rounding). This helper is new but is **shared**: Phase
   10's private helper now delegates to it, so the two phases cannot drift.
2. Only that window of B03 and B08 is read.
3. `roi_mask()` applies the pixel-centre inclusion rule; the comparison uses
   geometry **and** data validity.
4. `core.statistics.calculate_roi_ndvi_stats()` (Phase 6, unchanged) produces
   the statistics over the valid pixels only.

**Cell budget:** `DEFAULT_MAX_CELLS = 2,000,000`. The whole 2048² scene is
4.19 M cells, so a selection covering it is **refused** with
`INSUFFICIENT_DATA` and a message naming both numbers — never silently
coarsened. The refusal carries provenance (`roi_cells`, `cell_budget`,
`refused`) so the UI can say *how* big the request was.

## 8. Provenance

Every result records: engine, the full index definition (name, formula,
citation, `min_denominator`, value range, version, limitations), the band
roles with indexes and documented band ids (B03/B08), the reflectance scale and
offset and their source, the band-role confidence and evidence, the native
resolution, the CRS, the ROI window actually read (col, row, w, h), the ROI
cell count, valid-pixel count, runtime, and the router decision (intent,
confidence, matched phrases).

## 9. Router behaviour

Deterministic phrase/token routing — no LLM. One new implemented intent and one
new planned intent.

| Query | Intent | Outcome |
|---|---|---|
| *"What is the NDWI of this area?"* | `NDWI_ROI_STATS` | runs |
| *"Calculate NDWI here."* | `NDWI_ROI_STATS` | runs |
| *"Show the water index for this ROI."* | `NDWI_ROI_STATS` | runs |
| *"What is the water index of this area?"* | `NDWI_ROI_STATS` | runs |
| *"Show NDWI for this region."* | `NDWI_ROI_STATS` | runs |
| *"What is the NDVI of this area?"* | `NDVI_ROI_STATS` | NDVI, unchanged |
| *"Did flooding happen?"* | `FLOOD_CHANGE` | UNSUPPORTED, unchanged |
| *"Flood detection using NDWI"* | `FLOOD_CHANGE` | UNSUPPORTED |
| *"Compare NDWI before and after."* | `TEMPORAL_NDWI` | UNSUPPORTED, named |
| *"Find cropland near water"* | `SPATIAL_QUERY` | Phase 9, unchanged |
| *"Find areas within 500 m of water"* | `SPATIAL_QUERY` | Phase 9, unchanged |

Three deliberate design points:

* **`"ndwi"` was a FLOOD_CHANGE phrase until now** — a plain NDWI question used
  to be refused as a flood question. It is now a *token* (weight 1) of
  `FLOOD_CHANGE` and a *phrase* (weight 3) of `NDWI_ROI_STATS`, so *"What is the
  NDWI of this area?"* is an index question while *"Flood detection using NDWI"*
  stays a flood question and stays unsupported.
* **Bare `"water"` is not an NDWI token.** Water proximity and water extent
  belong to the Phase 9 spatial query and to the unsupported flood intent.
* **`TEMPORAL_NDWI` exists so that "compare NDWI" cannot fall through** to the
  NDVI change engine. Answering an NDWI question with an NDVI difference would
  be a fabricated result, so the temporal-NDWI phrasings are matched and
  refused by name.

Statuses used: `NEEDS_ROI` (no selection), `UNSUPPORTED` (band roles unknown),
`NO_VALID_PIXELS` (nothing measurable), `INSUFFICIENT_DATA` (over the cell
budget), `ERROR` (contract violation, never a traceback), `OK`.

## 10. UI behaviour

* New layer **`NDWI — Water Index`**, added alongside — never replacing —
  `Raster footprint`, `True colour (RGB)`, `False colour (NIR-R-G)`, the NDVI
  layers, the Phase 9 spatial layers and the Phase 10 temporal layers.
* A **continuous** legend (diverging blue ramp, −1 … +1) headed
  *"Continuous index — no water threshold"*. No categorical water/non-water
  legend exists.
* The result panel reports cells in selection, valid cells, invalid/masked
  cells, valid %, and min / max / mean / median / std, followed by the caveat.
* **The caveat is mandatory and appears twice** (result panel and map legend):

  > NDWI is a spectral index. This result does not by itself establish flood
  > extent, water availability, or water quality.

* A *"How this index was computed"* expander shows the formula, the citation,
  the bands, the reflectance scaling, the denominator guard, the ROI window and
  the band-role confidence.
* The NDWI legend is the lowest-priority analysis legend: spatial, suitability
  and temporal (each of which changes what the map *means* more than one more
  continuous index) still win.

## 11. Real-data verification

Reproduce with `PYTHONPATH=. python scripts/verify_phase11_real_data.py`.

**Scene:** `s2_s2b-36ruv-20230806-0-l2a_2048px.tif` — Sentinel-2B L2A,
2023-08-06, tile 36RUV, EPSG:32636, 2048² @ 10 m, bands
B02/B03/B04/B08 (uint16 DN, nodata 0, DN × 0.0001 + 0), Copernicus Open Data
Licence. Band roles resolved at *high* confidence from band descriptions.

**Hand check — 8×8 block at row 1000, column 1000** (raw DN read straight from
the file; no engine code involved):

```
green DN 363   -> reflectance 0.0363
nir   DN 3480  -> reflectance 0.3480
NDWI = (0.0363 - 0.3480) / (0.0363 + 0.3480) = -0.811085
```

The engine over an ROI equal to that block: **64/64 cells match**, maximum
absolute difference **1.092 × 10⁻⁷** (float32 rounding). `RESULT: MATCH`.

**One real ROI — 5.12 km window (512 × 512 px at 10 m), 2000 m from the scene
corner:**

```
ROI cells      262,144
valid cells    262,144  (100.00 % of the ROI)
NDWI min/max   -0.9115 / +0.4058
NDWI mean      -0.6924
NDWI median    -0.7652
NDWI std dev    0.1831
window read    [198, 198, 516, 516]  (col, row, w, h) -- not the whole scene
runtime        81 ms
```

These are reported as measured index values. **No scientific interpretation of
them is offered**: the mean is strongly negative (a vegetated/agricultural
window) and the maximum is positive, and the application draws no conclusion
about water from either.

## 12. Tests

```
tests/test_phase11_ndwi.py    67 passed
full suite (pytest tests -q) 620 passed, 0 failed   (553 before + 67 new)
browser smoke test           21 passed, 0 failed
```

Coverage against the required list — numerical (formula, hand-computed values,
reflectance scaling, denominator guard, negatives, 0/0 invalid, constant
values, NaN/nodata, valid-pixel intersection, statistics), band handling
(B03/B08 selection, missing green → `UNSUPPORTED`, missing NIR →
`UNSUPPORTED`, unlabelled bands never guessed), ROI (ROI-only statistics,
outside coverage, 2 M-cell budget, windowed reads), router (NDWI, NDVI
unchanged, "water index", flood unsupported, temporal NDWI unsupported, water
proximity still Phase 9), real data (8×8 hand check, 5.12 km ROI, band roles
from metadata) and the NDVI delegation gate.

### Test modifications and why

Two tests were updated, both reported in the Phase 11 design before any code
was written:

* `tests/test_phase7_router.py::test_planned_intents_have_no_handler` — asserts
  the intent census. Phase 11 adds `NDWI_ROI_STATS` to *available* and
  `TEMPORAL_NDWI` to *planned*, so the expected sets changed. The test's
  invariant (every planned intent has no handler) is untouched.
* `tests/test_phase8_router.py::test_crop_suitability_is_registered_and_no_longer_planned`
  — same census, same reason.

A third, tiny consequence: the registry contract that an unavailable message
**ends with "not available yet."** forced the TEMPORAL_NDWI message to put its
explanation first. No Phase 1–10 test was weakened.

## 13. Limitations

1. **NDWI is an index, not a water body, a flood extent, a water depth, a water
   availability or a water-quality measurement.**
2. **No threshold is applied.** Nothing in Phase 11 classifies a pixel as water.
3. **Over-detection over built-up surfaces** is a known property of NDWI
   (McFeeters 1996); MNDWI (Green/SWIR) is preferred for water mapping and needs
   a SWIR band this sample does not carry.
4. **No per-pixel cloud or cloud-shadow masking.** Scene-level cloud cover is
   reported in the scene provenance instead.
5. **No atmospheric correction** beyond the source product's L2A processing.
6. **Sub-pixel water and water under canopy are not resolved** at 10 m.
7. **Cell budget 2,000,000** — a larger selection is refused, not coarsened.
8. **Band roles come from metadata only.** A raster with unnamed bands is
   refused rather than guessed.
9. **Local-file source only** (the ROI-windowed read needs a path); an uploaded
   file has no path and is refused with `UNSUPPORTED`.
10. **Single date.** Two-date NDWI is a separate, unimplemented analysis.
11. **Statistics only** — no NDWI map classification, no area of water, no
    change, no trend.

## 14. Explicitly unsupported

Refused, not approximated: flood detection · flood change · water-extent change
between dates · SAR analysis · SAR change detection · water-body segmentation ·
automatic water extraction · water proximity (that is the Phase 9 spatial
query) · temporal NDWI · time series · water-quality inference · irrigation
inference · groundwater inference.

*"Did flooding happen?"* is **not** answered from NDWI: it routes to
`FLOOD_CHANGE` and returns the existing unsupported response.
*"Compare NDWI before and after."* routes to `TEMPORAL_NDWI` and is refused by
name.

## 15. Compatibility with Phases 1–10

* **Phase 9 (frozen):** `core/spatial_query.py`, `analyses/spatial_query.py`,
  `core/suitability.py`, `analyses/crop_suitability.py`, `config/spatial/*` and
  `docs/PHASE9_REPORT.md` were not modified. *"Find cropland near water"* and
  *"within 500 m of water"* still route to `SPATIAL_QUERY`; irrigation,
  groundwater, salinity and flood conditions still abort the query with
  `UNSUPPORTED_CONDITION`.
* **Phase 10 (frozen behaviour):** `core/temporal.py`, `config/temporal/` and
  the NDVI-change semantics are untouched. `analyses/ndvi_change.py` changed in
  exactly one way — its private ROI-grid helper now delegates to the new shared
  `core.alignment.native_roi_grid()` — and its 44 tests still pass.
* **Phase 1–8:** the whole 553-test suite passes unchanged apart from the two
  census assertions described in §12. `compute_ndvi` is bit-for-bit identical
  (§4).
* **Future work, deliberately not started:** NDWI temporal comparison, flood
  detection, water segmentation, multi-index reasoning, natural-language
  multi-condition spectral reasoning, VLM/GeoChat integration, SAR, time
  series, anomaly detection.

## 16. Architectural tradeoffs

* **Generic engine + NDVI delegation** was chosen over "a second function beside
  the first". The risk (changing NDVI) is bounded by an executable copy of the
  Phase 3 implementation kept inside the test file; if that test ever fails, the
  correct response is to revert NDVI to its own implementation, not to relax the
  test.
* **Metadata-only band gating** was chosen over a second confirmation checkbox
  like NDVI's. The bundled Sentinel-2 sample names its bands explicitly, the
  roles resolve at *high* confidence, and the engine refuses with
  `UNSUPPORTED` whenever they do not — so no new status and no new UI gate were
  needed.
* **`ROINDVIStats` was extended, not duplicated.** It gained `index_name`
  (default `"ndvi"`), plus optional `raster`, `mask` and `runtime_ms`. Every
  existing caller and test sees what it saw before.
* **The NDWI spec is appended last in the registry**, because `suggestions()`
  walks the registry in insertion order and the Phase 8 cotton examples must
  stay inside `suggestions(limit=8)`.
