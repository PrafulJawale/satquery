# Phase 12 — Multi-Index Geospatial Reasoning and Evidence Composition

**Status:** implemented and verified. **688 tests pass** (620 before Phase 12,
68 new), **35/35 browser checks pass**, and both real-data compositions are
reproduced by an independent hand calculation.

---

## 1. Objective

Answer questions that combine several things this system already measures, in one
deterministic step:

> "Find cropland with NDVI greater than 0.6 and NDWI less than -0.4."
> "Show areas with vegetation decrease near permanent water."
> "Find areas with NDVI decrease and NDWI statistics."

Phase 12 adds **no new measurement**. It composes three existing capabilities:

| Contribution | Comes from |
|---|---|
| categorical spatial conditions (land cover, water, proximity) | Phase 9 |
| single-date spectral index thresholds (NDVI, NDWI) | Phase 11 |
| temporal NDVI change classes (increase / stable / decrease / insufficient) | Phase 10 |

It is not a model, not a classifier and not a scoring function. It is boolean
algebra over masks that already exist, with the provenance of every input
carried through to the answer.

### The scientific boundary

> **A combined condition is geographic evidence, not causal attribution.**

That sentence is stored once (`CAVEAT` in `analyses/multi_condition.py`), leads
the result's limitations, is printed in the answer panel, and is printed again on
the map legend. It is reproduced verbatim in this document, in the UI and in the
engine so the three cannot drift apart.

What follows from it, and is enforced in the wording of every result:

* NDVI decrease **together with** water proximity does not establish flooding.
* NDVI decrease **together with** a high NDWI does not establish flooding.
* A low NDVI **together with** a high NDWI does not classify an area as water.
* NDVI decrease does not establish crop failure, drought, deforestation or damage.
* NDWI does not establish water availability, water quality or flood extent.

---

## 2. Files added and changed

**Added**

| File | Role |
|---|---|
| `core/multi_condition.py` | parser and mask algebra (no raster I/O, no engine imports) |
| `analyses/multi_condition.py` | the composition engine and `MultiConditionResult` |
| `config/multi/thresholds.yml` | the threshold policy, conventions and limitations |
| `tests/test_phase12_multi_condition.py` | 68 tests |
| `scripts/verify_phase12_real_data.py` | real-data evidence, with a hand calculation |
| `scripts/verify_phase12_browser.py` | 29 browser checks against a live server |

**Changed**

| File | Change |
|---|---|
| `analyses/base.py` | `Status.NEEDS_THRESHOLD` added |
| `analyses/registry.py` | `MULTI_CONDITION` spec appended last; `route(..., convention=None)` |
| `analyses/__init__.py` | re-exports `run_multi_condition`, `MultiConditionResult` |
| `core/router.py` | `Intent.MULTI_CONDITION`, composition guard, `QueryIntent.composed` |
| `ui/map.py` | composed-condition palette, RGBA and legend |
| `ui/components.py` | `render_multi_condition()` panel and the convention opt-in |
| `app.py` | map layer, legend, session state, panel dispatch, `IndexContext` roles |
| `tests/test_phase7_router.py`, `tests/test_phase8_router.py` | intent census updated (approved) |

**Untouched, deliberately:** `core/spatial_query.py`, `analyses/spatial_query.py`,
`core/spatial.py`, `core/datasources/worldcover.py`, `config/spatial/*`,
`core/temporal.py`, `analyses/ndvi_change.py`, `config/temporal/*`,
`core/indices.py`, `core/index_definitions.py`, `config/indices/*`,
`analyses/ndvi.py`, `analyses/ndwi.py`, `core/alignment.py`, `core/router.py`
vocabulary of Phases 1–11, all basemap code, and `docs/PHASE9_REPORT.md`,
`docs/PHASE10.md`, `docs/PHASE11.md`.

---

## 3. Supported grammar

Parsing lives in `core/multi_condition.py` and is deterministic: regular
expressions over a normalised sentence, never a language model.

| Sentence | Interpretation |
|---|---|
| Find cropland with high NDVI. | cropland **AND** NDVI — *threshold missing* |
| Find cropland with low NDWI. | cropland **AND** NDWI — *threshold missing* |
| Find cropland with high NDVI and low NDWI. | cropland **AND** NDVI **AND** NDWI |
| Find cropland with NDVI greater than 0.6. | cropland **AND** NDVI > 0.6 |
| Show areas with vegetation decrease near permanent water. | water proximity **AND** NDVI decrease |
| Find cropland with vegetation increase. | cropland **AND** NDVI increase |
| Find areas with NDVI decrease and NDWI statistics. | NDVI decrease, **plus** NDWI measured over the match |

Three rules keep the grammar honest:

1. **A quality word is not a threshold.** "high NDVI" becomes a real condition
   whose threshold is *absent*; the engine refuses it (§4).
2. **A bare index mention is not a condition.** "What is the NDVI of this area?"
   does **not** become a composition — a condition needs a qualifier (a number,
   a relative phrase, or a quality word). That is what keeps `NDVI_ROI_STATS`
   and `NDWI_ROI_STATS` on their own intents.
3. **A two-date comparison is not a threshold request.** "Compare NDWI before and
   after" is left to the existing temporal refusal, not hijacked into a
   composition.

---

## 4. Threshold policy — one is never invented

Every threshold in a result carries a provenance string. There are exactly three
origins, and the engine otherwise refuses:

| Provenance | Source | Wording in the UI |
|---|---|---|
| `user_specified` | a number in the query | "from your query" |
| `config_convention` | a labelled convention, **opt-in only** | "a display/query convention … not a scientific classification" |
| `relative` | the ROI's own distribution (median / p25 / p75) | "relative to this area" |

The default policy in `config/multi/thresholds.yml` is
`require_explicit_threshold`. A bare "high NDVI" therefore returns:

```
NEEDS_THRESHOLD — NDVI was asked for without a threshold, and no threshold will
be invented. Ask for e.g. NDVI greater than 0.6, use a relative form ('above the
median of this area'), or enable a labelled convention in the UI.
```

The refusal is actionable rather than terminal: the UI offers the labelled
conventions as buttons, each printed with its own note ("Display/query
convention, not a validated agronomic threshold"), and each applied **only to
that one question, only when clicked**.

The opt-in travels through `route(query, context, convention=...)`. `QueryIntent`
is frozen, so the opt-in builds a **new** intent (`dataclasses.replace`) instead
of mutating the parsed one; if anything goes wrong the question is left exactly
as the parser found it — a failed opt-in can never become a free threshold. The
browser test clicks the button and asserts the result is labelled
"a display/query convention … enabled by you, and not a scientific
classification".

A relative threshold ("NDVI above the median of this area") is computed from the
selected area and is always reported as relative to it. It is never worded as a
land-cover or water rule.

---

## 5. The condition model

Each condition is a `ComposedCondition`:

| Field | Meaning |
|---|---|
| `kind` | `spatial` / `spectral` / `temporal` |
| `name`, `label` | machine name; one line for the UI |
| `source_analysis` | `worldcover`, `core.indices:ndvi`, `analyses.ndvi_change` |
| `parameters` | e.g. `{classes: [40]}`, `{distance_m: 1000}`, `{class: decrease}` |
| `threshold` | a `ThresholdSpec` (index, operator, value, provenance, detail) |
| `negate` | whether the condition is inverted |
| `spatial_condition` | the Phase 9 condition object, when spatial — **never re-derived** |
| `evidence`, `interpretation`, `limitations` | the matched text, a plain sentence, and what it does not mean |

Arrays are **never** combined merely because their shapes match. Each mask is
built from its own source, on one agreed grid, and records where it came from.

---

## 6. Grid and alignment rules

Before any two masks are combined, the engine verifies that the grids are the
same grid: **CRS, affine transform, shape, cell size, bounds and ROI**. `Grid`
comparisons are delegated to `core.spatial.require_compatible`, which raises
`GridMismatchError` rather than quietly resampling.

* The composition grid is chosen **once**, before any mask exists:
  * a temporal condition fixes it — Phase 10 owns the change raster, so the
    composition adopts **its** grid (`grid_source: phase10_temporal`);
  * otherwise it is the native Sentinel-2 grid over the ROI, cut by
    `core.alignment.native_roi_grid` (`grid_source: native_roi_grid`).
* `native_roi_grid` adds a two-cell rim so the ROI can never be clipped by
  rounding. Those rim cells are **not** inside the ROI and are therefore
  reported as UNKNOWN, never as matches.
* Any alignment performed is **recorded** in `result.alignment`, including the
  method, the transform, the resolution and `resampled: false`. Nothing is
  resampled silently; where the two scenes share a grid (the bundled pair), the
  record says so explicitly.
* Coarser native resolution is used where sources differ; nodata is preserved
  as UNKNOWN in every mask.

---

## 7. Three-valued logic

Every cell is one of three states, using the Phase 9 codes:

| Code | State | Drawn as |
|---|---|---|
| 2 | TRUE — satisfies every condition | teal |
| 1 | FALSE — measured, does not satisfy | faint grey |
| 0 | UNKNOWN — insufficient data | amber |

Composition follows the three-valued algebra of `core.spatial.combine_all`:

* **AND**: FALSE if any input is FALSE; otherwise UNKNOWN if any input is
  UNKNOWN; otherwise TRUE.
* **OR**: TRUE if any input is TRUE; otherwise UNKNOWN if any input is UNKNOWN;
  otherwise FALSE. (So `FALSE OR UNKNOWN` is UNKNOWN — an undecided operand is
  not discarded just because the operator is permissive.)
* **NOT** preserves UNKNOWN exactly, and double negation is the identity.

**UNKNOWN is never counted as a match and never counted as a non-match.** A
composition in which nothing could be decided reports
`status: insufficient_data`, never "no matches found".

---

## 8. Spatial conditions — reused, not re-derived

Spatial conditions are parsed by Phase 9's own parser
(`core.spatial_query.parse_spatial_query`) and evaluated with Phase 9's own mask
functions (`land_cover_mask`, `water_mask_from_land_cover`, `proximity_mask`).
Semantics are unchanged: water = class 80, cropland = 40, built-up = 50,
"outside water" = NOT WATER, proximity default 1000 m.

A proximity condition needs the rim *outside* the ROI, so the land cover is
fetched once on an **expanded** grid (`analyses.spatial_query.expanded_analysis_grid`)
and the result is cropped back to the ROI. One fetch serves every spatial
condition in the composition.

---

## 9. Spectral conditions

* **NDVI and NDWI only.** No new index, no second engine.
* Computed through `core.indices` from the **native analytical bands**, never
  from an RGB or false-colour composite.
* The band roles come from the same `IndexContext` Phase 11 uses; the app now
  passes **all** resolved roles so any supported index can be composed
  (previously only green and NIR were handed over).
* Each condition carries index, operator, threshold, threshold provenance,
  valid mask, source scene/date and ROI.

---

## 10. Temporal conditions

Only the four Phase 10 classes are composable: **increase, stable, decrease,
insufficient**. The engine calls `compare_ndvi` and adopts its raster and its
grid; it never re-classifies change and never substitutes a date.

A temporal composition carries the **before and after dates**, the threshold
convention used by Phase 10, the valid mask, the alignment record and the
provenance. `insufficient` stays UNKNOWN through the conjunction.

---

## 11. Explicitly not implemented

Temporal NDWI · flood or water change · land-cover change · crop-suitability
change · time series · anomaly detection · any new spectral index · NDRE, EVI,
SAVI, NDBI, NBR · any claim of causation.

"Did flooding happen?" still routes to `FLOOD_CHANGE` and is refused with
"…not available yet." "Compare NDWI before and after" still routes to
`TEMPORAL_NDWI` and is refused by name. Both refusals are re-checked in the
browser test.

---

## 12. The result contract

`MultiConditionResult` carries: status, query, normalized query, operator,
conditions, condition results (each with its mask, counts and provenance),
combined mask, unknown mask, matched / non-matching / unknown counts, analysed
cells, matched area and fraction, source analyses, source dates, the grid, the
alignment record, threshold provenance, index summaries, limitations, warnings,
provenance and runtime.

Outcomes are distinguishable from one another:

| Outcome | Result status | Execution status |
|---|---|---|
| matches found | `ok` | `OK` |
| evaluated, nothing matched | `zero_matches` | `OK` |
| nothing could be evaluated | `insufficient_data` | `OK` (message says so) |
| threshold missing | — | `NEEDS_THRESHOLD` |
| no two dates selected | — | `NEEDS_TWO_DATES` |
| condition not measured | — | `UNSUPPORTED_CONDITION` |
| no area selected | — | `NEEDS_ROI` |
| ROI outside the scene | — | `INSUFFICIENT_DATA` |
| ROI above the cell budget | — | `INSUFFICIENT_DATA` |

Attached evidence (form 6) is reported **over the matched cells** and labelled
"cells matching the combined condition — not a filter": it is a measurement of
the answer, never a second condition.

---

## 13. Router behaviour

The router stays deterministic. The composition guard runs after scoring and
before the intent is built: a request that scores for `SPATIAL_QUERY` **and**
carries a composable condition becomes `MULTI_CONDITION`; a `MULTI_CONDITION`
with no conditions falls back to `SPATIAL_QUERY`.

| Query | Intent |
|---|---|
| What is the NDVI here? | `NDVI_ROI_STATS` |
| What is the NDWI here? | `NDWI_ROI_STATS` |
| Compare NDVI before and after. | `NDVI_CHANGE_ROI` |
| Find cropland with NDVI greater than 0.6 | `MULTI_CONDITION` |
| Find cropland near water | `SPATIAL_QUERY` |
| Find cropland with high NDVI | `MULTI_CONDITION` (→ `NEEDS_THRESHOLD`) |
| Did flooding happen? | `FLOOD_CHANGE` (unavailable by design) |
| Compare NDWI before and after | `TEMPORAL_NDWI` (unavailable by design) |

`MULTI_CONDITION` is appended **last** in the registry so the suggestion list —
including "Can I grow cotton here?" — is unchanged.

---

## 14. UI behaviour

**Map.** A dedicated layer, *Composed conditions (MATCH / NO MATCH / UNKNOWN)*,
is **added** to the layer control; every pre-existing layer stays. MATCH is teal
so it cannot be confused with the Phase 9 green; NO MATCH and UNKNOWN reuse the
application-wide grey and amber, because grey and amber mean one thing here. The
legend names the expression, each threshold with its provenance, the dates, the
analysis grid, and how many cells were undecided.

**Panel.** Normalised query and operator; four numbers side by side — matching,
measured-not-matching, **undecided**, matched area; per-condition lines with
operator, threshold, provenance and per-condition counts; attached evidence in
its own expander; sources, dates and grid; limitations. The boundary statement
is printed above the fold, not hidden in a collapsed expander.

---

## 15. Real-data verification

`PYTHONPATH=. python scripts/verify_phase12_real_data.py`, scene
`s2_s2b-36ruv-20230806-0-l2a_2048px.tif` (2048 × 2048 @ 10 m, EPSG:32636),
ROI 5.12 km × 5.12 km = 26.21 km².

**Composition A — cropland AND NDVI > 0.6 AND NDWI < -0.4** (single date)

```
conditions   land cover class [40] (Cropland) AND NDVI > 0.6 AND NDWI < -0.4
             cropland  220,646 match /  41,498 no-match / 4,112 undecided
             NDVI>0.6  212,835 match /  49,309 no-match / 4,112 undecided
             NDWI<-0.4 233,377 match /  28,767 no-match / 4,112 undecided
matched      204,521 cells = 20.452 km²   (grid 516 x 516 @ 10 m, +2-cell rim)
undecided      4,112 cells — never counted as non-matches
alignment    {'grid_source': 'native_roi_grid'}
sources      ESA WorldCover 2021 v200, core.indices:ndvi, core.indices:ndwi
runtime      93–113 ms
```

Independent hand calculation from the raw digital numbers (float64, outside the
engine): NDVI > 0.6 in 212,846 cells, NDWI < -0.4 in 233,377 cells. The engine
agrees cell for cell **except** for 11 and 2 cells respectively — every one of
them sitting exactly on the threshold, where a float32 index and a float64
recomputation fall on opposite sides of a strict `>`. Zero cells disagree away
from the threshold, and zero matched cells violate either condition.

**Composition B — NDVI decrease near permanent water** (2023-01-18 → 2023-08-06)

```
conditions   within 1000 m of mapped permanent water (class 80)
             AND NDVI change class = decrease
             water proximity  30,494 match / 4,112 undecided
             NDVI decrease    30,771 match / 4,112 undecided
matched        4,511 cells = 0.451 km²
dates          2023-01-18 → 2023-08-06
alignment      identical_grid, resampled: false
runtime        ~400 ms (first call fetches WorldCover)
```

**Attached evidence.** "Find areas with NDVI decrease and NDWI statistics."
matched 30,771 cells; NDWI over the match: mean −0.5022, median −0.5333,
30,771 valid cells, labelled "cells matching the combined condition — not a
filter".

**The opt-in, on real data.** "Find cropland with high NDVI" is refused; with
`convention="ndvi_high"` the same question returns 204,521 cells, and the
threshold is recorded as `config_convention` / `convention:ndvi_high`.

**The refusals**, all reproduced on real data: missing threshold →
`NEEDS_THRESHOLD`; no dates → `NEEDS_TWO_DATES`; ROI outside the scene →
`INSUFFICIENT_DATA` ("does not overlap the loaded scene"); 20 km ROI →
`INSUFFICIENT_DATA` (4,008,004 cells against a 2,000,000 budget, refused rather
than coarsened).

---

## 16. Tests

`tests/test_phase12_multi_condition.py` — 68 tests, mirroring the brief:

| Group | Tests | Covers |
|---|---|---|
| Parsing and routing | 01–10 | the routing table; explicit and missing thresholds; opt-in conventions; relative wording; statistics as attached evidence; generic words that must not misroute |
| Mask logic | 11–22 | AND / OR / NOT over all three states; UNKNOWN ≠ FALSE; empty-match vs all-unknown; CRS, transform, shape, grid and ROI mismatches refused |
| Spectral | 23–27 | NDVI and NDWI masks against hand arithmetic; `>`/`≥`/`<`/`≤` boundaries; invalid cells stay UNKNOWN; threshold provenance |
| Temporal | 28–32 | increase / decrease / stable; `insufficient` stays UNKNOWN; dates and alignment recorded; missing pair refused |
| Spatial | 33–38 | cropland + index; negation ("but not water"); the expanded proximity grid; temporal ∧ spatial; masks identical to Phase 9's |
| Contract and resources | 39–53 | ROI-first grid; 2M-cell budget; windowed reads only; the convention opt-in; no causal claim; full regression of the registry, the router and the suggestion list |

Full suite: **688 passed** (620 before Phase 12 + 68 new). Two intent-census
tests were updated to include `MULTI_CONDITION` — reported and approved before
the change, with each invariant preserved.

---

## 17. Limitations and resource rules

* **ROI-first.** The composition grid is the ROI (plus the two-cell rim), never
  the scene: 26.21 km² analysed as 262,144 cells, not the scene's 4.19 M.
* **2,000,000-cell budget**, the same as Phase 10's. Exceeding it refuses; it
  does not coarsen, because coarsening changes the question.
* **Windowed reads.** Index rasters are read through a window; the browser and
  unit tests assert no unwindowed read ever happens.
* **One WorldCover fetch per composition**, cached, on the expanded grid when a
  proximity condition is present.
* **Layer order unchanged.** The composition layer is added; basemaps and every
  earlier layer behave exactly as before.
* **Boundaries, not causes.** Results are coincidences of measured conditions on
  one verified grid. Nothing in this phase assesses flooding, damage, crop
  failure, drought, deforestation, water availability, water quality or flood
  extent — and the interface says so every time it shows a number.
