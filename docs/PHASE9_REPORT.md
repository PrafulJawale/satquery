# Phase 9 — Multi-condition spatial query (FINAL REPORT)

**Status: COMPLETE.** Every gate passed: engine verified on real satellite data,
UI and map layer integrated, genuine browser interaction tested, full regression
green, Phase 8 still frozen.

Phase 9 turns a sentence like *"Find cropland near water"* into an auditable
spatial query — conditions → masks → three-valued logic → a result with a map
layer and an explanation. It is deterministic orchestration over already
verified engines, never a new model.

> **Phase 9 does not implement flood detection, irrigation detection,
> groundwater detection, LLM query planning, or VLM/GeoChat reasoning.**

---

## 1. Objective

Let a user combine **mappable** conditions in natural language and get a
geographic answer with evidence, while refusing — loudly — anything the system
does not measure.

* cotton suitability (from the frozen Phase 8 engine)
* land-cover classes (cropland, built-up) from ESA WorldCover
* permanent water (WorldCover class 80)
* proximity to permanent water (metres, projected CRS)

combined with AND / OR / NOT over **three-valued** masks, so that missing data
survives to the answer instead of being flattened into "false".

## 2. Architecture

```
sentence
   │
   ├─ core/router.py            intent + intent-level blocking (irrigation…)
   │        │
   │        └─ core/spatial_query.py   the STRUCTURED QUERY (models + parser,
   │                                   no raster code, no Streamlit)
   │
   ├─ analyses/registry.py      intent  →  engine  (the only binding place)
   │
   └─ analyses/spatial_query.py the ORCHESTRATOR
            │
            ├─ analyses/crop_suitability.py   Phase 8 engine, unchanged
            ├─ core/datasources/worldcover.py existing datasource, unchanged
            └─ core/spatial.py                masks, grids, EDT proximity,
                                              three-valued logic
            │
            └─ SpatialQueryResult  →  ui/components.py + ui/map.py
```

| File | Role |
|---|---|
| `core/spatial_query.py` | `SpatialQuery` / `SpatialCondition` models + deterministic parser; config-driven |
| `core/spatial.py` | `Grid`, `GridMask`, compatibility checks, `crop_mask`, water / land-cover / proximity masks, three-valued AND/OR/NOT |
| `analyses/spatial_query.py` | orchestrator + `SpatialQueryResult` + engine-level refusal paths |
| `config/spatial/conditions.yml` | thresholds, classes, distances, clamps |
| `config/spatial/patterns.yml` | vocabulary: condition synonyms, operators, unsupported topics |
| `analyses/registry.py` | one row binds `SPATIAL_QUERY` to the orchestrator |
| `ui/components.py` | result panel: interpretation, summary, why, methodology, limitations |
| `ui/map.py` | categorical palette + legend for the result layer |

**Nothing was duplicated or rewritten:** the cotton condition calls the Phase 8
engine; the land-cover conditions use the existing WorldCover datasource.

## 3. Query grammar

Not a general-purpose parser — a small, auditable grammar:

```
[selection verb] <condition> [and|or|but not] <condition> ...
```

* **Operators**: `and` / `plus` / `together with` (AND, the default),
  `or` (OR), `but not` / `excluding` / `outside` / `not water` (NOT).
* **Distances**: `near water` (default 1 000 m), `within 500 m of water`,
  `within 1 km`; clamped to 30–10 000 m and reported when clamped.
* **Refusals instead of guesses**: an unattached negation, or mixed AND/OR in
  one sentence, yields `NEEDS_CLARIFICATION` — never a guess.
* Keyword matching is word-bounded, so *"suitable **for** cotton"* is not read
  as the OR operator.

Examples (verified parses):

| Query | Structured result |
|---|---|
| `Find cropland near water` | `land_cover([40]) AND water_proximity(<= 1000 m, class 80)` |
| `Find cropland within 500 m of water` | `land_cover([40]) AND water_proximity(<= 500 m, class 80)` |
| `Find cropland excluding water` | `land_cover([40]) AND NOT water(class [80])` |
| `Find cotton areas near water but not built-up` | `cotton(class >= 3, rainfed) AND water_proximity(<= 1000 m) AND NOT land_cover([50])` |
| `Find cropland or cotton areas` | `crop_suitability(…) OR land_cover([40])` |
| `Can I grow cotton near water?` | same conditions + an explicit note that water ≠ irrigation |

## 4. Supported conditions

| Condition | Meaning | Source |
|---|---|---|
| `CROP_SUITABILITY` | Phase 8 class **≥ 3**, scenario **rainfed** | Phase 8 engine (unchanged) |
| `LAND_COVER_CLASS` | membership of a WorldCover class (40 cropland, 50 built-up) | WorldCover 2021 v200 |
| `WATER` | the cell **is** mapped permanent water (class 80) | WorldCover 2021 v200 |
| `WATER_PROXIMITY` | cell centre within **N metres** of class 80 | derived by EDT |

## 5. Unsupported conditions

`irrigation` · `groundwater` · `flood / flood-prone` · `salinity`

Each produces a structured `UNSUPPORTED` condition that **aborts the query**:
no dataset is read, no partial answer is returned, and the reply names the
supported alternative *as a different quantity*. Two paths feed the same
refusal:

* the spatial parser (selection-form queries), and
* `QueryIntent.blocked_by` → `Status.UNSUPPORTED_CONDITION` **before any engine
  runs**, so *"Can I grow cotton with irrigation?"* can never be answered with a
  rainfed verdict.

Flood deliberately keeps its own `FLOOD_CHANGE` intent and its existing Phase 7
message; it is not a verdict blocker.

## 6. Spatial semantics

* Everything is computed on **one analysis grid** (30 m, EPSG:32636 for these
  AOIs). `Grid.require_compatible()` checks CRS, transform, dimensions, pixel
  width, pixel height and array shape; a mismatch raises rather than combining.
* Distances are **metres** in the projected CRS — never degrees, never a pixel
  count. A geographic CRS or non-square cells raise `DistanceGridError`.
* Area = `|a|·|e|` from the transform (**900 m²** at 30 m) — never an assumed
  10 × 10 m.
* Counting happens **inside the ROI**; the window buffer is excluded from all
  three counters, so it is never mistaken for missing data.

## 7. Three-valued mask logic

Every mask is (`match`, `valid`); `INSUFFICIENT` is `not valid` and is never
collapsed to FALSE. Strong Kleene semantics, unit-tested case by case:

| AND | TRUE | FALSE | NA | | OR | TRUE | FALSE | NA | | NOT | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRUE | TRUE | FALSE | NA | | TRUE | TRUE | TRUE | TRUE | | TRUE | → FALSE |
| FALSE | FALSE | FALSE | **FALSE** | | FALSE | TRUE | FALSE | **NA** | | FALSE | → TRUE |
| NA | NA | FALSE | NA | | NA | TRUE | NA | NA | | NA | → NA |

`FALSE AND NA = FALSE` (one failing condition is decisive) and
`FALSE OR NA = NA` (the unknown could still match).

On real data this is load-bearing: on AOI-2 *cotton AND near water* returns
**800 insufficient** cells — `INSUFFICIENT AND TRUE = INSUFFICIENT` — instead of
reporting them as non-matches.

## 8. Water definition

`WATER` = **WorldCover class 80**, the cell itself, 2021 epoch.

* TRUE = class 80 · FALSE = any other valid class · NA = nodata.
* **Wetland (class 90) is not water** — 10.66 % (AOI-1) and 12.55 % (AOI-2) of
  cells are class 90 and are excluded by design.
* It is not flooding, not irrigation, not groundwater, not soil moisture.

## 9. Water-proximity definition

`WATER_PROXIMITY(≤ N m)`: distance from the **cell centre** to the nearest
class-80 cell, computed with `scipy.ndimage.distance_transform_edt` using
`sampling=(|e|, |a|)` from the transform, so the output is metres.

**The window-edge rule (conservative, by decision):** a cell may be called
*"farther than N m from water"* only when the source window is large enough to
establish it — the nearest unknown cell or window edge must also be farther than
N. Otherwise the cell is `INSUFFICIENT_DATA`. **"No water in the window" is
never treated as proof of "far from water."**

## 10. Buffered-window strategy

Phase 8's grid carries only a 2-cell (60 m) buffer, which would manufacture an
insufficient-data rim for a 1 000 m proximity query. Decision:

* Phase 8 is **not** touched — no buffer argument, no grid change.
* Phase 9 reads WorldCover on a **pixel-aligned buffered window** of
  `ceil(distance / resolution) + 2` cells per side (**36 cells = 1 080 m**),
  computes proximity there, and **crops** back to the analysis grid.
* `crop_mask` slices, never resamples: it refuses CRS changes, cell-size
  changes, half-cell offsets and targets that are not contained.
* The window is capped at 2 000 000 cells; if the cap ever bites, the engine
  says so and falls back to the conservative rule.
* Effect: AOI-1 105→177→105, AOI-2 687→759→687; **0 cells** withheld on either
  AOI (a synthetic test pins this: the rim that was 45 % insufficient is now 0).

## 11. AOI-1 verification (3 × 3 km, 10 000 ROI cells)

Zero-water regression anchor. Source: WorldCover tile `N30E030`, 10 m, v200,
nearest-neighbour, **windowed read from cache**.

| class | cells | share |
|---|---:|---:|
| 40 Cropland | 9 210 | 83.54 % |
| 50 Built-up | 570 | 5.17 % |
| 90 Wetland | 1 175 | 10.66 % |
| **80 Permanent water** | **0** | **0.00 %** |

| case | matched | insufficient | share |
|---|---:|---:|---:|
| cotton suitability alone | 0 | 0 | 0.00 % |
| permanent water alone | 0 | 0 | 0.00 % |
| near permanent water (1 000 m) | **1 047** | 0 | 10.47 % |
| cotton AND near permanent water | 0 | 0 | 0.00 % |
| cropland AND near permanent water | **884** | 0 | 8.84 % |
| cropland AND NOT permanent water | **8 335** | 0 | 83.35 % |
| cropland AND water (impossible) | 0 | 0 | 0.00 % |
| irrigation / flood | refused — nothing computed | | |

AOI-1 contains **no** mapped permanent water; the 1 047 near-water cells are
near water mapped **just outside** the 3 km box but inside the 1 080 m buffered
window (0 water cells in the 105 × 105 window vs 72 in the 171 × 171 window) —
the buffered read working as intended.

## 12. AOI-2 verification (687 × 687 @ 30 m, 466 489 ROI cells)

| class | cells | share |
|---|---:|---:|
| 40 Cropland | 326 369 | 69.15 % |
| 50 Built-up | 47 522 | 10.07 % |
| 80 Permanent water | **33 329** | 7.06 % |
| 90 Wetland | 59 219 | 12.55 % |

| case | matched | area | insufficient | share |
|---|---:|---:|---:|---:|
| cotton suitability alone | 0 | — | **1 589** | 0.00 % |
| permanent water alone | **32 731** | 29.458 km² | 0 | 7.02 % |
| near permanent water (1 000 m) | **165 809** | 149.228 km² | 0 | 35.54 % |
| cotton AND near permanent water | 0 | — | **800** | 0.00 % |
| cropland AND near permanent water | **78 450** | 70.605 km² | 0 | 16.82 % |
| cropland AND NOT permanent water | **322 951** | 290.656 km² | 0 | 69.23 % |
| cropland AND water (impossible) | 0 | — | 0 | 0.00 % |
| irrigation / flood | refused — nothing computed | | | |

**Zero matches here is the correct scientific outcome**, not a failure: Phase 8's
verified rainfed verdict for this sample is **Unsuitable** (score 0.66), so no
cell reaches class ≥ 3. No AOI was altered to manufacture a positive result.

## 13. Independent verification

The engine was not checked against its own intermediate outputs.

| method | result |
|---|---|
| **Proximity, brute force** — nearest-water distance by explicit coordinate arithmetic (`min √(dx²+dy²)`), water search box expanded by `distance + window` cells so nothing nearer is missed; no EDT involved | **AOI-1: 6 400 / 6 400 = 100.0000 %** agreement · **AOI-2: 6 400 / 6 400 = 100.0000 %** |
| distance on water pixels | **0.000 m** on both AOIs |
| threshold boundary | just inside 1 000 m → engine TRUE on **80/80** (AOI-1) and **160/160** (AOI-2); just outside → **0** claimed TRUE |
| units | distances in metres (0–2 100 m / 0–1 236 m sampled); walking *k* cells east of water never exceeds `k × 30 m` (**0 violations**) |
| **Combinations recomputed in NumPy** from the raw arrays (replicating the buffered read + crop for the cotton case) | `cotton AND near-water`, `cropland AND near-water`, `cropland AND NOT water` — **cell-for-cell identical** on both AOIs |
| **Cotton recomputed** from the Phase 8 class raster as `class ≥ 3` | **identical** on 11 025 (AOI-1) and 471 969 (AOI-2) cells; insufficient 1 025 / 7 069 match exactly |
| **Phase 8 agreement** | `class_fractions` AOI-1 `Unsuitable 1.0`; AOI-2 `Insufficient 0.00341 / Unsuitable 0.99659` (≈1 589 cells) — Phase 8 thresholds, weights, formulas and sources unchanged |
| **JRC GSW cross-check** (verification only) | WorldCover water 7.06 % vs GSW "any water 1984–2021" 13.89 %; **both 32 748**, WorldCover-only 581, GSW-only 32 791 — the expected direction, explained by ephemeral water, never reconciled and never used as the engine |
| **Semantic regression** | `NOT WATER` ≠ `NOT WATER_PROXIMITY` on real data: 8 335 vs 1 243 cells (AOI-1), 322 951 vs 204 735 (AOI-2) |

## 14. UI behaviour

`Ask SatQuery` renders a spatial-query result as five blocks:

1. **Query** — the sentence as typed.
2. **Interpreted as** — each understood condition with the numbers that make it
   checkable, e.g.
   ```
   ✓ Cropland — WorldCover class 40
   ✓ Near permanent water — within 1,000 m
   Operator: AND — every condition must hold.
   ```
   and for cotton:
   ```
   ✓ Cotton suitability — Phase 8 class ≥ 3
     Scenario: rainfed (data-backed), crop: cotton
   ✓ Near permanent water — within 1,000 m
   ```
3. **Result** — one of four explicit states:
   * **MATCH** — conditions satisfied somewhere;
   * **NO MATCH** — every analysed cell was decided and none matched;
   * **INSUFFICIENT DATA** — source coverage prevented a conclusion;
   * **UNSUPPORTED** — the system does not measure the requested condition.
4. **Summary** — matched cells, matched area (km²), matched fraction, analysis
   resolution, and the proximity distance where one was used. The wording is
   always *"Within the selected ROI, X % of analysed cells satisfy all requested
   conditions"* — never *"no suitable land exists"*.
5. **Why / evidence · Data & methodology · Limitations** — collapsible, so the
   main panel stays readable.

## 15. Browser verification

Genuine Playwright interaction against the running app: the page is driven with
mouse and keyboard, the ROI is drawn with real mouse events and coordinates come
from Leaflet (never guessed). Log: `artifacts/phase9_browser_log.txt`;
screenshots: `artifacts/phase9_browser_*.png`.

| scenario | checks |
|---|---|
| **Q1** `Find cropland near water` | submitted · interpretation shown (cropland class 40, near-water 1 000 m, operator AND) · result metrics shown (matched cells / area / fraction) · **map layer appears** in the layer control · **legend** shows all three classes · methodology (WorldCover, nearest-neighbour, resolution) and limitations (not irrigation / not groundwater / not flood risk / wetland ≠ water) open |
| **Q2** `Find areas suitable for cotton near water` | cotton condition shown with **class ≥ 3** and **rainfed** · near-water shown · result reports **zero matches** · presented as **NO MATCH**, never as an error · **no false insufficient-data claim** · never says "no suitable land exists" |
| **Q3** `Find cotton land with reliable irrigation` | **UNSUPPORTED** panel · names irrigation · states nothing was computed and no proxy substituted · **no partial spatial result** · names water proximity as a *different* quantity |
| **Q4** `What is the NDVI of this area?` | the Phase 7 NDVI workflow still runs end to end |
| **map + legend** | the layer control lists *Spatial query result* · the legend names **MATCH / NO MATCH / INSUFFICIENT DATA** · "no match … is a result, not a failure" · "insufficient data … never counted as a non-match" |

Result: **PASS — 86 checks over 5 isolated browser sessions, 0 failed**
(29/29 cropland near water · 18/18 cotton near water · 14/14 unsupported
irrigation · 14/14 map + legend · 11/11 NDVI regression).

Each session ran with its **own freshly started Streamlit server and its own
browser**, then tore both down. The sandbox has ≈2 GB of RAM; running several
scenarios in one session let the kernel OOM-kill the app mid-run, which produced
stale readings that looked like failures. The isolation is what makes these
numbers trustworthy: every count below was produced by the query under test,
and the cotton scenario additionally asserts that the metrics on screen differ
from the snapshot taken *before* the question was asked.

## 16. Regression results

| suite | result |
|---|---|
| `pytest tests -q` (Phases 1–9) | **506 passed, 0 failed** |
| `python scripts/verify_phase8.py` | **15 / 15** — Phase 8 unchanged |
| `python scripts/verify_phase9.py` | **10/10 cases per AOI**, independent checks 100 % agreement, plus a 9-query gallery per AOI (§ below) |
| `python scripts/browser_test_phase9.py` | **86 checks, 0 failed** across 5 isolated sessions (29/29, 18/18, 14/14, 14/14, 11/11) |
| `python scripts/browser_test_phase8.py` | **44 / 44** — Phase 8 UI unaffected by the new panel |
| `python scripts/verify_basemap.py` | **20 / 20** — every base provider loads, offline option is clean, a blocked provider degrades to grey placeholders and recovers |
| Phase 6 outside-raster scenario (verbatim) | **8 / 8** — "does not overlap the raster", no statistics, no zero-filled values |

### Representative queries (real data, `verify_phase9.py` gallery)

Each query was parsed and executed end to end on both AOIs. AOI-1 is the
3 × 3 km ROI (10 000 cells, **no class-80 water in the source**); AOI-2 is the
wider 687 × 687 grid (466 489 cells, 33 329 class-80 cells).

| query | conditions parsed | AOI-1 matched / insufficient | AOI-2 matched / insufficient |
|---|---|---|---|
| Find cropland near water | land cover + water proximity | 884 / 0 | 78 450 / 0 |
| Find areas suitable for cotton near water | crop suitability + water proximity | 0 / 0 | 0 / **800** |
| Find cropland excluding water | land cover + **NOT water** | 8 335 / 0 | 322 951 / 0 |
| Find permanent water | water | 0 / 0 | 32 731 / 0 |
| Find built-up areas near water | water proximity + land cover | 35 / 0 | 19 103 / 0 |
| Find cropland but not permanent water | land cover + **NOT water** | 8 335 / 0 | 322 951 / 0 |
| Where are the suitable agricultural areas near water? | land cover + water proximity | 884 / 0 | 78 450 / 0 |
| Find cotton land with reliable irrigation | **unsupported** | refused, nothing computed | refused, nothing computed |
| Find areas that are not flood-prone | **unsupported** | refused, nothing computed | refused, nothing computed |

Three properties are visible in that table and are the point of Phase 9:
"excluding water" binds to the **water class**, not to proximity (8 335 = every
cropland cell in AOI-1, because AOI-1 has no water at all); cotton near water
reports **zero matches with 800 insufficient cells counted separately** on
AOI-2, so a zero is never confused with missing data; and the two unsupported
queries are refused without any computation and without a proxy.

One Phase 7 test was updated, not bent: `test_planned_intents_have_no_handler`
asserted the set of *available* specs; Phase 9 graduated `SPATIAL_QUERY` into it,
exactly as Phase 8 did for cotton. One Phase 8 test moved a single query
(*"Where can I grow cotton in this region?"*) from `CROP_SUITABILITY` to
`SPATIAL_QUERY` — the approved routing semantics — and pinned the cotton
condition to `crop=cotton, min_class=3, scenario=rainfed`. No Phase 8 behaviour,
threshold or weight changed.

## 17. Performance

| | |
|---|---|
| ROI-first | every read is windowed on the analysis grid or its buffer |
| Global raster downloads | **none** |
| Caching | existing caches reused (WorldCover, SoilGrids, DEM, WorldClim); a warm verification run takes **5.2 s** for both AOIs and all cases |
| Cold Phase 8 cotton run, AOI-2 (472 k cells) | 30.1 s |
| Peak memory | tracemalloc **83.1 MB**, process RSS **202.5 MB** |
| Duplicate work | one WorldCover read per run serves every land-cover condition; the class-80 mask is built once and reused by every proximity condition |
| Copies | largest intermediate ≈ 759 × 759 (2.3 MB); the web-mercator copy is display-only and cached |
| Discipline | pytest and the browser test were not run concurrently |

## 18. Provenance

Every result carries per-condition provenance: dataset, version, tile URL,
native resolution, licencing, temporal basis, resampling method, and what the
layer was used for.

* **Cotton** — Phase 8 rainfed screening (config `config/crops/cotton.yml`);
  native: land cover 10 m, soil 250 m, climate ~1 km, topography 30 m.
* **Land cover / water** — ESA WorldCover 10 m 2021 **v200**, tile `N30E030`,
  CC BY 4.0 (doi 10.5281/zenodo.7254221), nearest-neighbour (categorical).
* **Water proximity** — derived from class 80 with
  `scipy.ndimage.distance_transform_edt`, metres, projected CRS, buffered
  window; recorded per result (`distance_m`, `pixel_size_m`, `water_cells_found`,
  `undecidable_cells`, `edge_rule`).
* **JRC GSW** — cross-check only, never the production engine.

The report never claims resampling creates information: each result states
*"Analysed at 30 m; inputs are native 10–1000 m, so finer detail than the
analysis grid is not resolved."*

## 19. Limitations

* Permanent-water proximity is **not** irrigation availability.
* Permanent-water proximity is **not** groundwater access.
* Permanent-water proximity is **not** a flood-risk assessment.
* Wetland (class 90) is not treated as permanent water.
* Cotton suitability is the Phase 8 **rainfed** scenario; irrigation is not
  modelled and salinity is not assessed.
* The current sample AOIs contain no cotton cells reaching class ≥ 3, so cotton
  queries here return zero matches.
* Zero matches do not imply that no such land exists outside the selected ROI.
* External raster coverage and nodata can produce insufficient-data regions.
* Water semantics come from WorldCover class 80 only.
* JRC GSW is an independent cross-check, not the production water engine.
* WorldCover thematic accuracy is ~75 % globally; class 40 means *any* crop in
  2021 — not cotton, not currently cultivated; v200 must not be mixed with v100.
* The 30 m grid can miss water features narrower than 30 m.
* Distances are between cell centres (a cell beside water is 30 m away, not 0).

## 20. Future extensions (out of scope for Phase 9)

* Flood / change detection as its own analysis (never inferred from water
  proximity).
* Irrigation and groundwater as measured datasets, when sources exist.
* Additional crops and additional land-cover classes.
* Area-per-class and per-condition map toggles; exporting the result mask.
* Persistent result history beyond the last five entries.

Explicitly **not** in Phase 9: new crop types, flood analysis, vegetation
change, irrigation or groundwater inference, yield prediction, LLM, VLM,
GeoChat, autonomous query planning, live satellite feeds, predictive claims, or
any arbitrary proxy for an unmeasured quantity.

---

## Final evidence

See the closing summary delivered with this report for test counts, browser
counts, the Phase 8 gate, AOI numbers, changed files, server status and the
scientific-integrity confirmation.
