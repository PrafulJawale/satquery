# Phase 9 — Multi-Condition Spatial Query: DESIGN (Checkpoint A)

**Status: DESIGN ONLY — no implementation yet. Nothing in `core/`, `analyses/`, `ui/` or
`app.py` has been changed for Phase 9.**

This document answers the ten design questions, records the research that produced them,
and ends with the decisions I need approved before writing code.

---

## 0. Evidence gathered before designing (measured, not assumed)

| Probe | Result | Consequence for the design |
|---|---|---|
| `core/router.py` | `Intent` enum with `NDVI_ROI_STATS` + `CROP_SUITABILITY` executable, `FLOOD_CHANGE`/`VEGETATION_CHANGE` planned; `QueryIntent` is a dataclass carrying `intent, confidence, matched, explanation, required_context`; matching is a data-driven `_PATTERN_TABLE` (phrases / tokens / negative) | a new intent is added the same way; the **condition grammar lives in its own module**, only the "this is a spatial query" decision is added to the router |
| `analyses/registry.py` | intent → handler + `REQUIRED_CONTEXT` + examples | Phase 9 is one new registry row; no router rewrite |
| `analyses/crop_suitability.py` | `CropSuitabilityScreening` exposes `grid`, `scenarios{…}.suitability_raster` (class codes), `raster_transform`, `raster_crs`, `land_cover`, `performance`, provenance | the cotton condition reuses the engine and **adopts its grid** — Phase 8 is not modified |
| `core/datasources/worldcover.py` | `fetch_land_cover(grid) -> LayerResult`, `CLASS_NAMES` (40 cropland, 50 built-up, 80 permanent water, 90 herbaceous wetland…), `EXCLUDED_BY_DEFAULT = (50, 80, 95, 70, 100, 90)` | cropland / water / built-up conditions reuse the **same cached layer** (cache identity already includes AOI signature + resolution + resampling + band) |
| `core/alignment.py` | `make_grid()` (snapped, metric, cell-capped), `read_into_grid()` (typed resampling), `roi_mask()` | one grid per query; ROI-only denominators |
| `scipy` | **1.17.1 installed** but *not declared* in `requirements.txt` | use `scipy.ndimage.distance_transform_edt`; **add `scipy>=1.11` to requirements.txt** |
| JRC Global Surface Water (GSW) | tile `occurrence_30E_40Nv1_4_2021.tif` (bounds 30–40 E, 30–40 N, 40000², EPSG:4326, 0.00025°, 256² blocks, no overviews) opens in **0.7 s** and a 240×240 windowed read takes **0.3 s** over HTTP. The earlier “zero-size window” failure was the **wrong tile** (`…30N…` covers 20–30 N) | GSW is technically usable, but it is **not needed** for the engine — see §5 |
| GSW over the Nile Delta ROI | max occurrence **13 %**, **0 cells ≥ 50 %** | corroborates the WorldCover finding below: this ROI has no mapped permanent water |
| WorldCover over the **Phase-8 3×3 km ROI** (105×105 @30 m = 11,025 cells) | cropland 9,210 (83.5 %), herbaceous wetland 1,175 (10.7 %), built-up 570 (5.2 %), **water (80) = 0** | the standard ROI yields a **legitimate zero-match** water condition — a required test case, but a useless demo |
| WorldCover over the **full bundled sample extent** (687×687 @30 m = 471,969 cells, under the 2 M cap) | water **7.06 %** (33,329 cells), built-up 10.07 %, cropland 69.15 %, wetland 12.55 % | **AOI-2** makes the proximity condition non-trivial and is the demo/verification AOI |

---

## 1. What constitutes a multi-condition query?

A query is a **spatial query** when it asks the system to *select geography* inside the drawn
ROI by combining **two or more conditions**, or **one condition plus a spatial predicate**
(proximity / containment / exclusion).

| Form | Example | Routed to |
|---|---|---|
| Yes/no screening verdict for the ROI | “Can I grow cotton here?” | **CROP_SUITABILITY** (Phase 8, unchanged) |
| Geographic selection | “Find areas suitable for cotton near water.” | **SPATIAL_QUERY** (Phase 9) |
| Geographic selection, single condition | “Find areas suitable for cotton.” | **SPATIAL_QUERY** with one condition (reuses the engine, adds the map layer + matched area) |

The distinction is **question form**, not vocabulary: *can/is/should* → screening verdict;
*find/show/where/which areas* → selection. This keeps the Phase 8 browser test (44/44) and
its answer format untouched, and gives Phase 9 one code path for 1..n conditions.

## 2. How is the query decomposed?

```
natural language
  → core/router.py            normalise, score, decide INTENT = SPATIAL_QUERY
  → core/spatial_query.py     parse_conditions()  →  SpatialQuery
                              (conditions, operator, distance, status, clarifications)
  → analyses/registry.py      SPATIAL_QUERY → analyses/spatial_query.py
  → engines                   crop suitability (Phase 8, unmodified) + WorldCover layers
  → core/spatial.py           alignment check → boolean/three-valued combination
  → SpatialQueryResult        counts, area, masks, provenance, timings
```

**The router is not turned into a bag of special cases.** It gains one intent, one pattern
entry and one optional field (`conditions`) on `QueryIntent`; the condition grammar, its
synonyms, distances and class codes live in `core/spatial_query.py` +
`config/spatial/*.yml`, exactly as thresholds live in `config/crops/cotton.yml` today.

**No free-form strings survive parsing.** Everything after parsing is:

```python
@dataclass(frozen=True)
class Condition:
    condition_type: str        # crop_suitability | land_cover_class | water_proximity | …
    parameters: Dict[str, Any] # {"crop": "cotton", "min_class": 3} / {"classes": [40]} / {"distance_m": 1000}
    required_analysis: str     # "crop_suitability" | "worldcover"
    status: str                # SUPPORTED | UNSUPPORTED_RELATED | UNSUPPORTED | AMBIGUOUS
    negate: bool = False

@dataclass
class SpatialQuery:
    original_query: str
    operator: str              # AND | OR  (NOT is a per-condition `negate` flag)
    conditions: Tuple[Condition, ...]
    status: str                # OK | NEEDS_CLARIFICATION | UNSUPPORTED_CONDITION | AMBIGUOUS
    clarification: str = ""
    interpretation_notes: Tuple[str, ...] = ()
```

## 3. Which existing analysis engines are reused?

| Condition | Reuses | How |
|---|---|---|
| `crop_suitability(crop=cotton)` | **Phase 8 engine, verbatim** (`run_crop_suitability` via the registry) | class raster from `scenarios["rainfed"]`; **no duplication, no modification** |
| `land_cover_class(40 cropland)` | `core/datasources/worldcover.fetch_land_cover(grid)` | nearest-neighbour on the query grid |
| `water_proximity(≤ d m)` | same WorldCover layer, class **80** | distance transform in the projected CRS |
| `land_cover_class(50 built-up)` (negated) | same WorldCover layer | one fetch serves every land-cover condition (memoised per run) |

**Cache reuse is automatic:** Phase 8 already cached WorldCover for AOI-1 with identity
`{dataset, version, variable, source_url, aoi_signature, resolution, resampling, band}`.
A Phase 9 query on the same grid therefore performs **zero new network fetches** for land
cover, and reuses the cached soil/climate subsets for the suitability condition.

## 4. How are raster masks combined?

Three-valued logic. Every condition returns a `GridMask`:

```python
@dataclass
class GridMask:
    match: np.ndarray   # bool  — condition satisfied
    valid: np.ndarray   # bool  — the condition could be evaluated here
    grid: AnalysisGrid  # CRS + transform + shape, always carried along
    condition_id: str
    native_resolution: str
```

Combination (documented, deterministic, never inferred):

| Operator | match | valid |
|---|---|---|
| `A AND B` | `A.match & B.match` | `A.valid & B.valid` |
| `A OR B` | `A.match \| B.match` | `A.valid & B.valid` |
| `NOT B` | `~B.match` | `B.valid` |

A cell is **MATCH**, **NO MATCH** or **INSUFFICIENT DATA** (= not `valid`); insufficient
cells are excluded from both numerator and denominator and reported separately.
Mixed `AND`+`OR` in one sentence is **not** resolved by guessing precedence — it returns a
clarification state.

## 5. How is “near water” represented?

> **Definition (stated verbatim in the UI):** “near water” = *within a configurable distance
> of mapped **permanent surface-water pixels** (ESA WorldCover 2021 v200, class 80, 10 m
> native), measured on the projected analysis grid.*
>
> It does **not** mean irrigation available, groundwater available, canal access, or reliable
> water supply. It is proximity to mapped surface water only.

**Source decision — WorldCover class 80, not JRC GSW.**

* WorldCover is already integrated, verified (Phase 8, 15/15), cached, and at **10 m**
  native — finer than GSW’s 30 m.
* GSW works with the correct tile (measured: 0.7 s open + 0.3 s windowed read), but it
  answers a *different* question (“how often was water present 1984–2021”), introduces an
  occurrence threshold to defend, and adds a dataset/licence/version to track.
* **Proposed:** GSW (occurrence ≥ 50 %) is used **only in verification** (§16) as an
  *independent* cross-check of the water mask, never in the engine. If it misbehaves in the
  sandbox during implementation, verification falls back to a shapely-buffer cross-check.

**Distance pipeline** (never in degrees):

```
water mask (bool, on the query grid)
→ scipy.ndimage.distance_transform_edt(~water) * pixel_size   # metres, from the affine
→ distance <= water_proximity_m                                # configurable
→ boolean proximity mask
```

* `water_proximity_m: 1000` lives in **`config/spatial/conditions.yml`**, overridable by
  “within 1 km of water” / “within 500 m”; a configurable maximum clamps absurd values and
  says so. **The final answer always states the distance actually used.**
* Convention: distance is measured **cell-centre to nearest water cell-centre**, so the
  boundary is accurate to about half a cell diagonal (≈21 m at 30 m) — documented, not hidden.
* Water cells themselves are at distance 0 and therefore **do** satisfy “near water”; they
  are independently excluded by the suitability condition (Phase 8 hard constraint), so the
  combination still selects *dry* land near water. Documented and unit-tested.
* Class 90 (herbaceous wetland) is **not** water by default; it is an opt-in list in the
  config, off, and labelled as such.

### 5b. Two different water conditions (CORRECTION 1 — they are not interchangeable)

| Condition | Means | Triggered by | Example |
|---|---|---|---|
| `WATER(class 80)` | **the cell itself** is mapped permanent water | the word "water" **without** a proximity cue | "excluding water" → `NOT water(class [80])` |
| `WATER_PROXIMITY(≤ N m)` | the cell is **within N metres** of such a cell | a **proximity cue** + water | "near water", "within 500 m of water" → `water_proximity(<= 500 m, class 80)` |

Proximity cues (`config/spatial/patterns.yml`): *near, within, close to, next to,
adjacent to, distance to, closer/nearer than*. A bare "water" mention therefore
**cannot** become a distance condition, and:

* `NOT WATER` (the cell is not mapped as water) **≠** `NOT WATER_PROXIMITY`
  (the cell is farther than the threshold from water).
* Both are covered by an explicit semantic-regression test.
* `WETLAND (class 90)` remains a separate class and is never water.

## 6. What happens when a requested condition is unsupported?

An **unsupported condition aborts the computation** — the system never returns a partial
result that the user could read as the answer to what they asked.

| Requested | Status | Behaviour |
|---|---|---|
| “reliable irrigation”, “irrigated land”, “enough water”, “canal access” | `UNSUPPORTED_RELATED` (nearest supported: water proximity) | *“I can evaluate proximity to mapped surface water, but reliable irrigation availability is not currently measured.”* — **no computation** |
| “outside flood-prone areas”, “flood risk” | `UNSUPPORTED` (FLOOD_CHANGE is planned) | *“Flood analysis is not available yet.”* — **never** approximated by permanent water |
| groundwater, salinity, soil depth, drainage, yield | `UNSUPPORTED` | named as not measured, nothing computed |

The clarification message always names the supported alternative without performing it.

**Correction 2 — unsupported requirements also block *verdict* questions.**
An explicit irrigation/groundwater/salinity requirement is detected **before any
engine runs** (router field `QueryIntent.blocked_by` → registry status
`UNSUPPORTED_CONDITION`), so

> "Can I grow cotton with irrigation?" · "Is this suitable for irrigated cotton?"

return *"Irrigation availability and reliability are not measured by this system.
The cotton screening that IS available evaluates the configured RAINFED
scenario…"* instead of a silent rainfed verdict. Selection-form queries
("Find cotton areas with irrigation") take exactly the same path, and
`irrigation` is never rewritten as `near water`. Flood is deliberately **not** a
verdict blocker: it keeps its own `FLOOD_CHANGE` intent and its existing Phase 7
message.

### Checkpoint C -- implementation notes

**Module layout**

| File | Role |
|---|---|
| `core/spatial.py` | three-valued mask algebra, grid compatibility, water + proximity masks (no I/O, no UI) |
| `analyses/spatial_query.py` | the orchestrator: structured query → masks → `SpatialQueryResult` |
| `analyses/registry.py` | one new row binds `Intent.SPATIAL_QUERY` to the orchestrator |

**Three-valued logic** (`core.spatial`, strong Kleene). Every mask carries
(`match`, `valid`); `INSUFFICIENT` is `not valid` and is never collapsed into
FALSE:

| AND | TRUE | FALSE | NA | | OR | TRUE | FALSE | NA | | NOT | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TRUE** | TRUE | FALSE | NA | | **TRUE** | TRUE | TRUE | TRUE | | TRUE | → | FALSE |
| **FALSE** | FALSE | FALSE | FALSE | | **FALSE** | TRUE | FALSE | NA | | FALSE | → | TRUE |
| **NA** | NA | FALSE | NA | | **NA** | TRUE | NA | NA | | NA | → | NA |

`FALSE AND NA = FALSE` (one failing condition is decisive) and
`FALSE OR NA = NA` (the unknown could still match) are both unit-tested.

**Distances are metres.** `scipy.ndimage.distance_transform_edt` runs on the
projected analysis grid with `sampling=(|e|, |a|)` taken from the transform, so
the output is metres for the true cell size. It refuses, loudly:
* a geographic CRS (`DistanceGridError` -- degrees are not metres);
* non-square cells (one scalar distance would be wrong in one direction);
* a negative distance.

**The window-edge rule.** Distance is only decidable inside the window. A cell
more than N metres from mapped water is reported **FALSE only when the nearest
unknown cell or the window edge is also farther than N**; otherwise it is
INSUFFICIENT, because water just outside the window cannot be ruled out. In
particular, a window containing no water at all yields *no* "far from water"
claim -- only INSUFFICIENT. When the engine owns the grid it requests
`buffer_cells = ceil(distance / resolution) + 1` so the ROI interior is exact;
when the Phase 8 engine owns the grid (its buffer is 2 cells) the engine emits
an explicit warning instead of guessing.

**Area** comes from `|a| * |e|` of the analysis transform -- never an assumed
10x10 m. The Phase 8 native/effective resolution distinction is carried over
(`source_resolutions`, `effective_resolution_note`).

**Zero matches** is a computed result, worded as
*"Within the selected ROI, 0% of analysed cells satisfy all requested
conditions"*, never "no such land exists".

**Performance.** One windowed WorldCover read per run, shared by every
land-cover-derived condition; the class-80 mask is built once and reused by
every proximity condition; all arrays are bounded to the analysis grid.

## 7. How is missing data distinguished from zero matches?

| Situation | Report |
|---|---|
| Conditions evaluated, no cell satisfies them | **“No cells satisfy all requested conditions.”** |
| Some cells could not be evaluated | reported as **Insufficient data** cells, excluded from the fraction, listed with the reason (nodata / excluded class / unresolvable layer) |
| A whole condition could not be evaluated | status `INSUFFICIENT_DATA`, the combination is not attempted |

The system never says “no suitable land exists” — absence of matches is not evidence of
absence of suitable land.

## 8. How are different raster resolutions aligned?

**One grid per query, decided once:**

1. If the cotton condition is present, run the Phase 8 engine **first** and **adopt its
   grid** (`result.grid`). This guarantees bit-identical CRS / transform / width / height for
   every mask without touching Phase 8 code.
2. Otherwise build the grid with `make_grid(roi.geometry_raster_crs, crs,
   analysis_resolution_m)` (30 m default, cell-capped, coarsening reported).

Every mask is produced on that grid: land cover via `fetch_land_cover(grid)`
(**nearest-neighbour** — never bilinear on class codes), the distance field computed in the
grid’s metric CRS with the pixel size read from the affine.

Before combining, `core/spatial.require_compatible()` **asserts** CRS, transform and shape
identity and raises `GridMismatchError` otherwise — arrays are **never** combined because
their shapes merely match (tests 7–9). Effective resolutions are reported per condition
(WorldCover 10 m native → 30 m grid, GSW 30 m if used in verification), and the
“resampling does not create detail” warning from Phase 8 is reused verbatim.

## 9. What does the final geographic result mean?

```python
@dataclass
class SpatialQueryResult:
    query: str
    expression: str              # "crop_suitability(cotton, >=Moderate) AND land_cover(40) AND water_proximity(<=1000 m)"
    operator: str
    conditions: Tuple[ConditionResult, ...]   # per condition: status, match fraction, resolution, provenance
    matched_cell_count: int
    matched_area_m2: float                    # cells x |a*e| from the affine
    matched_fraction: float                   # matched / EVALUATED cells inside the ROI
    evaluated_cell_count: int
    insufficient_cell_count: int
    result_mask: Any                          # codes: 2 MATCH, 1 NO MATCH, 0 INSUFFICIENT
    raster_transform: Any; raster_crs: Any
    grid: AnalysisGrid
    analysis_resolution: float
    effective_resolutions: Dict[str, str]
    warnings: List[str]
    provenance: List[Dict[str, Any]]
    performance: Dict[str, Any]
```

* The denominator is **cells inside the ROI** (`roi_mask`, `all_touched=False`), not the
  whole grid — the grid carries a 2-cell buffer that must never enter a percentage.
* The answer is location-aware: *“Within the selected ROI, 18.4 % of analysed cells
  (≈1.66 km²) satisfy all requested conditions”* followed by a per-condition ✓ list.
* `to_dict()` excludes raw arrays; the conversational answer contains no NumPy.

## 10. How is provenance preserved?

Every condition contributes its own `SourceRecord` (dataset, version, variable, native
resolution, native CRS, units, temporal period, source URL, licence, access date, nodata,
resampling, processing, limitations) — reused unchanged from the engines that produced it.
Added by Phase 9 and stored with the result:

* the **distance method** (algorithm, grid CRS, pixel size used, threshold, centre-to-centre
  convention),
* the **combination expression** as an auditable string,
* the **operator** and per-condition match fractions,
* per-stage **timings** (parse / engine / alignment / combination / map / total) and peak RSS,
  in the same `performance` convention as Phase 8.

---

## 11. MVP conditions

| Condition | Source | Notes |
|---|---|---|
| `crop_suitability(crop=cotton, min_class=3)` | Phase 8 engine, **rainfed scenario** | see the assumption guard below |
| `land_cover_class(40)` cropland | WorldCover | “cropland” is **not** “suitable” — separate condition, separate evidence |
| `water_proximity(≤ 1000 m)` | WorldCover class 80 | §5 |
| `land_cover_class(50)` built-up, negated | WorldCover | “but not built-up” |

**Assumption guard (important):** the cotton condition always uses the **rainfed
(data-backed)** scenario. The irrigation-assumed scenario is a hypothetical sensitivity
analysis; stacking it on top of a water-proximity condition would compound two assumptions
and quietly imply irrigation. If the wording implies irrigation, the query goes to
clarification instead (§12/§13).

**“Suitable” threshold:** default `min_class = 3` (*Moderately suitable or better*),
configurable; the answer states the interpretation explicitly. Class 0 (Insufficient data)
is never treated as “not suitable”.

## 12. Boolean logic

`AND` (default; “and”, “also”, “but”, “with”), `OR` (“or”, “either”), `NOT` as a per-condition
flag (“not”, “outside”, “excluding”, “away from”, “free of”). Supported patterns:

* `cotton AND cropland`
* `cotton AND near_water`
* `cotton AND cropland AND near_water`
* `cropland AND NOT near_water`
* `cotton AND NOT built_up`

Mixed `AND`/`OR` → clarification (no silent precedence).

## 13. Ambiguity

| Query | Handling |
|---|---|
| “Find areas suitable for cotton near water.” | Interpret as **suitability + water proximity**, and state: *“Interpreting ‘near water’ as proximity to mapped surface water, not irrigation availability.”* |
| “Can I grow cotton near water?” | Same interpretation **plus** the disclaimer; the screening verdict (Phase 8) is reported alongside the map |
| “Can I grow cotton with irrigation?” / “Is this suitable for irrigated cotton?” | **Blocked** (verdict form) — no engine runs; the reply states that only the **rainfed** scenario is evaluated |
| “Find cotton land with reliable irrigation.” | **Clarification** — “I can evaluate proximity to mapped surface water, but reliable irrigation availability is not currently measured.” Nothing computed. |
| A query mixing supply language with proximity | **Clarification** (the interpretation changes the scientific meaning) |

## 14. Performance plan

ROI-first, as always: no global reads, one grid per query, cached AOI subsets reused
(WorldCover is shared by three conditions and by Phase 8). Measured in
`scripts/performance_phase9.py` and stored in `result.performance`:
`parse_seconds`, `engine_seconds` (per condition), `alignment_seconds`,
`combination_seconds`, `map_seconds`, `total_seconds`, `peak_rss_mb`, `cells`.
Expectation to confirm: on a warm cache the whole query is sub-second; a cold grid costs the
same public-data fetch Phase 8 already measured (≈317 s at 11 k cells, ≈250 MB peak).
AOI-2 is 472 k cells — 43× more cells, but the external *source* windows grow far less
(WorldClim ≈1 km, SoilGrids 250 m), so fetch cost should stay in the same order; this is
measured, not assumed.

## 15. Test plan (synthetic first, all hand-calculable)

`tests/test_phase9_spatial.py` + `tests/test_phase9_router.py` + `tests/test_phase9_alignment.py`:

1. A AND B · 2. A OR B · 3. A AND NOT B · 4. zero-match · 5. all-match · 6. partial-match ·
7. mismatched CRS → `GridMismatchError` · 8. mismatched transform → `GridMismatchError` ·
9. mismatched resolution → `GridMismatchError` · 10. water proximity distance (hand-computed
on a 10×10 mask at a known pixel size, plus a distance-parse test for “1 km”) ·
11. ROI clipping (buffer cells excluded from the fraction) · 12. insufficient-data condition
propagates · 13. unsupported condition → clarification, nothing computed · 14. ambiguous
condition → clarification · 15. cached vs uncached execution (same numbers, and zero network
on the second run — proven by timing/`from_cache` flags).
Plus: the five §12 query examples parse to the expected structured conditions; the
irrigation query refuses; Phase 8 patterns are unaffected.

## 16. Real-data verification (`scripts/verify_phase9.py`)

Two AOIs on the bundled Nile Delta sample:

* **AOI-1** — the Phase-8 3×3 km ROI (105×105 @30 m). Expected: cropland ≈83.5 %,
  water **0 cells** → the water condition legitimately yields **zero matches** (test case 4,
  and a regression anchor because every number here is already independently verified).
* **AOI-2** — the full sample extent (687×687 @30 m = 471,969 cells, 7.06 % water) → a
  non-trivial proximity mask and combination.

Independently recomputed **outside** the implementation: WorldCover re-read with plain
windowed rasterio; water mask counted; **distance computed with shapely buffers**
(vectorise → union → buffer(1000) → rasterise) instead of scipy’s EDT; combined mask, cell
count and area hand-checked against `cells × |a·e|`. Optionally cross-check the water mask
against **JRC GSW occurrence ≥ 50 %** and report agreement honestly (they measure different
things: a 2021 snapshot at 10 m vs 1984–2021 frequency at 30 m).

## 17. Browser test (`scripts/browser_test_phase9.py`)

Real Chromium, real mouse and keyboard: draw ROI → type *“Find areas suitable for cotton
near water.”* → submit → verify (1) multiple conditions recognised, (2) result appears,
(3) the map layer, (4) its legend (MATCH / NO MATCH / INSUFFICIENT DATA), (5) area and count,
(6) the methodology explanation, (7) that “near water” is explicitly *not* called irrigation.
Then *“Find cotton land with reliable irrigation.”* must **not** claim irrigation
availability and must compute nothing.

## 18. Map layer

New, dedicated symbology — not NDVI, not the suitability ramp:

| State | Colour |
|---|---|
| MATCH | violet (`#7b2cbf`), opaque |
| NO MATCH | **transparent** (the basemap stays visible; the layer highlights matches) |
| INSUFFICIENT DATA | pale grey, low opacity |

`ui/map.py::spatial_query_rgba()` + `spatial_query_legend_html()` (the legend also prints the
condition expression and the distance used). Display copy only: nearest-neighbour
reprojection to web mercator, never used for any reported number.

## 19. UI

Same conversational flow, one new panel `render_spatial_query`:
**Query → Interpreted conditions (with the water disclaimer) → Result (fraction + area +
per-condition ✓) → Map → “Why?” (boolean logic, distance method, alignment) → “Data &
methodology” (datasets, grid, resolutions, timings) → “Limitations”** (near water ≠
irrigation; land cover is a 2021 snapshot; model soil/climate inputs; not a recommendation).
No dashboard, no new navigation.

## 20. Not implemented

Flood detection · time-series change · irrigation detection · groundwater · ML query planning ·
LLM/VLM reasoning · yield prediction. Phase 9 is deterministic multi-condition spatial
reasoning over verified engines.

## 21. Files planned (after approval)

```
docs/PHASE9_DESIGN.md                 (this file)
config/spatial/conditions.yml         distance, classes, "suitable" threshold, limits
config/spatial/patterns.yml           condition synonyms, operator words, unsupported phrases
core/spatial_query.py                 SpatialQuery / Condition / parse_conditions  (no Streamlit)
core/spatial.py                       GridMask, require_compatible, combine, water_proximity
analyses/spatial_query.py             SpatialQueryResult + run_spatial_query
analyses/registry.py                  +1 row (SPATIAL_QUERY)
core/router.py                        +1 intent, +1 pattern entry, +conditions field
ui/map.py                             spatial_query_rgba(), spatial_query_legend_html()
ui/components.py                      render_spatial_query()
app.py                                overlay + chat branch (mirrors the Phase 8 pattern)
tests/test_phase9_*.py                ≥15 synthetic cases + parsing cases
scripts/verify_phase9.py              independent real-data recomputation (AOI-1 + AOI-2)
scripts/performance_phase9.py         per-stage timings + peak memory
requirements.txt                      + scipy>=1.11
```

Phase 8 methodology is **not** modified unless a regression is discovered.

---

## 22. Decisions I need from you (Checkpoint A)

1. **“Suitable” threshold** — default `min_class = 3` (*Moderately suitable or better*), or
   `2` (*Marginal or better*)? *(recommendation: 3, always stated in the answer)*
2. **Water definition** — permanent water (class 80) only, or should herbaceous wetland
   (90) be included? *(recommendation: 80 only; 90 stays an opt-in flag)*
3. **Demo/verification AOI** — keep only the 3×3 km ROI (water = 0 cells), or add the full
   sample extent (472 k cells, 7.1 % water) as AOI-2? *(recommendation: add AOI-2; AOI-1
   stays as the zero-match regression anchor)*
4. **Do water cells themselves count as “near water”** (distance 0)? *(recommendation: yes,
   documented; they are excluded anyway by the suitability condition)*
5. **Mixed AND + OR** — clarify, or apply a fixed precedence? *(recommendation: clarify)*
6. **JRC GSW cross-check** in verification only (one extra public dataset, read-only)?
   *(recommendation: yes, with a shapely-buffer fallback if the fetch misbehaves)*
7. **Single-condition “Find areas suitable for cotton.”** — route to the new layer (matched
   area + map) or leave it to Phase 8? *(recommendation: new layer, reusing the engine)*
