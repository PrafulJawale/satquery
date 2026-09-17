# Phase 9 — Checkpoint D: real-data verification

**Scope: verification only.** No Streamlit UI, no map layer, no browser test,
no new flood/irrigation/groundwater inference, no LLM. JRC GSW was used once as
an independent cross-check and is not part of the engine.

Machine-readable evidence: `docs/PHASE9D_evidence.json`
Reproduce with: `python scripts/verify_phase9.py` (≈5 s with warm caches).

Regression: **501 tests pass** (496 + 5 new) · Phase 8 gate **15/15** ·
Phase 8 thresholds, weights and source handling unchanged.

---

## 1. AOI definitions

| | AOI-1 | AOI-2 |
|---|---|---|
| Role | Phase 8 zero-water regression anchor | positive water-proximity case |
| Box (EPSG:32636) | `385940, 3450560 → 388940, 3453560` | `377200, 3441820 → 397680, 3462300` |
| Size | 3 000 × 3 000 m = **9.000 km²** | 20 480 × 20 480 m = **419.430 km²** |
| Analysis grid | **105 × 105** @ 30 m (11 025 cells) | **687 × 687** @ 30 m (471 969 cells) |
| Transform | `(30, 0, 385860, 0, -30, 3453630)` | `(30, 0, 377130, 0, -30, 3462360)` |
| Window bounds | `385860, 3450480, 389010, 3453630` | `377130, 3441750, 397740, 3462360` |
| Cells inside ROI | 10 000 | 466 489 |
| CRS | EPSG:32636 (UTM 36N, projected, metres) | same |

AOI-1 was **not** altered to manufacture matches. AOI-2 is the already verified
full sample extent; its water fraction (7.06 %) reproduces the Checkpoint-A
measurement exactly.

## 2. Source datasets and provenance

| | |
|---|---|
| Dataset | ESA WorldCover 10 m 2021 **v200** |
| Tile | `ESA_WorldCover_10m_2021_v200_N30E030_Map.tif` |
| URL | `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N30E030_Map.tif` |
| Licence | CC BY 4.0 (ESA WorldCover project, doi 10.5281/zenodo.7254221) |
| Native | 10 m, EPSG:4326, epoch 2021 |
| Read | **windowed** on the analysis grid only — no global raster is ever downloaded |
| Resampling | **nearest neighbour** (categorical; never averaged) |
| Access | cached (`from cache: True`, 0.01–0.02 s per window) |
| Class semantics | 40 = cropland, 50 = built-up, 80 = permanent water, 90 = herbaceous wetland (**not** water), 0/NaN = nodata |
| Cotton | the existing Phase 8 engine (rainfed scenario), unchanged |

Class counts (full analysis window, independently counted from the raw array):

| class | AOI-1 | | AOI-2 | |
|---|---:|---|---:|---|
| 10 Tree cover | 5 | 0.05 % | 774 | 0.16 % |
| 20 Shrubland | 16 | 0.15 % | 360 | 0.08 % |
| 30 Grassland | 47 | 0.43 % | 2 619 | 0.55 % |
| 40 Cropland | **9 210** | 83.54 % | **326 369** | 69.15 % |
| 50 Built-up | 570 | 5.17 % | 47 522 | 10.07 % |
| 60 Bare/sparse | 2 | 0.02 % | 1 777 | 0.38 % |
| 80 Permanent water | **0** | 0.00 % | **33 329** | 7.06 % |
| 90 Herbaceous wetland | 1 175 | 10.66 % | 59 219 | 12.55 % |
| nodata | 0 | 0.00 % | 0 | 0.00 % |

AOI-1's 83.54 % cropland reproduces the Phase 8 figure; AOI-2's 69.15 % /
10.07 % / 7.06 % / 12.55 % reproduce Checkpoint-A exactly.

## 3. Resolutions and CRS

* Analysis (effective) resolution: **30 m** on both AOIs.
* Source (native) resolutions — WorldCover **10 m**, soil 250 m, climate ~1 km,
  topography 30 m. Reported per result as `source_resolutions` plus
  `effective_resolution_note`: *"Analysed at 30 m; inputs are native 10–1000 m,
  so finer detail than the analysis grid is not resolved."*
* CRS: EPSG:32636 (projected). Distances are computed in **metres**; the
  proximity engine refuses a geographic CRS outright.
* Cell area is taken from the transform (`|a|·|e|` = **900 m²**), never assumed.

## 4. Requested proximity distance and buffer

* Default distance: **1 000 m** (`config/spatial/conditions.yml`).
* Buffer: `ceil(1000 / 30) + 2 = ` **36 cells per side = 1 080 m**.
  The extra 2 cells are alignment/reprojection margin; the buffer must be at
  least the requested distance (Decision 1).
* Resulting windows: AOI-1 105 → **177 × 177** (cropped back to 105 × 105);
  AOI-2 687 → **759 × 759** (cropped back to 687 × 687).
* Cropping is exact: the expanded window keeps the transform coefficients and
  moves the origin by whole cells, so `crop_mask` slices rather than resamples
  (it refuses CRS changes, cell-size changes, half-cell offsets and targets that
  are not contained).
* Cap: 2 000 000 cells — both windows are far below it (31 k and 576 k).

## 5. The nine verification cases

AOI-1 (10 000 ROI cells):

| # | case | status | matched | insufficient | matched % |
|---|---|---|---:|---:|---:|
| 1 | cotton suitability alone | OK | 0 | 0 | 0.00 % |
| 2 | permanent water alone | OK | 0 | 0 | 0.00 % |
| 3 | near permanent water (1 000 m) | OK | **1 047** | 0 | 10.47 % |
| 4 | cotton AND near permanent water | OK | 0 | 0 | 0.00 % |
| 5 | cropland AND near permanent water | OK | **884** | 0 | 8.84 % |
| 6 | cropland AND NOT permanent water | OK | **8 335** | 0 | 83.35 % |
| 7 | cropland AND water (impossible) | OK | 0 | 0 | 0.00 % |
| 8 | cotton alone — insufficient data | OK | 0 | **0** (in ROI) | 0.00 % |
| 9a | irrigation | **UNSUPPORTED_CONDITION** | — no computation — | | |
| 9b | flood-prone | **UNSUPPORTED_CONDITION** | — no computation — | | |

AOI-2 (466 489 ROI cells):

| # | case | status | matched | area | insufficient | matched % |
|---|---|---|---:|---:|---:|---:|
| 1 | cotton suitability alone | OK | 0 | — | **1 589** | 0.00 % |
| 2 | permanent water alone | OK | **32 731** | 29.458 km² | 0 | 7.02 % |
| 3 | near permanent water (1 000 m) | OK | **165 809** | 149.228 km² | 0 | 35.54 % |
| 4 | cotton AND near permanent water | OK | 0 | — | **800** | 0.00 % |
| 5 | cropland AND near permanent water | OK | **78 450** | 70.605 km² | 0 | 16.82 % |
| 6 | cropland AND NOT permanent water | OK | **322 951** | 290.656 km² | 0 | 69.23 % |
| 7 | cropland AND water (impossible) | OK | 0 | — | 0 | 0.00 % |
| 8 | cotton alone — insufficient data | OK | 0 | — | **1 589** | 0.00 % |
| 9a | irrigation | **UNSUPPORTED_CONDITION** | — no computation — | | | |
| 9b | flood-prone | **UNSUPPORTED_CONDITION** | — no computation — | | | |

Every figure is computed inside the ROI; the window buffer is excluded from all
three counters.

## 6. The required distinction

| situation | evidence |
|---|---|
| **No permanent water in the source** | AOI-1: class-80 count = **0** in the whole window; case 2 reports 0 matched **and 0 insufficient**, i.e. every cell was decided, and the message says *"0 % of analysed cells satisfy all requested conditions … a statement about the selected area, not about the surrounding region."* |
| **Zero valid matches** | Case 7 (`cropland AND water`) = 0 matched with 0 insufficient on both AOIs: a cell cannot hold two WorldCover classes, so the mask is empty **by construction**, not from missing data. |
| **Insufficient data** | AOI-2 cotton: **1 589** cells are Phase 8 class 0 inside the ROI; they are reported as insufficient, never as non-matching. Case 4: **800** cells are insufficient because `INSUFFICIENT AND TRUE = INSUFFICIENT` (three-valued logic doing real work). |
| **Unsupported condition** | 9a and 9b: `UNSUPPORTED_CONDITION`, `result is None`, and the provider call counters stay empty — no dataset was read, no proxy was substituted. |

The engine never says "no suitable land exists": the wording is always
*"Within the selected ROI, X % of analysed cells satisfy all requested
conditions"*.

## 7. Independent verification

### 7.1 Water proximity — brute force, not the engine's own method

Distances were recomputed by explicit coordinate arithmetic
(`min over water cells of sqrt(dx² + dy²)`) on an 80 × 80 sample window placed
around the water cell in the sparsest neighbourhood, with the water search box
expanded by `distance + window` cells so no nearer water outside the box could
be missed. This does not use `scipy.ndimage.distance_transform_edt`.

| check | AOI-1 | AOI-2 |
|---|---|---|
| window | 171 × 171 @ 30 m | 753 × 753 @ 30 m |
| water cells in window | 72 | 44 233 |
| sample window | 80 × 80, distances 0 – 2 100 m | 80 × 80, distances 0 – 1 236 m |
| distance on water pixels | **0.000 m** (max) | **0.000 m** (max) |
| cells compared (engine-decidable) | 6 400 | 6 400 |
| **agreement with brute force** | **6 400 / 6 400 = 100.0000 %** | **6 400 / 6 400 = 100.0000 %** |
| just inside 1 000 m (must be TRUE) | 80 cells → engine TRUE on **80** | 160 cells → engine TRUE on **160** |
| just outside 1 000 m (must not be TRUE) | 80 cells → engine TRUE on **0** | 160 cells → engine TRUE on **0** |
| unit sanity (k cells east of water) | never farther than `k × 30 m`; **0 violations** | never farther than `k × 30 m`; **0 violations** |
| units | metres (0–2 100 m range) | metres (0–1 236 m range) |
| CRS | EPSG:32636, projected | EPSG:32636, projected |

Tolerance: none needed — the comparison is on integer cell counts and boolean
masks; the distance test is the exact `distance ≤ 1000 m` predicate.

Two earlier "failures" were bugs in the *check*, not the engine: (a) the brute
force only saw water inside the sample box (6 cells disagreed near the box
edge — fixed by expanding the water search box, after which agreement is
100 %); (b) the first sample windows never reached 1 000 m, so the boundary
test was vacuous (fixed by choosing the sparsest neighbourhood).

### 7.2 Cotton — compared against the Phase 8 engine itself

The cotton mask was recomputed in plain NumPy from the Phase 8 class raster as
`valid = class ≥ 1`, `match = class ≥ 3`:

| | AOI-1 | AOI-2 |
|---|---|---|
| cells | 11 025 | 471 969 |
| engine mask == (`class ≥ 3`) | **identical** | **identical** |
| TRUE cells engine / recomputed | 0 / 0 | 0 / 0 |
| INSUFFICIENT engine / recomputed | 1 025 / 1 025 | 7 069 / 7 069 |
| Phase 8 classes inside ROI | {0: 0, 1: 10 000} | {0: 1 589, 1: 464 900} |

Phase 8's own `class_fractions` agree: AOI-1 `Unsuitable 1.0`, AOI-2
`Insufficient data 0.00341 / Unsuitable 0.99659` (0.00341 × 466 489 ≈ 1 590 ≈
1 589 counted). No Phase 8 threshold, weight, formula or source was changed.

### 7.3 Spatial combinations — recomputed from the raw arrays

Each combination was rebuilt in NumPy from the raw WorldCover array (and, for
the cotton case, from a replicated buffered read + crop, exactly as Decision 1
prescribes) and compared **cell by cell** inside the ROI:

| combination | AOI-1 | AOI-2 |
|---|---|---|
| cotton AND near-water | identical (0 TRUE) | **identical** (0 TRUE, 800 insufficient) |
| cropland AND near-water | **identical** (884 TRUE) | **identical** (78 450 TRUE) |
| cropland AND NOT water | **identical** (8 335 TRUE) | **identical** (322 951 TRUE) |
| NOT water vs NOT near-water | 8 335 vs 1 243 → **not identical** | 322 951 vs 204 735 → **not identical** |

The last line is the required semantic regression on real data: `NOT WATER`
(cells that are not class 80) and `NOT WATER_PROXIMITY` (cells farther than
1 000 m from water) give materially different answers, so the Checkpoint-B
correction holds end to end.

### 7.4 JRC Global Surface Water — cross-check (AOI-2, verification only)

| | cells | share |
|---|---:|---:|
| WorldCover 2021 permanent water | 33 329 | 7.06 % |
| GSW occurrence > 0 (any time 1984–2021) | — | 13.89 % |
| both | 32 748 | |
| WorldCover only | 581 | |
| GSW only (ephemeral/seasonal) | 32 791 | |

98.3 % of WorldCover's water cells were also seen by GSW at some point in
1984–2021 — the expected direction. The ~2× larger GSW figure is explained by
GSW counting *any* observation in 38 years (seasonal flooding, wet years,
flooded cropland), while WorldCover is one 2021 epoch of **permanent** water.
They answer different questions; GSW was never used as the engine, and the
disagreement is reported rather than reconciled.

## 8. Performance

| | |
|---|---|
| Total wall time (both AOIs, all cases, warm caches) | **5.2 s** |
| Cold Phase 8 cotton run on AOI-2 (472 k cells) | 30.1 s |
| tracemalloc peak | **83.1 MB** |
| process peak RSS | **202.5 MB** |
| ROI-first | every read is windowed on the analysis grid (or its buffer) |
| Global raster downloads | **none** — one WorldCover tile window + cached soil/climate/DEM |
| Duplicate reads | none: one WorldCover read per run serves all conditions; the class-80 mask is built once and reused |
| Array bounds | largest intermediate = 759 × 759 float32/uint8 ≈ 2.3 MB |

## 9. Warnings, limitations and remaining uncertainty

**Warnings the engine emits (visible in every result):**
* the buffered-window note: *"Permanent-water proximity was computed on a
  buffered WorldCover window (177×177 cells, +36 cells per side = 1080 m) and
  cropped back to the 105×105 analysis grid."*
* the conservative edge rule: a cell is only called "farther than N m from
  water" when the source window is large enough to establish it; otherwise it is
  `INSUFFICIENT_DATA`. On these AOIs the buffer makes every ROI cell decidable,
  so 0 cells were withheld — **no water found in a window is never reported as
  proof of "far from water"**.

**Limitations:**
1. *WorldCover accuracy* — ~75 % thematic accuracy globally; class 40 means any
   crop in 2021 (not cotton, not currently cultivated); v200 must never be mixed
   with v100.
2. *Proximity ≠ irrigation* — the distance is to mapped **permanent** water. It
   is not irrigation, not groundwater, not canal access, not flood risk, not
   soil moisture, and no such claim is made anywhere.
3. *Cotton ≥ 3 is empty on both AOIs* — Phase 8's verified rainfed verdict for
   this region is **Unsuitable** (score 0.66, driven by its documented limiting
   factor). Cases 1, 4 and 8 are therefore genuine zero-matches, not engine
   failures; Phase 9 does not re-grade Phase 8. A positive cotton ∩ proximity
   case needs an AOI where Phase 8 reaches class ≥ 3.
4. *AOI-1 has no water of its own* — the 1 047 cells matched in case 3 are near
   water mapped **outside** the 3 km box but inside the 1 080 m buffered window
   (0 class-80 cells in the 105 × 105 window vs 72 in the 171 × 171 window).
   That is the buffered read working as intended, and it is the reason the
   rim is not reported as insufficient data.
5. *30 m grid vs 10 m source* — nearest-neighbour resampling can miss water
   features narrower than 30 m; the native/effective resolution note is carried
   in every result.
6. *Wetland is not water* — 10.66 % (AOI-1) and 12.55 % (AOI-2) of cells are
   class 90 and are excluded from the water mask by design.
7. *Cell-centre distances* — a cell adjacent to water is 30 m away, not 0 m;
   distances are between cell centres.
8. *GSW cross-check resolution* — reprojected from EPSG:4326 (~30 m here) with
   nearest neighbour; used only as a sanity check.

**Remaining uncertainty:** whether Phase 8's "Unsuitable" verdict for this
region is right is a Phase 8 question and is out of scope here; Phase 9
consumes it unchanged. If a future AOI yields cotton class ≥ 3, cases 1, 4 and
8 should be re-run there to exercise the positive cotton ∩ proximity path on
real data (currently covered by synthetic tests only).

## 10. What was NOT done

No Streamlit UI, no map layer, no browser test, no new flood/irrigation/
groundwater analysis, no LLM/VLM, no autonomous planning. `app.py` and `ui/*`
are untouched (verified by content). GSW appears only in §7.4.
