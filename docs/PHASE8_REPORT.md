# Phase 8 — Experimental Crop-Suitability Screening: FINAL REPORT

**SatQuery AI · SIH 2026 prototype · Phase 8 of 8**
Date: 2026-09-14 · Workspace: `/home/user/satquery`

> **Framing, repeated once here so it governs everything below.**
> This is **experimental crop-suitability *screening***. It is not a crop
> recommendation, not a yield or profit prediction, not a soil-fertility or
> irrigation-sufficiency diagnosis, and not a forecast. It returns one of five
> classes for the area you drew, with the evidence behind it, and it tells you
> what it could **not** assess. Every threshold is labelled with where it came
> from. Results must be validated with local agronomic and field information.

---

## A. What was built

A second analysis behind the same conversational interface. You draw an area,
you ask “Can I grow cotton here?”, and the router sends it to a new engine that:

1. reads **only the ROI window** of four public datasets (nothing global is downloaded),
2. puts them on **one common 30 m analysis grid in a metric CRS**,
3. converts each factor to a **membership 0–1** with a trapezoidal function,
4. combines them with a **weighted mean**, then applies **four separate gates**,
5. returns a **class, score, confidence, limiting factors, missing factors and a raster**,
6. draws the classes on the map with **their own legend**, and explains itself in
   an expandable “Why?” and “Data & methodology” panel.

Cotton is implemented; any other crop is refused with
*“Only cotton suitability is currently supported.”* and nothing is computed.

Verification summary (details in §K–M):

| Gate | Result |
|---|---|
| Synthetic unit tests (hand-calculable) | **67 passed** |
| Full suite, Phases 1–8 | **333 passed** |
| Real data vs independent recomputation | **15 / 15 checks** |
| Browser, Phase 8 (real mouse + keyboard) | **44 / 44 checks** |
| Browser, Phase 7 regression (NDVI + router) | **28 / 28 checks** |
| Budget: peak memory on a cold network run | **250 MB** (limit ≈2 GB) |

---

## B. Datasets, licences and provenance

| Layer | Dataset | Native | Temporal | Licence | Used for |
|---|---|---|---|---|---|
| Land cover | ESA WorldCover 10 m 2021 **v200** | 10 m | static (2021) | CC BY 4.0 · doi 10.5281/zenodo.7254221 | hard constraints (built-up, water, wetland…) |
| Soil | SoilGrids **2.0** (ISRIC) pH, clay/sand/silt | 250 m | static prediction | CC BY 4.0 · Poggio et al. 2021 | pH (critical), texture (supporting) |
| Climate | WorldClim **2.1** 30″ monthly tmin/tmax/prec | ~1 km | **climatological normals 1970–2000** | CC BY 4.0 | water (critical), temperature / GDD (critical) |
| Topography | Copernicus DEM **GLO-30** | 30 m | static | Copernicus/Open data licence | slope (supporting), elevation context |

**Every one is model output or a static map — none is a field measurement.**
Each fetched subset is written to `data/external/<source>/…tif` with a **JSON
provenance sidecar** carrying the 14 fields of `SourceRecord` (dataset, version,
variable, native resolution, native CRS, units, temporal period, source URL,
licence, access date, nodata, resampling, processing, limitations) plus a cache
identity of `{dataset, version, variable, source_url, aoi_signature, resolution,
resampling, band}`. `data/external/` is in `.gitignore` — **external data is
never committed**, and a subset with a different identity can never be silently reused.

---

## C. Factor inventory (availability decided before anything was coded)

| Category | Factor | Status | Why |
|---|---|---|---|
| **Hard constraint** | Land cover (built-up / water / mangrove / wetland / snow / moss) | excluded, never scored | not land that can carry a crop |
| **Critical agronomic** | Growing-season rainfall (Apr–Oct) | scored, veto if outside 450–1500 mm | cotton cannot complete the cycle otherwise |
| **Critical agronomic** | Temperature (approx. GDD₁₅.₆) | scored | thermal time to maturity |
| **Critical agronomic** | Topsoil pH (0–5 cm, 5–15 cm) | scored, veto outside 5.0–9.5 | nutrient availability, FAO-style N class |
| **Supporting** | Texture (USDA class from sand+silt+clay) | scored | water-holding / workability |
| **Supporting** | Slope (Horn 3×3, %) | scored | machinery, erosion, drainage proxy |
| Context only | Elevation | **not scored** | reported for terrain context |
| **Not assessed** | Salinity (ECe) | unavailable at 250 m globally | named as unassessed — never scored as 0 |
| **Not assessed** | Irrigation availability / water source | no dataset | hypothetical Scenario B only |
| **Not assessed** | Rooting depth, drainage, organic matter, nutrients, pests, variety, sowing date | out of scope | stated in the answer |

---

## D. Spatial and temporal alignment

**Spatial.** One `AnalysisGrid`: ROI bbox → metric CRS (UTM zone of the ROI) →
snapped outward onto a **30 m** lattice (`analysis_resolution_m` in
`config/crops/cotton.yml`), with a 2-cell buffer. If the grid would exceed the
**2,000,000-cell** cap the cell size is doubled and the change is **reported**
(requested vs effective + reason) — never applied silently.

Three different resolutions are always reported separately, because resampling
does not create detail:

* **native** — LC 10 m, DEM 30 m, soil 250 m, climate ~1 km
* **analysis grid** — 30 m (or the coarsened effective value)
* **effective information** — ≈ the coarsest input; a warning is emitted whenever a
  layer is ≥4× coarser than the grid (“The soil layer is native 250 m but is shown
  on a 30 m analysis grid…”).

**Resampling is typed:** continuous layers → bilinear; **categorical (land cover)
→ nearest-neighbour only**, so class codes are never averaged.

**Temporal — explicitly MIXED.** The UI says: *Climatological + static (mixed):
1970–2000 climate normals, a static 2021 land-cover map, static soil predictions
and a static DEM.* Nothing here is “today”, and nothing is a forecast.

---

## E. The model

1. **Membership** — one trapezoidal function per factor: 0 outside the absolute
   range, 1 inside the optimum, linear ramps between.
2. **Score** — weighted mean of the *available* memberships; weights from
   `cotton.yml` — temperature 0.30, water 0.30, pH 0.15, texture 0.15,
   slope 0.10 — renormalised when a factor is missing.
3. **Four gates, in this order:**
   * *hard land-cover constraint* → `Unsuitable`
   * *critical-factor veto* (temperature, water or pH outside its absolute
     tolerance) → `Unsuitable` regardless of the score
   * *missing critical factor* → `INSUFFICIENT_DATA`, **no score at all** (a zero is
     never substituted)
   * *assumed mandatory factor* (the hypothetical irrigation scenario) → capped at
     **Moderately suitable**, confidence **Low**
4. **Classes** — Highly suitable ≥ 0.75, Moderately suitable ≥ 0.55,
   Marginal ≥ 0.35, Unsuitable < 0.35, plus **Insufficient data** (not a score).

**No automatic “weakest factor demotes the class” rule.** The user asked for that
rule to be removed and it was: the weighted score is the model, and limiting
factors are *reported* so a reader can challenge them. The trade-off is stated
openly — a weighted mean is compensatory, so a very weak factor can be partly
offset by strong ones; the veto thresholds (not the weights) are what protect
against severe single-factor failure.

---

## F. Thresholds and where each one comes from

| Item | Values (cotton) | Status | Source / adaptation |
|---|---|---|---|
| Growing-season rainfall | **262 / 437 / 700 / 875 mm** (absolute / optimum / optimum / absolute) | **experimental (prorated from literature)** | FAO EcoCrop annual 750–1200 preferred, 450–1500 tolerated, scaled by the 7/12 season fraction (0.583). Documented caveat: prorating assumes rain is spread evenly, so in winter-rainfall climates (including the Nile Delta) it **overstates** effective growing-season rain — the water factor is optimistic there, not pessimistic |
| pH (topsoil) | **5.0 / 6.0 / 7.5 / 9.5** | literature-backed | FAO EcoCrop / *Gossypium hirsutum*: 6.0–7.5 preferred, 5.0–9.5 tolerated; 9.5 is a screening ceiling, not a guarantee |
| Temperature (monthly mean) | **15 / 22 / 32 / 42 °C** | **literature-adapted** | FAO EcoCrop optimum 22–36 °C, absolute 15–42 °C; the upper *optimum* was reduced 36 → 32 because a **monthly mean** of 36 °C implies daily maxima well above 40 °C (36 stays the absolute maximum) |
| GDD₁₅.₆ | base 15.6 °C, **reported context only — never scored** | approximation | monthly-mean thermal sum; literature target 1200–1400 GDD₁₅.₆ (UF/IFAS ABE381; 10.1002/agj2.21086). It has no thresholds in the model |
| Texture | USDA class → membership (loam 1.00, silt loam 0.90, clay loam 0.85, silty clay loam 0.80, sandy clay loam 0.70, sandy loam / silt / clay 0.55, sandy clay 0.45, loamy sand 0.20, sand 0.10) | **experimental** | FAO land-evaluation texture ratings for upland crops, adapted: heavy clays scored **below** the FAO water-holding rating because cotton is sensitive to poor drainage and crusting. A **proxy**, not a soil-physical assessment |
| Slope | **0 / 0 / 2 / 8 %** | literature-backed | FAO “Soil and terrain suitability for surface irrigation”: optimum <2 %, 2–8 % acceptable (slope classes A 0–2, B 2–5, C 5–8 %) |
| Weights | temperature 0.30, water 0.30, pH 0.15, texture 0.15, slope 0.10 | **experimental weighting scheme** | constructed for this prototype, not literature-derived |
| Class breaks | **0.35 / 0.55 / 0.75** | **experimental screening thresholds** | coarse, evenly spaced breaks on the 0–1 scale — deliberately coarse because the memberships are coarse |

Every row above also exists **in the UI**, in a threshold-provenance table inside
the “Why?” panel, with the columns *Item / Status / Values / Source*, where
`EXPERIMENTAL` means “constructed for this screening, not literature-derived”.

**No number that changes a result lives in Python.** Thresholds, weights, class
breaks, the land-cover policy and the growing-season window are all read from
`config/crops/cotton.yml`.

---

## G. Water: the two-scenario design

Scenario A (**rainfed, data-backed, primary**) scores measured growing-season
rainfall. Scenario B (**irrigation-assumed**) is a **hypothetical sensitivity
analysis**: the water factor is removed, the rest are re-weighted, the class is
capped at *Moderately suitable*, confidence is **Low**, and the panel opens with
“Hypothetical sensitivity analysis … This is NOT a suitability assessment of the
land”. Irrigation is never detected and never assumed to exist without saying so.

Both are shown as separate tabs, and the answer reports **growing-season
precipitation and the annual total** with the months used (Apr–Oct for cotton) —
so “73 mm/year” can never be mistaken for “73 mm in the growing season”.

---

## H. Missing and unverified data

| Situation | Status | Behaviour |
|---|---|---|
| No ROI drawn | `NEEDS_ROI` | no computation, prescribed message |
| Critical factor missing | `INSUFFICIENT_DATA` | **no score** (never zero-filled) |
| Supporting factor missing | `PARTIAL_DATA` | weight removed + renormalised, listed, confidence reduced |
| Water assumed | `PARTIAL_DATA` + `ASSUMED` | capped class, Low confidence, warning text |
| Layer ≥4× coarser than the grid | `OK` + warning | effective resolution stated |
| Unmeasured constraint | — | listed under *not assessed* (salinity, depth, drainage, irrigation) |

**Computed limiting factors and unassessed risks are two different lists.**
The answer never says “the soil is suitable” on the basis of pH + texture;
it says salinity, rooting depth and drainage were not assessed. The map also
separates **Unsuitable** (measured and limiting) from **Insufficient data**
(not measured — never treated as unsuitable).

---

## I. Architecture

```
app.py                         Streamlit shell: map + chat + overlay (only UI file touched)
ui/map.py                      suitability_rgba(), suitability_legend_html()   (new)
ui/components.py               render_crop_suitability() + explainers          (new)
analyses/crop_suitability.py   the engine: grid → layers → scenarios → provenance
analyses/registry.py           CROP_SUITABILITY flipped on — no router rewrite
core/suitability.py            membership, veto, score, caps, classes, threshold provenance
core/alignment.py              AnalysisGrid, make_grid, read_into_grid, roi_mask, compute_slope
core/datasources/{base,worldcover,soilgrids,worldclim,copernicus_dem}.py
config/crops/cotton.yml        every threshold, weight, class break and rule
```

* Everything from `core/` and `analyses/` down is **Streamlit-free** and testable
  without a browser.
* The router was **not rewritten** for this feature: the intent moved from
  `planned` to `available` and the registry binds it (a test enforces the
  router stays numpy/rasterio-free).
* Memory discipline: ROI → bbox → windowed read → reproject subset → close the
  source. Measured peak on a cold network run: **250 MB**.

---

## J. What the user sees

Draw ROI → ask → the answer shows **scenario / class / confidence / score /
computed limiting factors / missing-and-unverified inputs** → the map gains a
**screening overlay with its own five-class legend** (blue/teal for suitable,
grey for unsuitable, faint grey for insufficient — deliberately unlike the NDVI
green/red ramp) → expandable **“Why?”** (model, the four gates, the threshold
table) → expandable **“Data & methodology”** (datasets, grid, temporal basis,
timings) → a closing disclaimer that this is screening, not advice.

---

## K. Verification 1 — synthetic tests first

Written **before** any real data was fetched, all hand-calculable:

| File | Tests | Covers |
|---|---|---|
| `tests/test_phase8_suitability.py` | 32 | membership arithmetic, veto, cap, missing/assumed handling, texture class, GDD approximation, threshold provenance |
| `tests/test_phase8_alignment.py` | 12 | grid snapping, cell cap + reported coarsening, ROI mask, categorical nearest-neighbour, slope (Horn) cases 18a–18f |
| `tests/test_phase8_router.py` | 23 | routing, crop policy, “no ROI” path, no network |

---

## L. Verification 2 — real data, checked against an independent calculation

`scripts/verify_phase8.py`, on the bundled Nile Delta sample (3 × 3 km ROI,
105 × 105 grid at 30 m = 11,025 cells). Part B recomputes the same numbers
**outside the implementation** — plain rasterio windowed reads and plain numpy.

| Quantity | Engine | Independent | |
|---|---|---|---|
| WorldCover dominant class | 40 (cropland) | 40 | ✅ |
| pH (0–5 cm) | 7.071 | 7.11 | ✅ ±0.35 |
| Clay / sand / silt | 37.3 / 35.4 / 29.1 % | 35.5 / 35.4 / 29.1 % | ✅ |
| Growing-season rain (Apr–Oct) | 13.21 mm | 13.2 mm | ✅ |
| Annual rain | 73.22 mm | 73.5 mm | ✅ |
| Slope | 0.719 % | 1.05 % (`np.gradient`) | ✅ |
| Approx. GDD₁₅.₆ | 1796 | 1797 | ✅ |
| Weighted score | 0.66 | **0.6572 hand-calculated** | ✅ |

**15 / 15 checks passed.** Outcome: rainfed **Unsuitable** (confidence High),
and the check confirms the class came from the **critical-factor veto**, not
from the score (0.66 would otherwise be “Moderately suitable”); the
irrigation-assumed scenario is **Moderately suitable** (0.94) and correctly
capped below Highly suitable.
Full output: `artifacts/phase8_verification.json`.

---

## M. Verification 3 — browser, regression, performance

**Browser, Phase 8 — 44/44** (`scripts/browser_test_phase8.py`): real Chromium,
real mouse drag to draw the ROI, real typing. Confirms the routed intent, the
screening panel, both scenario tabs (including clicking the irrigation tab and
finding the hypothetical-analysis warning), the missing-data language, both
expanders opened by clicking, the overlay checkbox, the map layer and the
five-class legend. “Can I grow rice here?” → *Only cotton suitability is
currently supported.*, nothing computed.

**Browser, Phase 7 regression — 28/28** (`scripts/browser_test_phase7.py`):
NDVI question → statistics + histogram + “not a crop-health diagnosis”;
flood (planned) and ambiguous questions still refused; deleting the ROI returns
the prescribed message. **The Phases 1–7 workflow is unchanged.**

**Unit regression — 333 passed** (16/22/33/32/69/36/46/67 per phase 1→8).

**Performance** (`scripts/performance_phase8.py`, 11,025 cells):

| Stage | Cold (network) | Warm (cached) |
|---|---|---|
| Source access (opening remote sources) | 138.6 s | 0.000 s |
| Windowed read + reprojection | 178.1 s | 0.000 s |
| — land cover / soil / climate / terrain | 3.4 / 165.9 / 143.5 / 4.1 s | 0.022 / 0.032 / 0.069 / 0.005 s |
| Read back from local cache | 0.0 s | 0.093 s |
| Factor computation | <0.001 s | <0.001 s |
| Suitability scoring + classification | 0.009 s | 0.009 s |
| Map overlay (display only) | 0.012 s | 0.012 s |
| Panel render | 0.365 s | 0.346 s |
| **Total engine** | **316.9 s** | **0.147 s** |
| **Peak process RSS** | **250 MB** | 174 MB |

All cost is the first fetch of public data; the analysis itself is milliseconds.
Both are far inside the ≈2 GB budget.

---

## N. Limitations, and what was deliberately not built

**Not built, on purpose:** no machine learning, no yield or profit model, no crop
recommendation, no live weather, no irrigation detection, no salinity estimation,
no economics, no time-series phenology, no Sentinel-1, and no LLM/VLM in the
scoring path — the engine is deterministic and hand-checkable.

**Scientific limits shown in the UI, not just here:**
soil inputs are model predictions, not lab results; climate is 1970–2000 normals,
not the season asked about; GDD is approximated from monthly means so daily
extremes are smoothed away; salinity is not assessed (first-order for irrigated
delta cotton); irrigation is assumed, never verified; the DEM is a surface model
so slope over flat ground is near its noise floor; soil (250 m) and climate
(~1 km) are coarse relative to a field; weights and several thresholds are
experimental; and the weighted score is compensatory (§E).
**Validation with local agronomic and field information is required** — the app
says so in the panel, the map legend and the answer.

---

## O. How to run it, and what Phase 9 should do

```bash
cd /home/user/satquery
pip install -r requirements.txt

python scripts/verify_phase8.py          # real data vs independent recomputation (15/15)
python scripts/browser_test_phase8.py    # 44/44  (needs: streamlit run app.py on :8501)
python scripts/browser_test_phase7.py    # 28/28  Phase 1-7 regression in a browser
python scripts/performance_phase8.py     # [--cold] per-stage timings + peak memory
python scripts/smoke_phase8_ui.py        # 22/22 fast UI smoke test (cached result)
pytest tests -q                          # 333 passed
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then: open the app → draw a rectangle → ask **“Can I grow cotton here?”**.

**Known rough edges / next steps (Phase 9+):**
the first query on a new grid fetches every layer (≈5 min cold) — a pre-fetch or
a progress bar would help; the grid snapping means a mouse-jittered ROI can
produce a slightly different grid and therefore a cache miss; only cotton exists,
but `config/crops/<crop>.yml` + the registry are all a second crop needs; soil
salinity and drainage remain unassessed; and the OOM margin in this 2 GB sandbox
is thin when a browser session and the 2048² sample are both live.
