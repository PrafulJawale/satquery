# Phase 8 — Crop Suitability Analysis: DATA-SOURCE & METHODOLOGY DESIGN

**Status: implemented and verified (see §13 and §14).** This document is the
design record; where code and prose disagree, the code is the later truth.
It is *experimental crop-suitability screening*, never a recommendation.

**Status: DESIGN ONLY. Nothing has been implemented and nothing has been downloaded.**
All numbers below were obtained by *transient windowed reads* over HTTP (`/vsicurl`,
`/vsizip//vsicurl`, `/vsigzip//vsicurl`) against the AOI of the bundled sample. No global
raster was downloaded, no file was written to disk, no dataset was cached.

Deliverable type: **experimental crop-suitability screening** (cotton only). Not a
recommendation, not a yield prediction, not a diagnosis, not a validated score.

---

## 0. Quick answer to the ten investigation questions

| # | Question | Answer |
|---|---|---|
| 1 | Factors genuinely required for cotton | Growing-season temperature, water supply, soil reaction (pH), soil texture/drainage, terrain slope, land-cover constraint. (Salinity + irrigation availability are *agronomically essential* but have **no usable public raster** → they become the two headline limitations.) |
| 2 | Obtainable from public data | Yes: land cover (10 m), soil pH/texture (250 m), climate normals (~1 km), topography (30 m). All four verified readable for this AOI. |
| 3 | Not currently available | Soil salinity/EC, irrigation availability & reliability, soil depth, drainage class, sub-surface waterlogging, daily temperature extremes, variety/pest/market data. |
| 4 | Reasonably approximated | GDD from monthly climatologies (standard approximation, coarser than daily); drainage risk from slope+texture; frost risk from coldest-month minimum. Each approximation is labelled. |
| 5 | What causes uncertainty | Coarse climate (1 km) and soil (250 m) relative to a 20 km AOI; salinity missing; irrigation assumed; climatology ≠ the season asked about; SoilGrids is a *model*, not a measurement. |
| 6 | Resolution of each dataset | Land cover 10 m · DEM 30 m · soil 250 m · climate ~1 km (30″) · CHIRPS 5.5 km (optional). |
| 7 | Spatial alignment plan | Everything → one common grid: ROI's UTM zone, 30 m, bilinear for continuous, **nearest-neighbour for categorical**, slope computed *after* reprojection. |
| 8 | Time period of each dataset | Climate = 1970–2000 normals (WorldClim 2.1) / 1981–2010 (CHELSA); soil = static model (2019); DEM = static (2011–2015); land cover = epoch 2021. |
| 9 | Can the Sentinel‑2 sample be combined? | **As the spatial AOI and grid reference only.** NDVI is *not* scored (see §6.3 — circularity). The screening is not date-specific. |
| 10 | Is the Nile Delta AOI a good cotton demo? | Yes — and it is a *better* demo than a benign one, because it forces the honest answer: physically excellent cotton land **that is rainfed-unsuitable (73 mm/yr) and only works under irrigation we cannot verify**. |

---

## 1. The AOI (measured, from the bundled sample)

```
sample        : data/sample/s2_s2b-36ruv-20230806-0-l2a_2048px.tif
grid          : EPSG:32636 (WGS84 / UTM 36N), 2048 x 2048 @ 10 m  ->  20.48 x 20.48 km
UTM extent    : 377200, 3441820 -> 397680, 3462300
WGS84 extent  : 31.10381 – 31.29054 N, 31.71233 – 31.92496 E
centre        : 31.19722 N, 31.81854 E       (Nile Delta, Egypt)
acquisition   : 2023-08-06 (single date — a snapshot, not a time series)
```

## 2. Measured values at the AOI centre (all from windowed reads)

| Layer | Value at AOI | Plausible? |
|---|---|---|
| ESA WorldCover 2021 v200 | class **40 = Cropland** | ✔ |
| SoilGrids pH (0–5 cm) | **7.10** (raw 71, ×0.1) | ✔ delta alluvium |
| SoilGrids clay / sand / silt | **36.7 / 34.9 / 28.4 %** (sum = 100 ✔) | ✔ heavy clay loam |
| SoilGrids CEC / bulk density | **21.6 cmol(c)/kg**, **1.27 kg/dm³** | ✔ |
| SoilGrids SOC | 60.9 g/kg (**6 % OC — implausible**) | ✘ → excluded, see §4.3 |
| Copernicus DEM (30 m) | mean **0.35 m**, range −2.5 … 10.4 m, σ 0.87 m | ✔ flat delta |
| WorldClim 2.1 BIO1 | **20.2 °C** annual mean | ✔ |
| WorldClim 2.1 BIO5 / BIO6 | **31.1 °C** warmest-month max / **7.6 °C** coldest-month min | ✔ |
| WorldClim 2.1 BIO12 | **73 mm/yr** annual precipitation | ✔ hyper-arid |
| WorldClim Jan tmin / tmax / prec | **7.6 °C / 17.9 °C / 17.7 mm** | ✔ |
| CHELSA v2.1 BIO1 (cross-check) | 2940.49 → K×10 → **20.9 °C** | ✔ agrees with WorldClim |
| CHIRPS 2023‑08 | **0.02 mm** (month) | ✔ units validated: Mumbai 2023‑07 = 951 mm |

**Read cost measured (windowed, AOI only):** WorldClim zipped monthly 2–7 s · CHELSA 3.6 s ·
SoilGrids VRT, WorldCover COG, Copernicus COG ≈ seconds. A full monthly climatology fetch
(12 months × 3 variables) ≈ 2–4 minutes, **once per AOI, then cached**.

---

## 3. Factor inventory and availability classification

Legend: **A-NOW** = derivable from data already in the repo · **A-PUB** = public dataset,
verified reachable · **FUTURE** = exists but not integrated in v1 · **N/A** = no usable source.

### 3.1 Remote sensing

| Factor | Source | Res | Class | In v1? | Why / why not |
|---|---|---|---|---|---|
| Land-cover constraint | ESA WorldCover 2021 v200 | 10 m | A-PUB | **Yes — hard constraint** | Built-up / permanent water / wetland are physical exclusions. Verified: AOI = cropland. |
| Vegetation condition (NDVI) | Sentinel‑2 (bundled) | 10–20 m | A-NOW | **No — context only** | Circularity: NDVI measures *what is already growing*, not the land's capacity. Scoring it would reward existing crops (§6.3). |
| Vegetation change / phenology | needs a time series | — | FUTURE | No | Single-date sample only. |

### 3.2 Climate

| Factor | Source | Res | Period | Class | In v1? |
|---|---|---|---|---|---|
| Growing-season temperature (GDD₁₅.₆) | WorldClim 2.1 monthly tmin/tmax | 30″ (~1 km) | 1970–2000 | A-PUB | **Yes — scored** |
| Heat stress (warmest-month max) | same / BIO5 | 30″ | 1970–2000 | A-PUB | **Yes — scored** |
| Cold / frost risk (coldest-month min) | same / BIO6 | 30″ | 1970–2000 | A-PUB | **Yes — scored** |
| Water supply (rainfall) | WorldClim 2.1 monthly prec / BIO12 | 30″ | 1970–2000 | A-PUB | **Yes — scored (rainfed scenario)** |
| Recent-year rainfall cross-check | CHIRPS v2.0 monthly | 0.05° (5.5 km) | 1981→ | FUTURE | Units verified (mm), nodata −9999 undeclared. Kept as an optional cross-check, not the primary. |
| Daily extremes, GDD from daily data | — | — | — | N/A | Monthly climatologies only → GDD is approximated. |

### 3.3 Soil

| Factor | Source | Res | Class | In v1? | Why / why not |
|---|---|---|---|---|---|
| pH (phh2o, 0–30 cm) | SoilGrids 2.0 | 250 m | A-PUB | **Yes — scored** | Cotton optimum 6.0–7.5, tolerance 5.0–9.5. Verified 7.10 ✔ |
| Texture (clay/sand/silt %) | SoilGrids 2.0 | 250 m | A-PUB | **Yes — scored** | Cotton: well-drained loam; clay 20–35 % optimum. Verified 36.7 % (mild). |
| Bulk density / CEC | SoilGrids 2.0 | 250 m | A-PUB | Context only | Supports the texture interpretation; no defensible cotton threshold. |
| Organic carbon (SOC) | SoilGrids 2.0 | 250 m | A-PUB | **Excluded** | Two defects found: (a) 6.1 % OC at this AOI is implausible; (b) the `soc` VRT grid is offset by exactly 3 px (750 m) from `phh2o`/`clay`/`sand` — the property VRTs are *not* on an identical grid. Reported as context with a caveat. |
| **Salinity / EC** | none usable | — | **N/A** | **No** | The single biggest gap for an irrigated delta. FAO GSASmap exists but is not a subsettable raster service here. |
| Soil depth, drainage class, waterlogging | none | — | N/A | No | Approximated indirectly (slope+texture → drainage *risk* flag, not scored). |
| Soil moisture (actual) | Sentinel‑1/2 or SMAP | — | FUTURE | No | Would be date-specific → breaks the climatological model. |

### 3.4 Topography

| Factor | Source | Res | Class | In v1? |
|---|---|---|---|---|
| Elevation | Copernicus DEM GLO‑30 (AWS Open Data) | 30 m (1/3600°) | A-PUB | **Yes — derived** |
| Slope | computed from the DEM **after** reprojection | 30 m | derived | **Yes — scored** (cotton: mechanisation + drainage) |

*Key discovery:* the working AWS key is
`https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N31_00_E031_00_DEM/Copernicus_DSM_COG_10_N31_00_E031_00_DEM.tif`
(28 MB, EPSG:4326, 3600², float32, no nodata set). The `COG_30_*` naming I first tried returns 404.
Fallback if it ever fails: AWS Terrain Tiles `elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif` (verified 200,
EPSG:3857, ~19 m) — and if both fail, topography degrades to `INSUFFICIENT_DATA` (which the
model handles by design, §7).

### 3.5 Water & land constraints

| Factor | Source | Class | In v1? |
|---|---|---|---|
| Permanent water / built-up exclusion | WorldCover (classes 80, 50) | A-PUB | **Yes — hard constraint** |
| Surface-water seasonality | JRC GSW occurrence (30 m) | FUTURE | No — windowed read unreliable; WorldCover already covers the constraint for v1. |
| **Irrigation availability / reliability** | none usable | **N/A** | **No → drives the `ASSUMED` water factor (§6.4)** |
| Groundwater / canal proximity | none | N/A | No |

---

## 4. Dataset provenance blocks (as implemented — access dates and URLs are
written into the JSON sidecar next to every cached subset)

### 4.1 ESA WorldCover 10 m 2021 v200
- **Variable:** discrete land cover (11 classes). **Units:** class code. **Res:** 10 m.
- **CRS:** EPSG:4326 (3°×3° COG tiles). **Coverage:** global. **Time:** 2021 epoch.
- **Source:** `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N30E030_Map.tif` (AWS Open Data, 47 MB/tile).
- **Licence:** CC BY 4.0, doi 10.5281/zenodo.7254221.
- **Relevance:** hard exclusions + "is this farmland at all?".
- **Limitations:** thematic accuracy ~75 % globally; class 40 "cropland" includes fallow; v100 (2020) and v200 (2021) use different algorithms — **never compare across versions**; no irrigated/rainfed distinction.

### 4.2 SoilGrids 2.0 (ISRIC)
- **Variables:** `phh2o`, `clay`, `sand`, `silt`, `cec`, `bdod`, `nitrogen`, `soc` at 0–5 / 5–15 / 15–30 / 30–60 / 60–100 / 100–200 cm; here **0–5 and 5–15 cm mean**.
- **Units (ISRIC mapped):** pH×10, g/kg (texture), mmol(c)/kg (CEC), cg/cm³ (BD), dg/kg (SOC). **The VRTs carry no GDAL scale metadata** (verified: `scales=(1.0,)`) → the conversion factor must be applied in code, from config, per property.
- **Res:** 250 m. **CRS:** Interrupted Goode Homolosine (verified by sampling 5 known sites — pH: Sahara 8.2, Congo 4.6, Iowa 6.6, Sahel 6.3 ✔).
- **Time:** static model output (2019 release; trained on ~2000–2017 profiles).
- **Source:** `https://files.isric.org/soilgrids/latest/data/<prop>/<prop>_<depth>_mean.vrt` (VRT 3.5 MB; windowed reads are cheap despite the global 250 m grid). Also WCS at `maps.isric.org`.
- **Licence:** CC BY 4.0; cite Poggio et al. 2021, SOIL 7:217–240.
- **Limitations:** it is a **machine-learning prediction, not a measurement**; uncertainty is large at 250 m and larger for SOC; grids are **not identical between properties** (750 m offset found for `soc`); no salinity/depth/drainage layers. The REST endpoint (`rest.isric.org`) returned **503** during probing → use the VRT/WCS route.

### 4.3 WorldClim 2.1 (primary climate)
- **Variables:** monthly `tmin`, `tmax`, `prec` (+ `bio` for cross-check). **Units:** °C (stored ×10, GDAL scale applied by rasterio — verified), mm.
- **Res:** 30 arc-sec (~1 km). **CRS:** EPSG:4326. **Time:** **1970–2000 climatological normals**.
- **Source:** `https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_{tmin,tmax,prec}.zip` (4–5 GB each, read via `/vsizip//vsicurl` — **never downloaded**; a windowed read costs 2–7 s).
- **Licence:** CC BY 4.0.
- **Limitations:** 1970–2000 normals are 25–55 years old; do not describe the 2023 season; station interpolation is weak where station density is low; 1 km pixels are effectively constant across a 20 km AOI.

### 4.4 Copernicus DEM GLO‑30 (topography)
- **Variable:** digital surface model (metres, EGM2008-ish vertical datum). **Res:** 30 m (1/3600°). **CRS:** EPSG:4326. **Time:** static, TanDEM‑X acquisitions ~2011–2015.
- **Source:** AWS Open Data `copernicus-dem-30m` bucket (see §3.4 for the working key).
- **Licence:** Copernicus Open Data Licence (free, attribution required).
- **Limitations:** it is a **DSM, not a bare-earth DTM** — buildings/canopy bias elevation (minor at 0–10 m in a flat delta, important in cities/forest); no nodata set → void handling must be explicit; slope from a DSM ≠ ground slope.

### 4.5 CHIRPS v2.0 (optional cross-check, FUTURE)
- **Variable:** monthly precipitation, mm. **Res:** 0.05° (≈5.5 km). **CRS:** EPSG:4326. **Time:** 1981→present, monthly.
- **Source:** `https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.YYYY.MM.tif.gz` (only `.gz` exists; `.tif` → 404).
- **Licence:** free for research/use with attribution (UCSB CHC).
- **Verified:** mm/month (Mumbai 2023‑07 = 951 mm ✔); **nodata = −9999 and is NOT declared** in the GeoTIFF → must be masked manually.
- **Limitation:** 5.5 km is coarser than the AOI's useful detail; satellite-gauge blended, so errors rise in arid regions where monthly totals are near zero.

---

## 5. Temporal and spatial alignment

### 5.1 Temporal — the model is explicitly **MIXED**
| Component | Temporal class | Period |
|---|---|---|
| Climate (temperature, rainfall) | climatological normals | 1970–2000 |
| Soil | static model | ~2019 |
| Topography | static | ~2011–2015 |
| Land cover | epoch | 2021 |
| Sentinel‑2 / NDVI | snapshot | 2023‑08‑06 — **provenance only, not scored** |

Consequence, stated plainly in every result: **this is a long-term, static-climatology screening.
It is not a seasonal forecast, not a statement about the 2023 season, and not "real-time".**
The datasets are combined because suitability of *land* is a long-term property, not because they
share an acquisition date. Every result carries `temporal_basis: "climatological + static (mixed)"`.

### 5.2 Spatial — one common analysis grid
1. **Target CRS** = the ROI's UTM zone (here **EPSG:32636**), so pixel size is in metres and slope maths is valid.
2. **Cell size** = **30 m** (default, configurable) — matches the DEM, is a common Sentinel‑2 resolution, and is finer than the soil (250 m) and climate (~1 km) inputs, so no source is up-sampled more than necessary.
3. **Extent** = ROI bounding box snapped outward to whole cells **+ 2-cell buffer** (needed for the 3×3 slope kernel).
4. **Cell cap** = 2,000,000 cells; if the ROI exceeds it, double the cell size and emit a warning (protects the ~2 GB sandbox).
5. **Windowed reads only** — each source is opened remotely, the ROI bbox (+buffer) transformed into the source CRS, and only that window is read/reprojected. No global raster is ever downloaded.
6. **Resampling rules (enforced by layer type, declared in config):**
   | Layer type | Resampling | Why |
   |---|---|---|
   | Continuous (climate, soil, DEM) | **bilinear** | Values are physical quantities; averaging is meaningful. |
   | **Categorical (land cover)** | **nearest neighbour — never bilinear** | Bilinear on class codes invents classes (e.g. 40→50). |
   | Slope | **not resampled — recomputed** from the reprojected DEM in the target CRS | Slope is direction-dependent; reprojecting a slope raster is wrong. |
7. **Slope** = Horn 3×3 gradient on the reprojected DEM (metres) → percent.
8. **Caching** — each fetched subset is written to `data/external/<source>/<aoi_hash>_<variable>.tif`
   with a **JSON provenance sidecar** (dataset, URL, variable, units, native resolution, native CRS,
   time period, licence, access date, resampling applied, limitations). Re-runs reuse the cache.

---

## 6. Proposed suitability model

Deterministic, transparent, hand-calculable, **no machine learning** (there is no labelled
dataset — a model trained on our own score would be circular).

### 6.1 Pipeline
```
ROI (Sentinel-2 grid) + crop config (cotton.yml)
   -> native per-cell analysis grid (UTM, 30 m)
   -> factor layers (windowed fetch -> reproject -> resample per type)
   -> membership normalisation  s_f in [0,1]        (trapezoidal, config-driven)
   -> hard land-cover gate                           (built-up / water / wetland)
   -> critical-factor veto                           (FAO "N" class)
   -> weighted mean  S = sum(w*s)/sum(w)             (available factors only)   <-- PRIMARY MODEL
   -> unverified-input cap                           (ASSUMED factor -> max "Moderately suitable")
   -> class from S only (no automatic downgrade)  +  limiting factors REPORTED, not applied
```

### 6.2 Membership (one formula, all factors)
```
s = 0                                 if v < abs_min or v > abs_max
s = (v - abs_min)/(opt_low - abs_min) if abs_min <= v < opt_low      (1 if opt_low == abs_min)
s = 1                                 if opt_low <= v <= opt_high
s = (abs_max - v)/(abs_max - opt_high) if opt_high < v <= abs_max
```
Config holds `(abs_min, opt_low, opt_high, abs_max)` per factor per crop — **no magic numbers in Python**.

### 6.3 Why NDVI is NOT a factor
NDVI describes *current* vegetation. Scoring it positively is circular ("suitable because
something green is already here") and would make a bare, fallow, perfectly suitable field look
unsuitable. It is therefore recorded as **context** (provenance + optional note) and excluded from
the score. The existing NDVI engine is not modified.

### 6.4 Water: the two-scenario design (the honest core of this model)
The AOI receives **73 mm/yr** against cotton's absolute minimum of ~450 mm. A single rainfed score
would just say "Unsuitable" and be technically right but useless; pretending irrigation exists
without data would be fabrication. Therefore **both** are computed from the same factor stack:

| Scenario | Water factor | Result meaning |
|---|---|---|
| **Rainfed (primary, data-complete)** | scored from precipitation climatology | The defensible, fully data-backed answer. |
| **Assumed-irrigated (secondary, clearly labelled)** | **NOT scored — marked `ASSUMED`**, its weight removed and the remainder renormalised | "If irrigation is available, the *remaining* factors give…". Class is **capped at Moderately suitable** and confidence is **Low** because a mandatory input is unverified. |

Crop config declares which factors are `critical` (veto power) and which are optional.

### 6.5 Cotton configuration (draft — every number traceable to §2/§6.6)

| Factor | Variable | abs_min | opt_low | opt_high | abs_max | weight | critical | Source of the range |
|---|---|---|---|---|---|---|---|---|
| Growing-season temperature | GDD₁₅.₆ (°C·d) | 900 | 1400 | 2600 | 3600 | 0.30 | yes | base 15.6 °C (60 °F); 1200–1400 DD min for a short-season crop; 343/584/630 DD to square/flower/boll |
| Water supply (rainfed) | growing-season rainfall (mm) | 450 | 750 | 1200 | 1500 | 0.30 | yes | FAO EcoCrop: 750–1200 preferred, 450–1500 tolerated |
| Soil reaction | pH (0–30 cm) | 5.0 | 6.0 | 7.5 | 9.5 | 0.15 | yes | FAO EcoCrop: 6.0–7.5 preferred, 5.0–9.5 tolerated |
| Soil texture | clay % | 10 | 20 | 35 | 55 | 0.15 | no | deep well-drained loam; heavy clay → drainage/tilth penalty |
| Terrain | slope (%) | 0 | 0 | 2 | 8 | 0.10 | no | mechanised cultivation; flat is optimal |

> **This table is the approved design draft.** Three rows changed during implementation,
> and **`config/crops/cotton.yml` is the authoritative source** (every number there also
> appears in the UI's threshold-provenance table and in `docs/PHASE8_REPORT.md` §F):
>
> 1. **Growing-season temperature** is no longer scored as GDD. The factor is the *mean of
>    per-month trapezoid memberships* over the season, on monthly mean temperature
>    (optimum 22–32 °C, absolute 15–42 °C — the upper optimum was reduced from 36 to 32 °C
>    because a monthly mean of 36 °C implies daily maxima above 40 °C). GDD₁₅.₆ is still
>    computed and reported, but as **context only — it is never scored**.
> 2. **Water supply** is prorated to the growing season: 262 / 437 / 700 / 875 mm instead
>    of the annual 450 / 750 / 1200 / 1500 mm (× 7/12 = 0.583), with the caveat recorded
>    in the config that prorating overstates effective seasonal rain in winter-rainfall
>    climates such as the Nile Delta.
> 3. **Soil texture** is scored from the **USDA texture class** derived from sand + silt +
>    clay (clay % alone was rejected as indefensible for cotton), with per-class
>    memberships listed in the config.
>
> The class breaks are 0.35 / 0.55 / 0.75 (§6.6 table below is the draft value).

**Weights are an EXPERIMENTAL WEIGHTING SCHEME, not literature-derived.** Rationale (documented,
not measured): temperature and water are the two first-order controls on cotton (they are the two
EcoCrop parameters that decide suitability outright); soil reaction and texture are second-order
(manageable by amendment/drainage); slope is minor for cotton except for mechanisation and
drainage. The scheme is visible in `config/crops/cotton.yml`, overridable, and reported in every
answer so a reviewer can challenge it.

### 6.6 Classification
| Class | Rule (threshold = **Experimental screening threshold**) |
|---|---|
| Highly suitable | S ≥ 0.75, no veto, no unverified-input cap |
| Moderately suitable | 0.55 ≤ S < 0.75, or capped by the unverified-input rule |
| Marginal | 0.35 ≤ S < 0.55 |
| Unsuitable | S < 0.35, or a hard land-cover constraint, or a critical veto |
| Insufficient data | any critical factor missing / no valid cells |

Classes only — **no "87.42 % suitable"**. Area fractions per class are reported.

### 6.6.1 The class depends on the score alone — no automatic "limiting-factor downgrade"

**Revision (approved):** the earlier proposal to drop one class whenever
`min(s_f) < 0.35` has been **removed**. A result is **not** demoted just because its weakest factor
membership happens to be low: that threshold had no separately justified scientific basis, and it
silently overrode the weighted score and the declared experimental weights, which the reader can see
and challenge.

What remains, as **four separate, individually justified rules**:

| Rule | Trigger | Scientific basis | Effect |
|---|---|---|---|
| Hard land-cover gate | built-up / water / wetland | Not land that can carry a crop at all | `Unsuitable` |
| Critical-factor veto | a critical factor is **outside its absolute tolerance** (s = 0) | FAO-style "N — not suitable": the crop cannot complete its cycle | `Unsuitable`, regardless of S |
| Unverified-input cap | a mandatory factor is `ASSUMED` (irrigation) | Overclaiming control: an unverified input must not yield a top class | max `Moderately suitable`, confidence Low |
| Missing data | a critical factor is unavailable | Never fabricate, never zero-fill | `INSUFFICIENT_DATA` — no score |

Everything between "absolute limit" and "optimum" is therefore handled by the **weighted mean
alone** — the score *is* the model.

**Known trade-off, stated openly:** a weighted mean is *compensatory* — strong factors can partly
offset a weak one, whereas Liebig's law of the minimum says the weakest factor governs. Concretely,
if water scores 0.1 and every other factor scores 1.0, the weighted mean is
0.30·0.1 + 0.70·1.0 = **0.73 → "Moderately suitable"**, even though water is nearly absent. The
critical veto only catches the s = 0 case. This is **not** hidden: every answer and every map
tooltip names the limiting factors with their values and memberships, and §11.10 records the
limitation. If the weighted mean alone ever proves too permissive, the fix is a *separately
justified* rule (e.g. a crop-specific minimum water requirement expressed as a veto threshold),
not an arbitrary 0.35 cut on the membership scale.

**Limiting factors are still reported — that was the point of the analysis.** For every result the
pipeline returns the factor(s) with the lowest membership, with their raw value, membership, and the
stated tolerance band, e.g.:

> "Cotton suitability is limited primarily by **water supply** (73 mm/yr; required 450–1500 mm)."

Only factors actually computed can be named; unassessed ones (salinity, irrigation) appear as
caveats, never as reasons.

### 6.7 Aggregation to the ROI
1. Per-cell: compute s_f, S, class → the **suitability raster** (for the map).
2. ROI-level: aggregate each factor over valid cells (**mean**, and majority class for land cover),
   then run the *same* scoring pipeline once → ROI score, class, limiting factors.
   (Means-then-scoring ≠ scoring-then-averaging; both the ROI class and the per-class area
   fractions from the raster are reported, and the difference is documented.)
3. `valid_fraction` = valid cells / ROI cells.

---

## 7. Missing-data handling (never a silent zero)

| Situation | State | Behaviour |
|---|---|---|
| ROI not drawn | `NEEDS_ROI` | No computation, prescribed message. |
| No overlap with a required dataset / no valid cells | `INSUFFICIENT_DATA` | No score. |
| A **critical** factor missing | `INSUFFICIENT_DATA` | No score (never defaulted to 0). |
| An **optional** factor missing | `PARTIAL_DATA` | Weight removed, remaining weights renormalised, factor listed in `missing_factors`, confidence reduced. |
| Water assumed (irrigated scenario) | `PARTIAL_DATA` + `ASSUMED` flag | Weight removed, class capped at *Moderately suitable*, confidence = Low, assumption text mandatory. |
| A factor is all-nodata over the ROI | `INSUFFICIENT_DATA` for that factor | Same as missing. |
| Data exists but is ≥10× coarser than the ROI | `OK` + warning | Confidence reduced; a note that the layer is effectively constant across the ROI. |

Every result carries `valid_factors`, `missing_factors`, `assumed_factors`, `valid_fraction`,
`confidence` (High/Medium/Low) and `warnings`. The **map distinguishes Unsuitable from
Insufficient data** (own colour + own legend entry).

## 8. Result model & router integration

```python
@dataclass
class CropSuitabilityResult:
    crop: str
    scenario: str                 # "rainfed" | "assumed_irrigated"
    score: float | None           # 0..1, None when insufficient
    classification: str           # Highly / Moderately / Marginal / Unsuitable / Insufficient data
    confidence: str               # High | Medium | Low
    valid_fraction: float
    factor_scores: dict[str, float | None]     # per-factor membership
    factor_values: dict[str, float | None]     # raw factor values (units in config)
    limiting_factors: list[LimitingFactor]     # name, value, membership, why-limiting
    missing_factors: list[str]
    assumed_factors: list[str]
    assumptions: list[str]
    provenance: list[ProvenanceRecord]         # dataset, url, variable, units, res, CRS, period, licence, access date, processing
    warnings: list[str]
    temporal_basis: str                        # "climatological + static (mixed)"
    suitability_raster: np.ndarray | None      # classified codes + mask, native analysis grid
    class_fractions: dict[str, float]
```
It is returned inside the Phase 7 `AnalysisExecution` (data/answer/warnings/artifacts), so **the
router executes it with no UI-specific logic**: add a handler, add patterns, flip the registry row —
the Phase 7 extensibility test already proves no router rewrite is needed. Context status mapping:
no ROI → `NEEDS_ROI`; no overlap → `INSUFFICIENT_DATA`; optional factor gone → `PARTIAL_DATA`;
all good → `OK`. Any crop ≠ cotton → *"Only cotton suitability is currently supported."*

## 9. Map & explanation
- Overlay on the ROI in the **common analysis grid** (reprojected to EPSG:3857 **only for web display**).
- Own 5-class palette and legend (Highly / Moderately / Marginal / Unsuitable / Insufficient data) — **never the NDVI colour scale** (a green "Highly suitable" next to a green NDVI would be read as the same thing).
- Analysis always runs on the native/common grid; only the display copy is warped.
- The answer always names the **limiting factors**, derived from the computed memberships:
  *"Cotton suitability is limited primarily by **water supply** (73 mm/yr vs 450–1500 mm required)."*
  Only factors actually computed can be named; unassessed ones appear as caveats, never as reasons.

## 10. Expected result at this AOI (VERIFIED — see §14 for the measurements)

| Scenario | Expectation |
|---|---|
| Rainfed | **Unsuitable** — critical veto on water (73 mm ≪ 450 mm abs min). Temperature (GDD ≈ 1800, well above 1400), pH 7.1 (optimum), slope ≈ 0 % (optimum), clay 36.7 % (mild) are all fine. |
| Assumed-irrigated | **Moderately suitable (capped)** — remaining factors score high, but water is `ASSUMED` and **salinity is `INSUFFICIENT_DATA`**, so confidence = Low and the class cannot exceed Moderately suitable. |

This is exactly the nuance a defensible demo should show: agronomically fine land whose limiting
factor is water, surfaced instead of hidden.

**Measured, not expected** (`scripts/verify_phase8.py`, 15/15 checks): rainfed **Unsuitable**,
confidence High, the class coming from the *critical-factor veto* and not from the score
(weighted score 0.66); irrigation-assumed **Moderately suitable**, score 0.94, confidence Low,
capped by the assumed-input rule.

## 11. Scientific limitations (stated in the UI, not just here)
1. Not a yield, profit, or crop-success prediction; not a farmer recommendation; not a soil-fertility or irrigation-sufficiency diagnosis.
2. Soil inputs are **model predictions** (SoilGrids), not laboratory measurements.
3. Climate inputs are **1970–2000 normals** — not the season being asked about; GDD is approximated from monthly means (daily extremes are smoothed away).
4. **Salinity is not assessed** — a first-order constraint for irrigated delta cotton.
5. **Irrigation is assumed, never verified**; no irrigation dataset is used.
6. DEM is a DSM (surface, not bare ground); slope over flat terrain is within DEM noise.
7. Climatology pixels (~1 km) and soil pixels (250 m) are coarse relative to field-scale decisions; a 20 km AOI may contain only a handful of independent climate values.
8. Weights are experimental; thresholds are screening thresholds; the output is one class among five, not a precise score.
9. Validation requires local agronomic/field information — the UI says so.
10. **The weighted score is compensatory.** Below the absolute-tolerance veto, a very weak factor is
    partly offset by strong ones (§6.6.1), so a "Moderately suitable" result can still sit on a
    nearly-absent factor. Limiting factors are always named; the veto thresholds, not the weights,
    are what protect against severe single-factor failure.

## 12. Implementation plan (COMPLETED)
1. `core/datasources/` — per-source fetchers (windowed, cached, provenance-writing).
2. `core/alignment.py` — common-grid definition + typed reprojection/resampling.
3. `config/crops/cotton.yml` — all thresholds/weights/units/sources (no numbers in Python).
4. `core/suitability.py` — membership, veto, weighted mean, caps, classes (Streamlit-free).
5. `analyses/crop_suitability.py` — handler + `CropSuitabilityResult` + answer/explanation text.
6. Registry flip + router patterns (no router rewrite).
7. Synthetic tests first (18 cases, hand-calculable rasters), then real data, then Playwright.

Built as above. Files: `core/datasources/{base,worldcover,soilgrids,worldclim,copernicus_dem}.py`,
`core/alignment.py`, `core/suitability.py`, `config/crops/cotton.yml`,
`analyses/crop_suitability.py`, `ui/map.py` (`suitability_legend_html`),
`ui/components.py` (`render_crop_suitability`), `app.py` (overlay + chat branch).
No number that changes a result lives in Python: every threshold, weight, class break,
land-cover rule and the growing-season window are read from `config/crops/cotton.yml`.

---

## 13. Where each of the 30 methodology corrections lives

| # | Correction | Implemented in |
|---|---|---|
| 1 | Screening framing everywhere | `analyses/crop_suitability.py` answer text; panel title "Experimental crop-suitability screening"; legend "Screening classes only — not a recommendation"; closing disclaimer |
| 2 | Analysis / native / **effective** resolution | `core/alignment.py::make_grid` (requested vs effective + `note`), `core/suitability.py` `_metres()` warning when a layer is ≥4× coarser than the grid, `native_resolutions` = {land cover 10 m, soil 250 m, climate ~1 km, topography 30 m} |
| 3 | GDD is a documented approximation | `config/crops/cotton.yml` `gdd_base_temp_c: 15.6` marked "context metric only — never scored"; `core/datasources/worldclim.py` monthly-mean thermal sum; unit-tested |
| 4 | Growing-season-aware rainfall | `growing_season_months` in `cotton.yml`; the engine reports annual **and** growing-season precipitation plus the months used |
| 5 | Assumed irrigation = hypothetical sensitivity analysis | Scenario B: `water_treatment: assumed`, confidence Low, `assumed_factors`, max class *Moderately suitable*, prominent `st.warning`; Scenario A rainfed is primary |
| 6 | Every threshold validated, or labelled | `config/crops/cotton.yml` carries `threshold_source` + `threshold_status` per item; `core/suitability.py::threshold_provenance()` emits a separate table with `threshold_type` (`optimum+absolute` / `weighting scheme` / `classification breaks` / `temporal window`) and status `literature-backed` / `experimental` / `assumed` |
| 7 | Weighted score + critical veto; no unjustified demotion | §6.6.1: the automatic "min membership < 0.35 → drop one class" rule was **removed** |
| 8 | Three factor categories | `category:` per factor in `cotton.yml` — `hard_constraint` (water / built-up / excluded land cover), `critical` (temperature, water, pH), `supporting` (texture, slope); the answer names which category excluded |
| 9 | Configurable land-cover policy | `land_cover_policy:` in `cotton.yml` (EXCLUDE / ALLOW / CONTEXT class lists); cropland is never read as "currently cotton" |
| 10 | Texture from sand+silt+clay | USDA texture class from the three fractions, labelled **"Soil texture proxy (USDA class)"** |
| 11 | No soil-suitability inference from pH+texture | Salinity, rooting depth and drainage are reported as *not assessed*; computed limiting factors and unassessed risks are separate lists |
| 12 | Elevation is context, not a factor | DEM used for slope + elevation context only; no elevation factor in `cotton.yml` |
| 13 | Slope methodology stated | `core/alignment.py::compute_slope` — DEM reprojected to the metric analysis grid first, Horn 3×3, spacing from the affine, edge replication documented (outer row/column ≈ half the true gradient), NaN propagates, output in % |
| 14 | Configurable grid + honest cap | `analysis_resolution_m: 30`; the 2,000,000-cell cap doubles the cell size and reports requested vs effective with the reason |
| 15 | Memory safety | ROI → bbox → windowed read → reproject subset → release (`read_sources_into_grid`); measured peak 250 MB on a cold network run |
| 16 | Cache + provenance, never committed | `data/external/<source>/…tif` + `.json` sidecar; `CACHE_KEY_FIELDS` = dataset, version, variable, source URL, AOI signature, resolution, resampling, band; `.gitignore` excludes `data/external/` |
| 17 | Machine-readable provenance per factor | `SourceRecord` (14 fields) **plus** the threshold-provenance table |
| 18 | Extended result model | `CropSuitabilityScreening` carries crop, scenario, score, classification, confidence, factor scores/values, computed limiting factors, missing/assumed factors, annual + growing-season precipitation, growing-season months, native resolutions, analysis resolution, valid fraction, class fractions, provenance, threshold provenance, assumptions, warnings, temporal basis, suitability raster — no raw arrays in the chat response |
| 19 | Wording comes from the evidence | Answer text is composed from computed values; missing data is never phrased as a negative |
| 20 | Registry activation only | `analyses/registry.py` flips `CROP_SUITABILITY`; `core/router.py` is unchanged apart from moving the intent from planned to available (a test enforces that the router was not rewritten for this feature) |
| 21 | UI order and honesty | ROI → ask → scenario / class / confidence / score / limiting factors / missing-unverified → map → expandable "Why?" → expandable "Data & methodology" → disclaimer; the panel never looks certified |
| 22 | Own legend, five classes | `ui/map.py::suitability_legend_html` + `suitability_rgba`, a blue/teal → grey palette that cannot be confused with the NDVI ramp; *Unsuitable* (measured and limiting) is distinguished from *Insufficient data* (not measured) |
| 23 | Synthetic tests first | `tests/test_phase8_suitability.py` (~30 hand-calculable cases), `tests/test_phase8_alignment.py` (grid, cap, ROI mask, categorical nearest-neighbour, slope 18a–18f), `tests/test_phase8_router.py` (23 routing/crop-policy cases, no network) |
| 24 | Real-data validation | `scripts/verify_phase8.py` — the engine vs an **independent** recomputation on the Nile Delta AOI: 15/15 |
| 25 | Genuine browser interaction | `scripts/browser_test_phase8.py` — 44 checks with a real mouse and keyboard, including "Can I grow rice here?" |
| 26 | Phase 1–8 regression | `pytest tests -q` → 333 passed; the NDVI workflow and the Phase 7 router tests are unchanged in behaviour |
| 27 | Timing reported separately | `result.performance` + `scripts/performance_phase8.py` (cold and warm), see §14 |
| 28 | Deliberately NOT built | No ML, no yield or profit model, no recommendations, no live weather, no irrigation detection, no salinity estimation, no economics, no phenology/time-series, no Sentinel-1, no LLM/VLM in the scoring path |
| 29 | Architecture | `core/datasources/`, `core/alignment.py`, `core/suitability.py`, `config/crops/cotton.yml`, `analyses/crop_suitability.py` — all Streamlit-free; only `ui/` and `app.py` touch Streamlit |
| 30 | Final report | Sections A–O in `docs/PHASE8_REPORT.md` |

---

## 14. Verification evidence

### 14.1 Real data, Nile Delta AOI (`scripts/verify_phase8.py`, 15/15)

Engine vs **independent** recomputation (plain rasterio windowed reads and plain numpy,
written without looking at the implementation):

| Quantity | Engine | Independent |
|---|---|---|
| WorldCover dominant class | 40 (cropland) | 40 |
| pH (0–5 cm) | 7.071 | 7.11 |
| Clay / sand / silt | 37.3 / 35.4 / 29.1 % | 35.5 / 35.4 / 29.1 % |
| Growing-season rain (Apr–Oct) | 13.21 mm | 13.2 mm |
| Annual rain | 73.22 mm | 73.5 mm |
| Slope | 0.719 % | 1.05 % (`np.gradient`) |
| Approx. GDD₁₅.₆ | 1796 | 1797 |
| Weighted score | 0.66 | **0.6572 hand-calculated** |

Outcome: rainfed **Unsuitable** (critical-factor veto on water, confidence High);
irrigation-assumed **Moderately suitable** (score 0.94, confidence Low, capped).
Full output: `artifacts/phase8_verification.json`.

### 14.2 Browser (`scripts/browser_test_phase8.py`, 44/44)

A real Chromium session draws an ROI with the mouse, types "Can I grow cotton here?",
and the app shows the routed intent, the screening panel, both scenario tabs, the
missing-data language, the "Why?" and "Data & methodology" explainers, the overlay
checkbox, the map layer and the five-class legend. "Can I grow rice here?" returns
"Only cotton suitability is currently supported." and computes nothing.

### 14.3 Performance (`scripts/performance_phase8.py`)

105 × 105 grid @ 30 m = 11,025 cells.

| Stage | Cold (network) | Warm (cached) |
|---|---|---|
| Source access (opening remote sources) | 138.6 s | 0.000 s |
| Windowed read + reprojection | 178.1 s | 0.000 s |
| Read back from local cache | 0.0 s | 0.093 s |
| — land cover / soil / climate / terrain | 3.4 / 165.9 / 143.5 / 4.1 s | 0.022 / 0.032 / 0.069 / 0.005 s |
| Factor computation | <0.001 s | <0.001 s |
| Suitability scoring + classification | 0.009 s | 0.009 s |
| Map overlay (display only) | 0.012 s | 0.012 s |
| Panel render | 0.365 s | 0.346 s |
| **Total (engine)** | **316.9 s** | **0.147 s** |
| **Peak process RSS** | **250 MB** | **174 MB** |

The cost is entirely in the first fetch of public data; the analysis itself is
milliseconds. Both are far inside the ≈2 GB budget.
