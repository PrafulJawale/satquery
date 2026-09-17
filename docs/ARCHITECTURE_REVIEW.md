# SatQuery AI — Architecture Review & Risk Register

Reviewed: project brief (Phases 1–7 + future architecture) and the proposed
`satquery/` layout. Verified against `rasterio 1.5.1 / GDAL 3.12.4` on the
bundled sample GeoTIFFs.

Severity: 🔴 will produce wrong numbers · 🟠 will produce wrong geometry or
crash on real data · 🟡 will slow you down or mislead users.
"Phase" = when it must be fixed.

---

## A. What is already right (keep these decisions)

| Decision | Why it's right |
|---|---|
| Python + Streamlit + rasterio + NumPy first | Streamlit gives a usable UI in ~100 lines; rasterio is the correct GDAL binding. No React yet is correct for a prototype. |
| "NL query → geospatial analysis", not "AI describes an image" | This is the actual product. It also keeps the LLM out of the maths. |
| Modest first pipeline (ingest → preview → NDVI → map → ROI → stats) | Each step is independently verifiable. |
| "Do not blindly resize to 504×504" | Correct. Resizing destroys both area statistics and the geographic link. |
| "Preserve CRS / transform / extent" | Correct, and it is the single most-violated rule in hobby geospatial code. |
| "Do not fabricate data / no fake accuracy" | Non-negotiable for an SIH demo; judges check. |
| Modular `core/` + `ui/` split | Right instinct — I sharpen it below. |

---

## B. Technical mistakes and hidden traps

### 🔴 T1 — NDVI from 8-bit or display-stretched data is not NDVI
NDVI is a *ratio of surface reflectances*. A GeoTIFF visual product has been
stretched (often non-linearly, per-band) for human eyes. Compute NDVI from it
and you get numbers in the right range with no physical meaning.

This matters for your own sample data: `RGB.byte.tif` is 8-bit. The app already
flags it (`8-BIT DATA` warning) and `Provenance.bands_are_physically_valid` is
`False` for it.

**Fix (Phase 3):** compute indices only from analysis-ready data
(Sentinel-2 **L2A**, Landsat Collection-2 **L2SP/L2SR**). Refuse, with an
explanation, when the dtype is 8-bit.

### 🔴 T2 — Reflectance scaling: multiplicative cancels, additive does not
NDVI = (NIR−Red)/(NIR+Red) is invariant to a *multiplicative* factor, which
makes people careless. It is **not** invariant to an *additive* offset:

- Sentinel-2 L2A: `ρ = DN / 10000`
- Landsat 8/9 C2 L2SP: `ρ = DN × 0.0000275 − 0.2`  ← **offset −0.2 exists**

Skip the offset and healthy vegetation silently shifts. **Fix (Phase 3):** read
`scale_factor` / `add_offset` from the file (or apply the known sensor
constant), convert to reflectance *first*, then compute indices, in `float32`.

### 🔴 T3 — Band count and band order are not band meaning
"Multispectral" does not tell you which band is NIR. Even the same sensor
ships different band stacks (Sentinel-2 L2A COGs are single-band-per-file;
Landsat ARD is one file per band; some tools stack, resample and reorder).
Our own `rgb1_fake_nir_epsg3857.tif` is named "NIR" and has **one** band.

**Fix (Phase 3):** a `BandSpec` the user confirms. Use three *hints*, never a
silent assumption:
1. GDAL band **descriptions** (`all-nodata.tif` → `blue, green, red, nir`),
2. `colorinterp` (`RGB.byte.tif` → `red, green, blue`),
3. preset profiles (S2 L2A, L8/9 C2) offered as a *guess to confirm*.

### 🟠 T4 — Bands inside one stack can have different ground resolutions
Sentinel-2: B2/B3/B4/B8 = 10 m, B5/B6/B7/B8A/B11/B12 = 20 m, B1/B9 = 60 m.
If a user uploads a 6-band stack built from these, `width`/`height` differ per
band and `(NIR − Red)` fails or, worse, silently broadcasts.

**Fix (Phase 3):** validate `(width, height, transform, crs)` equality; if
mismatched, resample to a common grid and *say which method was used*
(`bilinear` for continuous data, never `nearest` for reflectance).

### 🟠 T5 — Web-map overlays need reprojection, not corner-fitting
Folium's `ImageOverlay` takes a PNG plus `[[lat_min, lon_min],[lat_max, lon_max]]`.
That rectangle is only correct when the raster is already in EPSG:3857/4326 and
*north-up*. Otherwise:
- rotated transforms (`b` or `d` ≠ 0) → wrong placement *and* wrong shape;
- UTM scenes → the lon/lat outline is a curved quadrilateral, not a rectangle;
- datums differ (NAD27 vs WGS84 ≈ tens of metres).

**Fix (Phase 4):** `rasterio.warp.calculate_default_transform` →
`reproject` to EPSG:3857 → use the *destination* transform's bounds for the
overlay. Our `rotated.tif` and `byte.tif` samples exist to prove the app
refuses to draw a wrong footprint instead of drawing one anyway.

### 🟠 T6 — Four-corner footprints understate coverage — ✅ fixed in Phase 1
See `densify_ring()`: 16 points per edge → 64 vertices before transforming.

### 🟠 T7 — Datum blindness — ✅ fixed in Phase 1
`byte.tif` is `EPSG:26711` (NAD27). A test asserts WGS84 bounds differ from the
native numbers by >1 unit, so "we forgot to reproject" fails the build.

### 🔴 T8 — Nodata, NaN and division by zero
`(NIR − Red)/(NIR + Red)` produces `0/0` on nodata pairs and explodes on
`−1/0`. Sentinel-2 L2A uses nodata `0`; Landsat fill is `0`. An unmasked mean
over a scene with black corners reports a perfectly plausible, entirely wrong
value. `all-nodata.tif` (every pixel nodata) is the test: a naive `mean()`
returns `0.0` and looks fine.

**Fix:** `np.errstate(divide="ignore", invalid="ignore")`, masked arrays, and
stats that always report **valid pixel count** alongside the mean. Already
wired into `window_stats()`; extend it in Phase 3/6.

### 🟠 T9 — Never buffer, measure distance or compute area in degrees
"Near water" means a 500 m buffer. Buffering an EPSG:4326 polygon by `0.005`
gives ~550 m at the equator and a different number 500 km north, and it is
*elliptical*, not circular.

**Fix (Phase 5/6):** buffer in a metric CRS, or use `pyproj.Geod` geodesic
buffers. For your AOI (Garhchiroli ≈ 80°E) the right projected CRS is
**UTM zone 44N = EPSG:32644**. For India-wide work consider the Survey of India
LCC (EPSG:7755) — verify it for your exact AOI before demoing.

### 🟠 T10 — Memory and the Streamlit rerun model
One Sentinel-2 band at 10 m = 10980² × uint16 ≈ **240 MB**. Four bands ≈ 1 GB.
Streamlit **re-executes the entire script on every widget interaction**.

**Fix:** cache the *description* and small derived arrays, never the open GDAL
handle; use overviews / `out_shape` decimated reads for anything displayed;
prefer COGs (tiled + overviews) so windowed reads are cheap. Already done in
`app.py` (`@st.cache_data` on metadata, windowed sanity read).

### 🟡 T11 — "Don't resize" ≠ "render the full-resolution array"
Correct rule, refined: **compute at native resolution, render decimated.**
A 10980×10980 PNG will kill the browser. Keep two things: the native-resolution
result array with its transform, and a downsampled *rendering* with its own
(separate) transform.

### 🟡 T12 — Streamlit upload mechanics — ✅ handled in Phase 1
`st.file_uploader` returns bytes, not a path → `rasterio.MemoryFile`.
Default upload cap is 200 MB (raised to 1 GB in `.streamlit/config.toml`).
Sentinel-2 SAFE `.jp2` files need GDAL's `JP2OpenJPEG` driver; if a file
won't open, the app says exactly that instead of failing silently.

### 🟡 T13 — "Real-time satellite imagery" is not achievable — and not needed
Sentinel-2 L2A typically appears on public STAC catalogues hours to ~2 days
after acquisition; revisit is 5 days (S2 A+B) / 16 days (Landsat). Say
"near real-time via public archives", never "real-time".
Realistic path later: **STAC search** (`earth-search`, Microsoft Planetary
Computer) → download the COG → run the exact same pipeline we're building now.
That's why `core/raster.py` must stay format-agnostic.

### 🔴 T14 — Crop suitability has no ground truth → no accuracy number
You listed vegetation, soil, rainfall, temperature, water, land use,
elevation/slope. Honest status of each:

| Factor | Realistic public source | Status |
|---|---|---|
| Soil (pH, OC, texture, depth) | SoilGrids 250 m (ISRIC) — REST/COG | ✅ global, usable |
| Rainfall | CHIRPS ~5 km (ClimateSERV API); IMD 0.25° gridded (registration) | ✅ usable |
| Temperature | WorldClim 2.1 (~1 km) — **climatology**, not this season; ERA5 (CDS, API key, slow) | ⚠️ partial |
| Elevation / slope | Copernicus DEM 30 m / SRTM 30 m (AWS open data) | ✅ usable |
| Land use / crop mask | ESA WorldCover 10 m, Esri/Dynamic World 10 m | ✅ usable |
| Water availability / proximity | JRC Global Surface Water (occurrence, 30 m) | ✅ usable |
| Flood-proneness | JRC GSW recurrence + DEM (HAND); official hazard maps are national/restricted | ⚠️ proxy only |
| **Ground truth (where cotton actually grows/yields)** | India: NRSC/Bhuvan, DES/APY district statistics, FASAL | ❌ the hard part |

Without ground truth you **cannot** quote an accuracy percentage — and your own
engineering rule #5 forbids inventing one.

**Honest MVP:** rule-based multi-criteria evaluation (FAO-style land
suitability: S1 / S2 / S3 / N) with literature-derived thresholds, *plus a map
of the limiting factor* ("suitable except for slope"). Label it
**experimental — not validated**. Adding a validation dataset later is what
turns it into a real product, and it's a great "future work" slide.

### 🟠 T15 — Optical flood detection fails exactly when floods happen
Monsoon cloud cover is near-total during Indian floods. Optical sensors see
clouds, not water. Additional traps:
- **NDWI** (McFeeters 1996, `(Green−NIR)/(Green+NIR)`) over-detects water in
  built-up areas → prefer **MNDWI** (Xu 2006, `(Green−SWIR)/(Green+SWIR)`) —
  needs SWIR (S2 B11 / L8 B6);
- a fixed `0.0` threshold is not transferable → use **Otsu** on the difference
  image, and report the threshold used;
- terrain shadow and cloud shadow look like water → mask with SCL / QA_PIXEL
  and exclude steep terrain;
- compare **same sensor, similar phenology**, or normalise radiometrically.
- The operational answer is **Sentinel-1 SAR (GRD, VV/VH)** — all-weather.
  Good "future work" line, and it's genuinely how flood services work.

### 🟠 T16 — VLM chips must keep their georeferencing
When Phase 9 chips a scene for a vision model: record the chip's `Window` and
derive its transform. Never convert a resized chip's pixel coordinates back
into geographic coordinates.

### 🟠 T17 — The LLM must never do the maths
The router extracts *structured intent* (analysis kind, bands, dates, ROI,
crop) → a deterministic function computes → the LLM only narrates the result
with the numbers it was handed. This also makes the system testable without an
LLM. Use a registry (`analyses/`) mapping an enum → callable.

---

## C. Proposed structure (small, justified changes)

```
satquery/
├── app.py                     # Streamlit entry point (thin)
├── core/                      # PURE Python. Must never import streamlit.
│   ├── models.py              # ✚ NEW: BandInfo, SpatialInfo, RasterInfo,
│   │                          #        Provenance, AnalysisResult
│   ├── raster.py              # ingestion, metadata, footprint, windowed reads
│   ├── preview.py             # ✚ Phase 2: stretch + RGB/FCC compositing
│   ├── indices.py             # Phase 3: NDVI, NDWI, MNDWI (+ scaling!)
│   ├── geo.py                 # Phase 4/5: reprojection, ROI masks, buffers
│   └── statistics.py          # Phase 6: zonal stats
├── analyses/                  # ✚ Phase 8: registry (name → callable) for the router
├── ui/
│   ├── components.py          # renders dicts only
│   └── map.py                 # Phase 4: folium builder
├── data/sample/               # + provenance.json  ✚ NEW
├── scripts/verify_phase1.py   # ✚ NEW: CLI verification harness
├── tests/                     # ✚ NEW: pytest
├── docs/
│   ├── ARCHITECTURE_REVIEW.md
│   └── PHASE1.md
├── requirements.txt
└── README.md
```

**Why these four changes:**

1. **`core/models.py`** — one vocabulary. Without `RasterInfo`/`AnalysisResult`,
   every new analysis invents its own return shape and the map code becomes a
   chain of `if analysis == ...`.
2. **`analyses/` registry** — the future query router can then map
   `intent → enum → callable` without importing modules dynamically or
   hand-writing a giant `if/elif`.
3. **`data/sample/provenance.json`** — makes "do not fabricate data" *machine
   checkable*: each sample declares `is_real_satellite_data`,
   `bands_are_physically_valid`, `good_for`, `not_good_for`.
4. **`scripts/` + `tests/`** — your brief asks "how do we verify it works?"
   for every phase. A CLI harness plus pytest is the answer, and it is how you
   stop Phase 5 from silently breaking Phase 3.

Everything else in your proposed layout was kept as-is.

---

## D. Phase plan (unchanged order, sharpened exit criteria)

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **1** ✅ | Ingestion + metadata + warnings | `verify_phase1.py` 0 failures; report = `rio info` |
| 2 | RGB / FCC preview (percentile stretch) | Preview is not washed out / not black; nodata transparent |
| 3 | NDVI (reflectance-aware, masked) | Hand-computed NDVI at 3 pixels matches code |
| 4 | Interactive map overlay | Overlay corners land within one pixel of `rio info` bounds |
| 5 | ROI drawing (polygon/rect/click) | Drawn polygon → pixel window is correct in a projected CRS |
| 6 | Zonal statistics | Stats over a rectangle of known size match a manual window read |
| 7 | End-to-end test | One real Sentinel-2 L2A COG run through 1→6 |
| 8+ | Query router, NDWI/change, suitability, VLM | Each behind the `analyses/` registry |

---

## E. Reference tables you will need from Phase 3

**Sentinel-2 L2A (COG) bands**

| Band | λ (nm) | GSD | Use |
|---|---|---|---|
| B02 | 490 | 10 m | Blue (water, RGB) |
| B03 | 560 | 10 m | Green (NDWI) |
| B04 | 665 | 10 m | **Red (NDVI)** |
| B05 | 705 | 20 m | Red-edge (NDRE) |
| B08 | 842 | 10 m | **NIR (NDVI, FCC)** |
| B11 | 1610 | 20 m | SWIR-1 (MNDWI) |
| B12 | 2190 | 20 m | SWIR-2 |
| SCL | — | 20 m | Scene classification (cloud/shadow mask) |

**Landsat 8/9 Collection-2 L2SP**

| Band | λ (nm) | GSD | Use |
|---|---|---|---|
| SR_B2 | 482 | 30 m | Blue |
| SR_B3 | 562 | 30 m | Green |
| SR_B4 | 655 | 30 m | **Red** |
| SR_B5 | 865 | 30 m | **NIR** |
| SR_B6 | 1609 | 30 m | SWIR-1 |
| QA_PIXEL | — | 30 m | Cloud / shadow / fill mask |

**Indices**

| Index | Formula | Notes |
|---|---|---|
| NDVI | (NIR−Red)/(NIR+Red) | Tucker 1979. Range −1…1. |
| NDRE | (NIR−RedEdge)/(NIR+RedEdge) | Less saturated in dense canopy. |
| NDWI | (Green−NIR)/(Green+NIR) | McFeeters 1996. Built-up over-detection. |
| MNDWI | (Green−SWIR)/(Green+SWIR) | Xu 2006. Preferred for water. |

**Do not memorise thresholds as universal truth.** "NDVI > 0.5 = healthy
vegetation" depends on sensor, phenology, atmosphere and land cover. When you
use one, print it on screen next to its source and call it a heuristic.

---

## F. Where to get real data for later phases

- **Microsoft Planetary Computer STAC** or **Earth Search (Element84)** — search
  Sentinel-2 L2A COGs by AOI + date, no key needed for the public collections.
- **AWS Open Data**: `sentinel-s2-l2a` (COGs), Copernicus DEM, ESA WorldCover.
- **ISRIC SoilGrids** — REST API / WCS for soil properties at 250 m.
- **ClimateSERV** — CHIRPS rainfall time series API.
- **JRC Global Surface Water** — water occurrence/recurrence, 30 m.
- Always keep the acquisition date, sensor and processing level next to every
  number you show. A geospatial result without provenance is an opinion.
