# SatQuery AI

**Explore satellite imagery and obtain evidence-backed geospatial insights through natural-language queries.**

SatQuery AI loads a GeoTIFF/COG, shows it on a 3D globe, and answers questions about it —
vegetation indices, water indices, change between two dated acquisitions, spatial conditions
(proximity, land cover, index ranges), and combinations of those conditions. Every answer is
computed from the raster's own pixels and is returned with the evidence behind it: what was
measured, what was assumed, what remained unknown, and what the result does **not** claim.

---

## Contents

- [Main capabilities](#main-capabilities)
- [What SatQuery AI does not do](#what-satquery-ai-does-not-do)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the application](#running-the-application)
- [Why `serve.py` and not `streamlit run app.py`](#why-serve-py-and-not-streamlit-run-app-py)
- [How the map tile proxy works](#how-the-map-tile-proxy-works)
- [The globe and the flat map](#the-globe-and-the-flat-map)
- [Place search](#place-search)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Tests and verification](#tests-and-verification)
- [Data, attribution and licensing](#data-attribution-and-licensing)
- [Limitations](#limitations)

---

## Main capabilities

**Imagery**
- GeoTIFF / COG ingestion with metadata, CRS, transform and quality checks.
- True-colour and false-colour composites with reflectance-aware stretching.
- Band-role detection from metadata (with its confidence and the evidence used), plus an
  explicit confirmation step before any index is computed.

**Indices and measurement**
- NDVI and NDWI as **continuous measurements on the native raster** — never on a resampled
  preview, never with a silent threshold.
- Statistics for a drawn area (count, mean, median, min/max, standard deviation, percentiles),
  reported together with how many cells were valid, invalid or outside the selection.

**Change over time**
- Choose two dated acquisitions; NDVI change is computed cell by cell over the interval you
  select. The application never picks a date for you.

**Spatial and multi-condition reasoning**
- Conditions on proximity, land cover and index ranges, evaluated per cell.
- Several conditions combined into one answer, with each condition reported separately and
  matches / measured non-matches / undecidable cells counted as distinct outcomes.

**Evidence**
- Every result carries an evidence package: the question as normalised, the data used, the
  thresholds and where each one came from, the counts behind the answer, the limits, and a
  JSON export of the whole package.

**Map**
- A 3D spherical globe as the primary map: drag to rotate, zoom towards a location, switch
  between satellite and street imagery, search for any place on Earth, draw an analysis area.
- A flat, projected detail mode for pixel-precise drawing and inspection (see below).
- Raster overlays (true colour, false colour, NDVI, NDWI, change, composed conditions,
  evidence layers) drawn in their true geographic position.

## What SatQuery AI does not do

- No real-time or near-real-time satellite imagery: SatQuery analyses the raster you load.
- No flood detection, crop-failure prediction, yield estimation or land-cover change detection.
- No causal attribution. Results are **geographic evidence, not causal explanations**.
- No prediction, forecasting or anomaly detection.
- No automatic selection of dates, thresholds or band mappings on the user's behalf: where a
  value must be supplied, the application asks instead of guessing.
- No invented data. Insufficient or missing data is reported as such — never as zero, never as
  a "no match".

## Requirements

| | |
|---|---|
| **Python** | 3.11 or newer (developed and tested on 3.13) |
| **Key OS packages** | GDAL — bundled in the `rasterio` wheels on Linux, macOS and Windows |
| **API keys** | **None.** No key, token or account is required for anything |
| **Network** | Needed on first use for base-map imagery and place search (see below) |

## Installation

```bash
git clone https://github.com/PrafulJawale/sihsat.git
cd sihsat

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

External datasets (land cover, soil, climate, elevation) are **not** bundled. They are downloaded
on first use by the modules in `core/datasources/` when a spatial condition needs them, cached
under `SATQUERY_EXTERNAL_DIR`, and described by a provenance sidecar that records source,
licence and retrieval. Until a dataset has been fetched, the conditions that need it report
insufficient data instead of a substitute.

## Running the application

```bash
python serve.py
```

Then open <http://localhost:8501>. The bundled Sentinel-2 sample scene is selected by default, so
the application is usable immediately.

Useful flags (passed straight through to Streamlit):

```bash
python serve.py --server.port 8600
```

## Why `serve.py` and not `streamlit run app.py`

`streamlit run app.py` starts the application but mounts **no** additional HTTP routes. The map
needs one: a same-origin proxy that serves base-map tiles. `serve.py` wraps Streamlit's internal
Starlette application factory, adds that single route, and then starts Streamlit unchanged:

```
/satquery-tiles/<provider>/<z>/<x>/<y>.png
```

Consequences of running `streamlit run app.py` instead:

- the globe renders its sphere and the raster overlays, but **the satellite/street backdrop is
  missing**; the interface says so explicitly and every analysis keeps working;
- place search and every measurement are unaffected.

## How the map tile proxy works

1. The browser asks **this application** for a tile, never a tile provider:
   `/satquery-tiles/satellite/5/16/10.png`.
2. The server fetches that tile from the provider, caches it on disk, and returns it with
   cache headers. Subsequent requests are answered from cache.
3. Providers are declared in one place — `tileserver.py::PROVIDERS`:

   | Key | Label | Source |
   |---|---|---|
   | `osm` | Streets | OpenStreetMap standard tiles |
   | `satellite` | Satellite | Esri World Imagery |

   Both are used with attribution, on the provider's terms, for low-volume interactive use.
4. A "Light" basemap that required an API key existed earlier and has been **removed**; nothing
   in the application can request it.

Because the proxy is same-origin, a locked-down browser or an offline-ish network cannot break
the map with a third-party 403, and the browser never holds a provider credential.

## The globe and the flat map

There is **one** map surface.

- **Globe (default).** A real 3D sphere (CesiumJS, loaded from a pinned CDN version), with
  drag-to-rotate, zoom-towards-location, satellite/street imagery, raster overlays, the scene
  footprint, the current selection, and a pin for a searched place.
- **Flat map (detail mode).** The projected, north-up view — the right surface for pixel-precise
  rectangle/polygon drawing and inspection. It is opened deliberately with **Map view → Flat
  map** (or **2D map** on the globe), and while it is open the globe is not rendered.

The two never render at the same time. Overlays use identical bounds in both, so a layer cannot
drift between views, and a selection drawn on the globe is handed to the analysis code in exactly
the same form as one drawn on the flat map.

If the 3D engine cannot load (no CDN access, WebGL unavailable), the map falls back to a
2D-rendered orthographic globe using the same proxied imagery, and offers the flat mode for
overlays and drawing.

## Place search

- Global, server-side, through the public **OpenStreetMap Nominatim** geocoder
  (`core/geocode.py`), with a descriptive User-Agent and a one-request-per-second limit.
- Accepts cities, regions, countries, landmarks and addresses where the geocoder supports them;
  several matches are listed and one is chosen.
- Moving the map is **navigation only**: it never reads the raster, changes the selection, or
  runs an analysis. No coordinates are invented — no match is reported as no match.

## Project structure

```
app.py                  Streamlit application: sections, map, chat, evidence
serve.py                Launches Streamlit WITH the same-origin tile proxy mounted
tileserver.py           Tile proxy: providers, disk cache, transparency placeholder
requirements.txt        Runtime dependencies

core/                   Analysis core (no UI, no Streamlit)
  raster.py             Windowed GeoTIFF/COG reads, metadata, CRS/transform handling
  bands.py              Band-role detection from metadata + evidence
  reflectance.py        Reflectance scaling and validation
  indices.py            NDVI / NDWI computation on native pixels
  index_definitions.py  Declarative index definitions (band roles, ranges)
  statistics.py         Native-raster ROI statistics
  geometry.py           CRS/affine conversions, pixel↔world mapping
  geo.py                Reprojection for display ONLY, PNG encoding
  roi.py                Drawn geometry -> validated selection (intersection in raster CRS)
  spatial.py            Proximity and land-cover masks (exact distance transform)
  spatial_query.py      Per-cell spatial conditions
  multi_condition.py    Composition of several conditions, masks and counts
  temporal.py           Scene discovery, dated scene pairs, change
  suitability.py        Experimental crop-suitability screening
  alignment.py          Grid alignment checks between acquisitions
  evidence.py           Evidence packages for results
  geocode.py            Nominatim place search (throttled, attributed)
  router.py             Text -> registered analysis (routing only, never computation)
  models.py             Result objects
  preview.py            Decimated preview reads for display
  samples.py            Bundled sample discovery and provenance
  datasources/          Fetchers + provenance for external datasets

analyses/               The registered analyses the router may select
  registry.py           What is available, what is not, and why
  ndvi.py  ndwi.py  ndvi_change.py  spatial_query.py  multi_condition.py
  crop_suitability.py  evidence.py  evidence_masks.py  base.py

ui/
  theme.py              Design system (typography, spacing, surfaces, states)
  components.py         Shared rendering blocks, welcome screen, chat turns
  map.py                Flat detail map (folium) and overlay objects
  evidence_panel.py     Evidence panel, layers, JSON export
  globe.py              Globe map wrapper: argument building and validation
  globe_frontend/       The globe/mapping component (no build step)

config/                 Data-driven configuration (indices, crops, spatial, temporal, multi)
scripts/                Verification, browser and maintenance scripts
tests/                  Automated tests (pytest)
docs/                   Architecture and design notes
data/sample/            Bundled sample scenes + provenance sidecars
assets/                 Product mark (favicon, logo, header)
.streamlit/config.toml  Streamlit theme and client settings
```

## Configuration

**Nothing is required.** There are no API keys and no mandatory environment variables.

Two optional paths can be overridden (see `.env.example`, names only, no values):

| Variable | Default | Purpose |
|---|---|---|
| `SATQUERY_TILE_CACHE` | `/tmp/satquery_tile_cache` | Where proxied map tiles are cached |
| `SATQUERY_EXTERNAL_DIR` | `data/external` | Where fetched external datasets are cached |

## Tests and verification

```bash
# unit / integration / AppTest suite (~800 tests, ~2 minutes)
python -m pytest -q
```

Browser verification (needs the application running at `http://127.0.0.1:8501` and Playwright
with Chromium: `pip install playwright && playwright install chromium`):

```bash
python serve.py                                   # in one terminal
python scripts/verify_ui_final.py                 # 38 checks: UI, globe, search, overlays, ROI
python scripts/verify_phase12_browser.py          # 35 checks: multi-condition composition
python scripts/verify_phase13_browser.py          # 30 checks: evidence generation
python scripts/verify_phase13_real_data.py        # 29 checks: evidence against the real scene
python scripts/verify_map_display.py              # 13 checks: the map really renders imagery
python scripts/inspect_ui.py                      # page inventory / wording audit for humans
```

Browser scripts drive the running application the way a user does; those that inspect Leaflet
layers open the flat detail mode first, because the globe is the default surface.

## Data, attribution and licensing

- Base map: © OpenStreetMap contributors (Streets) and Esri World Imagery (Satellite), used
  through this application's own proxy with attribution on screen.
- Place search: OpenStreetMap via Nominatim, used within its usage policy (low volume,
  descriptive User-Agent, at most one request per second).
- Bundled sample: Sentinel-2 MSI Level-2A surface reflectance (Copernicus Sentinel data),
  windowed to 2048 × 2048 px at 10 m; `data/sample/*.provenance.json` records the exact source,
  processing level and what the data is and is not valid for.
- External datasets fetched at runtime are **not** committed; each carries a provenance sidecar.

## Limitations

- **The raster is the boundary.** Analysis runs only where the loaded, georeferenced raster has
  data. Elsewhere the application says so; it never extrapolates.
- **Display versus analysis.** Previews, overlays and composites are resampled for the screen.
  Every number in an answer is computed on the native raster; the interface keeps the two apart.
- **External datasets.** Spatial conditions that need land cover, soil, climate or elevation
  data download those datasets on first use (see `core/datasources/`) and cache them locally;
  if a download is unavailable, the affected condition reports insufficient data rather than a
  substitute.
- **Unsupported questions are refused.** Flood change and temporal NDWI, for example, are
  registered as *not available* and are answered as such — never approximated.
- **The globe needs a CDN on first load** for the 3D engine (pinned version), with a
  2D-rendered fallback; base-map imagery also needs network access unless tiles are cached.
- **Crop-suitability screening is experimental**: a transparent, weighted screening of the
  conditions it states — not a recommendation, yield prediction or soil diagnosis.
