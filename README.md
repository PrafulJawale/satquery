# SatQuery AI

**Explore satellite imagery and obtain evidence-backed geospatial insights through natural-language queries.**

SatQuery AI loads a GeoTIFF/COG, shows it on a 3D globe, and answers questions about it—vegetation indices, water indices, change between two dated acquisitions, spatial conditions such as proximity, land cover and index ranges, and combinations of those conditions.

Every answer is computed from the raster's own pixels and is returned with the evidence behind it: what was measured, what was assumed, what remained unknown, and what the result does **not** claim.

---

## Contents

* [Main capabilities](#main-capabilities)
* [What SatQuery AI does not do](#what-satquery-ai-does-not-do)
* [Requirements](#requirements)
* [Installation](#installation)
* [Running the application](#running-the-application)
* [Why `serve.py` and not `streamlit run app.py`](#why-servepy-and-not-streamlit-run-apppy)
* [How the map tile proxy works](#how-the-map-tile-proxy-works)
* [The globe and the flat map](#the-globe-and-the-flat-map)
* [Place search](#place-search)
* [Project structure](#project-structure)
* [Configuration](#configuration)
* [Tests and verification](#tests-and-verification)
* [Data, attribution and licensing](#data-attribution-and-licensing)
* [Limitations](#limitations)

---

## Main capabilities

### Imagery

* GeoTIFF / COG ingestion with metadata, CRS, transform and quality checks.
* True-colour and false-colour composites with reflectance-aware stretching.
* Band-role detection from metadata, including confidence and the evidence used.
* An explicit confirmation step before any index is computed.

### Indices and measurement

* NDVI and NDWI as **continuous measurements on the native raster**—never on a resampled preview and never with a silent threshold.
* Statistics for a drawn area, including count, mean, median, minimum, maximum, standard deviation and percentiles.
* Results report how many cells were valid, invalid or outside the selection.

### Change over time

* Choose two dated acquisitions.
* NDVI change is computed cell by cell over the selected interval.
* The application never picks a date for the user.

### Spatial and multi-condition reasoning

* Conditions on proximity, land cover and index ranges, evaluated per cell.
* Several conditions can be combined into one answer.
* Each condition is reported separately, with matches, measured non-matches and undecidable cells counted as distinct outcomes.

### Evidence

Every result carries an evidence package containing:

* The normalised question.
* The data used.
* The thresholds and where each threshold came from.
* The counts behind the answer.
* The limitations.
* A JSON export of the complete evidence package.

### Map

* A 3D spherical globe as the primary map.
* Drag to rotate and zoom towards a location.
* Switch between satellite and street imagery.
* Search for any place on Earth.
* Draw an analysis area.
* Use raster overlays such as true colour, false colour, NDVI, NDWI, change, composed conditions and evidence layers.
* A flat, projected detail mode for pixel-precise drawing and inspection.

## What SatQuery AI does not do

* No real-time or near-real-time satellite imagery. SatQuery analyses the raster that you load.
* No flood detection, crop-failure prediction, yield estimation or land-cover change detection.
* No causal attribution. Results are **geographic evidence, not causal explanations**.
* No prediction, forecasting or anomaly detection.
* No automatic selection of dates, thresholds or band mappings on the user's behalf. Where a value must be supplied, the application asks instead of guessing.
* No invented data. Insufficient or missing data is reported as such—never as zero and never as a “no match.”

## Requirements

| Requirement         | Details                                                              |
| ------------------- | -------------------------------------------------------------------- |
| **Python**          | 3.11 or newer; developed and tested on 3.13                          |
| **Key OS packages** | GDAL is bundled in the `rasterio` wheels on Linux, macOS and Windows |
| **API keys**        | **None.** No key, token or account is required                       |
| **Network**         | Needed on first use for base-map imagery and place search            |

## Installation

```bash
git clone https://github.com/PrafulJawale/sihsat.git
cd sihsat

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

External datasets such as land cover, soil, climate and elevation data are **not** bundled.

They are downloaded on first use by the modules in `core/datasources/` when a spatial condition needs them. They are cached under `SATQUERY_EXTERNAL_DIR` and described by a provenance sidecar that records the source, licence and retrieval information.

Until a dataset has been fetched, the conditions that need it report insufficient data instead of using a substitute.

## Running the application

```bash
python serve.py
```

Then open:

```text
http://localhost:8501
```

The bundled Sentinel-2 sample scene is selected by default, so the application is usable immediately.

Useful flags can be passed directly through to Streamlit:

```bash
python serve.py --server.port 8600
```

## Why `serve.py` and not `streamlit run app.py`

`streamlit run app.py` starts the application but does not mount the additional HTTP route required by the map.

The map needs a same-origin proxy that serves base-map tiles. `serve.py` wraps Streamlit's internal Starlette application factory, adds that single route, and then starts Streamlit unchanged:

```text
/satquery-tiles/<provider>/<z>/<x>/<y>.png
```

If you run `streamlit run app.py` instead:

* The globe and raster overlays still render.
* The satellite/street backdrop is missing.
* The interface reports the missing backdrop explicitly.
* Place search and all measurements remain unaffected.

## How the map tile proxy works

1. The browser requests a tile from this application rather than directly from a tile provider:

   ```text
   /satquery-tiles/satellite/5/16/10.png
   ```

2. The server fetches the tile from the provider, caches it on disk and returns it with cache headers.

3. Subsequent requests are answered from the cache.

4. Providers are declared in one place: `tileserver.py::PROVIDERS`.

   | Key         | Label     | Source                       |
   | ----------- | --------- | ---------------------------- |
   | `osm`       | Streets   | OpenStreetMap standard tiles |
   | `satellite` | Satellite | Esri World Imagery           |

   Both providers are used with attribution and according to the provider's terms for low-volume interactive use.

5. A “Light” basemap that previously required an API key has been removed. Nothing in the application can request it.

Because the proxy is same-origin, a locked-down browser or an offline-style network cannot break the map with a third-party 403, and the browser never holds a provider credential.

## The globe and the flat map

There is **one** map surface.

### Globe — default

The globe is a real 3D sphere using CesiumJS, loaded from a pinned CDN version. It supports:

* Drag-to-rotate navigation.
* Zooming towards a location.
* Satellite and street imagery.
* Raster overlays.
* Scene footprint.
* Current selection.
* A pin for a searched place.

### Flat map — detail mode

The flat map is the projected, north-up view. It is intended for pixel-precise rectangle or polygon drawing and inspection.

It can be opened deliberately using:

```text
Map view → Flat map
```

or:

```text
2D map
```

on the globe.

While the flat map is open, the globe is not rendered.

The two views never render at the same time. Overlays use identical bounds in both views, so a layer cannot drift between views. A selection drawn on the globe is handed to the analysis code in exactly the same form as one drawn on the flat map.

If the 3D engine cannot load because CDN access is unavailable or WebGL is unsupported, the map falls back to a 2D-rendered orthographic globe using the same proxied imagery. The flat mode remains available for overlays and drawing.

## Place search

* Global, server-side search through the public OpenStreetMap Nominatim geocoder in `core/geocode.py`.
* Uses a descriptive User-Agent and a one-request-per-second limit.
* Accepts cities, regions, countries, landmarks and addresses where the geocoder supports them.
* Several matches can be listed, and the user chooses one.
* Moving the map is navigation only. It never reads the raster, changes the selection or runs an analysis.
* No coordinates are invented. No match is reported as no match.

## Project structure

```text
app.py                  Streamlit application: sections, map, chat, evidence
serve.py                Launches Streamlit with the same-origin tile proxy
tileserver.py           Tile proxy: providers, disk cache, transparency placeholder

requirements.txt        Runtime dependencies

core/                   Analysis core with no UI or Streamlit dependency
    raster.py           Windowed GeoTIFF/COG reads, metadata, CRS and transform handling
    bands.py            Band-role detection from metadata and evidence
    reflectance.py      Reflectance scaling and validation
    indices.py          NDVI / NDWI computation on native pixels
    index_definitions.py
                        Declarative index definitions, band roles and ranges
    statistics.py       Native-raster ROI statistics
    geometry.py         CRS/affine conversions and pixel/world mapping
    geo.py              Reprojection for display only and PNG encoding
    roi.py              Drawn geometry to validated selection
    spatial.py          Proximity and land-cover masks
    spatial_query.py    Per-cell spatial conditions
    multi_condition.py  Composition of several conditions, masks and counts
    temporal.py         Scene discovery, dated scene pairs and change
    suitability.py      Experimental crop-suitability screening
    alignment.py        Grid alignment checks between acquisitions
    evidence.py         Evidence packages for results
    geocode.py          Nominatim place search
    router.py           Text to registered analysis routing
    models.py           Result objects
    preview.py          Decimated preview reads for display
    samples.py          Bundled sample discovery and provenance
    datasources/        Fetchers and provenance for external datasets

analyses/               Registered analyses the router may select
    registry.py         What is available, unavailable and why
    ndvi.py
    ndwi.py
    ndvi_change.py
    spatial_query.py
    multi_condition.py
    crop_suitability.py
    evidence.py
    evidence_masks.py
    base.py

ui/
    theme.py            Design system: typography, spacing, surfaces and states
    components.py       Shared rendering blocks, welcome screen and chat turns
    map.py              Flat detail map and overlay objects
    evidence_panel.py   Evidence panel, layers and JSON export
    globe.py            Globe map wrapper, argument building and validation
    globe_frontend/     Globe and mapping component with no build step

config/                 Data-driven configuration
    indices
    crops
    spatial
    temporal
    multi

scripts/                Verification, browser and maintenance scripts
tests/                  Automated tests
docs/                   Architecture and design notes

data/sample/            Bundled sample scenes and provenance sidecars
assets/                 Product mark, favicon, logo and header

.streamlit/
    config.toml         Streamlit theme and client settings
```

## Configuration

**Nothing is required.** There are no API keys and no mandatory environment variables.

Two optional paths can be overridden. See `.env.example` for the variable names only; do not place secret values in the repository.

| Variable                | Default                    | Purpose                                              |
| ----------------------- | -------------------------- | ---------------------------------------------------- |
| `SATQUERY_TILE_CACHE`   | `/tmp/satquery_tile_cache` | Directory where proxied map tiles are cached         |
| `SATQUERY_EXTERNAL_DIR` | `data/external`            | Directory where fetched external datasets are cached |
| `SATQUERY_SESSION_DIR`  | `sessions`                 | Directory where session files, checkpoints and archives are stored |

## Tests and verification

Run the unit, integration and AppTest suite:

```bash
python -m pytest -q
```

The suite contains approximately 800 tests and takes around two minutes.

Browser verification requires the application to be running at `http://127.0.0.1:8501` and Playwright with Chromium installed:

```bash
pip install playwright
playwright install chromium
```

Run the verification scripts:

```bash
python serve.py
```

In another terminal:

```bash
python scripts/verify_ui_final.py
python scripts/verify_phase12_browser.py
python scripts/verify_phase13_browser.py
python scripts/verify_phase13_real_data.py
python scripts/verify_map_display.py
python scripts/inspect_ui.py
```

The browser scripts drive the running application as a user would.

Scripts that inspect Leaflet layers open the flat detail mode first because the globe is the default surface.

## Data, attribution and licensing

* **Base map:** © OpenStreetMap contributors for streets and Esri World Imagery for satellite imagery. Both are used through this application's own proxy with attribution displayed on screen.
* **Place search:** OpenStreetMap through Nominatim, used within its usage policy with a descriptive User-Agent and a maximum of one request per second.
* **Bundled sample:** Sentinel-2 MSI Level-2A surface reflectance, using Copernicus Sentinel data, windowed to 2048 × 2048 pixels at 10 m. Files under `data/sample/` include provenance sidecars that record the exact source, processing level and intended validity.
* **External datasets:** Datasets fetched at runtime are not committed. Each dataset carries a provenance sidecar.

## Limitations

* **The raster is the boundary.** Analysis runs only where the loaded, georeferenced raster contains data. Elsewhere the application reports insufficient coverage and never extrapolates.
* **Display versus analysis.** Previews, overlays and composites are resampled for the screen. Every number in an answer is computed on the native raster. The interface keeps display and analysis data separate.
* **External datasets.** Spatial conditions requiring land cover, soil, climate or elevation data download those datasets on first use through `core/datasources/`. If a download is unavailable, the affected condition reports insufficient data rather than using a substitute.
* **Unsupported questions are refused.** Flood change and temporal NDWI, for example, are registered as unavailable and are answered as such rather than approximated.
* **The globe needs a CDN on first load** for the pinned 3D engine. A 2D-rendered fallback is available. Base-map imagery also needs network access unless tiles are cached.
* **Crop-suitability screening is experimental.** It is a transparent, weighted screening of the conditions it states—not a recommendation, yield prediction or soil diagnosis.
