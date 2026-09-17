# AGENTS.md

SatQuery AI: Streamlit app that analyzes a loaded GeoTIFF/COG and answers natural-language geospatial questions with evidence. Python 3.11+ (developed on 3.13), no API keys, no pyproject/setup/lint/CI config.

## Run and verify

- Install: `python -m venv .venv` then `pip install -r requirements.txt` (rasterio wheels bundle GDAL). Activate `.venv` first.
- Launch the app with `python serve.py` — **never `streamlit run app.py`**. `serve.py` monkeypatches `create_starlette_app` to mount the same-origin tile proxy `/satquery-tiles/<provider>/<z>/<x>/<y>.png` required for the map backdrop. Without it the globe/overlays render but base-map imagery silently disappears.
- Tests (from repo root): `python -m pytest -q`. ~805 tests, ~2 min. `tests/conftest.py` inserts the repo root on `sys.path` (workaround until packaging exists) and sets `SATQUERY_DISABLE_SERVER_BASEMAP=1` so unit tests never hit the network.
- No linter/formatter/typecheck exists. Verification is the pytest suite plus browser scripts.
- Browser verification: Playwright is **not** in `requirements.txt` — `pip install playwright` + `playwright install chromium` are required, and the app must be running at `http://127.0.0.1:8501`. Scripts under `scripts/verify_*.py` drive the running app. Scripts that inspect Leaflet layers must open flat detail mode first (the globe is the default surface).

## Architecture

- `core/` — analysis engine with **no Streamlit/UI dependency**. `ui/` — Streamlit layer. `analyses/` — handlers the router selects. `config/` — data-driven YAML (indices, conditions, thresholds, crops).
- `analyses/registry.py` is the only place intent is bound to code. Adding an analysis = add one registry entry + engine module; `core/router.py`, `app.py`, and the UI must not change. There is no `if "ndvi" in query:` anywhere.
- `core/datasources/` fetches external datasets (worldcover, soilgrids, worldclim, copernicus_dem) at runtime and caches them under `SATQUERY_EXTERNAL_DIR` (default `data/external`, gitignored) with provenance sidecars. Until fetched, dependent conditions report insufficient data rather than substituting.
- Optional env vars (names only, no secrets): `SATQUERY_TILE_CACHE`, `SATQUERY_EXTERNAL_DIR`. `.env` is gitignored; see `.env.example`.

## Product invariants (do not break)

- Analysis numbers are computed **on the native raster from physical reflectance** (float), never from resampled/display-stretched previews. `core/preview.py` decimates only for the screen; display and analysis data are kept separate.
- Never invent data, dates, thresholds, or band mappings. When something is missing or ambiguous the app asks or reports "unknown/insufficient" — never zero, never a guess. Refuse (with an explanation) 8-bit/visual datasets for indices.
- UI hides Python tracebacks (`showErrorDetails = false` in `.streamlit/config.toml`) — debug via the terminal, not the page.

## Read before touching indices, geometry, or change detection

`docs/ARCHITECTURE_REVIEW.md` is the graded risk register for this codebase (reflectance scaling offsets, band-order traps, reprojection vs corner-fitting, grid alignment). `docs/PHASE*.md` are per-phase design/verification notes.