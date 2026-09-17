# Phase 1 — GeoTIFF Ingestion

> Build order: **CORRECT → WORKING → UNDERSTANDABLE → IMPRESSIVE.**
> Phase 1 does exactly one thing: read a GeoTIFF and tell the truth about it.

---

## 1. What we are building

A module (`core/raster.py`) that opens a GeoTIFF and returns a complete,
serialisable description of it — plus a thin Streamlit page that shows that
description, and a verification harness that proves the description is right.

No pixels are displayed. No index is computed. That is intentional.

If we cannot trust our CRS, transform, bounds and nodata handling, then every
later feature — preview, NDVI, map overlay, ROI statistics — is quietly wrong,
and it will be wrong in a way that *looks* plausible. That is the failure mode
we are buying insurance against.

---

## 2. Why we need it

Every later phase reads the same three things, over and over:

| Later feature | What it needs from ingestion |
|---|---|
| RGB preview | band count + dtypes + a decimated read |
| NDVI | the Red and NIR bands + nodata + dtype validity |
| Map overlay | CRS + transform + a **correct** WGS84 extent |
| ROI statistics | transform (to convert a drawn polygon into pixel rows/cols) |
| Change detection | two rasters on a **common grid**, same CRS |
| Crop suitability | every one of the above, plus external rasters aligned to them |

Ingestion is the contract. If the contract is vague, each feature invents its
own idea of what "the raster" is, and the map stops agreeing with the numbers.

---

## 3. How it works

```
file (path or upload bytes)
        │
        ▼
rasterio.open()  /  MemoryFile(upload_bytes).open()
        │
        ▼
RasterInfo  ──►  .to_dict()  ──►  Streamlit UI (or JSON API, later)
```

### The three numbers that matter most

**CRS (Coordinate Reference System)** answers *"what do these coordinates
mean?"* `EPSG:32644` = WGS 84 / UTM zone 44N (which covers Garhchiroli, ~80°E).
`EPSG:4326` = plain longitude/latitude. Web maps speak EPSG:4326 for
coordinates and EPSG:3857 (Web Mercator) for the imagery underneath them.

**Affine transform** is the 6-number recipe mapping pixel → world:

```
x = a·col + b·row + c
y = d·col + e·row + f
```

For a normal north-up image: `a` = pixel width, `e` = **−**pixel height
(y decreases as row increases), `b = d = 0`. If `b` or `d` are not zero the
image is rotated, and a simple "put this PNG between opposite corners" overlay
is geometrically wrong.

**Nodata** is the value that means "no observation". Sentinel-2 L2A uses `0`.
If you compute a mean without masking nodata, every black scene corner and
cloud gap is counted as a real measurement of the Earth.

### What the code does, file by file

| File | Responsibility |
|---|---|
| `core/models.py` | `BandInfo`, `SpatialInfo`, `RasterInfo`, `Provenance`, `AnalysisResult` — the shared vocabulary. Zero Streamlit imports. |
| `core/raster.py` | Opening (disk + in-memory uploads), metadata extraction, densified WGS84 footprint, area, warnings, small-window pixel sanity check. |
| `core/samples.py` | Loads `data/sample/provenance.json` — where each sample came from and what it may/may not be used for. |
| `ui/components.py` | Renders dicts. Knows nothing about rasterio. |
| `app.py` | Streamlit shell: pick a source → describe → display. |
| `scripts/verify_phase1.py` | CLI harness: asserts invariants + known values. |
| `tests/` | 16 pytest cases covering the same invariants. |

### Three details worth understanding

**1. Densified footprints.** Projecting a UTM raster into lon/lat bends its
straight edges into curves. Sampling only the 4 corners understates the real
coverage. So we insert 16 points along each edge before transforming → 64
vertices. (`densify_ring()` in `core/raster.py`.)

**2. Datums are not interchangeable.** `byte.tif` is `EPSG:26711` — NAD27,
not WGS84. The difference is tens of metres on the ground. A test asserts the
transform actually moves the coordinates, so "we forgot to reproject" fails
loudly instead of silently shifting every result north-west.

**3. Warnings are part of the data, not decoration.** `RasterInfo.warnings`
carries structural caveats the UI is *required* to display: no CRS, rotated
transform, no nodata declared, 8-bit display product, non-square pixels. Phase 3
will refuse to compute NDVI on an 8-bit visual product — the warning is what
makes that refusal explainable.

---

## 4. How to verify it works

```bash
# 1. invariants + known values across all bundled samples
python scripts/verify_phase1.py
#    -> RESULT: 77 passed, 0 failed

# 2. unit tests
python -m pytest tests -q
#    -> 16 passed

# 3. your own file, checked the same way
python scripts/verify_phase1.py /path/to/your/scene.tif

# 4. independent cross-check with GDAL's own CLI (ships with rasterio)
rio info data/sample/RGB.byte.tif
```

Then open the app (`streamlit run app.py`) and confirm:

- [ ] Width / height / band count / dtype match `rio info` **exactly**.
- [ ] CRS and bounds match `rio info` **exactly**.
- [ ] WGS84 bounds put your scene where you know it is. Copy
      `min lat min lon` into Google Earth — for `RGB.byte.tif` you should land
      near 24.56°N, 77.76°W (Bahamas).
- [ ] The caveats listed match what you already know about the file.
- [ ] `python -m pytest tests -q` is still green after any change.

**Phase 1 is done when you can upload a GeoTIFF you have never used in this
project, and the app's report agrees with `rio info` line for line.**

---

## 5. What is deliberately NOT in Phase 1

- Pixel display or stretching → Phase 2
- NDVI / NDWI → Phase 3
- Any map → Phase 4
- Drawing / clicking an area → Phase 5
- Zonal statistics → Phase 6
- Natural-language understanding → Phase 8+

## 6. Known limitations (written down so we don't forget)

1. Uploads are held fully in RAM via `rasterio.MemoryFile`. Fine for
   prototype-sized files; a production build would spool to disk.
2. `looks_like_cog` is a heuristic (tiled + overviews + ≥256 block), not a
   certification.
3. Band meaning is **not** inferred. `all-nodata.tif` has bands literally named
   `blue, green, red, nir`, but `RGB.byte.tif` has unnamed bands with
   `colorinterp = red, green, blue`. Phase 3 will use descriptions +
   colorinterp + preset profiles as *hints the user confirms*, never as silent
   assumptions.
4. No reprojection yet — so rotated or non-WGS84 rasters are *described*
   correctly but not yet *displayed* correctly. That is Phase 4's job.
