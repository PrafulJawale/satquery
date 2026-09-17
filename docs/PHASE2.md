# Phase 2 — Satellite Visualisation

> Build order: **CORRECT → WORKING → UNDERSTANDABLE → IMPRESSIVE.**
> Phase 2 renders true-colour and false-colour images. It measures nothing.

---

## 1. Which dataset I selected, and why

I needed a **real** multispectral raster with a genuine near-infrared band.
Nothing synthetic, nothing guessed.

### Selected: Copernicus Sentinel-2 L2A, `S2B_36RUV_20230806_0_L2A`

| Property | Value |
|---|---|
| **Source** | AWS Open Data COG archive (`sentinel-s2-l2a-cogs`), located via the Earth Search STAC API |
| **Product** | `S2B_MSIL2A_20230806T082609_N0509_R021_T36RTV_20230806T122257.SAFE` |
| **Platform** | Sentinel-2B (MSI) |
| **Acquisition** | **2023-08-06 08:41:42 UTC** |
| **Tile** | MGRS **36RUV** (Nile Delta, Egypt) |
| **Processing level** | **L2A** = bottom-of-atmosphere surface reflectance |
| **Cloud cover** | 0.20 % |
| **Nodata** | 0 % of the tile |
| **CRS** | **EPSG:32636** (WGS 84 / UTM zone 36N) |
| **Bands acquired** | B02, B03, B04, B08 (blue, green, red, NIR) |
| **Resolution** | **10 m** (all four bands) |
| **Extracted window** | 2048 × 2048 px = **20.48 km** square, col_off=7720, row_off=3774 |
| **Local file** | `data/sample/s2_s2b-36ruv-20230806-0-l2a_2048px.tif` (23.5 MB, uint16) |
| **Licence** | Copernicus Open Data Licence — free to use, copy and distribute, attribution requested |
| **Reproduce it** | `python scripts/fetch_sentinel2_sample.py` |

**Why this one:**

1. **It is real, and lawfully reusable.** Copernicus data is open; the COG
   archive is public and needs no credentials.
2. **Small.** A full 10 m band is 10980² ≈ 240 MB. Because these are
   *Cloud-Optimised* GeoTIFFs, GDAL can fetch a 2048² window with a handful of
   HTTP range requests — 23 MB for all four bands, ~6 seconds.
3. **10 m for every band we need.** Red and NIR at the same GSD means NDVI in
   Phase 3 needs no resampling.
4. **It contains the things we will demo.** Measured on the extracted window:
   **75 % dense vegetation** (mean NDVI 0.84), **10.4 % surface water**
   (NIR reflectance 0.013), some bare/sparse land — one window that supports
   vegetation, water and later change-detection demos.
5. **Not Garhchiroli.** As instructed, the code hard-codes no AOI. The fetch
   script takes `--bbox` / `--datetime` / `--item-id`, so you can pull your own
   area any time.

**How the window was chosen (not just "crop the middle"):** the script reads a
256×256 overview of the whole tile, computes NDVI and NDWI, and scores every
candidate window by vegetation fraction + water presence + texture − nodata
penalty. It picked a window with `veg_frac=0.84, water_frac=0.095`.

### Provenance is machine-readable

`data/sample/provenance.json` and the sidecar
`data/sample/s2_…_2048px.tif.provenance.json` record the item id, datetime,
tile, CRS, bands, window offsets, scale factor and licence. If you move or
share the GeoTIFF, keep the sidecar with it.

---

## 2. Exactly which bands are used

Band order in the local file is **blue, green, red, NIR** (i.e. NOT alphabetical
and NOT the usual "1,2,3 = RGB" assumption — a deliberate trap avoided by
reading descriptions).

| Composite | Display R | Display G | Display B | Purpose |
|---|---|---|---|---|
| **True colour (RGB)** | band 3 (B04 red, 665 nm) | band 2 (B03 green, 560 nm) | band 1 (B02 blue, 490 nm) | Natural-looking context: fields, roads, water, settlements |
| **False colour (CIR)** | band 4 (B08 NIR, 842 nm) | band 3 (B04 red) | band 2 (B03 green) | Vegetation vigour at a glance |

Reading false colour: healthy vegetation reflects NIR strongly → it appears
**bright red / magenta** (brighter = more vigorous canopy). Water absorbs NIR →
**near black**. Bare soil and built-up land appear grey, cyan or brown. It is a
**qualitative** view — the measurement is NDVI in Phase 3.

### Band meaning is detected, never assumed

`core/bands.py` produces a guess **with the evidence and a confidence level**:

| Evidence | Strength |
|---|---|
| GDAL band descriptions (`B08_nir_842nm`, `SR_B5`) | high |
| A literal wavelength in the description (`842nm`) | high |
| GDAL colour interpretation (`red`/`green`/`blue`) | medium |
| Word tokens in descriptions (`blue`, `nir`) | medium |
| Band-count profile (13 bands → Sentinel-2 order) | low (flagged as an assumption) |
| Nothing usable → assume 1,2,3 = R,G,B | low (flagged, NDVI blocked) |

For our Sentinel-2 file this yields **HIGH confidence** with
`blue=1, green=2, red=3, nir=4` and profile `sentinel-2-l2a`.
For `RGB.byte.tif` (no descriptions, only colour interpretation) it yields
`red=1, green=2, blue=3`, **no NIR**, and a warning that NDVI is unavailable.
The UI shows the evidence and requires **explicit user confirmation** before
any index is computed (Phase 3 gate).

---

## 3. Preprocessing / resampling actually required

| Step | Needed? | Detail |
|---|---|---|
| Band alignment | **No** | All four bands are 10 m on an identical 10980² grid, verified by the fetch script (it aborts if any band is not 10 m or is a different size) |
| Reflectance scaling for display | **No** | Display uses raw DN with a percentile stretch. Scaling matters for *indices* (Phase 3): ρ = DN/10000 |
| Cloud masking | Not for this scene | 0.2 % cloud, 0 % nodata. Phase 3+ will use the SCL band |
| **Nodata masking** | **Yes** | `nodata = 0`. It must be excluded from stretch bounds or the black point collapses toward 0 and the image washes out |
| **Decimation for display** | **Yes** | 2048² source → ~1225² rendered. Analysis stays at native resolution; only rendering is decimated |
| **Transform recalculation** | **Yes** | The rendered image gets its own Affine = native × decimation factor. Verified to floating-point tolerance |
| Resampling method | `average` | Not `nearest`: nearest-neighbour aliasing makes bright pixels flicker in and out. GDAL ≥ 3.1 also excludes nodata from the average |
| Reprojection for the web | **Not yet** | Native CRS is UTM 36N; web maps need EPSG:3857/4326. That is Phase 4 — so no map is drawn yet |

### Two things real data taught us here

**1. The metadata lied about the reflectance offset.** The STAC
`raster:bands` block advertises `scale: 0.0001, offset: -0.1`. Applying it
makes the median pixel **−0.045 reflectance** — physically impossible. Measured
DN percentiles (median 552, p2 74) confirm this baseline-05 product has
`BOA_ADD_OFFSET = 0`, i.e. **ρ = DN/10000**. Blindly trusting metadata would
have broken every index in Phase 3. We default to offset 0, keep the claim in
the provenance file, and Phase 3 will re-check it.

**2. The band *names* lie.** On Earth Search, asset key `nir08` is **B8A at
20 m**; the 10 m NIR band `B08` is under the key `nir`. The fetch script
asserts GSD instead of trusting names.

---

## 4. What Phase 2 added

| File | Responsibility |
|---|---|
| `core/bands.py` | **NEW.** Band-role detection with evidence + confidence; sensor profiles; reflectance scale/offset for Phase 3 |
| `core/preview.py` | **NEW.** Decimated read, nodata masking, percentile stretch, RGB/FCC compositing, histograms, PNG encoding |
| `ui/components.py` | Band inspector, guess banner, band-mapping controls, composite renderer, real histograms |
| `app.py` | Phase 2 layout + sidebar render controls (stretch, mode, resolution) |
| `scripts/fetch_sentinel2_sample.py` | **NEW.** Reproducible STAC download with automatic window selection and provenance capture |
| `scripts/verify_phase2.py` | **NEW.** 58 correctness checks |
| `tests/test_phase2_preview.py`, `tests/test_app_smoke.py` | **NEW.** 21 unit tests + 6 headless app smoke tests |

### The rule this phase exists to enforce

> **Compute at native resolution. Render decimated.**

`PreviewResult` therefore carries **two** geometries: the source shape and the
display shape with its own Affine. Mixing them up is the classic
"the map disagrees with the numbers" bug, and the transform is verified in
tests against `native × decimation`.

---

## 5. How to verify it works

```bash
python scripts/verify_phase2.py     # 58 passed, 0 failed
python scripts/verify_phase1.py     # 90 passed, 0 failed  (no regression)
python -m pytest tests -q           # 44 passed
```

Then in the app (`streamlit run app.py`):

- [ ] **Band inspector** shows 4 bands named `B02_blue_490nm … B08_nir_842nm`.
- [ ] Confidence banner reads **HIGH / sentinel-2-l2a**, and the evidence
      expander explains why.
- [ ] True colour looks like a landscape (grey-brown fields, dark water).
- [ ] False colour shows vegetated fields in **red/magenta** and water **black**.
- [ ] Uncheck "I confirm this band mapping" → composites are explicitly labelled
      *visualisation only*.
- [ ] Switch the sample to `RGB.byte.tif` → the false-colour panel refuses to
      render and explains that a 3-band file has no NIR band.
- [ ] Switch to `all-nodata.tif` → everything is transparent, no invented pixels.
- [ ] Move the percentile sliders → the image re-renders and the bounds shown in
      "details" change with it.
- [ ] Rendered pixel size = 10 m × decimation factor (shown in details).

**Phase 2 is done when** you can point the app at *any* GeoTIFF you have not
seen, and it (a) identifies bands or admits it cannot, (b) renders both
composites or explains precisely why it cannot, and (c) never paints nodata as
data.

---

## 6. What Phase 2 deliberately does NOT do

- **No NDVI / NDWI.** NDVI is Phase 3, and it will be gated on the confirmed
  band mapping introduced here.
- **No map.** Reprojection to web CRSs is Phase 4. The downloaded PNG has no
  world file — the transform is printed next to it instead of being implied.
- **No change detection.** Only one acquisition date is bundled.
- **No interpretation thresholds.** Phase 3 will print any threshold it uses,
  with its source, as a heuristic.
- **No claims about crop suitability or floods.** Future work.

### Known limitations

1. Band statistics are computed on the **decimated** read — indicative, not
   authoritative (exact statistics: Phase 6).
2. The stretch is **display only** and is never an input to any computation.
3. A display pixel is treated as valid only when all three channels are valid
   (QGIS/Earth-Engine convention); partial-nodata pixels are masked entirely.
4. Uploads are held in RAM (`rasterio.MemoryFile`) — fine at prototype scale.
5. `streamlit>=1.50` is now required (`width="stretch"` replaces the deprecated
   `use_container_width`).
