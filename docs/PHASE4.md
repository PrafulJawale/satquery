# Phase 4 — Interactive map with correct reprojection

*What we are building, why the obvious approach is wrong, how it works, what the
code does, and how to verify it.*

---

## 1. What we are building

Phases 1–3 gave us numbers: metadata, band meanings, reflectance, NDVI. None of
it had a *place*. Phase 4 puts the raster on a real map, in the real world, so
that a future answer ("your vegetation is stressed **here**") can point at
something.

The requirement is deceptively narrow: **draw the raster where it actually is.**
Everything in this phase exists to make that sentence true and provable.

---

## 2. Why this is the easiest thing in GIS to get wrong

Our Sentinel-2 sample stores pixels like this:

```
CRS        EPSG:32636  (WGS 84 / UTM zone 36N)
transform  | 10.00, 0.00, 377200.00 |
           | 0.00,-10.00, 3441820.00|
           | 0.00,  0.00,      1.00 |
bounds     easting 377200–397680, northing 3421140–3441820   (metres)
```

Leaflet (the library inside Folium) understands **only** latitude/longitude
(EPSG:4326) on top of Web Mercator tiles (EPSG:3857). UTM eastings/northings are
just numbers to it. Hand it `377200, 3441820` as if they were lon/lat and the
image lands somewhere off the coast of Somalia — or nowhere at all, because
`latitude = 3441820` is not a latitude.

### The trap: "just transform the four corners"

The tempting shortcut is to convert the four corners to lon/lat and draw a
rectangle between them. It fails in three separate ways:

1. **Straight lines do not stay straight.** A raster edge that is straight in
   UTM is a *curve* in Web Mercator. The four corners of the source grid are not
   the extremes of the destination shape — the true extent bulges slightly
   outside the corner quad. Measured on our scene, a north-up UTM grid is
   rotated ~0.62° in Web Mercator, which puts ~8 display pixels of the raster
   outside any corner-derived rectangle.
2. **A rotated grid has no rectangle at all.** If the affine has rotation or
   shear, the footprint is a general quadrilateral. Its bounding box is a
   strictly larger area — up to ~40 % for a strongly rotated grid. Nothing in the
   corner quad tells you which part of that box is data and which part is empty.
3. **You never learn the destination grid.** `ImageOverlay` needs the bounds of
   the *image you are drawing*, i.e. the corners of the grid you resampled into.
   Those come from the destination transform, not from the source corners.

So we do not transform four corners. We let GDAL compute the destination grid,
and we derive the bounds from that grid.

---

## 3. The architecture rule: analysis ≠ visualisation

```
                     ┌──────────────────────── ANALYSIS (native) ───────────┐
  GeoTIFF            │  core.indices.ndvi_from_dataset                      │
  EPSG:32636  ──────►│      ↓                                               │
  2048×2048, 10 m    │  AnalysisResult  (native grid, native CRS)           │
                     │      ↓                    ↓                          │
                     │  statistics          GeoTIFF export                  │
                     └──────────────────────────┬───────────────────────────┘
                                                │  one-way
                     ┌──────────────────────────▼── VISUALISATION (display) ┐
                     │  core.geo.reproject_array / reproject_bands          │
                     │      ↓                                               │
                     │  WebRaster  (EPSG:3857, 776×773, remembers source)   │
                     │      ↓                                               │
                     │  web_rgba_from_* → uint8 RGBA (invalid → alpha 0)    │
                     │      ↓                                               │
                     │  folium ImageOverlay → Leaflet                       │
                     └──────────────────────────────────────────────────────┘
```

The arrow only ever points right. Consequences, all enforced in code and tests:

* `reproject_array` **never mutates its input** (test: `test_reprojection_leaves_the_input_alone`).
* Every statistic, every number in the UI, and the GeoTIFF download come from the
  native array. The map's reprojected copy is never read back into analysis.
* Resampling to the display grid changes values slightly — measured median
  difference **0.021 NDVI** on this scene — which is exactly why it must never be
  analysed. The map is for *where*, the native result is for *how much*.

---

## 4. How the pipeline works, step by step

| Step | Code | What happens |
|---|---|---|
| 1 | `ds.crs`, `ds.transform` | Read the CRS and affine. If the CRS is missing → `MissingCRSError`, and the app says "I cannot place this" instead of guessing. |
| 2 | `destination_grid()` → `rasterio.warp.calculate_default_transform` | GDAL works out the destination transform, width and height that cover the source. This is the authoritative answer to "what grid should I resample into?" |
| 3 | `max_pixels` cap | If that grid is too big, scale it down — with `math.ceil`, so the grid always **covers** the footprint. (Bug found in review: `round` shrank the grid ~12 m inside the true footprint and silently mis-placed the overlay.) |
| 4 | `rasterio.warp.reproject(..., resampling=Resampling.average)` | Resample. `average` (not `nearest`) so a 10 m raster downsampled for the screen does not alias into noise. Masked pixels are cast to `float32` **before** `filled(np.nan)` — filling a `uint16` masked array with NaN raises `TypeError`. |
| 5 | `WebRaster` | The result plus its provenance: `source_crs`, `source_transform`, `source_shape`, destination grid, and the `leaflet_bounds` derived from the destination corners. |
| 6 | `web_rgba_from_bands` / `web_rgba_from_values` | Percentile stretch (identical to the Phase 2 panel) → uint8 RGBA. **Invalid pixels get alpha 0** — transparent, never green, never zero-as-data. |
| 7 | `folium.raster_layers.ImageOverlay` | `image=<ndarray>`, `bounds=[[lat_min, lon_min], [lat_max, lon_max]]`, **`mercator_project=False`**. |
| 8 | `raster_footprint()` | Each of the four grid edges is densified into 32 segments, every vertex transformed to WGS84, then closed into a Shapely polygon → GeoJSON. Densification is what makes the curved edge correct. |

### Two details worth remembering

**`mercator_project=False`.** Folium's `mercator_project=True` only re-wraps the
bounding box into Plate Carrée and stretches the image to fit; it cannot
represent a UTM grid. We already reprojected properly, so we hand Leaflet a real
EPSG:3857 image with its real EPSG:3857 bounds and tell it to leave the pixels
alone. (Verified: the ndarray is encoded to a base64 PNG data URI and the RGBA
alpha channel survives — a pixel with alpha 128 comes back as alpha 128.)

**Bounds come from the destination grid.** `WebRaster.leaflet_bounds` is
`transform * (0, 0)` and `transform * (width, height)` converted to EPSG:4326.
Not the source corners. That is the whole point of section 2.

---

## 5. Evidence from the real scene

Scene: **Sentinel-2B L2A, `S2B_36RUV_20230806_0_L2A`, Nile Delta, 2023-08-06,
EPSG:32636, 10 m, 2048 × 2048.**

| Measurement | Value | Why it matters |
|---|---|---|
| Footprint vertices | **93** (not 4) | densified edges, so curved edges are followed |
| Footprint area | **419.6 km²** (geodesic) | matches 20.48 km × 20.48 km = 419.4 km² |
| Footprint bbox | lon **31.70983 – 31.92705**, lat **31.10381 – 31.29054** | equals the independently computed `grid_bounds_wgs84` for this north-up case |
| Scene centre pixel (1024,1024) | **31.19722 °N, 31.81854 °E** | Nile Delta, as it should be |
| Round trip | centre → lon/lat → pixel **(1024, 1024)** | the inverse transform is consistent |
| Display grid (RGB and NDVI) | **776 × 773** @ 600 k px cap | identical `leaflet_bounds` for both layers, so they stack exactly |
| `leaflet_bounds` | `[[31.1039213, 31.7098282], [31.2905365, 31.9271490]]` | computed from the destination grid |
| Display bounds ⊇ footprint | **yes, on all four sides** | `max_pixels` must cover, never crop |
| Opaque overlay pixels inside the footprint | **99.80 %** | the overlay really is on the raster |
| The 0.2 % that are outside | worst case **12.4 m = 1.24 source pixels** | boundary resampling, not a positioning error |
| NDVI at the same lon/lat, native vs display | median difference **0.021** | agreement; and why display ≠ analysis |
| Valid fraction, native → display | 0.999996 → 0.97995 | the ~2 % loss is the UTM→Mercator grid rotation: a north-up UTM grid is *not* north-up in Web Mercator, so the covering grid has thin empty wedges at top and bottom. Geometrically correct — this is precisely what corner-fitting gets wrong. |

---

## 6. What the code does

### `core/geo.py` (new, Streamlit-free)

| Function / class | Job |
|---|---|
| `MissingCRSError` | raised instead of guessing coordinates |
| `WebRaster` | reprojected array + its provenance + `leaflet_bounds` |
| `destination_grid()` | `calculate_default_transform` + the `max_pixels` cap |
| `reproject_array()` | array → display grid (used for the in-memory NDVI result) |
| `reproject_bands()` | dataset → display grid (used for RGB / false colour) |
| `web_rgba_from_bands()`, `web_rgba_from_values()` | stretch / colour-map → uint8 RGBA with alpha |
| `rgba_to_png_bytes()` | standalone PNG export |
| `raster_footprint()`, `footprint_feature()` | densified WGS84 polygon → GeoJSON |
| `grid_bounds_wgs84()` | independent cross-check of the footprint |
| `pixel_to_lonlat()`, `lonlat_to_pixel()` | forward and inverse georeferencing (map clicks) |

### `ui/map.py` (new)

`BASE_TILES` (OSM / Esri World Imagery / CartoDB Positron), `MapOverlay`,
`colourbar_css()` (samples a matplotlib colour map into a CSS gradient),
`ndvi_legend_html()` (states *"Not an RGB / satellite photo"*, dataset, date,
vmin/vmax, *"invalid pixels are transparent, never green"*, source CRS → display
CRS), `rgb_legend_html()`, and `build_map()` (tile layers + footprint GeoJSON +
N image overlays + layer control + click popup + live `lon/lat:` readout +
`fit_bounds`).

### `app.py` — section "4 · Interactive map"

Base-map picker, True colour / False colour / NDVI toggles, overlay-resolution
slider, opacity, footprint toggle, reset-view button, `st_folium`, and a
click-to-inspect panel that walks a clicked lon/lat **backwards** into the native
grid (`lonlat_to_pixel`) and reads the actual stored NDVI there. The
"Reprojection details" expander prints both CRSs and the raw `WebRaster`
metadata. If the raster has no CRS, the app shows an explicit error — it never
guesses a position.

---

## 7. How to verify

```bash
python scripts/verify_phase4.py     # 37 checks — CRS, transform, reprojection, placement, rendering
python scripts/verify_phase1.py     # 90 checks — must stay green
python scripts/verify_phase2.py     # 58 checks
python scripts/verify_phase3.py     # 54 checks
python -m pytest tests -q           # 113 tests
```

`verify_phase4.py` also writes two artefacts:

* **`artifacts/phase4_map.html`** — a standalone Leaflet page (2.7 MB) with the
  real overlays and footprint. Open it in a browser; no server needed. The
  overlays are embedded as data URIs, so they draw even offline (only the base
  tiles need the network).
* **`artifacts/phase4_smoke.png`** — a static figure: the footprint vs the bounds
  handed to Leaflet, every opaque overlay pixel re-projected to lon/lat and
  scattered over the footprint, and the exact RGBA images Leaflet receives.

New tests: `tests/test_phase4_geo.py` (23) and `tests/test_phase4_map.py` (9),
plus one app smoke test. The map tests assert on the generated HTML (overlay
bounds strings, embedded PNG magic bytes, layer control, legend text) because
that is what Leaflet will actually read.

---

## 8. Deliberately NOT built

No ROI/polygon selection, no crop suitability, no flood or change detection, no
query router, no GeoChat/VLM, no live satellite search. Phase 4 is geographic
correctness and interactivity only.

**geopandas was deliberately not added.** Footprint → GeoJSON, geodesic area,
containment and masking are all done with Shapely + PyProj, which rasterio
already depends on. It stays an optional dependency until we need real vector
overlays (soil/land-use polygons) for crop suitability.

---

## 9. Limitations

* **The display copy slightly differs from native values** (median 0.021 NDVI).
  Expected, and the reason analysis never uses it.
* **~2 % of edge pixels are lost** to the UTM→Mercator grid rotation. The
  coverage grid is intentionally larger than the data; the wedges are
  transparent, not painted.
* **Average resampling blurs.** At 10 m → ~26 m display pixels, fine field
  boundaries soften. This is a display artefact; zooming does not add detail.
* **One PNG per layer**, embedded as a base64 data URI (≈1.3 MB per overlay).
  Fine for a 2048² window; a full 10980² Sentinel-2 tile would need a tiled
  server (TiTiler / `localtileserver`), not `ImageOverlay`.
* **No antimeridian or polar handling.** A scene crossing 180° would produce a
  nonsense bbox. Not exercised by this sample.
* **CRS-less rasters get no map at all** — by design. A plausible-looking wrong
  position is worse than a clear error.
* Base tiles need internet in the browser; the app itself runs fully offline.

---

## 10. Next

**Phase 5 — area selection**: draw or click a region on this map and resolve it
to a mask *in the native raster grid* (the inverse of what Phase 4 does for
display), ready for zonal statistics in Phase 6. Say **NEXT**.
