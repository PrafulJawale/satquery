# Phase 6 — ROI-based raster statistics (zonal statistics)

*What we are building, why the obvious approaches are wrong, how it works, what
the code does, and how to verify it.*

---

## 1. What we are building

Phase 5 gave the user the ability to say **"this area"**. Phase 6 answers the
question that sentence implies: **"what are the numbers inside this area?"**

```
GeoTIFF → NDVI (native) ─────────────────────────┐
                                                 ├── pixels inside the ROI
user draws a region → geometry in the RASTER CRS ─┘
                                                 ↓
                          ROI NDVI Analysis: counts, statistics, histogram
```

Phase 6 measures. It does **not** judge: no crop suitability, no flood/change
detection, no "this field is healthy". Those are Phase 7+ and each of them needs
this layer to be correct first.

---

## 2. The decisions, and why the obvious alternatives are wrong

**1. Why must the statistics come from the native raster, not the map layer?**
The map shows a *reprojected copy*: resampled to Web Mercator, usually capped at
a few hundred thousand pixels, averaged during resampling. Every number computed
from it would be a number about the picture, not about the Earth. The rule is
therefore absolute:

> `AnalysisResult.array` (native NDVI) + `AnalysisResult.mask` + native
> `transform` + native `crs` are the **only** inputs to a statistic.
> The reprojected array is a display artefact and is never measured.

**2. Why "pixels inside the ROI" and "valid pixels inside the ROI" separately?**
They answer different questions. The first is *geometry*: how many pixel centres
does my polygon cover? The second is *data*: how many of those actually carry an
NDVI value (not nodata, not NaN, not ±inf)? In a scene with a nodata edge, a
selection can cover 10 000 pixels and contain 9 412 valid ones. Collapsing the
two into one number is how a dashboard ends up reporting a mean over data that
does not exist.

**3. Why `all_touched=False` (pixel-centre convention) by default?**
`rasterio.features.geometry_mask` can either include every pixel the polygon
*touching* (`all_touched=True`) or only pixels whose **centre** falls inside
(`all_touched=False`). The centre convention is the one that matches how a
raster sample represents a location: the value of a pixel is the value *at its
centre*, so a pixel is "in" a region if its sample point is. `all_touched`
inflates the count along every edge. The convention is a parameter, it is
reported in the result (`all_touched`), and it is tested (`test_phase6_roi_stats.py`).
Boundary behaviour: a centre lying **exactly** on the polygon edge is included
(closed test) — documented by `test_pixel_centre_exactly_on_the_boundary_is_included`.

**4. Why rasterise only the ROI's window?**
Rasterising a polygon over a full 2048 × 2048 scene costs the same whether the
polygon is 10 pixels or 4 million. So:

```python
window = rasterio.windows.from_bounds(*roi.bounds, transform=transform)   # float
window = window.round_offsets() grown OUTWARD (floor/ceil)                # never crops
sub_transform = rasterio.windows.transform(window, transform)
mask = geometry_mask([roi], out_shape=window.shape, transform=sub_transform, invert=True)
```

The window is grown outward so rounding can never cut the ROI, and the mask is
pasted back into a full-shape boolean array. Cost becomes O(ROI). Measured on
the sample scene: **4.2 ms** for a 300 m ROI versus **229 ms** for a
whole-scene ROI (and 3.1 ms of that is the bare rasterisation — the rest is
percentiles over 4.2 million values). The test suite asserts the windowed mask
is **bit-identical** to a full-array rasterisation, so the optimisation cannot
silently change the answer.

**5. Why derive pixel size from the affine instead of assuming 10 m?**
Because our sample happens to be 10 m and the next file will not be. For a
general (rotated/sheared) affine, the column step is the vector `(a, d)` and the
row step is `(b, e)`:

```
pixel width  = hypot(a, d) × metres_per_unit(crs)
pixel height = hypot(b, e) × metres_per_unit(crs)
pixel area   = |a·e − b·d| × metres_per_unit(crs)²
```

Verified: 10 m grid → 10.0 × 10.0 m, 100 m²; 20 × 10 m grid → 200 m²; rotated
`Affine(10, 3, …, 2, −10, …)` → **106 m²** = |10·(−10) − 3·2|. Areas of the ROI
geometry reuse Phase 5's `core.geometry.area_m2` (planar for projected CRS,
geodesic on WGS84 for geographic CRS) — never lon/lat differences.

**6. Why does a CRS mismatch raise instead of "just working"?**
Because a polygon in EPSG:4326 masked against a raster in EPSG:32636 does not
fail loudly — it produces a plausible-looking mask over the wrong part of the
world. `calculate_roi_ndvi_stats` compares the declared `roi_crs` with the
raster CRS and raises `GeometryError` naming both.

**7. Why does empty mean "no pixels", not zero?**
A mean NDVI of `0.0000` is a real measurement (bare rock / water). Printing it
for a region that contains no pixels at all is fabrication. Two distinct
messages exist and each is reachable and tested:

- `NO_PIXELS_MESSAGE` → "No raster pixels fall inside the selected area."
- `NO_VALID_MESSAGE` → "No valid NDVI pixels were found inside the selected area."

In both cases `ROINDVIStats.stats` is `None` — not a dict of zeros. The ROI's own
geometric area is still reported (it is a fact about the polygon, not about the
raster).

---

## 3. The code

| Location | What it does |
|---|---|
| `core/statistics.py` | `ROINDVIStats`, `pixel_geometry()`, `_roi_window()`, `roi_pixel_mask()`, `calculate_roi_ndvi_stats()`, `NO_PIXELS_MESSAGE`, `NO_VALID_MESSAGE` — **no Streamlit import** |
| `ui/components.py` | `render_roi_analysis()` — the only place the numbers become pixels |
| `app.py` | wires the panel; optional display-only "analysed pixels" overlay |
| `tests/test_phase6_roi_stats.py` | 36 unit/integration tests, all synthetic and hand-checkable |
| `scripts/verify_phase6.py` | 54 checks on the real Sentinel-2 tile with independent cross-checks |
| `scripts/browser_test_phase6.py` | 50 checks driving real Chromium, comparing the UI against a recomputation |

### The contract

```python
res = calculate_roi_ndvi_stats(
    ndvi,                       # native NDVI array (H, W), NaN where invalid
    roi_geometry,               # shapely geometry ALREADY IN THE RASTER CRS
    transform,                  # native affine
    valid_mask,                 # AnalysisResult.mask (optional)
    crs=crs, roi_crs=roi_crs,   # mismatch -> GeometryError
    all_touched=False,          # pixel-centre convention
    percentiles=(5, 25, 50, 75, 95),
)

res.pixels_inside_roi   # geometry: pixel centres inside the polygon
res.valid_pixels        # data: those that also have a usable value
res.invalid_pixels      # pixels_inside_roi - valid_pixels
res.valid_fraction
res.stats               # dict with min/max/mean/median/std/sum + percentiles, or None
res.area_m2             # area of the ROI (planar or geodesic, method reported)
res.pixel_width / pixel_height / pixel_area_m2   # derived from the affine
res.valid_area_m2       # valid_pixels × pixel_area_m2
res.window              # the pixel window actually rasterised
res.crs / res.transform / res.all_touched / res.message / res.warnings
res.to_dict()           # JSON-safe, adds hectares and km²
```

`describe_valid()` (Phase 3) computes the statistics, so the ROI path reuses the
same, already-tested code and cannot drift from the scene-wide numbers.

### What the panel shows

Area (ha/km²) · native resolution (m × m, from the affine) · pixels inside /
valid / invalid / valid % · mean, median, min, max, std · P5, P25, P75, P95 ·
one sentence of plain description ("mean NDVI of 0.76 over 76 784 valid pixels")
· a histogram of **valid** NDVI values only. Invalid pixels are excluded from the
histogram rather than bucketed at zero. No thresholds, no classification, no
suitability claim.

---

## 4. How to verify it

```bash
cd satquery
python -m pytest tests -q                 # 220 passed
python scripts/verify_phase6.py           # 54/54 checks, real Sentinel-2 tile
streamlit run app.py                      # then:
python scripts/browser_test_phase6.py     # 50/50 checks in real Chromium
```

### What `verify_phase6.py` proves (54/54)

Every number is cross-checked against a **second, independently written**
implementation, not against another call of the same function:

* pixel counts vs `matplotlib.path.Path.contains_points` over the pixel centres
  (a different library, a different algorithm);
* mean/median/std/min/max/percentiles vs plain `numpy` over the same pixels;
* the windowed mask vs a full-array `geometry_mask` (bit-identical);
* areas vs the affine and vs the Phase 5 `area_m2`;
* an ROI covering one nodata pixel of the real scene → "no valid NDVI pixels";
* a CRS mismatch → `GeometryError` naming both CRS;
* no health / crop / suitability wording anywhere in the messages.

Reported for the real tile (EPSG:32636, 2048 × 2048, 10 m):

| ROI | pixels inside | valid | mean | median | std | P5 | P95 |
|---|---|---|---|---|---|---|---|
| centre, 1 000 × 1 000 m | 10 000 | 10 000 | 0.7772 | 0.8495 | 0.1951 | 0.2588 | 0.9307 |
| hanging off the NW corner | 7 000 | 7 000 | 0.8017 | 0.8701 | 0.1685 | 0.3811 | 0.9333 |
| 50 km outside the raster | 0 | 0 | — | — | — | — | — |

### What the browser test proves (50/50)

Real mouse drags on the Leaflet map; coordinates come from Leaflet's own
`latLngToContainerPoint`, never from guessed fractions of the screen. The test
reads the drawn shape's lat/lon back out of the browser, recomputes the ROI
statistics **inside the test process** from the GeoTIFF with numpy + matplotlib,
and compares with what the browser displays:

```
drawn shape (from Leaflet): lat 31.17815..31.20458, lon 31.79649..31.82671
expected (recomputed here): inside=84,349  mean=0.8069  median=0.8862  std=0.1928
shown in the browser       : inside=84,351  mean=0.8069  median=0.8862  std=0.1928
```

(The 2-pixel difference is the rectangle's projected edge: the test densifies
the WGS84 rectangle, the app densifies the same edge in `core.roi` — the
tolerance is ±2 px / 2 %.)

Scenarios: draw → panel appears with all 13 metrics + histogram; the numbers
match the independent recomputation; a second rectangle changes them; deleting
every shape removes the whole analysis; a rectangle outside the raster produces
"does not overlap" and no statistics.

---

## 5. Known limitations

1. **Single ROI, single index.** One selection, NDVI only. No multi-date, no
   NDWI, no comparison between regions — all Phase 7+.
2. **The map component can go silent after every shape is deleted.** While the
   NDVI image overlay is displayed, deleting the last shape leaves
   streamlit-folium silent: a shape drawn afterwards appears on the map but
   never reaches Python, and neither a second draw nor a zoom revives it. With
   the NDVI layer hidden, the identical flow works (message in ~8 s). It is a
   mapping-layer interaction, not a statistics bug: the numbers are computed in
   Python and are covered by 36 unit tests, 54 verifier checks and the
   independent recomputation. Workaround in the app: **Reset view**, or hide the
   NDVI layer. The browser test works around it and reports it.
3. **Memory.** One session with the 2048² scene loaded and NDVI computed costs
   ≈0.7 GB RSS. This sandbox has ≈2 GB, so a second concurrent session (or a
   headless Chromium plus a full `pytest` run) can trip the OOM killer. Restart
   the server before a browser run; don't run `pytest` while the app is up.
4. **`all_touched` is exposed but not offered in the UI.** The centre convention
   is the default; the flag exists for future use and is tested.
5. **Area of a geographic (lon/lat) raster** uses a geodesic WGS84
   approximation of the polygon; the *pixel* area for such a raster is derived
   from the affine in degrees² and is therefore approximate. Projected scenes
   (the normal case) are exact.
6. **Nodata is whatever the file declares** (Phase 2/3), plus non-finite values.
   Clouds are not masked — Sentinel-2 L2A ships a scene classification layer
   that we do not read yet, so cloudy pixels are valid NDVI (near 0 or negative).
