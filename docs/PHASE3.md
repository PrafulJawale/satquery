# Phase 3 — NDVI

> Build order: **CORRECT → WORKING → UNDERSTANDABLE → IMPRESSIVE.**
> Phase 3 computes a real measurement. Everything before it was preparation.

---

## 1. Exact NDVI data flow

```
DatasetReader  +  USER-CONFIRMED (red_index, nir_index)
        │
        ▼
detect_reflectance_spec()          propose: file tags -> sensor profile -> raw
        │                          validate: apply to real pixels, check physics
        ▼
ReflectanceSpec                    scale, offset, source, evidence, warnings
        │
        ▼  to_reflectance()        rho = DN * scale + offset   (float32)
        │
        ▼
build_valid_mask()                 nodata ∪ NaN ∪ ±inf  (masked BEFORE scaling)
        │
        ▼
compute_ndvi()                     denom guard |NIR+Red| >= 1e-6
        │                          division under np.errstate
        │                          invalid -> NaN, never 0
        ▼
AnalysisResult                     array (NaN where invalid) + bool mask
        │                          + CRS + transform + native shape
        │                          + stats + counts + reflectance + provenance
        ├──────────────► statistics / export / future router
        ▼
decimate_array()  ──► ndvi_to_rgba()  ──► matplotlib figure
   (display copy, own transform)      (invalid = alpha 0)     (labelled NDVI, not RGB)
```

The gate is at the very top: **no confirmation → no computation**. Even a
high-confidence automatic detection is treated as a suggestion.

### Router-readiness

`core/indices.py` imports nothing from Streamlit. In Phase 8 the router will
call exactly:

```python
result, spec, report = ndvi_from_dataset(ds, red_index, nir_index, profile=profile)
text = result.summary_lines()      # factual sentences, values already computed
```

The model picks *which* analysis to run; it never computes a value.

---

## 2. Reflectance / scaling design

### Three tiers, in order of trust

| Tier | Source | Example |
|---|---|---|
| 1 | **The file declares it** — GDAL `ds.scales` / `ds.offsets`, or `scale_factor`/`add_offset` tags | — |
| 2 | **Sensor profile** | `sentinel-2-l2a` → 1e-4 / 0 · `landsat-c2-l2` → 2.75e-5 / −0.2 |
| 3 | **Raw DN** (unknown) | `is_reflectance=False`, caveat attached |

### Then empirical validation — never blind application

`validate_spec()` applies the candidate spec to a sample of real valid DN and
checks physics:

* if >1 % of pixels fall below −0.01 reflectance **and** an offset was applied →
  **reject the offset**, retry at 0, and record why;
* if >1 % exceed 1.6 reflectance → warn that the scale looks wrong.

This is not theoretical for us: the STAC catalogue for our own sample advertises
`offset: -0.1`, and applying it makes the median pixel **−0.045 reflectance**.
The check rejects it, and the UI shows the rejection in red.

### Current dataset: exactly how it is converted

```
rho = DN / 10000          (scale = 0.0001, offset = 0)
```

Source: sensor profile `sentinel-2-l2a`, chosen because the file declares no
scaling. **Not hard-coded in the NDVI function** — it lives in
`SENSOR_PROFILES` in `core/reflectance.py`, is overridable in the UI, and
Landsat support is already present (unverified, flagged as such).

### Why convert at all, if NDVI is scale-invariant?

With offset 0, NDVI from DN *is* NDVI from reflectance — the factor cancels.
A test asserts this (`test_ndvi_is_invariant_to_pure_multiplicative_scaling`).
We still convert, because:

1. it makes the offset assumption explicit and testable instead of implicit;
2. sensors with a real offset (Landsat C2: −0.2) **need** it — a test shows the
   result genuinely differs there;
3. later phases compare absolute reflectance against thresholds.

---

## 3. Masking strategy

A pixel is **invalid** if any of these hold (evaluated on raw DN, before scaling):

| Condition | Handling |
|---|---|
| equals the band's declared nodata | masked (NaN nodata handled specially) |
| NaN | masked (`np.isfinite`) |
| ±infinity | masked (`np.isfinite`) |
| \|NIR + Red\| < 1e-6 (undefined ratio) | masked — **not** set to 0 |
| masked by GDAL | masked (`masked=True` reads) |

Invariants, all covered by tests:

* invalid pixels are stored as **NaN** *and* flagged `False` in `result.mask`;
* `result.mask == np.isfinite(result.array)` exactly;
* **no invalid pixel is ever 0.0** — a zero would read as "bare rock/water";
* counts (total / valid / invalid / valid %) are always reported, even when the
  scene is empty;
* `stats is None` when there are no valid pixels — no mean, no zeros.

Real-data proof: the Sentinel-2 window contains 16 pixels where **NIR = 0
(nodata) while Red carries a value**. They are excluded. Treating NIR = 0 as a
measurement would have produced NDVI = −1.0, i.e. confidently labelled
"water" — a fabricated conclusion from missing data.

---

## 4. Output structure

`AnalysisResult` (one shape for every future analysis):

| Field | Meaning |
|---|---|
| `array` | float32, native resolution, NaN where invalid |
| `mask` | bool, True where valid |
| `crs`, `transform` | carried unchanged from the source |
| `native_shape` | (height, width) of the analysis raster |
| `value_range` | theoretical (−1, 1) |
| `observed_range` | measured (min, max), None if empty |
| `stats` | min, max, mean, median, std, valid_pixels, percentiles — **None if empty** |
| `counts` | total / valid / invalid / valid % / invalid % — always present |
| `reflectance` | the `ReflectanceSpec` actually applied |
| `bands_used` | which band played red, which played nir |
| `provenance` | dataset, datetime, sensor, tile, raster geometry |
| `caveats` | out-of-range values, raw-DN usage, spec warnings… |
| `colormap` | rendering hint (UI only) |

`to_dict()` excludes pixel arrays (safe for caching/JSON); `summary_lines()`
returns factual sentences for the future chat layer.

---

## 5. Risks identified

| Risk | Mitigation / status |
|---|---|
| **Wrong band mapping** produces plausible-looking NDVI | Hard user gate; evidence shown; never auto-accepted |
| **Bad metadata offset** (already hit once) | Empirical validation rejects impossible offsets |
| **Native-resolution memory**: a full 10 m Sentinel-2 band is 10980² × 4 B ≈ 480 MB as float32 | Fine at prototype scale (2048² ≈ 16 MB); chunked/windowed NDVI is future work |
| **Clouds / shadows** inflate or depress NDVI | Not masked in Phase 3 (scene is 0.2 % cloud); SCL/QA masking is Phase 4+ |
| **Atmosphere / BRDF** differences between scenes | L2A mitigates; cross-date comparison needs extra care (change detection phase) |
| **Decimated rendering** hides thin nodata | Display uses NaN-aware block averages; counts always come from the native array |
| **Threshold misuse** | Only continuous values by default; illustrative classes are off by default and carry a "NOT validated" caveat |
| **float32 precision** | Adequate for NDVI (verified against float64 hand calculations: max diff 2.8e-08) |

---

## 6. What was implemented

| File | Change |
|---|---|
| `core/reflectance.py` | **NEW** — profiles, proposal tiers, empirical validation |
| `core/indices.py` | **NEW** — masking, `compute_ndvi()`, `ndvi_from_dataset()`, illustrative classification |
| `core/statistics.py` | **NEW** — `describe_valid()` (None when empty), `count_summary()`, `fraction_within()` |
| `core/models.py` | `AnalysisResult` extended: mask, counts, observed range, reflectance, bands_used, provenance, `summary_lines()` |
| `core/preview.py` | `decimate_array()`, `ndvi_to_rgba()`, `render_ndvi_figure()`, `ndvi_histogram()`, `analysis_to_geotiff_bytes()` |
| `ui/components.py` | reflectance panel, pixel accounting, stats, classes, figure, provenance |
| `app.py` | Phase 3 section: gate → reflectance → compute → stats → map → export |
| `scripts/verify_phase3.py` | **NEW** — 54 checks |
| `tests/test_phase3_ndvi.py` | **NEW** — 33 tests |
| `tests/test_app_smoke.py` | +3 tests: gate locked, NDVI after confirmation, pixel accounting |

Phase 1 and Phase 2 code was **not rewritten**; only additive changes were made
(one bug fix in `count_summary` and one in `fraction_within`).

---

## 7. Verification

```
python scripts/verify_phase1.py  →  90 passed, 0 failed
python scripts/verify_phase2.py  →  58 passed, 0 failed
python scripts/verify_phase3.py  →  54 passed, 0 failed
python -m pytest tests -q        →  80 passed
```

Includes an **independent hand-check**: NDVI recomputed from raw DN at five
randomly chosen real pixels matches the engine to **2.8e-08**.

### Sample NDVI statistics — Sentinel-2B, Nile Delta, 2023-08-06

| Quantity | Value |
|---|---|
| Total pixels | 4,194,304 (2048 × 2048 at 10 m) |
| Valid | 4,194,288 (99.9996 %) |
| Invalid | 16 (NIR nodata where Red had data) |
| Min | **−0.9928** (open water) |
| Max | **0.9995** (dense irrigated crop) |
| Mean | **0.6116** |
| Median | **0.8450** |
| Std dev | **0.4402** |
| p5 / p25 / p50 / p75 / p95 | −0.413 / 0.408 / 0.845 / 0.906 / 0.938 |
| Reflectance | ρ = DN/10000, offset 0 (validated: 0 % negative) |

Illustrative class shares (**NOT validated**, off by default):
water/non-vegetated 10.1 %, bare/built 9.3 %, sparse 7.5 %, dense 73.1 %.
These are consistent with the independent measurements made when the sample was
fetched (10.4 % water, 75 % vegetation), which is a good cross-check.

---

## 8. Limitations

1. **No vegetation-health thresholds.** Continuous values only. Any class shown
   is illustrative, off by default, and carries its caveat.
2. **No cloud/shadow masking** yet (this scene is 0.2 % cloud, so it does not
   bite here — it will on other scenes).
3. **No map overlay yet** (Phase 4). The GeoTIFF export is georeferenced and
   opens in QGIS; the on-screen figure is not a map.
4. **Whole-scene only** — ROI statistics are Phase 5/6.
5. **One sensor family validated.** The Landsat profile is implemented but
   **unverified** against a real Landsat file; it is flagged accordingly.
6. **Single date** — no change detection yet.

---

## 9. How to check it yourself

```bash
streamlit run app.py
```

- NDVI panel starts **locked**; read the evidence, then tick the confirmation.
- Reflectance panel shows `rho = DN x 0.0001` and the validation numbers.
- Pixel accounting: 4,194,304 total / 4,194,288 valid / 16 invalid.
- Statistics: min −0.993 → max 0.999, mean 0.612, median 0.845.
- Map: continuous RdYlGn, colourbar labelled "NDVI", invalid pixels transparent,
  footer states it is **not** an RGB image and names dataset + date.
- Switch the sample to `all-nodata.tif`: pixel accounting reports 100 % invalid
  and **no statistics at all** — the correct, non-fabricated outcome.
