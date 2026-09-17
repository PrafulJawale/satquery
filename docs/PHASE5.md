# Phase 5 — Interactive area (ROI) selection

*What we are building, why the obvious approaches are wrong, how it works, what
the code does, and how to verify it.*

---

## 1. What we are building

Phases 1–4 gave us a correctly placed map. Phase 5 adds the verb that makes the
whole prototype useful later: **"this area"**.

```
GeoTIFF → metadata → RGB/FCC → NDVI → reprojected map → USER DRAWS AN AREA
                                                              ↓
                                        validated geometry in the RASTER CRS
                                                              ↓
                                              (Phase 6: zonal statistics)
```

Phase 5 deliberately stops at geometry. No statistics, no crop suitability, no
"can I grow cotton here" — only a validated region, ready for analysis.

---

## 2. The five questions, answered

**1. Why is the drawn geometry in EPSG:4326?**
Leaflet knows one coordinate system: lon/lat in WGS84, drawn on Web Mercator
tiles. `layer.toGeoJSON()` emits GeoJSON, and the GeoJSON specification mandates
CRS84 (= EPSG:4326, lon/lat order). This is a property of the browser, not a
choice we make.

**2. Why is the raster in a projected CRS like EPSG:32636?**
Satellite products are delivered in a local projected CRS (UTM) because that is
what makes "10 m per pixel" and "4.2 hectares" meaningful. Our scene's
coordinates are *metres of easting/northing*: `377200, 3441820`.

**3. Why transform into the raster CRS before masking or statistics?**
Phase 6 will ask "which pixels are inside this polygon?" using
`rasterio.features.geometry_mask(..., transform=ds.transform)`, and that
transform maps pixel (row, col) → coordinates **in the raster CRS**. The polygon
must be in the same CRS. Intersecting with the footprint also has to happen where
the footprint is exact — and the footprint is defined by the affine in the raster
CRS.

**4. Why is treating lon/lat as pixel coordinates wrong?**
- *Magnitude:* centre pixel (1024,1024) is `387640 E, 3431480 N`, but lon/lat
  `31.82, 31.20`. Using the latter as easting/northing moves the shape ~400 km.
- *Shape:* degrees are not square — 1° of longitude is 111 km at the equator and
  ~95 km at 31°N.
- *Order:* GeoJSON is (lon, lat); the affine wants (easting, northing). Swapping
  them yields a mirrored polygon that can still overlap the raster — a
  plausible-looking wrong answer, the worst kind.

**5. How the coordinates are converted**

```
Leaflet.Draw GeoJSON (EPSG:4326)
  → core.geometry.parse_geometry()        shapely geometry
  → ensure_polygonal()                    reject Point / LineString
  → repair()                              shapely.validation.make_valid (reported)
  → transform_geometry(4326 → raster CRS) pyproj Transformer, always_xy=True
  → intersection with native_footprint()  TRUE polygon intersection
  → ROISelection                          geometry_raster_crs + area
```

The original drawn geometry is kept in EPSG:4326 for display and provenance; the
raster-CRS copy is what analysis will use.

---

## 3. What the libraries actually do (verified, not assumed)

Everything below was checked against the installed `streamlit-folium 0.27.4`
frontend bundle and `leaflet-draw 1.0.2`.

| Finding | Consequence for the design |
|---|---|
| `all_drawings` = `window.drawnItems.toGeoJSON().features`, recomputed on every draw event | `all_drawings` is **authoritative**; the state is derived from it, never from deltas |
| `all_drawings` is `None` until the component reports, `[]` when nothing is left | `None` = "has not reported" → **keep** the selection; `[]` = "reported empty" → **clear** it |
| `last_active_drawing` is set from `t.layer.toGeoJSON()` — on a delete event that is the **deleted** shape | Using it as "the selection" is exactly how a stale ROI survives → we never do |
| streamlit-folium's React app sets `window.drawnItems` itself from folium's feature group | Our alias is only needed for standalone `m.save()` HTML (artefacts) |
| **leaflet-draw 1.0.2 fires `deleted` on the layer** and only fires map-level `draw:deleted` when the user clicks "Save" | Without a bridge, a deleted shape stays in `all_drawings` forever → **the delete bridge** (below) |
| streamlit-folium injects the body with `innerHTML`, so `<script>` tags never execute (verified), but an `<img onerror>` handler does | The bridge is injected as an `<img onerror>` handler |
| folium renders `root.script` children *before* the FeatureGroup variable exists | Any alias must be deferred to `DOMContentLoaded` |

### The delete bridge

```js
// leaflet-draw fires 'deleted' on the layer; streamlit-folium listens for the
// map-level 'draw:deleted'. Bridge them so a delete reaches Python immediately.
fg.on('layeradd', function (e) {
  e.layer.on('deleted', function () {
    setTimeout(function () { fg._map.fire('draw:deleted', {sq_bridge: 1}); }, 0);
  });
});
```

Measured before/after: with one shape drawn and then deleted,
`window.drawnItems` had **0 layers** while `all_drawings` still reported **1**
(broken); with the bridge, `all_drawings` reports **0** and the panel clears.

---

## 4. Validation, repair, rejection

| Input | Behaviour |
|---|---|
| Rectangle | Arrives as a GeoJSON `Polygon` (Leaflet serialises rectangles that way) — accepted |
| Polygon | Accepted |
| Self-intersecting | Repaired with `shapely.validation.make_valid()`; the repair is **reported** in the panel, never silent |
| Zero-area / degenerate | Rejected: "The drawn shape encloses no area." |
| Point / LineString | Rejected: "…is not an area. Draw a rectangle or a polygon." |
| Empty / nothing drawn | No selection; the panel shows the prompt |
| MultiPolygon | Accepted; number of parts and the summed area are reported |

No silent buffering, no silent ring-dropping: if nothing polygonal survives
repair, the geometry is rejected rather than quietly emptied.

## 5. Intersection, clipping, area

- **True polygon intersection**, never a bounding-box test. On a rotated grid the
  footprint is a quadrilateral; a selection can sit inside its bounding box and
  outside the raster. That case is refused (test: `test_selection_outside_a_rotated_footprint_but_inside_its_bbox`).
- **Clipping** happens in the raster CRS, against `core.geo.native_footprint()`
  (the exact parallelogram of the affine — rotation and shear included). The
  clipped part is what is stored; the original is kept for provenance.
- **Area:**
  - projected CRS → planar area, converted to m² with the CRS axis unit factor
    (so a feet-based CRS does not silently produce "m²");
  - geographic CRS → **geodesic** area via `pyproj.Geod` (never degrees²);
  - the method used is recorded and shown ("planar (EPSG:32636, metre)" /
    "geodesic (WGS84 ellipsoid)"), and labelled approximate.
- Straight lon/lat edges are densified (~0.01°) before reprojection so they
  follow the projection's curve.

## 6. Session state

`core.roi.update_roi_state()` is a Streamlit-free state machine over a plain dict:

```
drawings is None  → keep the selection, flag it as "map was redrawn"
drawings == []    → clear (delete really deletes — no stale ROI)
drawings == [..]  → recompute (last shape wins)
signature unchanged → return the SAME object (no recompute, no flicker)
```

The signature covers the drawings **and a raster key** (path/size/transform/CRS),
so switching datasets invalidates the selection. A "Clear selection" button
clears the state and bumps the map key so the component remounts and the drawn
layer really disappears.

---

## 7. Verification

```bash
python scripts/verify_phase1.py     #  90 checks   (regression)
python scripts/verify_phase2.py     #  58 checks   (regression)
python scripts/verify_phase3.py     #  54 checks   (regression)
python scripts/verify_phase4.py     #  37 checks   (regression)
python scripts/verify_phase5.py     #  43 checks   (ROI)
python -m pytest tests -q           # 184 tests
python scripts/browser_test_phase5.py   # 20 checks in a real Chromium browser
```

`verify_phase5.py` writes `artifacts/phase5_roi.png` (inside / partial / outside,
plus the rotated-grid bbox trap) and `artifacts/phase5_draw.html` (a standalone
page with the draw tools).

### Browser test — this one is genuine

`scripts/browser_test_phase5.py` launches Chromium, clicks Leaflet.Draw's
rectangle tool, **drags the mouse**, and checks what the app reports. Coordinates
are not guessed: the raster overlay's lat/lon bounds are read out of Leaflet and
converted with `latLngToContainerPoint`.

Result of the last run: **20 passed, 0 failed** — rectangle tool present, polygon
tool present, marker/polyline/circle absent; draw inside → "Selection detected",
100 % inside, geometry stored in EPSG:32636; draw again → the newer shape
replaces the older; delete every shape → panel returns to the empty state with
**no stale ROI**; straddling selection → "Only the portion inside the available
raster…"; outside selection → "Selected area does not overlap the available
raster."

Screenshots: `artifacts/phase5_browser_{inside,replace,deleted,partial,outside,final}.png`.

### Manual checklist (if you want to confirm it yourself)

1. Draw a rectangle over the fields → panel shows area in ha/km² and "100.0%".
2. Draw another → the new shape is used; a "2 shapes are drawn" note appears.
3. Press the delete tool, click each shape → the panel returns to "Draw a
   rectangle or polygon…".
4. Draw across the raster edge → "Only the portion inside the available raster…".
5. Zoom out and draw away from the image → "Selected area does not overlap the
   available raster."
6. Press **Clear selection** → everything resets and the outline disappears.

---

## 8. Known limitations

* **A map redraw loses the outline.** Changing a map control (base map, opacity)
  regenerates the map HTML, which makes Streamlit recreate the component iframe;
  the drawn shape disappears from the map. The selection itself is kept (state is
  only cleared by an explicit empty report) and the panel says so. Redraw it, or
  press **Clear selection**.
* **A new drawing re-renders nothing, but any control change resets pan/zoom**
  (the map re-fits to the raster extent on load).
* **Single ROI.** Multiple shapes can be drawn; only the most recent is used and
  the panel says so. Multi-ROI comparison is future work.
* **No antimeridian / polar handling** in the transform path.
* **Circle tool is off** by design (centre+radius is not a polygon and
  streamlit-folium's circle→polygon approximation is not verifiable here).
* **No statistics yet** — Phase 6 will rasterise the stored geometry.
* The delete bridge works around a leaflet-draw 1.0.2 behaviour; if folium ever
  ships leaflet-draw ≥ 1.0.3 the bridge becomes redundant (it is a no-op if the
  map-level event already fires).

---

## 9. Next

**Phase 6 — zonal statistics**: rasterise the stored `geometry_raster_crs`
against the native pixel grid (`rasterio.features.geometry_mask`) and summarise
the NDVI values inside it, with the same invalid-pixel discipline as Phase 3.
