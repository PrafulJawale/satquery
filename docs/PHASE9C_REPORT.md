# Phase 9 — Checkpoint B corrections + Checkpoint C

Status: **complete, stopped at the gate.** No UI, no map layer, no browser test,
no real-data verification. Checkpoint D needs explicit approval.

Tests: **496 passed** (was 392 at the end of Checkpoint B, 333 before Phase 9).
Phase 9 tests: **162** (79 parser, 65 spatial-core, 18 engine).

---

## Part 1 — the three required corrections

### Correction 1 — WATER is not WATER_PROXIMITY

`core/spatial_query.py`, `config/spatial/patterns.yml`

A proximity condition now requires a **proximity cue** (`near, within, close to,
next to, adjacent to, distance to, closer/nearer than`) *plus* the word water.
A bare "water" mention is the **water class**, not a distance.

| Query | Parsed as |
|---|---|
| `Find cropland excluding water` | `land_cover([40]) AND NOT water(class [80])` |
| `Find cropland not water` | `land_cover([40]) AND NOT water(class [80])` |
| `Find cotton areas outside water` | `crop_suitability(cotton, class >= 3, rainfed) AND NOT water(class [80])` |
| `Find cropland near water` | `land_cover([40]) AND water_proximity(<= 1000 m, class 80)` |
| `Find cropland within 500 m of water` | `land_cover([40]) AND water_proximity(<= 500 m, class 80)` |
| `Find water` | `water(class [80])` |

`NOT WATER` ≠ `NOT WATER_PROXIMITY` is asserted at parser level, mask level and
engine level (three separate regression tests). Wetland (90) stays separate.

Two supporting fixes were needed to make the split work:
* `_negated()` now handles negation phrases that **overlap** the condition
  ("not water") as well as those that precede it ("but not built-up").
* `Affine.to_gdal()` returns GDAL order `(c, a, b, f, d, e)`; using it made
  `Grid` read the **origin** as the pixel size (376140 m × 0 m). Fixed to
  `(a, b, c, d, e, f)`.

### Correction 2 — explicit irrigation blocks verdict queries

`core/router.py` (+`blocked_by`), `analyses/registry.py` (+check),
`analyses/base.py` (+`Status.UNSUPPORTED_CONDITION`), `config/spatial/patterns.yml`

`detect_unsupported_requirements()` runs for **every** intent; the result lands in
`QueryIntent.blocked_by` and the registry refuses **before any engine runs**:

```
Can I grow cotton with irrigation?        -> UNSUPPORTED_CONDITION (nothing computed)
Is this suitable for irrigated cotton?    -> UNSUPPORTED_CONDITION
Can I grow cotton if irrigation is available? -> UNSUPPORTED_CONDITION
Find cotton areas with irrigation         -> UNSUPPORTED_CONDITION
cotton near groundwater                   -> UNSUPPORTED_CONDITION
Can I grow cotton here?                   -> not blocked (unchanged)
Show flood areas.                         -> not blocked (Phase 7 message unchanged)
```

The message states that the available cotton screening evaluates the configured
**rainfed** scenario and does not measure irrigation availability. `irrigation`
is never rewritten as `near water`. Tested for both verdict and selection forms,
including "no fetch happened" (the provider call counters stay empty).

### Correction 3 — Phase 8 semantics preserved

`Where can I grow cotton in this region?` stays routed to `SPATIAL_QUERY`; a test
pins the resulting condition to `crop=cotton`, `min_class=3`, `scenario=rainfed`,
`required_analysis=crop_suitability` (delegate, do not re-implement).
Phase 8 files are untouched: 0 Phase-9 references in `core/suitability.py`,
`analyses/crop_suitability.py`, `config/crops/cotton.yml`, `core/alignment.py`,
`core/datasources/worldcover.py`.

---

## Part 2 — Checkpoint C

### C1/C2 — `core/spatial.py` (572 lines)

`Grid` (CRS, transform, width, height, cell size, note, source resolutions) +
`GridMask` (`match`/`valid`). `require_compatible()` checks CRS, transform,
dimensions, pixel width, pixel height, array shape and raises
`GridMismatchError` rather than combining mismatched grids.

### C5 — three-valued logic (strong Kleene), unit-tested

| AND | TRUE | FALSE | NA | | OR | TRUE | FALSE | NA | | NOT | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRUE | TRUE | FALSE | NA | | TRUE | TRUE | TRUE | TRUE | | TRUE | → FALSE |
| FALSE | FALSE | FALSE | **FALSE** | | FALSE | TRUE | FALSE | **NA** | | FALSE | → TRUE |
| NA | NA | FALSE | NA | | NA | TRUE | NA | NA | | NA | → NA |

All 9 AND cases, 9 OR cases and 3 NOT cases are parametrised tests.

### C3 — WATER

`water_mask_from_land_cover()`: TRUE = class 80, FALSE = any other valid class,
INSUFFICIENT = nodata/NaN. Wetland 90 is never water (tested).

### C4 — WATER PROXIMITY

`proximity_mask()` uses `scipy.ndimage.distance_transform_edt` with
`sampling=(|e|, |a|)` from the transform, so the result is **metres**. It refuses
a geographic CRS, non-square cells and negative distances
(`DistanceGridError`).

**The window-edge rule (a scientific choice, stated explicitly).** A distance is
only decidable inside the window, so a cell farther than N metres from mapped
water is FALSE **only when the nearest unknown cell or the window edge is also
farther than N**; otherwise it is INSUFFICIENT. A window with no water yields no
"far from water" claim at all. When the engine owns the grid it requests
`buffer_cells = ceil(distance / resolution) + 1` so the ROI interior is exact.

### C6/C7 — `SpatialQueryResult` (`analyses/spatial_query.py`, 580 lines)

Exposes query, conditions, operator, matched/non-matching/insufficient cell
counts, matched area and fraction, result mask, per-condition results, warnings,
provenance, analysis resolution, source resolutions and status.
Area = `|a| * |e|` of the analysis transform (900 m² at 30 m — never an assumed
10×10 m), with Phase 8's native/effective resolution note.

Zero matches is worded as a fact about the ROI:

> "Within the selected ROI, 0% of analysed cells satisfy all requested
> conditions — ... That is a statement about the selected area, not about the
> surrounding region."

and is distinguished from insufficient data (`RESULT_ZERO_MATCHES` vs
`Status.INSUFFICIENT_DATA`).

### C8 — synthetic tests (65 + 18, no network)

Water (all / none / mixed / nodata / wetland), proximity (metre scaling, 0 m,
threshold boundary, 2-D geometry, non-square arrays, no-water, nodata, geographic
CRS, non-square pixels, negative distance), logic truth tables, combinations
(`cotton AND water`, `cotton AND near-water`, `cropland AND near-water`,
`cropland AND NOT water`, `cotton OR cropland`, `cotton AND NOT water`),
edge cases (zero / all / partial / insufficient, incompatible CRS, transform,
dimensions, resolution), and engine-level runs with injected synthetic layers.

### C9 — registry

`Intent.SPATIAL_QUERY` → `run_spatial_query`. It orchestrates: Phase 8 cotton
engine for suitability conditions, WorldCover for land-cover/water conditions,
`core.spatial` for proximity and combination. `example_queries=()` — suggestions
stay unchanged until the UI exists (Checkpoint D).

### C10 — performance

One windowed WorldCover read per run, shared by all conditions; the class-80 mask
is built once and reused; arrays bounded to the analysis grid (a 3 km ROI at 30 m
= ~10–29 k cells). A test asserts the provider is called exactly once even for a
three-condition query.

---

## Validation

1. **New tests:** 162 Phase 9 tests pass (79 parser / 65 spatial / 18 engine).
2. **Full suite: 496 passed**, 0 failed, 68 s.
3. **Phase 8 unchanged:** all Phase 8 tests pass; 0 Phase-9 references in the
   Phase 8 engine/config files.
4. **Changed files:** `core/spatial.py` (new), `analyses/spatial_query.py` (new),
   `core/spatial_query.py`, `core/router.py`, `config/spatial/{conditions,patterns}.yml`,
   `analyses/registry.py`, `analyses/base.py` (+1 status),
   `tests/test_phase9_{parser,spatial,engine}.py` (2 new),
   `tests/test_phase8_router.py`, `tests/test_phase7_router.py` (+1 graduation),
   `docs/PHASE9_DESIGN.md`, `scripts/demo_phase9_masks.py` (new).
   `app.py` and `ui/*` are **untouched**.
5. **Counts:** 496 total (333 → 392 → 496).
6. **Representative structured queries:** see the tables above and
   `python scripts/demo_phase9_masks.py`.
7. **Representative mask results** (synthetic, 3 km ROI, 30 m, 4×4 water block):

   | Query | matched | area km² | match % | insuff % |
   |---|---:|---:|---:|---:|
   | Find cropland | 9,954 | 8.959 | 99.8 | 0.3 |
   | Find water | 16 | 0.014 | 0.2 | 0.3 |
   | Find cropland near water | 3,900 | 3.510 | 39.0 | 0.0 |
   | Find cropland within 500 m of water | 1,068 | 0.961 | 10.7 | 0.0 |
   | Find cropland excluding water | 9,954 | 8.959 | 99.8 | 0.3 |
   | Find cropland excluding areas near water | 6,084 | 5.476 | 60.8 | 0.0 |
   | Find cotton areas | 7,900 | 7.110 | 82.3 | 4.0 |
   | Find cotton areas near water | 3,786 | 3.407 | 69.0 | 45.1 |
   | Find cotton areas with irrigation | — | — | — | refused |

   Note the last two rows: `excluding water` (9,954) differs from
   `excluding areas near water` (6,084) — the correction, visible numerically.

8. **No UI or browser work.** `app.py`, `ui/map.py`, `ui/components.py` untouched
   (verified by content, not mtime).
9. **No real-data verification.** No WorldCover/GSW/SoilGrids tile was fetched;
   every layer in every test is synthetic.

## Two things to decide before Checkpoint D

1. **The cotton path cannot be given a proximity buffer.** Phase 8 owns the grid
   when a cotton condition is present (2-cell = 60 m buffer), so a 1 km
   proximity query marks the rim INSUFFICIENT (45.1% in the demo) and emits a
   warning. Options: (a) accept it, (b) let Phase 9 fetch WorldCover on a wider
   grid and crop to the cotton grid, (c) pass a buffer argument into Phase 8
   (touches Phase 8).
2. **The edge rule is conservative by design.** It converts "unknown because the
   window ends" into INSUFFICIENT rather than a possibly-wrong FALSE. If Checkpoint
   D should prefer a more permissive rule, say so and I will change it.
