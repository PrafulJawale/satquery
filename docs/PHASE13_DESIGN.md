# Phase 13 — Design report (Step 1: inspect before modifying)

**Status: design only. No implementation code has been written.**

The conclusion of the inspection is the most important line in this document:

> Every fact Phase 13 must expose is **already carried** by the frozen
> Phase 9–12 result objects. Phase 13 can therefore be a **read-only
> interpretation layer** over existing results — no Phase 9–12 engine, result
> class, mask-algebra rule or scientific wording needs to change.

---

## 1. What already exists

### 1.1 Result contracts

| Object | Fields that matter to Phase 13 |
|---|---|
| `analyses.multi_condition.MultiConditionResult` (30 fields) | `conditions` (`ComposedCondition`s), `condition_results` (per-condition counts + provenance), `combined_mask`, `condition_masks`, `grid`, `roi_mask`, the four cell counts, `matched_area_m2`, fractions, `index_summaries`, `source_analyses`, `source_dates`, `alignment`, `threshold_provenance`, `analysis_resolution`, `source_resolutions`, `limitations`, `warnings`, `provenance`, `performance` |
| `analyses.spatial_query.SpatialQueryResult` (25) | `conditions`, `condition_masks`, `grid`, counts, `analysis_resolution`, `effective_resolution_note`, `provenance`, `performance` |
| `analyses.ndvi_change.NDVIChangeResult` (33) | `before/after_date`, `before/after_scene`, counts, `change_raster`, `class_raster`, `crs`, `transform`, `resolution`, `thresholds`, `alignment`, `provenance`, `limitations` |
| `core.statistics.ROINDVIStats` (NDVI/NDWI) | `pixels_inside_roi`, `valid_pixels`, `invalid_pixels`, `valid_fraction`, `stats`, `area_m2`, `pixel_area_m2`, `crs`, `transform`, `window`, `index_name` |
| `analyses.base.AnalysisExecution` | `intent`, `status`, `query`, `normalized_query`, `confidence`, `explanation`, `matched`, `result`, `message`, `warnings`, `provenance` + `to_dict()` |
| `core.multi_condition.ComposedCondition` | `kind`, `name`, `label`, `source_analysis`, `parameters` (classes, `distance_m`, `class`), `threshold` (`ThresholdSpec`), `negate`, `spatial_condition`, `evidence`, `interpretation`, `limitations` + `to_dict()` |

### 1.2 Provenance that already travels end-to-end

* **Execution level** — `AnalysisExecution.provenance` carries `router`, `engine`, and per-intent extras.
* **Result level** — `MultiConditionResult.provenance` keys: `engine`, `config_version`, `conditions`, `router`, `sources`, `thresholds`, `alignment`.
* **Condition level** — each `condition_results[i]["provenance"]`, e.g. spatial `{'classes': [40], 'clipped_to_roi': True}`; spectral `{'operator': '>', 'threshold': 0.6, 'dataset': …, 'reflectance_scale': 0.0001, 'reflectance_offset': 0.0, 'clipped_to_roi': True}`.
* **Threshold level** — `ThresholdSpec` with `provenance` ∈ {`user_specified`, `config_convention`, `relative`} and `detail` (e.g. `your query`, `convention:ndvi_high`, `relative:median`).
* **Grid / alignment level** — `Grid.to_dict()` (crs, transform, width, height, resolution, source resolutions, note) and `AlignmentRecord` (method, crs, transform, width, height, resolution, resampling, target, note).
* **Evidence statistics** — `index_summaries[index]` with `mean`, `median`, `min`, `max`, `std`, `valid_pixels`, and `computed_over` ("cells matching the combined condition").
* **Performance** — `performance = {'runtime_ms': …, 'grid_cells': …}`.

### 1.3 Existing conventions Phase 13 must follow

| Convention | Where it is established |
|---|---|
| `to_dict()` on every result/dataclass | `analyses/base.py`, all engines, `core/spatial.py` |
| JSON fallback `json.dumps(value, default=str)` | `core/preview.py`, `core/roi.py` |
| `st.download_button(label, data=bytes, file_name=…, mime=…, width="stretch")` | `app.py` (GeoTIFF export), `ui/components.py` (PNG export) |
| Frozen dataclasses for parsed structures | `QueryIntent`, `ComposedCondition`, `IndexDefinition` |
| Refusal statuses as first-class outcomes | `analyses.base.Status` (`NEEDS_THRESHOLD`, `INSUFFICIENT_DATA`, `NEEDS_TWO_DATES`, `UNSUPPORTED_CONDITION`, …) |
| Three-valued states 2/1/0 + `combine_all` | `core/spatial.py`, reused by Phase 12 |
| Boundary wording in ONE constant | `analyses.multi_condition.CAVEAT` |
| Panels per intent, expanders for detail, `st.caption` for provenance | `ui/components.py` |
| Layers added, never substituted; one legend per layer family | `ui/map.py`, `app.py` |

### 1.4 UI integration points

* `app.py:1810-1832` — the chat-history loop that dispatches to `render_spatial_query`, `render_multi_condition`, `render_ndwi_stats`, `render_ndvi_change`, `render_crop_suitability`.
* `app.py:1257-1360` — the map/overlay section: `web_spatial_from_mask(...)` → `MapOverlay(...)`; existing composed-condition layer and legend.
* `ui/components.py::_render_why` / `_render_methodology` — the Phase 9 precedent for "why this result" prose.
* `st.session_state["last_composition"]` — the live result object the map section draws from.

---

## 2. What can be reused unchanged

1. `MultiConditionResult` and the other result objects — read them, never restate them.
2. `CAVEAT` (imported from `analyses.multi_condition`; **no second copy**).
3. `ComposedCondition.parameters` for the "Class: 40 / Distance: 1000 m / change class" lines.
4. `Grid.to_dict()`, `GridMask.to_dict()`, `AlignmentRecord` → dict.
5. `composition_rgba` / `composition_legend_html` / `COMPOSITION_STATE_COLOURS` for evidence layers.
6. `web_spatial_from_mask` + `MapOverlay` for reprojection.
7. `st.download_button` for export.
8. `Status`, `state_counts`, and Phase 12's own unknown/insufficient wording rules.

---

## 3. What is missing today

| Gap | Why it matters |
|---|---|
| No **evidence record type** | Provenance is spread across four result shapes; nothing has a stable ID, source dataset, band/index, condition, operator, threshold, grid identity and counts in one serializable place. |
| No explicit **lineage** object | The chain query → intent → conditions → sources → grid → mask → combined → statistics → answer is *implied* by the code, not represented. |
| No **deterministic explanation generator** | Answer text is currently assembled ad hoc inside each panel. |
| No **per-condition "why"** with parameters | The panel shows labels and counts, not "Class: 40", "Distance: 1000 m", "Origin: from your query" grouped per condition. |
| No **per-condition evidence layers** | Only the combined mask is on the map. |
| No consolidated **reproducibility block** | Grid/transform/alignment/budget exist in the result but are shown only inside collapsed engine-specific expanders. |
| No **JSON export** | No machine-readable evidence package exists at all. |
| No shared **JSON sanitiser** | `MultiConditionResult.to_dict()` embeds live `GridMask` objects in `condition_results[…]["mask"]`, so the existing dict is **not** JSON-safe as-is. |
| No Phase 13 tests / verification / docs | Steps 13–16. |

---

## 4. Where Phase 13 integrates

```
core/evidence.py        NEW   contract: EvidenceRecord, Lineage, EvidencePackage
                              (facts only — no arrays, no engines, no UI imports)
analyses/evidence.py    NEW   build_evidence(result) -> EvidencePackage
                              explain(package) -> deterministic text blocks
ui/components.py        MOD   render_evidence_package(): Answer / measured /
                              why / sources / details / limitations / export
ui/map.py               MOD*  (only if a second legend variant is needed —
                              the existing composition palette is reused as-is)
app.py                  MOD   dispatch after each intent panel; opt-in evidence
                              layers; export download button
analyses/__init__.py    MOD   re-export the two new helpers (house convention)
tests/ scripts/ docs/   NEW   Step 13–16 deliverables
```

`core/evidence.py` imports **nothing** from `analyses.*` — the same layering rule
`core/multi_condition.py` obeys. `analyses/evidence.py` is the only place that
reads engine result objects.

---

## 5. Design of the contract (Step 2)

```python
@dataclass(frozen=True)
class EvidenceRecord:
    id: str                 # deterministic: f"{kind}:{name}" (+ index, stable order)
    kind: str               # "spatial" | "spectral" | "temporal" | "combined" | "statistics"
    label: str              # as measured, never reinterpreted
    source_analysis: str    # "worldcover" | "core.indices:ndvi" | "analyses.ndvi_change"
    source_dataset: str     # scene path / dataset label / "unavailable"
    source_dates: Tuple[str, ...]
    band_or_index: str      # "ndvi" | "ndwi" | "land cover class" | ""
    condition: str          # "ndvi_gt" | "land_cover_class" | "water_proximity"
    parameters: Dict        # {"classes": [40]} | {"distance_m": 1000} | {"class": "decrease"}
    operator: Optional[str]
    threshold: Optional[float]
    threshold_provenance: Optional[Dict]     # ThresholdSpec.to_dict()
    negated: bool
    grid: Dict              # Grid.to_dict()
    counts: Dict            # matched / non_matching / insufficient / total
    area_m2: Optional[float]
    fraction: Optional[float]
    runtime_ms: Optional[float]
    limitations: Tuple[str, ...]
    provenance: Dict        # the mask/engine provenance dict, verbatim

    def to_dict(self) -> Dict[str, Any]      # JSON-safe by construction
```

* **Facts only.** An evidence record never holds a raster, a mask or a pointer
  to one; the map layers read arrays from the live result object at render time.
  This keeps the export small and the contract honest.
* **Lineage** is a separate frozen structure with the nine named links of
  Step 3 (`query → normalized → intent → conditions → sources → grid → mask →
  combined → statistics → answer`), each holding the value it names.
* **`EvidencePackage`** = `{query, normalized_query, intent, status, lineage,
  records[], combined record, statistics[], sources[], grid, alignment,
  threshold_provenance, unknown handling, limitations, boundary, runtime}` plus
  `to_json()`.

---

## 6. Explanation generation (Step 4)

`analyses/evidence.py::explain(package)` returns a dict of **text blocks** built
by templates — no branching on data values beyond the four outcome classes:

| Outcome | Wording (deterministic) |
|---|---|
| matches found | "X cells satisfy <expression>; Y cells were measured and do not; Z were undecided." |
| zero matches | "No cells satisfy <expression>; every measured cell was evaluated and did not match." |
| partial unknown | "X cells matched; Y cells were undecided and were **not** counted as matches." |
| all unknown | "No cells could be evaluated with sufficient data." — never "no matches". |

Every block ends with the boundary sentence pulled from the reused `CAVEAT`
whenever the package is a composed geographic condition.

---

## 7. Unknown and threshold semantics (Steps 9–10)

Preserved by construction: the record stores `insufficient` as its own counter,
the package records `unknown_handling`, and the wording table above is selected
on `analysed_cell_count == 0` (all unknown) rather than on `matched == 0`.
Thresholds are emitted as `{value, operator, provenance, detail}` verbatim from
`ThresholdSpec`; no renderer may print "validated" — the only permitted wordings
are those already used in Phase 12.

---

## 8. Map evidence layers (Step 6)

* Per-condition layers are **opt-in** (a checkbox group); the combined layer
  stays exactly as it is, defaulting to on.
* Names carry provenance: `Evidence — Cropland (WorldCover class 40)`,
  `Evidence — NDVI > 0.6`, `Evidence — NDVI decrease (2023-01-18 → 2023-08-06)`,
  alongside the unchanged `Composed conditions (MATCH / NO MATCH / UNKNOWN)`.
* Same three-valued colours, so amber continues to mean exactly one thing.
* Risk to handle: `web_spatial_from_mask` is `@st.cache_data(max_entries=3)`;
  adding up to N condition layers would evict the combined layer. A separate
  cached helper with its own budget will be added rather than raising the
  existing one (so Phase 12's cache behaviour is untouched).

---

## 9. Export (Step 8)

* In-app: `st.download_button("Download evidence package (JSON)", …)` following
  the existing export convention, file name `satquery_evidence_<intent>.json`.
* On disk: `scripts/verify_phase13_real_data.py` writes
  `artifacts/phase13_evidence_caseA.json` / `…_caseB.json` so the package is
  inspectable as a workspace artefact.
* Content: only facts already in the result + the deterministic explanation
  blocks. Arrays are excluded; where a fact is unavailable the value is the
  string `"unavailable"` — never invented.
* JSON safety: a shared sanitiser converts numpy scalars, tuples, CRS objects
  and datetimes via `default=str`, and drops the `mask` keys that
  `MultiConditionResult.to_dict()` currently embeds.

---

## 10. Files

**Added:** `core/evidence.py`, `analyses/evidence.py`,
`tests/test_phase13_evidence.py`, `scripts/verify_phase13_real_data.py`,
`scripts/verify_phase13_browser.py`, `docs/PHASE13.md`,
`artifacts/phase13_evidence_case{A,B}.json`.

**Changed:** `ui/components.py` (new panel + why-blocks + export),
`app.py` (dispatch, opt-in evidence layers, download button),
`analyses/__init__.py` (re-export), and `ui/map.py` **only if** a legend variant
is needed (likely none).

**Deliberately untouched:** `core/multi_condition.py`,
`analyses/multi_condition.py`, `analyses/spatial_query.py`,
`core/spatial_query.py`, `core/spatial.py`, `analyses/ndvi_change.py`,
`core/temporal.py`, `analyses/ndvi.py`, `analyses/ndwi.py`, `core/indices.py`,
`core/index_definitions.py`, `core/alignment.py`, `analyses/base.py`,
`analyses/registry.py`, `core/router.py`, `config/**`, `docs/PHASE9_REPORT.md`,
`docs/PHASE10.md`, `docs/PHASE11.md`, and every PPT file.

*(If a gap is later found that genuinely requires touching a frozen file, it
will be reported first rather than changed.)*

---

## 11. Open decisions before implementation

1. **Scope of the package** — every intent (NDVI/NDWI/temporal/spatial/
   composition), with a deep per-condition breakdown only for compositions
   *(recommended)*, or compositions only?
2. **Evidence map layers** — opt-in checkbox group, off by default
   *(recommended)*, or on by default?
3. **Export** — in-app download **and** JSON artefacts written by the
   verification script *(recommended)*, or in-app download only?
