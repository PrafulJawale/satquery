# Phase 13 — Evidence-Backed Geospatial Answers and Result Explainability

Phase 12 made SatQuery able to answer *"find cropland with NDVI > 0.6 and NDWI <
−0.4"*. Phase 13 makes it able to **show its work**: every number in every
answer can now be traced back to the query that asked for it, the analysis that
produced it, the scene and date it came from, the grid it was computed on, the
threshold that decided it, and the cells that could not be decided at all.

**Phase 13 adds an explanation of an existing measurement. It does not add a
measurement, an index, a threshold, a flood assessment, a crop-failure finding
or a causal claim.**

### The scientific boundary (unchanged, and now printed by the evidence layer)

A combined condition is geographic evidence, not causal attribution. Phase 13
reuses that one sentence from `analyses.multi_condition.CAVEAT`; it does not
create a second copy, so the panel, the legend and the exported JSON cannot
drift apart.

---

## 1. Objective

| # | Requirement | Where it lives |
|---|---|---|
| 1 | A serializable evidence contract (facts only) | `core/evidence.py` |
| 2 | Full lineage for every answer: query → intent → conditions → analysis → scene → grid → mask → statistics → answer | `core/evidence.py::Lineage`, `analyses/evidence.py` |
| 3 | A deterministic explanation generator — no LLM | `analyses/evidence.py::explain` |
| 4 | "Why this result?" per condition | `ui/evidence_panel.py` |
| 5 | Optional per-condition evidence map layers | `ui/evidence_panel.py`, `analyses/evidence_masks.py` |
| 6 | Reproducibility / analysis-details block | `analyses/evidence.py::grid_rows` |
| 7 | Machine-readable JSON export | `EvidencePackage.to_json()` + download button |
| 8 | Phase 12 unknown/threshold semantics preserved exactly | everywhere (no engine change) |
| 9 | Scientific-language guard tests | `tests/test_phase13_evidence.py` |
| 10 | Existing UI unchanged | `app.py` (append-only) |
| 11–15 | Tests, real-data and browser verification, this document | `tests/`, `scripts/`, `docs/` |

Three decisions shaped the implementation (all confirmed before coding):

1. **every intent** gets an evidence package — NDVI, NDWI, temporal, spatial,
   composition — with the deep per-condition breakdown wherever one exists;
2. per-condition map layers are **opt-in and off by default**;
3. the JSON export is an **in-app download only** — the verification scripts
   write no JSON artefacts to `artifacts/`.

---

## 2. The evidence contract (`core/evidence.py`)

Three immutable dataclasses, in the same style as the Phase 9–12 result
objects (`frozen=True`, `to_dict()`, no behaviour that interprets anything):

* **`EvidenceRecord`** — one measured thing. Identity (`id`, `kind`, `label`),
  source (`source_analysis`, `source_dataset`, `source_dates`, `band_or_index`),
  the decision (`condition`, `parameters`, `operator`, `threshold`,
  `threshold_provenance`, `negated`), the geometry (`grid`), the outcome
  (`counts`, `area_m2`, `fraction`, `runtime_ms`) and its own `limitations`.
  Kinds: `spatial`, `spectral`, `temporal`, `combined`, `statistics`.
* **`Lineage`** — the chain, link by link: query → normalized query → intent →
  conditions → source analyses → grid/ROI → condition masks → combined mask →
  statistics → final answer. `Lineage.steps()` renders it as (label, value)
  pairs so it can be printed or asserted against.
* **`EvidencePackage`** — the auditable package: the records, the combined
  record, the statistics, the sources, the grid, the alignment, the threshold
  provenance, the counts, the limitations, the boundary and the generated
  explanation.

Rules the module enforces:

* **facts only** — no record stores an array and none stores a conclusion;
* **`"unavailable"`, never a guess** — a fact that does not exist is the string
  `UNAVAILABLE`, not a plausible substitute and not a zero;
* **three-valued, always** — `insufficient` sits beside `matched` and
  `non_matching` in every count block (`counts_of()`), so an undecided cell can
  never silently become a non-match downstream;
* **JSON-safe by construction** — `json_safe()` converts numpy scalars, drops
  arrays (data, not evidence) and reports anything unrecognised as
  `"unavailable"` instead of printing a repr that looks like data;
* **deterministic** — `to_json()` sorts keys, so the same result exports the
  same bytes every time.

`core/evidence.py` imports nothing from `analyses.*` — the same layering rule
`core/multi_condition.py` obeys. The builders that read engine results live in
`analyses/evidence.py`.

---

## 3. Evidence lineage: from the question to the number

For **"find cropland with NDVI > 0.6 and NDWI < −0.4"** over the bundled scene
the package reconstructs the whole chain, and every step is a value that can be
printed or asserted:

| Step | Value |
|---|---|
| User query | `Find cropland with NDVI greater than 0.6 and NDWI less than -0.4` |
| Normalized query | `find cropland with ndvi greater than 0 6 and ndwi less than -0 4` |
| Intent | `MULTI_CONDITION` |
| Conditions | `land_cover_class AND ndvi_gt AND ndwi_lt` |
| Source analyses | `ESA WorldCover 2021 v200`, `core.indices:ndvi`, `core.indices:ndwi` |
| Grid / ROI | `516 x 516 cells at 10.0 m (EPSG:32636)` |
| Condition masks | one per condition, on that one verified grid |
| Combined mask | `three-valued AND` |
| Statistics | NDVI/NDWI measured over the match, never used as a filter |
| Final answer | `204,521 cells satisfy … over 262,144 measured cells (4,112 undecided)` |

and, per condition:

| Condition | Source | Parameter | Provenance |
|---|---|---|---|
| land cover class [40] (Cropland) | ESA WorldCover 2021 v200 | Class: 40 | ESA WorldCover 2021 v200 |
| NDVI > 0.6 | `core.indices:ndvi` | Threshold: > 0.6 | from your query. |
| NDWI < −0.4 | `core.indices:ndwi` | Threshold: < −0.4 | from your query. |

Cropland is traced to **ESA WorldCover 2021 v200 class 40**; NDVI and NDWI are
traced to the analytical bands through the **Phase 11 index engine**; both
thresholds are traced to the **user's own query**; the combination is traced to
**three-valued AND**. The answer is never reported as a bare number.

---

## 4. The explanation generator — templates, not a model

`analyses/evidence.py::explain()` produces seven blocks from the package:

1. **what was asked** — the normalized query;
2. **what was evaluated** — one line per condition, each naming its analysis,
   its parameter and where the parameter came from;
3. **what was found** — matched / measured non-matching / unknown cells, area
   and fraction;
4. **how it was evaluated** — "three-valued AND logic on a common 10 m grid",
   with the rule that a missing cell stays UNKNOWN and is never counted as a
   non-match;
5. **evidence sources** — the datasets and scenes;
6. **unknown handling** — its own sentence, always;
7. **limitations and the boundary** — copied from the result, never reworded.

There is no LLM, no embedding, no retrieval and no index. The same package
always produces the same text (`test_24`).

Three wording rules are enforced by tests, because they are the three ways an
honest answer can be turned into a dishonest one:

* **"nothing could be measured" is never printed as "nothing matched"** — the
  outcome is chosen on `analysed_cells`, never on `matched_cells` alone
  (`all_unknown` → *"No cells could be evaluated with sufficient data."*);
* **a measurement is not described as a filter** — *"What is the NDVI here?"*
  produces *"NDVI was measured over … valid cells"*, never "cells satisfy …";
* **a classification is not described as a selection** — a Phase 10 change
  result says *"classified … on the shared grid"*, because Phase 10 labels every
  cell rather than selecting some.

---

## 5. The "Why this result?" panel

Rendered by `ui/evidence_panel.py` **below** the existing panel for every
answered intent. Layout, per the brief: the primary answer stays one sentence
and everything else lives in expanders — except the boundary, which is printed
in the open for composed conditions.

```
Why this result?
Answer — 204,521 cells satisfy land cover class [40] (Cropland) AND NDVI > 0.6
         AND NDWI < -0.4 over 262,144 measured cells (4,112 undecided).
What was measured
  · land cover class [40] (Cropland) — from ESA WorldCover 2021 v200, Class: 40
  · NDVI > 0.6 — from core.indices:ndvi, > 0.6 (from your query.)
⚠️ A combined condition is geographic evidence, not causal attribution.
[Matched / valid cells] [Measured, not matching] [Undecided cells] [Matched area]
> Why these cells matched      (Condition · Source · Threshold/parameter ·
> Evidence sources              Provenance · Matched · Not matching · Undecided)
> Analysis details             (query → grid → thresholds → runtime)
> Limitations                  (the engine's own sentences, verbatim)
[ ] Show individual condition layers on the map
[ Export evidence (JSON) ]
```

The per-condition table carries exactly the columns the brief names: condition,
source, threshold/parameter, provenance, matched, measured non-matching and
unknown.

---

## 6. Evidence map inspection (opt-in)

A checkbox — **off by default** — adds one display layer per condition, named
with its provenance:

* `Evidence — land cover class [40] (Cropland)`
* `Evidence — NDVI > 0.6`
* `Evidence — NDWI < -0.4`

Rules that protect the existing map:

* the **combined layer stays**, and every pre-existing layer (footprint, true
  colour, false colour, NDVI, NDWI, suitability, change) stays;
* nothing is added until the user switches it on;
* the layer set belongs to **one answer** — a new query clears it;
* the layers are **the engine's own masks**, read back from
  `condition_results[i]["mask"]` (or rebuilt from the `condition_masks` state
  codes through `GridMask.from_state`). No threshold is re-applied and no
  pixel is re-classified, so a layer can never disagree with its count — a test
  asserts the match count of each layer equals the count in its record;
* a temporal result has no condition masks, so its classes are derived from the
  `class_raster` Phase 10 already computed (code 0 = insufficient, never a
  match);
* the reprojection uses its **own** cached helper (`web_evidence_mask`) so it
  cannot evict the combined layer from the map's cache; outside-ROI cells are
  transparent because they were never analysed.

---

## 7. Reproducibility / analysis details

`grid_rows()` produces the block the brief lists, in order:

```
Query · Normalized query · Intent · Analysis dates · CRS · Grid · Resolution ·
Transform · Alignment method · Resampling · Analysed cells · Undecided cells ·
Matched area · Unknown handling · Runtime · Threshold — <name> (one per threshold)
```

Nothing is invented: when the alignment is unknown the block says
**`unavailable`** rather than claiming "computed on the native grid"; when a
single-date result records no scene date it says *"not recorded for this
result"*. A composition stores its alignment per source, so the block unwraps
the one that produced the grid (`temporal: identical_grid, resampled false`).

---

## 8. Machine-readable export

`EvidencePackage.to_json()` — schema `satquery-evidence/1`, sorted keys,
11–12 KB for a three-condition answer. It contains the result facts, the
lineage, the records, the sources, the grid, the alignment, the threshold
provenance, the limitations, the boundary and the generated explanation —
**and nothing else**. A test asserts the export's top-level keys are a subset of
the agreed list.

The UI exposes it as one download button only
(`satquery_evidence_<intent>_<n>.json`, `application/json`); as agreed, the
verification scripts write no JSON artefacts.

---

## 9. Preserved semantics

| Guarantee | How Phase 13 keeps it |
|---|---|
| UNKNOWN is never a match and never a non-match | `counts_of()` carries `insufficient` beside `matched`/`non_matching`; `matched + non_matching == analysed` is asserted |
| insufficient data is never reported as zero matches | `unknown_handling_for()` returns `all_unknown` only when nothing was decided, and the wording branch is chosen on `analysed_cells` |
| thresholds stay labelled | `threshold_origin_text()` prints "from your query.", "a display/query convention (*x*) — enabled by you, and **not a scientific classification**", or "relative to this area" |
| no silent resampling | the grid and alignment are reported, not assumed; an unknown alignment says `unavailable` |
| grids are never reinterpreted | Phase 13 reads `result.grid`; it never builds one |
| Phase 9/10/11/12 semantics | untouched — no engine file was modified |

---

## 10. The scientific-language guard

`test_32` runs **five intents through the real explanation path** — composition,
temporal composition, NDVI statistics, NDWI statistics, Phase 9 spatial, Phase
10 temporal — and asserts the generated text contains none of:

`flooded`, `flood extent`, `crop failure`, `drought`, `deforestation`, `damage`,
`water availability`, `water quality`, `caused by`, `because of`, `cause of`,
`scientific classification`, `scientifically validated`, `validated threshold`.

The scan covers what Phase 13 **generates**, not the engine's limitations
(those sentences are where the denials live — *"NDWI does not establish water
availability…"* — and they are copied verbatim, never reworded). The browser
script makes the same check against the rendered page.

---

## 11. Files added, changed, untouched

**Added**

| File | Purpose |
|---|---|
| `core/evidence.py` | the contract: `EvidenceRecord`, `Lineage`, `EvidencePackage`, `json_safe`, `counts_of`, `unknown_handling_for` |
| `analyses/evidence.py` | the builders (one per result shape), `explain()`, `condition_rows`, `source_rows`, `grid_rows`, `threshold_origin_text` |
| `analyses/evidence_masks.py` | the per-condition masks behind the evidence layers |
| `ui/evidence_panel.py` | the panel, the opt-in layers, the export control |
| `tests/test_phase13_evidence.py` | 48 tests |
| `scripts/verify_phase13_real_data.py` | 29 real-data checks |
| `scripts/verify_phase13_browser.py` | 30 browser checks |
| `docs/PHASE13.md` | this document |

**Changed**

| File | Change |
|---|---|
| `app.py` | +import of the panel; +one `render_evidence(entry)` call in the chat loop (append-only); +the opt-in evidence-layer block in the overlay section |

**Untouched** — every Phase 9/10/11 file, `core/multi_condition.py`,
`analyses/multi_condition.py`, `analyses/spatial_query.py`,
`core/spatial_query.py`, `core/spatial.py`, `analyses/ndvi_change.py`,
`core/temporal.py`, `analyses/ndvi.py`, `analyses/ndwi.py`, `core/indices.py`,
`core/index_definitions.py`, `core/alignment.py`, `analyses/base.py`,
`analyses/registry.py`, `core/router.py`, `config/**`,
`docs/PHASE9_REPORT.md`, `docs/PHASE10.md`, `docs/PHASE11.md`, all PPT files.

Phase 13 is a **read-only interpretation layer**: it reads frozen result
objects and never re-runs, re-implements or re-decides an analysis.

---

## 12. Integration points in `app.py`

* chat-history render loop (~line 1867): `render_evidence(entry)` is called
  after the existing panel, for every answered entry;
* overlay section (~line 1348): after the combined layer is appended, the
  opt-in evidence layers are appended;
* session state: `evidence_layers_on` (bool) and `evidence_layers_owner`
  (`id(result)`), so a layer set belongs to one answer and toggling it triggers
  exactly one rerun.

Every rendering path is wrapped so a display-only failure can never break an
answer: a layer that cannot be drawn is reported as unavailable.

---

## 13. Tests — `tests/test_phase13_evidence.py` (48)

| Group | Tests | Covers |
|---|---|---|
| Contract / serialization | 01–08 | fields, immutability, unknown classes, zero denominators, `json_safe`, round-trip, determinism, no extra conclusions |
| Lineage | 09–14 | WorldCover class 40, index engine + user thresholds, three-valued AND, temporal dates, the full chain, "not just the final number" |
| Explanation | 15–24 | counts, area, dates, thresholds/origin, zero matches, all-unknown, partial unknown, statistics ≠ filter, classification ≠ selection, determinism |
| Unknown handling | 25–28 | unknown never a match, nothing-measured wording, grid metadata, missing metadata |
| Scientific boundary | 29–33 | caveat identity (`is`), boundary presence, banned words over five intents, conventions stay labelled |
| Reproducibility | 34–36 | condition rows, details rows, threshold provenance rows |
| Layers / integration / regression | 37–45 | layers are the engine's masks, temporal class masks, refusals have no package, entry ↔ execution parity, no mutation, panel API (layer helper exercised), Phase 12 results unchanged, two real-data cases |

**Full suite: 736 passed** (688 before Phase 13 + 48 new). No existing
assertion was weakened.

---

## 14. Real-data verification — `scripts/verify_phase13_real_data.py` (29/29)

Same bundled scene, same 5.12 km ROI as Phase 12, every number derived from the
existing engine:

* **Case A** — cropland AND NDVI > 0.6 AND NDWI < −0.4 → **204,521 cells /
  20.452 km² / 4,112 undecided**, three records (one spatial, two spectral),
  thresholds traced to the user's query, grid 516 × 516 @ 10 m;
* **Case B** — NDVI decrease AND near permanent water → **4,511 cells /
  0.451 km²**, dates **2023-01-18 → 2023-08-06**, alignment `identical_grid`
  with `resampled false`, both source analyses named;
* an independent hand calculation of the ROI from raw band values is printed
  alongside, so the three states can be checked without trusting the engine;
* both packages export byte-for-byte deterministically (11.5 KB / 11.4 KB) and
  generate no causal claim;
* a refused query (`Find cropland with high NDVI`) produces **no package at
  all**.

---

## 15. Browser verification — `scripts/verify_phase13_browser.py` (30/30)

One fresh server, one drawn ROI: the evidence panel renders; the Phase 12 panel
renders unchanged and reports the same numbers; the combined layer and every
pre-existing overlay remain; no evidence layer appears until the switch is
clicked, then one appears per condition with provenance in its name; both
analysis dates are displayed for Case B; the undecided count is displayed; the
boundary is on screen; the export produces
`satquery_evidence_multi_condition_0.json`; NDVI statistics still answer;
flooding and temporal NDWI are still unsupported *by name*; a bare "high NDVI"
is still refused.

Every question is **verified against the transcript and re-asked** if Streamlit
swallowed the keystrokes during a rerun.

**Phase 12 regression after the `app.py` change: 35/35 browser checks and the
real-data script still pass.**

---

## 16. Known limitations

* **The evidence is only as good as the analysis it describes.** Phase 13
  cannot detect an error it did not make; it can only make every step visible.
* **Per-condition layers are display copies.** They are reprojected for the map
  with nearest-neighbour resampling; the counts were computed on the analysis
  grid, never on the reprojected image. This is stated in the layer's help text.
* **A single-date composition records no scene date**, because the engine does
  not carry one; the details block says so rather than guessing from a file
  name.
* **Phase 9 per-condition counts are the ones Phase 9 reports** (buffered
  window grid); Phase 13 copies them verbatim instead of rescaling them to the
  ROI.
* **Phase 10 evidence layers are not offered**: the change map already shows the
  classes, and re-encoding them would add a second rendering of the same
  classification.
* **The export is a snapshot of one answer**, tied to the session that produced
  it; it is not a provenance graph of the whole project.
* **Memory**: with five answers open, each carrying its own evidence panel, the
  Streamlit process in this 2 GB sandbox can be OOM-killed under repeated
  browser automation. It is a sandbox limit, not an application defect, but it
  is why the browser script verifies a fresh server.

---

## 17. What Phase 13 still does not do

Flood or flood-extent detection, crop-failure or drought assessment,
deforestation, damage assessment, land-cover change, anomaly detection,
prediction, forecasting and causal inference are all still out of scope — and
now also **out of the wording**: the guard test fails if any of those claims
appears in a generated explanation.

No new spectral index, no new dataset, no new threshold, no embedding, no
vector index, no language or vision model, and no Phase 9–12 semantics were
reinterpreted. Phase 13 is the last planned phase.
