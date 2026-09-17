# Phase 7 — End-to-end query understanding + analysis router

*What we are building, why the router must not do mathematics, how it works, and
how to verify it.*

---

## 1. What we are building

Phases 1–6 gave SatQuery correct capabilities that the **user** had to assemble
by hand. Phase 7 adds the conversational entry point:

```
USER QUERY → QUERY INTENT → CORRECT EXISTING ANALYSIS → RESULT → MAP / UI EVIDENCE
```

"Understand the request" becomes a layer of its own. Nothing new is computed:
every number still comes from the Phase 6 NDVI engine.

---

## 2. Why a router, and why separate from the engines

| Router (`core/router.py`) | Engine (`analyses/ndvi.py` → `core/statistics.py`) |
|---|---|
| Understands a sentence | Understands pixels |
| Returns *what was asked* | Returns *what is true* |
| Cheap, pure text → data | Raster/geometry I/O |
| Must be explainable | Must be numerically verifiable |
| Will be swapped for an LLM later | Must never be re-verified because the parser changed |

Mixing them makes both unverifiable: every parser tweak would risk the
mathematics, and every numerical bug would only be debuggable through the UI.
The rule is enforced, not just intended:

* `core/router.py` imports no numpy/rasterio/shapely/pyproj (`verify_phase7.py`
  parses the file's AST **and** imports it in a clean subprocess to prove it);
* `analyses/*` never touch `st.session_state`;
* `app.py` contains no `if "ndvi" in query:` — the only binding lives in the
  registry.

---

## 3. Why deterministic matching instead of an LLM

* **Auditable** — every answer carries the pattern that fired and a confidence.
* **Exactly testable** — including the ambiguous and unsupported cases.
* **Offline, free, instant** — no key, no network, no latency.
* **Honest by construction** — "can I grow cotton here?" matches a *declared but
  engineless* intent and gets "not available yet" instead of a plausible story.

**Swapping in an LLM later** means writing another function with the same
signature:

```python
parse_query(text) -> QueryIntent
```

constrained to emit only `Intent` members the registry knows. `app.py`, the UI,
the engines and every existing test stay untouched — they depend on
`QueryIntent`, not on how it was produced.

---

## 4. Intent model

```python
Intent.NDVI_ROI_STATS        # implemented  (Phase 6 engine)
Intent.CROP_SUITABILITY      # planned: recognised, NO handler
Intent.FLOOD_CHANGE          # planned: recognised, NO handler
Intent.VEGETATION_CHANGE     # planned: recognised, NO handler
Intent.UNKNOWN               # parser outcome, never an engine
```

`QueryIntent` is the parser's structured output:

```python
QueryIntent(intent, confidence, original_query, normalized_query,
            matched, explanation, required_context, rejected)
```

**Statuses are deliberately separate from intents** — the intent is *what the
user asked*, the status is *whether it could be answered*:

| Status | Meaning |
|---|---|
| `OK` | executed, `result` present |
| `NEEDS_ROI` | understood, but nothing is selected |
| `NEEDS_NDVI_CONFIRMATION` | understood, but the Phase 3 band gate is closed |
| `NO_VALID_PIXELS` | executed; the selection holds no usable NDVI pixel |
| `UNSUPPORTED` | intent exists, engine does not (yet) — nothing was computed |
| `UNKNOWN` | the parser refused to guess |
| `ERROR` | the engine raised; nothing is reported |

---

## 5. How matching works (evidence-scored, not string equality)

1. **Normalize**: NFKC → lower → strip punctuation → collapse whitespace → apply
   a small synonym map (*"vegetation index" → "ndvi"*, *"analyse" → "analyze"*).
2. **Score**: phrase hit `+3`, token hit `+1`, negative hit `−2`
   (e.g. *change / difference / flood / suitability / weather* veto the
   single-date NDVI), confidence = `clamp(score / 4)`.
3. **Decide**: accept when confidence ≥ 0.34 **and** the winner is not tied.
   A tie means the sentence is genuinely ambiguous → `UNKNOWN`.
4. **Explain**: every accepted intent records which patterns fired; the UI and
   the tests both use it.

`Tell me about this area.` → 0.00 → `UNKNOWN`.
`show me the data` → 0.25 (one weak token) → `UNKNOWN`.
`What is the NDVI of this area?` → 1.00, matched `('ndvi', 'ndvi of')`.

---

## 6. The registry — the only place intents are bound to code

```python
REGISTRY = {
    Intent.NDVI_ROI_STATS: AnalysisSpec(handler=run_ndvi_roi_stats,
                                        requires=("roi", "ndvi_confirmed"), ...),
    Intent.CROP_SUITABILITY: AnalysisSpec(handler=None,
                                          unavailable_message="Crop suitability analysis is not available yet."),
    ...
}
```

`route(query, context)` is the single entry point:

```
parse → unknown? → UNSUPPORTED/UNKNOWN
      → no handler? → UNSUPPORTED (planned, computes nothing)
      → validate context → NEEDS_ROI / NEEDS_NDVI_CONFIRMATION
      → handler(context, intent) → AnalysisExecution
```

`AnalysisExecution` is structured: `intent, status, query, normalized_query,
confidence, explanation, matched, result, message, warnings, provenance`. The UI
renders from it; nothing is scraped from a widget.

---

## 7. Context validation

| Situation | Answer |
|---|---|
| No ROI | "Please select an area on the map first." (`NEEDS_ROI`, `result=None`) |
| NDVI not confirmed | "Please confirm the detected satellite bands before running NDVI analysis." |
| ROI with no pixels | "No raster pixels fall inside the selected area." (`NO_VALID_PIXELS`) |
| ROI with only invalid pixels | "No valid NDVI pixels were found inside the selected area." |
| Unsupported | "Crop suitability analysis is not available yet." |
| Ambiguous | "I could not match that to an available analysis. I can currently analyze NDVI (a vegetation index) for a selected area. Try asking: …" |

The context is built **once** in `app.py` (`AnalysisContext`) and handed to the
router as plain data — engines never read session state.

---

## 8. UI

New section **5 · Ask SatQuery**: a chat input plus the last five turns. Each
turn shows **Query → Intent (+confidence and the matched pattern) → Answer**, and
for a successful NDVI execution it renders the *same* `render_roi_analysis()`
panel used in section 4 — so the chat cannot drift from the analysis it
describes, because it does not re-implement it. The map, the layers and the ROI
are untouched; the router only requests results.

Wording rule: "vegetation index", "NDVI value", "observed NDVI". The answer adds
an explicit caveat that it is **not** a crop-health, yield or disease diagnosis.

---

## 9. How to verify it

```bash
python -m pytest tests -q                    # 266 passed
python scripts/verify_phase7.py              # 45/45 checks (real Sentinel-2)
python scripts/browser_test_phase7.py        # 28/28 checks in real Chromium
```

### End-to-end identity (the central claim)

On the real tile, routing changes nothing:

```
query   : "What is the NDVI of this area?"
intent  : NDVI_ROI_STATS (confidence 1.00)
answer  : Selected area has a mean NDVI of 0.7772 over 10,000 valid pixels
stats   : inside=10,000 valid=10,000 mean=0.7772 median=0.8495 std=0.1951
          min=0.0235 max=0.9652 P5=0.2588 P95=0.9307
area    : 100.00 ha, valid 100.00 ha

[PASS] the whole structured result is identical to calling Phase 6 directly
[PASS] pixel count matches an independent point-in-polygon test (10,000)
[PASS] mean NDVI matches the independent recomputation (0.777236 / delta 0.0e+00)
```

`tests/test_phase7_end_to_end.py` asserts dict-level equality between
`route(...)` and `calculate_roi_ndvi_stats(...)`; `verify_phase7.py` re-derives
the numbers again with `matplotlib.path.Path` + numpy.

### Browser (28/28)

Confirm bands → draw an ROI with the mouse → ask the question → intent, answer,
statistics and histogram appear (mean 0.8102 over 73 367 valid pixels) →
"Can I grow cotton here?" shows `CROP_SUITABILITY / not available yet` →
"Show flood areas." shows `FLOOD_CHANGE / not available yet` →
"Tell me about this area." is refused → delete the ROI → the same NDVI question
answers "Please select an area on the map first."

---

## 10. Plugging in a future analysis (the architectural test)

Adding NDWI later requires exactly three steps and **no** change to the router,
`app.py` or the UI:

1. `analyses/ndwi.py` — `run_ndwi_roi_stats(context, query) -> AnalysisExecution`
   wrapping the NDWI engine, with `Status` handling for missing context;
2. `core/router.py` — add `Intent.NDWI_ROI_STATS`, its phrases/tokens/negatives,
   and its `REQUIRED_CONTEXT` entry;
3. `analyses/registry.py` — one `AnalysisSpec` row (handler, requirements,
   example queries).

The UI picks it up automatically: the suggestion line under the chat input is
generated from `suggestions()`, which iterates the registry. Today's planned
intents (`CROP_SUITABILITY`, `FLOOD_CHANGE`, `VEGETATION_CHANGE`) are already
wired this way — recognised, advertised as unavailable, and incapable of
producing a number.

---

## 11. Known limitations

1. **One executable intent.** Only `NDVI_ROI_STATS` has an engine; the rest are
   declared placeholders.
2. **Keyword/phrase vocabulary, not language understanding.** Paraphrases outside
   the pattern table fall through to `UNKNOWN` by design. Synonyms are added as
   data, not code changes.
3. **English only, single-turn.** No multi-turn context (e.g. "and for that other
   area?") and no follow-up refinement.
4. **No query history persisted** across sessions (last five turns, in memory).
5. **The router cannot ask a clarifying question** — it reports `UNKNOWN` with a
   suggestion instead.
6. **Unchanged from Phase 6**: the map component can go silent after every shape
   is deleted while the NDVI layer is shown (workaround: **Reset view**), and one
   session with the 2048² scene costs ≈0.7 GB in a ≈2 GB sandbox.
7. **Clouds are still not masked**, so cloudy pixels count as valid NDVI.
