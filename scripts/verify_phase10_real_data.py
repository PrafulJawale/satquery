"""Phase 10 -- reproducible REAL-DATA verification of the NDVI change engine.

This is the evidence behind the numbers in docs/PHASE10.md. It runs entirely on
the two bundled Copernicus Sentinel-2 acquisitions -- no synthetic data -- and
prints, in one place:

    1. the two scenes, their dates, platforms, tiles and licences
    2. an INDEPENDENT hand-check on an 8x8 block (raw DN -> NDVI -> delta)
    3. the engine's result over a 5.12 km window of the real scene
    4. the classification breakdown and the alignment that was used

Run it with the Streamlit server STOPPED if the sandbox is short of memory.

    python scripts/verify_phase10_real_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from analyses.ndvi_change import compare_ndvi, compose_message
from core.temporal import ScenePair, SceneRef, grids_identical, load_temporal_config

REPO = Path(__file__).resolve().parent.parent
BEFORE = REPO / "data/sample" / "s2_s2b-36ruv-20230118-0-l2a_2048px.tif"
AFTER = REPO / "data/sample" / "s2_s2b-36ruv-20230806-0-l2a_2048px.tif"

ROW, COL, SIZE = 1000, 1000, 8          # the hand-check block


def line(char: str = "-", n: int = 74) -> str:
    return char * n


def main() -> int:
    if not (BEFORE.exists() and AFTER.exists()):
        print("the two Sentinel-2 scenes are not present -- run "
              "scripts/fetch_sentinel2_second_scene.py")
        return 1

    before = SceneRef.from_path(str(BEFORE))
    after = SceneRef.from_path(str(AFTER))
    cfg = load_temporal_config()

    print(line("="))
    print("PHASE 10 -- REAL-DATA VERIFICATION")
    print(line("="))

    # -- 1. the two acquisitions ------------------------------------------- #
    print("\n1. THE TWO ACQUISITIONS")
    for role, scene in (("before", before), ("after", after)):
        src = scene.source
        print(f"   {role:6s} {src['item_id']}")
        print(f"          date={src['datetime'][:10]}  platform={src['platform']}  "
              f"tile={src['mgrs_tile']}  EPSG:{src['epsg']}")
        print(f"          cloud={src['cloud_cover']}  "
              f"shape={scene.width}x{scene.height} @ {scene.resolution:g} m  "
              f"scale={scene.scale:g}  nodata={scene.nodata}")
        print(f"          bands={src['bands']}")
        print(f"          licence: {src['license']}")
    print(f"   identical grid (no resampling needed): "
          f"{grids_identical(before, after)}")

    # -- 2. independent hand-check ------------------------------------------ #
    print("\n2. INDEPENDENT HAND-CHECK (raw digital numbers, no engine code)")
    with rasterio.open(str(BEFORE)) as ds:
        tr = ds.transform
        win = rasterio.windows.Window(COL, ROW, SIZE, SIZE)
        rb = ds.read(3, window=win).astype("float64")
        nb = ds.read(4, window=win).astype("float64")
    with rasterio.open(str(AFTER)) as ds:
        ra = ds.read(3, window=win).astype("float64")
        na = ds.read(4, window=win).astype("float64")

    def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        r, n = red / 10000.0, nir / 10000.0
        den = n + r
        return np.where(np.abs(den) > 1e-6, (n - r) / np.where(den == 0, 1, den), np.nan)

    nd_b, nd_a = ndvi(rb, nb), ndvi(ra, na)
    expected = nd_a - nd_b
    print(f"   block: {SIZE}x{SIZE} px at row {ROW}, col {COL} "
          f"(DN -> reflectance = DN / 10000)")
    print(f"   pixel (0,0) before: red={rb[0, 0]:.0f} nir={nb[0, 0]:.0f} "
          f"-> NDVI = ({nb[0, 0] / 10000:.4f} - {rb[0, 0] / 10000:.4f}) / "
          f"({nb[0, 0] / 10000:.4f} + {rb[0, 0] / 10000:.4f}) = {nd_b[0, 0]:+.6f}")
    print(f"   pixel (0,0) after : red={ra[0, 0]:.0f} nir={na[0, 0]:.0f} "
          f"-> NDVI = {nd_a[0, 0]:+.6f}")
    print(f"   pixel (0,0) delta : {expected[0, 0]:+.6f}")

    # -- 3. the engine over a real 5.12 km window ---------------------------- #
    print("\n3. THE ENGINE OVER A REAL 5.12 km WINDOW (512 x 512 px at 10 m)")
    minx, miny, maxx, maxy = before.bounds
    geom_box = (minx + 2000.0, miny + 2000.0,
                minx + 2000.0 + 5120.0, miny + 2000.0 + 5120.0)
    from shapely.geometry import box

    roi_geom = box(*geom_box)
    result, status, message, warnings = compare_ndvi(
        ScenePair(before=before, after=after), roi_geom)
    if result is None:
        print(f"   REFUSED: {status.value} -- {message}")
        return 1

    d = result.to_dict()
    print(f"   status                 {status.value}")
    print(f"   ROI cells              {d['roi_cell_count']:,}")
    print(f"   comparable cells       {d['valid_pixel_count']:,} "
          f"({d['valid_fraction_of_roi'] * 100:.1f}% of the ROI)")
    print(f"   insufficient cells     {d['insufficient_count']:,}")
    print(f"   NDVI before  mean/median {d['before_mean']:+.4f} / "
          f"{d['before_median']:+.4f}  (std {d['before_std']:.4f})")
    print(f"   NDVI after   mean/median {d['after_mean']:+.4f} / "
          f"{d['after_median']:+.4f}  (std {d['after_std']:.4f})")
    print(f"   delta NDVI   mean/median {d['delta_mean']:+.4f} / "
          f"{d['delta_median']:+.4f}  (std {d['delta_std']:.4f}, "
          f"range {d['delta_min']:+.3f} .. {d['delta_max']:+.3f})")
    print(f"   increase {d['increased_pct']:5.1f}%   stable {d['stable_pct']:5.1f}%   "
          f"decrease {d['decreased_pct']:5.1f}%   (threshold "
          f"+/-{d['thresholds']['increase']:g})")
    print(f"   alignment              {d['alignment']['method']}, "
          f"{d['alignment']['resolution']:g} m, resampling="
          f"{d['alignment']['resampling']}")
    print(f"   runtime                {d['provenance']['runtime_ms']:.0f} ms")

    # -- 4. does the engine reproduce the hand-check? ------------------------ #
    print("\n4. DOES THE ENGINE REPRODUCE THE HAND-CHECK?")
    hb = box(tr.c + COL * 10.0, tr.f - (ROW + SIZE) * 10.0,
            tr.c + (COL + SIZE) * 10.0, tr.f - ROW * 10.0)
    hand, hstatus, hmessage, _ = compare_ndvi(
        ScenePair(before=before, after=after), hb)
    ok = False
    if hand is not None:
        delta = np.asarray(hand.change_raster)
        got = delta[np.isfinite(delta)]
        ok = got.size == SIZE * SIZE and np.allclose(
            got, expected.ravel(), rtol=1e-5, atol=1e-6)
        print(f"   engine cells: {got.size} (expected {SIZE * SIZE})")
        print(f"   max |engine - hand-computed| = "
              f"{np.max(np.abs(got - expected.ravel())):.3e}"
              if got.size == SIZE * SIZE else "   shape mismatch")
    print(f"   RESULT: {'MATCH' if ok else 'MISMATCH -- ' + str(hmessage)}")

    # -- 5. the wording that goes to the user -------------------------------- #
    print("\n5. THE WORDING SHOWN TO THE USER")
    print()
    for para in compose_message(result).split("\n\n"):
        print("   " + para.strip())
        print()

    print(line("="))
    print("PROVENANCE")
    print(line("="))
    print(json.dumps({k: v for k, v in d["provenance"].items()
                      if k not in ("scenes",)}, indent=2)[:2000])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
