"""Fetch a SECOND real Sentinel-2 acquisition for Phase 10 temporal analysis.

WHY THIS SCRIPT EXISTS
----------------------
Phase 1-9 bundle exactly one Sentinel-2 acquisition, and the shipped
provenance says so:

    "not_good_for": ["change detection (only one date is bundled)"]

Phase 10 compares NDVI between two dates, so it needs a genuine second
acquisition of the *same* ground area. This script downloads one and pins it
to the SAME pixel window as the reference scene, using the SAME STAC
collection, so the two rasters share a CRS, an affine transform and a shape.

THAT PIN IS THE WHOLE POINT. Two Sentinel-2 L2A products from the same MGRS
tile are published on the same 10 m UTM grid, so reading the same
(col_off, row_off, width, height) window out of both yields an *identical*
transform. The real-data Phase 10 path therefore needs NO resampling, which
is the one case where a pixel-by-pixel difference is scientifically honest
without an interpolation step. The alignment code in core/temporal.py still
handles the "grids differ" case -- it is just not exercised by this pair.

Nothing here is synthetic: every pixel is Copernicus Sentinel-2 data.

USAGE
-----
    python scripts/fetch_sentinel2_second_scene.py
    python scripts/fetch_sentinel2_second_scene.py --item-id S2B_36RUV_20230227_0_L2A

Licence: Copernicus Open Data Licence (free, attribution requested).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.transform import array_bounds
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
PROVENANCE_FILE = SAMPLE_DIR / "provenance.json"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Reuse the audited Phase 2 fetcher instead of reimplementing the STAC/GDAL
# plumbing. It is import-safe: everything runs under main().
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from fetch_sentinel2_sample import (  # noqa: E402
    BAND_ASSETS,
    GDAL_ENV,
    fetch_band_windows,
    stac_get_item,
    write_stack,
)

# The scene Phase 1-9 already ships. Its window is the reference.
REFERENCE_ITEM_ID = "S2B_36RUV_20230806_0_L2A"
REFERENCE_FILENAME = "s2_s2b-36ruv-20230806-0-l2a_2048px.tif"

# Winter acquisition of the same MGRS tile: same platform (sentinel-2b),
# same processing level (L2A), 0.06% cloud, 0% nodata. Chosen to give a
# genuine seasonal NDVI contrast against the August reference scene.
DEFAULT_ITEM_ID = "S2B_36RUV_20230118_0_L2A"


# --------------------------------------------------------------------------- #
# reference window
# --------------------------------------------------------------------------- #
def read_reference_window(path: Path) -> Dict[str, Any]:
    """Return the exact window + georeferencing of the scene we already ship.

    Read from the raster itself, not from the sidecar JSON: the raster is the
    ground truth for the grid the app actually analyses.
    """
    if not path.exists():
        raise SystemExit(
            f"reference scene not found: {path}\n"
            "run scripts/fetch_sentinel2_sample.py first"
        )
    with rasterio.open(path) as ds:
        return {
            "crs": ds.crs,
            "transform": ds.transform,
            "width": ds.width,
            "height": ds.height,
            "bounds": ds.bounds,
            "res": ds.res,
        }


def window_from_sidecar(path: Path) -> Optional[Dict[str, int]]:
    """The (col_off, row_off) the reference window was cut from, if recorded."""
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data.get("technical", {}).get("window")


def window_from_bounds(src_ds: rasterio.DatasetReader, bounds: Tuple[float, float, float, float]) -> Window:
    """Cut a window matching `bounds` from `src_ds`, snapped to its grid."""
    win = rasterio.windows.from_bounds(*bounds, transform=src_ds.transform)
    col_off = int(round(win.col_off))
    row_off = int(round(win.row_off))
    width = int(round(win.width))
    height = int(round(win.height))
    # Clamp to the source so a partially-covered date cannot read out of range.
    col_off = max(0, min(col_off, src_ds.width - 1))
    row_off = max(0, min(row_off, src_ds.height - 1))
    width = max(1, min(width, src_ds.width - col_off))
    height = max(1, min(height, src_ds.height - row_off))
    return Window(col_off=col_off, row_off=row_off, width=width, height=height)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item-id", default=DEFAULT_ITEM_ID,
                    help="STAC item id of the SECOND scene (default: 2023-01-18, same tile)")
    ap.add_argument("--reference", default=REFERENCE_FILENAME,
                    help="filename of the already-shipped scene inside data/sample/")
    ap.add_argument("--out", default=None, help="output filename inside data/sample/")
    ap.add_argument("--no-provenance", action="store_true")
    args = ap.parse_args(argv)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    reference_path = SAMPLE_DIR / args.reference

    print("1. reading the reference grid we must match")
    reference = read_reference_window(reference_path)
    sidecar_window = window_from_sidecar(reference_path)
    print(f"   {args.reference}: {reference['width']}x{reference['height']} "
          f"@ {reference['res'][0]:g} m, {reference['crs']}")
    print(f"   bounds={tuple(round(float(v), 3) for v in reference['bounds'])}")
    if sidecar_window:
        print(f"   recorded window: {sidecar_window}")

    print("2. locating the second scene via STAC (earth-search.aws.element84.com)")
    item = stac_get_item(args.item_id)
    props = item["properties"]
    item_id = item["id"]
    print(f"   using {item_id}")
    print(f"   datetime={props.get('datetime')}  platform={props.get('platform')}")
    print(f"   cloud={props.get('eo:cloud_cover')}  tile={props.get('grid:code')}  EPSG:{props.get('proj:epsg')}")

    if props.get("grid:code") != "MGRS-36RUV":
        print(f"   WARNING: tile {props.get('grid:code')} differs from the reference tile MGRS-36RUV")
    if props.get("proj:epsg") != 32636:
        print(f"   WARNING: EPSG:{props.get('proj:epsg')} differs from the reference EPSG:32636")

    hrefs = {key: item["assets"][key]["href"] for key, _b, _w, _n in BAND_ASSETS}

    print("3. pinning the window to the reference footprint (no resampling)")
    with Env(**GDAL_ENV):
        with rasterio.open(hrefs["red"]) as ds:
            src_w, src_h, src_res, src_crs = ds.width, ds.height, ds.res[0], ds.crs
            if sidecar_window and (src_w, src_h) == (10980, 10980):
                # Same full-tile grid as the reference: reuse the pixel offsets.
                window = Window(col_off=int(sidecar_window["col_off"]),
                                row_off=int(sidecar_window["row_off"]),
                                width=int(sidecar_window["width"]),
                                height=int(sidecar_window["height"]))
                print(f"   reused pixel offsets col_off={window.col_off} row_off={window.row_off}")
            else:
                window = window_from_bounds(ds, tuple(reference["bounds"]))
                print(f"   fell back to bounds-derived window col_off={window.col_off} row_off={window.row_off}")
            if abs(src_res - 10.0) > 1e-6:
                raise SystemExit(f"expected a 10 m band, got {src_res} m")
            if src_crs != reference["crs"]:
                raise SystemExit(f"CRS mismatch: new scene {src_crs} vs reference {reference['crs']}")
            if (int(window.width), int(window.height)) != (reference["width"], reference["height"]):
                raise SystemExit(
                    f"window size mismatch: {int(window.width)}x{int(window.height)} vs "
                    f"reference {reference['width']}x{reference['height']}"
                )

    print("4. downloading (COG ranged reads)")
    stack, meta = fetch_band_windows(hrefs, window)
    if stack.shape[1:] != (reference["height"], reference["width"]):
        raise SystemExit(f"shape mismatch after read: {stack.shape[1:]}")

    # THE GUARANTEE Phase 10 relies on: identical grid => honest subtraction.
    same_grid = (
        meta["crs"] == reference["crs"]
        and meta["transform"] == reference["transform"]
    )
    print(f"   transform identical to reference: {same_grid}")
    if not same_grid:
        print("   WARNING: grids differ -- Phase 10 will resample explicitly")

    out_name = args.out or f"s2_{item_id.lower().replace('_', '-')}_{int(window.width)}px.tif"
    out_path = SAMPLE_DIR / out_name
    tags = {
        "source": "Copernicus Sentinel-2 L2A (AWS Open Data COG archive)",
        "stac_item_id": item_id,
        "stac_collection": "sentinel-2-l2a",
        "stac_api": "https://earth-search.aws.element84.com/v1",
        "product_uri": str(props.get("s2:product_uri", "")),
        "platform": str(props.get("platform", "")),
        "datetime": str(props.get("datetime", "")),
        "eo_cloud_cover": str(props.get("eo:cloud_cover", "")),
        "mgrs_tile": str(props.get("grid:code", "")),
        "processing_baseline": str(props.get("s2:processing_baseline", "")),
        "extraction_window": f"col_off={int(window.col_off)},row_off={int(window.row_off)},"
                             f"width={int(window.width)},height={int(window.height)}",
        "bands": "1=B02 blue 490nm, 2=B03 green 560nm, 3=B04 red 665nm, 4=B08 nir 842nm",
        "pixel_size_m": "10",
        "reflectance_scale": "0.0001 (DN/10000); BOA_ADD_OFFSET verified as 0 for this baseline",
        "nodata": "0",
        "phase10_role": "second acquisition for temporal NDVI comparison",
        "grid_matches_reference": str(same_grid),
        "license": "Copernicus Sentinel data, Copernicus Open Data Licence (free, attribution requested)",
    }
    write_stack(stack, meta, out_path, tags)
    size_mb = out_path.stat().st_size / 1e6
    print(f"5. wrote {out_path.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")

    bounds = array_bounds(stack.shape[1], stack.shape[2], meta["transform"])
    valid_frac = float(np.mean((stack > 0).all(axis=0)))
    summary = {
        "name": f"Sentinel-2 L2A {props.get('grid:code')} window ({props.get('datetime', '')[:10]})",
        "filename": out_name,
        "source_url": f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{item_id}",
        "what_it_is": (
            f"Real Copernicus Sentinel-2 MSI Level-2A surface-reflectance window: "
            f"{int(window.width)}x{int(window.height)} px at 10 m "
            f"({int(window.width) * 10 / 1000:.1f} km), 4 bands "
            f"(B02 blue, B03 green, B04 red, B08 nir), uint16 DN, EPSG:{props.get('proj:epsg')}, "
            f"nodata=0. Cut from the SAME pixel window as {args.reference}, so the two "
            f"scenes share one CRS, affine transform and shape."
        ),
        "is_real_satellite_data": True,
        "bands_are_physically_valid": True,
        "good_for": ["metadata", "rgb_preview", "false_colour_composite", "ndvi", "temporal ndvi change"],
        "not_good_for": ["cloud-covered pixels are not masked (scene-wide cloud is "
                         f"{props.get('eo:cloud_cover')}); no atmospheric correction beyond L2A"],
        "notes": (
            "Reflectance = DN/10000 (BOA_ADD_OFFSET 0, verified empirically -- see docs/PHASE2.md). "
            "Band order is blue, green, red, nir -- NOT wavelength order. Acquired by the same "
            "sensor and processing level as the reference scene to keep the comparison like-for-like."
        ),
        "technical": {
            "item_id": item_id,
            "datetime": props.get("datetime"),
            "platform": props.get("platform"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "mgrs_tile": props.get("grid:code"),
            "epsg": props.get("proj:epsg"),
            "bounds_native": [round(float(v), 3) for v in bounds],
            "shape": [int(stack.shape[1]), int(stack.shape[2])],
            "bands": ["B02_blue_490nm", "B03_green_560nm", "B04_red_665nm", "B08_nir_842nm"],
            "dtype": str(stack.dtype),
            "size_mb": round(size_mb, 2),
            "valid_pixel_fraction": round(valid_frac, 6),
            "scale_to_reflectance": 0.0001,
            "reflectance_offset": 0.0,
            "license": "Copernicus Open Data Licence",
            "sampled_by": "scripts/fetch_sentinel2_second_scene.py",
            "grid_matches_reference": same_grid,
            "window": {
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
            },
        },
    }
    (SAMPLE_DIR / f"{out_name}.provenance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if not args.no_provenance:
        data = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8")) if PROVENANCE_FILE.exists() else {"samples": {}}
        data.setdefault("samples", {})[out_name] = {k: v for k, v in summary.items() if k != "technical"}
        PROVENANCE_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print("6. provenance.json updated")

    print("\nDone. Two genuine acquisitions of the same window now exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
