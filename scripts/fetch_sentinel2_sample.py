"""Fetch a small REAL multispectral sample from public Sentinel-2 L2A COGs.

WHY THIS SCRIPT EXISTS
----------------------
Phase 2 (RGB / false-colour preview) and Phase 3 (NDVI) need a raster with a
genuine near-infrared band. Guessing one is not acceptable, so we download one
from the AWS Open Data "sentinel-s2-l2a-cogs" archive via the Earth Search STAC
API. Nothing here is synthetic: every pixel comes from a Copernicus Sentinel-2
scene, and the sidecar provenance file records exactly where it came from.

WHAT IT DOWNLOADS
-----------------
A square window (default 2048 x 2048 px = 20.48 km at 10 m) of four 10 m bands
-- B02 blue, B03 green, B04 red, B08 nir -- stacked into ONE 4-band uint16
GeoTIFF in its native UTM CRS. A full Sentinel-2 band is ~240 MB; a windowed
COG read is a few MB, because a Cloud-Optimised GeoTIFF lets GDAL fetch only
the byte ranges it needs.

WINDOW SELECTION
----------------
We do not just crop the middle of the tile. The script reads a 256x256 overview
of the whole tile, computes NDVI and NDWI, and picks the window that contains
the most vegetation *and* some surface water -- so the sample is useful for
vegetation, water and (later) change-detection demos.

USAGE
-----
    python scripts/fetch_sentinel2_sample.py                     # default scene
    python scripts/fetch_sentinel2_sample.py --size 1024         # smaller file
    python scripts/fetch_sentinel2_sample.py --search            # search for a scene
    python scripts/fetch_sentinel2_sample.py --bbox 77 20 78 21 --datetime 2024-01-01/2024-03-31

Licence: the Sentinel-2 L2A COG archive is Copernicus Sentinel data,
distributed under the Copernicus Open Data Licence (free, no restrictions on
use or redistribution, attribution requested). See data/sample/provenance.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.env import Env
from rasterio.transform import array_bounds
from rasterio.windows import Window, transform as window_transform

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "data" / "sample"
PROVENANCE_FILE = SAMPLE_DIR / "provenance.json"

STAC_URL = "https://earth-search.aws.element84.com/v1/search"

# Reproducible default: Nile Delta, Egypt. Chosen because it is cloud-free,
# has 0% nodata, and contains irrigated farmland, canals and Mediterranean
# coastline -- i.e. vegetation AND water in one small window.
DEFAULT_ITEM_ID = "S2B_36RUV_20230806_0_L2A"
DEFAULT_BBOX = [30.7, 31.2, 31.0, 31.5]
DEFAULT_DATETIME = "2023-06-01T00:00:00Z/2023-09-30T23:59:59Z"

# Semantic STAC asset key -> (band file, wavelength, human name).
# VERIFIED empirically: on Earth Search v1 the key "nir" is B08 (10 m) while
# "nir08" is B8A (20 m). The names are misleading, so we assert the ground
# sampling distance instead of trusting them.
BAND_ASSETS = [
    ("blue", "B02", 490, "blue"),
    ("green", "B03", 560, "green"),
    ("red", "B04", 665, "red"),
    ("nir", "B08", 842, "nir"),
]

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",  # don't list the whole bucket
    AWS_NO_SIGN_REQUEST="YES",                 # public bucket, no credentials
    GDAL_HTTP_MULTIPLEX="YES",
    VSI_CACHE="TRUE",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
)


# --------------------------------------------------------------------------- #
# STAC
# --------------------------------------------------------------------------- #
def stac_search(
    bbox: List[float], datetime_range: str, max_cloud: float, limit: int = 10
) -> List[Dict[str, Any]]:
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": limit,
        "query": {
            "eo:cloud_cover": {"lt": max_cloud},
            "s2:nodata_pixel_percentage": {"lt": 5},
        },
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    req = urllib.request.Request(
        STAC_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8")).get("features", [])


def stac_get_item(item_id: str, collections: str = "sentinel-2-l2a") -> Dict[str, Any]:
    url = f"https://earth-search.aws.element84.com/v1/collections/{collections}/items/{item_id}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# window selection
# --------------------------------------------------------------------------- #
def pick_window(hrefs: Dict[str, str], size: int, tile: int, probe: int = 256) -> Window:
    """Choose the most demonstration-useful window, using overviews only."""
    with Env(**GDAL_ENV):
        with rasterio.open(hrefs["red"]) as ds:
            tile_w, tile_h = ds.width, ds.height
        red = _overview_read(hrefs["red"], probe)
        nir = _overview_read(hrefs["nir"], probe)
        green = _overview_read(hrefs["green"], probe)

    valid = (red > 0) & (nir > 0) & (green > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(valid, (nir - red) / np.where((nir + red) == 0, np.nan, (nir + red)), np.nan)
        ndwi = np.where(valid, (green - nir) / np.where((green + nir) == 0, np.nan, (green + nir)), np.nan)

    win_px = max(1, int(round(size / tile_w * probe)))
    best_score, best = -1e9, None
    for r in range(0, probe - win_px + 1, 4):
        for c in range(0, probe - win_px + 1, 4):
            sub_ndvi = ndvi[r : r + win_px, c : c + win_px]
            sub_ndwi = ndwi[r : r + win_px, c : c + win_px]
            sub_valid = valid[r : r + win_px, c : c + win_px]
            nodata_frac = 1.0 - sub_valid.mean()
            if sub_valid.mean() < 0.98:          # skip windows touching scene edges
                continue
            veg = float(np.nanmean(sub_ndvi > 0.4))
            water = float(np.nanmean(sub_ndwi > 0.05))
            texture = float(np.nanstd(sub_ndvi))
            score = 1.0 * veg + 3.0 * min(water, 0.08) + 0.5 * texture - 10.0 * nodata_frac
            if score > best_score:
                best_score, best = score, (r, c, veg, water, texture)

    if best is None:                              # degenerate tile -> centre
        off = max(0, (tile_h - size) // 2)
        return Window(col_off=off, row_off=off, width=size, height=size)

    r, c, veg, water, texture = best
    scale = tile_w / probe
    col_off = int(min(max(0, round(c * scale)), tile_w - size))
    row_off = int(min(max(0, round(r * scale)), tile_h - size))
    print(
        f"   window chosen: veg_frac={veg:.2f} water_frac={water:.3f} "
        f"ndvi_std={texture:.2f} score={best_score:.3f}"
    )
    return Window(col_off=col_off, row_off=row_off, width=size, height=size)


def _overview_read(href: str, probe: int) -> np.ndarray:
    with rasterio.open(href) as ds:
        return ds.read(1, out_shape=(1, probe, probe), resampling=Resampling.average).astype("float32")


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def fetch_band_windows(
    hrefs: Dict[str, str], window: Window
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read the same window from every band; returns (stack, meta)."""
    arrays: List[np.ndarray] = []
    meta: Dict[str, Any] = {}
    with Env(**GDAL_ENV):
        for key, _band, _wl, _name in BAND_ASSETS:
            with rasterio.open(hrefs[key]) as ds:
                if ds.width != ds.height:
                    raise RuntimeError(f"{key}: unexpected non-square source {ds.width}x{ds.height}")
                if abs(ds.res[0] - 10.0) > 1e-6:
                    raise RuntimeError(f"{key}: expected a 10 m band, got {ds.res[0]} m")
                arr = ds.read(1, window=window)
                arrays.append(arr)
                if not meta:
                    meta = {
                        "crs": ds.crs,
                        "transform": window_transform(window, ds.transform),
                        "dtype": ds.dtypes[0],
                        "nodata": ds.nodatavals[0],
                        "descriptions": [f"{b}_{n}_{w}nm" for _k, b, w, n in BAND_ASSETS],
                    }
    return np.stack(arrays, axis=0), meta


def write_stack(stack: np.ndarray, meta: Dict[str, Any], out_path: Path, tags: Dict[str, str]) -> None:
    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": stack.shape[0],
        "dtype": stack.dtype,
        "crs": meta["crs"],
        "transform": meta["transform"],
        "nodata": meta["nodata"],
        "compress": "DEFLATE",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stack)
        for i, desc in enumerate(meta["descriptions"], start=1):
            dst.set_band_description(i, desc)
        dst.update_tags(**tags)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item-id", default=DEFAULT_ITEM_ID, help="STAC item id (default: reproducible Nile Delta scene)")
    ap.add_argument("--search", action="store_true", help="search for a scene instead of using --item-id")
    ap.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX, help="min_lon min_lat max_lon max_lat")
    ap.add_argument("--datetime", default=DEFAULT_DATETIME, help="STAC datetime range")
    ap.add_argument("--max-cloud", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=2048, help="edge of the square window, in 10 m pixels")
    ap.add_argument("--out", default=None, help="output filename inside data/sample/")
    ap.add_argument("--no-provenance", action="store_true", help="do not update provenance.json")
    args = ap.parse_args(argv)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print("1. locating a Sentinel-2 L2A scene via STAC (earth-search.aws.element84.com)")
    if args.search:
        feats = stac_search(args.bbox, args.datetime, args.max_cloud)
        if not feats:
            print("   no scene matched the search; try a wider bbox/datetime or higher --max-cloud")
            return 1
        for f in feats[:5]:
            print(
                f"   {f['id']}  cloud={f['properties'].get('eo:cloud_cover'):.4f}  "
                f"nodata%={f['properties'].get('s2:nodata_pixel_percentage')}"
            )
        item = feats[0]
    else:
        item = stac_get_item(args.item_id)

    props = item["properties"]
    item_id = item["id"]
    print(f"   using {item_id}")
    print(f"   datetime={props.get('datetime')}  platform={props.get('platform')}")
    print(f"   cloud={props.get('eo:cloud_cover')}  tile={props.get('grid:code')}  EPSG:{props.get('proj:epsg')}")

    hrefs = {key: item["assets"][key]["href"] for key, _b, _w, _n in BAND_ASSETS}
    print("2. bands (asset key -> file):")
    for key, band, wl, name in BAND_ASSETS:
        print(f"   {key:6s} -> {hrefs[key].rsplit('/', 1)[-1]:8s} ({wl} nm, {name})")

    print(f"3. choosing a {args.size}x{args.size} window (10 m px = {args.size * 10 / 1000:.1f} km)")
    window = pick_window(hrefs, args.size, tile=10980)

    print("4. downloading (COG ranged reads, ~a few MB)")
    stack, meta = fetch_band_windows(hrefs, window)

    out_name = args.out or f"s2_{item_id.lower().replace('_', '-')}_{args.size}px.tif"
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
        "license": "Copernicus Sentinel data, Copernicus Open Data Licence (free, attribution requested)",
    }
    write_stack(stack, meta, out_path, tags)
    size_mb = out_path.stat().st_size / 1e6
    print(f"5. wrote {out_path.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")

    bounds = array_bounds(stack.shape[1], stack.shape[2], meta["transform"])
    summary = {
        "name": f"Sentinel-2 L2A {props.get('grid:code')} window ({props.get('datetime', '')[:10]})",
        "filename": out_name,
        "source_url": f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{item_id}",
        "what_it_is": (
            f"Real Copernicus Sentinel-2 MSI Level-2A surface-reflectance window: "
            f"{args.size}x{args.size} px at 10 m ({args.size * 10 / 1000:.1f} km), 4 bands "
            f"(B02 blue, B03 green, B04 red, B08 nir), uint16 DN, EPSG:{props.get('proj:epsg')}, "
            f"nodata=0. Extracted from the AWS Open Data COG archive."
        ),
        "is_real_satellite_data": True,
        "bands_are_physically_valid": True,
        "good_for": ["metadata", "rgb_preview", "false_colour_composite", "ndvi", "ndwi", "map overlay"],
        "not_good_for": ["change detection (only one date is bundled)"],
        "notes": (
            "Reflectance = DN/10000. The STAC raster:bands metadata advertises offset -0.1, but "
            "applying it makes ~98% of pixels negative, so BOA_ADD_OFFSET is 0 for this "
            "baseline (verified empirically -- see docs/PHASE2.md). Bands 1-4 are NOT in "
            "wavelength order: order is blue, green, red, nir."
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
            "scale_to_reflectance": 0.0001,
            "reflectance_offset": 0.0,
            "license": "Copernicus Open Data Licence",
            "sampled_by": "scripts/fetch_sentinel2_sample.py",
            "window": {
                "col_off": int(window.col_off),
                "row_off": int(window.row_off),
                "width": int(window.width),
                "height": int(window.height),
            },
        },
    }
    (SAMPLE_DIR / f"{out_name}.provenance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_provenance:
        data = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8")) if PROVENANCE_FILE.exists() else {"samples": {}}
        data.setdefault("samples", {})[out_name] = {k: v for k, v in summary.items() if k != "technical"}
        PROVENANCE_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print("6. provenance.json updated")

    print("\nDone. This is REAL satellite data -- keep the provenance sidecar with the file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
