"""Phase 1 verification harness.

WHY THIS SCRIPT EXISTS
----------------------
A Streamlit page can *look* fine while being wrong. This script is the
un-glamorous check that Phase 1 is actually correct: it reads every bundled
sample, asserts the invariants that matter for later phases, and compares
against values we know from `gdalinfo`.

Run:
    python scripts/verify_phase1.py                  # all bundled samples
    python scripts/verify_phase1.py path/to/a.tif    # your own file(s)

Exits 0 only if every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from core.raster import describe_path, open_dataset, window_stats
from core.samples import SAMPLE_DIR, list_samples, sample_path

# Known-good values for data/sample/RGB.byte.tif, read with gdalinfo.
# If these ever change, the sample file was replaced -- investigate, don't just
# update the numbers.
KNOWN_RGB_BYTE = {
    "width": 791,
    "height": 718,
    "count": 3,
    "epsg": 32618,
    "bounds": (101985.0, 2611485.0, 339315.0, 2826915.0),
    "nodata": 0.0,
}

_results: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> bool:
    _results.append((bool(condition), message))
    print(f"   [{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)


def verify_file(path: Path, known: dict | None = None) -> None:
    print(f"\n=== {path.name} ===")
    try:
        info = describe_path(str(path), label=path.name)
    except Exception as exc:
        check(False, f"{path.name}: could not be opened ({exc})")
        return

    sp = info.spatial
    print(
        f"   {info.width}x{info.height} px | {info.count} band(s) | {', '.join(info.dtypes)} | "
        f"driver={info.driver} | {info.estimated_full_read_mb:.1f} MB full read"
    )
    print(f"   CRS: {sp.crs_name} (EPSG:{sp.crs_epsg}) | {sp.resolution_label}")
    print(f"   North-up: {sp.is_north_up} | Tiled: {info.tiled} | COG-like: {info.looks_like_cog}")
    if sp.bounds_wgs84:
        print(f"   WGS84 bounds: {tuple(round(v, 6) for v in sp.bounds_wgs84)}")
    if sp.approx_area_km2:
        print(f"   Approx coverage: {sp.approx_area_km2:,.2f} km2")
    for w in info.warnings:
        print(f"   ! {w}")

    # ---- invariants that every later phase depends on -----------------------
    check(info.width > 0 and info.height > 0, "dimensions are positive")
    check(info.count >= 1, "at least one band")
    check(len(info.bands) == info.count, "band records match band count")
    check(
        all(b.index == i + 1 for i, b in enumerate(info.bands)),
        "band indices are 1-based and contiguous",
    )

    if sp.has_crs:
        check(sp.footprint_wgs84 is not None, "footprint exists when CRS exists")
        check(sp.bounds_wgs84 is not None, "WGS84 bounds exist when CRS exists")
        if sp.bounds_wgs84:
            lon_min, lat_min, lon_max, lat_max = sp.bounds_wgs84
            check(-180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0, "longitudes in [-180, 180]")
            check(-90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0, "latitudes in [-90, 90]")
        if sp.footprint_wgs84:
            check(len(sp.footprint_wgs84) == 64, "footprint is densified (64 vertices = 16 per edge)")
        if sp.is_projected:
            check(
                not sp.is_geographic and sp.linear_units is not None,
                "projected CRS reports linear units",
            )
    else:
        check(sp.footprint_wgs84 is None, "no footprint fabricated for a CRS-less raster")
        check(
            any(w.startswith("NO CRS") for w in info.warnings),
            "missing CRS raises an explicit warning",
        )

    if sp.is_rotated:
        check(
            any(w.startswith("ROTATED") for w in info.warnings),
            "rotated transform raises an explicit warning",
        )

    # ---- pixel-level sanity -------------------------------------------------
    try:
        with open_dataset(str(path)) as ds:
            stats = window_stats(ds, band=1)
        print(
            f"   centre window: {stats['valid_pixels']:,} valid px | "
            f"min={stats['min']} mean={stats['mean']} max={stats['max']}"
        )
        check(
            (stats["all_nodata"] and stats["valid_pixels"] == 0)
            or (not stats["all_nodata"] and stats["valid_pixels"] > 0),
            "nodata accounting is self-consistent",
        )
        if stats["mean"] is not None:
            check(np.isfinite(stats["mean"]), "mean over valid pixels is finite (NaN did not leak)")
    except Exception as exc:
        check(False, f"windowed read failed: {exc}")

    # ---- known-value regression --------------------------------------------
    if known:
        check(info.width == known["width"], f"width == {known['width']}")
        check(info.height == known["height"], f"height == {known['height']}")
        check(info.count == known["count"], f"band count == {known['count']}")
        check(sp.crs_epsg == known["epsg"], f"EPSG == {known['epsg']}")
        check(
            tuple(round(v, 3) for v in sp.bounds_native) == tuple(round(v, 3) for v in known["bounds"]),
            "native bounds match gdalinfo",
        )
        check(info.nodata_per_band[0] == known["nodata"], f"nodata == {known['nodata']}")
        check(
            any("8-BIT" in w for w in info.warnings),
            "8-bit display product is flagged as not index-valid",
        )

    # ---- serialisation ------------------------------------------------------
    try:
        import json

        json.dumps(info.to_dict())
        check(True, "RasterInfo serialises to JSON (Streamlit/API safe)")
    except Exception as exc:
        check(False, f"RasterInfo is not JSON-serialisable: {exc}")


def main(argv: list[str]) -> int:
    targets: list[tuple[Path, dict | None]] = []
    if len(argv) > 1:
        for arg in argv[1:]:
            targets.append((Path(arg).expanduser().resolve(), None))
    else:
        names = list_samples()
        if not names:
            print(f"No samples found in {SAMPLE_DIR}")
            return 1
        for name in names:
            known = KNOWN_RGB_BYTE if name == "RGB.byte.tif" else None
            targets.append((sample_path(name), known))

    print("=" * 78)
    print("SatQuery AI -- Phase 1 verification: GeoTIFF ingestion")
    print("=" * 78)

    for path, known in targets:
        verify_file(path, known)

    passed = sum(1 for ok, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 78)
    print(f"RESULT: {passed} passed, {failed} failed  ({len(targets)} file(s))")
    print("=" * 78)
    for ok, msg in _results:
        if not ok:
            print(f"  FAILED: {msg}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
