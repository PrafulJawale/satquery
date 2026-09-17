"""Phase 2 verification harness: band inspection + RGB / false-colour rendering.

Run:
    python scripts/verify_phase2.py
    python scripts/verify_phase2.py path/to/scene.tif

Exits 0 only if every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import rasterio

from core.bands import describe_guess, guess_band_roles
from core.preview import (
    band_histograms,
    decimated_read,
    encode_png,
    make_preview,
    percentile_bounds,
    stretch_to_uint8,
    valid_mask,
)
from core.raster import describe_path, open_dataset
from core.samples import list_samples, sample_path

_results: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> bool:
    _results.append((bool(condition), message))
    print(f"   [{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)


S2_FILE = next((n for n in list_samples() if n.startswith("s2_")), None)


# --------------------------------------------------------------------------- #
def verify_band_detection() -> None:
    print("\n=== A. band role detection ===")

    if S2_FILE:
        path = sample_path(S2_FILE)
        info = describe_path(str(path))
        g = guess_band_roles(info)
        print(f"   {S2_FILE}: {describe_guess(g)}")
        check(g.roles.get("blue") == 1, "blue -> band 1 (from description 'B02_blue_490nm')")
        check(g.roles.get("green") == 2, "green -> band 2")
        check(g.roles.get("red") == 3, "red -> band 3")
        check(g.roles.get("nir") == 4, "nir -> band 4 (B08)")
        check(g.confidence == "high", f"confidence is 'high' (got '{g.confidence}')")
        check(g.profile == "sentinel-2-l2a", f"profile detected (got '{g.profile}')")
        check(g.reflectance_scale == 1e-4, "reflectance scale 1e-4 carried for Phase 3")
        check(not g.needs_confirmation, "high-confidence guess does not demand confirmation")

    # A real 3-band file with no band descriptions: only colour-interp to go on.
    info = describe_path(str(sample_path("RGB.byte.tif")))
    g = guess_band_roles(info)
    print(f"   RGB.byte.tif: {describe_guess(g)}")
    check(g.roles.get("red") == 1 and g.roles.get("green") == 2 and g.roles.get("blue") == 3,
          "RGB from GDAL colour interpretation")
    check(g.band("nir") is None, "no NIR band invented for a 3-band file")
    check(any("near-infrared" in w for w in g.warnings), "missing NIR is flagged as a warning")

    # Named 4-band reflectance file (all pixels nodata, but names are real).
    info = describe_path(str(sample_path("all-nodata.tif")))
    g = guess_band_roles(info)
    print(f"   all-nodata.tif: {describe_guess(g)}")
    check(g.roles.get("nir") == 4, "nir -> band 4 from band description 'nir'")

    # Single band: nothing to guess, must not fabricate.
    info = describe_path(str(sample_path("byte.tif")))
    g = guess_band_roles(info)
    print(f"   byte.tif: {describe_guess(g)}")
    check(g.confidence in ("none", "low"), f"single band => low/no confidence (got '{g.confidence}')")
    check(g.band("nir") is None, "no NIR for a 1-band file")


# --------------------------------------------------------------------------- #
def verify_preview(path: Path, has_nir: bool, label: str) -> None:
    print(f"\n=== B. rendering: {label} ===")
    with open_dataset(str(path)) as ds:
        nir_idx = None
        if has_nir:
            nir_idx = guess_band_roles(describe_path(str(path))).band("nir")

        res = make_preview(ds, (3, 2, 1), ("red", "green", "blue"), max_pixels=600_000)
        print(f"   source {res.source_shape} -> display {res.display_shape} "
              f"(decimation 1/{res.decimation_factor:.2f}), nodata {100 * res.nodata_fraction:.2f}%")

        check(res.image.dtype == np.uint8, "image is uint8")
        check(res.image.ndim == 3 and res.image.shape[2] == 3, "image shape is (H, W, 3)")
        check(res.display_shape[0] * res.display_shape[1] <= 600_000 * 1.05,
              "decimated read respects max_pixels")

        # The rendered transform must be the native transform scaled by the
        # decimation factor -- this is what keeps Phase 4's overlay aligned.
        sx_native = abs(ds.transform.a)
        sx_display = abs(res.transform.a)
        expected = sx_native * (ds.width / res.display_shape[1])
        check(abs(sx_display - expected) < 1e-6,
              f"display pixel size = native x decimation ({sx_display:.4f} ~= {expected:.4f})")

        # Stretch bounds must be computed on VALID pixels only.
        for ch, (lo, hi), st in zip(res.channels, res.stretch_bounds, res.band_stats):
            nodata = st["nodata"]
            if nodata is not None and st["valid_pixels"] > 0 and st["min"] is not None:
                check(lo >= float(st["min"]) - 1e-6, f"{ch}: stretch low >= observed min")
                if nodata != 0:
                    check(lo > float(nodata), f"{ch}: nodata value excluded from stretch bounds")

        # Per-band stretch should use most of the 8-bit range.
        bright = res.image[res.alpha > 0]
        if bright.size:
            check(int(bright.max()) >= 250, f"stretched highlights reach ~255 (got {int(bright.max())})")
            check(int(bright.min()) <= 5, f"stretched shadows reach ~0 (got {int(bright.min())})")

        # Nodata must be transparent, never painted as black data.
        if res.nodata_fraction > 0:
            check(int(res.alpha.min()) == 0, "nodata pixels are transparent in alpha")

        if nir_idx:
            fcc = make_preview(ds, (nir_idx, 3, 2), ("nir", "red", "green"), max_pixels=600_000)
            check(fcc.image.shape == res.image.shape, "false-colour has the same display shape")
            check(fcc.stretch_bounds != res.stretch_bounds, "FCC uses its own stretch bounds")
            print(f"   FCC channels {fcc.channels} bounds "
                  f"{[(round(a, 1), round(b, 1)) for a, b in fcc.stretch_bounds]}")

        png = encode_png(res.image, res.alpha)
        check(png[:8] == b"\x89PNG\r\n\x1a\n", "PNG encoding produces a valid PNG byte stream")
        check(len(png) > 1000, f"PNG has plausible size ({len(png) / 1024:.0f} KB)")

        hists = band_histograms(ds, (1, 2), bins=32, max_pixels=300_000)
        check(len(hists) == 2 and len(hists[0]["counts"]) == 32, "histograms return the requested bin count")
        check(sum(hists[0]["counts"]) == hists[0]["valid_pixels"] or hists[0]["valid_pixels"] == 0,
              "histogram counts sum to the number of valid pixels")


# --------------------------------------------------------------------------- #
def verify_pure_functions() -> None:
    print("\n=== C. pure functions (synthetic arrays, no files) ===")
    # A ramp with a nodata ring: exercises stretching and masking deterministically.
    arr = np.tile(np.arange(100, 1100, dtype=np.uint16), (10, 1))
    arr[0, :] = 0                      # nodata row
    mask = valid_mask(arr, 0)
    check(mask.sum() == arr.size - arr.shape[1], "valid_mask excludes the nodata row")

    lo, hi = percentile_bounds(arr[mask], 2, 98)
    check(abs(lo - np.percentile(arr[mask], 2)) < 1e-6, "percentile_bounds matches numpy percentiles")
    check(hi > lo, "stretch bounds are ordered")

    out = stretch_to_uint8(arr, mask, lo, hi)
    check(out.dtype == np.uint8, "stretch_to_uint8 returns uint8")
    check(int(out[0, :].max()) == 0, "nodata pixels are forced to 0")
    check(int(out[mask].max()) == 255, "the 98th percentile maps to 255")
    check(int(out[mask].min()) == 0, "the 2nd percentile maps to 0")

    flat = np.full((8, 8), 5000, dtype=np.uint16)
    lo_c, hi_c = percentile_bounds(flat, 2, 98)
    check(hi_c > lo_c, "a constant band still yields usable bounds (no divide-by-zero)")

    empty = np.array([], dtype=np.uint16)
    lo_e, hi_e = percentile_bounds(empty, 2, 98)
    check(hi_e > lo_e, "an empty band degrades gracefully")


# --------------------------------------------------------------------------- #
def verify_edge_cases() -> None:
    print("\n=== D. edge cases must not crash ===")
    for name in ("rotated.tif", "float_nan.tif", "all-nodata.tif"):
        path = sample_path(name)
        with open_dataset(str(path)) as ds:
            bands = (1, 1, 1) if ds.count == 1 else (3, 2, 1)
            res = make_preview(ds, bands, ("a", "b", "c"), max_pixels=100_000)
            check(res.image.ndim == 3, f"{name}: preview renders without crashing")
            if name == "all-nodata.tif":
                check(res.nodata_fraction > 0.99, f"{name}: all pixels reported as nodata")
                check(int(res.alpha.max()) == 0, f"{name}: alpha fully transparent (nothing painted as data)")
                check(int(res.image.max()) == 0, f"{name}: no pixel values fabricated")


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    print("=" * 78)
    print("SatQuery AI -- Phase 2 verification: band inspection + visualisation")
    print("=" * 78)

    if len(argv) > 1:
        for arg in argv[1:]:
            p = Path(arg).resolve()
            with open_dataset(str(p)) as ds:
                g = guess_band_roles(describe_path(str(p)))
            print(f"   {p.name}: {describe_guess(g)}")
            verify_preview(p, has_nir=g.band("nir") is not None, label=p.name)
    else:
        verify_band_detection()
        if S2_FILE:
            verify_preview(sample_path(S2_FILE), has_nir=True, label=S2_FILE)
        verify_preview(sample_path("RGB.byte.tif"), has_nir=False, label="RGB.byte.tif (8-bit, no NIR)")
        verify_pure_functions()
        verify_edge_cases()

    passed = sum(1 for ok, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 78)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 78)
    for ok, msg in _results:
        if not ok:
            print(f"  FAILED: {msg}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
