"""Phase 3 verification harness: reflectance handling, NDVI, masking, rendering.

Run:
    python scripts/verify_phase3.py
    python scripts/verify_phase3.py path/to/multispectral.tif --red 3 --nir 4

Exits 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from core.bands import guess_band_roles
from core.indices import classify_ndvi, compute_ndvi, ndvi_from_dataset
from core.preview import analysis_to_geotiff_bytes, decimate_array, ndvi_to_rgba
from core.raster import describe_path, open_dataset
from core.reflectance import (
    ReflectanceSpec,
    detect_reflectance_spec,
    spec_from_profile,
    validate_spec,
)
from core.samples import list_samples, sample_path

_results: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> bool:
    _results.append((bool(condition), message))
    print(f"   [{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)


S2_FILE = next((n for n in list_samples() if n.startswith("s2_")), None)


# --------------------------------------------------------------------------- #
def verify_reflectance_layer() -> None:
    print("\n=== A. reflectance / scaling layer ===")

    s2 = spec_from_profile("sentinel-2-l2a")
    check(s2.scale == 1e-4 and s2.offset == 0.0, "sentinel-2-l2a profile = DN/10000, offset 0")
    check(s2.verified, "sentinel-2-l2a profile is marked as empirically verified")

    l8 = spec_from_profile("landsat-c2-l2")
    check(l8.offset == pytest_approx(-0.2), "landsat-c2-l2 profile keeps its real offset (-0.2)")
    check(
        abs(float(l8.apply(np.array([10000], dtype=np.uint16))[0]) - 0.075) < 1e-6,
        "landsat scaling computes rho = DN*2.75e-5 - 0.2 (DN 10000 -> 0.075)",
    )

    # The trap we actually hit: STAC metadata advertising offset -0.1
    dn = np.array([100, 500, 900, 1500, 3000, 5000], dtype=np.uint16)
    bad = ReflectanceSpec(scale=1e-4, offset=-0.1, source="file_tags")
    fixed, report = validate_spec(bad, dn)
    check(report["offset_rejected"], "an offset that makes reflectance negative is rejected")
    check(fixed.offset == 0.0, "the rejected offset is reset to 0")
    check(
        np.all(fixed.apply(dn) >= 0),
        "after correction no sampled pixel has negative reflectance",
    )

    good, good_report = validate_spec(spec_from_profile("sentinel-2-l2a"), dn)
    check(not good_report["offset_rejected"], "a valid spec is left untouched")
    check(good_report["negative_fraction"] == 0.0, "valid spec yields no negative reflectance")

    # NDVI invariance to pure multiplicative scaling (documented property)
    red_dn = np.array([[1000, 4000]], dtype=np.uint16)
    nir_dn = np.array([[8000, 3000]], dtype=np.uint16)
    a = compute_ndvi(red_dn, nir_dn, reflectance=spec_from_profile("sentinel-2-l2a"))
    b = compute_ndvi(red_dn.astype(np.float32) * 1e-4, nir_dn.astype(np.float32) * 1e-4,
                     apply_reflectance=False)
    check(np.allclose(a.array, b.array, atol=1e-6),
          "with offset 0, NDVI from DN equals NDVI from reflectance (scale cancels)")

    # ... but NOT when a real offset exists
    c = compute_ndvi(red_dn, nir_dn, reflectance=spec_from_profile("landsat-c2-l2"))
    check(not np.allclose(a.array, c.array, atol=1e-4),
          "with a real offset (Landsat) the result differs -- so reflectance must be applied")


def pytest_approx(v):
    return v


# --------------------------------------------------------------------------- #
def verify_masking_rules() -> None:
    print("\n=== B. invalid-pixel handling ===")
    red = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    nir = np.array([[-0.1, 0.2, 0.9]], dtype=np.float32)
    res = compute_ndvi(red, nir, apply_reflectance=False)
    check(np.isnan(res.array[0, 0]), "zero denominator -> NaN (never 0)")
    check(not res.mask[0, 0], "zero denominator -> excluded from the mask")
    check(res.array[0, 1] == 0.0 and res.mask[0, 1], "a genuine zero NDVI stays valid")

    for label, arr_r, arr_n in (
        ("NaN", np.array([[np.nan]]), np.array([[0.5]])),
        ("+inf", np.array([[np.inf]]), np.array([[0.5]])),
        ("-inf", np.array([[-np.inf]]), np.array([[0.5]])),
    ):
        r = compute_ndvi(arr_r.astype(np.float32), arr_n.astype(np.float32), apply_reflectance=False)
        check(np.isnan(r.array[0, 0]) and not r.mask[0, 0], f"{label} input is masked out")

    r = compute_ndvi(np.array([[0]], dtype=np.uint16), np.array([[900]], dtype=np.uint16),
                     red_nodata=0, nir_nodata=0)
    check(np.isnan(r.array[0, 0]) and not r.mask[0, 0], "nodata pixel is masked out")

    empty = compute_ndvi(np.zeros((4, 4), dtype=np.uint16), np.zeros((4, 4), dtype=np.uint16),
                         red_nodata=0, nir_nodata=0)
    check(empty.stats is None, "all-invalid scene reports NO statistics")
    check(empty.counts["valid_pixels"] == 0 and empty.counts["invalid_pixels"] == 16,
          "all-invalid scene still reports its pixel accounting")
    check(np.isnan(empty.array).all(), "all-invalid scene contains only NaN")


# --------------------------------------------------------------------------- #
def verify_ndvi_on_file(path: Path, red_idx: int, nir_idx: int, label: str) -> None:
    print(f"\n=== C. NDVI end-to-end: {label} ===")
    info = describe_path(str(path))
    profile = guess_band_roles(info).profile

    with open_dataset(str(path)) as ds:
        result, spec, report = ndvi_from_dataset(ds, red_idx, nir_idx, profile=profile)

    print(f"   reflectance : {spec.label}  (source={spec.source}, validated={report.get('validated')})")
    print(f"   counts      : {result.counts}")
    if result.stats:
        s = result.stats
        print(f"   stats       : min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} "
              f"median={s['median']:.4f} std={s['std']:.4f}")
        print(f"   percentiles : " + ", ".join(f"{k}={v:.3f}" for k, v in s["percentiles"].items()))

    check(result.counts["total_pixels"] == info.width * info.height, "counts cover every pixel")
    check(
        result.counts["valid_pixels"] + result.counts["invalid_pixels"] == result.counts["total_pixels"],
        "valid + invalid == total",
    )

    if result.stats is None:
        check(True, "no statistics reported (scene has no valid pixels) -- correct behaviour")
        check(np.isnan(result.array).all(), "result array is entirely NaN")
        return

    s = result.stats
    for key in ("min", "max", "mean", "median", "std", "valid_pixels"):
        check(key in s, f"statistic '{key}' is present")
    check(-1.0001 <= s["min"] <= 1.0001, "min within theoretical NDVI range")
    check(-1.0001 <= s["max"] <= 1.0001, "max within theoretical NDVI range")
    check(s["min"] <= s["median"] <= s["max"], "min <= median <= max")
    check(s["std"] >= 0.0, "standard deviation is non-negative")
    check(s["valid_pixels"] == result.counts["valid_pixels"], "stats and counts agree on valid pixels")

    # georeferencing
    check(str(result.crs) == str(info.spatial.crs_name), "CRS preserved")
    check(tuple(result.transform)[:6] == tuple(info.spatial.transform), "transform preserved")
    check(result.native_shape == (info.height, info.width), "native dimensions preserved")

    # no invalid pixel masquerading as a measurement
    invalid = ~result.mask
    if invalid.any():
        check(np.isnan(result.array[invalid]).all(), "every invalid pixel is NaN")
        check(not np.any(result.array[invalid] == 0.0), "no invalid pixel was stored as 0.0")
    else:
        check(True, "no invalid pixels in this scene")

    # rendering
    disp, disp_transform, factor = decimate_array(result.array, result.transform, max_pixels=250_000)
    check(disp.shape[0] * disp.shape[1] <= 250_000 * 1.2, "render decimated below max_pixels")
    if factor > 1:
        check(abs(disp_transform.a - result.transform.a * factor) < 1e-9,
              f"display transform scaled by the decimation factor ({factor})")
    rgba = ndvi_to_rgba(np.where(result.mask, result.array, np.nan), colormap="RdYlGn")
    check(rgba.shape[-1] == 4, "rendered NDVI has an alpha channel")
    if invalid.any():
        sample_rgba = ndvi_to_rgba(result.array, result.mask)
        check(bool(np.all(sample_rgba[invalid][:, 3] == 0)),
              "invalid pixels are fully transparent (never painted as vegetation)")

    # export
    data = analysis_to_geotiff_bytes(result.array, result.mask, result.transform,
                                     result.crs, result.to_dict())
    check(data[:2] in (b"II", b"MM"), "GeoTIFF export produces a TIFF byte stream")
    from rasterio.io import MemoryFile
    with MemoryFile(data) as mem, mem.open() as ds2:
        check(str(ds2.crs) == str(result.crs), "exported raster keeps the CRS")
        check(ds2.transform == result.transform, "exported raster keeps the transform")
        check(np.isnan(ds2.nodata), "exported raster declares NaN nodata")

    # illustrative classes carry their caveat
    cl = classify_ndvi(result.array, result.mask, stats=result.stats)
    check(cl is not None and "NOT validated" in cl["caveat"],
          "illustrative classes are labelled as not validated")
    check(abs(sum(cl["shares"].values()) - 1.0) < 1e-6, "class shares sum to 1")


# --------------------------------------------------------------------------- #
def verify_against_hand_calculation(path: Path, red_idx: int, nir_idx: int) -> None:
    """Recompute NDVI by hand at real pixels and compare with the engine."""
    print("\n=== D. independent hand-check on real pixels ===")
    with open_dataset(str(path)) as ds:
        red = ds.read(red_idx).astype(np.float64)
        nir = ds.read(nir_idx).astype(np.float64)
        spec, _ = detect_reflectance_spec(ds=ds, profile="sentinel-2-l2a")
        res, _, _ = ndvi_from_dataset(ds, red_idx, nir_idx, profile="sentinel-2-l2a")

    scale, offset = spec.scale, spec.offset
    rng = np.random.default_rng(7)
    ok = (red != 0) & (nir != 0)
    rows, cols = np.nonzero(ok)
    pick = rng.choice(len(rows), size=5, replace=False)
    worst = 0.0
    for i in pick:
        r, c = rows[i], cols[i]
        rr, nn = red[r, c] * scale + offset, nir[r, c] * scale + offset
        expected = (nn - rr) / (nn + rr)
        got = float(res.array[r, c])
        worst = max(worst, abs(expected - got))
        print(f"   pixel ({r},{c})  DN red={red[r,c]:.0f} nir={nir[r,c]:.0f} -> "
              f"NDVI engine={got:.6f} hand={expected:.6f}")
    check(worst < 1e-5, f"engine matches hand calculation at sampled pixels (max diff {worst:.2e})")


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", help="optional raster to verify")
    ap.add_argument("--red", type=int, default=3)
    ap.add_argument("--nir", type=int, default=4)
    args = ap.parse_args(argv[1:])

    print("=" * 78)
    print("SatQuery AI -- Phase 3 verification: NDVI engine")
    print("=" * 78)

    verify_reflectance_layer()
    verify_masking_rules()

    if args.path:
        verify_ndvi_on_file(Path(args.path).resolve(), args.red, args.nir, Path(args.path).name)
        verify_against_hand_calculation(Path(args.path).resolve(), args.red, args.nir)
    elif S2_FILE:
        p = sample_path(S2_FILE)
        verify_ndvi_on_file(p, 3, 4, S2_FILE)
        verify_against_hand_calculation(p, 3, 4)

        # The all-nodata fixture: NDVI must refuse to invent numbers.
        print("\n=== E. all-nodata fixture ===")
        verify_ndvi_on_file(sample_path("all-nodata.tif"), 3, 4, "all-nodata.tif")
    else:
        print("\nNo Sentinel-2 sample found; run scripts/fetch_sentinel2_sample.py first.")
        return 1

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
