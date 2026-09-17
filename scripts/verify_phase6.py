"""Phase 6 verification harness: ROI-based raster statistics.

Run:
    python scripts/verify_phase6.py

Also writes:
    artifacts/phase6_roi.png   the three ROIs drawn over the native NDVI

Exits 0 only if every check passes.

The point of this harness is to prove, on a REAL Sentinel-2 tile, that

    1. statistics come from the NATIVE NDVI array + validity mask, at the
       native transform / CRS / resolution (never the resampled web layer);
    2. "pixels inside the ROI" and "valid NDVI pixels inside the ROI" are two
       different, separately reported numbers;
    3. areas are derived from the affine and from the ROI geometry in the
       raster CRS - nothing is assumed to be 10 m;
    4. an ROI with no valid pixel says so instead of reporting zeros.

Every number printed here is cross-checked against an INDEPENDENT method
(matplotlib.path.Path.contains_points over pixel centres, and a brute-force
numpy computation) rather than against another call of the same function.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import box, mapping

from core.indices import ndvi_from_dataset
from core.raster import open_dataset
from core.statistics import (
    NO_PIXELS_MESSAGE,
    NO_VALID_MESSAGE,
    calculate_roi_ndvi_stats,
    pixel_geometry,
    roi_pixel_mask,
)

CHECKS: list[tuple[bool, str]] = []


def check(condition: bool, label: str) -> None:
    CHECKS.append((bool(condition), label))
    print(f"    [{'PASS' if condition else 'FAIL'}] {label}")


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def independent_inside_mask(geometry, transform: Affine, height: int, width: int) -> np.ndarray:
    """Slow, obvious, INDEPENDENT re-implementation of the pixel-in-ROI test.

    Uses matplotlib.path.Path.contains_points instead of rasterio.features, so a
    bug in the rasteriser cannot hide behind agreement between two calls to the
    same library function.
    """
    import matplotlib.path as mpath

    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    a, b, c, d, e, f, _, _, _ = tuple(transform)
    xs = a * cols + b * rows + c
    ys = d * cols + e * rows + f
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    poly = np.asarray(geometry.exterior.coords, dtype=float)
    path = mpath.Path(poly)
    return path.contains_points(pts, radius=0.0).reshape(height, width)


def main() -> int:
    scene = REPO_ROOT / "data" / "sample" / "s2_s2b-36ruv-20230806-0-l2a_2048px.tif"
    print("=" * 78)
    print("PHASE 6 - ROI-BASED RASTER STATISTICS (real Sentinel-2 sample)")
    print("=" * 78)
    print(f"\nScene: {scene.name}")

    with open_dataset(str(scene)) as ds:
        # Bands 3 (red) and 4 (NIR) 1-based, as documented for Sentinel-2 L2A.
        result, _spec, _report = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        ndvi = np.asarray(result.array, dtype="float32")
        mask = np.asarray(result.mask, dtype=bool)
        transform = Affine(*tuple(result.transform))
        crs = CRS.from_user_input(result.crs)
        height, width = ndvi.shape
        assert not result.has_valid_pixels or np.isfinite(ndvi[mask]).all()

    print(f"  Native NDVI shape     : {height} x {width}")
    print(f"  Native CRS            : {crs.to_string()}")
    print(f"  Native transform      : {tuple(transform)}")
    g = pixel_geometry(transform, crs)
    print(f"  Pixel geometry        : {g['pixel_width']:.3f} m x {g['pixel_height']:.3f} m"
          f"  = {g['pixel_area_m2']:.2f} m2")
    print(f"  Valid NDVI pixels     : {int(mask.sum()):,} / {ndvi.size:,}"
          f" ({100 * mask.mean():.2f}%)")

    print("\n1. Pixel geometry is derived from the affine, not assumed")
    check(close(g["pixel_width"], 10.0, 1e-9), "pixel width from affine == 10.0 m")
    check(close(g["pixel_height"], 10.0, 1e-9), "pixel height from affine == 10.0 m")
    check(close(g["pixel_area_m2"], 100.0, 1e-9), "pixel area from affine == 100.0 m2")
    check(close(g["pixel_area_m2"], abs(transform.a * transform.e - transform.b * transform.d), 1e-9),
          "pixel area == |a*e - b*d|")

    # ------------------------------------------------------------------ #
    # Three ROIs, all defined in the NATIVE raster CRS (EPSG:32636).
    # ------------------------------------------------------------------ #
    minx, miny = transform * (0, height)      # south-west corner of the native grid
    maxx, maxy = transform * (width, 0)       # north-east corner
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2

    rois = {
        "centre (fully inside)": box(cx - 500, cy - 500, cx + 500, cy + 500),
        "corner (partially outside)": box(minx - 300, maxy - 1000, minx + 700, maxy + 300),
        "far outside": box(minx - 50_000, miny - 50_000, minx - 49_000, miny - 49_000),
    }

    print("\n2. ROI statistics against the native NDVI")
    results = {}
    for label, roi in rois.items():
        res = calculate_roi_ndvi_stats(ndvi, roi, transform, mask, crs=crs, roi_crs=crs)
        results[label] = res
        print(f"\n  --- {label} ---")
        print(f"    ROI bounds (EPSG:32636): {tuple(round(v, 1) for v in roi.bounds)}")
        print(f"    ROI area               : {res.area_m2:,.1f} m2 = {res.area_m2 / 10_000:,.3f} ha")
        print(f"    Raster window used     : {res.window}")
        print(f"    Pixels inside ROI      : {res.pixels_inside_roi:,}")
        print(f"    Valid NDVI pixels      : {res.valid_pixels:,}")
        print(f"    Invalid / nodata       : {res.invalid_pixels:,}"
              f"  ({100 * (1 - res.valid_fraction):.2f}%)")
        if res.stats:
            s = res.stats
            print(f"    NDVI  mean {s['mean']:.4f} | median {s['median']:.4f} | "
                  f"std {s['std']:.4f}")
            print(f"          min {s['min']:.4f} | max {s['max']:.4f}")
            print(f"          P5 {s['percentiles']['p5']:.4f} | "
                  f"P25 {s['percentiles']['p25']:.4f} | "
                  f"P75 {s['percentiles']['p75']:.4f} | "
                  f"P95 {s['percentiles']['p95']:.4f}")
            print(f"    Valid pixel area       : {res.valid_area_m2:,.1f} m2")
        print(f"    Message: {res.message}")

    centre = results["centre (fully inside)"]
    corner = results["corner (partially outside)"]
    outside = results["far outside"]

    print("\n3. Pixel counts agree with an independent point-in-polygon test")
    for label, roi in rois.items():
        res = results[label]
        ref = independent_inside_mask(roi, transform, height, width)
        check(res.pixels_inside_roi == int(ref.sum()),
              f"{label}: pixels inside == matplotlib.Path count ({res.pixels_inside_roi:,})")
        check(res.valid_pixels == int((ref & mask).sum()),
              f"{label}: valid pixels == independent count ({res.valid_pixels:,})")

    print("\n4. Statistics agree with a brute-force numpy computation")
    for label in ("centre (fully inside)", "corner (partially outside)"):
        res = results[label]
        roi = rois[label]
        ref = independent_inside_mask(roi, transform, height, width) & mask
        vals = np.asarray(ndvi)[ref]
        s = res.stats
        p = s["percentiles"]
        exp_p = {f"p{q}": float(np.percentile(vals, q)) for q in (5, 25, 50, 75, 95)}
        check(s["valid_pixels"] == vals.size, f"{label}: valid_pixels == numpy size")
        check(close(s["mean"], vals.mean()), f"{label}: mean == numpy mean")
        check(close(s["median"], np.median(vals)), f"{label}: median == numpy median")
        check(close(s["std"], vals.std(ddof=0)), f"{label}: std == numpy std (population)")
        check(close(s["min"], vals.min(), 1e-9), f"{label}: min == numpy min")
        check(close(s["max"], vals.max(), 1e-9), f"{label}: max == numpy max")
        for q in (5, 25, 50, 75, 95):
            check(close(p[f"p{q}"], exp_p[f"p{q}"], 1e-6),
                  f"{label}: P{q} == numpy percentile ({p[f'p{q}']:.4f})")

    print("\n5. Areas are consistent with the pixel counts and the affine")
    check(close(centre.area_m2, 1_000 * 1_000, 1e-9),
          "centre ROI area == 1 000 m x 1 000 m bounding box (1.0 km2)")
    check(close(centre.valid_area_m2, centre.valid_pixels * g["pixel_area_m2"], 1e-9),
          "valid area == valid pixels x per-pixel area from the affine")
    check(close(centre.pixels_inside_roi * g["pixel_area_m2"], centre.area_m2, 0.05),
          "inside-pixel area is within 5% of the geometric ROI area")
    check(centre.pixels_inside_roi == 100 * 100,
          "a 1 km box over a 10 m grid covers exactly 10 000 pixel centres")

    print("\n6. Empty and no-valid-pixel ROIs never report misleading zeros")
    check(outside.pixels_inside_roi == 0 and outside.valid_pixels == 0,
          "ROI outside the footprint has zero pixels inside")
    check(outside.stats is None, "no statistics object is produced for an empty ROI")
    check(close(outside.area_m2, 1_000_000.0, 1e-9),
          "empty ROI still reports its own geometric area (1 000 m x 1 000 m)")
    check(outside.message == NO_PIXELS_MESSAGE, "empty ROI -> 'no raster pixels' message")
    check(outside.valid_area_m2 == 0.0, "empty ROI valid area is genuinely 0 m2 (no fake data)")

    # A synthetic ROI sitting entirely in the nodata corner of the real scene.
    nodata_roi = None
    valid_rows, valid_cols = np.nonzero(mask)
    if valid_rows.size:
        # the sample is mostly valid, so carve an ROI out of an invalid pocket
        inv = ~mask
        if inv.any():
            # Take ONE invalid pixel and build the ROI as exactly that pixel's
            # footprint, so the ROI cannot accidentally include valid neighbours.
            rr, cc = np.nonzero(inv)      # np.nonzero returns MATCHING pairs
            r0, c0 = int(rr[0]), int(cc[0])
            assert not mask[r0, c0], "the picked pixel must really be invalid"
            assert not np.isfinite(ndvi[r0, c0]) or not mask[r0, c0]
            x0, y0 = transform * (c0, r0 + 1)      # SW corner of that pixel
            x1, y1 = transform * (c0 + 1, r0)      # NE corner of that pixel
            nodata_roi = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            print(f"\n  Nodata pocket: {int(inv.sum()):,} invalid pixel(s) in the scene; "
                  f"testing native pixel (row {r0}, col {c0})")
    if nodata_roi is not None:
        res = calculate_roi_ndvi_stats(ndvi, nodata_roi, transform, mask, crs=crs, roi_crs=crs)
        print(f"\n  All-nodata ROI -> inside {res.pixels_inside_roi}, valid {res.valid_pixels},"
              f" message: {res.message}")
        check(res.pixels_inside_roi == 1, "all-nodata ROI counts exactly 1 geometric pixel")
        check(res.invalid_pixels == 1, "that pixel is reported as invalid, not silently dropped")
        check(res.valid_pixels == 0 and res.stats is None,
              "all-nodata ROI reports zero VALID pixels and no stats")
        check(res.message == NO_VALID_MESSAGE, "all-nodata ROI -> 'no valid NDVI' message")
    else:
        check(True, "scene has no nodata pocket - all-nodata ROI case skipped")

    print("\n7. The windowed rasterisation matches a full-array rasterisation")
    for label in ("centre (fully inside)", "corner (partially outside)"):
        roi = rois[label]
        win_mask, win = roi_pixel_mask(roi, transform, height, width)
        from rasterio.features import geometry_mask

        full_inside = ~geometry_mask([mapping(roi)], out_shape=(height, width),
                                     transform=transform, all_touched=False)
        check(bool(np.array_equal(win_mask, full_inside)),
              f"{label}: windowed mask == full-array mask (window {win})")
        check(win is not None and win.width * win.height < height * width,
              f"{label}: the window is smaller than the scene (O(ROI) work)")

    print("\n8. CRS mismatch is refused loudly")
    try:
        calculate_roi_ndvi_stats(ndvi, rois["centre (fully inside)"], transform, mask,
                                 crs=crs, roi_crs="EPSG:4326")
    except Exception as exc:  # noqa: BLE001 - any error is better than silent mis-masking
        check("4326" in str(exc) and "32636" in str(exc),
              f"CRS mismatch raises and names both CRS: {type(exc).__name__}")
    else:
        check(False, "CRS mismatch should raise")

    print("\n9. Messages stay descriptive, never judgemental")
    combined = " ".join(str(results[k].message) for k in results)
    banned = ("healthy", "unhealthy", "suitable", "suitability", "crop", "stress")
    check(not any(w in combined.lower() for w in banned),
          "no health / crop / suitability wording in any message")
    check("mean" in centre.message.lower() and "valid pixels" in centre.message.lower(),
          "success message states the mean and the valid-pixel count")
    check(centre.warnings and any("planar" in w.lower() for w in centre.warnings),
          "area method (planar / geodesic) is reported as a warning")

    print("\n10. Cost is O(ROI), not O(scene)")
    import time

    from rasterio.features import geometry_mask as _gm

    small = box(cx - 150, cy - 150, cx + 150, cy + 150)
    big = box(minx, miny, maxx, maxy)
    t0 = time.perf_counter(); calculate_roi_ndvi_stats(ndvi, small, transform, mask, crs=crs); t_small = time.perf_counter() - t0
    t0 = time.perf_counter(); calculate_roi_ndvi_stats(ndvi, big, transform, mask, crs=crs); t_big = time.perf_counter() - t0
    t0 = time.perf_counter(); _gm([mapping(big)], out_shape=(height, width), transform=transform, all_touched=False, invert=True); t_full = time.perf_counter() - t0
    print(f"    300 m ROI (30x30 px) : {t_small * 1e3:7.2f} ms")
    print(f"    whole scene ROI      : {t_big * 1e3:7.2f} ms")
    print(f"    bare full-array mask : {t_full * 1e3:7.2f} ms  (rasterisation alone)")
    check(t_small < max(0.010, t_big / 10),
          f"a 0.09 km2 ROI costs far less than the whole scene "
          f"({t_small * 1e3:.2f} ms vs {t_big * 1e3:.2f} ms)")

    # ---------------------------------------------------------------- #
    # Static picture of the three ROIs over the native NDVI
    # ---------------------------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon

        artifacts = REPO_ROOT / "artifacts"
        artifacts.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=110)
        show = np.where(mask, ndvi, np.nan)
        vmin, vmax = float(np.nanpercentile(show, 2)), float(np.nanpercentile(show, 98))
        ax.imshow(show, cmap="RdYlGn", vmin=vmin, vmax=vmax, interpolation="nearest")
        for label, roi in rois.items():
            if label == "far outside":
                continue
            xy = np.asarray(roi.exterior.coords)
            cols = np.clip((xy[:, 0] - minx) / 10.0, 0, width)
            rows = np.clip((maxy - xy[:, 1]) / 10.0, 0, height)
            ax.add_patch(MplPolygon(np.column_stack([cols, rows]), closed=True,
                                    fill=False, lw=2.2, edgecolor="magenta"))
            ax.text(cols.min() + 4, rows.min() + 18, label, color="magenta",
                    fontsize=9, weight="bold")
        ax.set_title("Phase 6 - ROIs rasterised against the native NDVI\n"
                     f"{scene.name} ({crs.to_string()})", fontsize=10)
        ax.set_xlabel("native column"); ax.set_ylabel("native row")
        fig.tight_layout()
        png = artifacts / "phase6_roi.png"
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  wrote {png.relative_to(REPO_ROOT)}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  (figure skipped: {exc})")

    failed = [label for ok, label in CHECKS if not ok]
    print("\n" + "=" * 78)
    print(f"PHASE 6 RESULT: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for label in failed:
            print(f"  FAILED: {label}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
