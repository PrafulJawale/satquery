"""Phase 4 verification harness: CRS handling, reprojection, map rendering.

Run:
    python scripts/verify_phase4.py

It also writes `artifacts/phase4_map.html` -- a standalone Leaflet page (real
base tiles, real overlays) that you can open in a browser as a visual check.

Exits 0 only if every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import rowcol

from core.geo import (
    WEB_MERCATOR,
    MissingCRSError,
    footprint_feature,
    grid_bounds_wgs84,
    lonlat_to_pixel,
    pixel_to_crs,
    pixel_to_lonlat,
    raster_footprint,
    reproject_array,
    reproject_bands,
    web_rgba_from_bands,
    web_rgba_from_values,
)
from core.indices import ndvi_from_dataset
from core.raster import open_dataset
from core.samples import list_samples, sample_path
from ui.map import MapOverlay, build_map, ndvi_legend_html

_results: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> bool:
    _results.append((bool(condition), message))
    print(f"   [{'PASS' if condition else 'FAIL'}] {message}")
    return bool(condition)


S2_FILE = next((n for n in list_samples() if n.startswith("s2_")), None)
UTM36 = CRS.from_epsg(32636)
NORTH_UP = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3441820.0)
ROTATED = Affine(10.0, 3.0, 377200.0, 2.0, -10.0, 3441820.0)


# --------------------------------------------------------------------------- #
def verify_crs() -> None:
    print("\n=== A. CRS handling (EPSG:32636 -> EPSG:4326) ===")
    from pyproj import Transformer

    x, y = pixel_to_crs(NORTH_UP, 1024, 1024)
    expect = Transformer.from_crs(UTM36, "EPSG:4326", always_xy=True).transform(x, y)
    got = pixel_to_lonlat(NORTH_UP, UTM36, 1024, 1024)
    check(abs(got[0] - expect[0]) < 1e-9 and abs(got[1] - expect[1]) < 1e-9,
          "pixel -> lon/lat matches an independent pyproj transform")

    # The whole point: UTM numbers are metres, not degrees.
    check(x > 300_000 and y > 3_000_000, "UTM easting/northing are metres (never feed these to Leaflet)")

    try:
        raster_footprint(NORTH_UP, None, 8, 8)
        check(False, "missing CRS should raise")
    except MissingCRSError:
        check(True, "missing CRS raises MissingCRSError instead of guessing coordinates")
    try:
        reproject_array(np.zeros((4, 4), np.float32), NORTH_UP, None)
        check(False, "reprojection without CRS should raise")
    except MissingCRSError:
        check(True, "reprojection refuses a CRS-less raster")

    if S2_FILE:
        with open_dataset(str(sample_path(S2_FILE))) as ds:
            lon, lat = pixel_to_lonlat(ds.transform, ds.crs, ds.width / 2, ds.height / 2)
        print(f"   scene centre: {lat:.5f} deg N, {lon:.5f} deg E")
        check(30.5 < lon < 32.5 and 30.5 < lat < 32.0, "the real scene lands in the Nile Delta")


# --------------------------------------------------------------------------- #
def verify_transform_and_footprint() -> None:
    print("\n=== B. transform -> geographic footprint ===")
    poly = raster_footprint(NORTH_UP, UTM36, 2048, 2048)
    check(len(poly.exterior.coords) > 4, f"footprint is densified ({len(poly.exterior.coords)} vertices, not 4 corners)")
    from pyproj import Geod

    area = abs(Geod(ellps="WGS84").geometry_area_perimeter(poly)[0]) / 1e6
    print(f"   footprint area: {area:.1f} km2 (expected ~419 km2 for 20.48 km square)")
    check(400 < area < 440, "footprint area matches the known ground extent")

    grid = grid_bounds_wgs84(NORTH_UP, UTM36, 2048, 2048)
    check(abs(poly.bounds[0] - grid[0]) < 1e-6 and abs(poly.bounds[2] - grid[2]) < 1e-6,
          "north-up footprint agrees with the grid bounds")

    rot = raster_footprint(ROTATED, UTM36, 512, 512)
    bbox_area = (rot.bounds[2] - rot.bounds[0]) * (rot.bounds[3] - rot.bounds[1])
    check(rot.area < bbox_area * 0.999,
          "a rotated grid produces a quadrilateral, NOT a bounding-box rectangle")

    with open_dataset(str(sample_path(S2_FILE))) as ds:
        lon, lat = pixel_to_lonlat(ds.transform, ds.crs, 1024.5, 1024.5)
        r, c = lonlat_to_pixel(ds.transform, ds.crs, lon, lat)
    check((r, c) == (1024, 1024), f"lon/lat -> pixel round-trips to the same pixel (got row {r}, col {c})")


# --------------------------------------------------------------------------- #
def verify_reprojection() -> None:
    print("\n=== C. reprojection ===")
    data = np.full((128, 128), 0.8, dtype=np.float32)
    data[:, :64] = np.nan
    web = reproject_array(data, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR, max_pixels=200_000)
    check(str(web.crs) == WEB_MERCATOR, "destination CRS is EPSG:3857")
    check(str(web.source_crs) == str(UTM36), "source CRS is remembered")
    check(web.source_transform == NORTH_UP, "source transform is remembered")
    check(web.mask.mean() == 0.5 or abs(web.mask.mean() - 0.5) < 0.06,
          f"nodata half stays nodata (valid fraction {web.mask.mean():.3f})")
    valid = web.array[web.mask]
    check(np.nanmin(valid) == 0.8 or abs(np.nanmin(valid) - 0.8) < 1e-3,
          "valid values survive resampling (nodata did not poison the average)")
    check(not np.any(np.nan_to_num(web.array, nan=9.9) == 0.0), "no invalid pixel became 0")

    before = data.copy()
    reproject_array(data, NORTH_UP, UTM36, dst_crs=WEB_MERCATOR)
    check(np.array_equal(np.nan_to_num(data, nan=-1), np.nan_to_num(before, nan=-1)),
          "reprojection does not mutate the input array (analysis stays native)")

    big = reproject_array(np.full((512, 512), 0.5, np.float32), NORTH_UP, UTM36,
                          dst_crs=WEB_MERCATOR, max_pixels=10_000)
    check(big.shape[0] * big.shape[1] <= 15_000, "max_pixels caps the display grid")

    ds_mem = MemoryFile()
    d = ds_mem.open(driver="GTiff", count=1, height=64, width=64, dtype="uint16",
                    crs=UTM36, transform=NORTH_UP, nodata=0)
    d.write(np.linspace(100, 5000, 64 * 64).astype("uint16").reshape(64, 64), 1)
    wb = reproject_bands(d, (1,), dst_crs=WEB_MERCATOR, max_pixels=100_000)
    d.close()
    ds_mem.close()
    check(wb.source_shape == (64, 64), "source shape preserved in metadata")
    lon_min, lat_min, lon_max, lat_max = wb.bounds_wgs84
    check(lon_min < lon_max and lat_min < lat_max, "destination bounds are ordered")
    check(wb.leaflet_bounds == [[lat_min, lon_min], [lat_max, lon_max]],
          "leaflet bounds are [[lat_min, lon_min], [lat_max, lon_max]]")


# --------------------------------------------------------------------------- #
def verify_geographic_placement() -> None:
    print("\n=== D. geographic placement of the NDVI result ===")
    if not S2_FILE:
        return
    with open_dataset(str(sample_path(S2_FILE))) as ds:
        res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        native_mask_frac = float(res.mask.mean())
        web = reproject_array(res.array, ds.transform, ds.crs, dst_crs=WEB_MERCATOR,
                              max_pixels=1_000_000)
        poly = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=64)

        # same place -> same value
        from pyproj import Transformer

        to_merc = Transformer.from_crs("EPSG:4326", WEB_MERCATOR, always_xy=True)
        rng = np.random.default_rng(11)
        diffs = []
        for _ in range(10):
            r = int(rng.integers(60, ds.height - 60))
            c = int(rng.integers(60, ds.width - 60))
            native = float(res.array[r, c])
            if not np.isfinite(native):
                continue
            lon, lat = pixel_to_lonlat(ds.transform, ds.crs, c + 0.5, r + 0.5)
            x, y = to_merc.transform(lon, lat)
            row_d, col_d = rowcol(web.transform, x, y)
            if 0 <= row_d < web.shape[0] and 0 <= col_d < web.shape[1]:
                v = float(web.array[row_d, col_d])
                if np.isfinite(v):
                    diffs.append(abs(v - native))
        check(bool(diffs), "sampled comparable pixels in both grids")
        print(f"   value differences at identical locations: median {np.median(diffs):.5f}")
        check(float(np.median(diffs)) < 0.05,
              "NDVI at the same lon/lat matches before and after reprojection")

    print(f"   native valid fraction {native_mask_frac:.5f} -> display {float(web.mask.mean()):.5f}")
    check(float(web.mask.mean()) > 0.95, "the display copy keeps nearly all valid pixels")
    print("   (the small loss is the UTM->Mercator grid rotation: a north-up UTM grid is")
    print("    not north-up in Web Mercator, so the covering grid has thin edge wedges)")

    lon_min, lat_min, lon_max, lat_max = web.bounds_wgs84
    fx_min, fy_min, fx_max, fy_max = poly.bounds
    check(lon_min <= fx_min + 1e-6 and lon_max >= fx_max - 1e-6
          and lat_min <= fy_min + 1e-6 and lat_max >= fy_max - 1e-6,
          "display bounds fully contain the true footprint")


def verify_overlay_coverage() -> None:
    """Every opaque overlay pixel, mapped back to lon/lat, must land inside the
    raster footprint. This is the headless equivalent of eyeballing the map."""
    print("\n=== F. overlay pixels vs the true footprint ===")
    if not S2_FILE:
        return
    from pyproj import Transformer
    from shapely.geometry import Point

    with open_dataset(str(sample_path(S2_FILE))) as ds:
        res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        web = reproject_array(res.array, ds.transform, ds.crs, dst_crs=WEB_MERCATOR,
                              max_pixels=600_000)
        poly = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=64)
        native_transform, native_crs = ds.transform, ds.crs

    to_wgs84 = Transformer.from_crs(WEB_MERCATOR, "EPSG:4326", always_xy=True)
    rows, cols = np.nonzero(web.mask)
    rng = np.random.default_rng(7)
    pick = rng.choice(len(rows), size=min(4000, len(rows)), replace=False)
    inside = 0
    for i in pick:
        x, y = web.transform * (cols[i] + 0.5, rows[i] + 0.5)
        lon, lat = to_wgs84.transform(x, y)
        inside += poly.contains(Point(lon, lat))
    frac = inside / len(pick)
    print(f"   {frac:.4f} of sampled opaque overlay pixels fall inside the footprint polygon")
    check(frac > 0.99, "opaque overlay pixels are inside the true footprint")

    # The few that fall outside are edge-resampling artefacts: they must be
    # within about one source pixel of the boundary, never a whole-scene offset.
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    src_pixel_m = abs(native_transform.a)
    outside_m = []
    for i in pick:
        x, y = web.transform * (cols[i] + 0.5, rows[i] + 0.5)
        lon, lat = to_wgs84.transform(x, y)
        p = Point(lon, lat)
        if not poly.contains(p):
            nearest = poly.exterior.interpolate(poly.exterior.project(p))
            outside_m.append(abs(geod.inv(lon, lat, nearest.x, nearest.y)[2]))
    if outside_m:
        worst = max(outside_m)
        print(f"   {len(outside_m)} of {len(pick)} pixels sit outside, worst by "
              f"{worst:.1f} m = {worst / src_pixel_m:.2f} source pixels")
        check(worst <= 2.0 * src_pixel_m,
              "stray pixels are within ~1 source pixel of the boundary (edge resampling)")
    else:
        print("   no opaque pixel falls outside the footprint")

    # ...and the footprint must be covered by the overlay, not merely touched.
    fx_min, fy_min, fx_max, fy_max = poly.bounds
    lon_min, lat_min, lon_max, lat_max = web.bounds_wgs84
    area_cover = ((min(lon_max, fx_max) - max(lon_min, fx_min))
                  * (min(lat_max, fy_max) - max(lat_min, fy_min)))
    area_foot = (fx_max - fx_min) * (fy_max - fy_min)
    check(area_cover / area_foot > 0.99,
          f"overlay bounds cover {area_cover / area_foot:.4f} of the footprint bbox")

    # Cross-check against the NATIVE grid: sample native pixels and confirm the
    # same lon/lat is inside the footprint. Both grids must agree.
    r_n = rng.integers(0, 2048, size=500)
    c_n = rng.integers(0, 2048, size=500)
    ok = 0
    for r, c in zip(r_n, c_n):
        lon, lat = pixel_to_lonlat(native_transform, native_crs, c + 0.5, r + 0.5)
        ok += poly.contains(Point(lon, lat))
    check(ok == 500, "every native pixel also lies inside the footprint (grids agree)")


def write_smoke_figure() -> None:
    """A static, network-free picture of exactly what Leaflet will draw."""
    if not S2_FILE:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    from shapely.geometry import Point

    with open_dataset(str(sample_path(S2_FILE))) as ds:
        res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        web_ndvi = reproject_array(res.array, ds.transform, ds.crs, dst_crs=WEB_MERCATOR,
                                   max_pixels=600_000)
        web_rgb = reproject_bands(ds, (3, 2, 1), dst_crs=WEB_MERCATOR, max_pixels=600_000)
        poly = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=64)

    rgb_rgba = web_rgba_from_bands(web_rgb, ((0.0, 3000.0),) * 3)
    ndvi_rgba = web_rgba_from_values(web_ndvi, -1.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle("Phase 4 visual smoke test -- what the map draws, and where",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    lx, ly = poly.exterior.xy
    ax.plot(lx, ly, color="#d62728", lw=1.6, label="footprint (densified, from CRS+transform)")
    (lat_min, lon_min), (lat_max, lon_max) = web_ndvi.leaflet_bounds
    ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            color="#1f77b4", lw=1.2, ls="--", label="overlay bounds given to Leaflet")
    ax.set_title("Placement in WGS84 (deg)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.ticklabel_format(useOffset=False)
    ax.text(0.02, 0.02, f"source CRS EPSG:32636 -> display {WEB_MERCATOR}",
            transform=ax.transAxes, fontsize=8, color="#555")

    ax = axes[0, 1]
    to_wgs84 = Transformer.from_crs(WEB_MERCATOR, "EPSG:4326", always_xy=True)
    rows, cols = np.nonzero(web_ndvi.mask)
    rng = np.random.default_rng(3)
    pick = rng.choice(len(rows), size=min(3000, len(rows)), replace=False)
    pts = np.array([to_wgs84.transform(*(web_ndvi.transform * (cols[i] + 0.5, rows[i] + 0.5)))
                    for i in pick])
    ax.scatter(pts[:, 0], pts[:, 1], s=2, color="#2ca02c", alpha=0.5,
               label="opaque overlay pixels")
    ax.plot(lx, ly, color="#d62728", lw=1.6, label="footprint")
    inside = np.mean([poly.contains(Point(x, y)) for x, y in pts])
    ax.set_title(f"Overlay pixels reprojected to lon/lat\n{inside:.4f} inside the footprint")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ax.ticklabel_format(useOffset=False)

    ax = axes[1, 0]
    ax.imshow(rgb_rgba)
    ax.set_title(f"True colour overlay ({rgb_rgba.shape[1]}x{rgb_rgba.shape[0]} px)\n"
                 "exactly the RGBA image Leaflet receives")
    ax.set_axis_off()

    ax = axes[1, 1]
    ax.imshow(ndvi_rgba)
    ax.set_title("NDVI overlay (RdYlGn, invalid = transparent)\nNOT an RGB image")
    ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir = REPO_ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "phase4_smoke.png"
    fig.savefig(str(out), dpi=110)
    plt.close(fig)
    print(f"   visual smoke-test figure written to {out.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
def verify_map_rendering() -> None:
    print("\n=== E. map rendering ===")
    if not S2_FILE:
        return
    with open_dataset(str(sample_path(S2_FILE))) as ds:
        res, _spec, _rep = ndvi_from_dataset(ds, 3, 4, profile="sentinel-2-l2a")
        web_ndvi = reproject_array(res.array, ds.transform, ds.crs, dst_crs=WEB_MERCATOR,
                                   max_pixels=600_000)
        web_rgb = reproject_bands(ds, (3, 2, 1), dst_crs=WEB_MERCATOR, max_pixels=600_000)
        poly = raster_footprint(ds.transform, ds.crs, ds.width, ds.height, segments_per_edge=32)

    bands = ((0.0, 3000.0), (0.0, 3000.0), (0.0, 3000.0))
    ov_rgb = MapOverlay("True colour (RGB)", web_rgba_from_bands(web_rgb, bands), web_rgb.leaflet_bounds)
    ov_ndvi = MapOverlay("NDVI", web_rgba_from_values(web_ndvi, -1.0, 1.0), web_ndvi.leaflet_bounds)

    check(web_rgb.leaflet_bounds == web_ndvi.leaflet_bounds,
          "RGB and NDVI overlays share identical geographic bounds")

    rgba = web_rgba_from_values(web_ndvi, -1.0, 1.0)
    invalid = ~web_ndvi.mask
    check(rgba[invalid][:, 3].max() == 0 if invalid.any() else True,
          "invalid NDVI pixels are fully transparent")
    check(rgba[invalid].max() == 0 if invalid.any() else True,
          "invalid NDVI pixels are not painted (never green)")
    valid_px = rgba[web_ndvi.mask]
    check(valid_px[:, 1].mean() > valid_px[:, 0].mean(), "high NDVI renders green-dominant")

    m = build_map(
        overlays=[ov_rgb, ov_ndvi],
        footprint=footprint_feature(poly, {"crs": "EPSG:32636", "scene": S2_FILE}),
        bounds=web_ndvi.leaflet_bounds,
        extra_html=ndvi_legend_html(dataset=S2_FILE, when="2023-08-06",
                                    source_crs="EPSG:32636", display_crs=WEB_MERCATOR),
    )
    html = m.get_root().render()
    check("data:image/png;base64," in html, "overlay images are embedded as PNG data URIs")
    check("Not an RGB" in html, "the legend states the NDVI layer is not an RGB image")
    check("Raster footprint" in html, "the footprint polygon is in the layer control")
    check("lon/lat:" in html, "live coordinate readout is present")

    out_dir = REPO_ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "phase4_map.html"
    m.save(str(out_file))
    print(f"   standalone map written to {out_file.relative_to(REPO_ROOT)} "
          f"({out_file.stat().st_size / 1e6:.2f} MB) -- open it in a browser")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 78)
    print("SatQuery AI -- Phase 4 verification: CRS, reprojection, map")
    print("=" * 78)
    verify_crs()
    verify_transform_and_footprint()
    verify_reprojection()
    verify_geographic_placement()
    verify_overlay_coverage()
    verify_map_rendering()
    write_smoke_figure()

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
    raise SystemExit(main())
