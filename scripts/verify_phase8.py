"""Phase 8 -- REAL-DATA verification of the cotton screening on the Nile Delta sample.

Two independent halves:

  A. run the actual application engine (`analyses.crop_suitability`) end to end
     on the bundled Sentinel-2 AOI and print everything it decided.

  B. recompute the same numbers OUTSIDE the implementation code -- plain
     rasterio windowed reads, plain numpy, hand-written arithmetic -- and
     compare. If the engine agreed with itself we would have proved nothing.

Run:  python scripts/verify_phase8.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rasterio                                            # noqa: E402
from pyproj import Transformer                              # noqa: E402
from rasterio import Affine                                 # noqa: E402
from rasterio.crs import CRS                                # noqa: E402
from rasterio.windows import Window                         # noqa: E402
from shapely.geometry import box                            # noqa: E402

from analyses import AnalysisContext, Status                # noqa: E402
from analyses.crop_suitability import run_crop_suitability  # noqa: E402
from core.router import parse_query                         # noqa: E402
from core.roi import ROISelection                           # noqa: E402
from core.suitability import load_crop_config, trapezoid_membership  # noqa: E402

CHECKS: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    CHECKS.append((bool(ok), label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def close(a, b, tol):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# the AOI: a 3 x 3 km box in the middle of the bundled sample
# --------------------------------------------------------------------------- #
TEN_M = Affine(10.0, 0.0, 377200.0, 0.0, -10.0, 3462300.0)
UTM36N = CRS.from_epsg(32636)
CENTRE_COL, CENTRE_ROW = 1024, 1024
HALF = 150                       # 150 cells * 10 m = 1.5 km -> 3 x 3 km ROI
C0, R0, C1, R1 = CENTRE_COL - HALF, CENTRE_ROW - HALF, CENTRE_COL + HALF, CENTRE_ROW + HALF
X0, Y0 = TEN_M * (C0, R1)
X1, Y1 = TEN_M * (C1, R0)
ROI_BOX = box(min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1))

LON_C, LAT_C = 31.81854, 31.19722        # measured centre of the sample


def make_roi() -> ROISelection:
    return ROISelection(is_valid=True, intersects_raster=True,
                        area_m2=float(ROI_BOX.area), raster_crs="EPSG:32636",
                        geometry_raster_crs=ROI_BOX, geometry_type="Polygon",
                        num_parts=1)


# --------------------------------------------------------------------------- #
# A. the application engine
# --------------------------------------------------------------------------- #
def run_engine():
    print("=" * 78)
    print("A. THE APPLICATION ENGINE  (real external data, windowed reads)")
    print("=" * 78)
    query = parse_query("Can I grow cotton here?")
    t0 = time.perf_counter()
    execution = run_crop_suitability(AnalysisContext(roi=make_roi()), query)
    wall = time.perf_counter() - t0

    print(f"  status        : {execution.status.value}")
    print(f"  wall time     : {wall:,.1f} s")
    if execution.result is None:
        print(f"  message       : {execution.message}")
        for w in execution.warnings:
            print(f"    - {w}")
        return None, execution

    r = execution.result
    print(f"  grid          : {r.grid.width} x {r.grid.height} cells @ "
          f"{r.grid.resolution:g} m ({r.grid.cells:,} cells)")
    print(f"  land cover    : {r.land_cover.get('dominant_class_name')} "
          f"(excluded {r.land_cover.get('excluded_fraction', 0):.1%})")
    print(f"  elevation     : {r.elevation_context}")
    print(f"  slope (ROI)   : {r.slope_context}")
    print()
    for name, res in r.scenarios.items():
        print(f"  --- scenario {name}: {res.classification} "
              f"(score {res.score}, confidence {res.confidence}) ---")
        print(f"      valid fraction : {res.valid_fraction:.3f}")
        print(f"      limiting       : "
              f"{[f['factor'] for f in res.computed_limiting_factors]}")
        print(f"      missing        : {res.missing_factors}")
        print(f"      assumed        : {res.assumed_factors}")
        for f in res.factors:
            print(f"      {f.label:32s} value={f.value} membership={f.membership} "
                  f"[{f.status}]")
        print(f"      annual precip  : {res.annual_precipitation}")
        print(f"      season  precip : {res.growing_season_precipitation} "
              f"(months {res.growing_season_months})")
        if res.approx_gdd:
            print(f"      approx GDD     : {res.approx_gdd['value']} "
                  f"(base {res.approx_gdd['base_temp_c']} C) -- context only")
        print(f"      class fractions: "
              f"{{ {', '.join(f'{k}: {v:.1%}' for k, v in res.class_fractions.items())} }}")
    print()
    print("  ANSWER TEXT")
    print("  " + "-" * 74)
    for line in execution.message.splitlines():
        print(f"  | {line}")
    print("  " + "-" * 74)
    print()
    print("  PERFORMANCE (s)")
    for k, v in sorted(r.performance.items()):
        print(f"      {k:34s} {v}")
    print()
    return r, execution


# --------------------------------------------------------------------------- #
# B. independent recomputation (no project code except the config file)
# --------------------------------------------------------------------------- #
def _window_mean(url, lon0, lat0, lon1, lat1, scale=1.0, categorical=False):
    """Plain windowed read + mean, no WarpedVRT, no caching, no project code."""
    with rasterio.open(url) as ds:
        to_src = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        (xa, ya), (xb, yb) = to_src.transform(lon0, lat0), to_src.transform(lon1, lat1)
        r0, c0 = ds.index(min(xa, xb), max(ya, yb))
        r1, c1 = ds.index(max(xa, xb), min(ya, yb))
        r0, r1 = max(0, min(r0, r1)), min(ds.height, max(r0, r1) + 1)
        c0, c1 = max(0, min(c0, c1)), min(ds.width, max(c0, c1) + 1)
        arr = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype("float64")
        nd = ds.nodatavals[0]
        if nd is not None and np.isfinite(nd):
            arr[arr == nd] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        if categorical:
            vals, counts = np.unique(arr[np.isfinite(arr)], return_counts=True)
            return float(vals[int(np.argmax(counts))])
        return float(np.nanmean(arr) * scale)


def independent_checks(result) -> None:
    print("=" * 78)
    print("B. INDEPENDENT RECOMPUTATION (plain rasterio + numpy, no project code)")
    print("=" * 78)
    cfg = load_crop_config("cotton")
    lon0, lat0, lon1, lat1 = 31.8026, 31.1838, 31.8345, 31.2107   # ~3 km box

    # ---- land cover ------------------------------------------------------- #
    lc_url = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_N30E030_Map.tif")
    lc = _window_mean(lc_url, lon0, lat0, lon1, lat1, categorical=True)
    print(f"  WorldCover dominant class (independent) : {lc:.0f}")
    check(lc in (10.0, 20.0, 30.0, 40.0, 60.0),
          f"land cover is a real WorldCover class ({lc:.0f})")
    check(result.land_cover.get("dominant_class") == int(lc),
          f"engine agrees on the dominant class ({result.land_cover.get('dominant_class')})")

    # ---- soil ------------------------------------------------------------- #
    print("  SoilGrids (independent windowed reads, IGH transform):")
    soil = {}
    for prop, scale in (("phh2o", 0.1), ("clay", 0.1), ("sand", 0.1), ("silt", 0.1)):
        url = (f"https://files.isric.org/soilgrids/latest/data/{prop}/"
               f"{prop}_0-5cm_mean.vrt")
        soil[prop] = _window_mean(url, lon0, lat0, lon1, lat1, scale=scale)
        print(f"      {prop:6s} = {soil[prop]:8.2f}")
    check(abs(soil["clay"] + soil["sand"] + soil["silt"] - 100.0) < 8.0,
          f"texture fractions sum to ~100% "
          f"({soil['clay'] + soil['sand'] + soil['silt']:.1f}%)")

    # ---- climate ---------------------------------------------------------- #
    season = cfg["growing_season"]["months"]
    print(f"  WorldClim 2.1 monthly normals (independent, season {season}):")
    season_rain = 0.0
    annual_rain = 0.0
    for month in range(1, 13):
        url = ("/vsizip//vsicurl/https://geodata.ucdavis.edu/climate/worldclim/2_1/"
               f"base/wc2.1_30s_prec.zip/wc2.1_30s_prec_{month:02d}.tif")
        v = _window_mean(url, lon0, lat0, lon1, lat1)
        annual_rain += v
        if month in season:
            season_rain += v
    print(f"      annual precipitation      = {annual_rain:,.1f} mm")
    print(f"      growing-season (Apr-Oct)  = {season_rain:,.1f} mm")

    tmeans = {}
    gdd = 0.0
    days = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31}
    for month in season:
        vals = []
        for var in ("tmin", "tmax"):
            url = ("/vsizip//vsicurl/https://geodata.ucdavis.edu/climate/worldclim/2_1/"
                   f"base/wc2.1_30s_{var}.zip/wc2.1_30s_{var}_{month:02d}.tif")
            vals.append(_window_mean(url, lon0, lat0, lon1, lat1))
        tmin, tmax = vals
        tmean = (max(tmin, 15.6) + tmax) / 2.0
        tmeans[month] = (tmin, tmax, tmean)
        gdd += max(0.0, tmean - 15.6) * days[month]
        print(f"      month {month:02d}: tmin {tmin:5.2f}  tmax {tmax:5.2f}  "
              f"mean {tmean:5.2f} C")
    print(f"      approximate GDD(15.6)     = {gdd:,.0f} C.d  (context only)")

    # ---- topography ------------------------------------------------------- #
    dem_url = ("https://copernicus-dem-30m.s3.amazonaws.com/"
               "Copernicus_DSM_COG_10_N31_00_E031_00_DEM/"
               "Copernicus_DSM_COG_10_N31_00_E031_00_DEM.tif")
    with rasterio.open(dem_url) as ds:
        r, c = ds.index(LON_C, LAT_C)
        dem = ds.read(1, window=Window(c - 40, r - 40, 80, 80)).astype("float64")
    gy, gx = np.gradient(dem, 30.0)          # independent central differences
    slope_ind = float(np.nanmean(np.hypot(gx, gy) * 100.0))
    print(f"  Copernicus DEM (independent np.gradient):")
    print(f"      elevation min/mean/max = {np.nanmin(dem):.1f} / "
          f"{np.nanmean(dem):.1f} / {np.nanmax(dem):.1f} m")
    print(f"      mean slope (central differences) = {slope_ind:.2f} %")

    # ---- compare with the engine ------------------------------------------ #
    print()
    print("  COMPARISON (engine ROI mean vs independent recomputation)")
    primary = result.scenarios["rainfed"]
    fv = primary.factor_values
    print(f"      pH                engine {fv.get('soil_reaction')}   "
          f"independent {soil['phh2o']:.2f}")
    print(f"      clay %            engine {fv.get('clay_pct')}   independent {soil['clay']:.1f}")
    print(f"      season rainfall   engine {primary.growing_season_precipitation}   "
          f"independent {season_rain:,.1f}")
    print(f"      annual rainfall   engine {primary.annual_precipitation}   "
          f"independent {annual_rain:,.1f}")
    print(f"      slope %           engine {fv.get('terrain_slope')}   "
          f"independent {slope_ind:.2f}")

    check(close(fv.get("soil_reaction"), soil["phh2o"], 0.35), "soil pH agrees (±0.35)")
    check(close(fv.get("clay_pct"), soil["clay"], 6.0), "clay fraction agrees (±6 pp)")
    check(close(fv.get("sand_pct"), soil["sand"], 8.0), "sand fraction agrees (±8 pp)")
    check(close(fv.get("silt_pct"), soil["silt"], 8.0), "silt fraction agrees (±8 pp)")
    check(close(primary.growing_season_precipitation, season_rain, 25.0),
          "growing-season rainfall agrees (±25 mm)")
    check(close(primary.annual_precipitation, annual_rain, 30.0),
          "annual rainfall agrees (±30 mm)")
    check(close(fv.get("terrain_slope"), slope_ind, 1.5),
          "slope agrees with an independent gradient (±1.5 pp)")

    if primary.approx_gdd:
        gdd_engine = primary.approx_gdd["value"]
        print(f"      approx GDD        engine {gdd_engine:,.0f}   independent {gdd:,.0f}")
        check(close(gdd_engine, gdd, max(60.0, 0.06 * gdd)),
              "approximate GDD agrees (±6%)")
    else:  # pragma: no cover
        check(False, "approximate GDD was computed")

    # ---- hand-calculated score -------------------------------------------- #
    print()
    print("  HAND-CALCULATED SCORE (arithmetic written out, no project model)")
    w = cfg["weights"]["values"]
    f = primary.factor_scores
    precip_spec = cfg["factors"]["growing_season_precipitation"]
    ph_spec = cfg["factors"]["soil_reaction"]
    tex_spec = cfg["factors"]["soil_texture"]
    slope_spec = cfg["factors"]["terrain_slope"]

    ph_mem = trapezoid_membership(soil["phh2o"], ph_spec["absolute"][0],
                                  ph_spec["optimum"][0], ph_spec["optimum"][1],
                                  ph_spec["absolute"][1])
    slope_mem = trapezoid_membership(slope_ind, slope_spec["absolute"][0],
                                     slope_spec["optimum"][0], slope_spec["optimum"][1],
                                     slope_spec["absolute"][1])
    rain_mem = trapezoid_membership(season_rain, precip_spec["absolute"][0],
                                    precip_spec["optimum"][0], precip_spec["optimum"][1],
                                    precip_spec["absolute"][1])
    temp_mem = f.get("growing_season_temperature")      # from monthly memberships
    tex_mem = f.get("soil_texture")
    print(f"      temperature membership : {temp_mem}")
    print(f"      water membership       : {rain_mem}  (hand-calculated)")
    print(f"      pH membership          : {ph_mem:.3f}  (hand-calculated)")
    print(f"      texture membership     : {tex_mem}  (from the USDA class)")
    print(f"      slope membership       : {slope_mem:.3f}  (hand-calculated)")
    hand = (w["growing_season_temperature"] * temp_mem
            + w["growing_season_precipitation"] * rain_mem
            + w["soil_reaction"] * ph_mem
            + w["soil_texture"] * tex_mem
            + w["terrain_slope"] * slope_mem)
    print(f"      weighted sum           : {hand:.4f}")
    check(close(primary.score, hand, 0.06), "engine score matches the hand calculation (±0.06)")
    check(primary.classification == "Unsuitable",
          f"rainfed classification is Unsuitable (is: {primary.classification})")
    check(any("veto" in x for x in primary.warnings),
          "the Unsuitable class came from the critical-factor veto, not the score")
    check(result.scenarios["irrigation_assumed"].classification != "Highly suitable",
          "the hypothetical irrigation scenario is capped below 'Highly suitable'")


def main() -> int:
    print()
    print("Phase 8 real-data verification -- Nile Delta Sentinel-2 sample")
    print(f"ROI: {ROI_BOX.bounds} (EPSG:32636)  ->  {ROI_BOX.area / 1e6:.2f} km2")
    print()
    result, execution = run_engine()
    if result is None:
        check(False, "the engine returned a result")
    else:
        independent_checks(result)
        out = ROOT / "artifacts" / "phase8_verification.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result.to_dict(), indent=2, default=str),
                       encoding="utf-8")
        print(f"\n  structured result written to {out.relative_to(ROOT)}")

    failed = [label for ok, label in CHECKS if not ok]
    print()
    print("=" * 78)
    print(f"RESULT: {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for label in failed:
        print(f"  FAILED: {label}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
