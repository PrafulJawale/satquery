"""Phase 10 -- the temporal scene-pair model, its validation and its alignment.

WHAT THIS MODULE IS
-------------------
A pair of satellite acquisitions is not "two rasters": it is two measurements of
the same ground, taken by (possibly) different sensors, on different days, and
possibly published on different grids. Subtracting them is only meaningful
after a series of explicit checks, so those checks live here rather than inside
the analysis engine.

LAYERING
--------
`core/` must not import `analyses/`. The outcome vocabulary therefore lives here
as `TemporalStatus`, whose *values* are the same strings as
`analyses.base.Status`; `analyses/ndvi_change.py` converts one to the other.
Nothing else in the app needs to know both exist.

THE FOUR THINGS THIS MODULE GUARANTEES
-------------------------------------
1. **No silent subtraction across grids.** `plan_alignment()` states, in a
   record, whether the two scenes share a grid. If they do, the difference is
   computed on the native grid. If they do not, BOTH are resampled onto one
   explicit common grid -- never one into the other, and never implicitly.
2. **No fabricated coverage.** `validate_pair()` refuses to proceed unless the
   ROI is covered by BOTH scenes and the scenes overlap each other.
3. **No resampling claim.** Resampling is recorded with its method and cell
   size, and the alignment record says in plain words that it adds no
   information.
4. **Nodata is per scene.** Validity is the INTERSECTION of the two scenes'
   valid masks; a pixel that is valid on one date only is not a change of zero,
   it is an unknown.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

TEMPORAL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "temporal"
)


# =========================================================================== #
# configuration
# =========================================================================== #
def load_temporal_config(name: str = "ndvi_change",
                         base_dir: str = TEMPORAL_CONFIG_DIR) -> Dict[str, Any]:
    """Load config/temporal/<name>.yml (mirrors core.spatial_query and core.suitability)."""
    import yaml

    path = os.path.join(base_dir, f"{name}.yml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Temporal configuration is not a mapping: {path}")
    return cfg


# =========================================================================== #
# outcome vocabulary (string-compatible with analyses.base.Status)
# =========================================================================== #
class TemporalStatus(str, Enum):
    """Why a temporal comparison could or could not be produced."""

    OK = "OK"
    NEEDS_TWO_DATES = "NEEDS_TWO_DATES"        # missing, equal or unordered dates
    NO_TEMPORAL_OVERLAP = "NO_TEMPORAL_OVERLAP"  # the two scenes share no ground
    UNSUPPORTED = "UNSUPPORTED"                # a scene lacks a required band
    ERROR = "ERROR"                            # the pair itself is unusable


# =========================================================================== #
# a single acquisition
# =========================================================================== #
@dataclass
class SceneRef:
    """One acquisition of one area, described well enough to be compared.

    Deliberately holds NO pixel data: building a SceneRef opens the file, reads
    its metadata and closes it again. Two of these cost kilobytes, not the
    ~34 MB a 2048x2048 uint16 stack would cost.
    """

    path: str
    label: str = ""
    date: Optional[date] = None
    red_index: int = 3          # 1-based rasterio band index
    nir_index: int = 4
    crs: Any = None
    transform: Optional[Affine] = None
    width: int = 0
    height: int = 0
    resolution: float = 0.0
    bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    band_count: int = 0
    band_names: Tuple[str, ...] = ()
    scale: float = 0.0001       # DN -> reflectance
    offset: float = 0.0
    nodata: Optional[float] = None
    source: Dict[str, Any] = field(default_factory=dict)   # provenance block

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_path(cls,
                  path: str,
                  *,
                  date: Optional[Any] = None,
                  label: Optional[str] = None,
                  red_index: Optional[int] = None,
                  nir_index: Optional[int] = None,
                  scale: Optional[float] = None,
                  offset: Optional[float] = None) -> "SceneRef":
        """Read a scene's metadata (never its pixels) from `path`.

        Band indexes, scale and acquisition date are taken from the sidecar
        provenance when it exists, because that is where the fetch scripts
        recorded them; explicit arguments override it.
        """
        sidecar: Dict[str, Any] = {}
        sidecar_path = f"{path}.provenance.json"
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as fh:
                    sidecar = json.load(fh)
            except (json.JSONDecodeError, OSError):
                sidecar = {}
        tech = sidecar.get("technical") or {}

        with rasterio.open(path) as ds:
            names = tuple(str(d or "") for d in ds.descriptions)
            # Prefer name-based detection (B04_red..., B08_nir...), fall back to
            # the documented 4-band sample order blue/green/red/nir.
            r_idx = red_index if red_index is not None else _band_by_name(names, "red") or 3
            n_idx = nir_index if nir_index is not None else _band_by_name(names, "nir") or 4
            res = float(ds.res[0])
            scene = cls(
                path=path,
                label=label or os.path.basename(path),
                date=parse_date(date if date is not None else tech.get("datetime")),
                red_index=int(r_idx),
                nir_index=int(n_idx),
                crs=ds.crs,
                transform=ds.transform,
                width=int(ds.width),
                height=int(ds.height),
                resolution=res,
                bounds=tuple(float(v) for v in ds.bounds),
                band_count=int(ds.count),
                band_names=names,
                scale=float(tech.get("scale_to_reflectance", scale if scale is not None else 0.0001)),
                offset=float(tech.get("reflectance_offset", offset if offset is not None else 0.0)),
                nodata=ds.nodatavals[0] if ds.nodatavals else None,
                source={
                    "item_id": tech.get("item_id", ""),
                    "datetime": tech.get("datetime", ""),
                    "platform": tech.get("platform", ""),
                    "cloud_cover": tech.get("cloud_cover"),
                    "mgrs_tile": tech.get("mgrs_tile", ""),
                    "epsg": tech.get("epsg"),
                    "license": tech.get("license", ""),
                    "source_url": sidecar.get("source_url", ""),
                    "bands": tech.get("bands", list(names)),
                },
            )
        return scene

    # -- derived ----------------------------------------------------------- #
    @property
    def has_required_bands(self) -> bool:
        """True when a red and a near-infrared band actually exist."""
        return (
            self.band_count > 0
            and 1 <= self.red_index <= self.band_count
            and 1 <= self.nir_index <= self.band_count
            and self.red_index != self.nir_index
        )

    @property
    def is_dated(self) -> bool:
        return self.date is not None

    @property
    def date_label(self) -> str:
        return self.date.isoformat() if self.date else "unknown date"

    @property
    def footprint(self):
        """The scene's extent as a shapely box, in the scene's own CRS."""
        return box(*self.bounds)

    @property
    def area_km2(self) -> float:
        minx, miny, maxx, maxy = self.bounds
        return abs((maxx - minx) * (maxy - miny)) / 1e6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "date": self.date.isoformat() if self.date else None,
            "red_index": self.red_index,
            "nir_index": self.nir_index,
            "crs": str(self.crs) if self.crs is not None else None,
            "transform": list(self.transform)[:6] if self.transform is not None else None,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "bounds": list(self.bounds),
            "band_count": self.band_count,
            "band_names": list(self.band_names),
            "scale": self.scale,
            "offset": self.offset,
            "nodata": self.nodata,
            "has_required_bands": self.has_required_bands,
            "source": dict(self.source),
        }


@dataclass
class ScenePair:
    """The two acquisitions of a temporal comparison, in chronological order."""

    before: Optional[SceneRef] = None
    after: Optional[SceneRef] = None

    @property
    def complete(self) -> bool:
        return self.before is not None and self.after is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
        }


# =========================================================================== #
# grid comparison and alignment
# =========================================================================== #
@dataclass
class AlignmentRecord:
    """How (or whether) the two scenes were brought onto one grid.

    This record is provenance, not decoration: it is returned with every result
    so the UI can say exactly what was resampled and why.
    """

    method: str = "none"                # identical_grid | common_grid_resampled | none
    crs: Optional[str] = None
    transform: Optional[Tuple[float, ...]] = None
    width: int = 0
    height: int = 0
    resolution: Optional[float] = None
    resampling: Optional[str] = None    # None when nothing was resampled
    target: str = ""                    # which scene's grid was adopted
    note: str = ""

    @property
    def resampled(self) -> bool:
        return self.method == "common_grid_resampled"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "crs": self.crs,
            "transform": list(self.transform) if self.transform else None,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "resampling": self.resampling,
            "target": self.target,
            "resampled": self.resampled,
            "note": self.note,
        }


#: The wording used whenever resampling happened. It must travel with the result:
#: a resampled difference is not a native-resolution measurement.
RESAMPLING_NOTE = (
    "Both scenes were resampled onto one common grid so that a pixel-by-pixel "
    "difference is defined. Resampling adds no information: the comparison is "
    "limited to the coarser of the two native resolutions."
)

IDENTICAL_NOTE = (
    "The two scenes share one CRS, affine transform and shape, so the difference "
    "was computed on the native grid with no resampling."
)


def grids_identical(a: SceneRef, b: SceneRef) -> bool:
    """True when the two scenes are pixel-for-pixel comparable as published.

    Same CRS, same affine (to floating-point tolerance), same shape. This is the
    only case in which subtracting two arrays is meaningful without an explicit
    resampling step.
    """
    if a.transform is None or b.transform is None:
        return False
    if (a.width, a.height) != (b.width, b.height):
        return False
    if a.crs is None or b.crs is None:
        return False
    try:
        same_crs = CRS.from_user_input(a.crs) == CRS.from_user_input(b.crs)
    except Exception:
        same_crs = str(a.crs) == str(b.crs)
    if not same_crs:
        return False
    return all(abs(float(x) - float(y)) <= 1e-9
               for x, y in zip(tuple(a.transform)[:6], tuple(b.transform)[:6]))


def plan_alignment(before: SceneRef,
                   after: SceneRef,
                   geometry: Any,
                   config: Optional[Dict[str, Any]] = None) -> AlignmentRecord:
    """Decide how the two scenes will be compared, and record the decision.

    * Identical grids  -> adopt the native grid. `resampling` stays None.
    * Different grids  -> build ONE common grid over the ROI at the COARSER of
      the two native cell sizes, in the CRS of the coarser scene.

    The ROI geometry limits how much is ever read: the comparison is ROI-first,
    exactly like the single-date analyses.
    """
    cfg = config or load_temporal_config()
    alg = cfg.get("alignment", {}) or {}
    identical = grids_identical(before, after)

    if identical:
        return AlignmentRecord(
            method="identical_grid",
            crs=str(before.crs),
            transform=tuple(float(v) for v in before.transform)[:6],
            width=before.width,
            height=before.height,
            resolution=float(before.resolution),
            resampling=None,
            target="both (native grid)",
            note=IDENTICAL_NOTE,
        )

    # Different grids: adopt the COARSER scene's grid. Ties go to `before`, so
    # the choice is deterministic and reproducible.
    if float(before.resolution) >= float(after.resolution):
        target_scene, other = before, after
        target_name = "before"
    else:
        target_scene, other = after, before
        target_name = "after"

    res = float(target_scene.resolution)
    minx, miny, maxx, maxy = geometry.bounds
    buf = int(alg.get("buffer_cells", 2) or 0)
    left = float(np.floor((minx - buf * res) / res) * res)
    bottom = float(np.floor((miny - buf * res) / res) * res)
    right = float(np.ceil((maxx + buf * res) / res) * res)
    top = float(np.ceil((maxy + buf * res) / res) * res)

    max_cells = int(alg.get("max_cells", 2_000_000) or 2_000_000)
    width = max(1, int(round((right - left) / res)))
    height = max(1, int(round((top - bottom) / res)))
    while width * height > max_cells:          # coarsen, never fail silently
        res *= 2.0
        width = max(1, int(round((right - left) / res)))
        height = max(1, int(round((top - bottom) / res)))

    return AlignmentRecord(
        method="common_grid_resampled",
        crs=str(target_scene.crs),
        transform=(res, 0.0, left, 0.0, -res, top),
        width=width,
        height=height,
        resolution=res,
        resampling=str(alg.get("resampling", "nearest")),
        target=f"{target_name} scene ({target_scene.label})",
        note=(
            f"{RESAMPLING_NOTE} The {target_name} scene sets the grid at {res:g} m "
            f"(the coarser native resolution); the other scene is {float(other.resolution):g} m."
        ),
    )


# =========================================================================== #
# validation
# =========================================================================== #
def covers(scene: SceneRef, geometry: Any, geometry_crs: Any = None) -> bool:
    """True when `geometry` lies inside the scene's footprint.

    `geometry_crs` is the CRS the geometry is currently expressed in. When it
    differs from the scene's CRS the geometry is transformed first; a geometry
    with no CRS is assumed to already be in the scene's CRS (the caller -- the
    ROI pipeline -- guarantees that for the primary raster).
    """
    if geometry is None or scene.transform is None:
        return False
    geom = geometry
    if geometry_crs is not None and scene.crs is not None:
        try:
            if CRS.from_user_input(geometry_crs) != CRS.from_user_input(scene.crs):
                geom = reproject_geometry(geometry, geometry_crs, scene.crs)
        except Exception:
            return False
    try:
        return bool(scene.footprint.covers(geom))
    except Exception:
        return False


def reproject_geometry(geometry: Any, src_crs: Any, dst_crs: Any) -> Any:
    """Reproject a shapely geometry between two CRSs."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True
    )
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geometry)


def validate_pair(pair: ScenePair,
                  geometry: Any,
                  geometry_crs: Any = None,
                  config: Optional[Dict[str, Any]] = None
                  ) -> Tuple[TemporalStatus, str, Tuple[str, ...]]:
    """Every check that stands between a request and a pixel difference.

    Returns `(status, message, warnings)`. `status is TemporalStatus.OK` means
    the pair may be compared; anything else means the caller must stop and show
    the message instead of a number.

    Order matters: the cheapest and most user-actionable failure is reported
    first, so the message tells the user what to DO.
    """
    cfg = config or load_temporal_config()
    warnings: List[str] = []

    if pair is None or not pair.complete:
        return (TemporalStatus.NEEDS_TWO_DATES,
                "Two acquisitions are required. Select a before scene and an "
                "after scene (the dates are never chosen automatically).",
                tuple(warnings))

    before, after = pair.before, pair.after

    # 1. required bands ---------------------------------------------------- #
    for role, scene in (("before", before), ("after", after)):
        if not scene.has_required_bands:
            return (TemporalStatus.UNSUPPORTED,
                    f"The {role} scene has no usable red and near-infrared band, "
                    f"so NDVI cannot be computed for it "
                    f"({scene.label}: {scene.band_count} band(s) available).",
                    tuple(warnings))

    # 2. dates -------------------------------------------------------------- #
    if not before.is_dated or not after.is_dated:
        missing = [r for r, s in (("before", before), ("after", after)) if not s.is_dated]
        return (TemporalStatus.NEEDS_TWO_DATES,
                f"No acquisition date is recorded for the {' and '.join(missing)} "
                f"scene. Both dates must be known before a change can be attributed "
                f"to a time interval.",
                tuple(warnings))
    if before.date == after.date:
        return (TemporalStatus.NEEDS_TWO_DATES,
                f"Both scenes are dated {before.date_label}. A comparison needs two "
                f"different acquisition dates.",
                tuple(warnings))
    if before.date > after.date:
        return (TemporalStatus.ERROR,
                f"The before date ({before.date_label}) is later than the after date "
                f"({after.date_label}). Assign the earlier acquisition as 'before'.",
                tuple(warnings))

    # 3. the two scenes must share ground ----------------------------------- #
    overlap = _footprint_overlap(before, after)
    if overlap is None or overlap <= 0:
        return (TemporalStatus.NO_TEMPORAL_OVERLAP,
                "The two scenes do not cover any common area, so no change can be "
                "measured between them.",
                tuple(warnings))

    # 4. the ROI must be covered by BOTH ------------------------------------- #
    if geometry is not None:
        uncovered = [r for r, s in (("before", before), ("after", after))
                     if not covers(s, geometry, geometry_crs)]
        if uncovered:
            return (TemporalStatus.NO_TEMPORAL_OVERLAP,
                    "The selected area is not covered by both acquisitions "
                    f"({', '.join(uncovered)}), so no change is reported for it.",
                    tuple(warnings))

    # 5. consistency warnings (do NOT block) --------------------------------- #
    if before.source.get("platform") and after.source.get("platform"):
        if before.source["platform"] != after.source["platform"]:
            warnings.append(
                f"The two scenes come from different platforms "
                f"({before.source['platform']} vs {after.source['platform']}); "
                f"band-response differences can contribute to the measured change."
            )
    if before.crs is not None and after.crs is not None:
        try:
            if CRS.from_user_input(before.crs) != CRS.from_user_input(after.crs):
                warnings.append(
                    f"The two scenes are published in different CRSs "
                    f"({before.crs} vs {after.crs}); both are reprojected onto one "
                    f"explicit common grid."
                )
        except Exception:
            pass
    if not grids_identical(before, after):
        warnings.append(RESAMPLING_NOTE)

    return TemporalStatus.OK, "", tuple(warnings)


def _footprint_overlap(a: SceneRef, b: SceneRef) -> Optional[float]:
    """Area common to both scenes, in the units of `a`'s CRS squared.

    Only compared when both share a CRS; a cross-CRS pair is handled by
    `plan_alignment()` (which reprojects) and by the coverage check above.
    """
    try:
        if a.crs is None or b.crs is None:
            return None
        if CRS.from_user_input(a.crs) != CRS.from_user_input(b.crs):
            # Cannot measure the area without reprojecting; a conservative
            # check on transformed bounds is enough to detect "no common area".
            geom = reproject_geometry(b.footprint, b.crs, a.crs)
            inter = a.footprint.intersection(geom)
        else:
            inter = a.footprint.intersection(b.footprint)
        return float(inter.area) if not inter.is_empty else 0.0
    except Exception:
        return None


# =========================================================================== #
# helpers
# =========================================================================== #
def parse_date(value: Any) -> Optional[date]:
    """Parse an ISO date/datetime (or a date) into a `datetime.date`.

    Returns None for anything unparseable or empty -- an unknown date must
    surface as NEEDS_TWO_DATES, never as a guessed one.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _band_by_name(names: Sequence[str], token: str) -> Optional[int]:
    """1-based index of the band whose description contains `token`."""
    lowered = [(i, str(n or "").lower()) for i, n in enumerate(names, start=1)]
    for idx, name in lowered:
        if token in name:
            return idx
    return None


def discover_scenes(sample_dir: str,
                    config: Optional[Dict[str, Any]] = None) -> List[SceneRef]:
    """Every acquisition in `sample_dir` that can take part in a comparison.

    A scene qualifies when it has a red and a near-infrared band AND a recorded
    acquisition date. Scenes that fail either test are skipped rather than
    offered: an undated scene cannot be ordered, and a band-less one cannot be
    compared. Sorted oldest first.
    """
    cfg = config or load_temporal_config()
    red_index = int((cfg.get("ndvi", {}) or {}).get("red_band_index", 3))
    nir_index = int((cfg.get("ndvi", {}) or {}).get("nir_band_index", 4))

    scenes: List[SceneRef] = []
    directory = str(sample_dir)
    if not os.path.isdir(directory):
        return scenes
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(directory, name)
        try:
            scene = SceneRef.from_path(path, red_index=red_index, nir_index=nir_index)
        except Exception:
            continue          # unreadable / not a raster: not a scene, skip it
        if scene.has_required_bands and scene.is_dated:
            scenes.append(scene)
    scenes.sort(key=lambda s: (s.date or date.min, s.label))
    return scenes


def pair_by_dates(scenes: Sequence[SceneRef],
                  before_date: Any,
                  after_date: Any) -> Optional[ScenePair]:
    """Build a ScenePair by matching two exact dates.

    Returns None when either date is absent from `scenes`: a temporal
    comparison must never silently substitute the "nearest" date.
    """
    want_before = parse_date(before_date)
    want_after = parse_date(after_date)
    if want_before is None or want_after is None:
        return None
    before = next((s for s in scenes if s.date == want_before), None)
    after = next((s for s in scenes if s.date == want_after), None)
    if before is None or after is None:
        return None
    return ScenePair(before=before, after=after)


__all__ = [
    "TEMPORAL_CONFIG_DIR",
    "load_temporal_config",
    "TemporalStatus",
    "SceneRef",
    "ScenePair",
    "AlignmentRecord",
    "IDENTICAL_NOTE",
    "RESAMPLING_NOTE",
    "grids_identical",
    "plan_alignment",
    "validate_pair",
    "covers",
    "reproject_geometry",
    "parse_date",
    "discover_scenes",
    "pair_by_dates",
]
