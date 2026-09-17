"""core/models.py -- the shared vocabulary of SatQuery.

WHY THIS FILE EXISTS
--------------------
Every later phase (preview, NDVI, NDWI, change detection, crop suitability,
query routing) has to agree on what "a raster" and "a result" look like.

If the UI, the analysis code and the tests all speak these same dataclasses,
then:

  * `ui/` never needs to import rasterio (it just renders dicts),
  * results are JSON-serialisable -> easy Streamlit caching, easy export,
  * we can later swap rasterio for rioxarray/xarray/stac without touching `ui/`.

DESIGN RULE: `core/` must never import Streamlit. It is pure, testable Python.
The UI converts these objects to dicts via `.to_dict()`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A geographic coordinate as (longitude, latitude) in EPSG:4326 order.
# NOTE: we standardise on (x, y) == (lon, lat) == (easting, northing).
# GDAL 3 changed PROJ axis order rules; rasterio's `warp.transform` always
# uses (lon, lat) so we follow that convention everywhere.
Coord = Tuple[float, float]


@dataclass(frozen=True)
class BandInfo:
    """Metadata for a single raster band (1-based index, like GDAL uses)."""

    index: int
    name: Optional[str]          # GDAL band description, often None
    dtype: str                   # e.g. "uint8", "uint16", "float32"
    nodata: Optional[float]      # declared nodata for this band (may be None)
    block_shape: Tuple[int, int]
    color_interp: Optional[str]  # GDAL's guess ("red"/"green"/"blue"/...); the only
                                 # clue to band meaning in some files

    @property
    def label(self) -> str:
        """Human label: use the band description if the file provides one."""
        return self.name if self.name else f"Band {self.index}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpatialInfo:
    """Everything about WHERE the raster lives on Earth.

    This is the part that most prototypes get wrong: CRS, transform and bounds
    must travel together. Never carry an array around without them.
    """

    has_crs: bool
    crs_epsg: Optional[int]
    crs_name: Optional[str]
    crs_wkt: Optional[str]
    is_geographic: bool
    is_projected: bool
    linear_units: Optional[str]          # "metre", "US survey foot", ... (projected only)

    transform: Tuple[float, ...]         # GDAL/Affine 6-tuple (a,b,c,d,e,f)
    pixel_size_x: float
    pixel_size_y: float
    is_north_up: bool                    # no rotation/skew -> safe for simple overlays
    is_rotated: bool                     # b != 0 or d != 0 -> needs reprojection for web maps

    bounds_native: Optional[Tuple[float, float, float, float]]  # in the file's own CRS
    bounds_wgs84: Optional[Tuple[float, float, float, float]]   # (min_lon, min_lat, max_lon, max_lat)
    footprint_wgs84: Optional[List[Coord]]  # densified outline, NOT just 4 corners
    approx_area_km2: Optional[float]

    @property
    def resolution_label(self) -> str:
        """e.g. '300.04 m x 300.04 m' or '0.0003 deg x 0.0003 deg'."""
        unit = self.linear_units or "deg"
        sx, sy = abs(self.pixel_size_x), abs(self.pixel_size_y)
        if not self.is_geographic and unit.startswith(("metre", "meter")):
            return f"{sx:.4g} m x {sy:.4g} m"
        if self.is_geographic:
            return f"{sx:.6g} deg x {sy:.6g} deg"
        return f"{sx:.4g} x {sy:.4g} {unit}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolution_label"] = self.resolution_label
        return d


@dataclass(frozen=True)
class RasterInfo:
    """Complete description of an opened GeoTIFF. No pixels are stored here."""

    source_label: str                 # what to show the user ("upload.tif", sample name)
    source_path: Optional[str]        # None for in-memory uploads
    file_size_bytes: Optional[int]

    driver: str
    width: int
    height: int
    count: int
    dtypes: Tuple[str, ...]
    bands: Tuple[BandInfo, ...]

    spatial: SpatialInfo

    nodata_per_band: Tuple[Optional[float], ...]
    tiled: bool
    overview_levels: Tuple[int, ...]
    block_shape: Tuple[int, int]
    compress: Optional[str]
    interleave: Optional[str]
    looks_like_cog: bool              # heuristic, see core/raster.py

    tags: Dict[str, str]              # a trimmed subset of GDAL metadata
    warnings: Tuple[str, ...]         # honest caveats the UI MUST surface
    estimated_full_read_mb: float

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    @property
    def is_multispectral_candidate(self) -> bool:
        """Heuristic only: >=4 bands is the usual minimum for red+NIR work.

        It is NOT proof that NIR exists. Band meaning must be confirmed by the
        user (or by reading band descriptions / STAC metadata). See README.
        """
        return self.count >= 4

    @property
    def is_analysis_ready_int(self) -> bool:
        """True when values look like raw sensor digital numbers / reflectance."""
        return any(d in ("uint16", "int16", "uint32", "int32", "float32", "float64")
                   for d in self.dtypes)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        d["total_pixels"] = self.total_pixels
        d["is_multispectral_candidate"] = self.is_multispectral_candidate
        d["is_analysis_ready_int"] = self.is_analysis_ready_int
        d["spatial"] = self.spatial.to_dict()
        d["bands"] = [b.to_dict() for b in self.bands]
        return d


@dataclass(frozen=True)
class Provenance:
    """Where a sample file came from. Honesty is a feature, not a nice-to-have."""

    name: str
    filename: str
    source_url: str
    what_it_is: str
    is_real_satellite_data: bool
    bands_are_physically_valid: bool   # False for 8-bit display products
    good_for: Tuple[str, ...]          # e.g. ("metadata", "rgb_preview")
    not_good_for: Tuple[str, ...]      # e.g. ("ndvi",)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisResult:
    """The contract for every analysis we add from Phase 3 onwards.

    NDVI, NDWI, change detection and crop suitability all return the SAME shape
    of object: pixels + validity mask + georeferencing + numbers + provenance.

    Without this shared shape, the map code turns into one `if` per analysis.

    TWO GEOMETRIES, NEVER CONFUSED
        `array` / `transform` describe the NATIVE-resolution analytical result.
        Rendering decimates a COPY; the display geometry lives on the render
        result, never in here.
    """

    name: str                        # "ndvi"
    label: str = ""                  # human title
    description: str = ""
    method: str = ""                 # formula + citation
    array: Any = None                # numpy.ndarray (2D float32); NaN where invalid
    mask: Any = None                 # bool ndarray: True where array is valid
    crs: Any = None
    transform: Any = None
    nodata: float = float("nan")
    value_range: Tuple[float, float] = (-1.0, 1.0)     # theoretical
    observed_range: Optional[Tuple[float, float]] = None  # measured (min, max)
    colormap: str = "viridis"
    stats: Optional[Dict[str, Any]] = None   # None => no valid pixels; report nothing
    counts: Dict[str, Any] = field(default_factory=dict)
    reflectance: Any = None          # ReflectanceSpec actually applied
    bands_used: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    caveats: Tuple[str, ...] = ()

    # ---------------------------------------------------------------- #
    @property
    def has_valid_pixels(self) -> bool:
        return bool(self.counts.get("valid_pixels", 0)) and self.stats is not None

    @property
    def native_shape(self) -> Optional[Tuple[int, int]]:
        return None if self.array is None else tuple(self.array.shape)  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        """Metadata only -- pixel arrays are intentionally excluded."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "method": self.method,
            "crs": str(self.crs) if self.crs is not None else None,
            "transform": tuple(self.transform)[:6] if self.transform is not None else None,
            "nodata": self.nodata,
            "value_range": list(self.value_range),
            "observed_range": list(self.observed_range) if self.observed_range else None,
            "colormap": self.colormap,
            "stats": self.stats,
            "counts": self.counts,
            "bands_used": self.bands_used,
            "reflectance": self.reflectance.to_dict() if self.reflectance is not None else None,
            "provenance": self.provenance,
            "caveats": list(self.caveats),
            "native_shape": list(self.native_shape) if self.native_shape else None,
            "has_valid_pixels": self.has_valid_pixels,
        }

    def summary_lines(self) -> list[str]:
        """Short, factual sentences -- what the query router will read aloud later."""
        if not self.has_valid_pixels:
            return [
                f"{self.label or self.name}: no valid pixels; no statistics to report.",
                *[f"Caveat: {c}" for c in self.caveats[:3]],
            ]
        s, c = self.stats, self.counts
        lines = [
            f"{self.label or self.name} over {c['valid_pixels']:,} valid pixels "
            f"({c['valid_percentage']:.1f}% of {c['total_pixels']:,}).",
            f"Mean {s['mean']:.3f}, median {s['median']:.3f}, "
            f"range {s['min']:.3f} to {s['max']:.3f}, std {s['std']:.3f}.",
            f"Method: {self.method}",
        ]
        lines.extend(f"Caveat: {cav}" for cav in self.caveats[:3])
        return lines
