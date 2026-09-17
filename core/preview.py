"""core/preview.py -- raster -> displayable image (PHASE 2).

THE CORE IDEA: COMPUTE NATIVE, RENDER DECIMATED
-----------------------------------------------
A Sentinel-2 scene is 10980x10980 px. We must not resize it to 504x504 *for
analysis* (that would destroy area statistics and the geographic link), and we
also cannot ship a 10980x10980 PNG to a browser (that is ~360 MB of RGB).

So we separate the two concerns:

    analysis  -> full resolution, native CRS + transform   (Phase 3+)
    rendering -> decimated read, its OWN transform recorded

The decimated image carries its own Affine, so when Phase 4 overlays it on a
map it uses the *display* transform, while statistics keep using the *native*
one. Mixing those up is the classic "map and numbers disagree" bug.

WHY PERCENTILE STRETCH (AND NOT MIN/MAX)
----------------------------------------
Satellite reflectance is concentrated in a narrow low range with a long tail
(a few clouds, a few bright rooftops). Scaling 0..max makes everything black.
We clip the 2nd-98th percentile by default, computed **on valid pixels only** --
nodata (0) must never participate, or it drags the low end down and the whole
image turns washed out.

Every stretch is *display only*. It changes no data and must never be fed back
into an index computation.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling

DEFAULT_MAX_PIXELS = 1_500_000      # ~1200x1200: crisp in a browser, fast to render
_PERCENTILE_SAMPLE_CAP = 1_000_000  # subsample above this for percentile speed


@dataclass
class PreviewResult:
    """A rendered image plus everything needed to place or explain it."""

    image: Any                              # uint8 (H, W, 3) RGB
    alpha: Any                              # uint8 (H, W): 255 valid, 0 nodata
    transform: Any                          # Affine of the *rendered* image
    crs: Any
    display_shape: Tuple[int, int]
    source_shape: Tuple[int, int]
    channels: Tuple[str, ...]               # e.g. ("red", "green", "blue")
    band_indices: Tuple[int, ...]
    stretch_bounds: Tuple[Tuple[float, float], ...]
    stretch_percentiles: Tuple[float, float]
    band_stats: List[Dict[str, Any]] = field(default_factory=list)
    nodata_fraction: float = 0.0
    decimation_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Metadata only -- pixel arrays are deliberately excluded."""
        return {
            "display_shape": list(self.display_shape),
            "source_shape": list(self.source_shape),
            "channels": list(self.channels),
            "band_indices": list(self.band_indices),
            "stretch_bounds": [list(b) for b in self.stretch_bounds],
            "stretch_percentiles": list(self.stretch_percentiles),
            "band_stats": self.band_stats,
            "nodata_fraction": self.nodata_fraction,
            "decimation_factor": self.decimation_factor,
            "transform": tuple(self.transform)[:6],
            "crs": str(self.crs) if self.crs is not None else None,
        }


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def decimated_read(
    ds: rasterio.DatasetReader,
    band_indices: Sequence[int],
    max_pixels: int = DEFAULT_MAX_PIXELS,
    resampling: Resampling = Resampling.average,
) -> Tuple[np.ndarray, Affine]:
    """Read bands decimated to at most `max_pixels`, with the matching transform.

    `average` resampling (not `nearest`) so that downsampling does not alias --
    one stray bright pixel would otherwise flicker in and out as you zoom.

    Note: GDAL >= 3.1 excludes nodata from the average, so decimation does not
    smear black scene edges into real pixels.
    """
    indices = list(band_indices)
    h, w = ds.height, ds.width
    total = h * w
    factor = math.sqrt(total / max_pixels) if total > max_pixels else 1.0
    out_h = max(1, int(round(h / factor)))
    out_w = max(1, int(round(w / factor)))

    arr = ds.read(indices, out_shape=(len(indices), out_h, out_w), resampling=resampling)

    # Display transform: pixel (col,row) of the image maps to (col*sx, row*sy)
    # of the source. Written out longhand because `Affine.__mul__` is deprecated.
    t = ds.transform
    sx, sy = w / out_w, h / out_h
    transform = Affine(t.a * sx, t.b * sy, t.c, t.d * sx, t.e * sy, t.f)
    return arr, transform


def valid_mask(band: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """Boolean mask of pixels that carry a real measurement."""
    if band.dtype.kind == "f":
        mask = np.isfinite(band)
    else:
        mask = np.ones(band.shape, dtype=bool)
    if nodata is not None:
        if isinstance(nodata, float) and math.isnan(nodata):
            mask &= ~np.isnan(band)
        else:
            mask &= band != nodata
    return mask


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def band_stats(band: np.ndarray, nodata: Optional[float], name: str) -> Dict[str, Any]:
    """Robust statistics over VALID pixels only.

    IMPORTANT: these are computed on the decimated preview, so they are
    indicative, not authoritative. Phase 6 computes exact zonal statistics.
    """
    mask = valid_mask(band, nodata)
    vals = band[mask]
    out: Dict[str, Any] = {
        "band": name,
        "dtype": str(band.dtype),
        "nodata": nodata,
        "valid_pixels": int(vals.size),
        "total_pixels": int(band.size),
        "valid_fraction": (float(vals.size) / float(band.size)) if band.size else 0.0,
    }
    if vals.size == 0:
        out.update({"min": None, "max": None, "mean": None, "std": None,
                    "p2": None, "p50": None, "p98": None})
        return out
    v = vals.astype(np.float64)
    out.update({
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "p2": float(np.percentile(v, 2)),
        "p50": float(np.percentile(v, 50)),
        "p98": float(np.percentile(v, 98)),
    })
    return out


def percentile_bounds(
    values: np.ndarray, low: float = 2.0, high: float = 98.0, seed: int = 0
) -> Tuple[float, float]:
    """Clip bounds for stretching, computed on valid values only.

    Two display traps are handled here, because both turn a normal-looking
    band into a BLACK image on the map:

    * NaN / +-Inf -- percentiles over them are NaN, so the stretch divides by
      NaN and every pixel lands on 0. They are not data, so they are dropped
      before the percentiles are taken.
    * a degenerate range -- a constant band (or one whose spread is below the
      precision of the values). An epsilon-only expansion keeps `hi > lo` but
      still maps every pixel to 0. The range is given a floor instead, so a
      band with no contrast displays as MID GREY -- what every GIS does with a
      single-value stretch -- rather than as a black rectangle.
    """
    v = np.asarray(values, dtype="float64").ravel()
    v = v[np.isfinite(v)]                      # NaN and infinities are not data
    if v.size == 0:
        return 0.0, 1.0
    if v.size > _PERCENTILE_SAMPLE_CAP:
        rng = np.random.default_rng(seed)
        v = v[rng.choice(v.size, _PERCENTILE_SAMPLE_CAP, replace=False)]
    lo = float(np.percentile(v, low))
    hi = float(np.percentile(v, high))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return 0.0, 1.0
    span = hi - lo
    floor = max(abs(lo), abs(hi), 1.0) * 1e-6
    if span < floor:                   # constant or below-precision spread
        centre = 0.5 * (lo + hi)
        lo, hi = centre - 0.5 * floor, centre + 0.5 * floor
    return lo, hi


def stretch_to_uint8(
    band: np.ndarray, mask: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Linear stretch to 0..255. Nodata pixels become 0 (shown as transparent).

    `mask` selects the pixels that carry a measurement; everything else stays
    transparent. A band with no contrast (`hi <= lo`, or bounds that are not
    finite) is painted MID GREY rather than stretched to black: no data was
    lost, so the display must not imply that there was none.
    """
    out = np.zeros(band.shape, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    span = float(hi) - float(lo)
    if not math.isfinite(span) or span <= 0.0:
        out[mask] = 128                        # single value -> mid grey
        return out
    scaled = (np.asarray(band, dtype="float32") - float(lo)) * (255.0 / span)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0)
    np.clip(scaled, 0.0, 255.0, out=scaled)
    out[mask] = scaled[mask].astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# compositing
# --------------------------------------------------------------------------- #
def composite(
    stack: np.ndarray,
    masks: Sequence[np.ndarray],
    channel_bands: Sequence[int],
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    stretch_mode: str = "per_band",
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[Tuple[float, float], ...]]:
    """Build an 8-bit RGB image from selected bands.

    channel_bands: index into `stack` for each display channel (R, G, B).
    stretch_mode:  "per_band" -> each channel stretched independently (default,
                    gives a natural-looking image);
                   "joint"    -> one range shared by all three channels, which
                    preserves relative brightness between bands and is the
                    honest choice when comparing two composites.
    """
    if len(channel_bands) != 3:
        raise ValueError("composite() needs exactly three channel band indices")

    bounds: List[Tuple[float, float]] = []
    if stretch_mode == "joint":
        pooled = [stack[i][masks[i]] for i in channel_bands]
        pooled = [p for p in pooled if p.size]
        lo, hi = percentile_bounds(np.concatenate(pooled) if pooled else np.array([]),
                                   low_pct, high_pct, seed=seed)
        bounds = [(lo, hi)] * 3
    else:
        for i in channel_bands:
            bounds.append(percentile_bounds(stack[i][masks[i]], low_pct, high_pct, seed=seed))

    rgb = np.dstack(
        [
            stretch_to_uint8(stack[i], masks[i], bounds[k][0], bounds[k][1])
            for k, i in enumerate(channel_bands)
        ]
    )

    # A display pixel counts as data only when ALL THREE channels are valid
    # (the convention used by QGIS and Earth Engine). Where any channel is
    # missing we zero the colour as well as the alpha, otherwise a pixel marked
    # transparent would still paint colour from its surviving channels -- a
    # semi-transparent ghost that looks like real data.
    all_valid = np.ones(stack.shape[1:], dtype=bool)
    for i in channel_bands:
        all_valid &= masks[i]
    rgb[~all_valid] = 0
    alpha = np.where(all_valid, 255, 0).astype(np.uint8)
    return rgb, alpha, tuple(bounds)


def make_preview(
    ds: rasterio.DatasetReader,
    band_indices: Sequence[int],
    channel_names: Sequence[str],
    max_pixels: int = DEFAULT_MAX_PIXELS,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    stretch_mode: str = "per_band",
) -> PreviewResult:
    """Read (decimated) + stretch + composite in one call.

    `band_indices` are 1-based raster bands; `channel_names` label them for the
    UI (e.g. ["nir", "red", "green"] for a false-colour composite).
    """
    stack, transform = decimated_read(ds, band_indices, max_pixels=max_pixels)
    nodatas = [ds.nodatavals[i - 1] for i in band_indices]
    masks = [valid_mask(stack[k], nodatas[k]) for k in range(stack.shape[0])]
    stats = [band_stats(stack[k], nodatas[k], str(channel_names[k])) for k in range(stack.shape[0])]

    rgb, alpha, bounds = composite(
        stack, masks, channel_bands=(0, 1, 2),
        low_pct=low_pct, high_pct=high_pct, stretch_mode=stretch_mode,
    )

    all_valid = np.ones(stack.shape[1:], dtype=bool)
    for m in masks:
        all_valid &= m

    return PreviewResult(
        image=rgb,
        alpha=alpha,
        transform=transform,
        crs=ds.crs,
        display_shape=rgb.shape[:2],
        source_shape=(ds.height, ds.width),
        channels=tuple(channel_names),
        band_indices=tuple(band_indices),
        stretch_bounds=bounds,
        stretch_percentiles=(low_pct, high_pct),
        band_stats=stats,
        nodata_fraction=1.0 - float(all_valid.mean()),
        decimation_factor=float(ds.height) / float(rgb.shape[0]),
    )


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def encode_png(rgb: np.ndarray, alpha: Optional[np.ndarray] = None) -> bytes:
    """Encode an RGB(A) array as PNG bytes, without touching the filesystem."""
    from PIL import Image

    if alpha is None:
        img = Image.fromarray(rgb, mode="RGB")
    else:
        img = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def histogram(rgb_or_band: np.ndarray, bins: int = 64, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram of a single band / channel, valid pixels only."""
    data = rgb_or_band if mask is None else rgb_or_band[mask]
    data = np.asarray(data).ravel()
    if data.size == 0:
        return np.zeros(bins), np.arange(bins)
    counts, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return counts, centers


# --------------------------------------------------------------------------- #
# inspection helpers (used by the band inspector UI)
# --------------------------------------------------------------------------- #
def band_label(ds: rasterio.DatasetReader, band_index: int) -> str:
    desc = ds.descriptions[band_index - 1] if ds.descriptions else None
    return desc.strip() if desc else f"band {band_index}"


def all_band_stats(
    ds: rasterio.DatasetReader, max_pixels: int = 1_000_000
) -> List[Dict[str, Any]]:
    """Decimated statistics for EVERY band, for the band inspector table.

    One decimated read for all bands is far cheaper than one read per band,
    because GDAL fetches the same overview blocks once.
    """
    indices = list(range(1, ds.count + 1))
    stack, _ = decimated_read(ds, indices, max_pixels=max_pixels)
    rows = []
    for k, i in enumerate(indices):
        s = band_stats(stack[k], ds.nodatavals[i - 1], band_label(ds, i))
        s["band"] = i
        s["band_name"] = band_label(ds, i)
        rows.append(s)
    return rows


def band_histograms(
    ds: rasterio.DatasetReader,
    band_indices: Sequence[int],
    bins: int = 48,
    max_pixels: int = 1_000_000,
) -> List[Dict[str, Any]]:
    """Real histograms of valid pixel values, computed on the decimated read."""
    indices = list(band_indices)
    stack, _ = decimated_read(ds, indices, max_pixels=max_pixels)
    out: List[Dict[str, Any]] = []
    for k, i in enumerate(indices):
        nd = ds.nodatavals[i - 1]
        mask = valid_mask(stack[k], nd)
        vals = stack[k][mask].astype(np.float64)
        if vals.size:
            counts, edges = np.histogram(vals, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2.0
            p2, p50, p98 = (float(np.percentile(vals, p)) for p in (2, 50, 98))
        else:
            counts, centers = np.zeros(bins), np.arange(bins)
            p2 = p50 = p98 = 0.0
        out.append(
            {
                "band": i,
                "band_name": band_label(ds, i),
                "counts": [int(c) for c in counts],
                "centers": [round(float(c), 2) for c in centers],
                "p2": p2,
                "p50": p50,
                "p98": p98,
                "valid_pixels": int(vals.size),
            }
        )
    return out


# =========================================================================== #
# PHASE 3 -- rendering an analysis result (NDVI)
#
# Rendering NEVER touches the native analysis array: it works on a decimated
# copy and carries its own transform, so the statistical result and the picture
# can never silently disagree.
# =========================================================================== #
def decimate_array(
    array: np.ndarray,
    transform: Affine,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Tuple[np.ndarray, Affine, int]:
    """Block-average a float array for display, NaN-aware.

    Uses nanmean (valid pixels only) rather than nearest-neighbour sampling, so
    a display pixel summarises its block instead of picking one arbitrary
    source pixel. A display pixel is NaN only when its whole block was invalid.

    Returns (display_array, display_transform, factor).
    """
    h, w = array.shape
    factor = int(math.ceil(math.sqrt((h * w) / max_pixels))) if h * w > max_pixels else 1
    if factor <= 1:
        return array.astype(np.float32, copy=True), transform, 1

    hh = (h // factor) * factor
    ww = (w // factor) * factor
    block = array[:hh, :ww].astype(np.float32)
    valid = np.isfinite(block)

    summed = np.where(valid, block, 0.0).reshape(hh // factor, factor, ww // factor, factor).sum(axis=(1, 3))
    counted = valid.reshape(hh // factor, factor, ww // factor, factor).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counted > 0, summed / np.maximum(counted, 1), np.nan).astype(np.float32)

    t = transform
    display_transform = Affine(t.a * factor, t.b * factor, t.c,
                               t.d * factor, t.e * factor, t.f)
    return out, display_transform, factor


def ndvi_to_rgba(
    array: np.ndarray,
    mask: Optional[np.ndarray] = None,
    colormap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """Colour an NDVI array, leaving INVALID pixels fully transparent.

    Transparency (not black, and definitely not green) is the whole point: a
    nodata pixel painted dark green would read as "healthy vegetation".
    """
    import matplotlib

    cmap = matplotlib.colormaps.get(colormap) or matplotlib.colormaps["viridis"]
    cmap = cmap.copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))          # masked -> alpha 0

    arr = np.asarray(array, dtype=np.float32)
    invalid = ~np.isfinite(arr)
    if mask is not None:
        invalid = invalid | ~np.asarray(mask, dtype=bool)

    if vmax <= vmin:
        vmax = vmin + 1.0
    normalised = (arr - vmin) / (vmax - vmin)
    normalised = np.clip(normalised, 0.0, 1.0).astype(np.float32)

    masked = np.ma.masked_array(normalised, mask=invalid)
    rgba = (cmap(masked) * 255.0).astype(np.uint8)
    return rgba


def render_ndvi_figure(
    display_array: np.ndarray,
    display_mask: Optional[np.ndarray],
    *,
    title: str = "NDVI",
    subtitle: str = "",
    footer: str = "",
    colormap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> Any:
    """Build a matplotlib figure: NDVI map + colourbar + provenance footer.

    The figure states in words that this is NDVI and not an RGB image, because
    a red/green NDVI map is easy to misread as a false-colour photograph.
    """
    import matplotlib
    matplotlib.use("Agg")            # headless: no display in a server
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    rgba = ndvi_to_rgba(display_array, display_mask, colormap, vmin, vmax)
    cmap = matplotlib.colormaps.get(colormap) or matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(7.2, 7.0), dpi=110)
    ax.imshow(rgba, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    if subtitle:
        ax.text(0.0, -0.045, subtitle, transform=ax.transAxes, fontsize=8.5, color="#444444")

    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label("NDVI  (dimensionless, -1 to +1)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if footer:
        fig.text(0.02, 0.015, footer, fontsize=7.2, color="#333333", va="bottom")

    fig.tight_layout()
    return fig


def figure_to_png(fig: Any) -> bytes:
    """Serialise a matplotlib figure to PNG bytes (no temp files)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=110, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def ndvi_histogram(array: np.ndarray, mask: Optional[np.ndarray] = None, bins: int = 60) -> Dict[str, Any]:
    """Real histogram of VALID NDVI values across the full theoretical range."""
    arr = np.asarray(array, dtype=np.float32)
    mask = np.isfinite(arr) if mask is None else np.asarray(mask, dtype=bool)
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"counts": [], "centers": [], "valid_pixels": 0}
    counts, edges = np.histogram(vals, bins=bins, range=(-1.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "counts": [int(c) for c in counts],
        "centers": [round(float(c), 4) for c in centers],
        "valid_pixels": int(vals.size),
    }


def analysis_to_geotiff_bytes(
    array: np.ndarray,
    mask: Optional[np.ndarray],
    transform: Any,
    crs: Any,
    meta: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Export an analysis result as a real, georeferenced single-band GeoTIFF.

    float32 + NaN nodata, source CRS and transform preserved, provenance written
    into GDAL tags. This is a geospatial product, not a screenshot: it will
    re-open in QGIS in the right place on Earth.
    """
    from rasterio.io import MemoryFile

    meta = meta or {}
    arr = np.asarray(array, dtype=np.float32)
    if mask is not None:
        arr = np.where(np.asarray(mask, dtype=bool), arr, np.nan).astype(np.float32)

    tags: Dict[str, str] = {
        "satquery_analysis": str(meta.get("name", "analysis")),
        "satquery_label": str(meta.get("label", "")),
        "satquery_method": str(meta.get("method", "")),
        "satquery_nodata": "NaN",
    }
    refl = meta.get("reflectance") or {}
    if refl:
        tags["satquery_reflectance"] = str(refl.get("label", ""))
    bands_used = meta.get("bands_used") or {}
    if bands_used:
        tags["satquery_bands"] = ", ".join(f"{k}={v}" for k, v in bands_used.items())
    prov = meta.get("provenance") or {}
    for key, value in prov.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        tags[f"satquery_{key}"[:60]] = text[:500]

    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
            dtype="float32", crs=crs, transform=transform, nodata=float("nan"),
            compress="DEFLATE", predictor=3, tiled=True, blockxsize=256, blockysize=256,
        ) as ds:
            ds.write(arr, 1)
            ds.set_band_description(1, str(meta.get("label") or meta.get("name") or "analysis"))
            ds.update_tags(**tags)
        return mem.read()
