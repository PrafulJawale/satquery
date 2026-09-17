"""core/roi.py -- region-of-interest selection (PHASE 5).

WHAT THIS MODULE DECIDES
------------------------
A shape drawn in the browser is only a *candidate*. This module answers, for
that candidate:

    is it an area at all?
    is it usable (valid, non-empty, non-degenerate)?
    does it touch the raster?
    if it hangs off the edge, what part is inside?
    how big is the usable part, in m2 / ha / km2?

It never touches pixels -- that is Phase 6 (zonal statistics). It produces
geometry that Phase 6 can rasterise directly with
`rasterio.features.geometry_mask(..., transform=ds.transform)`.

THE ONE-WAY RULE (same as Phase 4)
----------------------------------
    drawn GeoJSON (EPSG:4326)  ->  validate  ->  transform to raster CRS
                               ->  intersect with the footprint (raster CRS)
                               ->  ROISelection

The original drawn geometry is kept for display and provenance; the raster-CRS
geometry is what analysis will use. Nothing here computes statistics.

STATE CONTRACT (verified against streamlit-folium 0.27.4 frontend)
------------------------------------------------------------------
`all_drawings` is `window.drawnItems.toGeoJSON().features`, recomputed on every
draw:created / draw:edited / draw:deleted event. Therefore:

    all_drawings is None -> the map component has not reported any drawing
                            activity (first mount, or remount after its HTML
                            changed). KEEP the existing selection.
    all_drawings == []   -> reported, and nothing is left on the map: CLEAR.
    all_drawings == [..] -> reported: use it (last shape = most recent).

`last_active_drawing` is deliberately NOT used as the source of truth: on a
delete event it still holds the *deleted* shape, which is exactly how a stale
ROI survives.

Pure Python. No Streamlit: the state helpers take and return a plain dict, which
is all `st.session_state` needs to be.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .geo import MissingCRSError
from .geometry import (
    CRS84,
    GeometryError,
    area_m2,
    crs_label,
    ensure_polygonal,
    format_area,
    geojson_of,
    parse_geometry,
    polygonal_parts,
    repair,
    transform_geometry,
)

EMPTY_MESSAGE = "Draw a rectangle or polygon on the map to select an area for analysis."
OUTSIDE_MESSAGE = "Analysis data is not available for this selected area."
# Kept for callers that reported the old wording; the map is global now, so
# a selection can sit anywhere on Earth and the honest answer is about the
# *analysis data*, not about a raster the user never chose.
CLIPPED_MESSAGE = (
    "Only the portion inside the available raster will be used for later analysis."
)
INVALID_MESSAGE = "Selection could not be used. Please draw a valid polygon or rectangle."

_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #
@dataclass
class ROISelection:
    """A validated (or rejected) user selection, ready for Phase 6.

    Shapely geometries are kept in memory for analysis; `to_dict()` returns a
    JSON-safe view for display, provenance and session state.
    """

    source: str = "user_drawn"
    geometry_type: str = ""
    num_parts: int = 0
    is_valid: bool = False
    intersects_raster: bool = False
    was_clipped: bool = False
    overlap_fraction: float = 0.0
    area_m2: float = 0.0
    area_hectares: float = 0.0
    area_km2: float = 0.0
    area_method: str = ""
    original_area_m2: float = 0.0
    raster_crs: str = ""
    original_geometry: Optional[BaseGeometry] = None      # EPSG:4326, as drawn
    geometry_raster_crs: Optional[BaseGeometry] = None    # usable part, raster CRS
    drawn_count: int = 0
    message: str = ""
    warnings: Tuple[str, ...] = ()

    # -- convenience ------------------------------------------------------- #
    @property
    def usable(self) -> bool:
        """True when there is geometry Phase 6 may analyse."""
        return bool(self.is_valid and self.intersects_raster and self.area_m2 > 0)

    @property
    def original_geojson(self) -> Optional[Dict[str, Any]]:
        return geojson_of(self.original_geometry)

    @property
    def clipped_geojson(self) -> Optional[Dict[str, Any]]:
        return geojson_of(self.geometry_raster_crs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "geometry_type": self.geometry_type,
            "num_parts": self.num_parts,
            "is_valid": self.is_valid,
            "intersects_raster": self.intersects_raster,
            "was_clipped": self.was_clipped,
            "overlap_fraction": self.overlap_fraction,
            "area_m2": self.area_m2,
            "area_hectares": self.area_hectares,
            "area_km2": self.area_km2,
            "area_method": self.area_method,
            "area_display": format_area(self.area_m2),
            "original_area_m2": self.original_area_m2,
            "original_area_display": format_area(self.original_area_m2),
            "raster_crs": self.raster_crs,
            "drawn_count": self.drawn_count,
            "message": self.message,
            "warnings": list(self.warnings),
            "original_geometry": self.original_geojson,
            "geometry_raster_crs": self.clipped_geojson,
        }


def _rejected(message: str, warnings: Sequence[str] = (), drawn_count: int = 0,
              raster_crs: str = "") -> ROISelection:
    return ROISelection(
        is_valid=False,
        intersects_raster=False,
        raster_crs=raster_crs,
        drawn_count=drawn_count,
        message=message,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# picking the active drawing
# --------------------------------------------------------------------------- #
def pick_active_drawing(
    drawings: Optional[Sequence[Any]],
) -> Tuple[Optional[Any], int]:
    """(most recent polygonal drawing, number of polygonal drawings).

    The last item wins: Leaflet's `drawnItems` keeps insertion order, so the
    newest shape is last. Deleting a shape removes it from the list, which is
    what makes replacement and deletion work without any delta tracking.
    """
    if not drawings:
        return None, 0
    polygonal: List[Any] = []
    for feature in drawings:
        try:
            geom = parse_geometry(feature)
        except GeometryError:
            continue
        if isinstance(geom, (Polygon, MultiPolygon)):
            polygonal.append(feature)
    if not polygonal:
        return None, 0
    return polygonal[-1], len(polygonal)


def drawing_signature(drawings: Optional[Sequence[Any]], raster_key: str = "") -> str:
    """Stable fingerprint of (which raster, which drawings).

    Used so that a Streamlit rerun triggered by an unrelated widget does not
    recompute -- or worse, silently drop -- the selection.
    """
    payload = json.dumps(
        {"raster": raster_key, "drawings": list(drawings or [])},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# the core decision
# --------------------------------------------------------------------------- #
def select_roi(
    drawings: Optional[Sequence[Any]],
    footprint_raster_crs: BaseGeometry,
    raster_crs: Any,
    raster_key: str = "",
    min_area_m2: float = 0.0,
) -> Optional[ROISelection]:
    """Turn the current drawings into an ROISelection (or None if nothing drawn).

    `footprint_raster_crs` must already be in `raster_crs` (use
    `core.geo.native_footprint`) -- the intersection is a true polygon
    intersection, never a bounding-box test, and never in degrees.
    """
    if raster_crs is None:
        raise MissingCRSError(
            "Raster has no CRS; a selection cannot be transformed or measured."
        )
    if footprint_raster_crs is None or footprint_raster_crs.is_empty:
        raise GeometryError("Raster footprint is empty; a selection cannot be checked.")

    feature, drawn_count = pick_active_drawing(drawings)
    if feature is None:
        if drawings:
            # Something is on the map but none of it is an area (would only
            # happen with tools we have disabled). Say so instead of silently
            # falling back to "nothing selected".
            return _rejected(
                "That drawing is not an area. Draw a rectangle or a polygon.",
                raster_crs=crs_label(raster_crs),
                drawn_count=len(drawings),
            )
        return None                      # nothing drawn, or everything deleted

    warnings: List[str] = []
    if drawn_count > 1:
        warnings.append(
            f"{drawn_count} shapes are drawn — only the most recent is used. "
            "Delete the others to avoid ambiguity."
        )

    crs_name = crs_label(raster_crs)

    # 1. parse + type gate ------------------------------------------------- #
    try:
        geom_wgs84 = ensure_polygonal(parse_geometry(feature))
    except GeometryError as exc:
        return _rejected(str(exc), warnings, drawn_count, crs_name)

    # 2. validity ---------------------------------------------------------- #
    try:
        geom_wgs84, repaired = repair(geom_wgs84)
        if repaired:
            warnings.append(
                "The drawn shape was self-intersecting and has been repaired "
                "(shapely make_valid)."
            )
    except GeometryError as exc:
        return _rejected(str(exc), warnings, drawn_count, crs_name)

    if geom_wgs84.is_empty or geom_wgs84.area <= 0:
        return _rejected(
            "The drawn shape encloses no area.", warnings, drawn_count, crs_name
        )

    # 3. into the raster CRS ----------------------------------------------- #
    geom_raster = transform_geometry(geom_wgs84, CRS84, raster_crs)

    # 4. intersect with the footprint (true polygon intersection) ---------- #
    inter = geom_raster.intersection(footprint_raster_crs)
    parts = polygonal_parts(inter)
    if not parts:
        return ROISelection(
            source="user_drawn",
            geometry_type=geom_wgs84.geom_type,
            num_parts=len(polygonal_parts(geom_wgs84)),
            is_valid=True,
            intersects_raster=False,
            raster_crs=crs_name,
            original_geometry=geom_wgs84,
            drawn_count=drawn_count,
            message=OUTSIDE_MESSAGE,
            warnings=tuple(warnings),
        )

    usable = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    area_usable, method = area_m2(usable, raster_crs)
    area_original, _ = area_m2(geom_raster, raster_crs)

    if area_usable <= 0 or (min_area_m2 and area_usable < min_area_m2):
        return _rejected(
            "The part of the selection inside the raster is too small to analyse.",
            warnings,
            drawn_count,
            crs_name,
        )

    overlap = (area_usable / area_original) if area_original > 0 else 0.0
    was_clipped = area_usable < area_original * (1.0 - 1e-9)

    if was_clipped:
        message = CLIPPED_MESSAGE
        warnings = list(warnings) + [
            f"{overlap * 100:.1f}% of the drawn area is inside the raster "
            f"({format_area(area_original)} drawn → {format_area(area_usable)} usable)."
        ]
    else:
        message = "Selection is fully inside the raster extent."

    return ROISelection(
        source="user_drawn",
        geometry_type=geom_wgs84.geom_type,
        num_parts=len(polygonal_parts(usable)),
        is_valid=True,
        intersects_raster=True,
        was_clipped=was_clipped,
        overlap_fraction=float(overlap),
        area_m2=float(area_usable),
        area_hectares=float(area_usable / 10_000.0),
        area_km2=float(area_usable / 1_000_000.0),
        area_method=method,
        original_area_m2=float(area_original),
        raster_crs=crs_name,
        original_geometry=geom_wgs84,
        geometry_raster_crs=usable,
        drawn_count=drawn_count,
        message=message,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# session-state machine (Streamlit-free: state is just a dict)
# --------------------------------------------------------------------------- #
ROI_SIG = "roi_signature"
ROI_OBJ = "roi"
ROI_STALE = "roi_map_stale"


def update_roi_state(
    state: Dict[str, Any],
    drawings: Optional[Sequence[Any]],
    footprint_raster_crs: BaseGeometry,
    raster_crs: Any,
    raster_key: str = "",
    min_area_m2: float = 0.0,
) -> Optional[ROISelection]:
    """Recompute the selection only when the input actually changed.

    Returns the current ROISelection (or None) and stores it in `state`.

        drawings is None  -> component has not reported: keep, flag as stale-map
        drawings == []    -> reported, nothing on the map: CLEAR (no stale ROI)
        drawings == [...] -> reported: use it
        unchanged         -> keep (no recompute, no flicker)
    """
    if drawings is None:
        # The map was (re)mounted and has not sent drawing data yet. Keeping the
        # selection is safe; the UI warns that the outline may be gone.
        state[ROI_STALE] = state.get(ROI_OBJ) is not None
        return state.get(ROI_OBJ)

    state[ROI_STALE] = False
    signature = drawing_signature(drawings, raster_key)
    if state.get(ROI_SIG) == signature and ROI_OBJ in state:
        return state[ROI_OBJ]

    selection = select_roi(
        drawings,
        footprint_raster_crs,
        raster_crs,
        raster_key=raster_key,
        min_area_m2=min_area_m2,
    )
    state[ROI_SIG] = signature
    state[ROI_OBJ] = selection
    return selection


def clear_roi_state(state: Dict[str, Any]) -> None:
    """Forget the selection completely (used by the Clear-selection button)."""
    state[ROI_SIG] = None
    state[ROI_OBJ] = None
    state[ROI_STALE] = False


def is_map_stale(state: Dict[str, Any]) -> bool:
    return bool(state.get(ROI_STALE))
