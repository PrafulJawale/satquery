"""Three-valued spatial mask algebra for Phase 9 (Checkpoint C).

This module is deliberately narrow: it turns ARRAYS INTO MASKS and combines
masks. It knows nothing about queries, Streamlit, or how a layer was fetched.

The central idea
----------------
A boolean mask cannot express "we do not know". Phase 8 already distinguishes
`INSUFFICIENT DATA` (class 0) from `UNSUITABLE` (class 1), and Phase 9 must not
destroy that distinction the moment two conditions are combined. So every mask
here is THREE-VALUED:

    TRUE          the condition is satisfied where data exists
    FALSE         the condition is not satisfied where data exists
    INSUFFICIENT  the required data is missing, so no claim is made

encoded as the pair (`match`, `valid`) -- INSUFFICIENT is simply `not valid` --
or as the integer codes below when combining. `FALSE` is never used as a
stand-in for "unknown": that would silently turn missing data into a
geographic claim.

Three-valued logic (strong Kleene), as specified for Phase 9:

    AND  TRUE  AND TRUE  = TRUE        OR  FALSE OR FALSE = FALSE
         TRUE  AND FALSE = FALSE           TRUE  OR FALSE = TRUE
         TRUE  AND NA    = NA              TRUE  OR NA    = TRUE
         FALSE AND NA    = FALSE           FALSE OR NA    = NA
                                       NOT TRUE  = FALSE
                                           FALSE = TRUE
                                           NA    = NA

`FALSE AND NA = FALSE` is the "one condition already fails, the unknown cannot
rescue it" rule; `FALSE OR NA = NA` is its dual. Both are tested explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .spatial_query import ConditionType, Operator

# --------------------------------------------------------------------------- #
# the three values
# --------------------------------------------------------------------------- #

#: Integer codes used when combining masks (uint8 arrays).
INSUFFICIENT: int = 0
FALSE: int = 1
TRUE: int = 2

STATE_LABELS: Dict[int, str] = {
    INSUFFICIENT: "insufficient data",
    FALSE: "does not match",
    TRUE: "matches",
}


class GridMismatchError(ValueError):
    """Raised when two masks do not share one analysis grid."""


class DistanceGridError(ValueError):
    """Raised when a distance cannot be computed in metres on this grid."""


# --------------------------------------------------------------------------- #
# the analysis grid
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Grid:
    """One analysis grid: the only grid on which masks may be combined.

    Mirrors `core.alignment.AnalysisGrid`, but with the transform normalised to
    a plain tuple so that equality is exact and testable.
    """

    crs: Any
    transform: Tuple[float, float, float, float, float, float]
    width: int
    height: int
    resolution_m: float
    requested_resolution: Optional[float] = None
    note: str = ""
    source_resolutions: Mapping[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_analysis_grid(cls, grid: Any, **extra: Any) -> "Grid":
        """Build from a Phase 8 `AnalysisGrid`."""
        transform = grid.transform
        # An Affine iterates as (a, b, c, d, e, f). `to_gdal()` would return
        # (c, a, b, f, d, e) -- GDAL order -- which would read the ORIGIN as
        # the pixel size, so it is never used here.
        transform = tuple(float(v) for v in tuple(transform))
        return cls(
            crs=grid.crs,
            transform=transform,  # type: ignore[arg-type]
            width=int(grid.width),
            height=int(grid.height),
            resolution_m=float(grid.resolution),
            requested_resolution=getattr(grid, "requested_resolution", None),
            note=str(getattr(grid, "note", "") or ""),
            source_resolutions=dict(extra.pop("source_resolutions", {}) or {}),
            **extra,
        )

    # -- geometry ----------------------------------------------------------- #
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_width_m(self) -> float:
        """|a| -- the east-west size of one cell, in CRS units."""
        return abs(float(self.transform[0]))

    @property
    def pixel_height_m(self) -> float:
        """|e| -- the north-south size of one cell, in CRS units."""
        return abs(float(self.transform[4]))

    @property
    def pixel_area_m2(self) -> float:
        """Area of one cell from the transform itself -- never assumed 10x10."""
        return self.pixel_width_m * self.pixel_height_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crs": str(self.crs),
            "transform": [float(v) for v in self.transform],
            "width": self.width,
            "height": self.height,
            "pixel_size_m": [self.pixel_width_m, self.pixel_height_m],
            "cell_area_m2": self.pixel_area_m2,
            "resolution_m": self.resolution_m,
            "requested_resolution_m": self.requested_resolution,
            "note": self.note,
            "source_resolutions": dict(self.source_resolutions),
        }

    # -- compatibility ------------------------------------------------------ #
    def mismatch_reason(self, other: "Grid") -> Optional[str]:
        """Why `other` may not be combined with this grid (None = compatible)."""
        if self.width != other.width or self.height != other.height:
            return (f"dimensions differ: {self.width}x{self.height} vs "
                    f"{other.width}x{other.height}")
        if tuple(self.transform) != tuple(other.transform):
            return (f"transform differs: {tuple(self.transform)} vs "
                    f"{tuple(other.transform)}")
        if not _same_crs(self.crs, other.crs):
            return f"CRS differs: {self.crs} vs {other.crs}"
        if not np.isclose(self.pixel_width_m, other.pixel_width_m,
                          rtol=1e-9, atol=1e-9):
            return (f"pixel width differs: {self.pixel_width_m} vs "
                    f"{other.pixel_width_m}")
        if not np.isclose(self.pixel_height_m, other.pixel_height_m,
                          rtol=1e-9, atol=1e-9):
            return (f"pixel height differs: {self.pixel_height_m} vs "
                    f"{other.pixel_height_m}")
        return None

    def is_compatible_with(self, other: "Grid") -> bool:
        return self.mismatch_reason(other) is None

    def require_compatible(self, other: "Grid") -> None:
        """Raise `GridMismatchError` unless the two grids are the same grid."""
        reason = self.mismatch_reason(other)
        if reason is not None:
            raise GridMismatchError(
                "Cannot combine spatial conditions computed on different "
                f"analysis grids: {reason}.")


def _same_crs(a: Any, b: Any) -> bool:
    """CRS equality that tolerates the many spellings of the same CRS."""
    if a is b:
        return True
    if str(a) == str(b):
        return True
    try:
        from pyproj import CRS
        return CRS.from_user_input(a).equals(CRS.from_user_input(b))
    except Exception:                                   # pragma: no cover
        return False


# --------------------------------------------------------------------------- #
# the mask
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GridMask:
    """A three-valued condition evaluated on one `Grid`.

    `match`  -- True where the condition holds (only meaningful where `valid`)
    `valid`  -- True where the underlying data actually exists
    """

    grid: Grid
    name: str
    source: str
    match: Any                       # np.ndarray[bool]
    valid: Any                       # np.ndarray[bool]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        match = np.asarray(self.match, dtype=bool)
        valid = np.asarray(self.valid, dtype=bool)
        if match.shape != tuple(self.grid.shape):
            raise ValueError(
                f"mask '{self.name}' has shape {match.shape} but the grid is "
                f"{self.grid.shape}")
        if valid.shape != match.shape:
            raise ValueError(
                f"mask '{self.name}': 'valid' shape {valid.shape} does not "
                f"match 'match' shape {match.shape}")
        object.__setattr__(self, "match", match)
        object.__setattr__(self, "valid", valid)
        # a cell cannot match where there is no data
        object.__setattr__(self, "match", match & valid)

    # -- three-valued view -------------------------------------------------- #
    @property
    def state(self) -> Any:
        """uint8 codes: 2 = TRUE, 1 = FALSE, 0 = INSUFFICIENT."""
        out = np.zeros(self.valid.shape, dtype=np.uint8)
        out[self.valid] = 1
        out[self.valid & self.match] = 2
        return out

    @classmethod
    def from_state(cls, state: Any, **kwargs: Any) -> "GridMask":
        state = np.asarray(state)
        return cls(match=(state == TRUE), valid=(state > INSUFFICIENT), **kwargs)

    # -- counting ----------------------------------------------------------- #
    @property
    def n_true(self) -> int:
        return int(np.count_nonzero(self.valid & self.match))

    @property
    def n_false(self) -> int:
        return int(np.count_nonzero(self.valid & ~self.match))

    @property
    def n_insufficient(self) -> int:
        return int(np.count_nonzero(~self.valid))

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def n_cells(self) -> int:
        return int(self.valid.size)

    def area_m2(self) -> float:
        """Matched area from the real cell area -- not from an assumed 10x10."""
        return float(self.n_true) * float(self.grid.pixel_area_m2)

    def counts(self, scope: Any = None) -> Dict[str, Any]:
        """Counts inside `scope` (normally the ROI).

        Cells OUTSIDE the scope are excluded from all three counters: they were
        never part of the question, so they must not be reported as
        "insufficient" -- that would inflate the unknown with geometry we
        deliberately ignored.
        """
        valid, match = self.valid, self.match
        if scope is None:
            scope = np.ones(valid.shape, dtype=bool)
        else:
            scope = np.asarray(scope, dtype=bool)
        valid = valid & scope
        match = match & scope
        n_true = int(np.count_nonzero(valid & match))
        n_false = int(np.count_nonzero(valid & ~match))
        n_ins = int(np.count_nonzero(scope & ~valid))
        n_valid = n_true + n_false
        return {
            "cells_analysed": n_valid,
            "cells_in_scope": int(np.count_nonzero(scope)),
            "matching_cells": n_true,
            "non_matching_cells": n_false,
            "insufficient_cells": n_ins,
            "valid_cells": n_valid,
            "matched_fraction": (float(n_true) / float(n_valid)
                                 if n_valid else 0.0),
            "insufficient_fraction": (float(n_ins) / float(n_valid + n_ins)
                                      if (n_valid + n_ins) else 0.0),
            "matched_area_m2": float(n_true) * float(self.grid.pixel_area_m2),
            "cell_area_m2": float(self.grid.pixel_area_m2),
        }

    def to_dict(self) -> Dict[str, Any]:
        counts = self.counts()
        return {
            "name": self.name,
            "source": self.source,
            "grid": self.grid.to_dict(),
            "counts": counts,
            "provenance": dict(self.provenance),
        }


# --------------------------------------------------------------------------- #
# combination
# --------------------------------------------------------------------------- #


def require_compatible(masks: Sequence[GridMask]) -> Grid:
    """One shared grid, or a `GridMismatchError`. Never silently combines."""
    if not masks:
        raise ValueError("require_compatible() needs at least one mask")
    grid = masks[0].grid
    for other in masks[1:]:
        grid.require_compatible(other.grid)
    return grid


def negate(mask: GridMask) -> GridMask:
    """NOT with three-valued semantics: NA stays NA."""
    return GridMask(
        grid=mask.grid,
        name=f"NOT {mask.name}",
        source=mask.source,
        match=~mask.match & mask.valid,
        valid=mask.valid,
        provenance={"negated_from": mask.name},
    )


def combine_all(masks: Sequence[GridMask],
                operator: Operator = Operator.AND) -> GridMask:
    """AND / OR over any number of masks, three-valued.

    AND: FALSE if any input is FALSE; otherwise NA if any input is NA;
         otherwise TRUE.
    OR:  TRUE if any input is TRUE; otherwise NA if any input is NA;
         otherwise FALSE.
    """
    if not masks:
        raise ValueError("combine_all() needs at least one mask")
    grid = require_compatible(masks)
    if len(masks) == 1:
        single = masks[0]
        if operator is Operator.AND:
            return single
        return single

    stacked = np.stack([m.state for m in masks], axis=0)
    if operator is Operator.AND:
        any_false = np.any(stacked == FALSE, axis=0)
        any_insufficient = np.any(stacked == INSUFFICIENT, axis=0)
        state = np.where(any_false, FALSE,
                         np.where(any_insufficient, INSUFFICIENT, TRUE))
        symbol = " AND "
    elif operator is Operator.OR:
        any_true = np.any(stacked == TRUE, axis=0)
        any_insufficient = np.any(stacked == INSUFFICIENT, axis=0)
        state = np.where(any_true, TRUE,
                         np.where(any_insufficient, INSUFFICIENT, FALSE))
        symbol = " OR "
    else:                                            # pragma: no cover
        raise ValueError(f"unsupported operator: {operator!r}")

    return GridMask.from_state(
        state.astype(np.uint8),
        grid=grid,
        name=symbol.join(m.name for m in masks),
        source="combination",
        provenance={"operator": operator.value,
                    "inputs": [m.name for m in masks]},
    )


def crop_mask(mask: GridMask, target: Grid) -> GridMask:
    """Slice a mask computed on a LARGER, identically aligned grid back to
    `target`.

    Used when a condition needs a buffered source window (proximity) but must
    be combined on the ROI's own grid. The two grids must share CRS, cell size
    and alignment to whole cells -- anything else is a GridMismatchError, never
    a silent resample.
    """
    source = mask.grid
    if not _same_crs(source.crs, target.crs):
        raise GridMismatchError(
            f"cannot crop: CRS differs ({source.crs} vs {target.crs})")
    for axis, mine, theirs in (("width", source.pixel_width_m,
                                target.pixel_width_m),
                               ("height", source.pixel_height_m,
                                target.pixel_height_m)):
        if not np.isclose(mine, theirs, rtol=1e-9, atol=1e-9):
            raise GridMismatchError(
                f"cannot crop: cell {axis} differs ({mine} vs {theirs})")
    if abs(float(source.transform[1])) > 1e-9 or abs(float(target.transform[1])) > 1e-9 \
            or abs(float(source.transform[3])) > 1e-9 \
            or abs(float(target.transform[3])) > 1e-9:
        raise GridMismatchError("cannot crop: rotated grids are not supported")

    # offsets in whole cells, from the two origins
    col = (float(target.transform[2]) - float(source.transform[2])) \
        / float(source.transform[0])
    row = (float(target.transform[5]) - float(source.transform[5])) \
        / float(source.transform[4])
    col0, row0 = int(round(col)), int(round(row))
    if abs(col - col0) > 1e-6 or abs(row - row0) > 1e-6:
        raise GridMismatchError(
            f"cannot crop: the grids are not aligned to whole cells "
            f"(offset {col:.6f}, {row:.6f} cells)")
    if col0 < 0 or row0 < 0 or col0 + target.width > source.width \
            or row0 + target.height > source.height:
        raise GridMismatchError(
            f"cannot crop: the target grid is not contained in the source "
            f"window (offset {col0},{row0}; target {target.width}x"
            f"{target.height}; source {source.width}x{source.height})")

    rows = slice(row0, row0 + target.height)
    cols = slice(col0, col0 + target.width)
    return GridMask(
        grid=target, name=mask.name, source=mask.source,
        match=mask.match[rows, cols], valid=mask.valid[rows, cols],
        provenance={**dict(mask.provenance),
                    "cropped_from": [source.height, source.width],
                    "crop_offset_cells": [row0, col0]},
    )


def apply_roi(mask: GridMask, inside: Any) -> GridMask:
    """Cells outside the ROI become INSUFFICIENT (they were never analysed)."""
    inside = np.asarray(inside, dtype=bool)
    if inside.shape != mask.valid.shape:
        raise ValueError(
            f"ROI mask shape {inside.shape} does not match the grid "
            f"{mask.valid.shape}")
    return GridMask(
        grid=mask.grid,
        name=mask.name,
        source=mask.source,
        match=mask.match & inside,
        valid=mask.valid & inside,
        provenance={**dict(mask.provenance), "clipped_to_roi": True},
    )


# --------------------------------------------------------------------------- #
# condition -> mask
# --------------------------------------------------------------------------- #


def mask_from_class_raster(codes: Any, grid: Grid, *, min_class: int,
                           name: str, source: str,
                           insufficient_code: int = 0,
                           valid_min: int = 1,
                           **provenance: Any) -> GridMask:
    """A suitability/categorical raster -> "class >= min_class".

    Cells below `valid_min` (Phase 8 class 0 = INSUFFICIENT DATA, and NaN) stay
    INSUFFICIENT rather than becoming FALSE.
    """
    codes = np.asarray(codes)
    finite = np.isfinite(codes) if codes.dtype.kind == "f" else np.ones(
        codes.shape, dtype=bool)
    valid = finite & (codes >= valid_min)
    match = valid & (codes >= int(min_class))
    return GridMask(
        grid=grid, name=name, source=source, match=match, valid=valid,
        provenance={"min_class": int(min_class),
                    "insufficient_code": int(insufficient_code),
                    **provenance},
    )


def water_mask_from_land_cover(classes: Any, grid: Grid, *,
                               water_class: int = 80,
                               name: Optional[str] = None,
                               source: str = "worldcover",
                               **provenance: Any) -> GridMask:
    """WATER: the cell itself is mapped permanent water (WorldCover class 80).

    Wetland (class 90) is a different class and is deliberately NOT water.
    """
    classes = np.asarray(classes)
    finite = (np.isfinite(classes) if classes.dtype.kind == "f"
              else np.ones(classes.shape, dtype=bool))
    valid = finite & (classes > 0)
    match = valid & (classes == int(water_class))
    return GridMask(
        grid=grid,
        name=name or f"water(class {int(water_class)})",
        source=source,
        match=match,
        valid=valid,
        provenance={"water_class": int(water_class),
                    "wetland_class_excluded": 90,
                    **provenance},
    )


def land_cover_mask(classes: Any, grid: Grid, *, classes_wanted: Sequence[int],
                    name: Optional[str] = None,
                    source: str = "worldcover",
                    **provenance: Any) -> GridMask:
    """"is one of these WorldCover classes"."""
    classes = np.asarray(classes)
    finite = (np.isfinite(classes) if classes.dtype.kind == "f"
              else np.ones(classes.shape, dtype=bool))
    valid = finite & (classes > 0)
    wanted = [int(c) for c in classes_wanted]
    match = valid & np.isin(classes, wanted)
    return GridMask(
        grid=grid,
        name=name or f"land_cover({wanted})",
        source=source,
        match=match,
        valid=valid,
        provenance={"classes": wanted, **provenance},
    )


def proximity_mask(water: GridMask, distance_m: float, *,
                   name: Optional[str] = None) -> GridMask:
    """WATER_PROXIMITY: distance to the nearest mapped water cell <= N metres.

    Distance is computed with `scipy.ndimage.distance_transform_edt` on the
    PROJECTED analysis grid, with the true cell size passed as `sampling`, so
    the result is in metres -- not in degrees and not in pixel counts.

    Guards (each raises `DistanceGridError` rather than returning a wrong map):
      * the CRS must be projected -- degrees are not metres;
      * the cells must be square -- an anisotropic grid cannot use one scalar
        distance per axis without lying about one direction.
    """
    grid = water.grid
    if _is_geographic(grid.crs):
        raise DistanceGridError(
            f"Distance cannot be computed in degrees: the analysis grid is "
            f"geographic ({grid.crs}). Reproject to a projected CRS first.")
    width_m, height_m = grid.pixel_width_m, grid.pixel_height_m
    if not np.isclose(width_m, height_m, rtol=1e-6, atol=1e-9):
        raise DistanceGridError(
            f"Distance needs square cells: this grid is {width_m} m x "
            f"{height_m} m, so a single scalar distance would be wrong in one "
            "direction.")
    if width_m <= 0 or height_m <= 0:
        raise DistanceGridError("The analysis grid has a non-positive cell size.")

    distance = float(distance_m)
    if distance < 0:
        raise DistanceGridError(f"Distance must be non-negative, got {distance}.")

    from scipy.ndimage import distance_transform_edt     # local: see note below

    source = water.match                     # class-80 cells; NA is not water
    known = water.valid                      # where land cover actually exists
    n_water = int(np.count_nonzero(source))

    # Distance TO the nearest mapped water cell, in metres, for every cell.
    # With no water inside the window there is nothing to measure from.
    if n_water:
        d_water = distance_transform_edt(~source, sampling=(height_m, width_m))
    else:
        d_water = np.full(source.shape, np.inf, dtype=float)

    # Distance to the nearest cell whose land cover is UNKNOWN, or to the edge
    # of the window. Water may exist beyond either, so a cell can only be
    # declared "farther than N m" if even the far side of that gap exceeds N.
    padded = np.pad(known, 1, mode="constant", constant_values=False)
    d_edge = distance_transform_edt(padded, sampling=(height_m, width_m))[1:-1, 1:-1]

    within = d_water <= distance
    # Not within N m of known water, but the unknown region is close enough
    # that water could be within N m -> the answer is unknown, not "far".
    undecidable = (~within) & (d_edge <= distance)

    return GridMask(
        grid=grid,
        name=name or f"water_proximity(<= {_fmt_metres(distance)} m)",
        source="derived:worldcover-class-80",
        match=within & known,
        valid=known & ~undecidable,
        provenance={
            "distance_m": distance,
            "method": "scipy.ndimage.distance_transform_edt on the projected "
                      "analysis grid, distance between cell centres, in metres",
            "pixel_size_m": [width_m, height_m],
            "source_water_class": water.provenance.get("water_class", 80),
            "water_cells_found": n_water,
            "edge_rule": "a cell is only FALSE when the nearest unknown cell or "
                         "the window edge is farther away than the threshold; "
                         "otherwise the answer is INSUFFICIENT, because water "
                         "just outside the window cannot be ruled out",
            "undecidable_cells": int(np.count_nonzero(undecidable & known)),
        },
    )


def _is_geographic(crs: Any) -> bool:
    """True when the CRS is lat/lon (so its units are degrees, not metres)."""
    try:
        from pyproj import CRS
        return bool(CRS.from_user_input(crs).is_geographic)
    except Exception:                                   # pragma: no cover
        return False                                    # unknown: do not block


def _fmt_metres(value: float) -> str:
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


__all__ = [
    "TRUE", "FALSE", "INSUFFICIENT", "STATE_LABELS",
    "GridMismatchError", "DistanceGridError",
    "Grid", "GridMask",
    "require_compatible", "negate", "combine_all", "apply_roi", "crop_mask",
    "mask_from_class_raster", "water_mask_from_land_cover", "land_cover_mask",
    "proximity_mask",
]
