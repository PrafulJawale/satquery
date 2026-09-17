"""Phase 13 -- the per-condition masks behind an evidence record.

This module is READ-ONLY with respect to the engines. Phase 9 and Phase 12
already keep the individual masks on their result -- as a `GridMask` on each
`condition_results` entry and as state codes in `condition_masks` -- so reading
them back here means Phase 13 adds per-condition inspection layers WITHOUT
re-running or re-implementing any Phase 9/10/12 mask algebra: the layer that is
drawn is the mask that produced the numbers.

A temporal result carries neither, so its change classes are derived from the
very `class_raster` Phase 10 computed, using the public class codes
(0 = insufficient, and NEVER a match).

Nothing here decides anything. If a mask is missing for a condition, that
condition simply has no layer.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.spatial import Grid, GridMask

__all__ = ["evidence_masks", "mask_for_record"]


def evidence_masks(result: Any) -> Dict[str, GridMask]:
    """Per-condition masks of a result, keyed by the condition NAME.

    The keys match the `condition` field of the Phase 13 evidence records, so
    a record and its layer can be paired without guessing.
    """
    if result is None:
        return {}

    masks = _masks_from_condition_results(result)
    if masks:
        return masks
    return _masks_from_state_codes(result) or _temporal_class_masks(result)


def _masks_from_condition_results(result: Any) -> Dict[str, GridMask]:
    """The live `GridMask` objects Phase 9/12 keep on each condition entry."""
    out: Dict[str, GridMask] = {}
    for entry in getattr(result, "condition_results", ()) or ():
        if not isinstance(entry, dict):
            continue
        mask = entry.get("mask")
        name = str(entry.get("name") or getattr(mask, "name", "") or "")
        if isinstance(mask, GridMask) and name:
            out[name] = mask
    return out


def _masks_from_state_codes(result: Any) -> Dict[str, GridMask]:
    """Rebuild masks from `condition_masks`, which stores state codes.

    The codes are Phase 9's own (2 TRUE / 1 FALSE / 0 INSUFFICIENT), so
    `GridMask.from_state` restores exactly the engine's three-valued mask --
    no threshold is re-applied and no pixel is re-classified.
    """
    stored = getattr(result, "condition_masks", None)
    if not isinstance(stored, dict) or not stored:
        return {}
    grid = _grid_of(result)
    if grid is None:
        return {}
    out: Dict[str, GridMask] = {}
    for name, state in stored.items():
        array = np.asarray(state)
        if array.ndim != 2 or array.shape != tuple(grid.shape):
            continue        # a mask from another grid is not this answer's mask
        try:
            out[str(name)] = GridMask.from_state(
                array, grid=grid, name=str(name),
                source=str(getattr(result, "source_analyses", "") or "engine"))
        except Exception:
            continue
    return out


def _grid_of(result: Any) -> Any:
    """The analysis `Grid`, rebuilt from the result's own grid dictionary."""
    grid = getattr(result, "grid", None)
    if grid is None:
        return None
    if isinstance(grid, Grid):
        return grid
    if not isinstance(grid, dict) or not grid.get("width"):
        return None
    try:
        return Grid(crs=grid.get("crs"),
                    transform=tuple(float(v) for v in grid["transform"]),
                    width=int(grid["width"]), height=int(grid["height"]),
                    resolution_m=float(grid.get("resolution_m") or 0.0))
    except Exception:
        return None


def _temporal_class_masks(result: Any) -> Dict[str, GridMask]:
    """One mask per NDVI change class, from the class raster Phase 10 made."""
    classes = getattr(result, "class_raster", None)
    if classes is None:
        return {}
    try:  # pragma: no cover - the codes are part of the Phase 10 contract
        from analyses.ndvi_change import (CHANGE_DECREASE, CHANGE_INCREASE,
                                          CHANGE_STABLE)
    except Exception:
        return {}

    array = np.asarray(classes)
    if array.ndim != 2:
        return {}
    roi = getattr(result, "roi_mask", None)
    inside = (np.ones(array.shape, dtype=bool) if roi is None
              else np.asarray(roi, dtype=bool))
    if inside.shape != array.shape:
        inside = np.ones(array.shape, dtype=bool)

    transform = getattr(result, "transform", None)
    crs = getattr(result, "crs", None)
    if transform is None or crs is None:
        return {}
    try:
        grid = Grid(crs=crs,
                    transform=tuple(float(v) for v in transform),
                    width=int(array.shape[1]),
                    height=int(array.shape[0]),
                    resolution_m=float(getattr(result, "resolution", 10.0) or 10.0))
    except Exception:
        return {}

    # Code 0 means "insufficient data": it is never a match and never valid.
    valid = inside & (array > 0)
    source = "analyses.ndvi_change"
    out: Dict[str, GridMask] = {}
    for code, name in ((CHANGE_INCREASE, "increase"),
                       (CHANGE_STABLE, "stable"),
                       (CHANGE_DECREASE, "decrease")):
        try:
            out[name] = GridMask(grid=grid, name=name, source=source,
                                 match=valid & (array == code), valid=valid,
                                 provenance={"class_code": int(code)})
        except Exception:
            continue
    return out


def mask_for_record(record: Any, result: Any) -> Any:
    """The mask belonging to one evidence record, or None.

    Matched by the record's `condition` (the condition name), then by label.
    """
    masks = evidence_masks(result)
    if not masks:
        return None
    key = str(getattr(record, "condition", "") or "")
    if key in masks:
        return masks[key]
    label = str(getattr(record, "label", "") or "")
    return masks.get(label)
