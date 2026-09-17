"""Phase 11 -- the spectral-index abstraction.

WHY THIS MODULE EXISTS
----------------------
Phases 3-10 each needed "an index", and each got its own function. Adding a
fourth (NDWI) that way would mean a fourth copy of the same denominator guard,
the same nodata handling, the same NaN bookkeeping and the same caveat wording
-- four places for one scientific convention to drift apart.

So an index is now DATA, not code:

    config/indices/ndvi.yml  --\
                                >-- IndexDefinition --\
    config/indices/ndwi.yml  --/                       \
                                                        -> core.indices.compute_index()
                                                             -> AnalysisResult

Adding NDBI or MNDWI later means adding one YAML file (and, for a non-normalised
difference such as SAVI, one `kind: custom` branch). Adding a *threshold* is a
separate, deliberate decision and lives in its own phase -- none of the
definitions below contain one.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
* No classifier. An index is continuous; "water if NDWI > X" is a scientific
  claim that Phase 11 does not make.
* No band-index guessing. Role resolution belongs to `core.bands`; this module
  only records the documented role -> band mapping for provenance.
* No Streamlit, no I/O beyond reading its own YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

INDEX_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "indices"
)

#: The two shapes an index definition may take today.
KIND_NORMALIZED_DIFFERENCE = "normalized_difference"
KIND_CUSTOM = "custom"          # reserved: SAVI / EVI-style formulae (future)


@dataclass(frozen=True)
class IndexDefinition:
    """Everything needed to compute and describe one spectral index.

    `numerator_role` / `denominator_role` are band ROLES ("red", "green",
    "nir", "swir1"), never band numbers: the number depends on the raster, and
    resolving it is `core.bands`' job.

    For `kind: normalized_difference` the value is

        (numerator - denominator) / (numerator + denominator)
    """

    name: str
    short_name: str
    display_name: str
    label: str
    kind: str = KIND_NORMALIZED_DIFFERENCE

    numerator_role: str = "nir"
    denominator_role: str = "red"
    required_roles: Tuple[str, ...] = ()
    role_labels: Dict[str, str] = field(default_factory=dict)

    sensor_bands: Dict[str, Dict[str, str]] = field(default_factory=dict)
    wavelengths_nm: Dict[str, Optional[int]] = field(default_factory=dict)

    formula: str = ""
    citation: str = ""

    min_denominator: float = 1e-6
    value_range: Tuple[float, float] = (-1.0, 1.0)
    colormap: str = "RdYlGn"

    reflectance_scale: Optional[float] = None
    reflectance_offset: Optional[float] = 0.0

    description: str = ""
    caveat: str = ""
    limitations: Tuple[str, ...] = ()
    version: str = ""

    # -- derived ----------------------------------------------------------- #
    @property
    def roles(self) -> Tuple[str, ...]:
        """Roles in the order they are reported (`bands_used`, valid mask)."""
        return tuple(self.required_roles) or (self.denominator_role, self.numerator_role)

    @property
    def is_normalized_difference(self) -> bool:
        return self.kind == KIND_NORMALIZED_DIFFERENCE

    def role_label(self, role: str) -> str:
        """Human label for a role, e.g. nir -> "NIR" (used in error messages)."""
        return self.role_labels.get(role, role.upper() if len(role) <= 4 else role.title())

    def band_id(self, role: str, sensor: Optional[str] = None) -> Optional[str]:
        """Documented band id for a role (e.g. ("nir", "sentinel-2-l2a") -> "B08")."""
        if sensor and sensor in self.sensor_bands:
            return self.sensor_bands[sensor].get(role)
        for mapping in self.sensor_bands.values():
            if role in mapping:
                return mapping[role]
        return None

    def band_ids_for(self, sensor: Optional[str]) -> Dict[str, str]:
        if sensor and sensor in self.sensor_bands:
            return dict(self.sensor_bands[sensor])
        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "short_name": self.short_name,
            "display_name": self.display_name,
            "label": self.label,
            "kind": self.kind,
            "numerator_role": self.numerator_role,
            "denominator_role": self.denominator_role,
            "required_roles": list(self.roles),
            "formula": self.formula,
            "citation": self.citation,
            "min_denominator": self.min_denominator,
            "value_range": list(self.value_range),
            "colormap": self.colormap,
            "reflectance": {
                "scale": self.reflectance_scale,
                "offset": self.reflectance_offset,
            },
            "description": self.description,
            "caveat": self.caveat,
            "limitations": list(self.limitations),
            "version": self.version,
        }


# =========================================================================== #
# loading
# =========================================================================== #
def load_index_config(name: str, base_dir: str = INDEX_CONFIG_DIR) -> Dict[str, Any]:
    """Load config/indices/<name>.yml (mirrors core.suitability / core.temporal)."""
    import yaml

    path = os.path.join(base_dir, f"{name}.yml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Index configuration is not a mapping: {path}")
    return cfg


def definition_from_config(cfg: Dict[str, Any]) -> IndexDefinition:
    """Build an IndexDefinition from a parsed YAML mapping."""
    reflectance = cfg.get("reflectance") or {}
    value_range = cfg.get("value_range") or [-1.0, 1.0]
    required = tuple(str(r) for r in (cfg.get("required_roles") or ()))
    return IndexDefinition(
        name=str(cfg.get("name", "")),
        short_name=str(cfg.get("short_name") or str(cfg.get("name", "")).upper()),
        display_name=str(cfg.get("display_name") or cfg.get("name", "")),
        label=str(cfg.get("label") or cfg.get("display_name") or cfg.get("name", "")),
        kind=str(cfg.get("kind") or KIND_NORMALIZED_DIFFERENCE),
        numerator_role=str(cfg.get("numerator_role", "nir")),
        denominator_role=str(cfg.get("denominator_role", "red")),
        required_roles=required,
        role_labels={str(k): str(v) for k, v in (cfg.get("role_labels") or {}).items()},
        sensor_bands={str(k): {str(a): str(b) for a, b in (v or {}).items()}
                      for k, v in (cfg.get("sensor_bands") or {}).items()},
        wavelengths_nm={str(k): (int(v) if v is not None else None)
                        for k, v in (cfg.get("wavelengths_nm") or {}).items()},
        formula=str(cfg.get("formula", "")),
        citation=str(cfg.get("citation", "")),
        min_denominator=float(cfg.get("min_denominator", 1e-6)),
        value_range=(float(value_range[0]), float(value_range[1])),
        colormap=str(cfg.get("colormap", "RdYlGn")),
        reflectance_scale=(float(reflectance["scale"]) if reflectance.get("scale") is not None else None),
        reflectance_offset=(float(reflectance.get("offset") or 0.0)),
        description=str(cfg.get("description", "")).strip(),
        caveat=str(cfg.get("caveat", "")).strip(),
        limitations=tuple(str(x) for x in (cfg.get("limitations") or ())),
        version=str(cfg.get("version", "")),
    )


@lru_cache(maxsize=16)
def get_index(name: str) -> IndexDefinition:
    """One definition, cached. Raises FileNotFoundError for an unknown index."""
    return definition_from_config(load_index_config(name))


def available_indices(base_dir: str = INDEX_CONFIG_DIR) -> Tuple[str, ...]:
    """Every index with a configuration file, sorted."""
    if not os.path.isdir(base_dir):
        return ()
    return tuple(sorted(
        f[:-4] for f in os.listdir(base_dir) if f.lower().endswith(".yml")
    ))


def all_indices(base_dir: str = INDEX_CONFIG_DIR) -> Dict[str, IndexDefinition]:
    return {name: get_index(name) for name in available_indices(base_dir)}


__all__ = [
    "INDEX_CONFIG_DIR",
    "KIND_NORMALIZED_DIFFERENCE",
    "KIND_CUSTOM",
    "IndexDefinition",
    "load_index_config",
    "definition_from_config",
    "get_index",
    "available_indices",
    "all_indices",
]
