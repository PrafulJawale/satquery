"""core/samples.py -- bundled sample GeoTIFFs and their provenance.

HONESTY RULE
------------
Every bundled file must declare where it came from and whether its pixel
values are physically valid for spectral analysis. A prototype that quietly
computes NDVI from an 8-bit JPEG-ish product produces numbers that LOOK right
and are wrong. The `provenance.json` file next to the data is the single source
of truth; this module just loads and validates it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import Provenance

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
PROVENANCE_FILE = SAMPLE_DIR / "provenance.json"


def load_provenance() -> Dict[str, Provenance]:
    """filename -> Provenance. Missing/invalid file -> empty dict (no crash)."""
    if not PROVENANCE_FILE.exists():
        return {}
    raw = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    out: Dict[str, Provenance] = {}
    for filename, item in raw.get("samples", {}).items():
        out[filename] = Provenance(
            name=item.get("name", filename),
            filename=filename,
            source_url=item.get("source_url", ""),
            what_it_is=item.get("what_it_is", ""),
            is_real_satellite_data=bool(item.get("is_real_satellite_data", False)),
            bands_are_physically_valid=bool(item.get("bands_are_physically_valid", False)),
            good_for=tuple(item.get("good_for", ())),
            not_good_for=tuple(item.get("not_good_for", ())),
            notes=item.get("notes", ""),
        )
    return out


def list_samples() -> List[str]:
    """Filenames of samples that actually exist on disk, richest first."""
    prov = load_provenance()
    return [name for name in prov if (SAMPLE_DIR / name).exists()]


def sample_path(filename: str) -> Path:
    path = SAMPLE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Sample not found: {path}")
    return path


def sample_provenance(filename: str) -> Optional[Provenance]:
    return load_provenance().get(filename)
