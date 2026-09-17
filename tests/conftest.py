"""Makes `core/` importable when pytest is run from the repository root.

    python -m pytest tests -q

(No pyproject/setup.py yet -- adding packaging is Phase 2+ work. Until then this
one sys.path line is the honest, minimal fix.)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Unit tests must not touch the network. `ui.map.build_street_basemap()` fetches
# OSM tiles server-side for the live app; every test asserts on map *structure*,
# so it is disabled here. The behaviour it provides is covered offline by
# `tests/test_phase4_map.py::test_street_basemap_layers_are_built_offline` (which
# monkeypatches the fetch) and end-to-end by `scripts/verify_basemap.py`.
import os

os.environ.setdefault("SATQUERY_DISABLE_SERVER_BASEMAP", "1")
