"""Paper analysis workspace."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src-layout package importable when running paper modules directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SRC_ROOT_TEXT = str(_SRC_ROOT)

if _SRC_ROOT.exists() and _SRC_ROOT_TEXT not in sys.path:
    sys.path.insert(0, _SRC_ROOT_TEXT)
