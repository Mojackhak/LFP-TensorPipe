"""
Pickle-based persistence helpers for LFP pipelines.

This module is intentionally small and dependency-light. It is used by interactive
step-by-step pipeline scripts under `pipeline/`.
"""

from __future__ import annotations

import cloudpickle as pickle
from pathlib import Path
import stat
from tempfile import NamedTemporaryFile
from typing import Any


def save_pkl(obj: Any, path: Path) -> None:
    """Atomically save a Python object to disk as a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            pickle.dump(obj, f)
        if target_mode is not None:
            temp_path.chmod(target_mode)
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def load_pkl(path: Path) -> Any:
    """Load a Python object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
