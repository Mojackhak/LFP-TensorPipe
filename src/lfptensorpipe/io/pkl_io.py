"""
Pickle-based persistence helpers for LFP pipelines.

This module is intentionally small and dependency-light. It is used by interactive
step-by-step pipeline scripts under `pipeline/`.
"""

from __future__ import annotations

import cloudpickle as pickle
from pathlib import Path
from typing import Any


def save_pkl(obj: Any, path: Path) -> None:
    """Save a Python object to disk as a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path: Path) -> Any:
    """Load a Python object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
