"""
Pickle-based persistence helpers for LFP pipelines.

This module is intentionally small and dependency-light. It is used by interactive
step-by-step pipeline scripts under `pipeline/`.
"""

from __future__ import annotations

import cloudpickle as pickle
from pathlib import Path
from typing import Any

from .atomic_file import atomic_binary_file


def save_pkl(obj: Any, path: Path) -> None:
    """Atomically save a Python object to disk as a pickle file."""
    with atomic_binary_file(path, create_mode=0o600) as stream:
        pickle.dump(obj, stream)


def load_pkl(path: Path) -> Any:
    """Load a Python object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
