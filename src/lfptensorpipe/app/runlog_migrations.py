"""Backward-compatible static import surface for run-log migration APIs."""

from __future__ import annotations

from .shared import runlog_migrations as _module

__all__ = getattr(
    _module,
    "__all__",
    [name for name in dir(_module) if not name.startswith("_")],
)


def __getattr__(name: str):
    return getattr(_module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_module)))
