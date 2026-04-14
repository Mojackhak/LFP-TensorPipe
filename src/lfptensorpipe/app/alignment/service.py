"""Stable alignment service import surface."""

from __future__ import annotations

from lfptensorpipe.app.alignment import workflows as _workflows

__all__ = getattr(
    _workflows,
    "__all__",
    [name for name in dir(_workflows) if not name.startswith("_")],
)


def __getattr__(name: str):
    return getattr(_workflows, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_workflows)))
