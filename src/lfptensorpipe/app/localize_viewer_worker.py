"""Backward-compatible static import surface for localize viewer worker APIs."""

from __future__ import annotations

from .localize import viewer_worker as _module

if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(_module.main())

__all__ = getattr(
    _module,
    "__all__",
    [name for name in dir(_module) if not name.startswith("_")],
)


def __getattr__(name: str):
    return getattr(_module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_module)))
