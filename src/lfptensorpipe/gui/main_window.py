"""Public MainWindow entrypoint with stable import path."""

from __future__ import annotations

from lfptensorpipe.gui.shell import main_window_logic as _main_window_logic

__all__ = getattr(
    _main_window_logic,
    "__all__",
    [name for name in dir(_main_window_logic) if not name.startswith("_")],
)


def __getattr__(name: str):
    return getattr(_main_window_logic, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_main_window_logic)))
