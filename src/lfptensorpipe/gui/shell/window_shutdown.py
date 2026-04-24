"""Shutdown helpers for app-owned auxiliary windows."""

from __future__ import annotations

from lfptensorpipe.gui.shell.preproc_plotting_backend import (
    active_mne_browser_tracking_tokens,
    active_mne_browser_widget_tokens,
)
from lfptensorpipe.gui.shell.common import Any, QApplication, QWidget


def close_auxiliary_windows(window: Any) -> None:
    """Best-effort close for auxiliary Qt/Matplotlib windows owned by the app."""
    excluded_tokens = active_mne_browser_tracking_tokens(window) | set(
        getattr(window, "_mne_browser_shutdown_excluded_tokens", set())
    )
    _close_tracked_plot_handles(window)
    _close_qt_top_level_widgets(window, excluded_tokens=excluded_tokens)
    _close_matplotlib_figures(excluded_tokens=excluded_tokens)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def _close_tracked_plot_handles(window: Any) -> None:
    seen: set[int] = set()
    for target, _ in list(getattr(window, "_plot_close_hooks", [])):
        _close_candidate(target, owner=window, seen=seen)
        _close_candidate(getattr(target, "fig", None), owner=window, seen=seen)
    plot_close_hooks = getattr(window, "_plot_close_hooks", None)
    if isinstance(plot_close_hooks, list):
        plot_close_hooks.clear()


def _close_qt_top_level_widgets(window: Any, *, excluded_tokens: set[int]) -> None:
    app = QApplication.instance()
    if app is None:
        return
    excluded_widget_tokens = active_mne_browser_widget_tokens(window) | excluded_tokens
    for widget in list(app.topLevelWidgets()):
        if (
            widget is window
            or not isinstance(widget, QWidget)
            or not widget.isVisible()
            or id(widget) in excluded_widget_tokens
        ):
            continue
        try:
            widget.close()
        except Exception:
            continue


def _close_matplotlib_figures(*, excluded_tokens: set[int]) -> None:
    try:
        import matplotlib.pyplot as plt

        for figure_number in list(plt.get_fignums()):
            figure = plt.figure(figure_number)
            if id(figure) in excluded_tokens:
                continue
            plt.close(figure)
    except Exception:
        return


def _close_candidate(candidate: Any, *, owner: Any, seen: set[int]) -> None:
    if candidate is None or candidate is owner:
        return
    token = id(candidate)
    if token in seen:
        return
    seen.add(token)

    close = getattr(candidate, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass

    figure = getattr(candidate, "figure", None)
    if figure is not None and figure is not candidate:
        _close_candidate(figure, owner=owner, seen=seen)

    canvas = getattr(candidate, "canvas", None)
    manager = getattr(canvas, "manager", None) if canvas is not None else None
    manager_window = getattr(manager, "window", None) if manager is not None else None
    if manager_window is not None and manager_window is not candidate:
        _close_candidate(manager_window, owner=owner, seen=seen)
