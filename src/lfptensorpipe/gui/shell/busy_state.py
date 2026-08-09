"""Busy-state helpers for MainWindow long-running actions."""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, TypeVar

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QWidget

T = TypeVar("T")
_GLOBAL_UI_LOCK_OWNERS = frozenset(("plot", "busy", "tensor"))


def global_ui_lock_owner(window: Any) -> str | None:
    owner = getattr(window, "_global_ui_lock_owner", None)
    return owner if owner in _GLOBAL_UI_LOCK_OWNERS else None


def set_global_ui_lock(
    window: Any,
    *,
    owner: str,
    lock: bool,
    exempt_widgets: Iterable[QWidget] = (),
) -> None:
    """Acquire or release the single main-window interaction lock."""
    if owner not in _GLOBAL_UI_LOCK_OWNERS:
        raise ValueError(f"Unknown global UI lock owner: {owner}")

    current_owner = global_ui_lock_owner(window)
    if lock:
        if current_owner == owner:
            return
        if current_owner is not None:
            raise RuntimeError(f"Global UI lock is already owned by {current_owner}.")

        window._global_ui_lock_owner = owner
        window._global_ui_lock_parent_states = []
        window._global_ui_lock_widgets = []
        window._global_ui_lock_actions = []

        for action in window.findChildren(QAction):
            try:
                if not action.isEnabled():
                    continue
                action.setEnabled(False)
                window._global_ui_lock_actions.append(action)
            except RuntimeError:
                continue

        menu_bar = window.menuBar()
        if menu_bar is not None:
            try:
                was_enabled = menu_bar.isEnabled()
                window._global_ui_lock_parent_states.append((menu_bar, was_enabled))
                if was_enabled:
                    menu_bar.setEnabled(False)
            except RuntimeError:
                pass

        if owner in {"plot", "busy"}:
            central_widget = window.centralWidget()
            if central_widget is not None:
                try:
                    was_enabled = central_widget.isEnabled()
                    window._global_ui_lock_parent_states.append(
                        (central_widget, was_enabled)
                    )
                    if was_enabled:
                        central_widget.setEnabled(False)
                except RuntimeError:
                    pass
            return

        exempt = tuple(widget for widget in exempt_widgets if widget is not None)
        central_widget = window.centralWidget()
        if central_widget is None:
            return
        for widget in central_widget.findChildren(QWidget):
            if any(
                widget is allowed
                or widget.isAncestorOf(allowed)
                or allowed.isAncestorOf(widget)
                for allowed in exempt
            ):
                continue
            try:
                if not widget.isEnabled():
                    continue
                widget.setEnabled(False)
                window._global_ui_lock_widgets.append(widget)
            except RuntimeError:
                continue
        return

    if current_owner != owner:
        return

    for widget, was_enabled in getattr(
        window,
        "_global_ui_lock_parent_states",
        [],
    ):
        try:
            widget.setEnabled(bool(was_enabled))
        except RuntimeError:
            continue
    window._global_ui_lock_parent_states = []

    for widget in getattr(window, "_global_ui_lock_widgets", []):
        try:
            widget.setEnabled(True)
        except RuntimeError:
            continue
    window._global_ui_lock_widgets = []

    for action in getattr(window, "_global_ui_lock_actions", []):
        try:
            action.setEnabled(True)
        except RuntimeError:
            continue
    window._global_ui_lock_actions = []
    window._global_ui_lock_owner = None


def on_busy_tick(window: Any, *, busy_frames: tuple[str, ...]) -> None:
    if window._busy_label is None:
        return
    window._busy_frame_idx = (window._busy_frame_idx + 1) % len(busy_frames)
    render_busy_message(window, busy_frames=busy_frames)


def render_busy_message(window: Any, *, busy_frames: tuple[str, ...]) -> None:
    if window._busy_label is None:
        return
    frame = busy_frames[window._busy_frame_idx]
    suffix = str(getattr(window, "_busy_suffix", "") or "").strip()
    message = f"{window._busy_label} | {frame}"
    if suffix:
        message = f"{message} {suffix}"
    window.statusBar().showMessage(message)


def start_busy(
    window: Any,
    *,
    label: str,
    busy_frames: tuple[str, ...],
    suffix: str | None = None,
) -> None:
    window._busy_label = label
    window._busy_suffix = str(suffix).strip() if suffix else None
    window._busy_frame_idx = 0
    set_busy_ui_lock(window, lock=True)
    window._busy_timer.start()
    render_busy_message(window, busy_frames=busy_frames)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def stop_busy(window: Any) -> None:
    window._busy_timer.stop()
    window._busy_label = None
    window._busy_suffix = None
    window._busy_frame_idx = 0
    set_busy_ui_lock(window, lock=False)


def set_busy_ui_lock(window: Any, *, lock: bool) -> None:
    set_global_ui_lock(window, owner="busy", lock=lock)


def run_with_busy(
    window: Any,
    *,
    label: str,
    work: Callable[[], T],
    busy_frames: tuple[str, ...],
    suffix: str | None = None,
) -> T:
    lock_owner = global_ui_lock_owner(window)
    if window._busy_label is not None or lock_owner is not None:
        active_label = window._busy_label or f"{lock_owner} lock"
        window.statusBar().showMessage(
            f"{active_label} is active; duplicate action ignored."
        )
        raise RuntimeError("Global UI lock is active.")

    result: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = work()
        except BaseException as exc:  # noqa: BLE001
            error["exc"] = exc

    start_busy(window, label=label, busy_frames=busy_frames, suffix=suffix)
    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    app = QApplication.instance()
    try:
        while worker.is_alive():
            if app is not None:
                app.processEvents()
            worker.join(timeout=0.05)
    finally:
        stop_busy(window)

    if "exc" in error:
        raise error["exc"]

    return result["value"]


__all__ = [
    "global_ui_lock_owner",
    "on_busy_tick",
    "render_busy_message",
    "start_busy",
    "stop_busy",
    "set_busy_ui_lock",
    "set_global_ui_lock",
    "run_with_busy",
]
