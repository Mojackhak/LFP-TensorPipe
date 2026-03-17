"""Busy-state helpers for MainWindow long-running actions."""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QAbstractButton, QApplication

T = TypeVar("T")


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
    if lock:
        window._busy_locked_buttons = []
        for button in window.findChildren(QAbstractButton):
            try:
                if not button.isEnabled():
                    continue
                button.setEnabled(False)
                window._busy_locked_buttons.append(button)
            except RuntimeError:
                continue

        window._busy_locked_actions = []
        for action in window.findChildren(QAction):
            try:
                if not action.isEnabled():
                    continue
                action.setEnabled(False)
                window._busy_locked_actions.append(action)
            except RuntimeError:
                continue
        return

    for button in window._busy_locked_buttons:
        try:
            button.setEnabled(True)
        except RuntimeError:
            continue
    window._busy_locked_buttons = []

    for action in window._busy_locked_actions:
        try:
            action.setEnabled(True)
        except RuntimeError:
            continue
    window._busy_locked_actions = []


def run_with_busy(
    window: Any,
    *,
    label: str,
    work: Callable[[], T],
    busy_frames: tuple[str, ...],
    suffix: str | None = None,
) -> T:
    if window._busy_label is not None:
        window.statusBar().showMessage(
            f"{window._busy_label} is running; duplicate action ignored."
        )
        raise RuntimeError("Busy lock is active.")

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
    "on_busy_tick",
    "render_busy_message",
    "start_busy",
    "stop_busy",
    "set_busy_ui_lock",
    "run_with_busy",
]
