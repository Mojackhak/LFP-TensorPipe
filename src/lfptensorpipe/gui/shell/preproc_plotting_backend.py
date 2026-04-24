"""Autosave and MNE-browser backend helpers for preprocess plotting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os

import numpy as np

from lfptensorpipe.app import invalidate_after_preproc_result_change
from lfptensorpipe.gui.shell.common import (
    Any,
    QApplication,
    Path,
    QObject,
    QTimer,
    QWidget,
    invalidate_downstream_preproc_steps,
)

PREPROC_PLOT_WINDOW_SIZE = (1200, 800)
_PREPROC_PLOT_DPI_FALLBACK = 96.0
_PLOT_CHANGE_TRACKED_STEPS = frozenset(
    ("filter", "annotations", "bad_segment_removal", "ecg_artifact_removal")
)


def _normalize_preproc_plot_orig_time(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[ns]"))
    if isinstance(value, (tuple, list)):
        normalized_items: list[Any] = []
        for item in value:
            if isinstance(item, (np.integer, int)):
                normalized_items.append(int(item))
                continue
            normalized_items.append(item)
        return tuple(normalized_items)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return repr(value)


def _normalize_preproc_plot_annotations(
    raw: Any,
) -> tuple[tuple[str, float, float], ...]:
    annotations = getattr(raw, "annotations", None)
    if annotations is None:
        return ()
    items: list[tuple[str, float, float]] = []
    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        items.append((str(description).strip(), float(onset), float(duration)))
    items.sort(key=lambda item: (item[1], item[2], item[0]))
    return tuple(items)


def _normalize_preproc_plot_bads(raw: Any) -> tuple[str, ...]:
    info = getattr(raw, "info", None)
    if not hasattr(info, "get"):
        return ()
    bads = info.get("bads", [])
    if not isinstance(bads, (list, tuple)):
        return ()
    normalized = {str(item).strip() for item in bads if str(item).strip()}
    return tuple(sorted(normalized))


def _preproc_plot_raw_signature(raw: Any) -> dict[str, Any]:
    data_getter = getattr(raw, "get_data", None)
    data_obj = (
        data_getter() if callable(data_getter) else np.empty((0, 0), dtype=np.float64)
    )
    data = np.ascontiguousarray(data_obj, dtype=np.float64)
    digest = hashlib.sha256(data.tobytes()).hexdigest()
    return {
        "ch_names": tuple(str(name) for name in getattr(raw, "ch_names", ())),
        "sfreq": float(getattr(raw, "info", {}).get("sfreq", 0.0)),
        "n_times": int(getattr(raw, "n_times", data.shape[-1] if data.ndim else 0)),
        "data_digest": digest,
        "annotations": _normalize_preproc_plot_annotations(raw),
        "bads": _normalize_preproc_plot_bads(raw),
        "orig_time": _normalize_preproc_plot_orig_time(
            getattr(getattr(raw, "annotations", None), "orig_time", None)
        ),
    }


def _preproc_plot_figsize_env_value(owner: Any) -> str:
    dpi_x = _PREPROC_PLOT_DPI_FALLBACK
    dpi_y = _PREPROC_PLOT_DPI_FALLBACK
    screen = None
    screen_getter = getattr(owner, "screen", None)
    if callable(screen_getter):
        try:
            screen = screen_getter()
        except Exception:
            screen = None
    if screen is None:
        app = QApplication.instance()
        if app is not None:
            try:
                screen = app.primaryScreen()
            except Exception:
                screen = None
    if screen is not None:
        dpi_x_getter = getattr(screen, "logicalDotsPerInchX", None)
        dpi_y_getter = getattr(screen, "logicalDotsPerInchY", None)
        if callable(dpi_x_getter):
            try:
                dpi_x = max(float(dpi_x_getter()), 1.0)
            except Exception:
                dpi_x = _PREPROC_PLOT_DPI_FALLBACK
        if callable(dpi_y_getter):
            try:
                dpi_y = max(float(dpi_y_getter()), 1.0)
            except Exception:
                dpi_y = _PREPROC_PLOT_DPI_FALLBACK
    width_px, height_px = PREPROC_PLOT_WINDOW_SIZE
    return f"{width_px / dpi_x:.6f},{height_px / dpi_y:.6f}"


def _resize_preproc_plot_window(browser: Any) -> None:
    resize = getattr(browser, "resize", None)
    if not callable(resize):
        return
    try:
        resize(*PREPROC_PLOT_WINDOW_SIZE)
    except Exception:
        pass


def _browser_tracking_targets(
    browser: Any,
) -> tuple[Any | None, QObject | None, Any | None]:
    figure = getattr(browser, "fig", None)
    if figure is None and hasattr(browser, "canvas"):
        figure = browser

    canvas = getattr(figure, "canvas", None)
    manager = getattr(canvas, "manager", None) if canvas is not None else None
    manager_window = getattr(manager, "window", None) if manager is not None else None

    qt_object: QObject | None = browser if isinstance(browser, QObject) else None
    if qt_object is None and isinstance(figure, QObject):
        qt_object = figure
    if qt_object is None and isinstance(manager_window, QObject):
        qt_object = manager_window
    return figure, qt_object, manager_window


def _restore_quit_on_last_window_closed(window: Any) -> None:
    app = QApplication.instance()
    if app is None:
        window._mne_browser_shutdown_prev_quit_on_last_window_closed = None
        return
    previous = getattr(
        window,
        "_mne_browser_shutdown_prev_quit_on_last_window_closed",
        None,
    )
    if previous is None:
        return
    try:
        app.setQuitOnLastWindowClosed(bool(previous))
    except Exception:
        pass
    window._mne_browser_shutdown_prev_quit_on_last_window_closed = None


def _finalize_app_shutdown(self) -> None:
    if getattr(self, "_finalizing_mainwindow_close", False):
        return
    self._mne_browser_shutdown_pending = False
    self._finalizing_mainwindow_close = True

    def _close_main_window() -> None:
        try:
            self.close()
        finally:
            _restore_quit_on_last_window_closed(self)
            app = QApplication.instance()
            if app is not None:
                QTimer.singleShot(0, app.quit)

    QTimer.singleShot(0, _close_main_window)


def _finalize_tracked_browser_close(self, token: int, event: Any | None = None) -> None:
    _ = event
    registry = getattr(self, "_active_mne_browsers", None)
    if not isinstance(registry, dict):
        return
    entry = registry.get(token)
    if entry is None or bool(entry.get("closed", False)):
        return
    entry["closed"] = True
    registry.pop(token, None)

    context = entry.get("context")
    raw = entry.get("raw")
    raw_path = entry.get("raw_path")
    step = entry.get("step")
    title_prefix = str(entry.get("title_prefix", "Plot"))

    try:
        if step == "raw":
            self.statusBar().showMessage(f"{title_prefix} plot closed.")
        elif (
            step in _PLOT_CHANGE_TRACKED_STEPS
            and context is not None
            and isinstance(raw_path, Path)
        ):
            opened_signature = entry.get("opened_signature")
            closed_signature = _preproc_plot_raw_signature(raw)
            if opened_signature != closed_signature:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw.save(str(raw_path), overwrite=True)
                invalidate_downstream_preproc_steps(context, step)
                invalidate_after_preproc_result_change(context, changed_step=step)
                self._refresh_stage_states_from_context()
                self._refresh_preproc_controls()
                self.statusBar().showMessage(
                    f"{title_prefix} plot closed: saved edited {raw_path.name}; downstream invalidated."
                )
            else:
                self.statusBar().showMessage(f"{title_prefix} plot closed.")
        elif step is None:
            self.statusBar().showMessage(f"{title_prefix} plot closed.")
    except Exception as exc:  # noqa: BLE001
        self._show_warning(
            f"{title_prefix} Plot",
            f"Plot close handling failed:\n{exc}",
        )
    finally:
        raw_close = getattr(raw, "close", None)
        if callable(raw_close):
            try:
                raw_close()
            except Exception:
                pass
        if getattr(self, "_mne_browser_shutdown_pending", False) and not registry:
            QTimer.singleShot(0, self._finalize_app_shutdown)


def _request_close_registered_browser(entry: dict[str, Any]) -> None:
    if bool(entry.get("close_requested", False)):
        return
    entry["close_requested"] = True
    for candidate_key in ("browser", "qt_object", "manager_window", "figure"):
        candidate = entry.get(candidate_key)
        close = getattr(candidate, "close", None)
        if callable(close):
            try:
                close()
                return
            except Exception:
                continue


def _request_close_all_mne_browsers(self) -> None:
    registry = getattr(self, "_active_mne_browsers", None)
    if not isinstance(registry, dict) or not registry:
        self._finalize_app_shutdown()
        return
    for entry in list(registry.values()):
        _request_close_registered_browser(entry)
    if not registry:
        self._finalize_app_shutdown()


def _defer_close_for_active_mne_browsers(self, event: Any) -> bool:
    if getattr(self, "_finalizing_mainwindow_close", False):
        return False
    registry = getattr(self, "_active_mne_browsers", None)
    if not isinstance(registry, dict) or not registry:
        return False
    try:
        event.ignore()
    except Exception:
        pass
    if getattr(self, "_mne_browser_shutdown_pending", False):
        return True

    self._mne_browser_shutdown_pending = True
    self._mne_browser_shutdown_excluded_tokens = active_mne_browser_tracking_tokens(
        self
    )
    self.statusBar().showMessage("Closing active preprocess plot windows...")
    app = QApplication.instance()
    if app is not None:
        try:
            previous = bool(app.quitOnLastWindowClosed())
        except Exception:
            previous = True
        self._mne_browser_shutdown_prev_quit_on_last_window_closed = previous
        try:
            app.setQuitOnLastWindowClosed(False)
        except Exception:
            pass
    QTimer.singleShot(0, self._request_close_all_mne_browsers)
    return True


def _attach_plot_autosave(
    self,
    *,
    browser: Any,
    raw: Any,
    raw_path: Path,
    step: str,
    title_prefix: str,
) -> None:
    _track_mne_browser(
        self,
        browser=browser,
        raw=raw,
        raw_path=raw_path,
        step=step,
        title_prefix=title_prefix,
    )


def _track_mne_browser(
    self,
    *,
    browser: Any,
    raw: Any,
    raw_path: Path,
    step: str | None,
    title_prefix: str,
) -> None:
    registry = getattr(self, "_active_mne_browsers", None)
    if not isinstance(registry, dict):
        registry = {}
        self._active_mne_browsers = registry

    figure, qt_object, manager_window = _browser_tracking_targets(browser)
    token = id(browser)
    entry: dict[str, Any] = {
        "browser": browser,
        "raw": raw,
        "raw_path": raw_path,
        "step": step,
        "title_prefix": title_prefix,
        "figure": figure,
        "qt_object": qt_object,
        "manager_window": manager_window,
        "context": self._record_context(),
        "opened_signature": (
            _preproc_plot_raw_signature(raw)
            if step in _PLOT_CHANGE_TRACKED_STEPS
            else None
        ),
        "close_requested": False,
        "closed": False,
    }
    registry[token] = entry

    def _on_close(event: Any | None = None) -> None:
        _finalize_tracked_browser_close(self, token, event)

    got_closed = getattr(browser, "gotClosed", None)
    used_mne_closed_signal = False
    if got_closed is not None and hasattr(got_closed, "connect"):
        try:
            got_closed.connect(_on_close)
            used_mne_closed_signal = True
        except Exception:
            used_mne_closed_signal = False

    if qt_object is not None and not used_mne_closed_signal:
        close_filter_cls = self._close_autosave_filter_class()
        close_filter = close_filter_cls(_on_close, qt_object)
        entry["close_filter"] = close_filter
        qt_object.installEventFilter(close_filter)
        try:
            qt_object.destroyed.connect(_on_close)
        except Exception:
            pass

    if (
        not used_mne_closed_signal
        and figure is not None
        and hasattr(figure, "canvas")
        and hasattr(figure.canvas, "mpl_connect")
    ):
        try:
            callback_id = figure.canvas.mpl_connect("close_event", _on_close)
            entry["mpl_close_callback_id"] = callback_id
        except Exception:
            pass


def _open_mne_raw_plot(
    self,
    raw_path: Path,
    title_prefix: str,
    *,
    autosave_step: str | None = None,
) -> None:
    if not self._enable_plots:
        return
    if getattr(self, "_mne_browser_shutdown_pending", False):
        self.statusBar().showMessage(
            f"{title_prefix} Plot unavailable: app shutdown is in progress."
        )
        return
    try:
        raw = self._read_raw_fif(raw_path, preload=True, verbose="ERROR")
        previous_plot_size = os.environ.get("MNE_BROWSE_RAW_SIZE")
        os.environ["MNE_BROWSE_RAW_SIZE"] = _preproc_plot_figsize_env_value(self)
        try:
            browser = raw.plot(block=False, title=f"{title_prefix}: {raw_path.name}")
        finally:
            if previous_plot_size is None:
                os.environ.pop("MNE_BROWSE_RAW_SIZE", None)
            else:
                os.environ["MNE_BROWSE_RAW_SIZE"] = previous_plot_size
        _resize_preproc_plot_window(browser)
        if autosave_step is not None:
            self._attach_plot_autosave(
                browser=browser,
                raw=raw,
                raw_path=raw_path,
                step=autosave_step,
                title_prefix=title_prefix,
            )
        else:
            _track_mne_browser(
                self,
                browser=browser,
                raw=raw,
                raw_path=raw_path,
                step=None,
                title_prefix=title_prefix,
            )
    except Exception as exc:
        self.statusBar().showMessage(f"{title_prefix} Plot failed: {exc}")


def active_mne_browser_tracking_tokens(window: Any) -> set[int]:
    registry = getattr(window, "_active_mne_browsers", None)
    if not isinstance(registry, dict):
        return set()
    tokens: set[int] = set()
    for entry in registry.values():
        for candidate_key in ("browser", "qt_object", "figure", "manager_window"):
            if entry.get(candidate_key) is not None:
                candidate = entry[candidate_key]
                tokens.add(id(candidate))
    return tokens


def active_mne_browser_widget_tokens(window: Any) -> set[int]:
    registry = getattr(window, "_active_mne_browsers", None)
    if not isinstance(registry, dict):
        return set()
    tokens: set[int] = set()
    for entry in registry.values():
        for candidate_key in ("browser", "qt_object", "figure", "manager_window"):
            candidate = entry.get(candidate_key)
            if isinstance(candidate, QWidget):
                tokens.add(id(candidate))
    return tokens
