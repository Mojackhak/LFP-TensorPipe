"""Autosave and MNE-browser backend helpers for preprocess plotting."""

from __future__ import annotations

import os

from lfptensorpipe.gui.shell.common import (
    Any,
    QApplication,
    Path,
    PathResolver,
    QObject,
    invalidate_downstream_preproc_steps,
    rawdata_input_fif_path,
)

PREPROC_PLOT_WINDOW_SIZE = (1200, 800)
_PREPROC_PLOT_DPI_FALLBACK = 96.0


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


def _attach_plot_autosave(
    self,
    *,
    browser: Any,
    raw: Any,
    raw_path: Path,
    step: str,
    title_prefix: str,
) -> None:
    figure = getattr(browser, "fig", None)
    if figure is None and hasattr(browser, "canvas"):
        figure = browser

    state = {"done": False}

    def _on_close(event: Any | None = None) -> None:
        _ = event
        if state["done"]:
            return
        state["done"] = True
        context = self._record_context()
        if context is None:
            return
        resolver = PathResolver(context)
        try:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw.save(str(raw_path), overwrite=True)
            if step == "raw":
                self._mark_preproc_step_runtime(
                    resolver=resolver,
                    step="raw",
                    completed=True,
                    input_path=str(rawdata_input_fif_path(context)),
                    output_path=str(raw_path),
                    message="Raw plot closed; auto-saved raw step artifact.",
                )
                invalidate_downstream_preproc_steps(context, "raw")
                self._refresh_stage_states_from_context()
            self._refresh_preproc_controls()
            self.statusBar().showMessage(
                f"{title_prefix} plot closed: auto-saved {raw_path.name}."
            )
        except Exception as exc:  # noqa: BLE001
            if step == "raw":
                self._mark_preproc_step_runtime(
                    resolver=resolver,
                    step="raw",
                    completed=False,
                    input_path=str(rawdata_input_fif_path(context)),
                    output_path=str(raw_path),
                    message=f"Raw plot auto-save failed: {exc}",
                )
                self._refresh_stage_states_from_context()
                self._refresh_preproc_controls()
            self._show_warning(
                f"{title_prefix} Plot",
                f"Auto-save failed on plot close:\n{exc}",
            )

    qt_object: QObject | None = browser if isinstance(browser, QObject) else None
    if qt_object is None and isinstance(figure, QObject):
        qt_object = figure
    used_mne_closed_signal = False
    got_closed = getattr(browser, "gotClosed", None)
    if got_closed is not None and hasattr(got_closed, "connect"):
        try:
            got_closed.connect(_on_close)
            self._plot_close_hooks.append((browser, "gotClosed"))
            used_mne_closed_signal = True
        except Exception:
            used_mne_closed_signal = False

    if qt_object is not None and not used_mne_closed_signal:
        close_filter_cls = self._close_autosave_filter_class()
        close_filter = close_filter_cls(_on_close, qt_object)
        qt_object.installEventFilter(close_filter)
        self._plot_close_hooks.append((qt_object, close_filter))
        try:
            qt_object.destroyed.connect(_on_close)
            self._plot_close_hooks.append((qt_object, "destroyed"))
        except Exception:
            pass

    if (
        not used_mne_closed_signal
        and figure is not None
        and hasattr(figure, "canvas")
        and hasattr(figure.canvas, "mpl_connect")
    ):
        callback_id = figure.canvas.mpl_connect("close_event", _on_close)
        self._plot_close_hooks.append((figure, callback_id))


def _open_mne_raw_plot(
    self,
    raw_path: Path,
    title_prefix: str,
    *,
    autosave_step: str | None = None,
) -> None:
    if not self._enable_plots:
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
    except Exception as exc:
        self.statusBar().showMessage(f"{title_prefix} Plot failed: {exc}")
