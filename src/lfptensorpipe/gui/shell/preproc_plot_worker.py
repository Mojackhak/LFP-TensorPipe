"""Dedicated worker process for preprocess MNE raw-browser plots."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication

_PARENT_POLL_INTERVAL_MS = 250
_SHUTDOWN_FALLBACK_MS = 250


def _write_stdout(message: str) -> None:
    print(message)


def _write_stderr(message: str) -> None:
    print(message, file=sys.stderr)


def process_exists(pid: int) -> bool:
    """Return whether a PID still exists."""
    if pid <= 0:
        return True
    try:
        import os

        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class _BrowserCloseFilter(QObject):
    """Event filter that reacts once to a Qt close event."""

    def __init__(
        self,
        on_close: Callable[[Any | None], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._on_close = on_close

    def eventFilter(self, watched: QObject, event: QEvent | None) -> bool:
        _ = watched
        if event is not None and event.type() == QEvent.Close:
            self._on_close(event)
        return False


class _PreprocPlotWorkerController:
    """Own the raw-browser lifecycle inside the dedicated worker process."""

    def __init__(
        self,
        *,
        app: QApplication,
        raw: Any,
        browser: Any,
        raw_path: Path,
        autosave: bool,
        parent_pid: int,
        auto_close_ms: int,
        stdout_writer: Callable[[str], None] = _write_stdout,
        stderr_writer: Callable[[str], None] = _write_stderr,
        process_exists_fn: Callable[[int], bool] = process_exists,
    ) -> None:
        self._app = app
        self._raw = raw
        self._browser = browser
        self._raw_path = raw_path
        self._autosave = autosave
        self._parent_pid = max(int(parent_pid), 0)
        self._auto_close_ms = max(int(auto_close_ms), 0)
        self._stdout_writer = stdout_writer
        self._stderr_writer = stderr_writer
        self._process_exists_fn = process_exists_fn
        self._close_filter: _BrowserCloseFilter | None = None
        self._parent_timer: QTimer | None = None
        self._closed = False
        self._shutdown_requested = False

    def install(self) -> None:
        self._install_close_hooks()
        if self._auto_close_ms > 0:
            QTimer.singleShot(self._auto_close_ms, self.request_shutdown)
        if self._parent_pid > 0:
            self._parent_timer = QTimer()
            self._parent_timer.setInterval(_PARENT_POLL_INTERVAL_MS)
            self._parent_timer.timeout.connect(self._poll_parent_process)
            self._parent_timer.start()

    def request_shutdown(self) -> None:
        """Ask the browser to close and finish the worker."""
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        close = getattr(self._browser, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        QTimer.singleShot(_SHUTDOWN_FALLBACK_MS, self._on_browser_closed)

    def _poll_parent_process(self) -> None:
        if self._parent_pid <= 0:
            return
        if not self._process_exists_fn(self._parent_pid):
            self.request_shutdown()

    def _install_close_hooks(self) -> None:
        figure = getattr(self._browser, "fig", None)
        if figure is None and hasattr(self._browser, "canvas"):
            figure = self._browser

        qt_object: QObject | None = (
            self._browser if isinstance(self._browser, QObject) else None
        )
        if qt_object is None and isinstance(figure, QObject):
            qt_object = figure

        got_closed = getattr(self._browser, "gotClosed", None)
        if got_closed is not None and hasattr(got_closed, "connect"):
            try:
                got_closed.connect(self._on_browser_closed)
                return
            except Exception:
                pass

        if qt_object is not None:
            self._close_filter = _BrowserCloseFilter(self._on_browser_closed, qt_object)
            qt_object.installEventFilter(self._close_filter)
            try:
                qt_object.destroyed.connect(self._on_browser_closed)
                return
            except Exception:
                pass

        canvas = getattr(figure, "canvas", None)
        if canvas is not None and hasattr(canvas, "mpl_connect"):
            try:
                canvas.mpl_connect("close_event", self._on_browser_closed)
            except Exception:
                pass

    def _on_browser_closed(self, event: Any | None = None) -> None:
        _ = event
        if self._closed:
            return
        self._closed = True
        if self._parent_timer is not None:
            self._parent_timer.stop()
        try:
            if self._autosave:
                self._raw_path.parent.mkdir(parents=True, exist_ok=True)
                self._raw.save(str(self._raw_path), overwrite=True)
                self._stdout_writer(f"Plot closed: auto-saved {self._raw_path.name}.")
            else:
                self._stdout_writer(f"Plot closed: {self._raw_path.name}.")
            self._app.exit(0)
        except Exception as exc:  # noqa: BLE001
            self._stderr_writer(f"Plot close failed: {exc}")
            self._app.exit(1)


def _ensure_app(
    app_factory: Callable[[list[str]], QApplication] | None = None,
) -> QApplication:
    app = QApplication.instance()
    if app is not None:
        return app
    if app_factory is not None:
        return app_factory(["lfptensorpipe-preproc-plot-worker"])
    return QApplication(["lfptensorpipe-preproc-plot-worker"])


def _install_signal_handlers(
    controller: _PreprocPlotWorkerController,
    *,
    signal_module: Any = signal,
) -> list[tuple[int, Any]]:
    installed: list[tuple[int, Any]] = []
    for signal_name in ("SIGTERM", "SIGINT"):
        signal_value = getattr(signal_module, signal_name, None)
        if signal_value is None:
            continue
        try:
            previous = signal_module.getsignal(signal_value)
            signal_module.signal(
                signal_value,
                lambda signum, frame: controller.request_shutdown(),
            )
        except Exception:
            continue
        installed.append((signal_value, previous))
    return installed


def _restore_signal_handlers(
    installed: list[tuple[int, Any]],
    *,
    signal_module: Any = signal,
) -> None:
    for signal_value, previous in installed:
        try:
            signal_module.signal(signal_value, previous)
        except Exception:
            continue


def run_preproc_plot_worker(
    raw_fif_path: str,
    *,
    title: str,
    autosave: bool = False,
    parent_pid: int = 0,
    auto_close_ms: int = 0,
    app_factory: Callable[[list[str]], QApplication] | None = None,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    controller_cls: type[_PreprocPlotWorkerController] = _PreprocPlotWorkerController,
    signal_module: Any = signal,
) -> int:
    """Open one MNE raw browser inside a dedicated worker process."""
    raw_path = Path(raw_fif_path).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    if read_raw_fif_fn is None:
        import mne

        read_raw_fif_fn = mne.io.read_raw_fif

    app = _ensure_app(app_factory=app_factory)
    raw = read_raw_fif_fn(raw_path, preload=True, verbose="ERROR")
    browser = raw.plot(block=False, title=title)
    controller = controller_cls(
        app=app,
        raw=raw,
        browser=browser,
        raw_path=raw_path,
        autosave=autosave,
        parent_pid=parent_pid,
        auto_close_ms=auto_close_ms,
    )
    controller.install()
    installed = _install_signal_handlers(controller, signal_module=signal_module)
    try:
        return app.exec()
    finally:
        _restore_signal_handlers(installed, signal_module=signal_module)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse worker-process arguments."""
    parser = argparse.ArgumentParser(prog="lfptensorpipe-preproc-plot-worker")
    parser.add_argument("--raw-fif-path", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--auto-close-ms", type=int, default=0)
    parser.add_argument("--autosave", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the preprocess raw-browser worker."""
    args = parse_args(argv)
    try:
        return run_preproc_plot_worker(
            str(args.raw_fif_path),
            title=str(args.title),
            autosave=bool(args.autosave),
            parent_pid=int(args.parent_pid),
            auto_close_ms=int(args.auto_close_ms),
        )
    except Exception as exc:  # noqa: BLE001
        _write_stderr(f"Preprocess plot worker failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
