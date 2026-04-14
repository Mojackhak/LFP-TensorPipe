"""Application entrypoint for LFP-TensorPipe desktop GUI."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from typing import Callable

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from lfptensorpipe.desktop_runtime import (
    LOCALIZE_VIEWER_WORKER_FLAG,
    RUNTIME_PLAN_WORKER_FLAG,
    TENSOR_WORKER_FLAG,
    detect_embedded_worker_flag,
    strip_embedded_worker_flag,
)
from lfptensorpipe.desktop_smoke import (
    run_smoke_demo_config_imports,
    run_smoke_demo_record_imports,
    run_smoke_demo_record_parsers,
    run_smoke_numerical_full_pipeline,
    run_smoke_numerical_preproc,
    run_smoke_preproc_ui,
    run_smoke_raw_plot,
    run_smoke_tensor_runtime,
)
from lfptensorpipe.gui import MainWindow
from lfptensorpipe.gui.icon_pipeline import preferred_runtime_icon_path

_NULL_STREAM_HANDLES: list[object] = []


def _ensure_console_streams() -> None:
    """Provide file-like console streams for GUI/frozen runs that expose None."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "write"):
            continue
        sink = open(os.devnull, "w", encoding="utf-8")
        _NULL_STREAM_HANDLES.append(sink)
        setattr(sys, stream_name, sink)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="lfptensorpipe")
    parser.add_argument(
        "--auto-close-ms",
        type=int,
        default=0,
        help="Auto-close the app after N milliseconds (for smoke validation).",
    )
    parser.add_argument(
        "--smoke-raw-plot-fif",
        default="",
        help="Internal smoke validation: open one raw FIF plot and auto-close it.",
    )
    parser.add_argument(
        "--smoke-raw-plot-close-ms",
        type=int,
        default=1500,
        help="Auto-close delay for --smoke-raw-plot-fif.",
    )
    parser.add_argument(
        "--smoke-demo-records-root",
        default="",
        help="Internal smoke validation: parse all repository demo records.",
    )
    parser.add_argument(
        "--smoke-demo-record-imports-root",
        default="",
        help="Internal smoke validation: import demo records through MainWindow.",
    )
    parser.add_argument(
        "--smoke-demo-configs-root",
        default="",
        help="Internal smoke validation: import demo configs through MainWindow.",
    )
    parser.add_argument(
        "--smoke-preproc-ui",
        action="store_true",
        help="Internal smoke validation: exercise packaged Preprocess-page handlers.",
    )
    parser.add_argument(
        "--smoke-numerical-preproc",
        action="store_true",
        help="Internal smoke validation: run Mojackhak numerical preprocess validation.",
    )
    parser.add_argument(
        "--smoke-numerical-full-pipeline",
        action="store_true",
        help="Internal smoke validation: run Mojackhak full-pipeline numerical validation.",
    )
    parser.add_argument(
        "--smoke-tensor-runtime",
        action="store_true",
        help="Internal smoke validation: run one packaged Build Tensor runtime path.",
    )
    parser.add_argument(
        "--smoke-project-root",
        default="",
        help="Project root for packaged config/preprocess smoke validation.",
    )
    parser.add_argument(
        "--smoke-subject",
        default="",
        help="Subject for packaged config/preprocess smoke validation.",
    )
    parser.add_argument(
        "--smoke-record",
        default="",
        help="Record for packaged config/preprocess smoke validation.",
    )
    parser.add_argument(
        "--smoke-reference-root",
        default="",
        help="Reference project root for packaged numerical validation.",
    )
    parser.add_argument(
        "--smoke-numerical-records-root",
        default="",
        help="Demo-record root for packaged numerical validation.",
    )
    parser.add_argument(
        "--smoke-numerical-configs-root",
        default="",
        help="Config root for packaged full-pipeline numerical validation.",
    )
    return parser.parse_args(argv)


def _run_embedded_worker(
    flag: str,
    worker_argv: list[str],
    *,
    tensor_worker_main: Callable[[list[str] | None], int] | None = None,
    runtime_plan_worker_main: Callable[[list[str] | None], int] | None = None,
    localize_viewer_worker_main: Callable[[list[str] | None], int] | None = None,
) -> int:
    if flag == TENSOR_WORKER_FLAG:
        if tensor_worker_main is None:
            from lfptensorpipe.app.tensor.worker import main as tensor_worker_main

        return tensor_worker_main(worker_argv)
    if flag == RUNTIME_PLAN_WORKER_FLAG:
        if runtime_plan_worker_main is None:
            from lfptensorpipe.app.tensor.runtime_plan_worker import (
                main as runtime_plan_worker_main,
            )

        return runtime_plan_worker_main(worker_argv)
    if flag == LOCALIZE_VIEWER_WORKER_FLAG:
        if localize_viewer_worker_main is None:
            from lfptensorpipe.app.localize_viewer_worker import (
                main as localize_viewer_worker_main,
            )

        return localize_viewer_worker_main(worker_argv)
    raise ValueError(f"Unsupported embedded worker flag: {flag}")


def main(
    argv: list[str] | None = None,
    *,
    tensor_worker_main: Callable[[list[str] | None], int] | None = None,
    runtime_plan_worker_main: Callable[[list[str] | None], int] | None = None,
    localize_viewer_worker_main: Callable[[list[str] | None], int] | None = None,
    smoke_raw_plot_main: Callable[[str, int], int] | None = None,
    smoke_demo_records_main: Callable[[str], int] | None = None,
    smoke_demo_record_imports_main: Callable[[str], int] | None = None,
    smoke_demo_configs_main: Callable[[str, str, str, str], int] | None = None,
    smoke_preproc_ui_main: Callable[[str, str, str], int] | None = None,
    smoke_numerical_preproc_main: Callable[[str, str, str, str], int] | None = None,
    smoke_numerical_full_pipeline_main: (
        Callable[[str, str, str, str, str], int] | None
    ) = None,
    smoke_tensor_runtime_main: Callable[[str, str, str, str], int] | None = None,
) -> int:
    """Launch the LFP-TensorPipe desktop shell."""
    multiprocessing.freeze_support()
    _ensure_console_streams()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    worker_flag = detect_embedded_worker_flag(argv_list)
    if worker_flag is not None:
        return _run_embedded_worker(
            worker_flag,
            strip_embedded_worker_flag(argv_list, worker_flag),
            tensor_worker_main=tensor_worker_main,
            runtime_plan_worker_main=runtime_plan_worker_main,
            localize_viewer_worker_main=localize_viewer_worker_main,
        )

    args = parse_args(argv_list)

    if args.smoke_raw_plot_fif:
        smoke_runner = smoke_raw_plot_main or (
            lambda path, close_ms: run_smoke_raw_plot(path, close_ms=close_ms)
        )
        return smoke_runner(
            str(args.smoke_raw_plot_fif),
            int(args.smoke_raw_plot_close_ms),
        )
    if args.smoke_demo_records_root:
        smoke_runner = smoke_demo_records_main or run_smoke_demo_record_parsers
        return smoke_runner(str(args.smoke_demo_records_root))
    if args.smoke_demo_record_imports_root:
        smoke_runner = smoke_demo_record_imports_main or run_smoke_demo_record_imports
        return smoke_runner(str(args.smoke_demo_record_imports_root))
    if args.smoke_tensor_runtime:
        smoke_runner = smoke_tensor_runtime_main or run_smoke_tensor_runtime
        return smoke_runner(
            str(args.smoke_demo_configs_root),
            str(args.smoke_project_root),
            str(args.smoke_subject),
            str(args.smoke_record),
        )
    if args.smoke_demo_configs_root:
        smoke_runner = smoke_demo_configs_main or run_smoke_demo_config_imports
        return smoke_runner(
            str(args.smoke_demo_configs_root),
            str(args.smoke_project_root),
            str(args.smoke_subject),
            str(args.smoke_record),
        )
    if args.smoke_preproc_ui:
        smoke_runner = smoke_preproc_ui_main or run_smoke_preproc_ui
        return smoke_runner(
            str(args.smoke_project_root),
            str(args.smoke_subject),
            str(args.smoke_record),
        )
    if args.smoke_numerical_preproc:
        smoke_runner = smoke_numerical_preproc_main or run_smoke_numerical_preproc
        return smoke_runner(
            str(args.smoke_reference_root),
            str(args.smoke_project_root),
            str(args.smoke_subject),
            str(args.smoke_numerical_records_root),
        )
    if args.smoke_numerical_full_pipeline:
        smoke_runner = (
            smoke_numerical_full_pipeline_main or run_smoke_numerical_full_pipeline
        )
        return smoke_runner(
            str(args.smoke_reference_root),
            str(args.smoke_project_root),
            str(args.smoke_subject),
            str(args.smoke_numerical_records_root),
            str(args.smoke_numerical_configs_root),
        )

    app = QApplication.instance()
    if app is None:
        app = QApplication(["lfptensorpipe", *argv_list])

    app.setStyle("Fusion")
    app_font = app.font()
    app_font.setFamily("Arial")
    app.setFont(QFont(app_font))
    app.setApplicationName("LFP-TensorPipe")
    runtime_icon = preferred_runtime_icon_path()
    if runtime_icon is not None:
        app_icon = QIcon(str(runtime_icon))
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)

    window = MainWindow()
    if runtime_icon is not None:
        window_icon = QIcon(str(runtime_icon))
        if not window_icon.isNull():
            window.setWindowIcon(window_icon)
    window.show()

    if args.auto_close_ms > 0:
        QTimer.singleShot(args.auto_close_ms, app.quit)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
