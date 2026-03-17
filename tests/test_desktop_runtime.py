"""Tests for frozen-app subprocess helpers."""

from __future__ import annotations

from lfptensorpipe.desktop_runtime import (
    LOCALIZE_VIEWER_WORKER_FLAG,
    RUNTIME_PLAN_WORKER_FLAG,
    TENSOR_WORKER_FLAG,
    build_worker_command,
    detect_embedded_worker_flag,
    strip_embedded_worker_flag,
)


def test_build_worker_command_uses_module_entry_in_source_mode() -> None:
    cmd = build_worker_command(
        module_name="lfptensorpipe.app.tensor.worker",
        embedded_flag=TENSOR_WORKER_FLAG,
        worker_args=["--request", "/tmp/request.json"],
        python_exec="/usr/bin/python3",
        frozen=False,
    )

    assert cmd == [
        "/usr/bin/python3",
        "-m",
        "lfptensorpipe.app.tensor.worker",
        "--request",
        "/tmp/request.json",
    ]


def test_build_worker_command_uses_embedded_flag_in_frozen_mode() -> None:
    cmd = build_worker_command(
        module_name="lfptensorpipe.app.tensor.worker",
        embedded_flag=TENSOR_WORKER_FLAG,
        worker_args=["--request", "/tmp/request.json"],
        python_exec="/Applications/LFP-TensorPipe.app/Contents/MacOS/LFP-TensorPipe",
        frozen=True,
    )

    assert cmd == [
        "/Applications/LFP-TensorPipe.app/Contents/MacOS/LFP-TensorPipe",
        TENSOR_WORKER_FLAG,
        "--request",
        "/tmp/request.json",
    ]


def test_detect_embedded_worker_flag_returns_first_known_flag() -> None:
    argv = ["--auto-close-ms", "1000", RUNTIME_PLAN_WORKER_FLAG]

    assert detect_embedded_worker_flag(argv) == RUNTIME_PLAN_WORKER_FLAG


def test_strip_embedded_worker_flag_removes_only_one_occurrence() -> None:
    argv = [
        LOCALIZE_VIEWER_WORKER_FLAG,
        "--csv-path",
        "/tmp/coords.csv",
        LOCALIZE_VIEWER_WORKER_FLAG,
    ]

    assert strip_embedded_worker_flag(argv, LOCALIZE_VIEWER_WORKER_FLAG) == [
        "--csv-path",
        "/tmp/coords.csv",
        LOCALIZE_VIEWER_WORKER_FLAG,
    ]
