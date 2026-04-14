"""Helpers for launching subprocesses from source and frozen desktop builds."""

from __future__ import annotations

from collections.abc import Sequence
import sys

TENSOR_WORKER_MODULE = "lfptensorpipe.app.tensor.worker"
TENSOR_WORKER_FLAG = "--run-tensor-worker"
RUNTIME_PLAN_WORKER_MODULE = "lfptensorpipe.app.tensor.runtime_plan_worker"
RUNTIME_PLAN_WORKER_FLAG = "--run-runtime-plan-worker"
LOCALIZE_VIEWER_WORKER_MODULE = "lfptensorpipe.app.localize_viewer_worker"
LOCALIZE_VIEWER_WORKER_FLAG = "--run-localize-viewer-worker"

EMBEDDED_WORKER_FLAGS = (
    TENSOR_WORKER_FLAG,
    RUNTIME_PLAN_WORKER_FLAG,
    LOCALIZE_VIEWER_WORKER_FLAG,
)


def is_frozen_app() -> bool:
    """Return whether the current process is running from a frozen bundle."""
    return bool(getattr(sys, "frozen", False))


def detect_embedded_worker_flag(argv: Sequence[str]) -> str | None:
    """Return the first embedded-worker flag found in argv."""
    for flag in EMBEDDED_WORKER_FLAGS:
        if flag in argv:
            return flag
    return None


def strip_embedded_worker_flag(argv: Sequence[str], flag: str) -> list[str]:
    """Return argv with one embedded-worker flag occurrence removed."""
    stripped: list[str] = []
    removed = False
    for item in argv:
        if item == flag and not removed:
            removed = True
            continue
        stripped.append(item)
    return stripped


def build_worker_command(
    *,
    module_name: str,
    embedded_flag: str,
    worker_args: Sequence[str],
    python_exec: str | None = None,
    frozen: bool | None = None,
) -> list[str]:
    """Build a subprocess command for source or frozen execution."""
    executable = python_exec or sys.executable
    use_embedded_entry = is_frozen_app() if frozen is None else bool(frozen)
    if use_embedded_entry:
        return [executable, embedded_flag, *worker_args]
    return [executable, "-m", module_name, *worker_args]
