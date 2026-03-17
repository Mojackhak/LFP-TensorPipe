"""Contact viewer launch helpers for Localize."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

from lfptensorpipe.desktop_runtime import (
    LOCALIZE_VIEWER_WORKER_FLAG,
    LOCALIZE_VIEWER_WORKER_MODULE,
    build_worker_command,
)

from .paths import localize_representative_csv_path


def can_open_contact_viewer(paths: Any) -> tuple[bool, str]:
    if not paths.leaddbs_dir.exists():
        return False, f"Invalid Lead-DBS path: {paths.leaddbs_dir}"
    if not paths.matlab_root.exists():
        return False, f"Invalid MATLAB installation path: {paths.matlab_root}"
    return True, "Contact viewer preconditions are satisfied."


def default_contact_viewer_launcher(
    csv_path: Path,
    atlas: str,
    paths: Any,
    *,
    python_exec: str = sys.executable,
    popen: Callable[..., Any] = subprocess.Popen,
    build_command_fn: Callable[..., list[str]] = build_worker_command,
) -> None:
    cmd = build_command_fn(
        module_name=LOCALIZE_VIEWER_WORKER_MODULE,
        embedded_flag=LOCALIZE_VIEWER_WORKER_FLAG,
        worker_args=[
            "--csv-path",
            str(csv_path),
            "--atlas",
            atlas,
            "--leaddbs-dir",
            str(paths.leaddbs_dir),
            "--matlab-root",
            str(paths.matlab_root),
        ],
        python_exec=python_exec,
    )
    popen(cmd, start_new_session=True)


def launch_contact_viewer(
    *,
    project_root: Path,
    subject: str,
    record: str,
    atlas: str,
    paths: Any,
    launcher: Callable[[Path, str, Any], None] | None,
    can_open_contact_viewer_fn: Callable[
        [Any], tuple[bool, str]
    ] = can_open_contact_viewer,
    default_launcher_fn: Callable[
        [Path, str, Any], None
    ] = default_contact_viewer_launcher,
) -> tuple[bool, str]:
    ok, message = can_open_contact_viewer_fn(paths)
    if not ok:
        return False, message

    csv_path = localize_representative_csv_path(project_root, subject, record)
    if not csv_path.is_file():
        return False, f"Missing Localize representative CSV: {csv_path}"

    try:
        run_launcher = launcher or default_launcher_fn
        run_launcher(csv_path, atlas, paths)
    except Exception as exc:
        return False, f"Contact Viewer launch failed: {exc}"
    return True, "Contact Viewer launched in independent process."
