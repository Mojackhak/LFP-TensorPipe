"""Localize panel runtime helpers."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable

import pandas as pd

from lfptensorpipe.anat.repcoords import (
    build_ordered_pair_representative_frame as _build_ordered_pair_repcoords_frame_impl,
    build_undirected_pair_representative_frame as _build_undirected_pair_repcoords_frame_impl,
)
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import read_run_log, read_ui_state
from .apply_runner import run_localize_apply as _run_localize_apply_impl
from .atlas_lookup import (
    build_repcoords_frame as _build_repcoords_frame_impl,
)
from .paths import (
    discover_atlases,
    discover_spaces,
    has_reconstruction_mat,
    infer_subject_space,
    infer_subject_spaces,
    load_localize_paths,
    localize_csv_path,
    localize_indicator_state,
    localize_log_path,
    localize_panel_state as _localize_panel_state_impl,
    localize_mat_path,
    localize_match_signature,
    localize_ordered_pair_representative_csv_path,
    localize_ordered_pair_representative_pkl_path,
    localize_representative_csv_path,
    localize_representative_pkl_path,
    localize_undirected_pair_representative_csv_path,
    localize_undirected_pair_representative_pkl_path,
    reconstruction_mat_path,
    reconstruction_root,
)
from .reconstruction import (
    load_reconstruction_contacts as _load_reconstruction_contacts_impl,
)
from .viewer import (
    can_open_contact_viewer as _can_open_contact_viewer_impl,
    default_contact_viewer_launcher as _default_contact_viewer_launcher_impl,
    launch_contact_viewer as _launch_contact_viewer_impl,
)


@dataclass(frozen=True)
class LocalizePaths:
    """Resolved runtime paths needed by Localize workflows."""

    leaddbs_dir: Path
    matlab_root: Path

    def __init__(
        self,
        *,
        leaddbs_dir: Path,
        matlab_root: Path | None = None,
        matlab_engine_path: Path | None = None,
    ) -> None:
        if matlab_root is None and matlab_engine_path is None:
            matlab_root = Path("__missing_matlab_root__")
        if matlab_root is None and matlab_engine_path is not None:
            from lfptensorpipe.matlab import infer_matlab_root

            matlab_root = (
                infer_matlab_root(Path(matlab_engine_path).expanduser())
                or Path(matlab_engine_path).expanduser()
            )
        object.__setattr__(self, "leaddbs_dir", Path(leaddbs_dir).expanduser())
        object.__setattr__(self, "matlab_root", Path(matlab_root).expanduser())

    @property
    def matlab_engine_path(self) -> Path:
        """Backward-compatible alias for the legacy MATLAB engine source dir."""
        candidate = self.matlab_root / "extern" / "engines" / "python"
        if candidate.exists():
            return candidate
        return self.matlab_root


LocalizeRuntimeRunner = Callable[
    [Path, str, str, str, list[str] | tuple[str, ...], LocalizePaths, Path, Path],
    None,
]
ContactViewerLauncher = Callable[[Path, str, LocalizePaths], None]


_ELSPEC_CACHE: dict[str, dict[str, Any]] = {}
_MATLAB_RUNTIME_LOCK = threading.Lock()
_MATLAB_RUNTIME_ENGINE: Any | None = None
_MATLAB_RUNTIME_KEY: tuple[str, str] | None = None
_MATLAB_RUNTIME_STATE = "idle"
_MATLAB_RUNTIME_MESSAGE = "Not started."
_MATLAB_TASK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="lfptp-matlab"
)
_MATLAB_TASK_TIMEOUT_S = 60.0
_MATLAB_SHUTDOWN_TIMEOUT_S = 5.0
_MATLAB_CONTROL_LOCK = threading.Lock()
_MATLAB_CONTROL_SEQ = 0
_MATLAB_CONTROL_LATEST: dict[str, int] = {}
_MATLAB_WARMUP_FUTURE: Future[tuple[bool, str]] | None = None
_MATLAB_WARMUP_KEY: tuple[str, str] | None = None
_MATLAB_STALE_CONTEXT_PREFIX = "stale-context:"


def _runtime_key(paths: LocalizePaths) -> tuple[str, str]:
    return (
        str(paths.leaddbs_dir.expanduser().resolve()),
        str(paths.matlab_root.expanduser().resolve()),
    )


def _set_matlab_runtime_status(state: str, message: str) -> None:
    global _MATLAB_RUNTIME_STATE, _MATLAB_RUNTIME_MESSAGE
    with _MATLAB_RUNTIME_LOCK:
        _MATLAB_RUNTIME_STATE = state
        _MATLAB_RUNTIME_MESSAGE = message


def matlab_runtime_status() -> tuple[str, str]:
    with _MATLAB_RUNTIME_LOCK:
        return _MATLAB_RUNTIME_STATE, _MATLAB_RUNTIME_MESSAGE


def clear_localize_runtime_cache() -> None:
    _ELSPEC_CACHE.clear()


def _completed_future(result: tuple[bool, str]) -> Future[tuple[bool, str]]:
    future: Future[tuple[bool, str]] = Future()
    future.set_result(result)
    return future


def _next_control_ticket(context_key: str) -> int:
    global _MATLAB_CONTROL_SEQ
    with _MATLAB_CONTROL_LOCK:
        _MATLAB_CONTROL_SEQ += 1
        ticket = _MATLAB_CONTROL_SEQ
        _MATLAB_CONTROL_LATEST[context_key] = ticket
    return ticket


def _is_latest_control_ticket(context_key: str, ticket: int) -> bool:
    with _MATLAB_CONTROL_LOCK:
        return _MATLAB_CONTROL_LATEST.get(context_key) == ticket


def is_stale_context_message(message: str | None) -> bool:
    return str(message or "").startswith(_MATLAB_STALE_CONTEXT_PREFIX)


def _submit_latest_control_task(
    context_key: str,
    fn: Callable[[], tuple[bool, str]],
) -> Future[tuple[bool, str]]:
    ticket = _next_control_ticket(context_key)

    def _runner() -> tuple[bool, str]:
        if not _is_latest_control_ticket(context_key, ticket):
            return False, f"{_MATLAB_STALE_CONTEXT_PREFIX} superseded by newer request."
        return fn()

    return _MATLAB_TASK_EXECUTOR.submit(_runner)


def _drop_matlab_engine() -> None:
    global _MATLAB_RUNTIME_ENGINE, _MATLAB_RUNTIME_KEY, _MATLAB_WARMUP_FUTURE, _MATLAB_WARMUP_KEY
    engine = None
    with _MATLAB_RUNTIME_LOCK:
        engine = _MATLAB_RUNTIME_ENGINE
        _MATLAB_RUNTIME_ENGINE = None
        _MATLAB_RUNTIME_KEY = None
        _MATLAB_WARMUP_FUTURE = None
        _MATLAB_WARMUP_KEY = None
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass


def _local_matlab_functions_dir() -> Path:
    module_file = Path(__file__).resolve()
    for parent in module_file.parents:
        candidate = parent / "src" / "lfptensorpipe" / "anat" / "leaddbs"
        if candidate.is_dir():
            return candidate
    return module_file.parents[0] / "anat" / "leaddbs"


def _ensure_matlab_engine_ready(
    paths: LocalizePaths,
    *,
    ensure_matlab_engine_fn: Callable[[Path], Any] | None = None,
    start_matlab_fn: Callable[[], Any] | None = None,
    matlab_functions_dir: Path | None = None,
) -> Any:
    global _MATLAB_RUNTIME_ENGINE, _MATLAB_RUNTIME_KEY
    if ensure_matlab_engine_fn is None:
        from lfptensorpipe.matlab import ensure_matlab_engine

        ensure_matlab_engine_fn = ensure_matlab_engine

    key = _runtime_key(paths)
    with _MATLAB_RUNTIME_LOCK:
        engine = _MATLAB_RUNTIME_ENGINE
        current_key = _MATLAB_RUNTIME_KEY
    if engine is not None and current_key == key:
        _set_matlab_runtime_status("ready", "Ready")
        return engine

    if engine is not None and current_key != key:
        _drop_matlab_engine()

    if not paths.leaddbs_dir.is_dir():
        message = f"Invalid Lead-DBS path: {paths.leaddbs_dir}"
        _set_matlab_runtime_status("failed", message)
        raise RuntimeError(message)
    if not paths.matlab_root.exists():
        message = f"Invalid MATLAB installation path: {paths.matlab_root}"
        _set_matlab_runtime_status("failed", message)
        raise RuntimeError(message)

    _set_matlab_runtime_status("starting", "Starting...")
    ensure_matlab_engine_fn(paths.matlab_root)

    if start_matlab_fn is None:
        import matlab.engine

        start_matlab_fn = matlab.engine.start_matlab

    fn_dir = matlab_functions_dir or _local_matlab_functions_dir()
    eng = start_matlab_fn()
    try:
        eng.addpath(eng.genpath(str(paths.leaddbs_dir)), nargout=0)
        if fn_dir.is_dir():
            eng.addpath(str(fn_dir), nargout=0)
    except Exception:
        try:
            eng.quit()
        except Exception:
            pass
        raise

    with _MATLAB_RUNTIME_LOCK:
        _MATLAB_RUNTIME_ENGINE = eng
        _MATLAB_RUNTIME_KEY = key
    _set_matlab_runtime_status("ready", "Ready")
    return eng


def _is_engine_disconnected_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "engine terminated",
        "engine is not running",
        "matlab has exited",
        "invalid object",
        "broken pipe",
        "connection reset",
    )
    return any(marker in text for marker in markers)


def _execute_matlab_task(paths: LocalizePaths, fn: Callable[[Any], Any]) -> Any:
    retry = False
    while True:
        eng = _ensure_matlab_engine_ready(paths)
        try:
            return fn(eng)
        except Exception as exc:
            if not retry and _is_engine_disconnected_error(exc):
                retry = True
                _drop_matlab_engine()
                continue
            _set_matlab_runtime_status("failed", f"{exc}")
            raise


def submit_matlab_task(paths: LocalizePaths, fn: Callable[[Any], Any]) -> Future[Any]:
    return _MATLAB_TASK_EXECUTOR.submit(_execute_matlab_task, paths, fn)


def warmup_matlab_async(
    paths: LocalizePaths,
    *,
    ensure_matlab_engine_ready_fn: Callable[[LocalizePaths], Any] | None = None,
) -> Future[tuple[bool, str]]:
    global _MATLAB_WARMUP_FUTURE, _MATLAB_WARMUP_KEY
    key = _runtime_key(paths)
    with _MATLAB_RUNTIME_LOCK:
        engine = _MATLAB_RUNTIME_ENGINE
        runtime_key = _MATLAB_RUNTIME_KEY
        warmup_future = _MATLAB_WARMUP_FUTURE
        warmup_key = _MATLAB_WARMUP_KEY
    if engine is not None and runtime_key == key:
        _set_matlab_runtime_status("ready", "Ready")
        return _completed_future((True, "MATLAB ready."))
    if warmup_future is not None and not warmup_future.done() and warmup_key == key:
        return warmup_future

    def _warmup() -> tuple[bool, str]:
        try:
            (ensure_matlab_engine_ready_fn or _ensure_matlab_engine_ready)(paths)
        except Exception as exc:  # noqa: BLE001
            message = f"MATLAB warmup failed: {exc}"
            _set_matlab_runtime_status("failed", message)
            return False, message
        _set_matlab_runtime_status("ready", "Ready")
        return True, "MATLAB ready."

    context_key = "runtime-control"
    future = _submit_latest_control_task(context_key, _warmup)
    with _MATLAB_RUNTIME_LOCK:
        _MATLAB_WARMUP_FUTURE = future
        _MATLAB_WARMUP_KEY = key
    return future


def reset_matlab_runtime(
    paths: LocalizePaths | None = None,
    *,
    ensure_matlab_engine_ready_fn: Callable[[LocalizePaths], Any] | None = None,
) -> Future[tuple[bool, str]]:
    global _MATLAB_WARMUP_FUTURE, _MATLAB_WARMUP_KEY
    clear_localize_runtime_cache()

    def _reset() -> tuple[bool, str]:
        _drop_matlab_engine()
        _set_matlab_runtime_status("idle", "Not started.")
        if paths is None:
            return True, "Runtime reset."
        try:
            (ensure_matlab_engine_ready_fn or _ensure_matlab_engine_ready)(paths)
        except Exception as exc:  # noqa: BLE001
            message = f"MATLAB warmup failed: {exc}"
            _set_matlab_runtime_status("failed", message)
            return False, message
        _set_matlab_runtime_status("ready", "Ready")
        return True, "MATLAB ready."

    context_key = "runtime-control"
    future = _submit_latest_control_task(context_key, _reset)
    with _MATLAB_RUNTIME_LOCK:
        _MATLAB_WARMUP_FUTURE = future
        _MATLAB_WARMUP_KEY = _runtime_key(paths) if paths is not None else None
    return future


def shutdown_matlab_runtime(timeout_s: float = _MATLAB_SHUTDOWN_TIMEOUT_S) -> None:
    def _shutdown() -> None:
        _drop_matlab_engine()
        _set_matlab_runtime_status("idle", "Not started.")

    future = _MATLAB_TASK_EXECUTOR.submit(_shutdown)
    try:
        future.result(timeout=timeout_s)
    except Exception:
        pass


def _load_match_payload_from_record_ui_state(
    project_root: Path, subject: str, record: str
) -> dict[str, Any] | None:
    context = RecordContext(
        project_root=project_root,
        subject=subject,
        record=record,
    )
    resolver = PathResolver(context)
    path = resolver.record_ui_state_path(create=False)
    if path.is_file():
        try:
            payload = read_ui_state(path)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            localize_node = payload.get("localize", {})
            if isinstance(localize_node, dict):
                match = localize_node.get("match")
                if isinstance(match, dict):
                    return dict(match)

    # Read-only legacy fallback for one release window.
    legacy_path = resolver.lfp_root / "lfptensorpipe_log.json"
    if not legacy_path.is_file():
        return None
    try:
        legacy_payload = read_run_log(legacy_path)
    except Exception:
        return None
    if not isinstance(legacy_payload, dict):
        return None
    params = legacy_payload.get("params")
    if not isinstance(params, dict):
        return None
    localize_node = params.get("localize", {})
    if not isinstance(localize_node, dict):
        return None
    match = localize_node.get("match")
    return dict(match) if isinstance(match, dict) else None


def load_reconstruction_contacts(
    project_root: Path,
    subject: str,
    paths: LocalizePaths,
) -> tuple[bool, str, dict[str, Any]]:
    return _load_reconstruction_contacts_impl(project_root, subject, paths)


def _build_repcoords_frame(
    *,
    project_root: Path,
    subject: str,
    record: str,
    space: str,
    atlas: str,
    region_names: list[str] | tuple[str, ...] | None = None,
    paths: LocalizePaths,
    reconstruction: dict[str, Any],
    mappings: list[dict[str, Any]],
) -> pd.DataFrame:
    return _build_repcoords_frame_impl(
        project_root=project_root,
        subject=subject,
        record=record,
        space=space,
        atlas=atlas,
        region_names=region_names,
        paths=paths,
        reconstruction=reconstruction,
        mappings=mappings,
    )


def _build_ordered_pair_repcoords_frame(channel_frame: pd.DataFrame) -> pd.DataFrame:
    return _build_ordered_pair_repcoords_frame_impl(channel_frame)


def _build_undirected_pair_repcoords_frame(channel_frame: pd.DataFrame) -> pd.DataFrame:
    return _build_undirected_pair_repcoords_frame_impl(channel_frame)


def run_localize_apply(
    *,
    project_root: Path,
    subject: str,
    record: str,
    space: str,
    atlas: str,
    selected_regions: list[str] | tuple[str, ...],
    paths: LocalizePaths | None = None,
    runtime_runner: LocalizeRuntimeRunner | None = None,
    read_only_project_root: Path | None = None,
    load_match_payload_fn: (
        Callable[[Path, str, str], dict[str, Any] | None] | None
    ) = None,
    load_reconstruction_contacts_fn: (
        Callable[[Path, str, Any], tuple[bool, str, dict[str, Any]]] | None
    ) = None,
    build_repcoords_frame_fn: Callable[..., pd.DataFrame] | None = None,
    build_ordered_pair_repcoords_frame_fn: (
        Callable[[pd.DataFrame], pd.DataFrame] | None
    ) = None,
    build_undirected_pair_repcoords_frame_fn: (
        Callable[[pd.DataFrame], pd.DataFrame] | None
    ) = None,
) -> tuple[bool, str]:
    _ = runtime_runner
    return _run_localize_apply_impl(
        project_root=project_root,
        subject=subject,
        record=record,
        space=space,
        atlas=atlas,
        selected_regions=selected_regions,
        paths=paths,
        read_only_project_root=read_only_project_root,
        load_match_payload=load_match_payload_fn
        or _load_match_payload_from_record_ui_state,
        load_reconstruction_contacts=load_reconstruction_contacts_fn
        or load_reconstruction_contacts,
        build_repcoords_frame=build_repcoords_frame_fn or _build_repcoords_frame,
        build_ordered_pair_repcoords_frame=build_ordered_pair_repcoords_frame_fn
        or _build_ordered_pair_repcoords_frame,
        build_undirected_pair_repcoords_frame=build_undirected_pair_repcoords_frame_fn
        or _build_undirected_pair_repcoords_frame,
    )


def localize_panel_state(
    project_root: Path,
    subject: str,
    record: str,
    *,
    atlas: Any,
    selected_regions: Any,
    match_payload: dict[str, Any] | None,
) -> str:
    return _localize_panel_state_impl(
        project_root,
        subject,
        record,
        atlas=atlas,
        selected_regions=selected_regions,
        match_payload=match_payload,
    )


def can_open_contact_viewer(paths: LocalizePaths) -> tuple[bool, str]:
    return _can_open_contact_viewer_impl(paths)


def _default_contact_viewer_launcher(
    csv_path: Path,
    atlas: str,
    paths: LocalizePaths,
    *,
    python_exec: str | None = None,
    popen: Callable[..., Any] | None = None,
) -> None:
    _default_contact_viewer_launcher_impl(
        csv_path,
        atlas,
        paths,
        python_exec=python_exec or sys.executable,
        popen=popen or subprocess.Popen,
    )


def launch_contact_viewer(
    *,
    project_root: Path,
    subject: str,
    record: str,
    atlas: str,
    paths: LocalizePaths,
    launcher: ContactViewerLauncher | None = None,
) -> tuple[bool, str]:
    return _launch_contact_viewer_impl(
        project_root=project_root,
        subject=subject,
        record=record,
        atlas=atlas,
        paths=paths,
        launcher=launcher,
        can_open_contact_viewer_fn=can_open_contact_viewer,
        default_launcher_fn=_default_contact_viewer_launcher,
    )


__all__ = [
    "ContactViewerLauncher",
    "LocalizePaths",
    "LocalizeRuntimeRunner",
    "can_open_contact_viewer",
    "clear_localize_runtime_cache",
    "discover_atlases",
    "discover_spaces",
    "has_reconstruction_mat",
    "infer_subject_space",
    "infer_subject_spaces",
    "is_stale_context_message",
    "launch_contact_viewer",
    "load_localize_paths",
    "load_reconstruction_contacts",
    "localize_csv_path",
    "localize_indicator_state",
    "localize_log_path",
    "localize_mat_path",
    "localize_match_signature",
    "localize_ordered_pair_representative_csv_path",
    "localize_ordered_pair_representative_pkl_path",
    "localize_panel_state",
    "localize_representative_csv_path",
    "localize_representative_pkl_path",
    "localize_undirected_pair_representative_csv_path",
    "localize_undirected_pair_representative_pkl_path",
    "matlab_runtime_status",
    "reconstruction_mat_path",
    "reconstruction_root",
    "reset_matlab_runtime",
    "run_localize_apply",
    "shutdown_matlab_runtime",
    "submit_matlab_task",
    "warmup_matlab_async",
]
