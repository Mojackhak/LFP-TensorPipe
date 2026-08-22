"""Localize panel runtime helpers."""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import pandas as pd

from lfptensorpipe.anat.repcoords import (
    build_ordered_pair_representative_frame as _build_ordered_pair_repcoords_frame_impl,
    build_undirected_pair_representative_frame as _build_undirected_pair_repcoords_frame_impl,
)
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import read_ui_state
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


class _DaemonSingleWorkerExecutor:
    """Run submitted Localize tasks in FIFO order on one daemon worker."""

    def __init__(self, *, thread_name_prefix: str) -> None:
        self._thread_name_prefix = thread_name_prefix
        self._queue: SimpleQueue[
            tuple[Future[Any], Callable[..., Any], tuple[Any, ...], dict[str, Any]]
        ] = SimpleQueue()
        self._start_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._shutdown_complete = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._worker: threading.Thread | None = None

    def submit(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        future: Future[Any] = Future()
        with self._start_lock:
            if self._shutdown.is_set():
                raise RuntimeError("cannot schedule new futures after shutdown")
            if self._worker is None:
                worker = threading.Thread(
                    target=self._run,
                    name=f"{self._thread_name_prefix}_0",
                    daemon=True,
                )
                worker.start()
                self._worker = worker
            self._queue.put((future, fn, args, kwargs))
        return future

    def _run_item(
        self,
        item: tuple[Future[Any], Callable[..., Any], tuple[Any, ...], dict[str, Any]],
    ) -> None:
        future, fn, args, kwargs = item
        result: Any = None
        try:
            if self._shutdown.is_set():
                future.cancel()
                return
            if not future.set_running_or_notify_cancel():
                return
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
            else:
                future.set_result(result)
        finally:
            del item, future, fn, args, kwargs, result

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            self._idle.clear()
            try:
                self._run_item(item)
            finally:
                self._idle.set()
                del item

    def cancel_pending(self) -> None:
        """Reject new submissions and cancel queued work that has not started."""
        with self._start_lock:
            self._shutdown.set()
            while True:
                try:
                    item = self._queue.get_nowait()
                except Empty:
                    break
                future = item[0]
                future.cancel()
                del item, future

    def is_idle(self) -> bool:
        return self._idle.is_set()

    def is_shutdown(self) -> bool:
        return self._shutdown.is_set()

    def mark_shutdown_complete(self) -> None:
        self._shutdown_complete.set()

    def shutdown_complete(self) -> bool:
        return self._shutdown_complete.is_set()


_ELSPEC_CACHE: dict[str, dict[str, Any]] = {}
_MATLAB_RUNTIME_LOCK = threading.Lock()
_MATLAB_RUNTIME_ENGINE: Any | None = None
_MATLAB_RUNTIME_KEY: tuple[str, str] | None = None
_MATLAB_RUNTIME_LAUNCH: Any | None = None
_MATLAB_RUNTIME_LAUNCH_PENDING = False
_MATLAB_RUNTIME_LAUNCH_PUBLISHED = threading.Event()
_MATLAB_RUNTIME_LAUNCH_PUBLISHED.set()
_MATLAB_RUNTIME_SHUTDOWN = threading.Event()
_MATLAB_RUNTIME_STATE = "idle"
_MATLAB_RUNTIME_MESSAGE = "Not started."
_MATLAB_TASK_EXECUTOR = _DaemonSingleWorkerExecutor(thread_name_prefix="lfptp-matlab")
_MATLAB_ENGINE_QUIT_EXECUTOR = _DaemonSingleWorkerExecutor(
    thread_name_prefix="lfptp-matlab-quit"
)
_MATLAB_ENGINE_QUIT_TASKS: dict[int, tuple[Any, Future[Any]]] = {}
_MATLAB_TASK_TIMEOUT_S = 60.0
_MATLAB_SHUTDOWN_TIMEOUT_S = 5.0
_MATLAB_CONTROL_LOCK = threading.Lock()
_MATLAB_CONTROL_SEQ = 0
_MATLAB_CONTROL_LATEST: dict[str, int] = {}
_MATLAB_WARMUP_FUTURE: Future[tuple[bool, str]] | None = None
_MATLAB_WARMUP_KEY: tuple[str, str] | None = None
_MATLAB_STALE_CONTEXT_PREFIX = "stale-context:"


def _matlab_runtime_is_shutting_down() -> bool:
    return _MATLAB_RUNTIME_SHUTDOWN.is_set() and _MATLAB_TASK_EXECUTOR.is_shutdown()


def _prepare_matlab_runtime_for_new_lifecycle() -> None:
    global _MATLAB_TASK_EXECUTOR
    with _MATLAB_RUNTIME_LOCK:
        executor = _MATLAB_TASK_EXECUTOR
        if not executor.is_shutdown():
            _MATLAB_RUNTIME_SHUTDOWN.clear()
            return
        if not executor.shutdown_complete() or not executor.is_idle():
            return
        _MATLAB_TASK_EXECUTOR = _DaemonSingleWorkerExecutor(
            thread_name_prefix="lfptp-matlab"
        )
        _MATLAB_RUNTIME_SHUTDOWN.clear()


def _runtime_key(paths: LocalizePaths) -> tuple[str, str]:
    return (
        str(paths.leaddbs_dir.expanduser().resolve()),
        str(paths.matlab_root.expanduser().resolve()),
    )


def _set_matlab_runtime_status(state: str, message: str) -> None:
    global _MATLAB_RUNTIME_STATE, _MATLAB_RUNTIME_MESSAGE
    with _MATLAB_RUNTIME_LOCK:
        if _matlab_runtime_is_shutting_down() and state in {"starting", "ready"}:
            return
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


def _failed_future(exc: BaseException) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_exception(exc)
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
    _prepare_matlab_runtime_for_new_lifecycle()
    if _matlab_runtime_is_shutting_down():
        return _completed_future((False, "MATLAB runtime is shutting down."))
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
    """Resolve the bundled or source `lfptensorpipe/anat/leaddbs` directory."""
    module_file = Path(__file__).resolve()
    package_relative = module_file.parents[2] / "anat" / "leaddbs"
    if package_relative.is_dir():
        return package_relative
    for parent in module_file.parents:
        candidate = parent / "src" / "lfptensorpipe" / "anat" / "leaddbs"
        if candidate.is_dir():
            return candidate
    return package_relative


def _ensure_matlab_engine_ready(
    paths: LocalizePaths,
    *,
    ensure_matlab_engine_fn: Callable[[Path], Any] | None = None,
    start_matlab_fn: Callable[[], Any] | None = None,
    matlab_functions_dir: Path | None = None,
) -> Any:
    global _MATLAB_RUNTIME_ENGINE, _MATLAB_RUNTIME_KEY
    global _MATLAB_RUNTIME_LAUNCH, _MATLAB_RUNTIME_LAUNCH_PENDING
    if _matlab_runtime_is_shutting_down():
        raise RuntimeError("MATLAB runtime is shutting down.")
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
    if _matlab_runtime_is_shutting_down():
        raise RuntimeError("MATLAB runtime is shutting down.")

    launch = None
    if start_matlab_fn is None:
        import matlab.engine

        with _MATLAB_RUNTIME_LOCK:
            if _matlab_runtime_is_shutting_down():
                raise RuntimeError("MATLAB runtime is shutting down.")
            _MATLAB_RUNTIME_LAUNCH_PENDING = True
            _MATLAB_RUNTIME_LAUNCH_PUBLISHED.clear()
        try:
            launch = matlab.engine.start_matlab(background=True)
        except BaseException:
            with _MATLAB_RUNTIME_LOCK:
                _MATLAB_RUNTIME_LAUNCH_PENDING = False
                _MATLAB_RUNTIME_LAUNCH_PUBLISHED.set()
            raise
        with _MATLAB_RUNTIME_LOCK:
            _MATLAB_RUNTIME_LAUNCH = launch
            _MATLAB_RUNTIME_LAUNCH_PENDING = False
            _MATLAB_RUNTIME_LAUNCH_PUBLISHED.set()
        try:
            eng = launch.result()
        except BaseException:
            with _MATLAB_RUNTIME_LOCK:
                if _MATLAB_RUNTIME_LAUNCH is launch:
                    _MATLAB_RUNTIME_LAUNCH = None
            raise
    else:
        eng = start_matlab_fn()

    fn_dir = matlab_functions_dir or _local_matlab_functions_dir()
    try:
        eng.addpath(eng.genpath(str(paths.leaddbs_dir)), nargout=0)
        if fn_dir.is_dir():
            eng.addpath(str(fn_dir), nargout=0)
    except Exception:
        try:
            eng.quit()
        except Exception:
            pass
        with _MATLAB_RUNTIME_LOCK:
            if _MATLAB_RUNTIME_LAUNCH is launch:
                _MATLAB_RUNTIME_LAUNCH = None
        raise

    with _MATLAB_RUNTIME_LOCK:
        stopping = _matlab_runtime_is_shutting_down()
        if not stopping:
            _MATLAB_RUNTIME_ENGINE = eng
            _MATLAB_RUNTIME_KEY = key
        if _MATLAB_RUNTIME_LAUNCH is launch:
            _MATLAB_RUNTIME_LAUNCH = None
    if stopping:
        try:
            eng.quit()
        except Exception:
            pass
        raise RuntimeError("MATLAB runtime is shutting down.")
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
        if _matlab_runtime_is_shutting_down():
            raise RuntimeError("MATLAB runtime is shutting down.")
        eng = _ensure_matlab_engine_ready(paths)
        try:
            result = fn(eng)
            if _matlab_runtime_is_shutting_down():
                raise RuntimeError("MATLAB runtime shut down before task completion.")
            return result
        except Exception as exc:
            if _matlab_runtime_is_shutting_down():
                raise RuntimeError("MATLAB runtime is shutting down.") from exc
            if not retry and _is_engine_disconnected_error(exc):
                retry = True
                _drop_matlab_engine()
                continue
            _set_matlab_runtime_status("failed", f"{exc}")
            raise


def submit_matlab_task(paths: LocalizePaths, fn: Callable[[Any], Any]) -> Future[Any]:
    _prepare_matlab_runtime_for_new_lifecycle()
    if _matlab_runtime_is_shutting_down():
        return _failed_future(RuntimeError("MATLAB runtime is shutting down."))
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
            if _matlab_runtime_is_shutting_down():
                return False, "MATLAB runtime is shutting down."
            message = f"MATLAB warmup failed: {exc}"
            _set_matlab_runtime_status("failed", message)
            return False, message
        if _matlab_runtime_is_shutting_down():
            return False, "MATLAB runtime is shutting down."
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
            if _matlab_runtime_is_shutting_down():
                return False, "MATLAB runtime is shutting down."
            message = f"MATLAB warmup failed: {exc}"
            _set_matlab_runtime_status("failed", message)
            return False, message
        if _matlab_runtime_is_shutting_down():
            return False, "MATLAB runtime is shutting down."
        _set_matlab_runtime_status("ready", "Ready")
        return True, "MATLAB ready."

    context_key = "runtime-control"
    future = _submit_latest_control_task(context_key, _reset)
    with _MATLAB_RUNTIME_LOCK:
        _MATLAB_WARMUP_FUTURE = future
        _MATLAB_WARMUP_KEY = _runtime_key(paths) if paths is not None else None
    return future


def shutdown_matlab_runtime(timeout_s: float = _MATLAB_SHUTDOWN_TIMEOUT_S) -> bool:
    """Stop only the MATLAB runtime owned by this app before GUI close."""
    global _MATLAB_RUNTIME_ENGINE, _MATLAB_RUNTIME_KEY, _MATLAB_RUNTIME_LAUNCH
    global _MATLAB_WARMUP_FUTURE, _MATLAB_WARMUP_KEY

    shutdown_executor = _MATLAB_TASK_EXECUTOR
    _MATLAB_RUNTIME_SHUTDOWN.set()
    _set_matlab_runtime_status("stopping", "Stopping...")
    shutdown_executor.cancel_pending()
    deadline = time.monotonic() + max(float(timeout_s), 0.0)

    with _MATLAB_RUNTIME_LOCK:
        launch_pending = _MATLAB_RUNTIME_LAUNCH_PENDING
    if launch_pending:
        remaining = max(deadline - time.monotonic(), 0.0)
        _MATLAB_RUNTIME_LAUNCH_PUBLISHED.wait(timeout=remaining)

    with _MATLAB_RUNTIME_LOCK:
        launch_pending = _MATLAB_RUNTIME_LAUNCH_PENDING
        launch = _MATLAB_RUNTIME_LAUNCH
        engine = _MATLAB_RUNTIME_ENGINE

    failure = ""
    engines_to_quit: list[Any] = []
    if launch_pending:
        failure = "MATLAB launch did not publish a cancellable handle in time."
    elif launch is not None:
        try:
            launch_cancelled = bool(launch.cancel())
        except Exception as exc:  # noqa: BLE001
            launch_cancelled = False
            failure = f"MATLAB launch cancellation failed: {exc}"
        if not launch_cancelled:
            while not failure:
                try:
                    launch_done = bool(launch.done())
                except Exception as exc:  # noqa: BLE001
                    failure = f"MATLAB launch status check failed: {exc}"
                    break
                if launch_done:
                    try:
                        launched_engine = launch.result()
                    except Exception:
                        launched_engine = None
                    if launched_engine is not None:
                        engines_to_quit.append(launched_engine)
                    break
                if time.monotonic() >= deadline:
                    failure = "MATLAB launch did not stop before the shutdown timeout."
                    break
                time.sleep(0.01)

    if engine is not None:
        engines_to_quit.append(engine)
    seen_engine_ids: set[int] = set()
    for owned_engine in engines_to_quit:
        engine_id = id(owned_engine)
        if engine_id in seen_engine_ids:
            continue
        seen_engine_ids.add(engine_id)
        with _MATLAB_RUNTIME_LOCK:
            quit_task = _MATLAB_ENGINE_QUIT_TASKS.get(engine_id)
            if quit_task is None:
                quit_future = _MATLAB_ENGINE_QUIT_EXECUTOR.submit(owned_engine.quit)
                _MATLAB_ENGINE_QUIT_TASKS[engine_id] = (owned_engine, quit_future)
            else:
                quit_future = quit_task[1]
        try:
            quit_future.result(timeout=max(deadline - time.monotonic(), 0.0))
        except FutureTimeoutError:
            failure = failure or "MATLAB Engine quit exceeded the shutdown timeout."
        except Exception as exc:  # noqa: BLE001
            with _MATLAB_RUNTIME_LOCK:
                _MATLAB_ENGINE_QUIT_TASKS.pop(engine_id, None)
            failure = failure or f"MATLAB Engine quit failed: {exc}"
        else:
            with _MATLAB_RUNTIME_LOCK:
                _MATLAB_ENGINE_QUIT_TASKS.pop(engine_id, None)

    if failure:
        _set_matlab_runtime_status("failed", f"MATLAB shutdown failed: {failure}")
        return False

    with _MATLAB_RUNTIME_LOCK:
        _MATLAB_RUNTIME_ENGINE = None
        _MATLAB_RUNTIME_KEY = None
        _MATLAB_RUNTIME_LAUNCH = None
        _MATLAB_WARMUP_FUTURE = None
        _MATLAB_WARMUP_KEY = None
        shutdown_executor.mark_shutdown_complete()
    _set_matlab_runtime_status("idle", "Not started.")
    return True


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

    return None


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
