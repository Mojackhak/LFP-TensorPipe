"""Run-log schema helpers, event history, and indicator derivation."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, ParamSpec, TypeVar

REQUIRED_LOG_KEYS = (
    "step",
    "completed",
    "timestamp_utc",
    "params",
    "input_path",
    "output_path",
    "message",
)
RUNLOG_SCHEMA_NAME = "lfptensorpipe.runlog"
RUNLOG_SCHEMA_KEY = "log_schema"
RUNLOG_VERSION_KEY = "log_version"
RUNLOG_SCHEMA_VERSION = 1
RUNLOG_HISTORY_KEY = "history"
RUNLOG_STATE_KEY = "state"

_RUN_LOG_READ_SNAPSHOT: ContextVar[dict[Path, dict[str, Any] | None] | None] = (
    ContextVar("run_log_read_snapshot", default=None)
)
_RUN_LOG_DERIVED_READ_SNAPSHOT: ContextVar[dict[tuple[str, Hashable], Any] | None] = (
    ContextVar("run_log_derived_read_snapshot", default=None)
)

_P = ParamSpec("_P")
_T = TypeVar("_T")


def cache_in_run_log_read_snapshot(
    key_builder: Callable[_P, Hashable | None],
    *,
    copy_result: bool = False,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Reuse one pure derived value inside an explicit read snapshot.

    A `None` key bypasses reuse. Mutable results must request independent copies.
    Calls outside a snapshot retain the decorated function's original behavior.
    """

    def decorator(function: Callable[_P, _T]) -> Callable[_P, _T]:
        namespace = f"{function.__module__}.{function.__qualname__}"

        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            snapshot = _RUN_LOG_DERIVED_READ_SNAPSHOT.get()
            if snapshot is None:
                return function(*args, **kwargs)
            key = key_builder(*args, **kwargs)
            if key is None:
                return function(*args, **kwargs)
            cache_key = (namespace, key)
            if cache_key not in snapshot:
                snapshot[cache_key] = function(*args, **kwargs)
            result = snapshot[cache_key]
            return deepcopy(result) if copy_result else result

        return wrapped

    return decorator


@dataclass(frozen=True)
class RunLogRecord:
    """Structured run-log payload for `lfptensorpipe_log.json`."""

    step: str
    completed: bool
    params: dict[str, Any] = field(default_factory=dict)
    input_path: str = ""
    output_path: str = ""
    message: str = ""
    timestamp_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serialized log payload with UTC timestamp."""
        timestamp = self.timestamp_utc or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return {
            "step": self.step,
            "completed": self.completed,
            "timestamp_utc": timestamp,
            "params": self.params,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "message": self.message,
        }


def _validate_run_log_event(payload: dict[str, Any]) -> list[str]:
    """Return validation errors for one event without envelope metadata."""
    errors: list[str] = []
    for key in REQUIRED_LOG_KEYS:
        if key not in payload:
            errors.append(f"Missing required key: {key}")

    if "completed" in payload and not isinstance(payload["completed"], bool):
        errors.append("Key 'completed' must be a bool.")
    if "params" in payload and not isinstance(payload["params"], dict):
        errors.append("Key 'params' must be a JSON object (dict).")

    for text_key in ("step", "timestamp_utc", "input_path", "output_path", "message"):
        if text_key in payload and not isinstance(payload[text_key], str):
            errors.append(f"Key '{text_key}' must be a string.")

    return errors


def validate_run_log(payload: dict[str, Any]) -> list[str]:
    """Return fixed v0.2.0 envelope validation errors."""
    errors = _validate_run_log_event(payload)

    if RUNLOG_SCHEMA_KEY not in payload:
        errors.append(f"Missing required key: {RUNLOG_SCHEMA_KEY}")
    elif payload[RUNLOG_SCHEMA_KEY] != RUNLOG_SCHEMA_NAME:
        errors.append(f"Key '{RUNLOG_SCHEMA_KEY}' must equal {RUNLOG_SCHEMA_NAME!r}.")
    if RUNLOG_VERSION_KEY not in payload:
        errors.append(f"Missing required key: {RUNLOG_VERSION_KEY}")
    else:
        version = payload[RUNLOG_VERSION_KEY]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != RUNLOG_SCHEMA_VERSION
        ):
            errors.append(
                f"Key '{RUNLOG_VERSION_KEY}' must equal {RUNLOG_SCHEMA_VERSION}."
            )

    return errors


def _stamp_run_log_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(payload)
    out[RUNLOG_SCHEMA_KEY] = RUNLOG_SCHEMA_NAME
    out[RUNLOG_VERSION_KEY] = RUNLOG_SCHEMA_VERSION
    return out


def _json_temporary_pid(path: Path, candidate: Path) -> int | None:
    prefix = f".{path.name}.pid-"
    suffix = ".tmp"
    name = candidate.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    pid_text, separator, token = name[len(prefix) : -len(suffix)].partition(".")
    if not separator or not token or not pid_text.isdigit():
        return None
    return int(pid_text)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _cleanup_dead_json_temporaries(path: Path) -> None:
    pattern = f".{path.name}.pid-*.tmp"
    for candidate in path.parent.glob(pattern):
        pid = _json_temporary_pid(path, candidate)
        if pid is None or _pid_is_alive(pid):
            continue
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    _cleanup_dead_json_temporaries(path)
    target_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.pid-{os.getpid()}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = Path(f.name)
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if target_mode is not None:
            temp_path.chmod(target_mode)
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _write_validated_run_log_payload(path: Path, payload: dict[str, Any]) -> None:
    errors = validate_run_log(payload)
    if errors:
        raise ValueError("; ".join(errors))
    _write_json_payload(path, payload)


def write_run_log(path: str | Path, record: RunLogRecord) -> Path:
    """Write a run log to disk with UTF-8 JSON encoding."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _stamp_run_log_metadata(record.to_dict())
    _write_validated_run_log_payload(out_path, payload)
    return out_path


def _coerce_record_payload(record: RunLogRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, RunLogRecord):
        payload = record.to_dict()
    elif isinstance(record, dict):
        payload = {key: record.get(key) for key in REQUIRED_LOG_KEYS}
    else:
        raise TypeError("record must be RunLogRecord or dict.")
    errors = _validate_run_log_event(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def _summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in REQUIRED_LOG_KEYS}


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def append_run_log_event(
    path: str | Path,
    record: RunLogRecord | dict[str, Any],
    *,
    state_patch: dict[str, Any] | None = None,
    source_path: str | Path | None = None,
) -> Path:
    """Append one event into `history` and mirror it into top-level summary.

    Top-level required fields always reflect the latest appended event.
    Optional state is stored under `state` and updated via deep-merge.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entry = _coerce_record_payload(record)

    existing_path = Path(source_path) if source_path is not None else out_path
    existing: dict[str, Any] | None
    try:
        existing = read_run_log(existing_path)
    except Exception:
        existing = None

    history: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    if isinstance(existing, dict):
        existing_history = existing.get(RUNLOG_HISTORY_KEY)
        if isinstance(existing_history, list):
            for item in existing_history:
                if isinstance(item, dict):
                    try:
                        history.append(_coerce_record_payload(item))
                    except Exception:
                        continue
        else:
            try:
                history.append(_coerce_record_payload(_summary_from_payload(existing)))
            except Exception:
                pass
        existing_state = existing.get(RUNLOG_STATE_KEY)
        if isinstance(existing_state, dict):
            state = deepcopy(existing_state)

    history.append(dict(entry))
    if state_patch is not None:
        if not isinstance(state_patch, dict):
            raise ValueError("state_patch must be a dict when provided.")
        state = _merge_dict(state, state_patch)

    payload = dict(entry)
    payload[RUNLOG_HISTORY_KEY] = history
    if state:
        payload[RUNLOG_STATE_KEY] = state
    payload = _stamp_run_log_metadata(payload)

    _write_validated_run_log_payload(out_path, payload)
    return out_path


def update_run_log_state(
    path: str | Path,
    *,
    state_patch: dict[str, Any],
) -> Path:
    """Deep-merge `state` without mutating history or top-level summary fields."""
    if not isinstance(state_patch, dict):
        raise ValueError("state_patch must be a dict.")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_run_log(out_path)
    if not isinstance(existing, dict):
        raise FileNotFoundError(f"Run log not found: {out_path}")

    state = existing.get(RUNLOG_STATE_KEY)
    merged_state = _merge_dict(state if isinstance(state, dict) else {}, state_patch)

    payload = deepcopy(existing)
    payload[RUNLOG_STATE_KEY] = merged_state
    payload.pop("migration_meta", None)
    payload = _stamp_run_log_metadata(payload)

    _write_validated_run_log_payload(out_path, payload)
    return out_path


def latest_run_log_entry(
    payload: dict[str, Any] | None,
    *,
    step: str | None = None,
) -> dict[str, Any] | None:
    """Return latest valid event, optionally filtered by `step`."""
    if not isinstance(payload, dict):
        return None

    candidates: list[dict[str, Any]] = []
    history = payload.get(RUNLOG_HISTORY_KEY)
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                try:
                    candidates.append(_coerce_record_payload(item))
                except Exception:
                    continue
    if not candidates:
        try:
            candidates = [_coerce_record_payload(_summary_from_payload(payload))]
        except Exception:
            candidates = []
    if not candidates:
        return None

    if step is None:
        return dict(candidates[-1])
    target = str(step).strip()
    if not target:
        return dict(candidates[-1])
    for item in reversed(candidates):
        if str(item.get("step", "")).strip() == target:
            return dict(item)
    return None


def read_ui_state(path: str | Path) -> dict[str, Any] | None:
    """Read record-level UI state JSON; return None when missing."""
    in_path = Path(path)
    _cleanup_dead_json_temporaries(in_path)
    if not in_path.exists():
        return None
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("UI state must be a JSON object.")
    return payload


def write_ui_state(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write record-level UI state JSON."""
    if not isinstance(payload, dict):
        raise ValueError("UI state payload must be a dict.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_payload(out_path, payload)
    return out_path


@contextmanager
def run_log_read_snapshot() -> Iterator[None]:
    """Reuse validated logs and pure derived reads in one read-only operation."""
    if _RUN_LOG_READ_SNAPSHOT.get() is not None:
        yield
        return

    token = _RUN_LOG_READ_SNAPSHOT.set({})
    derived_token = _RUN_LOG_DERIVED_READ_SNAPSHOT.set({})
    try:
        yield
    finally:
        _RUN_LOG_DERIVED_READ_SNAPSHOT.reset(derived_token)
        _RUN_LOG_READ_SNAPSHOT.reset(token)


def read_run_log(path: str | Path) -> dict[str, Any] | None:
    """Read and validate a fixed v0.2.0 run-log envelope without write-back."""
    in_path = Path(path)
    snapshot = _RUN_LOG_READ_SNAPSHOT.get()
    if snapshot is not None and in_path in snapshot:
        return deepcopy(snapshot[in_path])

    _cleanup_dead_json_temporaries(in_path)
    if not in_path.exists():
        if snapshot is not None:
            snapshot[in_path] = None
        return None

    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Run log must be a JSON object.")

    errors = validate_run_log(payload)
    if errors:
        raise ValueError("; ".join(errors))
    if snapshot is not None:
        snapshot[in_path] = payload
        return deepcopy(payload)
    return payload


def indicator_from_log(path: str | Path) -> str:
    """Map log state to one of: `gray`, `yellow`, `green`."""
    payload = read_run_log(path)
    if payload is None:
        return "gray"
    return "green" if payload["completed"] else "yellow"


__all__ = [
    "REQUIRED_LOG_KEYS",
    "RUNLOG_HISTORY_KEY",
    "RUNLOG_SCHEMA_KEY",
    "RUNLOG_SCHEMA_NAME",
    "RUNLOG_SCHEMA_VERSION",
    "RUNLOG_STATE_KEY",
    "RUNLOG_VERSION_KEY",
    "RunLogRecord",
    "append_run_log_event",
    "indicator_from_log",
    "latest_run_log_entry",
    "read_run_log",
    "read_ui_state",
    "update_run_log_state",
    "validate_run_log",
    "write_run_log",
    "write_ui_state",
]
