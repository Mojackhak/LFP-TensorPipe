"""Run-log schema helpers, event history, and indicator derivation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

from .runlog_migrations import (
    RUNLOG_MIGRATION_META_KEY,
    RUNLOG_SCHEMA_KEY,
    RUNLOG_VERSION_KEY,
    stamp_run_log_metadata,
    upgrade_run_log_payload,
)

REQUIRED_LOG_KEYS = (
    "step",
    "completed",
    "timestamp_utc",
    "params",
    "input_path",
    "output_path",
    "message",
)
RUNLOG_HISTORY_KEY = "history"
RUNLOG_STATE_KEY = "state"


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


def validate_run_log(payload: dict[str, Any]) -> list[str]:
    """Return schema validation errors; empty list means valid."""
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

    if RUNLOG_SCHEMA_KEY in payload and not isinstance(payload[RUNLOG_SCHEMA_KEY], str):
        errors.append(f"Key '{RUNLOG_SCHEMA_KEY}' must be a string.")
    if RUNLOG_VERSION_KEY in payload:
        version = payload[RUNLOG_VERSION_KEY]
        if isinstance(version, bool) or not isinstance(version, int):
            errors.append(f"Key '{RUNLOG_VERSION_KEY}' must be an integer.")
        elif version < 1:
            errors.append(f"Key '{RUNLOG_VERSION_KEY}' must be >= 1.")
    if (
        RUNLOG_MIGRATION_META_KEY in payload
        and not isinstance(payload[RUNLOG_MIGRATION_META_KEY], dict)
    ):
        errors.append(f"Key '{RUNLOG_MIGRATION_META_KEY}' must be a JSON object (dict).")

    return errors


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        temp_path = Path(f.name)
    temp_path.replace(path)


def _write_validated_run_log_payload(path: Path, payload: dict[str, Any]) -> None:
    errors = validate_run_log(payload)
    if errors:
        raise ValueError("; ".join(errors))
    _write_json_payload(path, payload)


def write_run_log(path: str | Path, record: RunLogRecord) -> Path:
    """Write a run log to disk with UTF-8 JSON encoding."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = stamp_run_log_metadata(record.to_dict())
    _write_validated_run_log_payload(out_path, payload)
    return out_path


def _coerce_record_payload(record: RunLogRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, RunLogRecord):
        payload = record.to_dict()
    elif isinstance(record, dict):
        payload = {key: record.get(key) for key in REQUIRED_LOG_KEYS}
    else:
        raise TypeError("record must be RunLogRecord or dict.")
    errors = validate_run_log(payload)
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
) -> Path:
    """Append one event into `history` and mirror it into top-level summary.

    Top-level required fields always reflect the latest appended event.
    Optional state is stored under `state` and updated via deep-merge.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entry = _coerce_record_payload(record)

    existing: dict[str, Any] | None
    try:
        existing = read_run_log(out_path)
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
    existing_migration_meta = (
        existing.get(RUNLOG_MIGRATION_META_KEY)
        if isinstance(existing, dict)
        else None
    )
    payload = stamp_run_log_metadata(payload)
    if isinstance(existing_migration_meta, dict):
        payload[RUNLOG_MIGRATION_META_KEY] = deepcopy(existing_migration_meta)

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
    existing_migration_meta = existing.get(RUNLOG_MIGRATION_META_KEY)
    payload = stamp_run_log_metadata(payload)
    if isinstance(existing_migration_meta, dict):
        payload[RUNLOG_MIGRATION_META_KEY] = deepcopy(existing_migration_meta)

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
    if not in_path.exists():
        return None
    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("UI state must be a JSON object.")
    return payload


def write_ui_state(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write record-level UI state JSON."""
    if not isinstance(payload, dict):
        raise ValueError("UI state payload must be a dict.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def read_run_log_raw(path: str | Path) -> dict[str, Any] | None:
    """Read a run-log payload from disk without migration or write-back."""
    in_path = Path(path)
    if not in_path.exists():
        return None

    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Run log must be a JSON object.")

    errors = validate_run_log(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def upgrade_run_log_file(path: str | Path) -> tuple[dict[str, Any] | None, bool]:
    """Upgrade one run-log file to the latest supported schema."""
    in_path = Path(path)
    payload = read_run_log_raw(in_path)
    if payload is None:
        return None, False
    upgraded, changed = upgrade_run_log_payload(payload)
    _write_needed = changed or upgraded != payload
    if _write_needed:
        _write_validated_run_log_payload(in_path, upgraded)
    return upgraded, _write_needed


def read_run_log(path: str | Path) -> dict[str, Any] | None:
    """Read, upgrade, and validate a run-log payload; return None when missing."""
    payload, _ = upgrade_run_log_file(path)
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
    "RUNLOG_STATE_KEY",
    "RunLogRecord",
    "append_run_log_event",
    "indicator_from_log",
    "latest_run_log_entry",
    "read_run_log",
    "read_run_log_raw",
    "read_ui_state",
    "update_run_log_state",
    "upgrade_run_log_file",
    "validate_run_log",
    "write_run_log",
    "write_ui_state",
]
