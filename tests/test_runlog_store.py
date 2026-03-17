"""Tests for run-log schema and indicator derivation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app.runlog_migrations import (
    RUNLOG_SCHEMA_KEY,
    RUNLOG_SCHEMA_NAME,
    RUNLOG_UPGRADES_KEY,
    RUNLOG_VERSION_KEY,
    register_run_log_migration,
)
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    append_run_log_event,
    indicator_from_log,
    latest_run_log_entry,
    read_run_log,
    read_run_log_raw,
    read_ui_state,
    validate_run_log,
    write_ui_state,
    write_run_log,
)


def test_indicator_states_follow_log_contract(tmp_path: Path) -> None:
    log_path = tmp_path / "lfptensorpipe_log.json"

    assert indicator_from_log(log_path) == "gray"

    write_run_log(
        log_path,
        RunLogRecord(
            step="filter", completed=False, input_path="in", output_path="out"
        ),
    )
    assert indicator_from_log(log_path) == "yellow"

    write_run_log(
        log_path,
        RunLogRecord(step="filter", completed=True, input_path="in", output_path="out"),
    )
    assert indicator_from_log(log_path) == "green"


def test_write_run_log_stamps_schema_metadata(tmp_path: Path) -> None:
    log_path = tmp_path / "lfptensorpipe_log.json"

    write_run_log(log_path, RunLogRecord(step="filter", completed=True))

    payload = read_run_log_raw(log_path)
    assert payload is not None
    assert payload[RUNLOG_SCHEMA_KEY] == RUNLOG_SCHEMA_NAME
    assert payload[RUNLOG_VERSION_KEY] == 1


def test_read_run_log_stamps_base_schema_metadata_on_first_read(tmp_path: Path) -> None:
    log_path = tmp_path / "lfptensorpipe_log.json"
    log_path.write_text(
        json.dumps(
            {
                "step": "filter",
                "completed": True,
                "timestamp_utc": "2026-03-15T00:00:00Z",
                "params": {},
                "input_path": "",
                "output_path": "",
                "message": "ok",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = read_run_log(log_path)

    assert payload is not None
    assert payload[RUNLOG_SCHEMA_KEY] == RUNLOG_SCHEMA_NAME
    assert payload[RUNLOG_VERSION_KEY] == 1
    persisted = read_run_log_raw(log_path)
    assert persisted == payload


def test_read_run_log_applies_registered_migration_and_is_idempotent(
    tmp_path: Path,
) -> None:
    step = "test_runlog_store_upgrade_step"
    register_run_log_migration(
        step,
        1,
        2,
        lambda payload: {
            **payload,
            "params": {**payload.get("params", {}), "migrated": True},
            "message": "upgraded",
        },
    )

    log_path = tmp_path / "lfptensorpipe_log.json"
    log_path.write_text(
        json.dumps(
            {
                RUNLOG_SCHEMA_KEY: RUNLOG_SCHEMA_NAME,
                RUNLOG_VERSION_KEY: 1,
                "step": step,
                "completed": True,
                "timestamp_utc": "2026-03-15T00:00:00Z",
                "params": {"value": 1},
                "input_path": "",
                "output_path": "",
                "message": "legacy",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    upgraded = read_run_log(log_path)
    assert upgraded is not None
    assert upgraded[RUNLOG_VERSION_KEY] == 2
    assert upgraded["params"]["migrated"] is True
    assert upgraded["migration_meta"][RUNLOG_UPGRADES_KEY] == [
        {"from_version": 1, "to_version": 2}
    ]

    persisted_once = log_path.read_text(encoding="utf-8")
    upgraded_again = read_run_log(log_path)
    assert upgraded_again == upgraded
    assert log_path.read_text(encoding="utf-8") == persisted_once


def test_read_run_log_validates_required_schema(tmp_path: Path) -> None:
    log_path = tmp_path / "bad_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({"step": "raw", "completed": True}, f)

    with pytest.raises(ValueError):
        read_run_log(log_path)


def test_validate_run_log_reports_type_errors() -> None:
    payload = {
        "step": 1,
        "completed": "yes",
        "timestamp_utc": "2026-02-13T00:00:00Z",
        "params": [],
        "input_path": 10,
        "output_path": "out",
        "message": "ok",
    }
    errors = validate_run_log(payload)
    assert "Key 'completed' must be a bool." in errors
    assert "Key 'params' must be a JSON object (dict)." in errors
    assert "Key 'step' must be a string." in errors
    assert "Key 'input_path' must be a string." in errors


def test_write_run_log_raises_when_payload_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        write_run_log(
            tmp_path / "bad_log.json",
            RunLogRecord(step=1, completed=True),  # type: ignore[arg-type]
        )


def test_read_run_log_raises_when_json_payload_not_object(tmp_path: Path) -> None:
    log_path = tmp_path / "bad_shape.json"
    log_path.write_text('["not-an-object"]', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_run_log(log_path)


def test_append_run_log_event_keeps_history_and_latest_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "lfptensorpipe_log.json"
    append_run_log_event(
        log_path,
        RunLogRecord(step="run_align_epochs", completed=True, message="run"),
        state_patch={"trial_config": {"method": "linear_warper"}},
    )
    append_run_log_event(
        log_path,
        RunLogRecord(step="build_raw_table", completed=False, message="finish failed"),
    )
    payload = read_run_log(log_path)
    assert payload is not None
    assert payload["step"] == "build_raw_table"
    assert payload["completed"] is False
    assert isinstance(payload.get("history"), list)
    assert len(payload["history"]) == 2
    state = payload.get("state", {})
    assert isinstance(state, dict)
    assert state.get("trial_config", {}).get("method") == "linear_warper"
    latest_align = latest_run_log_entry(payload, step="run_align_epochs")
    assert latest_align is not None
    assert latest_align["step"] == "run_align_epochs"


def test_ui_state_read_write_roundtrip(tmp_path: Path) -> None:
    state_path = tmp_path / "lfptensorpipe_ui_state.json"
    write_ui_state(state_path, {"localize": {"match": {"completed": True}}})
    loaded = read_ui_state(state_path)
    assert loaded == {"localize": {"match": {"completed": True}}}
