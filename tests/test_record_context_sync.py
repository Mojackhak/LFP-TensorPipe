"""Tests for record-context stage-state synchronization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app.record_log_migration import upgrade_record_run_logs
from lfptensorpipe.app.runlog_migrations import (
    RUNLOG_SCHEMA_KEY,
    RUNLOG_SCHEMA_NAME,
    RUNLOG_VERSION_KEY,
    register_run_log_migration,
)
from lfptensorpipe.app.dataset_index import scan_stage_states
from lfptensorpipe.app.runlog_store import RunLogRecord, write_run_log


def test_scan_stage_states_from_record_logs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    base = project / "derivatives" / "lfptensorpipe" / subject / record

    write_run_log(
        base / "preproc" / "finish" / "lfptensorpipe_log.json",
        RunLogRecord(step="finish", completed=True),
    )
    write_run_log(
        base / "tensor" / "raw_power" / "lfptensorpipe_log.json",
        RunLogRecord(step="raw_power", completed=True),
    )
    write_run_log(
        base / "alignment" / "gait" / "lfptensorpipe_log.json",
        RunLogRecord(step="run", completed=False),
    )
    write_run_log(
        base / "features" / "gait" / "lfptensorpipe_log.json",
        RunLogRecord(step="derive", completed=True),
    )

    states = scan_stage_states(project, subject, record)
    assert states == {
        "preproc": "green",
        "tensor": "green",
        "alignment": "yellow",
        "features": "green",
    }


def test_scan_stage_states_defaults_to_gray_when_logs_missing(tmp_path: Path) -> None:
    states = scan_stage_states(tmp_path, "sub-001", "runA")
    assert states == {
        "preproc": "gray",
        "tensor": "gray",
        "alignment": "gray",
        "features": "gray",
    }


def test_scan_stage_states_prefers_green_metric_over_stale_stage_yellow(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    base = project / "derivatives" / "lfptensorpipe" / subject / record

    write_run_log(
        base / "tensor" / "raw_power" / "lfptensorpipe_log.json",
        RunLogRecord(step="raw_power", completed=True),
    )
    write_run_log(
        base / "tensor" / "lfptensorpipe_log.json",
        RunLogRecord(step="build_tensor", completed=False),
    )

    states = scan_stage_states(project, subject, record)
    assert states["tensor"] == "green"


def test_upgrade_record_run_logs_rewrites_selected_record_logs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    step = "test_record_upgrade_step"
    register_run_log_migration(
        step,
        1,
        2,
        lambda payload: {
            **payload,
            "params": {**payload.get("params", {}), "migrated": True},
        },
    )

    log_path = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "custom"
        / "lfptensorpipe_log.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        (
            "{\n"
            f'  "{RUNLOG_SCHEMA_KEY}": "{RUNLOG_SCHEMA_NAME}",\n'
            f'  "{RUNLOG_VERSION_KEY}": 1,\n'
            f'  "step": "{step}",\n'
            '  "completed": true,\n'
            '  "timestamp_utc": "2026-03-15T00:00:00Z",\n'
            '  "params": {},\n'
            '  "input_path": "",\n'
            '  "output_path": "",\n'
            '  "message": "legacy"\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    summary = upgrade_record_run_logs(project, subject, record)

    assert summary.scanned_count == 1
    assert summary.upgraded_count == 1
    assert summary.failed_count == 0
    assert f'"{RUNLOG_VERSION_KEY}": 2' in log_path.read_text(encoding="utf-8")
