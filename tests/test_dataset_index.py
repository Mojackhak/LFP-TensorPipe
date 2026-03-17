"""Tests for dataset discovery and demo-root resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app.dataset_index import (
    _aggregate_states,
    _read_override_value,
    discover_records,
    discover_subjects,
    resolve_demo_data_root,
    resolve_demo_data_source_readonly,
)


def test_resolve_demo_data_root_from_override_file(tmp_path: Path) -> None:
    override = tmp_path / "AGENTS.override.md"
    override.write_text(
        (
            "## Local-only settings\n"
            "DEMO_DATA_ROOT = /tmp/my_demo_root\n"
            "DEMO_DATA_SOURCE_READONLY = /tmp/my_demo_ro\n"
        ),
        encoding="utf-8",
    )

    root = resolve_demo_data_root(override_file=override)
    assert root == Path("/tmp/my_demo_root").resolve()
    read_only = resolve_demo_data_source_readonly(override_file=override)
    assert read_only == Path("/tmp/my_demo_ro").resolve()


def test_discover_subjects_dedups_and_sorts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "derivatives" / "lfptensorpipe" / "sub-003").mkdir(parents=True)
    (project / "derivatives" / "leaddbs" / "sub-001").mkdir(parents=True)
    (project / "sourcedata" / "sub-002").mkdir(parents=True)
    (project / "rawdata" / "sub-001").mkdir(parents=True)

    subjects = discover_subjects(project)
    assert subjects == ["sub-001", "sub-002", "sub-003"]


def test_discover_records_dedups_and_sorts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    (project / "derivatives" / "lfptensorpipe" / subject / "run_a").mkdir(parents=True)
    (project / "derivatives" / "lfptensorpipe" / subject / "run_b").mkdir(parents=True)

    records = discover_records(project, subject)
    assert records == ["run_a", "run_b"]


def test_resolve_demo_data_root_uses_fallback_when_override_missing(
    tmp_path: Path,
) -> None:
    fallback = tmp_path / "fallback_demo"
    resolved = resolve_demo_data_root(
        override_file=tmp_path / "missing.override.md",
        fallback_root=fallback,
    )
    assert resolved == fallback.resolve()


def test_resolve_demo_data_root_uses_environment_when_override_not_passed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_root = tmp_path / "env_demo"
    env_read_only = tmp_path / "env_demo_ro"
    monkeypatch.setenv("DEMO_DATA_ROOT", str(env_root))
    monkeypatch.setenv("DEMO_DATA_SOURCE_READONLY", str(env_read_only))

    assert resolve_demo_data_root() == env_root.resolve()
    assert resolve_demo_data_source_readonly() == env_read_only.resolve()


def test_read_override_value_returns_none_for_empty_assignment(tmp_path: Path) -> None:
    override = tmp_path / "AGENTS.override.md"
    override.write_text("DEMO_DATA_ROOT =    \n", encoding="utf-8")
    assert _read_override_value(override, "DEMO_DATA_ROOT") is None


def test_resolve_demo_data_source_readonly_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    override = tmp_path / "AGENTS.override.md"
    override.write_text("DEMO_DATA_ROOT = /tmp/demo\n", encoding="utf-8")
    assert resolve_demo_data_source_readonly(override_file=override) is None


def test_aggregate_states_returns_gray_when_logs_missing_or_gray(
    tmp_path: Path,
) -> None:
    assert _aggregate_states([]) == "gray"
    assert (
        _aggregate_states([tmp_path / "missing_a.json", tmp_path / "missing_b.json"])
        == "gray"
    )
