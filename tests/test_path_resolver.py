"""Tests for record-scoped path resolution contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext


def test_record_stage_roots_are_resolved_by_contract(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    resolver = PathResolver(
        RecordContext(project_root=project_root, subject="sub-001", record="recA")
    )

    assert resolver.preproc_root == (
        project_root / "derivatives" / "lfptensorpipe" / "sub-001" / "recA" / "preproc"
    )
    assert resolver.tensor_root == (
        project_root / "derivatives" / "lfptensorpipe" / "sub-001" / "recA" / "tensor"
    )
    assert resolver.alignment_root == (
        project_root
        / "derivatives"
        / "lfptensorpipe"
        / "sub-001"
        / "recA"
        / "alignment"
    )
    assert resolver.features_root == (
        project_root / "derivatives" / "lfptensorpipe" / "sub-001" / "recA" / "features"
    )
    assert resolver.record_ui_state_path() == (
        project_root
        / "derivatives"
        / "lfptensorpipe"
        / "sub-001"
        / "recA"
        / "lfptensorpipe_ui_state.json"
    )


def test_preproc_step_validation_and_creation(tmp_path: Path) -> None:
    resolver = PathResolver(
        RecordContext(project_root=tmp_path, subject="sub-001", record="recA")
    )
    out = resolver.preproc_step_dir("raw", create=True)
    assert out.exists()

    with pytest.raises(ValueError):
        resolver.preproc_step_dir("unknown_step", create=False)


def test_metric_and_paradigm_key_validation_rejects_empty_values(
    tmp_path: Path,
) -> None:
    resolver = PathResolver(
        RecordContext(project_root=tmp_path, subject="sub-001", record="recA")
    )

    with pytest.raises(ValueError, match="Metric key cannot be empty"):
        resolver.tensor_metric_dir("   ", create=False)

    with pytest.raises(ValueError, match="Trial slug cannot be empty"):
        resolver.alignment_paradigm_dir("   ", create=False)


def test_ensure_record_roots_skips_tensor_unless_requested(tmp_path: Path) -> None:
    resolver = PathResolver(
        RecordContext(project_root=tmp_path, subject="sub-001", record="recA")
    )

    resolver.ensure_record_roots()
    assert resolver.preproc_root.exists()
    assert resolver.alignment_root.exists()
    assert resolver.features_root.exists()
    assert not resolver.tensor_root.exists()

    resolver.ensure_record_roots(include_tensor=True)
    assert resolver.tensor_root.exists()
