"""Tests for shared YAML config storage behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app.config_store import (
    AppConfigStore,
    CONFIG_FILE_BASENAMES,
    FEATURES_CONFIG_FILENAME,
    STATE_FILE_BASENAMES,
)


def test_ensure_core_files_creates_expected_yamls(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    for filename in CONFIG_FILE_BASENAMES:
        assert (repo_root / "configs" / filename).exists()
    for filename in STATE_FILE_BASENAMES:
        assert (repo_root / "state" / filename).exists()

    expected_record_defaults = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1] / "src" / "configs" / "record.yml"
        ).read_text(encoding="utf-8")
    )
    assert store.read_yaml("record.yml", default={}) == expected_record_defaults


def test_write_and_read_yaml_roundtrip(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    store = AppConfigStore(repo_root=repo_root)
    payload = {"alpha": 1, "beta": {"nested": True}}
    store.write_yaml("tensor.yml", payload)

    loaded = store.read_yaml("tensor.yml", default={})
    assert loaded == payload


def test_ensure_core_files_bootstraps_features_defaults(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    loaded = store.read_yaml(FEATURES_CONFIG_FILENAME, default={})

    assert isinstance(loaded, dict)
    assert "derive_param_cfg" in loaded
    assert (repo_root / "configs" / FEATURES_CONFIG_FILENAME).exists()


def test_recent_projects_fifo_and_pruning(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    existing = []
    for name in ("proj_a", "proj_b", "proj_c"):
        project = tmp_path / name
        project.mkdir()
        existing.append(project)

    store.append_recent_project(existing[0], max_items=2)
    store.append_recent_project(existing[1], max_items=2)
    values = store.append_recent_project(existing[2], max_items=2)

    expected = [str(existing[2].resolve()), str(existing[1].resolve())]
    assert values == expected

    missing = tmp_path / "proj_missing"
    store.write_yaml(
        "recent_projects.yml",
        {"recent_projects": [str(missing.resolve()), str(existing[0].resolve())]},
    )
    assert store.load_recent_projects() == [str(existing[0].resolve())]


def test_path_for_rejects_non_yaml_suffix(tmp_path: Path) -> None:
    store = AppConfigStore(repo_root=tmp_path / "repo")
    with pytest.raises(ValueError, match="must end with '.yml'"):
        store.path_for("tensor.json")


def test_path_for_routes_recent_projects_to_state_dir(tmp_path: Path) -> None:
    store = AppConfigStore(repo_root=tmp_path / "repo")
    assert (
        store.path_for("recent_projects.yml")
        == tmp_path / "repo" / "state" / "recent_projects.yml"
    )


def test_read_yaml_returns_deepcopied_default_for_missing_and_empty(
    tmp_path: Path,
) -> None:
    store = AppConfigStore(repo_root=tmp_path / "repo")
    missing_default = {"items": []}

    loaded_missing = store.read_yaml("tensor.yml", missing_default)
    assert loaded_missing == {"items": []}
    assert loaded_missing is not missing_default
    loaded_missing["items"].append("x")
    assert missing_default == {"items": []}

    empty_default = {"paradigms": []}
    store.path_for("alignment.yml").write_text("", encoding="utf-8")
    loaded_empty = store.read_yaml("alignment.yml", empty_default)
    assert loaded_empty == {"paradigms": []}
    assert loaded_empty is not empty_default


def test_load_recent_projects_skips_entries_that_raise_oserror(
    tmp_path: Path,
) -> None:
    class _ResolveFailStore(AppConfigStore):
        def _resolve_recent_project_path(self, value: str | Path) -> Path:
            if str(value).startswith("/boom/"):
                raise OSError("simulated path resolution failure")
            return super()._resolve_recent_project_path(value)

    store = _ResolveFailStore(repo_root=tmp_path / "repo")
    good = tmp_path / "good_project"
    good.mkdir()
    store.write_yaml(
        "recent_projects.yml",
        {"recent_projects": [str(good.resolve()), "/boom/project"]},
    )
    assert store.load_recent_projects() == [str(good.resolve())]


def test_append_recent_project_dedups_duplicates_from_loaded_state(
    tmp_path: Path,
) -> None:
    proj_a = tmp_path / "proj_a"
    proj_b = tmp_path / "proj_b"
    proj_c = tmp_path / "proj_c"
    proj_a.mkdir()
    proj_b.mkdir()
    proj_c.mkdir()

    a = str(proj_a.resolve())
    b = str(proj_b.resolve())

    class _DuplicateStateStore(AppConfigStore):
        def load_recent_projects(self) -> list[str]:
            return [a, a, b]

    store = _DuplicateStateStore(repo_root=tmp_path / "repo")

    values = store.append_recent_project(proj_c, max_items=5)
    assert values == [str(proj_c.resolve()), a, b]
