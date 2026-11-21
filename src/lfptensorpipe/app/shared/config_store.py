"""YAML-backed application config storage for LFP-TensorPipe."""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

APP_STORAGE_DIRNAME = "LFP-TensorPipe"
CONFIG_DEFAULTS_PACKAGE = "lfptensorpipe.resources.config_defaults"
FEATURES_CONFIG_FILENAME = "features.yml"
LOCALIZATION_CONFIG_FILENAME = "localization.yml"

CONFIG_FILE_BASENAMES = (
    "alignment.yml",
    "alignment_preview.yml",
    FEATURES_CONFIG_FILENAME,
    "features_plot.yml",
    LOCALIZATION_CONFIG_FILENAME,
    "paths.yml",
    "preproc.yml",
    "record.yml",
    "tensor.yml",
)
STATE_FILE_BASENAMES = ("recent_projects.yml",)
MANAGED_FILE_SUBDIRS = {
    **{filename: "configs" for filename in CONFIG_FILE_BASENAMES},
    **{filename: "state" for filename in STATE_FILE_BASENAMES},
}


def _default_app_root() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_STORAGE_DIRNAME
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / APP_STORAGE_DIRNAME
        return home / "AppData" / "Local" / APP_STORAGE_DIRNAME
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg_config_home:
        return Path(xdg_config_home) / APP_STORAGE_DIRNAME
    return home / ".config" / APP_STORAGE_DIRNAME


def _repo_default_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def _default_template_text(filename: str) -> str:
    try:
        resource = resources.files(CONFIG_DEFAULTS_PACKAGE).joinpath(filename)
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    source_path = _repo_default_config_dir() / filename
    if source_path.is_file():
        return source_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Default config template not found for {filename}")


def _default_template_payload(filename: str) -> Any:
    payload = yaml.safe_load(_default_template_text(filename))
    return {} if payload is None else payload


class AppConfigStore:
    """Read/write app configs under the platform user-storage root."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self.app_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else _default_app_root()
        )
        self.config_dir = self.app_root / "configs"
        self.state_dir = self.app_root / "state"
        self.cache_dir = self.app_root / "cache"
        self.logs_dir = self.app_root / "logs"
        for directory in (
            self.app_root,
            self.config_dir,
            self.state_dir,
            self.cache_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, filename: str) -> Path:
        """Return the managed absolute path for one logical YAML filename."""
        if not filename.endswith(".yml"):
            raise ValueError("Config filename must end with '.yml'.")
        subdir = MANAGED_FILE_SUBDIRS.get(filename)
        if subdir is None:
            raise ValueError(f"Unknown managed config filename: {filename}")
        if subdir == "configs":
            return self.config_dir / filename
        return self.state_dir / filename

    def ensure_core_files(self) -> None:
        """Create any missing managed YAML files from packaged defaults."""
        for filename in (*CONFIG_FILE_BASENAMES, *STATE_FILE_BASENAMES):
            path = self.path_for(filename)
            if path.exists():
                continue
            self.write_yaml(filename, _default_template_payload(filename))

    def read_yaml(self, filename: str, default: Any) -> Any:
        """Load YAML data or return a deep-copied default when missing/empty."""
        path = self.path_for(filename)
        if not path.exists():
            return deepcopy(default)

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return deepcopy(default)
        return data

    def write_yaml(self, filename: str, data: Any) -> Path:
        """Persist YAML data using UTF-8 encoding and atomic replacement."""
        path = self.path_for(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                temp_path = Path(f.name)
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)
            temp_path.replace(path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return path

    def _resolve_recent_project_path(self, value: str | Path) -> Path:
        """Resolve one recent-project entry to an absolute path."""
        return Path(value).expanduser().resolve()

    def load_recent_projects(self) -> list[str]:
        """Return recent project paths with missing paths pruned."""
        return self._refresh_recent_projects()

    def append_recent_project(
        self, project_path: str | Path, max_items: int = 5
    ) -> list[str]:
        """Add one project path to FIFO recent-projects and persist the result."""
        return self._refresh_recent_projects(
            newest=self._resolve_recent_project_path(project_path), max_items=max_items
        )

    def _refresh_recent_projects(
        self, *, newest: Path | None = None, max_items: int | None = None
    ) -> list[str]:
        payload = self.read_yaml("recent_projects.yml", {})
        if not isinstance(payload, dict):
            payload = {}
        previous = payload.get("recent_projects", [])
        paths = [str(newest)] if newest is not None and newest.exists() else []
        for value in previous:
            if max_items is not None and len(paths) >= max_items:
                break
            try:
                path = self._resolve_recent_project_path(value)
            except OSError:
                continue
            entry = str(path)
            if entry not in paths and path.exists():
                paths.append(entry)

        if paths != previous or (
            newest is not None and "recent_projects" not in payload
        ):
            payload["recent_projects"] = paths
            self.write_yaml("recent_projects.yml", payload)
        return paths
