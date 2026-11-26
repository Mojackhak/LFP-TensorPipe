"""Cached runtime software identity for logs, provenance sidecars, and exports."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

PACKAGE_NAME = "lfp-tensorpipe"
TRACKED_PACKAGES = (
    PACKAGE_NAME,
    "numpy",
    "scipy",
    "pandas",
    "mne",
    "mne-connectivity",
    "specparam",
    "fooof",
    "autoreject",
    "scikit-learn",
)
_GIT_TAG_VERSION_RE = re.compile(r"^v?(\d+\.\d+.*)$")


def _source_repo_root() -> Path | None:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) <= 4:
        return None
    candidate = resolved.parents[4]
    if (candidate / ".git").exists():
        return candidate
    return None


def _frozen_build_info() -> dict[str, Any] | None:
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    path = Path(meipass) / "build-info.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _git_output(args: list[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _resolve_version(
    *,
    packages: dict[str, str],
    git_describe: str | None,
    build_info: dict[str, Any] | None,
) -> str:
    if isinstance(git_describe, str) and git_describe:
        match = _GIT_TAG_VERSION_RE.match(git_describe)
        if match:
            return match.group(1)
    if isinstance(build_info, dict):
        built_version = build_info.get("version")
        if isinstance(built_version, str) and built_version.strip():
            return built_version.strip()
    installed = packages.get(PACKAGE_NAME)
    if isinstance(installed, str) and installed.strip():
        return installed.strip()
    return "unknown"


@lru_cache(maxsize=1)
def _cached_software_identity() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass

    build_info = _frozen_build_info()
    git_commit: str | None = None
    git_describe: str | None = None
    working_tree_modified: bool | None = None

    repo_root = None if getattr(sys, "_MEIPASS", None) else _source_repo_root()
    if repo_root is not None and shutil.which("git"):
        git_commit = _git_output(["rev-parse", "HEAD"], cwd=repo_root)
        git_describe = _git_output(["describe", "--tags", "--always"], cwd=repo_root)
        status = _git_output(
            [
                "status",
                "--porcelain",
                "--",
                "src/lfptensorpipe",
                "src/configs",
                "pyproject.toml",
            ],
            cwd=repo_root,
        )
        if status is not None:
            working_tree_modified = bool(status)
    elif isinstance(build_info, dict):
        source_commit = build_info.get("source_commit")
        if isinstance(source_commit, str) and source_commit.strip():
            git_commit = source_commit.strip()
        source_dirty = build_info.get("source_dirty")
        if isinstance(source_dirty, bool):
            working_tree_modified = source_dirty

    result: dict[str, Any] = {
        "name": PACKAGE_NAME,
        "version": _resolve_version(
            packages=packages,
            git_describe=git_describe,
            build_info=build_info,
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
    if git_commit:
        result["git_commit"] = git_commit
    if git_describe:
        result["git_describe"] = git_describe
    if working_tree_modified is not None:
        result["working_tree_modified"] = working_tree_modified
    return result


def software_identity() -> dict[str, Any]:
    """Return a fresh copy of the cached runtime software identity."""
    return deepcopy(_cached_software_identity())


def clear_software_identity_cache() -> None:
    """Clear the cached software identity (intended for tests)."""
    _cached_software_identity.cache_clear()


__all__ = [
    "PACKAGE_NAME",
    "TRACKED_PACKAGES",
    "clear_software_identity_cache",
    "software_identity",
]
