"""Path helpers for the PD paper workspace."""

from __future__ import annotations

from pathlib import Path

from paper.pd.specs import DEFAULT_PROJECT_ROOT


def is_appledouble_path(path: str | Path) -> bool:
    """Return whether a path is a macOS AppleDouble metadata companion."""
    return Path(path).name.startswith("._")


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    """Return the PD project root."""
    if project_root is None:
        return DEFAULT_PROJECT_ROOT
    return Path(project_root).expanduser().resolve()


def derivatives_root(project_root: str | Path | None = None) -> Path:
    """Return the lfptensorpipe derivatives root."""
    return resolve_project_root(project_root) / "derivatives" / "lfptensorpipe"


def summary_root(project_root: str | Path | None = None, *, create: bool = False) -> Path:
    """Return the summary output root."""
    root = resolve_project_root(project_root) / "summary"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def summary_table_root(
    project_root: str | Path | None = None,
    *,
    create: bool = False,
) -> Path:
    """Return the summary table output root."""
    root = summary_root(project_root, create=create) / "table"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root
