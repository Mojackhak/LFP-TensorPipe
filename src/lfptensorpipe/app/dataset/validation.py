"""Validation and path helpers for dataset actions."""

from __future__ import annotations

from pathlib import Path
import re

from lfptensorpipe.app.dataset_index import (
    STANDARD_RECORD_SCOPES,
    discover_subjects,
    standard_record_scope_roots,
)

SUBJECT_PATTERN = re.compile(r"^sub-[A-Za-z0-9]+$")
RECORD_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
# Delete exposes exactly the standard record scopes, in layout order.
RECORD_DELETE_SCOPES = STANDARD_RECORD_SCOPES


def validate_subject_name(subject: str) -> tuple[bool, str]:
    """Validate `sub-xxx` subject naming contract."""
    value = subject.strip()
    if not value:
        return False, "Subject name cannot be empty."
    if not SUBJECT_PATTERN.fullmatch(value):
        return False, "Subject must match pattern: sub-[A-Za-z0-9]+"
    return True, value


def validate_record_name(record: str) -> tuple[bool, str]:
    """Validate record naming contract."""
    value = record.strip()
    if not value:
        return False, "Record name cannot be empty."
    if not RECORD_PATTERN.fullmatch(value):
        return False, "Record must match pattern: [A-Za-z0-9_-]+"
    return True, value


def create_subject(project_root: Path, subject: str) -> tuple[bool, str]:
    """Create one subject in `sourcedata/` and `rawdata/`."""
    ok, normalized = validate_subject_name(subject)
    if not ok:
        return False, normalized
    if not project_root.exists():
        return False, f"Project path does not exist: {project_root}"
    if normalized in discover_subjects(project_root):
        return False, f"Subject already exists: {normalized}"

    sourcedata_dir = project_root / "sourcedata" / normalized
    rawdata_dir = project_root / "rawdata" / normalized
    sourcedata_dir.mkdir(parents=True, exist_ok=True)
    rawdata_dir.mkdir(parents=True, exist_ok=True)
    return True, f"Subject created: {normalized}"


def record_artifact_roots(
    project_root: Path, subject: str, record: str
) -> tuple[Path, ...]:
    """Return all known record roots for legacy-aware record operations."""
    return (
        project_root / "derivatives" / "lfptensorpipe" / subject / record,
        project_root / "sourcedata" / subject / record,
        project_root / "sourcedata" / subject / "lfp" / record,
        project_root / "rawdata" / subject / record,
        project_root / "rawdata" / subject / "ses-postop" / "lfp" / record,
    )


def record_delete_scope_paths(
    project_root: Path, subject: str, record: str
) -> dict[str, Path]:
    """Return standard record roots keyed by delete scope."""
    return {
        scope: root / record
        for scope, root in standard_record_scope_roots(project_root, subject).items()
    }


def rawdata_record_fif_path(project_root: Path, subject: str, record: str) -> Path:
    """Return standardized raw FIF output path for one record import."""
    return (
        record_delete_scope_paths(project_root, subject, record)["rawdata"]
        / "raw"
        / "raw.fif"
    )


def derivatives_record_root(project_root: Path, subject: str, record: str) -> Path:
    """Return derivatives record root for one imported record."""
    return record_delete_scope_paths(project_root, subject, record)["derivatives"]


def sourcedata_record_raw_dir(project_root: Path, subject: str, record: str) -> Path:
    """Return sourcedata original-file directory for non-FIF imports."""
    return (
        record_delete_scope_paths(project_root, subject, record)["sourcedata"] / "raw"
    )
