"""Validation and path helpers for dataset actions."""

from __future__ import annotations

from pathlib import Path
import re

from lfptensorpipe.app.dataset_index import discover_subjects

SUBJECT_PATTERN = re.compile(r"^sub-[A-Za-z0-9]+$")
RECORD_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


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
    """Return all known record roots for deletion scan."""
    return (
        project_root / "derivatives" / "lfptensorpipe" / subject / record,
        project_root / "sourcedata" / subject / record,
        project_root / "sourcedata" / subject / "lfp" / record,
        project_root / "rawdata" / subject / record,
        project_root / "rawdata" / subject / "ses-postop" / "lfp" / record,
    )


def rawdata_record_fif_path(project_root: Path, subject: str, record: str) -> Path:
    """Return standardized raw FIF output path for one record import."""
    return (
        project_root
        / "rawdata"
        / subject
        / "ses-postop"
        / "lfp"
        / record
        / "raw"
        / "raw.fif"
    )


def derivatives_record_root(project_root: Path, subject: str, record: str) -> Path:
    """Return derivatives record root for one imported record."""
    return project_root / "derivatives" / "lfptensorpipe" / subject / record


def sourcedata_record_raw_dir(project_root: Path, subject: str, record: str) -> Path:
    """Return sourcedata original-file directory for non-FIF imports."""
    return project_root / "sourcedata" / subject / "lfp" / record / "raw"
