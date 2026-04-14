"""Import runners for dataset records."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable

ValidateNameFn = Callable[[str], tuple[bool, str]]
DiscoverFn = Callable[..., list[str]]
LoadRawFn = Callable[..., tuple[Any, bool]]
ApplyBipolarFn = Callable[..., Any]


def import_record_from_raw(
    *,
    project_root: Path,
    subject: str,
    record: str,
    raw: Any,
    source_path: Path,
    is_fif_input: bool,
    result_cls: type,
    validate_subject_name_fn: ValidateNameFn,
    validate_record_name_fn: ValidateNameFn,
    discover_subjects_fn: DiscoverFn,
    discover_records_fn: DiscoverFn,
    rawdata_record_fif_path_fn: Callable[[Path, str, str], Path],
    derivatives_record_root_fn: Callable[[Path, str, str], Path],
    sourcedata_record_raw_dir_fn: Callable[[Path, str, str], Path],
    read_only_project_root: Path | None = None,
):
    """Persist one already-parsed raw into standardized record paths."""
    if read_only_project_root is not None:
        if project_root.resolve() == read_only_project_root.resolve():
            return result_cls(
                ok=False,
                message=f"Project is read-only: {project_root}",
            )

    ok, normalized_subject = validate_subject_name_fn(subject)
    if not ok:
        return result_cls(ok=False, message=normalized_subject)
    ok, normalized_record = validate_record_name_fn(record)
    if not ok:
        return result_cls(ok=False, message=normalized_record)
    if not project_root.exists():
        return result_cls(ok=False, message=f"Missing project path: {project_root}")
    if normalized_subject not in discover_subjects_fn(project_root):
        return result_cls(ok=False, message=f"Missing subject: {normalized_subject}")
    if normalized_record in discover_records_fn(project_root, normalized_subject):
        return result_cls(
            ok=False,
            message=f"Record already exists: {normalized_subject}/{normalized_record}",
        )
    if not source_path.exists() or not source_path.is_file():
        return result_cls(ok=False, message=f"Missing source file: {source_path}")

    raw_fif_path = rawdata_record_fif_path_fn(
        project_root, normalized_subject, normalized_record
    )
    derivatives_root = derivatives_record_root_fn(
        project_root, normalized_subject, normalized_record
    )
    derivatives_root.mkdir(parents=True, exist_ok=True)
    raw_fif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw.save(str(raw_fif_path), overwrite=True)
    except Exception as exc:
        return result_cls(ok=False, message=f"Failed to save raw.fif: {exc}")

    source_copy_path: Path | None = None
    if not bool(is_fif_input):
        source_copy_path = (
            sourcedata_record_raw_dir_fn(project_root, normalized_subject, normalized_record)
            / source_path.name
        )
        source_copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, source_copy_path)

    return result_cls(
        ok=True,
        message=f"Record imported: {normalized_subject}/{normalized_record}",
        raw_fif_path=raw_fif_path,
        sourcedata_copy_path=source_copy_path,
    )


def import_record(
    *,
    project_root: Path,
    subject: str,
    record: str,
    source_path: Path,
    csv_sr: float | None,
    csv_unit: str,
    bipolar_pairs: tuple[tuple[str, str], ...],
    bipolar_names: tuple[str, ...],
    result_cls: type,
    validate_subject_name_fn: ValidateNameFn,
    validate_record_name_fn: ValidateNameFn,
    discover_subjects_fn: DiscoverFn,
    discover_records_fn: DiscoverFn,
    load_raw_from_source_fn: LoadRawFn,
    apply_bipolar_reference_fn: ApplyBipolarFn,
    rawdata_record_fif_path_fn: Callable[[Path, str, str], Path],
    derivatives_record_root_fn: Callable[[Path, str, str], Path],
    sourcedata_record_raw_dir_fn: Callable[[Path, str, str], Path],
    read_only_project_root: Path | None = None,
):
    """Import one record and normalize into standardized raw FIF."""
    if read_only_project_root is not None:
        if project_root.resolve() == read_only_project_root.resolve():
            return result_cls(
                ok=False,
                message=f"Project is read-only: {project_root}",
            )

    ok, normalized_subject = validate_subject_name_fn(subject)
    if not ok:
        return result_cls(ok=False, message=normalized_subject)
    ok, normalized_record = validate_record_name_fn(record)
    if not ok:
        return result_cls(ok=False, message=normalized_record)
    if not project_root.exists():
        return result_cls(ok=False, message=f"Missing project path: {project_root}")
    if normalized_subject not in discover_subjects_fn(project_root):
        return result_cls(ok=False, message=f"Missing subject: {normalized_subject}")
    if normalized_record in discover_records_fn(project_root, normalized_subject):
        return result_cls(
            ok=False,
            message=f"Record already exists: {normalized_subject}/{normalized_record}",
        )
    if not source_path.exists() or not source_path.is_file():
        return result_cls(ok=False, message=f"Missing source file: {source_path}")

    try:
        raw, is_fif_input = load_raw_from_source_fn(
            source_path,
            csv_sr=csv_sr,
            csv_unit=csv_unit,
        )
        raw = apply_bipolar_reference_fn(
            raw,
            bipolar_pairs,
            bipolar_names if bipolar_names else None,
        )
    except Exception as exc:
        return result_cls(ok=False, message=f"Failed to load source: {exc}")

    raw_fif_path = rawdata_record_fif_path_fn(
        project_root, normalized_subject, normalized_record
    )
    derivatives_root = derivatives_record_root_fn(
        project_root, normalized_subject, normalized_record
    )
    derivatives_root.mkdir(parents=True, exist_ok=True)
    raw_fif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw.save(str(raw_fif_path), overwrite=True)
    except Exception as exc:
        return result_cls(ok=False, message=f"Failed to save raw.fif: {exc}")

    source_copy_path: Path | None = None
    if not is_fif_input:
        source_copy_path = (
            sourcedata_record_raw_dir_fn(project_root, normalized_subject, normalized_record)
            / source_path.name
        )
        source_copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, source_copy_path)

    return result_cls(
        ok=True,
        message=f"Record imported: {normalized_subject}/{normalized_record}",
        raw_fif_path=raw_fif_path,
        sourcedata_copy_path=source_copy_path,
    )
