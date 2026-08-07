"""Import runners for dataset records."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable

from lfptensorpipe.io.timeline import (
    format_timeline_normalization,
    normalize_raw_timeline,
)

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
    persist_import_sync_artifacts_fn: Callable[..., Any] | None = None,
    sync_state: Any | None = None,
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
    sync_root = derivatives_root / "import"
    source_copy_path: Path | None = None
    raw_record_root = raw_fif_path.parents[1]

    # Normally a no-op: parse_record_source already normalized. It still fires for
    # callers that reach this API without going through the parse choke point.
    raw, timeline_report = normalize_raw_timeline(raw)
    timeline_summary = format_timeline_normalization(timeline_report)

    try:
        derivatives_root.mkdir(parents=True, exist_ok=True)
        raw_fif_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(str(raw_fif_path), overwrite=True)

        if not bool(is_fif_input):
            source_copy_path = (
                sourcedata_record_raw_dir_fn(
                    project_root, normalized_subject, normalized_record
                )
                / source_path.name
            )
            source_copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, source_copy_path)

        if sync_state is not None:
            if persist_import_sync_artifacts_fn is None:
                raise RuntimeError("Missing persist_import_sync_artifacts_fn.")
            persist_import_sync_artifacts_fn(
                project_root=project_root,
                subject=normalized_subject,
                record=normalized_record,
                raw_fif_path=raw_fif_path,
                sync_state=sync_state,
            )
    except Exception as exc:
        for root in (
            sync_root,
            source_copy_path.parents[1] if source_copy_path is not None else None,
            raw_record_root,
            derivatives_root,
        ):
            if root is None or not root.exists():
                continue
            shutil.rmtree(root, ignore_errors=True)
        return result_cls(ok=False, message=f"Failed to import record: {exc}")

    message = f"Record imported: {normalized_subject}/{normalized_record}"
    if timeline_summary:
        message = f"{message} ({timeline_summary})"
    return result_cls(
        ok=True,
        message=message,
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
    raw, timeline_report = normalize_raw_timeline(raw)
    timeline_summary = format_timeline_normalization(timeline_report)
    try:
        raw.save(str(raw_fif_path), overwrite=True)
    except Exception as exc:
        return result_cls(ok=False, message=f"Failed to save raw.fif: {exc}")

    source_copy_path: Path | None = None
    if not is_fif_input:
        source_copy_path = (
            sourcedata_record_raw_dir_fn(
                project_root, normalized_subject, normalized_record
            )
            / source_path.name
        )
        source_copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, source_copy_path)

    message = f"Record imported: {normalized_subject}/{normalized_record}"
    if timeline_summary:
        message = f"{message} ({timeline_summary})"
    return result_cls(
        ok=True,
        message=message,
        raw_fif_path=raw_fif_path,
        sourcedata_copy_path=source_copy_path,
    )
