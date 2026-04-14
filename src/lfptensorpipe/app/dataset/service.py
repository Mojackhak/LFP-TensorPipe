"""Dataset mutation services for project/subject/record actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from lfptensorpipe.app.dataset_index import discover_records, discover_subjects
from .delete_runner import delete_record as _delete_record_impl
from .import_runner import (
    import_record as _import_record_impl,
    import_record_from_raw as _import_record_from_raw_impl,
)
from .source_parser import (
    _apply_bipolar_reference as _apply_bipolar_reference_impl,
    _is_fif_like_path as _is_fif_like_path_impl,
    _load_raw_from_source as _load_raw_from_source_impl,
    _validate_bipolar_pairs as _validate_bipolar_pairs_impl,
    apply_reset_reference as _apply_reset_reference_impl,
    load_import_channel_names as _load_import_channel_names_impl,
    parse_record_source as _parse_record_source_impl,
)
from .validation import (
    create_subject as _create_subject_impl,
    derivatives_record_root as _derivatives_record_root_impl,
    rawdata_record_fif_path as _rawdata_record_fif_path_impl,
    record_artifact_roots as _record_artifact_roots_impl,
    sourcedata_record_raw_dir as _sourcedata_record_raw_dir_impl,
    validate_record_name as _validate_record_name_impl,
    validate_subject_name as _validate_subject_name_impl,
)


@dataclass(frozen=True)
class RecordImportResult:
    """Result payload for one record import action."""

    ok: bool
    message: str
    raw_fif_path: Path | None = None
    sourcedata_copy_path: Path | None = None


@dataclass(frozen=True)
class RecordDeleteResult:
    """Result payload for one record deletion action."""

    ok: bool
    message: str
    deleted_paths: tuple[Path, ...] = ()


def _is_fif_like_path(path: Path) -> bool:
    return _is_fif_like_path_impl(path)


def validate_subject_name(subject: str) -> tuple[bool, str]:
    return _validate_subject_name_impl(subject)


def validate_record_name(record: str) -> tuple[bool, str]:
    return _validate_record_name_impl(record)


def create_subject(project_root: Path, subject: str) -> tuple[bool, str]:
    return _create_subject_impl(project_root, subject)


def record_artifact_roots(
    project_root: Path, subject: str, record: str
) -> tuple[Path, ...]:
    return _record_artifact_roots_impl(project_root, subject, record)


def rawdata_record_fif_path(project_root: Path, subject: str, record: str) -> Path:
    return _rawdata_record_fif_path_impl(project_root, subject, record)


def derivatives_record_root(project_root: Path, subject: str, record: str) -> Path:
    return _derivatives_record_root_impl(project_root, subject, record)


def sourcedata_record_raw_dir(project_root: Path, subject: str, record: str) -> Path:
    return _sourcedata_record_raw_dir_impl(project_root, subject, record)


def _validate_bipolar_pairs(
    raw: Any,
    bipolar_pairs: tuple[tuple[str, str], ...],
    bipolar_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    return _validate_bipolar_pairs_impl(raw, bipolar_pairs, bipolar_names)


def _apply_bipolar_reference(
    raw: Any,
    bipolar_pairs: tuple[tuple[str, str], ...],
    bipolar_names: tuple[str, ...] | None = None,
    *,
    set_bipolar_reference_fn: Any | None = None,
) -> Any:
    return _apply_bipolar_reference_impl(
        raw,
        bipolar_pairs,
        bipolar_names,
        set_bipolar_reference_fn=set_bipolar_reference_fn,
    )


def _load_raw_from_source(
    source_path: Path,
    *,
    csv_sr: float | None,
    csv_unit: str,
) -> tuple[Any, bool]:
    return _load_raw_from_source_impl(source_path, csv_sr=csv_sr, csv_unit=csv_unit)


def parse_record_source(
    *,
    import_type: str,
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str], bool]:
    return _parse_record_source_impl(
        import_type=import_type,
        paths=paths,
        options=options,
    )


def apply_reset_reference(
    raw: Any,
    reset_rows: tuple[tuple[str, str, str], ...],
) -> Any:
    return _apply_reset_reference_impl(raw, reset_rows)


def load_import_channel_names(
    source_path: Path,
    *,
    csv_sr: float | None = None,
    csv_unit: str = "V",
) -> list[str]:
    return _load_import_channel_names_impl(
        source_path,
        csv_sr=csv_sr,
        csv_unit=csv_unit,
    )


def import_record_from_raw(
    *,
    project_root: Path,
    subject: str,
    record: str,
    raw: Any,
    source_path: Path,
    is_fif_input: bool,
    read_only_project_root: Path | None = None,
) -> RecordImportResult:
    return _import_record_from_raw_impl(
        project_root=project_root,
        subject=subject,
        record=record,
        raw=raw,
        source_path=source_path,
        is_fif_input=is_fif_input,
        result_cls=RecordImportResult,
        validate_subject_name_fn=validate_subject_name,
        validate_record_name_fn=validate_record_name,
        discover_subjects_fn=discover_subjects,
        discover_records_fn=discover_records,
        rawdata_record_fif_path_fn=rawdata_record_fif_path,
        derivatives_record_root_fn=derivatives_record_root,
        sourcedata_record_raw_dir_fn=sourcedata_record_raw_dir,
        read_only_project_root=read_only_project_root,
    )


def import_record(
    *,
    project_root: Path,
    subject: str,
    record: str,
    source_path: Path,
    csv_sr: float | None = None,
    csv_unit: str = "V",
    bipolar_pairs: tuple[tuple[str, str], ...] = (),
    bipolar_names: tuple[str, ...] = (),
    read_only_project_root: Path | None = None,
    load_raw_from_source_fn: Any | None = None,
    apply_bipolar_reference_fn: Any | None = None,
) -> RecordImportResult:
    return _import_record_impl(
        project_root=project_root,
        subject=subject,
        record=record,
        source_path=source_path,
        csv_sr=csv_sr,
        csv_unit=csv_unit,
        bipolar_pairs=bipolar_pairs,
        bipolar_names=bipolar_names,
        result_cls=RecordImportResult,
        validate_subject_name_fn=validate_subject_name,
        validate_record_name_fn=validate_record_name,
        discover_subjects_fn=discover_subjects,
        discover_records_fn=discover_records,
        load_raw_from_source_fn=load_raw_from_source_fn or _load_raw_from_source,
        apply_bipolar_reference_fn=apply_bipolar_reference_fn
        or _apply_bipolar_reference,
        rawdata_record_fif_path_fn=rawdata_record_fif_path,
        derivatives_record_root_fn=derivatives_record_root,
        sourcedata_record_raw_dir_fn=sourcedata_record_raw_dir,
        read_only_project_root=read_only_project_root,
    )


def delete_record(
    *,
    project_root: Path,
    subject: str,
    record: str,
    read_only_project_root: Path | None = None,
    record_artifact_roots_fn: Any | None = None,
    rmtree_fn: Any | None = None,
) -> RecordDeleteResult:
    return _delete_record_impl(
        project_root=project_root,
        subject=subject,
        record=record,
        result_cls=RecordDeleteResult,
        validate_subject_name_fn=validate_subject_name,
        validate_record_name_fn=validate_record_name,
        record_artifact_roots_fn=record_artifact_roots_fn or record_artifact_roots,
        rmtree_fn=rmtree_fn or shutil.rmtree,
        read_only_project_root=read_only_project_root,
    )
