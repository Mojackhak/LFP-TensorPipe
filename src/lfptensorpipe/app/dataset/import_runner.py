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
RecordScopePathsFn = Callable[[Path, str, str], dict[str, Path]]


def _standard_record_roots(
    *,
    project_root: Path,
    subject: str,
    record: str,
    record_scope_paths_fn: RecordScopePathsFn,
) -> dict[str, Path]:
    """Return every standard record root keyed by scope.

    The scope set comes from the injected mapping rather than a local copy, so a
    new standard scope is covered by the occupancy check automatically instead of
    being skipped until a second list is updated.
    """
    return dict(record_scope_paths_fn(project_root, subject, record))


def _occupied_record_roots(standard_roots: dict[str, Path]) -> tuple[Path, ...]:
    return tuple(path for path in standard_roots.values() if path.exists())


def _record_conflict_message(
    *,
    subject: str,
    record: str,
    occupied_roots: tuple[Path, ...],
) -> str:
    paths = "\n".join(f"- {path}" for path in occupied_roots)
    return (
        f"Record already exists: {subject}/{record}.\n"
        f"Occupied standard roots:\n{paths}"
    )


def _persist_record_import(
    *,
    project_root: Path,
    subject: str,
    record: str,
    raw: Any,
    source_path: Path,
    is_fif_input: bool,
    result_cls: type,
    record_scope_paths_fn: RecordScopePathsFn,
    rawdata_record_fif_path_fn: Callable[[Path, str, str], Path],
    sourcedata_record_raw_dir_fn: Callable[[Path, str, str], Path],
    persist_import_sync_artifacts_fn: Callable[..., Any] | None = None,
    sync_state: Any | None = None,
):
    """Persist an import and roll back only record roots created by this call."""
    standard_roots = _standard_record_roots(
        project_root=project_root,
        subject=subject,
        record=record,
        record_scope_paths_fn=record_scope_paths_fn,
    )
    occupied_roots = _occupied_record_roots(standard_roots)
    if occupied_roots:
        return result_cls(
            ok=False,
            message=_record_conflict_message(
                subject=subject,
                record=record,
                occupied_roots=occupied_roots,
            ),
        )

    raw_fif_path = rawdata_record_fif_path_fn(project_root, subject, record)
    source_copy_path: Path | None = None
    owned_roots: list[Path] = []

    try:
        # Normally a no-op: parse_record_source already normalized. It still fires
        # for callers that reach this API without going through the parse choke point.
        raw, timeline_report = normalize_raw_timeline(raw)
        timeline_summary = format_timeline_normalization(timeline_report)

        derivatives_root = standard_roots["derivatives"]
        derivatives_root.mkdir(parents=True, exist_ok=False)
        owned_roots.append(derivatives_root)

        rawdata_root = standard_roots["rawdata"]
        rawdata_root.mkdir(parents=True, exist_ok=False)
        owned_roots.append(rawdata_root)
        raw_fif_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(str(raw_fif_path))

        if not bool(is_fif_input):
            sourcedata_root = standard_roots["sourcedata"]
            sourcedata_root.mkdir(parents=True, exist_ok=False)
            owned_roots.append(sourcedata_root)
            source_copy_path = (
                sourcedata_record_raw_dir_fn(project_root, subject, record)
                / source_path.name
            )
            source_copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, source_copy_path)

        if sync_state is not None:
            if persist_import_sync_artifacts_fn is None:
                raise RuntimeError("Missing persist_import_sync_artifacts_fn.")
            persist_import_sync_artifacts_fn(
                project_root=project_root,
                subject=subject,
                record=record,
                raw_fif_path=raw_fif_path,
                sync_state=sync_state,
            )
    except Exception as exc:  # noqa: BLE001
        for root in reversed(owned_roots):
            if not root.exists():
                continue
            try:
                shutil.rmtree(root)
            except Exception:  # noqa: BLE001
                continue

        remaining_roots = _occupied_record_roots(standard_roots)
        message = f"Failed to import record: {exc}"
        if remaining_roots:
            remaining = "\n".join(f"- {path}" for path in remaining_roots)
            message = (
                f"{message}\n"
                f"Standard record roots still present after cleanup:\n{remaining}"
            )
        return result_cls(ok=False, message=message)

    message = f"Record imported: {subject}/{record}"
    if timeline_summary:
        message = f"{message} ({timeline_summary})"
    return result_cls(
        ok=True,
        message=message,
        raw_fif_path=raw_fif_path,
        sourcedata_copy_path=source_copy_path,
    )


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
    record_scope_paths_fn: RecordScopePathsFn,
    rawdata_record_fif_path_fn: Callable[[Path, str, str], Path],
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

    standard_roots = _standard_record_roots(
        project_root=project_root,
        subject=normalized_subject,
        record=normalized_record,
        record_scope_paths_fn=record_scope_paths_fn,
    )
    occupied_roots = _occupied_record_roots(standard_roots)
    if occupied_roots:
        return result_cls(
            ok=False,
            message=_record_conflict_message(
                subject=normalized_subject,
                record=normalized_record,
                occupied_roots=occupied_roots,
            ),
        )
    if not source_path.exists() or not source_path.is_file():
        return result_cls(ok=False, message=f"Missing source file: {source_path}")

    return _persist_record_import(
        project_root=project_root,
        subject=normalized_subject,
        record=normalized_record,
        raw=raw,
        source_path=source_path,
        is_fif_input=is_fif_input,
        result_cls=result_cls,
        record_scope_paths_fn=record_scope_paths_fn,
        rawdata_record_fif_path_fn=rawdata_record_fif_path_fn,
        sourcedata_record_raw_dir_fn=sourcedata_record_raw_dir_fn,
        persist_import_sync_artifacts_fn=persist_import_sync_artifacts_fn,
        sync_state=sync_state,
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
    record_scope_paths_fn: RecordScopePathsFn,
    load_raw_from_source_fn: LoadRawFn,
    apply_bipolar_reference_fn: ApplyBipolarFn,
    rawdata_record_fif_path_fn: Callable[[Path, str, str], Path],
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

    standard_roots = _standard_record_roots(
        project_root=project_root,
        subject=normalized_subject,
        record=normalized_record,
        record_scope_paths_fn=record_scope_paths_fn,
    )
    occupied_roots = _occupied_record_roots(standard_roots)
    if occupied_roots:
        return result_cls(
            ok=False,
            message=_record_conflict_message(
                subject=normalized_subject,
                record=normalized_record,
                occupied_roots=occupied_roots,
            ),
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
    except Exception as exc:  # noqa: BLE001
        return result_cls(ok=False, message=f"Failed to load source: {exc}")

    return _persist_record_import(
        project_root=project_root,
        subject=normalized_subject,
        record=normalized_record,
        raw=raw,
        source_path=source_path,
        is_fif_input=is_fif_input,
        result_cls=result_cls,
        record_scope_paths_fn=record_scope_paths_fn,
        rawdata_record_fif_path_fn=rawdata_record_fif_path_fn,
        sourcedata_record_raw_dir_fn=sourcedata_record_raw_dir_fn,
    )
