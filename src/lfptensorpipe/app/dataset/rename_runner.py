"""Rename runner for dataset records."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import pandas as pd
import yaml

from lfptensorpipe.app.features.table_io import _save_table_xlsx
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl

ValidateNameFn = Callable[[str], tuple[bool, str]]
MovePathFn = Callable[[Path, Path], None]
DeletePathFn = Callable[[Path], None]

_JSON_CONTRACT_PATTERNS = (
    "**/lfptensorpipe_log.json",
    "**/*_backend_log.json",
)
_YAML_CONTRACT_PATTERNS = (
    "**/config.yml",
    "**/config.yaml",
)
_LOCALIZE_TABLE_BASENAMES = (
    "channel_representative_coords",
    "channel_pair_ordered_representative_coords",
    "channel_pair_undirected_representative_coords",
)
_RECORD_ROOT_ROLES = (
    "derivatives",
    "rawdata",
    "sourcedata",
)
_RECORD_RENAME_OPERATION = "record_rename"


@dataclass(frozen=True)
class RecordRenameResult:
    """Result payload for one record rename action."""

    ok: bool
    message: str
    moved_paths: tuple[Path, ...] = ()
    updated_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RecordRenameRecoveryResult:
    """Result payload for one subject-scoped interrupted rename check."""

    ok: bool
    recovered: bool
    message: str
    marker_path: Path | None = None
    old_record: str | None = None
    new_record: str | None = None
    conflicting_paths: tuple[Path, ...] = ()


def _default_move_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def _default_delete_path(path: Path) -> None:
    path.unlink()


def _record_rename_marker_path(project_root: Path, subject: str) -> Path:
    return (
        project_root
        / "derivatives"
        / "lfptensorpipe"
        / f".record-rename-{subject}.json"
    )


def _record_rename_writing_path(marker_path: Path) -> Path:
    return marker_path.with_name(f"{marker_path.name}.writing")


def _cleanup_lone_record_rename_writing(marker_path: Path) -> None:
    if marker_path.exists():
        return
    try:
        _record_rename_writing_path(marker_path).unlink(missing_ok=True)
    except OSError:
        pass


def _write_record_rename_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writing_path = _record_rename_writing_path(path)
    try:
        with writing_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        writing_path.replace(path)
    except Exception:
        try:
            writing_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _roots_by_role(roots: tuple[Path, ...]) -> dict[str, Path]:
    if len(roots) != len(_RECORD_ROOT_ROLES):
        raise ValueError(
            "Record root resolver must return exactly "
            f"{len(_RECORD_ROOT_ROLES)} paths."
        )
    return dict(zip(_RECORD_ROOT_ROLES, roots))


def _resolved_record_roots(
    *,
    project_root: Path,
    subject: str,
    record: str,
    record_artifact_roots_fn: Callable[[Path, str, str], tuple[Path, ...]],
) -> tuple[Path, ...]:
    return tuple(
        Path(path).expanduser().resolve()
        for path in record_artifact_roots_fn(project_root, subject, record)
    )


def _record_rename_marker_payload(
    *,
    subject: str,
    old_record: str,
    new_record: str,
    roots: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "operation": _RECORD_RENAME_OPERATION,
        "subject": subject,
        "old_record": old_record,
        "new_record": new_record,
        "roots": list(roots),
    }


def _load_record_rename_marker(
    path: Path,
    *,
    expected_subject: str,
    validate_subject_name_fn: ValidateNameFn,
    validate_record_name_fn: ValidateNameFn,
) -> tuple[str, str, tuple[str, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read record rename marker {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Record rename marker must be an object: {path}")
    expected_keys = {"operation", "subject", "old_record", "new_record", "roots"}
    if set(payload) != expected_keys:
        raise ValueError(f"Record rename marker has invalid fields: {path}")
    if payload.get("operation") != _RECORD_RENAME_OPERATION:
        raise ValueError(f"Record rename marker has invalid operation: {path}")

    raw_subject = payload.get("subject")
    raw_old_record = payload.get("old_record")
    raw_new_record = payload.get("new_record")
    if not all(
        isinstance(value, str)
        for value in (raw_subject, raw_old_record, raw_new_record)
    ):
        raise ValueError(f"Record rename marker names must be strings: {path}")

    ok, subject = validate_subject_name_fn(raw_subject)
    if not ok or subject != expected_subject:
        raise ValueError(f"Record rename marker subject mismatch: {path}")
    ok, old_record = validate_record_name_fn(raw_old_record)
    if not ok:
        raise ValueError(f"Record rename marker has invalid old record: {path}")
    ok, new_record = validate_record_name_fn(raw_new_record)
    if not ok or new_record == old_record:
        raise ValueError(f"Record rename marker has invalid new record: {path}")

    raw_roots = payload.get("roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError(f"Record rename marker has no root roles: {path}")
    if not all(isinstance(value, str) for value in raw_roots):
        raise ValueError(f"Record rename marker root roles must be strings: {path}")
    roots = tuple(raw_roots)
    if len(set(roots)) != len(roots) or any(
        role not in _RECORD_ROOT_ROLES for role in roots
    ):
        raise ValueError(f"Record rename marker has invalid root roles: {path}")
    return old_record, new_record, roots


def _is_metadata_sidecar(path: Path) -> bool:
    """Return True for macOS AppleDouble sidecar files (``._*``).

    Non-APFS volumes (exFAT/NTFS/SMB) grow binary ``._<name>`` companions for
    every file; contract globs must not treat them as JSON/YAML payloads.
    """
    return path.name.startswith("._")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _write_yaml_atomic(path: Path, payload: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _rewrite_path_prefix(
    value: str,
    replacements: tuple[tuple[str, str], ...],
) -> str:
    for old_prefix, new_prefix in replacements:
        if value == old_prefix:
            return new_prefix
        if value.startswith(old_prefix + "/"):
            return new_prefix + value[len(old_prefix) :]
    return value


def _rewrite_serialized_value(
    value: Any,
    *,
    old_record: str,
    new_record: str,
    path_replacements: tuple[tuple[str, str], ...],
) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) == "record" and isinstance(item, str) and item == old_record:
                out[key] = new_record
                changed = True
                continue
            new_item, item_changed = _rewrite_serialized_value(
                item,
                old_record=old_record,
                new_record=new_record,
                path_replacements=path_replacements,
            )
            out[key] = new_item
            changed |= item_changed
        return (out, True) if changed else (value, False)

    if isinstance(value, list):
        changed = False
        out: list[Any] = []
        for item in value:
            new_item, item_changed = _rewrite_serialized_value(
                item,
                old_record=old_record,
                new_record=new_record,
                path_replacements=path_replacements,
            )
            out.append(new_item)
            changed |= item_changed
        return (out, True) if changed else (value, False)

    if isinstance(value, str):
        new_value = _rewrite_path_prefix(value, path_replacements)
        return (new_value, new_value != value)

    return value, False


def _rewrite_json_contract_file(
    path: Path,
    *,
    old_record: str,
    new_record: str,
    path_replacements: tuple[tuple[str, str], ...],
) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse JSON contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON contract must be an object: {path}")
    updated, changed = _rewrite_serialized_value(
        payload,
        old_record=old_record,
        new_record=new_record,
        path_replacements=path_replacements,
    )
    if changed:
        _write_json_atomic(path, updated)
    return changed


def _rewrite_yaml_contract_file(
    path: Path,
    *,
    old_record: str,
    new_record: str,
    path_replacements: tuple[tuple[str, str], ...],
) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to parse YAML contract {path}: {exc}") from exc
    if payload is None:
        return False
    updated, changed = _rewrite_serialized_value(
        payload,
        old_record=old_record,
        new_record=new_record,
        path_replacements=path_replacements,
    )
    if changed:
        _write_yaml_atomic(path, updated)
    return changed


def _rewrite_dataframe_record_columns(
    frame: pd.DataFrame,
    *,
    old_record: str,
    new_record: str,
) -> tuple[pd.DataFrame, bool]:
    updated = frame.copy()
    changed = False
    for column in ("record", "Record"):
        if column not in updated.columns:
            continue
        mask = updated[column] == old_record
        if bool(mask.any()):
            updated.loc[mask, column] = new_record
            changed = True
    return updated, changed


def _rewrite_pickle_dataframe(
    path: Path,
    *,
    old_record: str,
    new_record: str,
) -> bool:
    try:
        payload = load_pkl(path)
    except Exception:
        return False
    if not isinstance(payload, pd.DataFrame):
        return False
    updated, changed = _rewrite_dataframe_record_columns(
        payload,
        old_record=old_record,
        new_record=new_record,
    )
    if changed:
        save_pkl(updated, path)
    return changed


def _rewrite_csv_dataframe(
    path: Path,
    *,
    old_record: str,
    new_record: str,
) -> bool:
    frame = pd.read_csv(path)
    updated, changed = _rewrite_dataframe_record_columns(
        frame,
        old_record=old_record,
        new_record=new_record,
    )
    if changed:
        updated.to_csv(path, index=False)
    return changed


def _rewrite_features_tables(
    features_root: Path,
    *,
    old_record: str,
    new_record: str,
) -> tuple[Path, ...]:
    updated_paths: list[Path] = []
    if not features_root.exists():
        return ()
    for path in sorted(features_root.glob("**/*.pkl")):
        try:
            payload = load_pkl(path)
        except Exception:
            continue
        if not isinstance(payload, pd.DataFrame):
            continue
        updated, changed = _rewrite_dataframe_record_columns(
            payload,
            old_record=old_record,
            new_record=new_record,
        )
        if not changed:
            continue
        save_pkl(updated, path)
        updated_paths.append(path)
        xlsx_path = path.with_suffix(".xlsx")
        if xlsx_path.exists():
            ok, message = _save_table_xlsx(updated, xlsx_path)
            if not ok:
                raise ValueError(f"Failed to rewrite XLSX table {xlsx_path}: {message}")
            updated_paths.append(xlsx_path)
    return tuple(updated_paths)


def _rewrite_derivatives_contracts(
    derivatives_root: Path,
    *,
    old_record: str,
    new_record: str,
    path_replacements: tuple[tuple[str, str], ...],
) -> tuple[Path, ...]:
    updated_paths: list[Path] = []
    if not derivatives_root.exists():
        return ()

    ui_state_path = derivatives_root / "lfptensorpipe_ui_state.json"
    if ui_state_path.exists() and _rewrite_json_contract_file(
        ui_state_path,
        old_record=old_record,
        new_record=new_record,
        path_replacements=path_replacements,
    ):
        updated_paths.append(ui_state_path)

    seen_json: set[Path] = set()
    for pattern in _JSON_CONTRACT_PATTERNS:
        for path in sorted(derivatives_root.glob(pattern)):
            if path in seen_json or _is_metadata_sidecar(path):
                continue
            seen_json.add(path)
            if _rewrite_json_contract_file(
                path,
                old_record=old_record,
                new_record=new_record,
                path_replacements=path_replacements,
            ):
                updated_paths.append(path)

    seen_yaml: set[Path] = set()
    for pattern in _YAML_CONTRACT_PATTERNS:
        for path in sorted(derivatives_root.glob(pattern)):
            if path in seen_yaml or _is_metadata_sidecar(path):
                continue
            seen_yaml.add(path)
            if _rewrite_yaml_contract_file(
                path,
                old_record=old_record,
                new_record=new_record,
                path_replacements=path_replacements,
            ):
                updated_paths.append(path)

    localize_root = derivatives_root / "localize"
    for basename in _LOCALIZE_TABLE_BASENAMES:
        pkl_path = localize_root / f"{basename}.pkl"
        if pkl_path.exists() and _rewrite_pickle_dataframe(
            pkl_path,
            old_record=old_record,
            new_record=new_record,
        ):
            updated_paths.append(pkl_path)
        csv_path = localize_root / f"{basename}.csv"
        if csv_path.exists() and _rewrite_csv_dataframe(
            csv_path,
            old_record=old_record,
            new_record=new_record,
        ):
            updated_paths.append(csv_path)

    alignment_root = derivatives_root / "alignment"
    if alignment_root.exists():
        for path in sorted(alignment_root.glob("*/*/na-raw.pkl")):
            if _rewrite_pickle_dataframe(
                path,
                old_record=old_record,
                new_record=new_record,
            ):
                updated_paths.append(path)

    updated_paths.extend(
        _rewrite_features_tables(
            derivatives_root / "features",
            old_record=old_record,
            new_record=new_record,
        )
    )
    return tuple(updated_paths)


def recover_record_rename(
    *,
    project_root: Path,
    subject: str,
    validate_subject_name_fn: ValidateNameFn,
    validate_record_name_fn: ValidateNameFn,
    record_artifact_roots_fn: Callable[[Path, str, str], tuple[Path, ...]],
    move_path_fn: MovePathFn = _default_move_path,
    delete_marker_fn: DeletePathFn = _default_delete_path,
    read_only_project_root: Path | None = None,
) -> RecordRenameRecoveryResult:
    """Roll one interrupted subject-scoped Record Rename back to its old name."""

    project_root = Path(project_root).expanduser().resolve()
    ok, normalized_subject = validate_subject_name_fn(subject)
    if not ok:
        return RecordRenameRecoveryResult(
            ok=False,
            recovered=False,
            message=normalized_subject,
        )

    marker_path = _record_rename_marker_path(project_root, normalized_subject)
    _cleanup_lone_record_rename_writing(marker_path)
    if not marker_path.exists():
        return RecordRenameRecoveryResult(
            ok=True,
            recovered=False,
            message="No interrupted record rename was found.",
        )

    try:
        old_record, new_record, root_roles = _load_record_rename_marker(
            marker_path,
            expected_subject=normalized_subject,
            validate_subject_name_fn=validate_subject_name_fn,
            validate_record_name_fn=validate_record_name_fn,
        )
    except ValueError as exc:
        return RecordRenameRecoveryResult(
            ok=False,
            recovered=False,
            message=str(exc),
            marker_path=marker_path,
        )

    if (
        read_only_project_root is not None
        and project_root == Path(read_only_project_root).expanduser().resolve()
    ):
        return RecordRenameRecoveryResult(
            ok=False,
            recovered=False,
            message=f"Project is read-only: {project_root}",
            marker_path=marker_path,
            old_record=old_record,
            new_record=new_record,
        )

    try:
        old_roots = _resolved_record_roots(
            project_root=project_root,
            subject=normalized_subject,
            record=old_record,
            record_artifact_roots_fn=record_artifact_roots_fn,
        )
        new_roots = _resolved_record_roots(
            project_root=project_root,
            subject=normalized_subject,
            record=new_record,
            record_artifact_roots_fn=record_artifact_roots_fn,
        )
        old_by_role = _roots_by_role(old_roots)
        new_by_role = _roots_by_role(new_roots)
    except Exception as exc:
        return RecordRenameRecoveryResult(
            ok=False,
            recovered=False,
            message=f"Failed to resolve record rename marker roots {marker_path}: {exc}",
            marker_path=marker_path,
            old_record=old_record,
            new_record=new_record,
        )

    pairs = tuple((role, old_by_role[role], new_by_role[role]) for role in root_roles)
    conflicts: list[Path] = []
    conflict_details: list[str] = []
    moves: list[tuple[Path, Path]] = []
    for role, old_path, new_path in pairs:
        old_exists = old_path.exists()
        new_exists = new_path.exists()
        if old_exists == new_exists:
            conflicts.extend((old_path, new_path))
            state = "both present" if old_exists else "both missing"
            conflict_details.append(
                f"{role}: {state}\n  old: {old_path}\n  new: {new_path}"
            )
        elif new_exists:
            moves.append((new_path, old_path))

    if conflicts:
        paths = tuple(conflicts)
        detail_text = "\n".join(conflict_details)
        return RecordRenameRecoveryResult(
            ok=False,
            recovered=False,
            message=(
                "Interrupted record rename cannot be rolled back because one or "
                f"more root pairs are ambiguous or missing. Marker: {marker_path}"
                f"\nConflicting root pairs:\n{detail_text}"
            ),
            marker_path=marker_path,
            old_record=old_record,
            new_record=new_record,
            conflicting_paths=paths,
        )

    try:
        for new_path, old_path in moves:
            move_path_fn(new_path, old_path)

        path_replacements = tuple(
            sorted(
                (
                    (str(new_path), str(old_path))
                    for old_path, new_path in zip(old_roots, new_roots)
                ),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        )
        _rewrite_derivatives_contracts(
            old_by_role["derivatives"],
            old_record=new_record,
            new_record=old_record,
            path_replacements=path_replacements,
        )
        delete_marker_fn(marker_path)
    except Exception as exc:
        return RecordRenameRecoveryResult(
            ok=False,
            recovered=False,
            message=(
                f"Failed to roll back interrupted record rename {old_record} -> "
                f"{new_record}. Marker retained at {marker_path}: {exc}"
            ),
            marker_path=marker_path,
            old_record=old_record,
            new_record=new_record,
        )

    return RecordRenameRecoveryResult(
        ok=True,
        recovered=True,
        message=(
            f"Interrupted record rename {old_record} -> {new_record} was rolled "
            "back. Run Rename again."
        ),
        marker_path=marker_path,
        old_record=old_record,
        new_record=new_record,
    )


def rename_record(
    *,
    project_root: Path,
    subject: str,
    record: str,
    new_record: str,
    result_cls: type,
    validate_subject_name_fn: ValidateNameFn,
    validate_record_name_fn: ValidateNameFn,
    record_artifact_roots_fn: Callable[[Path, str, str], tuple[Path, ...]],
    move_path_fn: MovePathFn = _default_move_path,
    delete_marker_fn: DeletePathFn = _default_delete_path,
    read_only_project_root: Path | None = None,
):
    """Rename all known record roots and repair known embedded record/path fields."""

    project_root = Path(project_root).expanduser().resolve()
    if read_only_project_root is not None:
        if project_root == Path(read_only_project_root).expanduser().resolve():
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

    ok, normalized_new_record = validate_record_name_fn(new_record)
    if not ok:
        return result_cls(ok=False, message=normalized_new_record)
    if normalized_new_record == normalized_record:
        return result_cls(
            ok=False,
            message="New record name must be different from the current record.",
        )

    marker_path = _record_rename_marker_path(project_root, normalized_subject)
    _cleanup_lone_record_rename_writing(marker_path)
    if marker_path.exists():
        return result_cls(
            ok=False,
            message=(
                "An interrupted record rename is pending for "
                f"{normalized_subject}: {marker_path}"
            ),
        )

    source_roots = _resolved_record_roots(
        project_root=project_root,
        subject=normalized_subject,
        record=normalized_record,
        record_artifact_roots_fn=record_artifact_roots_fn,
    )
    target_roots = _resolved_record_roots(
        project_root=project_root,
        subject=normalized_subject,
        record=normalized_new_record,
        record_artifact_roots_fn=record_artifact_roots_fn,
    )
    try:
        _roots_by_role(source_roots)
        _roots_by_role(target_roots)
    except ValueError as exc:
        return result_cls(ok=False, message=str(exc))

    for path in target_roots:
        if path.exists():
            return result_cls(
                ok=False,
                message=f"Target record artifacts already exist: {path}",
            )

    existing_pairs = tuple(
        (role, src, dst)
        for role, src, dst in zip(
            _RECORD_ROOT_ROLES,
            source_roots,
            target_roots,
        )
        if src.exists()
    )
    if not existing_pairs:
        return result_cls(
            ok=False,
            message=(
                "No record artifacts found for "
                f"{normalized_subject}/{normalized_record}."
            ),
        )

    path_replacements = tuple(
        sorted(
            ((str(src), str(dst)) for src, dst in zip(source_roots, target_roots)),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )
    moved_pairs: list[tuple[Path, Path]] = []

    try:
        marker_payload = _record_rename_marker_payload(
            subject=normalized_subject,
            old_record=normalized_record,
            new_record=normalized_new_record,
            roots=tuple(role for role, _src, _dst in existing_pairs),
        )
        _write_record_rename_marker(marker_path, marker_payload)

        for _role, src, dst in existing_pairs:
            move_path_fn(src, dst)
            moved_pairs.append((src, dst))

        derivatives_target_root = target_roots[0]
        updated_paths = _rewrite_derivatives_contracts(
            derivatives_target_root,
            old_record=normalized_record,
            new_record=normalized_new_record,
            path_replacements=path_replacements,
        )
        delete_marker_fn(marker_path)
    except Exception as exc:
        rollback_result = recover_record_rename(
            project_root=project_root,
            subject=normalized_subject,
            validate_subject_name_fn=validate_subject_name_fn,
            validate_record_name_fn=validate_record_name_fn,
            record_artifact_roots_fn=record_artifact_roots_fn,
            move_path_fn=move_path_fn,
            delete_marker_fn=delete_marker_fn,
            read_only_project_root=read_only_project_root,
        )
        message = (
            f"Failed to rename record {normalized_subject}/{normalized_record}: {exc}"
        )
        if rollback_result.ok and rollback_result.recovered:
            message += " Rollback applied."
        elif marker_path.exists():
            message += f" Rollback incomplete: {rollback_result.message}"
        else:
            message += " Rollback applied."
        return result_cls(ok=False, message=message)

    return result_cls(
        ok=True,
        message=(
            f"Renamed record {normalized_subject}/{normalized_record} "
            f"to {normalized_new_record}."
        ),
        moved_paths=tuple(dst for _, dst in moved_pairs),
        updated_paths=tuple(sorted(updated_paths)),
    )
