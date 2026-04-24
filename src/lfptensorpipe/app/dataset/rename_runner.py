"""Rename runner for dataset records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import pandas as pd
import yaml

from lfptensorpipe.app.features.table_io import _save_table_xlsx
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl

ValidateNameFn = Callable[[str], tuple[bool, str]]
MovePathFn = Callable[[Path, Path], None]

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


@dataclass(frozen=True)
class RecordRenameResult:
    """Result payload for one record rename action."""

    ok: bool
    message: str
    moved_paths: tuple[Path, ...] = ()
    updated_paths: tuple[Path, ...] = ()


def _default_move_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


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
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
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
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
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
            if path in seen_json:
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
            if path in seen_yaml:
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

    source_roots = tuple(
        Path(path).expanduser().resolve()
        for path in record_artifact_roots_fn(
            project_root, normalized_subject, normalized_record
        )
    )
    target_roots = tuple(
        Path(path).expanduser().resolve()
        for path in record_artifact_roots_fn(
            project_root, normalized_subject, normalized_new_record
        )
    )

    for path in target_roots:
        if path.exists():
            return result_cls(
                ok=False,
                message=f"Target record artifacts already exist: {path}",
            )

    existing_pairs = tuple(
        (src, dst) for src, dst in zip(source_roots, target_roots) if src.exists()
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
    reverse_path_replacements = tuple((new, old) for old, new in path_replacements)
    moved_pairs: list[tuple[Path, Path]] = []

    try:
        for src, dst in existing_pairs:
            move_path_fn(src, dst)
            moved_pairs.append((src, dst))

        derivatives_target_root = target_roots[0]
        updated_paths = _rewrite_derivatives_contracts(
            derivatives_target_root,
            old_record=normalized_record,
            new_record=normalized_new_record,
            path_replacements=path_replacements,
        )
    except Exception as exc:
        rollback_errors: list[str] = []
        for src, dst in reversed(moved_pairs):
            if not dst.exists():
                continue
            try:
                move_path_fn(dst, src)
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_errors.append(f"{dst} -> {src}: {rollback_exc}")
        derivatives_source_root = source_roots[0]
        if derivatives_source_root.exists():
            try:
                _rewrite_derivatives_contracts(
                    derivatives_source_root,
                    old_record=normalized_new_record,
                    new_record=normalized_record,
                    path_replacements=reverse_path_replacements,
                )
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_errors.append(f"content rollback: {rollback_exc}")
        message = (
            f"Failed to rename record {normalized_subject}/{normalized_record}: {exc}"
        )
        if rollback_errors:
            message += f" Rollback incomplete: {'; '.join(rollback_errors)}"
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
