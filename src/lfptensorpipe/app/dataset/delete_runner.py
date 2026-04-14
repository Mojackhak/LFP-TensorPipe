"""Delete runner for dataset records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

ValidateNameFn = Callable[[str], tuple[bool, str]]


def delete_record(
    *,
    project_root: Path,
    subject: str,
    record: str,
    result_cls: type,
    validate_subject_name_fn: ValidateNameFn,
    validate_record_name_fn: ValidateNameFn,
    record_artifact_roots_fn: Callable[[Path, str, str], tuple[Path, ...]],
    rmtree_fn: Callable[..., Any],
    read_only_project_root: Path | None = None,
):
    """Delete all known record roots for one project+subject+record."""

    def _rmtree_ignore_missing_entries(path: Path) -> None:
        def _onerror(_func: Any, _target: str, exc_info: Any) -> None:
            exc = exc_info[1]
            if isinstance(exc, FileNotFoundError):
                return
            raise exc

        try:
            rmtree_fn(path, onerror=_onerror)
        except FileNotFoundError:
            return

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

    deleted: list[Path] = []
    for path in record_artifact_roots_fn(
        project_root, normalized_subject, normalized_record
    ):
        if not path.exists():
            continue
        try:
            if path.is_dir():
                _rmtree_ignore_missing_entries(path)
            else:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        except Exception as exc:
            return result_cls(
                ok=False,
                message=f"Failed to delete record artifact: {path} ({exc})",
                deleted_paths=tuple(deleted),
            )
        deleted.append(path)

    if not deleted:
        return result_cls(
            ok=False,
            message=f"No record artifacts found for {normalized_subject}/{normalized_record}.",
            deleted_paths=(),
        )
    return result_cls(
        ok=True,
        message=f"Deleted record artifacts for {normalized_subject}/{normalized_record}.",
        deleted_paths=tuple(deleted),
    )
