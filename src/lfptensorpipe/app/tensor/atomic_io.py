"""Atomic artifact write helpers for tensor outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from uuid import uuid4


def write_outputs_atomically(
    outputs: list[tuple[Path, Callable[[Path], None]]],
    *,
    replace_fn: Callable[[Path, Path], Path] | None = None,
) -> None:
    """Write multiple artifacts atomically with rollback on failure."""
    token = uuid4().hex
    tmp_paths: list[tuple[Path, Path]] = []
    backup_paths: dict[Path, Path] = {}
    committed: list[Path] = []

    runtime_replace = replace_fn or (lambda src, dst: src.replace(dst))

    try:
        for target_path, writer in outputs:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = target_path.parent / f".{target_path.name}.tmp-{token}"
            writer(tmp_path)
            tmp_paths.append((target_path, tmp_path))

        for target_path, _ in tmp_paths:
            if target_path.exists():
                backup_path = target_path.parent / f".{target_path.name}.bak-{token}"
                runtime_replace(target_path, backup_path)
                backup_paths[target_path] = backup_path

        for target_path, tmp_path in tmp_paths:
            runtime_replace(tmp_path, target_path)
            committed.append(target_path)

        for backup_path in backup_paths.values():
            backup_path.unlink(missing_ok=True)
    except Exception:
        for target_path in committed:
            target_path.unlink(missing_ok=True)
        for target_path, backup_path in backup_paths.items():
            if backup_path.exists():
                runtime_replace(backup_path, target_path)
        raise
    finally:
        for _, tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)
        for backup_path in backup_paths.values():
            backup_path.unlink(missing_ok=True)


__all__ = ["write_outputs_atomically"]
