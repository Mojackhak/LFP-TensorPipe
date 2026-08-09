"""Atomic artifact write helpers for tensor outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .logging import TENSOR_RUN_ID_ENV
from .transaction_manifest import (
    TRANSACTION_MANIFEST_SCHEMA,
    transaction_manifest_path,
    write_transaction_manifest,
)


def write_outputs_atomically(
    outputs: list[tuple[Path, Callable[[Path], None]]],
    *,
    replace_fn: Callable[[Path, Path], Path] | None = None,
) -> None:
    """Write multiple artifacts atomically with rollback on failure."""
    if not outputs:
        return

    token = uuid4().hex
    prepared_outputs: list[tuple[Path, Path, Path, Callable[[Path], None]]] = []
    had_original: dict[Path, bool] = {}
    committed: list[Path] = []
    promotion_complete = False
    runtime_replace = replace_fn or (lambda src, dst: src.replace(dst))
    run_id = os.environ.get(TENSOR_RUN_ID_ENV, "").strip()

    for raw_target_path, writer in outputs:
        target_path = raw_target_path.expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.parent / f".{target_path.name}.tmp-{token}"
        backup_path = target_path.parent / f".{target_path.name}.bak-{token}"
        prepared_outputs.append((target_path, tmp_path, backup_path, writer))
        had_original[target_path] = target_path.exists()

    manifest_path = None
    manifest_payload = None
    if run_id:
        first_target = prepared_outputs[0][0]
        manifest_path = transaction_manifest_path(
            first_target,
            run_id=run_id,
            token=token,
        )
        manifest_payload = {
            "schema": TRANSACTION_MANIFEST_SCHEMA,
            "run_id": run_id,
            "phase": "prepared",
            "entries": [
                {
                    "target": str(target_path),
                    "temporary": str(tmp_path),
                    "backup": str(backup_path),
                    "had_original": had_original[target_path],
                }
                for target_path, tmp_path, backup_path, _ in prepared_outputs
            ],
        }
        try:
            write_transaction_manifest(manifest_path, manifest_payload)
        except Exception:
            manifest_path.with_name(f"{manifest_path.name}.writing").unlink(
                missing_ok=True
            )
            raise

    try:
        for _, tmp_path, _, writer in prepared_outputs:
            writer(tmp_path)

        for target_path, _, backup_path, _ in prepared_outputs:
            if had_original[target_path]:
                runtime_replace(target_path, backup_path)

        for target_path, tmp_path, _, _ in prepared_outputs:
            runtime_replace(tmp_path, target_path)
            committed.append(target_path)

        if manifest_path is not None and manifest_payload is not None:
            manifest_payload["phase"] = "committed"
            write_transaction_manifest(manifest_path, manifest_payload)
        promotion_complete = True
    except Exception:
        if promotion_complete:
            raise
        rollback_complete = False
        try:
            for target_path in committed:
                target_path.unlink(missing_ok=True)
            for target_path, _, backup_path, _ in prepared_outputs:
                if backup_path.exists():
                    runtime_replace(backup_path, target_path)
            rollback_complete = True
        finally:
            if rollback_complete:
                for _, tmp_path, backup_path, _ in prepared_outputs:
                    tmp_path.unlink(missing_ok=True)
                    backup_path.unlink(missing_ok=True)
                if manifest_path is not None:
                    manifest_path.unlink(missing_ok=True)
                    manifest_path.with_name(f"{manifest_path.name}.writing").unlink(
                        missing_ok=True
                    )
        raise
    else:
        if manifest_path is None:
            for _, tmp_path, backup_path, _ in prepared_outputs:
                tmp_path.unlink(missing_ok=True)
                backup_path.unlink(missing_ok=True)
            return
        try:
            for _, tmp_path, backup_path, _ in prepared_outputs:
                tmp_path.unlink(missing_ok=True)
                backup_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            manifest_path.with_name(f"{manifest_path.name}.writing").unlink(
                missing_ok=True
            )
        except OSError:
            # The committed manifest owns cleanup at the quiescent run boundary.
            return


__all__ = ["write_outputs_atomically"]
