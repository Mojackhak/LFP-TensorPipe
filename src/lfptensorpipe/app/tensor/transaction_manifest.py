"""Run-scoped recovery manifests for multi-artifact Tensor transactions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRANSACTION_MANIFEST_SCHEMA = "lfptensorpipe.tensor-output-transaction"
TRANSACTION_MANIFEST_PREFIX = ".lfptensorpipe-txn-"


def transaction_manifest_path(
    first_target: Path,
    *,
    run_id: str,
    token: str,
) -> Path:
    return first_target.parent / f"{TRANSACTION_MANIFEST_PREFIX}{run_id}-{token}.json"


def write_transaction_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create or update one transaction manifest."""
    temporary = path.with_name(f"{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _scoped_path(value: Any, *, validation_root: Path) -> Path:
    path = Path(str(value)).expanduser().resolve()
    root = validation_root.expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Transaction path is outside the Tensor root: {path}")
    return path


def _load_manifest(
    path: Path,
    *,
    validation_root: Path,
    run_id: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Transaction manifest must be an object: {path}")
    if payload.get("schema") != TRANSACTION_MANIFEST_SCHEMA:
        raise ValueError(f"Unexpected transaction manifest schema: {path}")
    if str(payload.get("run_id", "")) != str(run_id):
        raise ValueError(f"Transaction manifest run ID mismatch: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Transaction manifest has no entries: {path}")
    normalized_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid transaction manifest entry: {path}")
        normalized_entries.append(
            {
                "target": _scoped_path(
                    entry.get("target"), validation_root=validation_root
                ),
                "temporary": _scoped_path(
                    entry.get("temporary"), validation_root=validation_root
                ),
                "backup": _scoped_path(
                    entry.get("backup"), validation_root=validation_root
                ),
                "had_original": bool(entry.get("had_original", False)),
            }
        )
    return {
        "phase": str(payload.get("phase", "prepared")),
        "entries": normalized_entries,
    }


def _rollback_prepared(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        target = entry["target"]
        temporary = entry["temporary"]
        backup = entry["backup"]
        had_original = bool(entry["had_original"])
        if had_original:
            if backup.exists():
                target.unlink(missing_ok=True)
                backup.replace(target)
            elif not target.exists():
                raise RuntimeError(
                    f"Cannot recover missing original Tensor artifact: {target}"
                )
        else:
            target.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)


def _complete_committed(entries: list[dict[str, Any]]) -> None:
    missing = [entry["target"] for entry in entries if not entry["target"].exists()]
    if missing:
        raise RuntimeError(
            "Committed Tensor transaction is missing target artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    for entry in entries:
        entry["temporary"].unlink(missing_ok=True)
        entry["backup"].unlink(missing_ok=True)


def recover_tensor_run_transactions(
    manifest_scan_root: Path,
    *,
    validation_root: Path,
    run_id: str,
) -> list[Path]:
    """Recover run transactions discovered below one owner-scoped directory."""
    scan_root = manifest_scan_root.expanduser().resolve()
    allowed_root = validation_root.expanduser().resolve()
    if not scan_root.is_relative_to(allowed_root):
        raise ValueError(
            f"Transaction manifest scan root is outside the Tensor root: {scan_root}"
        )
    if not scan_root.exists():
        return []
    pattern = f"{TRANSACTION_MANIFEST_PREFIX}{run_id}-*.json"
    recovered: list[Path] = []
    for manifest_path in sorted(scan_root.rglob(pattern)):
        payload = _load_manifest(
            manifest_path,
            validation_root=allowed_root,
            run_id=run_id,
        )
        entries = payload["entries"]
        if payload["phase"] == "committed":
            _complete_committed(entries)
        else:
            _rollback_prepared(entries)
        manifest_path.unlink(missing_ok=True)
        manifest_path.with_name(f"{manifest_path.name}.writing").unlink(missing_ok=True)
        recovered.append(manifest_path)
    for writing_path in scan_root.rglob(f"{pattern}.writing"):
        writing_path.unlink(missing_ok=True)
    return recovered


__all__ = [
    "TRANSACTION_MANIFEST_SCHEMA",
    "recover_tensor_run_transactions",
    "transaction_manifest_path",
    "write_transaction_manifest",
]
