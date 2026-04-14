"""Record-scoped run-log upgrade helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .path_resolver import PathResolver, RecordContext
from .runlog_store import upgrade_run_log_file


@dataclass(frozen=True)
class RecordRunLogUpgradeSummary:
    """Summary of one record-scoped upgrade scan."""

    scanned_count: int = 0
    upgraded_count: int = 0
    failed_count: int = 0


def record_run_log_paths(
    project_root: Path,
    subject: str,
    record: str,
) -> tuple[Path, ...]:
    """List all run-log files stored under one record root."""
    resolver = PathResolver(
        RecordContext(project_root=project_root, subject=subject, record=record)
    )
    if not resolver.lfp_root.exists():
        return ()
    return tuple(sorted(resolver.lfp_root.rglob("lfptensorpipe_log.json")))


def upgrade_record_run_logs(
    project_root: Path,
    subject: str,
    record: str,
) -> RecordRunLogUpgradeSummary:
    """Upgrade all discovered run logs for one selected record."""
    scanned_count = 0
    upgraded_count = 0
    failed_count = 0

    for log_path in record_run_log_paths(project_root, subject, record):
        scanned_count += 1
        try:
            _, changed = upgrade_run_log_file(log_path)
        except Exception:
            failed_count += 1
            continue
        if changed:
            upgraded_count += 1

    return RecordRunLogUpgradeSummary(
        scanned_count=scanned_count,
        upgraded_count=upgraded_count,
        failed_count=failed_count,
    )


__all__ = [
    "RecordRunLogUpgradeSummary",
    "record_run_log_paths",
    "upgrade_record_run_logs",
]
