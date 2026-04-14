"""Shared app-layer orchestration utilities.

This package hosts only cross-domain app infrastructure such as config, path,
and run-log adapters. Domain workflows must not be moved here as a shortcut.
"""

from __future__ import annotations

from .config_store import AppConfigStore
from .dataset_index import (
    discover_records,
    discover_subjects,
    resolve_demo_data_root,
    resolve_demo_data_source_readonly,
    scan_stage_states,
)
from .downstream_invalidation import (
    invalidate_after_alignment_finish,
    invalidate_after_alignment_run,
    invalidate_after_localize_result_change,
    invalidate_after_preproc_result_change,
    invalidate_after_tensor_result_change,
)
from .path_resolver import PathResolver, RecordContext
from .record_log_migration import (
    RecordRunLogUpgradeSummary,
    record_run_log_paths,
    upgrade_record_run_logs,
)
from .runlog_migrations import (
    RUNLOG_ALL_STEPS,
    RUNLOG_BASE_VERSION,
    RUNLOG_MIGRATION_META_KEY,
    RUNLOG_SCHEMA_KEY,
    RUNLOG_SCHEMA_NAME,
    RUNLOG_UPGRADES_KEY,
    RUNLOG_VERSION_KEY,
    RunLogMigrationSpec,
    current_run_log_version,
    infer_run_log_version,
    register_run_log_migration,
    register_run_log_version,
    stamp_run_log_metadata,
    upgrade_run_log_payload,
)
from .runlog_store import (
    RunLogRecord,
    append_run_log_event,
    indicator_from_log,
    latest_run_log_entry,
    read_run_log,
    read_run_log_raw,
    read_ui_state,
    update_run_log_state,
    upgrade_run_log_file,
    write_run_log,
    write_ui_state,
)

__all__ = [
    "AppConfigStore",
    "discover_records",
    "discover_subjects",
    "invalidate_after_alignment_finish",
    "invalidate_after_alignment_run",
    "invalidate_after_localize_result_change",
    "invalidate_after_preproc_result_change",
    "invalidate_after_tensor_result_change",
    "resolve_demo_data_root",
    "resolve_demo_data_source_readonly",
    "scan_stage_states",
    "PathResolver",
    "RecordContext",
    "RecordRunLogUpgradeSummary",
    "RunLogRecord",
    "RUNLOG_ALL_STEPS",
    "RUNLOG_BASE_VERSION",
    "RUNLOG_MIGRATION_META_KEY",
    "RUNLOG_SCHEMA_KEY",
    "RUNLOG_SCHEMA_NAME",
    "RUNLOG_UPGRADES_KEY",
    "RUNLOG_VERSION_KEY",
    "append_run_log_event",
    "current_run_log_version",
    "indicator_from_log",
    "infer_run_log_version",
    "latest_run_log_entry",
    "read_run_log",
    "read_run_log_raw",
    "read_ui_state",
    "record_run_log_paths",
    "register_run_log_migration",
    "register_run_log_version",
    "update_run_log_state",
    "write_run_log",
    "write_ui_state",
    "RunLogMigrationSpec",
    "stamp_run_log_metadata",
    "upgrade_record_run_logs",
    "upgrade_run_log_file",
    "upgrade_run_log_payload",
]
