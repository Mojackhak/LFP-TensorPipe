"""Shared app-layer orchestration utilities.

This package hosts only cross-domain app infrastructure such as config, path,
and run-log adapters. Domain workflows must not be moved here as a shortcut.
"""

from __future__ import annotations

from .config_store import AppConfigStore
from .dataset_index import (
    STANDARD_RECORD_SCOPES,
    discover_records,
    discover_subjects,
    standard_record_scope_roots,
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
from .generation_lineage import (
    GENERATION_RECEIPT_SEMANTICS,
    GENERATION_RECEIPT_SEMANTICS_KEY,
    INPUT_GENERATIONS_KEY,
    LOCALIZE_GENERATION_REF,
    RESULT_GENERATION_ID_KEY,
    GenerationLineage,
    accepted_result_generation_id,
    alignment_generation_ref,
    input_generation_receipts_match,
    new_result_generation_id,
    params_with_generation_lineage,
    params_with_input_generation_receipts,
    parse_generation_lineage,
    preproc_generation_ref,
    tensor_generation_ref,
)
from .path_resolver import PathResolver, RecordContext
from .runlog_store import (
    RUNLOG_SCHEMA_KEY,
    RUNLOG_SCHEMA_NAME,
    RUNLOG_SCHEMA_VERSION,
    RUNLOG_VERSION_KEY,
    RunLogRecord,
    append_run_log_event,
    indicator_from_log,
    latest_run_log_entry,
    read_run_log,
    read_ui_state,
    update_run_log_state,
    write_run_log,
    write_ui_state,
)

__all__ = [
    "GENERATION_RECEIPT_SEMANTICS",
    "GENERATION_RECEIPT_SEMANTICS_KEY",
    "INPUT_GENERATIONS_KEY",
    "LOCALIZE_GENERATION_REF",
    "RESULT_GENERATION_ID_KEY",
    "GenerationLineage",
    "AppConfigStore",
    "STANDARD_RECORD_SCOPES",
    "discover_records",
    "discover_subjects",
    "standard_record_scope_roots",
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
    "RunLogRecord",
    "RUNLOG_SCHEMA_KEY",
    "RUNLOG_SCHEMA_NAME",
    "RUNLOG_SCHEMA_VERSION",
    "RUNLOG_VERSION_KEY",
    "accepted_result_generation_id",
    "alignment_generation_ref",
    "append_run_log_event",
    "indicator_from_log",
    "input_generation_receipts_match",
    "latest_run_log_entry",
    "new_result_generation_id",
    "params_with_generation_lineage",
    "params_with_input_generation_receipts",
    "parse_generation_lineage",
    "preproc_generation_ref",
    "read_run_log",
    "read_ui_state",
    "update_run_log_state",
    "write_run_log",
    "write_ui_state",
    "tensor_generation_ref",
]
