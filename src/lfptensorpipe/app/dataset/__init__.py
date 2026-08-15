"""Dataset orchestration package.

Domain logic here coordinates indexing/import/delete flows and delegates low-level
IO/compute to backend modules.
"""

from __future__ import annotations

from .service import (
    RECORD_DELETE_SCOPES,
    RecordDeleteResult,
    RecordImportResult,
    RecordRenameRecoveryResult,
    RecordRenameResult,
    apply_reset_reference,
    build_import_sync_seed,
    build_import_synced_raw,
    create_subject,
    delete_record,
    estimate_import_sync,
    import_record,
    import_record_from_raw,
    load_import_channel_names,
    parse_record_source,
    persist_import_sync_artifacts,
    record_delete_scope_paths,
    recover_record_rename,
    rename_record,
    validate_record_name,
    validate_subject_name,
)

__all__ = [
    "RECORD_DELETE_SCOPES",
    "RecordDeleteResult",
    "RecordImportResult",
    "RecordRenameRecoveryResult",
    "RecordRenameResult",
    "apply_reset_reference",
    "build_import_sync_seed",
    "build_import_synced_raw",
    "create_subject",
    "delete_record",
    "estimate_import_sync",
    "import_record",
    "import_record_from_raw",
    "load_import_channel_names",
    "parse_record_source",
    "persist_import_sync_artifacts",
    "record_delete_scope_paths",
    "recover_record_rename",
    "rename_record",
    "validate_record_name",
    "validate_subject_name",
]
