"""Dataset orchestration package.

Domain logic here coordinates indexing/import/delete flows and delegates low-level
IO/compute to backend modules.
"""

from __future__ import annotations

from .service import (
    RecordDeleteResult,
    RecordImportResult,
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
    rename_record,
    validate_record_name,
    validate_subject_name,
)

__all__ = [
    "RecordDeleteResult",
    "RecordImportResult",
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
    "rename_record",
    "validate_record_name",
    "validate_subject_name",
]
