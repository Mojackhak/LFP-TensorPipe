"""Stable application-layer import surface.

`lfptensorpipe.app` is a convenience API for GUI-facing orchestration
entrypoints. Exports are resolved lazily so importing one app submodule does not
eagerly load every optional runtime dependency.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_DATASET_EXPORTS = (
    "RecordDeleteResult",
    "RecordImportResult",
    "apply_reset_reference",
    "create_subject",
    "delete_record",
    "import_record",
    "import_record_from_raw",
    "load_import_channel_names",
    "parse_record_source",
    "validate_record_name",
    "validate_subject_name",
)

_LOCALIZE_EXPORTS = (
    "ContactViewerLauncher",
    "LocalizePaths",
    "LocalizeRuntimeRunner",
    "can_open_contact_viewer",
    "clear_localize_runtime_cache",
    "discover_atlases",
    "discover_spaces",
    "has_reconstruction_mat",
    "infer_subject_space",
    "is_stale_context_message",
    "launch_contact_viewer",
    "localize_csv_path",
    "matlab_runtime_status",
    "reset_matlab_runtime",
    "shutdown_matlab_runtime",
    "submit_matlab_task",
    "load_localize_paths",
    "load_reconstruction_contacts",
    "localize_indicator_state",
    "localize_match_signature",
    "localize_panel_state",
    "localize_mat_path",
    "localize_ordered_pair_representative_csv_path",
    "localize_ordered_pair_representative_pkl_path",
    "localize_representative_csv_path",
    "localize_representative_pkl_path",
    "localize_undirected_pair_representative_csv_path",
    "localize_undirected_pair_representative_pkl_path",
    "warmup_matlab_async",
    "run_localize_apply",
)

_PREPROC_EXPORTS = (
    "ECG_METHODS",
    "FINISH_SOURCE_PRIORITY",
    "PREPROC_STEPS",
    "apply_bad_segment_step",
    "apply_annotations_step",
    "apply_ecg_step",
    "apply_filter_step",
    "apply_finish_step",
    "bootstrap_raw_step_from_rawdata",
    "default_filter_advance_params",
    "invalidate_downstream_preproc_steps",
    "load_annotations_csv_rows",
    "mark_preproc_step",
    "normalize_filter_advance_params",
    "preproc_annotations_panel_state",
    "preproc_ecg_panel_state",
    "preproc_filter_panel_state",
    "preproc_step_config_path",
    "preproc_step_log_path",
    "preproc_step_raw_path",
    "rawdata_input_fif_path",
    "resolve_finish_source",
    "write_preproc_step_config",
)

_ALIGNMENT_EXPORTS = (
    "AlignmentMethodSpec",
    "ALIGNMENT_METHODS",
    "ALIGNMENT_METHODS_BY_KEY",
    "ALIGNMENT_METHODS_BY_LABEL",
    "alignment_epoch_inspector_state",
    "alignment_metric_tensor_warped_path",
    "alignment_method_panel_state",
    "alignment_paradigm_log_path",
    "alignment_trial_stage_state",
    "create_alignment_paradigm",
    "default_alignment_method_params",
    "delete_alignment_paradigm",
    "finish_alignment_epochs",
    "load_alignment_annotation_labels",
    "load_alignment_epoch_picks",
    "load_alignment_epoch_rows",
    "load_alignment_method_default_params",
    "load_alignment_method_defaults",
    "load_alignment_paradigms",
    "persist_alignment_epoch_picks",
    "run_align_epochs",
    "save_alignment_method_default_params",
    "save_alignment_paradigms",
    "update_alignment_paradigm",
    "validate_alignment_method_params",
)

_FEATURES_EXPORTS = (
    "extract_features_indicator_state",
    "features_panel_state",
    "features_derivatives_log_path",
    "features_derivatives_root",
    "features_normalization_log_path",
    "features_normalization_root",
    "invalidate_normalization_logs",
    "load_derive_defaults",
    "normalization_indicator_state",
    "run_extract_features",
    "run_normalization",
)

_SHARED_EXPORTS = (
    "AppConfigStore",
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
    "discover_records",
    "discover_subjects",
    "invalidate_after_alignment_finish",
    "invalidate_after_alignment_run",
    "invalidate_after_localize_result_change",
    "invalidate_after_preproc_result_change",
    "invalidate_after_tensor_result_change",
    "indicator_from_log",
    "infer_run_log_version",
    "latest_run_log_entry",
    "read_run_log",
    "read_run_log_raw",
    "resolve_demo_data_root",
    "resolve_demo_data_source_readonly",
    "record_run_log_paths",
    "register_run_log_migration",
    "register_run_log_version",
    "scan_stage_states",
    "stamp_run_log_metadata",
    "read_ui_state",
    "update_run_log_state",
    "upgrade_record_run_logs",
    "upgrade_run_log_file",
    "upgrade_run_log_payload",
    "write_run_log",
    "write_ui_state",
)

_TENSOR_EXPORTS = (
    "DEFAULT_TENSOR_BANDS",
    "DEFAULT_TENSOR_NOTCH_WIDTH",
    "TENSOR_METRICS",
    "TENSOR_METRICS_BY_KEY",
    "TensorFilterInheritance",
    "TensorFrequencyBounds",
    "TensorMetricSpec",
    "build_tensor_metric_notch_payload",
    "compute_tensor_metric_filter_notch_warnings",
    "default_tensor_metric_notch_params",
    "load_burst_baseline_annotation_labels",
    "load_tensor_filter_inheritance",
    "load_tensor_filter_metric_notch_params",
    "load_tensor_frequency_defaults",
    "normalize_tensor_metric_notch_params",
    "resolve_tensor_frequency_bounds",
    "resolve_tensor_metric_interest_range",
    "run_build_tensor",
    "tensor_metric_config_path",
    "tensor_metric_log_path",
    "tensor_metric_panel_state",
    "tensor_metric_tensor_path",
    "validate_tensor_frequency_params",
)

_MODULE_EXPORTS = (
    (".dataset", _DATASET_EXPORTS),
    (".localize", _LOCALIZE_EXPORTS),
    (".preproc", _PREPROC_EXPORTS),
    (".alignment", _ALIGNMENT_EXPORTS),
    (".features", _FEATURES_EXPORTS),
    (".shared", _SHARED_EXPORTS),
    (".tensor", _TENSOR_EXPORTS),
)

_EXPORT_TO_MODULE = {
    name: module_name
    for module_name, export_names in _MODULE_EXPORTS
    for name in export_names
}

__all__ = [name for _, export_names in _MODULE_EXPORTS for name in export_names]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
