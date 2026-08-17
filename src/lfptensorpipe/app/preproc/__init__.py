"""Preprocess orchestration package.

App layer controls stage flow; numerical signal operations belong in
`lfptensorpipe.preproc` backend modules.
"""

from __future__ import annotations

from .service import (
    ECG_METHODS,
    FINISH_SOURCE_PRIORITY,
    PREPROC_STEPS,
    apply_bad_segment_step,
    apply_annotations_step,
    apply_ecg_step,
    apply_filter_step,
    apply_finish_step,
    bootstrap_raw_step_from_rawdata,
    default_ecg_method_params,
    default_ecg_params_by_method,
    default_filter_advance_params,
    ecg_method_runtime_kwargs,
    filter_nyquist_warning,
    invalidate_downstream_preproc_steps,
    load_annotations_csv_rows,
    mark_preproc_step,
    normalize_ecg_method_params,
    normalize_ecg_params_by_method,
    normalize_filter_advance_params,
    normalize_filter_runtime_params,
    preproc_step_config_path,
    preproc_step_log_path,
    preproc_step_raw_path,
    rawdata_input_fif_path,
    resolve_finish_source,
    resolve_preproc_step_source,
    write_preproc_step_config,
)
from .indicator import (
    preproc_annotations_panel_state,
    preproc_ecg_panel_state,
    preproc_filter_panel_state,
    preproc_step_indicator_state,
)

__all__ = [
    "ECG_METHODS",
    "FINISH_SOURCE_PRIORITY",
    "PREPROC_STEPS",
    "apply_bad_segment_step",
    "apply_annotations_step",
    "apply_ecg_step",
    "apply_filter_step",
    "apply_finish_step",
    "bootstrap_raw_step_from_rawdata",
    "default_ecg_method_params",
    "default_ecg_params_by_method",
    "default_filter_advance_params",
    "ecg_method_runtime_kwargs",
    "filter_nyquist_warning",
    "invalidate_downstream_preproc_steps",
    "load_annotations_csv_rows",
    "mark_preproc_step",
    "normalize_ecg_method_params",
    "normalize_ecg_params_by_method",
    "normalize_filter_advance_params",
    "normalize_filter_runtime_params",
    "preproc_step_config_path",
    "preproc_step_log_path",
    "preproc_step_raw_path",
    "preproc_annotations_panel_state",
    "preproc_ecg_panel_state",
    "preproc_filter_panel_state",
    "preproc_step_indicator_state",
    "rawdata_input_fif_path",
    "resolve_finish_source",
    "resolve_preproc_step_source",
    "write_preproc_step_config",
]
