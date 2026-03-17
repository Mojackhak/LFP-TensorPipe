"""Alignment orchestration package.

App layer coordinates paradigm/method selection and persistence; warping
implementations should live in backend modules.
"""

from __future__ import annotations

from .service import (
    ALIGNMENT_METHODS,
    ALIGNMENT_METHODS_BY_KEY,
    ALIGNMENT_METHODS_BY_LABEL,
    AlignmentMethodSpec,
    alignment_metric_tensor_warped_path,
    alignment_epoch_inspector_state,
    alignment_method_panel_state,
    alignment_paradigm_log_path,
    alignment_trial_stage_state,
    create_alignment_paradigm,
    default_alignment_method_params,
    delete_alignment_paradigm,
    finish_alignment_epochs,
    load_alignment_annotation_labels,
    load_alignment_epoch_picks,
    load_alignment_epoch_rows,
    load_alignment_method_default_params,
    load_alignment_method_defaults,
    load_alignment_paradigms,
    persist_alignment_epoch_picks,
    run_align_epochs,
    save_alignment_method_default_params,
    save_alignment_paradigms,
    update_alignment_paradigm,
    validate_alignment_method_params,
)

__all__ = [
    "ALIGNMENT_METHODS",
    "ALIGNMENT_METHODS_BY_KEY",
    "ALIGNMENT_METHODS_BY_LABEL",
    "AlignmentMethodSpec",
    "alignment_metric_tensor_warped_path",
    "alignment_epoch_inspector_state",
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
]
