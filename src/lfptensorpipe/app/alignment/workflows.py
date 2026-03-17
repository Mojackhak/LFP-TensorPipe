"""Align-Epochs stage service helpers."""

from __future__ import annotations

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import (
    preproc_step_log_path,
    preproc_step_raw_path,
)
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    append_run_log_event,
    indicator_from_log,
    read_run_log,
    update_run_log_state,
)
from lfptensorpipe.app.tensor_service import (
    tensor_metric_log_path,
    tensor_metric_tensor_path,
)
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl
from lfptensorpipe.lfp.common import infer_sfreq_from_times
from lfptensorpipe.lfp.warp import (
    concat_warper,
    linear_warper,
    pad_warper,
    stack_warper,
)
from lfptensorpipe.lfp.warp.metadata import build_warped_tensor_metadata
from lfptensorpipe.tabular.tensor_slices import split_tensor4d_to_nested_df
from .config_repo import (
    _append_alignment_history,
    _extract_trial_config_from_history,
    _finish_time_axis_values,
    _load_trial_config_from_log,
    _normalize_method_defaults,
    _normalize_paradigm,
    _read_alignment_payload,
    _resolve_alignment_method_key,
    _trial_config_from_payload,
    create_alignment_paradigm,
    delete_alignment_paradigm,
    load_alignment_method_default_params,
    load_alignment_method_defaults,
    load_alignment_paradigms,
    save_alignment_method_default_params,
    save_alignment_paradigms,
    update_alignment_paradigm,
)
from .indicator import (
    alignment_epoch_inspector_state,
    alignment_method_panel_state,
    alignment_trial_stage_state,
)
from .paths import (
    alignment_metric_tensor_warped_path,
    alignment_paradigm_dir,
    alignment_paradigm_log_path,
    alignment_trial_raw_table_path,
    alignment_warp_fn_path,
    alignment_warp_labels_path,
    features_raw_log_path,
    features_raw_table_path,
)
from .validation import (
    ALIGNMENT_METHODS,
    ALIGNMENT_METHODS_BY_KEY,
    ALIGNMENT_METHODS_BY_LABEL,
    DEFAULT_DROP_FIELDS,
    DEFAULT_DROP_MODE,
    AlignmentMethodSpec,
    _normalize_anchors,
    _normalize_annotations,
    _normalize_drop_fields,
    _normalize_duration_range,
    _normalize_nonnegative_float,
    _normalize_sample_rate,
    _normalize_slug,
    default_alignment_method_params,
    validate_alignment_method_params,
)
from .warping import (
    _build_warper,
    _coerce_alignment_tensor,
    _completed_tensor_metrics,
    _filter_raw_annotations_by_duration,
    _float_pair_list,
    _load_raw_for_warp,
    _load_unique_annotation_labels,
    _localize_ordered_pair_representative_pkl_path,
    _localize_representative_csv_path,
    _localize_representative_pkl_path,
    _localize_undirected_pair_representative_pkl_path,
    _merge_channel_representative_coords,
    _merge_representative_coords_for_metric,
    _repcoord_conflict_column_name,
    _resolve_target_n_samples,
    load_alignment_annotation_labels,
)

__all__ = [
    "AlignmentMethodSpec",
    "ALIGNMENT_METHODS",
    "ALIGNMENT_METHODS_BY_KEY",
    "ALIGNMENT_METHODS_BY_LABEL",
    "DEFAULT_DROP_FIELDS",
    "DEFAULT_DROP_MODE",
    "default_alignment_method_params",
    "validate_alignment_method_params",
    "_normalize_sample_rate",
    "_normalize_nonnegative_float",
    "_normalize_duration_range",
    "_normalize_annotations",
    "_normalize_drop_fields",
    "_normalize_anchors",
    "_normalize_slug",
    "_read_alignment_payload",
    "_normalize_method_defaults",
    "_normalize_paradigm",
    "_trial_config_from_payload",
    "_extract_trial_config_from_history",
    "_load_trial_config_from_log",
    "_resolve_alignment_method_key",
    "_finish_time_axis_values",
    "_append_alignment_history",
    "load_alignment_paradigms",
    "save_alignment_paradigms",
    "load_alignment_method_defaults",
    "load_alignment_method_default_params",
    "save_alignment_method_default_params",
    "create_alignment_paradigm",
    "delete_alignment_paradigm",
    "update_alignment_paradigm",
    "_localize_ordered_pair_representative_pkl_path",
    "_localize_representative_csv_path",
    "_localize_representative_pkl_path",
    "_localize_undirected_pair_representative_pkl_path",
    "_repcoord_conflict_column_name",
    "_merge_channel_representative_coords",
    "_merge_representative_coords_for_metric",
    "_load_unique_annotation_labels",
    "load_alignment_annotation_labels",
    "_load_raw_for_warp",
    "_completed_tensor_metrics",
    "_float_pair_list",
    "_filter_raw_annotations_by_duration",
    "_build_warper",
    "_resolve_target_n_samples",
    "_coerce_alignment_tensor",
    "run_align_epochs",
    "finish_alignment_epochs",
    "load_alignment_epoch_picks",
    "load_alignment_epoch_rows",
    "persist_alignment_epoch_picks",
    "alignment_paradigm_dir",
    "alignment_paradigm_log_path",
    "alignment_method_panel_state",
    "alignment_epoch_inspector_state",
    "alignment_trial_stage_state",
    "alignment_warp_fn_path",
    "alignment_warp_labels_path",
    "alignment_metric_tensor_warped_path",
    "alignment_trial_raw_table_path",
    "features_raw_table_path",
    "features_raw_log_path",
    "load_pkl",
    "save_pkl",
    "AppConfigStore",
    "PathResolver",
    "RecordContext",
    "preproc_step_log_path",
    "preproc_step_raw_path",
    "RunLogRecord",
    "append_run_log_event",
    "indicator_from_log",
    "read_run_log",
    "update_run_log_state",
    "tensor_metric_log_path",
    "tensor_metric_tensor_path",
    "infer_sfreq_from_times",
    "build_warped_tensor_metadata",
    "split_tensor4d_to_nested_df",
    "linear_warper",
    "pad_warper",
    "stack_warper",
    "concat_warper",
]


def run_align_epochs(*args, **kwargs):
    from .runtime import run_align_epochs as _run_align_epochs

    return _run_align_epochs(*args, **kwargs)


def finish_alignment_epochs(*args, **kwargs):
    from .finish import finish_alignment_epochs as _finish_alignment_epochs

    return _finish_alignment_epochs(*args, **kwargs)


def load_alignment_epoch_rows(*args, **kwargs):
    from .epoch_view import load_alignment_epoch_rows as _load_alignment_epoch_rows

    return _load_alignment_epoch_rows(*args, **kwargs)


def load_alignment_epoch_picks(*args, **kwargs):
    from .epoch_view import load_alignment_epoch_picks as _load_alignment_epoch_picks

    return _load_alignment_epoch_picks(*args, **kwargs)


def persist_alignment_epoch_picks(*args, **kwargs):
    from .epoch_view import (
        persist_alignment_epoch_picks as _persist_alignment_epoch_picks,
    )

    return _persist_alignment_epoch_picks(*args, **kwargs)
