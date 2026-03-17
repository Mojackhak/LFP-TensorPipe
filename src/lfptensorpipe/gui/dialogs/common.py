"""Shared dialog support for MainWindow-related dialogs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lfptensorpipe.app import (
    ALIGNMENT_METHODS,
    DEFAULT_TENSOR_BANDS,
    ECG_METHODS,
    TENSOR_METRICS,
    AppConfigStore,
    PathResolver,
    RecordContext,
    alignment_metric_tensor_warped_path,
    alignment_paradigm_log_path,
    apply_annotations_step,
    apply_bad_segment_step,
    apply_ecg_step,
    apply_filter_step,
    apply_finish_step,
    apply_reset_reference,
    bootstrap_raw_step_from_rawdata,
    build_tensor_metric_notch_payload,
    can_open_contact_viewer,
    compute_tensor_metric_filter_notch_warnings,
    create_alignment_paradigm,
    create_subject,
    default_alignment_method_params,
    default_filter_advance_params,
    default_tensor_metric_notch_params,
    delete_alignment_paradigm,
    delete_record,
    discover_atlases,
    discover_records,
    discover_subjects,
    extract_features_indicator_state,
    finish_alignment_epochs,
    has_reconstruction_mat,
    import_record_from_raw,
    infer_subject_space,
    invalidate_downstream_preproc_steps,
    is_stale_context_message,
    launch_contact_viewer,
    load_alignment_annotation_labels,
    load_alignment_epoch_rows,
    load_alignment_paradigms,
    load_annotations_csv_rows,
    load_burst_baseline_annotation_labels,
    load_derive_defaults,
    load_localize_paths,
    load_reconstruction_contacts,
    load_tensor_filter_metric_notch_params,
    load_tensor_frequency_defaults,
    localize_indicator_state,
    localize_representative_csv_path,
    mark_preproc_step,
    matlab_runtime_status,
    normalize_filter_advance_params,
    normalize_tensor_metric_notch_params,
    parse_record_source,
    preproc_step_log_path,
    preproc_step_raw_path,
    rawdata_input_fif_path,
    read_run_log,
    read_ui_state,
    reset_matlab_runtime,
    resolve_demo_data_root,
    resolve_demo_data_source_readonly,
    resolve_finish_source,
    run_align_epochs,
    run_build_tensor,
    run_extract_features,
    run_localize_apply,
    save_alignment_method_default_params,
    scan_stage_states,
    shutdown_matlab_runtime,
    tensor_metric_log_path,
    update_alignment_paradigm,
    validate_alignment_method_params,
    validate_record_name,
    validate_tensor_frequency_params,
    warmup_matlab_async,
    write_ui_state,
)
from lfptensorpipe.app.runlog_store import indicator_from_log
from lfptensorpipe.gui.dialogs import (
    all_possible_pairs as _dialog_all_possible_pairs,
    auto_channel_pair as _dialog_auto_channel_pair,
    auto_contact_index_side as _dialog_auto_contact_index_side,
    checked_item_texts as _dialog_checked_item_texts,
    normalize_pair as _dialog_normalize_pair,
    set_all_check_state as _dialog_set_all_check_state,
)
from lfptensorpipe.gui.dialogs.common_constants import *  # noqa: F403
from lfptensorpipe.gui.dialogs.common_state import *  # noqa: F403
from lfptensorpipe.gui.shell import (
    actions as _shell_actions,
    busy_state as _shell_busy_state,
    routing as _shell_routing,
)
from lfptensorpipe.gui.stages import (
    alignment_panel as _stage_alignment_panel,
    features_panel as _stage_features_panel,
    localize_panel as _stage_localize_panel,
    preproc_panel as _stage_preproc_panel,
    tensor_panel as _stage_tensor_panel,
)
from lfptensorpipe.gui.state import (
    deep_merge_dict as _state_deep_merge_dict,
    default_preproc_filter_basic_params as _state_default_preproc_filter_basic_params,
    default_preproc_viz_psd_params as _state_default_preproc_viz_psd_params,
    default_preproc_viz_tfr_params as _state_default_preproc_viz_tfr_params,
    nested_get as _state_nested_get,
    nested_set as _state_nested_set,
    normalize_filter_notches_config as _state_normalize_filter_notches_config,
    normalize_preproc_filter_basic_params as _state_normalize_preproc_filter_basic_params,
    normalize_preproc_viz_psd_params as _state_normalize_preproc_viz_psd_params,
    normalize_preproc_viz_tfr_params as _state_normalize_preproc_viz_tfr_params,
)
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl
from lfptensorpipe.stats.preproc.normalize import normalize_df_by_baseline
from lfptensorpipe.stats.preproc.transform import transform_df

ACTION_PAYLOAD_ROLE = Qt.UserRole + 1


def make_action_table_item(
    text: str,
    payload: Any,
    *,
    tool_tip: str = "",
) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    item.setTextAlignment(Qt.AlignCenter)
    item.setData(ACTION_PAYLOAD_ROLE, payload)
    item.setToolTip(tool_tip)
    return item


__all__ = [
    name for name in globals() if not (name.startswith("__") and name.endswith("__"))
]
