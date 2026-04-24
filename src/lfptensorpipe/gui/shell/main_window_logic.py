"""Main application shell and stage routing for LFP-TensorPipe."""

from __future__ import annotations

from lfptensorpipe.gui.shell import common as _shell_common
from lfptensorpipe.gui.shell import window_shutdown as _window_shutdown
from lfptensorpipe.gui.shell.common import (
    Any,
    AppConfigStore,
    BUSY_FRAMES,
    BUSY_INTERVAL_MS,
    BUTTON_TEXT_HORIZONTAL_PADDING,
    Callable,
    INDICATOR_COLORS,
    LEFT_WIDTH_MAX,
    LEFT_WIDTH_MIN,
    LEFT_WIDTH_RATIO,
    Path,
    QAbstractButton,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTimer,
    QWidget,
    ROOT_MARGIN,
    ROOT_SPACING,
    STAGE_CONTENT_MIN_WIDTH,
    STAGE_SPECS,
    T,
    WINDOW_DEFAULT_SIZE,
    WINDOW_MIN_SIZE,
    _shell_busy_state,
    _shell_routing,
    load_localize_paths,
    pd,
    resolve_demo_data_root,
    resolve_demo_data_source_readonly,
    shutdown_matlab_runtime,
)
from lfptensorpipe.gui.dialogs.annotation_configure import AnnotationConfigureDialog
from lfptensorpipe.gui.dialogs.alignment_method_params import (
    AlignmentMethodParamsDialog,
)
from lfptensorpipe.gui.dialogs.autosave_filter import _CloseAutosaveFilter
from lfptensorpipe.gui.dialogs.baseline_range import BaselineRangeConfigureDialog
from lfptensorpipe.gui.dialogs.channel_pair import ChannelPairDialog
from lfptensorpipe.gui.dialogs.channel_select import ChannelSelectDialog
from lfptensorpipe.gui.dialogs.dataset_types import (
    ParsedImportPreview,
    ResetReferenceRow,
)
from lfptensorpipe.gui.dialogs.feature_axis import FeatureAxisConfigureDialog
from lfptensorpipe.gui.dialogs.features_plot_advance import FeaturesPlotAdvanceDialog
from lfptensorpipe.gui.dialogs.filter_advance import FilterAdvanceDialog
from lfptensorpipe.gui.dialogs.localize_atlas import LocalizeAtlasDialog
from lfptensorpipe.gui.dialogs.localize_match import LocalizeMatchDialog
from lfptensorpipe.gui.dialogs.paths_config import PathsConfigDialog
from lfptensorpipe.gui.dialogs.qc_advance import QcAdvanceDialog
from lfptensorpipe.gui.dialogs.record_import import RecordImportDialog
from lfptensorpipe.gui.dialogs.reset_reference import ResetReferenceDialog
from lfptensorpipe.gui.dialogs.tensor_bands import TensorBandsConfigureDialog
from lfptensorpipe.gui.dialogs.tensor_channel_select import TensorChannelSelectDialog
from lfptensorpipe.gui.dialogs.tensor_metric_advance import TensorMetricAdvanceDialog
from lfptensorpipe.gui.dialogs.tensor_pair_select import TensorPairSelectDialog
from lfptensorpipe.gui.shell.tensor_logic import MainWindowTensorMixin
from lfptensorpipe.gui.shell.alignment_logic import MainWindowAlignmentMixin
from lfptensorpipe.gui.shell.features_logic import MainWindowFeaturesMixin
from lfptensorpipe.gui.shell.preproc_logic import MainWindowPreprocMixin
from lfptensorpipe.gui.shell.runtime_dependencies import (
    MainWindowRuntimeDependenciesMixin,
)
from lfptensorpipe.gui.shell.stage_state import MainWindowStageStateMixin
from lfptensorpipe.gui.shell.dataset_localize_logic import (
    MainWindowDatasetLocalizeMixin,
)
from lfptensorpipe.gui.shell import main_window_layout as _main_window_layout

__all__ = [
    "AnnotationConfigureDialog",
    "AlignmentMethodParamsDialog",
    "_CloseAutosaveFilter",
    "BaselineRangeConfigureDialog",
    "ChannelPairDialog",
    "ChannelSelectDialog",
    "ParsedImportPreview",
    "ResetReferenceRow",
    "FeatureAxisConfigureDialog",
    "FeaturesPlotAdvanceDialog",
    "FilterAdvanceDialog",
    "LocalizeAtlasDialog",
    "LocalizeMatchDialog",
    "PathsConfigDialog",
    "QcAdvanceDialog",
    "RecordImportDialog",
    "ResetReferenceDialog",
    "TensorBandsConfigureDialog",
    "TensorChannelSelectDialog",
    "TensorMetricAdvanceDialog",
    "TensorPairSelectDialog",
    "MainWindow",
]


def __getattr__(name: str):
    return getattr(_shell_common, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_shell_common)))


class MainWindow(
    MainWindowRuntimeDependenciesMixin,
    MainWindowStageStateMixin,
    MainWindowTensorMixin,
    MainWindowAlignmentMixin,
    MainWindowFeaturesMixin,
    MainWindowPreprocMixin,
    MainWindowDatasetLocalizeMixin,
    QMainWindow,
):
    """Top-level window with persistent left panel and stage pages."""

    def __init__(
        self,
        *,
        config_store: AppConfigStore | None = None,
        demo_data_root: Path | None = None,
        auto_load_dataset: bool = True,
        enable_plots: bool = True,
    ) -> None:
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            app.setStyle("Fusion")
            app_font = app.font()
            app_font.setFamily("Arial")
            app.setFont(app_font)
        self.setWindowTitle("LFP-TensorPipe")
        self.setMinimumSize(*WINDOW_MIN_SIZE)
        self.resize(*WINDOW_DEFAULT_SIZE)

        self._config_store = config_store or AppConfigStore()
        self._demo_data_root = demo_data_root or resolve_demo_data_root()
        self._demo_data_source_readonly = resolve_demo_data_source_readonly()
        self._localize_paths = load_localize_paths(self._config_store)
        self._current_project: Path | None = None
        self._current_subject: str | None = None
        self._current_record: str | None = None
        self._enable_plots = enable_plots
        self._annotations_edit_mode = False

        self._stage_raw_states = {spec.key: "gray" for spec in STAGE_SPECS}
        self._stage_states = {spec.key: "gray" for spec in STAGE_SPECS}
        self._stage_buttons: dict[str, QPushButton] = {}
        self._stage_indicators: dict[str, QLabel] = {}
        self._stage_page_index: dict[str, int] = {}
        self._active_stage_key = STAGE_SPECS[0].key
        self._settings_configs_action: QAction | None = None

        self._project_combo: QComboBox | None = None
        self._subject_combo: QComboBox | None = None
        self._record_list: QListWidget | None = None
        self._project_add_button: QPushButton | None = None
        self._subject_add_button: QPushButton | None = None
        self._record_add_button: QPushButton | None = None
        self._record_delete_button: QPushButton | None = None
        self._record_rename_button: QPushButton | None = None
        self._preproc_raw_plot_button: QPushButton | None = None
        self._preproc_filter_advance_button: QPushButton | None = None
        self._preproc_filter_apply_button: QPushButton | None = None
        self._preproc_filter_plot_button: QPushButton | None = None
        self._preproc_filter_notches_edit: QLineEdit | None = None
        self._preproc_filter_low_freq_edit: QLineEdit | None = None
        self._preproc_filter_high_freq_edit: QLineEdit | None = None
        self._preproc_annotations_table: QTableWidget | None = None
        self._preproc_annotations_edit_button: QPushButton | None = None
        self._preproc_annotations_save_button: QPushButton | None = None
        self._preproc_annotations_import_button: QPushButton | None = None
        self._preproc_annotations_plot_button: QPushButton | None = None
        self._preproc_bad_segment_apply_button: QPushButton | None = None
        self._preproc_bad_segment_plot_button: QPushButton | None = None
        self._preproc_ecg_method_combo: QComboBox | None = None
        self._preproc_ecg_channels_button: QPushButton | None = None
        self._preproc_ecg_available_channels: tuple[str, ...] = ()
        self._preproc_ecg_selected_channels: tuple[str, ...] = ()
        self._preproc_ecg_apply_button: QPushButton | None = None
        self._preproc_ecg_plot_button: QPushButton | None = None
        self._preproc_finish_apply_button: QPushButton | None = None
        self._preproc_finish_plot_button: QPushButton | None = None
        self._preproc_step_indicators: dict[str, QLabel] = {}
        self._preproc_viz_step_combo: QComboBox | None = None
        self._preproc_viz_channels_button: QPushButton | None = None
        self._preproc_viz_available_channels: tuple[str, ...] = ()
        self._preproc_viz_selected_channels: tuple[str, ...] = ()
        self._preproc_viz_psd_advance_button: QPushButton | None = None
        self._preproc_viz_psd_plot_button: QPushButton | None = None
        self._preproc_viz_tfr_advance_button: QPushButton | None = None
        self._preproc_viz_tfr_plot_button: QPushButton | None = None
        self._tensor_metric_checks: dict[str, QCheckBox] = {}
        self._tensor_metric_name_buttons: dict[str, QPushButton] = {}
        self._tensor_metric_indicators: dict[str, QLabel] = {}
        self._tensor_bands_table: QTableWidget | None = None
        self._tensor_param_metric_combo: QComboBox | None = None
        metric_specs = tuple(self._stage_tensor_metric_specs())
        self._tensor_active_metric_key: str = (
            metric_specs[0].key if metric_specs else "raw_power"
        )
        self._tensor_metric_params_form: QFormLayout | None = None
        self._tensor_basic_param_widgets: dict[str, QWidget] = {}
        self._tensor_low_freq_edit: QLineEdit | None = None
        self._tensor_high_freq_edit: QLineEdit | None = None
        self._tensor_step_edit: QLineEdit | None = None
        self._tensor_time_resolution_edit: QLineEdit | None = None
        self._tensor_hop_edit: QLineEdit | None = None
        self._tensor_method_combo: QComboBox | None = None
        self._tensor_freq_range_edit: QLineEdit | None = None
        self._tensor_percentile_edit: QLineEdit | None = None
        self._tensor_min_cycles_basic_edit: QLineEdit | None = None
        self._tensor_bands_configure_button: QPushButton | None = None
        self._tensor_advance_button: QPushButton | None = None
        self._tensor_channels_button: QPushButton | None = None
        self._tensor_pairs_button: QPushButton | None = None
        self._tensor_available_channels: tuple[str, ...] = ()
        self._tensor_selected_channels_by_metric: dict[str, tuple[str, ...]] = {}
        self._tensor_selected_pairs_by_metric: dict[
            str, tuple[tuple[str, str], ...]
        ] = {}
        self._tensor_metric_params: dict[str, dict[str, Any]] = {}
        self._tensor_metric_title_label: QLabel | None = None
        self._tensor_metric_notice_label: QLabel | None = None
        self._tensor_mask_edge_checkbox: QCheckBox | None = None
        self._tensor_import_button: QPushButton | None = None
        self._tensor_export_button: QPushButton | None = None
        self._tensor_run_button: QPushButton | None = None
        self._alignment_paradigm_list: QListWidget | None = None
        self._alignment_paradigm_add_button: QPushButton | None = None
        self._alignment_paradigm_delete_button: QPushButton | None = None
        self._alignment_method_combo: QComboBox | None = None
        self._alignment_method_params_button: QPushButton | None = None
        self._alignment_import_button: QPushButton | None = None
        self._alignment_export_button: QPushButton | None = None
        self._alignment_method_indicator: QLabel | None = None
        self._alignment_n_samples_edit: QLineEdit | None = None
        self._alignment_method_description_label: QLabel | None = None
        self._alignment_run_button: QPushButton | None = None
        self._alignment_epoch_inspector_indicator: QLabel | None = None
        self._alignment_epoch_metric_combo: QComboBox | None = None
        self._alignment_epoch_channel_combo: QComboBox | None = None
        self._alignment_epoch_table: QTableWidget | None = None
        self._alignment_select_all_button: QPushButton | None = None
        self._alignment_preview_button: QPushButton | None = None
        self._alignment_merge_location_status_label: QLabel | None = None
        self._alignment_finish_button: QPushButton | None = None
        self._alignment_epoch_rows: list[dict[str, Any]] = []
        self._alignment_paradigms: list[dict[str, Any]] = []
        self._shared_stage_trial_slug_value: str | None = None
        self._syncing_shared_trial_selection = False
        self._features_paradigm_list: QListWidget | None = None
        self._features_paradigm_add_button: QPushButton | None = None
        self._features_paradigm_delete_button: QPushButton | None = None
        self._features_extract_indicator: QLabel | None = None
        self._features_run_extract_button: QPushButton | None = None
        self._features_refresh_button: QPushButton | None = None
        self._features_available_table: QTableWidget | None = None
        self._features_filter_feature_edit: QLineEdit | None = None
        self._features_axis_metric_combo: QComboBox | None = None
        self._features_axis_bands_button: QPushButton | None = None
        self._features_axis_times_button: QPushButton | None = None
        self._features_axis_apply_all_button: QPushButton | None = None
        self._features_import_button: QPushButton | None = None
        self._features_export_button: QPushButton | None = None
        self._features_subset_band_combo: QComboBox | None = None
        self._features_subset_channel_combo: QComboBox | None = None
        self._features_subset_region_combo: QComboBox | None = None
        self._features_plot_button: QPushButton | None = None
        self._features_plot_advance_button: QPushButton | None = None
        self._features_plot_export_button: QPushButton | None = None
        self._features_x_label_edit: QLineEdit | None = None
        self._features_y_label_edit: QLineEdit | None = None
        self._features_cbar_label_edit: QLineEdit | None = None
        self._features_paradigms: list[dict[str, Any]] = []
        self._features_files: list[dict[str, Any]] = []
        self._features_filtered_files: list[dict[str, Any]] = []
        self._features_axes_by_metric: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._features_trial_params_by_slug: dict[str, dict[str, Any]] = {}
        self._features_plot_advance_params: dict[str, Any] = (
            self._load_features_plot_advance_defaults()
        )
        self._features_last_plot_figure: Any | None = None
        self._features_last_plot_data: pd.DataFrame | None = None
        self._features_last_plot_name: str = ""
        self._space_value_edit: QLineEdit | None = None
        self._localize_atlas_button: QPushButton | None = None
        self._localize_atlas_summary_label: QLabel | None = None
        self._localize_apply_button: QPushButton | None = None
        self._localize_import_button: QPushButton | None = None
        self._localize_export_button: QPushButton | None = None
        self._localize_match_button: QPushButton | None = None
        self._localize_match_status_label: QLabel | None = None
        self._localize_matlab_status_label: QLabel | None = None
        self._localize_elmodel_edit: QLineEdit | None = None
        self._contact_viewer_button: QPushButton | None = None
        self._localize_indicator: QLabel | None = None
        self._localize_inferred_space: str | None = None
        self._localize_space_error: str = ""
        self._localize_reconstruction_exists = False
        self._localize_available_atlases: tuple[str, ...] = ()
        self._localize_region_names_by_atlas: dict[str, tuple[str, ...]] = {}
        self._localize_selected_atlas: str | None = None
        self._localize_selected_regions: tuple[str, ...] = ()
        self._localize_match_completed = False
        self._localize_match_payload: dict[str, Any] | None = None
        self._localize_reconstruction_summary: dict[str, Any] | None = None
        self._localize_matlab_failures_shown: set[str] = set()
        self._localize_matlab_timer = QTimer(self)
        self._localize_matlab_timer.setInterval(250)
        self._localize_matlab_timer.timeout.connect(self._poll_localize_matlab_status)
        self._left_column_widget: QWidget | None = None
        self._preproc_filter_advance_params = self._load_filter_advance_defaults()
        self._preproc_viz_psd_params = self._load_preproc_viz_psd_defaults()
        self._preproc_viz_tfr_params = self._load_preproc_viz_tfr_defaults()
        self._preproc_viz_last_step: str | None = None
        self._plot_close_hooks: list[tuple[Any, Any]] = []
        self._active_mne_browsers: dict[int, dict[str, Any]] = {}
        self._mne_browser_shutdown_pending = False
        self._mne_browser_shutdown_prev_quit_on_last_window_closed: bool | None = None
        self._mne_browser_shutdown_excluded_tokens: set[int] = set()
        self._finalizing_mainwindow_close = False
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(BUSY_INTERVAL_MS)
        self._busy_timer.timeout.connect(self._on_busy_tick)
        self._busy_label: str | None = None
        self._busy_suffix: str | None = None
        self._busy_frame_idx = 0
        self._busy_locked_buttons: list[QAbstractButton] = []
        self._busy_locked_actions: list[QAction] = []
        self._tensor_run_state: dict[str, Any] | None = None
        self._tensor_run_locked_buttons: list[QAbstractButton] = []
        self._tensor_run_locked_actions: list[QAction] = []
        self._tensor_run_poll_timer = QTimer(self)
        self._tensor_run_poll_timer.setInterval(150)
        self._tensor_run_poll_timer.timeout.connect(self._poll_tensor_run_process)
        self._record_param_dirty_keys: set[str] = set()
        self._record_param_syncing = False

        self._build_ui()
        self._bind_record_param_dirty_signals()
        self._localize_matlab_timer.start()
        self._refresh_localize_matlab_status()
        self._schedule_matlab_warmup()
        if auto_load_dataset:
            self._initialize_dataset_context()
        else:
            self._set_empty_record_context()
        self.route_to_stage(self._active_stage_key)

    def route_to_stage(self, stage_key: str) -> None:
        _shell_routing.route_to_stage(self, stage_key, stage_specs=STAGE_SPECS)

    def _build_ui(self) -> None:
        _main_window_layout.build_ui(
            self,
            root_margin=ROOT_MARGIN,
            root_spacing=ROOT_SPACING,
            stage_content_min_width=STAGE_CONTENT_MIN_WIDTH,
        )

    def _build_menu_bar(self) -> None:
        _main_window_layout.build_menu_bar(self)

    def _compute_left_column_width(self, window_width: int) -> int:
        return _main_window_layout.compute_left_column_width(
            window_width,
            left_width_ratio=LEFT_WIDTH_RATIO,
            left_width_min=LEFT_WIDTH_MIN,
            left_width_max=LEFT_WIDTH_MAX,
        )

    def _update_left_column_width(self) -> None:
        _main_window_layout.update_left_column_width(
            self,
            left_width_ratio=LEFT_WIDTH_RATIO,
            left_width_min=LEFT_WIDTH_MIN,
            left_width_max=LEFT_WIDTH_MAX,
        )

    def _apply_panel_title_style(self) -> None:
        _main_window_layout.apply_panel_title_style(self, button_cls=QPushButton)

    def _enforce_button_text_fit(self) -> None:
        _main_window_layout.enforce_button_text_fit(
            self,
            button_text_horizontal_padding=BUTTON_TEXT_HORIZONTAL_PADDING,
        )

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._update_left_column_width()

    def _on_busy_tick(self) -> None:
        _shell_busy_state.on_busy_tick(self, busy_frames=BUSY_FRAMES)

    def _render_busy_message(self) -> None:
        _shell_busy_state.render_busy_message(self, busy_frames=BUSY_FRAMES)

    def _start_busy(self, label: str, *, suffix: str | None = None) -> None:
        _shell_busy_state.start_busy(
            self,
            label=label,
            busy_frames=BUSY_FRAMES,
            suffix=suffix,
        )

    def _stop_busy(self) -> None:
        _shell_busy_state.stop_busy(self)

    def _set_busy_ui_lock(self, lock: bool) -> None:
        _shell_busy_state.set_busy_ui_lock(self, lock=lock)

    def _run_with_busy(
        self, label: str, work: Callable[[], T], *, suffix: str | None = None
    ) -> T:
        return _shell_busy_state.run_with_busy(
            self,
            label=label,
            work=work,
            busy_frames=BUSY_FRAMES,
            suffix=suffix,
        )

    def closeEvent(self, event: Any) -> None:
        if self._defer_close_for_active_mne_browsers(event):
            return
        try:
            self._shutdown_tensor_run()
        finally:
            try:
                _window_shutdown.close_auxiliary_windows(self)
            finally:
                try:
                    self._persist_record_params_snapshot_on_close()
                finally:
                    try:
                        shutdown_matlab_runtime(timeout_s=5.0)
                    finally:
                        super().closeEvent(event)

    @staticmethod
    def _normalize_feature_axis_rows(
        value: Any,
        *,
        min_start: float,
        max_end: float | None = None,
        allow_duplicate_names: bool = False,
    ) -> list[dict[str, float | str]]:
        return _main_window_layout.normalize_feature_axis_rows(
            value,
            min_start=min_start,
            max_end=max_end,
            allow_duplicate_names=allow_duplicate_names,
        )

    def _page_title(self, text: str) -> QLabel:
        return _shell_routing.page_title(text)

    def _make_indicator_label(self, state: str) -> QLabel:
        return _shell_routing.make_indicator_label(
            state,
            indicator_colors=INDICATOR_COLORS,
        )

    def _placeholder_block(self, title: str) -> QGroupBox:
        return _shell_routing.placeholder_block(title)

    def _refresh_stage_controls(self) -> None:
        _shell_routing.refresh_stage_controls(
            self,
            stage_specs=STAGE_SPECS,
            indicator_colors=INDICATOR_COLORS,
        )

    @staticmethod
    def _set_indicator_color(indicator: QLabel, state: str) -> None:
        _shell_routing.set_indicator_color(
            indicator,
            state,
            indicator_colors=INDICATOR_COLORS,
        )
