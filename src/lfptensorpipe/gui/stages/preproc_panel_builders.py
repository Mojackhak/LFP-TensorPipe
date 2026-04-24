"""Preprocess stage section builders."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from lfptensorpipe.gui.stages.indicator_group_box import IndicatorGroupBox


def build_preprocess_page(
    self,
    *,
    page_margin: int,
    page_spacing: int,
) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(page_margin, page_margin, page_margin, page_margin)
    layout.setSpacing(page_spacing)

    layout.addWidget(self._page_title("Preprocess Signal"))

    columns = QWidget()
    columns_layout = QHBoxLayout(columns)
    columns_layout.setContentsMargins(0, 0, 0, 0)
    columns_layout.setSpacing(page_spacing)

    left_col = QVBoxLayout()
    left_col.setSpacing(page_spacing)
    left_col.addWidget(self._build_preproc_raw_block(), stretch=1)
    left_col.addWidget(self._build_preproc_filter_block(), stretch=1)
    left_col.addWidget(self._build_preproc_bad_segment_block(), stretch=1)
    left_col.addWidget(self._build_preproc_finish_block(), stretch=1)

    right_col = QVBoxLayout()
    right_col.setSpacing(page_spacing)
    right_col.addWidget(self._build_preproc_annotations_block(), stretch=3)
    right_col.addWidget(self._build_preproc_ecg_block(), stretch=1)

    left_widget = QWidget()
    left_widget.setLayout(left_col)
    right_widget = QWidget()
    right_widget.setLayout(right_col)

    columns_layout.addWidget(left_widget, stretch=1)
    columns_layout.addWidget(right_widget, stretch=1)
    layout.addWidget(columns, stretch=1)

    layout.addWidget(self._build_preproc_visualization_block())
    return page


def build_preproc_raw_block(self, *, grid_spacing: int) -> QGroupBox:
    block = IndicatorGroupBox("0. Raw")
    self._register_preproc_indicator("raw", indicator=block.indicator_label())
    layout = QHBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(grid_spacing)
    self._preproc_raw_plot_button = QPushButton("Plot")
    self._preproc_raw_plot_button.clicked.connect(self._on_preproc_raw_plot)
    self._preproc_raw_plot_button.setEnabled(False)
    self._preproc_raw_plot_button.setToolTip("Plot raw input for the current record.")
    layout.addWidget(self._preproc_raw_plot_button)
    layout.addStretch(1)
    return block


def build_preproc_filter_block(self, *, grid_spacing: int) -> QGroupBox:
    block = IndicatorGroupBox("1. Filter")
    self._register_preproc_indicator("filter", indicator=block.indicator_label())
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(grid_spacing)

    notches_row = QWidget()
    notches_layout = QHBoxLayout(notches_row)
    notches_layout.setContentsMargins(0, 0, 0, 0)
    notches_layout.setSpacing(grid_spacing)
    notches_layout.addWidget(QLabel("Notches"))
    self._preproc_filter_notches_edit = QLineEdit("50,100")
    self._preproc_filter_notches_edit.setPlaceholderText("e.g. 50,100")
    self._preproc_filter_notches_edit.setToolTip("Comma-separated notch centers in Hz.")
    self._preproc_filter_notches_edit.setMinimumWidth(150)
    self._preproc_filter_notches_edit.setSizePolicy(
        QSizePolicy.Expanding, QSizePolicy.Fixed
    )
    notches_layout.addWidget(self._preproc_filter_notches_edit, stretch=1)
    layout.addWidget(notches_row)

    freq_row = QWidget()
    freq_layout = QHBoxLayout(freq_row)
    freq_layout.setContentsMargins(0, 0, 0, 0)
    freq_layout.setSpacing(grid_spacing)
    freq_layout.addWidget(QLabel("Low freq"))
    self._preproc_filter_low_freq_edit = QLineEdit("1")
    self._preproc_filter_low_freq_edit.setPlaceholderText("Hz")
    self._preproc_filter_low_freq_edit.setToolTip("High-pass cutoff in Hz.")
    self._preproc_filter_low_freq_edit.setMinimumWidth(45)
    freq_layout.addWidget(self._preproc_filter_low_freq_edit, stretch=1)

    freq_layout.addWidget(QLabel("High freq"))
    self._preproc_filter_high_freq_edit = QLineEdit("200")
    self._preproc_filter_high_freq_edit.setPlaceholderText("Hz")
    self._preproc_filter_high_freq_edit.setToolTip("Low-pass cutoff in Hz.")
    self._preproc_filter_high_freq_edit.setMinimumWidth(45)
    freq_layout.addWidget(self._preproc_filter_high_freq_edit, stretch=1)
    freq_layout.addStretch(1)
    layout.addWidget(freq_row)
    self._apply_filter_basic_params_to_fields(self._load_filter_basic_defaults())

    action_row = QWidget()
    action_layout = QHBoxLayout(action_row)
    action_layout.setContentsMargins(0, 0, 0, 0)
    action_layout.setSpacing(grid_spacing)
    self._preproc_filter_advance_button = QPushButton("Advance")
    self._preproc_filter_apply_button = QPushButton("Apply")
    self._preproc_filter_plot_button = QPushButton("Plot")
    self._preproc_filter_advance_button.setToolTip("Open advanced filter parameters.")
    self._preproc_filter_apply_button.setToolTip("Run filter with current parameters.")
    self._preproc_filter_plot_button.setToolTip("Plot filtered output.")
    self._preproc_filter_advance_button.clicked.connect(self._on_preproc_filter_advance)
    self._preproc_filter_apply_button.clicked.connect(self._on_preproc_filter_apply)
    self._preproc_filter_plot_button.clicked.connect(self._on_preproc_filter_plot)
    self._preproc_filter_advance_button.setEnabled(False)
    self._preproc_filter_apply_button.setEnabled(False)
    self._preproc_filter_plot_button.setEnabled(False)
    action_layout.addWidget(self._preproc_filter_advance_button)
    action_layout.addWidget(self._preproc_filter_apply_button)
    action_layout.addWidget(self._preproc_filter_plot_button)
    action_layout.addStretch(1)
    layout.addWidget(action_row)
    return block


def build_preproc_finish_block(self, *, grid_spacing: int) -> QGroupBox:
    block = IndicatorGroupBox("5. Finish")
    self._register_preproc_indicator("finish", indicator=block.indicator_label())
    layout = QHBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(grid_spacing)
    self._preproc_finish_apply_button = QPushButton("Apply")
    self._preproc_finish_plot_button = QPushButton("Plot")
    self._preproc_finish_apply_button.setToolTip(
        "Run the finish step and write the final preprocess output."
    )
    self._preproc_finish_plot_button.setToolTip("Plot finish-step output.")
    self._preproc_finish_apply_button.clicked.connect(self._on_preproc_finish_apply)
    self._preproc_finish_plot_button.clicked.connect(self._on_preproc_finish_plot)
    self._preproc_finish_apply_button.setEnabled(False)
    self._preproc_finish_plot_button.setEnabled(False)
    layout.addWidget(self._preproc_finish_apply_button)
    layout.addWidget(self._preproc_finish_plot_button)
    layout.addStretch(1)
    return block


def build_preproc_bad_segment_block(self, *, grid_spacing: int) -> QGroupBox:
    block = IndicatorGroupBox("3. Bad Segment Removal")
    self._register_preproc_indicator(
        "bad_segment_removal", indicator=block.indicator_label()
    )
    layout = QHBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(grid_spacing)
    self._preproc_bad_segment_apply_button = QPushButton("Apply")
    self._preproc_bad_segment_plot_button = QPushButton("Plot")
    self._preproc_bad_segment_apply_button.setToolTip("Run bad-segment removal.")
    self._preproc_bad_segment_plot_button.setToolTip("Plot bad-segment-removal output.")
    self._preproc_bad_segment_apply_button.clicked.connect(
        self._on_preproc_bad_segment_apply
    )
    self._preproc_bad_segment_plot_button.clicked.connect(
        self._on_preproc_bad_segment_plot
    )
    self._preproc_bad_segment_apply_button.setEnabled(False)
    self._preproc_bad_segment_plot_button.setEnabled(False)
    layout.addWidget(self._preproc_bad_segment_apply_button)
    layout.addWidget(self._preproc_bad_segment_plot_button)
    layout.addStretch(1)
    return block


def build_preproc_ecg_block(
    self,
    *,
    grid_spacing: int,
    ecg_methods: tuple[str, ...],
) -> QGroupBox:
    block = IndicatorGroupBox("4. ECG Artifact Removal")
    self._register_preproc_indicator(
        "ecg_artifact_removal", indicator=block.indicator_label()
    )
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(grid_spacing)

    method_row = QWidget()
    method_row_layout = QHBoxLayout(method_row)
    method_row_layout.setContentsMargins(0, 0, 0, 0)
    method_row_layout.setSpacing(grid_spacing)
    method_row_layout.addWidget(QLabel("Method"))
    self._preproc_ecg_method_combo = QComboBox()
    for method in ecg_methods:
        self._preproc_ecg_method_combo.addItem(method, method)
    self._preproc_ecg_method_combo.setToolTip("Method used for ECG artifact removal.")
    default_index = self._preproc_ecg_method_combo.findData("svd")
    if default_index >= 0:
        self._preproc_ecg_method_combo.setCurrentIndex(default_index)
    method_row_layout.addWidget(self._preproc_ecg_method_combo, stretch=1)
    layout.addWidget(method_row)

    channels_row = QWidget()
    channels_row_layout = QHBoxLayout(channels_row)
    channels_row_layout.setContentsMargins(0, 0, 0, 0)
    channels_row_layout.setSpacing(grid_spacing)
    channels_row_layout.addWidget(QLabel("Channels"))
    self._preproc_ecg_channels_button = QPushButton("Select Channels")
    self._preproc_ecg_channels_button.setToolTip(
        "Choose channels used for ECG artifact removal."
    )
    self._preproc_ecg_channels_button.clicked.connect(
        self._on_preproc_ecg_channels_select
    )
    channels_row_layout.addWidget(self._preproc_ecg_channels_button, stretch=1)
    layout.addWidget(channels_row)

    action_row = QWidget()
    action_row_layout = QHBoxLayout(action_row)
    action_row_layout.setContentsMargins(0, 0, 0, 0)
    action_row_layout.setSpacing(grid_spacing)
    self._preproc_ecg_apply_button = QPushButton("Apply")
    self._preproc_ecg_plot_button = QPushButton("Plot")
    self._preproc_ecg_apply_button.setToolTip("Run ECG artifact removal.")
    self._preproc_ecg_plot_button.setToolTip("Plot ECG-cleaned output.")
    self._preproc_ecg_apply_button.clicked.connect(self._on_preproc_ecg_apply)
    self._preproc_ecg_plot_button.clicked.connect(self._on_preproc_ecg_plot)
    self._preproc_ecg_apply_button.setEnabled(False)
    self._preproc_ecg_plot_button.setEnabled(False)
    action_row_layout.addWidget(self._preproc_ecg_apply_button)
    action_row_layout.addWidget(self._preproc_ecg_plot_button)
    action_row_layout.addStretch(1)
    layout.addWidget(action_row)

    return block


def build_preproc_visualization_block(self, *, grid_spacing: int) -> QGroupBox:
    block = QGroupBox("Visualization")
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(grid_spacing)

    step_row = QWidget()
    step_row_layout = QHBoxLayout(step_row)
    step_row_layout.setContentsMargins(0, 0, 0, 0)
    step_row_layout.setSpacing(grid_spacing)
    step_row_layout.addWidget(QLabel("Step"))
    self._preproc_viz_step_combo = QComboBox()
    self._preproc_viz_step_combo.currentIndexChanged.connect(
        self._on_preproc_viz_step_changed
    )
    self._preproc_viz_step_combo.setToolTip(
        "Choose which preprocess output to visualize."
    )
    step_row_layout.addWidget(self._preproc_viz_step_combo, stretch=1)
    layout.addWidget(step_row)

    psd_row = QWidget()
    psd_row_layout = QHBoxLayout(psd_row)
    psd_row_layout.setContentsMargins(0, 0, 0, 0)
    psd_row_layout.setSpacing(grid_spacing)
    psd_row_layout.addWidget(QLabel("PSD"))
    self._preproc_viz_psd_advance_button = QPushButton("Advance")
    self._preproc_viz_psd_plot_button = QPushButton("Plot")
    self._preproc_viz_psd_advance_button.setToolTip("Open PSD plotting parameters.")
    self._preproc_viz_psd_plot_button.setToolTip(
        "Plot PSD for the selected step and channels."
    )
    self._preproc_viz_psd_advance_button.clicked.connect(
        self._on_preproc_viz_psd_advance
    )
    self._preproc_viz_psd_plot_button.clicked.connect(self._on_preproc_viz_psd_plot)
    psd_row_layout.addWidget(self._preproc_viz_psd_advance_button)
    psd_row_layout.addWidget(self._preproc_viz_psd_plot_button)
    psd_row_layout.addStretch(1)
    layout.addWidget(psd_row)

    tfr_row = QWidget()
    tfr_row_layout = QHBoxLayout(tfr_row)
    tfr_row_layout.setContentsMargins(0, 0, 0, 0)
    tfr_row_layout.setSpacing(grid_spacing)
    tfr_row_layout.addWidget(QLabel("TFR"))
    self._preproc_viz_tfr_advance_button = QPushButton("Advance")
    self._preproc_viz_tfr_plot_button = QPushButton("Plot")
    self._preproc_viz_tfr_advance_button.setToolTip("Open TFR plotting parameters.")
    self._preproc_viz_tfr_plot_button.setToolTip(
        "Plot TFR for the selected step and channels."
    )
    self._preproc_viz_tfr_advance_button.clicked.connect(
        self._on_preproc_viz_tfr_advance
    )
    self._preproc_viz_tfr_plot_button.clicked.connect(self._on_preproc_viz_tfr_plot)
    tfr_row_layout.addWidget(self._preproc_viz_tfr_advance_button)
    tfr_row_layout.addWidget(self._preproc_viz_tfr_plot_button)
    tfr_row_layout.addStretch(1)
    layout.addWidget(tfr_row)

    channels_row = QWidget()
    channels_row_layout = QHBoxLayout(channels_row)
    channels_row_layout.setContentsMargins(0, 0, 0, 0)
    channels_row_layout.setSpacing(grid_spacing)
    channels_row_layout.addWidget(QLabel("Channels"))
    self._preproc_viz_channels_button = QPushButton("Select Channels")
    self._preproc_viz_channels_button.setToolTip(
        "Choose channels used for PSD/TFR plotting."
    )
    self._preproc_viz_channels_button.clicked.connect(
        self._on_preproc_viz_channels_select
    )
    channels_row_layout.addWidget(self._preproc_viz_channels_button, stretch=1)
    layout.addWidget(channels_row)

    return block


def build_preproc_status_row(self, step: str, *, grid_spacing: int) -> QWidget:
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(grid_spacing)
    row_layout.addWidget(QLabel("Status"))
    row_layout.addWidget(self._register_preproc_indicator(step))
    row_layout.addStretch(1)
    return row
