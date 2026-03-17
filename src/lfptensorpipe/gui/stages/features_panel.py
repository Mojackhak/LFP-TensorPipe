"""Features page and widget builders extracted from MainWindow."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from lfptensorpipe.gui.stages.indicator_group_box import IndicatorGroupBox
from lfptensorpipe.gui.stages.shared_trials_panel import build_trials_panel

PAGE_MARGIN = 0
PAGE_SPACING = 0
GRID_SPACING = 0


def _page_margin(owner) -> int:
    getter = getattr(owner, "_stage_panel_page_margin", None)
    return PAGE_MARGIN if getter is None else int(getter())


def _page_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_page_spacing", None)
    return PAGE_SPACING if getter is None else int(getter())


def _grid_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_grid_spacing", None)
    return GRID_SPACING if getter is None else int(getter())


def _build_features_page(self) -> QWidget:
    page = QWidget()
    layout = QHBoxLayout(page)
    page_margin = _page_margin(self)
    page_spacing = _page_spacing(self)
    layout.setContentsMargins(page_margin, page_margin, page_margin, page_margin)
    layout.setSpacing(page_spacing)

    left = QWidget()
    left.setMinimumWidth(0)
    left.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    left_layout = QVBoxLayout(left)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(page_spacing)
    left_layout.addWidget(self._page_title("Extract Features"))
    left_layout.addWidget(self._build_features_paradigm_block())
    left_layout.addWidget(self._build_features_phases_block(), stretch=1)

    right = QWidget()
    right.setMinimumWidth(0)
    right.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(page_spacing)
    right_layout.addWidget(self._build_features_available_block())
    right_layout.addWidget(self._build_features_subset_block())
    right_layout.addWidget(self._build_features_plot_block())
    right_layout.addStretch(1)

    layout.addWidget(left, stretch=1)
    layout.addWidget(right, stretch=1)
    layout.setStretch(0, 1)
    layout.setStretch(1, 1)
    return page


def _build_features_paradigm_block(self) -> QGroupBox:
    return build_trials_panel(
        self,
        block_attr="_features_trials_block",
        list_attr="_features_paradigm_list",
        add_button_attr="_features_paradigm_add_button",
        delete_button_attr="_features_paradigm_delete_button",
        action_row_attr="_features_trials_action_row",
        on_selected=self._on_features_paradigm_selected,
        add_enabled=False,
        delete_enabled=False,
        list_tooltip="Select a trial with finished alignment outputs.",
        add_tooltip="Create a new feature trial entry.",
        delete_tooltip="Delete the selected feature trial entry.",
    )


def _build_features_phases_block(self) -> QGroupBox:
    block = IndicatorGroupBox("Features")
    layout = QFormLayout(block)
    layout.setLabelAlignment(Qt.AlignLeft)
    layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    block.setContentsMargins(8, 8, 8, 8)
    self._features_extract_indicator = block.indicator_label()
    self._set_indicator_color(self._features_extract_indicator, "gray")
    self._features_extract_indicator.setToolTip(
        "Feature extraction state: gray=not run, yellow=stale or failed, green=current axes and trial have successful outputs."
    )

    self._features_axis_metric_combo = QComboBox()
    self._features_axis_metric_combo.currentIndexChanged.connect(
        self._on_features_axis_metric_changed
    )
    self._features_axis_metric_combo.setToolTip(
        "Select the metric whose bands/phases you are editing."
    )
    layout.addRow("Metric", self._features_axis_metric_combo)

    self._features_axis_bands_button = QPushButton("Bands Configure... (0)")
    self._features_axis_bands_button.setToolTip(
        "Configure named frequency bands for the selected metric."
    )
    self._features_axis_bands_button.clicked.connect(self._on_features_axis_bands)
    layout.addRow("Bands", self._features_axis_bands_button)

    self._features_axis_times_button = QPushButton("Phases Configure... (0)")
    self._features_axis_times_button.setToolTip(
        "Configure named phases for the selected metric."
    )
    self._features_axis_times_button.clicked.connect(self._on_features_axis_times)
    layout.addRow("Phases", self._features_axis_times_button)

    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(_grid_spacing(self))
    self._features_axis_apply_all_button = QPushButton("Apply to All Metrics")
    self._features_axis_apply_all_button.setToolTip(
        "Copy current band/phase settings to every metric in the selected trial."
    )
    self._features_axis_apply_all_button.clicked.connect(
        self._on_features_axis_apply_all
    )
    row_layout.addWidget(self._features_axis_apply_all_button)
    self._features_run_extract_button = QPushButton("Extract Features")
    self._features_run_extract_button.setToolTip(
        "Run feature extraction for the selected trial."
    )
    self._features_run_extract_button.clicked.connect(self._on_features_run_extract)
    row_layout.addWidget(self._features_run_extract_button)
    row_layout.addStretch(1)
    layout.addRow(row)

    config_row = QWidget()
    config_layout = QHBoxLayout(config_row)
    config_layout.setContentsMargins(0, 0, 0, 0)
    config_layout.setSpacing(_grid_spacing(self))
    self._features_import_button = QPushButton("Import Configs...")
    self._features_import_button.setToolTip("Load feature configuration.")
    self._features_import_button.clicked.connect(self._on_features_import_config)
    self._features_export_button = QPushButton("Export Configs...")
    self._features_export_button.setToolTip("Save feature configuration.")
    self._features_export_button.clicked.connect(self._on_features_export_config)
    config_layout.addWidget(self._features_import_button, 1)
    config_layout.addWidget(self._features_export_button, 1)
    layout.addRow(config_row)
    return block


def _build_features_available_block(self) -> QGroupBox:
    block = QGroupBox("Available Features")
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    grid_spacing = _grid_spacing(self)
    layout.setSpacing(grid_spacing)

    filter_row = QWidget()
    filter_layout = QHBoxLayout(filter_row)
    filter_layout.setContentsMargins(0, 0, 0, 0)
    filter_layout.setSpacing(grid_spacing)
    filter_layout.addWidget(QLabel("Search"))
    self._features_filter_feature_edit = QLineEdit()
    self._features_filter_feature_edit.textChanged.connect(
        self._refresh_features_available_files
    )
    self._features_filter_feature_edit.setToolTip("Filter available feature files.")
    filter_layout.addWidget(self._features_filter_feature_edit, stretch=1)
    self._features_refresh_button = QPushButton("Refresh Features")
    self._features_refresh_button.setToolTip("Rescan generated feature files.")
    self._features_refresh_button.clicked.connect(self._on_features_refresh_files)
    filter_layout.addWidget(self._features_refresh_button)
    layout.addWidget(filter_row)

    self._features_available_table = QTableWidget(0, 3)
    self._features_available_table.setHorizontalHeaderLabels(
        ["Metric", "Feature", "Property"]
    )
    self._features_available_table.verticalHeader().setVisible(False)
    self._features_available_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    self._features_available_table.setSelectionMode(QAbstractItemView.SingleSelection)
    self._features_available_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    self._features_available_table.setToolTip(
        "Available feature outputs for the selected trial."
    )
    self._features_available_table.itemSelectionChanged.connect(
        self._on_features_available_selection_changed
    )
    header = self._features_available_table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    header.setSectionResizeMode(1, QHeaderView.Stretch)
    header.setSectionResizeMode(2, QHeaderView.Stretch)
    self._features_available_table.setMinimumHeight(180)
    layout.addWidget(self._features_available_table, stretch=1)
    return block


def _build_features_subset_block(self) -> QGroupBox:
    block = QGroupBox("Subset Selection")
    layout = QFormLayout(block)
    layout.setLabelAlignment(Qt.AlignLeft)
    layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    block.setContentsMargins(8, 8, 8, 8)

    self._features_subset_band_combo = QComboBox()
    self._features_subset_band_combo.setEnabled(False)
    self._features_subset_band_combo.setToolTip(
        "Filter the selected feature payload by band before plotting. "
        "Available bands are constrained by the current channel and region selection."
    )
    self._features_subset_band_combo.currentIndexChanged.connect(
        self._on_features_subset_changed
    )
    layout.addRow("Band", self._features_subset_band_combo)

    self._features_subset_channel_combo = QComboBox()
    self._features_subset_channel_combo.setEnabled(False)
    self._features_subset_channel_combo.setToolTip(
        "Filter the selected feature payload by channel before plotting. "
        "Available channels are constrained by the current band and region selection."
    )
    self._features_subset_channel_combo.currentIndexChanged.connect(
        self._on_features_subset_changed
    )
    layout.addRow("Channel", self._features_subset_channel_combo)

    self._features_subset_region_combo = QComboBox()
    self._features_subset_region_combo.setEnabled(False)
    self._features_subset_region_combo.setToolTip(
        "Filter the selected feature payload by region before plotting. "
        "Available regions are constrained by the current band and channel selection."
    )
    self._features_subset_region_combo.currentIndexChanged.connect(
        self._on_features_subset_changed
    )
    layout.addRow("Region", self._features_subset_region_combo)
    return block


def _build_features_plot_block(self) -> QGroupBox:
    block = QGroupBox("Plot Settings")
    layout = QFormLayout(block)
    layout.setLabelAlignment(Qt.AlignLeft)
    layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    block.setContentsMargins(8, 8, 8, 8)

    self._features_x_label_edit = QLineEdit()
    self._features_y_label_edit = QLineEdit()
    self._features_cbar_label_edit = QLineEdit()
    self._features_x_label_edit.setToolTip(
        "Override the x-axis label. Leave blank to use the default."
    )
    self._features_y_label_edit.setToolTip(
        "Override the y-axis label. Leave blank to use the default."
    )
    self._features_cbar_label_edit.setToolTip(
        "Override the colorbar label. Leave blank to use the default."
    )
    layout.addRow("X label", self._features_x_label_edit)
    layout.addRow("Y label", self._features_y_label_edit)
    layout.addRow("Colorbar label", self._features_cbar_label_edit)

    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(_grid_spacing(self))
    self._features_plot_advance_button = QPushButton("Advance")
    self._features_plot_advance_button.setToolTip(
        "Open advanced plot settings."
    )
    self._features_plot_advance_button.clicked.connect(self._on_features_plot_advance)
    row_layout.addWidget(self._features_plot_advance_button)
    self._features_plot_button = QPushButton("Plot")
    self._features_plot_button.setToolTip("Plot the selected feature.")
    self._features_plot_button.clicked.connect(self._on_features_plot)
    row_layout.addWidget(self._features_plot_button)
    self._features_plot_export_button = QPushButton("Export")
    self._features_plot_export_button.setToolTip(
        "Export the last plotted figure and data."
    )
    self._features_plot_export_button.clicked.connect(self._on_features_plot_export)
    row_layout.addWidget(self._features_plot_export_button)
    row_layout.addStretch(1)
    layout.addRow(row)
    return block
