"""Alignment page and widget builders extracted from MainWindow."""

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
ALIGNMENT_METHODS: tuple[object, ...] = ()


def _page_margin(owner) -> int:
    getter = getattr(owner, "_stage_panel_page_margin", None)
    return PAGE_MARGIN if getter is None else int(getter())


def _page_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_page_spacing", None)
    return PAGE_SPACING if getter is None else int(getter())


def _grid_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_grid_spacing", None)
    return GRID_SPACING if getter is None else int(getter())


def _alignment_methods(owner) -> tuple[object, ...]:
    getter = getattr(owner, "_stage_alignment_methods", None)
    return ALIGNMENT_METHODS if getter is None else tuple(getter())


def _build_alignment_page(self) -> QWidget:
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
    left_layout.addWidget(self._page_title("Align Epochs"))
    left_layout.addWidget(self._build_alignment_paradigm_block())
    left_layout.addWidget(self._build_alignment_method_block(), stretch=1)

    right = QWidget()
    right.setMinimumWidth(0)
    right.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(page_spacing)
    right_layout.addWidget(self._build_alignment_epoch_inspector_block(), stretch=1)

    layout.addWidget(left, stretch=1)
    layout.addWidget(right, stretch=1)
    layout.setStretch(0, 1)
    layout.setStretch(1, 1)
    return page


def _build_alignment_paradigm_block(self) -> QGroupBox:
    return build_trials_panel(
        self,
        block_attr="_alignment_trials_block",
        list_attr="_alignment_paradigm_list",
        add_button_attr="_alignment_paradigm_add_button",
        delete_button_attr="_alignment_paradigm_delete_button",
        action_row_attr="_alignment_trials_action_row",
        on_selected=self._on_alignment_paradigm_selected,
        add_enabled=True,
        delete_enabled=True,
        on_add=self._on_alignment_paradigm_add,
        on_delete=self._on_alignment_paradigm_delete,
        list_tooltip="Select the trial/paradigm used for alignment.",
        add_tooltip="Create a new alignment trial.",
        delete_tooltip="Delete the selected alignment trial.",
    )


def _build_alignment_method_block(self) -> QGroupBox:
    block = IndicatorGroupBox("Method + Params")
    layout = QFormLayout(block)
    layout.setLabelAlignment(Qt.AlignLeft)
    layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    block.setContentsMargins(8, 8, 8, 8)
    self._alignment_method_indicator = block.indicator_label()
    self._set_indicator_color(self._alignment_method_indicator, "gray")
    self._alignment_method_indicator.setToolTip(
        "Alignment method state: gray=not run, yellow=stale or failed, "
        "green=current params have successful alignment outputs."
    )

    self._alignment_method_combo = QComboBox()
    for item in _alignment_methods(self):
        self._alignment_method_combo.addItem(item.display_name, item.key)
        row = self._alignment_method_combo.count() - 1
        self._alignment_method_combo.setItemData(
            row,
            self._alignment_method_description_text(item.key),
            Qt.ToolTipRole,
        )
    self._alignment_method_combo.currentIndexChanged.connect(
        self._on_alignment_method_changed
    )
    layout.addRow("Method", self._alignment_method_combo)

    run_row = QWidget()
    run_layout = QHBoxLayout(run_row)
    run_layout.setContentsMargins(0, 0, 0, 0)
    run_layout.setSpacing(_grid_spacing(self))
    self._alignment_method_params_button = QPushButton("Params")
    self._alignment_method_params_button.setToolTip("Open method parameter settings.")
    self._alignment_method_params_button.clicked.connect(
        self._on_alignment_method_params
    )
    self._alignment_run_button = QPushButton("Align Epochs")
    self._alignment_run_button.setToolTip(
        "Run alignment and save warped tensors for the selected trial."
    )
    self._alignment_run_button.clicked.connect(self._on_alignment_run)
    run_layout.addWidget(self._alignment_method_params_button)
    run_layout.addWidget(self._alignment_run_button)
    run_layout.addStretch(1)
    layout.addRow(run_row)

    self._alignment_method_description_label = QLabel("")
    self._alignment_method_description_label.setWordWrap(True)
    self._alignment_method_description_label.setStyleSheet("color: #555555;")
    layout.addRow("", self._alignment_method_description_label)

    config_row = QWidget()
    config_layout = QHBoxLayout(config_row)
    config_layout.setContentsMargins(0, 0, 0, 0)
    config_layout.setSpacing(_grid_spacing(self))
    self._alignment_import_button = QPushButton("Import Configs...")
    self._alignment_import_button.setToolTip(
        "Load alignment config for the selected trial."
    )
    self._alignment_import_button.clicked.connect(self._on_alignment_import_config)
    self._alignment_export_button = QPushButton("Export Configs...")
    self._alignment_export_button.setToolTip(
        "Save alignment config for the selected trial."
    )
    self._alignment_export_button.clicked.connect(self._on_alignment_export_config)
    config_layout.addWidget(self._alignment_import_button, 1)
    config_layout.addWidget(self._alignment_export_button, 1)
    layout.addRow(config_row)

    self._update_alignment_method_description()
    return block


def _build_alignment_epoch_inspector_block(self) -> QGroupBox:
    block = IndicatorGroupBox("Epoch Inspector")
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    grid_spacing = _grid_spacing(self)
    layout.setSpacing(grid_spacing)
    self._alignment_epoch_inspector_indicator = block.indicator_label()
    self._set_indicator_color(self._alignment_epoch_inspector_indicator, "gray")
    self._alignment_epoch_inspector_indicator.setToolTip(
        "Epoch inspector state: gray=not ready, yellow=stale or failed, "
        "green=current picked epochs have preview-ready outputs."
    )

    metric_row = QWidget()
    metric_layout = QHBoxLayout(metric_row)
    metric_layout.setContentsMargins(0, 0, 0, 0)
    metric_layout.setSpacing(grid_spacing)
    metric_layout.addWidget(QLabel("Metric"))
    self._alignment_epoch_metric_combo = QComboBox()
    self._alignment_epoch_metric_combo.currentIndexChanged.connect(
        self._on_alignment_epoch_metric_changed
    )
    self._alignment_epoch_metric_combo.setToolTip(
        "Select the metric shown in Epoch Inspector."
    )
    metric_layout.addWidget(self._alignment_epoch_metric_combo, stretch=1)
    metric_layout.addWidget(QLabel("Channel"))
    self._alignment_epoch_channel_combo = QComboBox()
    self._alignment_epoch_channel_combo.setToolTip(
        "Select channel used for preview averaging."
    )
    metric_layout.addWidget(self._alignment_epoch_channel_combo, stretch=1)
    layout.addWidget(metric_row)

    self._alignment_epoch_table = QTableWidget(0, 5)
    self._alignment_epoch_table.setHorizontalHeaderLabels(
        ["Pick", "Epoch", "Duration (s)", "Start (s)", "End (s)"]
    )
    self._alignment_epoch_table.verticalHeader().setVisible(False)
    self._alignment_epoch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    header = self._alignment_epoch_table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.Stretch)
    header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
    self._alignment_epoch_table.setMinimumHeight(220)
    self._alignment_epoch_table.setToolTip(
        "Epochs detected in the last run. Pick controls Preview and Finish outputs."
    )
    self._alignment_epoch_table.itemChanged.connect(
        self._on_alignment_epoch_item_changed
    )
    layout.addWidget(self._alignment_epoch_table, stretch=1)

    action_block = QWidget()
    action_block_layout = QVBoxLayout(action_block)
    action_block_layout.setContentsMargins(0, 0, 0, 0)
    action_block_layout.setSpacing(grid_spacing)

    action_row = QWidget()
    action_layout = QHBoxLayout(action_row)
    action_layout.setContentsMargins(0, 0, 0, 0)
    action_layout.setSpacing(grid_spacing)
    self._alignment_select_all_button = QPushButton("Select All")
    self._alignment_select_all_button.setToolTip("Toggle all epoch picks on or off.")
    self._alignment_preview_button = QPushButton("Preview")
    self._alignment_preview_button.setToolTip("Show average map from picked epochs.")
    self._alignment_finish_button = QPushButton("Finish")
    self._alignment_finish_button.setToolTip(
        "Build raw tables from picked epochs only."
    )
    self._alignment_select_all_button.clicked.connect(self._on_alignment_select_all)
    self._alignment_preview_button.clicked.connect(self._on_alignment_preview)
    self._alignment_finish_button.clicked.connect(self._on_alignment_finish)
    action_layout.addWidget(self._alignment_select_all_button)
    action_layout.addWidget(self._alignment_preview_button)
    action_layout.addWidget(self._alignment_finish_button)
    action_layout.addStretch(1)
    action_block_layout.addWidget(action_row)

    self._alignment_merge_location_status_label = QLabel(
        "Merge Location Info: Not Ready"
    )
    self._alignment_merge_location_status_label.setToolTip(
        "Representative-coordinate merge readiness follows the current record Localize state."
    )
    action_block_layout.addWidget(self._alignment_merge_location_status_label)

    layout.addWidget(action_block)
    return block
