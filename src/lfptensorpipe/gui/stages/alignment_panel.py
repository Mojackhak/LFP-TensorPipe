"""Alignment page and widget builders extracted from MainWindow."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QMenu,
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
    left_layout.addWidget(self._build_alignment_paradigm_block(), stretch=1)
    left_layout.addWidget(self._build_alignment_method_block())
    left_layout.addWidget(self._build_alignment_epoch_inspector_block())

    right = QWidget()
    right.setMinimumWidth(0)
    right.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(page_spacing)
    right_layout.addWidget(_epoch_list(self), stretch=1)

    left.setMinimumWidth(258)
    layout.addWidget(left, stretch=4)
    layout.addWidget(right, stretch=6)
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
    layout = QGridLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(_grid_spacing(self))
    self._alignment_method_indicator = block.indicator_label()
    self._set_indicator_color(self._alignment_method_indicator, "gray")
    self._alignment_method_combo = QComboBox()
    self._alignment_method_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    for item in _alignment_methods(self):
        self._alignment_method_combo.addItem(item.display_name, item.key)
        self._alignment_method_combo.setItemData(
            self._alignment_method_combo.count() - 1,
            self._alignment_method_description_text(item.key),
            Qt.ToolTipRole,
        )
    self._alignment_method_combo.currentIndexChanged.connect(
        self._on_alignment_method_changed
    )
    self._alignment_method_params_button = QPushButton("Params")
    self._alignment_method_params_button.clicked.connect(
        self._on_alignment_method_params
    )
    self._alignment_run_button = QPushButton("Align Epochs")
    self._alignment_run_button.clicked.connect(self._on_alignment_run)
    configs = QPushButton("Configs")
    menu = QMenu(configs)
    self._alignment_import_button = menu.addAction(
        "Import Configs…", self._on_alignment_import_config
    )
    self._alignment_export_button = menu.addAction(
        "Export Configs…", self._on_alignment_export_config
    )
    configs.setMenu(menu)
    layout.addWidget(self._alignment_method_combo, 0, 0)
    layout.addWidget(self._alignment_method_params_button, 0, 1)
    layout.addWidget(configs, 1, 0)
    layout.addWidget(self._alignment_run_button, 1, 1)
    layout.setColumnStretch(0, 2)
    layout.setColumnStretch(1, 1)
    self._update_alignment_method_description()
    return block


def _build_alignment_epoch_inspector_block(self) -> QGroupBox:
    block = IndicatorGroupBox("Epoch Inspector")
    layout = QGridLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(_grid_spacing(self))
    self._alignment_epoch_inspector_indicator = block.indicator_label()
    self._set_indicator_color(self._alignment_epoch_inspector_indicator, "gray")
    buttons = (
        ("_qc_run", "Run QC", self._run_epoch_qc),
        ("_qc_advance", "Advance", self._edit_epoch_qc),
        ("_qc_import", "Import Configs…", self._import_epoch_qc),
        ("_qc_export", "Export Configs…", self._export_epoch_qc),
        ("_qc_view", "View flagged", self._view_flagged),
        ("_qc_exclude", "Exclude flagged", self._exclude_flagged),
        (
            "_alignment_select_all_button",
            "Select all Include",
            self._on_alignment_select_all,
        ),
        ("_view_all", "Select all View", self._toggle_epoch_view),
        ("_alignment_preview_button", "Preview", self._on_alignment_preview),
        ("_alignment_finish_button", "Finish", self._on_alignment_finish),
    )
    for i, (attr, title, handler) in enumerate(buttons):
        button = QPushButton(title)
        setattr(self, attr, button)
        button.clicked.connect(handler)
        layout.addWidget(button, i // 2, i % 2)
    return block


def _epoch_list(self):
    block = QGroupBox("Epoch list")
    layout = QVBoxLayout(block)
    layout.setContentsMargins(6, 8, 6, 6)
    table = self._alignment_epoch_table = QTableWidget(0, 5)
    table.setHorizontalHeaderLabels(["Include", "Epoch", "Duration", "View", "Notes"])
    table.horizontalHeaderItem(2).setToolTip("Duration in seconds")
    table.verticalHeader().hide()
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.NoSelection)
    table.setTextElideMode(Qt.ElideRight)
    header = table.horizontalHeader()
    header.setMinimumSectionSize(35)
    for col in (0, 2, 3):
        header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
    for col in (1, 4):
        header.setSectionResizeMode(col, QHeaderView.Stretch)
    table.itemChanged.connect(self._on_alignment_epoch_item_changed)
    table.cellDoubleClicked.connect(self._open_epoch_cell)
    layout.addWidget(table, 1)
    self._alignment_merge_location_status_label = QLabel(
        "Merge Location Info: Not Ready"
    )
    layout.addWidget(self._alignment_merge_location_status_label)
    return block
