"""Preprocess annotation table builders and helpers."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lfptensorpipe.gui.stages.indicator_group_box import IndicatorGroupBox


def build_preproc_annotations_block(self, *, grid_spacing: int) -> QGroupBox:
    block = IndicatorGroupBox("2. Annotations")
    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(grid_spacing)
    self._register_preproc_indicator("annotations", indicator=block.indicator_label())

    self._preproc_annotations_table = QTableWidget(0, 3)
    self._preproc_annotations_table.setHorizontalHeaderLabels(
        ["Description", "Onset", "Duration"]
    )
    header = self._preproc_annotations_table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Stretch)
    self._preproc_annotations_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    self._preproc_annotations_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    self._preproc_annotations_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    self._preproc_annotations_table.setSizePolicy(
        QSizePolicy.Expanding, QSizePolicy.Expanding
    )
    self._preproc_annotations_table.setToolTip(
        "Current annotation rows for the record."
    )
    row_height = self._preproc_annotations_table.verticalHeader().defaultSectionSize()
    header_height = max(
        self._preproc_annotations_table.horizontalHeader().height(),
        self._preproc_annotations_table.horizontalHeader().sizeHint().height(),
    )
    table_min_height = header_height + (2 * row_height) + 24
    self._preproc_annotations_table.setMinimumHeight(table_min_height)
    table_panel = QFrame()
    table_panel.setObjectName("annotations_table_panel")
    table_panel.setFrameShape(QFrame.StyledPanel)
    table_panel.setAutoFillBackground(True)
    table_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    table_panel.setMinimumHeight(table_min_height + 8)
    table_panel.setStyleSheet(
        "QFrame#annotations_table_panel {"
        "background-color: #FFFFFF;"
        "border: 1px solid #C6C6C6;"
        "border-radius: 4px;"
        "}"
        "QFrame#annotations_table_panel QTableWidget {"
        "border: none;"
        "background-color: #FFFFFF;"
        "}"
    )
    table_panel_layout = QVBoxLayout(table_panel)
    table_panel_layout.setContentsMargins(4, 4, 4, 4)
    table_panel_layout.setSpacing(0)
    table_panel_layout.addWidget(self._preproc_annotations_table)
    layout.addWidget(table_panel, stretch=1)
    layout.addSpacing(6)

    actions_row = QWidget()
    actions_row.setObjectName("annotations_actions_row")
    actions_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    action_layout = QHBoxLayout(actions_row)
    action_layout.setContentsMargins(0, 0, 0, 0)
    action_layout.setSpacing(grid_spacing)
    self._preproc_annotations_edit_button = QPushButton("Configure...")
    self._preproc_annotations_save_button = QPushButton("Apply")
    self._preproc_annotations_import_button = None
    self._preproc_annotations_plot_button = QPushButton("Plot")
    self._preproc_annotations_edit_button.clicked.connect(
        self._on_preproc_annotations_edit
    )
    self._preproc_annotations_save_button.clicked.connect(
        self._on_preproc_annotations_save
    )
    self._preproc_annotations_plot_button.clicked.connect(
        self._on_preproc_annotations_plot
    )
    self._preproc_annotations_save_button.setEnabled(False)
    self._preproc_annotations_plot_button.setEnabled(False)
    self._preproc_annotations_edit_button.setToolTip("Open the annotation editor.")
    self._preproc_annotations_save_button.setToolTip(
        "Write current annotations to the pipeline."
    )
    self._preproc_annotations_plot_button.setToolTip(
        "Plot annotations over the signal."
    )
    actions_row.setFixedHeight(
        self._preproc_annotations_edit_button.sizeHint().height() + 10
    )
    action_layout.addWidget(self._preproc_annotations_edit_button)
    action_layout.addWidget(self._preproc_annotations_save_button)
    action_layout.addWidget(self._preproc_annotations_plot_button)
    action_layout.addStretch(1)
    layout.addWidget(actions_row)
    return block


def set_annotations_editable(self, editable: bool) -> None:
    _ = editable
    self._annotations_edit_mode = False
    if self._preproc_annotations_table is not None:
        self._preproc_annotations_table.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )
    if self._preproc_annotations_edit_button is not None:
        self._preproc_annotations_edit_button.setText("Configure...")


def annotations_table_rows(self) -> tuple[list[dict[str, Any]], list[int]]:
    if self._preproc_annotations_table is None:
        return [], []
    rows: list[dict[str, Any]] = []
    invalid_rows: list[int] = []
    for row_idx in range(self._preproc_annotations_table.rowCount()):
        description_item = self._preproc_annotations_table.item(row_idx, 0)
        onset_item = self._preproc_annotations_table.item(row_idx, 1)
        duration_item = self._preproc_annotations_table.item(row_idx, 2)
        description = (
            description_item.text() if description_item is not None else ""
        ).strip()
        onset_text = (onset_item.text() if onset_item is not None else "").strip()
        duration_text = (
            duration_item.text() if duration_item is not None else ""
        ).strip()

        if not description and not onset_text and not duration_text:
            continue
        rows.append(
            {
                "description": description,
                "onset": onset_text,
                "duration": duration_text,
                "_row_idx": row_idx,
            }
        )
        try:
            onset_value = float(onset_text)
            duration_value = float(duration_text)
        except Exception:
            invalid_rows.append(row_idx)
            continue
        if not description or onset_value < 0.0 or duration_value < 0.0:
            invalid_rows.append(row_idx)
    return rows, invalid_rows


def highlight_annotation_rows(self, invalid_rows: list[int]) -> None:
    if self._preproc_annotations_table is None:
        return
    invalid_set = set(invalid_rows)
    for row_idx in range(self._preproc_annotations_table.rowCount()):
        for col_idx in range(self._preproc_annotations_table.columnCount()):
            item = self._preproc_annotations_table.item(row_idx, col_idx)
            if item is None:
                item = QTableWidgetItem("")
                self._preproc_annotations_table.setItem(row_idx, col_idx, item)
            if row_idx in invalid_set:
                item.setBackground(Qt.red)
            else:
                item.setBackground(Qt.transparent)


def append_annotation_rows(self, rows: list[dict[str, Any]]) -> None:
    if self._preproc_annotations_table is None:
        return
    for row in rows:
        row_idx = self._preproc_annotations_table.rowCount()
        self._preproc_annotations_table.insertRow(row_idx)
        self._preproc_annotations_table.setItem(
            row_idx, 0, QTableWidgetItem(str(row.get("description", "")))
        )
        self._preproc_annotations_table.setItem(
            row_idx, 1, QTableWidgetItem(str(row.get("onset", "")))
        )
        self._preproc_annotations_table.setItem(
            row_idx, 2, QTableWidgetItem(str(row.get("duration", "")))
        )


def reset_annotations_table(self) -> None:
    if self._preproc_annotations_table is not None:
        self._preproc_annotations_table.setRowCount(0)
    self._set_annotations_editable(True)
