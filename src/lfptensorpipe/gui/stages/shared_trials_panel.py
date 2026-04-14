"""Shared trials-panel builder for Align Epochs and Extract Features."""

from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

GRID_SPACING = 0


def _grid_spacing(owner) -> int:
    getter = getattr(owner, "_stage_panel_grid_spacing", None)
    return GRID_SPACING if getter is None else int(getter())


def build_trials_panel(
    owner,
    *,
    block_attr: str,
    list_attr: str,
    add_button_attr: str,
    delete_button_attr: str,
    action_row_attr: str,
    on_selected: Callable[[int], None],
    add_enabled: bool,
    delete_enabled: bool,
    on_add: Callable[[], None] | None = None,
    on_delete: Callable[[], None] | None = None,
    list_tooltip: str = "Select the trial/paradigm used by this page.",
    add_tooltip: str = "Create a new trial entry.",
    delete_tooltip: str = "Delete the selected trial entry.",
) -> QGroupBox:
    block = QGroupBox("Trials")
    setattr(owner, block_attr, block)

    layout = QVBoxLayout(block)
    layout.setContentsMargins(8, 8, 8, 8)
    grid_spacing = _grid_spacing(owner)
    layout.setSpacing(grid_spacing)

    list_widget = QListWidget()
    list_widget.currentRowChanged.connect(on_selected)
    list_widget.setToolTip(list_tooltip)
    setattr(owner, list_attr, list_widget)
    layout.addWidget(list_widget, stretch=1)

    row = QWidget()
    setattr(owner, action_row_attr, row)
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(grid_spacing)

    add_button = QPushButton("+")
    add_button.setEnabled(add_enabled)
    add_button.setToolTip(add_tooltip)
    if on_add is not None:
        add_button.clicked.connect(on_add)
    setattr(owner, add_button_attr, add_button)
    row_layout.addWidget(add_button)

    delete_button = QPushButton("-")
    delete_button.setEnabled(delete_enabled)
    delete_button.setToolTip(delete_tooltip)
    if on_delete is not None:
        delete_button.clicked.connect(on_delete)
    setattr(owner, delete_button_attr, delete_button)
    row_layout.addWidget(delete_button)

    row_layout.addStretch(1)
    layout.addWidget(row)
    return block
