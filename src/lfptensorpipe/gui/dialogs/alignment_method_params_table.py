"""Anchor-table and annotation-list helpers for the alignment params dialog."""

from __future__ import annotations

from .common import *  # noqa: F403


def _on_add_anchor_row(dialog) -> None:
    if dialog._anchor_label_combo is None or dialog._anchor_percent_edit is None:
        return
    if not dialog._annotation_labels:
        dialog._show_information("Align Epochs Params", "No labels available.")
        return
    label = dialog._anchor_label_combo.currentData()
    if not isinstance(label, str) or not label.strip():
        dialog._show_warning("Align Epochs Params", "Select a valid event name label.")
        return
    percent_text = dialog._anchor_percent_edit.text().strip()
    try:
        percent = float(percent_text)
    except Exception:
        dialog._show_warning("Align Epochs Params", "target percent must be numeric.")
        return
    if percent < 0.0 or percent > 100.0:
        dialog._show_warning(
            "Align Epochs Params", "target percent must be within [0, 100]."
        )
        return
    rows = dialog._anchor_rows_from_table()
    rows.append((label, percent))
    dialog._set_anchor_rows(rows)
    _validate_all_table_cells(dialog)


def _on_remove_anchor_row(dialog, row: int | None = None) -> None:
    if row is None:
        row = dialog._anchors_table.currentRow()
    if row < 0 or row >= dialog._anchors_table.rowCount():
        return
    rows = dialog._anchor_rows_from_table()
    if row >= len(rows):
        return
    rows.pop(row)
    dialog._set_anchor_rows(rows)
    _validate_all_table_cells(dialog)


def _on_select_all_annotations(dialog) -> None:
    for row in range(dialog._annotation_list.count()):
        item = dialog._annotation_list.item(row)
        if item is not None:
            item.setCheckState(Qt.Checked)


def _on_clear_annotations(dialog) -> None:
    for row in range(dialog._annotation_list.count()):
        item = dialog._annotation_list.item(row)
        if item is not None:
            item.setCheckState(Qt.Unchecked)


def _set_cell_error(
    dialog,
    item: QTableWidgetItem | None,
    *,
    error: str | None,
) -> None:
    if item is None:
        return
    if error:
        item.setBackground(dialog._ERROR_BG)
        item.setData(dialog._CELL_ERROR_ROLE, error)
        item.setToolTip(error)
        return
    item.setData(Qt.BackgroundRole, None)
    item.setData(dialog._CELL_ERROR_ROLE, None)
    item.setToolTip("")


def _validate_anchor_cell(dialog, row: int, col: int) -> None:
    item = dialog._anchors_table.item(row, col)
    if item is None:
        return
    text = item.text().strip()
    error: str | None = None
    if col == 0:
        if not text:
            error = "event name cannot be empty."
    elif col == 1:
        try:
            value = float(text)
            if value < 0.0 or value > 100.0:
                error = "target percent must be within [0, 100]."
        except Exception:
            error = "target percent must be numeric."
    _set_cell_error(dialog, item, error=error)


def _validate_all_table_cells(dialog) -> None:
    dialog._table_validation_error = False
    if dialog._method_key == "linear_warper":
        for row in range(dialog._anchors_table.rowCount()):
            for col in (0, 1):
                _validate_anchor_cell(dialog, row, col)
                item = dialog._anchors_table.item(row, col)
                if item is not None and item.data(dialog._CELL_ERROR_ROLE):
                    dialog._table_validation_error = True


def _on_anchors_item_changed(dialog, item: QTableWidgetItem) -> None:
    if dialog._method_key != "linear_warper":
        return
    _validate_anchor_cell(dialog, item.row(), item.column())
    _validate_all_table_cells(dialog)


def _parse_optional_float(text: str) -> float | None:
    token = text.strip()
    if token == "":
        return None
    return float(token)
