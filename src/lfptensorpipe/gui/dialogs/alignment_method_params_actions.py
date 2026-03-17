"""Save/default actions for the alignment params dialog."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from lfptensorpipe.app import (
    default_alignment_method_params,
    save_alignment_method_default_params,
    validate_alignment_method_params,
)

from .alignment_method_params_table import (
    _parse_optional_float,
    _validate_all_table_cells,
)


def _collect_candidate_params(dialog) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "drop_bad": dialog._drop_bad_check.isChecked(),
        "drop_fields": ["bad", "edge"],
        "sample_rate": float(dialog._sample_rate_edit.text().strip()),
    }
    if dialog._method_key == "linear_warper":
        anchors: dict[float, str] = {}
        for row in range(dialog._anchors_table.rowCount()):
            label_item = dialog._anchors_table.item(row, 0)
            percent_item = dialog._anchors_table.item(row, 1)
            if percent_item is None or label_item is None:
                continue
            label = label_item.text().strip()
            if not label:
                continue
            percent = float(percent_item.text().strip())
            if percent in anchors:
                raise ValueError("target percent values must be unique.")
            anchors[percent] = label
        candidate["anchors_percent"] = anchors
        candidate["epoch_duration_range"] = [
            _parse_optional_float(dialog._duration_min_edit.text()),
            _parse_optional_float(dialog._duration_max_edit.text()),
        ]
        candidate["linear_warp"] = dialog._linear_warp_check.isChecked()
        candidate["percent_tolerance"] = float(
            dialog._percent_tolerance_edit.text().strip()
        )
        return candidate
    if dialog._method_key == "pad_warper":
        annotations: list[str] = []
        for row in range(dialog._annotation_list.count()):
            item = dialog._annotation_list.item(row)
            if item is not None and item.checkState() == Qt.Checked:
                annotations.append(item.text().strip())
        candidate["annotations"] = [item for item in annotations if item]
        candidate["pad_left"] = float(dialog._pad_left_edit.text().strip())
        candidate["anno_left"] = float(dialog._anno_left_edit.text().strip())
        candidate["anno_right"] = float(dialog._anno_right_edit.text().strip())
        candidate["pad_right"] = float(dialog._pad_right_edit.text().strip())
        candidate["duration_range"] = [
            float(dialog._duration_min_edit.text().strip()),
            float(dialog._duration_max_edit.text().strip()),
        ]
        return candidate

    annotations: list[str] = []
    for row in range(dialog._annotation_list.count()):
        item = dialog._annotation_list.item(row)
        if item is not None and item.checkState() == Qt.Checked:
            annotations.append(item.text().strip())
    candidate["annotations"] = [item for item in annotations if item]
    if dialog._method_key == "stack_warper":
        candidate["duration_range"] = [
            float(dialog._duration_min_edit.text().strip()),
            float(dialog._duration_max_edit.text().strip()),
        ]
    return candidate


def _on_save(dialog) -> None:
    _validate_all_table_cells(dialog)
    if dialog._table_validation_error:
        dialog._show_warning(
            "Align Epochs Params",
            "Fix highlighted table cells before saving.",
        )
        return
    try:
        candidate = _collect_candidate_params(dialog)
    except Exception as exc:  # noqa: BLE001
        dialog._show_warning("Align Epochs Params", f"Invalid parameters:\n{exc}")
        return
    ok, normalized, message = validate_alignment_method_params(
        dialog._method_key,
        candidate,
        annotation_labels=dialog._annotation_labels,
    )
    if not ok:
        dialog._show_warning("Align Epochs Params", message)
        return
    dialog._selected_params = normalized
    dialog.accept()


def _on_set_as_default(dialog) -> None:
    _validate_all_table_cells(dialog)
    if dialog._table_validation_error:
        dialog._show_warning(
            "Align Epochs Params",
            "Fix highlighted table cells before setting defaults.",
        )
        return
    try:
        candidate = _collect_candidate_params(dialog)
    except Exception as exc:  # noqa: BLE001
        dialog._show_warning("Align Epochs Params", f"Invalid parameters:\n{exc}")
        return
    ok, message, normalized = save_alignment_method_default_params(
        dialog._config_store,
        method_key=dialog._method_key,
        method_params=candidate,
        annotation_labels=dialog._annotation_labels,
    )
    if not ok or normalized is None:
        dialog._show_warning("Align Epochs Params", message)
        return
    dialog._apply_common(normalized)
    dialog._build_method_ui(normalized)
    dialog._show_information("Align Epochs Params", "Default params saved.")


def _on_restore_default(dialog) -> None:
    restored = default_alignment_method_params(dialog._method_key)
    dialog._apply_common(restored)
    dialog._build_method_ui(restored)
