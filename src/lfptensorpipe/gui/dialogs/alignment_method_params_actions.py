"""Save/default actions for the alignment params dialog."""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import Qt
from lfptensorpipe.app import (
    load_alignment_method_default_params,
    save_alignment_method_default_params,
    validate_alignment_method_params,
)

from .alignment_method_params_table import (
    _parse_optional_float,
    _validate_all_table_cells,
)


def _draft_float(text: str) -> float | str | None:
    token = text.strip()
    if not token:
        return None
    try:
        value = float(token)
    except Exception:
        return token
    return value if math.isfinite(value) else token


def _collect_candidate_params(
    dialog,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    parse_float = float if strict else _draft_float
    candidate: dict[str, Any] = {
        "drop_bad": dialog._drop_bad_check.isChecked(),
        "drop_fields": ["bad", "edge"],
        "sample_rate": parse_float(dialog._sample_rate_edit.text().strip()),
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
        if strict:
            candidate["epoch_duration_range"] = [
                _parse_optional_float(dialog._duration_min_edit.text()),
                _parse_optional_float(dialog._duration_max_edit.text()),
            ]
        else:
            candidate["epoch_duration_range"] = [
                _draft_float(dialog._duration_min_edit.text()),
                _draft_float(dialog._duration_max_edit.text()),
            ]
        candidate["linear_warp"] = dialog._linear_warp_check.isChecked()
        candidate["percent_tolerance"] = parse_float(
            dialog._percent_tolerance_edit.text()
        )
        return candidate
    if dialog._method_key == "pad_warper":
        annotations: list[str] = []
        for row in range(dialog._annotation_list.count()):
            item = dialog._annotation_list.item(row)
            if item is not None and item.checkState() == Qt.Checked:
                annotations.append(item.text().strip())
        candidate["annotations"] = [item for item in annotations if item]
        candidate["pad_left"] = parse_float(dialog._pad_left_edit.text())
        candidate["anno_left"] = parse_float(dialog._anno_left_edit.text())
        candidate["anno_right"] = parse_float(dialog._anno_right_edit.text())
        candidate["pad_right"] = parse_float(dialog._pad_right_edit.text())
        candidate["duration_range"] = [
            parse_float(dialog._duration_min_edit.text()),
            parse_float(dialog._duration_max_edit.text()),
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
            parse_float(dialog._duration_min_edit.text()),
            parse_float(dialog._duration_max_edit.text()),
        ]
    return candidate


def _filter_restored_params_for_labels(
    method_key: str,
    params: dict[str, Any],
    *,
    annotation_labels: list[str],
) -> dict[str, Any]:
    restored = dict(params)
    available = {str(item).strip() for item in annotation_labels if str(item).strip()}
    if method_key in {"pad_warper", "stack_warper", "concat_warper"}:
        annotations = restored.get("annotations", [])
        restored["annotations"] = (
            [
                str(item).strip()
                for item in annotations
                if str(item).strip() in available
            ]
            if isinstance(annotations, list)
            else []
        )
        return restored

    if method_key != "linear_warper":
        return restored
    raw_anchors = restored.get("anchors_percent", {})
    anchors = (
        {
            percent: str(label).strip()
            for percent, label in raw_anchors.items()
            if str(label).strip() in available
        }
        if isinstance(raw_anchors, dict)
        else {}
    )
    if anchors:
        candidate = dict(restored)
        candidate["anchors_percent"] = anchors
        ok, _, _ = validate_alignment_method_params(method_key, candidate)
        if not ok:
            anchors = {}
    restored["anchors_percent"] = anchors
    return restored


def _on_save(dialog) -> None:
    _validate_all_table_cells(dialog)
    if dialog._table_validation_error:
        dialog._show_warning(
            "Align Epochs Params",
            "Fix highlighted table cells before saving.",
        )
        return
    candidate = _collect_candidate_params(dialog, strict=False)
    dialog._selected_params = candidate
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
    if dialog._method_key != "linear_warper" and not candidate.get("annotations"):
        dialog._refresh_annotation_validation()
        dialog._show_warning(
            "Align Epochs Params",
            "At least one annotation is required before setting defaults.",
        )
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
    restored = load_alignment_method_default_params(
        dialog._config_store,
        method_key=dialog._method_key,
    )
    restored = _filter_restored_params_for_labels(
        dialog._method_key,
        restored,
        annotation_labels=dialog._annotation_labels,
    )
    dialog._apply_common(restored)
    dialog._build_method_ui(restored)
