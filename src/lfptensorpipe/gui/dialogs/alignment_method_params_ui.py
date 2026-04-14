"""Method-specific UI builders for the alignment params dialog."""

from __future__ import annotations

from typing import Any

from .common import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QPushButton,
    QTableWidgetItem,
    QWidget,
    Qt,
    make_action_table_item,
)
from .alignment_method_params_table import (
    _validate_all_table_cells,
)


def _clear_method_ui(dialog) -> None:
    while dialog._method_layout.count() > 0:
        item = dialog._method_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)


def _build_method_ui(dialog, params: dict[str, Any]) -> None:
    _clear_method_ui(dialog)
    if dialog._method_key == "linear_warper":
        _build_linear_ui(dialog, params)
    elif dialog._method_key == "pad_warper":
        _build_pad_ui(dialog, params)
    else:
        _build_stack_concat_ui(dialog, params)


def _build_linear_ui(dialog, params: dict[str, Any]) -> None:
    anchors_label = QLabel("anchors percent")
    dialog._method_layout.addWidget(anchors_label)
    anchors = params.get("anchors_percent", {})
    if not isinstance(anchors, dict):
        anchors = {}
    rows: list[tuple[str, float]] = []
    for key, label in anchors.items():
        try:
            rows.append((str(label), float(key)))
        except Exception:  # noqa: BLE001
            continue
    rows.sort(key=lambda item: item[1])
    dialog._set_anchor_rows(rows)
    dialog._method_layout.addWidget(dialog._anchors_table)

    anchor_draft = QFrame()
    anchor_draft_layout = QGridLayout(anchor_draft)
    anchor_draft_layout.setContentsMargins(0, 0, 0, 0)
    anchor_draft_layout.setHorizontalSpacing(6)
    anchor_draft_layout.setVerticalSpacing(4)
    anchor_draft_layout.addWidget(QLabel("event name"), 0, 0)
    dialog._anchor_label_combo = QComboBox()
    dialog._anchor_percent_edit = QLineEdit()
    dialog._reset_label_combo(dialog._anchor_label_combo)
    dialog._anchor_label_combo.setToolTip("Annotation label used at this anchor.")
    anchor_draft_layout.addWidget(dialog._anchor_label_combo, 0, 1)
    anchor_draft_layout.addWidget(QLabel("target percent"), 0, 2)
    dialog._anchor_percent_edit.setToolTip(
        "Anchor position in [0, 100]. Must include 0 and 100."
    )
    anchor_draft_layout.addWidget(dialog._anchor_percent_edit, 0, 3)
    add_anchor_button = QPushButton("Add Anchor")
    add_anchor_button.setToolTip("Add an anchor row.")
    add_anchor_button.clicked.connect(dialog._on_add_anchor_row)
    has_labels = bool(dialog._visible_annotation_labels)
    add_anchor_button.setEnabled(has_labels)
    dialog._anchor_label_combo.setEnabled(has_labels)
    dialog._anchor_percent_edit.setEnabled(has_labels)
    dialog._anchor_percent_edit.setText("50")
    anchor_draft_layout.addWidget(add_anchor_button, 0, 4)
    dialog._method_layout.addWidget(anchor_draft)

    duration_range = params.get("epoch_duration_range", [None, None])
    if not isinstance(duration_range, (list, tuple)) or len(duration_range) != 2:
        duration_range = [None, None]
    dialog._duration_min_edit.setText(
        "" if duration_range[0] is None else f"{float(duration_range[0]):g}"
    )
    dialog._duration_max_edit.setText(
        "" if duration_range[1] is None else f"{float(duration_range[1]):g}"
    )
    dialog._linear_warp_check.setChecked(bool(params.get("linear_warp", True)))
    dialog._percent_tolerance_edit.setText(
        f"{float(params.get('percent_tolerance', 15.0)):g}"
    )
    dialog._duration_min_edit.setToolTip("Optional minimum epoch duration in seconds.")
    dialog._duration_max_edit.setToolTip("Optional maximum epoch duration in seconds.")
    dialog._linear_warp_check.setToolTip(
        "Enable piecewise linear warp between anchors."
    )
    dialog._percent_tolerance_edit.setToolTip(
        "Allowed anchor timing deviation in percent (>= 0)."
    )

    form = QFormLayout()
    form.addRow("epoch duration min", dialog._duration_min_edit)
    form.addRow("epoch duration max", dialog._duration_max_edit)
    form.addRow("", dialog._linear_warp_check)
    form.addRow("percent tolerance", dialog._percent_tolerance_edit)
    dialog._method_layout.addLayout(form)
    _validate_all_table_cells(dialog)


def _build_pad_ui(dialog, params: dict[str, Any]) -> None:
    selected = params.get("annotations", [])
    if not isinstance(selected, list):
        selected = []
    selected_set = {str(item).strip() for item in selected if str(item).strip()}
    labels = list(dialog._visible_annotation_labels)
    for label in selected_set:
        lowered = label.lower()
        if label not in labels and not any(
            token in lowered for token in dialog._hidden_drop_fields
        ):
            labels.append(label)
    dialog._annotation_list.clear()
    for label in labels:
        item = QListWidgetItem(label)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if label in selected_set else Qt.Unchecked)
        item.setToolTip("Include this label in epoch building.")
        dialog._annotation_list.addItem(item)
    dialog._method_layout.addWidget(QLabel("annotations"))
    dialog._method_layout.addWidget(dialog._annotation_list)
    _add_annotation_actions(dialog)

    dialog._pad_left_edit = QLineEdit()
    dialog._anno_left_edit = QLineEdit()
    dialog._anno_right_edit = QLineEdit()
    dialog._pad_right_edit = QLineEdit()
    try:
        dialog._pad_left_edit.setText(f"{float(params.get('pad_left', 0.5)):g}")
        dialog._anno_left_edit.setText(f"{float(params.get('anno_left', 0.5)):g}")
        dialog._anno_right_edit.setText(f"{float(params.get('anno_right', 0.5)):g}")
        dialog._pad_right_edit.setText(f"{float(params.get('pad_right', 0.5)):g}")
    except Exception:
        dialog._pad_left_edit.setText("0.5")
        dialog._anno_left_edit.setText("0.5")
        dialog._anno_right_edit.setText("0.5")
        dialog._pad_right_edit.setText("0.5")
    dialog._pad_left_edit.setToolTip("Seconds before annotation start.")
    dialog._anno_left_edit.setToolTip("Seconds after annotation start.")
    dialog._anno_right_edit.setToolTip("Seconds before annotation end.")
    dialog._pad_right_edit.setToolTip("Seconds after annotation end.")

    duration_range = params.get("duration_range", [0.0, 1_000_000.0])
    if not isinstance(duration_range, (list, tuple)) or len(duration_range) != 2:
        duration_range = [0.0, 1_000_000.0]
    dialog._duration_min_edit.setText(f"{float(duration_range[0]):g}")
    dialog._duration_max_edit.setText(f"{float(duration_range[1]):g}")
    dialog._duration_min_edit.setToolTip(
        "Minimum annotation duration in seconds (>= 0)."
    )
    dialog._duration_max_edit.setToolTip(
        "Maximum annotation duration in seconds (>= duration min)."
    )
    form = QFormLayout()
    form.addRow("pad left", dialog._pad_left_edit)
    form.addRow("anno left", dialog._anno_left_edit)
    form.addRow("anno right", dialog._anno_right_edit)
    form.addRow("pad right", dialog._pad_right_edit)
    form.addRow("duration min", dialog._duration_min_edit)
    form.addRow("duration max", dialog._duration_max_edit)
    dialog._method_layout.addLayout(form)
    _validate_all_table_cells(dialog)


def _build_stack_concat_ui(dialog, params: dict[str, Any]) -> None:
    selected = params.get("annotations", [])
    if not isinstance(selected, list):
        selected = []
    selected_set = {str(item).strip() for item in selected if str(item).strip()}
    labels = list(dialog._visible_annotation_labels)
    for label in selected_set:
        lowered = label.lower()
        if label not in labels and not any(
            token in lowered for token in dialog._hidden_drop_fields
        ):
            labels.append(label)
    dialog._annotation_list.clear()
    for label in labels:
        item = QListWidgetItem(label)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if label in selected_set else Qt.Unchecked)
        item.setToolTip("Include this label in epoch building.")
        dialog._annotation_list.addItem(item)
    dialog._method_layout.addWidget(QLabel("annotations"))
    dialog._method_layout.addWidget(dialog._annotation_list)
    _add_annotation_actions(dialog)

    if dialog._method_key == "concat_warper":
        return

    duration_range = params.get("duration_range", [0.0, 1_000_000.0])
    if not isinstance(duration_range, (list, tuple)) or len(duration_range) != 2:
        duration_range = [0.0, 1_000_000.0]
    dialog._duration_min_edit.setText(f"{float(duration_range[0]):g}")
    dialog._duration_max_edit.setText(f"{float(duration_range[1]):g}")
    dialog._duration_min_edit.setToolTip(
        "Minimum annotation duration in seconds (>= 0)."
    )
    dialog._duration_max_edit.setToolTip(
        "Maximum annotation duration in seconds (>= duration min)."
    )
    form = QFormLayout()
    form.addRow("duration min", dialog._duration_min_edit)
    form.addRow("duration max", dialog._duration_max_edit)
    dialog._method_layout.addLayout(form)


def _add_annotation_actions(dialog) -> None:
    annotation_actions = QWidget()
    annotation_actions_layout = QHBoxLayout(annotation_actions)
    annotation_actions_layout.setContentsMargins(0, 0, 0, 0)
    annotation_actions_layout.setSpacing(6)
    select_all_button = QPushButton("Select All")
    clear_button = QPushButton("Clear")
    select_all_button.setToolTip("Select all labels.")
    clear_button.setToolTip("Unselect all labels.")
    select_all_button.clicked.connect(dialog._on_select_all_annotations)
    clear_button.clicked.connect(dialog._on_clear_annotations)
    annotation_actions_layout.addWidget(select_all_button)
    annotation_actions_layout.addWidget(clear_button)
    annotation_actions_layout.addStretch(1)
    dialog._method_layout.addWidget(annotation_actions)


def _reset_label_combo(dialog, combo: QComboBox | None) -> None:
    if combo is None:
        return
    combo.blockSignals(True)
    combo.clear()
    if not dialog._visible_annotation_labels:
        combo.addItem("No labels available", None)
    else:
        for label in dialog._visible_annotation_labels:
            combo.addItem(label, label)
    combo.blockSignals(False)


def _set_anchor_rows(dialog, rows: list[tuple[str, float]]) -> None:
    sorted_rows = sorted(rows, key=lambda item: item[1])
    dialog._anchors_table.blockSignals(True)
    dialog._anchors_table.setRowCount(0)
    for label, percent in sorted_rows:
        row = dialog._anchors_table.rowCount()
        dialog._anchors_table.insertRow(row)
        dialog._anchors_table.setItem(row, 0, QTableWidgetItem(str(label)))
        dialog._anchors_table.setItem(row, 1, QTableWidgetItem(f"{float(percent):g}"))
        dialog._anchors_table.setItem(
            row,
            2,
            make_action_table_item(
                "Del",
                row,
                tool_tip="Delete this anchor.",
            ),
        )
    dialog._anchors_table.blockSignals(False)


def _anchor_rows_from_table(dialog) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for row in range(dialog._anchors_table.rowCount()):
        label_item = dialog._anchors_table.item(row, 0)
        percent_item = dialog._anchors_table.item(row, 1)
        if label_item is None or percent_item is None:
            continue
        label = label_item.text().strip()
        if not label:
            continue
        rows.append((label, float(percent_item.text().strip())))
    return rows
