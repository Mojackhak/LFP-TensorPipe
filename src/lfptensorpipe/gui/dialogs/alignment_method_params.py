"""Alignment method-params dialog."""

from __future__ import annotations

from typing import Any

from .common import (
    AppConfigStore,
    QAbstractItemView,
    QCheckBox,
    QColor,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    Qt,
)
from .alignment_method_params_actions import (
    _collect_candidate_params as _collect_candidate_params_impl,
    _on_restore_default as _on_restore_default_impl,
    _on_save as _on_save_impl,
    _on_set_as_default as _on_set_as_default_impl,
)
from .alignment_method_params_table import (
    _on_add_anchor_row as _on_add_anchor_row_impl,
    _on_anchors_item_changed as _on_anchors_item_changed_impl,
    _on_clear_annotations as _on_clear_annotations_impl,
    _on_remove_anchor_row as _on_remove_anchor_row_impl,
    _on_select_all_annotations as _on_select_all_annotations_impl,
    _parse_optional_float as _parse_optional_float_impl,
    _set_cell_error as _set_cell_error_impl,
    _validate_all_table_cells as _validate_all_table_cells_impl,
    _validate_anchor_cell as _validate_anchor_cell_impl,
)
from .alignment_method_params_ui import (
    _anchor_rows_from_table as _anchor_rows_from_table_impl,
    _build_linear_ui as _build_linear_ui_impl,
    _build_method_ui as _build_method_ui_impl,
    _build_pad_ui as _build_pad_ui_impl,
    _build_stack_concat_ui as _build_stack_concat_ui_impl,
    _clear_method_ui as _clear_method_ui_impl,
    _reset_label_combo as _reset_label_combo_impl,
    _set_anchor_rows as _set_anchor_rows_impl,
)


def _normalize_hidden_drop_fields(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ("bad", "edge")
    tokens: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item).strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens or ("bad", "edge"))


def _filter_visible_annotation_labels(
    annotation_labels: list[str],
    *,
    hidden_drop_fields: tuple[str, ...],
) -> list[str]:
    visible: list[str] = []
    seen: set[str] = set()
    for item in annotation_labels:
        label = str(item).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        lowered = label.lower()
        if any(token in lowered for token in hidden_drop_fields):
            continue
        visible.append(label)
    return visible


class AlignmentMethodParamsDialog(QDialog):
    """Method-specific parameter editor for Align Epochs."""

    _CELL_ERROR_ROLE = Qt.UserRole + 101
    _ERROR_BG = QColor("#ffd6d6")

    def __init__(
        self,
        *,
        method_key: str,
        session_params: dict[str, Any],
        annotation_labels: list[str],
        config_store: AppConfigStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._method_key = method_key
        self._config_store = config_store
        self._annotation_labels = [
            str(item).strip() for item in annotation_labels if str(item).strip()
        ]
        self._hidden_drop_fields = _normalize_hidden_drop_fields(
            session_params.get("drop_fields", ["bad", "edge"])
        )
        self._visible_annotation_labels = _filter_visible_annotation_labels(
            self._annotation_labels,
            hidden_drop_fields=self._hidden_drop_fields,
        )
        self._selected_params: dict[str, Any] | None = None
        self._table_validation_error = False
        self.setWindowTitle("Align Epochs Params")
        self.resize(620, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self._drop_bad_check = QCheckBox("drop bad/edge")
        self._drop_bad_check.setToolTip(
            "Drop epochs overlapping annotations containing 'bad' or 'edge' (substring match)."
        )
        self._sample_rate_edit = QLineEdit("5")
        self._sample_rate_edit.setToolTip(self._sample_rate_tooltip_text())

        common_form = QFormLayout()
        common_form.addRow(self._sample_rate_label_text(), self._sample_rate_edit)
        common_form.addRow("", self._drop_bad_check)
        root.addLayout(common_form)

        self._duration_min_edit = QLineEdit()
        self._duration_max_edit = QLineEdit()
        self._linear_warp_check = QCheckBox("linear warp")
        self._percent_tolerance_edit = QLineEdit()
        self._pad_s_edit = QLineEdit()

        self._anchors_table = QTableWidget(0, 3)
        self._anchors_table.setHorizontalHeaderLabels(
            ["event name", "target percent", "action"]
        )
        self._anchors_table.verticalHeader().setVisible(False)
        self._anchors_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._anchors_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        anchors_header = self._anchors_table.horizontalHeader()
        anchors_header.setSectionResizeMode(0, QHeaderView.Stretch)
        anchors_header.setSectionResizeMode(1, QHeaderView.Stretch)
        anchors_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._anchors_table.setMinimumHeight(120)
        self._anchors_table.setToolTip("Map target percent (0-100) to event labels.")
        anchors_header_event = self._anchors_table.horizontalHeaderItem(0)
        if anchors_header_event is not None:
            anchors_header_event.setToolTip("Annotation label used at this anchor.")
        anchors_header_percent = self._anchors_table.horizontalHeaderItem(1)
        if anchors_header_percent is not None:
            anchors_header_percent.setToolTip(
                "Anchor position in [0, 100]. Must include 0 and 100."
            )
        self._anchors_table.cellClicked.connect(self._on_anchors_table_clicked)
        self._anchors_table.itemChanged.connect(self._on_anchors_item_changed)
        self._anchor_label_combo: QComboBox | None = None
        self._anchor_percent_edit: QLineEdit | None = None

        self._pad_left_edit: QLineEdit | None = None
        self._anno_left_edit: QLineEdit | None = None
        self._anno_right_edit: QLineEdit | None = None
        self._pad_right_edit: QLineEdit | None = None

        self._annotation_list = QListWidget()
        self._annotation_list.setMinimumHeight(140)
        self._annotation_list.setToolTip("Select annotation labels to keep.")

        self._method_box = QGroupBox("Method Params")
        self._method_layout = QVBoxLayout(self._method_box)
        self._method_layout.setContentsMargins(8, 8, 8, 8)
        self._method_layout.setSpacing(6)
        root.addWidget(self._method_box, stretch=1)

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(6)
        self._set_default_button = QPushButton("Set as Default")
        self._restore_default_button = QPushButton("Restore Default")
        self._set_default_button.clicked.connect(self._on_set_as_default)
        self._restore_default_button.clicked.connect(self._on_restore_default)
        self._set_default_button.setToolTip(
            "Save current method parameters as defaults for this alignment method."
        )
        self._restore_default_button.setToolTip(
            "Restore saved defaults for this alignment method."
        )
        action_layout.addWidget(self._set_default_button)
        action_layout.addWidget(self._restore_default_button)
        action_layout.addStretch(1)
        root.addWidget(action_row)

        footer = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        footer.accepted.connect(self._on_save)
        footer.rejected.connect(self.reject)
        save_button = footer.button(QDialogButtonBox.Save)
        cancel_button = footer.button(QDialogButtonBox.Cancel)
        if save_button is not None:
            save_button.setToolTip(
                "Use the current method parameters for the selected trial."
            )
        if cancel_button is not None:
            cancel_button.setToolTip("Close without changing method parameters.")
        root.addWidget(footer)

        self._apply_common(session_params)
        self._build_method_ui(session_params)

    @property
    def selected_params(self) -> dict[str, Any] | None:
        return self._selected_params

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _show_information(self, title: str, message: str) -> int:
        return QMessageBox.information(self, title, message)

    def _sample_rate_label_text(self) -> str:
        if self._method_key in {"pad_warper", "concat_warper"}:
            return "sample rate (Hz)"
        return "sample rate (n/%)"

    def _sample_rate_tooltip_text(self) -> str:
        if self._method_key in {"pad_warper", "concat_warper"}:
            return "Target sampling density in Hz. Warped n_samples = round(sample_rate * total time)."
        return "Target sampling density over 0-100% timeline. Warped n_samples = round(sample_rate * 100)."

    def _apply_common(self, params: dict[str, Any]) -> None:
        self._drop_bad_check.setChecked(bool(params.get("drop_bad", True)))
        default_rate = (
            50.0 if self._method_key in {"pad_warper", "concat_warper"} else 5.0
        )
        self._sample_rate_edit.setText(
            f"{float(params.get('sample_rate', default_rate)):g}"
        )

    def _clear_method_ui(self) -> None:
        _clear_method_ui_impl(self)

    def _build_method_ui(self, params: dict[str, Any]) -> None:
        _build_method_ui_impl(self, params)

    def _build_linear_ui(self, params: dict[str, Any]) -> None:
        _build_linear_ui_impl(self, params)

    def _build_pad_ui(self, params: dict[str, Any]) -> None:
        _build_pad_ui_impl(self, params)

    def _build_stack_concat_ui(self, params: dict[str, Any]) -> None:
        _build_stack_concat_ui_impl(self, params)

    def _reset_label_combo(self, combo: QComboBox | None) -> None:
        _reset_label_combo_impl(self, combo)

    def _set_anchor_rows(self, rows: list[tuple[str, float]]) -> None:
        _set_anchor_rows_impl(self, rows)

    def _anchor_rows_from_table(self) -> list[tuple[str, float]]:
        return _anchor_rows_from_table_impl(self)

    def _on_add_anchor_row(self) -> None:
        _on_add_anchor_row_impl(self)

    def _on_remove_anchor_row(self, row: int | None = None) -> None:
        _on_remove_anchor_row_impl(self, row)

    def _on_select_all_annotations(self) -> None:
        _on_select_all_annotations_impl(self)

    def _on_clear_annotations(self) -> None:
        _on_clear_annotations_impl(self)

    @staticmethod
    def _set_cell_error(
        item: QTableWidgetItem | None,
        *,
        error: str | None,
    ) -> None:
        _set_cell_error_impl(
            AlignmentMethodParamsDialog,
            item,
            error=error,
        )

    def _validate_anchor_cell(self, row: int, col: int) -> None:
        _validate_anchor_cell_impl(self, row, col)

    def _validate_all_table_cells(self) -> None:
        _validate_all_table_cells_impl(self)

    def _on_anchors_item_changed(self, item: QTableWidgetItem) -> None:
        _on_anchors_item_changed_impl(self, item)

    def _on_anchors_table_clicked(self, row: int, column: int) -> None:
        if column != 2:
            return
        self._on_remove_anchor_row(row)

    @staticmethod
    def _parse_optional_float(text: str) -> float | None:
        return _parse_optional_float_impl(text)

    def _collect_candidate_params(self) -> dict[str, Any]:
        return _collect_candidate_params_impl(self)

    def _on_save(self) -> None:
        _on_save_impl(self)

    def _on_set_as_default(self) -> None:
        _on_set_as_default_impl(self)

    def _on_restore_default(self) -> None:
        _on_restore_default_impl(self)
