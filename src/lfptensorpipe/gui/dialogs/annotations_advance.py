"""Annotations Advance dialog."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AnnotationsAdvanceDialog(QDialog):
    """Edit record-scoped Annotations review parameters."""

    def __init__(
        self,
        *,
        session_params: dict[str, Any] | None = None,
        mark_filter_edges_available: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annotations Advance")
        self.setModal(True)
        self._selected_params: dict[str, bool] | None = None
        self._mark_filter_edges_available = bool(mark_filter_edges_available)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self._mark_filter_edges_check = QCheckBox()
        tooltip = (
            "Mark prior Filter support only around BAD support added or expanded "
            "by Annotations. This does not refilter or change the time axis."
        )
        label = QLabel("mark filter edges")
        label.setToolTip(tooltip)
        self._mark_filter_edges_check.setToolTip(tooltip)
        self._mark_filter_edges_check.setChecked(
            self._mark_filter_edges_available
            and isinstance(session_params, dict)
            and session_params.get("mark_filter_edges") is True
        )
        self._mark_filter_edges_check.setEnabled(self._mark_filter_edges_available)
        form.addRow(label, self._mark_filter_edges_check)
        root.addLayout(form)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        save_button.setToolTip("Save this policy for the current record.")
        cancel_button.setToolTip("Close without changing record parameters.")
        save_button.clicked.connect(self._on_save)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        button_layout.addStretch(1)
        root.addWidget(button_row)

    @property
    def selected_params(self) -> dict[str, bool] | None:
        return (
            dict(self._selected_params)
            if isinstance(self._selected_params, dict)
            else None
        )

    def _on_save(self) -> None:
        self._selected_params = {
            "mark_filter_edges": bool(
                self._mark_filter_edges_available
                and self._mark_filter_edges_check.isChecked()
            )
        }
        self.accept()
