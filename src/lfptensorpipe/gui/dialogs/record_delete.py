"""Record delete scope dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from lfptensorpipe.app import RECORD_DELETE_SCOPES, record_delete_scope_paths

_SCOPE_LABELS = {
    "derivatives": "Derivatives",
    "rawdata": "Rawdata",
    "sourcedata": "Sourcedata",
}


class RecordDeleteDialog(QDialog):
    """Select standard record roots for permanent deletion."""

    def __init__(
        self,
        *,
        project_root: Path,
        subject: str,
        record: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Delete Record")
        self.setModal(True)
        self.resize(760, 280)
        self._scope_checkboxes: dict[str, QCheckBox] = {}
        self._scope_paths = record_delete_scope_paths(project_root, subject, record)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        context_label = QLabel(f"Record: {subject} / {record}")
        root.addWidget(context_label)

        warning_label = QLabel(
            "Delete selected data permanently. Files are not moved to Trash "
            "and cannot be restored with Undo."
        )
        warning_label.setWordWrap(True)
        root.addWidget(warning_label)

        for scope in RECORD_DELETE_SCOPES:
            checkbox = QCheckBox(_SCOPE_LABELS[scope])
            checkbox.setObjectName(f"recordDeleteScope_{scope}")
            checkbox.setChecked(scope == "derivatives")
            checkbox.toggled.connect(self._update_delete_enabled)
            root.addWidget(checkbox)

            path_label = QLabel(str(self._scope_paths[scope]))
            path_label.setObjectName(f"recordDeletePath_{scope}")
            path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            path_label.setWordWrap(True)
            root.addWidget(path_label)
            self._scope_checkboxes[scope] = checkbox

        footer = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._delete_button = footer.addButton(
            "Delete",
            QDialogButtonBox.DestructiveRole,
        )
        self._delete_button.setObjectName("recordDeleteConfirm")
        self._delete_button.clicked.connect(self.accept)
        footer.rejected.connect(self.reject)
        root.addWidget(footer)
        self._update_delete_enabled()

    @property
    def selected_scopes(self) -> tuple[str, ...]:
        return tuple(
            scope
            for scope in RECORD_DELETE_SCOPES
            if self._scope_checkboxes[scope].isChecked()
        )

    def _update_delete_enabled(self) -> None:
        self._delete_button.setEnabled(bool(self.selected_scopes))
