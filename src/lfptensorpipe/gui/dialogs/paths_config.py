"""Paths configuration dialog."""
# ruff: noqa: F403,F405

from __future__ import annotations

import sys

from .common import *  # noqa: F403


class PathsConfigDialog(QDialog):
    """Settings dialog for localize runtime path config."""

    def __init__(
        self,
        *,
        current_paths: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configs")
        self.setModal(True)
        self.resize(700, 180)
        self._selected_paths: dict[str, str] | None = None
        self._path_edits: dict[str, QLineEdit] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        for key, label in LOCALIZE_PATH_FIELD_LABELS.items():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            edit = QLineEdit()
            edit.setText(str(current_paths.get(key, "")).strip())
            browse_button = QPushButton("Browse...")
            browse_button.clicked.connect(
                lambda checked=False, field_key=key: self._on_browse_directory(
                    field_key
                )
            )
            field_tooltip = LOCALIZE_PATH_FIELD_TOOLTIPS.get(key, "")
            if field_tooltip:
                edit.setToolTip(field_tooltip)
            browse_button.setToolTip(f"Browse for {label}.")
            row_layout.addWidget(edit, stretch=1)
            row_layout.addWidget(browse_button, stretch=0)
            form.addRow(label, row)
            self._path_edits[key] = edit

        root.addLayout(form)

        footer = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        footer.accepted.connect(self._on_save)
        footer.rejected.connect(self.reject)
        save_button = footer.button(QDialogButtonBox.Save)
        cancel_button = footer.button(QDialogButtonBox.Cancel)
        if save_button is not None:
            save_button.setToolTip("Validate and save path settings to app storage.")
        if cancel_button is not None:
            cancel_button.setToolTip("Close without saving path changes.")
        root.addWidget(footer)

    @property
    def selected_paths(self) -> dict[str, str] | None:
        return self._selected_paths

    def _show_warning(self, title: str, message: str) -> int:
        return QMessageBox.warning(self, title, message)

    def _select_existing_directory(self, title: str, start_dir: str) -> str:
        return QFileDialog.getExistingDirectory(self, title, start_dir)

    def _use_matlab_bundle_browser(self, field_key: str) -> bool:
        return field_key == "matlab_root" and sys.platform == "darwin"

    def _select_matlab_installation_path(self, title: str, start_dir: str) -> str:
        browse_root = start_dir or "/Applications"
        dialog = QFileDialog(self, title, browse_root)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        if dialog.exec() != QDialog.Accepted:
            return ""
        selected_files = dialog.selectedFiles()
        return selected_files[0] if selected_files else ""

    def _browse_start_dir(self, field_key: str, current_text: str) -> str:
        if not current_text:
            return "/Applications" if self._use_matlab_bundle_browser(field_key) else ""
        start_path = Path(current_text).expanduser()
        if (
            self._use_matlab_bundle_browser(field_key)
            and start_path.suffix.lower() == ".app"
            and start_path.parent.exists()
        ):
            return str(start_path.parent)
        if start_path.exists() and start_path.is_dir():
            return str(start_path)
        if start_path.parent.exists():
            return str(start_path.parent)
        return "/Applications" if self._use_matlab_bundle_browser(field_key) else ""

    def _browse_directory_for_field(
        self,
        field_key: str,
        title: str,
        start_dir: str,
    ) -> str:
        if self._use_matlab_bundle_browser(field_key):
            return self._select_matlab_installation_path(title, start_dir)
        return self._select_existing_directory(title, start_dir)

    def _on_browse_directory(self, field_key: str) -> None:
        edit = self._path_edits.get(field_key)
        if edit is None:
            return
        current_text = edit.text().strip()
        title = f"Select {LOCALIZE_PATH_FIELD_LABELS.get(field_key, field_key)}"
        start_dir = self._browse_start_dir(field_key, current_text)
        selected = self._browse_directory_for_field(field_key, title, start_dir)
        if not selected:
            return
        edit.setText(str(Path(selected).expanduser()))

    def _collect_validated_paths(self) -> dict[str, str]:
        errors: list[str] = []
        normalized: dict[str, str] = {}
        for key, label in LOCALIZE_PATH_FIELD_LABELS.items():
            edit = self._path_edits.get(key)
            raw_value = edit.text().strip() if edit is not None else ""
            if not raw_value:
                errors.append(f"{label} is required.")
                continue
            resolved = Path(raw_value).expanduser()
            if not resolved.exists():
                errors.append(f"{label} does not exist: {resolved}")
                continue
            if not resolved.is_dir():
                errors.append(f"{label} must be an existing directory: {resolved}")
                continue
            normalized[key] = str(resolved.resolve())
        if errors:
            raise ValueError("\n".join(errors))
        return normalized

    def _on_save(self) -> None:
        try:
            validated = self._collect_validated_paths()
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Configs", f"Invalid paths:\n{exc}")
            return
        self._selected_paths = validated
        self.accept()
