"""Dataset-context create/delete MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Path,
    QDialog,
    QMessageBox,
    validate_record_name,
)


class MainWindowDatasetContextActionsMixin:
    def _on_project_add(self) -> None:
        start_dir = str(self._current_project or self._demo_data_root)
        selected = self._select_existing_directory(
            "Select project folder",
            start_dir,
        )
        if not selected:
            return
        project_root = Path(selected).expanduser().resolve()
        if not project_root.exists():
            self._show_warning(
                "Project +",
                f"Project path does not exist:\n{project_root}",
            )
            return

        recent_projects = self._config_store.append_recent_project(project_root)
        self._set_combo_values(self._project_combo, recent_projects, "Select project")
        if self._project_combo is not None:
            index = self._project_combo.findData(str(project_root))
            if index >= 0:
                self._project_combo.setCurrentIndex(index)
        self.statusBar().showMessage(f"Project added: {project_root}")

    def _on_subject_add(self) -> None:
        if self._current_project is None:
            self._show_warning("Subject +", "Select a project first.")
            return
        subject, ok = self._prompt_text(
            "Subject +",
            "Subject name (sub-xxx):",
        )
        if not ok:
            return
        normalized = subject.strip()
        created, message = self._create_subject_runtime(
            self._current_project,
            normalized,
        )
        if not created:
            self._show_warning("Subject +", message)
            return

        subjects = self._discover_subjects_runtime(self._current_project)
        self._set_combo_values(self._subject_combo, subjects, "Select subject")
        if self._subject_combo is not None:
            index = self._subject_combo.findData(normalized)
            if index >= 0:
                self._subject_combo.setCurrentIndex(index)
        self.statusBar().showMessage(message)

    def _on_record_add(self) -> None:
        if self._current_project is None or self._current_subject is None:
            self._show_warning("Record +", "Select project and subject first.")
            return

        existing_records = tuple(
            self._discover_records_runtime(
                self._current_project,
                self._current_subject,
            )
        )
        dialog = self._create_record_import_dialog(
            project_root=self._current_project,
            existing_records=existing_records,
            default_import_type=self._load_record_import_last_type(),
            config_store=self._config_store,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        preview = dialog.parsed_preview
        if preview is None:
            self._show_warning("Record +", "Parse result is missing.")
            return

        ok, normalized_record = validate_record_name(dialog.selected_record_name)
        if not ok:
            self._show_warning("Record +", normalized_record)
            return

        raw_to_import = preview.raw
        if dialog.use_reset_reference and dialog.reset_rows:
            rows = tuple(
                (row.anode, row.cathode, row.name) for row in dialog.reset_rows
            )
            try:
                raw_to_import = self._apply_reset_reference_runtime(raw_to_import, rows)
            except Exception as exc:  # noqa: BLE001
                self._show_warning(
                    "Record +",
                    f"Failed to apply reset reference:\n{exc}",
                )
                return

        result = self._run_with_busy(
            "Record Import",
            lambda: self._import_record_from_raw_runtime(
                project_root=self._current_project,
                subject=self._current_subject,
                record=normalized_record,
                raw=raw_to_import,
                source_path=preview.source_path,
                is_fif_input=preview.is_fif_input,
                read_only_project_root=self._demo_data_source_readonly,
            ),
        )
        if not result.ok:
            self._show_warning("Record +", result.message)
            return

        self._save_record_import_last_type(dialog.selected_import_type)
        records = self._discover_records_runtime(
            self._current_project,
            self._current_subject,
        )
        self._set_record_values(records)
        if self._select_record_item(normalized_record):
            self.statusBar().showMessage(result.message)
            return
        self._set_empty_record_context()
        self.statusBar().showMessage(f"{result.message} 请手动选择 record。")

    def _on_record_delete(self) -> None:
        if (
            self._current_project is None
            or self._current_subject is None
            or self._current_record is None
        ):
            self._show_warning(
                "Record -",
                "Select project, subject, and record first.",
            )
            return

        confirm = self._ask_question(
            "Record -",
            f"Delete all artifacts for record '{self._current_record}'?",
        )
        if confirm != QMessageBox.Yes:
            return

        result = self._run_with_busy(
            "Record Delete",
            lambda: self._delete_record_runtime(
                project_root=self._current_project,
                subject=self._current_subject,
                record=self._current_record,
                read_only_project_root=self._demo_data_source_readonly,
            ),
        )
        if not result.ok:
            self._show_warning("Record -", result.message)
            return

        records = self._discover_records_runtime(
            self._current_project,
            self._current_subject,
        )
        self._set_record_values(records)
        self._set_empty_record_context()
        self.statusBar().showMessage(result.message)
