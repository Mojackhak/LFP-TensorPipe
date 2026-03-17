"""Dataset-context selection and state MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Path,
    QComboBox,
    QListWidgetItem,
    Qt,
    STAGE_SPECS,
    scan_stage_states,
)


class MainWindowDatasetContextSelectionMixin:
    def _autosave_outgoing_record_snapshot(
        self,
        *,
        next_project: Path | None,
        next_subject: str | None,
        next_record: str | None,
    ) -> None:
        context = self._record_context()
        if context is None:
            return
        normalized_subject = (
            str(next_subject).strip() if isinstance(next_subject, str) else None
        )
        normalized_record = (
            str(next_record).strip() if isinstance(next_record, str) else None
        )
        if (
            next_project == context.project_root
            and normalized_subject == context.subject
            and normalized_record == context.record
        ):
            return
        try:
            self._persist_record_params_snapshot(reason="record_context_switch")
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                f"Outgoing record autosave skipped: {exc}"
            )

    def _initialize_dataset_context(self) -> None:
        self._config_store.ensure_core_files()
        recent_projects = self._config_store.load_recent_projects()

        if not recent_projects and self._demo_data_root.exists():
            recent_projects = self._config_store.append_recent_project(
                self._demo_data_root
            )

        self._set_combo_values(self._project_combo, recent_projects, "Select project")
        if recent_projects:
            self._project_combo.setCurrentIndex(1)
        else:
            self._set_empty_record_context()
            self._refresh_localize_controls()
        self._refresh_dataset_action_state()

    def _set_combo_values(
        self, combo: QComboBox | None, values: list[str], placeholder: str
    ) -> None:
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(placeholder, None)
        for value in values:
            combo.addItem(value, value)
        combo.blockSignals(False)

    def _set_record_values(self, records: list[str]) -> None:
        if self._record_list is None:
            return
        self._record_list.blockSignals(True)
        self._record_list.clear()
        for record in records:
            item = QListWidgetItem(record)
            item.setData(Qt.UserRole, record)
            self._record_list.addItem(item)
        self._record_list.blockSignals(False)

    def _select_record_item(self, record: str) -> bool:
        if self._record_list is None:
            return False
        for idx in range(self._record_list.count()):
            item = self._record_list.item(idx)
            if item is None:
                continue
            if str(item.data(Qt.UserRole) or item.text()) == record:
                self._record_list.setCurrentRow(idx)
                return True
        return False

    def _selected_record_value(self) -> str | None:
        if self._record_list is None:
            return None
        item = self._record_list.currentItem()
        if item is None:
            return None
        raw = item.data(Qt.UserRole)
        value = str(raw if raw is not None else item.text()).strip()
        return value or None

    def _on_project_changed(self, index: int) -> None:
        if self._project_combo is None:
            return
        project_value = self._project_combo.itemData(index)
        next_project = Path(str(project_value)) if project_value else None
        self._autosave_outgoing_record_snapshot(
            next_project=next_project,
            next_subject=None,
            next_record=None,
        )
        if not project_value:
            self._current_project = None
            self._current_subject = None
            self._set_combo_values(self._subject_combo, [], "Select subject")
            self._set_record_values([])
            self._set_empty_record_context()
            self._refresh_localize_controls()
            self._refresh_dataset_action_state()
            return

        project_root = Path(str(project_value))
        if not project_root.exists():
            self.statusBar().showMessage(f"Missing project path: {project_root}")
            self._current_project = None
            self._current_subject = None
            self._set_empty_record_context()
            self._refresh_localize_controls()
            self._refresh_dataset_action_state()
            return

        self._current_project = project_root
        self._current_subject = None
        self._config_store.append_recent_project(project_root)
        subjects = self._discover_subjects_runtime(project_root)
        self._set_combo_values(self._subject_combo, subjects, "Select subject")
        self._set_record_values([])
        if subjects and self._subject_combo is not None:
            self._subject_combo.setCurrentIndex(1)
        else:
            self.statusBar().showMessage(f"Project selected: {project_root}")
            self._set_empty_record_context()
            self._refresh_localize_controls()
            self._refresh_dataset_action_state()

    def _on_subject_changed(self, index: int) -> None:
        if self._subject_combo is None or self._current_project is None:
            return
        subject = self._subject_combo.itemData(index)
        next_subject = str(subject).strip() if subject else None
        self._autosave_outgoing_record_snapshot(
            next_project=self._current_project,
            next_subject=next_subject,
            next_record=None,
        )
        if not subject:
            self._current_subject = None
            self._set_record_values([])
            self._set_empty_record_context()
            self._refresh_localize_controls()
            self._refresh_dataset_action_state()
            return

        self._current_subject = str(subject)
        self._refresh_localize_controls()
        records = self._discover_records_runtime(
            self._current_project,
            self._current_subject,
        )
        self._set_record_values(records)
        self.statusBar().showMessage(
            f"Project: {self._current_project} | Subject: {self._current_subject}"
        )
        self._set_empty_record_context()
        self._refresh_dataset_action_state()

    def _on_record_changed(self) -> None:
        if (
            self._record_list is None
            or self._current_project is None
            or self._current_subject is None
        ):
            return
        record = self._selected_record_value()
        self._autosave_outgoing_record_snapshot(
            next_project=self._current_project,
            next_subject=self._current_subject,
            next_record=record,
        )
        if not record:
            self._current_record = None
            self._reset_annotations_table()
            self._set_empty_record_context()
            self._refresh_localize_controls()
            self._refresh_dataset_action_state()
            return

        self._current_record = str(record)
        self._shared_stage_trial_slug_value = None
        self._features_trial_params_by_slug = {}
        self._preproc_viz_last_step = None
        self._reset_annotations_table()
        migration_summary = self._upgrade_record_run_logs_runtime(
            self._current_project,
            self._current_subject,
            self._current_record,
        )
        stage_states = scan_stage_states(
            self._current_project, self._current_subject, self._current_record
        )
        self._set_stage_state_maps(stage_states)
        self._refresh_stage_controls()
        status_message = (
            "Context: "
            f"{self._current_project} | {self._current_subject} | {self._current_record}"
        )
        if getattr(migration_summary, "upgraded_count", 0):
            status_message += (
                f" | Upgraded {int(migration_summary.upgraded_count)} log(s)"
            )
        if getattr(migration_summary, "failed_count", 0):
            status_message += (
                f" | Failed to upgrade {int(migration_summary.failed_count)} log(s)"
            )
        self.statusBar().showMessage(status_message)
        self._refresh_localize_controls()
        self._refresh_dataset_action_state()
        self._refresh_preproc_controls()
        self._set_tensor_frequency_defaults_from_context(self._record_context())
        self._refresh_tensor_controls()
        self._reload_alignment_paradigms()
        self._refresh_alignment_controls()
        self._reload_features_paradigms()
        self._refresh_features_controls()
        self._sync_record_params_from_logs(include_master=True, clear_dirty=True)

    def _set_empty_record_context(self) -> None:
        self._current_record = None
        self._shared_stage_trial_slug_value = None
        self._features_trial_params_by_slug = {}
        self._preproc_viz_last_step = None
        self._record_param_dirty_keys.clear()
        self._localize_match_payload = None
        self._reset_annotations_table()
        empty_stage_states = {spec.key: "gray" for spec in STAGE_SPECS}
        self._stage_raw_states = dict(empty_stage_states)
        self._stage_states = dict(empty_stage_states)
        self._refresh_stage_controls()
        self._refresh_localize_controls()
        self._refresh_dataset_action_state()
        self._refresh_preproc_controls()
        self._set_tensor_frequency_defaults_from_context(None)
        self._refresh_tensor_controls()
        self._reload_alignment_paradigms()
        self._refresh_alignment_controls()
        self._reload_features_paradigms()
        self._refresh_features_controls()

    def _refresh_dataset_action_state(self) -> None:
        has_project = self._current_project is not None
        has_subject = has_project and self._current_subject is not None
        has_record = has_subject and self._current_record is not None

        if self._project_add_button is not None:
            self._project_add_button.setEnabled(True)
        if self._subject_add_button is not None:
            self._subject_add_button.setEnabled(has_project)
        if self._record_add_button is not None:
            self._record_add_button.setEnabled(has_subject)
        if self._record_delete_button is not None:
            self._record_delete_button.setEnabled(has_record)
