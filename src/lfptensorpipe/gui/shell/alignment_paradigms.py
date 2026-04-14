"""Alignment paradigm selection MainWindow methods."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem, QMessageBox


class MainWindowAlignmentParadigmsMixin:
    def _shared_stage_trial_slug(self) -> str | None:
        slug = self._shared_stage_trial_slug_value
        if not isinstance(slug, str):
            return None
        normalized = slug.strip()
        return normalized or None

    def _set_shared_stage_trial_slug(self, slug: str | None) -> None:
        normalized = str(slug).strip() if isinstance(slug, str) else ""
        self._shared_stage_trial_slug_value = normalized or None

    def _alignment_row_for_slug(self, slug: str | None) -> int:
        if self._alignment_paradigm_list is None or not isinstance(slug, str):
            return -1
        for idx in range(self._alignment_paradigm_list.count()):
            item = self._alignment_paradigm_list.item(idx)
            if item is not None and item.data(Qt.UserRole) == slug:
                return idx
        return -1

    def _sync_alignment_paradigm_selection(self, slug: str | None) -> int:
        if self._alignment_paradigm_list is None:
            return -1
        row = self._alignment_row_for_slug(slug)
        self._alignment_paradigm_list.blockSignals(True)
        try:
            if row >= 0:
                self._alignment_paradigm_list.setCurrentRow(row)
            else:
                self._alignment_paradigm_list.setCurrentRow(-1)
        finally:
            self._alignment_paradigm_list.blockSignals(False)
        return row

    def _refresh_alignment_selected_paradigm(self) -> None:
        paradigm = self._current_alignment_paradigm()
        context = self._record_context()
        if paradigm is None:
            if self._alignment_method_description_label is not None:
                self._alignment_method_description_label.setText("")
            self._set_alignment_epoch_rows([])
            self._refresh_alignment_metric_combo()
            self._refresh_alignment_controls()
            return

        method_key = str(paradigm.get("method", "stack_warper"))
        method_idx = (
            self._alignment_method_combo.findData(method_key)
            if self._alignment_method_combo is not None
            else -1
        )
        if self._alignment_method_combo is not None:
            self._alignment_method_combo.blockSignals(True)
            if method_idx >= 0:
                self._alignment_method_combo.setCurrentIndex(method_idx)
            self._alignment_method_combo.blockSignals(False)
        self._update_alignment_method_description()

        if context is not None:
            rows = self._load_alignment_epoch_rows_runtime(
                context,
                paradigm_slug=paradigm["slug"],
            )
            self._set_alignment_epoch_rows(rows)
        else:
            self._set_alignment_epoch_rows([])
        self._refresh_alignment_metric_combo()
        self._refresh_alignment_controls()

    def _current_alignment_paradigm_slug(self) -> str | None:
        shared_slug = self._shared_stage_trial_slug()
        if isinstance(shared_slug, str) and shared_slug:
            return shared_slug
        if self._alignment_paradigm_list is None:
            return None
        item = self._alignment_paradigm_list.currentItem()
        if item is None:
            return None
        slug = item.data(Qt.UserRole)
        if not isinstance(slug, str):
            return None
        return slug

    def _current_alignment_paradigm(self) -> dict[str, Any] | None:
        slug = self._current_alignment_paradigm_slug()
        if slug is None:
            return None
        return next(
            (item for item in self._alignment_paradigms if item.get("slug") == slug),
            None,
        )

    def _reload_alignment_paradigms(self, preferred_slug: str | None = None) -> None:
        self._alignment_paradigms = self._load_alignment_paradigms_runtime(
            self._config_store,
            context=self._record_context(),
        )
        if self._alignment_paradigm_list is None:
            return
        current_slug = preferred_slug or self._shared_stage_trial_slug()
        self._alignment_paradigm_list.blockSignals(True)
        self._alignment_paradigm_list.clear()
        for item in self._alignment_paradigms:
            title = str(item.get("name", item.get("slug", "")))
            slug = str(item.get("slug", ""))
            list_item = QListWidgetItem(title)
            list_item.setData(Qt.UserRole, slug)
            self._alignment_paradigm_list.addItem(list_item)
        self._alignment_paradigm_list.blockSignals(False)

        if self._alignment_paradigm_list.count() == 0:
            self._set_shared_stage_trial_slug(None)
            self._set_alignment_epoch_rows([])
            self._refresh_alignment_metric_combo()
            self._refresh_alignment_controls()
            return

        target_idx = 0
        if current_slug:
            current_idx = self._alignment_row_for_slug(current_slug)
            if current_idx >= 0:
                target_idx = current_idx
        self._alignment_paradigm_list.setCurrentRow(target_idx)
        self._on_alignment_paradigm_selected(target_idx)

    def _on_alignment_paradigm_add(self) -> None:
        context = self._record_context()
        if context is None:
            self._show_warning(
                "Trial +",
                "Select project/subject/record first.",
            )
            return
        name, ok = self._prompt_text(
            "Trial +",
            "Trial name:",
        )
        if not ok:
            return
        created, message, entry = self._create_alignment_paradigm_runtime(
            self._config_store,
            name=name,
            context=context,
        )
        if not created:
            self._show_warning("Trial +", message)
            return
        preferred_slug = str(entry.get("slug")) if isinstance(entry, dict) else None
        self._reload_alignment_paradigms(preferred_slug=preferred_slug)
        self._reload_features_paradigms(preferred_slug=self._shared_stage_trial_slug())
        self.statusBar().showMessage(message)

    def _on_alignment_paradigm_delete(self) -> None:
        context = self._record_context()
        slug = self._current_alignment_paradigm_slug()
        paradigm = self._current_alignment_paradigm()
        if context is None or slug is None or paradigm is None:
            return
        title = str(paradigm.get("name", slug))
        confirm = self._ask_question(
            "Trial -",
            f"Delete trial '{title}'?",
        )
        if confirm != QMessageBox.Yes:
            return
        ok, message = self._delete_alignment_paradigm_runtime(
            self._config_store,
            slug=slug,
            context=context,
        )
        if not ok:
            self._show_warning("Trial -", message)
            return
        self._set_shared_stage_trial_slug(None)
        self._reload_alignment_paradigms()
        self._reload_features_paradigms(preferred_slug=self._shared_stage_trial_slug())
        self.statusBar().showMessage(message)

    def _on_alignment_paradigm_selected(self, row: int) -> None:
        if self._syncing_shared_trial_selection:
            return
        previous_slug = self._shared_stage_trial_slug()
        slug = None
        if self._alignment_paradigm_list is not None and row >= 0:
            item = self._alignment_paradigm_list.item(row)
            if item is not None:
                value = item.data(Qt.UserRole)
                if isinstance(value, str) and value.strip():
                    slug = value.strip()
        if (
            not self._record_param_syncing
            and previous_slug is not None
            and previous_slug != slug
        ):
            self._capture_features_trial_params(previous_slug)
        self._set_shared_stage_trial_slug(slug)
        self._syncing_shared_trial_selection = True
        try:
            self._sync_features_paradigm_selection(self._shared_stage_trial_slug())
        finally:
            self._syncing_shared_trial_selection = False
        self._refresh_alignment_selected_paradigm()
        if not self._record_param_syncing and previous_slug != slug:
            self._restore_features_trial_params(
                slug if self._features_trial_is_selectable(slug) else None,
                respect_dirty_keys=False,
            )
        self._refresh_stage_states_from_context()
        self._refresh_features_controls()
