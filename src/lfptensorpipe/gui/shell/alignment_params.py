"""Alignment method parameter MainWindow methods."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog


class MainWindowAlignmentParamsMixin:
    @staticmethod
    def _trial_method_params_candidate(
        paradigm: dict[str, object],
        *,
        method_key: str,
    ) -> dict[str, object]:
        raw_cache = paradigm.get("method_params_by_method", {})
        if isinstance(raw_cache, dict):
            cached = raw_cache.get(method_key)
            if isinstance(cached, dict):
                return dict(cached)
        raw_params = paradigm.get("method_params", {})
        if isinstance(raw_params, dict) and paradigm.get("method") == method_key:
            return dict(raw_params)
        return {}

    @staticmethod
    def _alignment_method_description_text(method_key: str) -> str:
        if method_key == "linear_warper":
            return "Aligns trials by anchor events and warps each trial to a shared 0-100% timeline."
        if method_key == "pad_warper":
            return "Builds each epoch by clipping around event start/end windows."
        if method_key == "concat_warper":
            return "Merges selected annotation segments into one continuous epoch, then resamples if needed."
        return "Keeps selected annotations as trials and resamples each trial to a common length."

    def _update_alignment_method_description(self) -> None:
        if self._alignment_method_description_label is None:
            return
        if self._alignment_method_combo is None:
            self._alignment_method_description_label.setText("")
            return
        method_key = self._alignment_method_combo.currentData()
        if not isinstance(method_key, str):
            self._alignment_method_description_label.setText("")
            return
        description = self._alignment_method_description_text(method_key)
        self._alignment_method_description_label.setText(description)
        self._alignment_method_combo.setToolTip(description)

    def _on_alignment_method_changed(self, index: int) -> None:
        _ = index
        slug = self._current_alignment_paradigm_slug()
        paradigm = self._current_alignment_paradigm()
        if slug is None or paradigm is None or self._alignment_method_combo is None:
            return
        method_key = self._alignment_method_combo.currentData()
        if not isinstance(method_key, str):
            return
        self._update_alignment_method_description()
        context = self._record_context()
        labels = (
            self._load_alignment_annotation_labels_runtime(context)
            if context is not None
            else []
        )
        params = self._trial_method_params_candidate(
            paradigm,
            method_key=method_key,
        )
        if not params:
            params = self._default_alignment_method_params_runtime(method_key)
        ok_norm, normalized_params, message_norm = (
            self._validate_alignment_method_params_runtime(
                method_key,
                params,
                annotation_labels=labels,
            )
        )
        if not ok_norm:
            self._show_warning("Align Epochs", message_norm)
            return
        ok, message = self._update_alignment_paradigm_runtime(
            self._config_store,
            slug=slug,
            method=method_key,
            method_params=normalized_params,
            context=context,
        )
        if not ok:
            self._show_warning("Align Epochs", message)
            return
        self._reload_alignment_paradigms(preferred_slug=slug)

    def _on_alignment_method_params(self) -> None:
        slug = self._current_alignment_paradigm_slug()
        paradigm = self._current_alignment_paradigm()
        if slug is None or paradigm is None:
            return
        if self._alignment_method_combo is None:
            return
        method_key = self._alignment_method_combo.currentData()
        if not isinstance(method_key, str):
            return
        context = self._record_context()
        labels = (
            self._load_alignment_annotation_labels_runtime(context)
            if context is not None
            else []
        )
        params = self._trial_method_params_candidate(
            paradigm,
            method_key=method_key,
        )
        if not params:
            params = self._default_alignment_method_params_runtime(method_key)
        dialog = self._create_alignment_method_params_dialog(
            method_key=method_key,
            session_params=params,
            annotation_labels=labels,
            config_store=self._config_store,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted or dialog.selected_params is None:
            return
        ok, message = self._update_alignment_paradigm_runtime(
            self._config_store,
            slug=slug,
            method=method_key,
            method_params=dialog.selected_params,
            context=context,
        )
        if not ok:
            self._show_warning("Align Epochs", message)
            return
        self._reload_alignment_paradigms(preferred_slug=slug)
        self._persist_record_params_snapshot(reason="alignment_method_params_save")
