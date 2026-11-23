"""Alignment run, preview, and finish MainWindow methods."""

from __future__ import annotations

from typing import Any


from lfptensorpipe.app import (
    PathResolver,
)
from lfptensorpipe.app.alignment.generation import (
    alignment_generation_rerun_message,
)


class MainWindowAlignmentRunMixin:
    @staticmethod
    def _alignment_merge_location_status_text(ready: bool) -> str:
        return (
            "Merge Location Info: Ready" if ready else "Merge Location Info: Not Ready"
        )

    def _refresh_alignment_panel_indicators(
        self,
        *,
        method_state: str,
        epoch_state: str,
    ) -> None:
        indicator_specs = (
            (
                self._alignment_method_indicator,
                method_state,
                "Method + Params state",
                "gray=not run for current config, yellow=stale or failed run, green=current config run succeeded.",
            ),
            (
                self._alignment_epoch_inspector_indicator,
                epoch_state,
                "Epoch Inspector state",
                "gray=not finished, yellow=stale/failed/mismatched picks, green=current picks finish succeeded.",
            ),
        )
        for indicator, state, prefix, detail in indicator_specs:
            if indicator is None:
                continue
            self._set_indicator_color(indicator, state)
            indicator.setToolTip(f"{prefix}: {state}. {detail}")

    def _refresh_alignment_merge_location_status(self, context: Any | None) -> bool:
        ready = False
        localize_state = "gray"
        if context is not None:
            try:
                localize_state = self._localize_indicator_state_runtime(
                    context.project_root,
                    context.subject,
                    context.record,
                )
            except Exception:
                localize_state = "gray"
            ready = localize_state == "green"
        if self._alignment_merge_location_status_label is None:
            return ready
        text = self._alignment_merge_location_status_text(ready)
        if ready and context is not None:
            from lfptensorpipe.app.localize.paths import (
                USER_LABELS,
                applied_source,
                localize_log_path,
            )
            from lfptensorpipe.app.runlog_store import read_run_log

            source = applied_source(
                read_run_log(
                    localize_log_path(
                        context.project_root, context.subject, context.record
                    )
                )
            )
            if source:
                text += " · " + ("User Labels" if source == USER_LABELS else "Lead-DBS")
        self._alignment_merge_location_status_label.setText(text)
        self._alignment_merge_location_status_label.setStyleSheet(
            "color: #1f7a1f;" if ready else "color: #666666;"
        )
        tooltip = (
            "Localize is green for the current record. Finish will attempt representative-coordinate merge."
            if ready
            else "Localize is not green for the current record. Finish will skip representative-coordinate merge."
        )
        if context is None:
            tooltip = "Select project, subject, and record. Finish will skip representative-coordinate merge until Localize is green."
        else:
            tooltip = f"{tooltip} Current Localize state: {localize_state}."
        self._alignment_merge_location_status_label.setToolTip(tooltip)
        return ready

    def _refresh_alignment_controls(self) -> None:
        context = self._record_context()
        tensor_ready = self._stage_states.get("tensor") == "green"
        has_context = context is not None
        has_paradigm = self._current_alignment_paradigm_slug() is not None
        paradigm = self._current_alignment_paradigm()
        editable = has_context and tensor_ready
        has_epoch_rows = bool(self._alignment_epoch_rows)
        method_state = "gray"
        epoch_state = "gray"
        if (
            has_context
            and has_paradigm
            and context is not None
            and paradigm is not None
        ):
            resolver = PathResolver(context)
            method_state = self._alignment_method_panel_state_runtime(
                resolver,
                paradigm=paradigm,
            )
            epoch_state = self._alignment_epoch_inspector_state_runtime(
                resolver,
                paradigm=paradigm,
                picked_epoch_indices=self._selected_alignment_epoch_indices(),
            )
        self._refresh_alignment_panel_indicators(
            method_state=method_state,
            epoch_state=epoch_state,
        )
        if (
            self._alignment_epoch_inspector_indicator is not None
            and context is not None
            and has_paradigm
        ):
            rerun_message = alignment_generation_rerun_message(
                PathResolver(context),
                trial_slug=str(self._current_alignment_paradigm_slug()),
                stage="finish",
            )
            if rerun_message is not None:
                self._alignment_epoch_inspector_indicator.setToolTip(
                    f"{self._alignment_epoch_inspector_indicator.toolTip()} "
                    f"{rerun_message}"
                )
        self._refresh_alignment_merge_location_status(context)

        if self._alignment_paradigm_add_button is not None:
            self._alignment_paradigm_add_button.setEnabled(has_context)
        if self._alignment_paradigm_delete_button is not None:
            self._alignment_paradigm_delete_button.setEnabled(editable and has_paradigm)
        if self._alignment_method_combo is not None:
            self._alignment_method_combo.setEnabled(editable and has_paradigm)
        if self._alignment_method_params_button is not None:
            self._alignment_method_params_button.setEnabled(editable and has_paradigm)
        if self._alignment_import_button is not None:
            self._alignment_import_button.setEnabled(has_context and has_paradigm)
        if self._alignment_export_button is not None:
            self._alignment_export_button.setEnabled(has_context and has_paradigm)
        if self._alignment_run_button is not None:
            self._alignment_run_button.setEnabled(editable and has_paradigm)
        if self._alignment_select_all_button is not None:
            self._alignment_select_all_button.setEnabled(has_epoch_rows)
        ready = has_epoch_rows and method_state == "green"
        self._refresh_epoch_review_controls(ready)
        if self._alignment_preview_button is not None:
            self._alignment_preview_button.setEnabled(ready)
        if self._alignment_finish_button is not None:
            self._alignment_finish_button.setEnabled(ready and bool(self._selected_alignment_epoch_indices()))

    def _on_alignment_run(self) -> None:
        context = self._record_context()
        if context is None:
            self.statusBar().showMessage(
                "Run Align Epochs unavailable: select project/subject/record."
            )
            return
        if self._stage_states.get("tensor") != "green":
            self._show_warning(
                "Run Align Epochs",
                "Build Tensor must be green before Align Epochs can run.",
            )
            return
        slug = self._current_alignment_paradigm_slug()
        paradigm = self._current_alignment_paradigm()
        if slug is None or paradigm is None:
            self._show_warning("Run Align Epochs", "Select one trial first.")
            return

        method_key = str(paradigm.get("method", "stack_warper"))
        if self._alignment_method_combo is not None:
            method_data = self._alignment_method_combo.currentData()
            if isinstance(method_data, str):
                method_key = method_data
        params = paradigm.get("method_params", {})
        if not isinstance(params, dict):
            params = self._default_alignment_method_params_runtime(method_key)
        else:
            params = dict(params)
        labels = self._load_alignment_annotation_labels_runtime(context)
        ok_params, normalized_params, message_params = (
            self._validate_alignment_method_params_runtime(
                method_key,
                params,
                annotation_labels=labels,
            )
        )
        if not ok_params:
            self._show_warning("Run Align Epochs", message_params)
            return
        if method_key != "linear_warper" and not normalized_params.get("annotations"):
            self._show_warning(
                "Run Align Epochs",
                "Select at least one annotation before running Align Epochs.",
            )
            return
        ok_update, message_update = self._update_alignment_paradigm_runtime(
            self._config_store,
            slug=slug,
            method=method_key,
            method_params=normalized_params,
            context=context,
        )
        if not ok_update:
            self._show_warning("Run Align Epochs", message_update)
            return
        self._reload_alignment_paradigms(preferred_slug=slug, submitted=True)

        ok, message, rows = self._run_with_busy(
            "Run Align Epochs",
            lambda: self._run_align_epochs_runtime(
                context,
                config_store=self._config_store,
                paradigm_slug=slug,
            ),
        )
        if ok:
            self._set_alignment_epoch_rows(rows)
        self._refresh_stage_states_from_context()
        self._refresh_alignment_controls()
        prefix = "Align Epochs OK" if ok else "Align Epochs failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        if not ok and message == "No completed tensor metrics available.":
            self._show_warning(
                "Run Align Epochs",
                "No usable Tensor results are available. Run or rerun Build Tensor, "
                "then try Align Epochs again.",
            )
        self._post_step_action_sync(reason="alignment_run")

    def _on_alignment_finish(self) -> None:
        context = self._record_context()
        slug = self._current_alignment_paradigm_slug()
        if context is None or slug is None:
            self.statusBar().showMessage(
                "Finish unavailable: select context and trial."
            )
            return
        resolver = PathResolver(context)
        paradigm = self._current_alignment_paradigm()
        if (
            self._alignment_method_panel_state_runtime(
                resolver,
                paradigm=paradigm,
            )
            != "green"
        ):
            self._show_warning(
                "Finish Align Epochs",
                "Run Align Epochs successfully before Finish.",
            )
            return
        picks = self._selected_alignment_epoch_indices()
        ok, message = self._run_with_busy(
            "Finish Align Epochs",
            lambda: self._finish_alignment_epochs_runtime(
                context,
                paradigm_slug=slug,
                picked_epoch_indices=picks,
            ),
        )
        self._refresh_stage_states_from_context()
        self._refresh_alignment_controls()
        prefix = "Finish OK" if ok else "Finish failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="alignment_finish")
