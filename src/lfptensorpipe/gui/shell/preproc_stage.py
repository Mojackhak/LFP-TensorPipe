"""Preprocess stage assembly and gating MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app.preproc.lineage import (
    preproc_step_is_skipped,
    preproc_step_lineage_is_current,
)
from lfptensorpipe.gui.shell.common import (
    Any,
    PathResolver,
    QGroupBox,
    QLabel,
    QWidget,
    _stage_preproc_panel,
    normalize_ecg_method_params,
    normalize_filter_advance_params,
    preproc_step_indicator_state,
    preproc_step_raw_path,
    rawdata_input_fif_path,
    resolve_finish_source,
    set_control_validation_error,
)


class MainWindowPreprocStageMixin:
    @staticmethod
    def _preproc_step_display_name(step: str) -> str:
        mapping = {
            "raw": "Raw",
            "filter": "Filter",
            "annotations": "Annotations",
            "ecg_artifact_removal": "ECG Artifact Removal",
            "finish": "Finish",
        }
        return mapping.get(step, step)

    def _build_preproc_raw_block(self) -> QGroupBox:
        return _stage_preproc_panel._build_preproc_raw_block(self)

    def _build_preproc_filter_block(self) -> QGroupBox:
        return _stage_preproc_panel._build_preproc_filter_block(self)

    def _build_preproc_annotations_block(self) -> QGroupBox:
        return _stage_preproc_panel._build_preproc_annotations_block(self)

    def _set_annotations_editable(self, editable: bool) -> None:
        _stage_preproc_panel._set_annotations_editable(self, editable)

    def _annotations_table_rows(self) -> tuple[list[dict[str, Any]], list[int]]:
        return _stage_preproc_panel._annotations_table_rows(self)

    def _highlight_annotation_rows(self, invalid_rows: list[int]) -> None:
        _stage_preproc_panel._highlight_annotation_rows(self, invalid_rows)

    def _append_annotation_rows(self, rows: list[dict[str, Any]]) -> None:
        _stage_preproc_panel._append_annotation_rows(self, rows)

    def _reset_annotations_table(self) -> None:
        _stage_preproc_panel._reset_annotations_table(self)

    def _build_preproc_finish_block(self) -> QGroupBox:
        return _stage_preproc_panel._build_preproc_finish_block(self)

    def _build_preproc_ecg_block(self) -> QGroupBox:
        return _stage_preproc_panel._build_preproc_ecg_block(self)

    def _build_preproc_visualization_block(self) -> QGroupBox:
        return _stage_preproc_panel._build_preproc_visualization_block(self)

    def _register_preproc_indicator(
        self, step: str, indicator: QLabel | None = None
    ) -> QLabel:
        if indicator is None:
            indicator = self._make_indicator_label("gray")
        else:
            self._set_indicator_color(indicator, "gray")
        indicator.setToolTip(
            f"{self._preproc_step_display_name(step)} state: "
            "gray=not run, yellow=stale, failed, or blocked, "
            "green=current inputs match successful output. Current: gray."
        )
        self._preproc_step_indicators[step] = indicator
        return indicator

    def _build_preproc_status_row(self, step: str) -> QWidget:
        return _stage_preproc_panel._build_preproc_status_row(self, step)

    def _set_preproc_step_indicator(self, step: str, state: str) -> None:
        indicator = self._preproc_step_indicators.get(step)
        if indicator is not None:
            self._set_indicator_color(indicator, state)
            indicator.setToolTip(
                f"{self._preproc_step_display_name(step)} state: "
                "gray=not run, yellow=stale, failed, or blocked, "
                f"green=current inputs match successful output. Current: {state}."
            )

    def _refresh_preproc_controls(self) -> None:
        context = self._record_context()
        if context is None:
            for step, indicator in self._preproc_step_indicators.items():
                self._set_indicator_color(indicator, "gray")
                indicator.setToolTip(
                    f"{self._preproc_step_display_name(step)} state: "
                    "gray=not run, yellow=stale, failed, or blocked, "
                    "green=current inputs match successful output. Current: gray."
                )
            if self._preproc_raw_plot_button is not None:
                self._preproc_raw_plot_button.setEnabled(False)
            if self._preproc_filter_advance_button is not None:
                self._preproc_filter_advance_button.setEnabled(False)
            if self._preproc_filter_apply_button is not None:
                self._preproc_filter_apply_button.setEnabled(False)
            if self._preproc_filter_plot_button is not None:
                self._preproc_filter_plot_button.setEnabled(False)
            if self._preproc_filter_skip_button is not None:
                self._preproc_filter_skip_button.setChecked(False)
                self._preproc_filter_skip_button.setEnabled(False)
            if self._preproc_filter_notches_edit is not None:
                self._preproc_filter_notches_edit.setEnabled(False)
            if self._preproc_filter_low_freq_edit is not None:
                self._preproc_filter_low_freq_edit.setEnabled(False)
            if self._preproc_filter_high_freq_edit is not None:
                self._preproc_filter_high_freq_edit.setEnabled(False)
            if self._preproc_annotations_edit_button is not None:
                self._preproc_annotations_edit_button.setEnabled(False)
            if self._preproc_annotations_advance_button is not None:
                self._preproc_annotations_advance_button.setEnabled(False)
            if self._preproc_annotations_save_button is not None:
                self._preproc_annotations_save_button.setEnabled(False)
            if self._preproc_annotations_import_button is not None:
                self._preproc_annotations_import_button.setEnabled(False)
            if self._preproc_annotations_plot_button is not None:
                self._preproc_annotations_plot_button.setEnabled(False)
            if self._preproc_annotations_skip_button is not None:
                self._preproc_annotations_skip_button.setChecked(False)
                self._preproc_annotations_skip_button.setEnabled(False)
            if self._preproc_ecg_advance_button is not None:
                self._preproc_ecg_advance_button.setEnabled(False)
            if self._preproc_ecg_apply_button is not None:
                self._preproc_ecg_apply_button.setEnabled(False)
            if self._preproc_ecg_plot_button is not None:
                self._preproc_ecg_plot_button.setEnabled(False)
            if self._preproc_ecg_skip_button is not None:
                self._preproc_ecg_skip_button.setChecked(False)
                self._preproc_ecg_skip_button.setEnabled(False)
            if self._preproc_ecg_method_combo is not None:
                self._preproc_ecg_method_combo.setEnabled(False)
            if self._preproc_finish_apply_button is not None:
                self._preproc_finish_apply_button.setEnabled(False)
            if self._preproc_finish_plot_button is not None:
                self._preproc_finish_plot_button.setEnabled(False)
            self._refresh_preproc_ecg_channel_state(None)
            self._refresh_preproc_visualization_controls(None)
            for control in (
                self._preproc_filter_notches_edit,
                self._preproc_filter_low_freq_edit,
                self._preproc_filter_high_freq_edit,
                self._preproc_filter_advance_button,
                self._preproc_ecg_advance_button,
                self._preproc_ecg_channels_button,
            ):
                set_control_validation_error(control, None)
            return

        valid_advance, normalized_advance, advance_message = (
            normalize_filter_advance_params(self._preproc_filter_advance_params)
        )
        self._sync_filter_model_notches(
            normalized_advance["notch_model"] if valid_advance else None
        )
        raw_input_exists = rawdata_input_fif_path(context).exists()
        resolver = PathResolver(context)
        raw_log_state = preproc_step_indicator_state(resolver, "raw")
        finish_log_state = preproc_step_indicator_state(resolver, "finish")
        filter_skipped = preproc_step_is_skipped(resolver, "filter")
        ecg_skipped = preproc_step_is_skipped(resolver, "ecg_artifact_removal")
        annotations_skipped = preproc_step_is_skipped(resolver, "annotations")
        self._set_preproc_step_indicator("raw", raw_log_state)
        filter_notches = (
            self._filter_manual_notches_text()
            if self._preproc_filter_notches_edit is not None
            else None
        )
        filter_low_freq = (
            self._preproc_filter_low_freq_edit.text()
            if self._preproc_filter_low_freq_edit is not None
            else None
        )
        filter_high_freq = (
            self._preproc_filter_high_freq_edit.text()
            if self._preproc_filter_high_freq_edit is not None
            else None
        )
        filter_panel_state = self._preproc_filter_panel_state_runtime(
            resolver,
            notches=filter_notches,
            l_freq=filter_low_freq,
            h_freq=filter_high_freq,
            advance_params=self._preproc_filter_advance_params,
        )
        filter_review_required = self._preproc_filter_review_required_runtime(resolver)
        filter_preview_exists = self._preproc_filter_preview_raw_path_runtime(
            resolver
        ).exists()
        filter_lineage_available = (
            not filter_skipped
            and preproc_step_lineage_is_current(
                resolver,
                "filter",
            )
        )
        if not filter_lineage_available:
            self._preproc_annotations_mark_filter_edges = False
        annotations_mark_filter_edges = self._preproc_annotations_mark_filter_edges
        if (
            not filter_lineage_available
            and self._preproc_ecg_review_params.get("mark_filter_edges") is True
        ):
            self._preproc_ecg_review_params = {"mark_filter_edges": False}
        annotation_rows, _ = self._annotations_table_rows()
        annotations_panel_state = self._preproc_annotations_panel_state_runtime(
            resolver,
            rows=annotation_rows,
            mark_filter_edges=annotations_mark_filter_edges,
        )
        ecg_method = (
            self._preproc_ecg_method_combo.currentData()
            if self._preproc_ecg_method_combo is not None
            else None
        )
        ecg_panel_state = self._preproc_ecg_panel_state_runtime(
            resolver,
            method=ecg_method,
            picks=list(self._preproc_ecg_selected_channels),
            method_kwargs=self._preproc_ecg_params_by_method.get(
                str(ecg_method),
            ),
            mark_filter_edges=getattr(self, "_preproc_ecg_review_params", {}).get(
                "mark_filter_edges",
                False,
            ),
        )
        base_states = {
            "raw": raw_log_state,
            "filter": filter_panel_state,
            "ecg_artifact_removal": ecg_panel_state,
            "annotations": annotations_panel_state,
            "finish": finish_log_state,
        }
        skipped_steps = {
            "filter": filter_skipped,
            "ecg_artifact_removal": ecg_skipped,
            "annotations": annotations_skipped,
        }
        blocked = False
        for step in (
            "raw",
            "filter",
            "ecg_artifact_removal",
            "annotations",
            "finish",
        ):
            base_state = base_states[step]
            self._set_preproc_step_indicator(
                step,
                "yellow" if blocked else base_state,
            )
            if base_state == "yellow" and not skipped_steps.get(step, False):
                blocked = True
        filter_raw_exists = preproc_step_raw_path(resolver, "filter").exists()
        annotations_raw_exists = preproc_step_raw_path(resolver, "annotations").exists()
        ecg_raw_exists = preproc_step_raw_path(
            resolver, "ecg_artifact_removal"
        ).exists()
        finish_source_exists = resolve_finish_source(context) is not None
        finish_raw_path = resolver.preproc_root / "finish" / "raw.fif"
        finish_raw_exists = finish_raw_path.exists()
        raw_step_exists = preproc_step_raw_path(resolver, "raw").exists()
        raw_ready = raw_log_state == "green" and raw_step_exists
        filter_route_ready = raw_ready and (
            filter_skipped or filter_panel_state != "yellow"
        )
        ecg_route_ready = filter_route_ready and (
            ecg_skipped or ecg_panel_state != "yellow"
        )
        annotations_route_ready = ecg_route_ready and (
            annotations_skipped or annotations_panel_state != "yellow"
        )

        for button, skipped in (
            (self._preproc_filter_skip_button, filter_skipped),
            (self._preproc_ecg_skip_button, ecg_skipped),
            (self._preproc_annotations_skip_button, annotations_skipped),
        ):
            if button is not None:
                button.setChecked(skipped)

        if self._preproc_raw_plot_button is not None:
            self._preproc_raw_plot_button.setEnabled(
                raw_input_exists or raw_step_exists
            )
        if self._preproc_filter_advance_button is not None:
            self._preproc_filter_advance_button.setEnabled(raw_ready)
        if self._preproc_filter_apply_button is not None:
            self._preproc_filter_apply_button.setEnabled(raw_ready)
        if self._preproc_filter_plot_button is not None:
            self._preproc_filter_plot_button.setEnabled(
                (filter_review_required and filter_preview_exists)
                or (filter_panel_state == "green" and filter_raw_exists)
            )
        if self._preproc_filter_skip_button is not None:
            self._preproc_filter_skip_button.setEnabled(
                filter_skipped or (raw_ready and filter_panel_state != "gray")
            )
        if self._preproc_filter_notches_edit is not None:
            self._preproc_filter_notches_edit.setEnabled(raw_ready)
        if self._preproc_filter_low_freq_edit is not None:
            self._preproc_filter_low_freq_edit.setEnabled(raw_ready)
        if self._preproc_filter_high_freq_edit is not None:
            self._preproc_filter_high_freq_edit.setEnabled(raw_ready)
        if raw_ready:
            self._refresh_preproc_filter_basic_validation()
        else:
            for control in (
                self._preproc_filter_notches_edit,
                self._preproc_filter_low_freq_edit,
                self._preproc_filter_high_freq_edit,
            ):
                set_control_validation_error(control, None)
        set_control_validation_error(
            self._preproc_filter_advance_button,
            advance_message if raw_ready and not valid_advance else None,
        )
        if self._preproc_annotations_edit_button is not None:
            self._preproc_annotations_edit_button.setEnabled(ecg_route_ready)
        if self._preproc_annotations_advance_button is not None:
            self._preproc_annotations_advance_button.setEnabled(ecg_route_ready)
        if self._preproc_annotations_save_button is not None:
            self._preproc_annotations_save_button.setEnabled(ecg_route_ready)
        if self._preproc_annotations_import_button is not None:
            self._preproc_annotations_import_button.setEnabled(ecg_route_ready)
        if self._preproc_annotations_plot_button is not None:
            self._preproc_annotations_plot_button.setEnabled(
                ecg_route_ready
                and annotations_panel_state == "green"
                and annotations_raw_exists
            )
        if self._preproc_annotations_skip_button is not None:
            self._preproc_annotations_skip_button.setEnabled(
                annotations_skipped
                or (ecg_route_ready and annotations_panel_state != "gray")
            )
        if self._preproc_ecg_advance_button is not None:
            self._preproc_ecg_advance_button.setEnabled(filter_route_ready)
        valid_ecg, _, ecg_message = normalize_ecg_method_params(
            str(ecg_method),
            self._preproc_ecg_params_by_method.get(str(ecg_method)),
        )
        ecg_editable = filter_route_ready
        set_control_validation_error(
            self._preproc_ecg_advance_button,
            ecg_message if ecg_editable and not valid_ecg else None,
        )
        set_control_validation_error(
            self._preproc_ecg_channels_button,
            (
                "Select at least one ECG channel."
                if ecg_editable
                and self._preproc_ecg_available_channels
                and not self._preproc_ecg_selected_channels
                else None
            ),
        )
        if self._preproc_ecg_apply_button is not None:
            self._preproc_ecg_apply_button.setEnabled(filter_route_ready)
        if self._preproc_ecg_plot_button is not None:
            self._preproc_ecg_plot_button.setEnabled(
                filter_route_ready and ecg_panel_state == "green" and ecg_raw_exists
            )
        if self._preproc_ecg_skip_button is not None:
            self._preproc_ecg_skip_button.setEnabled(
                ecg_skipped or (filter_route_ready and ecg_panel_state != "gray")
            )
        if self._preproc_ecg_method_combo is not None:
            self._preproc_ecg_method_combo.setEnabled(filter_route_ready)
        if self._preproc_finish_apply_button is not None:
            self._preproc_finish_apply_button.setEnabled(
                annotations_route_ready and finish_source_exists
            )
        if self._preproc_finish_plot_button is not None:
            self._preproc_finish_plot_button.setEnabled(
                annotations_route_ready
                and finish_log_state == "green"
                and finish_raw_exists
            )
        self._refresh_preproc_ecg_channel_state(context)
        self._refresh_preproc_visualization_controls(context)
