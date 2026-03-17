"""Preprocess stage assembly and gating MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    PathResolver,
    QGroupBox,
    QLabel,
    QWidget,
    _stage_preproc_panel,
    indicator_from_log,
    preproc_step_log_path,
    preproc_step_raw_path,
    rawdata_input_fif_path,
    resolve_finish_source,
)


class MainWindowPreprocStageMixin:
    @staticmethod
    def _preproc_step_display_name(step: str) -> str:
        mapping = {
            "raw": "Raw",
            "filter": "Filter",
            "annotations": "Annotations",
            "bad_segment_removal": "Bad Segment Removal",
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

    def _build_preproc_bad_segment_block(self) -> QGroupBox:
        return _stage_preproc_panel._build_preproc_bad_segment_block(self)

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
            "gray=not run, yellow=stale or failed, "
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
                "gray=not run, yellow=stale or failed, "
                f"green=current inputs match successful output. Current: {state}."
            )

    def _refresh_preproc_controls(self) -> None:
        context = self._record_context()
        if context is None:
            for step, indicator in self._preproc_step_indicators.items():
                self._set_indicator_color(indicator, "gray")
                indicator.setToolTip(
                    f"{self._preproc_step_display_name(step)} state: "
                    "gray=not run, yellow=stale or failed, "
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
            if self._preproc_filter_notches_edit is not None:
                self._preproc_filter_notches_edit.setEnabled(False)
            if self._preproc_filter_low_freq_edit is not None:
                self._preproc_filter_low_freq_edit.setEnabled(False)
            if self._preproc_filter_high_freq_edit is not None:
                self._preproc_filter_high_freq_edit.setEnabled(False)
            if self._preproc_annotations_edit_button is not None:
                self._preproc_annotations_edit_button.setEnabled(False)
            if self._preproc_annotations_save_button is not None:
                self._preproc_annotations_save_button.setEnabled(False)
            if self._preproc_annotations_import_button is not None:
                self._preproc_annotations_import_button.setEnabled(False)
            if self._preproc_annotations_plot_button is not None:
                self._preproc_annotations_plot_button.setEnabled(False)
            if self._preproc_bad_segment_apply_button is not None:
                self._preproc_bad_segment_apply_button.setEnabled(False)
            if self._preproc_bad_segment_plot_button is not None:
                self._preproc_bad_segment_plot_button.setEnabled(False)
            if self._preproc_ecg_apply_button is not None:
                self._preproc_ecg_apply_button.setEnabled(False)
            if self._preproc_ecg_plot_button is not None:
                self._preproc_ecg_plot_button.setEnabled(False)
            if self._preproc_ecg_method_combo is not None:
                self._preproc_ecg_method_combo.setEnabled(False)
            if self._preproc_finish_apply_button is not None:
                self._preproc_finish_apply_button.setEnabled(False)
            if self._preproc_finish_plot_button is not None:
                self._preproc_finish_plot_button.setEnabled(False)
            self._refresh_preproc_ecg_channel_state(None)
            self._refresh_preproc_visualization_controls(None)
            return

        raw_input_exists = rawdata_input_fif_path(context).exists()
        resolver = PathResolver(context)
        raw_log_state = indicator_from_log(preproc_step_log_path(resolver, "raw"))
        filter_log_state = indicator_from_log(preproc_step_log_path(resolver, "filter"))
        annotations_log_state = indicator_from_log(
            preproc_step_log_path(resolver, "annotations")
        )
        bad_segment_log_state = indicator_from_log(
            preproc_step_log_path(resolver, "bad_segment_removal")
        )
        ecg_log_state = indicator_from_log(
            preproc_step_log_path(resolver, "ecg_artifact_removal")
        )
        finish_log_state = indicator_from_log(preproc_step_log_path(resolver, "finish"))
        self._set_preproc_step_indicator("raw", raw_log_state)
        filter_notches = (
            self._preproc_filter_notches_edit.text()
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
        annotation_rows, _ = self._annotations_table_rows()
        annotations_panel_state = self._preproc_annotations_panel_state_runtime(
            resolver,
            rows=annotation_rows,
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
        )
        self._set_preproc_step_indicator("filter", filter_panel_state)
        self._set_preproc_step_indicator("annotations", annotations_panel_state)
        self._set_preproc_step_indicator("bad_segment_removal", bad_segment_log_state)
        self._set_preproc_step_indicator("ecg_artifact_removal", ecg_panel_state)
        self._set_preproc_step_indicator("finish", finish_log_state)
        filter_raw_exists = preproc_step_raw_path(resolver, "filter").exists()
        annotations_raw_exists = preproc_step_raw_path(resolver, "annotations").exists()
        bad_segment_raw_exists = preproc_step_raw_path(
            resolver, "bad_segment_removal"
        ).exists()
        ecg_raw_exists = preproc_step_raw_path(
            resolver, "ecg_artifact_removal"
        ).exists()
        finish_source_exists = resolve_finish_source(context) is not None
        finish_raw_path = resolver.preproc_root / "finish" / "raw.fif"
        finish_raw_exists = finish_raw_path.exists()
        raw_step_exists = preproc_step_raw_path(resolver, "raw").exists()

        if self._preproc_raw_plot_button is not None:
            self._preproc_raw_plot_button.setEnabled(
                raw_input_exists or raw_step_exists
            )
        if self._preproc_filter_advance_button is not None:
            self._preproc_filter_advance_button.setEnabled(raw_log_state == "green")
        if self._preproc_filter_apply_button is not None:
            self._preproc_filter_apply_button.setEnabled(raw_log_state == "green")
        if self._preproc_filter_plot_button is not None:
            self._preproc_filter_plot_button.setEnabled(
                filter_log_state == "green" and filter_raw_exists
            )
        if self._preproc_filter_notches_edit is not None:
            self._preproc_filter_notches_edit.setEnabled(raw_log_state == "green")
        if self._preproc_filter_low_freq_edit is not None:
            self._preproc_filter_low_freq_edit.setEnabled(raw_log_state == "green")
        if self._preproc_filter_high_freq_edit is not None:
            self._preproc_filter_high_freq_edit.setEnabled(raw_log_state == "green")
        if self._preproc_annotations_edit_button is not None:
            self._preproc_annotations_edit_button.setEnabled(
                filter_log_state == "green"
            )
        if self._preproc_annotations_save_button is not None:
            self._preproc_annotations_save_button.setEnabled(
                filter_log_state == "green"
            )
        if self._preproc_annotations_import_button is not None:
            self._preproc_annotations_import_button.setEnabled(
                filter_log_state == "green"
            )
        if self._preproc_annotations_plot_button is not None:
            self._preproc_annotations_plot_button.setEnabled(
                annotations_log_state == "green" and annotations_raw_exists
            )
        if self._preproc_bad_segment_apply_button is not None:
            self._preproc_bad_segment_apply_button.setEnabled(
                annotations_log_state == "green"
            )
        if self._preproc_bad_segment_plot_button is not None:
            self._preproc_bad_segment_plot_button.setEnabled(
                bad_segment_log_state == "green" and bad_segment_raw_exists
            )
        if self._preproc_ecg_apply_button is not None:
            self._preproc_ecg_apply_button.setEnabled(bad_segment_log_state == "green")
        if self._preproc_ecg_plot_button is not None:
            self._preproc_ecg_plot_button.setEnabled(
                ecg_log_state == "green" and ecg_raw_exists
            )
        if self._preproc_ecg_method_combo is not None:
            self._preproc_ecg_method_combo.setEnabled(bad_segment_log_state == "green")
        if self._preproc_finish_apply_button is not None:
            self._preproc_finish_apply_button.setEnabled(finish_source_exists)
        if self._preproc_finish_plot_button is not None:
            self._preproc_finish_plot_button.setEnabled(finish_raw_exists)
        self._refresh_preproc_ecg_channel_state(context)
        self._refresh_preproc_visualization_controls(context)
