"""Tensor refresh and control MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    RecordContext,
    TENSOR_COMMON_BASIC_METRIC_KEYS,
    load_tensor_frequency_defaults,
)


class MainWindowTensorStateRefreshMixin:
    def _on_tensor_metric_selection_changed(self, state: int) -> None:
        _ = state
        self._refresh_tensor_controls()

    def _set_tensor_frequency_defaults_from_context(
        self, context: RecordContext | None
    ) -> None:
        self._ensure_tensor_metric_state_from_defaults(context)
        if context is not None:
            low_freq, high_freq, step_hz = load_tensor_frequency_defaults(context)
            for metric_key in TENSOR_COMMON_BASIC_METRIC_KEYS:
                params = dict(self._tensor_metric_params.get(metric_key, {}))
                params.setdefault("low_freq_hz", float(low_freq))
                params.setdefault("high_freq_hz", float(high_freq))
                params.setdefault("freq_step_hz", float(step_hz))
                self._tensor_metric_params[metric_key] = params
        self._apply_active_tensor_params_to_panel()

    def _refresh_tensor_metric_indicators(self, context: RecordContext | None) -> None:
        if context is None:
            for metric_key, indicator in self._tensor_metric_indicators.items():
                self._set_indicator_color(indicator, "gray")
                indicator.setToolTip(
                    f"{self._tensor_metric_display_name(metric_key)} state: "
                    "gray=not run, yellow=stale or failed, "
                    "green=current params match successful output. Current: gray."
                )
            return
        mask_edge_effects = (
            self._tensor_mask_edge_checkbox.isChecked()
            if self._tensor_mask_edge_checkbox is not None
            else True
        )
        for metric_key, indicator in self._tensor_metric_indicators.items():
            metric_params = (
                self._active_tensor_panel_indicator_params()
                if metric_key == self._tensor_active_metric_key
                else self._tensor_metric_params.get(metric_key, {})
            )
            state = self._tensor_metric_panel_state_runtime(
                context,
                metric_key=metric_key,
                metric_params=metric_params,
                mask_edge_effects=mask_edge_effects,
            )
            self._set_indicator_color(indicator, state)
            indicator.setToolTip(
                f"{self._tensor_metric_display_name(metric_key)} state: "
                "gray=not run, yellow=stale or failed, "
                f"green=current params match successful output. Current: {state}."
            )

    def _refresh_tensor_metric_indicators_from_draft(self) -> None:
        context = self._record_context()
        self._refresh_tensor_metric_indicators(context)

    def _refresh_tensor_controls(self) -> None:
        context = self._record_context()
        self._ensure_tensor_metric_state_from_defaults(context)
        self._refresh_tensor_channel_state(context)
        self._apply_active_tensor_params_to_panel()
        self._refresh_tensor_metric_indicators(context)
        if self._tensor_run_is_active():
            self._refresh_tensor_controls_for_active_run()
            return
        preproc_ready = self._stage_states.get("preproc") == "green"
        editable = context is not None and preproc_ready
        metric_key = self._tensor_active_metric_key
        metric_spec = self._tensor_metric_spec(metric_key)
        metric_supported = bool(metric_spec and metric_spec.supported)
        visible_basic_rows = self._refresh_tensor_basic_param_row_visibility()
        self._refresh_tensor_selector_row_visibility()

        for row_metric_key, checkbox in self._tensor_metric_checks.items():
            spec = self._tensor_metric_spec(row_metric_key)
            checkbox.setEnabled(editable and bool(spec and spec.supported))
            if not checkbox.isEnabled():
                checkbox.setChecked(False)
        for name_button in self._tensor_metric_name_buttons.values():
            name_button.setEnabled(context is not None)

        selected_metrics = self._selected_tensor_metrics()
        for row_key, widget in self._tensor_basic_param_widgets.items():
            widget.setEnabled(editable and row_key in visible_basic_rows)

        requires_channels = self._tensor_metric_requires_channel_selector(metric_key)
        requires_pairs = self._tensor_metric_pair_mode(metric_key) is not None
        if self._tensor_channels_button is not None:
            self._tensor_channels_button.setEnabled(
                editable and requires_channels and bool(self._tensor_available_channels)
            )
        if self._tensor_pairs_button is not None:
            self._tensor_pairs_button.setEnabled(
                editable and requires_pairs and bool(self._tensor_available_channels)
            )
            self._refresh_tensor_pair_button_text()
        if self._tensor_advance_button is not None:
            self._tensor_advance_button.setEnabled(editable and metric_supported)
        if self._tensor_mask_edge_checkbox is not None:
            self._tensor_mask_edge_checkbox.setEnabled(editable)
        if self._tensor_import_button is not None:
            self._tensor_import_button.setEnabled(context is not None)
        if self._tensor_export_button is not None:
            self._tensor_export_button.setEnabled(context is not None)
        if self._tensor_run_button is not None:
            self._tensor_run_button.setText("Build Tensor")
            self._tensor_run_button.setEnabled(editable and bool(selected_metrics))
