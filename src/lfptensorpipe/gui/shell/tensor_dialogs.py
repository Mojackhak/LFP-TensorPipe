"""Tensor dialog and interactive configuration MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.dialogs.tensor_bands import TensorBandsConfigureDialog
from lfptensorpipe.gui.shell.common import (
    Any,
    build_tensor_metric_notch_payload,
    compute_tensor_metric_filter_notch_warnings,
    QDialog,
    QMessageBox,
    load_tensor_filter_metric_notch_params,
    TENSOR_UNDIRECTED_METRIC_KEYS,
)


class MainWindowTensorDialogsMixin:
    def _inherit_tensor_metric_notches_from_filter(self, context: Any) -> bool:
        inherited_payload = load_tensor_filter_metric_notch_params(context)
        self._ensure_tensor_metric_state_from_defaults(context)
        changed = False
        for spec in self._stage_tensor_metric_specs():
            params = dict(self._tensor_metric_params.get(spec.key, {}))
            params.update(inherited_payload)
            normalized = dict(params)
            normalized.update(
                build_tensor_metric_notch_payload(
                    normalized.get("notches"),
                    normalized.get("notch_widths"),
                )
            )
            if normalized != self._tensor_metric_params.get(spec.key, {}):
                self._apply_tensor_metric_payload(spec.key, normalized)
                changed = True
        if changed:
            self._apply_active_tensor_params_to_panel()
            self._refresh_tensor_controls()
            self._mark_record_param_dirty("tensor.metric_params")
        return changed

    def _tensor_metric_notch_warnings(
        self,
        context: Any,
        metric_key: str,
        params: dict[str, Any],
    ) -> list[str]:
        return compute_tensor_metric_filter_notch_warnings(
            context,
            metric_key,
            params,
        )

    def _show_tensor_metric_notch_warning(
        self,
        metric_key: str,
        warnings: list[str],
    ) -> None:
        if not warnings:
            return
        self._show_warning(
            "Tensor Advance",
            (
                f"{self._tensor_metric_display_name(metric_key)} diverges from the "
                "completed Preprocess filter notch baseline:\n- "
                + "\n- ".join(warnings)
            ),
        )

    def _confirm_tensor_preflight_notch_warnings(
        self,
        warnings_by_metric: dict[str, list[str]],
    ) -> bool:
        if not warnings_by_metric:
            return True

        lines = [
            "Selected metrics diverge from the completed Preprocess filter notch baseline:",
            "",
        ]
        for metric_key, warnings in warnings_by_metric.items():
            lines.append(f"{self._tensor_metric_display_name(metric_key)}:")
            lines.extend(f"- {item}" for item in warnings)
            lines.append("")
        lines.append("Continue anyway?")

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle("Run Build Tensor")
        dialog.setText("\n".join(lines))
        continue_button = dialog.addButton("Continue", QMessageBox.AcceptRole)
        dialog.addButton("Cancel", QMessageBox.RejectRole)
        dialog.exec()
        return dialog.clickedButton() is continue_button

    def _on_tensor_channels_select(self) -> None:
        metric_key = self._tensor_active_metric_key
        self._commit_active_tensor_panel_to_params()
        if not self._tensor_metric_requires_channel_selector(metric_key):
            self._show_information(
                "Tensor Channels",
                "Channels are only available for power and burst metrics.",
            )
            return
        if not self._tensor_available_channels:
            self._show_warning(
                "Tensor Channels",
                "No channels available for selection.",
            )
            return

        default_channels = self._tensor_default_selected_channels_for_metric(
            metric_key,
            available_channels=self._tensor_available_channels,
        )

        def _save_channel_defaults(chosen_channels: tuple[str, ...]) -> None:
            self._tensor_save_default_channels(chosen_channels)
            metric_defaults = self._tensor_prepare_metric_default_payload(
                metric_key,
                {
                    **self._tensor_metric_params.get(metric_key, {}),
                    "selected_channels": list(chosen_channels),
                },
            )
            self._save_tensor_metric_default_params(metric_key, metric_defaults)
            self.statusBar().showMessage("Tensor channel defaults updated.")

        dialog = self._create_tensor_channel_select_dialog(
            title="Tensor Channels",
            channels=self._tensor_available_channels,
            session_selected=self._tensor_selected_channels_by_metric.get(
                metric_key, ()
            ),
            default_selected=default_channels,
            set_default_callback=_save_channel_defaults,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = tuple(dialog.selected_channels)
        self._tensor_selected_channels_by_metric[metric_key] = chosen
        self._mark_record_param_dirty("tensor.selectors")
        if self._tensor_channels_button is not None:
            self._tensor_channels_button.setText(
                self._format_channel_button_text(
                    "Select Channels",
                    chosen,
                    self._tensor_available_channels,
                )
            )
        self._commit_active_tensor_panel_to_params()
        self._refresh_tensor_metric_indicators_from_draft()

    def _on_tensor_pairs_select(self) -> None:
        metric_key = self._tensor_active_metric_key
        self._commit_active_tensor_panel_to_params()
        mode = self._tensor_metric_pair_mode(metric_key)
        if mode is None:
            self._show_information(
                "Tensor Pairs",
                "Pairs are only available for connectivity metrics.",
            )
            return
        if not self._tensor_available_channels:
            self._show_warning(
                "Tensor Pairs",
                "No channels available for pair selection.",
            )
            return

        directed = mode == "directed"
        session_pairs = self._tensor_selected_pairs_by_metric.get(metric_key, ())
        default_pairs = self._tensor_default_selected_pairs_for_metric(
            metric_key,
            directed=directed,
            available_channels=self._tensor_available_channels,
        )

        def _save_pair_defaults(
            chosen_pairs: tuple[tuple[str, str], ...],
        ) -> None:
            self._tensor_save_default_pairs(chosen_pairs, directed=directed)
            metric_defaults = self._tensor_prepare_metric_default_payload(
                metric_key,
                {
                    **self._tensor_metric_params.get(metric_key, {}),
                    "selected_pairs": [
                        [source, target] for source, target in chosen_pairs
                    ],
                },
            )
            self._save_tensor_metric_default_params(metric_key, metric_defaults)
            self.statusBar().showMessage("Tensor pair defaults updated.")

        dialog = self._create_tensor_pair_select_dialog(
            title="Tensor Pairs",
            channel_names=self._tensor_available_channels,
            session_pairs=session_pairs,
            default_pairs=default_pairs,
            directed=directed,
            set_default_callback=_save_pair_defaults,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = self._filter_tensor_pairs(
            dialog.selected_pairs,
            available_channels=self._tensor_available_channels,
            directed=directed,
        )
        self._tensor_selected_pairs_by_metric[metric_key] = chosen
        self._mark_record_param_dirty("tensor.selectors")
        self._refresh_tensor_pair_button_text()
        self._commit_active_tensor_panel_to_params()
        self._refresh_tensor_metric_indicators_from_draft()

    @staticmethod
    def _tensor_metric_advanced_keys(metric_key: str) -> tuple[str, ...]:
        if metric_key == "raw_power":
            return (
                "method",
                "min_cycles",
                "max_cycles",
                "time_bandwidth",
                "notches",
                "notch_widths",
            )
        if metric_key == "periodic_aperiodic":
            return (
                "method",
                "min_cycles",
                "max_cycles",
                "time_bandwidth",
                "freq_smooth_enabled",
                "freq_smooth_sigma",
                "time_smooth_enabled",
                "time_smooth_kernel_size",
                "aperiodic_mode",
                "peak_width_limits_hz",
                "max_n_peaks",
                "min_peak_height",
                "peak_threshold",
                "fit_qc_threshold",
                "notches",
                "notch_widths",
            )
        if metric_key in TENSOR_UNDIRECTED_METRIC_KEYS:
            return (
                "method",
                "mt_bandwidth",
                "min_cycles",
                "max_cycles",
                "notches",
                "notch_widths",
            )
        if metric_key == "trgc":
            return (
                "method",
                "mt_bandwidth",
                "min_cycles",
                "max_cycles",
                "gc_n_lags",
                "group_by_samples",
                "round_ms",
                "notches",
                "notch_widths",
            )
        if metric_key == "psi":
            return (
                "method",
                "mt_bandwidth",
                "min_cycles",
                "max_cycles",
                "notches",
                "notch_widths",
            )
        if metric_key == "burst":
            return (
                "thresholds_path",
                "thresholds",
                "min_cycles",
                "max_cycles",
                "notches",
                "notch_widths",
            )
        return ()

    def _on_tensor_metric_advance(self) -> None:
        metric_key = self._tensor_active_metric_key
        self._commit_active_tensor_panel_to_params()
        session_params = dict(self._tensor_metric_params.get(metric_key, {}))
        default_params = self._tensor_effective_metric_defaults(
            metric_key,
            context=self._record_context(),
            available_channels=self._tensor_available_channels,
        )

        def _save_metric_defaults(full_params: dict[str, Any]) -> None:
            prepared = self._tensor_prepare_metric_default_payload(
                metric_key, full_params
            )
            self._save_tensor_metric_default_params(metric_key, prepared)
            self.statusBar().showMessage("Tensor metric defaults updated.")

        burst_baseline_annotations: tuple[str, ...] = ()
        if metric_key == "burst":
            context = self._record_context()
            if context is not None:
                loaded_labels = self._load_burst_baseline_annotation_labels_runtime(
                    context
                )
                if isinstance(loaded_labels, list):
                    burst_baseline_annotations = tuple(
                        str(item).strip() for item in loaded_labels if str(item).strip()
                    )
        dialog = self._create_tensor_metric_advance_dialog(
            metric_key=metric_key,
            metric_label=self._tensor_metric_display_name(metric_key),
            session_params=session_params,
            default_params=default_params,
            burst_baseline_annotations=burst_baseline_annotations,
            set_default_callback=_save_metric_defaults,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        payload = dialog.selected_params or {}
        self._apply_tensor_metric_payload(metric_key, payload)
        self._apply_active_tensor_params_to_panel()
        self._refresh_tensor_pair_button_text()
        self._refresh_tensor_bands_button_text()
        if self._tensor_metric_requires_channel_selector(metric_key) or (
            self._tensor_metric_pair_mode(metric_key) is not None
        ):
            self._mark_record_param_dirty("tensor.selectors")
        self._mark_record_param_dirty("tensor.metric_params")
        self._refresh_tensor_metric_indicators_from_draft()
        context = self._record_context()
        if context is not None and getattr(dialog, "selected_action", "save") == "save":
            warnings = self._tensor_metric_notch_warnings(
                context,
                metric_key,
                dict(self._tensor_metric_params.get(metric_key, {})),
            )
            self._show_tensor_metric_notch_warning(metric_key, warnings)

    def _refresh_tensor_bands_button_text(self) -> None:
        if self._tensor_bands_configure_button is None:
            return
        metric_key = self._tensor_active_metric_key
        if metric_key not in {"psi", "burst"}:
            self._tensor_bands_configure_button.setText("Bands Configure...")
            return
        params = self._tensor_metric_params.get(metric_key, {})
        bands = self._normalize_tensor_bands_rows(params.get("bands"))
        self._tensor_bands_configure_button.setText(
            f"Bands Configure... ({len(bands)})"
        )

    def _on_tensor_bands_configure(self) -> None:
        metric_key = self._tensor_active_metric_key
        self._commit_active_tensor_panel_to_params()
        if metric_key not in {"psi", "burst"}:
            self._show_information(
                "Bands Configure",
                "Bands are only editable for PSI and Burst.",
            )
            return
        params = dict(self._tensor_metric_params.get(metric_key, {}))
        current_bands = self._normalize_tensor_bands_rows(params.get("bands"))
        if not current_bands:
            current_bands = self._load_tensor_metric_bands_defaults(metric_key)
        default_params = self._tensor_effective_metric_defaults(
            metric_key,
            context=self._record_context(),
            available_channels=self._tensor_available_channels,
        )
        default_bands = self._normalize_tensor_bands_rows(default_params.get("bands"))
        if not default_bands:
            default_bands = self._load_tensor_metric_bands_defaults(metric_key)

        def _save_bands_defaults(
            chosen_bands: tuple[dict[str, float | str], ...],
        ) -> None:
            metric_defaults = self._tensor_prepare_metric_default_payload(
                metric_key,
                {
                    **self._tensor_metric_params.get(metric_key, {}),
                    "bands": [dict(item) for item in chosen_bands],
                },
            )
            self._save_tensor_metric_default_params(metric_key, metric_defaults)
            self.statusBar().showMessage("Tensor bands defaults updated.")

        dialog = TensorBandsConfigureDialog(
            title=f"{self._tensor_metric_display_name(metric_key)} Bands",
            current_bands=tuple(dict(item) for item in current_bands),
            default_bands=tuple(dict(item) for item in default_bands),
            set_default_callback=_save_bands_defaults,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        params["bands"] = [dict(item) for item in dialog.selected_bands]
        self._tensor_metric_params[metric_key] = params
        self._refresh_tensor_bands_button_text()
        self._mark_record_param_dirty("tensor.metric_params")
        self._refresh_tensor_metric_indicators_from_draft()
