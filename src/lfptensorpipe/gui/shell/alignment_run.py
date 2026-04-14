"""Alignment run, preview, and finish MainWindow methods."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt

from lfptensorpipe.app import (
    PathResolver,
)


class MainWindowAlignmentRunMixin:
    @staticmethod
    def _alignment_merge_location_status_text(ready: bool) -> str:
        return (
            "Merge Location Info: Ready"
            if ready
            else "Merge Location Info: Not Ready"
        )

    @staticmethod
    def _alignment_preview_transform_mode(metric_key: str) -> str | None:
        key = str(metric_key).strip().lower()
        if key in {"raw_power", "periodic"}:
            return "dB"
        if key == "burst":
            return "log10"
        if key == "aperiodic":
            return "zscore"
        return None

    def _tighten_alignment_preview_figure(
        self,
        fig: Any,
        *,
        pad_in: float = 0.03,
    ) -> None:
        canvas = getattr(fig, "canvas", None)
        if canvas is None:
            return
        try:
            from matplotlib.transforms import Bbox

            canvas.draw()
            renderer = canvas.get_renderer()
            tight_bbox = fig.get_tightbbox(renderer)
            if tight_bbox is None:
                return
            bbox = Bbox.from_extents(
                float(tight_bbox.x0),
                float(tight_bbox.y0),
                float(tight_bbox.x1),
                float(tight_bbox.y1),
            )
            if (
                not np.all(np.isfinite([bbox.x0, bbox.y0, bbox.x1, bbox.y1]))
                or bbox.width <= 0.0
                or bbox.height <= 0.0
            ):
                return

            fig_w_in, fig_h_in = fig.get_size_inches()
            new_w_in = float(bbox.width) + 2.0 * float(pad_in)
            new_h_in = float(bbox.height) + 2.0 * float(pad_in)
            if new_w_in <= 0.0 or new_h_in <= 0.0:
                return

            for ax in fig.axes:
                pos = ax.get_position()
                ax.set_position(
                    [
                        ((pos.x0 * fig_w_in) - bbox.x0 + pad_in) / new_w_in,
                        ((pos.y0 * fig_h_in) - bbox.y0 + pad_in) / new_h_in,
                        (pos.width * fig_w_in) / new_w_in,
                        (pos.height * fig_h_in) / new_h_in,
                    ]
                )

            for text in getattr(fig, "texts", []):
                if text.get_transform() is not fig.transFigure:
                    continue
                x_frac, y_frac = text.get_position()
                text.set_position(
                    (
                        ((float(x_frac) * fig_w_in) - bbox.x0 + pad_in) / new_w_in,
                        ((float(y_frac) * fig_h_in) - bbox.y0 + pad_in) / new_h_in,
                    )
                )

            for legend in getattr(fig, "legends", []):
                anchor_box = legend.get_bbox_to_anchor()
                if anchor_box is None:
                    continue
                anchor_box = anchor_box.transformed(fig.transFigure.inverted())
                legend.set_bbox_to_anchor(
                    Bbox.from_bounds(
                        ((float(anchor_box.x0) * fig_w_in) - bbox.x0 + pad_in)
                        / new_w_in,
                        ((float(anchor_box.y0) * fig_h_in) - bbox.y0 + pad_in)
                        / new_h_in,
                        (float(anchor_box.width) * fig_w_in) / new_w_in,
                        (float(anchor_box.height) * fig_h_in) / new_h_in,
                    ),
                    transform=fig.transFigure,
                )

            fig.set_size_inches(new_w_in, new_h_in, forward=True)
            if hasattr(canvas, "draw_idle"):
                canvas.draw_idle()
        except Exception:
            return

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
        self._alignment_merge_location_status_label.setText(
            self._alignment_merge_location_status_text(ready)
        )
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
        has_metric_output = (
            self._alignment_epoch_metric_combo is not None
            and isinstance(self._alignment_epoch_metric_combo.currentData(), str)
        )
        has_channel_output = (
            self._alignment_epoch_channel_combo is not None
            and isinstance(self._alignment_epoch_channel_combo.currentData(), int)
        )
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
        self._refresh_alignment_select_all_button_label()
        if self._alignment_preview_button is not None:
            self._alignment_preview_button.setEnabled(
                has_epoch_rows and has_metric_output and has_channel_output
            )
        if self._alignment_finish_button is not None:
            self._alignment_finish_button.setEnabled(
                has_epoch_rows and has_metric_output and method_state == "green"
            )
        if self._alignment_epoch_metric_combo is not None:
            self._alignment_epoch_metric_combo.setEnabled(has_metric_output)
        if self._alignment_epoch_channel_combo is not None:
            self._alignment_epoch_channel_combo.setEnabled(has_channel_output)

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
        self._reload_alignment_paradigms(preferred_slug=slug)

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
        self._refresh_alignment_metric_combo()
        self._refresh_stage_states_from_context()
        self._refresh_alignment_controls()
        prefix = "Align Epochs OK" if ok else "Align Epochs failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="alignment_run")

    def _on_alignment_select_all(self) -> None:
        if self._alignment_epoch_table is None:
            return
        table = self._alignment_epoch_table
        row_count = table.rowCount()
        should_select_all = False
        for row_idx in range(row_count):
            item = table.item(row_idx, 0)
            if item is None or item.checkState() != Qt.Checked:
                should_select_all = True
                break
        table.blockSignals(True)
        try:
            for row_idx in range(table.rowCount()):
                item = table.item(row_idx, 0)
                if item is not None:
                    item.setCheckState(
                        Qt.Checked if should_select_all else Qt.Unchecked
                    )
        finally:
            table.blockSignals(False)
        self._refresh_alignment_select_all_button_label()
        self._refresh_alignment_controls()
        self._persist_alignment_epoch_picks_state()
        self._mark_record_param_dirty("alignment.picks")

    def _on_alignment_preview(self) -> None:
        if not self._enable_plots:
            return
        context = self._record_context()
        slug = self._current_alignment_paradigm_slug()
        if (
            context is None
            or slug is None
            or self._alignment_epoch_metric_combo is None
            or self._alignment_epoch_channel_combo is None
        ):
            return
        metric_key = self._alignment_epoch_metric_combo.currentData()
        if not isinstance(metric_key, str):
            return
        self._refresh_alignment_epoch_channel_combo()
        path = self._alignment_metric_tensor_warped_path_runtime(
            PathResolver(context),
            slug,
            metric_key,
        )
        if not path.exists():
            self.statusBar().showMessage(
                "Preview unavailable: missing tensor_warped.pkl."
            )
            return
        try:
            payload = self._load_pickle(path)
            if not isinstance(payload, dict):
                raise ValueError("Invalid tensor payload.")
            tensor = np.asarray(payload.get("tensor"), dtype=float)
            if tensor.ndim != 4:
                raise ValueError(f"Expected 4D tensor, got {tensor.shape}.")
            picks = self._selected_alignment_epoch_indices()
            valid = [idx for idx in picks if 0 <= idx < tensor.shape[0]]
            if not valid:
                raise ValueError("No picked epochs available for preview.")
            channel_index = self._alignment_epoch_channel_combo.currentData()
            if not isinstance(channel_index, (int, np.integer)):
                raise ValueError("No channel selected for preview.")
            channel_idx = int(channel_index)
            if channel_idx < 0 or channel_idx >= tensor.shape[1]:
                raise ValueError(f"Preview channel index out of range: {channel_idx}.")
            mean_map = np.nanmean(tensor[valid, channel_idx, :, :], axis=0)
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            axes = meta.get("axes", {}) if isinstance(meta, dict) else {}
            freq_axis = np.asarray(axes.get("freq"))
            time_axis = np.asarray(axes.get("percent", axes.get("time")))
            if freq_axis.ndim != 1 or time_axis.ndim != 1:
                raise ValueError("Warped metadata is missing freq/time axes.")
            if freq_axis.size != tensor.shape[2] or time_axis.size != tensor.shape[3]:
                raise ValueError("Warped metadata axis lengths do not match tensor.")

            freq_labels = freq_axis.tolist()
            time_labels = time_axis.tolist()
            preview_df = pd.DataFrame(
                {
                    "Value": [
                        pd.DataFrame(mean_map, index=freq_labels, columns=time_labels)
                    ]
                }
            )
            transform_mode = self._alignment_preview_transform_mode(metric_key)
            if transform_mode is not None:
                preview_df = self._transform_dataframe(
                    preview_df,
                    value_col="Value",
                    mode=transform_mode,
                )
            has_string_y = any(isinstance(item, (str, np.str_)) for item in freq_labels)
            y_log = False
            if not has_string_y:
                y_numeric = pd.to_numeric(np.asarray(freq_labels), errors="coerce")
                y_log = bool(
                    np.all(np.isfinite(y_numeric))
                    and np.all(np.asarray(y_numeric, dtype=float) > 0.0)
                )
            preview_params = self._load_alignment_preview_plot_params(metric_key)
            fig = self._plot_single_effect_df(
                preview_df,
                value_col="Value",
                x_label="Percent (%)",
                y_label="Frequency",
                y_log=y_log,
                title=None,
                cmap="viridis",
                boxsize=preview_params["boxsize"],
                title_fontsize=preview_params["font_size"],
                axis_label_fontsize=preview_params["font_size"],
                tick_label_fontsize=preview_params["tick_label_size"],
                x_label_offset_mm=preview_params["x_label_offset_mm"],
                y_label_offset_mm=preview_params["y_label_offset_mm"],
                colorbar_pad_mm=preview_params["colorbar_pad_mm"],
                cbar_label_offset_mm=preview_params["cbar_label_offset_mm"],
                colorbar_label=preview_params["colorbar_label"],
            )
            self._tighten_alignment_preview_figure(fig)
            fig.show()
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Preview", f"Preview failed:\n{exc}")

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
