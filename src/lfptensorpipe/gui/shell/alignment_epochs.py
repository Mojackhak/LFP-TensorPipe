"""Alignment epoch table and preview-source MainWindow methods."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidgetItem

from lfptensorpipe.app import PathResolver


class MainWindowAlignmentEpochsMixin:
    @staticmethod
    def _format_alignment_epoch_numeric_cell(value: Any) -> str:
        try:
            numeric = float(value)
        except Exception:
            return ""
        if not np.isfinite(numeric):
            return ""
        return f"{numeric:.3f}"

    def _set_alignment_epoch_rows(self, rows: list[dict[str, Any]]) -> None:
        self._alignment_epoch_rows = list(rows)
        if self._alignment_epoch_table is None:
            self._refresh_alignment_select_all_button_label()
            return
        self._alignment_epoch_table.blockSignals(True)
        self._alignment_epoch_table.setRowCount(len(self._alignment_epoch_rows))
        for row_idx, row in enumerate(self._alignment_epoch_rows):
            pick_item = QTableWidgetItem("")
            pick_item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
            )
            pick_state = bool(row.get("pick", True))
            pick_item.setCheckState(Qt.Checked if pick_state else Qt.Unchecked)
            self._alignment_epoch_table.setItem(row_idx, 0, pick_item)
            self._alignment_epoch_table.setItem(
                row_idx,
                1,
                QTableWidgetItem(str(row.get("epoch_label", f"epoch_{row_idx:03d}"))),
            )
            self._alignment_epoch_table.setItem(
                row_idx,
                2,
                QTableWidgetItem(
                    self._format_alignment_epoch_numeric_cell(
                        row.get("duration_s", np.nan)
                    )
                ),
            )
            self._alignment_epoch_table.setItem(
                row_idx,
                3,
                QTableWidgetItem(
                    self._format_alignment_epoch_numeric_cell(
                        row.get("start_t", np.nan)
                    )
                ),
            )
            self._alignment_epoch_table.setItem(
                row_idx,
                4,
                QTableWidgetItem(
                    self._format_alignment_epoch_numeric_cell(
                        row.get("end_t", np.nan)
                    )
                ),
            )
        self._alignment_epoch_table.blockSignals(False)
        self._refresh_alignment_select_all_button_label()

    def _all_alignment_epochs_picked(self) -> bool:
        if self._alignment_epoch_table is None:
            return False
        row_count = self._alignment_epoch_table.rowCount()
        if row_count <= 0:
            return False
        for row_idx in range(row_count):
            pick_item = self._alignment_epoch_table.item(row_idx, 0)
            if pick_item is None or pick_item.checkState() != Qt.Checked:
                return False
        return True

    def _refresh_alignment_select_all_button_label(self) -> None:
        if self._alignment_select_all_button is None:
            return
        self._alignment_select_all_button.setText(
            "Deselect All" if self._all_alignment_epochs_picked() else "Select All"
        )

    def _on_alignment_epoch_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        self._refresh_alignment_select_all_button_label()
        self._refresh_alignment_controls()
        self._persist_alignment_epoch_picks_state()

    def _selected_alignment_epoch_indices(self) -> list[int]:
        if self._alignment_epoch_table is None:
            return []
        selected: list[int] = []
        for row_idx, row in enumerate(self._alignment_epoch_rows):
            pick_item = self._alignment_epoch_table.item(row_idx, 0)
            if pick_item is None:
                continue
            if pick_item.checkState() != Qt.Checked:
                continue
            selected.append(int(row.get("epoch_index", row_idx)))
        return selected

    def _refresh_alignment_metric_combo(self) -> None:
        if self._alignment_epoch_metric_combo is None:
            return
        context = self._record_context()
        slug = self._current_alignment_paradigm_slug()
        current_metric = self._alignment_epoch_metric_combo.currentData()
        metrics: list[str] = []
        if context is not None and slug:
            resolver = PathResolver(context)
            paradigm_dir = resolver.alignment_paradigm_dir(slug, create=False)
            if paradigm_dir.exists():
                for metric_dir in sorted(
                    path for path in paradigm_dir.iterdir() if path.is_dir()
                ):
                    if (metric_dir / "tensor_warped.pkl").exists():
                        metrics.append(metric_dir.name)
        self._alignment_epoch_metric_combo.blockSignals(True)
        self._alignment_epoch_metric_combo.clear()
        if not metrics:
            self._alignment_epoch_metric_combo.addItem("No metric output", None)
        else:
            for metric_key in metrics:
                self._alignment_epoch_metric_combo.addItem(metric_key, metric_key)
            if isinstance(current_metric, str):
                idx = self._alignment_epoch_metric_combo.findData(current_metric)
                if idx >= 0:
                    self._alignment_epoch_metric_combo.setCurrentIndex(idx)
        self._alignment_epoch_metric_combo.blockSignals(False)
        self._refresh_alignment_epoch_channel_combo()

    def _on_alignment_epoch_metric_changed(self, index: int) -> None:
        _ = index
        self._refresh_alignment_epoch_channel_combo()
        self._refresh_alignment_controls()

    def _refresh_alignment_epoch_channel_combo(self) -> None:
        if self._alignment_epoch_channel_combo is None:
            return
        current_channel = self._alignment_epoch_channel_combo.currentData()
        channels: list[str] = []

        context = self._record_context()
        slug = self._current_alignment_paradigm_slug()
        metric_key = (
            self._alignment_epoch_metric_combo.currentData()
            if self._alignment_epoch_metric_combo is not None
            else None
        )
        if context is not None and slug and isinstance(metric_key, str):
            path = self._alignment_metric_tensor_warped_path_runtime(
                PathResolver(context),
                slug,
                metric_key,
            )
            if path.exists():
                try:
                    payload = self._load_pickle(path)
                    if isinstance(payload, dict):
                        tensor = np.asarray(payload.get("tensor"))
                        if tensor.ndim == 4:
                            n_channels = int(tensor.shape[1])
                            meta = payload.get("meta", {})
                            axes = (
                                meta.get("axes", {}) if isinstance(meta, dict) else {}
                            )
                            channel_axis = (
                                axes.get("channel") if isinstance(axes, dict) else None
                            )
                            if isinstance(channel_axis, (list, tuple, np.ndarray)):
                                axis_labels = [str(item) for item in channel_axis]
                                channels.extend(axis_labels[:n_channels])
                            if len(channels) < n_channels:
                                channels.extend(
                                    f"channel_{idx:03d}"
                                    for idx in range(len(channels), n_channels)
                                )
                except Exception:
                    channels = []

        self._alignment_epoch_channel_combo.blockSignals(True)
        self._alignment_epoch_channel_combo.clear()
        if not channels:
            self._alignment_epoch_channel_combo.addItem("No channel", None)
        else:
            for idx, label in enumerate(channels):
                self._alignment_epoch_channel_combo.addItem(label, idx)
            if isinstance(current_channel, (int, np.integer)):
                selected_idx = self._alignment_epoch_channel_combo.findData(
                    int(current_channel)
                )
                if selected_idx >= 0:
                    self._alignment_epoch_channel_combo.setCurrentIndex(selected_idx)
        self._alignment_epoch_channel_combo.blockSignals(False)

    def _collect_alignment_pick_indices(self) -> list[int]:
        picks: list[int] = []
        if self._alignment_epoch_table is None:
            return picks
        for row_idx in range(self._alignment_epoch_table.rowCount()):
            item = self._alignment_epoch_table.item(row_idx, 0)
            if item is None or item.checkState() != Qt.Checked:
                continue
            row_payload = (
                self._alignment_epoch_rows[row_idx]
                if row_idx < len(self._alignment_epoch_rows)
                else {"epoch_index": row_idx}
            )
            epoch_index = int(row_payload.get("epoch_index", row_idx))
            picks.append(epoch_index)
        return picks

    def _persist_alignment_epoch_picks_state(self) -> None:
        if self._record_param_syncing:
            return
        context = self._record_context()
        slug = self._current_alignment_paradigm_slug()
        if context is None or slug is None:
            return
        try:
            self._persist_alignment_epoch_picks_runtime(
                context,
                paradigm_slug=slug,
                picked_epoch_indices=self._collect_alignment_pick_indices(),
            )
        except Exception:
            return
