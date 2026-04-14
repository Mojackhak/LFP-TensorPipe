"""Record-parameter alignment apply MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    Qt,
    _nested_get,
)


class MainWindowRecordParamsApplyAlignmentMixin:
    def _apply_record_params_alignment_snapshot(self, snapshot: dict[str, Any]) -> int:
        skipped = 0

        if "alignment.paradigm" not in self._record_param_dirty_keys:
            slug = _nested_get(snapshot, ("alignment", "trial_slug"))
            if not isinstance(slug, str) or not slug:
                legacy_slug = _nested_get(snapshot, ("alignment", "paradigm_slug"))
                if isinstance(legacy_slug, str) and legacy_slug:
                    slug = legacy_slug
            if (
                isinstance(slug, str)
                and slug
                and self._alignment_paradigm_list is not None
            ):
                for idx in range(self._alignment_paradigm_list.count()):
                    item = self._alignment_paradigm_list.item(idx)
                    if item is None:
                        continue
                    if item.data(Qt.UserRole) == slug:
                        self._alignment_paradigm_list.setCurrentRow(idx)
                        break
        else:
            skipped += 1

        if "alignment.method" not in self._record_param_dirty_keys:
            method = _nested_get(snapshot, ("alignment", "method"))
            if isinstance(method, str) and self._alignment_method_combo is not None:
                idx = self._alignment_method_combo.findData(method)
                if idx < 0:
                    idx = 0
                self._alignment_method_combo.setCurrentIndex(idx)
        else:
            skipped += 1

        if "alignment.sample_rate" not in self._record_param_dirty_keys:
            sample_rate = _nested_get(snapshot, ("alignment", "sample_rate"))
            if sample_rate is not None and self._alignment_n_samples_edit is not None:
                try:
                    self._alignment_n_samples_edit.setText(f"{float(sample_rate):g}")
                except Exception:
                    pass
        else:
            skipped += 1

        if "alignment.epoch_metric" not in self._record_param_dirty_keys:
            epoch_metric = _nested_get(snapshot, ("alignment", "epoch_metric"))
            if (
                isinstance(epoch_metric, str)
                and self._alignment_epoch_metric_combo is not None
            ):
                idx = self._alignment_epoch_metric_combo.findData(epoch_metric)
                if idx >= 0:
                    self._alignment_epoch_metric_combo.setCurrentIndex(idx)
        else:
            skipped += 1

        if "alignment.epoch_channel" not in self._record_param_dirty_keys:
            epoch_channel = _nested_get(snapshot, ("alignment", "epoch_channel"))
            if (
                isinstance(epoch_channel, (int, float))
                and self._alignment_epoch_channel_combo is not None
            ):
                idx = self._alignment_epoch_channel_combo.findData(int(epoch_channel))
                if idx >= 0:
                    self._alignment_epoch_channel_combo.setCurrentIndex(idx)
        else:
            skipped += 1

        if "alignment.picks" not in self._record_param_dirty_keys:
            picks = _nested_get(snapshot, ("alignment", "picked_epoch_indices"))
            if isinstance(picks, list) and self._alignment_epoch_table is not None:
                selected = {
                    int(item) for item in picks if isinstance(item, (int, float))
                }
                for row_idx in range(self._alignment_epoch_table.rowCount()):
                    item = self._alignment_epoch_table.item(row_idx, 0)
                    if item is None:
                        continue
                    row_payload = (
                        self._alignment_epoch_rows[row_idx]
                        if row_idx < len(self._alignment_epoch_rows)
                        else {"epoch_index": row_idx}
                    )
                    epoch_index = int(row_payload.get("epoch_index", row_idx))
                    item.setCheckState(
                        Qt.Checked if epoch_index in selected else Qt.Unchecked
                    )
        else:
            skipped += 1

        return skipped
