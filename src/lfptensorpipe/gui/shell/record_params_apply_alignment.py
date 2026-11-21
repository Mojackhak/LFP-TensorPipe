"""Record-parameter alignment apply MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    Qt,
    _nested_get,
)


def _restore_method_draft(paradigm, saved, method):
    cache = dict(paradigm.get("method_params_by_method", {}))
    saved_cache = saved.get("method_params_by_method")
    if isinstance(saved_cache, dict):
        cache.update({k: v for k, v in saved_cache.items() if isinstance(v, dict)})
    params = saved.get("method_params", cache.get(method))
    if params is None and paradigm.get("method") == method:
        params = paradigm.get("method_params")
    if not isinstance(params, dict):
        return None
    cache[method] = params
    draft = {
        "method": method,
        "method_params": params,
        "method_params_by_method": cache,
    }
    paradigm.update(draft)
    return draft


class MainWindowRecordParamsApplyAlignmentMixin:
    def _apply_record_params_alignment_snapshot(self, snapshot: dict[str, Any]) -> int:
        skipped = 0

        if "alignment.paradigm" not in self._record_param_dirty_keys:
            slug = _nested_get(snapshot, ("alignment", "trial_slug"))
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
            alignment = snapshot.get("alignment", {})
            self._alignment_method_drafts = {}
            inactive = alignment.get("trial_params_by_slug")
            combo = self._alignment_method_combo
            if isinstance(inactive, dict) and inactive and combo is not None:
                for paradigm in self._alignment_paradigms:
                    slug = paradigm["slug"]
                    saved = inactive.get(slug)
                    if (
                        isinstance(saved, dict)
                        and combo.findData(saved.get("method")) >= 0
                    ):
                        draft = _restore_method_draft(paradigm, saved, saved["method"])
                        if draft is not None:
                            self._alignment_method_drafts[slug] = draft
            method = alignment.get("method")
            if isinstance(method, str) and combo is not None:
                index = combo.findData(method)
                combo.blockSignals(True)
                try:
                    combo.setCurrentIndex(max(index, 0))
                finally:
                    combo.blockSignals(False)
                current = self._current_alignment_paradigm()
                if isinstance(current, dict):
                    draft = _restore_method_draft(
                        current, alignment, combo.currentData()
                    )
                    slug = current.get("slug", alignment.get("trial_slug"))
                    if draft is not None and isinstance(slug, str):
                        self._alignment_method_drafts[slug] = draft
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
