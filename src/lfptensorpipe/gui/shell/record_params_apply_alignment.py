"""Record-parameter alignment apply MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    Qt,
)


def _restore_method_draft(paradigm, saved):
    method = saved["method"]
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
        alignment = snapshot.get("alignment", {})
        owner = alignment.get("trial_slug")

        if "alignment.paradigm" not in self._record_param_dirty_keys:
            if (
                isinstance(owner, str)
                and owner
                and self._alignment_paradigm_list is not None
            ):
                for idx in range(self._alignment_paradigm_list.count()):
                    item = self._alignment_paradigm_list.item(idx)
                    if item is None:
                        continue
                    if item.data(Qt.UserRole) == owner:
                        self._alignment_paradigm_list.setCurrentRow(idx)
                        break
        else:
            skipped += 1

        current = self._current_alignment_paradigm()
        owns_active = bool(
            owner and isinstance(current, dict) and current.get("slug", owner) == owner
        )
        if "alignment.method" not in self._record_param_dirty_keys:
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
                        draft = _restore_method_draft(paradigm, saved)
                        if draft is not None:
                            self._alignment_method_drafts[slug] = draft
            active_draft = (
                alignment
                if owns_active
                else self._alignment_method_drafts.get(
                    current.get("slug") if current else None, {}
                )
            )
            method = active_draft.get("method")
            index = (
                combo.findData(method)
                if isinstance(method, str) and combo is not None
                else -1
            )
            if index >= 0:
                combo.blockSignals(True)
                try:
                    combo.setCurrentIndex(index)
                finally:
                    combo.blockSignals(False)
                if owns_active:
                    draft = _restore_method_draft(current, alignment)
                    if draft is not None:
                        self._alignment_method_drafts[owner] = draft
        else:
            skipped += 1

        # A trial-owned Preview takes precedence over older one-channel record settings.
        metric = alignment.get("epoch_metric")
        if (
            isinstance(metric, str)
            and owns_active
            and not self._epoch_review.get("preview")
        ):
            self._epoch_review["preview"] = {
                "metric": metric,
                "channel_index": alignment.get("epoch_channel"),
            }

        if "alignment.picks" not in self._record_param_dirty_keys:
            picks = alignment.get("picked_epoch_indices")
            if (
                owns_active
                and isinstance(picks, list)
                and self._alignment_epoch_table is not None
            ):
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
