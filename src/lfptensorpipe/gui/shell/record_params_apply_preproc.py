"""Record-parameter preprocess apply MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    QTableWidgetItem,
    _nested_get,
    default_filter_advance_params,
    default_preproc_viz_psd_params,
    default_preproc_viz_tfr_params,
    default_ecg_params_by_method,
    default_ecg_review_params,
    normalize_ecg_review_params,
)


class MainWindowRecordParamsApplyPreprocMixin:
    def _apply_record_params_preproc_snapshot(self, snapshot: dict[str, Any]) -> int:
        skipped = 0

        if "preproc.filter" not in self._record_param_dirty_keys:
            basic = _nested_get(snapshot, ("preproc", "filter", "basic"))
            if isinstance(basic, dict):
                self._apply_filter_basic_params_to_fields(basic)
            advance = _nested_get(snapshot, ("preproc", "filter", "advance"))
            if isinstance(advance, dict):
                defaults = default_filter_advance_params()
                self._preproc_filter_advance_params = {
                    key: advance[key] if key in advance else value
                    for key, value in defaults.items()
                }
        else:
            skipped += 1

        if "preproc.annotations" not in self._record_param_dirty_keys:
            rows = _nested_get(snapshot, ("preproc", "annotations", "rows"))
            if isinstance(rows, list) and self._preproc_annotations_table is not None:
                self._preproc_annotations_table.blockSignals(True)
                self._preproc_annotations_table.setRowCount(0)
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    row_idx = self._preproc_annotations_table.rowCount()
                    self._preproc_annotations_table.insertRow(row_idx)
                    self._preproc_annotations_table.setItem(
                        row_idx,
                        0,
                        QTableWidgetItem(str(row.get("description", ""))),
                    )
                    self._preproc_annotations_table.setItem(
                        row_idx,
                        1,
                        QTableWidgetItem(str(row.get("onset", ""))),
                    )
                    self._preproc_annotations_table.setItem(
                        row_idx,
                        2,
                        QTableWidgetItem(str(row.get("duration", ""))),
                    )
                self._preproc_annotations_table.blockSignals(False)
                self._highlight_annotation_rows([])
            mark_filter_edges = _nested_get(
                snapshot,
                ("preproc", "annotations", "mark_filter_edges"),
            )
            if isinstance(mark_filter_edges, bool):
                self._preproc_annotations_mark_filter_edges = mark_filter_edges
        else:
            skipped += 1

        if "preproc.ecg" not in self._record_param_dirty_keys:
            params_by_method = _nested_get(
                snapshot,
                ("preproc", "ecg", "params_by_method"),
            )
            defaults_by_method = default_ecg_params_by_method()
            self._preproc_ecg_params_by_method = {}
            for method_key, defaults in defaults_by_method.items():
                raw_method = (
                    params_by_method.get(method_key)
                    if isinstance(params_by_method, dict)
                    else None
                )
                candidate = dict(defaults)
                if isinstance(raw_method, dict):
                    candidate.update(
                        {key: raw_method[key] for key in defaults if key in raw_method}
                    )
                self._preproc_ecg_params_by_method[method_key] = candidate
            review_params = _nested_get(snapshot, ("preproc", "ecg", "review"))
            valid_review, normalized_review, review_message = (
                normalize_ecg_review_params(review_params)
            )
            if valid_review:
                self._preproc_ecg_review_params = normalized_review
            else:
                self._preproc_ecg_review_params = default_ecg_review_params()
                self._show_ecg_params_warning_once(
                    "Invalid record ECG review parameters were replaced in memory: "
                    f"{review_message}"
                )
            method = _nested_get(snapshot, ("preproc", "ecg", "method"))
            if isinstance(method, str) and self._preproc_ecg_method_combo is not None:
                idx = self._preproc_ecg_method_combo.findData(method)
                if idx < 0:
                    idx = self._preproc_ecg_method_combo.findData("svd")
                    self._show_ecg_params_warning_once(
                        "Unknown record ECG method was replaced in memory: " f"{method}"
                    )
                if idx < 0:
                    idx = 0
                self._preproc_ecg_method_combo.setCurrentIndex(idx)
            selected_channels = _nested_get(
                snapshot, ("preproc", "ecg", "selected_channels")
            )
            if isinstance(selected_channels, list):
                self._preproc_ecg_selected_channels = tuple(
                    str(item) for item in selected_channels if str(item).strip()
                )
        else:
            skipped += 1

        if "preproc.viz" not in self._record_param_dirty_keys:
            psd_params = _nested_get(snapshot, ("preproc", "viz", "psd_params"))
            if isinstance(psd_params, dict):
                defaults = default_preproc_viz_psd_params()
                self._preproc_viz_psd_params = {
                    key: psd_params[key] if key in psd_params else value
                    for key, value in defaults.items()
                }
            tfr_params = _nested_get(snapshot, ("preproc", "viz", "tfr_params"))
            if isinstance(tfr_params, dict):
                defaults = default_preproc_viz_tfr_params()
                self._preproc_viz_tfr_params = {
                    key: tfr_params[key] if key in tfr_params else value
                    for key, value in defaults.items()
                }
            step = _nested_get(snapshot, ("preproc", "viz", "selected_step"))
            if isinstance(step, str):
                self._preproc_viz_last_step = step
            viz_channels = _nested_get(
                snapshot, ("preproc", "viz", "selected_channels")
            )
            if isinstance(viz_channels, list):
                self._preproc_viz_selected_channels = tuple(
                    str(item) for item in viz_channels if str(item).strip()
                )
        else:
            skipped += 1

        return skipped
