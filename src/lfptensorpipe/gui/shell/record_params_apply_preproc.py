"""Record-parameter preprocess apply MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    QTableWidgetItem,
    _nested_get,
    normalize_filter_advance_params,
    normalize_preproc_viz_psd_params,
    normalize_preproc_viz_tfr_params,
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
                ok_advance, normalized_advance, _ = normalize_filter_advance_params(
                    advance
                )
                if ok_advance:
                    self._preproc_filter_advance_params = normalized_advance
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
        else:
            skipped += 1

        if "preproc.ecg" not in self._record_param_dirty_keys:
            method = _nested_get(snapshot, ("preproc", "ecg", "method"))
            if isinstance(method, str) and self._preproc_ecg_method_combo is not None:
                idx = self._preproc_ecg_method_combo.findData(method)
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
                ok_psd, normalized_psd, _ = normalize_preproc_viz_psd_params(psd_params)
                if ok_psd:
                    self._preproc_viz_psd_params = normalized_psd
            tfr_params = _nested_get(snapshot, ("preproc", "viz", "tfr_params"))
            if isinstance(tfr_params, dict):
                ok_tfr, normalized_tfr, _ = normalize_preproc_viz_tfr_params(tfr_params)
                if ok_tfr:
                    self._preproc_viz_tfr_params = normalized_tfr
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
