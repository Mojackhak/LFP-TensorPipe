"""Record-parameter dirty-state MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    QCheckBox,
    QComboBox,
    QLineEdit,
    QTableWidget,
)


class MainWindowRecordParamsDirtyMixin:
    def _mark_record_param_dirty(self, key: str) -> None:
        if self._record_param_syncing:
            return
        self._record_param_dirty_keys.add(key)

    def _bind_record_param_dirty_signals(self) -> None:
        def bind_line(
            edit: QLineEdit | None,
            key: str,
            *,
            after_change=None,
        ) -> None:
            if edit is None:
                return
            edit.textEdited.connect(
                lambda _text, dirty_key=key: self._mark_record_param_dirty(dirty_key)
            )
            if after_change is not None:
                edit.textEdited.connect(lambda _text, callback=after_change: callback())

        def bind_combo(
            combo: QComboBox | None,
            key: str,
            *,
            after_change=None,
        ) -> None:
            if combo is None:
                return
            combo.currentIndexChanged.connect(
                lambda _idx, dirty_key=key: self._mark_record_param_dirty(dirty_key)
            )
            if after_change is not None:
                combo.activated.connect(lambda _idx, callback=after_change: callback())

        def bind_check(
            box: QCheckBox | None,
            key: str,
            *,
            after_change=None,
        ) -> None:
            if box is None:
                return
            box.stateChanged.connect(
                lambda _state, dirty_key=key: self._mark_record_param_dirty(dirty_key)
            )
            if after_change is not None:
                box.clicked.connect(lambda _checked, callback=after_change: callback())

        def bind_table(
            table: QTableWidget | None,
            key: str,
            *,
            after_change=None,
        ) -> None:
            if table is None:
                return
            table.cellChanged.connect(
                lambda _row, _col, dirty_key=key: self._mark_record_param_dirty(
                    dirty_key
                )
            )
            if after_change is not None:
                table.cellChanged.connect(
                    lambda _row, _col, callback=after_change: callback()
                )

        bind_line(
            self._preproc_filter_notches_edit,
            "preproc.filter",
            after_change=self._refresh_preproc_controls,
        )
        bind_line(
            self._preproc_filter_low_freq_edit,
            "preproc.filter",
            after_change=self._refresh_preproc_controls,
        )
        bind_line(
            self._preproc_filter_high_freq_edit,
            "preproc.filter",
            after_change=self._refresh_preproc_controls,
        )
        bind_table(
            self._preproc_annotations_table,
            "preproc.annotations",
            after_change=self._refresh_preproc_controls,
        )
        bind_combo(
            self._preproc_ecg_method_combo,
            "preproc.ecg",
            after_change=self._refresh_preproc_controls,
        )
        bind_combo(self._preproc_viz_step_combo, "preproc.viz")

        bind_line(
            self._tensor_low_freq_edit,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_line(
            self._tensor_high_freq_edit,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_line(
            self._tensor_step_edit,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_line(
            self._tensor_time_resolution_edit,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_line(
            self._tensor_hop_edit,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_line(
            self._tensor_freq_range_edit,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_line(
            self._tensor_percentile_edit,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_line(
            self._tensor_min_cycles_basic_edit,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_combo(
            self._tensor_method_combo,
            "tensor.metric_params",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        bind_check(
            self._tensor_mask_edge_checkbox,
            "tensor.mask_edge_effects",
            after_change=self._refresh_tensor_metric_indicators_from_draft,
        )
        for checkbox in self._tensor_metric_checks.values():
            checkbox.stateChanged.connect(
                lambda _state: self._mark_record_param_dirty("tensor.selected_metrics")
            )

        bind_combo(self._alignment_method_combo, "alignment.method")
        bind_line(self._alignment_n_samples_edit, "alignment.sample_rate")
        bind_combo(self._alignment_epoch_metric_combo, "alignment.epoch_metric")
        bind_combo(self._alignment_epoch_channel_combo, "alignment.epoch_channel")
        bind_table(self._alignment_epoch_table, "alignment.picks")

        bind_line(self._features_filter_feature_edit, "features.filters")
        bind_combo(self._features_axis_metric_combo, "features.axes")
        bind_combo(self._features_subset_band_combo, "features.subset")
        bind_combo(self._features_subset_channel_combo, "features.subset")
        bind_combo(self._features_subset_region_combo, "features.subset")
        bind_line(self._features_x_label_edit, "features.plot_labels")
        bind_line(self._features_y_label_edit, "features.plot_labels")
        bind_line(self._features_cbar_label_edit, "features.plot_labels")
