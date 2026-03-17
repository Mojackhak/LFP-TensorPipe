"""Preprocess apply/edit action MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    QDialog,
    preproc_step_raw_path,
)


class MainWindowPreprocActionsMixin:
    def _on_preproc_filter_advance(self) -> None:
        context = self._record_context()
        if context is None:
            self.statusBar().showMessage(
                "Filter Advance unavailable: select project/subject/record."
            )
            return

        default_params = self._load_filter_advance_defaults()
        default_basic_params = self._load_filter_basic_defaults()

        def _save_filter_defaults(advance_params: dict[str, Any]) -> None:
            notches, low_freq, high_freq = self._collect_filter_runtime_params()
            self._save_filter_advance_defaults(advance_params)
            self._save_filter_basic_defaults(
                {"notches": notches, "l_freq": low_freq, "h_freq": high_freq}
            )
            self.statusBar().showMessage(
                "Filter Advance defaults saved to app storage."
            )
            self._persist_record_params_snapshot(
                reason="preproc_filter_advance_default"
            )

        dialog = self._create_filter_advance_dialog(
            session_params=self._preproc_filter_advance_params,
            default_params=default_params,
            set_default_callback=_save_filter_defaults,
            parent=self,
        )
        if hasattr(dialog, "set_restore_callback"):
            dialog.set_restore_callback(
                lambda: self._apply_filter_basic_params_to_fields(default_basic_params)
            )
        if dialog.exec() != QDialog.Accepted:
            return
        if dialog.selected_params is None:
            return

        self._preproc_filter_advance_params = dict(dialog.selected_params)
        self.statusBar().showMessage("Filter Advance session parameters updated.")
        self._refresh_preproc_controls()
        self._persist_record_params_snapshot(reason="preproc_filter_advance_save")

    def _on_preproc_filter_apply(self) -> None:
        context = self._record_context()
        if context is None:
            self.statusBar().showMessage(
                "Filter Apply unavailable: select project/subject/record."
            )
            return
        try:
            notches, low_freq, high_freq = self._collect_filter_runtime_params()
        except Exception as exc:  # noqa: BLE001
            self._show_warning(
                "Filter Apply",
                f"Invalid filter parameters:\n{exc}",
            )
            self.statusBar().showMessage(
                f"Filter Apply failed: invalid parameters ({exc})"
            )
            return
        ok, message = self._run_with_busy(
            "Filter Apply",
            lambda: self._apply_filter_step_runtime(
                context,
                advance_params=self._preproc_filter_advance_params,
                notches=notches,
                l_freq=low_freq,
                h_freq=high_freq,
            ),
        )
        self._refresh_stage_states_from_context()
        self._refresh_preproc_controls()
        prefix = "Filter OK" if ok else "Filter failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="preproc_filter_apply")

    def _on_preproc_annotations_edit(self) -> None:
        if self._preproc_annotations_table is None:
            return
        context = self._record_context()
        dialog = self._create_annotation_configure_dialog(
            session_rows=self._collect_annotations_rows_for_params(),
            project_root=(context.project_root if context is not None else None),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self._preproc_annotations_table.blockSignals(True)
        self._preproc_annotations_table.setRowCount(0)
        self._append_annotation_rows(list(dialog.selected_rows))
        self._preproc_annotations_table.blockSignals(False)
        self._highlight_annotation_rows([])
        self._mark_record_param_dirty("preproc.annotations")
        self._refresh_preproc_controls()
        self.statusBar().showMessage("Annotations configured.")

    def _on_preproc_annotations_save(self) -> None:
        context = self._record_context()
        if context is None:
            self.statusBar().showMessage(
                "Annotations Apply unavailable: select project/subject/record."
            )
            return

        rows, invalid_rows = self._annotations_table_rows()
        self._highlight_annotation_rows(invalid_rows)
        if invalid_rows:
            self._mark_annotations_validation_failure(
                action="Apply",
                invalid_rows=invalid_rows,
            )
            self._show_warning(
                "Annotations Apply",
                "Invalid rows highlighted. Ensure description is non-empty, onset is numeric >= 0, and duration is numeric >= 0.",
            )
            self.statusBar().showMessage(
                "Annotations Apply failed: invalid rows highlighted."
            )
            self._refresh_preproc_controls()
            return

        clean_rows = [
            {
                "description": row["description"],
                "onset": row["onset"],
                "duration": row["duration"],
            }
            for row in rows
        ]
        ok, message = self._run_with_busy(
            "Annotations Apply",
            lambda: self._apply_annotations_step_runtime(
                context,
                rows=clean_rows,
            ),
        )
        self._refresh_stage_states_from_context()
        self._refresh_preproc_controls()
        prefix = "Annotations OK" if ok else "Annotations failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="preproc_annotations_apply")

    def _mark_annotations_validation_failure(
        self,
        *,
        action: str,
        invalid_rows: list[int],
    ) -> None:
        context = self._record_context()
        if context is None:
            return
        resolver = PathResolver(context)
        src = preproc_step_raw_path(resolver, "filter")
        dst = preproc_step_raw_path(resolver, "annotations")
        self._mark_preproc_step_runtime(
            resolver=resolver,
            step="annotations",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Annotations {action} blocked: invalid rows {invalid_rows}.",
        )

    def _on_preproc_annotations_import_csv(self) -> None:
        selected_file, _ = self._open_file_name(
            "Import Annotations CSV",
            (
                str(self._current_project)
                if self._current_project is not None
                else str(Path.home())
            ),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not selected_file:
            return
        ok, rows, message = self._load_annotations_csv_rows_runtime(Path(selected_file))
        if not ok:
            self._show_warning("Import CSV", message)
            return
        self._append_annotation_rows(rows)
        self._highlight_annotation_rows([])
        self._refresh_preproc_controls()
        self.statusBar().showMessage("Annotations CSV imported.")

    def _on_preproc_bad_segment_apply(self) -> None:
        context = self._record_context()
        if context is None:
            self.statusBar().showMessage(
                "Bad Segment Apply unavailable: select project/subject/record."
            )
            return
        ok, message = self._run_with_busy(
            "Bad Segment Apply",
            lambda: self._apply_bad_segment_step_runtime(context),
        )
        self._refresh_stage_states_from_context()
        self._refresh_preproc_controls()
        prefix = "Bad Segment OK" if ok else "Bad Segment failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="preproc_bad_segment_apply")

    def _on_preproc_ecg_apply(self) -> None:
        context = self._record_context()
        if context is None:
            self.statusBar().showMessage(
                "ECG Apply unavailable: select project/subject/record."
            )
            return
        method = "svd"
        if self._preproc_ecg_method_combo is not None:
            method = str(self._preproc_ecg_method_combo.currentData() or "svd")
        picks = list(self._preproc_ecg_selected_channels)
        if not picks:
            self._show_warning(
                "ECG Apply",
                "Select at least one ECG pick channel.",
            )
            return
        ok, message = self._run_with_busy(
            "ECG Apply",
            lambda: self._apply_ecg_step_runtime(
                context,
                method=method,
                picks=picks,
            ),
        )
        self._refresh_stage_states_from_context()
        self._refresh_preproc_controls()
        prefix = "ECG OK" if ok else "ECG failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="preproc_ecg_apply")

    def _on_preproc_finish_apply(self) -> None:
        context = self._record_context()
        if context is None:
            self.statusBar().showMessage(
                "Finish Apply unavailable: select project/subject/record."
            )
            return
        ok, message = self._run_with_busy(
            "Finish Apply",
            lambda: self._apply_finish_step_runtime(context),
        )
        self._refresh_stage_states_from_context()
        self._refresh_preproc_controls()
        prefix = "Finish OK" if ok else "Finish failed"
        self._post_step_action_sync(reason="preproc_finish_apply")
        if ok and hasattr(self, "_inherit_tensor_metric_notches_from_filter"):
            try:
                if self._inherit_tensor_metric_notches_from_filter(context):
                    self._persist_record_params_snapshot(
                        reason="preproc_finish_apply_tensor_notches"
                    )
            except Exception as exc:  # noqa: BLE001
                self._show_warning(
                    "Finish Apply",
                    f"Finish succeeded, but Tensor notch inheritance failed:\n{exc}",
                )
        self.statusBar().showMessage(f"{prefix}: {message}")
