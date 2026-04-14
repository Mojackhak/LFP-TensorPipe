"""Features run and file-refresh MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    QTableWidgetItem,
    pd,
)


class MainWindowFeaturesRunMixin:
    def _refresh_features_controls(self) -> None:
        self._refresh_features_indicators()
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        has_context = context is not None
        has_slug = isinstance(slug, str)
        alignment_green = self._stage_states.get("alignment") == "green"
        metric_keys = self._features_metric_keys_for_selected_trial()
        metrics_ready = bool(metric_keys)
        subset_ready = False
        if metrics_ready:
            subset_ready, _ = self._validate_features_axes_for_run(metric_keys)
        selected_file = self._selected_features_file()

        if self._features_run_extract_button is not None:
            self._features_run_extract_button.setEnabled(
                has_context
                and has_slug
                and alignment_green
                and metrics_ready
                and subset_ready
            )
        if self._features_import_button is not None:
            self._features_import_button.setEnabled(has_context and has_slug)
        if self._features_export_button is not None:
            self._features_export_button.setEnabled(has_context and has_slug)
        if self._features_refresh_button is not None:
            self._features_refresh_button.setEnabled(has_context and has_slug)
        self._refresh_features_axis_buttons()
        if self._features_plot_button is not None:
            self._features_plot_button.setEnabled(selected_file is not None)
        if self._features_plot_advance_button is not None:
            self._features_plot_advance_button.setEnabled(selected_file is not None)
        if self._features_plot_export_button is not None:
            self._features_plot_export_button.setEnabled(
                self._features_last_plot_figure is not None
                and isinstance(self._features_last_plot_data, pd.DataFrame)
                and not self._features_last_plot_data.empty
            )
        self._refresh_features_subset_options()

    def _on_features_run_extract(self) -> None:
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        if context is None or slug is None:
            self.statusBar().showMessage(
                "Extract Features unavailable: select context and trial."
            )
            return
        metric_keys = self._features_metric_keys_for_selected_trial()
        if not metric_keys:
            self._show_warning(
                "Extract Features",
                "No alignment raw tables found. Run Align Epochs Finish first.",
            )
            return
        ok_axes, message_axes = self._validate_features_axes_for_run(metric_keys)
        if not ok_axes:
            self._show_warning("Extract Features", message_axes)
            return
        axes_by_metric = self._collect_features_axes_for_run(metric_keys)
        ok, message = self._run_with_busy(
            "Extract Features",
            lambda: self._run_extract_features_runtime(
                context,
                paradigm_slug=slug,
                config_store=self._config_store,
                axes_by_metric=axes_by_metric,
            ),
        )
        self._refresh_stage_states_from_context()
        self._refresh_features_indicators()
        self._refresh_features_available_files()
        self._refresh_features_controls()
        prefix = "Extract Features OK" if ok else "Extract Features failed"
        self.statusBar().showMessage(f"{prefix}: {message}")
        self._post_step_action_sync(reason="features_extract_run")

    def _on_features_refresh_files(self) -> None:
        self._refresh_features_available_files()
        self._refresh_features_controls()
        self.statusBar().showMessage("Available features refreshed.")

    def _refresh_features_available_files(
        self,
        *_: Any,
        preferred_relative_stem: str | None = None,
    ) -> None:
        if self._features_available_table is None:
            return
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        files: list[dict[str, Any]] = []
        if context is not None and isinstance(slug, str):
            resolver = PathResolver(context)
            root = resolver.features_root / slug
            if root.exists():
                dedupe_rows: set[tuple[str, str, str]] = set()
                for path in sorted(root.glob("**/*.pkl")):
                    try:
                        payload = self._load_pickle(path)
                    except Exception:
                        continue
                    if not isinstance(payload, pd.DataFrame):
                        continue
                    if "Value" not in payload.columns:
                        continue
                    rel = path.relative_to(root)
                    metric_key = rel.parts[0] if len(rel.parts) > 1 else ""
                    derived_type = self._parse_derived_type_from_stem(path.stem)
                    if not derived_type:
                        continue
                    reducer = self._parse_reducer_from_stem(path.stem, derived_type)
                    dedupe_key = (metric_key, derived_type, reducer)
                    if dedupe_key in dedupe_rows:
                        continue
                    dedupe_rows.add(dedupe_key)
                    files.append(
                        {
                            "feature": derived_type,
                            "property": reducer,
                            "display_feature": derived_type,
                            "stem": path.stem,
                            "relative_stem": str(rel.with_suffix("")),
                            "derived_type": derived_type,
                            "metric": metric_key,
                            "path": path,
                        }
                    )
        files.sort(
            key=lambda item: (
                str(item.get("metric", "")).lower(),
                str(item.get("feature", "")).lower(),
                str(item.get("property", "")).lower(),
            )
        )
        self._features_files = files

        feature_filter = (
            self._features_filter_feature_edit.text().strip().lower()
            if self._features_filter_feature_edit is not None
            else ""
        )

        filtered: list[dict[str, Any]] = []
        for row in files:
            haystack = " ".join(
                [
                    str(row.get("metric", "")),
                    str(row.get("feature", "")),
                    str(row.get("property", "")),
                    str(row.get("stem", "")),
                ]
            ).lower()
            if feature_filter and feature_filter not in haystack:
                continue
            filtered.append(row)
        selected = self._selected_features_file()
        selected_relative_stem = (
            str(preferred_relative_stem).strip()
            if isinstance(preferred_relative_stem, str)
            else ""
        )
        if not selected_relative_stem and isinstance(selected, dict):
            selected_relative_stem = str(selected.get("relative_stem", "")).strip()
        selected_path = (
            Path(selected["path"])
            if isinstance(selected, dict) and "path" in selected
            else None
        )
        self._features_filtered_files = filtered
        self._features_available_table.blockSignals(True)
        self._features_available_table.setRowCount(len(filtered))
        for row_idx, row in enumerate(filtered):
            self._features_available_table.setItem(
                row_idx, 0, QTableWidgetItem(str(row.get("metric", "")))
            )
            self._features_available_table.setItem(
                row_idx, 1, QTableWidgetItem(str(row.get("feature", "")))
            )
            self._features_available_table.setItem(
                row_idx, 2, QTableWidgetItem(str(row.get("property", "")))
            )
        self._features_available_table.clearSelection()
        restored_selection = False
        if selected_relative_stem:
            for row_idx, row in enumerate(filtered):
                if str(row.get("relative_stem", "")).strip() == selected_relative_stem:
                    self._features_available_table.setCurrentCell(row_idx, 0)
                    restored_selection = True
                    break
        if not restored_selection and selected_path is not None:
            for row_idx, row in enumerate(filtered):
                if Path(row["path"]) == selected_path:
                    self._features_available_table.setCurrentCell(row_idx, 0)
                    break
        self._features_available_table.blockSignals(False)
        self._apply_features_plot_label_placeholders()
        self._refresh_features_controls()
