"""Features axis configuration MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app.alignment.generation import accepted_alignment_artifact_paths
from lfptensorpipe.app.features.derive_axes import feature_band_support_error
from lfptensorpipe.gui.shell.common import (
    FEATURE_AUTO_BAND_METRICS,
    Any,
    PathResolver,
    QDialog,
    np,
    pd,
    set_control_validation_error,
    tensor_metric_log_path,
)


class MainWindowFeaturesAxesMixin:
    @staticmethod
    def _features_metric_uses_auto_bands(metric_key: str) -> bool:
        return metric_key.strip().lower() in FEATURE_AUTO_BAND_METRICS

    def _features_alignment_raw_payload(self, metric_key: str) -> pd.DataFrame | None:
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        if context is None or not isinstance(slug, str):
            return None
        resolver = PathResolver(context)
        accepted_paths = dict(
            accepted_alignment_artifact_paths(
                resolver,
                trial_slug=slug,
                stage="finish",
            )
        )
        path = accepted_paths.get(metric_key)
        if path is None or not path.is_file():
            return None
        try:
            payload = self._load_pickle(path)
        except Exception:
            return None
        if not isinstance(payload, pd.DataFrame):
            return None
        if "Value" not in payload.columns:
            return None
        return payload

    def _features_auto_band_names_from_alignment_raw(
        self, metric_key: str
    ) -> list[str]:
        payload = self._features_alignment_raw_payload(metric_key)
        if payload is None:
            return []
        source: Any = None
        for item in payload["Value"].tolist():
            if isinstance(item, (pd.Series, pd.DataFrame)):
                source = item
                break
        if source is None:
            return []
        names: list[str] = []
        seen: set[str] = set()
        for value in source.index.unique().tolist():
            name = str(value).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def _features_frequency_support_for_accepted_metric(
        self, metric_key: str
    ) -> tuple[float, float] | None:
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        if context is None or not isinstance(slug, str):
            return None
        resolver = PathResolver(context)
        accepted_metrics = {
            accepted_metric
            for accepted_metric, _path in accepted_alignment_artifact_paths(
                resolver,
                trial_slug=slug,
                stage="finish",
            )
        }
        if metric_key not in accepted_metrics:
            return None
        params = self._read_completed_log_params(
            tensor_metric_log_path(resolver, metric_key)
        )
        try:
            low = float(params["low_freq"])
            high = float(params["high_freq"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(low) or not np.isfinite(high) or high < low:
            return None
        return low, high

    def _features_manual_band_support_error(
        self,
        metric_key: str,
        bands: list[dict[str, Any]],
    ) -> str:
        support = self._features_frequency_support_for_accepted_metric(metric_key)
        if support is None:
            return ""
        return feature_band_support_error(metric_key, bands, support)

    def _normalized_features_axes_for_metric(
        self, metric_key: str
    ) -> dict[str, list[dict[str, Any]]]:
        node = self._features_axes_by_metric.get(metric_key, {})
        if not isinstance(node, dict):
            node = {}
        raw_bands: Any = node.get("bands")
        raw_times: Any = node.get("times")
        if self._features_metric_uses_auto_bands(metric_key):
            raw_bands = []
        elif not isinstance(raw_bands, list):
            raw_bands = self._load_features_axis_defaults(
                metric_key=metric_key,
                axis_key="bands",
            )
        if not isinstance(raw_times, list):
            raw_times = self._load_features_axis_defaults(
                metric_key=metric_key,
                axis_key="times",
            )
        bands = self._normalize_feature_axis_rows(
            raw_bands,
            min_start=0.0,
            max_end=None,
            allow_duplicate_names=False,
        )
        times = self._normalize_feature_axis_rows(
            raw_times,
            min_start=0.0,
            max_end=100.0,
            allow_duplicate_names=True,
        )
        normalized = {
            "bands": [dict(item) for item in bands],
            "times": [dict(item) for item in times],
        }
        self._features_axes_by_metric[metric_key] = normalized
        return normalized

    def _refresh_features_axis_metric_combo(self) -> None:
        combo = self._features_axis_metric_combo
        if combo is None:
            return
        metrics = self._features_metric_keys_for_selected_trial()
        current = combo.currentData()
        preferred = current if isinstance(current, str) else ""
        if not preferred:
            slug = self._shared_stage_trial_slug()
            cached = (
                self._features_trial_params_by_slug.get(slug)
                if isinstance(slug, str)
                else None
            )
            if isinstance(cached, dict):
                preferred = str(cached.get("active_metric", "")).strip()
        combo.blockSignals(True)
        combo.clear()
        for metric_key in metrics:
            combo.addItem(metric_key, metric_key)
        if preferred:
            idx = combo.findData(preferred)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if combo.count() > 0 and combo.currentIndex() < 0:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

        for metric_key in metrics:
            self._normalized_features_axes_for_metric(metric_key)
        self._refresh_features_axis_buttons()

    def _current_features_axis_metric(self) -> str | None:
        if self._features_axis_metric_combo is None:
            return None
        metric_key = self._features_axis_metric_combo.currentData()
        if not isinstance(metric_key, str) or not metric_key:
            return None
        return metric_key

    def _refresh_features_axis_buttons(self) -> None:
        metric_key = self._current_features_axis_metric()
        has_metric = isinstance(metric_key, str)
        if self._features_axis_bands_button is not None:
            if has_metric and metric_key is not None:
                if self._features_metric_uses_auto_bands(metric_key):
                    count_bands = len(
                        self._features_auto_band_names_from_alignment_raw(metric_key)
                    )
                    self._features_axis_bands_button.setText(
                        f"Bands Auto ({count_bands})"
                    )
                    self._features_axis_bands_button.setEnabled(False)
                    set_control_validation_error(self._features_axis_bands_button, None)
                else:
                    bands = self._normalized_features_axes_for_metric(metric_key)[
                        "bands"
                    ]
                    count_bands = len(bands)
                    support_error = self._features_manual_band_support_error(
                        metric_key,
                        bands,
                    )
                    validation_error = support_error or (
                        None if count_bands else "At least one band is required."
                    )
                    self._features_axis_bands_button.setText(
                        f"Bands Configure... ({count_bands})"
                    )
                    self._features_axis_bands_button.setEnabled(True)
                    set_control_validation_error(
                        self._features_axis_bands_button,
                        validation_error,
                    )
            else:
                self._features_axis_bands_button.setText("Bands Configure... (0)")
                self._features_axis_bands_button.setEnabled(False)
                set_control_validation_error(self._features_axis_bands_button, None)
        if self._features_axis_times_button is not None:
            count_times = (
                len(self._normalized_features_axes_for_metric(metric_key)["times"])
                if has_metric and metric_key is not None
                else 0
            )
            self._features_axis_times_button.setText(
                f"Phases Configure... ({count_times})"
            )
            self._features_axis_times_button.setEnabled(has_metric)
            set_control_validation_error(
                self._features_axis_times_button,
                (
                    None
                    if not has_metric or count_times
                    else "At least one phase is required."
                ),
            )
        if self._features_axis_apply_all_button is not None:
            self._features_axis_apply_all_button.setEnabled(has_metric)

    def _on_features_axis_metric_changed(self, _row: int) -> None:
        slug = self._shared_stage_trial_slug()
        active_metric = self._current_features_axis_metric()
        cached = (
            self._features_trial_params_by_slug.get(slug)
            if isinstance(slug, str)
            else None
        )
        if isinstance(cached, dict) and isinstance(active_metric, str):
            cached["active_metric"] = active_metric
        self._refresh_features_axis_buttons()
        self._refresh_features_controls()

    def _on_features_axis_bands(self) -> None:
        metric_key = self._current_features_axis_metric()
        if metric_key is None:
            return
        if self._features_metric_uses_auto_bands(metric_key):
            self.statusBar().showMessage(
                f"{metric_key} bands inherit Value index from alignment na-raw.pkl."
            )
            return
        axes = self._normalized_features_axes_for_metric(metric_key)
        default_rows = self._load_features_axis_defaults(
            metric_key=metric_key,
            axis_key="bands",
        )
        dialog = self._create_feature_axis_configure_dialog(
            title=f"{metric_key} Bands",
            item_label="Band",
            current_rows=tuple(dict(item) for item in axes["bands"]),
            default_rows=tuple(dict(item) for item in default_rows),
            set_default_callback=lambda rows: self._save_features_axis_defaults(
                metric_key=metric_key,
                axis_key="bands",
                rows=[dict(item) for item in rows],
            ),
            min_start=0.0,
            max_end=None,
            allow_duplicate_names=False,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        axes["bands"] = [dict(item) for item in dialog.selected_rows]
        self._features_axes_by_metric[metric_key] = axes
        if dialog.selected_action == "set_default":
            self.statusBar().showMessage(f"{metric_key} bands defaults saved.")
        self._mark_record_param_dirty("features.axes")
        self._refresh_features_axis_buttons()
        self._refresh_features_controls()

    def _on_features_axis_times(self) -> None:
        metric_key = self._current_features_axis_metric()
        if metric_key is None:
            return
        axes = self._normalized_features_axes_for_metric(metric_key)
        default_rows = self._load_features_axis_defaults(
            metric_key=metric_key,
            axis_key="times",
        )
        dialog = self._create_feature_axis_configure_dialog(
            title=f"{metric_key} Phases",
            item_label="Phase",
            current_rows=tuple(dict(item) for item in axes["times"]),
            default_rows=tuple(dict(item) for item in default_rows),
            set_default_callback=lambda rows: self._save_features_axis_defaults(
                metric_key=metric_key,
                axis_key="times",
                rows=[dict(item) for item in rows],
            ),
            min_start=0.0,
            max_end=100.0,
            allow_duplicate_names=True,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        axes["times"] = [dict(item) for item in dialog.selected_rows]
        self._features_axes_by_metric[metric_key] = axes
        if dialog.selected_action == "set_default":
            self.statusBar().showMessage(f"{metric_key} phases defaults saved.")
        self._mark_record_param_dirty("features.axes")
        self._refresh_features_axis_buttons()
        self._refresh_features_controls()

    def _on_features_axis_apply_all(self) -> None:
        metric_key = self._current_features_axis_metric()
        if metric_key is None:
            return
        source_axes = self._normalized_features_axes_for_metric(metric_key)
        source_uses_auto_bands = self._features_metric_uses_auto_bands(metric_key)
        for target_metric in self._features_metric_keys_for_selected_trial():
            target_axes = self._normalized_features_axes_for_metric(target_metric)
            target_uses_auto_bands = self._features_metric_uses_auto_bands(
                target_metric
            )
            if not source_uses_auto_bands and not target_uses_auto_bands:
                target_axes["bands"] = [dict(item) for item in source_axes["bands"]]
            elif target_uses_auto_bands:
                target_axes["bands"] = []
            target_axes["times"] = [dict(item) for item in source_axes["times"]]
            self._features_axes_by_metric[target_metric] = target_axes
        self._mark_record_param_dirty("features.axes")
        self._refresh_features_axis_buttons()
        self._refresh_features_controls()
        self.statusBar().showMessage(
            f"Applied axis selection of {metric_key} to all metrics."
        )

    def _validate_features_axes_for_run(
        self,
        metric_keys: list[str],
    ) -> tuple[bool, str]:
        for metric_key in metric_keys:
            axes = self._normalized_features_axes_for_metric(metric_key)
            if self._features_metric_uses_auto_bands(metric_key):
                if not self._features_auto_band_names_from_alignment_raw(metric_key):
                    return (
                        False,
                        f"{metric_key}: no band labels found from na-raw.pkl Value index.",
                    )
            elif not axes["bands"]:
                return False, f"{metric_key}: configure at least one band interval."
            else:
                support_error = self._features_manual_band_support_error(
                    metric_key,
                    axes["bands"],
                )
                if support_error:
                    return False, support_error
            if not axes["times"]:
                return False, f"{metric_key}: configure at least one phase interval."
        return True, ""

    def _collect_features_axes_for_run(self, metric_keys: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric_key in metric_keys:
            axes = self._normalized_features_axes_for_metric(metric_key)
            bands = (
                []
                if self._features_metric_uses_auto_bands(metric_key)
                else [dict(item) for item in axes["bands"]]
            )
            out[metric_key] = {
                "bands": bands,
                "times": [dict(item) for item in axes["times"]],
            }
        return out
