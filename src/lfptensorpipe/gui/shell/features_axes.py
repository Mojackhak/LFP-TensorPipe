"""Features axis configuration MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    FEATURE_AUTO_BAND_METRICS,
    PathResolver,
    QDialog,
    pd,
)


class MainWindowFeaturesAxesMixin:
    @staticmethod
    def _features_metric_uses_auto_bands(metric_key: str) -> bool:
        return metric_key.strip().lower() in FEATURE_AUTO_BAND_METRICS

    def _features_auto_band_names_from_alignment_raw(
        self, metric_key: str
    ) -> list[str]:
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        if context is None or not isinstance(slug, str):
            return []
        path = PathResolver(context).alignment_root / slug / metric_key / "na-raw.pkl"
        if not path.exists():
            return []
        try:
            payload = self._load_pickle(path)
        except Exception:
            return []
        if not isinstance(payload, pd.DataFrame):
            return []
        if "Value" not in payload.columns:
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
        combo.blockSignals(True)
        combo.clear()
        for metric_key in metrics:
            combo.addItem(metric_key, metric_key)
        if isinstance(current, str):
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if combo.count() > 0 and combo.currentIndex() < 0:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

        valid_keys = set(metrics)
        self._features_axes_by_metric = {
            key: value
            for key, value in self._features_axes_by_metric.items()
            if key in valid_keys
        }
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
                else:
                    count_bands = len(
                        self._normalized_features_axes_for_metric(metric_key)["bands"]
                    )
                    self._features_axis_bands_button.setText(
                        f"Bands Configure... ({count_bands})"
                    )
                    self._features_axis_bands_button.setEnabled(True)
            else:
                self._features_axis_bands_button.setText("Bands Configure... (0)")
                self._features_axis_bands_button.setEnabled(False)
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
        if self._features_axis_apply_all_button is not None:
            self._features_axis_apply_all_button.setEnabled(has_metric)

    def _on_features_axis_metric_changed(self, _row: int) -> None:
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
            if not source_uses_auto_bands:
                target_axes["bands"] = [dict(item) for item in source_axes["bands"]]
            elif self._features_metric_uses_auto_bands(target_metric):
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
