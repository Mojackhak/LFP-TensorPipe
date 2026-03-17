"""Features config import/export MainWindow methods."""

from __future__ import annotations

import json

from lfptensorpipe.gui.shell.common import Any, Path, PathResolver, RecordContext

FEATURES_CONFIG_SCHEMA = "lfptensorpipe.features-config"
FEATURES_CONFIG_VERSION = 1
FEATURES_CONFIG_FILE_NAME = "lfptensorpipe_features_config.json"


class MainWindowFeaturesConfigMixin:
    @staticmethod
    def _features_config_json_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return MainWindowFeaturesConfigMixin._features_config_json_value(
                    item_method()
                )
            except Exception:
                pass
        if isinstance(value, dict):
            return {
                str(key): MainWindowFeaturesConfigMixin._features_config_json_value(
                    item
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                MainWindowFeaturesConfigMixin._features_config_json_value(item)
                for item in value
            ]
        raise TypeError(f"Unsupported features config value: {type(value).__name__}")

    @staticmethod
    def _features_config_slug_token(slug: str) -> str:
        token = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in str(slug).strip()
        )
        return token.strip("-_")

    def _features_config_default_path(
        self,
        context: RecordContext,
        *,
        trial_slug: str,
    ) -> Path:
        token = self._features_config_slug_token(trial_slug)
        filename = (
            FEATURES_CONFIG_FILE_NAME
            if not token
            else f"lfptensorpipe_features_{token}_config.json"
        )
        return PathResolver(context).lfp_root / filename

    def _build_features_config_export_payload(self) -> dict[str, Any]:
        metric_keys = self._features_metric_keys_for_selected_trial()
        active_metric = self._current_features_axis_metric() or ""
        if active_metric not in metric_keys:
            active_metric = metric_keys[0] if metric_keys else ""
        axes_by_metric: dict[str, dict[str, Any]] = {}
        for metric_key in metric_keys:
            axes = self._normalized_features_axes_for_metric(metric_key)
            axes_by_metric[metric_key] = {
                "bands": (
                    []
                    if self._features_metric_uses_auto_bands(metric_key)
                    else [dict(item) for item in axes["bands"]]
                ),
                "times": [dict(item) for item in axes["times"]],
            }
        return {
            "schema": FEATURES_CONFIG_SCHEMA,
            "version": FEATURES_CONFIG_VERSION,
            "features": {
                "active_metric": active_metric,
                "axes_by_metric": self._features_config_json_value(axes_by_metric),
            },
        }

    def _normalize_features_config_import_payload(
        self,
        payload: dict[str, Any],
        *,
        available_metrics: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(payload, dict):
            raise ValueError("Features config must be a JSON object.")
        if payload.get("schema") != FEATURES_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported features config schema: {payload.get('schema')!r}."
            )
        if payload.get("version") != FEATURES_CONFIG_VERSION:
            raise ValueError(
                f"Unsupported features config version: {payload.get('version')!r}."
            )

        features_node = payload.get("features")
        if not isinstance(features_node, dict):
            raise ValueError("Features config is missing required `features` object.")

        axes_by_metric = features_node.get("axes_by_metric")
        if not isinstance(axes_by_metric, dict):
            raise ValueError(
                "Features config is missing required `features.axes_by_metric` object."
            )

        warnings: list[str] = []
        available_set = set(available_metrics)
        ignored_metrics: list[str] = []
        auto_band_metrics: list[str] = []
        normalized_axes: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for metric_key_raw, axes_node in axes_by_metric.items():
            metric_key = str(metric_key_raw).strip()
            if not metric_key:
                continue
            if metric_key not in available_set:
                ignored_metrics.append(metric_key)
                continue
            if not isinstance(axes_node, dict):
                continue
            raw_bands = axes_node.get("bands")
            if self._features_metric_uses_auto_bands(metric_key):
                if isinstance(raw_bands, list) and raw_bands:
                    auto_band_metrics.append(metric_key)
                bands: list[dict[str, Any]] = []
            else:
                bands = [
                    dict(item)
                    for item in self._normalize_feature_axis_rows(
                        raw_bands,
                        min_start=0.0,
                        max_end=None,
                        allow_duplicate_names=False,
                    )
                ]
            times = [
                dict(item)
                for item in self._normalize_feature_axis_rows(
                    axes_node.get("times"),
                    min_start=0.0,
                    max_end=100.0,
                    allow_duplicate_names=True,
                )
            ]
            normalized_axes[metric_key] = {
                "bands": bands,
                "times": times,
            }

        if ignored_metrics:
            warnings.append(
                "Ignored unavailable metric(s): " + ", ".join(sorted(ignored_metrics))
            )
        if auto_band_metrics:
            warnings.append(
                "Ignored imported band rows for auto-band metric(s): "
                + ", ".join(sorted(set(auto_band_metrics)))
            )
        if not normalized_axes:
            raise ValueError(
                "Features config has no applicable metric config for the current selected trial."
            )

        active_metric = str(features_node.get("active_metric", "")).strip()
        if active_metric and active_metric not in normalized_axes:
            warnings.append(
                f"Ignored unavailable active metric selection: {active_metric}."
            )
            active_metric = ""
        if not active_metric:
            current_metric = self._current_features_axis_metric()
            if isinstance(current_metric, str) and current_metric in normalized_axes:
                active_metric = current_metric
            else:
                active_metric = next(
                    (
                        metric_key
                        for metric_key in available_metrics
                        if metric_key in normalized_axes
                    ),
                    next(iter(normalized_axes)),
                )

        return {
            "active_metric": active_metric,
            "axes_by_metric": normalized_axes,
        }, warnings

    def _on_features_export_config(self) -> None:
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        if (
            context is None
            or slug is None
            or self._features_paradigm_list is None
            or self._current_features_paradigm_slug() is None
        ):
            self._show_warning(
                "Export Configs",
                "Select project, subject, record, and one trial before exporting Features configs.",
            )
            return

        default_path = self._features_config_default_path(context, trial_slug=slug)
        file_path_text, _ = self._save_file_name(
            "Export Features Configs",
            str(default_path.resolve()),
            "JSON files (*.json);;All files (*)",
        )
        if not file_path_text:
            return

        export_path = Path(file_path_text)
        if not export_path.suffix:
            export_path = export_path.with_suffix(".json")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            payload = self._build_features_config_export_payload()
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Export Configs", f"Export failed:\n{exc}")
            return

        self.statusBar().showMessage(f"Exported Features config: {export_path.name}")

    def _on_features_import_config(self) -> None:
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        if (
            context is None
            or slug is None
            or self._features_paradigm_list is None
            or self._current_features_paradigm_slug() is None
        ):
            self._show_warning(
                "Import Configs",
                "Select project, subject, record, and one trial before importing Features configs.",
            )
            return

        default_path = self._features_config_default_path(context, trial_slug=slug)
        file_path_text, _ = self._open_file_name(
            "Import Features Configs",
            str(default_path.parent.resolve()),
            "JSON files (*.json);;All files (*)",
        )
        if not file_path_text:
            return

        import_path = Path(file_path_text)
        available_metrics = self._features_metric_keys_for_selected_trial()
        try:
            with import_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            features_snapshot, warnings = (
                self._normalize_features_config_import_payload(
                    payload,
                    available_metrics=available_metrics,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Import Configs", f"Import failed:\n{exc}")
            return

        self._features_axes_by_metric = {
            str(key): {
                "bands": [dict(item) for item in value.get("bands", [])],
                "times": [dict(item) for item in value.get("times", [])],
            }
            for key, value in features_snapshot["axes_by_metric"].items()
            if isinstance(value, dict)
        }
        self._refresh_features_axis_metric_combo()
        combo = self._features_axis_metric_combo
        if combo is not None:
            active_metric = str(features_snapshot.get("active_metric", "")).strip()
            idx = combo.findData(active_metric)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._mark_record_param_dirty("features.axes")
        self._refresh_features_controls()

        persisted = self._persist_record_params_snapshot(
            reason="features_import_config"
        )
        self.statusBar().showMessage(f"Imported Features config: {import_path.name}")
        if warnings:
            self._show_information(
                "Import Configs",
                "Features config imported with warnings:\n- " + "\n- ".join(warnings),
            )
        if not persisted:
            self._show_warning(
                "Import Configs",
                "Features config imported, but persisting record UI state failed.",
            )


__all__ = [
    "FEATURES_CONFIG_FILE_NAME",
    "FEATURES_CONFIG_SCHEMA",
    "FEATURES_CONFIG_VERSION",
    "MainWindowFeaturesConfigMixin",
]
