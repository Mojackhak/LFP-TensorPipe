"""Features config import/export MainWindow methods."""

from __future__ import annotations

from lfptensorpipe.app.features.phase_time import phase_limit, validate_phase_bounds

import json

from lfptensorpipe.app.shared.page_config import json_value, page_config_node
from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    QMessageBox,
    RecordContext,
)
from lfptensorpipe.io.atomic_file import write_json_atomically

FEATURES_CONFIG_SCHEMA = "lfptensorpipe.features-config"
FEATURES_CONFIG_VERSION = 1
FEATURES_CONFIG_FILE_NAME = "lfptensorpipe_features_config.json"


class MainWindowFeaturesConfigMixin:
    _features_config_json_value = staticmethod(json_value)

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
        if not metric_keys:
            raise ValueError(
                "A current Align Finish with at least one metric is required to export Features."
            )
        axes_ok, axes_message = self._validate_features_axes_for_run(metric_keys)
        if not axes_ok:
            raise ValueError(axes_message)
        active_metric = self._current_features_axis_metric() or ""
        if active_metric not in metric_keys:
            active_metric = metric_keys[0] if metric_keys else ""
        axes_by_metric: dict[str, dict[str, Any]] = {}
        for metric_key in metric_keys:
            axes = self._normalized_features_axes_for_metric(metric_key)
            axes_by_metric[metric_key] = {
                "time_unit": axes.get("time_unit", "percent"),
                "bands": (
                    []
                    if self._features_metric_uses_auto_bands(metric_key)
                    else [dict(item) for item in axes["bands"]]
                ),
                "times": [dict(item) for item in axes["times"]],
            }
        payload = {
            "schema": FEATURES_CONFIG_SCHEMA,
            "version": FEATURES_CONFIG_VERSION,
            "features": {
                "active_metric": active_metric,
                "axes_by_metric": self._features_config_json_value(axes_by_metric),
            },
        }
        context = self._record_context()
        slug = self._current_features_paradigm_slug()
        if context and slug and metric_keys:
            from lfptensorpipe.app.features.execution import execution_snapshot
            from lfptensorpipe.app.alignment.generation import (
                accepted_alignment_metrics,
            )

            ready = accepted_alignment_metrics(
                PathResolver(context), trial_slug=slug, stage="finish"
            )
            if set(metric_keys) <= set(ready):
                from lfptensorpipe.app.features.indicator import (
                    features_panel_state,
                    features_derivatives_log_path,
                )
                from lfptensorpipe.app.shared.runlog_store import read_run_log

                resolver = PathResolver(context)
                accepted = None
                if (
                    features_panel_state(
                        resolver, trial_slug=slug, axes_by_metric=axes_by_metric
                    )
                    == "green"
                ):
                    accepted = read_run_log(
                        features_derivatives_log_path(resolver, trial_slug=slug)
                    )["params"]
                payload["features"]["execution"] = execution_snapshot(
                    context, slug, metric_keys, self._config_store, accepted=accepted
                )
        return payload

    def _normalize_features_config_import_payload(
        self,
        payload: dict[str, Any],
        *,
        available_metrics: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        features_node = page_config_node(payload, "features")
        execution = features_node.get("execution")
        collapse = execution.get("collapse") if isinstance(execution, dict) else None
        interval_mode = (
            collapse.get("time_interval_mode") if isinstance(collapse, dict) else None
        )

        axes_by_metric = features_node.get("axes_by_metric")
        if not isinstance(axes_by_metric, dict):
            raise ValueError(
                "Features config is missing required `features.axes_by_metric` object."
            )

        warnings: list[str] = []
        available_set = set(available_metrics)
        ignored_metrics: list[str] = []
        auto_band_metrics: list[str] = []
        duration = None
        normalized_axes: dict[str, dict[str, Any]] = {}
        for metric_key_raw, axes_node in axes_by_metric.items():
            metric_key = str(metric_key_raw).strip()
            if not metric_key:
                continue
            if metric_key not in available_set:
                ignored_metrics.append(metric_key)
                continue
            if not isinstance(axes_node, dict):
                axes_node = {}
            if (
                interval_mode == "absolute"
                and "time_unit" not in axes_node
                and axes_node.get("times")
            ):
                from lfptensorpipe.app.features.phase_time import (
                    resolve_stored_phase_axes,
                )

                context = self._record_context()
                if context is not None:
                    axes_node = resolve_stored_phase_axes(
                        PathResolver(context),
                        self._current_features_paradigm_slug(),
                        metric_key,
                        axes_node,
                        interval_mode,
                    )
            raw_bands = axes_node.get("bands")
            if self._features_metric_uses_auto_bands(metric_key):
                if isinstance(raw_bands, list) and raw_bands:
                    auto_band_metrics.append(metric_key)
                bands: list[dict[str, Any]] = []
            else:
                normalized_bands = [
                    dict(item)
                    for item in self._normalize_feature_axis_rows(
                        raw_bands,
                        min_start=0.0,
                        max_end=None,
                        allow_duplicate_names=False,
                    )
                ]
                if isinstance(raw_bands, list) and len(normalized_bands) < len(
                    raw_bands
                ):
                    warnings.append(f"Removed invalid axis row(s): {metric_key}.bands.")
                bands = normalized_bands
                if not bands:
                    defaults = self._load_features_axis_defaults(
                        metric_key=metric_key,
                        axis_key="bands",
                    )
                    if not defaults:
                        raise ValueError(
                            f"{metric_key} band selection is required and has no safe automatic default."
                        )
                    bands = [dict(item) for item in defaults]
                    category = (
                        "Missing default restored"
                        if "bands" not in axes_node
                        else "Invalid value restored"
                    )
                    warnings.append(f"{category}: {metric_key}.bands.")
            raw_times = axes_node.get("times")
            unit = axes_node.get("time_unit", "percent")
            if unit == "s" and duration is None:
                duration = self._features_seconds_duration()
            limit = phase_limit(unit, duration)
            if unit == "s" and isinstance(raw_times, list):
                validate_phase_bounds(raw_times, unit, duration)
            times = [
                dict(item)
                for item in self._normalize_feature_axis_rows(
                    raw_times,
                    min_start=0.0,
                    max_end=limit,
                    allow_duplicate_names=True,
                )
            ]
            if isinstance(raw_times, list) and len(times) < len(raw_times):
                warnings.append(f"Removed invalid axis row(s): {metric_key}.times.")
            if not times:
                defaults = self._load_features_axis_defaults(
                    metric_key=metric_key,
                    axis_key="times",
                    time_unit=unit,
                )
                if not defaults:
                    raise ValueError(
                        f"{metric_key} phase selection is required and has no safe automatic default."
                    )
                times = [dict(item) for item in defaults]
                validate_phase_bounds(times, unit, duration)
                category = (
                    "Missing default restored"
                    if "times" not in axes_node
                    else "Invalid value restored"
                )
                warnings.append(f"{category}: {metric_key}.times.")
            normalized_axes[metric_key] = {
                **({"time_unit": unit} if "time_unit" in axes_node else {}),
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
        for metric_key in available_metrics:
            if metric_key in normalized_axes:
                continue
            bands = (
                []
                if self._features_metric_uses_auto_bands(metric_key)
                else [
                    dict(item)
                    for item in self._load_features_axis_defaults(
                        metric_key=metric_key,
                        axis_key="bands",
                    )
                ]
            )
            times = [
                dict(item)
                for item in self._load_features_axis_defaults(
                    metric_key=metric_key,
                    axis_key="times",
                )
            ]
            if not self._features_metric_uses_auto_bands(metric_key) and not bands:
                raise ValueError(
                    f"{metric_key} band selection is missing and has no safe automatic default."
                )
            if not times:
                raise ValueError(
                    f"{metric_key} phase selection is missing and has no safe automatic default."
                )
            normalized_axes[metric_key] = {"bands": bands, "times": times}
            restored_axes = (
                "times"
                if self._features_metric_uses_auto_bands(metric_key)
                else "bands/times"
            )
            warnings.append(f"Missing default restored: {metric_key}.{restored_axes}.")
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
        try:
            payload = self._build_features_config_export_payload()
            write_json_atomically(export_path, payload)
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

        preview_lines = [
            "Review the Features config normalization before importing:",
            "",
        ]
        if warnings:
            preview_lines.extend(f"- {warning}" for warning in warnings)
        else:
            preview_lines.append("- No normalization was required.")
        preview_lines.extend(["", "Apply this imported configuration?"])
        confirmed = self._ask_question(
            "Import Features Configs",
            "\n".join(preview_lines),
            buttons=QMessageBox.Yes | QMessageBox.No,
            default_button=QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            self.statusBar().showMessage("Features config import cancelled.")
            return

        self._features_axes_by_metric.update(
            {
                key: {"time_unit": "percent", **axes}
                for key, axes in features_snapshot["axes_by_metric"].items()
            }
        )
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
