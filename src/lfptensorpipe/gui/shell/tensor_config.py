"""Tensor config import/export MainWindow methods."""

from __future__ import annotations

import json

from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    RecordContext,
    TENSOR_METRICS,
    build_tensor_metric_notch_payload,
)

TENSOR_CONFIG_SCHEMA = "lfptensorpipe.tensor-config"
TENSOR_CONFIG_VERSION = 3
TENSOR_CONFIG_FILE_NAME = "lfptensorpipe_tensor_config.json"
TENSOR_DIRTY_KEYS = {
    "tensor.active_metric",
    "tensor.mask_edge_effects",
    "tensor.metric_params",
    "tensor.selected_metrics",
    "tensor.selectors",
}
TENSOR_CONFIG_FIELDS_BY_METRIC: dict[str, tuple[str, ...]] = {
    "raw_power": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "min_cycles",
        "max_cycles",
        "time_bandwidth",
        "notches",
        "notch_widths",
        "selected_channels",
    ),
    "periodic_aperiodic": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "freq_range_hz",
        "min_cycles",
        "max_cycles",
        "time_bandwidth",
        "freq_smooth_enabled",
        "freq_smooth_sigma",
        "time_smooth_enabled",
        "time_smooth_kernel_size",
        "aperiodic_mode",
        "peak_width_limits_hz",
        "max_n_peaks",
        "min_peak_height",
        "peak_threshold",
        "fit_qc_threshold",
        "notches",
        "notch_widths",
        "selected_channels",
    ),
    "coherence": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_bandwidth",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_widths",
        "selected_pairs",
    ),
    "plv": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_bandwidth",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_widths",
        "selected_pairs",
    ),
    "ciplv": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_bandwidth",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_widths",
        "selected_pairs",
    ),
    "pli": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_bandwidth",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_widths",
        "selected_pairs",
    ),
    "wpli": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_bandwidth",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_widths",
        "selected_pairs",
    ),
    "trgc": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_bandwidth",
        "min_cycles",
        "max_cycles",
        "gc_n_lags",
        "group_by_samples",
        "round_ms",
        "notches",
        "notch_widths",
        "selected_pairs",
    ),
    "psi": (
        "bands",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_bandwidth",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_widths",
        "selected_pairs",
    ),
    "burst": (
        "bands",
        "percentile",
        "baseline_keep",
        "min_cycles",
        "max_cycles",
        "hop_s",
        "decim",
        "thresholds",
        "notches",
        "notch_widths",
        "selected_channels",
    ),
}


class MainWindowTensorConfigMixin:
    @staticmethod
    def _tensor_config_supported_specs() -> tuple[Any, ...]:
        return tuple(
            spec for spec in TENSOR_METRICS if bool(getattr(spec, "supported", False))
        )

    @classmethod
    def _tensor_config_supported_metric_keys(cls) -> tuple[str, ...]:
        return tuple(spec.key for spec in cls._tensor_config_supported_specs())

    @staticmethod
    def _tensor_config_default_active_metric() -> str:
        return "raw_power"

    @staticmethod
    def _tensor_config_json_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return MainWindowTensorConfigMixin._tensor_config_json_value(
                    item_method()
                )
            except Exception:
                pass
        if isinstance(value, dict):
            return {
                str(key): MainWindowTensorConfigMixin._tensor_config_json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                MainWindowTensorConfigMixin._tensor_config_json_value(item)
                for item in value
            ]
        raise TypeError(f"Unsupported tensor config value: {type(value).__name__}")

    def _tensor_config_default_path(self, context: RecordContext | None) -> Path:
        if context is None:
            return Path.cwd() / TENSOR_CONFIG_FILE_NAME
        return PathResolver(context).lfp_root / TENSOR_CONFIG_FILE_NAME

    def _collect_tensor_config_metric_params(self, metric_key: str) -> dict[str, Any]:
        whitelist = TENSOR_CONFIG_FIELDS_BY_METRIC.get(metric_key, ())
        params = dict(self._tensor_metric_params.get(metric_key, {}))
        out: dict[str, Any] = {}
        pair_mode = self._tensor_metric_pair_mode(metric_key)

        for key in whitelist:
            if key == "notches":
                out[key] = build_tensor_metric_notch_payload(
                    params.get("notches"),
                    params.get("notch_widths"),
                )["notches"]
                continue
            if key == "notch_widths":
                out[key] = self._tensor_config_json_value(
                    build_tensor_metric_notch_payload(
                        params.get("notches"),
                        params.get("notch_widths"),
                    )["notch_widths"]
                )
                continue
            if key not in params:
                continue
            value = params.get(key)
            if key == "selected_channels":
                out[key] = list(self._coerce_tensor_channels(value))
                continue
            if key == "selected_pairs":
                pairs = self._coerce_tensor_pairs(
                    value,
                    directed=(pair_mode == "directed"),
                )
                out[key] = [[source, target] for source, target in pairs]
                continue
            if key == "bands":
                out[key] = [
                    dict(item) for item in self._normalize_tensor_bands_rows(value)
                ]
                continue
            out[key] = self._tensor_config_json_value(value)
        return out

    def _build_tensor_config_export_payload(self) -> dict[str, Any]:
        self._commit_active_tensor_panel_to_params()
        self._sync_tensor_selector_maps_into_metric_params()

        supported_metric_keys = self._tensor_config_supported_metric_keys()
        active_metric = (
            self._tensor_active_metric_key
            if self._tensor_active_metric_key in supported_metric_keys
            else self._tensor_config_default_active_metric()
        )
        return {
            "schema": TENSOR_CONFIG_SCHEMA,
            "version": TENSOR_CONFIG_VERSION,
            "tensor": {
                "selected_metrics": [
                    metric_key
                    for metric_key in self._selected_tensor_metrics_snapshot()
                    if metric_key in supported_metric_keys
                ],
                "active_metric": active_metric,
                "mask_edge_effects": bool(
                    self._tensor_mask_edge_checkbox.isChecked()
                    if self._tensor_mask_edge_checkbox is not None
                    else True
                ),
                "metric_params": {
                    metric_key: self._collect_tensor_config_metric_params(metric_key)
                    for metric_key in supported_metric_keys
                },
            },
        }

    def _normalize_tensor_config_metric_params(
        self,
        metric_key: str,
        node: dict[str, Any],
        *,
        available_channels: tuple[str, ...],
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(node, dict):
            raise ValueError(f"tensor.metric_params.{metric_key} must be an object.")
        if metric_key == "periodic_aperiodic":
            removed_keys = sorted(
                key for key in ("smooth_enabled", "kernel_size") if key in node
            )
            if removed_keys:
                raise ValueError(
                    "tensor.metric_params.periodic_aperiodic contains removed keys: "
                    + ", ".join(removed_keys)
                    + "."
                )

        whitelist = TENSOR_CONFIG_FIELDS_BY_METRIC.get(metric_key, ())
        pair_mode = self._tensor_metric_pair_mode(metric_key)
        warnings: list[str] = []
        out: dict[str, Any] = {}

        for key in whitelist:
            if key not in node:
                continue
            value = node.get(key)
            if key == "selected_channels":
                if not isinstance(value, list):
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.selected_channels must be a list."
                    )
                normalized = self._coerce_tensor_channels(value)
                filtered = tuple(
                    channel
                    for channel in normalized
                    if channel in set(available_channels)
                )
                dropped = len(normalized) - len(filtered)
                if dropped > 0:
                    warnings.append(
                        f"{self._tensor_metric_display_name(metric_key)} ignored {dropped} unavailable channel(s)."
                    )
                out[key] = [str(item) for item in filtered]
                continue
            if key == "selected_pairs":
                if not isinstance(value, list):
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.selected_pairs must be a list."
                    )
                normalized_pairs = self._coerce_tensor_pairs(
                    value,
                    directed=(pair_mode == "directed"),
                )
                filtered_pairs = self._filter_tensor_pairs(
                    normalized_pairs,
                    available_channels=available_channels,
                    directed=(pair_mode == "directed"),
                )
                dropped = len(normalized_pairs) - len(filtered_pairs)
                if dropped > 0:
                    warnings.append(
                        f"{self._tensor_metric_display_name(metric_key)} ignored {dropped} unavailable pair(s)."
                    )
                out[key] = [[source, target] for source, target in filtered_pairs]
                continue
            if key == "bands":
                if not isinstance(value, list):
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.bands must be a list."
                    )
                out[key] = [
                    dict(item) for item in self._normalize_tensor_bands_rows(value)
                ]
                continue
            if key == "notches":
                out[key] = build_tensor_metric_notch_payload(
                    value,
                    node.get("notch_widths"),
                )["notches"]
                continue
            if key == "notch_widths":
                out[key] = build_tensor_metric_notch_payload(
                    node.get("notches"),
                    value,
                )["notch_widths"]
                continue
            if key == "thresholds":
                if value is not None and not isinstance(value, list):
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.thresholds must be a list or null."
                    )
                out[key] = self._tensor_config_json_value(value)
                continue
            if key == "baseline_keep":
                if value is None:
                    out[key] = None
                    continue
                if not isinstance(value, list):
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.baseline_keep must be a list or null."
                    )
                labels: list[str] = []
                seen: set[str] = set()
                for item in value:
                    label = str(item).strip()
                    if not label or label in seen:
                        continue
                    seen.add(label)
                    labels.append(label)
                out[key] = labels or None
                continue
            out[key] = self._tensor_config_json_value(value)

        out.update(
            build_tensor_metric_notch_payload(
                out.get("notches"),
                out.get("notch_widths"),
            )
        )
        return out, warnings

    def _normalize_tensor_config_import_payload(
        self,
        payload: dict[str, Any],
        *,
        available_channels: tuple[str, ...],
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(payload, dict):
            raise ValueError("Tensor config must be a JSON object.")
        if payload.get("schema") != TENSOR_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported tensor config schema: {payload.get('schema')!r}."
            )
        version = payload.get("version")
        if version != TENSOR_CONFIG_VERSION:
            raise ValueError(
                f"Unsupported tensor config version: {payload.get('version')!r}."
            )

        tensor_node = payload.get("tensor")
        if not isinstance(tensor_node, dict):
            raise ValueError("Tensor config is missing required `tensor` object.")

        metric_params = tensor_node.get("metric_params")
        if not isinstance(metric_params, dict):
            raise ValueError(
                "Tensor config is missing required `tensor.metric_params` object."
            )

        supported_metric_keys = self._tensor_config_supported_metric_keys()
        missing = [
            metric_key
            for metric_key in supported_metric_keys
            if metric_key not in metric_params
        ]
        if missing:
            raise ValueError(
                "Tensor config is missing metric definitions for: "
                + ", ".join(missing)
                + "."
            )

        warnings: list[str] = []
        unknown_metric_keys = [
            str(metric_key)
            for metric_key in metric_params.keys()
            if str(metric_key) not in supported_metric_keys
        ]
        if unknown_metric_keys:
            warnings.append(
                "Ignored unknown tensor metric keys: "
                + ", ".join(sorted(unknown_metric_keys))
                + "."
            )

        raw_selected_metrics = tensor_node.get("selected_metrics")
        if not isinstance(raw_selected_metrics, list):
            raise ValueError("Tensor config `tensor.selected_metrics` must be a list.")
        selected_metrics: list[str] = []
        unknown_selected_metrics: list[str] = []
        for item in raw_selected_metrics:
            metric_key = str(item).strip()
            if not metric_key:
                continue
            if metric_key not in supported_metric_keys:
                unknown_selected_metrics.append(metric_key)
                continue
            if metric_key in selected_metrics:
                continue
            selected_metrics.append(metric_key)
        if unknown_selected_metrics:
            warnings.append(
                "Ignored unknown selected metrics: "
                + ", ".join(sorted(set(unknown_selected_metrics)))
                + "."
            )

        raw_active_metric = tensor_node.get("active_metric")
        if (
            not isinstance(raw_active_metric, str)
            or raw_active_metric not in supported_metric_keys
        ):
            active_metric = self._tensor_config_default_active_metric()
            warnings.append(
                "Invalid or missing active metric; falling back to raw_power."
            )
        else:
            active_metric = raw_active_metric

        mask_edge_effects = tensor_node.get("mask_edge_effects")
        if not isinstance(mask_edge_effects, bool):
            raise ValueError(
                "Tensor config `tensor.mask_edge_effects` must be a boolean."
            )

        normalized_metric_params: dict[str, dict[str, Any]] = {}
        for metric_key in supported_metric_keys:
            normalized_params, metric_warnings = (
                self._normalize_tensor_config_metric_params(
                    metric_key,
                    metric_params.get(metric_key, {}),
                    available_channels=available_channels,
                )
            )
            normalized_metric_params[metric_key] = normalized_params
            warnings.extend(metric_warnings)

        return (
            {
                "selected_metrics": list(selected_metrics),
                "active_metric": active_metric,
                "mask_edge_effects": bool(mask_edge_effects),
                "metric_params": normalized_metric_params,
            },
            warnings,
        )

    def _apply_tensor_import_snapshot(
        self,
        context: RecordContext,
        tensor_snapshot: dict[str, Any],
    ) -> None:
        non_tensor_dirty_keys = {
            key
            for key in self._record_param_dirty_keys
            if not key.startswith("tensor.")
        }
        self._record_param_syncing = True
        try:
            self._record_param_dirty_keys.clear()
            self._record_param_dirty_keys.update(non_tensor_dirty_keys)
            self._apply_record_params_tensor_snapshot(
                context,
                {"tensor": dict(tensor_snapshot)},
            )
        finally:
            self._record_param_syncing = False
        self._refresh_tensor_controls()

    def _persist_imported_tensor_snapshot(
        self,
        context: RecordContext,
        tensor_snapshot: dict[str, Any],
    ) -> bool:
        ok_existing, existing_payload, _ = self._load_record_params_payload(context)
        payload = (
            dict(existing_payload)
            if ok_existing and isinstance(existing_payload, dict)
            else {}
        )
        payload["tensor"] = dict(tensor_snapshot)
        ok = self._write_record_params_payload(
            context,
            params=payload,
            reason="tensor_import_config",
        )
        if ok:
            self._record_param_dirty_keys.difference_update(TENSOR_DIRTY_KEYS)
            return True
        self._record_param_dirty_keys.update(TENSOR_DIRTY_KEYS)
        return False

    def _on_tensor_export_config(self) -> None:
        context = self._record_context()
        if context is None:
            self._show_warning(
                "Export Configs",
                "Select project, subject, and record before exporting Tensor configs.",
            )
            return

        default_path = self._tensor_config_default_path(context)
        file_path_text, _ = self._save_file_name(
            "Export Tensor Configs",
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
            payload = self._build_tensor_config_export_payload()
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Export Configs", f"Export failed:\n{exc}")
            return

        self.statusBar().showMessage(f"Exported Tensor config: {export_path.name}")

    def _on_tensor_import_config(self) -> None:
        context = self._record_context()
        if context is None:
            self._show_warning(
                "Import Configs",
                "Select project, subject, and record before importing Tensor configs.",
            )
            return

        default_path = self._tensor_config_default_path(context)
        file_path_text, _ = self._open_file_name(
            "Import Tensor Configs",
            str(default_path.parent.resolve()),
            "JSON files (*.json);;All files (*)",
        )
        if not file_path_text:
            return

        import_path = Path(file_path_text)
        try:
            with import_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            tensor_snapshot, warnings = self._normalize_tensor_config_import_payload(
                payload,
                available_channels=tuple(self._tensor_available_channels),
            )
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Import Configs", f"Import failed:\n{exc}")
            return

        self._apply_tensor_import_snapshot(context, tensor_snapshot)
        if not self._persist_imported_tensor_snapshot(context, tensor_snapshot):
            self._show_warning(
                "Import Configs",
                "Tensor config imported into the current session, but persisting record UI state failed.",
            )
            return

        self.statusBar().showMessage(f"Imported Tensor config: {import_path.name}")
        if warnings:
            self._show_information(
                "Import Configs",
                "Tensor config imported with warnings:\n- " + "\n- ".join(warnings),
            )


__all__ = [
    "MainWindowTensorConfigMixin",
    "TENSOR_CONFIG_FIELDS_BY_METRIC",
    "TENSOR_CONFIG_FILE_NAME",
    "TENSOR_CONFIG_SCHEMA",
    "TENSOR_CONFIG_VERSION",
]
