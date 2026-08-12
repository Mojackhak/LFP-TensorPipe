"""Tensor config import/export MainWindow methods."""

from __future__ import annotations

from copy import deepcopy
import json
import math

from lfptensorpipe.app.tensor.cpu_budget import (
    DEFAULT_TENSOR_CPU_PERCENT,
    normalize_tensor_cpu_percent,
)
from lfptensorpipe.app.tensor.frequency import (
    validate_periodic_aperiodic_notch_bounds,
)
from lfptensorpipe.gui.shell.common import (
    Any,
    Path,
    PathResolver,
    RecordContext,
    TENSOR_METRICS,
    build_tensor_metric_notch_payload,
)
from lfptensorpipe.io.burst_thresholds import normalize_burst_threshold_payload

TENSOR_CONFIG_SCHEMA = "lfptensorpipe.tensor-config"
TENSOR_CONFIG_VERSION = 4
TENSOR_CONFIG_LEGACY_VERSION = 3
TENSOR_CONFIG_FILE_NAME = "lfptensorpipe_tensor_config.json"
TENSOR_MULTITAPER_METRIC_KEYS = frozenset(
    {
        "raw_power",
        "periodic_aperiodic",
        "coherence",
        "imcoh_abs",
        "plv",
        "ciplv",
        "pli",
        "wpli",
        "trgc",
        "psi",
    }
)
TENSOR_DIRTY_KEYS = {
    "tensor.active_metric",
    "tensor.mask_edge_effects",
    "tensor.cpu_percent",
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
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "notches",
        "notch_radii",
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
        "mt_time_bandwidth_product",
        "mt_min_cycles",
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
        "notch_radii",
        "selected_channels",
    ),
    "coherence": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "imcoh_abs": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "plv": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "ciplv": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "pli": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "wpli": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "trgc": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "min_cycles",
        "max_cycles",
        "gc_n_lags",
        "group_by_samples",
        "round_ms",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "psi": (
        "freq_step_hz",
        "bands",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "burst": (
        "bands",
        "percentile",
        "baseline_keep",
        "min_cycles",
        "max_cycles",
        "thresholds",
        "notches",
        "notch_radii",
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
                    params.get("notch_radii"),
                )["notches"]
                continue
            if key == "notch_radii":
                out[key] = self._tensor_config_json_value(
                    build_tensor_metric_notch_payload(
                        params.get("notches"),
                        params.get("notch_radii"),
                    )["notch_radii"]
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
            if key == "thresholds":
                out[key] = (
                    normalize_burst_threshold_payload(value)
                    if value is not None
                    else None
                )
                continue
            out[key] = self._tensor_config_json_value(value)
        return out

    def _build_tensor_config_export_payload(self) -> dict[str, Any]:
        self._commit_active_tensor_panel_to_params()
        self._sync_tensor_selector_maps_into_metric_params()

        validate_periodic_aperiodic_notch_bounds(
            dict(self._tensor_metric_params.get("periodic_aperiodic", {}))
        )

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
                "cpu_percent": normalize_tensor_cpu_percent(
                    self._tensor_cpu_percent_edit.text().strip()
                    if self._tensor_cpu_percent_edit is not None
                    else DEFAULT_TENSOR_CPU_PERCENT
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
        legacy_notch_fields: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(node, dict):
            raise ValueError(f"tensor.metric_params.{metric_key} must be an object.")
        if "notch_widths" in node:
            raise ValueError(
                f"tensor.metric_params.{metric_key}.notch_widths was removed "
                "from Build Tensor schema 4. Use notch_radii instead."
            )
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
                    node.get("notch_radii"),
                    legacy_mismatched_list_broadcast=legacy_notch_fields,
                )["notches"]
                continue
            if key == "notch_radii":
                out[key] = build_tensor_metric_notch_payload(
                    node.get("notches"),
                    value,
                    legacy_mismatched_list_broadcast=legacy_notch_fields,
                )["notch_radii"]
                continue
            if key == "thresholds":
                if value is None:
                    out[key] = None
                else:
                    try:
                        out[key] = normalize_burst_threshold_payload(value)
                    except ValueError as exc:
                        raise ValueError(
                            "tensor.metric_params.burst.thresholds is invalid: "
                            + str(exc)
                        ) from exc
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
            if key == "mt_time_bandwidth_product":
                try:
                    parsed = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.{key} must be a number."
                    ) from exc
                if not math.isfinite(parsed) or parsed < 2.0:
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.{key} must be finite and >= 2."
                    )
                out[key] = parsed
                continue
            if key == "mt_min_cycles":
                try:
                    parsed = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.{key} must be a number."
                    ) from exc
                if not math.isfinite(parsed) or parsed <= 0.0:
                    raise ValueError(
                        f"tensor.metric_params.{metric_key}.{key} must be finite and > 0."
                    )
                out[key] = parsed
                continue
            out[key] = self._tensor_config_json_value(value)

        out.update(
            build_tensor_metric_notch_payload(
                out.get("notches"),
                out.get("notch_radii"),
                legacy_mismatched_list_broadcast=legacy_notch_fields,
            )
        )
        return out, warnings

    def _normalize_tensor_config_import_payload(
        self,
        payload: dict[str, Any],
        *,
        context: RecordContext | None,
        available_channels: tuple[str, ...],
    ) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(payload, dict):
            raise ValueError("Tensor config must be a JSON object.")
        if payload.get("schema") != TENSOR_CONFIG_SCHEMA:
            raise ValueError(
                f"Unsupported tensor config schema: {payload.get('schema')!r}."
            )
        version = payload.get("version")
        if version not in {TENSOR_CONFIG_LEGACY_VERSION, TENSOR_CONFIG_VERSION}:
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
        metric_params_by_key = dict(metric_params)
        missing = [
            metric_key
            for metric_key in supported_metric_keys
            if metric_key not in metric_params_by_key
        ]
        required_missing = (
            [metric_key for metric_key in missing if metric_key != "imcoh_abs"]
            if version == TENSOR_CONFIG_LEGACY_VERSION
            else list(missing)
        )
        if required_missing:
            raise ValueError(
                "Tensor config is missing metric definitions for: "
                + ", ".join(required_missing)
                + "."
            )
        imcoh_abs_defaulted = (
            version == TENSOR_CONFIG_LEGACY_VERSION and "imcoh_abs" in missing
        )
        if imcoh_abs_defaulted:
            metric_params_by_key["imcoh_abs"] = self._tensor_effective_metric_defaults(
                "imcoh_abs",
                context=context,
                available_channels=available_channels,
            )

        warnings: list[str] = []
        if version == TENSOR_CONFIG_LEGACY_VERSION:
            for metric_key in supported_metric_keys:
                node = metric_params_by_key.get(metric_key)
                if not isinstance(node, dict):
                    continue
                normalized_legacy = dict(node)
                legacy_radius_present = "notch_widths" in normalized_legacy
                legacy_radius_value = normalized_legacy.pop("notch_widths", None)
                if not (imcoh_abs_defaulted and metric_key == "imcoh_abs"):
                    normalized_legacy.pop("notch_radii", None)
                if legacy_radius_present:
                    normalized_legacy["notch_radii"] = legacy_radius_value
                metric_params_by_key[metric_key] = normalized_legacy
            for metric_key in TENSOR_MULTITAPER_METRIC_KEYS:
                node = metric_params_by_key.get(metric_key)
                if not isinstance(node, dict):
                    continue
                normalized_legacy = dict(node)
                normalized_legacy.pop("time_bandwidth", None)
                normalized_legacy.pop("mt_bandwidth", None)
                normalized_legacy["mt_time_bandwidth_product"] = 4.0
                normalized_legacy["mt_min_cycles"] = 3.0
                metric_params_by_key[metric_key] = normalized_legacy
            warnings.append(
                "Version 3 Tensor config was imported.\n\n"
                "Legacy Build Tensor notch_widths values were preserved unchanged "
                "as notch_radii because those values already represented a radius."
                "\n\nLegacy Multitaper time_bandwidth and mt_bandwidth values "
                "were ignored. Version 4 defaults were applied: time-bandwidth "
                "product = 4.0 and minimum cycles = 3.0. Existing Multitaper "
                "tensors must be recomputed.\n\nRe-export the configuration to "
                "save the canonical version 4 fields."
            )
        else:
            for metric_key in TENSOR_MULTITAPER_METRIC_KEYS:
                node = metric_params_by_key.get(metric_key)
                if not isinstance(node, dict):
                    continue
                missing_mt_fields = [
                    field_name
                    for field_name in (
                        "mt_time_bandwidth_product",
                        "mt_min_cycles",
                    )
                    if field_name not in node
                ]
                if missing_mt_fields:
                    raise ValueError(
                        f"tensor.metric_params.{metric_key} is missing required "
                        "Multitaper fields: " + ", ".join(missing_mt_fields) + "."
                    )
        unknown_metric_keys = [
            str(metric_key)
            for metric_key in metric_params_by_key.keys()
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
        cpu_percent = normalize_tensor_cpu_percent(
            tensor_node.get("cpu_percent", DEFAULT_TENSOR_CPU_PERCENT)
        )

        normalized_metric_params: dict[str, dict[str, Any]] = {}
        for metric_key in supported_metric_keys:
            normalized_params, metric_warnings = (
                self._normalize_tensor_config_metric_params(
                    metric_key,
                    metric_params_by_key.get(metric_key, {}),
                    available_channels=available_channels,
                    legacy_notch_fields=(version == TENSOR_CONFIG_LEGACY_VERSION),
                )
            )
            normalized_metric_params[metric_key] = normalized_params
            if not (imcoh_abs_defaulted and metric_key == "imcoh_abs"):
                warnings.extend(metric_warnings)

        if imcoh_abs_defaulted:
            coherence_params = normalized_metric_params["coherence"]
            imcoh_abs_params = normalized_metric_params["imcoh_abs"]
            for field_name in ("notches", "notch_radii"):
                if field_name in coherence_params:
                    imcoh_abs_params[field_name] = deepcopy(
                        coherence_params[field_name]
                    )

        validate_periodic_aperiodic_notch_bounds(
            normalized_metric_params["periodic_aperiodic"]
        )

        return (
            {
                "selected_metrics": list(selected_metrics),
                "active_metric": active_metric,
                "mask_edge_effects": bool(mask_edge_effects),
                "cpu_percent": cpu_percent,
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
                context=context,
                available_channels=tuple(self._tensor_available_channels),
            )
        except Exception as exc:  # noqa: BLE001
            self._show_warning("Import Configs", f"Import failed:\n{exc}")
            return

        self._apply_tensor_import_snapshot(context, tensor_snapshot)
        self._record_param_dirty_keys.update(TENSOR_DIRTY_KEYS)

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
