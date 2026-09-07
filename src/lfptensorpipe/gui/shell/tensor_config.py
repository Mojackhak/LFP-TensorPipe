"""Tensor config import/export MainWindow methods."""

from __future__ import annotations

import json
import re
from copy import deepcopy

from lfptensorpipe.app.shared.page_config import json_value, page_config_node
from lfptensorpipe.app.tensor import service as tensor_service
from lfptensorpipe.app.tensor.config import (
    TENSOR_CONFIG_FIELDS_BY_METRIC,
    TENSOR_CONFIG_FILE_NAME,
    TENSOR_CONFIG_LEGACY_VERSION,
    TENSOR_CONFIG_SCHEMA,
    TENSOR_CONFIG_VERSION,
    TENSOR_DIRTY_KEYS,
    TENSOR_MULTITAPER_METRIC_KEYS,
    convert_legacy_tensor_metric,
    normalize_tensor_config_metric_params,
)
from lfptensorpipe.app.tensor.cpu_budget import (
    DEFAULT_TENSOR_CPU_PERCENT,
    normalize_tensor_cpu_percent,
)
from lfptensorpipe.app.tensor.frequency import (
    validate_periodic_aperiodic_notch_bounds,
)
from lfptensorpipe.app.tensor.orchestration_plan_validation import (
    prepare_metric_plan_inputs,
    validate_metric_storage_params,
)
from lfptensorpipe.gui.shell.common import (
    TENSOR_METRICS,
    Any,
    Path,
    PathResolver,
    QMessageBox,
    RecordContext,
    build_tensor_metric_notch_payload,
)
from lfptensorpipe.io.burst_thresholds import normalize_burst_threshold_payload


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

    _tensor_config_json_value = staticmethod(json_value)

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

        supported_metric_keys = self._tensor_config_supported_metric_keys()
        context = self._record_context()
        if context is None:
            raise ValueError("Select a record before exporting Tensor configs.")
        validation_errors: list[str] = []
        for metric_key in supported_metric_keys:
            params = dict(self._tensor_metric_params.get(metric_key, {}))
            try:
                prepare_metric_plan_inputs(
                    tensor_service,
                    context,
                    metric_key=metric_key,
                    metric_label=self._tensor_metric_display_name(metric_key),
                    metric_params=params,
                )
            except Exception as exc:  # noqa: BLE001
                validation_errors.append(
                    f"{self._tensor_metric_display_name(metric_key)}: {exc}"
                )
        if validation_errors:
            raise ValueError(
                "Tensor config contains invalid drafts:\n- "
                + "\n- ".join(validation_errors)
            )

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

    def _tensor_import_metric_defaults(
        self,
        metric_key: str,
        *,
        context: RecordContext | None,
        available_channels: tuple[str, ...],
    ) -> dict[str, Any] | None:
        getter = getattr(self, "_tensor_effective_metric_defaults", None)
        if not callable(getter):
            return None
        defaults = getter(
            metric_key,
            context=context,
            available_channels=available_channels,
        )
        return dict(defaults) if isinstance(defaults, dict) else None

    @staticmethod
    def _tensor_import_error_fields(
        message: str,
        whitelist: tuple[str, ...],
    ) -> list[str]:
        lowered = str(message).lower()
        if "selected channel" in lowered:
            return ["selected_channels"]
        if "selected pair" in lowered:
            return ["selected_pairs"]
        if "hilbert filter method" in lowered and "hilbert_filter_method" in whitelist:
            return ["hilbert_filter_method"]
        outer_context, separator, _nested_context = lowered.partition(":")
        literal_scopes = (outer_context, lowered) if separator else (lowered,)
        for scope in literal_scopes:
            literal_matches = [
                (match.start(), key)
                for key in whitelist
                if (
                    match := re.search(
                        rf"(?<![a-z0-9_]){re.escape(key.lower())}(?![a-z0-9_])",
                        scope,
                    )
                )
            ]
            if literal_matches:
                return [min(literal_matches, key=lambda item: item[0])[1]]
        if "band" in lowered and "bands" in whitelist:
            return ["bands"]
        if "specparam freq range" in lowered or "within specparam" in lowered:
            return ["freq_range_hz"]
        if "frequency" in lowered and not any(
            key.lower() in lowered for key in whitelist
        ):
            return [
                key
                for key in ("low_freq_hz", "high_freq_hz", "freq_step_hz")
                if key in whitelist
            ]
        return []

    def _repair_tensor_config_metric_params(
        self,
        metric_key: str,
        node: dict[str, Any],
        *,
        context: RecordContext | None,
        available_channels: tuple[str, ...],
        legacy_notch_fields: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        defaults = self._tensor_import_metric_defaults(
            metric_key,
            context=context,
            available_channels=available_channels,
        )
        if defaults is None:
            return self._normalize_tensor_config_metric_params(
                metric_key,
                node,
                available_channels=available_channels,
                legacy_notch_fields=legacy_notch_fields,
            )

        whitelist = TENSOR_CONFIG_FIELDS_BY_METRIC.get(metric_key, ())
        working = dict(node)
        warnings: list[str] = []
        unknown_fields = sorted(key for key in working if key not in whitelist)
        if unknown_fields:
            warnings.append(
                f"Removed unavailable values for {metric_key}: "
                + ", ".join(unknown_fields)
                + "."
            )
        working = {key: value for key, value in working.items() if key in whitelist}

        for key in whitelist:
            if key in working:
                continue
            if key not in defaults:
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.{key} is missing and has no safe default."
                )
            working[key] = deepcopy(defaults[key])
            warnings.append(f"Missing default restored for {metric_key}.{key}.")

        max_attempts = len(whitelist) + 4
        for _attempt in range(max_attempts):
            try:
                normalized, metric_warnings = (
                    self._normalize_tensor_config_metric_params(
                        metric_key,
                        working,
                        available_channels=available_channels,
                        legacy_notch_fields=legacy_notch_fields,
                    )
                )
                allow_empty_filtered_selectors = (
                    not normalized.get("selected_channels")
                    and any(
                        "unavailable channel(s)" in warning
                        for warning in metric_warnings
                    )
                ) or (
                    not normalized.get("selected_pairs")
                    and any(
                        "unavailable pair(s)" in warning for warning in metric_warnings
                    )
                )
                validate_metric_storage_params(
                    metric_key=metric_key,
                    metric_label=self._tensor_metric_display_name(metric_key),
                    metric_params=normalized,
                    allow_empty_selectors=allow_empty_filtered_selectors,
                )
                if context is not None:
                    prepare_metric_plan_inputs(
                        tensor_service,
                        context,
                        metric_key=metric_key,
                        metric_label=self._tensor_metric_display_name(metric_key),
                        metric_params=normalized,
                        allow_empty_selectors=allow_empty_filtered_selectors,
                    )
            except Exception as exc:  # noqa: BLE001
                fields = self._tensor_import_error_fields(str(exc), whitelist)
                fields = [
                    key
                    for key in fields
                    if key in defaults and working.get(key) != defaults.get(key)
                ]
                if not fields:
                    raise ValueError(
                        f"tensor.metric_params.{metric_key} has no safe repair: {exc}"
                    ) from exc
                for key in fields:
                    working[key] = deepcopy(defaults[key])
                    warnings.append(f"Invalid value restored for {metric_key}.{key}.")
                continue
            warnings.extend(metric_warnings)
            return normalized, warnings
        raise ValueError(
            f"tensor.metric_params.{metric_key} could not be repaired safely."
        )

    _normalize_tensor_config_metric_params = staticmethod(
        normalize_tensor_config_metric_params
    )

    def _normalize_tensor_config_import_payload(
        self,
        payload: dict[str, Any],
        *,
        context: RecordContext | None,
        available_channels: tuple[str, ...],
    ) -> tuple[dict[str, Any], list[str]]:
        tensor_node = page_config_node(payload, "tensor")
        version = payload["version"]

        metric_params = tensor_node.get("metric_params")
        if not isinstance(metric_params, dict):
            raise ValueError(
                "Tensor config is missing required `tensor.metric_params` object."
            )

        supported_metric_keys = self._tensor_config_supported_metric_keys()
        metric_params_by_key = dict(metric_params)
        warnings: list[str] = []
        missing = [
            metric_key
            for metric_key in supported_metric_keys
            if metric_key not in metric_params_by_key
        ]
        imcoh_abs_defaulted = (
            version == TENSOR_CONFIG_LEGACY_VERSION and "imcoh_abs" in missing
        )
        for metric_key in missing:
            defaults = self._tensor_import_metric_defaults(
                metric_key,
                context=context,
                available_channels=available_channels,
            )
            if defaults is None:
                raise ValueError(
                    "Tensor config is missing metric definitions for: "
                    + ", ".join(missing)
                    + "."
                )
            metric_params_by_key[metric_key] = defaults
            warnings.append(
                f"Missing default restored for tensor.metric_params.{metric_key}."
            )

        if version == TENSOR_CONFIG_LEGACY_VERSION:
            for metric_key in supported_metric_keys:
                node = metric_params_by_key.get(metric_key)
                if not isinstance(node, dict):
                    continue
                metric_params_by_key[metric_key] = convert_legacy_tensor_metric(
                    metric_key,
                    node,
                    keep_current_notches=imcoh_abs_defaulted
                    and metric_key == "imcoh_abs",
                )
            warnings.append(
                "Legacy conversion: Version 3 Tensor config was imported.\n\n"
                "Legacy Build Tensor notch_widths values were preserved unchanged "
                "as notch_radii because those values already represented a radius."
                "\n\nLegacy Multitaper time_bandwidth and mt_bandwidth values "
                "were ignored. Version 4 defaults were applied: time-bandwidth "
                "product = 4.0, minimum cycles = 3.0, and maximum cycles = none. "
                "Existing Multitaper "
                "tensors must be recomputed.\n\nRe-export the configuration to "
                "save the canonical version 4 fields."
            )
        else:
            for metric_key in TENSOR_MULTITAPER_METRIC_KEYS:
                node = metric_params_by_key.get(metric_key)
                if not isinstance(node, dict):
                    continue
                normalized_node = dict(node)
                normalized_node.setdefault("mt_max_cycles", None)
                missing_mt_fields = [
                    field_name
                    for field_name in (
                        "mt_time_bandwidth_product",
                        "mt_min_cycles",
                    )
                    if field_name not in normalized_node
                ]
                if missing_mt_fields:
                    defaults = self._tensor_import_metric_defaults(
                        metric_key,
                        context=context,
                        available_channels=available_channels,
                    )
                    if defaults is None or any(
                        field_name not in defaults for field_name in missing_mt_fields
                    ):
                        raise ValueError(
                            f"tensor.metric_params.{metric_key} is missing required "
                            "Multitaper fields: " + ", ".join(missing_mt_fields) + "."
                        )
                    for field_name in missing_mt_fields:
                        normalized_node[field_name] = deepcopy(defaults[field_name])
                        warnings.append(
                            "Missing default restored for "
                            f"{metric_key}.{field_name}."
                        )
                metric_params_by_key[metric_key] = normalized_node
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
            raw_selected_metrics = []
            warnings.append(
                "Invalid or missing selected_metrics restored to an empty selection."
            )
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
            mask_edge_effects = True
            warnings.append("Invalid or missing mask_edge_effects restored to true.")
        if "cpu_percent" in tensor_node:
            raw_cpu_percent = tensor_node.get("cpu_percent")
        else:
            raw_cpu_percent = DEFAULT_TENSOR_CPU_PERCENT
            warnings.append("Missing CPU (%) restored to 75.")
        try:
            cpu_percent = normalize_tensor_cpu_percent(raw_cpu_percent)
        except ValueError:
            cpu_percent = DEFAULT_TENSOR_CPU_PERCENT
            warnings.append("Invalid CPU (%) restored to 75.")

        normalized_metric_params: dict[str, dict[str, Any]] = {}
        for metric_key in supported_metric_keys:
            normalized_params, metric_warnings = (
                self._repair_tensor_config_metric_params(
                    metric_key,
                    metric_params_by_key.get(metric_key, {}),
                    context=context,
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

        try:
            payload = self._build_tensor_config_export_payload()
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
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

        preview_lines = [
            "Review the Tensor config normalization before importing:",
            "",
        ]
        if warnings:
            preview_lines.extend(f"- {warning}" for warning in warnings)
        else:
            preview_lines.append("- No normalization was required.")
        preview_lines.extend(["", "Apply this imported configuration?"])
        confirmed = self._ask_question(
            "Import Tensor Configs",
            "\n".join(preview_lines),
            buttons=QMessageBox.Yes | QMessageBox.No,
            default_button=QMessageBox.No,
        )
        if confirmed != QMessageBox.Yes:
            self.statusBar().showMessage("Tensor config import cancelled.")
            return

        self._apply_tensor_import_snapshot(context, tensor_snapshot)
        self._record_param_dirty_keys.update(TENSOR_DIRTY_KEYS)

        self.statusBar().showMessage(f"Imported Tensor config: {import_path.name}")


__all__ = [
    "MainWindowTensorConfigMixin",
    "TENSOR_CONFIG_FIELDS_BY_METRIC",
    "TENSOR_CONFIG_FILE_NAME",
    "TENSOR_CONFIG_SCHEMA",
    "TENSOR_CONFIG_VERSION",
]
