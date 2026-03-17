"""Features defaults and persistence MainWindow methods."""

from __future__ import annotations

import math

from lfptensorpipe.app.config_store import FEATURES_CONFIG_FILENAME
from lfptensorpipe.gui.shell.common import (
    Any,
    FEATURES_AXES_DEFAULTS_KEY,
    FEATURE_PLOT_BASELINE_MODES,
    FEATURE_PLOT_COLORMAPS,
    FEATURE_PLOT_NORMALIZE_MODES,
    FEATURE_PLOT_TRANSFORM_MODES,
    normalize_feature_plot_transform_mode,
)

_FEATURES_DEFAULTS_CONFIG_FILENAME = FEATURES_CONFIG_FILENAME
_FEATURES_PLOT_CONFIG_FILENAME = "features_plot.yml"
_FEATURES_PLOT_BOXSIZE_DEFAULT = (60.0, 50.0)
_FEATURES_PLOT_FONT_SIZE_DEFAULT = 16.0
_FEATURES_PLOT_TICK_LABEL_SIZE_DEFAULT = 12.0
_FEATURES_PLOT_X_LABEL_OFFSET_MM_DEFAULT = 10.0
_FEATURES_PLOT_Y_LABEL_OFFSET_MM_DEFAULT = 18.0
_FEATURES_PLOT_COLORBAR_PAD_MM_DEFAULT = 4.0
_FEATURES_PLOT_CBAR_LABEL_OFFSET_MM_DEFAULT = 18.0
_FEATURES_PLOT_LEGEND_POSITION_DEFAULT = "outside_right"


def _positive_float_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return float(default)
    return float(parsed)


def _boxsize_or_default(
    value: Any,
    default: tuple[float, float],
) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width = _positive_float_or_default(value[0], default[0])
        height = _positive_float_or_default(value[1], default[1])
        return (width, height)
    return tuple(float(item) for item in default)


def _legend_position_or_default(value: Any, default: str) -> str:
    token = str(value).strip().lower() if value is not None else ""
    return token or str(default).strip().lower()


class MainWindowFeaturesDefaultsMixin:
    def _default_features_plot_params(self) -> dict[str, Any]:
        return {
            "boxsize": tuple(_FEATURES_PLOT_BOXSIZE_DEFAULT),
            "font_size": float(_FEATURES_PLOT_FONT_SIZE_DEFAULT),
            "tick_label_size": float(_FEATURES_PLOT_TICK_LABEL_SIZE_DEFAULT),
            "x_label_offset_mm": float(_FEATURES_PLOT_X_LABEL_OFFSET_MM_DEFAULT),
            "y_label_offset_mm": float(_FEATURES_PLOT_Y_LABEL_OFFSET_MM_DEFAULT),
            "colorbar_pad_mm": float(_FEATURES_PLOT_COLORBAR_PAD_MM_DEFAULT),
            "cbar_label_offset_mm": float(_FEATURES_PLOT_CBAR_LABEL_OFFSET_MM_DEFAULT),
            "legend_position": str(_FEATURES_PLOT_LEGEND_POSITION_DEFAULT),
        }

    def _normalize_features_plot_params(
        self,
        raw_params: dict[str, Any] | None,
        *,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        params = raw_params if isinstance(raw_params, dict) else {}
        return {
            "boxsize": _boxsize_or_default(params.get("boxsize"), defaults["boxsize"]),
            "font_size": _positive_float_or_default(
                params.get("font_size"),
                float(defaults["font_size"]),
            ),
            "tick_label_size": _positive_float_or_default(
                params.get("tick_label_size"),
                float(defaults["tick_label_size"]),
            ),
            "x_label_offset_mm": _positive_float_or_default(
                params.get("x_label_offset_mm"),
                float(defaults["x_label_offset_mm"]),
            ),
            "y_label_offset_mm": _positive_float_or_default(
                params.get("y_label_offset_mm"),
                float(defaults["y_label_offset_mm"]),
            ),
            "colorbar_pad_mm": _positive_float_or_default(
                params.get("colorbar_pad_mm"),
                float(defaults["colorbar_pad_mm"]),
            ),
            "cbar_label_offset_mm": _positive_float_or_default(
                params.get("cbar_label_offset_mm"),
                float(defaults["cbar_label_offset_mm"]),
            ),
            "legend_position": _legend_position_or_default(
                params.get("legend_position"),
                str(defaults["legend_position"]),
            ),
        }

    def _load_features_plot_params(
        self,
        metric_key: str,
        derived_type: str,
    ) -> dict[str, Any]:
        defaults = self._default_features_plot_params()
        payload = self._config_store.read_yaml(
            _FEATURES_PLOT_CONFIG_FILENAME,
            default={},
        )
        if not isinstance(payload, dict):
            return defaults
        plot_defaults = payload.get("plot_defaults", {})
        if not isinstance(plot_defaults, dict):
            return defaults

        normalized = defaults

        default_node = plot_defaults.get("default")
        if isinstance(default_node, dict):
            normalized = self._normalize_features_plot_params(
                default_node,
                defaults=normalized,
            )

        derived_type_token = str(derived_type).strip().lower()
        by_derived_type = plot_defaults.get("by_derived_type", {})
        if isinstance(by_derived_type, dict):
            derived_node = by_derived_type.get(derived_type_token)
            if isinstance(derived_node, dict):
                normalized = self._normalize_features_plot_params(
                    derived_node,
                    defaults=normalized,
                )

        metric_token = str(metric_key).strip()
        by_metric = plot_defaults.get("by_metric", {})
        if isinstance(by_metric, dict):
            metric_node = by_metric.get(metric_token)
            if isinstance(metric_node, dict):
                normalized = self._normalize_features_plot_params(
                    metric_node,
                    defaults=normalized,
                )

        by_metric_and_derived_type = plot_defaults.get("by_metric_and_derived_type", {})
        if isinstance(by_metric_and_derived_type, dict):
            metric_derived_node = by_metric_and_derived_type.get(metric_token)
            if isinstance(metric_derived_node, dict):
                derived_metric_node = metric_derived_node.get(derived_type_token)
                if isinstance(derived_metric_node, dict):
                    normalized = self._normalize_features_plot_params(
                        derived_metric_node,
                        defaults=normalized,
                    )

        return normalized

    def _load_features_axis_defaults(
        self,
        *,
        metric_key: str,
        axis_key: str,
    ) -> list[dict[str, Any]]:
        if axis_key == "bands" and self._features_metric_uses_auto_bands(metric_key):
            return []
        payload = self._config_store.read_yaml(
            _FEATURES_DEFAULTS_CONFIG_FILENAME,
            default={},
        )
        if not isinstance(payload, dict):
            payload = {}
        node = payload.get(FEATURES_AXES_DEFAULTS_KEY, {})
        if not isinstance(node, dict):
            node = {}
        bucket_name = "bands_by_metric" if axis_key == "bands" else "times_by_metric"
        bucket = node.get(bucket_name, {})
        if not isinstance(bucket, dict):
            bucket = {}
        raw_rows = bucket.get(metric_key)
        if not isinstance(raw_rows, list):
            raw_rows = bucket.get("default")
        normalized = self._normalize_feature_axis_rows(
            raw_rows,
            min_start=0.0,
            max_end=(100.0 if axis_key == "times" else None),
            allow_duplicate_names=(axis_key == "times"),
        )
        return [dict(item) for item in normalized]

    def _save_features_axis_defaults(
        self,
        *,
        metric_key: str,
        axis_key: str,
        rows: list[dict[str, Any]],
    ) -> None:
        if axis_key == "bands" and self._features_metric_uses_auto_bands(metric_key):
            return
        normalized = self._normalize_feature_axis_rows(
            rows,
            min_start=0.0,
            max_end=(100.0 if axis_key == "times" else None),
            allow_duplicate_names=(axis_key == "times"),
        )
        payload = self._config_store.read_yaml(
            _FEATURES_DEFAULTS_CONFIG_FILENAME,
            default={},
        )
        if not isinstance(payload, dict):
            payload = {}
        node = payload.get(FEATURES_AXES_DEFAULTS_KEY, {})
        if not isinstance(node, dict):
            node = {}
        bucket_name = "bands_by_metric" if axis_key == "bands" else "times_by_metric"
        bucket = node.get(bucket_name, {})
        if not isinstance(bucket, dict):
            bucket = {}
        bucket[str(metric_key)] = [dict(item) for item in normalized]
        node[bucket_name] = bucket
        payload[FEATURES_AXES_DEFAULTS_KEY] = node
        self._config_store.write_yaml(_FEATURES_DEFAULTS_CONFIG_FILENAME, payload)

    def _load_features_plot_advance_defaults(self) -> dict[str, Any]:
        payload = self._load_derive_defaults_runtime(self._config_store)
        node = payload.get("plot_advance_defaults", {})
        if not isinstance(node, dict):
            node = {}
        transform_mode = normalize_feature_plot_transform_mode(
            node.get("transform_mode", "none")
        )
        normalize_mode = str(node.get("normalize_mode", "none")).strip()
        baseline_mode = str(node.get("baseline_mode", "mean")).strip()
        colormap = str(node.get("colormap", "viridis")).strip()
        x_log = bool(node.get("x_log", False))
        y_log = bool(node.get("y_log", False))
        ranges = self._normalize_feature_axis_rows(
            [
                {"name": f"r{idx}", "start": item[0], "end": item[1]}
                for idx, item in enumerate(node.get("baseline_percent_ranges", []))
                if isinstance(item, (list, tuple)) and len(item) == 2
            ],
            min_start=0.0,
            max_end=100.0,
        )
        parsed_ranges = [[float(item["start"]), float(item["end"])] for item in ranges]
        if transform_mode not in FEATURE_PLOT_TRANSFORM_MODES:
            transform_mode = "none"
        if normalize_mode not in FEATURE_PLOT_NORMALIZE_MODES:
            normalize_mode = "none"
        if baseline_mode not in FEATURE_PLOT_BASELINE_MODES:
            baseline_mode = "mean"
        if colormap not in FEATURE_PLOT_COLORMAPS:
            colormap = "viridis"
        return {
            "transform_mode": normalize_feature_plot_transform_mode(transform_mode),
            "normalize_mode": normalize_mode,
            "baseline_mode": baseline_mode,
            "baseline_percent_ranges": parsed_ranges,
            "colormap": colormap,
            "x_log": x_log,
            "y_log": y_log,
        }

    def _save_features_plot_advance_defaults(self, params: dict[str, Any]) -> None:
        payload = self._config_store.read_yaml(
            _FEATURES_DEFAULTS_CONFIG_FILENAME,
            default={},
        )
        if not isinstance(payload, dict):
            payload = {}
        payload["plot_advance_defaults"] = {
            "transform_mode": normalize_feature_plot_transform_mode(
                params.get("transform_mode", "none")
            ),
            "normalize_mode": str(params.get("normalize_mode", "none")),
            "baseline_mode": str(params.get("baseline_mode", "mean")),
            "baseline_percent_ranges": [
                [float(item[0]), float(item[1])]
                for item in params.get("baseline_percent_ranges", [])
                if isinstance(item, (list, tuple)) and len(item) == 2
            ],
            "colormap": str(params.get("colormap", "viridis")),
            "x_log": bool(params.get("x_log", False)),
            "y_log": bool(params.get("y_log", False)),
        }
        self._config_store.write_yaml(_FEATURES_DEFAULTS_CONFIG_FILENAME, payload)
