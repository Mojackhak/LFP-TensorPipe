"""Private defaults for Align Epochs preview plotting."""

from __future__ import annotations

from typing import Any

import numpy as np

from lfptensorpipe.app.tensor import TENSOR_METRICS_BY_KEY

_ALIGNMENT_PREVIEW_CONFIG_FILENAME = "alignment_preview.yml"
_ALIGNMENT_PREVIEW_BOXSIZE_DEFAULT = (60.0, 50.0)
_ALIGNMENT_PREVIEW_FONT_SIZE_DEFAULT = 16.0
_ALIGNMENT_PREVIEW_TICK_LABEL_SIZE_DEFAULT = 12.0
_ALIGNMENT_PREVIEW_X_LABEL_OFFSET_MM_DEFAULT = 10.0
_ALIGNMENT_PREVIEW_Y_LABEL_OFFSET_MM_DEFAULT = 18.0
_ALIGNMENT_PREVIEW_COLORBAR_PAD_MM_DEFAULT = 4.0
_ALIGNMENT_PREVIEW_CBAR_LABEL_OFFSET_MM_DEFAULT = 18.0
_ALIGNMENT_PREVIEW_LABEL_OVERRIDES = {
    "periodic": "Periodic",
    "aperiodic": "Aperiodic",
}


def _alignment_preview_metric_label(metric_key: str) -> str:
    key = str(metric_key).strip()
    spec = TENSOR_METRICS_BY_KEY.get(key)
    if spec is not None:
        return spec.display_name
    return _ALIGNMENT_PREVIEW_LABEL_OVERRIDES.get(key, key or "Metric")


def _positive_float_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed) or parsed <= 0.0:
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


class MainWindowAlignmentDefaultsMixin:
    def _default_alignment_preview_plot_params(
        self,
        metric_key: str,
    ) -> dict[str, Any]:
        return {
            "boxsize": tuple(_ALIGNMENT_PREVIEW_BOXSIZE_DEFAULT),
            "font_size": float(_ALIGNMENT_PREVIEW_FONT_SIZE_DEFAULT),
            "tick_label_size": float(_ALIGNMENT_PREVIEW_TICK_LABEL_SIZE_DEFAULT),
            "x_label_offset_mm": float(_ALIGNMENT_PREVIEW_X_LABEL_OFFSET_MM_DEFAULT),
            "y_label_offset_mm": float(_ALIGNMENT_PREVIEW_Y_LABEL_OFFSET_MM_DEFAULT),
            "colorbar_pad_mm": float(_ALIGNMENT_PREVIEW_COLORBAR_PAD_MM_DEFAULT),
            "cbar_label_offset_mm": float(
                _ALIGNMENT_PREVIEW_CBAR_LABEL_OFFSET_MM_DEFAULT
            ),
            "colorbar_label": _alignment_preview_metric_label(metric_key),
        }

    def _normalize_alignment_preview_plot_params(
        self,
        metric_key: str,
        raw_params: dict[str, Any] | None,
        *,
        defaults: dict[str, Any],
    ) -> dict[str, Any]:
        params = raw_params if isinstance(raw_params, dict) else {}
        label_default = _alignment_preview_metric_label(metric_key)
        colorbar_label_raw = params.get("colorbar_label", defaults["colorbar_label"])
        colorbar_label = (
            str(colorbar_label_raw).strip() if colorbar_label_raw is not None else ""
        )
        if not colorbar_label:
            colorbar_label = label_default
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
            "colorbar_label": colorbar_label,
        }

    def _load_alignment_preview_plot_params(self, metric_key: str) -> dict[str, Any]:
        defaults = self._default_alignment_preview_plot_params(metric_key)
        payload = self._config_store.read_yaml(
            _ALIGNMENT_PREVIEW_CONFIG_FILENAME,
            default={},
        )
        if not isinstance(payload, dict):
            return defaults
        metric_defaults = payload.get("metric_defaults", {})
        if not isinstance(metric_defaults, dict):
            return defaults

        normalized = defaults
        default_node = metric_defaults.get("default")
        if isinstance(default_node, dict):
            normalized = self._normalize_alignment_preview_plot_params(
                metric_key,
                default_node,
                defaults=normalized,
            )
        metric_node = metric_defaults.get(str(metric_key).strip())
        if isinstance(metric_node, dict):
            normalized = self._normalize_alignment_preview_plot_params(
                metric_key,
                metric_node,
                defaults=normalized,
            )
        return normalized
