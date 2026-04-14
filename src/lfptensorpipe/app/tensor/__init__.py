"""Build-tensor orchestration package.

App layer performs validation and routing; tensor computations should live in
`lfptensorpipe.lfp` backend modules.
"""

from __future__ import annotations

from .frequency import (
    DEFAULT_TENSOR_BANDS,
    DEFAULT_TENSOR_NOTCH_WIDTH,
    TensorFilterInheritance,
    TensorFrequencyBounds,
    build_tensor_metric_notch_payload,
    compute_tensor_metric_filter_notch_warnings,
    default_tensor_metric_notch_params,
    load_tensor_filter_inheritance,
    load_tensor_filter_metric_notch_params,
    load_tensor_frequency_defaults,
    normalize_tensor_metric_notch_params,
    resolve_tensor_frequency_bounds,
    resolve_tensor_metric_interest_range,
    validate_tensor_frequency_params,
)
from .annotation_source import load_burst_baseline_annotation_labels
from .indicator import tensor_metric_panel_state
from .io import (
    tensor_metric_config_path,
    tensor_metric_log_path,
    tensor_metric_tensor_path,
)
from .params import TENSOR_METRICS, TENSOR_METRICS_BY_KEY, TensorMetricSpec
from .service import run_build_tensor

__all__ = [
    "DEFAULT_TENSOR_BANDS",
    "DEFAULT_TENSOR_NOTCH_WIDTH",
    "TENSOR_METRICS",
    "TENSOR_METRICS_BY_KEY",
    "TensorFilterInheritance",
    "TensorFrequencyBounds",
    "TensorMetricSpec",
    "build_tensor_metric_notch_payload",
    "compute_tensor_metric_filter_notch_warnings",
    "default_tensor_metric_notch_params",
    "load_burst_baseline_annotation_labels",
    "load_tensor_filter_inheritance",
    "load_tensor_filter_metric_notch_params",
    "load_tensor_frequency_defaults",
    "normalize_tensor_metric_notch_params",
    "resolve_tensor_frequency_bounds",
    "resolve_tensor_metric_interest_range",
    "run_build_tensor",
    "tensor_metric_config_path",
    "tensor_metric_log_path",
    "tensor_metric_panel_state",
    "tensor_metric_tensor_path",
    "validate_tensor_frequency_params",
]
