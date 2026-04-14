"""Stable tensor workflow surface with patch-friendly helper wrappers."""

from __future__ import annotations

# ruff: noqa: F401

from itertools import combinations, permutations

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import (
    preproc_step_log_path,
    preproc_step_raw_path,
)
from lfptensorpipe.app.runlog_store import indicator_from_log, read_run_log
from lfptensorpipe.io.pkl_io import save_pkl

from .atomic_io import write_outputs_atomically as _write_outputs_atomically
from .annotation_source import load_burst_baseline_annotation_labels
from .coercion import (
    _as_bool,
    _as_float,
    _as_int,
    _as_optional_float,
    _as_optional_int,
    _normalize_metric_method,
    _sanitize_metric_params_for_logs,
)
from .frequency import (
    DEFAULT_TENSOR_BANDS,
    DEFAULT_TENSOR_NOTCH_WIDTH,
    TensorFilterInheritance,
    TensorFrequencyBounds,
    _apply_dynamic_edge_mask_strict,
    _build_frequency_grid,
    _compute_mask_radii_seconds,
    _compute_notch_intervals,
    _cut_frequency_grid_by_intervals,
    _cycles_from_time_resolution,
    _effective_n_jobs_payload,
    _expand_notch_widths,
    _load_finish_nyquist_hz,
    _parse_positive_float_tuple,
    _psi_band_radii_seconds,
    build_tensor_metric_notch_payload,
    compute_tensor_metric_filter_notch_warnings,
    default_tensor_metric_notch_params,
    load_tensor_filter_inheritance,
    load_tensor_filter_metric_notch_params,
    )
from .orchestration import run_build_tensor
from .logging import (
    write_metric_config as _write_metric_config,
    write_metric_log as _write_metric_log,
    write_metric_log_to_path as _write_metric_log_to_path,  # noqa: F401
    write_stage_log as _write_stage_log,
    write_stage_log_to_path as _write_stage_log_to_path,  # noqa: F401
)
from .params import (
    TENSOR_BAND_REQUIRED_KEYS,
    TENSOR_CHANNEL_SELECTOR_KEYS,
    TENSOR_COMMON_BASIC_KEYS,
    TENSOR_DIRECTED_SELECTOR_KEYS,
    TENSOR_METRICS,
    TENSOR_METRICS_BY_KEY,
    TENSOR_UNDIRECTED_SELECTOR_KEYS,
    TensorMetricSpec,
)
from .paths import (
    tensor_metric_config_path,
    tensor_metric_log_path,
    tensor_stage_log_path,
    tensor_metric_tensor_path,
)
from .runner_dispatch import (
    _run_burst_metric,
    _run_periodic_aperiodic_metric,
    _run_psi_metric,
    _run_raw_power_metric,
    _run_trgc_backend_metric,
    _run_trgc_finalize_metric,
    _run_trgc_metric,
    _run_undirected_connectivity_metric,
)
from .selectors import (
    normalize_metric_bands as _normalize_metric_bands,
    normalize_metric_channels as _normalize_metric_channels,
    normalize_metric_pairs as _normalize_metric_pairs,
    normalize_selected_pairs as _normalize_selected_pairs,
)
from .validators import validate_bands as _validate_bands


def resolve_tensor_frequency_bounds(
    context,
    *,
    load_tensor_filter_inheritance_fn=None,
    load_finish_nyquist_hz_fn=None,
):
    inheritance = (load_tensor_filter_inheritance_fn or load_tensor_filter_inheritance)(
        context
    )
    min_low = float(inheritance.low_freq)
    max_high = float(inheritance.high_freq)
    nyquist = (load_finish_nyquist_hz_fn or _load_finish_nyquist_hz)(context)
    if nyquist is not None:
        max_high = min(max_high, float(nyquist))
    if max_high <= min_low:
        max_high = min_low + 1.0
    return TensorFrequencyBounds(
        min_low_freq=min_low,
        max_high_freq=max_high,
        filter_low_freq=float(inheritance.low_freq),
        filter_high_freq=float(inheritance.high_freq),
        nyquist_freq=nyquist,
    )


def validate_tensor_frequency_params(
    context,
    *,
    low_freq,
    high_freq,
    step_hz,
    resolve_tensor_frequency_bounds_fn=None,
    build_frequency_grid_fn=None,
):
    bounds = (resolve_tensor_frequency_bounds_fn or resolve_tensor_frequency_bounds)(
        context
    )
    low = float(low_freq)
    high = float(high_freq)
    step = float(step_hz)

    if low <= 0.0:
        return False, "Low freq must be > 0 Hz.", bounds
    if low < bounds.min_low_freq:
        return (
            False,
            (
                f"Low freq must be >= {bounds.min_low_freq:g} Hz "
                "(inherited from Preprocess Filter low freq)."
            ),
            bounds,
        )
    if high > bounds.max_high_freq:
        upper_source = (
            "min(preproc filter high freq, Nyquist)"
            if bounds.nyquist_freq is not None
            else "preproc filter high freq"
        )
        return (
            False,
            f"High freq must be <= {bounds.max_high_freq:g} Hz ({upper_source}).",
            bounds,
        )
    if high <= low:
        return False, "High freq must be greater than Low freq.", bounds
    if step <= 0.0:
        return False, "Step must be > 0.", bounds
    try:
        (build_frequency_grid_fn or _build_frequency_grid)(low, high, step)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), bounds
    return True, "", bounds


def load_tensor_frequency_defaults(
    context,
    *,
    resolve_tensor_frequency_bounds_fn=None,
):
    bounds = (resolve_tensor_frequency_bounds_fn or resolve_tensor_frequency_bounds)(
        context
    )
    low_freq = max(float(bounds.min_low_freq), 0.5)
    high_freq = float(bounds.max_high_freq)
    step_hz = 0.5
    if high_freq <= low_freq:
        high_freq = low_freq + 1.0
    return low_freq, high_freq, step_hz
