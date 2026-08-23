"""Validation and normalization helpers for tensor runtime plans."""

from __future__ import annotations

from typing import Any

import numpy as np

from lfptensorpipe.lfp.burst.semantics import (
    normalize_burst_method,
    normalize_hilbert_filter_method,
)

from .frequency import (
    build_tensor_metric_notch_payload,
    validate_periodic_aperiodic_notch_bounds,
)
from .orchestration_plan_models import MetricPlanInputs
from .params import DEFAULT_TENSOR_BANDS

_MISSING = object()
_SPECTRAL_METRIC_KEYS = frozenset(
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
_COMMON_FREQUENCY_METRIC_KEYS = frozenset(
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
    }
)
_METRIC_PASSTHROUGH_KEYS: dict[str, frozenset[str]] = {
    "raw_power": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "notches",
            "notch_radii",
            "selected_channels",
        }
    ),
    "periodic_aperiodic": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "freq_range_hz",
            "notches",
            "notch_radii",
            "selected_channels",
        }
    ),
    "coherence": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "notches",
            "notch_radii",
            "selected_pairs",
        }
    ),
    "imcoh_abs": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "notches",
            "notch_radii",
            "selected_pairs",
        }
    ),
    "plv": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "notches",
            "notch_radii",
            "selected_pairs",
        }
    ),
    "ciplv": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "notches",
            "notch_radii",
            "selected_pairs",
        }
    ),
    "pli": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "notches",
            "notch_radii",
            "selected_pairs",
        }
    ),
    "wpli": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "notches",
            "notch_radii",
            "selected_pairs",
        }
    ),
    "trgc": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "notches",
            "notch_radii",
            "selected_pairs",
        }
    ),
    "psi": frozenset(
        {
            "freq_step_hz",
            "bands",
            "notches",
            "notch_radii",
            "selected_pairs",
        }
    ),
    "burst": frozenset(
        {
            "bands",
            "notches",
            "notch_radii",
            "selected_channels",
            "thresholds",
            "thresholds_source_path",
            "baseline_keep",
            "boundary_isolated_filter",
            "method",
            "freq_step_hz",
            "morlet_n_cycles",
            "mt_n_cycles",
            "mt_time_bandwidth_product",
            "hilbert_filter_method",
            "hilbert_edge_tolerance_pct",
        }
    ),
}


def _provided_value(
    params: dict[str, Any],
    key: str,
    *,
    default: Any = _MISSING,
) -> Any:
    raw = params.get(key, _MISSING)
    if raw is _MISSING:
        if default is _MISSING:
            raise ValueError(f"{key} is required.")
        return default
    if raw is None:
        raise ValueError(f"{key} must not be empty.")
    return raw


def _finite_float(
    params: dict[str, Any],
    key: str,
    *,
    default: Any = _MISSING,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = False,
    maximum_inclusive: bool = False,
) -> float:
    raw = _provided_value(params, key, default=default)
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{key} must be numeric.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric.") from exc
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite.")
    if minimum is not None:
        below = value < minimum if minimum_inclusive else value <= minimum
        if below:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{key} must be {operator} {minimum:g}.")
    if maximum is not None:
        above = value > maximum if maximum_inclusive else value >= maximum
        if above:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{key} must be {operator} {maximum:g}.")
    return value


def _optional_finite_float(
    params: dict[str, Any],
    key: str,
) -> float | None:
    raw = params.get(key, _MISSING)
    if raw is _MISSING or raw is None:
        return None
    return _finite_float(params, key)


def _positive_integer(
    params: dict[str, Any],
    key: str,
    *,
    default: Any = _MISSING,
) -> int:
    raw = _provided_value(params, key, default=default)
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"{key} must be a positive integer.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer.") from exc
    if not np.isfinite(value) or not value.is_integer() or value < 1.0:
        raise ValueError(f"{key} must be a positive integer.")
    return int(value)


def _optional_positive_integer(
    params: dict[str, Any],
    key: str,
) -> int | None:
    raw = params.get(key, _MISSING)
    if raw is _MISSING or raw is None:
        return None
    return _positive_integer(params, key)


def _strict_bool(
    params: dict[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    raw = _provided_value(params, key, default=default)
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    if isinstance(raw, (int, np.integer)) and int(raw) in {0, 1}:
        return bool(raw)
    token = str(raw).strip().lower()
    if token in {"true", "yes", "on", "1"}:
        return True
    if token in {"false", "no", "off", "0"}:
        return False
    raise ValueError(f"{key} must be boolean.")


def _spectral_method(
    params: dict[str, Any],
    *,
    metric_label: str,
) -> str:
    raw = _provided_value(params, "method", default="morlet")
    method = str(raw).strip().lower()
    if method not in {"morlet", "multitaper"}:
        raise ValueError(
            f"{metric_label} method must be 'morlet' or 'multitaper', got: {raw!r}"
        )
    return method


def _finite_pair(
    params: dict[str, Any],
    key: str,
    *,
    default: tuple[float, float],
    positive_low: bool,
) -> tuple[float, float]:
    raw = _provided_value(params, key, default=default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"{key} must contain exactly two numeric values.")
    pair_params = {"low": raw[0], "high": raw[1]}
    low = _finite_float(pair_params, "low")
    high = _finite_float(pair_params, "high")
    if positive_low and low <= 0.0:
        raise ValueError(f"{key} must satisfy 0 < low < high.")
    if high <= low:
        relation = "0 < low < high" if positive_low else "high > low"
        raise ValueError(f"{key} must satisfy {relation}.")
    return low, high


def _max_n_peaks(params: dict[str, Any]) -> float:
    raw = _provided_value(params, "max_n_peaks", default=float("inf"))
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError("max_n_peaks must be a non-negative integer or inf.")
    if isinstance(raw, str) and raw.strip().lower() == "inf":
        return float("inf")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_n_peaks must be a non-negative integer or inf.") from exc
    if np.isposinf(value):
        return float("inf")
    if not np.isfinite(value) or not value.is_integer() or value < 0.0:
        raise ValueError("max_n_peaks must be a non-negative integer or inf.")
    return value


def _normalize_metric_compute_params(
    *,
    metric_key: str,
    metric_label: str,
    metric_params: dict[str, Any],
) -> dict[str, Any]:
    passthrough_keys = _METRIC_PASSTHROUGH_KEYS.get(metric_key, frozenset())
    normalized = {
        key: value for key, value in metric_params.items() if key in passthrough_keys
    }
    raw_notches = metric_params.get("notches")
    raw_notch_radii = metric_params.get("notch_radii", _MISSING)
    if raw_notches and raw_notch_radii is None:
        raise ValueError("notch_radii must not be empty when notches are configured.")
    normalized.update(
        build_tensor_metric_notch_payload(
            raw_notches,
            None if raw_notch_radii is _MISSING else raw_notch_radii,
        )
    )

    if metric_key in _SPECTRAL_METRIC_KEYS:
        method = _spectral_method(metric_params, metric_label=metric_label)
        normalized["method"] = method
        normalized["time_resolution_s"] = _finite_float(
            metric_params,
            "time_resolution_s",
            default=0.5,
            minimum=0.0,
        )
        normalized["hop_s"] = _finite_float(
            metric_params,
            "hop_s",
            default=0.025,
            minimum=0.0,
        )
        if method == "morlet":
            min_cycles = _finite_float(
                metric_params,
                "min_cycles",
                default=3.0,
                minimum=0.0,
            )
            max_cycles = _optional_finite_float(metric_params, "max_cycles")
            if max_cycles is not None and max_cycles < min_cycles:
                raise ValueError("max_cycles must be >= min_cycles.")
            normalized["min_cycles"] = min_cycles
            normalized["max_cycles"] = max_cycles
        else:
            normalized["mt_time_bandwidth_product"] = _finite_float(
                metric_params,
                "mt_time_bandwidth_product",
                default=4.0,
                minimum=2.0,
                minimum_inclusive=True,
            )
            mt_min_cycles = _finite_float(
                metric_params,
                "mt_min_cycles",
                default=3.0,
                minimum=0.0,
            )
            mt_max_cycles = _optional_finite_float(metric_params, "mt_max_cycles")
            if mt_max_cycles is not None and mt_max_cycles < mt_min_cycles:
                raise ValueError("mt_max_cycles must be >= mt_min_cycles.")
            normalized["mt_min_cycles"] = mt_min_cycles
            normalized["mt_max_cycles"] = mt_max_cycles

    if metric_key == "periodic_aperiodic":
        freq_smooth_enabled = _strict_bool(
            metric_params,
            "freq_smooth_enabled",
            default=True,
        )
        normalized["freq_smooth_enabled"] = freq_smooth_enabled
        if freq_smooth_enabled:
            normalized["freq_smooth_sigma"] = _finite_float(
                metric_params,
                "freq_smooth_sigma",
                default=1.5,
                minimum=0.0,
            )

        time_smooth_enabled = _strict_bool(
            metric_params,
            "time_smooth_enabled",
            default=True,
        )
        normalized["time_smooth_enabled"] = time_smooth_enabled
        if time_smooth_enabled:
            normalized["time_smooth_kernel_size"] = _optional_positive_integer(
                metric_params,
                "time_smooth_kernel_size",
            )

        mode_raw = _provided_value(metric_params, "aperiodic_mode", default="fixed")
        aperiodic_mode = str(mode_raw).strip().lower()
        if aperiodic_mode not in {"fixed", "knee"}:
            raise ValueError("aperiodic_mode must be 'fixed' or 'knee'.")
        normalized["aperiodic_mode"] = aperiodic_mode
        normalized["peak_width_limits_hz"] = list(
            _finite_pair(
                metric_params,
                "peak_width_limits_hz",
                default=(2.0, 12.0),
                positive_low=True,
            )
        )
        max_n_peaks = _max_n_peaks(metric_params)
        normalized["max_n_peaks"] = "inf" if np.isposinf(max_n_peaks) else max_n_peaks
        normalized["min_peak_height"] = _finite_float(
            metric_params,
            "min_peak_height",
            default=0.0,
            minimum=0.0,
            minimum_inclusive=True,
        )
        normalized["peak_threshold"] = _finite_float(
            metric_params,
            "peak_threshold",
            default=2.0,
            minimum=0.0,
        )
        normalized["fit_qc_threshold"] = _finite_float(
            metric_params,
            "fit_qc_threshold",
            default=0.6,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=True,
            maximum_inclusive=True,
        )

    if metric_key == "trgc":
        normalized["gc_n_lags"] = _positive_integer(
            metric_params,
            "gc_n_lags",
            default=20,
        )
        group_by_samples = _strict_bool(
            metric_params,
            "group_by_samples",
            default=False,
        )
        normalized["group_by_samples"] = group_by_samples
        if not group_by_samples:
            normalized["round_ms"] = _finite_float(
                metric_params,
                "round_ms",
                default=50.0,
                minimum=0.0,
            )

    if metric_key == "burst":
        method = normalize_burst_method(metric_params.get("method", "hilbert"))
        normalized["method"] = method
        if method == "hilbert":
            normalized["hilbert_filter_method"] = normalize_hilbert_filter_method(
                metric_params.get("hilbert_filter_method", "iir")
            )
            normalized["hilbert_edge_tolerance_pct"] = _finite_float(
                metric_params,
                "hilbert_edge_tolerance_pct",
                default=10.0,
                minimum=0.0,
                maximum=100.0,
            )
        elif method == "morlet":
            normalized["freq_step_hz"] = _finite_float(
                metric_params,
                "freq_step_hz",
                default=1.0,
                minimum=0.0,
            )
            normalized["morlet_n_cycles"] = _finite_float(
                metric_params,
                "morlet_n_cycles",
                default=6.0,
                minimum=0.0,
            )
        else:
            normalized["freq_step_hz"] = _finite_float(
                metric_params,
                "freq_step_hz",
                default=1.0,
                minimum=0.0,
            )
            normalized["mt_n_cycles"] = _finite_float(
                metric_params,
                "mt_n_cycles",
                default=7.0,
                minimum=0.0,
            )
            normalized["mt_time_bandwidth_product"] = _finite_float(
                metric_params,
                "mt_time_bandwidth_product",
                default=4.0,
                minimum=2.0,
                minimum_inclusive=True,
            )
        min_cycles = _finite_float(
            metric_params,
            "min_cycles",
            default=2.0,
            minimum=0.0,
        )
        max_cycles = _optional_finite_float(metric_params, "max_cycles")
        if max_cycles is not None and max_cycles < min_cycles:
            raise ValueError("max_cycles must be >= min_cycles.")
        normalized["min_cycles"] = min_cycles
        normalized["max_cycles"] = max_cycles
        normalized["boundary_isolated_filter"] = _strict_bool(
            metric_params,
            "boundary_isolated_filter",
            default=True,
        )
        normalized["hop_s"] = None
        normalized["decim"] = 1
        if metric_params.get("thresholds") is None:
            normalized["percentile"] = _finite_float(
                metric_params,
                "percentile",
                default=75.0,
                minimum=0.0,
                maximum=100.0,
            )
        else:
            normalized.pop("baseline_keep", None)

    return normalized


def _normalize_metric_selectors(
    svc: Any,
    *,
    metric_key: str,
    metric_label: str,
    metric_params: dict[str, Any],
    allow_empty_selectors: bool = False,
) -> tuple[list[str] | None, list[tuple[str, str]] | None]:
    if "spectral_mode" in metric_params:
        raise ValueError(
            f"{metric_label} no longer supports 'spectral_mode'. "
            "Use 'method' with values 'morlet' or 'multitaper'."
        )
    metric_channels = svc._normalize_metric_channels(
        metric_params.get("selected_channels")
    )
    empty_channel_draft = isinstance(
        metric_params.get("selected_channels"), list
    ) and not metric_params.get("selected_channels")
    if (
        metric_key in svc.TENSOR_CHANNEL_SELECTOR_KEYS
        and not metric_channels
        and not (allow_empty_selectors and empty_channel_draft)
    ):
        raise ValueError(f"{metric_label} requires at least one selected channel.")
    metric_pairs = svc._normalize_metric_pairs(metric_params.get("selected_pairs"))
    directed_or_undirected = (
        svc.TENSOR_UNDIRECTED_SELECTOR_KEYS | svc.TENSOR_DIRECTED_SELECTOR_KEYS
    )
    empty_pair_draft = isinstance(
        metric_params.get("selected_pairs"), list
    ) and not metric_params.get("selected_pairs")
    if (
        metric_key in directed_or_undirected
        and not metric_pairs
        and not (allow_empty_selectors and empty_pair_draft)
    ):
        raise ValueError(f"{metric_label} requires at least one selected pair.")
    return metric_channels, metric_pairs


def _resolve_metric_frequency_params(
    svc: Any,
    context: Any,
    *,
    metric_label: str,
    metric_key: str,
    metric_params: dict[str, Any],
) -> tuple[float, float, float]:
    default_low, default_high, default_step = svc.load_tensor_frequency_defaults(
        context
    )
    if metric_key not in svc.TENSOR_COMMON_BASIC_KEYS:
        metric_low = float(default_low)
        metric_high = float(default_high)
        metric_step = (
            _finite_float(
                metric_params,
                "freq_step_hz",
                default=default_step,
                minimum=0.0,
            )
            if metric_key == "psi"
            else float(default_step)
        )
        if metric_key == "psi":
            freq_ok, freq_message, _ = svc.validate_tensor_frequency_params(
                context,
                low_freq=metric_low,
                high_freq=metric_high,
                step_hz=metric_step,
            )
            if not freq_ok:
                raise ValueError(freq_message)
            metric_params["freq_step_hz"] = metric_step
        return metric_low, metric_high, metric_step

    metric_low = _finite_float(
        metric_params,
        "low_freq_hz",
        default=default_low,
        minimum=0.0,
    )
    metric_high = _finite_float(
        metric_params,
        "high_freq_hz",
        default=default_high,
    )
    metric_step = _finite_float(
        metric_params,
        "freq_step_hz",
        default=default_step,
        minimum=0.0,
    )
    freq_ok, freq_message, _ = svc.validate_tensor_frequency_params(
        context,
        low_freq=metric_low,
        high_freq=metric_high,
        step_hz=metric_step,
    )
    if not freq_ok:
        raise ValueError(f"{metric_label}: {freq_message}")
    metric_params["low_freq_hz"] = metric_low
    metric_params["high_freq_hz"] = metric_high
    metric_params["freq_step_hz"] = metric_step
    return metric_low, metric_high, metric_step


def _strict_metric_bands(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("At least one band is required.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Band row {index + 1} must be an object.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Band row {index + 1} has empty name.")
        if name in seen:
            raise ValueError(f"Duplicate band name: {name}")
        seen.add(name)
        row = {"start": item.get("start"), "end": item.get("end")}
        start = _finite_float(row, "start", minimum=0.0)
        end = _finite_float(row, "end")
        if end <= start:
            raise ValueError(f"Band row {index + 1} must satisfy 0 < start < end.")
        normalized.append({"name": name, "start": start, "end": end})
    return normalized


def _resolve_metric_bands(
    svc: Any,
    *,
    metric_key: str,
    metric_params: dict[str, Any],
) -> list[dict[str, Any]]:
    if metric_key not in svc.TENSOR_BAND_REQUIRED_KEYS:
        return svc._normalize_metric_bands(metric_params.get("bands"))
    raw_bands = metric_params.get("bands", _MISSING)
    if raw_bands is _MISSING:
        raw_bands = [dict(item) for item in DEFAULT_TENSOR_BANDS]
    elif raw_bands is None:
        raise ValueError("bands must not be empty.")
    metric_bands = _strict_metric_bands(raw_bands)
    metric_params["bands"] = [dict(item) for item in metric_bands]
    return metric_bands


def _resolve_periodic_aperiodic_inputs(
    svc: Any,
    context: Any,
    *,
    metric_low: float,
    metric_high: float,
    metric_step: float,
    metric_params: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    parsed_freq_range = _finite_pair(
        metric_params,
        "freq_range_hz",
        default=(metric_low, metric_high),
        positive_low=True,
    )
    metric_params["freq_range_hz"] = list(parsed_freq_range)
    spec_low, spec_high = parsed_freq_range
    if metric_low < spec_low or metric_high > spec_high:
        raise ValueError(
            "Low/high frequency must stay within SpecParam freq range (inclusive)."
        )

    spec_ok, spec_message, _ = svc.validate_tensor_frequency_params(
        context,
        low_freq=spec_low,
        high_freq=spec_high,
        step_hz=metric_step,
    )
    if not spec_ok:
        raise ValueError(
            "SpecParam freq range is out of bounds for preprocess/Nyquist "
            f"constraints: {spec_message}"
        )
    parsed_peak_width_limits = tuple(metric_params["peak_width_limits_hz"])
    max_n_peaks = float(metric_params["max_n_peaks"])
    validate_periodic_aperiodic_notch_bounds(metric_params)
    return parsed_freq_range, parsed_peak_width_limits, max_n_peaks


def validate_metric_storage_params(
    *,
    metric_key: str,
    metric_label: str,
    metric_params: dict[str, Any],
    allow_empty_selectors: bool = False,
) -> None:
    """Validate stored values, optionally retaining an empty selector draft."""
    normalized = _normalize_metric_compute_params(
        metric_key=metric_key,
        metric_label=metric_label,
        metric_params=metric_params,
    )
    if metric_key in _COMMON_FREQUENCY_METRIC_KEYS:
        low = _finite_float(metric_params, "low_freq_hz", minimum=0.0)
        high = _finite_float(metric_params, "high_freq_hz")
        _finite_float(metric_params, "freq_step_hz", minimum=0.0)
        if high <= low:
            raise ValueError("high_freq_hz must be greater than low_freq_hz.")
    elif metric_key == "psi":
        _finite_float(metric_params, "freq_step_hz", minimum=0.0)

    if metric_key == "periodic_aperiodic":
        freq_range = _finite_pair(
            metric_params,
            "freq_range_hz",
            default=(
                float(metric_params["low_freq_hz"]),
                float(metric_params["high_freq_hz"]),
            ),
            positive_low=True,
        )
        if low < freq_range[0] or high > freq_range[1]:
            raise ValueError(
                "Low/high frequency must stay within SpecParam freq range (inclusive)."
            )
        validate_periodic_aperiodic_notch_bounds(
            normalized | {"freq_range_hz": list(freq_range)}
        )

    if metric_key in {"psi", "burst"}:
        raw_bands = metric_params.get("bands", _MISSING)
        if raw_bands is _MISSING:
            raw_bands = [dict(item) for item in DEFAULT_TENSOR_BANDS]
        elif raw_bands is None:
            raise ValueError("bands must not be empty.")
        _strict_metric_bands(raw_bands)

    if "selected_channels" in metric_params:
        channels = metric_params.get("selected_channels")
        if not isinstance(channels, list) or (
            not channels and not allow_empty_selectors
        ):
            raise ValueError(f"{metric_label} requires at least one selected channel.")
    if "selected_pairs" in metric_params:
        pairs = metric_params.get("selected_pairs")
        if not isinstance(pairs, list) or (not pairs and not allow_empty_selectors):
            raise ValueError(f"{metric_label} requires at least one selected pair.")


def prepare_metric_plan_inputs(
    svc: Any,
    context: Any,
    *,
    metric_key: str,
    metric_label: str,
    metric_params: dict[str, Any],
    allow_empty_selectors: bool = False,
) -> MetricPlanInputs:
    """Normalize plan inputs, requiring compute-ready selectors by default."""
    normalized_params = _normalize_metric_compute_params(
        metric_key=metric_key,
        metric_label=metric_label,
        metric_params=metric_params,
    )
    metric_channels, metric_pairs = _normalize_metric_selectors(
        svc,
        metric_key=metric_key,
        metric_label=metric_label,
        metric_params=normalized_params,
        allow_empty_selectors=allow_empty_selectors,
    )
    if metric_channels is not None:
        normalized_params["selected_channels"] = list(metric_channels)
    if metric_pairs is not None:
        normalized_params["selected_pairs"] = [list(pair) for pair in metric_pairs]
    metric_low, metric_high, metric_step = _resolve_metric_frequency_params(
        svc,
        context,
        metric_label=metric_label,
        metric_key=metric_key,
        metric_params=normalized_params,
    )
    metric_bands = _resolve_metric_bands(
        svc,
        metric_key=metric_key,
        metric_params=normalized_params,
    )

    parsed_freq_range = None
    parsed_peak_width_limits = (2.0, 12.0)
    max_n_peaks = float(np.inf)
    if metric_key == "periodic_aperiodic":
        (
            parsed_freq_range,
            parsed_peak_width_limits,
            max_n_peaks,
        ) = _resolve_periodic_aperiodic_inputs(
            svc,
            context,
            metric_low=metric_low,
            metric_high=metric_high,
            metric_step=metric_step,
            metric_params=normalized_params,
        )

    return MetricPlanInputs(
        metric_key=metric_key,
        metric_params=normalized_params,
        metric_channels=metric_channels,
        metric_pairs=metric_pairs,
        metric_low=metric_low,
        metric_high=metric_high,
        metric_step=metric_step,
        metric_bands=metric_bands,
        parsed_freq_range=parsed_freq_range,
        parsed_peak_width_limits=parsed_peak_width_limits,
        max_n_peaks=max_n_peaks,
    )
