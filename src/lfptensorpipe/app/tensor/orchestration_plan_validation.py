"""Validation and normalization helpers for tensor runtime plans."""

from __future__ import annotations

from typing import Any

import numpy as np

from .orchestration_plan_models import MetricPlanInputs


def _normalize_metric_selectors(
    svc: Any,
    *,
    metric_key: str,
    metric_label: str,
    metric_params: dict[str, Any],
) -> tuple[list[str] | None, list[tuple[str, str]] | None]:
    if "spectral_mode" in metric_params:
        raise ValueError(
            f"{metric_label} no longer supports 'spectral_mode'. "
            "Use 'method' with values 'morlet' or 'multitaper'."
        )
    metric_channels = svc._normalize_metric_channels(
        metric_params.get("selected_channels")
    )
    if metric_key in svc.TENSOR_CHANNEL_SELECTOR_KEYS and metric_channels == []:
        raise ValueError(f"{metric_label} requires at least one selected channel.")
    metric_pairs = svc._normalize_metric_pairs(metric_params.get("selected_pairs"))
    directed_or_undirected = (
        svc.TENSOR_UNDIRECTED_SELECTOR_KEYS | svc.TENSOR_DIRECTED_SELECTOR_KEYS
    )
    if metric_key in directed_or_undirected and metric_pairs == []:
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
    if metric_key not in svc.TENSOR_COMMON_BASIC_KEYS:
        metric_low, metric_high, metric_step = svc.load_tensor_frequency_defaults(
            context
        )
        return float(metric_low), float(metric_high), float(metric_step)

    metric_low = svc._as_optional_float(metric_params.get("low_freq_hz"))
    metric_high = svc._as_optional_float(metric_params.get("high_freq_hz"))
    metric_step = svc._as_optional_float(metric_params.get("freq_step_hz"))
    if metric_low is None or metric_high is None or metric_step is None:
        raise ValueError(f"{metric_label} requires low/high/step frequency parameters.")
    freq_ok, freq_message, _ = svc.validate_tensor_frequency_params(
        context,
        low_freq=metric_low,
        high_freq=metric_high,
        step_hz=metric_step,
    )
    if not freq_ok:
        raise ValueError(freq_message)
    return float(metric_low), float(metric_high), float(metric_step)


def _resolve_metric_bands(
    svc: Any,
    *,
    metric_key: str,
    metric_params: dict[str, Any],
) -> list[dict[str, Any]]:
    metric_bands = svc._normalize_metric_bands(metric_params.get("bands"))
    if metric_key in svc.TENSOR_BAND_REQUIRED_KEYS:
        valid_bands, band_message = svc._validate_bands(metric_bands)
        if not valid_bands:
            raise ValueError(band_message)
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
    freq_range = metric_params.get("freq_range_hz")
    parsed_freq_range: tuple[float, float] | None = None
    if isinstance(freq_range, (list, tuple)) and len(freq_range) == 2:
        lo = svc._as_optional_float(freq_range[0])
        hi = svc._as_optional_float(freq_range[1])
        if lo is not None and hi is not None and hi > lo:
            parsed_freq_range = (float(lo), float(hi))
    if parsed_freq_range is None:
        raise ValueError(
            "Periodic/APeriodic requires SpecParam freq range as two numeric "
            "values with high > low."
        )

    spec_low, spec_high = parsed_freq_range
    if float(metric_low) < float(spec_low) or float(metric_high) > float(spec_high):
        raise ValueError(
            "Low/high frequency must stay within SpecParam freq range (inclusive)."
        )

    spec_ok, spec_message, _ = svc.validate_tensor_frequency_params(
        context,
        low_freq=float(spec_low),
        high_freq=float(spec_high),
        step_hz=float(metric_step),
    )
    if not spec_ok:
        raise ValueError(
            "SpecParam freq range is out of bounds for preprocess/Nyquist "
            f"constraints: {spec_message}"
        )

    peak_width_limits = metric_params.get("peak_width_limits_hz")
    parsed_peak_width_limits = (2.0, 12.0)
    if isinstance(peak_width_limits, (list, tuple)) and len(peak_width_limits) == 2:
        lo = svc._as_optional_float(peak_width_limits[0])
        hi = svc._as_optional_float(peak_width_limits[1])
        if lo is not None and hi is not None and hi > lo:
            parsed_peak_width_limits = (float(lo), float(hi))

    raw_max_n_peaks = metric_params.get("max_n_peaks")
    if isinstance(raw_max_n_peaks, str) and raw_max_n_peaks.lower() == "inf":
        max_n_peaks = float(np.inf)
    else:
        max_n_peaks = svc._as_float(raw_max_n_peaks, float(np.inf))

    return parsed_freq_range, parsed_peak_width_limits, max_n_peaks


def prepare_metric_plan_inputs(
    svc: Any,
    context: Any,
    *,
    metric_key: str,
    metric_label: str,
    metric_params: dict[str, Any],
) -> MetricPlanInputs:
    metric_channels, metric_pairs = _normalize_metric_selectors(
        svc,
        metric_key=metric_key,
        metric_label=metric_label,
        metric_params=metric_params,
    )
    metric_low, metric_high, metric_step = _resolve_metric_frequency_params(
        svc,
        context,
        metric_label=metric_label,
        metric_key=metric_key,
        metric_params=metric_params,
    )
    metric_bands = _resolve_metric_bands(
        svc,
        metric_key=metric_key,
        metric_params=metric_params,
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
            metric_params=metric_params,
        )

    return MetricPlanInputs(
        metric_key=metric_key,
        metric_params=dict(metric_params),
        metric_channels=metric_channels,
        metric_pairs=metric_pairs,
        metric_low=float(metric_low),
        metric_high=float(metric_high),
        metric_step=float(metric_step),
        metric_bands=metric_bands,
        parsed_freq_range=parsed_freq_range,
        parsed_peak_width_limits=parsed_peak_width_limits,
        max_n_peaks=float(max_n_peaks),
    )
