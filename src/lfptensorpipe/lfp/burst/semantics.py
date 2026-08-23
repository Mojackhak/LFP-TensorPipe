"""Shared Burst tensor value semantics."""

from __future__ import annotations

import math
from typing import Any, Mapping

BURST_NATIVE_DECIM = 1
BURST_NATIVE_HOP_S = None
BURST_SAMPLE_SUPPORT_KEY = "burst_sample_support"
BURST_SAMPLE_SUPPORT = "left_edge_half_open"
BURST_VALUE_SEMANTICS: Mapping[str, Any] = {
    "non_burst_value": 0.0,
    "invalid_value": "nan",
    "burst_value": "threshold_normalized_magnitude",
    "accepted_min_exclusive": 1.0,
    "threshold_requirement": "finite_positive",
    "notch_interpolation": "none",
    "time_grid": "native_sampling_rate",
}


def burst_value_semantics() -> dict[str, Any]:
    """Return a serialization-safe copy of the current Burst value contract."""
    return dict(BURST_VALUE_SEMANTICS)


def has_current_burst_value_semantics(value: object) -> bool:
    """Return whether a serialized value declares the current Burst contract."""
    return isinstance(value, Mapping) and dict(value) == dict(BURST_VALUE_SEMANTICS)


def _positive_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive finite number.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be a positive finite number.")
    return parsed


def normalize_burst_method(value: Any) -> str:
    """Return one supported Burst estimator method name."""
    method = str(value).strip().lower()
    if method not in {"hilbert", "morlet", "multitaper"}:
        raise ValueError("Burst method must be 'hilbert', 'morlet', or 'multitaper'.")
    return method


def burst_estimator_signature(
    *,
    method: Any,
    filter_order: Any = 4,
    hilbert_edge_tolerance_pct: Any = 10.0,
    freq_step_hz: Any = 1.0,
    morlet_n_cycles: Any = 6.0,
    mt_n_cycles: Any = 7.0,
    mt_time_bandwidth_product: Any = 4.0,
) -> dict[str, Any]:
    """Return the canonical active-method Burst estimator signature."""
    method_name = normalize_burst_method(method)
    if method_name == "hilbert":
        if isinstance(filter_order, bool):
            raise ValueError("filter_order must be a positive integer.")
        try:
            order = int(filter_order)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("filter_order must be a positive integer.") from exc
        if order <= 0 or float(order) != float(filter_order):
            raise ValueError("filter_order must be a positive integer.")
        tolerance = _positive_finite(
            hilbert_edge_tolerance_pct,
            field="hilbert_edge_tolerance_pct",
        )
        if tolerance >= 100.0:
            raise ValueError(
                "hilbert_edge_tolerance_pct must be finite and in (0, 100)."
            )
        return {
            "method": "hilbert",
            "filter_order": order,
            "edge_tolerance_pct": tolerance,
            "phase": "zero",
            "band_magnitude": "real_subband_sum_then_hilbert_abs",
        }
    step = _positive_finite(freq_step_hz, field="freq_step_hz")
    if method_name == "morlet":
        return {
            "method": "morlet",
            "freq_step_hz": step,
            "n_cycles": _positive_finite(
                morlet_n_cycles,
                field="morlet_n_cycles",
            ),
            "band_magnitude": "sqrt_mean_power",
        }
    product = _positive_finite(
        mt_time_bandwidth_product,
        field="mt_time_bandwidth_product",
    )
    if product < 2.0:
        raise ValueError("mt_time_bandwidth_product must be finite and >= 2.")
    return {
        "method": "multitaper",
        "freq_step_hz": step,
        "n_cycles": _positive_finite(mt_n_cycles, field="mt_n_cycles"),
        "time_bandwidth_product": product,
        "band_magnitude": "sqrt_mean_power",
    }


def normalize_burst_estimator_signature(value: Any) -> dict[str, Any]:
    """Validate a serialized Burst estimator signature without compatibility."""
    if not isinstance(value, Mapping):
        raise ValueError("Burst estimator signature must be an object.")
    method = normalize_burst_method(value.get("method"))
    if method == "hilbert":
        expected_keys = {
            "method",
            "filter_order",
            "edge_tolerance_pct",
            "phase",
            "band_magnitude",
        }
        if set(value) != expected_keys:
            raise ValueError("Hilbert estimator signature has invalid keys.")
        normalized = burst_estimator_signature(
            method=method,
            filter_order=value.get("filter_order"),
            hilbert_edge_tolerance_pct=value.get("edge_tolerance_pct"),
        )
    elif method == "morlet":
        expected_keys = {"method", "freq_step_hz", "n_cycles", "band_magnitude"}
        if set(value) != expected_keys:
            raise ValueError("Morlet estimator signature has invalid keys.")
        normalized = burst_estimator_signature(
            method=method,
            freq_step_hz=value.get("freq_step_hz"),
            morlet_n_cycles=value.get("n_cycles"),
        )
    else:
        expected_keys = {
            "method",
            "freq_step_hz",
            "n_cycles",
            "time_bandwidth_product",
            "band_magnitude",
        }
        if set(value) != expected_keys:
            raise ValueError("Multitaper estimator signature has invalid keys.")
        normalized = burst_estimator_signature(
            method=method,
            freq_step_hz=value.get("freq_step_hz"),
            mt_n_cycles=value.get("n_cycles"),
            mt_time_bandwidth_product=value.get("time_bandwidth_product"),
        )
    if dict(value) != normalized:
        raise ValueError("Burst estimator signature is not canonical.")
    return normalized


def has_compatible_burst_value_semantics(
    value: object,
    *,
    bands_segments_hz: object,
) -> bool:
    """Return whether the value uses the current contract without migration."""
    _ = bands_segments_hz
    return has_current_burst_value_semantics(value)


__all__ = [
    "BURST_NATIVE_DECIM",
    "BURST_NATIVE_HOP_S",
    "BURST_SAMPLE_SUPPORT",
    "BURST_SAMPLE_SUPPORT_KEY",
    "BURST_VALUE_SEMANTICS",
    "burst_estimator_signature",
    "burst_value_semantics",
    "has_compatible_burst_value_semantics",
    "has_current_burst_value_semantics",
    "normalize_burst_estimator_signature",
    "normalize_burst_method",
]
