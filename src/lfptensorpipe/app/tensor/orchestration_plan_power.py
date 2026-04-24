"""Power-family runtime plan builders for Build Tensor."""

from __future__ import annotations

from typing import Any

from .orchestration_execution import RuntimePlan


def plan_raw_power(
    svc: Any,
    context: Any,
    *,
    metric_low: float,
    metric_high: float,
    metric_step: float,
    metric_bands: list[dict[str, Any]],
    metric_channels: list[str] | None,
    metric_params: dict[str, Any],
    mask_edge_effects: bool,
) -> RuntimePlan:
    return RuntimePlan(
        plan_key="raw_power",
        metric_label=svc.TENSOR_METRICS_BY_KEY["raw_power"].display_name,
        runner_key="raw_power",
        runner_kwargs={
            "low_freq": float(metric_low),
            "high_freq": float(metric_high),
            "step_hz": float(metric_step),
            "mask_edge_effects": mask_edge_effects,
            "bands": metric_bands,
            "selected_channels": metric_channels,
            "method": str(metric_params.get("method", "morlet")),
            "time_resolution_s": svc._as_float(
                metric_params.get("time_resolution_s"), 0.5
            ),
            "hop_s": svc._as_float(metric_params.get("hop_s"), 0.025),
            "min_cycles": svc._as_optional_float(metric_params.get("min_cycles"), 3.0),
            "max_cycles": svc._as_optional_float(metric_params.get("max_cycles")),
            "time_bandwidth": svc._as_float(metric_params.get("time_bandwidth"), 1.0),
            "notches": metric_params.get("notches"),
            "notch_widths": metric_params.get(
                "notch_widths",
                svc.DEFAULT_TENSOR_NOTCH_WIDTH,
            ),
        },
    )


def plan_periodic_aperiodic(
    svc: Any,
    context: Any,
    *,
    metric_low: float,
    metric_high: float,
    metric_step: float,
    metric_bands: list[dict[str, Any]],
    metric_channels: list[str] | None,
    metric_params: dict[str, Any],
    mask_edge_effects: bool,
    parsed_freq_range: tuple[float, float],
    parsed_peak_width_limits: tuple[float, float],
    max_n_peaks: float,
) -> RuntimePlan:
    return RuntimePlan(
        plan_key="periodic_aperiodic",
        metric_label=svc.TENSOR_METRICS_BY_KEY["periodic_aperiodic"].display_name,
        runner_key="periodic_aperiodic",
        runner_kwargs={
            "low_freq": float(metric_low),
            "high_freq": float(metric_high),
            "step_hz": float(metric_step),
            "mask_edge_effects": mask_edge_effects,
            "bands": metric_bands,
            "selected_channels": metric_channels,
            "method": str(metric_params.get("method", "morlet")),
            "time_resolution_s": svc._as_float(
                metric_params.get("time_resolution_s"), 0.5
            ),
            "hop_s": svc._as_float(metric_params.get("hop_s"), 0.025),
            "min_cycles": svc._as_optional_float(metric_params.get("min_cycles"), 3.0),
            "max_cycles": svc._as_optional_float(metric_params.get("max_cycles")),
            "time_bandwidth": svc._as_float(metric_params.get("time_bandwidth"), 1.0),
            "freq_range_hz": parsed_freq_range,
            "freq_smooth_enabled": bool(metric_params.get("freq_smooth_enabled", True)),
            "freq_smooth_sigma": svc._as_optional_float(
                metric_params.get("freq_smooth_sigma"),
                1.5,
            ),
            "time_smooth_enabled": bool(metric_params.get("time_smooth_enabled", True)),
            "time_smooth_kernel_size": svc._as_optional_int(
                metric_params.get("time_smooth_kernel_size")
            ),
            "aperiodic_mode": str(metric_params.get("aperiodic_mode", "fixed")),
            "peak_width_limits_hz": parsed_peak_width_limits,
            "max_n_peaks": max_n_peaks,
            "min_peak_height": svc._as_float(metric_params.get("min_peak_height"), 0.0),
            "peak_threshold": svc._as_float(metric_params.get("peak_threshold"), 2.0),
            "fit_qc_threshold": svc._as_float(
                metric_params.get("fit_qc_threshold"), 0.6
            ),
            "notches": metric_params.get("notches"),
            "notch_widths": metric_params.get(
                "notch_widths",
                svc.DEFAULT_TENSOR_NOTCH_WIDTH,
            ),
        },
    )


__all__ = ["plan_periodic_aperiodic", "plan_raw_power"]
