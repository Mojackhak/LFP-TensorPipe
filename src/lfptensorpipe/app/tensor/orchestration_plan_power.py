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
    method = str(metric_params["method"])
    runner_kwargs = {
        "low_freq": float(metric_low),
        "high_freq": float(metric_high),
        "step_hz": float(metric_step),
        "mask_edge_effects": mask_edge_effects,
        "bands": metric_bands,
        "selected_channels": metric_channels,
        "method": method,
        "time_resolution_s": float(metric_params["time_resolution_s"]),
        "hop_s": float(metric_params["hop_s"]),
        "notches": metric_params["notches"],
        "notch_radii": metric_params["notch_radii"],
    }
    if method == "morlet":
        runner_kwargs.update(
            {
                "min_cycles": float(metric_params["min_cycles"]),
                "max_cycles": metric_params["max_cycles"],
                "mt_time_bandwidth_product": 4.0,
                "mt_min_cycles": 3.0,
            }
        )
    else:
        runner_kwargs.update(
            {
                "mt_time_bandwidth_product": float(
                    metric_params["mt_time_bandwidth_product"]
                ),
                "mt_min_cycles": float(metric_params["mt_min_cycles"]),
                "min_cycles": 3.0,
                "max_cycles": None,
            }
        )
    return RuntimePlan(
        plan_key="raw_power",
        metric_label=svc.TENSOR_METRICS_BY_KEY["raw_power"].display_name,
        runner_key="raw_power",
        runner_kwargs=runner_kwargs,
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
    method = str(metric_params["method"])
    runner_kwargs = {
        "low_freq": float(metric_low),
        "high_freq": float(metric_high),
        "step_hz": float(metric_step),
        "mask_edge_effects": mask_edge_effects,
        "bands": metric_bands,
        "selected_channels": metric_channels,
        "method": method,
        "time_resolution_s": float(metric_params["time_resolution_s"]),
        "hop_s": float(metric_params["hop_s"]),
        "freq_range_hz": parsed_freq_range,
        "freq_smooth_enabled": bool(metric_params["freq_smooth_enabled"]),
        "time_smooth_enabled": bool(metric_params["time_smooth_enabled"]),
        "aperiodic_mode": str(metric_params["aperiodic_mode"]),
        "peak_width_limits_hz": parsed_peak_width_limits,
        "max_n_peaks": max_n_peaks,
        "min_peak_height": float(metric_params["min_peak_height"]),
        "peak_threshold": float(metric_params["peak_threshold"]),
        "fit_qc_threshold": float(metric_params["fit_qc_threshold"]),
        "notches": metric_params["notches"],
        "notch_radii": metric_params["notch_radii"],
    }
    if method == "morlet":
        runner_kwargs.update(
            {
                "min_cycles": float(metric_params["min_cycles"]),
                "max_cycles": metric_params["max_cycles"],
                "mt_time_bandwidth_product": 4.0,
                "mt_min_cycles": 3.0,
            }
        )
    else:
        runner_kwargs.update(
            {
                "mt_time_bandwidth_product": float(
                    metric_params["mt_time_bandwidth_product"]
                ),
                "mt_min_cycles": float(metric_params["mt_min_cycles"]),
                "min_cycles": 3.0,
                "max_cycles": None,
            }
        )
    if bool(metric_params["freq_smooth_enabled"]):
        runner_kwargs["freq_smooth_sigma"] = float(metric_params["freq_smooth_sigma"])
    if bool(metric_params["time_smooth_enabled"]):
        runner_kwargs["time_smooth_kernel_size"] = metric_params[
            "time_smooth_kernel_size"
        ]
    return RuntimePlan(
        plan_key="periodic_aperiodic",
        metric_label=svc.TENSOR_METRICS_BY_KEY["periodic_aperiodic"].display_name,
        runner_key="periodic_aperiodic",
        runner_kwargs=runner_kwargs,
    )


__all__ = ["plan_periodic_aperiodic", "plan_raw_power"]
