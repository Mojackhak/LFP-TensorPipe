"""Connectivity-family runtime plan builders for Build Tensor."""

from __future__ import annotations

from typing import Any

from .orchestration_execution import RuntimePlan
from .runners.connectivity_trgc import (
    TRGC_FINALIZE_PLAN_KEY,
    TRGC_GC_BACKEND_PLAN_KEY,
    TRGC_GC_TR_BACKEND_PLAN_KEY,
)


def _spectral_runner_kwargs(metric_params: dict[str, Any]) -> dict[str, Any]:
    method = str(metric_params["method"])
    kwargs: dict[str, Any] = {
        "time_resolution_s": float(metric_params["time_resolution_s"]),
        "hop_s": float(metric_params["hop_s"]),
        "method": method,
    }
    if method == "morlet":
        kwargs.update(
            {
                "min_cycles": float(metric_params["min_cycles"]),
                "max_cycles": metric_params["max_cycles"],
                "mt_time_bandwidth_product": 4.0,
                "mt_min_cycles": 3.0,
            }
        )
    else:
        kwargs.update(
            {
                "mt_time_bandwidth_product": float(
                    metric_params["mt_time_bandwidth_product"]
                ),
                "mt_min_cycles": float(metric_params["mt_min_cycles"]),
                "min_cycles": 3.0,
                "max_cycles": None,
            }
        )
    return kwargs


def plan_undirected(
    svc: Any,
    context: Any,
    *,
    metric_key: str,
    metric_low: float,
    metric_high: float,
    metric_step: float,
    metric_bands: list[dict[str, Any]],
    metric_channels: list[str] | None,
    metric_pairs: list[tuple[str, str]] | None,
    metric_params: dict[str, Any],
    mask_edge_effects: bool,
) -> RuntimePlan:
    _ = context
    connectivity_metric_map = {
        "coherence": "coh",
        "imcoh_abs": "imcoh_abs",
        "plv": "plv",
        "ciplv": "ciplv",
        "pli": "pli",
        "wpli": "wpli",
    }
    return RuntimePlan(
        plan_key=metric_key,
        metric_label=svc.TENSOR_METRICS_BY_KEY[metric_key].display_name,
        runner_key="undirected_connectivity",
        runner_kwargs={
            "metric_key": metric_key,
            "connectivity_metric": connectivity_metric_map[metric_key],
            "low_freq": float(metric_low),
            "high_freq": float(metric_high),
            "step_hz": float(metric_step),
            "mask_edge_effects": mask_edge_effects,
            "bands": metric_bands,
            "selected_channels": metric_channels,
            "selected_pairs": metric_pairs,
            **_spectral_runner_kwargs(metric_params),
            "notches": metric_params["notches"],
            "notch_radii": metric_params["notch_radii"],
        },
    )


def plan_trgc(
    svc: Any,
    context: Any,
    *,
    metric_low: float,
    metric_high: float,
    metric_step: float,
    metric_bands: list[dict[str, Any]],
    metric_channels: list[str] | None,
    metric_pairs: list[tuple[str, str]] | None,
    metric_params: dict[str, Any],
    mask_edge_effects: bool,
) -> dict[str, RuntimePlan]:
    _ = context
    metric_label = svc.TENSOR_METRICS_BY_KEY["trgc"].display_name

    def _backend_plan(
        plan_key: str, backend_method: str, label_suffix: str
    ) -> RuntimePlan:
        plan = RuntimePlan(
            plan_key=plan_key,
            metric_label=f"{metric_label} {label_suffix}",
            runner_key="trgc_backend",
            runner_kwargs={
                "backend_method": str(backend_method),
                "low_freq": float(metric_low),
                "high_freq": float(metric_high),
                "step_hz": float(metric_step),
                "mask_edge_effects": mask_edge_effects,
                "bands": metric_bands,
                "selected_channels": metric_channels,
                "selected_pairs": metric_pairs,
                **_spectral_runner_kwargs(metric_params),
                "gc_n_lags": int(metric_params["gc_n_lags"]),
                "group_by_samples": bool(metric_params["group_by_samples"]),
                "notches": metric_params["notches"],
                "notch_radii": metric_params["notch_radii"],
            },
            log_metric_key="trgc",
        )
        if not bool(metric_params["group_by_samples"]):
            plan.runner_kwargs["round_ms"] = float(metric_params["round_ms"])
        else:
            plan.runner_kwargs["round_ms"] = 50.0
        return plan

    return {
        TRGC_GC_BACKEND_PLAN_KEY: _backend_plan(
            TRGC_GC_BACKEND_PLAN_KEY,
            "gc",
            "GC Backend",
        ),
        TRGC_GC_TR_BACKEND_PLAN_KEY: _backend_plan(
            TRGC_GC_TR_BACKEND_PLAN_KEY,
            "gc_tr",
            "GC_TR Backend",
        ),
        TRGC_FINALIZE_PLAN_KEY: RuntimePlan(
            plan_key=TRGC_FINALIZE_PLAN_KEY,
            metric_label=metric_label,
            runner_key="trgc_finalize",
            runner_kwargs={"mask_edge_effects": mask_edge_effects},
            phase=1,
            dependencies=(TRGC_GC_BACKEND_PLAN_KEY, TRGC_GC_TR_BACKEND_PLAN_KEY),
        ),
    }


def plan_psi(
    svc: Any,
    context: Any,
    *,
    metric_low: float,
    metric_high: float,
    metric_step: float,
    metric_bands: list[dict[str, Any]],
    metric_channels: list[str] | None,
    metric_pairs: list[tuple[str, str]] | None,
    metric_params: dict[str, Any],
    mask_edge_effects: bool,
) -> RuntimePlan:
    _ = context
    return RuntimePlan(
        plan_key="psi",
        metric_label=svc.TENSOR_METRICS_BY_KEY["psi"].display_name,
        runner_key="psi",
        runner_kwargs={
            "low_freq": float(metric_low),
            "high_freq": float(metric_high),
            "step_hz": float(metric_step),
            "mask_edge_effects": mask_edge_effects,
            "bands": metric_bands,
            "selected_channels": metric_channels,
            "selected_pairs": metric_pairs,
            **_spectral_runner_kwargs(metric_params),
            "notches": metric_params["notches"],
            "notch_radii": metric_params["notch_radii"],
        },
    )


__all__ = ["plan_psi", "plan_trgc", "plan_undirected"]
