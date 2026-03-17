"""Connectivity-family runtime plan builders for Build Tensor."""

from __future__ import annotations

from typing import Any

from .orchestration_execution import RuntimePlan
from .runners.connectivity_trgc import (
    TRGC_FINALIZE_PLAN_KEY,
    TRGC_GC_BACKEND_PLAN_KEY,
    TRGC_GC_TR_BACKEND_PLAN_KEY,
)


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
            "time_resolution_s": svc._as_float(
                metric_params.get("time_resolution_s"), 0.5
            ),
            "hop_s": svc._as_float(metric_params.get("hop_s"), 0.025),
            "method": str(metric_params.get("method", "morlet")),
            "mt_bandwidth": svc._as_optional_float(metric_params.get("mt_bandwidth")),
            "min_cycles": svc._as_optional_float(
                metric_params.get("min_cycles"), 3.0
            ),
            "max_cycles": svc._as_optional_float(metric_params.get("max_cycles")),
            "notches": metric_params.get("notches"),
            "notch_widths": metric_params.get(
                "notch_widths",
                svc.DEFAULT_TENSOR_NOTCH_WIDTH,
            ),
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

    def _backend_plan(plan_key: str, backend_method: str, label_suffix: str) -> RuntimePlan:
        return RuntimePlan(
            plan_key=plan_key,
            metric_label=f"{metric_label} {label_suffix}",
            runner_key="trgc_backend",
            runner_kwargs={
                "backend_method": str(backend_method),
                "low_freq": float(metric_low),
                "high_freq": float(metric_high),
                "step_hz": float(metric_step),
                "bands": metric_bands,
                "selected_channels": metric_channels,
                "selected_pairs": metric_pairs,
                "time_resolution_s": svc._as_float(
                    metric_params.get("time_resolution_s"), 0.5
                ),
                "hop_s": svc._as_float(metric_params.get("hop_s"), 0.025),
                "method": str(metric_params.get("method", "morlet")),
                "mt_bandwidth": svc._as_optional_float(
                    metric_params.get("mt_bandwidth")
                ),
                "min_cycles": svc._as_optional_float(
                    metric_params.get("min_cycles"), 3.0
                ),
                "max_cycles": svc._as_optional_float(
                    metric_params.get("max_cycles")
                ),
                "gc_n_lags": svc._as_int(metric_params.get("gc_n_lags"), 20),
                "group_by_samples": svc._as_bool(
                    metric_params.get("group_by_samples"), False
                ),
                "round_ms": svc._as_float(metric_params.get("round_ms"), 50.0),
                "notches": metric_params.get("notches"),
                "notch_widths": metric_params.get(
                    "notch_widths",
                    svc.DEFAULT_TENSOR_NOTCH_WIDTH,
                ),
            },
            log_metric_key="trgc",
        )

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
            "time_resolution_s": svc._as_float(
                metric_params.get("time_resolution_s"), 0.5
            ),
            "hop_s": svc._as_float(metric_params.get("hop_s"), 0.025),
            "method": str(metric_params.get("method", "morlet")),
            "mt_bandwidth": svc._as_optional_float(metric_params.get("mt_bandwidth")),
            "min_cycles": svc._as_optional_float(
                metric_params.get("min_cycles"), 3.0
            ),
            "max_cycles": svc._as_optional_float(metric_params.get("max_cycles")),
            "notches": metric_params.get("notches"),
            "notch_widths": metric_params.get(
                "notch_widths",
                svc.DEFAULT_TENSOR_NOTCH_WIDTH,
            ),
        },
    )


__all__ = ["plan_psi", "plan_trgc", "plan_undirected"]
