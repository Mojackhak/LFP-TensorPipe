"""Burst-family runtime plan builder for Build Tensor."""

from __future__ import annotations

from typing import Any

from lfptensorpipe.lfp.burst.semantics import (
    BURST_NATIVE_DECIM,
    BURST_NATIVE_HOP_S,
)

from .orchestration_execution import RuntimePlan


def _normalize_baseline_keep(value: Any) -> list[str] | None:
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else [value]
    labels: list[str] = []
    seen: set[str] = set()
    for item in items:
        label = str(item).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels or None


def plan_burst(
    svc: Any,
    context: Any,
    *,
    metric_low: float,
    metric_high: float,
    metric_bands: list[dict[str, Any]],
    metric_channels: list[str] | None,
    metric_params: dict[str, Any],
    mask_edge_effects: bool,
) -> RuntimePlan:
    method = str(metric_params.get("method", "hilbert"))
    runner_kwargs = {
        "low_freq": float(metric_low),
        "high_freq": float(metric_high),
        "mask_edge_effects": mask_edge_effects,
        "bands": metric_bands,
        "selected_channels": metric_channels,
        "method": method,
        "boundary_isolated_filter": bool(
            metric_params.get("boundary_isolated_filter", True)
        ),
        "min_cycles": float(metric_params["min_cycles"]),
        "max_cycles": metric_params["max_cycles"],
        "hop_s": BURST_NATIVE_HOP_S,
        "decim": BURST_NATIVE_DECIM,
        "thresholds": metric_params.get("thresholds"),
        "notches": metric_params["notches"],
        "notch_radii": metric_params["notch_radii"],
        "thresholds_source_path": (
            str(metric_params.get("thresholds_source_path"))
            if metric_params.get("thresholds_source_path") is not None
            else None
        ),
    }
    if method == "hilbert":
        runner_kwargs["hilbert_filter_method"] = str(
            metric_params.get("hilbert_filter_method", "iir")
        )
        runner_kwargs["hilbert_edge_tolerance_pct"] = float(
            metric_params.get("hilbert_edge_tolerance_pct", 10.0)
        )
    elif method == "morlet":
        runner_kwargs["step_hz"] = float(metric_params.get("freq_step_hz", 1.0))
        runner_kwargs["morlet_n_cycles"] = float(
            metric_params.get("morlet_n_cycles", 6.0)
        )
    else:
        runner_kwargs["step_hz"] = float(metric_params.get("freq_step_hz", 1.0))
        runner_kwargs["mt_n_cycles"] = float(metric_params.get("mt_n_cycles", 7.0))
        runner_kwargs["mt_time_bandwidth_product"] = float(
            metric_params.get("mt_time_bandwidth_product", 4.0)
        )
    if metric_params.get("thresholds") is None:
        runner_kwargs["percentile"] = float(metric_params["percentile"])
        runner_kwargs["baseline_keep"] = _normalize_baseline_keep(
            metric_params.get("baseline_keep")
        )
    return RuntimePlan(
        plan_key="burst",
        metric_label=svc.TENSOR_METRICS_BY_KEY["burst"].display_name,
        runner_key="burst",
        runner_kwargs=runner_kwargs,
    )


__all__ = ["plan_burst"]
