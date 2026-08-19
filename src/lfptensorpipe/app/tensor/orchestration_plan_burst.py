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
    runner_kwargs = {
        "low_freq": float(metric_low),
        "high_freq": float(metric_high),
        "mask_edge_effects": mask_edge_effects,
        "bands": metric_bands,
        "selected_channels": metric_channels,
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
