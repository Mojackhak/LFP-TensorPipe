"""Burst-family runtime plan builder for Build Tensor."""

from __future__ import annotations

from typing import Any

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
    metric_step: float,
    metric_bands: list[dict[str, Any]],
    metric_channels: list[str] | None,
    metric_params: dict[str, Any],
    mask_edge_effects: bool,
) -> RuntimePlan:
    return RuntimePlan(
        plan_key="burst",
        metric_label=svc.TENSOR_METRICS_BY_KEY["burst"].display_name,
        runner_key="burst",
        runner_kwargs={
            "low_freq": float(metric_low),
            "high_freq": float(metric_high),
            "step_hz": float(metric_step),
            "mask_edge_effects": mask_edge_effects,
            "bands": metric_bands,
            "selected_channels": metric_channels,
            "percentile": svc._as_float(metric_params.get("percentile"), 75.0),
            "baseline_keep": _normalize_baseline_keep(
                metric_params.get("baseline_keep")
            ),
            "min_cycles": svc._as_float(metric_params.get("min_cycles"), 2.0),
            "max_cycles": svc._as_optional_float(metric_params.get("max_cycles")),
            "hop_s": svc._as_optional_float(metric_params.get("hop_s")),
            "decim": svc._as_optional_int(metric_params.get("decim")),
            "thresholds": metric_params.get("thresholds"),
            "notches": metric_params.get("notches"),
            "notch_widths": metric_params.get(
                "notch_widths",
                svc.DEFAULT_TENSOR_NOTCH_WIDTH,
            ),
            "thresholds_source_path": (
                str(metric_params.get("thresholds_path"))
                if metric_params.get("thresholds_path") is not None
                else None
            ),
        },
    )


__all__ = ["plan_burst"]
