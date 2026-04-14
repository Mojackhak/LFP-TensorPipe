"""Tensor orchestration entrypoint (Build Tensor stage)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_tensor_result_change,
)

from .orchestration_execution import (
    apply_effective_parallel_policy,
    execute_runtime_plans,
)
from .orchestration_merge import merge_metric_params_map
from .orchestration_plans import build_runtime_plans


def _service_with_overrides(service_overrides: Mapping[str, Any] | None) -> Any:
    from . import service as svc

    if not service_overrides:
        return svc

    class _ServiceFacade:
        def __getattr__(self, name: str) -> Any:
            if name in service_overrides:
                return service_overrides[name]
            return getattr(svc, name)

    return _ServiceFacade()


def run_build_tensor(
    context: RecordContext,
    *,
    selected_metrics: list[str],
    metric_params_map: dict[str, dict[str, Any]] | None = None,
    low_freq: float | None = None,
    high_freq: float | None = None,
    step_hz: float | None = None,
    mask_edge_effects: bool = True,
    bands: list[dict[str, Any]] | None = None,
    selected_channels: list[str] | None = None,
    selected_pairs: dict[str, list[tuple[str, str]]] | None = None,
    service_overrides: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Run Build Tensor for selected metrics.

    Preferred input contract is ``metric_params_map`` (per-metric full params).
    Legacy global arguments are accepted for backward compatibility in tests.
    """
    svc = _service_with_overrides(service_overrides)

    resolver = svc.PathResolver(context)
    resolver.ensure_record_roots(include_tensor=False)

    metrics = [item for item in selected_metrics if str(item).strip()]
    if not metrics:
        return False, "No tensor metric selected."

    if svc.indicator_from_log(svc.preproc_step_log_path(resolver, "finish")) != "green":
        return False, "Preprocess finish must be green before Build Tensor."

    merged_metric_params_map = merge_metric_params_map(
        svc,
        context,
        metrics=metrics,
        metric_params_map=metric_params_map,
        low_freq=low_freq,
        high_freq=high_freq,
        step_hz=step_hz,
        bands=bands,
        selected_channels=selected_channels,
        selected_pairs=selected_pairs,
    )
    plan_state = build_runtime_plans(
        svc,
        context,
        resolver,
        metrics=metrics,
        merged_metric_params_map=merged_metric_params_map,
        mask_edge_effects=mask_edge_effects,
    )

    policy_n_jobs, policy_outer_n_jobs = apply_effective_parallel_policy(
        plan_state.runtime_plans,
        plan_state.effective_n_jobs_map,
    )
    runtime_results = execute_runtime_plans(
        svc,
        resolver,
        context,
        runtime_plans=plan_state.runtime_plans,
        merged_metric_params_map=merged_metric_params_map,
        policy_n_jobs=policy_n_jobs,
        policy_outer_n_jobs=policy_outer_n_jobs,
        force_in_process=bool(service_overrides),
    )

    overall_ok = plan_state.overall_ok
    messages = list(plan_state.messages)
    metric_statuses = dict(plan_state.metric_statuses)
    for metric_key in metrics:
        result = runtime_results.get(metric_key)
        if result is None:
            continue
        ok, message, metric_label = result
        overall_ok = overall_ok and ok
        messages.append(f"{metric_label}: {message}")
        metric_statuses[metric_key] = "success" if ok else "failed_runtime"

    result_message = "; ".join(messages)
    svc._write_stage_log(
        resolver,
        completed=overall_ok,
        params={
            "selected_metrics": metrics,
            "mask_edge_effects": bool(mask_edge_effects),
            "metric_params_map": {
                key: svc._sanitize_metric_params_for_logs(value)
                for key, value in merged_metric_params_map.items()
            },
            "metric_statuses": metric_statuses,
            "effective_n_jobs": plan_state.effective_n_jobs_map,
        },
        input_path=str(svc.preproc_step_raw_path(resolver, "finish")),
        output_path=str(resolver.tensor_root),
        message=result_message,
    )
    if any(status == "success" for status in metric_statuses.values()):
        invalidate_after_tensor_result_change(context, metric_keys=metrics)
    return overall_ok, result_message
