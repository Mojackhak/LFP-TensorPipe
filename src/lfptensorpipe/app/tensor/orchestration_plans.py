"""Runtime-plan building facade for Build Tensor orchestration."""

from __future__ import annotations

from typing import Any

from .orchestration_execution import RuntimePlan
from .orchestration_plan_dispatch import build_runtime_plan
from .orchestration_plan_models import TensorPlanBuildResult
from .orchestration_plan_validation import prepare_metric_plan_inputs


def write_metric_invalid(
    svc: Any,
    resolver: Any,
    *,
    metric_key: str,
    metric_label: str,
    params: dict[str, Any],
    reason: str,
) -> None:
    svc._write_metric_log(
        resolver,
        metric_key,
        completed=False,
        params=svc._sanitize_metric_params_for_logs(params),
        input_path=str(svc.preproc_step_raw_path(resolver, "finish")),
        output_path=str(svc.tensor_metric_tensor_path(resolver, metric_key)),
        message=f"{metric_label} failed: {reason}",
    )


def build_runtime_plans(
    svc: Any,
    context: Any,
    resolver: Any,
    *,
    metrics: list[str],
    merged_metric_params_map: dict[str, dict[str, Any]],
    mask_edge_effects: bool,
) -> TensorPlanBuildResult:
    overall_ok = True
    messages: list[str] = []
    metric_statuses: dict[str, str] = {}
    effective_n_jobs_map: dict[str, dict[str, int]] = {}
    runtime_plans: dict[str, RuntimePlan] = {}
    finish_path = svc.preproc_step_raw_path(resolver, "finish")

    for metric_key in metrics:
        spec = svc.TENSOR_METRICS_BY_KEY.get(metric_key)
        metric_output_path = svc.tensor_metric_tensor_path(resolver, metric_key)
        effective_n_jobs_map.setdefault(metric_key, {"n_jobs": 1, "outer_n_jobs": 1})

        if spec is None:
            svc._write_metric_log(
                resolver,
                metric_key,
                completed=False,
                params={},
                input_path=str(finish_path),
                output_path=str(metric_output_path),
                message="Unknown metric key.",
            )
            overall_ok = False
            messages.append(f"{metric_key}: unknown metric")
            metric_statuses[metric_key] = "failed_unknown_metric"
            continue

        if not spec.supported:
            svc._write_metric_log(
                resolver,
                metric_key,
                completed=False,
                params={},
                input_path=str(finish_path),
                output_path=str(metric_output_path),
                message="Metric not implemented in current slice.",
            )
            overall_ok = False
            messages.append(f"{spec.display_name}: not implemented")
            metric_statuses[metric_key] = "failed_not_implemented"
            continue

        metric_params = dict(merged_metric_params_map.get(metric_key, {}))
        try:
            prepared = prepare_metric_plan_inputs(
                svc,
                context,
                metric_key=metric_key,
                metric_label=spec.display_name,
                metric_params=metric_params,
            )
        except Exception as exc:  # noqa: BLE001
            write_metric_invalid(
                svc,
                resolver,
                metric_key=metric_key,
                metric_label=spec.display_name,
                params=metric_params,
                reason=str(exc),
            )
            overall_ok = False
            messages.append(f"{spec.display_name}: {exc}")
            metric_statuses[metric_key] = "failed_invalid_params"
            continue

        try:
            runtime_plans.update(
                build_runtime_plan(
                    svc,
                    context,
                    prepared=prepared,
                    mask_edge_effects=mask_edge_effects,
                )
            )
        except KeyError:
            svc._write_metric_log(
                resolver,
                metric_key,
                completed=False,
                params={},
                input_path=str(finish_path),
                output_path=str(metric_output_path),
                message="Unsupported metric handler.",
            )
            overall_ok = False
            messages.append(f"{spec.display_name}: unsupported handler")
            metric_statuses[metric_key] = "failed_unsupported_handler"

    return TensorPlanBuildResult(
        overall_ok=overall_ok,
        messages=messages,
        metric_statuses=metric_statuses,
        effective_n_jobs_map=effective_n_jobs_map,
        runtime_plans=runtime_plans,
    )


__all__ = [
    "TensorPlanBuildResult",
    "build_runtime_plans",
    "write_metric_invalid",
]
