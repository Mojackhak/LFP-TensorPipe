"""Tensor orchestration entrypoint (Build Tensor stage)."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any
from uuid import uuid4

from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_tensor_result_change,
)

from .cpu_budget import DEFAULT_TENSOR_CPU_PERCENT, derive_global_compute_slots
from .cancellation import raise_if_tensor_cancellation_requested
from .orchestration_execution import (
    apply_effective_parallel_policy,
    execute_runtime_plans,
)
from .orchestration_merge import merge_metric_params_map
from .orchestration_plans import build_runtime_plans
from .logging import TENSOR_RUN_ID_ENV
from .lineage import (
    TENSOR_INPUT_LINEAGE_ENV,
    capture_tensor_input_generation,
    serialize_tensor_input_lineage,
)
from .paths import tensor_output_metric_keys


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
    cpu_percent: float = DEFAULT_TENSOR_CPU_PERCENT,
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
    raise_if_tensor_cancellation_requested()
    try:
        normalized_cpu_percent, detected_cpu_count, global_compute_slots = (
            derive_global_compute_slots(cpu_percent)
        )
    except ValueError as exc:
        return False, str(exc)

    provided_metric_params = (
        metric_params_map if isinstance(metric_params_map, dict) else {}
    )
    for metric_key in metrics:
        metric_params = provided_metric_params.get(metric_key)
        if isinstance(metric_params, dict) and "notch_widths" in metric_params:
            return (
                False,
                f"metric_params_map.{metric_key}.notch_widths was removed from "
                "Build Tensor schema 4. Use notch_radii instead.",
            )

    input_generations = capture_tensor_input_generation(resolver)
    if input_generations is None:
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
    prior_lineage_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV)
    prior_run_id = os.environ.get(TENSOR_RUN_ID_ENV)
    installed_run_id = not str(prior_run_id or "").strip()
    os.environ[TENSOR_INPUT_LINEAGE_ENV] = serialize_tensor_input_lineage(
        context,
        input_generations,
    )
    if installed_run_id:
        os.environ[TENSOR_RUN_ID_ENV] = uuid4().hex
    try:
        runtime_results = execute_runtime_plans(
            svc,
            resolver,
            context,
            runtime_plans=plan_state.runtime_plans,
            merged_metric_params_map=merged_metric_params_map,
            policy_n_jobs=policy_n_jobs,
            policy_outer_n_jobs=policy_outer_n_jobs,
            global_compute_slots=global_compute_slots,
            force_in_process=bool(service_overrides),
        )
    finally:
        if prior_lineage_payload is None:
            os.environ.pop(TENSOR_INPUT_LINEAGE_ENV, None)
        else:
            os.environ[TENSOR_INPUT_LINEAGE_ENV] = prior_lineage_payload
        if installed_run_id:
            if prior_run_id is None:
                os.environ.pop(TENSOR_RUN_ID_ENV, None)
            else:
                os.environ[TENSOR_RUN_ID_ENV] = prior_run_id
    raise_if_tensor_cancellation_requested()

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
    try:
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
                "cpu_percent": normalized_cpu_percent,
                "detected_cpu_count": detected_cpu_count,
                "global_compute_slots": global_compute_slots,
            },
            input_path=str(svc.preproc_step_raw_path(resolver, "finish")),
            output_path=str(resolver.tensor_root),
            message=result_message,
        )
    except Exception as exc:  # noqa: BLE001
        summary_warning = f"Build Tensor stage summary warning: {exc}"
        result_message = (
            f"{result_message}; {summary_warning}"
            if result_message
            else summary_warning
        )
    changed_metric_keys = tensor_output_metric_keys(
        [
            metric_key
            for metric_key, status in metric_statuses.items()
            if status == "success"
        ]
    )
    if changed_metric_keys:
        invalidate_after_tensor_result_change(
            context,
            metric_keys=changed_metric_keys,
        )
    return overall_ok, result_message
