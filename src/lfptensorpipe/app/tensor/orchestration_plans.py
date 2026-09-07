"""Runtime-plan building facade for Build Tensor orchestration."""

from __future__ import annotations

from typing import Any

from .orchestration_execution import RuntimePlan
from .orchestration_plan_dispatch import build_runtime_plan
from .orchestration_plan_models import TensorPlanBuildResult
from .orchestration_plan_validation import prepare_metric_plan_inputs
from .selectors import TensorChannelInventory


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
    channel_inventory: TensorChannelInventory,
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
        exclusion_message = ""
        try:
            if metric_key in svc.TENSOR_CHANNEL_SELECTOR_KEYS:
                effective_channels, excluded_channels = svc._select_usable_channels(
                    metric_params.get("selected_channels"),
                    inventory=channel_inventory,
                )
                metric_params["selected_channels"] = list(effective_channels)
                if excluded_channels:
                    names = ", ".join(excluded_channels)
                    exclusion_message = (
                        f"{spec.display_name}: excluded Finish bad channel(s): {names}"
                    )
                    if not effective_channels:
                        raise ValueError(
                            f"{spec.display_name} has no usable selected channels "
                            "after excluding raw.info['bads']: " + names
                        )
            elif metric_key in (
                svc.TENSOR_UNDIRECTED_SELECTOR_KEYS | svc.TENSOR_DIRECTED_SELECTOR_KEYS
            ):
                effective_pairs, excluded_pairs = svc._select_usable_pairs(
                    metric_params.get("selected_pairs"),
                    inventory=channel_inventory,
                )
                if effective_pairs is not None:
                    metric_params["selected_pairs"] = [
                        list(pair) for pair in effective_pairs
                    ]
                if excluded_pairs:
                    involved_bad_channels = [
                        name
                        for name in channel_inventory.bad_channels
                        if any(name in pair for pair in excluded_pairs)
                    ]
                    names = ", ".join(involved_bad_channels)
                    exclusion_message = (
                        f"{spec.display_name}: excluded {len(excluded_pairs)} "
                        f"pair(s) containing Finish bad channel(s): {names}"
                    )
                    if not effective_pairs:
                        raise ValueError(
                            f"{spec.display_name} has no usable selected pairs "
                            "after excluding raw.info['bads']: " + names
                        )
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

        merged_metric_params_map[metric_key] = dict(prepared.metric_params)
        if exclusion_message:
            messages.append(exclusion_message)

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
