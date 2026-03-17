"""Cancellation helpers for Build Tensor child-process runs."""

from __future__ import annotations

from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import preproc_step_raw_path
from lfptensorpipe.app.runlog_store import read_run_log

from .coercion import _sanitize_metric_params_for_logs
from .logging import write_metric_log, write_stage_log
from .paths import tensor_metric_log_path, tensor_metric_tensor_path

CANCELLED_RUN_STATUS = "cancelled"
BUILD_TENSOR_CANCELLED_MESSAGE = "Build Tensor cancelled by user."


def _log_run_id(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    params = payload.get("params")
    if not isinstance(params, dict):
        return ""
    return str(params.get("run_id", "")).strip()


def _finished_status(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    params = payload.get("params")
    run_status = ""
    if isinstance(params, dict):
        run_status = str(params.get("run_status", "")).strip()
    if run_status:
        return run_status
    return "success" if bool(payload.get("completed")) else "failed_runtime"


def backfill_cancelled_build_tensor_run(
    context: RecordContext,
    *,
    selected_metrics: list[str],
    metric_params_map: dict[str, dict[str, Any]],
    mask_edge_effects: bool,
    run_id: str,
    message: str = BUILD_TENSOR_CANCELLED_MESSAGE,
) -> dict[str, str]:
    """Mark unfinished selected metrics as cancelled after child-process exit."""

    resolver = PathResolver(context)
    resolver.ensure_record_roots(include_tensor=True)
    metric_statuses: dict[str, str] = {}
    sanitized_metric_params_map = {
        key: _sanitize_metric_params_for_logs(value)
        for key, value in metric_params_map.items()
    }
    cancelled_metrics: list[str] = []

    for metric_key in selected_metrics:
        payload = read_run_log(tensor_metric_log_path(resolver, metric_key))
        if _log_run_id(payload) == run_id:
            metric_statuses[metric_key] = _finished_status(payload)
            continue
        params = dict(sanitized_metric_params_map.get(metric_key, {}))
        params["run_status"] = CANCELLED_RUN_STATUS
        params["cancelled"] = True
        write_metric_log(
            resolver,
            metric_key,
            completed=False,
            params=params,
            input_path=str(preproc_step_raw_path(resolver, "finish")),
            output_path=str(tensor_metric_tensor_path(resolver, metric_key)),
            message=message,
        )
        metric_statuses[metric_key] = CANCELLED_RUN_STATUS
        cancelled_metrics.append(metric_key)

    if cancelled_metrics:
        write_stage_log(
            resolver,
            completed=False,
            params={
                "selected_metrics": list(selected_metrics),
                "mask_edge_effects": bool(mask_edge_effects),
                "metric_params_map": sanitized_metric_params_map,
                "metric_statuses": metric_statuses,
                "run_status": CANCELLED_RUN_STATUS,
                "cancelled": True,
            },
            input_path=str(preproc_step_raw_path(resolver, "finish")),
            output_path=str(resolver.tensor_root),
            message=message,
        )
    return metric_statuses


__all__ = [
    "BUILD_TENSOR_CANCELLED_MESSAGE",
    "CANCELLED_RUN_STATUS",
    "backfill_cancelled_build_tensor_run",
]
