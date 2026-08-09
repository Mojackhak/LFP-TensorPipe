"""Child-process entrypoint for a single Build Tensor runtime plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from lfptensorpipe.app.path_resolver import RecordContext

from .logging import TENSOR_BENCHMARK_TRACE_PATH_ENV, TENSOR_RUN_ID_ENV
from .orchestration_execution import RuntimePlan, run_runtime_plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lfptensorpipe-runtime-plan-worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Worker payload must be a JSON object.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_benchmark_trace(payload: dict[str, Any]) -> None:
    """Append one optional benchmark event without affecting runtime results."""
    trace_path = os.environ.get(TENSOR_BENCHMARK_TRACE_PATH_ENV, "").strip()
    if not trace_path:
        return
    line = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        fd = os.open(trace_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    except OSError:
        return
    try:
        try:
            os.write(fd, line)
        except OSError:
            return
    finally:
        os.close(fd)


def _record_context(payload: dict[str, Any]) -> RecordContext:
    context = payload.get("context")
    if not isinstance(context, dict):
        raise ValueError("Worker payload is missing `context`.")
    project_root = Path(str(context.get("project_root", ""))).expanduser().resolve()
    subject = str(context.get("subject", "")).strip()
    record = str(context.get("record", "")).strip()
    if not subject or not record:
        raise ValueError("Worker payload requires non-empty `subject` and `record`.")
    return RecordContext(project_root=project_root, subject=subject, record=record)


def _merged_metric_params_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_map = payload.get("merged_metric_params_map", {})
    if not isinstance(raw_map, dict):
        raise ValueError(
            "Worker payload `merged_metric_params_map` must be a JSON object."
        )
    metric_params_map: dict[str, dict[str, Any]] = {}
    for metric_key, params in raw_map.items():
        if not isinstance(params, dict):
            raise ValueError(
                f"Worker payload metric params must be objects: {metric_key!r}"
            )
        metric_params_map[str(metric_key)] = dict(params)
    return metric_params_map


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request_path = Path(args.request).expanduser().resolve()
    result_path = Path(args.result).expanduser().resolve()
    runtime_plan: RuntimePlan | None = None
    start_perf_ns: int | None = None
    start_cpu_ns: int | None = None
    try:
        payload = _read_json(request_path)
        run_id = str(payload.get("run_id", "")).strip()
        if run_id:
            os.environ[TENSOR_RUN_ID_ENV] = run_id

        context = _record_context(payload)
        runtime_plan = RuntimePlan.from_payload(payload.get("runtime_plan", {}))
        merged_metric_params_map = _merged_metric_params_map(payload)
        policy_n_jobs = int(payload.get("policy_n_jobs", 1))
        policy_outer_n_jobs = int(payload.get("policy_outer_n_jobs", 1))

        from . import service as svc

        resolver = svc.PathResolver(context)
        resolver.ensure_record_roots(include_tensor=False)
        start_perf_ns = time.perf_counter_ns()
        start_cpu_ns = time.process_time_ns()
        ok, message, metric_label = run_runtime_plan(
            svc,
            resolver,
            context,
            runtime_plan=runtime_plan,
            merged_metric_params_map=merged_metric_params_map,
            policy_n_jobs=policy_n_jobs,
            policy_outer_n_jobs=policy_outer_n_jobs,
        )
    except Exception as exc:  # noqa: BLE001
        if runtime_plan is not None and start_perf_ns is not None:
            end_perf_ns = time.perf_counter_ns()
            end_cpu_ns = time.process_time_ns()
            _append_benchmark_trace(
                {
                    "run_id": os.environ.get(TENSOR_RUN_ID_ENV, "").strip(),
                    "plan_key": runtime_plan.plan_key,
                    "log_metric_key": (
                        runtime_plan.log_metric_key or runtime_plan.plan_key
                    ),
                    "pid": os.getpid(),
                    "ok": False,
                    "start_perf_ns": start_perf_ns,
                    "end_perf_ns": end_perf_ns,
                    "elapsed_s": (end_perf_ns - start_perf_ns) / 1e9,
                    "worker_cpu_s": (end_cpu_ns - int(start_cpu_ns or 0)) / 1e9,
                    "exception_type": type(exc).__name__,
                }
            )
        _write_json(
            result_path,
            {
                "ok": False,
                "message": str(exc),
                "exception_type": type(exc).__name__,
            },
        )
        return 2

    end_perf_ns = time.perf_counter_ns()
    end_cpu_ns = time.process_time_ns()
    _append_benchmark_trace(
        {
            "run_id": os.environ.get(TENSOR_RUN_ID_ENV, "").strip(),
            "plan_key": runtime_plan.plan_key,
            "log_metric_key": runtime_plan.log_metric_key or runtime_plan.plan_key,
            "pid": os.getpid(),
            "ok": bool(ok),
            "start_perf_ns": int(start_perf_ns),
            "end_perf_ns": end_perf_ns,
            "elapsed_s": (end_perf_ns - int(start_perf_ns)) / 1e9,
            "worker_cpu_s": (end_cpu_ns - int(start_cpu_ns)) / 1e9,
        }
    )

    _write_json(
        result_path,
        {
            "ok": bool(ok),
            "message": str(message),
            "metric_label": metric_label,
        },
    )
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(main())


__all__ = ["TENSOR_BENCHMARK_TRACE_PATH_ENV", "main", "parse_args"]
