"""Tensor run-log and config persistence helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import RunLogRecord, write_run_log

from .paths import tensor_metric_log_path, tensor_stage_log_path

TENSOR_RUN_ID_ENV = "LFPTENSORPIPE_TENSOR_RUN_ID"


def _log_params_with_runtime_metadata(params: dict[str, Any]) -> dict[str, Any]:
    payload = dict(params)
    run_id = os.environ.get(TENSOR_RUN_ID_ENV, "").strip()
    if run_id:
        payload["run_id"] = run_id
    return payload


def build_metric_log_record(
    metric_key: str,
    *,
    completed: bool,
    params: dict[str, Any],
    input_path: str,
    output_path: str,
    message: str,
) -> RunLogRecord:
    return RunLogRecord(
        step=metric_key,
        completed=completed,
        params=_log_params_with_runtime_metadata(params),
        input_path=input_path,
        output_path=output_path,
        message=message,
    )


def build_stage_log_record(
    *,
    completed: bool,
    params: dict[str, Any],
    input_path: str,
    output_path: str,
    message: str,
) -> RunLogRecord:
    return RunLogRecord(
        step="build_tensor",
        completed=completed,
        params=_log_params_with_runtime_metadata(params),
        input_path=input_path,
        output_path=output_path,
        message=message,
    )


def write_metric_log_to_path(
    path: Path,
    metric_key: str,
    *,
    completed: bool,
    params: dict[str, Any],
    input_path: str,
    output_path: str,
    message: str,
) -> None:
    write_run_log(
        path,
        build_metric_log_record(
            metric_key,
            completed=completed,
            params=params,
            input_path=input_path,
            output_path=output_path,
            message=message,
        ),
    )


def write_stage_log_to_path(
    path: Path,
    *,
    completed: bool,
    params: dict[str, Any],
    input_path: str,
    output_path: str,
    message: str,
) -> None:
    write_run_log(
        path,
        build_stage_log_record(
            completed=completed,
            params=params,
            input_path=input_path,
            output_path=output_path,
            message=message,
        ),
    )


def write_metric_log(
    resolver: PathResolver,
    metric_key: str,
    *,
    completed: bool,
    params: dict[str, Any],
    input_path: str,
    output_path: str,
    message: str,
) -> None:
    write_metric_log_to_path(
        tensor_metric_log_path(resolver, metric_key, create=True),
        metric_key,
        completed=completed,
        params=params,
        input_path=input_path,
        output_path=output_path,
        message=message,
    )


def write_stage_log(
    resolver: PathResolver,
    *,
    completed: bool,
    params: dict[str, Any],
    input_path: str,
    output_path: str,
    message: str,
) -> None:
    write_stage_log_to_path(
        tensor_stage_log_path(resolver, create=True),
        completed=completed,
        params=params,
        input_path=input_path,
        output_path=output_path,
        message=message,
    )


def write_metric_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


__all__ = [
    "TENSOR_RUN_ID_ENV",
    "build_metric_log_record",
    "build_stage_log_record",
    "write_metric_log",
    "write_metric_log_to_path",
    "write_stage_log",
    "write_stage_log_to_path",
    "write_metric_config",
]
