"""Alignment tensor inventory and payload coercion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import indicator_from_log
from lfptensorpipe.app.tensor_service import (
    tensor_metric_log_path,
    tensor_metric_tensor_path,
)


def _completed_tensor_metrics(resolver: PathResolver) -> list[str]:
    metrics: list[str] = []
    if not resolver.tensor_root.exists():
        return metrics
    seen: set[str] = set()
    for metric_dir in sorted(
        path for path in resolver.tensor_root.iterdir() if path.is_dir()
    ):
        metric_key = metric_dir.name
        if metric_key in {"periodic_aperiodic", "aperiodic"}:
            continue
        state = indicator_from_log(tensor_metric_log_path(resolver, metric_key))
        if state != "green":
            continue
        if not tensor_metric_tensor_path(resolver, metric_key).exists():
            continue
        if metric_key in seen:
            continue
        seen.add(metric_key)
        metrics.append(metric_key)
    if (
        "periodic" in seen
        and "aperiodic" not in seen
        and tensor_metric_tensor_path(resolver, "aperiodic").exists()
    ):
        metrics.insert(metrics.index("periodic") + 1, "aperiodic")
    return metrics


def _coerce_alignment_tensor(
    payload: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    tensor_raw = payload.get("tensor")
    meta_raw = payload.get("meta")
    if not isinstance(meta_raw, dict):
        raise ValueError("Alignment input tensor meta is missing.")
    tensor = np.asarray(tensor_raw, dtype=float)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(
            f"Alignment currently expects (channel,freq,time) or (1,channel,freq,time), got {tensor.shape}."
        )
    return tensor, meta_raw


__all__ = ["_coerce_alignment_tensor", "_completed_tensor_metrics"]
