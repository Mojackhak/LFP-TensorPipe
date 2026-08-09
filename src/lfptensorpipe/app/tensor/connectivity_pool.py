"""Shared process-pool primitives for Build Tensor connectivity tasks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Iterable
import warnings

import numpy as np

from .cancellation import raise_if_tensor_cancellation_requested

_WORKER_RAW: Any | None = None
_THREADPOOL_LIMITER: Any | None = None
_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class ConnectivityChunk:
    """One deterministic frequency-group and contiguous-time task."""

    family: str
    owner_id: str
    group_id: int
    time_indices: tuple[int, ...]
    weight: float

    @property
    def start_index(self) -> int:
        return int(self.time_indices[0])


def initialize_connectivity_worker(raw_path: str) -> None:
    """Open the record Raw once for the lifetime of a connectivity worker."""
    global _THREADPOOL_LIMITER, _WORKER_RAW

    for name in _NATIVE_THREAD_ENV_VARS:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITER = threadpool_limits(limits=1)
        _THREADPOOL_LIMITER.__enter__()
    except ImportError:
        _THREADPOOL_LIMITER = None

    import mne

    _WORKER_RAW = mne.io.read_raw_fif(
        str(Path(raw_path)),
        preload=False,
        verbose="ERROR",
    )


def run_connectivity_pool_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one indexed connectivity block using the worker-local Raw."""
    raise_if_tensor_cancellation_requested()
    if _WORKER_RAW is None:
        raise RuntimeError("Connectivity worker Raw was not initialized.")

    from lfptensorpipe.lfp.connectivity.grid import grid

    start_perf_ns = time.perf_counter_ns()
    start_cpu_ns = time.process_time_ns()
    grid_kwargs = dict(payload["grid_kwargs"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|invalid value) encountered in det",
            category=RuntimeWarning,
            module=r"numpy\.linalg\._linalg",
        )
        blocks, _ = grid(
            _WORKER_RAW,
            **grid_kwargs,
            method=payload["methods"],
            task_group_id=int(payload["group_id"]),
            task_time_indices=list(payload["time_indices"]),
            return_task_blocks=True,
            outer_n_jobs=1,
        )
    if len(blocks) != 1:
        raise RuntimeError(
            "Connectivity task must return exactly one non-empty indexed block."
        )
    end_perf_ns = time.perf_counter_ns()
    end_cpu_ns = time.process_time_ns()
    block = dict(blocks[0])
    block.update(
        {
            "task_id": str(payload["task_id"]),
            "family": str(payload["family"]),
            "owner_id": str(payload["owner_id"]),
            "start_perf_ns": int(start_perf_ns),
            "end_perf_ns": int(end_perf_ns),
            "elapsed_s": (end_perf_ns - start_perf_ns) / 1e9,
            "worker_cpu_s": (end_cpu_ns - start_cpu_ns) / 1e9,
            "pid": os.getpid(),
        }
    )
    raise_if_tensor_cancellation_requested()
    return block


def _contiguous_runs(indices: Iterable[int]) -> list[np.ndarray]:
    values = np.asarray(list(indices), dtype=int)
    if values.size == 0:
        return []
    values = np.unique(values)
    split_at = np.flatnonzero(np.diff(values) != 1) + 1
    return [item for item in np.split(values, split_at) if item.size]


def _allocate_chunk_counts(
    units: list[dict[str, Any]],
    *,
    target_count: int,
) -> list[int]:
    counts = [1 for _ in units]
    remaining = min(
        max(int(target_count) - len(units), 0),
        sum(int(unit["time_indices"].size) - 1 for unit in units),
    )
    while remaining > 0:
        eligible = [
            index
            for index, unit in enumerate(units)
            if counts[index] < int(unit["time_indices"].size)
        ]
        if not eligible:
            break
        total_weight = sum(float(units[index]["weight"]) for index in eligible)
        if total_weight <= 0:
            total_weight = float(len(eligible))
        quotas = {
            index: remaining
            * (
                float(units[index]["weight"])
                if float(units[index]["weight"]) > 0
                else 1.0
            )
            / total_weight
            for index in eligible
        }
        added = 0
        for index in eligible:
            capacity = int(units[index]["time_indices"].size) - counts[index]
            increment = min(int(np.floor(quotas[index])), capacity, remaining - added)
            if increment > 0:
                counts[index] += increment
                added += increment
        remaining -= added
        if remaining <= 0:
            break
        ranked = sorted(
            eligible,
            key=lambda index: (
                -(quotas[index] - np.floor(quotas[index])),
                str(units[index]["family"]),
                str(units[index]["owner_id"]),
                int(units[index]["group_id"]),
                int(units[index]["time_indices"][0]),
            ),
        )
        progressed = False
        for index in ranked:
            if remaining <= 0:
                break
            if counts[index] >= int(units[index]["time_indices"].size):
                continue
            counts[index] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return counts


def build_connectivity_chunks(
    group_descriptors: list[dict[str, Any]],
    *,
    global_compute_slots: int,
) -> list[ConnectivityChunk]:
    """Allocate deterministic weighted chunks without crossing support gaps."""
    units: list[dict[str, Any]] = []
    for descriptor in group_descriptors:
        for run in _contiguous_runs(descriptor["valid_time_indices"]):
            units.append(
                {
                    "family": str(descriptor["family"]),
                    "owner_id": str(descriptor["owner_id"]),
                    "group_id": int(descriptor["group_id"]),
                    "time_indices": run,
                    "weight": float(descriptor["weight_per_time"]) * int(run.size),
                }
            )
    if not units:
        return []

    target_count = max(len(units), 4 * max(int(global_compute_slots), 1))
    chunk_counts = _allocate_chunk_counts(units, target_count=target_count)
    chunks: list[ConnectivityChunk] = []
    for unit, chunk_count in zip(units, chunk_counts):
        for time_chunk in np.array_split(unit["time_indices"], int(chunk_count)):
            if not time_chunk.size:
                continue
            chunk_weight = float(unit["weight"]) * (
                int(time_chunk.size) / int(unit["time_indices"].size)
            )
            chunks.append(
                ConnectivityChunk(
                    family=str(unit["family"]),
                    owner_id=str(unit["owner_id"]),
                    group_id=int(unit["group_id"]),
                    time_indices=tuple(int(item) for item in time_chunk.tolist()),
                    weight=float(chunk_weight),
                )
            )
    return sorted(
        chunks,
        key=lambda item: (
            -float(item.weight),
            item.family,
            item.owner_id,
            int(item.group_id),
            int(item.start_index),
        ),
    )


__all__ = [
    "ConnectivityChunk",
    "build_connectivity_chunks",
    "initialize_connectivity_worker",
    "run_connectivity_pool_task",
]
