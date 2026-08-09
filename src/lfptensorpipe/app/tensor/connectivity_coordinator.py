"""In-process coordinator for shared Build Tensor connectivity computation."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
import os
import time
from typing import Any, Callable, Mapping

import numpy as np

from lfptensorpipe.app.path_resolver import RecordContext

from .cancellation import (
    BuildTensorCancellationRequested,
    raise_if_tensor_cancellation_requested,
)
from .connectivity_pool import (
    ConnectivityChunk,
    build_connectivity_chunks,
    initialize_connectivity_worker,
    run_connectivity_pool_task,
)
from .logging import TENSOR_BENCHMARK_TRACE_PATH_ENV, TENSOR_RUN_ID_ENV

RuntimeResult = tuple[bool, str, str]
UNDIRECTED_RUNNER_KEY = "undirected_connectivity"
TRGC_RUNNER_KEYS = frozenset({"trgc_backend", "trgc_finalize"})
CONNECTIVITY_RUNNER_KEYS = frozenset({UNDIRECTED_RUNNER_KEY, *TRGC_RUNNER_KEYS})

_UNDIRECTED_ESTIMATOR_BY_METRIC = {
    "coherence": "cohy",
    "imcoh_abs": "cohy",
    "plv": "plv",
    "ciplv": "ciplv",
    "pli": "pli",
    "wpli": "wpli",
}
_SHARED_UNDIRECTED_KEYS = (
    "low_freq",
    "high_freq",
    "step_hz",
    "mask_edge_effects",
    "selected_channels",
    "selected_pairs",
    "time_resolution_s",
    "hop_s",
    "method",
    "mt_time_bandwidth_product",
    "mt_min_cycles",
    "min_cycles",
    "max_cycles",
    "notches",
    "notch_radii",
)


def _freeze_signature_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_signature_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, np.ndarray):
        return _freeze_signature_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_signature_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def shared_compute_signature(runtime_plan: Any) -> tuple[Any, ...]:
    """Return the exact normalized inputs eligible for numerical sharing."""
    kwargs = runtime_plan.runner_kwargs
    return tuple(
        (key, _freeze_signature_value(kwargs.get(key)))
        for key in _SHARED_UNDIRECTED_KEYS
    )


def metric_output_signature(runtime_plan: Any) -> tuple[Any, ...]:
    """Return the logical metric and metric-local output contract."""
    kwargs = runtime_plan.runner_kwargs
    return (
        str(runtime_plan.plan_key),
        str(kwargs.get("connectivity_metric", "")),
        _freeze_signature_value(kwargs.get("bands", [])),
    )


def minimal_undirected_estimators(metric_keys: list[str]) -> tuple[str, ...]:
    """Return the stable minimal MNE estimator union for selected metrics."""
    requested = {
        _UNDIRECTED_ESTIMATOR_BY_METRIC[str(metric_key)] for metric_key in metric_keys
    }
    return tuple(
        estimator
        for estimator in ("cohy", "plv", "ciplv", "pli", "wpli")
        if estimator in requested
    )


def _append_trace(payload: dict[str, Any]) -> None:
    trace_path = os.environ.get(TENSOR_BENCHMARK_TRACE_PATH_ENV, "").strip()
    if not trace_path:
        return
    line = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
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


def _normalize_pairs(
    svc: Any,
    raw: Any,
    *,
    selected_channels: list[str] | None,
    selected_pairs: list[tuple[str, str]] | None,
    directed: bool,
) -> tuple[list[str], list[tuple[str, str]]]:
    available_channels = set(raw.ch_names)
    if selected_pairs is None:
        picks = [
            name
            for name in (selected_channels or raw.ch_names)
            if name in available_channels
        ]
        if len(picks) < 2:
            raise ValueError("Connectivity requires at least 2 valid channels.")
        if directed:
            from itertools import permutations

            pairs = list(permutations(picks, 2))
        else:
            from itertools import combinations

            pairs = list(combinations(picks, 2))
        return picks, pairs

    pairs = svc._normalize_selected_pairs(
        selected_pairs,
        available_channels=available_channels,
        directed=directed,
    )
    if not pairs:
        raise ValueError("No valid selected connectivity pairs are available.")
    picks = [name for name in raw.ch_names if any(name in pair for pair in pairs)]
    if len(picks) < 2:
        raise ValueError("Connectivity requires at least 2 valid channels.")
    return picks, pairs


def _prepare_undirected_cohort(
    svc: Any,
    context: RecordContext,
    plans: list[Any],
    *,
    cohort_id: str,
) -> dict[str, Any]:
    first_plan = plans[0]
    kwargs = first_plan.runner_kwargs
    resolver = svc.PathResolver(context)
    input_path = svc.preproc_step_raw_path(resolver, "finish")
    if svc.indicator_from_log(svc.preproc_step_log_path(resolver, "finish")) != "green":
        raise ValueError("Missing green preproc finish log.")
    if not input_path.exists():
        raise ValueError("Missing preproc finish raw input.")

    import mne

    raw = mne.io.read_raw_fif(str(input_path), preload=False, verbose="ERROR")
    try:
        picks, pairs = _normalize_pairs(
            svc,
            raw,
            selected_channels=kwargs.get("selected_channels"),
            selected_pairs=kwargs.get("selected_pairs"),
            directed=False,
        )
        nyquist = float(raw.info["sfreq"]) / 2.0
        low_freq = float(kwargs["low_freq"])
        applied_high = min(float(kwargs["high_freq"]), nyquist)
        if applied_high <= low_freq:
            raise ValueError("High frequency is invalid for current Nyquist frequency.")
        freqs_full = svc._build_frequency_grid(
            low_freq,
            applied_high,
            float(kwargs["step_hz"]),
        )
        notch_payload = svc.build_tensor_metric_notch_payload(
            kwargs.get("notches"),
            kwargs.get("notch_radii", svc.DEFAULT_TENSOR_NOTCH_RADIUS),
        )
        runtime_notches = tuple(float(item) for item in notch_payload["notches"])
        runtime_radii = svc._expand_notch_radii(
            notch_payload["notch_radii"], len(runtime_notches)
        )
        notch_intervals = svc._compute_notch_intervals(
            low_freq=low_freq,
            high_freq=applied_high,
            notches=runtime_notches,
            notch_radii=runtime_radii,
        )
        freqs_compute = freqs_full
        if notch_intervals:
            retained, removed_mask = svc._cut_frequency_grid_by_intervals(
                freqs_full, notch_intervals
            )
            if bool(np.any(removed_mask)):
                if retained.size < 2:
                    raise ValueError(
                        "Notch exclusion removed too many bins; reduce notch radii or widen the frequency range."
                    )
                freqs_compute = retained

        method_norm = svc._normalize_metric_method(
            kwargs.get("method", "morlet"),
            metric_label=first_plan.metric_label,
        )
        mask_radii = svc._compute_mask_radii_seconds(
            freqs_full,
            method=method_norm,
            time_resolution_s=float(kwargs["time_resolution_s"]),
            min_cycles=kwargs.get("min_cycles"),
            max_cycles=kwargs.get("max_cycles"),
            mt_time_bandwidth_product=float(kwargs["mt_time_bandwidth_product"]),
            mt_min_cycles=float(kwargs["mt_min_cycles"]),
        )
        annotation_skip_radius_s = (
            float(np.min(mask_radii))
            if bool(kwargs.get("mask_edge_effects", True))
            else None
        )
        grid_kwargs = {
            "freqs": [float(item) for item in freqs_compute.tolist()],
            "time_resolution_s": float(kwargs["time_resolution_s"]),
            "hop_s": float(kwargs["hop_s"]),
            "pairs": [(str(a), str(b)) for a, b in pairs],
            "multivariate": False,
            "ordered_pairs": False,
            "picks": list(picks),
            "spectral_mode": (
                "cwt_morlet" if method_norm == "morlet" else "multitaper"
            ),
            "mt_time_bandwidth_product": float(kwargs["mt_time_bandwidth_product"]),
            "mt_min_cycles": float(kwargs["mt_min_cycles"]),
            "min_cycles": kwargs.get("min_cycles"),
            "max_cycles": kwargs.get("max_cycles"),
            "annotation_skip_radius_s": annotation_skip_radius_s,
        }
        estimators = minimal_undirected_estimators(
            [str(plan.plan_key) for plan in plans]
        )
        from lfptensorpipe.lfp.connectivity.grid import grid

        description, metadata = grid(
            raw,
            **grid_kwargs,
            method=list(estimators),
            return_task_description=True,
            outer_n_jobs=1,
        )
        if len(estimators) == 1:
            metadata = {estimators[0]: metadata}
    finally:
        raw.close()

    group_descriptors = []
    pair_count = int(description["shape"][1])
    for group in description["groups"]:
        group_descriptors.append(
            {
                "family": "undirected",
                "owner_id": cohort_id,
                "group_id": int(group["group_id"]),
                "valid_time_indices": group["valid_time_indices"],
                "weight_per_time": (
                    len(group["call_frequency_indices"]) * pair_count * len(estimators)
                ),
            }
        )
    return {
        "cohort_id": cohort_id,
        "plans": plans,
        "estimators": estimators,
        "grid_kwargs": grid_kwargs,
        "description": description,
        "metadata": metadata,
        "group_descriptors": group_descriptors,
        "task_results": [],
        "failures": [],
    }


def _prepare_trgc(
    context: RecordContext,
    backend_plans: dict[str, Any],
) -> dict[str, Any]:
    from .runners.connectivity_trgc import _prepare_trgc_backend_inputs
    from lfptensorpipe.lfp.connectivity.grid import grid

    first_plan = next(iter(backend_plans.values()))
    runner_kwargs = dict(first_plan.runner_kwargs)
    runner_kwargs.pop("backend_method", None)
    prepared = _prepare_trgc_backend_inputs(context, **runner_kwargs)
    raw = prepared["raw"]
    grid_kwargs = {
        "freqs": [float(item) for item in prepared["freqs_compute"].tolist()],
        "time_resolution_s": float(prepared["time_resolution_s"]),
        "hop_s": float(prepared["hop_s"]),
        "pairs": [tuple(item) for item in prepared["pairs_compute"]],
        "multivariate": True,
        "ordered_pairs": True,
        "spectral_mode": str(prepared["spectral_mode_use"]),
        "mt_time_bandwidth_product": float(prepared["mt_time_bandwidth_product"]),
        "mt_min_cycles": float(prepared["mt_min_cycles"]),
        "min_cycles": prepared["min_cycles"],
        "max_cycles": prepared["max_cycles"],
        "gc_n_lags": int(prepared["gc_n_lags"]),
        "group_by_samples": bool(prepared["group_by_samples"]),
        "round_ms": float(prepared["round_ms"]),
        "picks": list(prepared["picks"]),
        "annotation_skip_radius_s": prepared["annotation_skip_radius_s"],
    }
    descriptions: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    try:
        for backend_method in ("gc", "gc_tr"):
            descriptions[backend_method], metadata[backend_method] = grid(
                raw,
                **grid_kwargs,
                method=backend_method,
                return_task_description=True,
                outer_n_jobs=1,
            )
    finally:
        raw.close()

    unordered_pair_count = max(len(prepared["pairs_compute"]) // 2, 1)
    group_descriptors: list[dict[str, Any]] = []
    for backend_method, description in descriptions.items():
        owner_id = f"trgc:{backend_method}"
        for group in description["groups"]:
            group_descriptors.append(
                {
                    "family": "trgc",
                    "owner_id": owner_id,
                    "group_id": int(group["group_id"]),
                    "valid_time_indices": group["valid_time_indices"],
                    "weight_per_time": (
                        len(group["call_frequency_indices"]) * unordered_pair_count
                    ),
                }
            )
    return {
        "backend_plans": backend_plans,
        "prepared": prepared,
        "grid_kwargs": grid_kwargs,
        "descriptions": descriptions,
        "metadata": metadata,
        "group_descriptors": group_descriptors,
        "task_results": {"gc": [], "gc_tr": []},
        "failures": {"gc": [], "gc_tr": []},
    }


def _assemble_blocks(
    description: dict[str, Any],
    methods: tuple[str, ...],
    task_results: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    shape = tuple(int(item) for item in description["shape"])
    output = {
        method: np.full(
            shape,
            complex(np.nan, np.nan) if method == "cohy" else np.nan,
            dtype=np.complex128 if method == "cohy" else float,
        )
        for method in methods
    }
    pair_idx = np.arange(shape[1], dtype=int)
    for result in task_results:
        time_idx = np.asarray(result["time_indices"], dtype=int)
        freq_idx = np.asarray(result["frequency_indices"], dtype=int)
        for method in methods:
            values = np.asarray(result["data"][method])
            expected = (time_idx.size, shape[1], freq_idx.size)
            if values.shape != expected:
                raise RuntimeError(
                    f"Connectivity task block shape mismatch: {values.shape} != {expected}."
                )
            output[method][0][np.ix_(pair_idx, freq_idx, time_idx)] = np.transpose(
                values, (1, 2, 0)
            )
    return output


def _metric_from_shared(
    metric_key: str,
    tensors: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    estimator = _UNDIRECTED_ESTIMATOR_BY_METRIC[metric_key]
    source = np.asarray(tensors[estimator])
    meta = deepcopy(metadata[estimator])
    params = dict(meta.get("params", {}) or {})
    if metric_key == "coherence":
        tensor = np.abs(source)
        params.update({"method": "coh", "method_internal": "coh"})
        params.pop("output_component", None)
    elif metric_key == "imcoh_abs":
        tensor = np.abs(np.imag(source))
        params.update(
            {
                "method": "imcoh_abs",
                "method_internal": "cohy",
                "output_component": "absolute_imaginary",
            }
        )
    else:
        tensor = np.asarray(source, dtype=float)
        params.update({"method": estimator, "method_internal": estimator})
        params.pop("output_component", None)
    meta["params"] = params
    return np.asarray(tensor, dtype=float), meta


def _precomputed_grid(
    tensor: np.ndarray,
    metadata: dict[str, Any],
) -> Callable[..., tuple[np.ndarray, dict[str, Any]]]:
    def _grid(*args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        _ = args, kwargs
        return np.asarray(tensor), deepcopy(metadata)

    return _grid


def _write_plan_failure(
    svc: Any,
    resolver: Any,
    plan: Any,
    *,
    message: str,
) -> RuntimeResult:
    metric_key = str(plan.log_metric_key or plan.plan_key)
    svc._write_metric_log(
        resolver,
        metric_key,
        completed=False,
        params=svc._sanitize_metric_params_for_logs(plan.runner_kwargs),
        input_path=str(svc.preproc_step_raw_path(resolver, "finish")),
        output_path=str(svc.tensor_metric_tensor_path(resolver, metric_key)),
        message=f"{plan.metric_label} failed: {message}",
    )
    return False, message, str(plan.metric_label)


def _task_payload(
    chunk: ConnectivityChunk,
    owner: dict[str, Any],
    *,
    sequence: int,
) -> dict[str, Any]:
    methods = (
        owner["estimators"]
        if chunk.family == "undirected"
        else (chunk.owner_id.split(":", 1)[1],)
    )
    return {
        "task_id": f"{chunk.family}:{chunk.owner_id}:{chunk.group_id}:{chunk.start_index}:{sequence}",
        "family": chunk.family,
        "owner_id": chunk.owner_id,
        "group_id": int(chunk.group_id),
        "time_indices": list(chunk.time_indices),
        "methods": list(methods) if len(methods) > 1 else methods[0],
        "grid_kwargs": owner["grid_kwargs"],
    }


def run_connectivity_coordinator(
    svc: Any,
    resolver: Any,
    context: RecordContext,
    *,
    runtime_plans: dict[str, Any],
    global_compute_slots: int,
    slot_gate: Any | None = None,
    executor_factory: Callable[..., Any] | None = None,
) -> dict[str, RuntimeResult]:
    """Compute all selected connectivity plans through one bounded pool."""
    results: dict[str, RuntimeResult] = {}
    undirected_plans = [
        plan
        for plan in runtime_plans.values()
        if plan.runner_key == UNDIRECTED_RUNNER_KEY
    ]
    backend_plans = {
        str(plan.runner_kwargs["backend_method"]): plan
        for plan in runtime_plans.values()
        if plan.runner_key == "trgc_backend"
    }
    finalize_plan = next(
        (plan for plan in runtime_plans.values() if plan.runner_key == "trgc_finalize"),
        None,
    )

    cohorts_by_signature: dict[tuple[Any, ...], list[Any]] = {}
    for plan in undirected_plans:
        cohorts_by_signature.setdefault(shared_compute_signature(plan), []).append(plan)

    cohorts: list[dict[str, Any]] = []
    group_descriptors: list[dict[str, Any]] = []
    for cohort_index, plans in enumerate(cohorts_by_signature.values()):
        cohort_id = f"undirected-{cohort_index:03d}"
        try:
            cohort = _prepare_undirected_cohort(
                svc,
                context,
                plans,
                cohort_id=cohort_id,
            )
        except BuildTensorCancellationRequested:
            raise
        except Exception as exc:  # noqa: BLE001
            for plan in plans:
                results[plan.plan_key] = _write_plan_failure(
                    svc, resolver, plan, message=str(exc)
                )
            continue
        cohorts.append(cohort)
        group_descriptors.extend(cohort["group_descriptors"])

    trgc_state: dict[str, Any] | None = None
    if backend_plans and finalize_plan is not None:
        try:
            trgc_state = _prepare_trgc(context, backend_plans)
        except BuildTensorCancellationRequested:
            raise
        except Exception as exc:  # noqa: BLE001
            results[finalize_plan.plan_key] = _write_plan_failure(
                svc, resolver, finalize_plan, message=str(exc)
            )
        else:
            group_descriptors.extend(trgc_state["group_descriptors"])

    chunks = build_connectivity_chunks(
        group_descriptors,
        global_compute_slots=int(global_compute_slots),
    )
    owner_by_id = {cohort["cohort_id"]: cohort for cohort in cohorts}
    if trgc_state is not None:
        owner_by_id["trgc:gc"] = {
            "grid_kwargs": trgc_state["grid_kwargs"],
            "estimators": ("gc",),
        }
        owner_by_id["trgc:gc_tr"] = {
            "grid_kwargs": trgc_state["grid_kwargs"],
            "estimators": ("gc_tr",),
        }

    task_cpu_s = 0.0
    task_wall_start_ns: int | None = None
    task_wall_end_ns: int | None = None
    if chunks:
        raise_if_tensor_cancellation_requested()
        executor_cls = executor_factory or ProcessPoolExecutor
        executor_kwargs: dict[str, Any] = {
            "max_workers": min(max(int(global_compute_slots), 1), len(chunks)),
            "initializer": initialize_connectivity_worker,
            "initargs": (str(svc.preproc_step_raw_path(resolver, "finish")),),
        }
        if executor_factory is None:
            executor_kwargs["mp_context"] = multiprocessing.get_context("spawn")
        future_to_chunk: dict[Future[Any], ConnectivityChunk] = {}
        with executor_cls(**executor_kwargs) as executor:
            for sequence, chunk in enumerate(chunks):
                raise_if_tensor_cancellation_requested()
                if slot_gate is not None:
                    slot_gate.acquire_connectivity()
                try:
                    future = executor.submit(
                        run_connectivity_pool_task,
                        _task_payload(
                            chunk,
                            owner_by_id[chunk.owner_id],
                            sequence=sequence,
                        ),
                    )
                except Exception:
                    if slot_gate is not None:
                        slot_gate.release()
                    raise
                if slot_gate is not None:
                    future.add_done_callback(lambda _future: slot_gate.release())
                future_to_chunk[future] = chunk

            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    task_result = future.result()
                except BuildTensorCancellationRequested:
                    for pending_future in future_to_chunk:
                        pending_future.cancel()
                    raise
                except Exception as exc:  # noqa: BLE001
                    if chunk.family == "undirected":
                        owner_by_id[chunk.owner_id]["failures"].append(str(exc))
                    elif trgc_state is not None:
                        backend = chunk.owner_id.split(":", 1)[1]
                        trgc_state["failures"][backend].append(str(exc))
                    continue
                start_ns = int(task_result["start_perf_ns"])
                end_ns = int(task_result["end_perf_ns"])
                task_wall_start_ns = (
                    start_ns
                    if task_wall_start_ns is None
                    else min(task_wall_start_ns, start_ns)
                )
                task_wall_end_ns = (
                    end_ns
                    if task_wall_end_ns is None
                    else max(task_wall_end_ns, end_ns)
                )
                task_cpu_s += float(task_result["worker_cpu_s"])
                if chunk.family == "undirected":
                    owner_by_id[chunk.owner_id]["task_results"].append(task_result)
                elif trgc_state is not None:
                    backend = chunk.owner_id.split(":", 1)[1]
                    trgc_state["task_results"][backend].append(task_result)

        raise_if_tensor_cancellation_requested()

    for cohort in cohorts:
        raise_if_tensor_cancellation_requested()
        plans = cohort["plans"]
        if cohort["failures"]:
            failure = "Connectivity cohort task failed: " + cohort["failures"][0]
            for plan in plans:
                results[plan.plan_key] = _write_plan_failure(
                    svc, resolver, plan, message=failure
                )
            continue
        try:
            tensors = _assemble_blocks(
                cohort["description"],
                cohort["estimators"],
                cohort["task_results"],
            )
        except BuildTensorCancellationRequested:
            raise
        except Exception as exc:  # noqa: BLE001
            for plan in plans:
                results[plan.plan_key] = _write_plan_failure(
                    svc, resolver, plan, message=str(exc)
                )
            continue

        cohort_start_ns = (
            min(int(item["start_perf_ns"]) for item in cohort["task_results"])
            if cohort["task_results"]
            else time.perf_counter_ns()
        )
        cohort_cpu_s = sum(
            float(item["worker_cpu_s"]) for item in cohort["task_results"]
        )
        for plan in plans:
            raise_if_tensor_cancellation_requested()
            metric_key = str(plan.plan_key)
            try:
                tensor, metadata = _metric_from_shared(
                    metric_key,
                    tensors,
                    cohort["metadata"],
                )
                kwargs = dict(plan.runner_kwargs)
                ok, message = svc._run_undirected_connectivity_metric(
                    context,
                    **kwargs,
                    n_jobs=1,
                    outer_n_jobs=1,
                    conn_grid_fn=_precomputed_grid(tensor, metadata),
                )
            except BuildTensorCancellationRequested:
                raise
            except Exception as exc:  # noqa: BLE001
                ok, message = False, f"Unexpected runtime failure: {exc}"
                _write_plan_failure(svc, resolver, plan, message=message)
            end_ns = time.perf_counter_ns()
            results[metric_key] = (bool(ok), str(message), str(plan.metric_label))
            _append_trace(
                {
                    "run_id": os.environ.get(TENSOR_RUN_ID_ENV, ""),
                    "plan_key": metric_key,
                    "log_metric_key": metric_key,
                    "cohort_id": cohort["cohort_id"],
                    "pid": os.getpid(),
                    "ok": bool(ok),
                    "start_perf_ns": cohort_start_ns,
                    "end_perf_ns": end_ns,
                    "elapsed_s": (end_ns - cohort_start_ns) / 1e9,
                    "worker_cpu_s": cohort_cpu_s,
                    "shared_worker_cpu_nonadditive": True,
                }
            )

    if trgc_state is not None and finalize_plan is not None:
        raise_if_tensor_cancellation_requested()
        backend_ok = True
        backend_starts: list[int] = []
        backend_cpu_total = 0.0
        for backend_method in ("gc", "gc_tr"):
            raise_if_tensor_cancellation_requested()
            plan = backend_plans[backend_method]
            failures = trgc_state["failures"][backend_method]
            task_results = trgc_state["task_results"][backend_method]
            if failures or not task_results:
                backend_ok = False
                message = (
                    "TRGC task failed: " + failures[0]
                    if failures
                    else "TRGC task produced no blocks."
                )
                results[plan.plan_key] = _write_plan_failure(
                    svc, resolver, plan, message=message
                )
                continue
            backend_start_ns = min(int(item["start_perf_ns"]) for item in task_results)
            backend_cpu_s = sum(float(item["worker_cpu_s"]) for item in task_results)
            backend_starts.append(backend_start_ns)
            backend_cpu_total += backend_cpu_s
            assembled = _assemble_blocks(
                trgc_state["descriptions"][backend_method],
                (backend_method,),
                task_results,
            )[backend_method]
            kwargs = dict(plan.runner_kwargs)
            ok, message = svc._run_trgc_backend_metric(
                context,
                **kwargs,
                n_jobs=1,
                outer_n_jobs=1,
                conn_grid_fn=_precomputed_grid(
                    assembled,
                    trgc_state["metadata"][backend_method],
                ),
            )
            backend_end_ns = time.perf_counter_ns()
            backend_ok = backend_ok and bool(ok)
            results[plan.plan_key] = (bool(ok), str(message), str(plan.metric_label))
            _append_trace(
                {
                    "run_id": os.environ.get(TENSOR_RUN_ID_ENV, ""),
                    "plan_key": str(plan.plan_key),
                    "log_metric_key": "trgc",
                    "pid": os.getpid(),
                    "ok": bool(ok),
                    "start_perf_ns": backend_start_ns,
                    "end_perf_ns": backend_end_ns,
                    "elapsed_s": (backend_end_ns - backend_start_ns) / 1e9,
                    "worker_cpu_s": backend_cpu_s,
                }
            )

        if backend_ok:
            raise_if_tensor_cancellation_requested()
            ok, message = svc._run_trgc_finalize_metric(
                context,
                **finalize_plan.runner_kwargs,
                n_jobs=1,
                outer_n_jobs=1,
            )
            final_end_ns = time.perf_counter_ns()
            trgc_start_ns = min(backend_starts)
            results[finalize_plan.plan_key] = (
                bool(ok),
                str(message),
                str(finalize_plan.metric_label),
            )
            _append_trace(
                {
                    "run_id": os.environ.get(TENSOR_RUN_ID_ENV, ""),
                    "plan_key": str(finalize_plan.plan_key),
                    "log_metric_key": "trgc",
                    "pid": os.getpid(),
                    "ok": bool(ok),
                    "start_perf_ns": trgc_start_ns,
                    "end_perf_ns": final_end_ns,
                    "elapsed_s": (final_end_ns - trgc_start_ns) / 1e9,
                    "worker_cpu_s": backend_cpu_total,
                    "shared_worker_cpu_nonadditive": True,
                }
            )
        else:
            results[finalize_plan.plan_key] = _write_plan_failure(
                svc,
                resolver,
                finalize_plan,
                message="TRGC finalization blocked by failed backend tasks.",
            )

    if task_wall_start_ns is not None and task_wall_end_ns is not None:
        _append_trace(
            {
                "run_id": os.environ.get(TENSOR_RUN_ID_ENV, ""),
                "plan_key": "connectivity_pool",
                "log_metric_key": "connectivity_pool",
                "pid": os.getpid(),
                "ok": not any(not value[0] for value in results.values()),
                "start_perf_ns": task_wall_start_ns,
                "end_perf_ns": task_wall_end_ns,
                "elapsed_s": (task_wall_end_ns - task_wall_start_ns) / 1e9,
                "worker_cpu_s": task_cpu_s,
                "task_count": len(chunks),
                "slot_costs": {"undirected": 1, "trgc": 1},
            }
        )
    return results


__all__ = [
    "CONNECTIVITY_RUNNER_KEYS",
    "metric_output_signature",
    "minimal_undirected_estimators",
    "run_connectivity_coordinator",
    "shared_compute_signature",
]
