"""Runtime-plan execution helpers for Build Tensor orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping

from lfptensorpipe.desktop_runtime import (
    RUNTIME_PLAN_WORKER_FLAG,
    RUNTIME_PLAN_WORKER_MODULE,
    build_worker_command,
)
from lfptensorpipe.app.path_resolver import RecordContext

from .cancellation import (
    BuildTensorCancellationRequested,
    raise_if_tensor_cancellation_requested,
    tensor_cancellation_requested,
)
from .logging import TENSOR_RUN_ID_ENV
from .runner_dispatch import invoke_runtime_plan_runner
from .transaction_manifest import recover_tensor_run_transactions

_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
RuntimeResult = tuple[bool, str, str]
ProcessPhaseExecutor = Callable[..., dict[str, RuntimeResult]]
INDEPENDENT_RUNTIME_PLAN_KEYS = frozenset(
    {"raw_power", "periodic_aperiodic", "psi", "burst"}
)


class ComputeSlotGate:
    """Coordinate deterministic independent-plan priority with connectivity tasks."""

    def __init__(self, capacity: int, *, pending_independent: int) -> None:
        self._capacity = max(int(capacity), 1)
        self._available = self._capacity
        self._pending_independent = max(int(pending_independent), 0)
        self._fatal_error: BaseException | None = None
        self._condition = threading.Condition()

    def try_acquire_independent(self) -> bool:
        with self._condition:
            if self._fatal_error is not None:
                return False
            if self._pending_independent <= 0 or self._available <= 0:
                return False
            self._pending_independent -= 1
            self._available -= 1
            return True

    def discard_pending_independent(self, count: int) -> None:
        with self._condition:
            self._pending_independent = max(
                self._pending_independent - max(int(count), 0), 0
            )
            self._condition.notify_all()

    def finish_independent_admission(self) -> None:
        """Release connectivity priority held by unadmitted independent work."""
        with self._condition:
            self._pending_independent = 0
            self._condition.notify_all()

    def fail_independent_admission(self, error: BaseException) -> None:
        """Close connectivity admission after a fatal independent scheduler error."""
        with self._condition:
            self._pending_independent = 0
            if self._fatal_error is None:
                self._fatal_error = error
            self._condition.notify_all()

    def _raise_if_failed(self) -> None:
        if self._fatal_error is None:
            return
        raise RuntimeError(
            "Build Tensor connectivity admission stopped because the "
            "independent scheduler failed."
        ) from self._fatal_error

    def acquire_connectivity(self) -> None:
        with self._condition:
            self._raise_if_failed()
            while self._pending_independent > 0 or self._available <= 0:
                self._condition.wait(timeout=0.05)
                raise_if_tensor_cancellation_requested()
                self._raise_if_failed()
            self._available -= 1

    def release(self) -> None:
        with self._condition:
            self._available += 1
            if self._available > self._capacity:
                raise RuntimeError(
                    "Build Tensor compute-slot budget was over-released."
                )
            self._condition.notify_all()


@dataclass(frozen=True)
class RuntimePlan:
    plan_key: str
    metric_label: str
    runner_key: str
    runner_kwargs: dict[str, Any]
    phase: int = 0
    dependencies: tuple[str, ...] = ()
    log_metric_key: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "plan_key": self.plan_key,
            "metric_label": self.metric_label,
            "runner_key": self.runner_key,
            "runner_kwargs": dict(self.runner_kwargs),
            "phase": int(self.phase),
            "dependencies": list(self.dependencies),
            "log_metric_key": self.log_metric_key,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RuntimePlan:
        if not isinstance(payload, Mapping):
            raise ValueError("Runtime plan payload must be a mapping.")
        plan_key = str(payload.get("plan_key", "")).strip()
        metric_label = str(payload.get("metric_label", "")).strip()
        runner_key = str(payload.get("runner_key", "")).strip()
        runner_kwargs = payload.get("runner_kwargs", {})
        dependencies = payload.get("dependencies", ())
        if not plan_key:
            raise ValueError("Runtime plan payload requires `plan_key`.")
        if not metric_label:
            raise ValueError("Runtime plan payload requires `metric_label`.")
        if not runner_key:
            raise ValueError("Runtime plan payload requires `runner_key`.")
        if not isinstance(runner_kwargs, Mapping):
            raise ValueError("Runtime plan payload `runner_kwargs` must be a mapping.")
        if not isinstance(dependencies, (list, tuple)):
            raise ValueError(
                "Runtime plan payload `dependencies` must be a list or tuple."
            )
        log_metric_key_raw = payload.get("log_metric_key")
        return cls(
            plan_key=plan_key,
            metric_label=metric_label,
            runner_key=runner_key,
            runner_kwargs=dict(runner_kwargs),
            phase=int(payload.get("phase", 0)),
            dependencies=tuple(str(item) for item in dependencies),
            log_metric_key=(
                str(log_metric_key_raw).strip()
                if log_metric_key_raw is not None
                else None
            ),
        )


def apply_effective_parallel_policy(
    runtime_plans: dict[str, RuntimePlan],
    effective_n_jobs_map: dict[str, dict[str, int]],
) -> tuple[int, int]:
    policy_n_jobs = 1
    policy_outer_n_jobs = 1
    for metric_key in list(runtime_plans):
        effective_n_jobs_map[metric_key] = {
            "n_jobs": int(policy_n_jobs),
            "outer_n_jobs": int(policy_outer_n_jobs),
        }
    return policy_n_jobs, policy_outer_n_jobs


def _runtime_plan_log_metric_key(runtime_plan: RuntimePlan) -> str:
    return runtime_plan.log_metric_key or runtime_plan.plan_key


def _runtime_plan_allows_process_backend(
    runtime_plan: RuntimePlan,
    *,
    platform: str | None = None,
) -> bool:
    _ = runtime_plan, platform
    return True


def _runtime_plan_metric_params(
    svc: Any,
    merged_metric_params_map: dict[str, dict[str, Any]],
    *,
    runtime_plan: RuntimePlan,
    policy_n_jobs: int,
    policy_outer_n_jobs: int,
) -> dict[str, Any]:
    metric_params = svc._sanitize_metric_params_for_logs(
        merged_metric_params_map.get(_runtime_plan_log_metric_key(runtime_plan), {})
    )
    metric_params.update(
        svc._effective_n_jobs_payload(
            n_jobs=int(policy_n_jobs),
            outer_n_jobs=int(policy_outer_n_jobs),
        )
    )
    return metric_params


def _write_runtime_plan_failure_log(
    svc: Any,
    resolver: Any,
    *,
    runtime_plan: RuntimePlan,
    merged_metric_params_map: dict[str, dict[str, Any]],
    policy_n_jobs: int,
    policy_outer_n_jobs: int,
    failure_message: str,
) -> None:
    log_metric_key = _runtime_plan_log_metric_key(runtime_plan)
    svc._write_metric_log(
        resolver,
        log_metric_key,
        completed=False,
        params=_runtime_plan_metric_params(
            svc,
            merged_metric_params_map,
            runtime_plan=runtime_plan,
            policy_n_jobs=policy_n_jobs,
            policy_outer_n_jobs=policy_outer_n_jobs,
        ),
        input_path=str(svc.preproc_step_raw_path(resolver, "finish")),
        output_path=str(svc.tensor_metric_tensor_path(resolver, log_metric_key)),
        message=f"{runtime_plan.metric_label} failed: {failure_message}",
    )


def _runtime_plan_failure_result(
    svc: Any,
    resolver: Any,
    *,
    runtime_plan: RuntimePlan,
    merged_metric_params_map: dict[str, dict[str, Any]],
    policy_n_jobs: int,
    policy_outer_n_jobs: int,
    failure_message: str,
) -> RuntimeResult:
    _write_runtime_plan_failure_log(
        svc,
        resolver,
        runtime_plan=runtime_plan,
        merged_metric_params_map=merged_metric_params_map,
        policy_n_jobs=policy_n_jobs,
        policy_outer_n_jobs=policy_outer_n_jobs,
        failure_message=failure_message,
    )
    return False, failure_message, runtime_plan.metric_label


def run_runtime_plan(
    svc: Any,
    resolver: Any,
    context: RecordContext,
    *,
    runtime_plan: RuntimePlan,
    merged_metric_params_map: dict[str, dict[str, Any]],
    policy_n_jobs: int,
    policy_outer_n_jobs: int,
) -> RuntimeResult:
    try:
        ok, message = invoke_runtime_plan_runner(
            svc,
            context,
            runner_key=runtime_plan.runner_key,
            runner_kwargs=runtime_plan.runner_kwargs,
            n_jobs=policy_n_jobs,
            outer_n_jobs=policy_outer_n_jobs,
        )
    except BuildTensorCancellationRequested:
        raise
    except Exception as exc:  # noqa: BLE001
        return _runtime_plan_failure_result(
            svc,
            resolver,
            runtime_plan=runtime_plan,
            merged_metric_params_map=merged_metric_params_map,
            policy_n_jobs=policy_n_jobs,
            policy_outer_n_jobs=policy_outer_n_jobs,
            failure_message=f"Unexpected runtime failure: {exc}",
        )
    return ok, message, runtime_plan.metric_label


def _runtime_plan_worker_env(
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    pythonpath_entries: list[str] = []
    for entry in sys.path:
        token = str(entry).strip()
        if token and token not in pythonpath_entries:
            pythonpath_entries.append(token)
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        token = entry.strip()
        if token and token not in pythonpath_entries:
            pythonpath_entries.append(token)
    if pythonpath_entries:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["MPLBACKEND"] = "Agg"
    for env_var in _NATIVE_THREAD_ENV_VARS:
        env[env_var] = "1"
    return env


def _runtime_plan_temp_json_path(stem: str) -> Path:
    fd, raw_path = tempfile.mkstemp(
        prefix=f"lfptensorpipe_{stem}_",
        suffix=".json",
    )
    os.close(fd)
    path = Path(raw_path)
    path.unlink(missing_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _serialize_context(context: RecordContext) -> dict[str, str]:
    return {
        "project_root": str(context.project_root),
        "subject": context.subject,
        "record": context.record,
    }


def _connectivity_fast_path_eligible(
    runtime_plans: Mapping[str, RuntimePlan],
) -> bool:
    """Return whether the explicit coordinator preserves the plan topology."""
    for plan in runtime_plans.values():
        if plan.runner_key == "trgc_finalize":
            if int(plan.phase) < 1 or not plan.dependencies:
                return False
            continue
        if int(plan.phase) != 0 or plan.dependencies:
            return False
    return True


def _execute_runtime_plan_phase_in_subprocesses(
    svc: Any,
    resolver: Any,
    context: RecordContext,
    *,
    runtime_plans: dict[str, RuntimePlan],
    merged_metric_params_map: dict[str, dict[str, Any]],
    runnable_keys: list[str],
    policy_n_jobs: int,
    policy_outer_n_jobs: int,
    global_compute_slots: int,
    slot_gate: ComputeSlotGate | None = None,
) -> dict[str, RuntimeResult]:
    runtime_results: dict[str, RuntimeResult] = {}
    process_states: dict[str, dict[str, Any]] = {}
    pending_keys = list(runnable_keys)
    run_id = os.environ.get(TENSOR_RUN_ID_ENV, "").strip()

    def _launch(metric_key: str) -> None:
        runtime_plan = runtime_plans[metric_key]
        request_path: Path | None = None
        result_path: Path | None = None
        try:
            request_path = _runtime_plan_temp_json_path(
                f"runtime_plan_request_{runtime_plan.plan_key}"
            )
            result_path = _runtime_plan_temp_json_path(
                f"runtime_plan_result_{runtime_plan.plan_key}"
            )
            request_payload = {
                "context": _serialize_context(context),
                "runtime_plan": runtime_plan.to_payload(),
                "merged_metric_params_map": merged_metric_params_map,
                "policy_n_jobs": int(policy_n_jobs),
                "policy_outer_n_jobs": int(policy_outer_n_jobs),
                "run_id": run_id,
            }
            _write_json(request_path, request_payload)
            process = subprocess.Popen(
                build_worker_command(
                    module_name=RUNTIME_PLAN_WORKER_MODULE,
                    embedded_flag=RUNTIME_PLAN_WORKER_FLAG,
                    worker_args=[
                        "--request",
                        str(request_path),
                        "--result",
                        str(result_path),
                    ],
                    python_exec=sys.executable,
                ),
                cwd=str(Path.cwd()),
                env=_runtime_plan_worker_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001
            if slot_gate is not None:
                slot_gate.release()
            if request_path is not None:
                request_path.unlink(missing_ok=True)
            if result_path is not None:
                result_path.unlink(missing_ok=True)
            runtime_results[metric_key] = _runtime_plan_failure_result(
                svc,
                resolver,
                runtime_plan=runtime_plan,
                merged_metric_params_map=merged_metric_params_map,
                policy_n_jobs=policy_n_jobs,
                policy_outer_n_jobs=policy_outer_n_jobs,
                failure_message=(
                    "Unexpected runtime failure: "
                    f"Failed to prepare or launch runtime plan worker: {exc}"
                ),
            )
            return
        process_states[metric_key] = {
            "process": process,
            "request_path": request_path,
            "result_path": result_path,
        }

    try:
        slot_cap = max(int(global_compute_slots), 1)
        while pending_keys or process_states:
            cancellation_requested = tensor_cancellation_requested()
            if cancellation_requested:
                if slot_gate is not None and pending_keys:
                    slot_gate.discard_pending_independent(len(pending_keys))
                pending_keys.clear()
                if not process_states:
                    raise_if_tensor_cancellation_requested()
            while (
                not cancellation_requested
                and pending_keys
                and len(process_states) < slot_cap
            ):
                if slot_gate is not None and not slot_gate.try_acquire_independent():
                    break
                _launch(pending_keys.pop(0))
            if not process_states:
                continue

            completed_keys: list[tuple[str, int]] = []
            for metric_key, state in process_states.items():
                exit_code = state["process"].poll()
                if exit_code is not None:
                    completed_keys.append((metric_key, int(exit_code)))
            if not completed_keys:
                time.sleep(0.02)
                continue

            for metric_key, exit_code in completed_keys:
                state = process_states.pop(metric_key)
                if slot_gate is not None:
                    slot_gate.release()
                runtime_plan = runtime_plans[metric_key]
                try:
                    result_payload = _read_json(state["result_path"])
                finally:
                    state["request_path"].unlink(missing_ok=True)
                    state["result_path"].unlink(missing_ok=True)
                if isinstance(result_payload, dict):
                    message = str(result_payload.get("message", "")).strip()
                    if not message:
                        message = f"{runtime_plan.metric_label} worker exited without a message."
                    runtime_results[metric_key] = (
                        bool(result_payload.get("ok", False)),
                        message,
                        runtime_plan.metric_label,
                    )
                else:
                    raise_if_tensor_cancellation_requested()
                    if run_id:
                        metric_key_for_log = _runtime_plan_log_metric_key(runtime_plan)
                        metric_root = svc.tensor_metric_tensor_path(
                            resolver,
                            metric_key_for_log,
                        ).parent
                        try:
                            recover_tensor_run_transactions(
                                metric_root,
                                validation_root=resolver.tensor_root,
                                run_id=run_id,
                            )
                        except Exception as exc:  # noqa: BLE001
                            raise RuntimeError(
                                f"{runtime_plan.metric_label} worker exited "
                                f"unexpectedly (code {exit_code}); transaction "
                                f"recovery failed: {exc}"
                            ) from exc
                    runtime_results[metric_key] = _runtime_plan_failure_result(
                        svc,
                        resolver,
                        runtime_plan=runtime_plan,
                        merged_metric_params_map=merged_metric_params_map,
                        policy_n_jobs=policy_n_jobs,
                        policy_outer_n_jobs=policy_outer_n_jobs,
                        failure_message=(
                            "Unexpected runtime failure: Runtime plan worker "
                            f"exited unexpectedly (code {exit_code})."
                        ),
                    )
    finally:
        if slot_gate is not None and pending_keys:
            slot_gate.discard_pending_independent(len(pending_keys))
        for state in process_states.values():
            state["request_path"].unlink(missing_ok=True)
            state["result_path"].unlink(missing_ok=True)

    raise_if_tensor_cancellation_requested()
    return runtime_results


def execute_runtime_plans(
    svc: Any,
    resolver: Any,
    context: RecordContext,
    *,
    runtime_plans: dict[str, RuntimePlan],
    merged_metric_params_map: dict[str, dict[str, Any]],
    policy_n_jobs: int,
    policy_outer_n_jobs: int,
    global_compute_slots: int = 1,
    force_in_process: bool = False,
    process_phase_executor: ProcessPhaseExecutor | None = None,
) -> dict[str, RuntimeResult]:
    if not force_in_process:
        from .connectivity_coordinator import (
            CONNECTIVITY_RUNNER_KEYS,
            run_connectivity_coordinator,
        )

        connectivity_plans = {
            key: plan
            for key, plan in runtime_plans.items()
            if plan.runner_key in CONNECTIVITY_RUNNER_KEYS
        }
        independent_plans = {
            key: plan
            for key, plan in runtime_plans.items()
            if plan.runner_key not in CONNECTIVITY_RUNNER_KEYS
        }
        if connectivity_plans and _connectivity_fast_path_eligible(runtime_plans):
            slot_gate = ComputeSlotGate(
                global_compute_slots,
                pending_independent=len(independent_plans),
            )

            def _run_connectivity() -> dict[str, RuntimeResult]:
                return run_connectivity_coordinator(
                    svc,
                    resolver,
                    context,
                    runtime_plans=connectivity_plans,
                    global_compute_slots=global_compute_slots,
                    slot_gate=slot_gate,
                )

            if not independent_plans:
                return _run_connectivity()

            process_phase_runner = (
                process_phase_executor or _execute_runtime_plan_phase_in_subprocesses
            )

            def _run_independent() -> dict[str, RuntimeResult]:
                try:
                    result = process_phase_runner(
                        svc,
                        resolver,
                        context,
                        runtime_plans=independent_plans,
                        merged_metric_params_map=merged_metric_params_map,
                        runnable_keys=list(independent_plans),
                        policy_n_jobs=policy_n_jobs,
                        policy_outer_n_jobs=policy_outer_n_jobs,
                        global_compute_slots=global_compute_slots,
                        slot_gate=slot_gate,
                    )
                except Exception as exc:  # noqa: BLE001
                    slot_gate.fail_independent_admission(exc)
                    raise
                slot_gate.finish_independent_admission()
                return result

            with ThreadPoolExecutor(max_workers=2) as executor:
                independent_future = executor.submit(_run_independent)
                connectivity_future = executor.submit(_run_connectivity)
                combined = independent_future.result()
                combined.update(connectivity_future.result())
                return combined

    runtime_results: dict[str, RuntimeResult] = {}
    use_process_backend = (not force_in_process) and len(runtime_plans) >= 2
    process_phase_runner = (
        process_phase_executor or _execute_runtime_plan_phase_in_subprocesses
    )

    def _run_metric_plan(metric_key: str) -> RuntimeResult:
        return run_runtime_plan(
            svc,
            resolver,
            context,
            runtime_plan=runtime_plans[metric_key],
            merged_metric_params_map=merged_metric_params_map,
            policy_n_jobs=policy_n_jobs,
            policy_outer_n_jobs=policy_outer_n_jobs,
        )

    phase_ids = sorted({int(plan.phase) for plan in runtime_plans.values()})
    for phase_id in phase_ids:
        raise_if_tensor_cancellation_requested()
        phase_keys = [
            metric_key
            for metric_key, runtime_plan in runtime_plans.items()
            if int(runtime_plan.phase) == phase_id
        ]
        runnable_keys: list[str] = []
        for metric_key in phase_keys:
            runtime_plan = runtime_plans[metric_key]
            failed_dependencies = [
                dependency
                for dependency in runtime_plan.dependencies
                if not runtime_results.get(dependency, (False, "", ""))[0]
            ]
            if not failed_dependencies:
                runnable_keys.append(metric_key)
                continue

            dependency_message = "Blocked by failed dependency: " + ", ".join(
                str(item) for item in failed_dependencies
            )
            runtime_results[metric_key] = _runtime_plan_failure_result(
                svc,
                resolver,
                runtime_plan=runtime_plan,
                merged_metric_params_map=merged_metric_params_map,
                policy_n_jobs=policy_n_jobs,
                policy_outer_n_jobs=policy_outer_n_jobs,
                failure_message=dependency_message,
            )

        if not runnable_keys:
            continue
        raise_if_tensor_cancellation_requested()
        runnable_keys = [
            metric_key
            for metric_key in runnable_keys
            if runtime_plans[metric_key].plan_key in INDEPENDENT_RUNTIME_PLAN_KEYS
        ] + [
            metric_key
            for metric_key in runnable_keys
            if runtime_plans[metric_key].plan_key not in INDEPENDENT_RUNTIME_PLAN_KEYS
        ]
        process_runnable_keys = runnable_keys
        in_process_runnable_keys: list[str] = []
        if use_process_backend:
            process_runnable_keys = [
                metric_key
                for metric_key in runnable_keys
                if _runtime_plan_allows_process_backend(runtime_plans[metric_key])
            ]
            in_process_runnable_keys = [
                metric_key
                for metric_key in runnable_keys
                if metric_key not in process_runnable_keys
            ]
        if process_runnable_keys and use_process_backend:
            runtime_results.update(
                process_phase_runner(
                    svc,
                    resolver,
                    context,
                    runtime_plans=runtime_plans,
                    merged_metric_params_map=merged_metric_params_map,
                    runnable_keys=process_runnable_keys,
                    policy_n_jobs=policy_n_jobs,
                    policy_outer_n_jobs=policy_outer_n_jobs,
                    global_compute_slots=global_compute_slots,
                )
            )
        runnable_in_process = (
            in_process_runnable_keys if use_process_backend else runnable_keys
        )
        if not runnable_in_process:
            continue
        if len(runnable_in_process) >= 2:
            with ThreadPoolExecutor(
                max_workers=min(
                    max(int(global_compute_slots), 1),
                    len(runnable_in_process),
                )
            ) as executor:
                future_to_metric = {
                    executor.submit(_run_metric_plan, metric_key): metric_key
                    for metric_key in runnable_in_process
                }
                for future in as_completed(future_to_metric):
                    metric_key = future_to_metric[future]
                    runtime_results[metric_key] = future.result()
            continue
        metric_key = runnable_in_process[0]
        runtime_results[metric_key] = _run_metric_plan(metric_key)

    return runtime_results


__all__ = [
    "ComputeSlotGate",
    "RuntimePlan",
    "RuntimeResult",
    "apply_effective_parallel_policy",
    "execute_runtime_plans",
    "run_runtime_plan",
]
