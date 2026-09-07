"""Child-process entrypoint for Build Tensor runs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.desktop_runtime import (
    TENSOR_WORKER_FLAG,
    TENSOR_WORKER_MODULE,
    build_worker_command,
)

from .cancellation import (
    BUILD_TENSOR_CANCELLED_MESSAGE,
    TENSOR_CANCEL_REQUEST_PATH_ENV,
    backfill_cancelled_build_tensor_run,
)
from .cpu_budget import DEFAULT_TENSOR_CPU_PERCENT
from .logging import TENSOR_RUN_ID_ENV
from .orchestration import run_build_tensor
from .process_tree import BuildTensorProcessTree, build_tensor_popen_kwargs
from .transaction_manifest import recover_tensor_run_transactions

TENSOR_RUN_COOPERATIVE_TIMEOUT_S = 5.0
TENSOR_RUN_TERMINATE_TIMEOUT_S = 5.0
TENSOR_RUN_DESCENDANT_DRAIN_TIMEOUT_S = 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lfptensorpipe-build-tensor-worker")
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


def _metric_params_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_map = payload.get("metric_params_map", {})
    if not isinstance(raw_map, dict):
        raise ValueError("Worker payload `metric_params_map` must be a JSON object.")
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
    payload = _read_json(request_path)
    run_id = str(payload.get("run_id", "")).strip()
    if run_id:
        os.environ[TENSOR_RUN_ID_ENV] = run_id
    cancel_path = str(payload.get("cancel_path", "")).strip()
    if cancel_path:
        os.environ[TENSOR_CANCEL_REQUEST_PATH_ENV] = cancel_path
    context = _record_context(payload)
    selected_metrics = [str(item) for item in list(payload.get("selected_metrics", []))]
    metric_params_map = _metric_params_map(payload)
    mask_edge_effects = bool(payload.get("mask_edge_effects", True))
    cpu_percent = payload.get("cpu_percent", DEFAULT_TENSOR_CPU_PERCENT)
    try:
        ok, message = run_build_tensor(
            context,
            selected_metrics=selected_metrics,
            metric_params_map=metric_params_map,
            mask_edge_effects=mask_edge_effects,
            cpu_percent=cpu_percent,
        )
    except Exception as exc:  # noqa: BLE001
        _write_json(
            result_path,
            {
                "ok": False,
                "message": str(exc),
                "exception_type": type(exc).__name__,
            },
        )
        return 2
    _write_json(result_path, {"ok": bool(ok), "message": str(message)})
    return 0 if ok else 1


def tensor_worker_env() -> dict[str, str]:
    env = dict(os.environ)
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
    return env


def tensor_temp_json_path(stem: str) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix=f"lfptensorpipe_{stem}_", suffix=".json")
    os.close(fd)
    path = Path(raw_path)
    path.unlink(missing_ok=True)
    return path


def write_tensor_worker_request(payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    request_path = tensor_temp_json_path("tensor_request")
    result_path = tensor_temp_json_path("tensor_result")
    cancel_path = tensor_temp_json_path("tensor_cancel")
    payload = dict(payload)
    payload["cancel_path"] = str(cancel_path)
    request_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return request_path, result_path, cancel_path


def read_tensor_worker_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def run_tensor_subprocess(
    context: RecordContext,
    *,
    selected_metrics: list[str],
    metric_params_map: dict[str, dict[str, Any]],
    mask_edge_effects: bool,
    cpu_percent: float,
) -> tuple[bool, str]:
    """Run the existing worker synchronously and own its descendants until exit."""
    run_id = uuid4().hex
    request_path, result_path, cancel_path = write_tensor_worker_request(
        {
            "context": {
                "project_root": str(context.project_root),
                "subject": context.subject,
                "record": context.record,
            },
            "selected_metrics": selected_metrics,
            "metric_params_map": metric_params_map,
            "mask_edge_effects": mask_edge_effects,
            "cpu_percent": cpu_percent,
            "run_id": run_id,
        }
    )
    process = None
    tree = None
    recovered = False
    cancelled = False
    stop_message = ""
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        process = subprocess.Popen(
            build_worker_command(
                module_name=TENSOR_WORKER_MODULE,
                embedded_flag=TENSOR_WORKER_FLAG,
                worker_args=[
                    "--request",
                    str(request_path),
                    "--result",
                    str(result_path),
                ],
                python_exec=sys.executable,
            ),
            env=tensor_worker_env(),
            stdin=subprocess.DEVNULL,
            **build_tensor_popen_kwargs(),
        )
        try:
            tree = BuildTensorProcessTree.attach(process)
        except BaseException:
            # Attachment can fail after launch; reap the leader we already own.
            process.kill()
            process.wait(timeout=TENSOR_RUN_TERMINATE_TIMEOUT_S)
            raise
        try:
            process.wait()
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            cancelled = True
            stop_message = BUILD_TENSOR_CANCELLED_MESSAGE
            cancel_path.write_text(run_id, encoding="utf-8")
            tree.wait_for_quiescence(TENSOR_RUN_COOPERATIVE_TIMEOUT_S)
        finally:
            # Repeated Ctrl+C must not interrupt owned-process recovery.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if not tree.wait_for_quiescence(TENSOR_RUN_DESCENDANT_DRAIN_TIMEOUT_S):
                stop_message = (
                    stop_message
                    or "Build Tensor worker exited with active descendants."
                )
                tree.terminate()
                if not tree.wait_for_quiescence(TENSOR_RUN_TERMINATE_TIMEOUT_S):
                    tree.force_kill()
                if not tree.wait_for_quiescence(TENSOR_RUN_DESCENDANT_DRAIN_TIMEOUT_S):
                    raise RuntimeError(
                        f"Build Tensor descendants are still active; recovery request retained at {request_path}."
                    )
            process.wait(timeout=0.0)
            tensor_root = PathResolver(context).tensor_root
            recover_tensor_run_transactions(
                tensor_root, validation_root=tensor_root, run_id=run_id
            )
            result = read_tensor_worker_result(result_path)
            if result is None:
                stop_message = (
                    stop_message
                    or f"Build Tensor worker exited without a result (code {process.returncode})."
                )
            if stop_message:
                backfill_cancelled_build_tensor_run(
                    context,
                    selected_metrics=selected_metrics,
                    metric_params_map=metric_params_map,
                    mask_edge_effects=mask_edge_effects,
                    run_id=run_id,
                    cpu_percent=cpu_percent,
                    message=stop_message,
                )
            recovered = True
        if cancelled:
            raise KeyboardInterrupt
        if stop_message:
            return False, stop_message
        return process.returncode == 0 and result.get("ok") is True, str(
            result.get("message") or "Build Tensor worker exited without a message."
        )
    finally:
        try:
            if tree is not None:
                tree.close()
            if recovered or tree is None:
                for path in (request_path, result_path, cancel_path):
                    path.unlink(missing_ok=True)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(main())
