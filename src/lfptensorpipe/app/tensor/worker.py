"""Child-process entrypoint for Build Tensor runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lfptensorpipe.app.path_resolver import RecordContext

from .logging import TENSOR_RUN_ID_ENV
from .orchestration import run_build_tensor


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
    context = _record_context(payload)
    selected_metrics = [str(item) for item in list(payload.get("selected_metrics", []))]
    metric_params_map = _metric_params_map(payload)
    mask_edge_effects = bool(payload.get("mask_edge_effects", True))
    try:
        ok, message = run_build_tensor(
            context,
            selected_metrics=selected_metrics,
            metric_params_map=metric_params_map,
            mask_edge_effects=mask_edge_effects,
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


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint guard
    raise SystemExit(main())
