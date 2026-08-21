"""Accepted-generation lineage for Build Tensor metric artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
from typing import Any, Concatenate, Mapping, ParamSpec, TypeVar, cast
from uuid import uuid4

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc.lineage import preproc_step_lineage_is_current
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.shared.atomic_outputs import OUTPUT_TRANSACTION_RUN_ID_ENV
from lfptensorpipe.app.shared.generation_lineage import (
    accepted_result_generation_id,
    input_generation_receipts_match,
    preproc_generation_ref,
)

from .paths import tensor_metric_log_path, tensor_metric_tensor_path

TENSOR_INPUT_LINEAGE_ENV = "LFPTENSORPIPE_TENSOR_INPUT_LINEAGE"
TENSOR_CURRENT_FINISH_REQUIRED_MESSAGE = (
    "Preprocess finish must be current before running a Tensor metric."
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_run_log(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _finish_log_path(resolver: PathResolver) -> Path:
    return resolver.preproc_step_dir("finish", create=False) / "lfptensorpipe_log.json"


def capture_tensor_input_generation(
    resolver: PathResolver,
) -> dict[str, str | None] | None:
    """Capture the current Preprocess Finish generation for one Build run."""
    if not preproc_step_lineage_is_current(resolver, "finish"):
        return None
    finish_payload = _read_payload(_finish_log_path(resolver))
    return {
        preproc_generation_ref("finish"): accepted_result_generation_id(finish_payload)
    }


def tensor_input_generation_matches(
    resolver: PathResolver,
    expected: Mapping[str, str | None],
) -> bool:
    """Return whether the captured Build input is still current."""
    current = capture_tensor_input_generation(resolver)
    return current is not None and dict(current) == dict(expected)


def serialize_tensor_input_lineage(
    context: RecordContext,
    input_generations: Mapping[str, str | None],
) -> str:
    """Serialize one parent-captured Build input snapshot for worker processes."""
    return json.dumps(
        {
            "context": {
                "project_root": str(context.project_root),
                "subject": context.subject,
                "record": context.record,
            },
            "input_generations": dict(input_generations),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@contextmanager
def tensor_runner_input_lineage(
    context: RecordContext,
    *,
    require_run_id: bool = False,
) -> Iterator[bool]:
    """Install one direct runner's Finish receipt before its first input read."""
    prior_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV)
    installed_lineage = not (prior_payload is not None and prior_payload.strip())
    if installed_lineage:
        input_generations = capture_tensor_input_generation(PathResolver(context))
        if input_generations is None:
            yield False
            return
        os.environ[TENSOR_INPUT_LINEAGE_ENV] = serialize_tensor_input_lineage(
            context,
            input_generations,
        )

    prior_run_id = os.environ.get(OUTPUT_TRANSACTION_RUN_ID_ENV)
    installed_run_id = require_run_id and not str(prior_run_id or "").strip()
    if installed_run_id:
        os.environ[OUTPUT_TRANSACTION_RUN_ID_ENV] = uuid4().hex
    try:
        yield True
    finally:
        if installed_run_id:
            if prior_run_id is None:
                os.environ.pop(OUTPUT_TRANSACTION_RUN_ID_ENV, None)
            else:
                os.environ[OUTPUT_TRANSACTION_RUN_ID_ENV] = prior_run_id
        if installed_lineage:
            if prior_payload is None:
                os.environ.pop(TENSOR_INPUT_LINEAGE_ENV, None)
            else:
                os.environ[TENSOR_INPUT_LINEAGE_ENV] = prior_payload


def tensor_runner_entry(
    runner: Callable[Concatenate[RecordContext, _P], _R],
) -> Callable[Concatenate[RecordContext, _P], _R]:
    """Wrap one public Tensor runner with direct-call Finish lineage ownership."""

    @wraps(runner)
    def wrapped(
        context: RecordContext,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        with tensor_runner_input_lineage(context) as ready:
            if not ready:
                return cast(
                    _R,
                    (False, TENSOR_CURRENT_FINISH_REQUIRED_MESSAGE),
                )
            return runner(context, *args, **kwargs)

    return cast(Callable[Concatenate[RecordContext, _P], _R], wrapped)


def trgc_runner_entry(
    runner: Callable[Concatenate[RecordContext, _P], _R],
) -> Callable[Concatenate[RecordContext, _P], _R]:
    """Wrap one TRGC runner with Finish lineage and non-empty run identity."""

    @wraps(runner)
    def wrapped(
        context: RecordContext,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R:
        with tensor_runner_input_lineage(context, require_run_id=True) as ready:
            if not ready:
                return cast(
                    _R,
                    (False, TENSOR_CURRENT_FINISH_REQUIRED_MESSAGE),
                )
            return runner(context, *args, **kwargs)

    return cast(Callable[Concatenate[RecordContext, _P], _R], wrapped)


def _decode_tensor_input_lineage(
    raw_payload: str,
) -> tuple[RecordContext, dict[str, str | None]]:
    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Build Tensor input lineage payload is invalid.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Build Tensor input lineage payload is invalid.")
    context_payload = payload.get("context")
    input_payload = payload.get("input_generations")
    if not isinstance(context_payload, dict) or not isinstance(input_payload, dict):
        raise RuntimeError("Build Tensor input lineage payload is invalid.")
    project_root = str(context_payload.get("project_root", "")).strip()
    subject = str(context_payload.get("subject", "")).strip()
    record = str(context_payload.get("record", "")).strip()
    if not project_root or not subject or not record:
        raise RuntimeError("Build Tensor input lineage payload is invalid.")
    normalized: dict[str, str | None] = {}
    for raw_ref, raw_generation_id in input_payload.items():
        ref = str(raw_ref).strip()
        if not ref or (
            raw_generation_id is not None and not isinstance(raw_generation_id, str)
        ):
            raise RuntimeError("Build Tensor input lineage payload is invalid.")
        normalized[ref] = raw_generation_id
    return (
        RecordContext(
            project_root=Path(project_root),
            subject=subject,
            record=record,
        ),
        normalized,
    )


def tensor_input_generations_from_environment() -> dict[str, str | None] | None:
    """Return the parent-captured Build receipt available to this worker."""
    raw_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV, "").strip()
    if not raw_payload:
        return None
    _context, input_generations = _decode_tensor_input_lineage(raw_payload)
    return input_generations


def require_tensor_input_generation_unchanged() -> None:
    """Abort candidate promotion when the Build input changed during compute."""
    raw_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV, "").strip()
    if not raw_payload:
        return
    context, expected = _decode_tensor_input_lineage(raw_payload)
    if not tensor_input_generation_matches(PathResolver(context), expected):
        raise RuntimeError(
            "Preprocess Finish changed while Build Tensor was running; "
            "candidate outputs were not accepted."
        )


def tensor_input_generation_changed_during_run() -> bool:
    """Return whether an active Build snapshot no longer matches Finish."""
    raw_payload = os.environ.get(TENSOR_INPUT_LINEAGE_ENV, "").strip()
    if not raw_payload:
        return False
    context, expected = _decode_tensor_input_lineage(raw_payload)
    return not tensor_input_generation_matches(PathResolver(context), expected)


def tensor_metric_lineage_is_current(
    resolver: PathResolver,
    metric_key: str,
) -> bool:
    """Return whether one persisted Tensor metric matches current Finish."""
    if not preproc_step_lineage_is_current(resolver, "finish"):
        return False
    log_path = tensor_metric_log_path(resolver, metric_key, create=False)
    artifact_path = tensor_metric_tensor_path(resolver, metric_key, create=False)
    payload = _read_payload(log_path)
    if (
        payload is None
        or payload.get("completed") is not True
        or not artifact_path.is_file()
    ):
        return False
    finish_payload = _read_payload(_finish_log_path(resolver))
    return input_generation_receipts_match(
        payload,
        expected={
            preproc_generation_ref("finish"): accepted_result_generation_id(
                finish_payload
            )
        },
    )


def tensor_metric_result_generation_id(
    resolver: PathResolver,
    metric_key: str,
) -> str | None:
    """Return a Tensor result ID only when the metric is recursively current."""
    if not tensor_metric_lineage_is_current(resolver, metric_key):
        return None
    return accepted_result_generation_id(
        _read_payload(tensor_metric_log_path(resolver, metric_key, create=False))
    )


__all__ = [
    "TENSOR_INPUT_LINEAGE_ENV",
    "TENSOR_CURRENT_FINISH_REQUIRED_MESSAGE",
    "capture_tensor_input_generation",
    "require_tensor_input_generation_unchanged",
    "serialize_tensor_input_lineage",
    "tensor_runner_entry",
    "tensor_runner_input_lineage",
    "trgc_runner_entry",
    "tensor_input_generation_matches",
    "tensor_input_generation_changed_during_run",
    "tensor_input_generations_from_environment",
    "tensor_metric_lineage_is_current",
    "tensor_metric_result_generation_id",
]
