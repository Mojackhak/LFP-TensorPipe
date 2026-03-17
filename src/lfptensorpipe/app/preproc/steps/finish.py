"""Finish-step source selection and apply helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

ReadRunLogFn = Callable[[Path], dict[str, Any] | None]
MarkStepFn = Callable[..., Any]


def resolve_finish_source(
    context: RecordContext,
    *,
    source_priority: tuple[str, ...],
    preproc_step_raw_path_fn: Callable[[PathResolver, str], Path],
    preproc_step_log_path_fn: Callable[[PathResolver, str], Path],
    read_run_log_fn: ReadRunLogFn,
) -> tuple[str, Path] | None:
    """Resolve highest-priority valid preprocess source for `finish`."""
    resolver = PathResolver(context)
    for step in source_priority:
        raw_path = preproc_step_raw_path_fn(resolver, step)
        log_path = preproc_step_log_path_fn(resolver, step)
        if not raw_path.exists() or not log_path.exists():
            continue
        payload = read_run_log_fn(log_path)
        if payload is None:
            continue
        if bool(payload.get("completed")):
            return step, raw_path
    return None


def apply_finish_step(
    context: RecordContext,
    *,
    resolve_finish_source_fn: Callable[[RecordContext], tuple[str, Path] | None],
    preproc_step_raw_path_fn: Callable[[PathResolver, str], Path],
    mark_preproc_step_fn: MarkStepFn,
) -> tuple[bool, str]:
    """Apply preprocess finish step by source-priority copy."""
    resolver = PathResolver(context)
    source = resolve_finish_source_fn(context)
    finish_raw_path = preproc_step_raw_path_fn(resolver, "finish")

    if source is None:
        mark_preproc_step_fn(
            resolver=resolver,
            step="finish",
            completed=False,
            input_path="",
            output_path=str(finish_raw_path),
            message="No valid source step available for finish.",
        )
        return False, "No valid source step available."

    source_step, source_path = source
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, finish_raw_path)
    mark_preproc_step_fn(
        resolver=resolver,
        step="finish",
        completed=True,
        params={"source_step": source_step},
        input_path=str(source_path),
        output_path=str(finish_raw_path),
        message=f"Finish raw copied from {source_step}.",
    )
    return True, f"Finish step completed using source: {source_step}."
