"""Preprocess-stage artifact and log lifecycle helpers."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import RunLogRecord, read_run_log, write_run_log
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_preproc_result_change,
)
from .paths import (
    preproc_step_config_path as _preproc_step_config_path_impl,
    preproc_step_log_path as _preproc_step_log_path_impl,
    preproc_step_raw_path as _preproc_step_raw_path_impl,
    rawdata_input_fif_path as _rawdata_input_fif_path_impl,
    write_preproc_step_config as _write_preproc_step_config_impl,
)
from .steps.annotations import (
    _normalize_annotation_rows as _normalize_annotation_rows_impl,
    apply_annotations_step as _apply_annotations_step_impl,
    load_annotations_csv_rows as _load_annotations_csv_rows_impl,
)
from .steps.bad_segment import apply_bad_segment_step as _apply_bad_segment_step_impl
from .steps.ecg import apply_ecg_step as _apply_ecg_step_impl
from .steps.finish import (
    apply_finish_step as _apply_finish_step_impl,
    resolve_finish_source as _resolve_finish_source_impl,
)
from .steps.filter import (
    _normalize_notch_widths as _normalize_notch_widths_impl,
    apply_filter_step as _apply_filter_step_impl,
    default_filter_advance_params as _default_filter_advance_params_impl,
    normalize_filter_advance_params as _normalize_filter_advance_params_impl,
)
from .steps.raw import bootstrap_raw_step_from_rawdata as _bootstrap_raw_step_impl
from .indicator import (
    preproc_annotations_panel_state as _preproc_annotations_panel_state_impl,
    preproc_ecg_panel_state as _preproc_ecg_panel_state_impl,
    preproc_filter_panel_state as _preproc_filter_panel_state_impl,
)

PREPROC_STEPS = (
    "raw",
    "filter",
    "annotations",
    "bad_segment_removal",
    "ecg_artifact_removal",
    "finish",
)

FINISH_SOURCE_PRIORITY = (
    "ecg_artifact_removal",
    "bad_segment_removal",
    "annotations",
    "filter",
    "raw",
)

ECG_METHODS = ("template", "perceive", "svd")


def default_filter_advance_params() -> dict[str, Any]:
    return _default_filter_advance_params_impl()


def _normalize_notch_widths(value: Any) -> float | list[float]:
    return _normalize_notch_widths_impl(value)


def normalize_filter_advance_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], str]:
    return _normalize_filter_advance_params_impl(params)


def rawdata_input_fif_path(context: RecordContext) -> Path:
    return _rawdata_input_fif_path_impl(context)


def preproc_step_raw_path(resolver: PathResolver, step: str) -> Path:
    return _preproc_step_raw_path_impl(resolver, step)


def preproc_step_log_path(resolver: PathResolver, step: str) -> Path:
    return _preproc_step_log_path_impl(resolver, step)


def preproc_step_config_path(resolver: PathResolver, step: str) -> Path:
    return _preproc_step_config_path_impl(resolver, step)


def write_preproc_step_config(
    *,
    resolver: PathResolver,
    step: str,
    config: dict[str, Any],
) -> Path:
    return _write_preproc_step_config_impl(resolver=resolver, step=step, config=config)


def mark_preproc_step(
    *,
    resolver: PathResolver,
    step: str,
    completed: bool,
    params: dict[str, Any] | None = None,
    input_path: str = "",
    output_path: str = "",
    message: str = "",
) -> Path:
    """Write one preprocess step log with schema-compliant payload."""
    log_path = preproc_step_log_path(resolver, step)
    return write_run_log(
        log_path,
        RunLogRecord(
            step=step,
            completed=completed,
            params=params or {},
            input_path=input_path,
            output_path=output_path,
            message=message,
        ),
    )


def bootstrap_raw_step_from_rawdata(context: RecordContext) -> tuple[bool, str]:
    return _bootstrap_raw_step_impl(
        context,
        rawdata_input_fif_path_fn=rawdata_input_fif_path,
        preproc_step_raw_path_fn=preproc_step_raw_path,
        mark_preproc_step_fn=mark_preproc_step,
    )


def invalidate_downstream_preproc_steps(
    context: RecordContext, changed_step: str
) -> list[Path]:
    """Rewrite downstream preprocess logs with `completed=false`."""
    resolver = PathResolver(context)
    if changed_step not in PREPROC_STEPS:
        raise ValueError(f"Unknown preprocess step: {changed_step}")
    changed_index = PREPROC_STEPS.index(changed_step)
    rewritten: list[Path] = []
    for step in PREPROC_STEPS[changed_index + 1 :]:
        log_path = mark_preproc_step(
            resolver=resolver,
            step=step,
            completed=False,
            input_path=str(preproc_step_raw_path(resolver, changed_step)),
            output_path=str(preproc_step_raw_path(resolver, step)),
            message=f"Invalidated by upstream step re-apply: {changed_step}",
        )
        rewritten.append(log_path)
    return rewritten


def resolve_finish_source(
    context: RecordContext,
    *,
    read_run_log_fn: Any | None = None,
) -> tuple[str, Path] | None:
    return _resolve_finish_source_impl(
        context,
        source_priority=FINISH_SOURCE_PRIORITY,
        preproc_step_raw_path_fn=preproc_step_raw_path,
        preproc_step_log_path_fn=preproc_step_log_path,
        read_run_log_fn=read_run_log_fn or read_run_log,
    )


def apply_finish_step(context: RecordContext) -> tuple[bool, str]:
    ok, message = _apply_finish_step_impl(
        context,
        resolve_finish_source_fn=resolve_finish_source,
        preproc_step_raw_path_fn=preproc_step_raw_path,
        mark_preproc_step_fn=mark_preproc_step,
    )
    if ok:
        invalidate_after_preproc_result_change(context, changed_step="finish")
    return ok, message


def apply_filter_step(
    context: RecordContext,
    *,
    advance_params: dict[str, Any] | None = None,
    notches: list[float] | tuple[float, ...] | None = None,
    l_freq: float | None = None,
    h_freq: float | None = None,
    thread_module: Any = threading,
    read_raw_fif_fn: Any | None = None,
    mark_lfp_bad_segments_fn: Any | None = None,
) -> tuple[bool, str]:
    ok, message = _apply_filter_step_impl(
        context,
        advance_params=advance_params,
        notches=notches,
        l_freq=l_freq,
        h_freq=h_freq,
        mark_preproc_step_fn=mark_preproc_step,
        invalidate_downstream_fn=invalidate_downstream_preproc_steps,
        thread_module=thread_module,
        read_raw_fif_fn=read_raw_fif_fn,
        mark_lfp_bad_segments_fn=mark_lfp_bad_segments_fn,
    )
    if ok:
        invalidate_after_preproc_result_change(context, changed_step="filter")
    return ok, message


def apply_bad_segment_step(
    context: RecordContext,
    *,
    read_raw_fif_fn: Any | None = None,
    filter_lfp_with_bad_annotations_fn: Any | None = None,
    add_head_tail_annotations_fn: Any | None = None,
) -> tuple[bool, str]:
    ok, message = _apply_bad_segment_step_impl(
        context,
        mark_preproc_step_fn=mark_preproc_step,
        invalidate_downstream_fn=invalidate_downstream_preproc_steps,
        read_raw_fif_fn=read_raw_fif_fn,
        filter_lfp_with_bad_annotations_fn=filter_lfp_with_bad_annotations_fn,
        add_head_tail_annotations_fn=add_head_tail_annotations_fn,
    )
    if ok:
        invalidate_after_preproc_result_change(
            context, changed_step="bad_segment_removal"
        )
    return ok, message


def apply_ecg_step(
    context: RecordContext,
    *,
    method: str = "svd",
    picks: list[str] | tuple[str, ...] | None = None,
    read_raw_fif_fn: Any | None = None,
    raw_call_ecgremover_fn: Any | None = None,
) -> tuple[bool, str]:
    ok, message = _apply_ecg_step_impl(
        context,
        method=method,
        picks=picks,
        ecg_methods=ECG_METHODS,
        mark_preproc_step_fn=mark_preproc_step,
        invalidate_downstream_fn=invalidate_downstream_preproc_steps,
        read_raw_fif_fn=read_raw_fif_fn,
        raw_call_ecgremover_fn=raw_call_ecgremover_fn,
    )
    if ok:
        invalidate_after_preproc_result_change(
            context, changed_step="ecg_artifact_removal"
        )
    return ok, message


def load_annotations_csv_rows(csv_path: Path) -> tuple[bool, list[dict[str, Any]], str]:
    return _load_annotations_csv_rows_impl(csv_path)


def _normalize_annotation_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    return _normalize_annotation_rows_impl(rows)


def apply_annotations_step(
    context: RecordContext,
    *,
    rows: list[dict[str, Any]],
    read_raw_fif_fn: Any | None = None,
    copy2_fn: Any | None = None,
) -> tuple[bool, str]:
    ok, message = _apply_annotations_step_impl(
        context,
        rows=rows,
        mark_preproc_step_fn=mark_preproc_step,
        invalidate_downstream_fn=invalidate_downstream_preproc_steps,
        read_raw_fif_fn=read_raw_fif_fn,
        copy2_fn=copy2_fn,
    )
    if ok:
        invalidate_after_preproc_result_change(context, changed_step="annotations")
    return ok, message


def preproc_filter_panel_state(
    resolver: PathResolver,
    *,
    notches: Any,
    l_freq: Any,
    h_freq: Any,
    advance_params: dict[str, Any] | None,
) -> str:
    return _preproc_filter_panel_state_impl(
        resolver,
        notches=notches,
        l_freq=l_freq,
        h_freq=h_freq,
        advance_params=advance_params,
    )


def preproc_annotations_panel_state(
    resolver: PathResolver,
    *,
    rows: list[dict[str, Any]],
) -> str:
    return _preproc_annotations_panel_state_impl(resolver, rows=rows)


def preproc_ecg_panel_state(
    resolver: PathResolver,
    *,
    method: Any,
    picks: Any,
) -> str:
    return _preproc_ecg_panel_state_impl(resolver, method=method, picks=picks)
