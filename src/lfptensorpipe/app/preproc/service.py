"""Preprocess-stage artifact and log lifecycle helpers."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import RunLogRecord, read_run_log, write_run_log
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_preproc_result_change,
)
from .paths import (
    preproc_filter_preview_raw_path as _preproc_filter_preview_raw_path_impl,
    preproc_step_config_path as _preproc_step_config_path_impl,
    preproc_step_log_path as _preproc_step_log_path_impl,
    preproc_step_raw_path as _preproc_step_raw_path_impl,
    preproc_step_routing_path as _preproc_step_routing_path_impl,
    rawdata_input_fif_path as _rawdata_input_fif_path_impl,
    write_preproc_step_routing as _write_preproc_step_routing_impl,
    write_preproc_step_config as _write_preproc_step_config_impl,
)
from .steps.annotations import (
    _normalize_annotation_rows as _normalize_annotation_rows_impl,
    apply_annotations_step as _apply_annotations_step_impl,
    load_annotations_csv_rows as _load_annotations_csv_rows_impl,
)
from .steps.ecg import (
    apply_ecg_step as _apply_ecg_step_impl,
    default_ecg_method_params as _default_ecg_method_params_impl,
    default_ecg_params_by_method as _default_ecg_params_by_method_impl,
    default_ecg_review_params as _default_ecg_review_params_impl,
    ecg_method_runtime_kwargs as _ecg_method_runtime_kwargs_impl,
    normalize_ecg_method_params as _normalize_ecg_method_params_impl,
    normalize_ecg_params_by_method as _normalize_ecg_params_by_method_impl,
    normalize_ecg_review_params as _normalize_ecg_review_params_impl,
)
from .steps.finish import apply_finish_step as _apply_finish_step_impl
from .steps.filter import (
    _normalize_notch_widths as _normalize_notch_widths_impl,
    apply_filter_step as _apply_filter_step_impl,
    finalize_filter_review as _finalize_filter_review_impl,
    default_filter_advance_params as _default_filter_advance_params_impl,
    filter_nyquist_warning as _filter_nyquist_warning_impl,
    normalize_filter_advance_params as _normalize_filter_advance_params_impl,
    normalize_filter_runtime_params as _normalize_filter_runtime_params_impl,
)
from .steps.raw import bootstrap_raw_step_from_rawdata as _bootstrap_raw_step_impl
from .indicator import (
    preproc_annotations_panel_state as _preproc_annotations_panel_state_impl,
    preproc_ecg_panel_state as _preproc_ecg_panel_state_impl,
    preproc_filter_panel_state as _preproc_filter_panel_state_impl,
    preproc_filter_review_required as _preproc_filter_review_required_impl,
    preproc_step_indicator_state as _preproc_step_indicator_state_impl,
)
from .lineage import capture_preproc_input_generation, preproc_step_is_skipped

logger = logging.getLogger(__name__)

PREPROC_STEPS = (
    "raw",
    "filter",
    "ecg_artifact_removal",
    "annotations",
    "finish",
)

OPTIONAL_PREPROC_STEPS = (
    "filter",
    "ecg_artifact_removal",
    "annotations",
)

FINISH_SOURCE_PRIORITY = (
    "annotations",
    "ecg_artifact_removal",
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


def normalize_filter_runtime_params(
    *,
    notches: Any,
    l_freq: Any,
    h_freq: Any,
) -> tuple[bool, dict[str, Any], str]:
    return _normalize_filter_runtime_params_impl(
        notches=notches,
        l_freq=l_freq,
        h_freq=h_freq,
    )


def filter_nyquist_warning(
    *,
    sfreq_hz: Any,
    notches: list[float] | tuple[float, ...],
    h_freq: float | None,
) -> str:
    return _filter_nyquist_warning_impl(
        sfreq_hz=sfreq_hz,
        notches=notches,
        h_freq=h_freq,
    )


def default_ecg_method_params(method: str) -> dict[str, Any]:
    return _default_ecg_method_params_impl(method)


def default_ecg_params_by_method() -> dict[str, dict[str, Any]]:
    return _default_ecg_params_by_method_impl()


def default_ecg_review_params() -> dict[str, bool]:
    return _default_ecg_review_params_impl()


def normalize_ecg_method_params(
    method: str,
    params: dict[str, Any] | None,
    *,
    base_params: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any], str]:
    return _normalize_ecg_method_params_impl(
        method,
        params,
        base_params=base_params,
    )


def normalize_ecg_params_by_method(
    params_by_method: Any,
    *,
    base_by_method: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, dict[str, dict[str, Any]], str]:
    return _normalize_ecg_params_by_method_impl(
        params_by_method,
        base_by_method=base_by_method,
    )


def normalize_ecg_review_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, bool], str]:
    return _normalize_ecg_review_params_impl(params)


def ecg_method_runtime_kwargs(
    method: str,
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], dict[str, Any], str]:
    return _ecg_method_runtime_kwargs_impl(method, params)


def rawdata_input_fif_path(context: RecordContext) -> Path:
    return _rawdata_input_fif_path_impl(context)


def preproc_step_raw_path(resolver: PathResolver, step: str) -> Path:
    return _preproc_step_raw_path_impl(resolver, step)


def preproc_step_log_path(resolver: PathResolver, step: str) -> Path:
    return _preproc_step_log_path_impl(resolver, step)


def preproc_step_config_path(resolver: PathResolver, step: str) -> Path:
    return _preproc_step_config_path_impl(resolver, step)


def preproc_step_routing_path(resolver: PathResolver, step: str) -> Path:
    return _preproc_step_routing_path_impl(resolver, step)


def preproc_filter_preview_raw_path(resolver: PathResolver) -> Path:
    return _preproc_filter_preview_raw_path_impl(resolver)


def write_preproc_step_config(
    *,
    resolver: PathResolver,
    step: str,
    config: dict[str, Any],
    path: Path | None = None,
) -> Path:
    return _write_preproc_step_config_impl(
        resolver=resolver,
        step=step,
        config=config,
        path=path,
    )


def mark_preproc_step(
    *,
    resolver: PathResolver,
    step: str,
    completed: bool,
    params: dict[str, Any] | None = None,
    input_path: str = "",
    output_path: str = "",
    message: str = "",
    log_path: Path | None = None,
) -> Path:
    """Write one preprocess step log with schema-compliant payload."""
    destination = log_path or preproc_step_log_path(resolver, step)
    return write_run_log(
        destination,
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
    ok, message = _bootstrap_raw_step_impl(
        context,
        rawdata_input_fif_path_fn=rawdata_input_fif_path,
        preproc_step_raw_path_fn=preproc_step_raw_path,
        mark_preproc_step_fn=mark_preproc_step,
    )
    if ok:
        invalidate_downstream_preproc_steps(context, "raw")
    return ok, message


def invalidate_downstream_preproc_steps(
    context: RecordContext, changed_step: str
) -> list[Path]:
    """Invalidate only previously successful downstream preprocess steps."""
    resolver = PathResolver(context)
    if changed_step not in PREPROC_STEPS:
        raise ValueError(f"Unknown preprocess step: {changed_step}")
    changed_index = PREPROC_STEPS.index(changed_step)
    rewritten: list[Path] = []
    for step in PREPROC_STEPS[changed_index + 1 :]:
        log_path = (
            resolver.preproc_step_dir(step, create=False) / "lfptensorpipe_log.json"
        )
        try:
            existing = read_run_log(log_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not inspect downstream log %s: %s", log_path, exc)
            continue
        if existing is None or not bool(existing.get("completed")):
            continue
        try:
            rewritten_path = mark_preproc_step(
                resolver=resolver,
                step=step,
                completed=False,
                input_path=str(
                    resolver.preproc_step_dir(changed_step, create=False) / "raw.fif"
                ),
                output_path=str(
                    resolver.preproc_step_dir(step, create=False) / "raw.fif"
                ),
                message=f"Invalidated by upstream step re-apply: {changed_step}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not invalidate downstream log %s: %s", log_path, exc)
            continue
        rewritten.append(rewritten_path)
    rewritten.extend(
        invalidate_after_preproc_result_change(
            context,
            changed_step=changed_step,
        )
    )
    return rewritten


def resolve_finish_source(
    context: RecordContext,
    *,
    read_run_log_fn: Any | None = None,
) -> tuple[str, Path] | None:
    return resolve_preproc_step_source(
        context,
        "finish",
        read_run_log_fn=read_run_log_fn,
    )


def resolve_preproc_step_source(
    context: RecordContext,
    target_step: str,
    *,
    read_run_log_fn: Any | None = None,
) -> tuple[str, Path] | None:
    """Resolve the nearest valid artifact before one preprocess target step."""
    if target_step not in PREPROC_STEPS:
        raise ValueError(f"Unknown preprocess step: {target_step}")
    target_index = PREPROC_STEPS.index(target_step)
    if target_index == 0:
        raise ValueError("Raw does not have a preprocess source step.")

    resolver = PathResolver(context)
    captured = capture_preproc_input_generation(resolver, target_step)
    if captured is None:
        return None
    source_step, _ = captured
    source_path = preproc_step_raw_path(resolver, source_step)
    if read_run_log_fn is not None:
        payload = read_run_log_fn(preproc_step_log_path(resolver, source_step))
        if not isinstance(payload, dict) or payload.get("completed") is not True:
            return None
    return source_step, source_path


def skip_preproc_step(context: RecordContext, step: str) -> tuple[bool, str]:
    """Toggle independent routing around one optional Preprocess step."""
    if step not in OPTIONAL_PREPROC_STEPS:
        raise ValueError(f"Preprocess step cannot be skipped: {step}")
    resolver = PathResolver(context)
    currently_skipped = preproc_step_is_skipped(resolver, step)
    if not currently_skipped:
        preview_pending = step == "filter" and preproc_filter_review_required(resolver)
        if (
            _preproc_step_indicator_state_impl(resolver, step) == "gray"
            and not preview_pending
        ):
            return False, f"{step} has no run state to skip."
        if capture_preproc_input_generation(resolver, step) is None:
            return (
                False,
                f"{step} cannot be skipped while an earlier step is unresolved.",
            )

    new_skipped = not currently_skipped
    _write_preproc_step_routing_impl(
        resolver=resolver,
        step=step,
        skipped=new_skipped,
    )
    invalidate_downstream_preproc_steps(context, step)
    action = "Skipped" if new_skipped else "Restored"
    return True, f"{action} preprocess step routing: {step}."


def _clear_preproc_step_skip(context: RecordContext, step: str) -> bool:
    resolver = PathResolver(context)
    if not preproc_step_is_skipped(resolver, step):
        return False
    _write_preproc_step_routing_impl(
        resolver=resolver,
        step=step,
        skipped=False,
    )
    return True


def apply_finish_step(
    context: RecordContext,
    *,
    read_raw_fif_fn: Any | None = None,
    add_head_tail_annotations_fn: Any | None = None,
) -> tuple[bool, str]:
    ok, message = _apply_finish_step_impl(
        context,
        resolve_finish_source_fn=resolve_finish_source,
        preproc_step_raw_path_fn=preproc_step_raw_path,
        mark_preproc_step_fn=mark_preproc_step,
        read_raw_fif_fn=read_raw_fif_fn,
        add_head_tail_annotations_fn=add_head_tail_annotations_fn,
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
        thread_module=thread_module,
        read_raw_fif_fn=read_raw_fif_fn,
        mark_lfp_bad_segments_fn=mark_lfp_bad_segments_fn,
    )
    if ok and _clear_preproc_step_skip(context, "filter"):
        invalidate_downstream_preproc_steps(context, "filter")
    return ok, message


def finalize_filter_review(
    context: RecordContext,
    *,
    reviewed_annotations: Any,
    reviewed_bads: list[str] | tuple[str, ...],
    read_raw_fif_fn: Any | None = None,
    finalize_reviewed_filter_fn: Any | None = None,
    review_source_is_current_fn: Any | None = None,
) -> tuple[bool, str]:
    ok, message = _finalize_filter_review_impl(
        context,
        reviewed_annotations=reviewed_annotations,
        reviewed_bads=reviewed_bads,
        mark_preproc_step_fn=mark_preproc_step,
        invalidate_downstream_fn=invalidate_downstream_preproc_steps,
        read_raw_fif_fn=read_raw_fif_fn,
        finalize_reviewed_filter_fn=finalize_reviewed_filter_fn,
        review_source_is_current_fn=review_source_is_current_fn,
    )
    if ok:
        _clear_preproc_step_skip(context, "filter")
    return ok, message


def apply_ecg_step(
    context: RecordContext,
    *,
    method: str = "svd",
    picks: list[str] | tuple[str, ...] | None = None,
    method_kwargs: dict[str, Any] | None = None,
    mark_filter_edges: bool = False,
    read_raw_fif_fn: Any | None = None,
    raw_call_ecgremover_fn: Any | None = None,
) -> tuple[bool, str]:
    ok, message = _apply_ecg_step_impl(
        context,
        source=resolve_preproc_step_source(context, "ecg_artifact_removal"),
        method=method,
        picks=picks,
        method_kwargs=method_kwargs,
        mark_filter_edges=mark_filter_edges,
        ecg_methods=ECG_METHODS,
        mark_preproc_step_fn=mark_preproc_step,
        invalidate_downstream_fn=invalidate_downstream_preproc_steps,
        read_raw_fif_fn=read_raw_fif_fn,
        raw_call_ecgremover_fn=raw_call_ecgremover_fn,
    )
    if ok:
        _clear_preproc_step_skip(context, "ecg_artifact_removal")
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
    mark_filter_edges: bool = False,
    read_raw_fif_fn: Any | None = None,
    copy2_fn: Any | None = None,
) -> tuple[bool, str]:
    ok, message = _apply_annotations_step_impl(
        context,
        source=resolve_preproc_step_source(context, "annotations"),
        rows=rows,
        mark_filter_edges=mark_filter_edges,
        mark_preproc_step_fn=mark_preproc_step,
        invalidate_downstream_fn=invalidate_downstream_preproc_steps,
        read_raw_fif_fn=read_raw_fif_fn,
        copy2_fn=copy2_fn,
    )
    if ok:
        _clear_preproc_step_skip(context, "annotations")
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


def preproc_filter_review_required(resolver: PathResolver) -> bool:
    return _preproc_filter_review_required_impl(resolver)


def preproc_annotations_panel_state(
    resolver: PathResolver,
    *,
    rows: list[dict[str, Any]],
    mark_filter_edges: Any = False,
) -> str:
    return _preproc_annotations_panel_state_impl(
        resolver,
        rows=rows,
        mark_filter_edges=mark_filter_edges,
    )


def preproc_ecg_panel_state(
    resolver: PathResolver,
    *,
    method: Any,
    picks: Any,
    method_kwargs: dict[str, Any] | None = None,
    mark_filter_edges: Any = False,
) -> str:
    return _preproc_ecg_panel_state_impl(
        resolver,
        method=method,
        picks=picks,
        method_kwargs=method_kwargs,
        mark_filter_edges=mark_filter_edges,
    )
