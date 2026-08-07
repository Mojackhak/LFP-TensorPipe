"""Preprocess source selection and finish-step apply helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

ReadRunLogFn = Callable[[Path], dict[str, Any] | None]
MarkStepFn = Callable[..., Any]
_SAVE_FORMATS = frozenset({"single", "double"})


def resolve_finish_source(
    context: RecordContext,
    *,
    source_priority: tuple[str, ...],
    preproc_step_raw_path_fn: Callable[[PathResolver, str], Path],
    preproc_step_log_path_fn: Callable[[PathResolver, str], Path],
    read_run_log_fn: ReadRunLogFn,
    required_step: str | None = None,
) -> tuple[str, Path] | None:
    """Resolve the highest-priority source when the required step is valid."""
    resolver = PathResolver(context)
    selected: tuple[str, Path] | None = None
    required_step_is_valid = required_step is None
    for step in source_priority:
        raw_path = preproc_step_raw_path_fn(resolver, step)
        log_path = preproc_step_log_path_fn(resolver, step)
        if not raw_path.exists() or not log_path.exists():
            continue
        payload = read_run_log_fn(log_path)
        if payload is None:
            continue
        if bool(payload.get("completed")):
            if selected is None:
                selected = (step, raw_path)
            if step == required_step:
                required_step_is_valid = True
    return selected if required_step_is_valid else None


def apply_finish_step(
    context: RecordContext,
    *,
    resolve_finish_source_fn: Callable[[RecordContext], tuple[str, Path] | None],
    preproc_step_raw_path_fn: Callable[[PathResolver, str], Path],
    mark_preproc_step_fn: MarkStepFn,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    add_head_tail_annotations_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Finalize the highest-priority source with physical endpoint markers."""
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
    raw = None
    raw_out = None
    try:
        import mne

        if read_raw_fif_fn is None:
            read_raw_fif_fn = mne.io.read_raw_fif
        if add_head_tail_annotations_fn is None:
            from lfptensorpipe.preproc.filter import add_head_tail_annotations

            add_head_tail_annotations_fn = add_head_tail_annotations

        raw = read_raw_fif_fn(str(source_path), preload=False, verbose="ERROR")
        # orig_format may be None, 'short', or 'int'; only float formats round-trip
        # without quantizing, and only these are valid save() options.
        source_format = str(raw.orig_format)
        save_format = source_format if source_format in _SAVE_FORMATS else "single"
        raw_out, edge_report = add_head_tail_annotations_fn(raw)

        finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_out.save(
            str(finish_raw_path),
            fmt=save_format,
            overwrite=True,
        )
        n_added = int(edge_report["n_added"])
        dropped = (
            int(edge_report.get("n_annotations_in", 0))
            + n_added
            - int(edge_report.get("n_annotations_out", 0))
        )
        message = f"Finish raw finalized from {source_step} with physical EDGE markers."
        if dropped:
            message = f"{message} Dropped {dropped} out-of-range annotation(s)."
        mark_preproc_step_fn(
            resolver=resolver,
            step="finish",
            completed=True,
            params={
                "source_step": source_step,
                "physical_edge_description": str(edge_report["description"]),
                "physical_edge_count": n_added,
                "physical_edge_onsets_sec": [
                    float(item) for item in edge_report["added_onsets_sec"]
                ],
                "physical_edge_durations_sec": [
                    float(item) for item in edge_report["added_durations_sec"]
                ],
                "source_first_time_sec": float(edge_report.get("first_time_sec", 0.0)),
                "dropped_annotations": dropped,
            },
            input_path=str(source_path),
            output_path=str(finish_raw_path),
            message=message,
        )
    except Exception as exc:
        mark_preproc_step_fn(
            resolver=resolver,
            step="finish",
            completed=False,
            input_path=str(source_path),
            output_path=str(finish_raw_path),
            message=f"Finish step failed: {exc}",
        )
        return False, f"Finish step failed: {exc}"
    finally:
        if raw_out is not None and raw_out is not raw and hasattr(raw_out, "close"):
            raw_out.close()
        if raw is not None and hasattr(raw, "close"):
            raw.close()

    return True, f"Finish step completed using source: {source_step}."
