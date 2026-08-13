"""Bad-segment-removal step runtime helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet

from ..paths import (
    preproc_step_config_path,
    preproc_step_log_path,
    preproc_step_raw_path,
    write_preproc_step_config,
)

MarkStepFn = Callable[..., Any]
InvalidateFn = Callable[[RecordContext, str], list[Any]]
_REMOVAL_PREFIXES = ("BAD", "EDGE")


def _select_bad_segment_annotations(annotations: Any) -> Any:
    """Return only BAD/EDGE annotations used as removal masks."""
    import mne

    if annotations is None or len(annotations) == 0:
        return mne.Annotations([], [], [], orig_time=None)

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []
    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        label = str(description).strip()
        if not label:
            continue
        if not label.upper().startswith(_REMOVAL_PREFIXES):
            continue
        onsets.append(float(onset))
        durations.append(float(duration))
        descriptions.append(label)

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=annotations.orig_time,
    )


def apply_bad_segment_step(
    context: RecordContext,
    *,
    source: tuple[str, Path] | None,
    mark_preproc_step_fn: MarkStepFn,
    invalidate_downstream_fn: InvalidateFn,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    filter_lfp_with_bad_annotations_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Apply bad-segment-removal step using function defaults."""
    from lfptensorpipe.preproc.filter import filter_lfp_with_bad_annotations

    resolver = PathResolver(context)
    dst = preproc_step_raw_path(resolver, "bad_segment_removal")

    if source is None:
        mark_preproc_step_fn(
            resolver=resolver,
            step="bad_segment_removal",
            completed=False,
            input_path="",
            output_path=str(dst),
            message="No valid preprocess input for bad-segment step.",
        )
        invalidate_downstream_fn(context, "bad_segment_removal")
        return False, "No valid preprocess input for bad-segment step."

    source_step, src = source

    try:
        import mne

        if read_raw_fif_fn is None:
            read_raw_fif_fn = mne.io.read_raw_fif
        runtime_filter = (
            filter_lfp_with_bad_annotations_fn or filter_lfp_with_bad_annotations
        )

        raw = read_raw_fif_fn(str(src), preload=True, verbose="ERROR")
        removal_annotations = _select_bad_segment_annotations(raw.annotations)
        filtered = runtime_filter(
            raw,
            bad_annotations=removal_annotations,
            bad_descs=_REMOVAL_PREFIXES,
            do_pre_filter=False,
            do_post_notch=False,
            do_post_filter=False,
            overlap_policy="compress",
            match_mode="substring",
            verbose=False,
        )
        if isinstance(filtered, tuple):
            raw_good = filtered[0]
            filter_report = filtered[-1] if isinstance(filtered[-1], dict) else {}
        else:
            raw_good = filtered
            filter_report = {}

        config_path = preproc_step_config_path(resolver, "bad_segment_removal")
        log_path = preproc_step_log_path(resolver, "bad_segment_removal")
        with AtomicOutputSet([dst, config_path, log_path]) as output_set:
            raw_good.save(str(output_set.staged_path(dst)), overwrite=True)
            write_preproc_step_config(
                resolver=resolver,
                step="bad_segment_removal",
                path=output_set.staged_path(config_path),
                config={
                    "filter_report": filter_report,
                },
            )
            mark_preproc_step_fn(
                resolver=resolver,
                step="bad_segment_removal",
                completed=True,
                params={"mode": "defaults"},
                input_path=str(src),
                output_path=str(dst),
                message=(
                    f"Bad Segment step completed using source {source_step} with "
                    "filter_lfp_with_bad_annotations defaults."
                ),
                log_path=output_set.staged_path(log_path),
            )
            output_set.commit()
        invalidate_downstream_fn(context, "bad_segment_removal")
    except Exception as exc:
        mark_preproc_step_fn(
            resolver=resolver,
            step="bad_segment_removal",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Bad Segment step failed: {exc}",
        )
        invalidate_downstream_fn(context, "bad_segment_removal")
        return False, f"Bad Segment step failed: {exc}"

    return True, "Bad Segment step completed."
