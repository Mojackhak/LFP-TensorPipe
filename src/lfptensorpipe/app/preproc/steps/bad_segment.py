"""Bad-segment-removal step runtime helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet
from lfptensorpipe.app.shared.generation_lineage import (
    new_result_generation_id,
    params_with_generation_lineage,
)

from ..paths import (
    preproc_step_config_path,
    preproc_step_log_path,
    preproc_step_raw_path,
    write_preproc_step_config,
)
from ..lineage import (
    PreprocInputGenerationChanged,
    capture_preproc_input_generation,
    preproc_input_generation_matches,
)

MarkStepFn = Callable[..., Any]
InvalidateFn = Callable[[RecordContext, str], list[Any]]
_REMOVAL_PREFIXES = ("BAD", "EDGE")
BAD_SEGMENT_MATCH_SEMANTICS_KEY = "bad_segment_match_semantics"
BAD_SEGMENT_MATCH_SEMANTICS = "case_insensitive_bad_edge_prefix"


def bad_segment_log_has_current_match_semantics(
    payload: dict[str, Any] | None,
) -> bool:
    """Return whether a completed-log payload uses current removal matching."""
    if not isinstance(payload, dict):
        return False
    params = payload.get("params")
    return bool(
        isinstance(params, dict)
        and params.get(BAD_SEGMENT_MATCH_SEMANTICS_KEY) == BAD_SEGMENT_MATCH_SEMANTICS
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
        return False, "No valid preprocess input for bad-segment step."

    source_step, src = source
    captured = capture_preproc_input_generation(resolver, "bad_segment_removal")
    if captured is None or captured[0] != source_step:
        return False, "Bad Segment source changed before input read."
    _, input_generations = captured
    result_generation_id = new_result_generation_id()

    try:
        import mne

        if read_raw_fif_fn is None:
            read_raw_fif_fn = mne.io.read_raw_fif
        runtime_filter = (
            filter_lfp_with_bad_annotations_fn or filter_lfp_with_bad_annotations
        )

        raw = read_raw_fif_fn(str(src), preload=True, verbose="ERROR")
        filtered = runtime_filter(
            raw,
            bad_descs=_REMOVAL_PREFIXES,
            do_pre_filter=False,
            do_post_notch=False,
            do_post_filter=False,
            overlap_policy="compress",
            match_mode="prefix",
            case_sensitive=False,
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
        with AtomicOutputSet(
            [dst, config_path, log_path],
            cleanup_stale_residues=True,
        ) as output_set:
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
                params=params_with_generation_lineage(
                    {
                        "mode": "defaults",
                        "source_step": source_step,
                        BAD_SEGMENT_MATCH_SEMANTICS_KEY: BAD_SEGMENT_MATCH_SEMANTICS,
                    },
                    result_generation_id=result_generation_id,
                    input_generations=input_generations,
                ),
                input_path=str(src),
                output_path=str(dst),
                message=(
                    f"Bad Segment step completed using source {source_step} with "
                    "filter_lfp_with_bad_annotations defaults."
                ),
                log_path=output_set.staged_path(log_path),
            )
            if not preproc_input_generation_matches(
                resolver,
                "bad_segment_removal",
                source_step=source_step,
                input_generations=input_generations,
            ):
                raise PreprocInputGenerationChanged(
                    "Bad Segment source changed during execution."
                )
            output_set.commit()
    except PreprocInputGenerationChanged as exc:
        return False, f"Bad Segment step failed: {exc}"
    except Exception as exc:
        mark_preproc_step_fn(
            resolver=resolver,
            step="bad_segment_removal",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Bad Segment step failed: {exc}",
        )
        return False, f"Bad Segment step failed: {exc}"

    invalidate_downstream_fn(context, "bad_segment_removal")
    return True, "Bad Segment step completed."
