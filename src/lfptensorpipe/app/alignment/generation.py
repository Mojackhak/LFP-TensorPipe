"""Accepted Alignment output-generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from lfptensorpipe.app.localize.paths import (
    localize_indicator_state,
    localize_result_generation_id,
)
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.preproc.lineage import preproc_step_lineage_is_current
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.shared.generation_lineage import (
    LOCALIZE_GENERATION_REF,
    accepted_result_generation_id,
    alignment_generation_ref,
    input_generation_receipts_match,
    preproc_generation_ref,
    tensor_generation_ref,
)
from lfptensorpipe.app.tensor.lineage import (
    tensor_metric_lineage_is_current,
    tensor_metric_result_generation_id,
)
from lfptensorpipe.lfp.burst.semantics import (
    BURST_SAMPLE_SUPPORT,
    BURST_SAMPLE_SUPPORT_KEY,
)

AlignmentGenerationStage = Literal["run", "finish"]

_ALIGNMENT_STAGE_STEP: dict[AlignmentGenerationStage, str] = {
    "run": "run_align_epochs",
    "finish": "build_raw_table",
}


class AlignmentInputGenerationChangedError(RuntimeError):
    """Raised when an Alignment candidate no longer matches captured inputs."""


ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE = (
    "Latest Align Run uses a legacy or incomplete metric manifest. "
    "Rerun Align Epochs before Finish."
)
ALIGNMENT_FINISH_MANIFEST_RERUN_MESSAGE = (
    "Latest Align Finish uses a legacy or incomplete metric manifest. "
    "Rerun Align Finish before Extract Features."
)
ALIGNMENT_CLIP_STITCH_GEOMETRY_RERUN_MESSAGE = (
    "Latest Clip/Stitch alignment uses legacy seam or physical-time geometry. "
    "Rerun Align Epochs and Finish before Extract Features."
)
ALIGNMENT_BURST_SAMPLE_SUPPORT_RERUN_MESSAGE = (
    "Latest Burst alignment uses legacy native sample-support semantics. "
    "Rerun Align Epochs and Finish before Extract Features."
)
ALIGNMENT_ZERO_DURATION_RERUN_MESSAGE = (
    "Latest Stack/Clip/Stitch alignment uses legacy point-event signal-support "
    "semantics. Rerun Align Epochs and Finish before Extract Features."
)
ALIGNMENT_LINEAR_EVENT_PAIRING_RERUN_MESSAGE = (
    "Latest Line Up Key Events alignment uses legacy event-pairing semantics. "
    "Rerun Align Epochs and Finish before Extract Features."
)


def _history_entries(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    history = payload.get("history")
    if isinstance(history, list):
        return [item for item in history if isinstance(item, dict)]
    return [payload]


def _latest_step(
    entries: list[dict[str, Any]],
    step: str,
) -> tuple[int, dict[str, Any]] | None:
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if str(entry.get("step", "")).strip() == step:
            return index, entry
    return None


def metrics_from_alignment_entry(entry: dict[str, Any]) -> list[str] | None:
    """Return the explicit non-empty metric manifest from one log event."""
    params = entry.get("params")
    if not isinstance(params, dict):
        return None
    raw_metrics = params.get("metrics")
    if not isinstance(raw_metrics, list):
        return None
    metrics: list[str] = []
    seen: set[str] = set()
    for raw_metric in raw_metrics:
        metric = str(raw_metric).strip()
        if not metric or metric in seen:
            return None
        seen.add(metric)
        metrics.append(metric)
    if not metrics:
        return None
    declared_count = params.get("n_metrics")
    if declared_count is not None:
        try:
            if int(declared_count) != len(metrics):
                return None
        except (TypeError, ValueError):
            return None
    return metrics


def latest_alignment_step_entry(
    resolver: PathResolver,
    *,
    trial_slug: str,
    step: str,
) -> tuple[int, dict[str, Any]] | None:
    """Return the latest event for one exact Alignment step."""
    slug = str(trial_slug).strip()
    target_step = str(step).strip()
    if not slug or not target_step:
        return None
    log_path = resolver.alignment_root / slug / "lfptensorpipe_log.json"
    try:
        payload = read_run_log(log_path)
    except Exception:
        return None
    return _latest_step(_history_entries(payload), target_step)


def _preproc_finish_result_generation_id(
    resolver: PathResolver,
) -> str | None:
    log_path = (
        resolver.preproc_step_dir("finish", create=False) / "lfptensorpipe_log.json"
    )
    try:
        payload = read_run_log(log_path)
    except Exception:
        return None
    return accepted_result_generation_id(payload)


def _run_artifacts_exist(
    resolver: PathResolver,
    *,
    trial_slug: str,
    metrics: list[str],
) -> bool:
    root = resolver.alignment_paradigm_dir(trial_slug, create=False)
    if not (root / "warp_fn.pkl").is_file() or not (root / "warp_labels.pkl").is_file():
        return False
    return all((root / metric / "tensor_warped.pkl").is_file() for metric in metrics)


def _finish_artifacts_exist(
    resolver: PathResolver,
    *,
    trial_slug: str,
    metrics: list[str],
) -> bool:
    root = resolver.alignment_paradigm_dir(trial_slug, create=False)
    return all((root / metric / "na-raw.pkl").is_file() for metric in metrics)


def capture_alignment_run_input_generations(
    resolver: PathResolver,
    *,
    metrics: list[str] | tuple[str, ...],
) -> dict[str, str | None] | None:
    """Capture Finish and manifest Tensor generations before Align input reads."""
    normalized_metrics = [str(metric).strip() for metric in metrics]
    if (
        not normalized_metrics
        or any(not metric for metric in normalized_metrics)
        or len(normalized_metrics) != len(set(normalized_metrics))
        or not preproc_step_lineage_is_current(resolver, "finish")
    ):
        return None

    input_generations: dict[str, str | None] = {
        preproc_generation_ref("finish"): _preproc_finish_result_generation_id(resolver)
    }
    for metric in normalized_metrics:
        if not tensor_metric_lineage_is_current(resolver, metric):
            return None
        input_generations[tensor_generation_ref(metric)] = (
            tensor_metric_result_generation_id(resolver, metric)
        )
    return input_generations


def alignment_run_input_generations_match(
    resolver: PathResolver,
    *,
    metrics: list[str] | tuple[str, ...],
    input_generations: dict[str, str | None],
) -> bool:
    """Recheck the exact original Align Run capture before promotion."""
    return (
        capture_alignment_run_input_generations(
            resolver,
            metrics=metrics,
        )
        == input_generations
    )


def capture_alignment_finish_input_generations(
    resolver: PathResolver,
    *,
    trial_slug: str,
    include_localize: bool,
) -> dict[str, str | None] | None:
    """Capture exact same-trial Run and the Localize result actually read."""
    slug = str(trial_slug).strip()
    if not slug or not alignment_stage_lineage_is_current(
        resolver,
        trial_slug=slug,
        stage="run",
    ):
        return None
    latest_run = latest_alignment_step_entry(
        resolver,
        trial_slug=slug,
        step="run_align_epochs",
    )
    if latest_run is None:
        return None
    current_localize_ready = (
        localize_indicator_state(
            resolver.context.project_root,
            resolver.context.subject,
            resolver.context.record,
        )
        == "green"
    )
    if current_localize_ready != bool(include_localize):
        return None

    input_generations: dict[str, str | None] = {
        alignment_generation_ref(slug, "run_align_epochs"): (
            accepted_result_generation_id(latest_run[1])
        )
    }
    if include_localize:
        input_generations[LOCALIZE_GENERATION_REF] = localize_result_generation_id(
            resolver.context.project_root,
            resolver.context.subject,
            resolver.context.record,
        )
    return input_generations


def alignment_finish_input_generations_match(
    resolver: PathResolver,
    *,
    trial_slug: str,
    include_localize: bool,
    input_generations: dict[str, str | None],
) -> bool:
    """Recheck the exact original Align Finish capture before promotion."""
    return (
        capture_alignment_finish_input_generations(
            resolver,
            trial_slug=trial_slug,
            include_localize=include_localize,
        )
        == input_generations
    )


def alignment_stage_lineage_is_current(
    resolver: PathResolver,
    *,
    trial_slug: str,
    stage: AlignmentGenerationStage,
) -> bool:
    """Return whether the latest exact Alignment stage is recursively current."""
    slug = str(trial_slug).strip()
    step = _ALIGNMENT_STAGE_STEP.get(stage)
    if not slug or step is None:
        return False
    latest = latest_alignment_step_entry(
        resolver,
        trial_slug=slug,
        step=step,
    )
    if latest is None or latest[1].get("completed") is not True:
        return False
    metrics = metrics_from_alignment_entry(latest[1])
    if metrics is None:
        return False

    if stage == "run":
        captured = capture_alignment_run_input_generations(
            resolver,
            metrics=metrics,
        )
        return (
            captured is not None
            and _run_artifacts_exist(
                resolver,
                trial_slug=slug,
                metrics=metrics,
            )
            and input_generation_receipts_match(latest[1], expected=captured)
        )

    latest_run = latest_alignment_step_entry(
        resolver,
        trial_slug=slug,
        step="run_align_epochs",
    )
    if (
        latest_run is None
        or latest[0] <= latest_run[0]
        or not alignment_stage_lineage_is_current(
            resolver,
            trial_slug=slug,
            stage="run",
        )
    ):
        return False
    run_metrics = metrics_from_alignment_entry(latest_run[1])
    if run_metrics is None or not set(metrics).issubset(run_metrics):
        return False
    params = latest[1].get("params")
    if not isinstance(params, dict):
        return False
    current_localize_ready = (
        localize_indicator_state(
            resolver.context.project_root,
            resolver.context.subject,
            resolver.context.record,
        )
        == "green"
    )
    stored_localize_ready = params.get("merge_location_info_ready")
    if isinstance(stored_localize_ready, bool):
        if stored_localize_ready != current_localize_ready:
            return False
        include_localize = stored_localize_ready
    else:
        include_localize = current_localize_ready
    captured = capture_alignment_finish_input_generations(
        resolver,
        trial_slug=slug,
        include_localize=include_localize,
    )
    return (
        captured is not None
        and _finish_artifacts_exist(
            resolver,
            trial_slug=slug,
            metrics=metrics,
        )
        and input_generation_receipts_match(latest[1], expected=captured)
    )


def alignment_stage_result_generation_id(
    resolver: PathResolver,
    *,
    trial_slug: str,
    stage: AlignmentGenerationStage,
) -> str | None:
    """Return the current accepted result ID for one exact Alignment stage."""
    if not alignment_stage_lineage_is_current(
        resolver,
        trial_slug=trial_slug,
        stage=stage,
    ):
        return None
    latest = latest_alignment_step_entry(
        resolver,
        trial_slug=trial_slug,
        step=_ALIGNMENT_STAGE_STEP[stage],
    )
    return accepted_result_generation_id(latest[1]) if latest is not None else None


def alignment_generation_requires_burst_sample_support_rerun(
    entry: dict[str, Any],
) -> bool:
    """Return whether one accepted Burst alignment predates current support."""
    metrics = metrics_from_alignment_entry(entry)
    if metrics is None or "burst" not in metrics:
        return False
    params = entry.get("params")
    return not (
        isinstance(params, dict)
        and params.get(BURST_SAMPLE_SUPPORT_KEY) == BURST_SAMPLE_SUPPORT
    )


def accepted_alignment_metrics(
    resolver: PathResolver,
    *,
    trial_slug: str,
    stage: AlignmentGenerationStage,
) -> list[str] | None:
    """Resolve metrics owned by the latest accepted Run or Finish event."""
    slug = str(trial_slug).strip()
    if not slug or not alignment_stage_lineage_is_current(
        resolver,
        trial_slug=slug,
        stage=stage,
    ):
        return None
    latest = latest_alignment_step_entry(
        resolver,
        trial_slug=slug,
        step=_ALIGNMENT_STAGE_STEP[stage],
    )
    if latest is None or latest[1].get("completed") is not True:
        return None
    return metrics_from_alignment_entry(latest[1])


def alignment_generation_rerun_message(
    resolver: PathResolver,
    *,
    trial_slug: str,
    stage: AlignmentGenerationStage,
) -> str | None:
    """Return actionable guidance for a successful legacy/incomplete event."""
    slug = str(trial_slug).strip()
    if not slug:
        return None
    log_path = resolver.alignment_root / slug / "lfptensorpipe_log.json"
    try:
        payload = read_run_log(log_path)
    except Exception:
        return None
    entries = _history_entries(payload)
    latest_run = _latest_step(entries, "run_align_epochs")
    if latest_run is not None and latest_run[1].get("completed") is True:
        params = latest_run[1].get("params")
        if (
            isinstance(params, dict)
            and str(params.get("method", "")) == "linear_warper"
        ):
            from .method_specs import (
                LINEAR_EVENT_PAIRING,
                LINEAR_EVENT_PAIRING_KEY,
            )

            if params.get(LINEAR_EVENT_PAIRING_KEY) != LINEAR_EVENT_PAIRING:
                return ALIGNMENT_LINEAR_EVENT_PAIRING_RERUN_MESSAGE
        if isinstance(params, dict) and str(params.get("method", "")) in {
            "pad_warper",
            "concat_warper",
        }:
            from .method_specs import (
                CLIP_STITCH_GEOMETRY,
                CLIP_STITCH_GEOMETRY_KEY,
            )

            if params.get(CLIP_STITCH_GEOMETRY_KEY) != CLIP_STITCH_GEOMETRY:
                return ALIGNMENT_CLIP_STITCH_GEOMETRY_RERUN_MESSAGE
        if alignment_generation_requires_burst_sample_support_rerun(latest_run[1]):
            return ALIGNMENT_BURST_SAMPLE_SUPPORT_RERUN_MESSAGE
        if (
            isinstance(params, dict)
            and metrics_from_alignment_entry(latest_run[1]) is not None
        ):
            from .method_specs import (
                ZERO_DURATION_ALIGNMENT,
                ZERO_DURATION_ALIGNMENT_KEY,
                ZERO_DURATION_ALIGNMENT_METHODS,
            )

            if (
                str(params.get("method", "")) in ZERO_DURATION_ALIGNMENT_METHODS
                and params.get(ZERO_DURATION_ALIGNMENT_KEY) != ZERO_DURATION_ALIGNMENT
            ):
                return ALIGNMENT_ZERO_DURATION_RERUN_MESSAGE
    step = "run_align_epochs" if stage == "run" else "build_raw_table"
    latest = _latest_step(entries, step)
    if (
        latest is None
        or latest[1].get("completed") is not True
        or metrics_from_alignment_entry(latest[1]) is not None
    ):
        return None
    return (
        ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE
        if stage == "run"
        else ALIGNMENT_FINISH_MANIFEST_RERUN_MESSAGE
    )


def accepted_alignment_artifact_paths(
    resolver: PathResolver,
    *,
    trial_slug: str,
    stage: AlignmentGenerationStage,
) -> list[tuple[str, Path]]:
    """Return declared artifact paths for one accepted Alignment generation."""
    metrics = accepted_alignment_metrics(
        resolver,
        trial_slug=trial_slug,
        stage=stage,
    )
    if metrics is None:
        return []
    filename = "tensor_warped.pkl" if stage == "run" else "na-raw.pkl"
    root = resolver.alignment_root / str(trial_slug).strip()
    return [(metric, root / metric / filename) for metric in metrics]


__all__ = [
    "ALIGNMENT_BURST_SAMPLE_SUPPORT_RERUN_MESSAGE",
    "ALIGNMENT_CLIP_STITCH_GEOMETRY_RERUN_MESSAGE",
    "ALIGNMENT_FINISH_MANIFEST_RERUN_MESSAGE",
    "ALIGNMENT_LINEAR_EVENT_PAIRING_RERUN_MESSAGE",
    "ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE",
    "ALIGNMENT_ZERO_DURATION_RERUN_MESSAGE",
    "AlignmentInputGenerationChangedError",
    "accepted_alignment_artifact_paths",
    "accepted_alignment_metrics",
    "alignment_finish_input_generations_match",
    "alignment_generation_requires_burst_sample_support_rerun",
    "alignment_generation_rerun_message",
    "alignment_run_input_generations_match",
    "alignment_stage_lineage_is_current",
    "alignment_stage_result_generation_id",
    "capture_alignment_finish_input_generations",
    "capture_alignment_run_input_generations",
    "latest_alignment_step_entry",
    "metrics_from_alignment_entry",
]
