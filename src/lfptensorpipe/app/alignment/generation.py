"""Accepted Alignment output-generation helpers."""

from __future__ import annotations

from math import ulp
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Literal

from lfptensorpipe.app.localize.paths import (
    localize_indicator_state,
    localize_result_generation_id,
)
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.preproc.finish_fingerprint import (
    annotation_dependency_digest,
    finish_fingerprint,
    finish_legacy_bridge,
)
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
from lfptensorpipe.app.shared.runlog_store import cache_in_run_log_read_snapshot
from lfptensorpipe.app.tensor.lineage import (
    tensor_metric_lineage_is_current,
    tensor_metric_result_generation_id,
)
from lfptensorpipe.lfp.burst.semantics import (
    BURST_SAMPLE_SUPPORT,
    BURST_SAMPLE_SUPPORT_KEY,
)
from lfptensorpipe.io.pkl_io import load_pkl

from .method_params import linear_anchor_selection_error
from .method_specs import (
    BURST_DISPLAY_GRID,
    BURST_DISPLAY_GRID_KEY,
    CLIP_BOUNDARY_PADDING_KEY,
    CLIP_STITCH_GEOMETRY,
    CLIP_STITCH_GEOMETRY_KEY,
    LINEAR_EVENT_PAIRING_KEY,
    LINEAR_WARP_GEOMETRY,
    LINEAR_WARP_GEOMETRY_KEY,
    ZERO_DURATION_ALIGNMENT,
    ZERO_DURATION_ALIGNMENT_KEY,
    ZERO_DURATION_ALIGNMENT_METHODS,
    linear_event_pairing_mode,
    DEFAULT_DROP_FIELDS,
    DEFAULT_DROP_MODE,
)
from .param_normalizers import _normalize_drop_fields
from .paths import alignment_warp_labels_path
from .warper_builder import _resolve_target_duration_s

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
ALIGNMENT_BURST_DISPLAY_RERUN_MESSAGE = (
    "Clip/Stitch Burst display bins in this trial use the earlier time grid. "
    "Rerun Align Epochs and Finish to refresh its display table."
)
ALIGNMENT_ZERO_DURATION_RERUN_MESSAGE = (
    "Latest Stack/Clip/Stitch alignment uses legacy point-event signal-support "
    "semantics. Rerun Align Epochs and Finish before Extract Features."
)
ALIGNMENT_LINEAR_WARP_GEOMETRY_RERUN_MESSAGE = (
    "Latest Line Up Key Events alignment uses legacy target-percent geometry. "
    "Rerun Align Epochs and Finish before Extract Features."
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


def _legacy_burst_sample_support(params: dict[str, Any], metrics: list[str]) -> bool:
    return (
        "burst" in metrics
        and params.get(BURST_SAMPLE_SUPPORT_KEY) != BURST_SAMPLE_SUPPORT
    )


def _run_semantics_rerun_message(
    resolver: PathResolver, slug: str, params: dict[str, Any], metrics: list[str] | None
) -> str | None:
    """Return the first incompatible semantic setting consumed by this Run."""
    method = params.get("method")
    method_params = params.get("method_params")
    if method == "linear_warper":
        if not isinstance(method_params, dict):
            return ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE
        anchor_error = linear_anchor_selection_error(method, method_params)
        if anchor_error:
            return f"{anchor_error} Rerun Align Epochs and Finish before Extract Features."
        if params.get(LINEAR_EVENT_PAIRING_KEY) != linear_event_pairing_mode(
            method_params
        ):
            return ALIGNMENT_LINEAR_EVENT_PAIRING_RERUN_MESSAGE
        if (
            method_params.get("linear_warp", True)
            and metrics
            and any(metric != "burst" for metric in metrics)
            and params.get(LINEAR_WARP_GEOMETRY_KEY) != LINEAR_WARP_GEOMETRY
        ):
            return ALIGNMENT_LINEAR_WARP_GEOMETRY_RERUN_MESSAGE
    clip_stitch = method in {"pad_warper", "concat_warper"}
    if clip_stitch and params.get(CLIP_STITCH_GEOMETRY_KEY) != CLIP_STITCH_GEOMETRY:
        return ALIGNMENT_CLIP_STITCH_GEOMETRY_RERUN_MESSAGE
    if metrics is None:
        return ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE
    if _legacy_burst_sample_support(params, metrics):
        return ALIGNMENT_BURST_SAMPLE_SUPPORT_RERUN_MESSAGE
    if (
        method in ZERO_DURATION_ALIGNMENT_METHODS
        and params.get(ZERO_DURATION_ALIGNMENT_KEY) != ZERO_DURATION_ALIGNMENT
    ):
        return ALIGNMENT_ZERO_DURATION_RERUN_MESSAGE
    if (
        clip_stitch
        and "burst" in metrics
        and params.get(BURST_DISPLAY_GRID_KEY) != BURST_DISPLAY_GRID
    ):
        return ALIGNMENT_BURST_DISPLAY_RERUN_MESSAGE
    if method == "pad_warper" and params.get(CLIP_BOUNDARY_PADDING_KEY) != "nan":
        from lfptensorpipe.lfp.warp.pad_concat import positive_pad_intervals

        try:
            labels = load_pkl(alignment_warp_labels_path(resolver, slug))
            target = _resolve_target_duration_s(
                method=method, method_params=method_params, epochs_by_label=labels
            )
            for epoch in labels["ALL"]:
                recorded = sum(b - a for a, b in positive_pad_intervals(epoch.events_t))
                tolerance = 32 * ulp(max(1.0, abs(epoch.end_t), target))
                if target - recorded > tolerance:
                    return (
                        "Latest Clip Around Event alignment stretched a recording-boundary "
                        "epoch. Rerun Align Epochs and Finish before Extract Features."
                    )
        except (
            OSError,
            EOFError,
            UnpicklingError,
            KeyError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            return ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE
    return None


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


def read_alignment_finish_payload(resolver: PathResolver) -> dict[str, Any] | None:
    """Read the Preprocess Finish event Align Epochs is about to depend on."""
    log_path = (
        resolver.preproc_step_dir("finish", create=False) / "lfptensorpipe_log.json"
    )
    try:
        payload = read_run_log(log_path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def alignment_annotation_selection(
    method: str,
    method_params: Any,
) -> dict[str, Any]:
    """Return which Finish annotations one trial's warper reads.

    This mirrors `_build_warper`: a warper reads the labels its method selects
    plus every annotation its drop rule matches. A warper that reads any other
    annotation must be reflected here, or a stale trial would read as current.
    """
    params = method_params if isinstance(method_params, dict) else {}
    drop_mode = str(params.get("drop_mode", DEFAULT_DROP_MODE)).strip().lower()
    if drop_mode not in {"exact", "substring"}:
        drop_mode = DEFAULT_DROP_MODE
    drop_fields = (
        _normalize_drop_fields(params.get("drop_fields", list(DEFAULT_DROP_FIELDS)))
        if bool(params.get("drop_bad", True))
        else []
    )

    labels: list[Any] | None = None
    if method == "linear_warper" and "cue_response" in params:
        config = params["cue_response"]
        pairs = config.get("event_pairs") if isinstance(config, dict) else None
        if isinstance(pairs, list) and all(isinstance(pair, dict) for pair in pairs):
            labels = [
                pair.get(key)
                for pair in pairs
                for key in ("cue_label", "response_label")
            ]
    elif method == "linear_warper" and "event_groups" in params:
        groups = params["event_groups"]
        if isinstance(groups, list) and all(
            isinstance(group, (list, tuple)) for group in groups
        ):
            labels = [label for group in groups for label in group]
    elif method == "linear_warper":
        anchors = params.get("anchors_percent")
        if isinstance(anchors, dict) and len(anchors) >= 2:
            labels = list(anchors.values())
    elif method in {"pad_warper", "concat_warper", "stack_warper"}:
        keep = params.get("annotations")
        if keep is None:
            labels = []
        elif isinstance(keep, list):
            labels = list(keep)

    exact = [str(label) for label in labels or [] if label is not None]
    return {
        "exact": exact + (drop_fields if drop_mode == "exact" else []),
        "substring": drop_fields if drop_mode == "substring" else [],
        # Without an explicit selection the warper falls back to the label set.
        "everything": labels is None,
    }


def _trial_finish_receipts(
    finish_payload: dict[str, Any] | None,
    *,
    method: str | None,
    method_params: Any,
) -> tuple[str, ...]:
    """Return the Finish receipts a current run of one trial may hold.

    The first one is canonical and is what a new Align Epochs run records.
    With a Finish fingerprint it names the record structure and the
    annotations this trial reads, so other annotation edits keep the trial.
    """
    generation_id = accepted_result_generation_id(finish_payload)
    fingerprint = finish_fingerprint(finish_payload)
    if fingerprint is None or not method:
        return () if generation_id is None else (generation_id,)
    selection = alignment_annotation_selection(method, method_params)

    def _receipt(view: dict[str, Any]) -> str:
        return annotation_dependency_digest(
            view["annotation_label_digests"],
            structure_digest=view["record_structure_digest"],
            **selection,
        )

    receipts = [_receipt(fingerprint)]
    if generation_id is not None:
        receipts.append(generation_id)
    bridge = finish_legacy_bridge(finish_payload)
    if bridge is not None and _receipt(bridge) == receipts[0]:
        receipts.append(str(bridge["finish_generation_id"]))
    return tuple(receipts)


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
    method: str | None = None,
    method_params: Any = None,
    finish_payload: dict[str, Any] | None = None,
) -> dict[str, str | None] | None:
    """Capture Finish and manifest Tensor generations before Align input reads.

    Pass the trial's method to capture the annotations it reads rather than
    the whole Finish event. Pass a `finish_payload` read before the Raw when
    the method is only known after reading it.
    """
    normalized_metrics = [str(metric).strip() for metric in metrics]
    if (
        not normalized_metrics
        or any(not metric for metric in normalized_metrics)
        or len(normalized_metrics) != len(set(normalized_metrics))
        or not preproc_step_lineage_is_current(resolver, "finish")
    ):
        return None

    if finish_payload is None:
        finish_payload = read_alignment_finish_payload(resolver)
    finish_receipts = _trial_finish_receipts(
        finish_payload,
        method=method,
        method_params=method_params,
    )
    input_generations: dict[str, str | None] = {
        preproc_generation_ref("finish"): (
            finish_receipts[0] if finish_receipts else None
        )
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
    method: str | None = None,
    method_params: Any = None,
) -> bool:
    """Recheck the exact original Align Run capture before promotion."""
    return (
        capture_alignment_run_input_generations(
            resolver,
            metrics=metrics,
            method=method,
            method_params=method_params,
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


@cache_in_run_log_read_snapshot(
    lambda resolver, *, trial_slug, stage: (resolver.context, trial_slug, stage)
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
        run_params = latest[1].get("params")
        if (
            _run_semantics_rerun_message(resolver, slug, run_params, metrics)
            is not None
        ):
            return False
        method = (
            str(run_params.get("method", "")).strip() or None
            if isinstance(run_params, dict)
            else None
        )
        method_params = (
            run_params.get("method_params") if isinstance(run_params, dict) else None
        )
        finish_payload = read_alignment_finish_payload(resolver)
        captured = capture_alignment_run_input_generations(
            resolver,
            metrics=metrics,
            method=method,
            method_params=method_params,
            finish_payload=finish_payload,
        )
        if captured is None or not _run_artifacts_exist(
            resolver,
            trial_slug=slug,
            metrics=metrics,
        ):
            return False
        finish_ref = preproc_generation_ref("finish")
        return any(
            input_generation_receipts_match(
                latest[1],
                expected={**captured, finish_ref: receipt},
            )
            for receipt in _trial_finish_receipts(
                finish_payload,
                method=method,
                method_params=method_params,
            )
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


@cache_in_run_log_read_snapshot(
    lambda resolver, *, trial_slug, stage: (resolver.context, trial_slug, stage)
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
    return metrics is not None and _legacy_burst_sample_support(
        entry["params"], metrics
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
        if isinstance(params, dict):
            message = _run_semantics_rerun_message(
                resolver, slug, params, metrics_from_alignment_entry(latest_run[1])
            )
            if message is not None:
                return message
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


@cache_in_run_log_read_snapshot(
    lambda resolver, *, trial_slug, stage: (resolver.context, trial_slug, stage),
    copy_result=True,
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
    "ALIGNMENT_LINEAR_WARP_GEOMETRY_RERUN_MESSAGE",
    "ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE",
    "ALIGNMENT_ZERO_DURATION_RERUN_MESSAGE",
    "AlignmentInputGenerationChangedError",
    "accepted_alignment_artifact_paths",
    "accepted_alignment_metrics",
    "alignment_annotation_selection",
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
    "read_alignment_finish_payload",
]
