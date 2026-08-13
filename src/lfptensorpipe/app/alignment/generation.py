"""Accepted Alignment output-generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log

AlignmentGenerationStage = Literal["run", "finish"]

ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE = (
    "Latest Align Run uses a legacy or incomplete metric manifest. "
    "Rerun Align Epochs before Finish."
)
ALIGNMENT_FINISH_MANIFEST_RERUN_MESSAGE = (
    "Latest Align Finish uses a legacy or incomplete metric manifest. "
    "Rerun Align Finish before Extract Features."
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


def accepted_alignment_metrics(
    resolver: PathResolver,
    *,
    trial_slug: str,
    stage: AlignmentGenerationStage,
) -> list[str] | None:
    """Resolve metrics owned by the latest accepted Run or Finish event."""
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
    if latest_run is None or latest_run[1].get("completed") is not True:
        return None
    run_metrics = metrics_from_alignment_entry(latest_run[1])
    if run_metrics is None:
        return None
    if stage == "run":
        return run_metrics

    latest_finish = _latest_step(entries, "build_raw_table")
    if (
        latest_finish is None
        or latest_finish[0] <= latest_run[0]
        or latest_finish[1].get("completed") is not True
    ):
        return None
    finish_metrics = metrics_from_alignment_entry(latest_finish[1])
    if finish_metrics is None or not set(finish_metrics).issubset(run_metrics):
        return None
    return finish_metrics


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
    "ALIGNMENT_FINISH_MANIFEST_RERUN_MESSAGE",
    "ALIGNMENT_RUN_MANIFEST_RERUN_MESSAGE",
    "accepted_alignment_artifact_paths",
    "accepted_alignment_metrics",
    "alignment_generation_rerun_message",
    "metrics_from_alignment_entry",
]
