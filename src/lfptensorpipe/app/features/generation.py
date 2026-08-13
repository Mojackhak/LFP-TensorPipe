"""Accepted Extract Features output-generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log

from .indicator import features_derivatives_log_path, features_derivatives_root

FEATURE_MANIFEST_RERUN_MESSAGE = (
    "Latest Extract Features result uses a legacy or incomplete output manifest. "
    "Rerun Extract Features."
)
NUMERIC_MEAN_SEMANTICS_KEY = "numeric_mean_semantics"
NUMERIC_MEAN_SEMANTICS = "continuous_interval_trapezoid"
NUMERIC_MEAN_RERUN_MESSAGE = (
    "Latest Extract Features result uses legacy numeric mean interval semantics. "
    "Rerun Extract Features."
)


def outputs_from_features_entry(
    entry: dict[str, Any],
) -> dict[str, list[Path]] | None:
    """Return the explicit relative output manifest from one successful event."""
    if entry.get("completed") is not True:
        return None
    params = entry.get("params")
    if not isinstance(params, dict):
        return None
    raw_outputs = params.get("outputs_by_metric")
    if not isinstance(raw_outputs, dict) or not raw_outputs:
        return None

    outputs: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for raw_metric, raw_paths in raw_outputs.items():
        metric = str(raw_metric).strip()
        if not metric or not isinstance(raw_paths, list):
            return None
        metric_paths: list[Path] = []
        for raw_path in raw_paths:
            relative = Path(str(raw_path).strip())
            if (
                not str(relative)
                or relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 2
                or relative.parts[0] != metric
                or relative.suffix != ".pkl"
                or relative in seen
            ):
                return None
            seen.add(relative)
            metric_paths.append(relative)
        outputs[metric] = metric_paths

    declared_metrics = params.get("metrics")
    if not isinstance(declared_metrics, list):
        return None
    metrics = [str(metric).strip() for metric in declared_metrics]
    if not metrics or len(metrics) != len(set(metrics)):
        return None
    if set(metrics) != set(outputs):
        return None
    if not any(outputs.values()):
        return None
    declared_count = params.get("saved_outputs")
    if declared_count is not None:
        try:
            if int(declared_count) != sum(len(paths) for paths in outputs.values()):
                return None
        except (TypeError, ValueError):
            return None
    return outputs


def accepted_feature_artifact_paths(
    resolver: PathResolver,
    *,
    trial_slug: str,
) -> list[tuple[str, Path]]:
    """Return only artifacts declared by the accepted Feature generation."""
    log_path = features_derivatives_log_path(resolver, trial_slug=trial_slug)
    try:
        payload = read_run_log(log_path)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    outputs = outputs_from_features_entry(payload)
    if outputs is None:
        return []
    root = features_derivatives_root(resolver, trial_slug=trial_slug)
    paths: list[tuple[str, Path]] = []
    for metric, relative_paths in outputs.items():
        for relative_path in relative_paths:
            path = root / relative_path
            if not path.is_file():
                return []
            paths.append((metric, path))
    return paths


def feature_generation_requires_numeric_mean_rerun(entry: dict[str, Any]) -> bool:
    """Return whether one accepted generation predates continuous mean support."""
    outputs = outputs_from_features_entry(entry)
    if outputs is None:
        return False
    affected = any(
        metric != "burst"
        and any(
            path.name in {"mean-spectral.pkl", "mean-trace.pkl", "mean-scalar.pkl"}
            for path in paths
        )
        for metric, paths in outputs.items()
    )
    if not affected:
        return False
    params = entry.get("params")
    return not (
        isinstance(params, dict)
        and params.get(NUMERIC_MEAN_SEMANTICS_KEY) == NUMERIC_MEAN_SEMANTICS
    )


def feature_generation_rerun_message(
    resolver: PathResolver,
    *,
    trial_slug: str,
) -> str | None:
    """Return actionable guidance for a successful legacy/incomplete event."""
    log_path = features_derivatives_log_path(resolver, trial_slug=trial_slug)
    try:
        payload = read_run_log(log_path)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("completed") is not True:
        return None
    if outputs_from_features_entry(payload) is not None:
        return (
            NUMERIC_MEAN_RERUN_MESSAGE
            if feature_generation_requires_numeric_mean_rerun(payload)
            else None
        )
    return FEATURE_MANIFEST_RERUN_MESSAGE


__all__ = [
    "FEATURE_MANIFEST_RERUN_MESSAGE",
    "NUMERIC_MEAN_RERUN_MESSAGE",
    "NUMERIC_MEAN_SEMANTICS",
    "NUMERIC_MEAN_SEMANTICS_KEY",
    "accepted_feature_artifact_paths",
    "feature_generation_requires_numeric_mean_rerun",
    "feature_generation_rerun_message",
    "outputs_from_features_entry",
]
