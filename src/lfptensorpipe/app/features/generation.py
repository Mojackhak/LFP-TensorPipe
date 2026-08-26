"""Accepted Extract Features output-generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from lfptensorpipe.app.alignment.generation import (
    alignment_stage_lineage_is_current,
    alignment_stage_result_generation_id,
    latest_alignment_step_entry,
)
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.shared.generation_lineage import (
    alignment_generation_ref,
    input_generation_receipts_match,
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

from .indicator import features_derivatives_log_path, features_derivatives_root

FEATURE_MANIFEST_RERUN_MESSAGE = (
    "Latest Extract Features result uses a legacy or incomplete output manifest. "
    "Rerun Extract Features."
)
NUMERIC_MEAN_SEMANTICS_KEY = "numeric_mean_semantics"
NUMERIC_MEAN_SEMANTICS = "continuous_interval_trapezoid_positive_width_runs"
NUMERIC_MEAN_RERUN_MESSAGE = (
    "Latest Extract Features result uses legacy numeric mean interval semantics. "
    "Rerun Extract Features."
)
BURST_SAMPLE_SUPPORT_RERUN_MESSAGE = (
    "Latest Extract Features result uses legacy Burst native sample-support "
    "semantics. Rerun Extract Features."
)
CLIP_STITCH_MEAN_SUPPORT_KEY = "clip_stitch_mean_support"
CLIP_STITCH_MEAN_SUPPORT = "exact_fragment_segment_local_endpoint_hold"
CLIP_STITCH_MEAN_SUPPORT_RERUN_MESSAGE = (
    "Latest Clip/Stitch Extract Features result uses legacy source-fragment "
    "mean support. Rerun Extract Features."
)
FEATURE_LINEAGE_RERUN_MESSAGE = (
    "An accepted Alignment or Tensor input changed after Extract Features. "
    "Rerun Extract Features."
)


def _native_feature_tensor_dependencies(
    *,
    alignment_method: str,
    outputs_by_metric: Mapping[str, Sequence[str | Path]],
) -> set[str]:
    tensor_metrics: set[str] = set()
    for raw_metric, raw_paths in outputs_by_metric.items():
        metric = str(raw_metric).strip()
        names = {Path(path).name for path in raw_paths}
        if metric == "burst" and any(name.endswith("-scalar.pkl") for name in names):
            tensor_metrics.add(metric)
        if (
            metric != "burst"
            and alignment_method in {"pad_warper", "concat_warper"}
            and names.intersection({"mean-spectral.pkl", "mean-scalar.pkl"})
        ):
            tensor_metrics.add(metric)
    return tensor_metrics


def capture_feature_input_generations(
    resolver: PathResolver,
    *,
    trial_slug: str,
    alignment_method: str,
    outputs_by_metric: Mapping[str, Sequence[str | Path]],
) -> dict[str, str | None] | None:
    """Capture the accepted generations directly read by one Feature run."""
    slug = str(trial_slug).strip()
    method = str(alignment_method).strip()
    if not slug or not method:
        return None
    if not alignment_stage_lineage_is_current(
        resolver,
        trial_slug=slug,
        stage="finish",
    ):
        return None
    accepted_run = latest_alignment_step_entry(
        resolver,
        trial_slug=slug,
        step="run_align_epochs",
    )
    accepted_run_params = (
        accepted_run[1].get("params") if accepted_run is not None else None
    )
    if (
        not isinstance(accepted_run_params, dict)
        or str(accepted_run_params.get("method", "")).strip() != method
    ):
        return None
    finish_generation = alignment_stage_result_generation_id(
        resolver,
        trial_slug=slug,
        stage="finish",
    )
    input_generations: dict[str, str | None] = {
        alignment_generation_ref(slug, "build_raw_table"): finish_generation
    }
    tensor_metrics = _native_feature_tensor_dependencies(
        alignment_method=method,
        outputs_by_metric=outputs_by_metric,
    )
    if not tensor_metrics:
        return input_generations
    run_generation = alignment_stage_result_generation_id(
        resolver,
        trial_slug=slug,
        stage="run",
    )
    if not alignment_stage_lineage_is_current(
        resolver,
        trial_slug=slug,
        stage="run",
    ):
        return None
    input_generations[alignment_generation_ref(slug, "run_align_epochs")] = (
        run_generation
    )
    for metric in sorted(tensor_metrics):
        tensor_generation = tensor_metric_result_generation_id(resolver, metric)
        if not tensor_metric_lineage_is_current(resolver, metric):
            return None
        input_generations[tensor_generation_ref(metric)] = tensor_generation
    return input_generations


def feature_input_generations_match(
    resolver: PathResolver,
    *,
    trial_slug: str,
    alignment_method: str,
    outputs_by_metric: Mapping[str, Sequence[str | Path]],
    input_generations: Mapping[str, str | None],
) -> bool:
    """Recheck the exact Feature input capture before artifact promotion."""
    return capture_feature_input_generations(
        resolver,
        trial_slug=trial_slug,
        alignment_method=alignment_method,
        outputs_by_metric=outputs_by_metric,
    ) == dict(input_generations)


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


@cache_in_run_log_read_snapshot(
    lambda resolver, *, trial_slug: (resolver.context, trial_slug),
    copy_result=True,
)
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
    if outputs is None or not _feature_entry_lineage_is_current(
        resolver,
        trial_slug=trial_slug,
        entry=payload,
        outputs=outputs,
    ):
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


def _feature_entry_lineage_is_current(
    resolver: PathResolver,
    *,
    trial_slug: str,
    entry: dict[str, Any],
    outputs: Mapping[str, Sequence[Path]],
) -> bool:
    params = entry.get("params")
    if not isinstance(params, dict):
        return False
    expected = capture_feature_input_generations(
        resolver,
        trial_slug=trial_slug,
        alignment_method=str(params.get("alignment_method", "")).strip(),
        outputs_by_metric=outputs,
    )
    return expected is not None and input_generation_receipts_match(
        entry,
        expected=expected,
    )


@cache_in_run_log_read_snapshot(
    lambda resolver, *, trial_slug: (resolver.context, trial_slug)
)
def feature_generation_lineage_is_current(
    resolver: PathResolver,
    *,
    trial_slug: str,
) -> bool:
    """Return whether the accepted Feature generation matches direct inputs."""
    log_path = features_derivatives_log_path(resolver, trial_slug=trial_slug)
    try:
        payload = read_run_log(log_path)
    except Exception:
        return False
    if not isinstance(payload, dict) or payload.get("completed") is not True:
        return False
    outputs = outputs_from_features_entry(payload)
    return outputs is not None and _feature_entry_lineage_is_current(
        resolver,
        trial_slug=trial_slug,
        entry=payload,
        outputs=outputs,
    )


def feature_generation_requires_numeric_mean_rerun(entry: dict[str, Any]) -> bool:
    """Return whether one accepted generation predates current mean semantics."""
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


def feature_generation_requires_burst_sample_support_rerun(
    entry: dict[str, Any],
) -> bool:
    """Return whether one accepted Burst generation predates current support."""
    outputs = outputs_from_features_entry(entry)
    if outputs is None or "burst" not in outputs:
        return False
    params = entry.get("params")
    return not (
        isinstance(params, dict)
        and params.get(BURST_SAMPLE_SUPPORT_KEY) == BURST_SAMPLE_SUPPORT
    )


def feature_generation_requires_clip_stitch_mean_support_rerun(
    entry: dict[str, Any],
) -> bool:
    """Return whether one accepted Clip/Stitch mean uses legacy support."""
    outputs = outputs_from_features_entry(entry)
    if outputs is None:
        return False
    params = entry.get("params")
    if not isinstance(params, dict) or params.get("alignment_method") not in {
        "pad_warper",
        "concat_warper",
    }:
        return False
    affected = any(
        metric != "burst"
        and any(path.name in {"mean-spectral.pkl", "mean-scalar.pkl"} for path in paths)
        for metric, paths in outputs.items()
    )
    return affected and params.get(CLIP_STITCH_MEAN_SUPPORT_KEY) != (
        CLIP_STITCH_MEAN_SUPPORT
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
        if not feature_generation_lineage_is_current(
            resolver,
            trial_slug=trial_slug,
        ):
            return FEATURE_LINEAGE_RERUN_MESSAGE
        if feature_generation_requires_burst_sample_support_rerun(payload):
            return BURST_SAMPLE_SUPPORT_RERUN_MESSAGE
        if feature_generation_requires_clip_stitch_mean_support_rerun(payload):
            return CLIP_STITCH_MEAN_SUPPORT_RERUN_MESSAGE
        return (
            NUMERIC_MEAN_RERUN_MESSAGE
            if feature_generation_requires_numeric_mean_rerun(payload)
            else None
        )
    return FEATURE_MANIFEST_RERUN_MESSAGE


__all__ = [
    "BURST_SAMPLE_SUPPORT_RERUN_MESSAGE",
    "CLIP_STITCH_MEAN_SUPPORT",
    "CLIP_STITCH_MEAN_SUPPORT_KEY",
    "CLIP_STITCH_MEAN_SUPPORT_RERUN_MESSAGE",
    "FEATURE_MANIFEST_RERUN_MESSAGE",
    "FEATURE_LINEAGE_RERUN_MESSAGE",
    "NUMERIC_MEAN_RERUN_MESSAGE",
    "NUMERIC_MEAN_SEMANTICS",
    "NUMERIC_MEAN_SEMANTICS_KEY",
    "accepted_feature_artifact_paths",
    "capture_feature_input_generations",
    "feature_generation_lineage_is_current",
    "feature_generation_requires_burst_sample_support_rerun",
    "feature_generation_requires_clip_stitch_mean_support_rerun",
    "feature_generation_requires_numeric_mean_rerun",
    "feature_generation_rerun_message",
    "feature_input_generations_match",
    "outputs_from_features_entry",
]
