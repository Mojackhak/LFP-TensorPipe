"""Helpers for invalidating downstream result logs after upstream rewrites."""

from __future__ import annotations

from pathlib import Path

from .path_resolver import PathResolver, RecordContext
from .runlog_store import RunLogRecord, append_run_log_event, read_run_log


def _append_invalidation_event(
    path: Path,
    *,
    step: str,
    input_path: str,
    output_path: str,
    message: str,
) -> Path | None:
    """Append a failed event only when the target log already exists."""
    try:
        payload = read_run_log(path)
    except Exception:
        payload = None
    if payload is None or not path.exists():
        return None
    append_run_log_event(
        path,
        RunLogRecord(
            step=step,
            completed=False,
            params={},
            input_path=input_path,
            output_path=output_path,
            message=message,
        ),
    )
    return path


def _tensor_metric_log_paths(resolver: PathResolver) -> list[Path]:
    if not resolver.tensor_root.exists():
        return []
    return sorted(resolver.tensor_root.glob("*/lfptensorpipe_log.json"))


def _alignment_log_paths(
    resolver: PathResolver,
    *,
    paradigm_slug: str | None = None,
) -> list[Path]:
    if not resolver.alignment_root.exists():
        return []
    slug = str(paradigm_slug or "").strip()
    if slug:
        path = resolver.alignment_root / slug / "lfptensorpipe_log.json"
        return [path] if path.exists() else []
    return sorted(resolver.alignment_root.glob("*/lfptensorpipe_log.json"))


def _features_log_paths(
    resolver: PathResolver,
    *,
    paradigm_slug: str | None = None,
) -> list[Path]:
    if not resolver.features_root.exists():
        return []
    slug = str(paradigm_slug or "").strip()
    if slug:
        path = resolver.features_root / slug / "lfptensorpipe_log.json"
        return [path] if path.exists() else []
    return sorted(resolver.features_root.glob("*/lfptensorpipe_log.json"))


def invalidate_after_preproc_result_change(
    context: RecordContext,
    *,
    changed_step: str,
) -> list[Path]:
    """Invalidate tensor/alignment/features results after a preproc rewrite."""
    resolver = PathResolver(context)
    message = f"Invalidated by upstream result change: preproc/{changed_step}."
    rewritten: list[Path] = []

    for path in _tensor_metric_log_paths(resolver):
        step = path.parent.name.strip() or "build_tensor_metric"
        updated = _append_invalidation_event(
            path,
            step=step,
            input_path=str(resolver.preproc_root),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)

    stage_log = resolver.tensor_root / "lfptensorpipe_log.json"
    updated = _append_invalidation_event(
        stage_log,
        step="build_tensor",
        input_path=str(resolver.preproc_root),
        output_path=str(resolver.tensor_root),
        message=message,
    )
    if updated is not None:
        rewritten.append(updated)

    for path in _alignment_log_paths(resolver):
        updated = _append_invalidation_event(
            path,
            step="run_align_epochs",
            input_path=str(resolver.tensor_root),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)

    for path in _features_log_paths(resolver):
        updated = _append_invalidation_event(
            path,
            step="run_extract_features",
            input_path=str(resolver.alignment_root),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)
    return rewritten


def invalidate_after_tensor_result_change(
    context: RecordContext,
    *,
    metric_keys: list[str] | tuple[str, ...],
) -> list[Path]:
    """Invalidate alignment/features results after Build Tensor rewrites metrics."""
    resolver = PathResolver(context)
    metrics_text = ", ".join(
        sorted({str(item).strip() for item in metric_keys if str(item).strip()})
    )
    metrics_suffix = f" ({metrics_text})" if metrics_text else ""
    message = f"Invalidated by upstream result change: tensor{metrics_suffix}."
    rewritten: list[Path] = []

    for path in _alignment_log_paths(resolver):
        updated = _append_invalidation_event(
            path,
            step="run_align_epochs",
            input_path=str(resolver.tensor_root),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)

    for path in _features_log_paths(resolver):
        updated = _append_invalidation_event(
            path,
            step="run_extract_features",
            input_path=str(resolver.alignment_root),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)
    return rewritten


def invalidate_after_localize_result_change(context: RecordContext) -> list[Path]:
    """Invalidate alignment-finish/features results after Localize Apply rewrites."""
    resolver = PathResolver(context)
    message = "Invalidated by upstream result change: localize/apply."
    rewritten: list[Path] = []

    for path in _alignment_log_paths(resolver):
        updated = _append_invalidation_event(
            path,
            step="build_raw_table",
            input_path=str(resolver.lfp_root / "localize"),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)

    for path in _features_log_paths(resolver):
        updated = _append_invalidation_event(
            path,
            step="run_extract_features",
            input_path=str(resolver.alignment_root),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)
    return rewritten


def invalidate_after_alignment_run(
    context: RecordContext,
    *,
    paradigm_slug: str,
) -> list[Path]:
    """Invalidate one trial's finish/features results after Align Epochs reruns."""
    resolver = PathResolver(context)
    slug = str(paradigm_slug).strip()
    if not slug:
        return []
    message = f"Invalidated by upstream result change: alignment/{slug}/run_align_epochs."
    rewritten: list[Path] = []

    for path in _alignment_log_paths(resolver, paradigm_slug=slug):
        updated = _append_invalidation_event(
            path,
            step="build_raw_table",
            input_path=str(path.parent),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)

    for path in _features_log_paths(resolver, paradigm_slug=slug):
        updated = _append_invalidation_event(
            path,
            step="run_extract_features",
            input_path=str(resolver.alignment_root / slug),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)
    return rewritten


def invalidate_after_alignment_finish(
    context: RecordContext,
    *,
    paradigm_slug: str,
) -> list[Path]:
    """Invalidate one trial's features results after Align Finish rewrites raw tables."""
    resolver = PathResolver(context)
    slug = str(paradigm_slug).strip()
    if not slug:
        return []
    message = f"Invalidated by upstream result change: alignment/{slug}/build_raw_table."
    rewritten: list[Path] = []

    for path in _features_log_paths(resolver, paradigm_slug=slug):
        updated = _append_invalidation_event(
            path,
            step="run_extract_features",
            input_path=str(resolver.alignment_root / slug),
            output_path=str(path.parent),
            message=message,
        )
        if updated is not None:
            rewritten.append(updated)
    return rewritten


__all__ = [
    "invalidate_after_alignment_finish",
    "invalidate_after_alignment_run",
    "invalidate_after_localize_result_change",
    "invalidate_after_preproc_result_change",
    "invalidate_after_tensor_result_change",
]
