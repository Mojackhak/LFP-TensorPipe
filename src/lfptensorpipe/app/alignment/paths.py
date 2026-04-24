"""Alignment artifact path helpers."""

from __future__ import annotations

from pathlib import Path

from lfptensorpipe.app.path_resolver import PathResolver


def alignment_paradigm_dir(
    resolver: PathResolver,
    paradigm_slug: str,
    *,
    create: bool = False,
) -> Path:
    """Return alignment paradigm root directory."""
    return resolver.alignment_paradigm_dir(paradigm_slug, create=create)


def alignment_paradigm_log_path(
    resolver: PathResolver,
    paradigm_slug: str,
    *,
    create: bool = False,
) -> Path:
    """Return alignment paradigm log path."""
    return (
        alignment_paradigm_dir(resolver, paradigm_slug, create=create)
        / "lfptensorpipe_log.json"
    )


def alignment_warp_fn_path(
    resolver: PathResolver,
    paradigm_slug: str,
    *,
    create: bool = False,
) -> Path:
    """Return saved warper callable path."""
    return (
        alignment_paradigm_dir(resolver, paradigm_slug, create=create) / "warp_fn.pkl"
    )


def alignment_warp_labels_path(
    resolver: PathResolver,
    paradigm_slug: str,
    *,
    create: bool = False,
) -> Path:
    """Return saved warp labels path."""
    return (
        alignment_paradigm_dir(resolver, paradigm_slug, create=create)
        / "warp_labels.pkl"
    )


def alignment_metric_tensor_warped_path(
    resolver: PathResolver,
    paradigm_slug: str,
    metric_key: str,
    *,
    create: bool = False,
) -> Path:
    """Return warped tensor path for one paradigm+metric."""
    return (
        alignment_paradigm_dir(resolver, paradigm_slug, create=create)
        / metric_key
        / "tensor_warped.pkl"
    )


def alignment_trial_raw_table_path(
    resolver: PathResolver,
    *,
    trial_slug: str,
    metric_key: str,
    create: bool = False,
) -> Path:
    """Return alignment-side raw table artifact path."""
    safe_metric = str(metric_key).strip() or "metric"
    out = alignment_paradigm_dir(resolver, trial_slug, create=create) / safe_metric
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out / "na-raw.pkl"


def features_raw_table_path(
    resolver: PathResolver,
    *,
    paradigm_slug: str,
    param: str,
    subparam: str,
    create: bool = False,
) -> Path:
    """Return legacy features/raw table artifact path."""
    safe_param = str(param).strip() or "param"
    safe_subparam = str(subparam).strip() or "default"
    out = resolver.features_root / "raw" / paradigm_slug / safe_param / safe_subparam
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out / f"{safe_subparam}-na-raw.pkl"


def features_raw_log_path(
    resolver: PathResolver,
    paradigm_slug: str,
    *,
    create: bool = False,
) -> Path:
    """Return legacy features/raw paradigm log path."""
    out = resolver.features_root / "raw" / paradigm_slug
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out / "lfptensorpipe_log.json"


__all__ = [
    "alignment_paradigm_dir",
    "alignment_paradigm_log_path",
    "alignment_warp_fn_path",
    "alignment_warp_labels_path",
    "alignment_metric_tensor_warped_path",
    "alignment_trial_raw_table_path",
    "features_raw_table_path",
    "features_raw_log_path",
]
