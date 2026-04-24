"""Tensor artifact path resolution helpers."""

from __future__ import annotations

from pathlib import Path

from lfptensorpipe.app.path_resolver import PathResolver

TENSOR_LOG_STORAGE_KEY_ALIASES: dict[str, str] = {
    "periodic_aperiodic": "periodic",
    "aperiodic": "periodic",
}
TENSOR_TENSOR_STORAGE_KEY_ALIASES: dict[str, str] = {
    "periodic_aperiodic": "periodic",
}
TENSOR_CONFIG_STORAGE_KEY_ALIASES: dict[str, str] = {
    "periodic_aperiodic": "periodic",
}


def tensor_metric_log_path(
    resolver: PathResolver, metric_key: str, *, create: bool = False
) -> Path:
    """Return metric log path under tensor root."""
    storage_key = TENSOR_LOG_STORAGE_KEY_ALIASES.get(metric_key, metric_key)
    return (
        resolver.tensor_metric_dir(storage_key, create=create)
        / "lfptensorpipe_log.json"
    )


def tensor_metric_tensor_path(
    resolver: PathResolver, metric_key: str, *, create: bool = False
) -> Path:
    """Return metric tensor pickle path."""
    storage_key = TENSOR_TENSOR_STORAGE_KEY_ALIASES.get(metric_key, metric_key)
    return resolver.tensor_metric_dir(storage_key, create=create) / "tensor.pkl"


def tensor_metric_config_path(
    resolver: PathResolver, metric_key: str, *, create: bool = False
) -> Path:
    """Return metric config path."""
    storage_key = TENSOR_CONFIG_STORAGE_KEY_ALIASES.get(metric_key, metric_key)
    return resolver.tensor_metric_dir(storage_key, create=create) / "config.yml"


def tensor_stage_log_path(resolver: PathResolver, *, create: bool = False) -> Path:
    """Return Build-Tensor stage log path under tensor root."""
    if create:
        resolver.tensor_root.mkdir(parents=True, exist_ok=True)
    return resolver.tensor_root / "lfptensorpipe_log.json"


__all__ = [
    "TENSOR_LOG_STORAGE_KEY_ALIASES",
    "TENSOR_TENSOR_STORAGE_KEY_ALIASES",
    "TENSOR_CONFIG_STORAGE_KEY_ALIASES",
    "tensor_metric_log_path",
    "tensor_metric_tensor_path",
    "tensor_metric_config_path",
    "tensor_stage_log_path",
]
