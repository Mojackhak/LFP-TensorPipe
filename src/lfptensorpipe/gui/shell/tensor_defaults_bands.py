"""Band-default helpers for tensor GUI defaults."""

from __future__ import annotations

from lfptensorpipe.app.tensor.selectors import (
    normalize_tensor_bands_rows as _normalize_tensor_bands_rows,
)
from lfptensorpipe.gui.shell.common import (
    DEFAULT_TENSOR_BANDS,
    TENSOR_BANDS_DEFAULTS_KEY,
    TENSOR_METRIC_DEFAULTS_KEY,
    Any,
)


def _load_tensor_bands_defaults(self) -> list[dict[str, Any]]:
    payload = self._config_store.read_yaml("tensor.yml", default={})
    defaults: list[dict[str, Any]] = [dict(item) for item in DEFAULT_TENSOR_BANDS]
    if not isinstance(payload, dict):
        payload = {}
    normalized = _normalize_tensor_bands_rows(payload.get(TENSOR_BANDS_DEFAULTS_KEY))
    if normalized:
        return [dict(item) for item in normalized]
    return defaults


def _load_tensor_metric_bands_defaults(
    self,
    metric_key: str,
) -> list[dict[str, float | str]]:
    payload = self._config_store.read_yaml("tensor.yml", default={})
    if not isinstance(payload, dict):
        payload = {}
    if metric_key not in {"psi", "burst"}:
        return [dict(item) for item in self._load_tensor_bands_defaults()]
    metric_defaults = payload.get(TENSOR_METRIC_DEFAULTS_KEY)
    if not isinstance(metric_defaults, dict):
        metric_defaults = {}
    metric_node = metric_defaults.get(metric_key)
    if not isinstance(metric_node, dict):
        metric_node = {}
    bands = _normalize_tensor_bands_rows(metric_node.get("bands"))
    if bands:
        return [dict(item) for item in bands]
    return [dict(item) for item in self._load_tensor_bands_defaults()]
