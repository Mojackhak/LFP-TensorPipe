"""Band-default helpers for tensor GUI defaults."""

from __future__ import annotations

from math import isfinite

from lfptensorpipe.gui.shell.common import (
    Any,
    DEFAULT_TENSOR_BANDS,
    TENSOR_BANDS_DEFAULTS_KEY,
    TENSOR_METRIC_DEFAULTS_KEY,
)


def _normalize_tensor_bands_rows(value: Any) -> list[dict[str, float | str]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, float | str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in names:
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except Exception:
            continue
        if not isfinite(start) or not isfinite(end) or start <= 0.0 or end <= start:
            continue
        names.add(name)
        normalized.append({"name": name, "start": float(start), "end": float(end)})
    return sorted(normalized, key=lambda item: float(item["start"]))


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
