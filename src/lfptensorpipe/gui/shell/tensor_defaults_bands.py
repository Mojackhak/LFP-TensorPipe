"""Band-default helpers for tensor GUI defaults."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    DEFAULT_TENSOR_BANDS,
    TENSOR_BANDS_DEFAULTS_KEY,
    TENSOR_BURST_BANDS_DEFAULTS_KEY,
    TENSOR_PSI_BANDS_DEFAULTS_KEY,
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
        if start <= 0.0 or end <= start:
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
    payload[TENSOR_BANDS_DEFAULTS_KEY] = defaults
    self._config_store.write_yaml("tensor.yml", payload)
    return defaults


def _load_tensor_metric_bands_defaults(
    self,
    metric_key: str,
) -> list[dict[str, float | str]]:
    payload = self._config_store.read_yaml("tensor.yml", default={})
    if not isinstance(payload, dict):
        payload = {}
    if metric_key == "psi":
        key = TENSOR_PSI_BANDS_DEFAULTS_KEY
    elif metric_key == "burst":
        key = TENSOR_BURST_BANDS_DEFAULTS_KEY
    else:
        return [dict(item) for item in self._load_tensor_bands_defaults()]
    bands = _normalize_tensor_bands_rows(payload.get(key))
    if bands:
        return [dict(item) for item in bands]
    base = [dict(item) for item in self._load_tensor_bands_defaults()]
    payload[key] = base
    self._config_store.write_yaml("tensor.yml", payload)
    return base


def _save_tensor_metric_bands_defaults(
    self,
    metric_key: str,
    bands: list[dict[str, Any]],
) -> None:
    if metric_key not in {"psi", "burst"}:
        return
    normalized = _normalize_tensor_bands_rows(bands)
    if not normalized:
        return
    key = (
        TENSOR_PSI_BANDS_DEFAULTS_KEY
        if metric_key == "psi"
        else TENSOR_BURST_BANDS_DEFAULTS_KEY
    )
    payload = self._config_store.read_yaml("tensor.yml", default={})
    if not isinstance(payload, dict):
        payload = {}
    payload[key] = [dict(item) for item in normalized]
    self._config_store.write_yaml("tensor.yml", payload)
