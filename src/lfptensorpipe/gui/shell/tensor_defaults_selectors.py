"""Selector-default helpers for tensor GUI defaults."""

from __future__ import annotations

from lfptensorpipe.gui.shell.common import (
    Any,
    TENSOR_SELECTOR_DEFAULTS_KEY,
)


def _tensor_default_selected_channels_for_metric(
    self,
    metric_key: str,
    *,
    available_channels: tuple[str, ...],
) -> tuple[str, ...]:
    if not self._tensor_metric_requires_channel_selector(metric_key):
        return ()
    metric_node = self._tensor_metric_default_override_node(metric_key)
    if "selected_channels" in metric_node:
        selected = self._coerce_tensor_channels(metric_node.get("selected_channels"))
    else:
        selector_defaults = self._tensor_load_default_channels()
        if selector_defaults is None:
            selected = tuple(available_channels)
        else:
            selected = self._coerce_tensor_channels(selector_defaults)
    if not available_channels:
        return selected
    allowed = set(available_channels)
    return tuple(channel for channel in selected if channel in allowed)


def _tensor_default_selected_pairs_for_metric(
    self,
    metric_key: str,
    *,
    directed: bool,
    available_channels: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    if self._tensor_metric_pair_mode(metric_key) is None:
        return ()
    metric_node = self._tensor_metric_default_override_node(metric_key)
    if "selected_pairs" in metric_node:
        pairs = self._coerce_tensor_pairs(
            metric_node.get("selected_pairs"), directed=directed
        )
    else:
        selector_defaults = self._tensor_load_default_pairs(directed=directed)
        if selector_defaults is None:
            pairs = ()
        else:
            pairs = self._coerce_tensor_pairs(selector_defaults, directed=directed)
    if not available_channels:
        return pairs
    return self._filter_tensor_pairs(
        pairs,
        available_channels=available_channels,
        directed=directed,
    )


def _tensor_read_selector_defaults_payload(self) -> dict[str, Any]:
    payload = self._config_store.read_yaml("tensor.yml", {})
    if not isinstance(payload, dict):
        return {}
    selectors = payload.get(TENSOR_SELECTOR_DEFAULTS_KEY)
    if not isinstance(selectors, dict):
        return {}
    return selectors


def _tensor_load_default_channels(self) -> tuple[str, ...] | None:
    selectors = self._tensor_read_selector_defaults_payload()
    if "channels" not in selectors:
        return None
    raw_channels = selectors.get("channels")
    if not isinstance(raw_channels, (list, tuple)):
        return ()
    deduped: list[str] = []
    for item in raw_channels:
        channel = str(item).strip()
        if not channel or channel in deduped:
            continue
        deduped.append(channel)
    return tuple(deduped)


def _tensor_load_default_pairs(
    self,
    *,
    directed: bool,
) -> tuple[tuple[str, str], ...] | None:
    selectors = self._tensor_read_selector_defaults_payload()
    key = "directed_pairs" if directed else "undirected_pairs"
    if key not in selectors:
        return None
    raw_pairs = selectors.get(key)
    if not isinstance(raw_pairs, (list, tuple)):
        return ()
    parsed: list[tuple[str, str]] = []
    for token in raw_pairs:
        pair = self._parse_tensor_pair_token(token)
        if pair is None:
            continue
        try:
            normalized = self._normalize_tensor_pair(
                pair[0], pair[1], directed=directed
            )
        except Exception:
            continue
        parsed.append(normalized)
    return tuple(parsed)


def _tensor_save_default_channels(self, channels: tuple[str, ...]) -> None:
    payload = self._config_store.read_yaml("tensor.yml", {})
    if not isinstance(payload, dict):
        payload = {}
    selectors = payload.get(TENSOR_SELECTOR_DEFAULTS_KEY)
    if not isinstance(selectors, dict):
        selectors = {}
    selectors["channels"] = [str(channel) for channel in channels]
    payload[TENSOR_SELECTOR_DEFAULTS_KEY] = selectors
    self._config_store.write_yaml("tensor.yml", payload)


def _tensor_save_default_pairs(
    self,
    pairs: tuple[tuple[str, str], ...],
    *,
    directed: bool,
) -> None:
    payload = self._config_store.read_yaml("tensor.yml", {})
    if not isinstance(payload, dict):
        payload = {}
    selectors = payload.get(TENSOR_SELECTOR_DEFAULTS_KEY)
    if not isinstance(selectors, dict):
        selectors = {}
    key = "directed_pairs" if directed else "undirected_pairs"
    if directed:
        selectors[key] = [f"({source},{target})" for source, target in pairs]
    else:
        selectors[key] = [f"{source}-{target}" for source, target in pairs]
    payload[TENSOR_SELECTOR_DEFAULTS_KEY] = selectors
    self._config_store.write_yaml("tensor.yml", payload)
