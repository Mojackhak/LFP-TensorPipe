"""Tensor selector/normalization helpers."""

from __future__ import annotations

from typing import Any


def normalize_metric_channels(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("selected_channels must be a list of channel names.")
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        channel = str(item).strip()
        if not channel or channel in seen:
            continue
        seen.add(channel)
        out.append(channel)
    return out


def normalize_metric_pairs(value: Any) -> list[tuple[str, str]] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("selected_pairs must be a list of [source, target] pairs.")
    out: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid selected pair format: {item!r}")
        source = str(item[0]).strip()
        target = str(item[1]).strip()
        if not source or not target:
            continue
        out.append((source, target))
    return out


def normalize_metric_bands(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except Exception:
            continue
        out.append({"name": name, "start": float(start), "end": float(end)})
    return out


def normalize_selected_pairs(
    selected_pairs: list[tuple[str, str]] | None,
    *,
    available_channels: set[str],
    directed: bool,
) -> list[tuple[str, str]]:
    if not selected_pairs:
        return []
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in selected_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid pair format: {pair!r}")
        source = str(pair[0]).strip()
        target = str(pair[1]).strip()
        if not source or not target:
            raise ValueError("Pair channels cannot be empty.")
        if source == target:
            raise ValueError(f"Self-pairs are not allowed: {source}")
        if source not in available_channels or target not in available_channels:
            raise ValueError(
                f"Selected pair includes unknown channel: {source}->{target}"
            )
        normalized_pair = (
            (source, target) if directed else tuple(sorted((source, target)))
        )
        if normalized_pair in seen:
            continue
        seen.add(normalized_pair)
        normalized.append(normalized_pair)
    return normalized


__all__ = [
    "normalize_metric_channels",
    "normalize_metric_pairs",
    "normalize_metric_bands",
    "normalize_selected_pairs",
]
