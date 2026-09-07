"""Tensor selector/normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorChannelInventory:
    """Ordered Finish channel inventory used by Tensor selectors."""

    all_channels: tuple[str, ...]
    bad_channels: tuple[str, ...]
    usable_channels: tuple[str, ...]


def tensor_channel_inventory_from_raw(raw: Any) -> TensorChannelInventory:
    """Return ordered all, bad, and usable channel names from one Raw."""
    all_channels = tuple(str(name) for name in raw.ch_names)
    available = set(all_channels)
    info = getattr(raw, "info", {})
    listed_bads = {str(name) for name in info.get("bads", ()) if str(name) in available}
    bad_channels = tuple(name for name in all_channels if name in listed_bads)
    usable_channels = tuple(name for name in all_channels if name not in listed_bads)
    return TensorChannelInventory(
        all_channels=all_channels,
        bad_channels=bad_channels,
        usable_channels=usable_channels,
    )


def load_tensor_channel_inventory(
    raw_path: Any,
    *,
    read_raw_fif_fn: Any | None = None,
) -> TensorChannelInventory:
    """Read one FIF header and return its Tensor channel inventory."""
    if read_raw_fif_fn is None:
        import mne

        read_raw_fif_fn = mne.io.read_raw_fif
    raw = read_raw_fif_fn(str(raw_path), preload=False, verbose="ERROR")
    try:
        return tensor_channel_inventory_from_raw(raw)
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()


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


def select_usable_channels(
    selected_channels: Any,
    *,
    inventory: TensorChannelInventory,
) -> tuple[list[str], tuple[str, ...]]:
    """Return effective channels and selected bad channels that were excluded."""
    normalized = normalize_metric_channels(selected_channels)
    if normalized is None:
        return list(inventory.usable_channels), tuple(inventory.bad_channels)
    usable = set(inventory.usable_channels)
    bad = set(inventory.bad_channels)
    effective = [name for name in normalized if name in usable]
    excluded = tuple(name for name in normalized if name in bad)
    return effective, excluded


def select_usable_pairs(
    selected_pairs: Any,
    *,
    inventory: TensorChannelInventory,
) -> tuple[list[tuple[str, str]] | None, tuple[tuple[str, str], ...]]:
    """Return effective pairs and pairs excluded by Finish bad channels."""
    normalized = normalize_metric_pairs(selected_pairs)
    if normalized is None:
        return None, ()
    bad = set(inventory.bad_channels)
    effective = [
        pair for pair in normalized if pair[0] not in bad and pair[1] not in bad
    ]
    excluded = tuple(pair for pair in normalized if pair[0] in bad or pair[1] in bad)
    return effective, excluded


__all__ = [
    "TensorChannelInventory",
    "load_tensor_channel_inventory",
    "normalize_metric_channels",
    "normalize_metric_pairs",
    "normalize_metric_bands",
    "normalize_selected_pairs",
    "select_usable_channels",
    "select_usable_pairs",
    "tensor_channel_inventory_from_raw",
]
