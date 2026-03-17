"""Reusable dialog logic helpers extracted from GUI classes."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget


def checked_item_texts(list_widget: QListWidget) -> tuple[str, ...]:
    selected: list[str] = []
    for row_idx in range(list_widget.count()):
        item = list_widget.item(row_idx)
        if item is not None and item.checkState() == Qt.Checked:
            selected.append(item.text())
    return tuple(selected)


def set_all_check_state(list_widget: QListWidget, *, checked: bool) -> None:
    target_state = Qt.Checked if checked else Qt.Unchecked
    for row_idx in range(list_widget.count()):
        item = list_widget.item(row_idx)
        if item is not None:
            item.setCheckState(target_state)


def normalize_pair(
    source: str,
    target: str,
    *,
    directed: bool,
) -> tuple[str, str]:
    src = str(source).strip()
    dst = str(target).strip()
    if not src or not dst:
        raise ValueError("Pair channels cannot be empty.")
    if src == dst:
        raise ValueError("Self-pairs are not allowed.")
    if directed:
        return src, dst
    return tuple(sorted((src, dst)))  # type: ignore[return-value]


def all_possible_pairs(
    channels: tuple[str, ...],
    *,
    directed: bool,
) -> list[tuple[str, str]]:
    values = list(channels)
    if directed:
        return [
            (source, target)
            for source in values
            for target in values
            if source != target
        ]
    out: list[tuple[str, str]] = []
    for source_idx, source in enumerate(values):
        for target in values[source_idx + 1 :]:
            out.append(tuple(sorted((source, target))))  # type: ignore[arg-type]
    return out


def auto_contact_index_side(contact_name: str) -> tuple[str | None, str | None]:
    text = str(contact_name).strip()
    match = re.match(
        r"^K([0-9A-Za-z]+)\s*\(([RL])\)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, None
    token = match.group(1).strip().upper()
    side = match.group(2).strip().upper()
    if not token:
        return None, None
    return token, side


def auto_channel_pair(channel_name: str) -> tuple[str, str, str | None] | None:
    match = re.match(
        r"^(\d+[A-Za-z]*)_(\d+[A-Za-z]*)(?:_([LR]))?$",
        channel_name.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    left = match.group(1).strip().upper()
    right = match.group(2).strip().upper()
    side = match.group(3).strip().upper() if match.group(3) else None
    if not left or not right:
        return None
    return left, right, side


__all__ = [
    "checked_item_texts",
    "set_all_check_state",
    "normalize_pair",
    "all_possible_pairs",
    "auto_contact_index_side",
    "auto_channel_pair",
]
