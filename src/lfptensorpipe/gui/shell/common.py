"""Shared shell support for MainWindow-related modules."""

from __future__ import annotations

from lfptensorpipe.app import load_annotations_csv_rows as _load_annotations_csv_rows
from lfptensorpipe.gui.dialogs import (
    all_possible_pairs as _all_possible_pairs,
    auto_channel_pair as _auto_channel_pair,
    auto_contact_index_side as _auto_contact_index_side,
    checked_item_texts as _checked_item_texts,
    normalize_pair as _normalize_pair,
    set_all_check_state as _set_all_check_state,
)
from lfptensorpipe.gui.dialogs import common as _common

__all__ = [
    name for name in dir(_common) if not (name.startswith("__") and name.endswith("__"))
]
_dialog_all_possible_pairs = _all_possible_pairs
_dialog_auto_channel_pair = _auto_channel_pair
_dialog_auto_contact_index_side = _auto_contact_index_side
_dialog_checked_item_texts = _checked_item_texts
_dialog_normalize_pair = _normalize_pair
_dialog_set_all_check_state = _set_all_check_state
load_annotations_csv_rows = _load_annotations_csv_rows

__all__ += [
    "_dialog_all_possible_pairs",
    "_dialog_auto_channel_pair",
    "_dialog_auto_contact_index_side",
    "_dialog_checked_item_texts",
    "_dialog_normalize_pair",
    "_dialog_set_all_check_state",
    "load_annotations_csv_rows",
]


def __getattr__(name: str):
    return getattr(_common, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_common)))
