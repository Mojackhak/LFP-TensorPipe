"""Dialog-local helper exports.

Dialog classes live in concrete dialog modules. This package re-exports only
small reusable dialog helpers, not the full dialog catalog.
"""

from __future__ import annotations

from .logic import (
    all_possible_pairs,
    auto_channel_pair,
    auto_contact_index_side,
    checked_item_texts,
    normalize_pair,
    set_all_check_state,
)

__all__ = [
    "all_possible_pairs",
    "auto_channel_pair",
    "auto_contact_index_side",
    "checked_item_texts",
    "normalize_pair",
    "set_all_check_state",
]
