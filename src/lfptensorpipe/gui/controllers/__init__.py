"""Compatibility-oriented main-window controller helpers.

This package exists for stable controller-style imports. New MainWindow logic
should be placed in focused `gui.shell` modules first.
"""

from __future__ import annotations

from .record_sync import (
    persist_record_params_snapshot,
    post_step_action_sync,
    sync_record_params_from_logs,
)
from .stage_controller import (
    make_indicator_label,
    page_title,
    placeholder_block,
    refresh_stage_controls,
    route_to_stage,
    set_indicator_color,
)

__all__ = [
    "persist_record_params_snapshot",
    "post_step_action_sync",
    "sync_record_params_from_logs",
    "make_indicator_label",
    "page_title",
    "placeholder_block",
    "refresh_stage_controls",
    "route_to_stage",
    "set_indicator_color",
]
