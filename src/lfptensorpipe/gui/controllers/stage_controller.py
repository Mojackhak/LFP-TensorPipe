"""Backward-compatible routing helpers; primary impl lives in gui.shell."""

from __future__ import annotations

from lfptensorpipe.gui.shell.routing import (
    make_indicator_label,
    page_title,
    placeholder_block,
    refresh_stage_controls,
    route_to_stage,
    set_indicator_color,
)

__all__ = [
    "set_indicator_color",
    "refresh_stage_controls",
    "route_to_stage",
    "page_title",
    "make_indicator_label",
    "placeholder_block",
]
