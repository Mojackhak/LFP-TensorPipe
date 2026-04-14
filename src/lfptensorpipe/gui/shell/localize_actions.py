"""Localize shell assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.localize_apply_actions import (
    MainWindowLocalizeApplyActionsMixin,
)
from lfptensorpipe.gui.shell.localize_config import MainWindowLocalizeConfigMixin
from lfptensorpipe.gui.shell.localize_match_actions import (
    MainWindowLocalizeMatchActionsMixin,
)
from lfptensorpipe.gui.shell.localize_state import MainWindowLocalizeStateMixin


class MainWindowLocalizeActionsMixin(
    MainWindowLocalizeStateMixin,
    MainWindowLocalizeMatchActionsMixin,
    MainWindowLocalizeConfigMixin,
    MainWindowLocalizeApplyActionsMixin,
):
    """Assembly mixin for localize state, match, and apply/viewer flow."""
