"""Dataset-context shell assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.dataset_context_actions import (
    MainWindowDatasetContextActionsMixin,
)
from lfptensorpipe.gui.shell.dataset_context_defaults import (
    MainWindowDatasetContextDefaultsMixin,
)
from lfptensorpipe.gui.shell.dataset_context_selection import (
    MainWindowDatasetContextSelectionMixin,
)


class MainWindowDatasetContextMixin(
    MainWindowDatasetContextDefaultsMixin,
    MainWindowDatasetContextSelectionMixin,
    MainWindowDatasetContextActionsMixin,
):
    """Assembly mixin for dataset defaults, selection, and add/delete actions."""
