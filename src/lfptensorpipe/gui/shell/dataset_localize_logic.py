"""Dataset/localize shell assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.dataset_context import MainWindowDatasetContextMixin
from lfptensorpipe.gui.shell.localize_actions import MainWindowLocalizeActionsMixin
from lfptensorpipe.gui.shell.localize_runtime import MainWindowLocalizeRuntimeMixin
from lfptensorpipe.gui.shell.record_params import MainWindowRecordParamsMixin


class MainWindowDatasetLocalizeMixin(
    MainWindowDatasetContextMixin,
    MainWindowRecordParamsMixin,
    MainWindowLocalizeRuntimeMixin,
    MainWindowLocalizeActionsMixin,
):
    """Assembly mixin for dataset, localize, and record-snapshot logic."""
