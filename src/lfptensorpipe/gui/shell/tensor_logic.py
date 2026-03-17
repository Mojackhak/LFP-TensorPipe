"""Tensor shell assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.tensor_config import MainWindowTensorConfigMixin
from lfptensorpipe.gui.shell.tensor_defaults import MainWindowTensorDefaultsMixin
from lfptensorpipe.gui.shell.tensor_dialogs import MainWindowTensorDialogsMixin
from lfptensorpipe.gui.shell.tensor_panels import MainWindowTensorPanelsMixin
from lfptensorpipe.gui.shell.tensor_run import MainWindowTensorRunMixin
from lfptensorpipe.gui.shell.tensor_state import MainWindowTensorStateMixin


class MainWindowTensorMixin(
    MainWindowTensorConfigMixin,
    MainWindowTensorDefaultsMixin,
    MainWindowTensorPanelsMixin,
    MainWindowTensorStateMixin,
    MainWindowTensorDialogsMixin,
    MainWindowTensorRunMixin,
):
    """Assembly mixin for tensor defaults, panels, state, dialogs, and run flow."""
