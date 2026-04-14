"""Tensor state shell assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.tensor_state_metrics import (
    MainWindowTensorStateMetricsMixin,
)
from lfptensorpipe.gui.shell.tensor_state_refresh import (
    MainWindowTensorStateRefreshMixin,
)
from lfptensorpipe.gui.shell.tensor_state_selectors import (
    MainWindowTensorStateSelectorsMixin,
)


class MainWindowTensorStateMixin(
    MainWindowTensorStateMetricsMixin,
    MainWindowTensorStateSelectorsMixin,
    MainWindowTensorStateRefreshMixin,
):
    """Assembly mixin for tensor metric state, selectors, and refresh flow."""
