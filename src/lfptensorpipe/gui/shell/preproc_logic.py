"""Preprocess shell assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.preproc_actions import MainWindowPreprocActionsMixin
from lfptensorpipe.gui.shell.preproc_channels import MainWindowPreprocChannelsMixin
from lfptensorpipe.gui.shell.preproc_defaults import MainWindowPreprocDefaultsMixin
from lfptensorpipe.gui.shell.preproc_plotting import MainWindowPreprocPlottingMixin
from lfptensorpipe.gui.shell.preproc_stage import MainWindowPreprocStageMixin


class MainWindowPreprocMixin(
    MainWindowPreprocDefaultsMixin,
    MainWindowPreprocStageMixin,
    MainWindowPreprocChannelsMixin,
    MainWindowPreprocActionsMixin,
    MainWindowPreprocPlottingMixin,
):
    """Assembly mixin for preprocess defaults, stage, channels, actions, and plotting."""
