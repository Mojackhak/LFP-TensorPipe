"""Alignment shell assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.alignment_config import MainWindowAlignmentConfigMixin
from lfptensorpipe.gui.shell.alignment_defaults import MainWindowAlignmentDefaultsMixin
from lfptensorpipe.gui.shell.alignment_epochs import MainWindowAlignmentEpochsMixin
from lfptensorpipe.gui.shell.alignment_paradigms import (
    MainWindowAlignmentParadigmsMixin,
)
from lfptensorpipe.gui.shell.alignment_params import MainWindowAlignmentParamsMixin
from lfptensorpipe.gui.shell.alignment_run import MainWindowAlignmentRunMixin


class MainWindowAlignmentMixin(
    MainWindowAlignmentParadigmsMixin,
    MainWindowAlignmentConfigMixin,
    MainWindowAlignmentParamsMixin,
    MainWindowAlignmentDefaultsMixin,
    MainWindowAlignmentEpochsMixin,
    MainWindowAlignmentRunMixin,
):
    """Assembly mixin for alignment paradigms, config, params, epochs, and run flow."""
