"""Features shell assembly mixin."""

from __future__ import annotations

from lfptensorpipe.gui.shell.features_axes import MainWindowFeaturesAxesMixin
from lfptensorpipe.gui.shell.features_config import MainWindowFeaturesConfigMixin
from lfptensorpipe.gui.shell.features_defaults import MainWindowFeaturesDefaultsMixin
from lfptensorpipe.gui.shell.features_plotting import MainWindowFeaturesPlottingMixin
from lfptensorpipe.gui.shell.features_run import MainWindowFeaturesRunMixin
from lfptensorpipe.gui.shell.features_subset import MainWindowFeaturesSubsetMixin
from lfptensorpipe.gui.shell.features_trials import MainWindowFeaturesTrialsMixin


class MainWindowFeaturesMixin(
    MainWindowFeaturesDefaultsMixin,
    MainWindowFeaturesTrialsMixin,
    MainWindowFeaturesSubsetMixin,
    MainWindowFeaturesAxesMixin,
    MainWindowFeaturesConfigMixin,
    MainWindowFeaturesRunMixin,
    MainWindowFeaturesPlottingMixin,
):
    """Assembly mixin for features defaults, trials, subset, axes, run, and plotting."""
