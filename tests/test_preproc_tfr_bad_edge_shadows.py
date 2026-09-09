"""Focused coverage for channel-aware Preprocess TFR annotation shadows."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

from lfptensorpipe.gui.shell.preproc_plotting_viz import (
    _on_preproc_viz_tfr_plot,
    _tfr_bad_edge_shadow_intervals,
)


def _annotated_raw() -> mne.io.RawArray:
    raw = mne.io.RawArray(
        np.ones((2, 40), dtype=float),
        mne.create_info(["A", "B"], sfreq=10.0, ch_types=["seeg", "seeg"]),
        first_samp=100,
        verbose="ERROR",
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[0.2, 0.5, 0.9, 1.4, 2.0, 2.5, 2.7, 3.0],
            duration=[0.3, 0.5, 0.6, 0.5, 0.4, 0.2, 0.0, 0.2],
            description=[
                "BAD_clip",
                "BAD_global",
                "edge_a",
                "BaD_b",
                "EDGE_b",
                "artifact BAD",
                "EDGE_point",
                "OTHER",
            ],
            ch_names=[(), (), ("A",), ("B",), ("B",), (), (), ()],
        )
    )
    return raw


def test_tfr_bad_edge_shadow_intervals_follow_channel_scope_and_union() -> None:
    raw = _annotated_raw()

    assert _tfr_bad_edge_shadow_intervals(
        raw,
        picks=["A"],
        display_start=0.25,
        display_stop=2.2,
    ) == [(0.25, 1.5)]
    assert _tfr_bad_edge_shadow_intervals(
        raw,
        picks=["B"],
        display_start=0.25,
        display_stop=2.2,
    ) == [(0.25, 1.0), (1.4, 1.9), (2.0, 2.2)]
    assert _tfr_bad_edge_shadow_intervals(
        raw,
        picks=["A", "B"],
        display_start=0.25,
        display_stop=2.2,
    ) == [(0.25, 1.9), (2.0, 2.2)]


class _TfrPlotHarness:
    def __init__(self, raw: mne.io.RawArray) -> None:
        self._raw = raw
        self._preproc_viz_tfr_params = {
            "fmin": 1.0,
            "fmax": 4.0,
            "n_freqs": 8,
            "decim": 4,
        }
        self._preproc_viz_selected_channels = ("A", "B")
        self._enable_plots = True
        self.figure = None
        self.warning = None

    def statusBar(self):
        return Mock()

    def _current_preproc_viz_source(self):
        return "raw", Path("/tmp/raw/raw.fif")

    def _read_raw_fif(self, path, *, preload, verbose):
        _ = path, preload, verbose
        return self._raw

    def _create_matplotlib_subplots(self):
        figure, axis = plt.subplots()
        figure.show = Mock()
        return figure, axis

    def _track_plot_figure(self, figure) -> None:
        self.figure = figure

    def _show_warning(self, title, message) -> None:
        self.warning = (title, message)


def test_tfr_plot_draws_union_shadows_for_merged_channels() -> None:
    harness = _TfrPlotHarness(_annotated_raw())

    _on_preproc_viz_tfr_plot(harness)

    assert harness.warning is None
    assert harness.figure is not None
    ax = harness.figure.axes[0]
    assert len(ax.patches) == 2
    assert all(patch.get_alpha() == 0.2 for patch in ax.patches)
    expected_color = matplotlib.colors.to_rgba("#F2000E", alpha=0.2)
    assert all(
        np.allclose(patch.get_facecolor(), expected_color) for patch in ax.patches
    )
    plt.close(harness.figure)
