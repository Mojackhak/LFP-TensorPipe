"""Focused tests for frozen-safe preprocess filter helpers."""

from __future__ import annotations

from types import ModuleType
import sys

import mne
import numpy as np

from lfptensorpipe.preproc.filter import BadAnnotationConfig, mark_lfp_bad_segments


def test_mark_lfp_bad_segments_keeps_autoreject_results_when_qc_save_fails(
    monkeypatch,
) -> None:
    info = mne.create_info(["C0"], sfreq=10.0, ch_types=["eeg"])
    data = np.tile(np.linspace(-5e-5, 5e-5, 10, dtype=float), 4)[np.newaxis, :]
    raw = mne.io.RawArray(data, info, verbose=False)
    cfg = BadAnnotationConfig(
        l_freq=1.0,
        h_freq=4.0,
        epoch_dur=1.0,
        p2p_thresh=(1e-6, 1e-3),
        notches=None,
        verbose=False,
    )

    fake_autoreject = ModuleType("autoreject")

    class FakeRejectLog:
        def __init__(self, *, bad_epochs, labels, ch_names):
            self.bad_epochs = bad_epochs
            self.labels = labels
            self.ch_names = ch_names

    def fake_compute_thresholds(*_args, **_kwargs):
        return {"C0": 1e-6}

    fake_autoreject.RejectLog = FakeRejectLog
    fake_autoreject.compute_thresholds = fake_compute_thresholds
    monkeypatch.setitem(sys.modules, "autoreject", fake_autoreject)

    def fake_save_reject_log(*_args, **_kwargs):
        raise AttributeError("'NoneType' object has no attribute 'write'")

    monkeypatch.setattr(
        "lfptensorpipe.preproc.filter._save_reject_log_plot_agg",
        fake_save_reject_log,
    )

    raw_marked, reject_log, summary = mark_lfp_bad_segments(
        raw,
        cfg,
        reject_plot_path="reject.png",
    )

    assert reject_log is not None
    assert summary["n_bad_autoreject"] == 4
    assert summary["autoreject_plot_error"].startswith("AttributeError:")
    assert "autoreject_plot_failed:" in str(raw_marked.info.get("description", ""))
