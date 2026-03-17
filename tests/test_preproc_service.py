"""Tests for preprocess service lifecycle helpers."""

from __future__ import annotations

# ruff: noqa: E402

import sys
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Work around NumPy/SciPy compatibility issues in this test environment where
# bool(_CopyMode.IF_NEEDED) raises during MNE imports.
_copy_mode = getattr(getattr(np, "_globals", None), "_CopyMode", None)
if _copy_mode is not None:

    def _compat_copy_mode_bool(self: object) -> bool:
        return self != _copy_mode.NEVER

    _copy_mode.__bool__ = _compat_copy_mode_bool

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.dataset_index import scan_stage_states
from lfptensorpipe.app import tensor_service as tensor_service_module
from lfptensorpipe.app.preproc_service import (
    apply_bad_segment_step,
    apply_annotations_step,
    apply_ecg_step,
    apply_filter_step,
    apply_finish_step,
    bootstrap_raw_step_from_rawdata,
    default_filter_advance_params,
    invalidate_downstream_preproc_steps,
    mark_preproc_step,
    load_annotations_csv_rows,
    normalize_filter_advance_params,
    preproc_annotations_panel_state,
    preproc_ecg_panel_state,
    preproc_filter_panel_state,
    preproc_step_config_path,
    preproc_step_log_path,
    preproc_step_raw_path,
    resolve_finish_source,
)
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    indicator_from_log,
    read_run_log,
    write_run_log,
)
from lfptensorpipe.app.tensor_service import (
    DEFAULT_TENSOR_BANDS,
    load_tensor_filter_inheritance,
    load_tensor_frequency_defaults,
    resolve_tensor_frequency_bounds,
    run_build_tensor,
    tensor_metric_config_path,
    tensor_metric_log_path,
    tensor_metric_tensor_path,
    validate_tensor_frequency_params,
)
from lfptensorpipe.io.pkl_io import load_pkl
from lfptensorpipe.lfp.interp.freq import (
    interpolate_freq_axis_transformed,
    interpolate_tensor_with_metadata_transformed,
)
from lfptensorpipe.lfp.smooth.smooth import smooth_axis
from lfptensorpipe.stats.preproc.normalize import baseline_normalize, normalize_df
from lfptensorpipe.stats.preproc.transform import transform_df
from lfptensorpipe.utils.transforms import apply_transform_array, get_transform_pair


def _context(project: Path) -> RecordContext:
    return RecordContext(project_root=project, subject="sub-001", record="runA")


def _fake_read_raw(raw_obj: Any):
    def _reader(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        return raw_obj

    return _reader


def _metric_runner_override(runner, /, **dependencies: Any):
    def _wrapped(context: RecordContext, **kwargs: Any) -> tuple[bool, str]:
        return runner(context, **kwargs, **dependencies)

    return _wrapped


def test_bootstrap_raw_step_from_rawdata(tmp_path: Path) -> None:
    context = _context(tmp_path)
    src = (
        tmp_path
        / "rawdata"
        / context.subject
        / "ses-postop"
        / "lfp"
        / context.record
        / "raw"
        / "raw.fif"
    )
    src.parent.mkdir(parents=True)
    src.write_text("dummy fif bytes", encoding="utf-8")

    ok, _ = bootstrap_raw_step_from_rawdata(context)
    assert ok

    resolver = PathResolver(context)
    assert preproc_step_raw_path(resolver, "raw").exists()
    assert indicator_from_log(preproc_step_log_path(resolver, "raw")) == "green"


def test_invalidate_downstream_steps_rewrites_logs_to_yellow(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    for step in ("filter", "annotations", "bad_segment_removal"):
        preproc_step_raw_path(resolver, step).write_text(step, encoding="utf-8")
        mark_preproc_step(
            resolver=resolver,
            step=step,
            completed=True,
            input_path="in",
            output_path="out",
            message="initial",
        )

    invalidate_downstream_preproc_steps(context, "filter")
    assert (
        indicator_from_log(preproc_step_log_path(resolver, "annotations")) == "yellow"
    )
    assert (
        indicator_from_log(preproc_step_log_path(resolver, "bad_segment_removal"))
        == "yellow"
    )
    assert preproc_step_raw_path(resolver, "annotations").exists()


def test_finish_source_priority_and_apply_finish(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    for step in ("raw", "filter", "annotations"):
        raw_path = preproc_step_raw_path(resolver, step)
        raw_path.write_text(step, encoding="utf-8")
        mark_preproc_step(
            resolver=resolver,
            step=step,
            completed=True,
            input_path=str(raw_path),
            output_path=str(raw_path),
            message="ok",
        )

    source = resolve_finish_source(context)
    assert source is not None
    assert source[0] == "annotations"

    ok, _ = apply_finish_step(context)
    assert ok
    finish_payload = read_run_log(preproc_step_log_path(resolver, "finish"))
    assert finish_payload is not None
    assert finish_payload["completed"] is True
    assert finish_payload["params"]["source_step"] == "annotations"


def test_apply_finish_fails_when_no_valid_source(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    ok, _ = apply_finish_step(context)
    assert not ok
    assert indicator_from_log(preproc_step_log_path(resolver, "finish")) == "yellow"


def test_apply_filter_step_writes_artifacts_and_invalidates_downstream(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "raw")
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("dummy fif bytes", encoding="utf-8")
    mark_preproc_step(
        resolver=resolver,
        step="raw",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="raw ready",
    )

    # Seed downstream logs to assert invalidation after filter apply.
    for step in (
        "annotations",
        "bad_segment_removal",
        "ecg_artifact_removal",
        "finish",
    ):
        mark_preproc_step(
            resolver=resolver,
            step=step,
            completed=True,
            input_path="in",
            output_path="out",
            message="seed",
        )

    tensor_log = tensor_metric_log_path(resolver, "raw_power")
    write_run_log(
        tensor_log,
        RunLogRecord(
            step="raw_power",
            completed=True,
            params={},
            input_path="in",
            output_path="out",
            message="tensor ready",
        ),
    )
    alignment_log = resolver.alignment_root / "gait" / "lfptensorpipe_log.json"
    write_run_log(
        alignment_log,
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path="out",
            message="align ready",
        ),
    )
    features_log = resolver.features_root / "gait" / "lfptensorpipe_log.json"
    write_run_log(
        features_log,
        RunLogRecord(
            step="run_extract_features",
            completed=True,
            params={},
            input_path="in",
            output_path="out",
            message="features ready",
        ),
    )

    import mne
    import numpy as np

    sfreq = 200.0
    info = mne.create_info(["CH1", "CH2"], sfreq=sfreq, ch_types=["dbs", "dbs"])
    data = np.zeros((2, 400), dtype=float)
    raw = mne.io.RawArray(data, info)
    raw.save(str(src), overwrite=True)

    called = {"reject_plot_path": None}

    def fake_marker(raw_obj, cfg, *, eeg_like_channels=None, reject_plot_path=None):
        _ = (cfg, eeg_like_channels)
        called["reject_plot_path"] = str(reject_plot_path)
        return raw_obj.copy(), None, {"n_epochs": 1}

    ok, _ = apply_filter_step(
        context,
        mark_lfp_bad_segments_fn=fake_marker,
    )
    assert ok

    assert preproc_step_raw_path(resolver, "filter").exists()
    assert preproc_step_config_path(resolver, "filter").exists()
    assert indicator_from_log(preproc_step_log_path(resolver, "filter")) == "green"
    assert called["reject_plot_path"] is not None
    assert called["reject_plot_path"].endswith("/preproc/filter/qc/reject.png")
    assert (
        indicator_from_log(preproc_step_log_path(resolver, "annotations")) == "yellow"
    )
    assert indicator_from_log(tensor_log) == "yellow"
    assert indicator_from_log(alignment_log) == "yellow"
    assert indicator_from_log(features_log) == "yellow"


def test_filter_advance_params_validation_and_apply_override(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    ok, defaults, message = normalize_filter_advance_params(None)
    assert ok
    assert message == ""
    assert defaults == default_filter_advance_params()
    assert defaults["notch_widths"] == 2.0

    valid, _, message = normalize_filter_advance_params(
        {"p2p_thresh": [0.2, 0.1], "epoch_dur": -1}
    )
    assert not valid
    assert message

    src = preproc_step_raw_path(resolver, "raw")
    src.parent.mkdir(parents=True, exist_ok=True)

    import mne
    import numpy as np

    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="raw",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="raw ready",
    )

    captured: dict[str, object] = {}

    def fake_marker(raw_obj, cfg, *, eeg_like_channels=None, reject_plot_path=None):
        _ = eeg_like_channels
        captured["notch_widths"] = cfg.notch_widths
        captured["epoch_dur"] = cfg.epoch_dur
        captured["p2p_thresh"] = cfg.p2p_thresh
        captured["autoreject_correct_factor"] = cfg.autoreject_correct_factor
        captured["reject_plot_path"] = str(reject_plot_path)
        return raw_obj.copy(), None, {"n_epochs": 1}

    ok, _ = apply_filter_step(
        context,
        advance_params={
            "notch_widths": [1.25, 1.5],
            "epoch_dur": 0.75,
            "p2p_thresh": [1e-7, 2e-4],
            "autoreject_correct_factor": 2.0,
        },
        mark_lfp_bad_segments_fn=fake_marker,
    )

    assert ok
    assert captured["notch_widths"] == [1.25, 1.5]
    assert captured["epoch_dur"] == 0.75
    assert captured["p2p_thresh"] == (1e-7, 2e-4)
    assert captured["autoreject_correct_factor"] == 2.0
    assert str(captured["reject_plot_path"]).endswith("/preproc/filter/qc/reject.png")


def test_preproc_panel_indicators_track_draft_staleness_and_restore(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    filter_params = {
        "low_freq": 1.0,
        "high_freq": 90.0,
        "notches": [50.0, 100.0],
        "notch_widths": 2.0,
        "epoch_dur": 2.0,
        "p2p_thresh": [1e-7, 2e-4],
        "autoreject_correct_factor": 1.0,
    }
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params=filter_params,
        input_path="in",
        output_path="out",
        message="filter ready",
    )
    advance_params = {
        "notch_widths": 2.0,
        "epoch_dur": 2.0,
        "p2p_thresh": [1e-7, 2e-4],
        "autoreject_correct_factor": 1.0,
    }
    assert (
        preproc_filter_panel_state(
            resolver,
            notches="50,100",
            l_freq="1",
            h_freq="90",
            advance_params=advance_params,
        )
        == "green"
    )
    assert (
        preproc_filter_panel_state(
            resolver,
            notches="50,100",
            l_freq="2",
            h_freq="90",
            advance_params=advance_params,
        )
        == "yellow"
    )
    assert (
        preproc_filter_panel_state(
            resolver,
            notches="50,100",
            l_freq="1",
            h_freq="90",
            advance_params=advance_params,
        )
        == "green"
    )

    annotations_csv = resolver.preproc_root / "annotations" / "annotations.csv"
    annotations_csv.parent.mkdir(parents=True, exist_ok=True)
    annotations_csv.write_text(
        "description,onset,duration\n" "task,1.0,2.5\n",
        encoding="utf-8",
    )
    mark_preproc_step(
        resolver=resolver,
        step="annotations",
        completed=True,
        params={"row_count": 1},
        input_path="in",
        output_path="out",
        message="annotations ready",
    )
    annotation_rows = [
        {"description": "task", "onset": "1.0", "duration": "2.5"},
    ]
    assert preproc_annotations_panel_state(resolver, rows=annotation_rows) == "green"
    assert (
        preproc_annotations_panel_state(
            resolver,
            rows=[{"description": "task", "onset": "1.5", "duration": "2.5"}],
        )
        == "yellow"
    )
    assert preproc_annotations_panel_state(resolver, rows=annotation_rows) == "green"

    mark_preproc_step(
        resolver=resolver,
        step="ecg_artifact_removal",
        completed=True,
        params={"method": "svd", "picks": ["ECG1", "ECG2"]},
        input_path="in",
        output_path="out",
        message="ecg ready",
    )
    assert (
        preproc_ecg_panel_state(
            resolver,
            method="svd",
            picks=["ECG2", "ECG1"],
        )
        == "green"
    )
    assert (
        preproc_ecg_panel_state(
            resolver,
            method="template",
            picks=["ECG2", "ECG1"],
        )
        == "yellow"
    )
    assert (
        preproc_ecg_panel_state(
            resolver,
            method="svd",
            picks=["ECG1", "ECG2"],
        )
        == "green"
    )

    mark_preproc_step(
        resolver=resolver,
        step="ecg_artifact_removal",
        completed=False,
        params={"method": "svd", "picks": ["ECG1", "ECG2"]},
        input_path="in",
        output_path="out",
        message="ecg failed",
    )
    assert (
        preproc_ecg_panel_state(
            resolver,
            method="svd",
            picks=["ECG1", "ECG2"],
        )
        == "yellow"
    )


def test_apply_filter_step_uses_runtime_filter_frequency_params(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "raw")
    src.parent.mkdir(parents=True, exist_ok=True)

    import mne
    import numpy as np

    info = mne.create_info(["CH1", "CH2"], sfreq=300.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info)
    raw.save(str(src), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="raw",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="raw ready",
    )

    captured: dict[str, object] = {}

    def fake_marker(raw_obj, cfg, *, eeg_like_channels=None, reject_plot_path=None):
        _ = (raw_obj, eeg_like_channels, reject_plot_path)
        captured["l_freq"] = cfg.l_freq
        captured["h_freq"] = cfg.h_freq
        captured["notches"] = cfg.notches
        return raw.copy(), None, {"n_epochs": 1}

    ok, _ = apply_filter_step(
        context,
        notches=[60.0, 120.0],
        l_freq=2.0,
        h_freq=130.0,
        mark_lfp_bad_segments_fn=fake_marker,
    )

    assert ok
    assert captured["l_freq"] == 2.0
    assert float(captured["h_freq"]) <= 130.0
    assert captured["notches"] == (60.0, 120.0)


def test_apply_filter_step_drops_notches_at_or_above_nyquist(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "raw")
    src.parent.mkdir(parents=True, exist_ok=True)

    import mne
    import numpy as np

    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="raw",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="raw ready",
    )

    captured: dict[str, object] = {}

    def fake_marker(raw_obj, cfg, *, eeg_like_channels=None, reject_plot_path=None):
        _ = (raw_obj, eeg_like_channels, reject_plot_path)
        captured["notches"] = cfg.notches
        return raw.copy(), None, {"n_epochs": 1}

    ok, _ = apply_filter_step(
        context,
        notches=[50.0, 100.0, 120.0],
        l_freq=1.0,
        h_freq=90.0,
        mark_lfp_bad_segments_fn=fake_marker,
    )

    assert ok
    assert captured["notches"] == (50.0,)
    payload = read_run_log(preproc_step_log_path(resolver, "filter"))
    assert payload is not None
    assert payload["params"]["dropped_notches"] == [100.0, 120.0]


def test_apply_filter_step_skips_reject_plot_path_in_background_thread(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "raw")
    src.parent.mkdir(parents=True, exist_ok=True)

    import mne
    import numpy as np

    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="raw",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="raw ready",
    )

    captured: dict[str, object] = {}

    def fake_marker(raw_obj, cfg, *, eeg_like_channels=None, reject_plot_path=None):
        _ = (raw_obj, cfg, eeg_like_channels)
        captured["reject_plot_path"] = reject_plot_path
        return raw.copy(), None, {"n_epochs": 1}

    class _ThreadStub:
        def __init__(self, name: str) -> None:
            self.name = name

    class _ThreadModule:
        @staticmethod
        def main_thread() -> _ThreadStub:
            return _ThreadStub("main")

        @staticmethod
        def current_thread() -> _ThreadStub:
            return _ThreadStub("worker")

    ok, _ = apply_filter_step(
        context,
        thread_module=_ThreadModule,
        mark_lfp_bad_segments_fn=fake_marker,
    )
    assert ok
    assert captured["reject_plot_path"] is None


def test_apply_bad_segment_step_writes_artifacts_and_invalidates_downstream(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "annotations")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)

    mark_preproc_step(
        resolver=resolver,
        step="annotations",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="annotations ready",
    )
    for step in ("ecg_artifact_removal", "finish"):
        mark_preproc_step(
            resolver=resolver,
            step=step,
            completed=True,
            input_path="in",
            output_path="out",
            message="seed",
        )

    calls = {"filtered": 0, "edge": 0}

    def fake_filter(raw_obj, **kwargs):
        _ = kwargs
        calls["filtered"] += 1
        return raw_obj.copy(), {"removed_pct": 0.0}

    def fake_add_edges(raw_obj, **kwargs):
        _ = kwargs
        calls["edge"] += 1
        return raw_obj.copy(), {"head_duration_sec": 0.0, "tail_duration_sec": 0.0}

    ok, _ = apply_bad_segment_step(
        context,
        filter_lfp_with_bad_annotations_fn=fake_filter,
        add_head_tail_annotations_fn=fake_add_edges,
    )
    assert ok
    assert calls["filtered"] == 1
    assert calls["edge"] == 1
    assert preproc_step_raw_path(resolver, "bad_segment_removal").exists()
    assert preproc_step_config_path(resolver, "bad_segment_removal").exists()
    assert (
        indicator_from_log(preproc_step_log_path(resolver, "bad_segment_removal"))
        == "green"
    )
    assert (
        indicator_from_log(preproc_step_log_path(resolver, "ecg_artifact_removal"))
        == "yellow"
    )


def test_apply_ecg_step_writes_artifacts_and_invalidates_downstream(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "bad_segment_removal")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)

    mark_preproc_step(
        resolver=resolver,
        step="bad_segment_removal",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="bad segment ready",
    )
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path="in",
        output_path="out",
        message="seed",
    )

    calls = {"method": None, "picks": None}

    def fake_raw_call(raw_obj, method, picks, **kwargs):
        _ = kwargs
        calls["method"] = method
        calls["picks"] = list(picks)
        return raw_obj.copy(), {}

    ok, _ = apply_ecg_step(
        context,
        method="template",
        raw_call_ecgremover_fn=fake_raw_call,
    )

    assert ok
    assert calls["method"] == "template"
    assert calls["picks"] == ["CH1", "CH2"]
    assert preproc_step_raw_path(resolver, "ecg_artifact_removal").exists()
    assert preproc_step_config_path(resolver, "ecg_artifact_removal").exists()
    assert (
        indicator_from_log(preproc_step_log_path(resolver, "ecg_artifact_removal"))
        == "green"
    )
    assert indicator_from_log(preproc_step_log_path(resolver, "finish")) == "yellow"


def test_apply_ecg_step_uses_selected_picks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "bad_segment_removal")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ["CH1", "CH2", "CH3"], sfreq=200.0, ch_types=["dbs", "dbs", "dbs"]
    )
    raw = mne.io.RawArray(np.zeros((3, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)

    mark_preproc_step(
        resolver=resolver,
        step="bad_segment_removal",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="bad segment ready",
    )

    captured: dict[str, object] = {}

    def fake_raw_call(raw_obj, method, picks, **kwargs):
        _ = (method, kwargs)
        captured["picks"] = list(picks)
        return raw_obj.copy(), {}

    ok, _ = apply_ecg_step(
        context,
        method="svd",
        picks=["CH2", "CH3"],
        raw_call_ecgremover_fn=fake_raw_call,
    )

    assert ok
    assert captured["picks"] == ["CH2", "CH3"]


def test_apply_annotations_step_writes_artifacts_and_invalidates_downstream(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "filter")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="filter ready",
    )
    for step in ("bad_segment_removal", "ecg_artifact_removal", "finish"):
        mark_preproc_step(
            resolver=resolver,
            step=step,
            completed=True,
            input_path="in",
            output_path="out",
            message="seed",
        )

    ok, _ = apply_annotations_step(
        context,
        rows=[{"description": "bad", "onset": "0.1", "duration": "0.2"}],
    )
    assert ok
    assert preproc_step_raw_path(resolver, "annotations").exists()
    assert preproc_step_config_path(resolver, "annotations").exists()
    assert indicator_from_log(preproc_step_log_path(resolver, "annotations")) == "green"
    assert (
        indicator_from_log(preproc_step_log_path(resolver, "bad_segment_removal"))
        == "yellow"
    )
    csv_path = resolver.preproc_step_dir("annotations", create=True) / "annotations.csv"
    assert csv_path.exists()
    assert "description,onset,duration" in csv_path.read_text(encoding="utf-8")


def test_apply_annotations_step_accepts_zero_duration(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "filter")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="filter ready",
    )

    ok, message = apply_annotations_step(
        context,
        rows=[{"description": "bad", "onset": "0.1", "duration": "0"}],
    )
    assert ok
    assert "completed" in message.lower()
    assert indicator_from_log(preproc_step_log_path(resolver, "annotations")) == "green"


def test_apply_annotations_step_inherits_filter_raw_before_writing(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "filter")
    src.parent.mkdir(parents=True, exist_ok=True)
    filter_info = mne.create_info(["CH1", "CH2"], sfreq=250.0, ch_types=["dbs", "dbs"])
    filter_raw = mne.io.RawArray(np.ones((2, 500), dtype=float), filter_info)
    filter_raw.save(str(src), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="filter ready",
    )

    dst = preproc_step_raw_path(resolver, "annotations")
    dst.parent.mkdir(parents=True, exist_ok=True)
    old_info = mne.create_info(["OLD"], sfreq=100.0, ch_types=["dbs"])
    old_raw = mne.io.RawArray(np.zeros((1, 100), dtype=float), old_info)
    old_raw.save(str(dst), overwrite=True)

    ok, _ = apply_annotations_step(
        context,
        rows=[{"description": "evt", "onset": "1.0", "duration": "0"}],
    )
    assert ok
    out = mne.io.read_raw_fif(str(dst), preload=False, verbose="ERROR")
    assert out.ch_names == ["CH1", "CH2"]
    assert float(out.info["sfreq"]) == 250.0


def test_apply_annotations_step_keeps_filter_annotations_and_appends_rows(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "filter")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.25],
            duration=[0.5],
            description=["BAD_auto"],
            orig_time=raw.annotations.orig_time,
        )
    )
    raw.save(str(src), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        input_path=str(src),
        output_path=str(src),
        message="filter ready",
    )

    ok, _ = apply_annotations_step(
        context,
        rows=[{"description": "evt", "onset": "1.0", "duration": "0"}],
    )
    assert ok

    dst = preproc_step_raw_path(resolver, "annotations")
    out = mne.io.read_raw_fif(str(dst), preload=False, verbose="ERROR")
    labels = list(out.annotations.description)
    assert "BAD_auto" in labels
    assert "evt" in labels
    assert len(labels) == 2


def test_load_annotations_csv_rows_reads_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "ann.csv"
    csv_path.write_text(
        "description,onset,duration\nbad,0.1,0.2\n",
        encoding="utf-8",
    )
    ok, rows, _ = load_annotations_csv_rows(csv_path)
    assert ok
    assert len(rows) == 1
    assert rows[0]["description"] == "bad"


def test_load_annotations_csv_rows_rejects_invalid_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "ann_invalid.csv"
    csv_path.write_text(
        "description,onset,duration\nbad,0.1,-0.1\n",
        encoding="utf-8",
    )
    ok, rows, message = load_annotations_csv_rows(csv_path)
    assert not ok
    assert rows == []
    assert "invalid rows" in message.lower()


def test_preproc_bootstrap_invalidate_and_finish_source_guard_branches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    ok, message = bootstrap_raw_step_from_rawdata(context)
    assert not ok
    assert "missing canonical rawdata input" in message.lower()
    assert indicator_from_log(preproc_step_log_path(resolver, "raw")) == "yellow"

    with pytest.raises(ValueError):
        invalidate_downstream_preproc_steps(context, "unknown_step")

    raw_path = preproc_step_raw_path(resolver, "raw")
    raw_path.write_text("dummy", encoding="utf-8")
    log_path = preproc_step_log_path(resolver, "raw")
    log_path.write_text("{}", encoding="utf-8")
    assert resolve_finish_source(context, read_run_log_fn=lambda path: None) is None


def test_filter_advance_and_apply_error_branches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    valid, _, message = normalize_filter_advance_params("bad")  # type: ignore[arg-type]
    assert not valid
    assert "dictionary" in message.lower()

    valid, _, _ = normalize_filter_advance_params({"notch_widths": 0})
    assert not valid
    valid, _, _ = normalize_filter_advance_params({"notch_widths": []})
    assert not valid
    valid, _, _ = normalize_filter_advance_params({"notch_widths": [1.0, 0.0]})
    assert not valid
    valid, _, _ = normalize_filter_advance_params({"notch_widths": "bad"})
    assert not valid

    valid, _, _ = normalize_filter_advance_params({"epoch_dur": "bad"})
    assert not valid
    valid, _, _ = normalize_filter_advance_params({"autoreject_correct_factor": 0.0})
    assert not valid
    valid, _, _ = normalize_filter_advance_params({"p2p_thresh": [1.0]})
    assert not valid
    valid, _, _ = normalize_filter_advance_params({"p2p_thresh": ["bad", "value"]})
    assert not valid
    valid, _, _ = normalize_filter_advance_params({"p2p_thresh": [1.0, 1.0]})
    assert not valid

    ok, message = apply_filter_step(context, advance_params={"notch_widths": 0})
    assert not ok
    assert "invalid filter advance params" in message.lower()

    ok, message = apply_filter_step(context, l_freq="bad")  # type: ignore[arg-type]
    assert not ok
    assert "invalid filter freq params" in message.lower()

    ok, message = apply_filter_step(context, l_freq=5.0, h_freq=4.0)
    assert not ok
    assert "require 0 <= low < high" in message.lower()

    ok, message = apply_filter_step(context, notches=["bad"])  # type: ignore[list-item]
    assert not ok
    assert "invalid filter notches" in message.lower()

    ok, message = apply_filter_step(context, notches=[0.0])
    assert not ok
    assert "values must be > 0" in message.lower()

    ok, message = apply_filter_step(context, l_freq=1.0, h_freq=90.0, notches=[50.0])
    assert not ok
    assert "missing preprocess raw input" in message.lower()

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "raw")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=20.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)

    ok, message = apply_filter_step(context, l_freq=9.999, h_freq=100.0, notches=[5.0])
    assert not ok
    assert "filter step failed" in message.lower()
    assert indicator_from_log(preproc_step_log_path(resolver, "filter")) == "yellow"


def test_apply_bad_segment_error_branches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    ok, message = apply_bad_segment_step(context)
    assert not ok
    assert "missing annotations raw input" in message.lower()

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "annotations")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)

    ok, _ = apply_bad_segment_step(
        context,
        filter_lfp_with_bad_annotations_fn=lambda raw_obj, **kwargs: raw_obj.copy(),
        add_head_tail_annotations_fn=lambda raw_obj, **kwargs: (
            raw_obj.copy(),
            {"edge_report": {}},
        ),
    )
    assert ok

    def _raise_edge(raw_obj, **kwargs):  # noqa: ANN001
        _ = (raw_obj, kwargs)
        raise RuntimeError("edge-fail")

    ok, message = apply_bad_segment_step(
        context,
        filter_lfp_with_bad_annotations_fn=lambda raw_obj, **kwargs: raw_obj.copy(),
        add_head_tail_annotations_fn=_raise_edge,
    )
    assert not ok
    assert "bad segment step failed" in message.lower()


def test_apply_ecg_guard_and_failure_branches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    ok, message = apply_ecg_step(context, method="unknown")
    assert not ok
    assert "unknown ecg method" in message.lower()

    ok, message = apply_ecg_step(context, method="svd")
    assert not ok
    assert "missing bad-segment raw input" in message.lower()

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "bad_segment_removal")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)

    ok, message = apply_ecg_step(context, method="svd", picks=[])
    assert not ok
    assert "no picks selected" in message.lower()

    ok, message = apply_ecg_step(context, method="svd", picks=["MISSING"])
    assert not ok
    assert "unknown ecg picks" in message.lower()

    captured: dict[str, Any] = {}

    def fake_raw_call(raw_obj, method, picks, **kwargs):
        _ = raw_obj
        captured["method"] = method
        captured["kwargs"] = dict(kwargs)
        return raw.copy(), {}

    ok, _ = apply_ecg_step(
        context,
        method="perceive",
        raw_call_ecgremover_fn=fake_raw_call,
    )
    assert ok
    assert captured["method"] == "perceive"

    def fail_raw_call(raw_obj, method, picks, **kwargs):  # noqa: ANN001
        _ = (raw_obj, method, picks, kwargs)
        raise RuntimeError("ecg-call-fail")

    ok, message = apply_ecg_step(
        context,
        method="template",
        raw_call_ecgremover_fn=fail_raw_call,
    )
    assert not ok
    assert "ecg step failed" in message.lower()


def test_apply_ecg_no_channels_branch(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "bad_segment_removal")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1"], sfreq=200.0, ch_types=["dbs"])
    raw = mne.io.RawArray(np.zeros((1, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)

    class _RawNoChannels:
        ch_names: list[str] = []

    ok, message = apply_ecg_step(
        context,
        method="svd",
        read_raw_fif_fn=lambda *args, **kwargs: _RawNoChannels(),
    )
    assert not ok
    assert "no channels available" in message.lower()


def test_annotations_csv_and_apply_error_branches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    missing_csv = tmp_path / "missing.csv"
    ok, rows, message = load_annotations_csv_rows(missing_csv)
    assert not ok and rows == []
    assert "does not exist" in message.lower()

    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("desc,onset,duration\nx,0,1\n", encoding="utf-8")
    ok, rows, message = load_annotations_csv_rows(bad_header)
    assert not ok and rows == []
    assert "header must contain" in message.lower()

    as_dir = tmp_path / "as_dir.csv"
    as_dir.mkdir()
    ok, rows, message = load_annotations_csv_rows(as_dir)
    assert not ok and rows == []
    assert "failed to read csv" in message.lower()

    invalid_numeric = tmp_path / "invalid_numeric.csv"
    invalid_numeric.write_text(
        "description,onset,duration\nbad,bad,1\n",
        encoding="utf-8",
    )
    ok, rows, message = load_annotations_csv_rows(invalid_numeric)
    assert not ok and rows == []
    assert "invalid rows" in message.lower()

    ok, message = apply_annotations_step(
        context,
        rows=[{"description": "evt", "onset": "0.1", "duration": "0.2"}],
    )
    assert not ok
    assert "missing filter raw input" in message.lower()

    import mne
    import numpy as np

    src = preproc_step_raw_path(resolver, "filter")
    src.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(src), overwrite=True)

    ok, message = apply_annotations_step(
        context,
        rows=[{"description": "evt", "onset": "bad", "duration": "0.2"}],
    )
    assert not ok
    assert "invalid annotation rows" in message.lower()

    ok, message = apply_annotations_step(
        context,
        rows=[{"description": "evt", "onset": "0.1", "duration": "0.2"}],
        read_raw_fif_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("read-fail")
        ),
    )
    assert not ok
    assert "annotations step failed" in message.lower()


def test_tensor_frequency_defaults_inherit_filter_and_nyquist(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=240.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 720), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)

    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 5.0,
            "high_freq": 180.0,
            "notches": [50.0, 100.0],
            "notch_widths": 2.5,
        },
        input_path="in",
        output_path="out",
        message="filter ready",
    )

    low_freq, high_freq, step_hz = load_tensor_frequency_defaults(context)
    assert low_freq == 5.0
    assert high_freq == 120.0
    assert step_hz == 0.5

    inheritance = load_tensor_filter_inheritance(context)
    assert inheritance.notches == (50.0, 100.0)
    assert inheritance.notch_widths == (2.5, 2.5)

    bounds = resolve_tensor_frequency_bounds(context)
    assert bounds.min_low_freq == 5.0
    assert bounds.max_high_freq == 120.0
    assert bounds.filter_high_freq == 180.0
    assert bounds.nyquist_freq == 120.0


def test_validate_tensor_frequency_params_enforces_filter_bounds(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)

    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 10.0,
            "high_freq": 80.0,
            "notches": [50.0],
            "notch_widths": 2.0,
        },
        input_path="in",
        output_path="out",
        message="filter ready",
    )

    ok, message, _ = validate_tensor_frequency_params(
        context,
        low_freq=5.0,
        high_freq=40.0,
        step_hz=1.0,
    )
    assert not ok
    assert ">= 10" in message

    ok, message, _ = validate_tensor_frequency_params(
        context,
        low_freq=12.0,
        high_freq=120.0,
        step_hz=1.0,
    )
    assert not ok
    assert "<= 80" in message

    ok, message, _ = validate_tensor_frequency_params(
        context,
        low_freq=12.0,
        high_freq=70.0,
        step_hz=1.0,
    )
    assert ok
    assert message == ""


def test_run_build_tensor_raw_power_success(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )
    filter_raw_path = preproc_step_raw_path(resolver, "filter")
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 1.0,
            "high_freq": 80.0,
            "notches": [50.0],
            "notch_widths": 2.0,
        },
        input_path=str(finish_raw_path),
        output_path=str(filter_raw_path),
        message="filter ready",
    )

    def fake_tfr_grid(raw_obj, **kwargs):
        _ = (raw_obj, kwargs)
        tensor = np.ones((1, 2, 8, 20), dtype=float)
        metadata = {
            "axes": {
                "channel": np.array(["CH1", "CH2"], dtype=object),
                "freq": np.linspace(2.0, 16.0, 8, dtype=float),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": "morlet"},
        }
        return tensor, metadata

    ok, message = run_build_tensor(
        context,
        selected_metrics=["raw_power"],
        low_freq=2.0,
        high_freq=80.0,
        step_hz=2.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        service_overrides={
            "_run_raw_power_metric": _metric_runner_override(
                tensor_service_module._run_raw_power_metric,
                tfr_grid_fn=fake_tfr_grid,
            )
        },
    )

    assert ok
    assert "Raw power" in message
    tensor_path = tensor_metric_tensor_path(resolver, "raw_power")
    assert tensor_path.exists()
    tensor_payload = load_pkl(tensor_path)
    assert isinstance(tensor_payload, dict)
    assert tensor_payload["tensor"].shape == (1, 2, 8, 20)
    assert indicator_from_log(tensor_metric_log_path(resolver, "raw_power")) == "green"
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "raw_power").read_text(encoding="utf-8")
    )
    assert config_payload["mask_edge_effects"] is True
    states = scan_stage_states(context.project_root, context.subject, context.record)
    assert states["tensor"] == "green"


def test_run_build_tensor_raw_power_applies_notch_interpolation(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 1.0,
            "high_freq": 90.0,
            "notches": [50.0],
            "notch_widths": 2.0,
        },
        input_path="in",
        output_path="out",
        message="filter ready",
    )

    captured: dict[str, Any] = {
        "compute_freqs": None,
        "interp_called": False,
        "full_freqs": None,
    }

    def fake_tfr_grid(raw_obj, **kwargs):
        _ = raw_obj
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        captured["compute_freqs"] = freqs
        tensor = np.ones((1, 2, freqs.size, 20), dtype=float)
        metadata = {
            "axes": {
                "channel": np.array(["CH1", "CH2"], dtype=object),
                "freq": freqs.copy(),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": "morlet"},
        }
        return tensor, metadata

    def fake_interp(
        tensor,
        metadata,
        *,
        freqs_out,
        axis,
        method,
        transform_mode,
        **kwargs,
    ):
        _ = kwargs
        assert axis == -2
        assert method == "linear"
        assert transform_mode == "dB"
        captured["interp_called"] = True
        captured["full_freqs"] = np.asarray(freqs_out, dtype=float)
        out = np.full(
            (tensor.shape[0], tensor.shape[1], len(freqs_out), tensor.shape[3]),
            2.0,
            dtype=float,
        )
        meta = dict(metadata)
        axes = dict(meta.get("axes", {}))
        axes["freq"] = np.asarray(freqs_out, dtype=float)
        axes["shape"] = out.shape
        meta["axes"] = axes
        return out, meta

    ok, _ = run_build_tensor(
        context,
        selected_metrics=["raw_power"],
        low_freq=45.0,
        high_freq=55.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        metric_params_map={
            "raw_power": {
                "notches": [50.0],
                "notch_widths": [2.0],
            }
        },
        service_overrides={
            "_run_raw_power_metric": _metric_runner_override(
                tensor_service_module._run_raw_power_metric,
                tfr_grid_fn=fake_tfr_grid,
                interpolate_freq_tensor_fn=fake_interp,
            )
        },
    )

    assert ok
    assert captured["compute_freqs"] is not None
    assert captured["interp_called"] is True
    assert 50.0 not in set(np.asarray(captured["compute_freqs"], dtype=float).tolist())
    assert int(np.asarray(captured["compute_freqs"]).size) < int(
        np.asarray(captured["full_freqs"]).size
    )
    tensor_payload = load_pkl(tensor_metric_tensor_path(resolver, "raw_power"))
    assert tensor_payload["tensor"].shape[2] == int(
        np.asarray(captured["full_freqs"]).size
    )
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "raw_power").read_text(encoding="utf-8")
    )
    assert config_payload["interpolation_applied"] is True
    assert config_payload["notches"] == [50.0]
    assert config_payload["notch_widths"] == [2.0]
    log_payload = read_run_log(tensor_metric_log_path(resolver, "raw_power"))
    assert log_payload is not None
    assert log_payload["params"]["interpolation_applied"] is True


def test_run_build_tensor_periodic_aperiodic_success(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    captured: dict[str, Any] = {}

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )

    def fake_tfr_grid(raw_obj, **kwargs):
        _ = (raw_obj, kwargs)
        tensor = np.ones((1, 2, 8, 20), dtype=float)
        metadata = {
            "axes": {
                "channel": np.array(["CH1", "CH2"], dtype=object),
                "freq": np.linspace(2.0, 16.0, 8, dtype=float),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": "morlet"},
        }
        return tensor, metadata

    def fake_decompose(tensor, freqs, **kwargs):
        _ = freqs
        report_dir = Path(kwargs["report_dir"])
        captured["report_dir"] = report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "specparam_e000_ch000.pdf").write_text(
            "dummy report",
            encoding="utf-8",
        )
        tfr_aperiodic = np.full_like(tensor, 1.0)
        tfr_periodic = np.full_like(tensor, 3.0)
        tfr_full_model = np.full_like(tensor, 4.0)
        params_tensor = np.zeros(
            (tensor.shape[0], tensor.shape[1], 4, tensor.shape[3]), dtype=float
        )
        params_meta = {
            "axes": {
                "freq": np.array(
                    ["offset", "knee", "exponent", "gof_rsquared"], dtype=object
                )
            }
        }
        params_tensor[:, :, 3, :] = 0.95
        return tfr_aperiodic, tfr_periodic, tfr_full_model, params_tensor, params_meta

    ok, message = run_build_tensor(
        context,
        selected_metrics=["periodic_aperiodic"],
        low_freq=2.0,
        high_freq=16.0,
        step_hz=2.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        service_overrides={
            "_run_periodic_aperiodic_metric": _metric_runner_override(
                tensor_service_module._run_periodic_aperiodic_metric,
                tfr_grid_fn=fake_tfr_grid,
                decompose_fn=fake_decompose,
            )
        },
    )

    assert ok
    assert "Periodic/APeriodic" in message
    tensor_path = tensor_metric_tensor_path(resolver, "periodic_aperiodic")
    assert tensor_path.exists()
    assert tensor_path.parent.name == "periodic"
    tensor_payload = load_pkl(tensor_path)
    assert isinstance(tensor_payload, dict)
    assert tensor_payload["tensor"].shape == (1, 2, 8, 20)
    aperiodic_path = tensor_metric_tensor_path(resolver, "aperiodic")
    assert aperiodic_path.exists()
    aperiodic_payload = load_pkl(aperiodic_path)
    assert isinstance(aperiodic_payload, dict)
    assert aperiodic_payload["tensor"].shape == (1, 2, 4, 20)
    assert (
        indicator_from_log(tensor_metric_log_path(resolver, "periodic_aperiodic"))
        == "green"
    )
    log_payload = read_run_log(tensor_metric_log_path(resolver, "periodic_aperiodic"))
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "periodic_aperiodic").read_text(
            encoding="utf-8"
        )
    )
    report_dir = tensor_path.parent / "specparam_report"
    assert captured["report_dir"] == report_dir
    assert report_dir.exists()
    assert sorted(path.name for path in report_dir.iterdir()) == [
        "specparam_e000_ch000.pdf"
    ]
    assert config_payload["output_component"] == "periodic"
    assert config_payload["periodic_tensor_path"] == str(tensor_path)
    assert config_payload["aperiodic_tensor_path"] == str(aperiodic_path)
    assert config_payload["specparam_report_dir"] == str(report_dir)
    assert config_payload["interpolation_applied"] is False
    assert log_payload["params"]["specparam_report_dir"] == str(report_dir)


def test_run_build_tensor_periodic_aperiodic_applies_notch_interpolation(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 1.0,
            "high_freq": 90.0,
            "notches": [50.0],
            "notch_widths": 2.0,
        },
        input_path="in",
        output_path="out",
        message="filter ready",
    )

    captured: dict[str, Any] = {
        "compute_freqs": None,
        "interp_called": False,
        "full_freqs": None,
    }

    def fake_tfr_grid(raw_obj, **kwargs):
        _ = raw_obj
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        captured["compute_freqs"] = freqs
        tensor = np.ones((1, 2, freqs.size, 20), dtype=float)
        metadata = {
            "axes": {
                "channel": np.array(["CH1", "CH2"], dtype=object),
                "freq": freqs.copy(),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": "morlet"},
        }
        return tensor, metadata

    def fake_decompose(tensor, freqs, **kwargs):
        _ = (freqs, kwargs)
        tfr_aperiodic = np.full_like(tensor, 1.0)
        tfr_periodic = np.full_like(tensor, 3.0)
        tfr_full_model = np.full_like(tensor, 4.0)
        params_tensor = np.zeros(
            (tensor.shape[0], tensor.shape[1], 4, tensor.shape[3]), dtype=float
        )
        params_meta = {
            "axes": {
                "freq": np.array(
                    ["offset", "knee", "exponent", "gof_rsquared"], dtype=object
                )
            }
        }
        params_tensor[:, :, 3, :] = 0.95
        return tfr_aperiodic, tfr_periodic, tfr_full_model, params_tensor, params_meta

    def fake_interp(
        tensor,
        metadata,
        *,
        freqs_out,
        axis,
        method,
        transform_mode,
        **kwargs,
    ):
        _ = kwargs
        assert axis == -2
        assert method == "linear"
        assert transform_mode == "dB"
        captured["interp_called"] = True
        captured["full_freqs"] = np.asarray(freqs_out, dtype=float)
        out = np.full(
            (tensor.shape[0], tensor.shape[1], len(freqs_out), tensor.shape[3]),
            2.0,
            dtype=float,
        )
        meta = dict(metadata)
        axes = dict(meta.get("axes", {}))
        axes["freq"] = np.asarray(freqs_out, dtype=float)
        axes["shape"] = out.shape
        meta["axes"] = axes
        return out, meta

    ok, _ = run_build_tensor(
        context,
        selected_metrics=["periodic_aperiodic"],
        low_freq=45.0,
        high_freq=55.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        metric_params_map={
            "periodic_aperiodic": {
                "notches": [50.0],
                "notch_widths": [2.0],
            }
        },
        service_overrides={
            "_run_periodic_aperiodic_metric": _metric_runner_override(
                tensor_service_module._run_periodic_aperiodic_metric,
                tfr_grid_fn=fake_tfr_grid,
                decompose_fn=fake_decompose,
                interpolate_freq_tensor_fn=fake_interp,
            )
        },
    )

    assert ok
    assert captured["compute_freqs"] is not None
    assert captured["interp_called"] is True
    assert 50.0 not in set(np.asarray(captured["compute_freqs"], dtype=float).tolist())
    assert int(np.asarray(captured["compute_freqs"]).size) < int(
        np.asarray(captured["full_freqs"]).size
    )
    tensor_payload = load_pkl(tensor_metric_tensor_path(resolver, "periodic_aperiodic"))
    assert tensor_payload["tensor"].shape[2] == int(
        np.asarray(captured["full_freqs"]).size
    )
    aperiodic_payload = load_pkl(tensor_metric_tensor_path(resolver, "aperiodic"))
    assert aperiodic_payload["tensor"].shape == (1, 2, 4, 20)
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "periodic_aperiodic").read_text(
            encoding="utf-8"
        )
    )
    assert config_payload["interpolation_applied"] is True
    assert config_payload["notches"] == [50.0]
    assert config_payload["notch_widths"] == [2.0]
    log_payload = read_run_log(tensor_metric_log_path(resolver, "periodic_aperiodic"))
    assert log_payload is not None
    assert log_payload["params"]["interpolation_applied"] is True
    assert log_payload["params"]["component"] == "periodic"
    assert log_payload["params"]["aperiodic_tensor_path"] == str(
        tensor_metric_tensor_path(resolver, "aperiodic")
    )


def test_run_build_tensor_coherence_success(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ["CH1", "CH2", "CH3"], sfreq=200.0, ch_types=["dbs", "dbs", "dbs"]
    )
    raw = mne.io.RawArray(np.zeros((3, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )
    filter_raw_path = preproc_step_raw_path(resolver, "filter")
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 1.0,
            "high_freq": 80.0,
            "notches": [50.0],
            "notch_widths": 2.0,
        },
        input_path=str(finish_raw_path),
        output_path=str(filter_raw_path),
        message="filter ready",
    )

    captured: dict[str, Any] = {}

    def fake_conn_grid(raw_obj, **kwargs):
        _ = raw_obj
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        pairs = list(kwargs["pairs"])
        captured["pairs"] = pairs
        tensor = np.full((1, len(pairs), freqs.size, 20), 0.5, dtype=float)
        metadata = {
            "axes": {
                "channel": np.asarray([f"{a}-{b}" for a, b in pairs], dtype=object),
                "freq": freqs.copy(),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": "coh"},
        }
        return tensor, metadata

    ok, message = run_build_tensor(
        context,
        selected_metrics=["coherence"],
        low_freq=2.0,
        high_freq=80.0,
        step_hz=2.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        service_overrides={
            "_run_undirected_connectivity_metric": _metric_runner_override(
                tensor_service_module._run_undirected_connectivity_metric,
                conn_grid_fn=fake_conn_grid,
            )
        },
    )

    assert ok
    assert "Coherence" in message
    assert captured["pairs"] == [("CH1", "CH2"), ("CH1", "CH3"), ("CH2", "CH3")]
    tensor_path = tensor_metric_tensor_path(resolver, "coherence")
    assert tensor_path.exists()
    tensor_payload = load_pkl(tensor_path)
    assert isinstance(tensor_payload, dict)
    assert tensor_payload["tensor"].shape == (1, 3, 40, 20)
    assert indicator_from_log(tensor_metric_log_path(resolver, "coherence")) == "green"
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "coherence").read_text(encoding="utf-8")
    )
    assert config_payload["method"] == "morlet"
    assert config_payload["connectivity_metric"] == "coh"
    assert config_payload["interpolation_applied"] is False


def test_run_build_tensor_plv_success_uses_plv_method(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ["CH1", "CH2", "CH3"], sfreq=200.0, ch_types=["dbs", "dbs", "dbs"]
    )
    raw = mne.io.RawArray(np.zeros((3, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 1.0,
            "high_freq": 80.0,
            "notches": [50.0],
            "notch_widths": 2.0,
        },
        input_path=str(finish_raw_path),
        output_path=str(preproc_step_raw_path(resolver, "filter")),
        message="filter ready",
    )

    captured: dict[str, Any] = {}

    def fake_conn_grid(raw_obj, **kwargs):
        _ = raw_obj
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        pairs = list(kwargs["pairs"])
        captured["pairs"] = pairs
        captured["method"] = kwargs["method"]
        tensor = np.full((1, len(pairs), freqs.size, 20), 0.5, dtype=float)
        metadata = {
            "axes": {
                "channel": np.asarray([f"{a}-{b}" for a, b in pairs], dtype=object),
                "freq": freqs.copy(),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": kwargs["method"]},
        }
        return tensor, metadata

    ok, message = run_build_tensor(
        context,
        selected_metrics=["plv"],
        low_freq=2.0,
        high_freq=80.0,
        step_hz=2.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        service_overrides={
            "_run_undirected_connectivity_metric": _metric_runner_override(
                tensor_service_module._run_undirected_connectivity_metric,
                conn_grid_fn=fake_conn_grid,
            )
        },
    )

    assert ok
    assert "PLV" in message
    assert captured["method"] == "plv"
    assert captured["pairs"] == [("CH1", "CH2"), ("CH1", "CH3"), ("CH2", "CH3")]
    tensor_path = tensor_metric_tensor_path(resolver, "plv")
    assert tensor_path.exists()
    tensor_payload = load_pkl(tensor_path)
    assert isinstance(tensor_payload, dict)
    assert tensor_payload["tensor"].shape == (1, 3, 40, 20)
    assert indicator_from_log(tensor_metric_log_path(resolver, "plv")) == "green"
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "plv").read_text(encoding="utf-8")
    )
    assert config_payload["method"] == "morlet"
    assert config_payload["connectivity_metric"] == "plv"
    assert config_payload["interpolation_applied"] is False


def test_run_build_tensor_trgc_success_derives_trgc_from_gc_and_gc_tr(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ["CH1", "CH2", "CH3"], sfreq=200.0, ch_types=["dbs", "dbs", "dbs"]
    )
    raw = mne.io.RawArray(np.zeros((3, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )

    captured: dict[str, Any] = {"calls": []}

    def fake_conn_grid(raw_obj, **kwargs):
        _ = raw_obj
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        pairs = list(kwargs["pairs"])
        method = str(kwargs["method"])
        captured["calls"].append(
            {
                "pairs": pairs,
                "method": method,
                "multivariate": kwargs["multivariate"],
                "ordered_pairs": kwargs["ordered_pairs"],
                "gc_n_lags": kwargs["gc_n_lags"],
                "group_by_samples": kwargs["group_by_samples"],
                "round_ms": kwargs["round_ms"],
            }
        )
        pair_values = {
            ("gc", ("CH1", "CH2")): 5.0,
            ("gc", ("CH2", "CH1")): 2.0,
            ("gc_tr", ("CH1", "CH2")): 4.0,
            ("gc_tr", ("CH2", "CH1")): 3.0,
        }
        tensor = np.empty((1, len(pairs), freqs.size, 20), dtype=float)
        for idx, pair in enumerate(pairs):
            tensor[:, idx, :, :] = pair_values[(method, pair)]
        metadata = {
            "axes": {
                "channel": list(pairs),
                "freq": freqs.copy(),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": method},
        }
        return tensor, metadata

    ok, message = run_build_tensor(
        context,
        selected_metrics=["trgc"],
        low_freq=2.0,
        high_freq=80.0,
        step_hz=2.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        selected_pairs={"trgc": [("CH1", "CH2")]},
        service_overrides={
            "_run_trgc_backend_metric": _metric_runner_override(
                tensor_service_module._run_trgc_backend_metric,
                conn_grid_fn=fake_conn_grid,
            )
        },
    )

    assert ok
    assert "TRGC" in message
    assert sorted(call["method"] for call in captured["calls"]) == ["gc", "gc_tr"]
    for call in captured["calls"]:
        assert call["multivariate"] is True
        assert call["ordered_pairs"] is True
        assert call["gc_n_lags"] == 20
        assert call["group_by_samples"] is False
        assert call["round_ms"] == 50.0
        assert call["pairs"] == [("CH1", "CH2"), ("CH2", "CH1")]
    tensor_path = tensor_metric_tensor_path(resolver, "trgc")
    assert tensor_path.exists()
    tensor_payload = load_pkl(tensor_path)
    assert isinstance(tensor_payload, dict)
    assert tensor_payload["tensor"].shape == (1, 1, 40, 20)
    assert np.allclose(tensor_payload["tensor"], 2.0)
    assert tensor_payload["meta"]["axes"]["channel"] == [("CH1", "CH2")]
    assert indicator_from_log(tensor_metric_log_path(resolver, "trgc")) == "green"
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "trgc").read_text(encoding="utf-8")
    )
    assert config_payload["method"] == "morlet"
    assert config_payload["connectivity_metric"] == "trgc"
    assert config_payload["backend_methods"] == ["gc", "gc_tr"]
    assert config_payload["gc_n_lags"] == 20
    assert config_payload["group_by_samples"] is False
    assert config_payload["round_ms"] == 50.0
    assert config_payload["selected_pairs"] == [["CH1", "CH2"]]
    assert config_payload["pairs_compute"] == [["CH1", "CH2"], ["CH2", "CH1"]]
    assert config_payload["interpolation_applied"] is False


def test_run_build_tensor_psi_success_uses_bands(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ["CH1", "CH2", "CH3"], sfreq=200.0, ch_types=["dbs", "dbs", "dbs"]
    )
    raw = mne.io.RawArray(np.zeros((3, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 1.0,
            "high_freq": 80.0,
            "notches": [50.0],
            "notch_widths": 2.0,
        },
        input_path=str(finish_raw_path),
        output_path=str(preproc_step_raw_path(resolver, "filter")),
        message="filter ready",
    )

    captured: dict[str, Any] = {}

    def fake_psi_grid(raw_obj, **kwargs):
        _ = raw_obj
        bands = dict(kwargs["bands"])
        pairs = list(kwargs["pairs"])
        captured["bands"] = bands
        captured["pairs"] = pairs
        captured["ordered_pairs"] = kwargs["ordered_pairs"]
        tensor = np.full((1, len(pairs), len(bands), 20), 0.5, dtype=float)
        metadata = {
            "axes": {
                "channel": np.asarray([f"{a}->{b}" for a, b in pairs], dtype=object),
                "freq": np.asarray(list(bands.keys()), dtype=object),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": "psi"},
        }
        return tensor, metadata

    ok, message = run_build_tensor(
        context,
        selected_metrics=["psi"],
        low_freq=40.0,
        high_freq=60.0,
        step_hz=2.0,
        mask_edge_effects=False,
        bands=[{"name": "alpha", "start": 45.0, "end": 55.0}],
        metric_params_map={
            "psi": {
                "notches": [50.0],
                "notch_widths": [2.0],
            }
        },
        service_overrides={
            "_run_psi_metric": _metric_runner_override(
                tensor_service_module._run_psi_metric,
                psi_grid_fn=fake_psi_grid,
            )
        },
    )

    assert ok
    assert "PSI" in message
    assert captured["ordered_pairs"] is True
    assert len(captured["pairs"]) == 6
    assert captured["bands"]["alpha"] == [(45.0, 48.0), (52.0, 55.0)]
    tensor_path = tensor_metric_tensor_path(resolver, "psi")
    assert tensor_path.exists()
    tensor_payload = load_pkl(tensor_path)
    assert isinstance(tensor_payload, dict)
    assert tensor_payload["tensor"].shape == (1, 6, len(captured["bands"]), 20)
    assert indicator_from_log(tensor_metric_log_path(resolver, "psi")) == "green"
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "psi").read_text(encoding="utf-8")
    )
    assert config_payload["method"] == "morlet"
    assert config_payload["connectivity_metric"] == "psi"
    assert config_payload["interpolation_applied"] is False
    assert config_payload["bands_used"]["alpha"] == [[45.0, 48.0], [52.0, 55.0]]


def test_run_build_tensor_burst_success_writes_thresholds(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ["CH1", "CH2", "CH3"], sfreq=200.0, ch_types=["dbs", "dbs", "dbs"]
    )
    raw = mne.io.RawArray(np.zeros((3, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )
    filter_raw_path = preproc_step_raw_path(resolver, "filter")
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        params={
            "low_freq": 1.0,
            "high_freq": 80.0,
            "notches": [50.0],
            "notch_widths": 2.0,
        },
        input_path=str(finish_raw_path),
        output_path=str(filter_raw_path),
        message="filter ready",
    )

    captured: dict[str, Any] = {}

    def fake_burst_grid(raw_obj, **kwargs):
        _ = raw_obj
        bands = dict(kwargs["bands"])
        picks = list(kwargs["picks"])
        captured["bands"] = bands
        captured["picks"] = picks
        captured["hop_s"] = kwargs["hop_s"]
        captured["decim"] = kwargs["decim"]
        tensor = np.full((1, len(picks), len(bands), 20), 0.5, dtype=float)
        metadata = {
            "axes": {
                "channel": np.asarray(picks, dtype=object),
                "freq": np.asarray(list(bands.keys()), dtype=object),
                "time": np.linspace(0.0, 1.0, 20, dtype=float),
                "shape": tensor.shape,
            },
            "params": {"method": "burst_grid"},
            "qc": {"thresholds": np.full((len(bands), len(picks)), 0.8, dtype=float)},
        }
        return tensor, metadata

    ok, message = run_build_tensor(
        context,
        selected_metrics=["burst"],
        low_freq=40.0,
        high_freq=60.0,
        step_hz=2.0,
        mask_edge_effects=True,
        bands=[{"name": "alpha", "start": 45.0, "end": 55.0}],
        metric_params_map={
            "burst": {
                "notches": [50.0],
                "notch_widths": [2.0],
            }
        },
        service_overrides={
            "_run_burst_metric": _metric_runner_override(
                tensor_service_module._run_burst_metric,
                burst_grid_fn=fake_burst_grid,
            )
        },
    )

    assert ok
    assert "Burst" in message
    assert captured["bands"]["alpha"] == [(45.0, 48.0), (52.0, 55.0)]
    assert captured["picks"] == ["CH1", "CH2", "CH3"]
    assert captured["hop_s"] is None
    assert captured["decim"] == 1
    tensor_path = tensor_metric_tensor_path(resolver, "burst")
    assert tensor_path.exists()
    tensor_payload = load_pkl(tensor_path)
    assert isinstance(tensor_payload, dict)
    assert tensor_payload["tensor"].shape == (1, 3, len(captured["bands"]), 20)
    metric_dir = resolver.tensor_metric_dir("burst", create=False)
    thresholds_path = metric_dir / "thresholds.pkl"
    assert thresholds_path.exists()
    assert np.asarray(load_pkl(thresholds_path)).shape == (len(captured["bands"]), 3)
    assert indicator_from_log(tensor_metric_log_path(resolver, "burst")) == "green"
    config_payload = yaml.safe_load(
        tensor_metric_config_path(resolver, "burst").read_text(encoding="utf-8")
    )
    assert config_payload["method"] == "burst_grid"
    assert config_payload["hop_s"] is None
    assert config_payload["decim"] == 1
    assert config_payload["bands_used"]["alpha"] == [[45.0, 48.0], [52.0, 55.0]]
    assert config_payload["thresholds_path"] is not None
    assert config_payload["interpolation_applied"] is False


def test_run_build_tensor_unknown_metric_marks_yellow(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )

    ok, _ = run_build_tensor(
        context,
        selected_metrics=["unknown_metric"],
        low_freq=2.0,
        high_freq=80.0,
        step_hz=2.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
    )

    assert not ok
    assert (
        indicator_from_log(tensor_metric_log_path(resolver, "unknown_metric"))
        == "yellow"
    )


def test_run_build_tensor_connectivity_without_pairs_blocks_run_even_with_old_green(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ["CH1", "CH2", "CH3"], sfreq=200.0, ch_types=["dbs", "dbs", "dbs"]
    )
    raw = mne.io.RawArray(np.zeros((3, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )
    old_log_path = tensor_metric_log_path(resolver, "coherence")
    write_run_log(
        old_log_path,
        RunLogRecord(
            step="coherence",
            completed=True,
            params={"legacy": True},
            input_path="in",
            output_path="out",
            message="old green log",
        ),
    )

    ok, message = run_build_tensor(
        context,
        selected_metrics=["coherence"],
        low_freq=2.0,
        high_freq=80.0,
        step_hz=2.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        selected_pairs={},
    )

    assert not ok
    assert "requires at least one selected pair" in message
    log_payload = read_run_log(old_log_path)
    assert log_payload is not None
    assert "requires at least one selected pair" in log_payload["message"]
    assert indicator_from_log(old_log_path) == "yellow"
    states = scan_stage_states(context.project_root, context.subject, context.record)
    assert states["tensor"] == "yellow"


def test_run_build_tensor_connectivity_without_pairs_blocks_run_no_old_green(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    import mne
    import numpy as np

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ["CH1", "CH2", "CH3"], sfreq=200.0, ch_types=["dbs", "dbs", "dbs"]
    )
    raw = mne.io.RawArray(np.zeros((3, 600), dtype=float), info)
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )

    metric_log_path = tensor_metric_log_path(resolver, "coherence")
    assert not resolver.tensor_root.exists()
    assert not metric_log_path.parent.exists()
    if metric_log_path.exists():
        metric_log_path.unlink()

    ok, message = run_build_tensor(
        context,
        selected_metrics=["coherence"],
        low_freq=2.0,
        high_freq=80.0,
        step_hz=2.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in DEFAULT_TENSOR_BANDS],
        selected_pairs={},
    )

    assert not ok
    assert "requires at least one selected pair" in message
    assert resolver.tensor_root.exists()
    assert metric_log_path.exists()
    assert indicator_from_log(metric_log_path) == "yellow"
    states = scan_stage_states(context.project_root, context.subject, context.record)
    assert states["tensor"] == "yellow"


def test_transform_df_supports_log10_nested_values() -> None:
    nested = pd.DataFrame(
        [[1.0, 10.0], [100.0, np.nan]],
        index=["f1", "f2"],
        columns=["t1", "t2"],
    )
    df = pd.DataFrame({"Value": [nested]})

    out = transform_df(df, mode="log10")
    value = out.loc[0, "Value"]

    assert isinstance(value, pd.DataFrame)
    np.testing.assert_allclose(
        value.to_numpy(dtype=float),
        np.array([[0.0, 1.0], [2.0, np.nan]], dtype=float),
        equal_nan=True,
    )


def test_transform_df_log10_marks_non_positive_values_as_nan() -> None:
    nested = pd.DataFrame(
        [[0.0, 10.0], [-3.0, np.nan]],
        index=["f1", "f2"],
        columns=["t1", "t2"],
    )
    df = pd.DataFrame({"Value": [nested]})

    out = transform_df(df, mode="log10")
    value = out.loc[0, "Value"]

    assert isinstance(value, pd.DataFrame)
    np.testing.assert_allclose(
        value.to_numpy(dtype=float),
        np.array([[np.nan, 1.0], [np.nan, np.nan]], dtype=float),
        equal_nan=True,
    )


def test_transform_df_zscore_is_rowwise_for_df_and_serieswise_for_series() -> None:
    nested_df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [5.0, 5.0, 5.0], [np.nan, 1.0, 2.0]],
        index=["r1", "r2", "r3"],
        columns=["c1", "c2", "c3"],
    )
    nested_series = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"], name="s")
    scalar = 5.0
    df = pd.DataFrame({"Value": [nested_df, nested_series, scalar]})

    out = transform_df(df, mode="zscore")

    value_df = out.loc[0, "Value"]
    value_series = out.loc[1, "Value"]
    value_scalar = out.loc[2, "Value"]

    assert isinstance(value_df, pd.DataFrame)
    assert isinstance(value_series, pd.Series)
    assert value_scalar == scalar

    np.testing.assert_allclose(
        value_df.loc["r1"].to_numpy(dtype=float),
        np.array([-1.22474487, 0.0, 1.22474487]),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        value_df.loc["r2"].to_numpy(dtype=float),
        np.array([np.nan, np.nan, np.nan]),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        value_df.loc["r3"].to_numpy(dtype=float),
        np.array([np.nan, -1.0, 1.0]),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        value_series.to_numpy(dtype=float),
        np.array([-1.22474487, 0.0, 1.22474487]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_apply_transform_array_returns_nan_for_domain_violations() -> None:
    np.testing.assert_allclose(
        apply_transform_array(np.array([0.0, 10.0, -1.0]), mode="dB"),
        np.array([np.nan, 10.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        apply_transform_array(np.array([-1.0, 0.0, 0.5, 1.0]), mode="logit"),
        np.array([np.nan, np.nan, 0.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        apply_transform_array(np.array([-1.0, 0.0, 0.5, 1.0]), mode="fisherz"),
        np.array([np.nan, 0.0, 0.54930614, np.nan]),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_numeric_transform_api_no_longer_exposes_legacy_eps_params() -> None:
    targets = [
        transform_df,
        normalize_df,
        baseline_normalize,
        smooth_axis,
        interpolate_freq_axis_transformed,
        interpolate_tensor_with_metadata_transformed,
        apply_transform_array,
        get_transform_pair,
    ]

    for target in targets:
        params = inspect.signature(target).parameters
        assert "eps" not in params
        assert "transform_eps" not in params


def test_baseline_normalize_returns_nan_when_denominator_is_zero() -> None:
    x = pd.DataFrame(
        [[0.0, 1.0, 2.0], [2.0, 4.0, 6.0]],
        index=["f1", "f2"],
        columns=["t1", "t2", "t3"],
    )

    out = baseline_normalize(
        x,
        baseline=slice(0, 1),
        mode="ratio",
    )

    assert isinstance(out, pd.DataFrame)
    np.testing.assert_allclose(
        out.loc["f1"].to_numpy(dtype=float),
        np.array([np.nan, np.nan, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out.loc["f2"].to_numpy(dtype=float),
        np.array([1.0, 2.0, 3.0]),
        equal_nan=True,
    )


def test_normalize_df_groupwise_zscore_returns_nan_for_zero_baseline_sd() -> None:
    df = pd.DataFrame(
        {
            "Group": ["A", "A"],
            "Condition": ["base", "task"],
            "Value": [
                pd.Series([1.0, 1.0, 1.0], index=["t1", "t2", "t3"]),
                pd.Series([1.0, 2.0, 3.0], index=["t1", "t2", "t3"]),
            ],
        }
    )

    out = normalize_df(
        df,
        group_cols="Group",
        baseline={"Condition": "base"},
        mode="zscore",
    )

    base_val = out.loc[0, "Value"]
    task_val = out.loc[1, "Value"]
    assert isinstance(base_val, pd.Series)
    assert isinstance(task_val, pd.Series)
    np.testing.assert_allclose(
        base_val.to_numpy(dtype=float),
        np.array([np.nan, np.nan, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        task_val.to_numpy(dtype=float),
        np.array([np.nan, np.nan, np.nan]),
        equal_nan=True,
    )


def test_normalize_df_ratio_uses_dynamic_tolerance_for_tiny_denominators() -> None:
    df = pd.DataFrame(
        {
            "Group": ["A", "A"],
            "Condition": ["base", "task"],
            "Value": [
                pd.Series([1e-12, 1.0, 1.0], index=["t1", "t2", "t3"]),
                pd.Series([2.0, 2.0, 3.0], index=["t1", "t2", "t3"]),
            ],
        }
    )

    out = normalize_df(
        df,
        group_cols="Group",
        baseline={"Condition": "base"},
        mode="ratio",
    )

    task_val = out.loc[1, "Value"]
    assert isinstance(task_val, pd.Series)
    np.testing.assert_allclose(
        task_val.to_numpy(dtype=float),
        np.array([np.nan, 2.0, 3.0]),
        equal_nan=True,
    )
