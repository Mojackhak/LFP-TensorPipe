"""Focused branch tests for tensor service helpers and preflight paths."""

from __future__ import annotations

import importlib
import inspect
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app import tensor_service
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import (
    preproc_step_log_path,
    preproc_step_raw_path,
)
from lfptensorpipe.app.tensor.frequency import TensorFilterInheritance
from lfptensorpipe.app.tensor import (
    orchestration_execution as orchestration_execution_module,
)
from lfptensorpipe.app.tensor.runners import connectivity_trgc as trgc_runner_module
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    indicator_from_log,
    read_run_log,
    write_run_log,
)
from lfptensorpipe.app.tensor.cancellation import (
    BUILD_TENSOR_CANCELLED_MESSAGE,
    backfill_cancelled_build_tensor_run,
)


def _context(project: Path) -> RecordContext:
    return RecordContext(project_root=project, subject="sub-001", record="runA")


def _mark_finish_green(resolver: PathResolver) -> None:
    finish_raw = preproc_step_raw_path(resolver, "finish")
    write_run_log(
        preproc_step_log_path(resolver, "finish"),
        RunLogRecord(
            step="finish",
            completed=True,
            params={},
            input_path=str(finish_raw),
            output_path=str(finish_raw),
            message="finish ready",
        ),
    )


def _run_metric(metric_key: str, context: RecordContext) -> tuple[bool, str]:
    bands = [dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS]
    if metric_key == "raw_power":
        return tensor_service._run_raw_power_metric(
            context,
            low_freq=2.0,
            high_freq=40.0,
            step_hz=1.0,
            mask_edge_effects=True,
            bands=bands,
            selected_channels=["CH1"],
        )
    if metric_key == "periodic_aperiodic":
        return tensor_service._run_periodic_aperiodic_metric(
            context,
            low_freq=2.0,
            high_freq=40.0,
            step_hz=1.0,
            mask_edge_effects=True,
            bands=bands,
            selected_channels=["CH1"],
        )
    if metric_key == "coherence":
        return tensor_service._run_undirected_connectivity_metric(
            context,
            metric_key="coherence",
            connectivity_metric="coh",
            low_freq=2.0,
            high_freq=40.0,
            step_hz=1.0,
            mask_edge_effects=True,
            bands=bands,
            selected_channels=["CH1", "CH2"],
            selected_pairs=None,
        )
    if metric_key == "trgc":
        return tensor_service._run_trgc_metric(
            context,
            low_freq=2.0,
            high_freq=40.0,
            step_hz=1.0,
            mask_edge_effects=True,
            bands=bands,
            selected_channels=["CH1", "CH2"],
            selected_pairs=None,
        )
    if metric_key == "psi":
        return tensor_service._run_psi_metric(
            context,
            low_freq=2.0,
            high_freq=40.0,
            step_hz=1.0,
            mask_edge_effects=True,
            bands=bands,
            selected_channels=["CH1", "CH2"],
            selected_pairs=None,
        )
    if metric_key == "burst":
        return tensor_service._run_burst_metric(
            context,
            low_freq=2.0,
            high_freq=40.0,
            step_hz=1.0,
            mask_edge_effects=True,
            bands=bands,
            selected_channels=["CH1"],
        )
    raise ValueError(f"Unhandled test metric key: {metric_key}")


def _prepare_finish_ready(context: RecordContext, resolver: PathResolver) -> Path:
    _mark_finish_green(resolver)
    finish_raw = preproc_step_raw_path(resolver, "finish")
    finish_raw.parent.mkdir(parents=True, exist_ok=True)
    finish_raw.write_text("fake", encoding="utf-8")
    return finish_raw


class _FakeRaw:
    def __init__(self, ch_names: tuple[str, ...], sfreq: float) -> None:
        self.ch_names = list(ch_names)
        self.info = {"sfreq": float(sfreq)}

    def close(self) -> None:
        return None


def _fake_read_raw(raw_obj: Any):
    def _reader(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        return raw_obj

    return _reader


def _metric_runner_override(runner, /, **dependencies: Any):
    def _wrapped(context: RecordContext, **kwargs: Any) -> tuple[bool, str]:
        return runner(context, **kwargs, **dependencies)

    return _wrapped


def test_load_burst_baseline_annotation_labels_filters_positive_durations(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)

    class _FakeAnnotationRaw:
        def __init__(self) -> None:
            self.annotations = type(
                "_Annotations",
                (),
                {
                    "duration": np.array([0.0, 0.5, 1.2, 0.8, -0.1], dtype=float),
                    "description": np.array(
                        ["zero", "Rest", "Task", "Rest", "neg"],
                        dtype=object,
                    ),
                },
            )()

        def close(self) -> None:
            return None

    labels = tensor_service.load_burst_baseline_annotation_labels(
        context,
        read_raw_fif_fn=lambda *args, **kwargs: _FakeAnnotationRaw(),
    )

    assert labels == ["Rest", "Task"]


def _periodic_prepared_input(
    *,
    raw: _FakeRaw,
    interpolation_applied: bool,
    freqs_compute: np.ndarray,
    freqs_model: np.ndarray,
    freqs_final: np.ndarray,
):
    periodic_models = importlib.import_module(
        "lfptensorpipe.app.tensor.runners.periodic_aperiodic_models"
    )
    return periodic_models.PeriodicAperiodicPreparedInput(
        raw=raw,
        picks=["CH1"],
        inheritance=TensorFilterInheritance(
            low_freq=1.0,
            high_freq=80.0,
            notches=(),
            notch_widths=(),
        ),
        runtime_notches=(),
        runtime_notch_widths=(),
        spec_low=float(freqs_model[0]),
        spec_high=float(freqs_model[-1]),
        freqs_model=np.asarray(freqs_model, dtype=float),
        freqs_final=np.asarray(freqs_final, dtype=float),
        freqs_compute=np.asarray(freqs_compute, dtype=float),
        notch_intervals=[],
        interpolation_applied=bool(interpolation_applied),
        method_norm="morlet",
    )


def _periodic_options(
    context: RecordContext,
    *,
    freq_smooth_enabled: bool = True,
    freq_smooth_sigma: float | None = 1.5,
    time_smooth_enabled: bool = True,
    time_smooth_kernel_size: int | None = 13,
):
    periodic_models = importlib.import_module(
        "lfptensorpipe.app.tensor.runners.periodic_aperiodic_models"
    )
    return periodic_models.PeriodicAperiodicOptions(
        context=context,
        low_freq=1.0,
        high_freq=3.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[],
        selected_channels=["CH1"],
        method="morlet",
        time_resolution_s=0.3,
        hop_s=0.025,
        min_cycles=3.0,
        max_cycles=13.0,
        time_bandwidth=1.0,
        freq_range_hz=(1.0, 3.0),
        freq_smooth_enabled=bool(freq_smooth_enabled),
        freq_smooth_sigma=freq_smooth_sigma,
        time_smooth_enabled=bool(time_smooth_enabled),
        time_smooth_kernel_size=time_smooth_kernel_size,
        aperiodic_mode="fixed",
        peak_width_limits_hz=(2.0, 12.0),
        max_n_peaks=8.0,
        min_peak_height=0.15,
        peak_threshold=2.0,
        fit_qc_threshold=0.6,
        notches=[],
        notch_widths=2.0,
        n_jobs=1,
        outer_n_jobs=1,
    )


def test_tensor_metric_paths_are_read_only_by_default(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    log_path = tensor_service.tensor_metric_log_path(resolver, "raw_power")
    tensor_path = tensor_service.tensor_metric_tensor_path(resolver, "raw_power")
    config_path = tensor_service.tensor_metric_config_path(resolver, "raw_power")

    assert log_path.name == "lfptensorpipe_log.json"
    assert tensor_path.name == "tensor.pkl"
    assert config_path.name == "config.yml"
    assert not resolver.tensor_root.exists()
    assert not log_path.parent.exists()

    periodic_alias_tensor = tensor_service.tensor_metric_tensor_path(
        resolver, "periodic_aperiodic"
    )
    periodic_alias_log = tensor_service.tensor_metric_log_path(
        resolver, "periodic_aperiodic"
    )
    aperiodic_tensor = tensor_service.tensor_metric_tensor_path(resolver, "aperiodic")
    assert periodic_alias_tensor.parent.name == "periodic"
    assert periodic_alias_log.parent.name == "periodic"
    assert aperiodic_tensor.parent.name == "aperiodic"


def test_write_outputs_atomically_rolls_back_on_commit_failure(tmp_path: Path) -> None:
    first = tmp_path / "a.pkl"
    second = tmp_path / "b.pkl"
    third = tmp_path / "c.yml"
    first.write_text("old-a", encoding="utf-8")
    second.write_text("old-b", encoding="utf-8")
    third.write_text("old-c", encoding="utf-8")

    def _replace_with_failure(source: Path, target: Path) -> Path:
        target_path = Path(target)
        if source.name.startswith(".b.pkl.tmp-") and target_path == second:
            raise OSError("inject commit failure")
        return source.replace(target_path)

    with pytest.raises(OSError, match="inject commit failure"):
        tensor_service._write_outputs_atomically(
            [
                (
                    first,
                    lambda out: out.write_text("new-a", encoding="utf-8"),
                ),
                (
                    second,
                    lambda out: out.write_text("new-b", encoding="utf-8"),
                ),
                (
                    third,
                    lambda out: out.write_text("new-c", encoding="utf-8"),
                ),
            ],
            replace_fn=_replace_with_failure,
        )

    assert first.read_text(encoding="utf-8") == "old-a"
    assert second.read_text(encoding="utf-8") == "old-b"
    assert third.read_text(encoding="utf-8") == "old-c"


def test_tensor_metric_panel_indicator_tracks_draft_staleness_and_restore(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=True,
            params={"low_freq": 1.0, "high_freq": 60.0, "notches": []},
            input_path="in",
            output_path="out",
            message="filter ready",
        ),
    )

    raw_power_log = {
        "low_freq": 1.0,
        "high_freq": 40.0,
        "step_hz": 1.0,
        "method": "morlet",
        "time_resolution_s": 0.5,
        "hop_s": 0.025,
        "min_cycles": 3.0,
        "max_cycles": None,
        "time_bandwidth": 1.0,
        "mask_edge_effects": True,
        "notches": [],
        "notch_widths": [],
        "selected_channels": ["CH1"],
    }
    write_run_log(
        tensor_service.tensor_metric_log_path(resolver, "raw_power"),
        RunLogRecord(
            step="run_build_tensor",
            completed=True,
            params=raw_power_log,
            input_path="in",
            output_path="out",
            message="raw power ready",
        ),
    )
    raw_power_params = {
        "low_freq_hz": 1.0,
        "high_freq_hz": 40.0,
        "freq_step_hz": 1.0,
        "method": "morlet",
        "time_resolution_s": 0.5,
        "hop_s": 0.025,
        "min_cycles": 3.0,
        "time_bandwidth": 1.0,
        "selected_channels": ["CH1"],
        "notches": [],
        "notch_widths": 2.0,
    }
    assert (
        tensor_service.tensor_metric_panel_state(
            context,
            metric_key="raw_power",
            metric_params=raw_power_params,
            mask_edge_effects=True,
        )
        == "green"
    )
    raw_power_params["low_freq_hz"] = "bad"
    assert (
        tensor_service.tensor_metric_panel_state(
            context,
            metric_key="raw_power",
            metric_params=raw_power_params,
            mask_edge_effects=True,
        )
        == "yellow"
    )
    raw_power_params["low_freq_hz"] = 1.0
    raw_power_params["low_freq_hz"] = 2.0
    assert (
        tensor_service.tensor_metric_panel_state(
            context,
            metric_key="raw_power",
            metric_params=raw_power_params,
            mask_edge_effects=True,
        )
        == "yellow"
    )
    raw_power_params["low_freq_hz"] = 1.0
    assert (
        tensor_service.tensor_metric_panel_state(
            context,
            metric_key="raw_power",
            metric_params=raw_power_params,
            mask_edge_effects=True,
        )
        == "green"
    )

    psi_log = {
        "low_freq": 1.0,
        "high_freq": 60.0,
        "step_hz": 0.5,
        "method": "morlet",
        "time_resolution_s": 0.5,
        "hop_s": 0.025,
        "mt_bandwidth": None,
        "min_cycles": 3.0,
        "max_cycles": None,
        "mask_edge_effects": True,
        "notches": [],
        "notch_widths": [],
        "bands_used": {"beta": [13.0, 30.0]},
        "selected_pairs": [["CH1", "CH2"]],
    }
    write_run_log(
        tensor_service.tensor_metric_log_path(resolver, "psi"),
        RunLogRecord(
            step="run_build_tensor",
            completed=True,
            params=psi_log,
            input_path="in",
            output_path="out",
            message="psi ready",
        ),
    )
    psi_params = {
        "method": "morlet",
        "time_resolution_s": 0.5,
        "hop_s": 0.025,
        "min_cycles": 3.0,
        "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
        "selected_pairs": [["CH1", "CH2"]],
        "notches": [],
        "notch_widths": 2.0,
    }
    assert (
        tensor_service.tensor_metric_panel_state(
            context,
            metric_key="psi",
            metric_params=psi_params,
            mask_edge_effects=True,
        )
        == "green"
    )
    psi_params["bands"] = [{"name": "beta", "start": 13.0, "end": 31.0}]
    assert (
        tensor_service.tensor_metric_panel_state(
            context,
            metric_key="psi",
            metric_params=psi_params,
            mask_edge_effects=True,
        )
        == "yellow"
    )
    write_run_log(
        tensor_service.tensor_metric_log_path(resolver, "psi"),
        RunLogRecord(
            step="run_build_tensor",
            completed=False,
            params=psi_log,
            input_path="in",
            output_path="out",
            message="psi failed",
        ),
    )
    assert (
        tensor_service.tensor_metric_panel_state(
            context,
            metric_key="psi",
            metric_params=psi_params,
            mask_edge_effects=True,
        )
        == "yellow"
    )


def test_backfill_cancelled_build_tensor_run_preserves_finished_metrics(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    finish_raw = _prepare_finish_ready(context, resolver)
    run_id = "run-123"

    write_run_log(
        tensor_service.tensor_metric_log_path(resolver, "raw_power"),
        RunLogRecord(
            step="raw_power",
            completed=True,
            params={"run_id": run_id, "selected_channels": ["CH1"]},
            input_path=str(finish_raw),
            output_path=str(
                tensor_service.tensor_metric_tensor_path(resolver, "raw_power")
            ),
            message="Raw power tensor computed.",
        ),
    )

    metric_statuses = backfill_cancelled_build_tensor_run(
        context,
        selected_metrics=["raw_power", "coherence"],
        metric_params_map={
            "raw_power": {"selected_channels": ["CH1"]},
            "coherence": {
                "selected_pairs": [["CH1", "CH2"]],
                "selected_channels": ["CH1", "CH2"],
            },
        },
        mask_edge_effects=True,
        run_id=run_id,
    )

    assert metric_statuses == {
        "raw_power": "success",
        "coherence": "cancelled",
    }
    raw_power_log = read_run_log(
        tensor_service.tensor_metric_log_path(resolver, "raw_power")
    )
    coherence_log = read_run_log(
        tensor_service.tensor_metric_log_path(resolver, "coherence")
    )
    stage_log = read_run_log(resolver.tensor_root / "lfptensorpipe_log.json")

    assert raw_power_log is not None and raw_power_log["completed"] is True
    assert coherence_log is not None and coherence_log["completed"] is False
    assert coherence_log["message"] == BUILD_TENSOR_CANCELLED_MESSAGE
    assert coherence_log["params"]["run_status"] == "cancelled"
    assert stage_log is not None and stage_log["completed"] is False
    assert stage_log["params"]["metric_statuses"]["raw_power"] == "success"
    assert stage_log["params"]["metric_statuses"]["coherence"] == "cancelled"


def test_tensor_helper_normalizers_and_grid_error_branches() -> None:
    assert tensor_service._as_float("bad", 3.0) == 3.0
    assert tensor_service._as_float(float("nan"), 3.0) == 3.0
    assert tensor_service._as_float("5.5", 3.0) == 5.5

    assert tensor_service._parse_positive_float_tuple(None) == ()
    assert tensor_service._parse_positive_float_tuple(["x", -1, 0, 2]) == (2.0,)

    assert tensor_service._expand_notch_widths([], 2) == (2.0, 2.0)
    assert tensor_service._expand_notch_widths([1.5], 3) == (1.5, 1.5, 1.5)
    assert tensor_service._expand_notch_widths([1.0, 2.0], 2) == (1.0, 2.0)
    assert tensor_service._expand_notch_widths([1.0, 2.0], 3) == (1.0, 1.0, 1.0)

    ok, message = tensor_service._validate_bands([])
    assert not ok and "At least one band" in message
    ok, message = tensor_service._validate_bands([{"name": "", "start": 1, "end": 2}])
    assert not ok and "empty name" in message
    ok, message = tensor_service._validate_bands(
        [{"name": "a", "start": 1, "end": 2}, {"name": "a", "start": 2, "end": 3}]
    )
    assert not ok and "Duplicate" in message
    ok, message = tensor_service._validate_bands(
        [{"name": "a", "start": "x", "end": 2}]
    )
    assert not ok and "invalid numeric" in message
    ok, message = tensor_service._validate_bands([{"name": "a", "start": 0, "end": 2}])
    assert not ok and "0 < start < end" in message
    ok, message = tensor_service._validate_bands(
        [{"name": "a", "start": 1, "end": 3}, {"name": "b", "start": 2, "end": 4}]
    )
    assert ok and message == ""

    with pytest.raises(ValueError, match="Low frequency"):
        tensor_service._build_frequency_grid(0.0, 10.0, 1.0)
    with pytest.raises(ValueError, match="greater than low"):
        tensor_service._build_frequency_grid(2.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="Step must be > 0"):
        tensor_service._build_frequency_grid(2.0, 4.0, 0.0)
    with pytest.raises(ValueError, match="at least two bins"):
        tensor_service._build_frequency_grid(2.0, 2.5, 2.0)
    with pytest.raises(ValueError, match="at least two unique bins"):
        tensor_service._build_frequency_grid(1.0, 1.0000004, 0.0000002)

    assert (
        tensor_service._normalize_selected_pairs(
            None, available_channels={"A", "B"}, directed=False
        )
        == []
    )
    with pytest.raises(ValueError, match="Invalid pair format"):
        tensor_service._normalize_selected_pairs(
            ["A-B"], available_channels={"A", "B"}, directed=False
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        tensor_service._normalize_selected_pairs(
            [(" ", "B")], available_channels={"A", "B"}, directed=False
        )
    with pytest.raises(ValueError, match="Self-pairs"):
        tensor_service._normalize_selected_pairs(
            [("A", "A")], available_channels={"A", "B"}, directed=False
        )
    with pytest.raises(ValueError, match="unknown channel"):
        tensor_service._normalize_selected_pairs(
            [("A", "C")], available_channels={"A", "B"}, directed=False
        )

    assert tensor_service._normalize_selected_pairs(
        [("B", "A"), ("A", "B")],
        available_channels={"A", "B"},
        directed=False,
    ) == [("A", "B")]
    assert tensor_service._normalize_selected_pairs(
        [("A", "B"), ("A", "B")],
        available_channels={"A", "B"},
        directed=True,
    ) == [("A", "B")]


def test_connectivity_grid_gc_pad_scale_default_and_cap() -> None:
    conn_grid_module = importlib.import_module("lfptensorpipe.lfp.connectivity.grid")

    signature = inspect.signature(conn_grid_module.grid)
    assert signature.parameters["gc_pad_scale"].default == 3.0

    assert conn_grid_module._gc_target_freqs(20, 3.0, 40) == 36
    assert conn_grid_module._gc_target_freqs(20, 3.0, 20) == 20
    with pytest.raises(ValueError, match="Need at least 12 frequency bins"):
        conn_grid_module._gc_target_freqs(20, 3.0, 10)
    assert tensor_service._compute_notch_intervals(
        low_freq=1.0,
        high_freq=5.0,
        notches=(50.0, 3.0),
        notch_widths=(1.0, 0.5),
    ) == [(2.5, 3.5)]


def test_tensor_filter_inheritance_and_bounds_fallback_branches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    _mark_finish_green(resolver)
    finish_raw = preproc_step_raw_path(resolver, "finish")
    finish_raw.parent.mkdir(parents=True, exist_ok=True)
    finish_raw.write_text("fake", encoding="utf-8")

    class _Raw:
        info = {"sfreq": 0.0}

        @staticmethod
        def close() -> None:
            return None

    assert (
        tensor_service._load_finish_nyquist_hz(
            context,
            read_raw_fif_fn=lambda *args, **kwargs: _Raw(),
        )
        is None
    )

    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=True,
            params={
                "low_freq": 20.0,
                "high_freq": 10.0,
                "notches": [50],
                "notch_widths": [],
            },
            input_path="in",
            output_path="out",
            message="ready",
        ),
    )
    inheritance = tensor_service.load_tensor_filter_inheritance(context)
    assert inheritance.low_freq == 20.0
    assert inheritance.high_freq == 21.0

    bounds = tensor_service.resolve_tensor_frequency_bounds(
        context,
        load_finish_nyquist_hz_fn=lambda _ctx: 5.0,
    )
    assert bounds.min_low_freq == 20.0
    assert bounds.max_high_freq == 21.0

    low, high, step = tensor_service.load_tensor_frequency_defaults(
        context,
        resolve_tensor_frequency_bounds_fn=lambda _ctx: tensor_service.TensorFrequencyBounds(
            min_low_freq=10.0,
            max_high_freq=10.0,
            filter_low_freq=10.0,
            filter_high_freq=10.0,
            nyquist_freq=None,
        ),
    )
    assert (low, high, step) == (10.0, 11.0, 0.5)


def test_validate_tensor_frequency_params_remaining_error_branches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)

    def resolve_bounds(_ctx):
        return tensor_service.TensorFrequencyBounds(
            min_low_freq=5.0,
            max_high_freq=20.0,
            filter_low_freq=5.0,
            filter_high_freq=20.0,
            nyquist_freq=None,
        )

    ok, message, _ = tensor_service.validate_tensor_frequency_params(
        context,
        low_freq=0.0,
        high_freq=10.0,
        step_hz=1.0,
        resolve_tensor_frequency_bounds_fn=resolve_bounds,
    )
    assert not ok and "Low freq must be > 0" in message

    ok, message, _ = tensor_service.validate_tensor_frequency_params(
        context,
        low_freq=6.0,
        high_freq=30.0,
        step_hz=1.0,
        resolve_tensor_frequency_bounds_fn=resolve_bounds,
    )
    assert not ok and "preproc filter high freq" in message

    ok, message, _ = tensor_service.validate_tensor_frequency_params(
        context,
        low_freq=10.0,
        high_freq=10.0,
        step_hz=1.0,
        resolve_tensor_frequency_bounds_fn=resolve_bounds,
    )
    assert not ok and "greater than Low freq" in message

    ok, message, _ = tensor_service.validate_tensor_frequency_params(
        context,
        low_freq=10.0,
        high_freq=11.0,
        step_hz=0.0,
        resolve_tensor_frequency_bounds_fn=resolve_bounds,
    )
    assert not ok and "Step must be > 0" in message

    ok, message, _ = tensor_service.validate_tensor_frequency_params(
        context,
        low_freq=10.0,
        high_freq=10.5,
        step_hz=2.0,
        resolve_tensor_frequency_bounds_fn=resolve_bounds,
    )
    assert not ok and "at least two bins" in message


@pytest.mark.parametrize(
    "metric_key",
    ["raw_power", "periodic_aperiodic", "coherence", "trgc", "psi", "burst"],
)
def test_metric_runners_fail_when_finish_not_green(
    tmp_path: Path,
    metric_key: str,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    ok, message = _run_metric(metric_key, context)

    assert not ok
    assert "Missing green preproc finish log" in message
    assert (
        indicator_from_log(tensor_service.tensor_metric_log_path(resolver, metric_key))
        == "yellow"
    )


@pytest.mark.parametrize(
    "metric_key",
    ["raw_power", "periodic_aperiodic", "coherence", "trgc", "psi", "burst"],
)
def test_metric_runners_fail_when_finish_raw_missing(
    tmp_path: Path,
    metric_key: str,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _mark_finish_green(resolver)

    ok, message = _run_metric(metric_key, context)

    assert not ok
    assert "Missing preproc finish raw input" in message
    assert (
        indicator_from_log(tensor_service.tensor_metric_log_path(resolver, metric_key))
        == "yellow"
    )


@pytest.mark.parametrize(
    "metric_key",
    ["raw_power", "periodic_aperiodic", "coherence", "trgc", "psi", "burst"],
)
def test_metric_runners_write_yellow_log_on_runtime_exception(
    tmp_path: Path,
    metric_key: str,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _mark_finish_green(resolver)
    _prepare_finish_ready(context, resolver)

    service_overrides: dict[str, object]
    metric_params_map: dict[str, dict[str, object]]
    if metric_key == "raw_power":
        service_overrides = {
            "_run_raw_power_metric": lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
        }
        metric_params_map = {
            "raw_power": {
                "low_freq_hz": 3.0,
                "high_freq_hz": 30.0,
                "freq_step_hz": 0.5,
                "selected_channels": ["CH1"],
            }
        }
    elif metric_key == "periodic_aperiodic":
        service_overrides = {
            "_run_periodic_aperiodic_metric": lambda *args, **kwargs: (
                _ for _ in ()
            ).throw(RuntimeError("boom"))
        }
        metric_params_map = {
            "periodic_aperiodic": {
                "low_freq_hz": 3.0,
                "high_freq_hz": 30.0,
                "freq_step_hz": 0.5,
                "selected_channels": ["CH1"],
                "freq_range_hz": [3.0, 30.0],
            }
        }
    elif metric_key == "coherence":
        service_overrides = {
            "_run_undirected_connectivity_metric": lambda *args, **kwargs: (
                _ for _ in ()
            ).throw(RuntimeError("boom"))
        }
        metric_params_map = {
            "coherence": {
                "low_freq_hz": 10.0,
                "high_freq_hz": 60.0,
                "freq_step_hz": 1.0,
                "selected_pairs": [["CH1", "CH2"]],
            }
        }
    elif metric_key == "trgc":
        service_overrides = {
            "_run_trgc_backend_metric": lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
        }
        metric_params_map = {
            "trgc": {
                "low_freq_hz": 10.0,
                "high_freq_hz": 60.0,
                "freq_step_hz": 1.0,
                "selected_pairs": [["CH1", "CH2"]],
            }
        }
    elif metric_key == "psi":
        service_overrides = {
            "_run_psi_metric": lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
        }
        metric_params_map = {
            "psi": {
                "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
                "selected_pairs": [["CH1", "CH2"]],
            }
        }
    else:
        service_overrides = {
            "_run_burst_metric": lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            )
        }
        metric_params_map = {
            "burst": {
                "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
                "selected_channels": ["CH1"],
            }
        }

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=[metric_key],
        mask_edge_effects=True,
        service_overrides=service_overrides,
        metric_params_map=metric_params_map,
    )

    assert not ok
    if metric_key == "trgc":
        assert "blocked by failed dependency" in message.lower()
    else:
        assert "unexpected runtime failure" in message.lower()
        assert "boom" in message
    assert (
        indicator_from_log(tensor_service.tensor_metric_log_path(resolver, metric_key))
        == "yellow"
    )


def test_run_build_tensor_top_level_remaining_branches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    bands = [dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS]

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=[],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
    )
    assert not ok and "No tensor metric selected" in message

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power"],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
    )
    assert not ok and "Preprocess finish must be green" in message

    _prepare_finish_ready(context, resolver)
    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["psi"],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[],
    )
    assert not ok and "At least one band" in message

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power"],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        service_overrides={
            "validate_tensor_frequency_params": lambda *args, **kwargs: (
                False,
                "freq invalid",
                None,
            )
        },
    )
    assert not ok and "freq invalid" in message

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["unknown_metric"],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
    )
    assert not ok and "unknown metric" in message

    unsupported_map = dict(tensor_service.TENSOR_METRICS_BY_KEY)
    unsupported_map["raw_power"] = tensor_service.TensorMetricSpec(
        "raw_power",
        "Raw power",
        "Power",
        supported=False,
    )
    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power"],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        service_overrides={"TENSOR_METRICS_BY_KEY": unsupported_map},
    )
    assert not ok and "not implemented" in message

    write_run_log(
        tensor_service.tensor_metric_log_path(resolver, "raw_power"),
        RunLogRecord(
            step="raw_power",
            completed=False,
            params={},
            input_path="in",
            output_path="out",
            message="yellow",
        ),
    )
    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power"],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=[],
    )
    assert not ok and "requires at least one selected channel" in message

    captured: dict[str, object] = {}
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

    def _fake_undirected(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        captured["pairs"] = kwargs.get("selected_pairs")
        return True, "ok"

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["coherence"],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_pairs={"coherence": [("CH1", "CH2")]},
        service_overrides={"_run_undirected_connectivity_metric": _fake_undirected},
    )
    assert ok
    assert captured["pairs"] == [("CH1", "CH2")]
    assert "Coherence: ok" in message
    assert indicator_from_log(alignment_log) == "yellow"
    assert indicator_from_log(features_log) == "yellow"

    custom_metric_map = dict(tensor_service.TENSOR_METRICS_BY_KEY)
    custom_metric_map["custom_metric"] = tensor_service.TensorMetricSpec(
        "custom_metric",
        "Custom",
        "Power",
        supported=True,
    )
    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["custom_metric"],
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        service_overrides={"TENSOR_METRICS_BY_KEY": custom_metric_map},
    )
    assert not ok and "unsupported handler" in message


def test_run_build_tensor_prefers_metric_params_map(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)

    captured: dict[str, tuple[float, float, float]] = {}

    def _fake_raw_power(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        captured["raw_power"] = (
            float(kwargs["low_freq"]),
            float(kwargs["high_freq"]),
            float(kwargs["step_hz"]),
        )
        return True, "ok"

    def _fake_coherence(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        captured["coherence"] = (
            float(kwargs["low_freq"]),
            float(kwargs["high_freq"]),
            float(kwargs["step_hz"]),
        )
        return True, "ok"

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power", "coherence"],
        mask_edge_effects=True,
        service_overrides={
            "_run_raw_power_metric": _fake_raw_power,
            "_run_undirected_connectivity_metric": _fake_coherence,
        },
        metric_params_map={
            "raw_power": {
                "low_freq_hz": 3.0,
                "high_freq_hz": 30.0,
                "freq_step_hz": 0.5,
                "selected_channels": ["CH1", "CH2"],
            },
            "coherence": {
                "low_freq_hz": 10.0,
                "high_freq_hz": 60.0,
                "freq_step_hz": 1.0,
                "selected_pairs": [["CH1", "CH2"]],
            },
        },
    )

    assert ok
    assert "Raw power: ok" in message
    assert "Coherence: ok" in message
    assert captured["raw_power"] == (3.0, 30.0, 0.5)
    assert captured["coherence"] == (10.0, 60.0, 1.0)


def test_run_build_tensor_single_runnable_uses_negative_one_jobs(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    captured: dict[str, tuple[int, int]] = {}

    def _fake_raw_power(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        captured["raw_power"] = (
            int(kwargs.get("n_jobs", 0)),
            int(kwargs.get("outer_n_jobs", 0)),
        )
        return True, "ok"

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power"],
        mask_edge_effects=True,
        service_overrides={"_run_raw_power_metric": _fake_raw_power},
        metric_params_map={
            "raw_power": {
                "low_freq_hz": 3.0,
                "high_freq_hz": 30.0,
                "freq_step_hz": 0.5,
                "selected_channels": ["CH1"],
            }
        },
    )

    assert ok
    assert "Raw power: ok" in message
    assert captured["raw_power"] == (-1, -1)

    stage_log = read_run_log(tensor_service.tensor_stage_log_path(resolver))
    assert isinstance(stage_log, dict)
    effective = (
        stage_log.get("params", {}).get("effective_n_jobs", {}).get("raw_power", {})
    )
    assert effective == {"n_jobs": -1, "outer_n_jobs": -1}


def test_run_build_tensor_multi_runnable_parallel_and_internal_jobs_one(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    lock = threading.Lock()
    active = {"count": 0}
    max_active = {"count": 0}
    captured: dict[str, tuple[int, int]] = {}

    def _mark_enter() -> None:
        with lock:
            active["count"] += 1
            max_active["count"] = max(max_active["count"], active["count"])

    def _mark_leave() -> None:
        with lock:
            active["count"] -= 1

    def _fake_raw_power(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _mark_enter()
        try:
            time.sleep(0.05)
            captured["raw_power"] = (
                int(kwargs.get("n_jobs", 0)),
                int(kwargs.get("outer_n_jobs", 0)),
            )
            return True, "ok"
        finally:
            _mark_leave()

    def _fake_coherence(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _mark_enter()
        try:
            time.sleep(0.05)
            captured["coherence"] = (
                int(kwargs.get("n_jobs", 0)),
                int(kwargs.get("outer_n_jobs", 0)),
            )
            return True, "ok"
        finally:
            _mark_leave()

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power", "coherence"],
        mask_edge_effects=True,
        service_overrides={
            "_run_raw_power_metric": _fake_raw_power,
            "_run_undirected_connectivity_metric": _fake_coherence,
        },
        metric_params_map={
            "raw_power": {
                "low_freq_hz": 3.0,
                "high_freq_hz": 30.0,
                "freq_step_hz": 0.5,
                "selected_channels": ["CH1"],
            },
            "coherence": {
                "low_freq_hz": 10.0,
                "high_freq_hz": 60.0,
                "freq_step_hz": 1.0,
                "selected_pairs": [["CH1", "CH2"]],
            },
        },
    )

    assert ok
    assert "Raw power: ok" in message
    assert "Coherence: ok" in message
    assert captured["raw_power"] == (1, 1)
    assert captured["coherence"] == (1, 1)
    assert max_active["count"] >= 2

    stage_log = read_run_log(tensor_service.tensor_stage_log_path(resolver))
    assert isinstance(stage_log, dict)
    effective = stage_log.get("params", {}).get("effective_n_jobs", {})
    assert effective.get("raw_power") == {"n_jobs": 1, "outer_n_jobs": 1}
    assert effective.get("coherence") == {"n_jobs": 1, "outer_n_jobs": 1}


def test_run_build_tensor_single_trgc_expands_to_backend_and_finalize_plans(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    lock = threading.Lock()
    active = {"count": 0}
    max_active = {"count": 0}
    completed: list[str] = []
    captured_backend_jobs: dict[str, tuple[int, int, bool, float]] = {}
    captured_finalize_jobs: dict[str, int] = {}

    def _mark_enter() -> None:
        with lock:
            active["count"] += 1
            max_active["count"] = max(max_active["count"], active["count"])

    def _mark_leave(label: str) -> None:
        with lock:
            completed.append(label)
            active["count"] -= 1

    def _fake_trgc_backend(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        backend_method = str(kwargs["backend_method"])
        _mark_enter()
        try:
            time.sleep(0.05)
            captured_backend_jobs[backend_method] = (
                int(kwargs.get("n_jobs", 0)),
                int(kwargs.get("outer_n_jobs", 0)),
                bool(kwargs.get("group_by_samples", True)),
                float(kwargs.get("round_ms", -1.0)),
            )
            return True, f"{backend_method} ok"
        finally:
            _mark_leave(backend_method)

    def _fake_trgc_finalize(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        captured_finalize_jobs["n_jobs"] = int(kwargs.get("n_jobs", 0))
        captured_finalize_jobs["outer_n_jobs"] = int(kwargs.get("outer_n_jobs", 0))
        assert sorted(completed) == ["gc", "gc_tr"]
        return True, "trgc ok"

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["trgc"],
        mask_edge_effects=True,
        service_overrides={
            "_run_trgc_backend_metric": _fake_trgc_backend,
            "_run_trgc_finalize_metric": _fake_trgc_finalize,
        },
        metric_params_map={
            "trgc": {
                "low_freq_hz": 3.0,
                "high_freq_hz": 30.0,
                "freq_step_hz": 0.5,
                "selected_pairs": [["CH1", "CH2"]],
                "group_by_samples": True,
                "round_ms": 75.0,
            }
        },
    )

    assert ok
    assert "TRGC: trgc ok" in message
    assert captured_backend_jobs == {
        "gc": (1, 1, True, 75.0),
        "gc_tr": (1, 1, True, 75.0),
    }
    assert captured_finalize_jobs == {"n_jobs": 1, "outer_n_jobs": 1}
    assert max_active["count"] >= 2

    stage_log = read_run_log(tensor_service.tensor_stage_log_path(resolver))
    assert isinstance(stage_log, dict)
    effective = stage_log.get("params", {}).get("effective_n_jobs", {})
    assert effective.get("trgc") == {"n_jobs": 1, "outer_n_jobs": 1}


def test_run_build_tensor_trgc_backends_share_top_level_phase_with_other_metrics(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    lock = threading.Lock()
    active = {"count": 0}
    max_active = {"count": 0}
    completed: list[str] = []
    captured: dict[str, tuple[int, int]] = {}

    def _mark_enter() -> None:
        with lock:
            active["count"] += 1
            max_active["count"] = max(max_active["count"], active["count"])

    def _mark_leave(label: str) -> None:
        with lock:
            completed.append(label)
            active["count"] -= 1

    def _fake_raw_power(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _mark_enter()
        try:
            time.sleep(0.05)
            captured["raw_power"] = (
                int(kwargs.get("n_jobs", 0)),
                int(kwargs.get("outer_n_jobs", 0)),
            )
            return True, "raw ok"
        finally:
            _mark_leave("raw_power")

    def _fake_trgc_backend(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        backend_method = str(kwargs["backend_method"])
        _mark_enter()
        try:
            time.sleep(0.05)
            captured[backend_method] = (
                int(kwargs.get("n_jobs", 0)),
                int(kwargs.get("outer_n_jobs", 0)),
            )
            return True, f"{backend_method} ok"
        finally:
            _mark_leave(backend_method)

    def _fake_trgc_finalize(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        assert sorted(completed) == ["gc", "gc_tr", "raw_power"]
        captured["trgc_finalize"] = (
            int(kwargs.get("n_jobs", 0)),
            int(kwargs.get("outer_n_jobs", 0)),
        )
        return True, "trgc ok"

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power", "trgc"],
        mask_edge_effects=True,
        service_overrides={
            "_run_raw_power_metric": _fake_raw_power,
            "_run_trgc_backend_metric": _fake_trgc_backend,
            "_run_trgc_finalize_metric": _fake_trgc_finalize,
        },
        metric_params_map={
            "raw_power": {
                "low_freq_hz": 3.0,
                "high_freq_hz": 30.0,
                "freq_step_hz": 0.5,
                "selected_channels": ["CH1"],
            },
            "trgc": {
                "low_freq_hz": 10.0,
                "high_freq_hz": 60.0,
                "freq_step_hz": 1.0,
                "selected_pairs": [["CH1", "CH2"]],
            },
        },
    )

    assert ok
    assert "Raw power: raw ok" in message
    assert "TRGC: trgc ok" in message
    assert captured["raw_power"] == (1, 1)
    assert captured["gc"] == (1, 1)
    assert captured["gc_tr"] == (1, 1)
    assert captured["trgc_finalize"] == (1, 1)
    assert max_active["count"] >= 3

    stage_log = read_run_log(tensor_service.tensor_stage_log_path(resolver))
    assert isinstance(stage_log, dict)
    effective = stage_log.get("params", {}).get("effective_n_jobs", {})
    assert effective.get("raw_power") == {"n_jobs": 1, "outer_n_jobs": 1}
    assert effective.get("trgc") == {"n_jobs": 1, "outer_n_jobs": 1}


def test_run_build_tensor_counts_only_valid_runnable_metrics_for_jobs_policy(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)

    captured: dict[str, tuple[int, int]] = {}

    def _fake_raw_power(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        captured["raw_power"] = (
            int(kwargs.get("n_jobs", 0)),
            int(kwargs.get("outer_n_jobs", 0)),
        )
        return True, "ok"

    ok, message = tensor_service.run_build_tensor(
        context,
        selected_metrics=["raw_power", "psi"],
        mask_edge_effects=True,
        service_overrides={"_run_raw_power_metric": _fake_raw_power},
        metric_params_map={
            "raw_power": {
                "low_freq_hz": 3.0,
                "high_freq_hz": 30.0,
                "freq_step_hz": 0.5,
                "selected_channels": ["CH1"],
            },
            "psi": {
                "bands": [],
                "time_resolution_s": 0.5,
                "hop_s": 0.025,
                "selected_pairs": [["CH1", "CH2"]],
            },
        },
    )

    assert not ok
    assert "PSI: At least one band is required." in message
    assert "Raw power: ok" in message
    assert captured["raw_power"] == (-1, -1)

    stage_log = read_run_log(tensor_service.tensor_stage_log_path(resolver))
    assert isinstance(stage_log, dict)
    effective = stage_log.get("params", {}).get("effective_n_jobs", {})
    assert effective.get("raw_power") == {"n_jobs": -1, "outer_n_jobs": -1}
    assert effective.get("psi") == {"n_jobs": 1, "outer_n_jobs": 1}


def test_runtime_plan_payload_round_trip() -> None:
    runtime_plan = orchestration_execution_module.RuntimePlan(
        plan_key="trgc_gc_backend",
        metric_label="TRGC GC Backend",
        runner_key="trgc_backend",
        runner_kwargs={
            "backend_method": "gc",
            "selected_pairs": [("CH1", "CH2")],
            "group_by_samples": True,
        },
        phase=1,
        dependencies=("raw_power", "coherence"),
        log_metric_key="trgc",
    )

    restored = orchestration_execution_module.RuntimePlan.from_payload(
        runtime_plan.to_payload()
    )

    assert restored == runtime_plan


def test_runtime_plan_worker_env_clamps_native_threads() -> None:
    env = orchestration_execution_module._runtime_plan_worker_env(
        {"PYTHONPATH": "/tmp/custom", "OMP_NUM_THREADS": "8"}
    )

    assert "/tmp/custom" in env["PYTHONPATH"]
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"


def test_execute_runtime_plans_uses_process_phase_executor_for_multi_plan() -> None:
    context = RecordContext(
        project_root=Path("/tmp/project"),
        subject="sub-001",
        record="runA",
    )
    runtime_plans = {
        "raw_power": orchestration_execution_module.RuntimePlan(
            plan_key="raw_power",
            metric_label="Raw power",
            runner_key="raw_power",
            runner_kwargs={"selected_channels": ["CH1"]},
        ),
        "coherence": orchestration_execution_module.RuntimePlan(
            plan_key="coherence",
            metric_label="Coherence",
            runner_key="undirected_connectivity",
            runner_kwargs={"selected_pairs": [("CH1", "CH2")]},
        ),
    }
    calls: list[dict[str, Any]] = []

    def _fake_process_phase_executor(  # noqa: ANN001
        svc,
        resolver,
        context_arg,
        *,
        runtime_plans,
        merged_metric_params_map,
        runnable_keys,
        policy_n_jobs,
        policy_outer_n_jobs,
    ):
        _ = (svc, resolver, runtime_plans, merged_metric_params_map)
        calls.append(
            {
                "context": context_arg,
                "runnable_keys": list(runnable_keys),
                "n_jobs": int(policy_n_jobs),
                "outer_n_jobs": int(policy_outer_n_jobs),
            }
        )
        return {
            metric_key: (
                True,
                f"{metric_key} ok",
                runtime_plans[metric_key].metric_label,
            )
            for metric_key in runnable_keys
        }

    runtime_results = orchestration_execution_module.execute_runtime_plans(
        object(),
        object(),
        context,
        runtime_plans=runtime_plans,
        merged_metric_params_map={},
        policy_n_jobs=1,
        policy_outer_n_jobs=1,
        process_phase_executor=_fake_process_phase_executor,
    )

    assert len(calls) == 1
    assert calls[0]["context"] == context
    assert set(calls[0]["runnable_keys"]) == {"raw_power", "coherence"}
    assert calls[0]["n_jobs"] == 1
    assert calls[0]["outer_n_jobs"] == 1
    assert runtime_results["raw_power"] == (True, "raw_power ok", "Raw power")
    assert runtime_results["coherence"] == (True, "coherence ok", "Coherence")


def test_execute_runtime_plans_allows_windows_periodic_aperiodic_process_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RecordContext(
        project_root=Path("/tmp/project"),
        subject="sub-001",
        record="runA",
    )
    runtime_plans = {
        "raw_power": orchestration_execution_module.RuntimePlan(
            plan_key="raw_power",
            metric_label="Raw power",
            runner_key="raw_power",
            runner_kwargs={"selected_channels": ["CH1"]},
        ),
        "periodic_aperiodic": orchestration_execution_module.RuntimePlan(
            plan_key="periodic_aperiodic",
            metric_label="Periodic/APeriodic",
            runner_key="periodic_aperiodic",
            runner_kwargs={"selected_channels": ["CH1"]},
        ),
    }
    process_calls: list[list[str]] = []
    in_process_calls: list[str] = []

    def _fake_process_phase_executor(  # noqa: ANN001
        svc,
        resolver,
        context_arg,
        *,
        runtime_plans,
        merged_metric_params_map,
        runnable_keys,
        policy_n_jobs,
        policy_outer_n_jobs,
    ):
        _ = (
            svc,
            resolver,
            context_arg,
            runtime_plans,
            merged_metric_params_map,
            policy_n_jobs,
            policy_outer_n_jobs,
        )
        process_calls.append(list(runnable_keys))
        return {
            metric_key: (
                True,
                f"{metric_key} ok",
                runtime_plans[metric_key].metric_label,
            )
            for metric_key in runnable_keys
        }

    def _fake_run_runtime_plan(  # noqa: ANN001
        svc,
        resolver,
        context_arg,
        *,
        runtime_plan,
        merged_metric_params_map,
        policy_n_jobs,
        policy_outer_n_jobs,
    ):
        _ = (
            svc,
            resolver,
            context_arg,
            merged_metric_params_map,
            policy_n_jobs,
            policy_outer_n_jobs,
        )
        in_process_calls.append(runtime_plan.plan_key)
        return True, f"{runtime_plan.plan_key} ok", runtime_plan.metric_label

    monkeypatch.setattr(orchestration_execution_module.sys, "platform", "win32")
    monkeypatch.setattr(
        orchestration_execution_module,
        "run_runtime_plan",
        _fake_run_runtime_plan,
    )

    runtime_results = orchestration_execution_module.execute_runtime_plans(
        object(),
        object(),
        context,
        runtime_plans=runtime_plans,
        merged_metric_params_map={},
        policy_n_jobs=1,
        policy_outer_n_jobs=1,
        process_phase_executor=_fake_process_phase_executor,
    )

    assert len(process_calls) == 1
    assert set(process_calls[0]) == {"raw_power", "periodic_aperiodic"}
    assert in_process_calls == []
    assert runtime_results["raw_power"] == (True, "raw_power ok", "Raw power")
    assert runtime_results["periodic_aperiodic"] == (
        True,
        "periodic_aperiodic ok",
        "Periodic/APeriodic",
    )


def test_raw_power_remaining_failure_shape_branches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)

    ok, message = tensor_service._run_raw_power_metric(
        context,
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["MISSING"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 100.0)),
    )
    assert not ok and "No valid channels selected for Raw power" in message

    ok, message = tensor_service._run_raw_power_metric(
        context,
        low_freq=6.0,
        high_freq=8.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 10.0)),
    )
    assert not ok and "High frequency is invalid" in message

    ok, message = tensor_service._run_raw_power_metric(
        context,
        low_freq=2.0,
        high_freq=3.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        notches=[2.5],
        notch_widths=10.0,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(2.5,),
            notch_widths=(10.0,),
        ),
    )
    assert not ok and "Notch exclusion removed too many bins" in message

    ok, message = tensor_service._run_raw_power_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(),
            notch_widths=(),
        ),
        tfr_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 3, 4), dtype=float),
            {"axes": {}},
        ),
    )
    assert ok and "computed" in message

    ok, message = tensor_service._run_raw_power_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(),
            notch_widths=(),
        ),
        tfr_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((3, 4), dtype=float),
            {"axes": {}},
        ),
    )
    assert not ok and "Unexpected Raw power tensor shape" in message


def test_periodic_aperiodic_remaining_branches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["MISSING"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 100.0)),
    )
    assert not ok and "No valid channels selected for Periodic/APeriodic" in message

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=6.0,
        high_freq=8.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 10.0)),
    )
    assert not ok and "exceeds Nyquist" in message

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=2.0,
        high_freq=3.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        notches=[2.5],
        notch_widths=10.0,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(2.5,),
            notch_widths=(10.0,),
        ),
    )
    assert not ok and "Notch exclusion removed too many bins" in message

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(),
            notch_widths=(),
        ),
        tfr_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((3, 4), dtype=float),
            {"axes": {"freq": np.array([1.0, 2.0]), "time": np.array([0.0, 1.0])}},
        ),
    )
    assert not ok and "Unexpected TFR tensor shape" in message

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(),
            notch_widths=(),
        ),
        tfr_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 1, 3, 4), dtype=float),
            {"axes": {"time": np.array([0.0, 1.0, 2.0, 3.0])}},
        ),
    )
    assert not ok and "missing/invalid frequency axis" in message

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(),
            notch_widths=(),
        ),
        tfr_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 1, 3, 4), dtype=float),
            {
                "axes": {
                    "freq": np.array([2.0, 3.0, 4.0]),
                    "time": np.array([0.0, 1.0, 2.0, 3.0]),
                    "channel": np.array(["CH1"], dtype=object),
                }
            },
        ),
        decompose_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 1, 3, 4), dtype=float),
            np.ones((1, 3, 4), dtype=float),
            np.ones((1, 1, 3, 4), dtype=float),
            np.ones((1, 1, 3, 2), dtype=float),
            {},
        ),
    )
    assert not ok and "Unexpected periodic tensor shape" in message

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=2.0,
        high_freq=4.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(),
            notch_widths=(),
        ),
        tfr_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 3, 4), dtype=float),
            {
                "axes": {
                    "freq": np.array([2.0, 3.0, 4.0]),
                    "time": np.array([0.0, 1.0, 2.0, 3.0]),
                    "channel": np.array(["CH1"], dtype=object),
                }
            },
        ),
        decompose_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 1, 3, 4), dtype=float),
            np.ones((1, 1, 3, 4), dtype=float),
            np.ones((1, 1, 3, 4), dtype=float),
            np.ones((1, 1, 3, 4), dtype=float),
            {
                "axes": {
                    "freq": np.array(
                        ["offset", "exponent", "gof_rsquared"], dtype=object
                    )
                }
            },
        ),
    )
    assert ok and "computed" in message


def test_periodic_aperiodic_report_dir_rerun_replaces_and_failure_restores(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    report_dir = (
        tensor_service.tensor_metric_tensor_path(
            resolver,
            "periodic_aperiodic",
            create=True,
        ).parent
        / "specparam_report"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stale.pdf").write_text("stale report", encoding="utf-8")

    def _fake_tfr_grid(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        tensor = np.ones((1, 1, freqs.size, 4), dtype=float)
        return tensor, {
            "axes": {
                "freq": freqs.copy(),
                "time": np.linspace(0.0, 1.0, 4, dtype=float),
                "channel": np.asarray(["CH1"], dtype=object),
                "shape": tensor.shape,
            }
        }

    def _fake_successful_decompose(tensor, freqs, **kwargs):  # noqa: ANN001, ANN003
        _ = freqs
        current_report_dir = Path(kwargs["report_dir"])
        assert current_report_dir == report_dir
        assert sorted(path.name for path in current_report_dir.iterdir()) == []
        (current_report_dir / "specparam_e000_ch000.pdf").write_text(
            "stable report",
            encoding="utf-8",
        )
        params_tensor = np.ones((1, 1, 3, tensor.shape[3]), dtype=float)
        params_meta = {
            "axes": {
                "freq": np.asarray(
                    ["offset", "exponent", "gof_rsquared"],
                    dtype=object,
                )
            }
        }
        return (
            np.ones_like(tensor),
            np.full_like(tensor, 2.0),
            np.full_like(tensor, 3.0),
            params_tensor,
            params_meta,
        )

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=2.0,
        high_freq=4.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(),
            notch_widths=(),
        ),
        tfr_grid_fn=_fake_tfr_grid,
        decompose_fn=_fake_successful_decompose,
    )
    assert ok and "computed" in message
    assert sorted(path.name for path in report_dir.iterdir()) == [
        "specparam_e000_ch000.pdf"
    ]
    assert (report_dir / "specparam_e000_ch000.pdf").read_text(encoding="utf-8") == (
        "stable report"
    )

    def _fake_failing_decompose(tensor, freqs, **kwargs):  # noqa: ANN001, ANN003
        _ = tensor, freqs
        current_report_dir = Path(kwargs["report_dir"])
        assert current_report_dir == report_dir
        assert sorted(path.name for path in current_report_dir.iterdir()) == []
        (current_report_dir / "partial.pdf").write_text(
            "partial report",
            encoding="utf-8",
        )
        raise RuntimeError("report export failed")

    ok, message = tensor_service._run_periodic_aperiodic_metric(
        context,
        low_freq=2.0,
        high_freq=4.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        load_tensor_filter_inheritance_fn=lambda _ctx: TensorFilterInheritance(
            low_freq=1.0,
            high_freq=100.0,
            notches=(),
            notch_widths=(),
        ),
        tfr_grid_fn=_fake_tfr_grid,
        decompose_fn=_fake_failing_decompose,
    )
    assert not ok and "report export failed" in message
    assert sorted(path.name for path in report_dir.iterdir()) == [
        "specparam_e000_ch000.pdf"
    ]
    assert (report_dir / "specparam_e000_ch000.pdf").read_text(encoding="utf-8") == (
        "stable report"
    )


def test_periodic_aperiodic_run_tfr_grid_smooths_in_db_domain(tmp_path: Path) -> None:
    compute_module = importlib.import_module(
        "lfptensorpipe.app.tensor.runners.periodic_aperiodic_compute"
    )
    captured: list[dict[str, object]] = []

    def _fake_smooth(tensor: np.ndarray, **kwargs: object) -> np.ndarray:
        captured.append(dict(kwargs))
        return np.asarray(tensor, dtype=float)

    prepared = _periodic_prepared_input(
        raw=_FakeRaw(("CH1",), 200.0),
        interpolation_applied=False,
        freqs_compute=np.array([1.0, 2.0], dtype=float),
        freqs_model=np.array([1.0, 2.0], dtype=float),
        freqs_final=np.array([1.0, 2.0], dtype=float),
    )
    options = _periodic_options(_context(tmp_path))

    tensor, metadata = compute_module._run_tfr_grid(
        prepared,
        options,
        tfr_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 1, 2, 3), dtype=float),
            {
                "axes": {
                    "freq": np.array([1.0, 2.0], dtype=float),
                    "time": np.array([0.0, 0.1, 0.2], dtype=float),
                    "channel": np.array(["CH1"], dtype=object),
                }
            },
        ),
        smooth_axis_fn=_fake_smooth,
    )

    assert np.asarray(tensor).shape == (1, 1, 2, 3)
    assert np.asarray(metadata["axes"]["freq"], dtype=float).tolist() == [1.0, 2.0]
    assert len(captured) == 2
    assert captured[0]["method"] == "gaussian"
    assert captured[0]["axis"] == -2
    assert captured[0]["sigma"] == 1.5
    assert captured[0]["transform_mode"] == "dB"
    assert captured[1]["method"] == "median"
    assert captured[1]["axis"] == -1
    assert captured[1]["kernel_size"] == 13
    assert captured[1]["transform_mode"] == "dB"


def test_periodic_aperiodic_run_tfr_grid_clips_interpolated_power_to_preinterp_floor(
    tmp_path: Path,
) -> None:
    compute_module = importlib.import_module(
        "lfptensorpipe.app.tensor.runners.periodic_aperiodic_compute"
    )

    prepared = _periodic_prepared_input(
        raw=_FakeRaw(("CH1",), 200.0),
        interpolation_applied=True,
        freqs_compute=np.array([1.0, 3.0], dtype=float),
        freqs_model=np.array([1.0, 2.0, 3.0], dtype=float),
        freqs_final=np.array([1.0, 2.0, 3.0], dtype=float),
    )
    options = _periodic_options(
        _context(tmp_path),
        freq_smooth_enabled=False,
        time_smooth_enabled=False,
    )

    tensor, _metadata = compute_module._run_tfr_grid(
        prepared,
        options,
        tfr_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.array([[[[0.25, 0.50], [1.00, 2.00]]]], dtype=float),
            {
                "axes": {
                    "freq": np.array([1.0, 3.0], dtype=float),
                    "time": np.array([0.0, 0.1], dtype=float),
                    "channel": np.array(["CH1"], dtype=object),
                }
            },
        ),
        interpolate_freq_tensor_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.array([[[[0.10, 0.30], [0.24, 0.60], [0.80, 1.20]]]], dtype=float),
            {
                "axes": {
                    "freq": np.array([1.0, 2.0, 3.0], dtype=float),
                    "time": np.array([0.0, 0.1], dtype=float),
                    "channel": np.array(["CH1"], dtype=object),
                }
            },
        ),
    )

    assert np.isclose(float(np.min(tensor)), 0.25)
    assert np.isclose(float(tensor[0, 0, 0, 0]), 0.25)
    assert np.isclose(float(tensor[0, 0, 1, 0]), 0.25)
    assert np.isclose(float(tensor[0, 0, 2, 1]), 1.20)


def test_undirected_connectivity_remaining_runtime_branches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    bands = [dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS]

    ok, message = tensor_service._run_undirected_connectivity_metric(
        context,
        metric_key="coherence",
        connectivity_metric="coh",
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
    )
    assert not ok and "requires at least 2 valid channels" in message

    ok, message = tensor_service._run_undirected_connectivity_metric(
        context,
        metric_key="coherence",
        connectivity_metric="coh",
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=None,
        selected_pairs=[("CH1", "CH2")],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        normalize_selected_pairs_fn=lambda *args, **kwargs: [("CH1", "CH2")],
    )
    assert not ok and "requires at least 2 valid channels" in message

    ok, message = tensor_service._run_undirected_connectivity_metric(
        context,
        metric_key="coherence",
        connectivity_metric="coh",
        low_freq=3.0,
        high_freq=10.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 4.0)),
    )
    assert not ok and "High frequency is invalid" in message

    ok, message = tensor_service._run_undirected_connectivity_metric(
        context,
        metric_key="coherence",
        connectivity_metric="coh",
        low_freq=2.0,
        high_freq=3.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        compute_notch_intervals_fn=lambda **kwargs: [(1.0, 10.0)],
    )
    assert not ok and "Notch exclusion removed too many bins" in message

    ok, message = tensor_service._run_undirected_connectivity_metric(
        context,
        metric_key="coherence",
        connectivity_metric="coh",
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        compute_notch_intervals_fn=lambda **kwargs: [(3.0, 3.0)],
        conn_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 3, 4), dtype=float),
            {
                "axes": {
                    "freq": np.array([2.0, 4.0, 5.0]),
                    "time": np.array([0, 1, 2, 3]),
                }
            },
        ),
        interpolate_freq_tensor_fn=lambda tensor, metadata, **kwargs: (  # noqa: ANN001
            np.asarray(tensor, dtype=float),
            metadata,
        ),
    )
    assert ok and "with notch interpolation" in message

    ok, message = tensor_service._run_undirected_connectivity_metric(
        context,
        metric_key="coherence",
        connectivity_metric="coh",
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        compute_notch_intervals_fn=lambda **kwargs: [],
        conn_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((2,), dtype=float),
            {},
        ),
    )
    assert not ok and "Unexpected Coherence tensor shape" in message


def test_trgc_psi_burst_remaining_runtime_branches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    bands = [dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS]

    ok, message = tensor_service._run_trgc_metric(
        context,
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
    )
    assert not ok and "requires at least 2 valid channels" in message

    ok, message = tensor_service._run_trgc_metric(
        context,
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=None,
        selected_pairs=[("CH1", "CH2")],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        normalize_selected_pairs_fn=lambda *args, **kwargs: [("CH1", "CH2")],
    )
    assert not ok and "requires at least 2 valid channels" in message

    ok, message = tensor_service._run_trgc_metric(
        context,
        low_freq=3.0,
        high_freq=10.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 4.0)),
    )
    assert not ok and "High frequency is invalid" in message

    ok, message = tensor_service._run_trgc_metric(
        context,
        low_freq=2.0,
        high_freq=3.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        compute_notch_intervals_fn=lambda **kwargs: [(1.0, 10.0)],
    )
    assert not ok and "Notch exclusion removed too many bins" in message

    def _fake_trgc_grid(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        pairs = list(kwargs["pairs"])
        return (
            np.ones((len(pairs), 3, 4), dtype=float),
            {
                "axes": {
                    "channel": list(pairs),
                    "freq": np.array([2.0, 4.0, 5.0]),
                    "time": np.array([0, 1, 2, 3]),
                }
            },
        )

    ok, message = tensor_service._run_trgc_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        compute_notch_intervals_fn=lambda **kwargs: [(3.0, 3.0)],
        conn_grid_fn=_fake_trgc_grid,
        interpolate_freq_tensor_fn=lambda tensor, metadata, **kwargs: (  # noqa: ANN001
            np.asarray(tensor, dtype=float),
            metadata,
        ),
    )
    assert ok and "with notch interpolation" in message

    ok, message = tensor_service._run_trgc_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        compute_notch_intervals_fn=lambda **kwargs: [],
        conn_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((2,), dtype=float),
            {},
        ),
    )
    assert not ok and "Unexpected TRGC tensor shape" in message

    ok, message = tensor_service._run_psi_metric(
        context,
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
    )
    assert not ok and "requires at least 2 valid channels" in message

    ok, message = tensor_service._run_psi_metric(
        context,
        low_freq=2.0,
        high_freq=40.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=None,
        selected_pairs=[("CH1", "CH2")],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        normalize_selected_pairs_fn=lambda *args, **kwargs: [("CH1", "CH2")],
    )
    assert not ok and "requires at least 2 valid channels" in message

    ok, message = tensor_service._run_psi_metric(
        context,
        low_freq=2.0,
        high_freq=4.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[
            {"name": "", "start": 1.0, "end": 2.0},
            {"name": "gamma", "start": 8.0, "end": 9.0},
        ],
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
    )
    assert not ok and "No valid band intersects" in message

    ok, message = tensor_service._run_psi_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        psi_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((2, 3, 4), dtype=float),
            {"axes": {"time": np.array([0.0, 1.0, 2.0, 3.0])}},
        ),
    )
    assert ok and "PSI tensor computed" in message

    ok, message = tensor_service._run_psi_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1", "CH2"],
        selected_pairs=None,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        psi_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((2,), dtype=float),
            {},
        ),
    )
    assert not ok and "Unexpected PSI tensor shape" in message

    ok, message = tensor_service._run_burst_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["MISSING"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
    )
    assert not ok and "requires at least 1 valid channel" in message

    ok, message = tensor_service._run_burst_metric(
        context,
        low_freq=2.0,
        high_freq=4.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[
            {"name": "", "start": 1.0, "end": 2.0},
            {"name": "gamma", "start": 8.0, "end": 9.0},
        ],
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
    )
    assert not ok and "No valid burst band intersects" in message

    ok, message = tensor_service._run_burst_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        burst_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((1, 3, 4), dtype=float),
            {"qc": {"thresholds": np.array([0.2, 0.4], dtype=float)}},
        ),
    )
    assert ok and "Burst tensor computed" in message

    ok, message = tensor_service._run_burst_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=["CH1"],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        burst_grid_fn=lambda *args, **kwargs: (  # noqa: ANN002, ANN003
            np.ones((2,), dtype=float),
            {},
        ),
    )
    assert not ok and "Unexpected Burst tensor shape" in message


def test_run_psi_metric_splits_bands_by_runtime_notch_intervals(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    finish_raw = _prepare_finish_ready(context, resolver)
    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=True,
            params={
                "low_freq": 1.0,
                "high_freq": 80.0,
                "notches": [50.0],
                "notch_widths": 2.0,
            },
            input_path=str(finish_raw),
            output_path=str(preproc_step_raw_path(resolver, "filter")),
            message="filter ready",
        ),
    )

    captured: dict[str, object] = {}

    def fake_psi_grid(raw_obj, **kwargs):  # noqa: ANN001, ANN003
        _ = raw_obj
        captured["bands"] = dict(kwargs["bands"])
        pairs = list(kwargs["pairs"])
        tensor = np.ones((1, len(pairs), len(kwargs["bands"]), 4), dtype=float)
        metadata = {
            "axes": {
                "channel": np.asarray([f"{a}->{b}" for a, b in pairs], dtype=object),
                "freq": np.asarray(list(kwargs["bands"].keys()), dtype=object),
                "time": np.linspace(0.0, 1.0, 4, dtype=float),
            }
        }
        return tensor, metadata

    ok, message = tensor_service._run_psi_metric(
        context,
        low_freq=40.0,
        high_freq=60.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[{"name": "alpha", "start": 45.0, "end": 55.0}],
        selected_channels=["CH1", "CH2"],
        selected_pairs=[("CH1", "CH2")],
        notches=[50.0],
        notch_widths=2.0,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        psi_grid_fn=fake_psi_grid,
    )

    assert ok and "PSI tensor computed" in message
    assert captured["bands"] == {"alpha": [(45.0, 48.0), (52.0, 55.0)]}


def test_run_burst_metric_uses_notch_safe_bands_and_default_time_grid(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    finish_raw = _prepare_finish_ready(context, resolver)
    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=True,
            params={
                "low_freq": 1.0,
                "high_freq": 80.0,
                "notches": [50.0],
                "notch_widths": 2.0,
            },
            input_path=str(finish_raw),
            output_path=str(preproc_step_raw_path(resolver, "filter")),
            message="filter ready",
        ),
    )

    captured: dict[str, object] = {}

    def fake_burst_grid(raw_obj, **kwargs):  # noqa: ANN001, ANN003
        _ = raw_obj
        captured["bands"] = dict(kwargs["bands"])
        captured["baseline_keep"] = kwargs["baseline_keep"]
        captured["baseline_match"] = kwargs["baseline_match"]
        captured["hop_s"] = kwargs["hop_s"]
        captured["decim"] = kwargs["decim"]
        tensor = np.ones((1, 1, len(kwargs["bands"]), 4), dtype=float)
        metadata = {
            "axes": {
                "channel": np.asarray(["CH1"], dtype=object),
                "freq": np.asarray(list(kwargs["bands"].keys()), dtype=object),
                "time": np.linspace(0.0, 1.0, 4, dtype=float),
            },
            "qc": {"thresholds": np.array([[0.2]], dtype=float)},
        }
        return tensor, metadata

    ok, message = tensor_service._run_burst_metric(
        context,
        low_freq=40.0,
        high_freq=60.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=[{"name": "alpha", "start": 45.0, "end": 55.0}],
        selected_channels=["CH1"],
        baseline_keep=["Rest"],
        notches=[50.0],
        notch_widths=2.0,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        burst_grid_fn=fake_burst_grid,
    )

    assert ok and "Burst tensor computed" in message
    assert captured["bands"] == {"alpha": [(45.0, 48.0), (52.0, 55.0)]}
    assert captured["baseline_keep"] == ["Rest"]
    assert captured["baseline_match"] == "exact"
    assert captured["hop_s"] is None
    assert captured["decim"] == 1


def test_compute_tensor_metric_filter_notch_warnings_uses_interest_range_rules(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    filter_output = preproc_step_raw_path(resolver, "filter")
    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=False,
            params={
                "low_freq": 1.0,
                "high_freq": 120.0,
                "notches": [50.0],
                "notch_widths": 2.0,
            },
            input_path=str(filter_output),
            output_path=str(filter_output),
            message="filter incomplete",
        ),
    )
    assert (
        tensor_service.compute_tensor_metric_filter_notch_warnings(
            context,
            "raw_power",
            {
                "low_freq_hz": 49.0,
                "high_freq_hz": 55.0,
                "notches": [],
                "notch_widths": 2.0,
            },
        )
        == []
    )

    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=True,
            params={
                "low_freq": 1.0,
                "high_freq": 120.0,
                "notches": [50.0, 100.0],
                "notch_widths": 2.0,
            },
            input_path=str(filter_output),
            output_path=str(filter_output),
            message="filter ready",
        ),
    )

    raw_power_warnings = tensor_service.compute_tensor_metric_filter_notch_warnings(
        context,
        "raw_power",
        {
            "low_freq_hz": 49.0,
            "high_freq_hz": 55.0,
            "notches": [],
            "notch_widths": 2.0,
        },
    )
    assert raw_power_warnings == ["Missing preprocess filter notch 50 Hz."]

    specparam_warnings = tensor_service.compute_tensor_metric_filter_notch_warnings(
        context,
        "periodic_aperiodic",
        {
            "freq_range_hz": [45.0, 55.0],
            "notches": [],
            "notch_widths": 2.0,
        },
    )
    assert specparam_warnings == ["Missing preprocess filter notch 50 Hz."]

    psi_warnings = tensor_service.compute_tensor_metric_filter_notch_warnings(
        context,
        "psi",
        {
            "bands": [
                {"name": "low", "start": 10.0, "end": 20.0},
                {"name": "high", "start": 70.0, "end": 80.0},
            ],
            "notches": [50.0],
            "notch_widths": 1.0,
        },
    )
    assert psi_warnings == ["Notch 50 Hz width is too small (metric=1, filter=2)."]


def test_raw_power_metric_records_runtime_and_inherited_filter_notches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    finish_raw = _prepare_finish_ready(context, resolver)
    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=True,
            params={
                "low_freq": 1.0,
                "high_freq": 80.0,
                "notches": [50.0],
                "notch_widths": 2.0,
            },
            input_path=str(finish_raw),
            output_path=str(preproc_step_raw_path(resolver, "filter")),
            message="filter ready",
        ),
    )

    import yaml

    def _fake_tfr_grid(raw_obj, **kwargs):  # noqa: ANN001, ANN003
        _ = raw_obj
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        tensor = np.ones((1, freqs.size, 4), dtype=float)
        metadata = {
            "axes": {
                "channel": np.asarray(["CH1"], dtype=object),
                "freq": freqs,
                "time": np.linspace(0.0, 1.0, 4, dtype=float),
            }
        }
        return tensor, metadata

    ok, message = tensor_service._run_raw_power_metric(
        context,
        low_freq=2.0,
        high_freq=20.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS],
        selected_channels=["CH1"],
        notches=[60.0],
        notch_widths=3.0,
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1",), 200.0)),
        tfr_grid_fn=_fake_tfr_grid,
    )

    assert ok and "Raw power tensor computed" in message

    config_path = tensor_service.tensor_metric_config_path(resolver, "raw_power")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    log_payload = read_run_log(
        tensor_service.tensor_metric_log_path(resolver, "raw_power")
    )

    assert config_payload["notches"] == [60.0]
    assert config_payload["notch_widths"] == [3.0]
    assert config_payload["inherited_filter_notches"] == [50.0]
    assert config_payload["inherited_filter_notch_widths"] == [2.0]
    assert log_payload["params"]["notches"] == [60.0]
    assert log_payload["params"]["notch_widths"] == [3.0]
    assert log_payload["params"]["inherited_filter_notches"] == [50.0]
    assert log_payload["params"]["inherited_filter_notch_widths"] == [2.0]


def test_run_trgc_metric_single_direction_pair_derives_and_crops(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    bands = [dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS]

    import yaml

    from lfptensorpipe.io.pkl_io import load_pkl

    calls: list[dict[str, object]] = []

    def _fake_trgc_backend(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        method = str(kwargs["method"])
        pairs = list(kwargs["pairs"])
        calls.append(
            {
                "method": method,
                "pairs": pairs,
                "group_by_samples": bool(kwargs["group_by_samples"]),
                "round_ms": float(kwargs["round_ms"]),
            }
        )
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        tensor = np.empty((1, len(pairs), freqs.size, 5), dtype=float)
        pair_values = {
            ("gc", ("CH1", "CH2")): 5.0,
            ("gc", ("CH2", "CH1")): 2.0,
            ("gc_tr", ("CH1", "CH2")): 4.0,
            ("gc_tr", ("CH2", "CH1")): 3.0,
        }
        for idx, pair in enumerate(pairs):
            tensor[:, idx, :, :] = pair_values[(method, pair)]
        return (
            tensor,
            {
                "axes": {
                    "channel": list(pairs),
                    "freq": freqs.copy(),
                    "time": np.linspace(0.0, 1.0, 5, dtype=float),
                    "shape": tensor.shape,
                },
                "params": {"method": method},
            },
        )

    ok, message = tensor_service._run_trgc_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=bands,
        selected_channels=None,
        selected_pairs=[("CH1", "CH2")],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
        compute_notch_intervals_fn=lambda **kwargs: [],
        conn_grid_fn=_fake_trgc_backend,
    )

    assert ok and "TRGC tensor computed" in message
    assert calls == [
        {
            "method": "gc",
            "pairs": [("CH1", "CH2"), ("CH2", "CH1")],
            "group_by_samples": False,
            "round_ms": 50.0,
        },
        {
            "method": "gc_tr",
            "pairs": [("CH1", "CH2"), ("CH2", "CH1")],
            "group_by_samples": False,
            "round_ms": 50.0,
        },
    ]

    tensor_payload = load_pkl(
        tensor_service.tensor_metric_tensor_path(resolver, "trgc")
    )
    assert tensor_payload["tensor"].shape == (1, 1, 4, 5)
    assert np.allclose(tensor_payload["tensor"], 2.0)
    assert tensor_payload["meta"]["axes"]["channel"] == [("CH1", "CH2")]

    config_payload = yaml.safe_load(
        tensor_service.tensor_metric_config_path(resolver, "trgc").read_text(
            encoding="utf-8"
        )
    )
    assert config_payload["connectivity_metric"] == "trgc"
    assert config_payload["selected_pairs"] == [["CH1", "CH2"]]
    assert config_payload["pairs_compute"] == [["CH1", "CH2"], ["CH2", "CH1"]]
    assert config_payload["group_by_samples"] is False
    assert config_payload["round_ms"] == 50.0


def test_compute_trgc_backend_tensor_suppresses_known_numpy_det_warning() -> None:
    def _fake_grid(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        warnings.warn_explicit(
            "invalid value encountered in det",
            RuntimeWarning,
            filename="/opt/anaconda3/envs/lfptp/lib/python3.11/site-packages/numpy/linalg/_linalg.py",
            lineno=2431,
            module="numpy.linalg._linalg",
        )
        warnings.warn_explicit(
            "keep me visible",
            RuntimeWarning,
            filename=str(Path(__file__)),
            lineno=1,
            module="tests.trgc_warning_probe",
        )
        return (
            np.ones((1, 2, 2, 3), dtype=float),
            {
                "axes": {
                    "channel": [("CH1", "CH2"), ("CH2", "CH1")],
                    "freq": np.array([2.0, 3.0], dtype=float),
                    "time": np.array([0.0, 0.1, 0.2], dtype=float),
                    "shape": (1, 2, 2, 3),
                },
                "params": {"method": "gc"},
            },
        )

    prepared = {
        "raw": _FakeRaw(("CH1", "CH2"), 200.0),
        "freqs_compute": np.array([2.0, 3.0], dtype=float),
        "time_resolution_s": 0.25,
        "hop_s": 0.05,
        "pairs_compute": [("CH1", "CH2"), ("CH2", "CH1")],
        "spectral_mode_use": "cwt_morlet",
        "mt_bandwidth": None,
        "min_cycles": None,
        "max_cycles": None,
        "gc_n_lags": 20,
        "group_by_samples": False,
        "round_ms": 50.0,
        "picks": ["CH1", "CH2"],
        "metric_label": "TRGC",
        "interpolation_applied": False,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensor4d, metadata = trgc_runner_module._compute_trgc_backend_tensor(
            prepared,
            backend_method="gc",
            conn_grid_fn=_fake_grid,
        )

    assert tensor4d.shape == (1, 2, 2, 3)
    assert metadata["params"]["method"] == "gc"
    assert [str(item.message) for item in caught] == ["keep me visible"]


def test_connectivity_metrics_selected_pairs_empty_list_branches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    _prepare_finish_ready(context, resolver)
    bands = [dict(item) for item in tensor_service.DEFAULT_TENSOR_BANDS]

    ok, message = tensor_service._run_undirected_connectivity_metric(
        context,
        metric_key="coherence",
        connectivity_metric="coh",
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=None,
        selected_pairs=[],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
    )
    assert not ok and "No valid selected pairs available for Coherence." in message

    ok, message = tensor_service._run_trgc_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=None,
        selected_pairs=[],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
    )
    assert not ok and "No valid selected directed pairs available for TRGC." in message

    ok, message = tensor_service._run_psi_metric(
        context,
        low_freq=2.0,
        high_freq=5.0,
        step_hz=1.0,
        mask_edge_effects=True,
        bands=bands,
        selected_channels=None,
        selected_pairs=[],
        read_raw_fif_fn=_fake_read_raw(_FakeRaw(("CH1", "CH2"), 200.0)),
    )
    assert not ok and "No valid selected directed pairs available for PSI." in message
