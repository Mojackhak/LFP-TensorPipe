"""Regression coverage for Finish bad-channel exclusion in Build Tensor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.tensor import indicator as indicator_module
from lfptensorpipe.app.tensor import service as tensor_service
from lfptensorpipe.app.tensor import orchestration_plans as plan_module
from lfptensorpipe.app.tensor.connectivity_coordinator import _normalize_pairs
from lfptensorpipe.app.tensor.runners import (
    connectivity_trgc,
    connectivity_undirected,
    periodic_aperiodic_prepare,
    raw_power,
)
from lfptensorpipe.app.tensor.selectors import (
    TensorChannelInventory,
    select_usable_channels,
    select_usable_pairs,
    tensor_channel_inventory_from_raw,
)


class _RawStub:
    def __init__(
        self,
        channels: tuple[str, ...],
        *,
        bads: tuple[str, ...] = (),
        sfreq: float = 100.0,
    ) -> None:
        self.ch_names = list(channels)
        self.info = {"bads": list(bads), "sfreq": float(sfreq)}
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _inventory() -> TensorChannelInventory:
    return TensorChannelInventory(
        all_channels=("A", "B", "C"),
        bad_channels=("B",),
        usable_channels=("A", "C"),
    )


def test_channel_inventory_and_channel_selection_preserve_order() -> None:
    inventory = tensor_channel_inventory_from_raw(
        _RawStub(("C", "B", "A"), bads=("B",))
    )

    assert inventory.all_channels == ("C", "B", "A")
    assert inventory.bad_channels == ("B",)
    assert inventory.usable_channels == ("C", "A")
    assert select_usable_channels(None, inventory=inventory) == (
        ["C", "A"],
        ("B",),
    )
    assert select_usable_channels(["A", "B", "C"], inventory=inventory) == (
        ["A", "C"],
        ("B",),
    )
    assert select_usable_channels([], inventory=inventory) == ([], ())


def test_pair_selection_excludes_every_pair_with_a_bad_endpoint() -> None:
    pairs, excluded = select_usable_pairs(
        [("A", "B"), ("A", "C"), ("B", "C")],
        inventory=_inventory(),
    )

    assert pairs == [("A", "C")]
    assert len(excluded) == 2
    assert all("B" in pair for pair in excluded)


class _PlanService:
    TENSOR_CHANNEL_SELECTOR_KEYS = tensor_service.TENSOR_CHANNEL_SELECTOR_KEYS
    TENSOR_UNDIRECTED_SELECTOR_KEYS = tensor_service.TENSOR_UNDIRECTED_SELECTOR_KEYS
    TENSOR_DIRECTED_SELECTOR_KEYS = tensor_service.TENSOR_DIRECTED_SELECTOR_KEYS
    TENSOR_METRICS_BY_KEY = tensor_service.TENSOR_METRICS_BY_KEY

    def __init__(self, root: Path) -> None:
        self.root = root
        self.failure_messages: list[str] = []

    @staticmethod
    def _select_usable_channels(*args: Any, **kwargs: Any):
        return select_usable_channels(*args, **kwargs)

    @staticmethod
    def _select_usable_pairs(*args: Any, **kwargs: Any):
        return select_usable_pairs(*args, **kwargs)

    def preproc_step_raw_path(self, _resolver: Any, _step: str) -> Path:
        return self.root / "finish" / "raw.fif"

    def tensor_metric_tensor_path(self, _resolver: Any, metric_key: str) -> Path:
        return self.root / "tensor" / metric_key / "tensor.pkl"

    @staticmethod
    def _sanitize_metric_params_for_logs(params: dict[str, Any]) -> dict[str, Any]:
        return dict(params)

    def _write_metric_log(self, *args: Any, **kwargs: Any) -> None:
        _ = args
        self.failure_messages.append(str(kwargs["message"]))


def test_runtime_plans_filter_all_eleven_metric_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = sorted(_PlanService.TENSOR_METRICS_BY_KEY)
    params: dict[str, dict[str, Any]] = {}
    for metric_key in metrics:
        if metric_key in _PlanService.TENSOR_CHANNEL_SELECTOR_KEYS:
            params[metric_key] = {"selected_channels": ["A", "B", "C"]}
        else:
            params[metric_key] = {
                "selected_pairs": [["A", "B"], ["A", "C"], ["B", "C"]]
            }

    def prepare_stub(
        _svc: Any,
        _context: Any,
        *,
        metric_key: str,
        metric_label: str,
        metric_params: dict[str, Any],
    ) -> Any:
        _ = metric_label
        return SimpleNamespace(
            metric_key=metric_key,
            metric_params=dict(metric_params),
        )

    captured: dict[str, dict[str, Any]] = {}

    def plan_stub(
        _svc: Any,
        _context: Any,
        *,
        prepared: Any,
        mask_edge_effects: bool,
    ) -> dict[str, Any]:
        _ = mask_edge_effects
        captured[prepared.metric_key] = dict(prepared.metric_params)
        return {prepared.metric_key: object()}

    monkeypatch.setattr(plan_module, "prepare_metric_plan_inputs", prepare_stub)
    monkeypatch.setattr(plan_module, "build_runtime_plan", plan_stub)
    service = _PlanService(tmp_path)
    result = plan_module.build_runtime_plans(
        service,
        object(),
        object(),
        metrics=metrics,
        merged_metric_params_map=params,
        mask_edge_effects=False,
        channel_inventory=_inventory(),
    )

    assert result.overall_ok is True
    assert set(result.runtime_plans) == set(metrics)
    for metric_key in _PlanService.TENSOR_CHANNEL_SELECTOR_KEYS:
        assert captured[metric_key]["selected_channels"] == ["A", "C"]
    for metric_key in (
        _PlanService.TENSOR_UNDIRECTED_SELECTOR_KEYS
        | _PlanService.TENSOR_DIRECTED_SELECTOR_KEYS
    ):
        assert captured[metric_key]["selected_pairs"] == [["A", "C"]]
    assert len(result.messages) == len(metrics)


def test_runtime_plan_does_not_replace_an_all_bad_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _PlanService(tmp_path)
    monkeypatch.setattr(
        plan_module,
        "prepare_metric_plan_inputs",
        lambda *_args, **kwargs: SimpleNamespace(
            metric_key=kwargs["metric_key"],
            metric_params=dict(kwargs["metric_params"]),
        ),
    )
    monkeypatch.setattr(
        plan_module,
        "build_runtime_plan",
        lambda *_args, **kwargs: {kwargs["prepared"].metric_key: object()},
    )
    params = {
        "raw_power": {"selected_channels": ["B"]},
        "coherence": {"selected_pairs": [["A", "C"]]},
    }

    result = plan_module.build_runtime_plans(
        service,
        object(),
        object(),
        metrics=["raw_power", "coherence"],
        merged_metric_params_map=params,
        mask_edge_effects=True,
        channel_inventory=_inventory(),
    )

    assert result.overall_ok is False
    assert "raw_power" not in result.runtime_plans
    assert "coherence" in result.runtime_plans
    assert params["raw_power"]["selected_channels"] == ["B"]
    assert any("raw.info['bads']" in item for item in service.failure_messages)


@pytest.mark.parametrize(
    ("metric_key", "params", "selector_key", "expected"),
    [
        (
            "raw_power",
            {"selected_channels": ["A", "B", "C"]},
            "selected_channels",
            ["A", "C"],
        ),
        (
            "coherence",
            {"selected_pairs": [["A", "B"], ["A", "C"]]},
            "selected_pairs",
            [["A", "C"]],
        ),
    ],
)
def test_indicator_validates_the_effective_selector(
    monkeypatch: pytest.MonkeyPatch,
    metric_key: str,
    params: dict[str, Any],
    selector_key: str,
    expected: list[Any],
) -> None:
    captured: dict[str, Any] = {}

    def prepare_stub(*_args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs["metric_params"])
        raise RuntimeError("stop after selector capture")

    monkeypatch.setattr(
        indicator_module,
        "prepare_metric_plan_inputs",
        prepare_stub,
    )

    with pytest.raises(RuntimeError, match="stop after selector capture"):
        indicator_module._current_metric_signature(
            object(),
            metric_key=metric_key,
            metric_params=params,
            mask_edge_effects=True,
            channel_inventory=_inventory(),
        )

    assert captured[selector_key] == expected


@pytest.mark.parametrize("selected_channels", [["B"], []])
def test_direct_raw_power_never_falls_back_to_all_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_channels: list[str],
) -> None:
    context = RecordContext(tmp_path, "sub-test", "record-test")
    input_path = PathResolver(context).preproc_root / "finish" / "raw.fif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch()
    raw = _RawStub(("A", "B"), bads=("B",))
    compute_called = False

    def compute_stub(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal compute_called
        compute_called = True
        raise AssertionError("TFR computation must not start.")

    monkeypatch.setattr(raw_power.svc, "indicator_from_log", lambda *_args: "green")
    monkeypatch.setattr(
        raw_power.svc, "_write_metric_log", lambda *_args, **_kwargs: None
    )

    ok, message = raw_power.run_raw_power_metric.__wrapped__(
        context,
        low_freq=1.0,
        high_freq=30.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[],
        selected_channels=selected_channels,
        read_raw_fif_fn=lambda *_args, **_kwargs: raw,
        tfr_grid_fn=compute_stub,
    )

    assert ok is False
    assert "No valid channels" in message
    assert compute_called is False


def test_direct_raw_power_passes_only_usable_channels_to_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RecordContext(tmp_path, "sub-test", "record-test")
    input_path = PathResolver(context).preproc_root / "finish" / "raw.fif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch()
    raw = _RawStub(("A", "B"), bads=("B",))
    captured_picks: list[str] = []

    def compute_stub(*_args: Any, **kwargs: Any) -> Any:
        captured_picks.extend(kwargs["picks"])
        raise RuntimeError("stop after picks")

    monkeypatch.setattr(raw_power.svc, "indicator_from_log", lambda *_args: "green")
    monkeypatch.setattr(
        raw_power.svc, "_write_metric_log", lambda *_args, **_kwargs: None
    )

    ok, message = raw_power.run_raw_power_metric.__wrapped__(
        context,
        low_freq=1.0,
        high_freq=30.0,
        step_hz=1.0,
        mask_edge_effects=False,
        bands=[],
        selected_channels=None,
        read_raw_fif_fn=lambda *_args, **_kwargs: raw,
        tfr_grid_fn=compute_stub,
        load_tensor_filter_inheritance_fn=lambda _context: SimpleNamespace(
            notches=(), notch_widths=()
        ),
    )

    assert ok is False
    assert "stop after picks" in message
    assert captured_picks == ["A"]


def test_periodic_aperiodic_closes_raw_when_all_channels_are_bad(
    tmp_path: Path,
) -> None:
    raw = _RawStub(("B",), bads=("B",))

    with pytest.raises(ValueError, match="No valid channels"):
        periodic_aperiodic_prepare.prepare_periodic_aperiodic_runtime(
            SimpleNamespace(selected_channels=None),
            SimpleNamespace(input_path=tmp_path / "raw.fif"),
            read_raw_fif_fn=lambda *_args, **_kwargs: raw,
        )

    assert raw.closed is True


@pytest.mark.parametrize(
    ("channels", "bads", "selected_pairs", "expected_error"),
    [
        (("B",), ("B",), None, "at least 2 valid channels"),
        (
            ("A", "B"),
            ("B",),
            [("A", "B")],
            "No valid selected directed pairs",
        ),
    ],
)
def test_trgc_closes_raw_when_bad_exclusion_rejects_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channels: tuple[str, ...],
    bads: tuple[str, ...],
    selected_pairs: list[tuple[str, str]] | None,
    expected_error: str,
) -> None:
    context = RecordContext(tmp_path, "sub-test", "record-test")
    input_path = PathResolver(context).preproc_root / "finish" / "raw.fif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch()
    raw = _RawStub(channels, bads=bads)
    monkeypatch.setattr(
        connectivity_trgc,
        "load_tensor_filter_inheritance",
        lambda _context: SimpleNamespace(notches=(), notch_widths=()),
    )
    monkeypatch.setattr(
        connectivity_trgc,
        "indicator_from_log",
        lambda *_args: "green",
    )

    with pytest.raises(ValueError, match=expected_error):
        connectivity_trgc._prepare_trgc_backend_inputs(
            context,
            low_freq=1.0,
            high_freq=30.0,
            step_hz=1.0,
            mask_edge_effects=False,
            bands=[],
            selected_channels=None,
            selected_pairs=selected_pairs,
            time_resolution_s=0.5,
            hop_s=0.025,
            method="morlet",
            mt_time_bandwidth_product=4.0,
            mt_min_cycles=3.0,
            mt_max_cycles=None,
            min_cycles=3.0,
            max_cycles=None,
            gc_n_lags=20,
            group_by_samples=False,
            round_ms=50.0,
            read_raw_fif_fn=lambda *_args, **_kwargs: raw,
        )

    assert raw.closed is True


def test_direct_connectivity_does_not_compute_an_all_bad_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = RecordContext(tmp_path, "sub-test", "record-test")
    input_path = PathResolver(context).preproc_root / "finish" / "raw.fif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.touch()
    raw = _RawStub(("A", "B", "C"), bads=("B",))
    compute_called = False

    def compute_stub(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal compute_called
        compute_called = True
        raise AssertionError("Connectivity computation must not start.")

    monkeypatch.setattr(
        connectivity_undirected.svc,
        "indicator_from_log",
        lambda *_args: "green",
    )
    monkeypatch.setattr(
        connectivity_undirected.svc,
        "_write_metric_log",
        lambda *_args, **_kwargs: None,
    )

    ok, message = (
        connectivity_undirected.run_undirected_connectivity_metric.__wrapped__(
            context,
            metric_key="coherence",
            connectivity_metric="coh",
            low_freq=1.0,
            high_freq=30.0,
            step_hz=1.0,
            mask_edge_effects=False,
            bands=[],
            selected_channels=None,
            selected_pairs=[("A", "B")],
            read_raw_fif_fn=lambda *_args, **_kwargs: raw,
            conn_grid_fn=compute_stub,
        )
    )

    assert ok is False
    assert "No valid selected pairs" in message
    assert compute_called is False


def test_shared_connectivity_coordinator_uses_only_usable_channels() -> None:
    raw = _RawStub(("A", "B", "C"), bads=("B",))

    picks, pairs = _normalize_pairs(
        tensor_service,
        raw,
        selected_channels=None,
        selected_pairs=None,
        directed=False,
    )

    assert picks == ["A", "C"]
    assert pairs == [("A", "C")]
    with pytest.raises(ValueError, match="No valid selected connectivity pairs"):
        _normalize_pairs(
            tensor_service,
            raw,
            selected_channels=None,
            selected_pairs=[("A", "B")],
            directed=False,
        )


def test_gui_inventory_exposes_only_usable_finish_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PySide6")
    mne = pytest.importorskip("mne")
    from PySide6.QtWidgets import QApplication

    from lfptensorpipe.gui.shell.main_window_logic import MainWindow

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    context = RecordContext(tmp_path, "sub-test", "record-test")
    finish_path = PathResolver(context).preproc_root / "finish" / "raw.fif"
    finish_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["A", "B", "C"], sfreq=100.0, ch_types="dbs")
    raw = mne.io.RawArray(np.zeros((3, 100)), info, verbose="ERROR")
    raw.info["bads"] = ["B"]
    raw.save(str(finish_path), overwrite=False, verbose="ERROR")
    raw.close()

    window = MainWindow(
        config_store=AppConfigStore(repo_root=tmp_path / "app-config"),
        demo_data_root=tmp_path / "demo",
        auto_load_dataset=False,
        enable_plots=False,
    )
    try:
        window._refresh_tensor_channel_state(context)
        assert window._tensor_available_channels == ("A", "C")
        assert window._tensor_channel_inventory.bad_channels == ("B",)
        assert "Excluded bad channel(s): B" in window._tensor_channels_button.toolTip()

        window._tensor_selected_channels_by_metric["raw_power"] = ("B",)
        window._tensor_selected_pairs_by_metric["coherence"] = (("A", "B"),)
        window._refresh_tensor_channel_state(context)
        assert window._tensor_selected_channels_by_metric["raw_power"] == ()
        assert window._tensor_selected_pairs_by_metric["coherence"] == ()

        second_context = RecordContext(tmp_path, "sub-test", "record-second")
        second_finish_path = (
            PathResolver(second_context).preproc_root / "finish" / "raw.fif"
        )
        second_finish_path.parent.mkdir(parents=True, exist_ok=True)
        second_info = mne.create_info(["D", "E"], sfreq=100.0, ch_types="dbs")
        second_raw = mne.io.RawArray(np.zeros((2, 100)), second_info, verbose="ERROR")
        second_raw.save(str(second_finish_path), overwrite=False, verbose="ERROR")
        second_raw.close()

        window._refresh_tensor_channel_state(second_context)
        assert window._tensor_available_channels == ("D", "E")
        assert window._tensor_selected_channels_by_metric["raw_power"] == (
            "D",
            "E",
        )
    finally:
        window.close()
        app.processEvents()
