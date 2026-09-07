"""Regression tests for Tensor Raw ownership and bad-channel state consistency."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from lfptensorpipe.app import tensor_metric_panel_state
from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import RunLogRecord, read_run_log, write_run_log
from lfptensorpipe.app.shared.generation_lineage import params_with_generation_lineage
from lfptensorpipe.app.tensor import indicator as indicator_module
from lfptensorpipe.app.tensor import service as tensor_service
from lfptensorpipe.app.tensor.lineage import tensor_metric_lineage_is_current
from lfptensorpipe.app.tensor.runners import connectivity_undirected, raw_power
from lfptensorpipe.app.tensor.selectors import (
    TensorChannelInventory,
    load_tensor_channel_inventory,
)
from lfptensorpipe.io.pkl_io import load_pkl


@pytest.mark.parametrize("metric_key", ["raw_power", "coherence"])
@pytest.mark.parametrize(
    "outcome",
    ["all_bad", "empty", "compute_error", "write_error", "success", "read_error"],
)
def test_runner_closes_each_opened_raw_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_key: str,
    outcome: str,
) -> None:
    context = RecordContext(tmp_path, "sub-test", "record-test")
    input_path = (
        PathResolver(context).preproc_step_dir("finish", create=True) / "raw.fif"
    )
    input_path.touch()
    raw = SimpleNamespace(
        ch_names=["C1", "C2", "C3"],
        info={"sfreq": 128.0, "bads": ["C3"]},
        close=Mock(),
    )
    reader = Mock(return_value=raw)
    if outcome == "read_error":
        reader.side_effect = OSError("test read failure")

    def compute(_raw: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        assert _raw is raw
        assert kwargs["picks"] == ["C1", "C2"]
        if outcome == "compute_error":
            raise RuntimeError("test compute failure")
        channels = kwargs.get("pairs", kwargs["picks"])
        freqs = np.asarray(kwargs["freqs"], dtype=float)
        return np.ones((1, len(channels), len(freqs), 2)), {
            "axes": {"channel": channels, "freq": freqs, "time": [0.0, 0.25]},
            "params": {},
        }

    compute_mock = Mock(side_effect=compute)
    writer = Mock()
    if outcome == "write_error":
        writer.side_effect = OSError("test write failure")
    failure_log = Mock()
    monkeypatch.setattr(tensor_service, "indicator_from_log", lambda *_args: "green")
    monkeypatch.setattr(tensor_service, "_write_metric_log", failure_log)
    monkeypatch.setattr(tensor_service, "_write_outputs_atomically", writer)
    monkeypatch.setattr(
        tensor_service,
        "load_tensor_filter_inheritance",
        lambda _context: SimpleNamespace(notches=(), notch_widths=()),
    )
    channels = ["C1", "C2", "C3"]
    pairs = [("C1", "C2"), ("C1", "C3")]
    if outcome == "all_bad":
        channels, pairs = ["C3"], [("C1", "C3")]
    elif outcome == "empty":
        channels, pairs = [], []
    kwargs = dict(
        low_freq=8.0,
        high_freq=24.0,
        step_hz=8.0,
        mask_edge_effects=False,
        bands=[],
        selected_channels=channels,
        read_raw_fif_fn=reader,
    )
    if metric_key == "raw_power":
        result = raw_power.run_raw_power_metric.__wrapped__(
            context, **kwargs, tfr_grid_fn=compute_mock
        )
    else:
        result = connectivity_undirected.run_undirected_connectivity_metric.__wrapped__(
            context,
            **kwargs,
            metric_key=metric_key,
            connectivity_metric="coh",
            selected_pairs=pairs,
            conn_grid_fn=compute_mock,
        )

    ok, message = result
    assert ok is (outcome == "success"), message
    assert raw.close.call_count == (0 if outcome == "read_error" else 1)
    assert compute_mock.call_count == (
        0 if outcome in {"all_bad", "empty", "read_error"} else 1
    )
    assert failure_log.call_count == (0 if ok else 1)
    if outcome.endswith("_error"):
        assert f"test {outcome.removesuffix('_error')} failure" in message


def _write_finish_record(context: RecordContext, *, bad_channel: str = "C3") -> Path:
    resolver = PathResolver(context)
    raw = mne.io.RawArray(
        np.random.default_rng(42).normal(size=(4, 640)) * 1e-6,
        mne.create_info(["C1", "C2", "C3", "C4"], 128.0, ch_types="dbs"),
        verbose="ERROR",
    )
    raw.info["bads"] = [bad_channel]
    try:
        for step, generation, params, inputs in (
            ("raw", "1" * 32, {}, {}),
            ("finish", "2" * 32, {"source_step": "raw"}, {"preproc/raw": "1" * 32}),
        ):
            step_dir = resolver.preproc_step_dir(step, create=True)
            path = step_dir / "raw.fif"
            raw.save(str(path), overwrite=False, verbose="ERROR")
            write_run_log(
                step_dir / "lfptensorpipe_log.json",
                RunLogRecord(
                    step=step,
                    completed=True,
                    params=params_with_generation_lineage(
                        params,
                        result_generation_id=generation,
                        input_generations=inputs,
                    ),
                    output_path=str(path),
                ),
            )
    finally:
        raw.close()
    return resolver.preproc_step_dir("finish", create=False) / "raw.fif"


@pytest.fixture(scope="module", params=["raw_power", "coherence"])
def accepted_metric(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> SimpleNamespace:
    metric_key = request.param
    context = RecordContext(
        tmp_path_factory.mktemp(metric_key), "sub-test", "record-test"
    )
    finish_path = _write_finish_record(context)
    kwargs = dict(
        low_freq=8.0,
        high_freq=24.0,
        step_hz=8.0,
        hop_s=0.25,
        mask_edge_effects=False,
        bands=[],
        selected_channels=["C1", "C2", "C3"],
    )
    if metric_key == "raw_power":
        result = raw_power.run_raw_power_metric(context, **kwargs)
        selector_key = "selected_channels"
        requested = ["C1", "C2", "C3"]
        effective = ["C1", "C2"]
        changed = ["C1", "C4"]
        rejected = ["C3"]
        expected_axis = ["C1", "C2"]
    else:
        result = connectivity_undirected.run_undirected_connectivity_metric(
            context,
            **kwargs,
            metric_key=metric_key,
            connectivity_metric="coh",
            selected_pairs=[("C1", "C2"), ("C1", "C3")],
        )
        selector_key = "selected_pairs"
        requested = [["C1", "C2"], ["C1", "C3"]]
        effective = [["C1", "C2"]]
        changed = [["C1", "C4"]]
        rejected = [["C1", "C3"]]
        expected_axis = [("C1", "C2")]
    assert result[0], result[1]
    resolver = PathResolver(context)
    assert tensor_metric_lineage_is_current(resolver, metric_key)
    log = read_run_log(tensor_service.tensor_metric_log_path(resolver, metric_key))
    assert log["params"][selector_key] == effective
    output = load_pkl(tensor_service.tensor_metric_tensor_path(resolver, metric_key))
    assert list(output["meta"]["axes"]["channel"]) == expected_axis
    return SimpleNamespace(
        context=context,
        metric_key=metric_key,
        selector_key=selector_key,
        effective=effective,
        changed=changed,
        rejected=rejected,
        params={
            "low_freq_hz": 8.0,
            "high_freq_hz": 24.0,
            "freq_step_hz": 8.0,
            "hop_s": 0.25,
            selector_key: requested,
        },
        finish_path=finish_path,
        inventory=load_tensor_channel_inventory(finish_path),
    )


def test_public_indicator_compares_effective_selectors(accepted_metric: Any) -> None:
    metric = accepted_metric
    for inventory_kwargs in ({}, {"channel_inventory": metric.inventory}):
        for selected, expected in (
            (metric.params[metric.selector_key], "green"),
            (metric.effective, "green"),
            (metric.changed, "yellow"),
            (metric.rejected, "yellow"),
            ([], "yellow"),
        ):
            assert (
                tensor_metric_panel_state(
                    metric.context,
                    metric_key=metric.metric_key,
                    metric_params={**metric.params, metric.selector_key: selected},
                    mask_edge_effects=False,
                    **inventory_kwargs,
                )
                == expected
            )
        assert (
            tensor_metric_panel_state(
                metric.context,
                metric_key=metric.metric_key,
                metric_params={**metric.params, "high_freq_hz": 16.0},
                mask_edge_effects=False,
                **inventory_kwargs,
            )
            == "yellow"
        )


def test_public_indicator_reuses_or_loads_inventory(
    accepted_metric: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = accepted_metric
    loader = Mock(wraps=load_tensor_channel_inventory)
    monkeypatch.setattr(
        indicator_module, "load_tensor_channel_inventory", loader, raising=False
    )
    kwargs = dict(
        metric_key=metric.metric_key,
        metric_params=metric.params,
        mask_edge_effects=False,
    )
    assert (
        tensor_metric_panel_state(
            metric.context, **kwargs, channel_inventory=metric.inventory
        )
        == "green"
    )
    loader.assert_not_called()
    empty_inventory = TensorChannelInventory(
        all_channels=metric.inventory.all_channels,
        bad_channels=metric.inventory.all_channels,
        usable_channels=(),
    )
    assert (
        tensor_metric_panel_state(
            metric.context, **kwargs, channel_inventory=empty_inventory
        )
        == "yellow"
    )
    loader.assert_not_called()
    assert tensor_metric_panel_state(metric.context, **kwargs) == "green"
    loader.assert_called_once_with(metric.finish_path)


def test_unreadable_inventory_does_not_validate_unfiltered_selectors(
    accepted_metric: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = accepted_metric
    loader = Mock(side_effect=OSError("test inventory read failure"))
    monkeypatch.setattr(
        indicator_module, "load_tensor_channel_inventory", loader, raising=False
    )
    assert (
        tensor_metric_panel_state(
            metric.context,
            metric_key=metric.metric_key,
            metric_params={**metric.params, metric.selector_key: metric.effective},
            mask_edge_effects=False,
        )
        == "yellow"
    )
    loader.assert_called_once_with(metric.finish_path)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("no_context", "gray"),
        ("no_log", "gray"),
        ("failed", "yellow"),
        ("stale", "yellow"),
    ],
)
def test_indicator_early_states_do_not_read_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    expected: str,
) -> None:
    context = RecordContext(tmp_path, "sub-test", "record-test")
    resolver = PathResolver(context)
    tensor_service.tensor_metric_log_path(resolver, "raw_power", create=True).touch()
    monkeypatch.setattr(
        indicator_module,
        "_read_payload",
        lambda _path: (None if state == "no_log" else {"completed": state != "failed"}),
    )
    monkeypatch.setattr(
        indicator_module, "tensor_metric_lineage_is_current", lambda *_args: False
    )
    loader = Mock(side_effect=AssertionError("Unexpected inventory read"))
    monkeypatch.setattr(
        indicator_module, "load_tensor_channel_inventory", loader, raising=False
    )
    assert (
        tensor_metric_panel_state(
            None if state == "no_context" else context,
            metric_key="raw_power",
            metric_params={},
            mask_edge_effects=False,
        )
        == expected
    )
    loader.assert_not_called()


def test_selector_tooltips_follow_record_clear_and_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from lfptensorpipe.gui.shell.main_window_logic import MainWindow

    app = QApplication.instance() or QApplication([])
    first = RecordContext(tmp_path, "sub-test", "first")
    second = RecordContext(tmp_path, "sub-test", "second")
    _write_finish_record(first, bad_channel="C3")
    _write_finish_record(second, bad_channel="C2")
    window = MainWindow(
        config_store=AppConfigStore(repo_root=tmp_path / "app-config"),
        demo_data_root=tmp_path / "demo",
        auto_load_dataset=False,
        enable_plots=False,
    )
    try:
        window._refresh_tensor_channel_state(first)
        buttons = (window._tensor_channels_button, window._tensor_pairs_button)
        for button in buttons:
            assert "Excluded bad channel(s): C3." in button.toolTip()
        window._refresh_tensor_channel_state(None)
        assert window._tensor_channel_inventory is None
        assert window._tensor_available_channels == ()
        for button in buttons:
            assert button.text().endswith("(0/0)")
            assert not button.isEnabled()
            assert button.toolTip() == (
                "Select a record to view available Tensor channels and pairs."
            )
        window._refresh_tensor_channel_state(second)
        for button in buttons:
            assert "Excluded bad channel(s): C2." in button.toolTip()
            assert "C3" not in button.toolTip()
    finally:
        window.close()
        app.processEvents()
