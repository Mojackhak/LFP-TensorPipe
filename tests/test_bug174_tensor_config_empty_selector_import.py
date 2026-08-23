"""Regression for preserving empty filtered selectors during config import."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.app.tensor import service as tensor_service
from lfptensorpipe.app.tensor.orchestration_plan_validation import (
    prepare_metric_plan_inputs,
    validate_metric_storage_params,
)
from lfptensorpipe.gui.shell.main_window_logic import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _payload() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    path = root / "demo/configs/tensor/lfptensorpipe_tensor_config.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("metric_key", "selector_key", "invalid_value", "warning_fragment"),
    [
        (
            "raw_power",
            "selected_channels",
            ["unavailable-channel"],
            "ignored 1 unavailable channel(s)",
        ),
        (
            "coherence",
            "selected_pairs",
            [["unavailable-source", "unavailable-target"]],
            "ignored 1 unavailable pair(s)",
        ),
    ],
)
def test_config_import_preserves_empty_filtered_selector_draft(
    tmp_path: Path,
    qapp: QApplication,
    metric_key: str,
    selector_key: str,
    invalid_value: object,
    warning_fragment: str,
) -> None:
    payload = _payload()
    available_channels = tuple(
        payload["tensor"]["metric_params"]["raw_power"]["selected_channels"]
    )
    payload["tensor"]["metric_params"][metric_key][selector_key] = invalid_value
    context = RecordContext(tmp_path, "sub-test", "record-test")
    window = MainWindow(
        config_store=AppConfigStore(repo_root=tmp_path / "app-config"),
        demo_data_root=tmp_path / "demo",
        auto_load_dataset=False,
        enable_plots=False,
    )
    try:
        normalized, warnings = window._normalize_tensor_config_import_payload(
            payload,
            context=context,
            available_channels=available_channels,
        )
    finally:
        window.close()
        qapp.processEvents()

    metric_params = normalized["metric_params"][metric_key]
    assert metric_params[selector_key] == []
    assert any(warning_fragment in warning for warning in warnings)
    assert f"Invalid value restored for {metric_key}.{selector_key}." not in warnings

    with pytest.raises(ValueError, match="requires at least one selected"):
        validate_metric_storage_params(
            metric_key=metric_key,
            metric_label=window._tensor_metric_display_name(metric_key),
            metric_params=metric_params,
        )
    with pytest.raises(ValueError, match="requires at least one selected"):
        prepare_metric_plan_inputs(
            tensor_service,
            context,
            metric_key=metric_key,
            metric_label=window._tensor_metric_display_name(metric_key),
            metric_params=metric_params,
        )


@pytest.mark.parametrize(
    ("metric_key", "selector_key"),
    [
        ("raw_power", "selected_channels"),
        ("coherence", "selected_pairs"),
    ],
)
def test_config_import_keeps_valid_nonempty_selectors(
    tmp_path: Path,
    qapp: QApplication,
    metric_key: str,
    selector_key: str,
) -> None:
    payload = _payload()
    expected = deepcopy(payload["tensor"]["metric_params"][metric_key][selector_key])
    available_channels = tuple(
        payload["tensor"]["metric_params"]["raw_power"]["selected_channels"]
    )
    window = MainWindow(
        config_store=AppConfigStore(repo_root=tmp_path / "app-config"),
        demo_data_root=tmp_path / "demo",
        auto_load_dataset=False,
        enable_plots=False,
    )
    try:
        normalized, warnings = window._normalize_tensor_config_import_payload(
            payload,
            context=RecordContext(tmp_path, "sub-test", "record-test"),
            available_channels=available_channels,
        )
    finally:
        window.close()
        qapp.processEvents()

    assert normalized["metric_params"][metric_key][selector_key] == expected
    assert not any(
        "unavailable channel(s)" in warning or "unavailable pair(s)" in warning
        for warning in warnings
    )


def test_explicit_empty_channel_selector_keeps_existing_default_repair(
    tmp_path: Path,
    qapp: QApplication,
) -> None:
    payload = _payload()
    expected = deepcopy(
        payload["tensor"]["metric_params"]["raw_power"]["selected_channels"]
    )
    available_channels = tuple(expected)
    payload["tensor"]["metric_params"]["raw_power"]["selected_channels"] = []
    window = MainWindow(
        config_store=AppConfigStore(repo_root=tmp_path / "app-config"),
        demo_data_root=tmp_path / "demo",
        auto_load_dataset=False,
        enable_plots=False,
    )
    try:
        normalized, warnings = window._normalize_tensor_config_import_payload(
            payload,
            context=RecordContext(tmp_path, "sub-test", "record-test"),
            available_channels=available_channels,
        )
    finally:
        window.close()
        qapp.processEvents()

    assert normalized["metric_params"]["raw_power"]["selected_channels"] == expected
    assert warnings == ["Invalid value restored for raw_power.selected_channels."]


@pytest.mark.parametrize("invalid_value", [[], "invalid"])
def test_empty_or_invalid_pair_selector_keeps_existing_hard_failure(
    tmp_path: Path,
    qapp: QApplication,
    invalid_value: object,
) -> None:
    payload = _payload()
    available_channels = tuple(
        payload["tensor"]["metric_params"]["raw_power"]["selected_channels"]
    )
    payload["tensor"]["metric_params"]["coherence"]["selected_pairs"] = invalid_value
    window = MainWindow(
        config_store=AppConfigStore(repo_root=tmp_path / "app-config"),
        demo_data_root=tmp_path / "demo",
        auto_load_dataset=False,
        enable_plots=False,
    )
    try:
        with pytest.raises(ValueError, match="has no safe repair"):
            window._normalize_tensor_config_import_payload(
                payload,
                context=RecordContext(tmp_path, "sub-test", "record-test"),
                available_channels=available_channels,
            )
    finally:
        window.close()
        qapp.processEvents()
