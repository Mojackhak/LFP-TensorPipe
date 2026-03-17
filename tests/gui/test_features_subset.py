import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QComboBox

from lfptensorpipe.gui.shell.features_subset import MainWindowFeaturesSubsetMixin


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _combo_values(combo: QComboBox) -> list[str]:
    return [str(combo.itemData(index) or "") for index in range(combo.count())]


class _SubsetHarness(MainWindowFeaturesSubsetMixin):
    def __init__(self, payload: pd.DataFrame | None) -> None:
        self._payload = payload
        self._features_subset_band_combo = QComboBox()
        self._features_subset_channel_combo = QComboBox()
        self._features_subset_region_combo = QComboBox()

    def _selected_features_payload(self) -> pd.DataFrame | None:
        return None if self._payload is None else self._payload.copy()


def _sample_payload() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Band": "alpha", "Channel": "Ch1", "R1_in": 1, "R2_in": 0},
            {"Band": "beta", "Channel": "Ch1", "R1_in": 1, "R2_in": 0},
            {"Band": "alpha", "Channel": "Ch2", "R1_in": 0, "R2_in": 1},
            {"Band": "gamma", "Channel": "Ch3", "R1_in": 0, "R2_in": 1},
        ]
    )


def test_sync_features_subset_options_constrains_other_candidates() -> None:
    harness = _SubsetHarness(_sample_payload())

    harness._sync_features_subset_options(
        preferred_selection={"band": "", "channel": "Ch1", "region": ""}
    )

    assert harness._features_subset_channel_combo.currentData() == "Ch1"
    assert _combo_values(harness._features_subset_band_combo) == ["", "alpha", "beta"]
    assert _combo_values(harness._features_subset_channel_combo) == [
        "",
        "Ch1",
        "Ch2",
        "Ch3",
    ]
    assert _combo_values(harness._features_subset_region_combo) == ["", "R1"]


def test_sync_features_subset_options_resets_invalid_combination_to_all() -> None:
    harness = _SubsetHarness(_sample_payload())

    harness._sync_features_subset_options(
        preferred_selection={"band": "", "channel": "Ch1", "region": "R2"}
    )

    assert harness._features_subset_band_combo.currentData() == ""
    assert harness._features_subset_channel_combo.currentData() == ""
    assert harness._features_subset_region_combo.currentData() == ""
    assert _combo_values(harness._features_subset_band_combo) == [
        "",
        "alpha",
        "beta",
        "gamma",
    ]
    assert _combo_values(harness._features_subset_channel_combo) == [
        "",
        "Ch1",
        "Ch2",
        "Ch3",
    ]
    assert _combo_values(harness._features_subset_region_combo) == ["", "R1", "R2"]


def test_apply_features_subset_filters_matches_current_selection() -> None:
    payload = _sample_payload()
    harness = _SubsetHarness(payload)
    harness._sync_features_subset_options(
        preferred_selection={"band": "beta", "channel": "Ch1", "region": "R1"}
    )

    filtered = harness._apply_features_subset_filters(payload)

    assert filtered.shape[0] == 1
    assert filtered.iloc[0]["Band"] == "beta"
    assert filtered.iloc[0]["Channel"] == "Ch1"
    assert int(filtered.iloc[0]["R1_in"]) == 1
