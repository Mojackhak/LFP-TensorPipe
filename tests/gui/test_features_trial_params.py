from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
)

from lfptensorpipe.app.shared.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.runlog_store import RunLogRecord, write_run_log
from lfptensorpipe.gui.shell import main_window_layout as _main_window_layout
from lfptensorpipe.gui.shell.features_axes import MainWindowFeaturesAxesMixin
from lfptensorpipe.gui.shell.features_run import MainWindowFeaturesRunMixin
from lfptensorpipe.gui.shell.features_subset import MainWindowFeaturesSubsetMixin
from lfptensorpipe.gui.shell.features_trials import MainWindowFeaturesTrialsMixin
from lfptensorpipe.gui.shell.record_params_apply_features import (
    MainWindowRecordParamsApplyFeaturesMixin,
)
from lfptensorpipe.gui.shell.record_params_snapshot_collect import (
    MainWindowRecordParamsSnapshotCollectMixin,
)
from lfptensorpipe.gui.shell.record_params_snapshot_logs import (
    MainWindowRecordParamsSnapshotLogsMixin,
)
from lfptensorpipe.gui.shell.record_params_store import MainWindowRecordParamsStoreMixin


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FeaturesTrialParamsHarness(
    MainWindowRecordParamsStoreMixin,
    MainWindowRecordParamsSnapshotLogsMixin,
    MainWindowRecordParamsSnapshotCollectMixin,
    MainWindowRecordParamsApplyFeaturesMixin,
    MainWindowFeaturesAxesMixin,
    MainWindowFeaturesSubsetMixin,
    MainWindowFeaturesRunMixin,
    MainWindowFeaturesTrialsMixin,
):
    def __init__(self, context: RecordContext, trial_slug: str) -> None:
        self._context = context
        self._trial_slug = trial_slug
        self._record_param_dirty_keys: set[str] = set()
        self._record_param_syncing = False
        self._features_axes_by_metric: dict[str, dict[str, list[dict[str, object]]]] = {}
        self._features_trial_params_by_slug: dict[str, dict[str, object]] = {}
        self._features_plot_advance_params = self._load_features_plot_advance_defaults()
        self._features_files: list[dict[str, object]] = []
        self._features_filtered_files: list[dict[str, object]] = []
        self._features_axis_metric_combo = QComboBox()
        self._features_filter_feature_edit = QLineEdit()
        self._features_subset_band_combo = QComboBox()
        self._features_subset_channel_combo = QComboBox()
        self._features_subset_region_combo = QComboBox()
        self._features_x_label_edit = QLineEdit()
        self._features_y_label_edit = QLineEdit()
        self._features_cbar_label_edit = QLineEdit()
        self._features_available_table = QTableWidget(0, 3)
        self._features_axis_bands_button = None
        self._features_axis_times_button = None
        self._features_axis_apply_all_button = None
        self._features_run_extract_button = None
        self._features_import_button = None
        self._features_export_button = None
        self._features_refresh_button = None
        self._features_plot_button = None
        self._features_plot_advance_button = None
        self._features_plot_export_button = None
        self._features_extract_indicator = None
        self._features_last_plot_figure = None
        self._features_last_plot_data = None

    def _record_context(self) -> RecordContext | None:
        return self._context

    def _shared_stage_trial_slug(self) -> str | None:
        return self._trial_slug

    def _current_alignment_paradigm_slug(self) -> str | None:
        return self._trial_slug

    def _current_features_paradigm_slug(self) -> str | None:
        return self._trial_slug

    @staticmethod
    def _load_pickle(path: str | Path) -> object:
        return pd.read_pickle(path)

    @staticmethod
    def _load_features_plot_advance_defaults() -> dict[str, object]:
        return {
            "transform_mode": "none",
            "normalize_mode": "none",
            "baseline_mode": "mean",
            "baseline_percent_ranges": [],
        }

    @staticmethod
    def _load_features_axis_defaults(*, metric_key: str, axis_key: str) -> list[dict[str, object]]:
        _ = (metric_key, axis_key)
        return []

    @staticmethod
    def _normalize_feature_axis_rows(
        value: object,
        *,
        min_start: float,
        max_end: float | None = None,
        allow_duplicate_names: bool = False,
    ) -> list[dict[str, float | str]]:
        return _main_window_layout.normalize_feature_axis_rows(
            value,
            min_start=min_start,
            max_end=max_end,
            allow_duplicate_names=allow_duplicate_names,
        )

    def _refresh_features_controls(self) -> None:
        return

    @staticmethod
    def statusBar() -> object:
        class _StatusBar:
            def showMessage(self, message: str) -> None:
                _ = message

        return _StatusBar()


class _FeaturesSharedTrialRestoreHarness(_FeaturesTrialParamsHarness):
    def __init__(self, context: RecordContext, trial_slug: str) -> None:
        super().__init__(context, trial_slug)
        self._shared_trial_slug = trial_slug
        self._features_paradigms = [
            {
                "name": trial_slug,
                "trial_slug": trial_slug,
                "slug": trial_slug,
            }
        ]
        self._features_paradigm_list = QListWidget()
        item = QListWidgetItem(trial_slug)
        item.setData(Qt.UserRole, trial_slug)
        self._features_paradigm_list.addItem(item)
        self._features_paradigm_list.setCurrentRow(-1)

    def _shared_stage_trial_slug(self) -> str | None:
        return self._shared_trial_slug

    def _current_features_paradigm_slug(self) -> str | None:
        return MainWindowFeaturesTrialsMixin._current_features_paradigm_slug(self)


def _prepare_trial_files(tmp_path: Path, *, trial_slug: str) -> RecordContext:
    context = RecordContext(
        project_root=tmp_path / "project",
        subject="subject-1",
        record="record-1",
    )
    resolver = PathResolver(context)
    alignment_metric_dir = resolver.alignment_root / trial_slug / "raw_power"
    alignment_metric_dir.mkdir(parents=True, exist_ok=True)
    (alignment_metric_dir / "na-raw.pkl").write_bytes(b"placeholder")
    return context


def _write_feature_payload(
    context: RecordContext,
    *,
    trial_slug: str,
    relative_stem: str,
) -> Path:
    resolver = PathResolver(context)
    output_path = resolver.features_root / trial_slug / f"{relative_stem}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = pd.DataFrame(
        {
            "Band": ["alpha", "beta"],
            "Channel": ["Ch1", "Ch2"],
            "R1_in": [True, False],
            "Value": [1.0, 2.0],
        }
    )
    payload.to_pickle(output_path)
    return output_path


def test_restore_features_trial_params_falls_back_to_completed_log_axes(
    qapp: QApplication, tmp_path: Path
) -> None:
    _ = qapp
    trial_slug = "trial-a"
    context = _prepare_trial_files(tmp_path, trial_slug=trial_slug)
    resolver = PathResolver(context)
    write_run_log(
        resolver.features_root / trial_slug / "lfptensorpipe_log.json",
        RunLogRecord(
            step="run_extract_features",
            completed=True,
            params={
                "trial_slug": trial_slug,
                "axes_by_metric": {
                    "raw_power": {
                        "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
                        "times": [{"name": "phase-1", "start": 5.0, "end": 25.0}],
                    }
                },
            },
            message="ok",
        ),
    )

    harness = _FeaturesTrialParamsHarness(context, trial_slug)

    harness._restore_features_trial_params(trial_slug, respect_dirty_keys=False)

    assert harness._features_axes_by_metric == {
        "raw_power": {
            "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
            "times": [{"name": "phase-1", "start": 5.0, "end": 25.0}],
        }
    }
    assert harness._features_axis_metric_combo.currentData() == "raw_power"


def test_apply_features_trial_params_restores_selected_feature_before_subset(
    qapp: QApplication, tmp_path: Path
) -> None:
    _ = qapp
    trial_slug = "trial-a"
    context = _prepare_trial_files(tmp_path, trial_slug=trial_slug)
    _write_feature_payload(
        context,
        trial_slug=trial_slug,
        relative_stem="raw_power/mean-trace",
    )

    source = _FeaturesTrialParamsHarness(context, trial_slug)
    source._refresh_features_available_files()
    source._features_available_table.setCurrentCell(0, 0)
    source._features_filter_feature_edit.setText("trace")
    source._sync_features_subset_options(
        preferred_selection={"band": "alpha", "channel": "Ch1", "region": "R1"}
    )
    source._features_x_label_edit.setText("Time")

    params = source._collect_current_features_trial_params(trial_slug)

    restored = _FeaturesTrialParamsHarness(context, trial_slug)
    restored._apply_features_trial_params_to_ui(params, respect_dirty_keys=False)

    assert params["selected_relative_stem"] == "raw_power/mean-trace"
    assert restored._selected_features_file() is not None
    assert restored._selected_features_file()["relative_stem"] == "raw_power/mean-trace"
    assert restored._current_features_subset_selection() == {
        "band": "alpha",
        "channel": "Ch1",
        "region": "R1",
    }
    assert restored._features_filter_feature_edit.text() == "trace"
    assert restored._features_x_label_edit.text() == "Time"


def test_features_trial_params_roundtrip_through_record_ui_state(
    qapp: QApplication, tmp_path: Path
) -> None:
    _ = qapp
    trial_slug = "trial-a"
    context = _prepare_trial_files(tmp_path, trial_slug=trial_slug)
    _write_feature_payload(
        context,
        trial_slug=trial_slug,
        relative_stem="raw_power/mean-trace",
    )

    source = _FeaturesTrialParamsHarness(context, trial_slug)
    source._refresh_features_available_files()
    source._features_available_table.setCurrentCell(0, 0)
    source._features_filter_feature_edit.setText("trace")
    source._sync_features_subset_options(
        preferred_selection={"band": "alpha", "channel": "Ch1", "region": "R1"}
    )

    payload = {"features": source._collect_features_record_params_snapshot()}
    assert source._write_record_params_payload(
        context,
        params=payload,
        reason="test_features_roundtrip",
    )

    ok, loaded_payload, message = source._load_record_params_payload(context)
    assert ok, message

    restored = _FeaturesTrialParamsHarness(context, trial_slug)
    restored._apply_record_params_features_snapshot(loaded_payload)

    assert restored._selected_features_file() is not None
    assert restored._selected_features_file()["relative_stem"] == "raw_power/mean-trace"
    assert restored._current_features_subset_selection() == {
        "band": "alpha",
        "channel": "Ch1",
        "region": "R1",
    }


def test_features_restore_uses_shared_trial_when_no_features_row_is_selected(
    qapp: QApplication, tmp_path: Path
) -> None:
    _ = qapp
    trial_slug = "trial-a"
    context = _prepare_trial_files(tmp_path, trial_slug=trial_slug)
    _write_feature_payload(
        context,
        trial_slug=trial_slug,
        relative_stem="raw_power/mean-trace",
    )

    source = _FeaturesTrialParamsHarness(context, trial_slug)
    source._refresh_features_available_files()
    source._features_available_table.setCurrentCell(0, 0)
    source._features_filter_feature_edit.setText("trace")
    source._sync_features_subset_options(
        preferred_selection={"band": "alpha", "channel": "Ch1", "region": "R1"}
    )
    source._features_x_label_edit.setText("Time")

    payload = {"features": source._collect_features_record_params_snapshot()}

    restored = _FeaturesSharedTrialRestoreHarness(context, trial_slug)
    assert restored._features_paradigm_list is not None
    assert restored._features_paradigm_list.currentRow() == -1

    restored._apply_record_params_features_snapshot(payload)

    assert restored._current_features_paradigm_slug() == trial_slug
    assert restored._selected_features_file() is not None
    assert restored._selected_features_file()["relative_stem"] == "raw_power/mean-trace"
    assert restored._features_filter_feature_edit.text() == "trace"
    assert restored._current_features_subset_selection() == {
        "band": "alpha",
        "channel": "Ch1",
        "region": "R1",
    }
    assert restored._features_x_label_edit.text() == "Time"
