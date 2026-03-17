"""Smoke tests for the first GUI vertical slice."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LFPTP_DISABLE_MATLAB_WARMUP", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QGroupBox,
    QLabel,
    QHeaderView,
    QTableWidgetItem,
)

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.gui.dialogs.common_constants import (
    normalize_feature_plot_transform_mode,
)
from lfptensorpipe.gui import main_window as main_window_module
from lfptensorpipe.gui.main_window import (
    AnnotationConfigureDialog,
    AlignmentMethodParamsDialog,
    BUSY_FRAMES,
    ChannelSelectDialog,
    ChannelPairDialog,
    FilterAdvanceDialog,
    LocalizeMatchDialog,
    MainWindow,
    PathsConfigDialog,
    QcAdvanceDialog,
    STAGE_SPECS,
    FeatureAxisConfigureDialog,
    FeaturesPlotAdvanceDialog,
    TensorChannelSelectDialog,
    TensorBandsConfigureDialog,
    TensorMetricAdvanceDialog,
    TensorPairSelectDialog,
    normalize_preproc_viz_psd_params,
    normalize_preproc_viz_tfr_params,
)


def _attach_warning_sink(dialog: object, bucket: list[str]) -> None:
    setattr(
        dialog, "_show_warning", lambda title, message: bucket.append(str(message)) or 0
    )


def _attach_info_sink(dialog: object, bucket: list[str]) -> None:
    setattr(
        dialog,
        "_show_information",
        lambda title, message: bucket.append(str(message)) or 0,
    )


def test_gui_shell_launches_with_expected_default_route() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    assert window.windowTitle() == "LFP-TensorPipe"
    assert window.width() == 900
    assert window.height() == 600
    assert window.minimumWidth() == 900
    assert window.minimumHeight() == 600
    assert app.style().objectName().lower() == "fusion"
    assert app.font().family() == "Arial"
    assert window.statusBar().currentMessage() == "Route: stage/preproc"
    settings_menu = window.menuBar().actions()
    assert any(action.text() == "Settings" for action in settings_menu)
    assert window._settings_configs_action is not None
    assert window._settings_configs_action.text() == "Configs"
    assert len(STAGE_SPECS) == 4
    expected_left = max(200, min(600, round(window.width() * 0.33)))
    assert window._left_column_widget is not None
    assert window._left_column_widget.width() == expected_left
    assert window._stage_stack.minimumWidth() == 400
    assert window._stage_buttons["preproc"].isEnabled()
    assert not window._stage_buttons["tensor"].isEnabled()
    assert not window._stage_buttons["alignment"].isEnabled()
    assert not window._stage_buttons["features"].isEnabled()
    assert window._record_delete_button.text() == "-"
    assert window._preproc_filter_advance_button is not None
    assert window._preproc_filter_advance_button.text() == "Advance"
    assert window._preproc_filter_notches_edit is not None
    defaults = window._load_filter_basic_defaults()
    expected_notches = window._format_filter_notches(defaults.get("notches", []))
    assert window._preproc_filter_notches_edit.text() == expected_notches
    assert window._preproc_filter_low_freq_edit is not None
    assert (
        window._preproc_filter_low_freq_edit.text()
        == f"{defaults.get('l_freq', 1.0):g}"
    )
    assert window._preproc_filter_high_freq_edit is not None
    assert (
        window._preproc_filter_high_freq_edit.text()
        == f"{defaults.get('h_freq', 200.0):g}"
    )
    header = window._preproc_annotations_table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.Stretch
    assert (
        window._preproc_annotations_table.verticalScrollBarPolicy()
        == Qt.ScrollBarAlwaysOn
    )
    assert (
        window._preproc_annotations_table.horizontalScrollBarPolicy()
        == Qt.ScrollBarAlwaysOn
    )
    table_parent = window._preproc_annotations_table.parentWidget()
    button_parent = window._preproc_annotations_edit_button.parentWidget()
    assert table_parent is not None
    assert button_parent is not None
    assert window._preproc_annotations_edit_button.text() == "Configure..."
    assert window._preproc_annotations_save_button.text() == "Apply"
    assert (
        window._preproc_annotations_table.editTriggers()
        == main_window_module.QAbstractItemView.NoEditTriggers
    )
    assert button_parent is not table_parent
    assert not table_parent.isAncestorOf(window._preproc_annotations_edit_button)
    assert window._preproc_ecg_channels_button is not None
    assert "Select Channels" in window._preproc_ecg_channels_button.text()
    assert window._preproc_viz_channels_button is not None
    assert "Select Channels" in window._preproc_viz_channels_button.text()
    assert window._tensor_run_button is not None
    assert window._tensor_run_button.text() == "Build Tensor"
    assert window._tensor_import_button is not None
    assert window._tensor_import_button.text() == "Import Configs..."
    assert window._tensor_export_button is not None
    assert window._tensor_export_button.text() == "Export Configs..."
    assert "raw_power" in window._tensor_metric_checks
    assert "periodic_aperiodic" in window._tensor_metric_checks
    assert "coherence" in window._tensor_metric_checks
    assert "plv" in window._tensor_metric_checks
    assert "trgc" in window._tensor_metric_checks
    assert "psi" in window._tensor_metric_checks
    assert "burst" in window._tensor_metric_checks
    assert window._tensor_metric_checks["raw_power"].text() == ""
    assert window._tensor_metric_name_buttons["raw_power"].text() == "Raw power"
    assert (
        window._tensor_metric_name_buttons["periodic_aperiodic"].text()
        == "Periodic/APeriodic (SpecParam)"
    )
    coherence_spec = next(
        item for item in main_window_module.TENSOR_METRICS if item.key == "coherence"
    )
    assert coherence_spec.supported is True
    assert window._tensor_bands_configure_button is not None
    assert "Bands Configure" in window._tensor_bands_configure_button.text()
    assert window._tensor_low_freq_edit is not None
    assert window._tensor_step_edit is not None
    assert window._tensor_channels_button is not None
    assert window._tensor_channels_button.text() == "Select Channels (0/0)"
    assert window._tensor_pairs_button is not None
    assert window._tensor_pairs_button.text() == "Select Pairs (0/0)"
    assert window._alignment_paradigm_list is not None
    assert window._alignment_trials_block is not None
    assert window._alignment_trials_action_row is not None
    assert window._alignment_method_combo is not None
    assert window._alignment_method_params_button is not None
    assert window._alignment_method_params_button.text() == "Params"
    assert window._alignment_import_button is not None
    assert window._alignment_import_button.text() == "Import Configs..."
    assert window._alignment_export_button is not None
    assert window._alignment_export_button.text() == "Export Configs..."
    assert window._alignment_method_indicator is not None
    method_panel = window._alignment_method_indicator.parentWidget()
    assert isinstance(method_panel, QGroupBox)
    assert method_panel.title() == "Method + Params"
    assert window._alignment_run_button is not None
    assert window._alignment_run_button.text() == "Align Epochs"
    assert window._alignment_epoch_table is not None
    assert window._alignment_epoch_table.columnCount() == 5
    assert (
        window._alignment_epoch_table.horizontalHeaderItem(2).text() == "Duration (s)"
    )
    assert window._alignment_epoch_inspector_indicator is not None
    epoch_panel = window._alignment_epoch_inspector_indicator.parentWidget()
    assert isinstance(epoch_panel, QGroupBox)
    assert epoch_panel.title() == "Epoch Inspector"
    assert window._alignment_finish_button is not None
    assert window._alignment_finish_button.text() == "Finish"
    assert window._features_paradigm_list is not None
    assert window._features_trials_block is not None
    assert window._features_trials_action_row is not None
    assert window._features_paradigm_add_button is not None
    assert window._features_paradigm_delete_button is not None
    assert not window._features_paradigm_add_button.isEnabled()
    assert not window._features_paradigm_delete_button.isEnabled()
    assert window._features_run_extract_button is not None
    assert window._features_run_extract_button.text() == "Extract Features"
    assert window._features_import_button is not None
    assert window._features_import_button.text() == "Import Configs..."
    assert window._features_export_button is not None
    assert window._features_export_button.text() == "Export Configs..."
    assert window._features_refresh_button is not None
    assert window._features_refresh_button.text() == "Refresh Features"
    assert window._features_axis_metric_combo is not None
    assert window._features_axis_bands_button is not None
    assert window._features_axis_times_button is not None
    assert window._features_axis_apply_all_button is not None
    assert window._features_axis_apply_all_button.text() == "Apply to All Metrics"
    assert window._features_extract_indicator is not None
    features_panel = window._features_extract_indicator.parentWidget()
    assert isinstance(features_panel, QGroupBox)
    assert features_panel.title() == "Features"
    assert window._features_subset_band_combo is not None
    assert window._features_subset_channel_combo is not None
    assert window._features_subset_region_combo is not None
    assert window._features_available_table is not None
    assert window._features_available_table.columnCount() == 3
    assert window._features_plot_advance_button is not None
    assert window._features_plot_advance_button.text() == "Advance"
    assert window._features_plot_button is not None
    assert window._features_plot_button.text() == "Plot"
    assert window._features_plot_export_button is not None
    assert window._features_plot_export_button.text() == "Export"
    assert window._localize_matlab_status_label is not None
    assert window._localize_matlab_status_label.text().startswith("MATLAB: ")
    assert window._localize_match_button is not None
    assert window._localize_match_button.text() == "Configure..."
    matlab_top = window._localize_matlab_status_label.mapToGlobal(
        window._localize_matlab_status_label.rect().topLeft()
    ).y()
    match_top = window._localize_match_button.mapToGlobal(
        window._localize_match_button.rect().topLeft()
    ).y()
    assert matlab_top < match_top
    contact_row = window._localize_apply_button.parentWidget()
    assert contact_row is not None
    apply_right = (
        window._localize_apply_button.geometry().x()
        + window._localize_apply_button.width()
    )
    contact_right = (
        window._contact_viewer_button.geometry().x()
        + window._contact_viewer_button.width()
    )
    assert apply_right <= contact_row.width()
    assert contact_right <= contact_row.width()
    config_row = window._localize_import_button.parentWidget()
    assert config_row is not None
    import_right = (
        window._localize_import_button.geometry().x()
        + window._localize_import_button.width()
    )
    export_right = (
        window._localize_export_button.geometry().x()
        + window._localize_export_button.width()
    )
    assert import_right <= config_row.width()
    assert export_right <= config_row.width()
    localize_panel = window._localize_indicator.parentWidget()
    assert isinstance(localize_panel, QGroupBox)
    assert localize_panel.title() == "Localize"
    assert not any(
        isinstance(label, QLabel) and label.text() == "Status"
        for label in localize_panel.findChildren(QLabel)
    )
    expected_preproc_titles = {
        "raw": "0. Raw",
        "filter": "1. Filter",
        "annotations": "2. Annotations",
        "bad_segment_removal": "3. Bad Segment Removal",
        "ecg_artifact_removal": "4. ECG Artifact Removal",
        "finish": "5. Finish",
    }
    for step, title in expected_preproc_titles.items():
        indicator = window._preproc_step_indicators.get(step)
        assert indicator is not None
        panel = indicator.parentWidget()
        assert isinstance(panel, QGroupBox)
        assert panel.title() == title
        assert indicator.isVisible()
        assert not any(
            isinstance(label, QLabel) and label.text() == "Status"
            for label in panel.findChildren(QLabel)
        )
    style_sheet = window.styleSheet()
    assert "QGroupBox::title" in style_sheet
    assert "font-weight: 700" in style_sheet
    match = re.search(r"font-size:\s*([0-9.]+)pt;", style_sheet)
    assert match is not None
    title_size = float(match.group(1))
    button_size = window._preproc_filter_apply_button.font().pointSizeF()
    assert button_size > 0
    assert abs(title_size - button_size) < 0.2

    window._start_busy("Smoke")
    busy_message = window.statusBar().currentMessage()
    assert "Smoke" in busy_message
    assert any(frame.strip() in busy_message for frame in BUSY_FRAMES)
    window._stop_busy()

    window.close()


def test_stage_refresh_keeps_active_blocked_page_without_auto_fallback() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    window._stage_states.update(
        {
            "preproc": "green",
            "tensor": "green",
            "alignment": "green",
            "features": "gray",
        }
    )
    window._refresh_stage_controls()
    window.route_to_stage("alignment")
    app.processEvents()

    assert window._active_stage_key == "alignment"

    window._stage_states.update(
        {
            "preproc": "yellow",
            "tensor": "yellow",
            "alignment": "yellow",
        }
    )
    window._refresh_stage_controls()
    app.processEvents()

    assert window._active_stage_key == "alignment"
    assert window._stage_stack.currentIndex() == window._stage_page_index["alignment"]
    assert not window._stage_buttons["alignment"].isEnabled()

    window.close()


def test_mainwindow_close_also_closes_auxiliary_windows() -> None:
    app = QApplication.instance() or QApplication([])

    class _BrowserStub:
        def __init__(self) -> None:
            self.close_called = False

        def close(self) -> None:
            self.close_called = True

    window = MainWindow(auto_load_dataset=False)
    window.show()
    auxiliary = QDialog()
    auxiliary.setWindowTitle("Auxiliary Plot")
    auxiliary.show()
    browser = _BrowserStub()
    window._plot_close_hooks.append((browser, "close"))
    app.processEvents()

    assert auxiliary.isVisible()
    assert browser.close_called is False

    window.close()
    app.processEvents()

    assert browser.close_called is True
    assert not auxiliary.isVisible()


def test_features_plot_advance_transform_options_use_ln_display_and_canonical_modes() -> (
    None
):
    app = QApplication.instance() or QApplication([])

    params = {
        "transform_mode": "log",
        "normalize_mode": "none",
        "baseline_mode": "mean",
        "baseline_percent_ranges": [],
        "colormap": "viridis",
        "x_log": False,
        "y_log": False,
    }
    dialog = FeaturesPlotAdvanceDialog(
        session_params=dict(params),
        default_params=dict(params),
        allow_x_log=True,
        allow_y_log=True,
        allow_normalize=True,
    )
    dialog.show()
    app.processEvents()

    labels = [
        dialog._transform_combo.itemText(idx)
        for idx in range(dialog._transform_combo.count())
    ]
    assert "ln" in labels
    assert "log10" in labels

    log_idx = dialog._transform_combo.findData("log")
    log10_idx = dialog._transform_combo.findData("log10")
    assert log_idx >= 0
    assert log10_idx >= 0
    assert dialog._transform_combo.itemText(log_idx) == "ln"

    dialog._transform_combo.setCurrentIndex(log10_idx)
    assert dialog._collect()["transform_mode"] == "log10"

    dialog._transform_combo.setCurrentIndex(log_idx)
    assert dialog._collect()["transform_mode"] == "log"

    dialog.close()


def test_normalize_feature_plot_transform_mode_accepts_ln_alias() -> None:
    assert normalize_feature_plot_transform_mode("ln") == "log"
    assert normalize_feature_plot_transform_mode("log") == "log"
    assert normalize_feature_plot_transform_mode("log10") == "log10"


def test_features_plot_runtime_normalizes_ln_alias_before_transform() -> None:
    import numpy as np
    import pandas as pd

    _ = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window._features_plot_advance_params = {
        "transform_mode": "ln",
        "normalize_mode": "none",
        "baseline_mode": "mean",
        "baseline_percent_ranges": [],
        "colormap": "viridis",
        "x_log": False,
        "y_log": False,
    }
    payload = pd.DataFrame(
        {"Value": [pd.Series([1.0, float(np.e)], index=["f1", "f2"], name="Value")]}
    )

    out = window._apply_features_plot_advance(payload, derived_type="spectral")
    value = out.loc[0, "Value"]

    assert isinstance(value, pd.Series)
    np.testing.assert_allclose(
        value.to_numpy(dtype=float),
        np.array([0.0, 1.0], dtype=float),
        equal_nan=True,
    )

    window.close()


def test_tensor_bottom_actions_use_two_rows() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    window._stage_stack.setCurrentIndex(window._stage_page_index["tensor"])
    app.processEvents()

    import_row = window._tensor_import_button.parentWidget()
    export_row = window._tensor_export_button.parentWidget()
    mask_row = window._tensor_mask_edge_checkbox.parentWidget()
    run_row = window._tensor_run_button.parentWidget()

    assert import_row is not None
    assert mask_row is not None
    assert import_row is export_row
    assert mask_row is run_row
    assert import_row is not mask_row
    assert import_row.parentWidget() is not None
    assert import_row.parentWidget() is mask_row.parentWidget()
    import_top = window._tensor_import_button.mapToGlobal(
        window._tensor_import_button.rect().topLeft()
    ).y()
    mask_top = window._tensor_mask_edge_checkbox.mapToGlobal(
        window._tensor_mask_edge_checkbox.rect().topLeft()
    ).y()
    assert import_top < mask_top

    window.close()


def test_alignment_method_actions_place_import_export_on_bottom_row() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    window._stage_stack.setCurrentIndex(window._stage_page_index["alignment"])
    app.processEvents()

    assert window._alignment_method_params_button is not None
    assert window._alignment_run_button is not None
    assert window._alignment_import_button is not None
    assert window._alignment_export_button is not None

    run_row = window._alignment_method_params_button.parentWidget()
    import_row = window._alignment_import_button.parentWidget()
    export_row = window._alignment_export_button.parentWidget()

    assert run_row is not None
    assert import_row is not None
    assert import_row is export_row
    assert import_row is not run_row
    assert import_row.parentWidget() is not None
    assert import_row.parentWidget() is run_row.parentWidget()

    run_top = window._alignment_method_params_button.mapToGlobal(
        window._alignment_method_params_button.rect().topLeft()
    ).y()
    import_top = window._alignment_import_button.mapToGlobal(
        window._alignment_import_button.rect().topLeft()
    ).y()
    export_top = window._alignment_export_button.mapToGlobal(
        window._alignment_export_button.rect().topLeft()
    ).y()
    assert run_top < import_top
    assert abs(import_top - export_top) <= 2

    assert window._alignment_method_indicator is not None
    assert window._alignment_epoch_inspector_indicator is not None
    assert window._alignment_method_indicator.isVisible()
    assert window._alignment_epoch_inspector_indicator.isVisible()

    window.close()


def test_busy_ui_lock_disables_buttons_and_actions_then_restores() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    preproc_button = window._stage_buttons["preproc"]
    settings_action = window._settings_configs_action
    assert settings_action is not None
    assert preproc_button.isEnabled()
    assert settings_action.isEnabled()

    window._start_busy("Busy Lock")
    app.processEvents()

    assert not preproc_button.isEnabled()
    assert not settings_action.isEnabled()

    window._stop_busy()
    app.processEvents()

    assert preproc_button.isEnabled()
    assert settings_action.isEnabled()
    window.close()


def test_run_with_busy_rejects_reentry_when_already_busy() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    window._busy_label = "Existing Busy Task"

    invoked = {"value": False}

    def _work() -> tuple[bool, str]:
        invoked["value"] = True
        return (True, "ok")

    with pytest.raises(RuntimeError, match="Busy lock is active."):
        window._run_with_busy("Another Task", _work)
    assert invoked["value"] is False
    assert "duplicate action ignored" in window.statusBar().currentMessage()

    window._busy_label = None
    window.close()


def test_localize_match_auto_match_uses_side_constraint_and_casefold() -> None:
    app = QApplication.instance() or QApplication([])

    dialog = LocalizeMatchDialog(
        channel_names=("0a_1b_L",),
        lead_specs=[
            {
                "display_name": "Lead R",
                "contacts": [
                    {"token": "R_K0A (R)", "contact_name": "K0A (R)"},
                    {"token": "R_K1B (R)", "contact_name": "K1B (R)"},
                ],
            },
            {
                "display_name": "Lead L",
                "contacts": [
                    {"token": "L_k0a (L)", "contact_name": "k0a (L)"},
                    {"token": "L_k1b (L)", "contact_name": "k1b (L)"},
                ],
            },
        ],
    )
    dialog.show()
    app.processEvents()

    dialog._on_auto_match()

    assert "0a_1b_L" in dialog._mapping
    assert dialog._mapping["0a_1b_L"]["anode"] == "L_k0a (L)"
    assert dialog._mapping["0a_1b_L"]["cathode"] == "L_k1b (L)"
    assert dialog._mapping["0a_1b_L"]["rep_coord"] == "Mid"


def test_localize_match_auto_match_skips_ambiguous_multi_lead_candidates() -> None:
    app = QApplication.instance() or QApplication([])

    dialog = LocalizeMatchDialog(
        channel_names=("0A_1B",),
        lead_specs=[
            {
                "display_name": "Lead R",
                "contacts": [
                    {"token": "R_K0A (R)", "contact_name": "K0A (R)"},
                    {"token": "R_K1B (R)", "contact_name": "K1B (R)"},
                ],
            },
            {
                "display_name": "Lead L",
                "contacts": [
                    {"token": "L_K0A (L)", "contact_name": "K0A (L)"},
                    {"token": "L_K1B (L)", "contact_name": "K1B (L)"},
                ],
            },
        ],
    )
    dialog.show()
    app.processEvents()

    dialog._on_auto_match()

    assert "0A_1B" not in dialog._mapping
    assert dialog._mapping == {}


def test_localize_match_uses_text_del_action_cells() -> None:
    app = QApplication.instance() or QApplication([])

    dialog = LocalizeMatchDialog(
        channel_names=("0_1",),
        lead_specs=[],
        current_payload={
            "mappings": [
                {
                    "channel": "0_1",
                    "anode": "K0A (L)",
                    "cathode": "K1A (L)",
                    "rep_coord": "Mid",
                }
            ]
        },
    )
    dialog.show()
    app.processEvents()

    assert dialog._mapping_table.cellWidget(0, 4) is None
    action_item = dialog._mapping_table.item(0, 4)
    assert action_item is not None
    assert action_item.text() == "Del"

    dialog._on_mapping_row_clicked(0, 4)
    assert dialog._mapping == {}


def test_localize_match_defaults_save_and_restore(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    lead_specs = [
        {
            "display_name": "Lead L",
            "contacts": [
                {"token": "L_K0 (L)", "contact_name": "K0 (L)"},
                {"token": "L_K1 (L)", "contact_name": "K1 (L)"},
            ],
        }
    ]

    dialog = LocalizeMatchDialog(
        channel_names=("0_1", "1_2"),
        lead_specs=lead_specs,
        config_store=store,
    )
    dialog.show()
    app.processEvents()
    dialog._mapping = {
        "0_1": {
            "anode": "L_K0 (L)",
            "cathode": "L_K1 (L)",
            "rep_coord": "Mid",
        }
    }
    dialog._on_set_default()

    saved = store.read_yaml("localization.yml", default={})
    assert saved["match_defaults"]["channels"] == ["0_1", "1_2"]
    assert saved["match_defaults"]["mappings"] == [
        {
            "channel": "0_1",
            "anode": "L_K0 (L)",
            "cathode": "L_K1 (L)",
            "rep_coord": "Mid",
        }
    ]

    restore_dialog = LocalizeMatchDialog(
        channel_names=("0_1", "1_2"),
        lead_specs=lead_specs,
        config_store=store,
    )
    restore_dialog.show()
    app.processEvents()
    restore_dialog._mapping = {
        "1_2": {
            "anode": "L_K1 (L)",
            "cathode": "L_K0 (L)",
            "rep_coord": "Anode",
        }
    }
    restore_dialog._on_restore_default()

    assert restore_dialog._mapping == {
        "0_1": {
            "anode": "L_K0 (L)",
            "cathode": "L_K1 (L)",
            "rep_coord": "Mid",
        }
    }
    assert restore_dialog._mapping_table.rowCount() == 1


def test_localize_match_restore_default_is_noop_when_incompatible(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    base_dialog = LocalizeMatchDialog(
        channel_names=("0_1",),
        lead_specs=[
            {
                "display_name": "Lead L",
                "contacts": [
                    {"token": "L_K0 (L)", "contact_name": "K0 (L)"},
                    {"token": "L_K1 (L)", "contact_name": "K1 (L)"},
                ],
            }
        ],
        config_store=store,
    )
    base_dialog._mapping = {
        "0_1": {
            "anode": "L_K0 (L)",
            "cathode": "L_K1 (L)",
            "rep_coord": "Mid",
        }
    }
    base_dialog._on_set_default()

    incompatible_dialog = LocalizeMatchDialog(
        channel_names=("0_1",),
        lead_specs=[
            {
                "display_name": "Lead L",
                "contacts": [
                    {"token": "L_K0X (L)", "contact_name": "K0X (L)"},
                    {"token": "L_K1X (L)", "contact_name": "K1X (L)"},
                ],
            }
        ],
        config_store=store,
    )
    incompatible_dialog.show()
    app.processEvents()
    incompatible_dialog._mapping = {
        "0_1": {
            "anode": "L_K0X (L)",
            "cathode": "L_K1X (L)",
            "rep_coord": "Cathode",
        }
    }

    incompatible_dialog._on_restore_default()

    assert incompatible_dialog._mapping == {
        "0_1": {
            "anode": "L_K0X (L)",
            "cathode": "L_K1X (L)",
            "rep_coord": "Cathode",
        }
    }


def test_preproc_viz_param_normalizers_cover_remaining_branches() -> None:
    ok, defaults, message = normalize_preproc_viz_psd_params(None)
    assert ok and message == ""
    assert defaults["average"] is True

    ok, _, message = normalize_preproc_viz_psd_params("bad")  # type: ignore[arg-type]
    assert not ok and "dictionary" in message
    ok, _, message = normalize_preproc_viz_psd_params({"fmin": "bad"})
    assert not ok and message
    ok, _, message = normalize_preproc_viz_psd_params({"fmin": -1.0})
    assert not ok and "fmin" in message
    ok, _, message = normalize_preproc_viz_psd_params({"fmin": 1.0, "fmax": 1.0})
    assert not ok and "fmax" in message
    ok, _, message = normalize_preproc_viz_psd_params({"n_fft": 8})
    assert not ok and "n_fft" in message
    ok, normalized, message = normalize_preproc_viz_psd_params(
        {"fmin": 2, "fmax": 40, "n_fft": 256, "average": "yes"}
    )
    assert ok and message == ""
    assert normalized["average"] is True

    ok, defaults, message = normalize_preproc_viz_tfr_params(None)
    assert ok and message == ""
    assert defaults["n_freqs"] == 40

    ok, _, message = normalize_preproc_viz_tfr_params("bad")  # type: ignore[arg-type]
    assert not ok and "dictionary" in message
    ok, _, message = normalize_preproc_viz_tfr_params({"fmin": "bad"})
    assert not ok and message
    ok, _, message = normalize_preproc_viz_tfr_params({"fmin": 0.0})
    assert not ok and "fmin" in message
    ok, _, message = normalize_preproc_viz_tfr_params({"fmin": 1.0, "fmax": 1.0})
    assert not ok and "fmax" in message
    ok, _, message = normalize_preproc_viz_tfr_params({"n_freqs": 3})
    assert not ok and "n_freqs" in message
    ok, _, message = normalize_preproc_viz_tfr_params({"decim": 0})
    assert not ok and "decim" in message
    ok, normalized, message = normalize_preproc_viz_tfr_params(
        {"fmin": 2, "fmax": 80, "n_freqs": 20, "decim": 2}
    )
    assert ok and message == ""
    assert normalized["decim"] == 2


def test_paths_config_dialog_validates_paths_and_saves(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    warnings: list[str] = []

    leaddbs_dir = tmp_path / "leaddbs"
    matlab_root = tmp_path / "matlab"
    leaddbs_dir.mkdir(parents=True)
    matlab_root.mkdir(parents=True)
    not_dir = tmp_path / "not_dir.txt"
    not_dir.write_text("x", encoding="utf-8")

    dialog = PathsConfigDialog(
        current_paths={
            "leaddbs_dir": "",
            "matlab_root": "",
        }
    )
    _attach_warning_sink(dialog, warnings)
    dialog.show()
    app.processEvents()

    dialog._on_save()
    assert any("is required" in item for item in warnings)

    dialog._path_edits["leaddbs_dir"].setText(str(not_dir))
    dialog._path_edits["matlab_root"].setText(str(matlab_root))
    dialog._on_save()
    assert any("must be an existing directory" in item for item in warnings)

    dialog._path_edits["leaddbs_dir"].setText(str(leaddbs_dir))
    dialog._path_edits["matlab_root"].setText(str(tmp_path / "missing"))
    dialog._on_save()
    assert any("does not exist" in item for item in warnings)

    dialog._path_edits["matlab_root"].setText(str(matlab_root))
    dialog._on_save()
    assert dialog.selected_paths == {
        "leaddbs_dir": str(leaddbs_dir.resolve()),
        "matlab_root": str(matlab_root.resolve()),
    }
    assert dialog.result() == QDialog.Accepted


def test_paths_config_dialog_browse_branches(
    tmp_path: Path,
) -> None:
    QApplication.instance() or QApplication([])
    target_dir = tmp_path / "selected_dir"
    target_dir.mkdir(parents=True)

    dialog = PathsConfigDialog(current_paths={})
    dialog._use_matlab_bundle_browser = (  # type: ignore[method-assign]
        lambda _field_key: False
    )

    dialog._select_existing_directory = (  # type: ignore[method-assign]
        lambda _title, _start_dir: str(target_dir)
    )
    dialog._on_browse_directory("leaddbs_dir")
    assert dialog._path_edits["leaddbs_dir"].text() == str(target_dir)

    before = dialog._path_edits["matlab_root"].text()
    dialog._select_existing_directory = (  # type: ignore[method-assign]
        lambda _title, _start_dir: ""
    )
    dialog._on_browse_directory("matlab_root")
    assert dialog._path_edits["matlab_root"].text() == before
    dialog.reject()


def test_paths_config_dialog_browse_uses_macos_matlab_selector(
    tmp_path: Path,
) -> None:
    QApplication.instance() or QApplication([])
    matlab_app = tmp_path / "MATLAB_R2024b.app"
    matlab_app.mkdir(parents=True)

    dialog = PathsConfigDialog(current_paths={"matlab_root": str(matlab_app)})

    captured: dict[str, str] = {}

    def _select_matlab_installation_path(_title: str, start_dir: str) -> str:
        captured["start_dir"] = start_dir
        return str(matlab_app)

    dialog._use_matlab_bundle_browser = (  # type: ignore[method-assign]
        lambda _field_key: True
    )
    dialog._select_matlab_installation_path = (  # type: ignore[method-assign]
        _select_matlab_installation_path
    )
    dialog._select_existing_directory = (  # type: ignore[method-assign]
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected selector"))
    )

    dialog._on_browse_directory("matlab_root")

    assert captured["start_dir"] == str(tmp_path)
    assert dialog._path_edits["matlab_root"].text() == str(matlab_app)
    dialog.reject()


def test_channel_pair_dialog_interactions_and_validation() -> None:
    app = QApplication.instance() or QApplication([])

    warnings: list[str] = []

    dialog = ChannelPairDialog(
        channel_names=["A", "B", "C"],
        current_pairs=(),
        current_names=(),
    )
    _attach_warning_sink(dialog, warnings)
    dialog.show()
    app.processEvents()

    dialog._on_remove_pair()

    def _item(token: str):
        for idx in range(dialog._channel_list.count()):
            current = dialog._channel_list.item(idx)
            if current is not None and current.text() == token:
                return current
        raise AssertionError(f"missing channel item: {token}")

    dialog._on_channel_clicked(_item("A"))
    assert "Anode selected" in dialog._status_label.text()
    dialog._on_channel_clicked(_item("A"))
    assert "cancelled" in dialog._status_label.text()

    dialog._on_channel_clicked(_item("A"))
    dialog._on_channel_clicked(_item("B"))
    assert dialog.selected_pairs == (("A", "B"),)
    dialog._on_pair_name_changed(dialog._pair_table.item(0, 0))
    name_item = dialog._pair_table.item(0, 1)
    assert name_item is not None
    name_item.setText(" ")
    dialog._on_pair_name_changed(name_item)
    assert dialog.selected_names == ("A-B",)

    dialog._pair_table.setCurrentCell(0, 0)
    dialog._on_remove_pair()
    assert dialog.selected_pairs == ()

    dialog._on_accept()
    assert any("At least one pair" in item for item in warnings)

    dialog._on_channel_clicked(_item("A"))
    dialog._on_channel_clicked(_item("B"))
    dialog._on_channel_clicked(_item("A"))
    dialog._on_channel_clicked(_item("C"))
    assert dialog.selected_pairs == (("A", "B"), ("A", "C"))
    first_name_item = dialog._pair_table.item(0, 1)
    second_name_item = dialog._pair_table.item(1, 1)
    assert first_name_item is not None and second_name_item is not None
    first_name_item.setText("dup")
    second_name_item.setText("dup")
    dialog._on_accept()
    assert any("must be unique" in item for item in warnings)

    first_name_item.setText("ab")
    second_name_item.setText("ac")
    dialog._on_accept()
    assert dialog.result() == QDialog.Accepted


def test_channel_pair_dialog_edge_branches() -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []

    dialog = ChannelPairDialog(
        channel_names=["A", "B", "C"],
        current_pairs=(("A", "B"), ("B", "C")),
        current_names=(" ",),
    )
    _attach_warning_sink(dialog, warnings)
    assert dialog.selected_names == ("A-B", "B-C")

    first_name_item = dialog._pair_table.item(0, 1)
    assert first_name_item is not None
    dialog._pair_table.takeItem(0, 0)
    first_name_item.setText("edge-name")

    second_pair_item = dialog._pair_table.item(1, 0)
    second_name_item = dialog._pair_table.item(1, 1)
    assert second_pair_item is not None and second_name_item is not None
    second_pair_item.setData(Qt.UserRole, "bad-token")
    second_name_item.setText("edge-name-2")

    dialog._sync_pair_names_from_table()
    dialog._on_pair_name_changed(first_name_item)
    dialog._on_pair_name_changed(second_name_item)

    dialog._pair_names[dialog.selected_pairs[0]] = ""
    dialog._on_accept()
    assert any("requires a non-empty name" in item for item in warnings)


def test_channel_select_and_tensor_channel_dialog_branches() -> None:
    app = QApplication.instance() or QApplication([])

    dialog = ChannelSelectDialog(
        title="Select Channels",
        channels=["A", "B", "C"],
        selected_channels=("B",),
    )
    dialog.show()
    app.processEvents()
    assert dialog.selected_channels == ("B",)
    dialog._on_select_all()
    assert set(dialog.selected_channels) == {"A", "B", "C"}
    dialog._on_clear()
    assert dialog.selected_channels == ()
    dialog.accept()
    assert dialog.result() == QDialog.Accepted

    tensor_dialog = TensorChannelSelectDialog(
        title="Tensor Channels",
        channels=("A", "B", "C"),
        session_selected=("A",),
        default_selected=("B", "C"),
    )
    tensor_dialog.show()
    app.processEvents()
    assert tensor_dialog.selected_action is None
    assert tensor_dialog.selected_channels == ("A",)
    tensor_dialog._on_select_all()
    assert set(tensor_dialog.selected_channels) == {"A", "B", "C"}
    tensor_dialog._on_clear()
    assert tensor_dialog.selected_channels == ()
    tensor_dialog._on_restore_defaults()
    assert tensor_dialog.selected_channels == ("B", "C")
    footer = tensor_dialog.layout().itemAt(tensor_dialog.layout().count() - 1).widget()
    assert footer is not None
    footer_layout = footer.layout()
    assert footer_layout is not None
    indices = {
        "set_default": -1,
        "restore_defaults": -1,
    }
    for idx in range(footer_layout.count()):
        item = footer_layout.itemAt(idx)
        widget = item.widget()
        if widget is None or not isinstance(widget, main_window_module.QPushButton):
            continue
        text = widget.text()
        if text == "Set as Default":
            indices["set_default"] = idx
        elif text == "Restore Defaults":
            indices["restore_defaults"] = idx
    assert indices["set_default"] >= 0
    assert indices["restore_defaults"] >= 0
    assert indices["set_default"] < indices["restore_defaults"]
    tensor_dialog._accept("set_default")
    assert tensor_dialog.selected_action == "set_default"
    assert tensor_dialog.result() != QDialog.Accepted


def test_tensor_pair_dialog_branches() -> None:
    app = QApplication.instance() or QApplication([])
    warnings: list[str] = []
    dialog = TensorPairSelectDialog(
        title="Pairs",
        channel_names=("A", "B", "C"),
        session_pairs=(("A", "B"), ("B", "A"), ("A", "A"), ("A", "D")),
        default_pairs=(("B", "C"),),
        directed=False,
    )
    dialog._show_warning = (  # type: ignore[method-assign]
        lambda title, message: warnings.append(str(message)) or 0
    )
    dialog.show()
    app.processEvents()
    assert dialog.selected_pairs == (("A", "B"),)
    assert dialog._display_pair(("A", "B")) == "A-B"

    with pytest.raises(ValueError, match="cannot be empty"):
        dialog._normalize_pair(" ", "A")
    with pytest.raises(ValueError, match="Self-pairs"):
        dialog._normalize_pair("A", "A")

    dialog._on_remove_pair(("A", "B"))
    assert dialog.selected_pairs == ()

    def _item(token: str):
        for idx in range(dialog._channel_list.count()):
            current = dialog._channel_list.item(idx)
            if current is not None and current.text() == token:
                return current
        raise AssertionError(f"missing channel item: {token}")

    dialog._on_channel_clicked(_item("A"))
    dialog._on_channel_clicked(_item("A"))
    assert dialog._draft_source_edit.text() == ""
    dialog._on_channel_clicked(_item("A"))
    dialog._on_channel_clicked(_item("B"))
    dialog._on_apply_draft()
    assert dialog.selected_pairs == (("A", "B"),)
    dialog._on_channel_clicked(_item("A"))
    dialog._on_channel_clicked(_item("B"))
    dialog._on_apply_draft()
    assert any("already exists" in item for item in warnings)
    dialog._on_clear_draft()
    dialog._on_channel_clicked(_item("B"))
    dialog._on_channel_clicked(_item("C"))
    dialog._on_apply_draft()
    dialog._on_channel_clicked(_item("A"))
    dialog._on_channel_clicked(_item("C"))
    dialog._on_apply_draft()
    assert dialog.selected_pairs == (("A", "B"), ("B", "C"), ("A", "C"))
    assert dialog._pair_table.item(1, 3).text() == "B-C"
    assert dialog._pair_table.item(2, 3).text() == "A-C"
    dialog._search_edit.setText("b-c")
    assert dialog._pair_table.rowCount() == 1
    assert dialog._pair_table.cellWidget(0, 4) is None
    action_item = dialog._pair_table.item(0, 4)
    assert action_item is not None
    assert action_item.text() == "Del"
    dialog._on_pair_table_clicked(0, 4)
    assert dialog.selected_pairs == (("A", "B"), ("A", "C"))
    dialog._search_edit.setText("")

    dialog._on_restore_defaults()
    assert dialog.selected_pairs == (("B", "C"),)
    footer = dialog.layout().itemAt(dialog.layout().count() - 1).widget()
    assert footer is not None
    footer_layout = footer.layout()
    assert footer_layout is not None
    indices = {
        "set_default": -1,
        "restore_defaults": -1,
    }
    for idx in range(footer_layout.count()):
        item = footer_layout.itemAt(idx)
        widget = item.widget()
        if widget is None or not isinstance(widget, main_window_module.QPushButton):
            continue
        text = widget.text()
        if text == "Set as Default":
            indices["set_default"] = idx
        elif text == "Restore Defaults":
            indices["restore_defaults"] = idx
    assert indices["set_default"] >= 0
    assert indices["restore_defaults"] >= 0
    assert indices["set_default"] < indices["restore_defaults"]
    dialog._on_add_all_pairs()
    assert dialog.selected_pairs == (("B", "C"), ("A", "B"), ("A", "C"))
    dialog._accept("save")
    assert dialog.selected_action == "save"
    assert dialog.result() == QDialog.Accepted

    directed = TensorPairSelectDialog(
        title="Directed",
        channel_names=("A", "B"),
        session_pairs=(("B", "A"),),
        default_pairs=(),
        directed=True,
    )
    directed._show_warning = (  # type: ignore[method-assign]
        lambda title, message: warnings.append(str(message)) or 0
    )
    directed.show()
    app.processEvents()
    assert directed._display_pair(("B", "A")) == "B -> A"
    assert directed.selected_pairs == (("B", "A"),)
    directed._on_add_all_pairs()
    assert directed.selected_pairs == (("B", "A"), ("A", "B"))


def test_tensor_metric_advance_restore_defaults_returns_full_payload() -> None:
    _ = QApplication.instance() or QApplication([])
    dialog = TensorMetricAdvanceDialog(
        metric_key="psi",
        metric_label="PSI",
        session_params={
            "time_resolution_s": 0.2,
            "hop_s": 0.01,
            "method": "morlet",
            "notches": [60.0],
            "notch_widths": 1.5,
            "selected_pairs": [["CH1", "CH2"]],
            "bands": [{"name": "alpha", "start": 8.0, "end": 12.0}],
        },
        default_params={
            "time_resolution_s": 0.5,
            "hop_s": 0.025,
            "method": "multitaper",
            "mt_bandwidth": 4.0,
            "notches": [50.0, 100.0],
            "notch_widths": 2.0,
            "selected_pairs": [["CH2", "CH1"]],
            "bands": [{"name": "theta", "start": 4.0, "end": 7.0}],
        },
    )

    dialog._on_restore_defaults()
    assert dialog._fields["notches"].text() == "50, 100"
    assert dialog._fields["notch_widths"].text() == "2"
    dialog._on_submit("save")
    assert dialog.selected_params is not None
    assert dialog.selected_params["time_resolution_s"] == 0.5
    assert dialog.selected_params["hop_s"] == 0.025
    assert dialog.selected_params["method"] == "multitaper"
    assert dialog.selected_params["notches"] == [50.0, 100.0]
    assert dialog.selected_params["notch_widths"] == 2.0
    assert dialog.selected_params["selected_pairs"] == [["CH2", "CH1"]]
    assert dialog.selected_params["bands"] == [
        {"name": "theta", "start": 4.0, "end": 7.0}
    ]


def test_burst_tensor_metric_advance_supports_baseline_annotation_dropdown() -> None:
    _ = QApplication.instance() or QApplication([])
    dialog = TensorMetricAdvanceDialog(
        metric_key="burst",
        metric_label="Burst",
        session_params={
            "baseline_keep": ["Rest"],
            "min_cycles": 2.0,
            "max_cycles": None,
        },
        default_params={
            "baseline_keep": ["Task"],
            "min_cycles": 3.0,
            "max_cycles": 5.0,
        },
        burst_baseline_annotations=("Rest", "Task"),
    )

    assert dialog._baseline_annotations_combo is not None
    assert dialog._baseline_annotations_combo.currentData() == "Rest"

    dialog._on_restore_defaults()
    assert dialog._baseline_annotations_combo.currentData() == "Task"

    dialog._baseline_annotations_combo.setCurrentIndex(1)
    dialog._on_submit("save")
    assert dialog.selected_params is not None
    assert dialog.selected_params["baseline_keep"] == ["Rest"]
    assert "baseline_match" not in dialog.selected_params


def test_trgc_tensor_metric_advance_supports_window_grouping_controls() -> None:
    _ = QApplication.instance() or QApplication([])
    dialog = TensorMetricAdvanceDialog(
        metric_key="trgc",
        metric_label="TRGC",
        session_params={
            "method": "morlet",
            "gc_n_lags": 20,
            "group_by_samples": False,
            "round_ms": 50.0,
        },
        default_params={
            "method": "multitaper",
            "mt_bandwidth": 4.0,
            "gc_n_lags": 24,
            "group_by_samples": True,
            "round_ms": 25.0,
        },
    )

    group_widget = dialog._fields["group_by_samples"]
    round_widget = dialog._fields["round_ms"]
    assert isinstance(group_widget, QCheckBox)
    assert group_widget.text() == ""
    assert (
        group_widget.toolTip()
        == "Group TRGC frequencies by exact window length in samples. Recommended only when you want grouping tied to the recording sample rate; for most runs leave this off and use Round ms."
    )
    assert group_widget.isChecked() is False
    assert round_widget.isEnabled() is True
    assert (
        round_widget.toolTip()
        == "Millisecond grid used to group TRGC window lengths when Group by samples is off. Recommended: keep 50 ms for most runs; smaller values preserve finer timing differences but can create more groups."
    )

    group_widget.setChecked(True)
    assert round_widget.isEnabled() is False

    dialog._on_restore_defaults()
    assert group_widget.isChecked() is True
    assert round_widget.isEnabled() is False
    assert float(round_widget.text()) == 25.0

    dialog._on_submit("save")
    assert dialog.selected_params is not None
    assert dialog.selected_params["group_by_samples"] is True
    assert dialog.selected_params["round_ms"] == 25.0
    assert dialog.selected_params["gc_n_lags"] == 24


def test_periodic_tensor_metric_advance_supports_split_smoothing_controls() -> None:
    _ = QApplication.instance() or QApplication([])
    dialog = TensorMetricAdvanceDialog(
        metric_key="periodic_aperiodic",
        metric_label="Periodic/APeriodic",
        session_params={
            "freq_smooth_enabled": False,
            "freq_smooth_sigma": 1.25,
            "time_smooth_enabled": True,
            "time_smooth_kernel_size": 11,
        },
        default_params={
            "freq_smooth_enabled": True,
            "freq_smooth_sigma": 1.5,
            "time_smooth_enabled": False,
            "time_smooth_kernel_size": 21,
        },
    )

    freq_widget = dialog._fields["freq_smooth_enabled"]
    freq_sigma_widget = dialog._fields["freq_smooth_sigma"]
    time_widget = dialog._fields["time_smooth_enabled"]
    time_kernel_widget = dialog._fields["time_smooth_kernel_size"]

    assert isinstance(freq_widget, QCheckBox)
    assert isinstance(time_widget, QCheckBox)
    assert freq_widget.isChecked() is False
    assert freq_sigma_widget.isEnabled() is False
    assert time_widget.isChecked() is True
    assert time_kernel_widget.isEnabled() is True

    freq_widget.setChecked(True)
    assert freq_sigma_widget.isEnabled() is True

    dialog._on_restore_defaults()
    assert freq_widget.isChecked() is True
    assert time_widget.isChecked() is False
    assert time_kernel_widget.isEnabled() is False

    dialog._on_submit("save")
    assert dialog.selected_params is not None
    assert dialog.selected_params["freq_smooth_enabled"] is True
    assert dialog.selected_params["freq_smooth_sigma"] == 1.5
    assert dialog.selected_params["time_smooth_enabled"] is False
    assert dialog.selected_params["time_smooth_kernel_size"] == 21


def test_tensor_config_export_payload_uses_whitelist_and_all_metrics(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    window._current_project = tmp_path / "project"
    window._current_subject = "sub-001"
    window._current_record = "recA"
    window._stage_states["preproc"] = "green"
    window._refresh_tensor_controls()
    window._tensor_metric_checks["raw_power"].setChecked(True)
    window._tensor_metric_checks["burst"].setChecked(True)
    window._tensor_mask_edge_checkbox.setChecked(False)
    window._tensor_low_freq_edit.setText("2")
    window._tensor_high_freq_edit.setText("60")
    window._tensor_step_edit.setText("1")
    window._tensor_time_resolution_edit.setText("0.5")
    window._tensor_hop_edit.setText("0.1")
    window._tensor_selected_channels_by_metric["raw_power"] = ("A", "B")
    window._tensor_selected_channels_by_metric["burst"] = ("B",)
    window._tensor_metric_params["raw_power"] = {
        "low_freq_hz": 2.0,
        "high_freq_hz": 60.0,
        "freq_step_hz": 1.0,
        "time_resolution_s": 0.5,
        "hop_s": 0.1,
        "method": "morlet",
        "min_cycles": 3.0,
        "max_cycles": 7.0,
        "time_bandwidth": 2.0,
        "selected_channels": ["A", "B"],
        "notches": [50.0],
        "n_freqs": 123,
    }
    window._tensor_metric_params["burst"] = {
        "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
        "percentile": 80.0,
        "baseline_keep": ["Rest"],
        "min_cycles": 4.0,
        "max_cycles": 8.0,
        "thresholds": [[1.0, 2.0]],
        "thresholds_path": "/tmp/thresholds.pkl",
        "selected_channels": ["B"],
    }
    window._tensor_metric_params["trgc"] = {
        "low_freq_hz": 2.0,
        "high_freq_hz": 60.0,
        "freq_step_hz": 1.0,
        "time_resolution_s": 0.5,
        "hop_s": 0.1,
        "method": "morlet",
        "min_cycles": 3.0,
        "max_cycles": None,
        "gc_n_lags": 20,
        "group_by_samples": False,
        "round_ms": 50.0,
        "selected_pairs": [["A", "B"]],
    }

    payload = window._build_tensor_config_export_payload()

    assert payload["schema"] == "lfptensorpipe.tensor-config"
    assert payload["version"] == 3
    assert payload["tensor"]["selected_metrics"] == ["raw_power", "burst"]
    assert payload["tensor"]["active_metric"] == "raw_power"
    assert payload["tensor"]["mask_edge_effects"] is False
    assert list(payload["tensor"]["metric_params"].keys()) == [
        spec.key for spec in main_window_module.TENSOR_METRICS if spec.supported
    ]
    assert payload["tensor"]["metric_params"]["raw_power"] == {
        "low_freq_hz": 2.0,
        "high_freq_hz": 60.0,
        "freq_step_hz": 1.0,
        "time_resolution_s": 0.5,
        "hop_s": 0.1,
        "method": "morlet",
        "min_cycles": 3.0,
        "max_cycles": 7.0,
        "time_bandwidth": 2.0,
        "notches": [50.0],
        "notch_widths": 2.0,
        "selected_channels": ["A", "B"],
    }
    assert "thresholds_path" not in payload["tensor"]["metric_params"]["burst"]
    assert payload["tensor"]["metric_params"]["burst"]["baseline_keep"] == ["Rest"]
    assert payload["tensor"]["metric_params"]["burst"]["notches"] == []
    assert payload["tensor"]["metric_params"]["burst"]["notch_widths"] == 2.0
    assert payload["tensor"]["metric_params"]["burst"]["thresholds"] == [[1.0, 2.0]]
    assert payload["tensor"]["metric_params"]["trgc"]["group_by_samples"] is False
    assert payload["tensor"]["metric_params"]["trgc"]["round_ms"] == 50.0


def test_tensor_config_import_payload_filters_selectors_and_warns() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    supported_metric_keys = [
        spec.key for spec in main_window_module.TENSOR_METRICS if spec.supported
    ]
    metric_params = {metric_key: {} for metric_key in supported_metric_keys}
    metric_params["raw_power"] = {
        "selected_channels": ["A", "Z"],
        "notches": [50.0],
        "notch_widths": 1.5,
    }
    metric_params["plv"] = {"selected_pairs": [["B", "A"], ["A", "Z"], ["B", "B"]]}
    metric_params["burst"] = {
        "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
        "baseline_keep": ["Rest"],
        "thresholds": [[1.0, 2.0]],
        "thresholds_path": "/tmp/ignore-me.pkl",
    }
    metric_params["unknown_metric"] = {"value": 1}

    payload = {
        "schema": "lfptensorpipe.tensor-config",
        "version": 3,
        "tensor": {
            "selected_metrics": ["raw_power", "unknown_metric", "plv"],
            "active_metric": "unknown_metric",
            "mask_edge_effects": True,
            "metric_params": metric_params,
        },
    }

    normalized, warnings = window._normalize_tensor_config_import_payload(
        payload,
        available_channels=("A", "B", "C"),
    )

    assert normalized["selected_metrics"] == ["raw_power", "plv"]
    assert normalized["active_metric"] == "raw_power"
    assert normalized["metric_params"]["raw_power"]["selected_channels"] == ["A"]
    assert normalized["metric_params"]["raw_power"]["notches"] == [50.0]
    assert normalized["metric_params"]["raw_power"]["notch_widths"] == 1.5
    assert normalized["metric_params"]["plv"]["selected_pairs"] == [["A", "B"]]
    assert normalized["metric_params"]["burst"]["baseline_keep"] == ["Rest"]
    assert normalized["metric_params"]["burst"]["notches"] == []
    assert normalized["metric_params"]["burst"]["notch_widths"] == 2.0
    assert normalized["metric_params"]["burst"]["thresholds"] == [[1.0, 2.0]]
    assert "thresholds_path" not in normalized["metric_params"]["burst"]
    assert any("unknown tensor metric keys" in warning for warning in warnings)
    assert any("unknown selected metrics" in warning for warning in warnings)
    assert any("falling back to raw_power" in warning for warning in warnings)
    assert any("unavailable channel" in warning for warning in warnings)
    assert any("unavailable pair" in warning for warning in warnings)


def test_tensor_config_import_payload_rejects_removed_periodic_smoothing_keys() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    supported_metric_keys = [
        spec.key for spec in main_window_module.TENSOR_METRICS if spec.supported
    ]
    metric_params = {metric_key: {} for metric_key in supported_metric_keys}
    metric_params["periodic_aperiodic"] = {
        "smooth_enabled": True,
        "kernel_size": 9,
    }

    with pytest.raises(ValueError, match="removed keys"):
        window._normalize_tensor_config_import_payload(
            {
                "schema": "lfptensorpipe.tensor-config",
                "version": 3,
                "tensor": {
                    "selected_metrics": [],
                    "active_metric": "raw_power",
                    "mask_edge_effects": True,
                    "metric_params": metric_params,
                },
            },
            available_channels=(),
        )


def test_tensor_config_import_payload_requires_all_supported_metrics() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    with pytest.raises(ValueError, match="missing metric definitions"):
        window._normalize_tensor_config_import_payload(
            {
                "schema": "lfptensorpipe.tensor-config",
                "version": 3,
                "tensor": {
                    "selected_metrics": [],
                    "active_metric": "raw_power",
                    "mask_edge_effects": True,
                    "metric_params": {"raw_power": {}},
                },
            },
            available_channels=(),
        )


def test_alignment_config_export_payload_contains_method_only() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    assert window._alignment_paradigm_list is not None
    assert window._alignment_method_combo is not None

    paradigm = {
        "name": "Gait",
        "slug": "gait",
        "trial_slug": "gait",
        "method": "stack_warper",
        "method_params": {
            "annotations": ["event"],
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "sample_rate": 0.4,
        },
        "annotation_filter": {},
    }
    window._alignment_paradigms = [paradigm]
    item = main_window_module.QListWidgetItem("Gait")
    item.setData(Qt.UserRole, "gait")
    window._alignment_paradigm_list.addItem(item)
    window._alignment_paradigm_list.setCurrentRow(0)
    method_idx = window._alignment_method_combo.findData("stack_warper")
    assert method_idx >= 0
    window._alignment_method_combo.setCurrentIndex(method_idx)

    payload = window._build_alignment_config_export_payload()

    assert payload["schema"] == "lfptensorpipe.alignment-config"
    assert payload["version"] == 1
    assert payload["alignment"] == {
        "method": "stack_warper",
        "method_params": {
            "annotations": ["event"],
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "drop_fields": ["bad", "edge"],
            "sample_rate": 0.4,
        },
    }

    window.close()


def test_alignment_config_import_payload_filters_labels_and_warns() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    payload = {
        "schema": "lfptensorpipe.alignment-config",
        "version": 1,
        "alignment": {
            "method": "linear_warper",
            "method_params": {
                "anchors_percent": {
                    "0": "event_start",
                    "50": "missing_mid",
                    "100": "event_end",
                },
                "epoch_duration_range": [0.0, 5.0],
                "linear_warp": True,
                "percent_tolerance": 15.0,
                "drop_bad": True,
                "sample_rate": 5.0,
            },
        },
    }

    normalized, warnings = window._normalize_alignment_config_import_payload(
        payload,
        annotation_labels=("event_start", "event_end"),
    )

    assert normalized["method"] == "linear_warper"
    assert normalized["method_params"]["anchors_percent"] == {
        0.0: "event_start",
        100.0: "event_end",
    }
    assert any("unavailable anchor label" in warning for warning in warnings)

    window.close()


def test_alignment_config_import_payload_allows_empty_annotations_after_filter() -> (
    None
):
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    payload = {
        "schema": "lfptensorpipe.alignment-config",
        "version": 1,
        "alignment": {
            "method": "stack_warper",
            "method_params": {
                "annotations": ["missing_event"],
                "duration_range": [0.0, 100.0],
                "drop_bad": False,
                "sample_rate": 0.4,
            },
        },
    }

    normalized, warnings = window._normalize_alignment_config_import_payload(
        payload,
        annotation_labels=("event",),
    )
    assert normalized["method"] == "stack_warper"
    assert normalized["method_params"] == {
        "annotations": [],
        "duration_range": [0.0, 100.0],
        "drop_bad": False,
        "drop_fields": ["bad", "edge"],
        "sample_rate": 0.4,
    }
    assert any("unavailable annotation label" in warning for warning in warnings)

    window.close()


def test_tensor_import_snapshot_persists_tensor_only_and_clears_tensor_dirty(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    project_root = tmp_path / "project"
    subject = "sub-001"
    record = "recA"
    resolver = main_window_module.PathResolver(
        main_window_module.RecordContext(
            project_root=project_root,
            subject=subject,
            record=record,
        )
    )
    resolver.lfp_root.mkdir(parents=True, exist_ok=True)
    ui_state_path = resolver.record_ui_state_path(create=True)
    ui_state_path.write_text(
        json.dumps(
            {
                "preproc": {"filter": {"basic": {"l_freq": 1.0}}},
                "tensor": {"selected_metrics": ["raw_power"]},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    window._current_project = project_root
    window._current_subject = subject
    window._current_record = record
    window._tensor_available_channels = ("A", "B")
    window._record_param_dirty_keys = {"features.axes", "tensor.metric_params"}
    tensor_snapshot = {
        "selected_metrics": ["burst"],
        "active_metric": "burst",
        "mask_edge_effects": False,
        "metric_params": {
            metric_key: {}
            for metric_key in (
                spec.key for spec in main_window_module.TENSOR_METRICS if spec.supported
            )
        },
    }
    tensor_snapshot["metric_params"]["burst"] = {
        "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
        "percentile": 80.0,
        "thresholds": [[1.0, 2.0]],
        "selected_channels": [],
    }

    context = window._record_context()
    assert context is not None

    window._apply_tensor_import_snapshot(context, tensor_snapshot)
    assert window._tensor_active_metric_key == "burst"
    assert window._tensor_selected_channels_by_metric["burst"] == ()
    assert "features.axes" in window._record_param_dirty_keys
    assert "tensor.metric_params" not in window._record_param_dirty_keys

    assert window._persist_imported_tensor_snapshot(context, tensor_snapshot) is True

    persisted = json.loads(ui_state_path.read_text(encoding="utf-8"))
    assert persisted["preproc"] == {"filter": {"basic": {"l_freq": 1.0}}}
    assert persisted["tensor"] == tensor_snapshot
    assert "features.axes" in window._record_param_dirty_keys
    assert not any(key.startswith("tensor.") for key in window._record_param_dirty_keys)


def test_tensor_and_feature_axis_dialogs_allow_overlap_ranges() -> None:
    _ = QApplication.instance() or QApplication([])

    tensor_dialog = TensorBandsConfigureDialog(
        title="Bands Configure",
        current_bands=(),
    )
    ok, message = tensor_dialog._validate_rows(
        [
            {"name": "BandA", "start": 1.0, "end": 4.0},
            {"name": "BandB", "start": 2.0, "end": 5.0},
        ]
    )
    assert ok and message == ""

    features_bands_dialog = FeatureAxisConfigureDialog(
        title="Bands Configure",
        item_label="Band",
        current_rows=(),
        default_rows=(),
        min_start=0.0,
        max_end=None,
    )
    valid_bands, normalized_bands, message_bands = features_bands_dialog._validate_rows(
        [
            {"name": "BandA", "start": 1.0, "end": 4.0},
            {"name": "BandB", "start": 2.0, "end": 6.0},
        ]
    )
    assert valid_bands and message_bands == ""
    assert [str(item["name"]) for item in normalized_bands] == ["BandA", "BandB"]

    features_phase_dialog = FeatureAxisConfigureDialog(
        title="Phases Configure",
        item_label="Phase",
        current_rows=(),
        default_rows=(),
        min_start=0.0,
        max_end=100.0,
        allow_duplicate_names=True,
    )
    valid_phases, normalized_phases, message_phases = (
        features_phase_dialog._validate_rows(
            [
                {"name": "strike_L", "start": 0.0, "end": 50.0},
                {"name": "stance_L", "start": 0.0, "end": 75.0},
                {"name": "strike_L", "start": 50.0, "end": 100.0},
            ]
        )
    )
    assert valid_phases and message_phases == ""
    assert [str(item["name"]) for item in normalized_phases] == [
        "strike_L",
        "stance_L",
        "strike_L",
    ]

    features_phase_dialog._rows = [dict(item) for item in normalized_phases]
    features_phase_dialog._render_table()
    features_phase_dialog._on_table_clicked(0, 4)
    assert [str(item["name"]) for item in features_phase_dialog._rows] == [
        "stance_L",
        "strike_L",
    ]

    features_phases_dialog = FeatureAxisConfigureDialog(
        title="Phases Configure",
        item_label="Phase",
        current_rows=(),
        default_rows=(),
        min_start=0.0,
        max_end=100.0,
    )
    valid_phases, normalized_phases, message_phases = (
        features_phases_dialog._validate_rows(
            [
                {"name": "PhaseA", "start": 0.0, "end": 60.0},
                {"name": "PhaseB", "start": 40.0, "end": 100.0},
            ]
        )
    )
    assert valid_phases and message_phases == ""
    assert [str(item["name"]) for item in normalized_phases] == ["PhaseA", "PhaseB"]


def test_tensor_bands_dialog_set_default_and_restore_defaults() -> None:
    _ = QApplication.instance() or QApplication([])
    saved_defaults: list[tuple[dict[str, float | str], ...]] = []
    dialog = TensorBandsConfigureDialog(
        title="Bands Configure",
        current_bands=({"name": "alpha", "start": 8.0, "end": 12.0},),
        default_bands=({"name": "theta", "start": 4.0, "end": 7.0},),
        set_default_callback=lambda rows: saved_defaults.append(
            tuple(dict(item) for item in rows)
        ),
    )

    footer = dialog.layout().itemAt(dialog.layout().count() - 1).widget()
    assert footer is not None
    footer_layout = footer.layout()
    assert footer_layout is not None
    indices = {
        "set_default": -1,
        "restore_defaults": -1,
    }
    for idx in range(footer_layout.count()):
        item = footer_layout.itemAt(idx)
        widget = item.widget()
        if widget is None or not isinstance(widget, main_window_module.QPushButton):
            continue
        text = widget.text()
        if text == "Set as Default":
            indices["set_default"] = idx
        elif text == "Restore Defaults":
            indices["restore_defaults"] = idx
    assert indices["set_default"] >= 0
    assert indices["restore_defaults"] >= 0
    assert indices["set_default"] < indices["restore_defaults"]

    dialog._on_clear_all()
    dialog._on_restore_defaults()
    assert dialog.selected_bands == ({"name": "theta", "start": 4.0, "end": 7.0},)

    dialog._bands = [{"name": "beta", "start": 13.0, "end": 30.0}]
    dialog._on_submit("set_default")
    assert dialog.selected_action == "set_default"
    assert saved_defaults == [({"name": "beta", "start": 13.0, "end": 30.0},)]
    dialog._on_clear_all()
    dialog._on_restore_defaults()
    assert dialog.selected_bands == ({"name": "beta", "start": 13.0, "end": 30.0},)

    dialog._on_submit("save")
    assert dialog.selected_action == "save"
    assert dialog.result() == QDialog.Accepted


def test_tensor_bands_configure_button_opens_real_dialog_without_name_error() -> None:
    app = QApplication.instance() or QApplication([])

    window = MainWindow(auto_load_dataset=False)
    window.show()
    app.processEvents()

    window._set_active_tensor_metric("psi")
    window._tensor_metric_params["psi"] = {
        "bands": [{"name": "alpha", "start": 8.0, "end": 12.0}],
    }
    window._apply_active_tensor_params_to_panel()
    window._refresh_tensor_bands_button_text()

    QTimer.singleShot(
        0,
        lambda: (
            app.activeModalWidget().reject()
            if isinstance(app.activeModalWidget(), QDialog)
            else None
        ),
    )

    window._on_tensor_bands_configure()

    assert window._tensor_metric_params["psi"]["bands"] == [
        {"name": "alpha", "start": 8.0, "end": 12.0}
    ]


def test_compact_dialog_del_action_cells_remove_rows() -> None:
    _ = QApplication.instance() or QApplication([])

    reset_dialog = main_window_module.ResetReferenceDialog(
        channel_names=("A", "B", "C"),
        current_rows=(
            main_window_module.ResetReferenceRow(
                anode="A",
                cathode="B",
                name="keep_ab",
            ),
        ),
    )
    assert reset_dialog._pair_table.cellWidget(0, 4) is None
    reset_action = reset_dialog._pair_table.item(0, 4)
    assert reset_action is not None
    assert reset_action.text() == "Del"
    reset_dialog._on_pair_table_clicked(0, 4)
    assert reset_dialog.selected_rows == ()

    tensor_dialog = TensorBandsConfigureDialog(
        title="Bands Configure",
        current_bands=({"name": "alpha", "start": 8.0, "end": 12.0},),
    )
    assert tensor_dialog._table.cellWidget(0, 4) is None
    tensor_action = tensor_dialog._table.item(0, 4)
    assert tensor_action is not None
    assert tensor_action.text() == "Del"
    tensor_dialog._on_table_clicked(0, 4)
    assert tensor_dialog.selected_bands == ()

    feature_dialog = FeatureAxisConfigureDialog(
        title="Bands Configure",
        item_label="Band",
        current_rows=({"name": "theta", "start": 4.0, "end": 8.0},),
        default_rows=(),
        min_start=0.0,
        max_end=None,
    )
    assert feature_dialog._table.cellWidget(0, 4) is None
    feature_action = feature_dialog._table.item(0, 4)
    assert feature_action is not None
    assert feature_action.text() == "Del"
    feature_dialog._on_table_clicked(0, 4)
    assert feature_dialog.selected_rows == ()

    baseline_dialog = main_window_module.BaselineRangeConfigureDialog(
        current_ranges=([0.0, 20.0], [30.0, 40.0]),
    )
    assert baseline_dialog._table.cellWidget(0, 3) is None
    baseline_action = baseline_dialog._table.item(0, 3)
    assert baseline_action is not None
    assert baseline_action.text() == "Del"
    baseline_dialog._on_table_clicked(0, 3)
    assert baseline_dialog.selected_ranges == ([30.0, 40.0],)


def test_reset_reference_dialog_supports_manual_unary_rows() -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []

    dialog = main_window_module.ResetReferenceDialog(
        channel_names=("A", "B", "C"),
        current_rows=(),
    )
    _attach_warning_sink(dialog, warnings)

    assert not dialog._draft_anode_edit.isReadOnly()

    dialog._draft_anode_edit.setText("A")
    dialog._draft_cathode_edit.setText("B")
    dialog._draft_name_edit.setText("neg_B")
    dialog._draft_anode_edit.clear()
    dialog._on_apply_draft()

    assert warnings == []
    assert dialog.selected_rows == (
        main_window_module.ResetReferenceRow(
            anode="",
            cathode="B",
            name="neg_B",
        ),
    )

    assert dialog._pair_table.item(0, 1).text() == "-"
    assert dialog._pair_table.item(0, 2).text() == "B"


def test_reset_reference_dialog_default_buttons_persist_to_record_config(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    dialog = main_window_module.RecordImportDialog(
        project_root=tmp_path,
        existing_records=(),
        default_import_type="Medtronic",
        config_store=store,
    )
    assert dialog._load_reset_reference_defaults() == ()

    row = main_window_module.ResetReferenceRow(
        anode="A",
        cathode="B",
        name="A_B",
    )
    reset_dialog = main_window_module.ResetReferenceDialog(
        channel_names=("A", "B", "C"),
        current_rows=(row,),
        default_rows=dialog._load_reset_reference_defaults(),
        set_default_callback=dialog._save_reset_reference_defaults,
    )

    footer = reset_dialog.layout().itemAt(reset_dialog.layout().count() - 1).widget()
    assert footer is not None
    footer_layout = footer.layout()
    assert footer_layout is not None
    labels = [
        footer_layout.itemAt(idx).widget().text()
        for idx in range(footer_layout.count())
        if footer_layout.itemAt(idx).widget() is not None
        and isinstance(
            footer_layout.itemAt(idx).widget(), main_window_module.QPushButton
        )
    ]
    assert "Set as Default" in labels
    assert "Restore Default" in labels

    reset_dialog._on_set_default()
    assert store.read_yaml("record.yml", default={}) == {
        "reset_reference_defaults": [
            {
                "anode": "A",
                "cathode": "B",
                "name": "A_B",
            }
        ]
    }

    reset_dialog._on_clear_all()
    assert reset_dialog.selected_rows == ()
    reset_dialog._on_restore_default()
    assert reset_dialog.selected_rows == (row,)

    store.write_yaml("record.yml", {})
    assert dialog._load_reset_reference_defaults() == ()


def test_record_import_dialog_loads_unary_reset_reference_defaults_with_empty_ui_semantics(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "record.yml",
        {
            "reset_reference_defaults": [
                {
                    "anode": None,
                    "cathode": "B",
                    "name": "neg_B",
                }
            ]
        },
    )

    dialog = main_window_module.RecordImportDialog(
        project_root=tmp_path,
        existing_records=(),
        default_import_type="Medtronic",
        config_store=store,
    )

    assert dialog._load_reset_reference_defaults() == (
        main_window_module.ResetReferenceRow(
            anode="",
            cathode="B",
            name="neg_B",
        ),
    )


def test_filter_and_qc_advance_dialog_branches() -> None:
    app = QApplication.instance() or QApplication([])
    warnings: list[str] = []

    session_params = {
        "notch_widths": [2.0, 3.0],
        "epoch_dur": 1.0,
        "p2p_thresh": [1e-6, 1e-3],
        "autoreject_correct_factor": 1.5,
    }
    default_params = {
        "notch_widths": 2.0,
        "epoch_dur": 2.0,
        "p2p_thresh": [1e-5, 1e-2],
        "autoreject_correct_factor": 1.2,
    }
    dialog = FilterAdvanceDialog(
        session_params=session_params,
        default_params=default_params,
    )
    _attach_warning_sink(dialog, warnings)
    dialog.show()
    app.processEvents()
    assert "Notch filter bandwidth (Hz)" in dialog._notch_widths_edit.toolTip()
    assert "Epoch length in seconds" in dialog._epoch_dur_edit.toolTip()
    assert (
        dialog._p2p_thresh_edit.toolTip()
        == "Peak-to-peak amplitude range in Volts: min,max. Epochs outside this "
        "range are marked BAD Must satisfy 0 <= min < max (e.g., 1e-6,1e-3)."
    )
    assert "AutoReject channel thresholds" in dialog._autoreject_factor_edit.toolTip()
    buttons = {
        button.text(): button.toolTip()
        for button in dialog.findChildren(main_window_module.QPushButton)
    }
    assert "Filter basic values" in buttons["Set as Default"]
    assert "Filter basic values" in buttons["Restore Defaults"]
    assert dialog._stringify_notch_widths([2.0, 3.0]) == "2, 3"
    with pytest.raises(ValueError, match="cannot be empty"):
        dialog._parse_notch_widths("")
    assert dialog._parse_notch_widths("1,2") == [1.0, 2.0]
    with pytest.raises(ValueError, match="two numbers"):
        dialog._parse_p2p_thresh("1")
    dialog._apply_to_fields({"p2p_thresh": "bad"})
    restore_calls = {"count": 0}
    dialog.set_restore_callback(
        lambda: restore_calls.__setitem__("count", restore_calls["count"] + 1)
    )
    dialog._on_restore_defaults()
    assert restore_calls["count"] == 1
    assert dialog._epoch_dur_edit.text() == "2"
    dialog._p2p_thresh_edit.setText("1")
    dialog._on_submit("save")
    assert any("Invalid parameters" in item for item in warnings)
    dialog._epoch_dur_edit.setText("-1")
    dialog._p2p_thresh_edit.setText("1e-6,1e-3")
    with pytest.raises(ValueError):
        dialog._collect_params()
    dialog._epoch_dur_edit.setText("1")
    dialog._p2p_thresh_edit.setText("1e-6,1e-3")
    dialog._on_submit("set_default")
    assert dialog.selected_action == "set_default"
    assert dialog.selected_params is not None

    with pytest.raises(ValueError, match="Unsupported QC mode"):
        QcAdvanceDialog(
            mode="bad",
            session_params={},
            default_params={},
        )

    psd_dialog = QcAdvanceDialog(
        mode="psd",
        session_params={"fmin": 1.0, "fmax": 50.0, "n_fft": 256, "average": True},
        default_params={"fmin": 2.0, "fmax": 40.0, "n_fft": 128, "average": False},
    )
    _attach_warning_sink(psd_dialog, warnings)
    psd_dialog.show()
    app.processEvents()
    psd_dialog._fmin_edit.setText("bad")
    psd_dialog._on_submit("save")
    assert any("Invalid parameters" in item for item in warnings)
    psd_dialog._on_restore_defaults()
    psd_dialog._on_submit("save")
    assert psd_dialog.selected_action == "save"
    assert psd_dialog.selected_params is not None

    tfr_dialog = QcAdvanceDialog(
        mode="tfr",
        session_params={"fmin": 1.0, "fmax": 40.0, "n_freqs": 20, "decim": 2},
        default_params={"fmin": 2.0, "fmax": 30.0, "n_freqs": 10, "decim": 1},
    )
    _attach_warning_sink(tfr_dialog, warnings)
    tfr_dialog.show()
    app.processEvents()
    tfr_dialog._decim_edit.setText("0")
    tfr_dialog._on_submit("save")
    assert any("Invalid parameters" in item for item in warnings)
    tfr_dialog._on_restore_defaults()
    tfr_dialog._on_submit("set_default")
    assert tfr_dialog.selected_action == "set_default"
    assert tfr_dialog.selected_params is not None


def test_annotation_configure_dialog_branches(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    warnings: list[str] = []
    load_state: dict[str, object] = {
        "path": "",
        "result": (False, [], "csv failed"),
    }

    dialog = AnnotationConfigureDialog(
        session_rows=[{"description": "evt", "onset": "0.1", "duration": "0.2"}],
        project_root=tmp_path,
    )
    _attach_warning_sink(dialog, warnings)
    dialog._open_file_name = (  # type: ignore[method-assign]
        lambda _title, _start_dir, _file_filter: (str(load_state["path"]), "CSV")
    )
    dialog._load_annotations_csv_rows = (  # type: ignore[method-assign]
        lambda _path: tuple(load_state["result"])
    )
    dialog.show()
    app.processEvents()
    assert dialog.minimumWidth() == 580
    assert dialog.minimumHeight() == 380
    assert dialog._rows_table is not None
    assert dialog._rows_table.rowCount() == 1
    assert dialog._rows_table.item(0, 1).text() == "evt"
    assert dialog._rows_table.columnWidth(2) == dialog._rows_table.columnWidth(4)
    assert dialog._rows_table.columnWidth(3) == dialog._rows_table.columnWidth(4)
    assert dialog._rows_table.columnWidth(5) == dialog._rows_table.columnWidth(4)

    dialog._search_edit.setText("missing")
    assert dialog._rows_table.rowCount() == 0
    dialog._search_edit.setText("")
    assert dialog._rows_table.rowCount() == 1

    dialog._draft_description_edit.setText("evt2")
    dialog._draft_start_edit.setText("1.0")
    dialog._draft_duration_edit.setText("0.5")
    dialog._draft_end_edit.setText("2.0")
    dialog._on_apply_draft()
    assert any(
        row["description"] == "evt2" and row["duration"] == 0.5
        for row in dialog.selected_rows
    )

    dialog._draft_description_edit.setText("evt3")
    dialog._draft_start_edit.setText("1.0")
    dialog._draft_duration_edit.setText("")
    dialog._draft_end_edit.setText("1.5")
    dialog._on_apply_draft()
    assert any(
        row["description"] == "evt3" and row["duration"] == 0.5
        for row in dialog.selected_rows
    )

    dialog._draft_description_edit.setText("evt_bad")
    dialog._draft_start_edit.setText("bad")
    dialog._draft_duration_edit.setText("1")
    dialog._on_apply_draft()
    assert any("Start must be a valid number." in item for item in warnings)

    dialog._on_import_annotations()
    load_state["path"] = tmp_path / "ann.csv"
    dialog._on_import_annotations()
    assert any("csv failed" in item for item in warnings)
    load_state["result"] = (
        True,
        [{"description": "evt4", "onset": 2.0, "duration": 1.0}],
        "ok",
    )
    dialog._on_import_annotations()
    assert any(row["description"] == "evt4" for row in dialog.selected_rows)
    dialog._search_edit.setText("evt4")
    assert dialog._rows_table.rowCount() == 1
    assert dialog._rows_table.cellWidget(0, 5) is None
    action_item = dialog._rows_table.item(0, 5)
    assert action_item is not None
    assert action_item.text() == "Del"
    dialog._on_rows_table_clicked(0, 5)
    assert all(row["description"] != "evt4" for row in dialog.selected_rows)
    dialog._search_edit.setText("")

    row_id = dialog._rows[0]["row_id"]
    dialog._on_remove_row(int(row_id))
    dialog._on_clear_draft()
    dialog._on_clear_all()
    assert dialog.selected_rows == ()
    dialog.accept()
    assert dialog.result() == QDialog.Accepted


def test_alignment_method_params_dialog_branches(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()
    warnings: list[str] = []
    infos: list[str] = []

    linear_dialog = AlignmentMethodParamsDialog(
        method_key="linear_warper",
        session_params={
            "anchors_percent": {"bad": "evt1", 20: "evt2"},
            "epoch_duration_range": "bad",
        },
        annotation_labels=["evt1", "evt2"],
        config_store=store,
    )
    _attach_warning_sink(linear_dialog, warnings)
    _attach_info_sink(linear_dialog, infos)
    linear_dialog.show()
    app.processEvents()
    assert linear_dialog._anchors_table.rowCount() >= 1
    assert linear_dialog._anchors_table.cellWidget(0, 2) is None
    anchor_action = linear_dialog._anchors_table.item(0, 2)
    assert anchor_action is not None
    assert anchor_action.text() == "Del"
    linear_dialog._anchor_percent_edit.setText("bad")
    linear_dialog._on_add_anchor_row()
    assert any("target percent must be numeric." in item for item in warnings)
    linear_dialog._anchor_percent_edit.setText("50")
    baseline_anchor_rows = linear_dialog._anchors_table.rowCount()
    linear_dialog._on_add_anchor_row()
    assert linear_dialog._anchors_table.rowCount() == baseline_anchor_rows + 1
    linear_dialog._on_anchors_table_clicked(
        linear_dialog._anchors_table.rowCount() - 1,
        2,
    )
    assert linear_dialog._anchors_table.rowCount() == baseline_anchor_rows
    linear_dialog._anchors_table.insertRow(linear_dialog._anchors_table.rowCount())
    linear_dialog._anchors_table.setItem(
        linear_dialog._anchors_table.rowCount() - 1,
        0,
        QTableWidgetItem(""),
    )
    linear_dialog._percent_tolerance_edit.setText("bad")
    linear_dialog._on_save()
    assert any(
        "Fix highlighted table cells before saving." in item for item in warnings
    )
    linear_dialog._anchors_table.removeRow(linear_dialog._anchors_table.rowCount() - 1)
    linear_dialog._percent_tolerance_edit.setText("10")
    linear_dialog._set_anchor_rows([("evt1", 0.0), ("evt2", 100.0)])
    linear_dialog._on_save()
    assert linear_dialog.selected_params is not None

    pad_dialog = AlignmentMethodParamsDialog(
        method_key="pad_warper",
        session_params={
            "annotations": ["evt1"],
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
            "duration_range": "bad",
            "sample_rate": 50.0,
        },
        annotation_labels=["evt1", "evt2"],
        config_store=store,
    )
    _attach_warning_sink(pad_dialog, warnings)
    _attach_info_sink(pad_dialog, infos)
    pad_dialog.show()
    app.processEvents()
    pad_dialog._on_select_all_annotations()
    pad_dialog._duration_min_edit.setText("0")
    pad_dialog._duration_max_edit.setText("1")
    params = pad_dialog._collect_candidate_params()
    assert sorted(params.get("annotations", [])) == ["evt1", "evt2"]
    assert float(params.get("pad_left", 0.0)) == 0.5
    pad_dialog._pad_left_edit.setText("bad")
    pad_dialog._duration_min_edit.setText("bad")
    pad_dialog._on_save()
    assert any("Invalid parameters" in item for item in warnings)
    pad_dialog._pad_left_edit.setText("0.5")
    pad_dialog._duration_min_edit.setText("0")
    pad_dialog._duration_max_edit.setText("1")
    pad_dialog._on_save()
    assert pad_dialog.selected_params is not None

    stack_dialog = AlignmentMethodParamsDialog(
        method_key="stack_warper",
        session_params={
            "annotations": ["evt3"],
            "duration_range": "bad",
        },
        annotation_labels=["evt1", "evt2"],
        config_store=store,
    )
    _attach_warning_sink(stack_dialog, warnings)
    _attach_info_sink(stack_dialog, infos)
    stack_dialog.show()
    app.processEvents()
    assert stack_dialog._parse_optional_float("") is None
    assert stack_dialog._parse_optional_float("1.5") == 1.5
    stack_dialog._on_select_all_annotations()
    stack_dialog._on_clear_annotations()
    stack_dialog._duration_min_edit.setText("bad")
    stack_dialog._on_save()
    assert any("Invalid parameters" in item for item in warnings)
    stack_dialog._duration_min_edit.setText("0")
    stack_dialog._duration_max_edit.setText("1")
    stack_dialog._on_save()
    assert stack_dialog.selected_params is not None
    assert stack_dialog.selected_params["annotations"] == []
    assert "pad_s" not in stack_dialog.selected_params

    stack_empty_dialog = AlignmentMethodParamsDialog(
        method_key="stack_warper",
        session_params={"annotations": "bad", "duration_range": "bad"},
        annotation_labels=[],
        config_store=store,
    )
    _attach_warning_sink(stack_empty_dialog, warnings)
    _attach_info_sink(stack_empty_dialog, infos)
    stack_empty_dialog.show()
    app.processEvents()
    stack_empty_dialog._duration_min_edit.setText("0")
    stack_empty_dialog._duration_max_edit.setText("1")
    stack_empty_dialog._on_save()
    assert stack_empty_dialog.selected_params is not None

    concat_dialog = AlignmentMethodParamsDialog(
        method_key="concat_warper",
        session_params={
            "annotations": ["evt1"],
        },
        annotation_labels=["evt1", "evt2"],
        config_store=store,
    )
    _attach_warning_sink(concat_dialog, warnings)
    _attach_info_sink(concat_dialog, infos)
    concat_dialog.show()
    app.processEvents()
    concat_dialog._on_select_all_annotations()
    concat_dialog._on_save()
    assert concat_dialog.selected_params is not None
    assert "pad_s" not in concat_dialog.selected_params
    assert "duration_range" not in concat_dialog.selected_params


def test_alignment_method_params_dialog_helper_edge_branches(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()

    linear_dialog = AlignmentMethodParamsDialog(
        method_key="linear_warper",
        session_params={"anchors_percent": "bad"},
        annotation_labels=["evt1", "evt2"],
        config_store=store,
    )
    linear_dialog._show_warning = lambda title, message: 0  # type: ignore[method-assign]
    linear_dialog._show_information = (  # type: ignore[method-assign]
        lambda title, message: 0
    )
    linear_dialog.show()
    app.processEvents()
    linear_dialog._clear_method_ui()
    linear_dialog._build_linear_ui({"anchors_percent": "bad"})
    assert linear_dialog._anchors_table.rowCount() == 0
    linear_dialog._set_anchor_rows([("evt1", 50.0), ("evt2", 50.0)])
    with pytest.raises(ValueError, match="unique"):
        linear_dialog._collect_candidate_params()

    pad_dialog = AlignmentMethodParamsDialog(
        method_key="pad_warper",
        session_params={
            "annotations": ["evt1"],
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
            "duration_range": [0.0, 1.0],
            "sample_rate": 50.0,
        },
        annotation_labels=["evt1", "evt2"],
        config_store=store,
    )
    pad_dialog._show_warning = lambda title, message: 0  # type: ignore[method-assign]
    pad_dialog._show_information = lambda title, message: 0  # type: ignore[method-assign]
    pad_dialog.show()
    app.processEvents()
    checked_before = [
        pad_dialog._annotation_list.item(i).checkState() == Qt.Checked
        for i in range(pad_dialog._annotation_list.count())
    ]
    assert checked_before.count(True) == 1
    pad_dialog._on_select_all_annotations()
    checked_after_select = [
        pad_dialog._annotation_list.item(i).checkState() == Qt.Checked
        for i in range(pad_dialog._annotation_list.count())
    ]
    assert all(checked_after_select)
    pad_dialog._on_clear_annotations()
    checked_after_clear = [
        pad_dialog._annotation_list.item(i).checkState() == Qt.Checked
        for i in range(pad_dialog._annotation_list.count())
    ]
    assert not any(checked_after_clear)

    pad_empty_dialog = AlignmentMethodParamsDialog(
        method_key="pad_warper",
        session_params={
            "annotations": [],
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
            "duration_range": [0.0, 1.0],
            "sample_rate": 50.0,
        },
        annotation_labels=[],
        config_store=store,
    )
    pad_empty_dialog.show()
    app.processEvents()
    assert pad_empty_dialog._annotation_list.count() == 0

    stack_default_dialog = AlignmentMethodParamsDialog(
        method_key="stack_warper",
        session_params={
            "annotations": [],
            "duration_range": [0.0, 1.0],
            "sample_rate": 0.5,
        },
        annotation_labels=["evt1", "evt2"],
        config_store=store,
    )
    stack_default_dialog.show()
    app.processEvents()
    stack_checked = [
        stack_default_dialog._annotation_list.item(i).checkState() == Qt.Checked
        for i in range(stack_default_dialog._annotation_list.count())
    ]
    assert stack_checked == [False, False]


def test_alignment_method_params_anchor_rows_auto_sort(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()

    dialog = AlignmentMethodParamsDialog(
        method_key="linear_warper",
        session_params={
            "anchors_percent": {100.0: "evt_end", 0.0: "evt_start"},
            "epoch_duration_range": [None, None],
        },
        annotation_labels=["evt_start", "evt_mid", "evt_end"],
        config_store=store,
    )
    dialog._show_warning = lambda title, message: 0  # type: ignore[method-assign]
    dialog._show_information = lambda title, message: 0  # type: ignore[method-assign]
    dialog.show()
    app.processEvents()

    initial_percents = [
        float(dialog._anchors_table.item(row, 1).text())
        for row in range(dialog._anchors_table.rowCount())
    ]
    assert initial_percents == [0.0, 100.0]

    dialog._anchor_label_combo.setCurrentText("evt_mid")
    dialog._anchor_percent_edit.setText("50")
    dialog._on_add_anchor_row()

    added_percents = [
        float(dialog._anchors_table.item(row, 1).text())
        for row in range(dialog._anchors_table.rowCount())
    ]
    added_labels = [
        dialog._anchors_table.item(row, 0).text()
        for row in range(dialog._anchors_table.rowCount())
    ]
    assert added_percents == [0.0, 50.0, 100.0]
    assert added_labels == ["evt_start", "evt_mid", "evt_end"]

    dialog._on_remove_anchor_row(1)
    percents_after_remove = [
        float(dialog._anchors_table.item(row, 1).text())
        for row in range(dialog._anchors_table.rowCount())
    ]
    assert percents_after_remove == [0.0, 100.0]


def test_alignment_method_params_dialog_filters_hidden_drop_field_labels(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()

    annotation_labels = [
        "EDGE_concat",
        "evt_start",
        "bad_trial",
        "evt_end",
    ]

    linear_dialog = AlignmentMethodParamsDialog(
        method_key="linear_warper",
        session_params={
            "anchors_percent": {0.0: "evt_start", 100.0: "evt_end"},
            "epoch_duration_range": [None, None],
            "drop_fields": ["bad", "edge"],
        },
        annotation_labels=annotation_labels,
        config_store=store,
    )
    linear_dialog.show()
    app.processEvents()
    linear_labels = [
        linear_dialog._anchor_label_combo.itemText(idx)
        for idx in range(linear_dialog._anchor_label_combo.count())
    ]
    assert linear_labels == ["evt_start", "evt_end"]

    expected_visible = ["evt_start", "evt_end"]
    dialog_cases = (
        (
            "pad_warper",
            {
                "annotations": ["evt_start", "EDGE_concat"],
                "pad_left": 0.5,
                "anno_left": 0.5,
                "anno_right": 0.5,
                "pad_right": 0.5,
                "duration_range": [0.0, 1.0],
                "sample_rate": 50.0,
                "drop_fields": ["bad", "edge"],
            },
        ),
        (
            "stack_warper",
            {
                "annotations": ["evt_start", "EDGE_concat"],
                "duration_range": [0.0, 1.0],
                "sample_rate": 0.5,
                "drop_fields": ["bad", "edge"],
            },
        ),
        (
            "concat_warper",
            {
                "annotations": ["evt_start", "EDGE_concat"],
                "sample_rate": 50.0,
                "drop_fields": ["bad", "edge"],
            },
        ),
    )
    for method_key, session_params in dialog_cases:
        dialog = AlignmentMethodParamsDialog(
            method_key=method_key,
            session_params=session_params,
            annotation_labels=annotation_labels,
            config_store=store,
        )
        dialog.show()
        app.processEvents()
        visible_labels = [
            dialog._annotation_list.item(idx).text()
            for idx in range(dialog._annotation_list.count())
        ]
        assert visible_labels == expected_visible
