"""Integration checks for minimal preprocess actions in MainWindow."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LFPTP_DISABLE_MATLAB_WARMUP", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QStyle,
    QTableWidgetItem,
    QWidget,
)
import mne

from lfptensorpipe.app.alignment_service import (
    alignment_paradigm_log_path,
    create_alignment_paradigm,
    update_alignment_paradigm,
)
from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import (
    mark_preproc_step,
    preproc_step_log_path,
    preproc_step_raw_path,
)
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    append_run_log_event,
    indicator_from_log,
    read_run_log,
    write_ui_state,
    write_run_log,
)
from lfptensorpipe.app.tensor_service import (
    tensor_metric_log_path,
    tensor_metric_tensor_path,
)
from lfptensorpipe.io.pkl_io import save_pkl
from lfptensorpipe.gui.main_window import MainWindow
import lfptensorpipe.gui.main_window as main_window_module
import lfptensorpipe.gui.shell.main_window_layout as main_window_layout_module


class OverrideMainWindow(MainWindow):
    def __init__(
        self, *args, overrides: dict[str, Any] | None = None, **kwargs
    ) -> None:
        self._overrides = dict(overrides or {})
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        if name not in {"_overrides", "__dict__", "__class__", "__getattribute__"}:
            try:
                overrides = object.__getattribute__(self, "_overrides")
            except AttributeError:
                overrides = {}
            if name in overrides:
                value = overrides[name]
                if hasattr(value, "__get__"):
                    return value.__get__(self, type(self))
                return value
        return super().__getattribute__(name)


def _select_record_context(window: MainWindow, *, subject: str, record: str) -> None:
    subject_combo = window._subject_combo
    assert subject_combo is not None
    subject_idx = subject_combo.findData(subject)
    assert subject_idx >= 0
    subject_combo.setCurrentIndex(subject_idx)
    assert window._select_record_item(record)
    assert window._current_record == record


def _build_window_with_preproc_steps(
    tmp_path: Path,
    *,
    steps: tuple[str, ...],
    enable_plots: bool = False,
    window_cls: type[MainWindow] = MainWindow,
    overrides: dict[str, Any] | None = None,
) -> tuple[MainWindow, RecordContext, PathResolver, dict[str, Path]]:
    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    step_paths: dict[str, Path] = {}
    for step in steps:
        raw_path = preproc_step_raw_path(resolver, step)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(str(raw_path), overwrite=True)
        mark_preproc_step(
            resolver=resolver,
            step=step,
            completed=True,
            input_path=str(raw_path),
            output_path=str(raw_path),
            message=f"{step} ready",
        )
        step_paths[step] = raw_path

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window_kwargs: dict[str, Any] = {
        "config_store": store,
        "demo_data_root": tmp_path / "missing_demo",
        "enable_plots": enable_plots,
    }
    if overrides is not None:
        window_kwargs["overrides"] = overrides
    window = window_cls(**window_kwargs)
    _select_record_context(window, subject=subject, record=record)
    return window, context, resolver, step_paths


def _build_alignment_window(
    tmp_path: Path,
    *,
    enable_plots: bool = False,
    window_cls: type[MainWindow] = MainWindow,
    overrides: dict[str, Any] | None = None,
) -> tuple[MainWindow, RecordContext, PathResolver, str]:
    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 500), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(onset=[0.5], duration=[0.8], description=["event"])
    )
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    created, _, entry = create_alignment_paradigm(store, name="Gait", context=context)
    assert created and isinstance(entry, dict)
    slug = str(entry["slug"])
    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
        method_params={
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.4,
        },
        context=context,
    )
    assert updated, message

    window_kwargs: dict[str, Any] = {
        "config_store": store,
        "demo_data_root": tmp_path / "missing_demo",
        "enable_plots": enable_plots,
    }
    if overrides is not None:
        window_kwargs["overrides"] = overrides
    window = window_cls(**window_kwargs)
    _select_record_context(window, subject=subject, record=record)
    return window, context, resolver, slug


def _seed_alignment_trial_finish_ready(
    resolver: PathResolver,
    slug: str,
    *,
    metrics: tuple[str, ...] = ("raw_power",),
) -> None:
    write_run_log(
        preproc_step_log_path(resolver, "finish"),
        RunLogRecord(
            step="finish",
            completed=True,
            params={},
            input_path="in",
            output_path=str(preproc_step_raw_path(resolver, "finish")),
            message="finish ready",
        ),
    )
    for metric_key in metrics:
        write_run_log(
            tensor_metric_log_path(resolver, metric_key),
            RunLogRecord(
                step=metric_key,
                completed=True,
                params={},
                input_path="in",
                output_path=str(tensor_metric_tensor_path(resolver, metric_key)),
                message=f"{metric_key} tensor ready",
            ),
        )
    trial_dir = resolver.alignment_paradigm_dir(slug)
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_cfg = {
        "name": slug,
        "slug": slug,
        "trial_slug": slug,
        "method": "stack_warper",
        "method_params": {
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.4,
        },
        "annotation_filter": {},
    }
    append_run_log_event(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={
                "trial_slug": slug,
                "name": slug,
                "method": "stack_warper",
                "method_params": dict(trial_cfg["method_params"]),
                "metrics": list(metrics),
            },
            input_path="in",
            output_path=str(trial_dir),
            message=f"{slug} alignment ready",
        ),
        state_patch={"trial_config": trial_cfg},
    )
    (trial_dir / "warp_fn.pkl").write_bytes(b"ok")
    (trial_dir / "warp_labels.pkl").write_bytes(b"ok")
    for metric_key in metrics:
        metric_dir = trial_dir / metric_key
        metric_dir.mkdir(parents=True, exist_ok=True)
        tensor_warped_path = metric_dir / "tensor_warped.pkl"
        if not tensor_warped_path.exists():
            tensor_warped_path.write_bytes(b"ok")
        raw_table_path = metric_dir / "na-raw.pkl"
        if not raw_table_path.exists():
            raw_table_path.write_bytes(b"ok")
    append_run_log_event(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="build_raw_table",
            completed=True,
            params={
                "trial_slug": slug,
                "picked_epoch_indices": [0],
                "merge_location_info_ready": False,
            },
            input_path=str(trial_dir),
            output_path=str(trial_dir),
            message=f"{slug} finish ready",
        ),
    )


def _create_ready_alignment_trial(
    store: AppConfigStore,
    context: RecordContext,
    resolver: PathResolver,
    *,
    name: str,
    metrics: tuple[str, ...] = ("raw_power",),
) -> str:
    created, message, entry = create_alignment_paradigm(
        store,
        name=name,
        context=context,
    )
    assert created and isinstance(entry, dict), message
    slug = str(entry["slug"])
    updated, update_message = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
        method_params={
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.4,
        },
        context=context,
    )
    assert updated, update_message
    for metric_key in metrics:
        raw_table_path = resolver.alignment_root / slug / metric_key / "na-raw.pkl"
        raw_table_path.parent.mkdir(parents=True, exist_ok=True)
        save_pkl(
            pd.DataFrame(
                [
                    {
                        "Value": pd.Series([1.0, 2.0], index=[13.0, 30.0]),
                    }
                ]
            ),
            raw_table_path,
        )
    _seed_alignment_trial_finish_ready(resolver, slug, metrics=metrics)
    return slug


def test_mainwindow_preproc_raw_and_finish_actions(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)
    rawdata_input = (
        project
        / "rawdata"
        / subject
        / "ses-postop"
        / "lfp"
        / record
        / "raw"
        / "raw.fif"
    )
    rawdata_input.parent.mkdir(parents=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(rawdata_input), overwrite=True)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    def fake_apply_filter_step(
        ctx,
        *,
        advance_params=None,
        notches=None,
        l_freq=None,
        h_freq=None,
    ):
        _ = (advance_params, notches, l_freq, h_freq)
        resolver = PathResolver(ctx)
        source_path = preproc_step_raw_path(resolver, "raw")
        filter_path = preproc_step_raw_path(resolver, "filter")
        filter_path.parent.mkdir(parents=True, exist_ok=True)
        filter_path.write_bytes(source_path.read_bytes())
        mark_preproc_step(
            resolver=resolver,
            step="filter",
            completed=True,
            input_path=str(source_path),
            output_path=str(filter_path),
            message="filter ok",
        )
        return True, "ok"

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
        overrides={
            "_apply_filter_step_runtime": lambda self, *args, **kwargs: fake_apply_filter_step(
                *args, **kwargs
            ),
        },
    )
    _select_record_context(window, subject=subject, record=record)

    assert window._preproc_raw_plot_button.isEnabled()
    window._on_preproc_raw_plot()

    raw_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "raw"
        / "lfptensorpipe_log.json"
    )
    assert indicator_from_log(raw_log) == "green"
    assert window._preproc_filter_advance_button is not None
    assert window._preproc_filter_advance_button.isEnabled()
    assert window._preproc_filter_apply_button.isEnabled()
    window._on_preproc_filter_apply()

    filter_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "filter"
        / "lfptensorpipe_log.json"
    )
    assert indicator_from_log(filter_log) == "green"
    assert window._preproc_finish_apply_button.isEnabled()

    window._on_preproc_finish_apply()

    finish_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "finish"
        / "lfptensorpipe_log.json"
    )
    assert indicator_from_log(finish_log) == "green"
    assert window._stage_states["preproc"] == "green"
    assert window._stage_buttons["tensor"].isEnabled()

    window.close()


def test_mainwindow_filter_apply_uses_session_advance_params(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    raw_path = preproc_step_raw_path(resolver, "raw")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="raw",
        completed=True,
        input_path=str(raw_path),
        output_path=str(raw_path),
        message="raw ready",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
        overrides={
            "_apply_filter_step_runtime": lambda self, *args, **kwargs: fake_apply_filter_step(
                *args, **kwargs
            ),
        },
    )
    _select_record_context(window, subject=subject, record=record)

    captured: dict[str, object] = {}
    desired_params = {
        "notch_widths": [1.1, 1.4],
        "epoch_dur": 0.8,
        "p2p_thresh": [1e-7, 2e-4],
        "autoreject_correct_factor": 2.2,
    }
    window._preproc_filter_advance_params = dict(desired_params)

    def fake_apply_filter_step(
        ctx,
        *,
        advance_params=None,
        notches=None,
        l_freq=None,
        h_freq=None,
    ):
        _ = ctx
        captured["advance_params"] = advance_params
        captured["notches"] = notches
        captured["l_freq"] = l_freq
        captured["h_freq"] = h_freq
        return True, "ok"

    window._on_preproc_filter_apply()

    assert captured["advance_params"] == desired_params
    assert captured["notches"] == [50.0, 100.0]
    assert captured["l_freq"] == 1.0
    assert captured["h_freq"] == 200.0
    window.close()


def test_mainwindow_preproc_bad_segment_apply_action(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    annotations_raw_path = preproc_step_raw_path(resolver, "annotations")
    annotations_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(annotations_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="annotations",
        completed=True,
        input_path=str(annotations_raw_path),
        output_path=str(annotations_raw_path),
        message="annotations ready",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)

    assert window._preproc_bad_segment_apply_button.isEnabled()
    window._on_preproc_bad_segment_apply()

    bad_segment_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "bad_segment_removal"
        / "lfptensorpipe_log.json"
    )
    assert indicator_from_log(bad_segment_log) == "green"
    assert window._preproc_bad_segment_plot_button.isEnabled()

    window.close()


def test_mainwindow_preproc_ecg_apply_action(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    bad_segment_raw_path = preproc_step_raw_path(resolver, "bad_segment_removal")
    bad_segment_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(bad_segment_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="bad_segment_removal",
        completed=True,
        input_path=str(bad_segment_raw_path),
        output_path=str(bad_segment_raw_path),
        message="bad segment ready",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)

    assert window._preproc_ecg_apply_button.isEnabled()
    if window._preproc_ecg_method_combo is not None:
        window._preproc_ecg_method_combo.setCurrentText("svd")

    def fake_apply_ecg_step(ctx, *, method="template", picks=None):
        _ = method
        _ = picks
        ecg_raw = preproc_step_raw_path(PathResolver(ctx), "ecg_artifact_removal")
        ecg_raw.parent.mkdir(parents=True, exist_ok=True)
        mne.io.RawArray(np.zeros((2, 400), dtype=float), info).save(
            str(ecg_raw), overwrite=True
        )
        mark_preproc_step(
            resolver=PathResolver(ctx),
            step="ecg_artifact_removal",
            completed=True,
            input_path=str(
                preproc_step_raw_path(PathResolver(ctx), "bad_segment_removal")
            ),
            output_path=str(ecg_raw),
            message="ecg done",
        )
        return True, "ECG step completed."

    window._on_preproc_ecg_apply()

    ecg_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "ecg_artifact_removal"
        / "lfptensorpipe_log.json"
    )
    assert indicator_from_log(ecg_log) == "green"
    assert window._preproc_ecg_plot_button.isEnabled()

    window.close()


def test_mainwindow_preproc_annotations_save_action(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    filter_raw_path = preproc_step_raw_path(resolver, "filter")
    filter_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(filter_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        input_path=str(filter_raw_path),
        output_path=str(filter_raw_path),
        message="filter ready",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)

    assert window._preproc_annotations_edit_button.isEnabled()
    assert window._preproc_annotations_edit_button.text() == "Configure..."
    assert window._preproc_annotations_save_button.isEnabled()
    assert window._preproc_annotations_table is not None
    if window._preproc_annotations_table.rowCount() == 0:
        window._preproc_annotations_table.insertRow(0)
    window._preproc_annotations_table.setItem(0, 0, QTableWidgetItem("BAD"))
    window._preproc_annotations_table.setItem(0, 1, QTableWidgetItem("0.1"))
    window._preproc_annotations_table.setItem(0, 2, QTableWidgetItem("0.2"))
    window._on_preproc_annotations_save()

    annotations_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "annotations"
        / "lfptensorpipe_log.json"
    )
    assert indicator_from_log(annotations_log) == "green"
    assert window._preproc_bad_segment_apply_button.isEnabled()

    window.close()


def test_mainwindow_annotations_save_invalid_writes_yellow_log(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    filter_raw_path = preproc_step_raw_path(resolver, "filter")
    filter_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(filter_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="filter",
        completed=True,
        input_path=str(filter_raw_path),
        output_path=str(filter_raw_path),
        message="filter ready",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
        overrides={
            "_show_warning": lambda self, _title, _message: 0,
        },
    )
    _select_record_context(window, subject=subject, record=record)

    assert window._preproc_annotations_table is not None
    if window._preproc_annotations_table.rowCount() == 0:
        window._preproc_annotations_table.insertRow(0)
    window._preproc_annotations_table.setItem(0, 0, QTableWidgetItem("BAD"))
    window._preproc_annotations_table.setItem(0, 1, QTableWidgetItem("0.1"))
    window._preproc_annotations_table.setItem(0, 2, QTableWidgetItem("-0.1"))
    window._on_preproc_annotations_save()

    annotations_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "annotations"
        / "lfptensorpipe_log.json"
    )
    assert indicator_from_log(annotations_log) == "yellow"
    assert window._preproc_annotations_edit_button.text() == "Configure..."
    window.close()


def test_mainwindow_raw_plot_autosave_hook_marks_log_and_invalidates_downstream(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    (project / "rawdata" / subject / "ses-postop" / "lfp" / record / "raw").mkdir(
        parents=True
    )
    (
        project
        / "rawdata"
        / subject
        / "ses-postop"
        / "lfp"
        / record
        / "raw"
        / "raw.fif"
    ).write_text(
        "seed",
        encoding="utf-8",
    )
    preproc_raw = preproc_step_raw_path(resolver, "raw")
    preproc_raw.parent.mkdir(parents=True, exist_ok=True)
    preproc_raw.write_text("old", encoding="utf-8")
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path="in",
        output_path="out",
        message="seed finish",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)

    class _CanvasStub:
        def __init__(self) -> None:
            self.callback = None

        def mpl_connect(self, event_name, callback):
            assert event_name == "close_event"
            self.callback = callback
            return 1

    class _FigureStub:
        def __init__(self) -> None:
            self.canvas = _CanvasStub()

    class _BrowserStub:
        def __init__(self) -> None:
            self.fig = _FigureStub()

    class _RawStub:
        def __init__(self) -> None:
            self.saved_paths: list[str] = []

        def save(self, path: str, overwrite: bool = False) -> None:
            _ = overwrite
            self.saved_paths.append(path)
            Path(path).write_text("saved", encoding="utf-8")

    raw_stub = _RawStub()
    browser_stub = _BrowserStub()
    window._attach_plot_autosave(
        browser=browser_stub,
        raw=raw_stub,
        raw_path=preproc_raw,
        step="raw",
        title_prefix="Raw",
    )
    assert browser_stub.fig.canvas.callback is not None
    browser_stub.fig.canvas.callback(None)

    raw_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "raw"
        / "lfptensorpipe_log.json"
    )
    finish_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "finish"
        / "lfptensorpipe_log.json"
    )
    assert raw_stub.saved_paths
    assert indicator_from_log(raw_log) == "green"
    assert indicator_from_log(finish_log) == "yellow"
    window.close()


def test_mainwindow_raw_plot_autosave_hook_handles_qt_close_event(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    (project / "rawdata" / subject / "ses-postop" / "lfp" / record / "raw").mkdir(
        parents=True
    )
    (
        project
        / "rawdata"
        / subject
        / "ses-postop"
        / "lfp"
        / record
        / "raw"
        / "raw.fif"
    ).write_text(
        "seed",
        encoding="utf-8",
    )
    preproc_raw = preproc_step_raw_path(resolver, "raw")
    preproc_raw.parent.mkdir(parents=True, exist_ok=True)
    preproc_raw.write_text("old", encoding="utf-8")
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path="in",
        output_path="out",
        message="seed finish",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)

    class _RawStub:
        def __init__(self) -> None:
            self.saved_paths: list[str] = []

        def save(self, path: str, overwrite: bool = False) -> None:
            _ = overwrite
            self.saved_paths.append(path)
            Path(path).write_text("saved", encoding="utf-8")

    raw_stub = _RawStub()
    browser_stub = QWidget()
    window._attach_plot_autosave(
        browser=browser_stub,
        raw=raw_stub,
        raw_path=preproc_raw,
        step="raw",
        title_prefix="Raw",
    )
    browser_stub.close()
    app.processEvents()

    raw_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "raw"
        / "lfptensorpipe_log.json"
    )
    finish_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "finish"
        / "lfptensorpipe_log.json"
    )
    assert raw_stub.saved_paths
    assert indicator_from_log(raw_log) == "green"
    assert indicator_from_log(finish_log) == "yellow"
    window.close()


def test_mainwindow_raw_plot_autosave_prefers_mne_gotclosed_signal(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    (project / "rawdata" / subject / "ses-postop" / "lfp" / record / "raw").mkdir(
        parents=True
    )
    (
        project
        / "rawdata"
        / subject
        / "ses-postop"
        / "lfp"
        / record
        / "raw"
        / "raw.fif"
    ).write_text(
        "seed",
        encoding="utf-8",
    )
    preproc_raw = preproc_step_raw_path(resolver, "raw")
    preproc_raw.parent.mkdir(parents=True, exist_ok=True)
    preproc_raw.write_text("old", encoding="utf-8")
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path="in",
        output_path="out",
        message="seed finish",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)

    class _RawStub:
        def __init__(self) -> None:
            self.saved_paths: list[str] = []

        def save(self, path: str, overwrite: bool = False) -> None:
            _ = overwrite
            self.saved_paths.append(path)
            Path(path).write_text("saved", encoding="utf-8")

    class _BrowserWithClosedSignal(QWidget):
        gotClosed = Signal()

    raw_stub = _RawStub()
    browser_stub = _BrowserWithClosedSignal()
    window._attach_plot_autosave(
        browser=browser_stub,
        raw=raw_stub,
        raw_path=preproc_raw,
        step="raw",
        title_prefix="Raw",
    )

    browser_stub.close()
    app.processEvents()
    assert not raw_stub.saved_paths

    browser_stub.gotClosed.emit()
    app.processEvents()
    assert raw_stub.saved_paths

    raw_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "raw"
        / "lfptensorpipe_log.json"
    )
    finish_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "finish"
        / "lfptensorpipe_log.json"
    )
    assert indicator_from_log(raw_log) == "green"
    assert indicator_from_log(finish_log) == "yellow"
    window.close()


def test_mainwindow_raw_plot_autosave_failure_marks_yellow_and_warns(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []

    window, _context, resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("raw",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0
        },
    )
    raw_path = step_paths["raw"]

    class _CanvasStub:
        def __init__(self) -> None:
            self.callback = None

        def mpl_connect(self, event_name, callback):
            assert event_name == "close_event"
            self.callback = callback
            return 1

    class _FigureStub:
        def __init__(self) -> None:
            self.canvas = _CanvasStub()

    class _BrowserStub:
        def __init__(self) -> None:
            self.fig = _FigureStub()

    class _RawFailStub:
        def save(self, path: str, overwrite: bool = False) -> None:
            _ = (path, overwrite)
            raise RuntimeError("save-failed")

    browser_stub = _BrowserStub()
    window._attach_plot_autosave(
        browser=browser_stub,
        raw=_RawFailStub(),
        raw_path=raw_path,
        step="raw",
        title_prefix="Raw",
    )
    assert browser_stub.fig.canvas.callback is not None
    browser_stub.fig.canvas.callback(None)

    assert indicator_from_log(preproc_step_log_path(resolver, "raw")) == "yellow"
    assert warnings and "Auto-save failed" in warnings[-1]
    window.close()


def test_mainwindow_raw_plot_autosave_context_none_and_browser_canvas_fallback(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, _context, _resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path, steps=("raw",), enable_plots=False
    )
    raw_path = step_paths["raw"]

    class _CanvasStub:
        def __init__(self) -> None:
            self.callback = None

        def mpl_connect(self, event_name, callback):
            assert event_name == "close_event"
            self.callback = callback
            return 1

    class _CanvasOnlyBrowser:
        def __init__(self) -> None:
            self.canvas = _CanvasStub()

    class _RawStub:
        def __init__(self) -> None:
            self.saved_paths: list[str] = []

        def save(self, path: str, overwrite: bool = False) -> None:
            _ = overwrite
            self.saved_paths.append(path)

    browser = _CanvasOnlyBrowser()
    raw = _RawStub()
    window._attach_plot_autosave(
        browser=browser,
        raw=raw,
        raw_path=raw_path,
        step="raw",
        title_prefix="Raw",
    )

    assert browser.canvas.callback is not None
    window._current_project = None
    window._current_subject = None
    window._current_record = None
    browser.canvas.callback(None)
    browser.canvas.callback(None)

    assert raw.saved_paths == []
    window.close()


def test_mainwindow_raw_plot_autosave_qobject_figure_signal_fallback_exceptions(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, _context, _resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path, steps=("filter",), enable_plots=False
    )
    raw_path = step_paths["filter"]

    class _CanvasStub:
        def __init__(self) -> None:
            self.callback = None

        def mpl_connect(self, event_name, callback):
            assert event_name == "close_event"
            self.callback = callback
            return 2

    class _FigureWidget(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.canvas = _CanvasStub()

        def __getattribute__(self, name: str):  # noqa: ANN204
            if name == "destroyed":
                return SimpleNamespace(
                    connect=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("destroyed-connect-failed")
                    )
                )
            return super().__getattribute__(name)

    class _BrokenClosedSignal:
        def connect(self, *_args, **_kwargs) -> None:
            raise RuntimeError("gotClosed-connect-failed")

    class _BrowserStub:
        def __init__(self, fig: QWidget) -> None:
            self.fig = fig
            self.gotClosed = _BrokenClosedSignal()

    class _RawStub:
        def __init__(self) -> None:
            self.saved_paths: list[str] = []

        def save(self, path: str, overwrite: bool = False) -> None:
            _ = overwrite
            self.saved_paths.append(path)
            Path(path).write_text("saved", encoding="utf-8")

    fig = _FigureWidget()
    browser = _BrowserStub(fig)
    raw = _RawStub()
    window._attach_plot_autosave(
        browser=browser,
        raw=raw,
        raw_path=raw_path,
        step="filter",
        title_prefix="Filter",
    )

    assert fig.canvas.callback is not None
    fig.canvas.callback(None)
    assert raw.saved_paths
    window.close()


def test_mainwindow_open_mne_raw_plot_success_and_error_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    resize_calls: list[tuple[int, int]] = []
    plot_calls: list[dict[str, object]] = []
    plot_env_values: list[str | None] = []

    class _BrowserStub:
        def __init__(self) -> None:
            self.fig = SimpleNamespace()

        def resize(self, width: int, height: int) -> None:
            resize_calls.append((width, height))

    def _plot(**kwargs):
        plot_calls.append(dict(kwargs))
        plot_env_values.append(os.environ.get("MNE_BROWSE_RAW_SIZE"))
        return _BrowserStub()

    raw_stub = SimpleNamespace(plot=_plot)
    attach_calls: list[dict[str, object]] = []
    state = {"fail": False}

    def _read_raw_fif(_self, *_args, **_kwargs):
        if state["fail"]:
            raise RuntimeError("plot-read-failed")
        return raw_stub

    window, _context, _resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("raw",),
        enable_plots=True,
        window_cls=OverrideMainWindow,
        overrides={
            "_read_raw_fif": _read_raw_fif,
            "_attach_plot_autosave": lambda self, **kwargs: attach_calls.append(
                dict(kwargs)
            ),
        },
    )
    raw_path = step_paths["raw"]

    previous_env = os.environ.get("MNE_BROWSE_RAW_SIZE")
    os.environ["MNE_BROWSE_RAW_SIZE"] = "1.0,1.0"
    window._open_mne_raw_plot(raw_path, "Raw", autosave_step="raw")
    assert attach_calls
    assert attach_calls[-1]["raw_path"] == raw_path
    assert attach_calls[-1]["step"] == "raw"
    assert plot_calls[-1]["block"] is False
    assert plot_calls[-1]["title"] == f"Raw: {raw_path.name}"
    assert plot_env_values[-1] != "1.0,1.0"
    assert resize_calls[-1] == (1200, 800)
    assert os.environ.get("MNE_BROWSE_RAW_SIZE") == "1.0,1.0"

    state["fail"] = True
    window._open_mne_raw_plot(raw_path, "Raw")
    assert "Raw Plot failed: plot-read-failed" in window.statusBar().currentMessage()
    if previous_env is None:
        os.environ.pop("MNE_BROWSE_RAW_SIZE", None)
    else:
        os.environ["MNE_BROWSE_RAW_SIZE"] = previous_env
    window.close()


def test_mainwindow_raw_plot_uses_existing_preproc_raw_without_bootstrap(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(
        parents=True, exist_ok=True
    )
    rawdata_path = (
        project
        / "rawdata"
        / subject
        / "ses-postop"
        / "lfp"
        / record
        / "raw"
        / "raw.fif"
    )
    rawdata_path.parent.mkdir(parents=True, exist_ok=True)
    rawdata_path.write_text("source", encoding="utf-8")

    preproc_raw = preproc_step_raw_path(resolver, "raw")
    preproc_raw.parent.mkdir(parents=True, exist_ok=True)
    preproc_raw.write_text("edited", encoding="utf-8")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
        overrides={
            "_bootstrap_raw_step_from_rawdata_runtime": (
                lambda self, *_args, **_kwargs: fake_bootstrap(_args[0])
            )
        },
    )
    _select_record_context(window, subject=subject, record=record)

    called = {"count": 0}

    def fake_bootstrap(_context):
        _ = _context
        called["count"] += 1
        return True, "bootstrapped"

    window._on_preproc_raw_plot()

    assert called["count"] == 0
    assert preproc_raw.read_text(encoding="utf-8") == "edited"
    assert indicator_from_log(preproc_step_log_path(resolver, "raw")) == "green"
    window.close()


def test_mainwindow_preproc_step_indicators_follow_step_logs(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(
        parents=True, exist_ok=True
    )
    rawdata_path = (
        project
        / "rawdata"
        / subject
        / "ses-postop"
        / "lfp"
        / record
        / "raw"
        / "raw.fif"
    )
    rawdata_path.parent.mkdir(parents=True, exist_ok=True)
    rawdata_path.write_text("source", encoding="utf-8")

    step_states = {
        "raw": True,
        "filter": False,
        "annotations": True,
        "bad_segment_removal": False,
        "ecg_artifact_removal": True,
        "finish": False,
    }
    for step, completed in step_states.items():
        raw_path = preproc_step_raw_path(resolver, step)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(step, encoding="utf-8")
        params: dict[str, Any] | None = None
        if step == "ecg_artifact_removal":
            params = {"method": "svd", "picks": []}
        mark_preproc_step(
            resolver=resolver,
            step=step,
            completed=completed,
            params=params,
            input_path=str(raw_path),
            output_path=str(raw_path),
            message="seed",
        )
    annotations_csv = resolver.preproc_root / "annotations" / "annotations.csv"
    annotations_csv.parent.mkdir(parents=True, exist_ok=True)
    annotations_csv.write_text(
        "description,onset,duration\n",
        encoding="utf-8",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)

    window._refresh_preproc_controls()
    expected_states = {
        "raw": "green",
        "filter": "yellow",
        "annotations": "green",
        "bad_segment_removal": "yellow",
        "ecg_artifact_removal": "green",
        "finish": "yellow",
    }
    expected_titles = {
        "raw": "0. Raw",
        "filter": "1. Filter",
        "annotations": "2. Annotations",
        "bad_segment_removal": "3. Bad Segment Removal",
        "ecg_artifact_removal": "4. ECG Artifact Removal",
        "finish": "5. Finish",
    }
    for step, expected_state in expected_states.items():
        indicator = window._preproc_step_indicators.get(step)
        assert indicator is not None
        panel = indicator.parentWidget()
        assert isinstance(panel, QGroupBox)
        assert panel.title() == expected_titles[step]
        expected_color = main_window_module.INDICATOR_COLORS[expected_state]
        assert f"background-color: {expected_color}" in indicator.styleSheet()
    window.close()


def test_mainwindow_filter_panel_indicator_updates_on_draft_edit_without_stage_regression(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    window, _context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=(
            "raw",
            "filter",
            "annotations",
            "bad_segment_removal",
            "ecg_artifact_removal",
            "finish",
        ),
        window_cls=OverrideMainWindow,
        overrides={
            "_preproc_filter_panel_state_runtime": (
                lambda self, resolver, *, notches, l_freq, h_freq, advance_params: (
                    "green" if str(l_freq).strip() == "1" else "yellow"
                )
            )
        },
    )

    indicator = window._preproc_step_indicators.get("filter")
    assert indicator is not None
    low_edit = window._preproc_filter_low_freq_edit
    assert low_edit is not None

    window._refresh_preproc_controls()
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['green']}"
        in indicator.styleSheet()
    )

    low_edit.setText("2")
    low_edit.textEdited.emit("2")
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['yellow']}"
        in indicator.styleSheet()
    )

    window._refresh_stage_states_from_context()
    assert window._stage_states["preproc"] == "green"

    low_edit.setText("1")
    low_edit.textEdited.emit("1")
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['green']}"
        in indicator.styleSheet()
    )

    window.close()


def test_mainwindow_tensor_metric_indicator_updates_on_draft_edit(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    window, _context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=(
            "raw",
            "filter",
            "annotations",
            "bad_segment_removal",
            "ecg_artifact_removal",
            "finish",
        ),
        window_cls=OverrideMainWindow,
        overrides={
            "_tensor_metric_panel_state_runtime": (
                lambda self, context, *, metric_key, metric_params, mask_edge_effects: (
                    "gray"
                    if metric_key != "raw_power"
                    else (
                        "yellow"
                        if str(metric_params.get("low_freq_hz", "")).strip() == "bad"
                        else (
                            "green"
                            if float(metric_params.get("low_freq_hz", 0.0)) == 1.0
                            else "yellow"
                        )
                    )
                )
            )
        },
    )

    window._set_active_tensor_metric("raw_power")
    window._refresh_tensor_controls()

    indicator = window._tensor_metric_indicators.get("raw_power")
    assert indicator is not None
    low_edit = window._tensor_low_freq_edit
    assert low_edit is not None
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['green']}"
        in indicator.styleSheet()
    )

    low_edit.setText("2")
    low_edit.textEdited.emit("2")
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['yellow']}"
        in indicator.styleSheet()
    )

    low_edit.setText("1")
    low_edit.textEdited.emit("1")
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['green']}"
        in indicator.styleSheet()
    )

    low_edit.setText("bad")
    low_edit.textEdited.emit("bad")
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['yellow']}"
        in indicator.styleSheet()
    )

    low_edit.setText("1")
    low_edit.textEdited.emit("1")
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['green']}"
        in indicator.styleSheet()
    )

    window.close()


def test_mainwindow_preproc_visualization_block_tracks_available_steps(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)

    raw_path = preproc_step_raw_path(resolver, "raw")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 400), dtype=float), info)
    raw.save(str(raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="raw",
        completed=True,
        input_path=str(raw_path),
        output_path=str(raw_path),
        message="raw ready",
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)

    assert window._preproc_viz_step_combo is not None
    assert window._preproc_viz_step_combo.count() >= 1
    assert window._preproc_viz_step_combo.currentData() == "raw"
    assert window._preproc_viz_channels_button is not None
    assert "(2/2)" in window._preproc_viz_channels_button.text()
    assert window._preproc_viz_psd_advance_button is not None
    assert window._preproc_viz_psd_advance_button.isEnabled()
    assert window._preproc_viz_tfr_plot_button is not None
    assert window._preproc_viz_tfr_plot_button.isEnabled()
    assert window._preproc_ecg_channels_button is not None
    assert window._preproc_ecg_method_combo is not None
    assert window._preproc_ecg_method_combo.currentData() == "svd"
    window.close()


def test_mainwindow_preproc_viz_runtime_controls_and_advance_dialogs(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []

    window, context, resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("raw", "filter"),
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0
        },
    )

    combo = window._preproc_viz_step_combo
    assert combo is not None
    raw_idx = combo.findData("raw")
    filter_idx = combo.findData("filter")
    assert raw_idx >= 0
    assert filter_idx >= 0

    window._preproc_viz_last_step = "filter"
    step_paths["raw"].unlink()
    combo.setCurrentIndex(raw_idx)
    window._on_preproc_viz_step_changed(raw_idx)
    assert combo.currentData() == "filter"
    assert any("Falling back" in item for item in warnings)

    window._preproc_viz_available_channels = ()
    window._on_preproc_viz_channels_select()
    assert any("No channels available for selection." in item for item in warnings)

    window._preproc_viz_available_channels = ("CH1", "CH2")
    window._preproc_viz_selected_channels = ("CH1",)
    window._overrides["_run_channel_selector"] = lambda self, **kwargs: None
    window._on_preproc_viz_channels_select()
    assert window._preproc_viz_selected_channels == ("CH1",)
    window._overrides["_run_channel_selector"] = lambda self, **kwargs: ("CH2",)
    window._on_preproc_viz_channels_select()
    assert window._preproc_viz_selected_channels == ("CH2",)
    assert window._preproc_viz_psd_plot_button is not None
    assert window._preproc_viz_psd_plot_button.isEnabled()

    window._refresh_preproc_ecg_channel_state(None)
    assert window._preproc_ecg_available_channels == ()
    assert window._preproc_ecg_selected_channels == ()
    window._refresh_preproc_ecg_channel_state(context)
    assert window._preproc_ecg_available_channels == ()

    bad_segment_raw_path = preproc_step_raw_path(resolver, "bad_segment_removal")
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    mne.io.RawArray(np.zeros((2, 200), dtype=float), info).save(
        str(bad_segment_raw_path), overwrite=True
    )
    window._refresh_preproc_ecg_channel_state(context)
    assert window._preproc_ecg_available_channels == ("CH1", "CH2")

    window._preproc_ecg_available_channels = ()
    window._on_preproc_ecg_channels_select()
    assert any("No channels available for selection." in item for item in warnings)

    window._preproc_ecg_available_channels = ("CH1", "CH2")
    window._preproc_ecg_selected_channels = ("CH1",)
    window._overrides["_run_channel_selector"] = lambda self, **kwargs: ("CH2",)
    window._on_preproc_ecg_channels_select()
    assert window._preproc_ecg_selected_channels == ("CH2",)

    window._overrides["_current_preproc_viz_source"] = lambda self: None
    window._on_preproc_viz_psd_advance()
    window._on_preproc_viz_tfr_advance()
    assert any("Select a valid visualization step first." in item for item in warnings)

    class _RejectedDialog:
        selected_params = None
        selected_action = None

        def __init__(
            self,
            *,
            mode: str,
            session_params: dict[str, object],
            default_params: dict[str, object],
            set_default_callback=None,  # noqa: ANN001
            parent: MainWindow,
        ) -> None:
            _ = (mode, session_params, default_params, set_default_callback, parent)

        def exec(self) -> int:
            return QDialog.Rejected

    window._overrides["_create_qc_advance_dialog"] = (
        lambda self, **kwargs: _RejectedDialog(**kwargs)
    )
    window._overrides["_current_preproc_viz_source"] = lambda self: (
        "filter",
        step_paths["filter"],
    )
    psd_before = dict(window._preproc_viz_psd_params)
    window._on_preproc_viz_psd_advance()
    assert window._preproc_viz_psd_params == psd_before

    captured_psd_defaults: dict[str, object] = {}
    window._overrides["_save_preproc_viz_psd_defaults"] = (
        lambda self, params: captured_psd_defaults.update(params)
    )

    class _AcceptedPsdDialog:
        def __init__(
            self,
            *,
            mode: str,
            session_params: dict[str, object],
            default_params: dict[str, object],
            set_default_callback=None,  # noqa: ANN001
            parent: MainWindow,
        ) -> None:
            _ = (session_params, default_params, parent)
            self._set_default_callback = set_default_callback
            assert mode == "psd"
            self.selected_action = "set_default"
            self.selected_params = {
                "fmin": 3.0,
                "fmax": 30.0,
                "n_fft": 128,
                "average": False,
            }

        def exec(self) -> int:
            if self._set_default_callback is not None:
                self._set_default_callback(dict(self.selected_params))
            return QDialog.Accepted

    window._overrides["_create_qc_advance_dialog"] = (
        lambda self, **kwargs: _AcceptedPsdDialog(**kwargs)
    )
    window._on_preproc_viz_psd_advance()
    assert captured_psd_defaults["fmin"] == 3.0
    assert window._preproc_viz_psd_params["average"] is False
    assert "session parameters updated" in window.statusBar().currentMessage().lower()

    class _AcceptedTfrDialog:
        def __init__(
            self,
            *,
            mode: str,
            session_params: dict[str, object],
            default_params: dict[str, object],
            set_default_callback=None,  # noqa: ANN001
            parent: MainWindow,
        ) -> None:
            _ = (session_params, default_params, set_default_callback, parent)
            assert mode == "tfr"
            self.selected_action = "save"
            self.selected_params = {
                "fmin": 2.0,
                "fmax": 40.0,
                "n_freqs": 20,
                "decim": 2,
            }

        def exec(self) -> int:
            return QDialog.Accepted

    window._overrides["_create_qc_advance_dialog"] = (
        lambda self, **kwargs: _AcceptedTfrDialog(**kwargs)
    )
    window._on_preproc_viz_tfr_advance()
    assert window._preproc_viz_tfr_params["decim"] == 2
    assert "session parameters updated" in window.statusBar().currentMessage().lower()

    window.close()


def test_mainwindow_preproc_viz_non_handler_source_helpers(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("raw", "filter"),
        enable_plots=False,
    )

    combo = window._preproc_viz_step_combo
    assert combo is not None

    original_combo = window._preproc_viz_step_combo
    window._preproc_viz_step_combo = None
    assert window._current_preproc_viz_step() is None
    window._refresh_preproc_visualization_controls(context)
    window._preproc_viz_step_combo = original_combo

    assert window._preproc_viz_step_combo is not None
    window._preproc_viz_step_combo.blockSignals(True)
    window._preproc_viz_step_combo.clear()
    window._preproc_viz_step_combo.addItem("Invalid", 123)
    window._preproc_viz_step_combo.setCurrentIndex(0)
    window._preproc_viz_step_combo.blockSignals(False)
    assert window._current_preproc_viz_step() is None
    assert window._current_preproc_viz_source() is None

    window._refresh_preproc_visualization_controls(context)
    source = window._current_preproc_viz_source()
    assert source is not None
    _, selected_path = source
    assert selected_path.exists()

    selected_path.unlink()
    assert window._current_preproc_viz_source() is None

    window._current_project = None
    assert window._current_preproc_viz_source() is None
    window.close()


def test_mainwindow_preproc_viz_plot_runtime_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []

    window, _context, _resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("raw",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0
        },
    )

    window._overrides["_current_preproc_viz_source"] = lambda self: None
    window._on_preproc_viz_psd_plot()
    window._on_preproc_viz_tfr_plot()
    assert any("No valid visualization source" in item for item in warnings)

    window._overrides["_current_preproc_viz_source"] = lambda self: (
        "raw",
        step_paths["raw"],
    )
    window._preproc_viz_selected_channels = ()
    window._on_preproc_viz_psd_plot()
    window._on_preproc_viz_tfr_plot()
    assert any("Select at least one channel" in item for item in warnings)

    warning_count = len(warnings)
    window._preproc_viz_selected_channels = ("CH1",)
    window._enable_plots = False
    window._on_preproc_viz_psd_plot()
    window._on_preproc_viz_tfr_plot()
    assert len(warnings) == warning_count

    window._enable_plots = True

    class _SpectrumStub:
        def __init__(self) -> None:
            self.average: bool | None = None

        def plot(self, *, average: bool) -> None:
            self.average = average

    class _RawPsdStub:
        def __init__(self) -> None:
            self.closed = False
            self.last_picks: list[str] | None = None
            self.spectrum = _SpectrumStub()

        def compute_psd(self, **kwargs):
            self.last_picks = list(kwargs.get("picks", []))
            return self.spectrum

        def close(self) -> None:
            self.closed = True

    raw_psd = _RawPsdStub()
    window._overrides["_read_raw_fif"] = lambda self, *_args, **_kwargs: raw_psd
    window._preproc_viz_psd_params = {
        "fmin": 1.0,
        "fmax": 40.0,
        "n_fft": 128,
        "average": True,
    }
    window._on_preproc_viz_psd_plot()
    assert raw_psd.last_picks == ["CH1"]
    assert raw_psd.spectrum.average is True
    assert raw_psd.closed

    def _raise_psd(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("psd failed")

    window._overrides["_read_raw_fif"] = lambda self, *_args, **_kwargs: _raise_psd()
    window._on_preproc_viz_psd_plot()
    assert any("PSD plot failed" in item for item in warnings)

    class _ImageStub:
        pass

    class _AxesStub:
        def __init__(self) -> None:
            self.imshow_called = False
            self.imshow_kwargs: dict[str, object] = {}
            self.yscale: str | None = None

        def imshow(self, *args, **kwargs):
            _ = (args, kwargs)
            self.imshow_called = True
            self.imshow_kwargs = dict(kwargs)
            return _ImageStub()

        def set_xlabel(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def set_ylabel(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def set_yscale(self, value: str) -> None:
            self.yscale = value

        def set_title(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

    class _FigureStub:
        def __init__(self) -> None:
            self.colorbar_label: str | None = None
            self.tight_layout_called = False
            self.show_called = False

        def colorbar(self, image, ax=None, label: str | None = None) -> None:
            _ = (image, ax)
            self.colorbar_label = label

        def tight_layout(self) -> None:
            self.tight_layout_called = True

        def show(self) -> None:
            self.show_called = True

    fig = _FigureStub()
    ax = _AxesStub()
    window._overrides["_create_matplotlib_subplots"] = lambda self: (fig, ax)
    window._overrides["_compute_tfr_array_morlet"] = (
        lambda self, *args, **kwargs: np.ones((1, 1, 3, 5), dtype=float)
    )

    class _RawTfrStub:
        def __init__(self) -> None:
            self.info = {"sfreq": 100.0}
            self.n_times = 500
            self.closed = False

        def get_data(self, picks, start: int, stop: int):
            _ = (picks, start, stop)
            return np.ones((1, 200), dtype=float)

        def close(self) -> None:
            self.closed = True

    raw_tfr = _RawTfrStub()
    window._overrides["_read_raw_fif"] = lambda self, *_args, **_kwargs: raw_tfr
    window._preproc_viz_tfr_params = {
        "fmin": 2.0,
        "fmax": 20.0,
        "n_freqs": 3,
        "decim": 2,
    }
    window._on_preproc_viz_tfr_plot()
    assert ax.imshow_called
    assert "norm" in ax.imshow_kwargs
    assert ax.yscale == "log"
    assert fig.colorbar_label == "Power (log scale)"
    assert fig.tight_layout_called
    assert fig.show_called
    assert raw_tfr.closed

    class _RawTfrEmptyStub(_RawTfrStub):
        def get_data(self, picks, start: int, stop: int):
            _ = (picks, start, stop)
            return np.zeros((0, 0), dtype=float)

    window._overrides["_read_raw_fif"] = (
        lambda self, *_args, **_kwargs: _RawTfrEmptyStub()
    )
    window._on_preproc_viz_tfr_plot()
    assert any("TFR plot failed" in item for item in warnings)

    window.close()


def test_mainwindow_tensor_run_uses_selected_metrics(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)

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

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)
    app.processEvents()

    assert window._stage_buttons["tensor"].isEnabled()
    assert window._tensor_run_button is not None
    assert not window._tensor_run_button.isEnabled()
    assert "raw_power" in window._tensor_metric_checks
    assert "periodic_aperiodic" in window._tensor_metric_checks
    assert "coherence" in window._tensor_metric_checks
    assert "plv" in window._tensor_metric_checks
    assert "trgc" in window._tensor_metric_checks
    assert "psi" in window._tensor_metric_checks
    assert "burst" in window._tensor_metric_checks
    assert window._tensor_metric_checks["raw_power"].isEnabled()
    assert window._tensor_metric_checks["periodic_aperiodic"].isEnabled()
    assert window._tensor_metric_checks["coherence"].isEnabled()
    assert window._tensor_metric_checks["plv"].isEnabled()
    assert window._tensor_metric_checks["trgc"].isEnabled()
    assert window._tensor_metric_checks["psi"].isEnabled()
    assert window._tensor_metric_checks["burst"].isEnabled()
    assert window._tensor_channels_button is not None
    assert window._tensor_channels_button.text().endswith("/2)")

    window._tensor_metric_checks["raw_power"].setChecked(True)
    window._tensor_metric_checks["periodic_aperiodic"].setChecked(True)
    app.processEvents()
    assert window._tensor_run_button.isEnabled()

    captured: dict[str, object] = {}

    def fake_launch_tensor_run_process(
        *,
        context,
        selected_metrics,
        metric_params_map,
        mask_edge_effects,
    ):
        captured["ctx"] = context
        captured["selected_metrics"] = selected_metrics
        captured["metric_params_map"] = metric_params_map
        captured["mask_edge_effects"] = mask_edge_effects

    window._launch_tensor_run_process = fake_launch_tensor_run_process  # type: ignore[method-assign]
    window._on_tensor_run()

    assert captured["ctx"] == context
    assert captured["selected_metrics"] == ["raw_power", "periodic_aperiodic"]
    metric_params_map = captured["metric_params_map"]
    assert isinstance(metric_params_map, dict)
    raw_power_params = metric_params_map["raw_power"]
    assert float(raw_power_params["low_freq_hz"]) > 0.0
    assert float(raw_power_params["high_freq_hz"]) > float(
        raw_power_params["low_freq_hz"]
    )
    assert float(raw_power_params["freq_step_hz"]) > 0.0
    assert raw_power_params["selected_channels"] == list(
        window._tensor_selected_channels_by_metric.get("raw_power", ())
    )
    window.close()


def test_mainwindow_tensor_run_passes_selected_pairs_for_connectivity(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)

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

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)
    app.processEvents()

    assert window._stage_buttons["tensor"].isEnabled()
    assert window._tensor_metric_checks["coherence"].isEnabled()
    window._tensor_metric_checks["coherence"].setChecked(True)
    window._tensor_selected_pairs_by_metric["coherence"] = (
        ("CH1", "CH2"),
        ("CH2", "CH3"),
    )
    app.processEvents()
    assert window._tensor_run_button is not None
    assert window._tensor_run_button.isEnabled()

    captured: dict[str, object] = {}

    def fake_launch_tensor_run_process(
        *,
        context,
        selected_metrics,
        metric_params_map,
        mask_edge_effects,
    ):
        _ = mask_edge_effects
        captured["ctx"] = context
        captured["selected_metrics"] = selected_metrics
        captured["metric_params_map"] = metric_params_map

    window._launch_tensor_run_process = fake_launch_tensor_run_process  # type: ignore[method-assign]
    window._on_tensor_run()

    assert captured["ctx"] == context
    assert captured["selected_metrics"] == ["coherence"]
    metric_params_map = captured["metric_params_map"]
    assert metric_params_map["coherence"]["selected_pairs"] == [
        ["CH1", "CH2"],
        ["CH2", "CH3"],
    ]
    window.close()


def test_mainwindow_tensor_run_busy_message_warns_for_trgc(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)

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

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)
    app.processEvents()

    class _FakeProcess:
        def __init__(self) -> None:
            self._alive = True

        def poll(self) -> int | None:
            return None if self._alive else 0

        def terminate(self) -> None:
            self._alive = False

        def kill(self) -> None:
            self._alive = False

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            self._alive = False
            return 0

    def _fake_launch_tensor_run_process(
        *,
        context,
        selected_metrics,
        metric_params_map,
        mask_edge_effects,
    ) -> None:
        _ = (context, selected_metrics, metric_params_map, mask_edge_effects)
        window._tensor_run_process = _FakeProcess()
        window._tensor_run_active_metrics = ["trgc"]
        window._tensor_run_busy_label = "Build Tensor"
        window._start_busy(
            "Build Tensor",
            suffix="This may take several hours.",
        )
        window._set_busy_ui_lock(True)

    window._tensor_metric_checks["trgc"].setChecked(True)
    app.processEvents()
    window._launch_tensor_run_process = _fake_launch_tensor_run_process  # type: ignore[method-assign]
    window._on_tensor_run()

    message = window.statusBar().currentMessage()
    assert "Build Tensor | Busy" in message
    assert "This may take several hours." in message

    window._shutdown_tensor_run()
    window.close()


def test_mainwindow_alignment_run_uses_selected_paradigm(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)

    import mne

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 800), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.5, 2.0, 3.0],
            duration=[0.6, 0.7, 0.8],
            description=["event", "event", "event"],
        )
    )
    raw.save(str(finish_raw_path), overwrite=True)
    mark_preproc_step(
        resolver=resolver,
        step="finish",
        completed=True,
        input_path=str(finish_raw_path),
        output_path=str(finish_raw_path),
        message="finish ready",
    )

    tensor_path = tensor_metric_tensor_path(resolver, "raw_power")
    tensor = np.ones((1, 2, 4, 60), dtype=float)
    save_pkl(
        {
            "tensor": tensor,
            "meta": {
                "axes": {
                    "channel": np.array(["CH1", "CH2"], dtype=object),
                    "freq": np.linspace(2.0, 12.0, 4, dtype=float),
                    "time": np.linspace(0.0, 3.0, 60, dtype=float),
                    "shape": tensor.shape,
                },
                "params": {},
            },
        },
        tensor_path,
    )
    write_run_log(
        tensor_metric_log_path(resolver, "raw_power"),
        RunLogRecord(
            step="raw_power",
            completed=True,
            params={},
            input_path="in",
            output_path=str(tensor_path),
            message="tensor ready",
        ),
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    created, _, entry = create_alignment_paradigm(store, name="Gait", context=context)
    assert created and isinstance(entry, dict)
    slug = str(entry["slug"])
    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
        method_params={
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.4,
        },
        context=context,
    )
    assert updated, message

    captured: dict[str, object] = {}

    def fake_run_align_epochs(ctx, *, config_store, paradigm_slug):
        captured["ctx"] = ctx
        captured["store"] = config_store
        captured["slug"] = paradigm_slug
        return (
            True,
            "ok",
            [
                {
                    "epoch_index": 0,
                    "epoch_label": "event",
                    "start_t": 0.5,
                    "end_t": 1.1,
                    "pick": True,
                }
            ],
        )

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
        overrides={
            "_run_align_epochs_runtime": lambda self, *args, **kwargs: fake_run_align_epochs(
                *args, **kwargs
            )
        },
    )
    _select_record_context(window, subject=subject, record=record)
    app.processEvents()

    assert window._stage_buttons["alignment"].isEnabled()
    assert window._alignment_paradigm_list is not None
    assert window._alignment_paradigm_list.count() == 1
    assert window._alignment_run_button is not None
    assert window._alignment_run_button.isEnabled()

    window._on_alignment_run()

    assert captured["ctx"] == context
    assert captured["store"] == store
    assert captured["slug"] == slug
    assert window._alignment_epoch_table is not None
    assert window._alignment_epoch_table.rowCount() == 1
    assert window._alignment_finish_button is not None
    assert not window._alignment_finish_button.isEnabled()
    window.close()


def test_mainwindow_alignment_runtime_handler_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []

    window, context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=True,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0
        },
    )

    assert window._alignment_paradigm_list is not None
    assert window._alignment_epoch_metric_combo is not None
    assert window._alignment_epoch_channel_combo is not None
    assert window._alignment_epoch_table is not None
    window._alignment_paradigm_list.setCurrentRow(0)
    window._stage_states["tensor"] = "green"

    state: dict[str, object] = {
        "validate_result": (
            True,
            {
                "annotations": ["event"],
                "mode": "exact",
                "duration_range": [0.0, 100.0],
                "drop_bad": False,
                "pad_s": 0.0,
                "sample_rate": 0.4,
            },
            "",
        ),
        "update_result": (True, "updated"),
        "run_result": (
            True,
            "runtime ok",
            [
                {
                    "epoch_index": 0,
                    "epoch_label": "event",
                    "duration_s": 0.8,
                    "start_t": 0.5,
                    "end_t": 1.3,
                    "pick": True,
                }
            ],
        ),
        "finish_result": (True, "finish ok"),
        "method_state": "yellow",
        "preview_payload": {},
    }

    window._overrides["_load_alignment_annotation_labels_runtime"] = (
        lambda self, *_args, **_kwargs: ["event"]
    )
    window._overrides["_validate_alignment_method_params_runtime"] = (
        lambda self, *_args, **_kwargs: state["validate_result"]
    )
    window._overrides["_update_alignment_paradigm_runtime"] = (
        lambda self, *_args, **_kwargs: state["update_result"]
    )
    window._overrides["_run_align_epochs_runtime"] = (
        lambda self, *_args, **_kwargs: state["run_result"]
    )
    window._overrides["_finish_alignment_epochs_runtime"] = (
        lambda self, *_args, **_kwargs: state["finish_result"]
    )
    window._overrides["_alignment_method_panel_state_runtime"] = (
        lambda self, *_args, **_kwargs: state["method_state"]
    )
    window._overrides["_run_with_busy"] = lambda self, _label, work: work()

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_alignment_run()
    assert "select project/subject/record" in window.statusBar().currentMessage()

    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record

    window._stage_states["tensor"] = "yellow"
    window._on_alignment_run()
    assert any("Build Tensor must be green" in item for item in warnings)

    window._stage_states["tensor"] = "green"
    alignment_list = window._alignment_paradigm_list
    assert alignment_list is not None
    window._set_shared_stage_trial_slug(None)
    window._alignment_paradigm_list = None
    window._on_alignment_run()
    assert any("Select one trial first." in item for item in warnings)
    window._alignment_paradigm_list = alignment_list
    window._alignment_paradigm_list.setCurrentRow(0)

    state["validate_result"] = (False, {}, "validate failed")
    window._on_alignment_run()
    assert any("validate failed" in item for item in warnings)

    state["validate_result"] = (
        True,
        {
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.4,
        },
        "",
    )
    state["update_result"] = (False, "update failed")
    window._on_alignment_run()
    assert any("update failed" in item for item in warnings)

    state["update_result"] = (True, "updated")
    state["run_result"] = (False, "runtime failed", [])
    window._on_alignment_run()
    assert "Align Epochs failed: runtime failed" in window.statusBar().currentMessage()

    window._stage_states["tensor"] = "green"
    state["run_result"] = (
        True,
        "runtime ok",
        [
            {
                "epoch_index": 0,
                "epoch_label": "event",
                "duration_s": 0.8,
                "start_t": 0.5,
                "end_t": 1.3,
                "pick": True,
            }
        ],
    )
    window._on_alignment_run()
    assert window._alignment_epoch_table.rowCount() == 1
    assert window._alignment_epoch_table.item(0, 2).text() == "0.800"
    assert window._alignment_epoch_table.item(0, 3).text() == "0.500"
    assert window._alignment_epoch_table.item(0, 4).text() == "1.300"
    assert "Align Epochs OK: runtime ok" in window.statusBar().currentMessage()

    window._set_alignment_epoch_rows(
        [
            {
                "epoch_index": 0,
                "epoch_label": "missing",
                "duration_s": np.nan,
                "start_t": np.nan,
                "end_t": np.nan,
                "pick": True,
            }
        ]
    )
    assert window._alignment_epoch_table.item(0, 2).text() == ""
    assert window._alignment_epoch_table.item(0, 3).text() == ""
    assert window._alignment_epoch_table.item(0, 4).text() == ""

    window._set_alignment_epoch_rows(
        [
            {
                "epoch_index": 0,
                "epoch_label": "event",
                "start_t": 0.1,
                "end_t": 0.9,
                "pick": False,
            },
            {
                "epoch_index": 1,
                "epoch_label": "event",
                "start_t": 1.0,
                "end_t": 1.8,
                "pick": False,
            },
        ]
    )
    assert window._alignment_select_all_button.text() == "Select All"
    window._on_alignment_select_all()
    assert window._alignment_select_all_button.text() == "Deselect All"
    for row_idx in range(window._alignment_epoch_table.rowCount()):
        item = window._alignment_epoch_table.item(row_idx, 0)
        assert item is not None
        assert item.checkState() == Qt.Checked
    window._on_alignment_select_all()
    assert window._alignment_select_all_button.text() == "Select All"
    for row_idx in range(window._alignment_epoch_table.rowCount()):
        item = window._alignment_epoch_table.item(row_idx, 0)
        assert item is not None
        assert item.checkState() == Qt.Unchecked
    first_item = window._alignment_epoch_table.item(0, 0)
    second_item = window._alignment_epoch_table.item(1, 0)
    assert first_item is not None
    assert second_item is not None
    first_item.setCheckState(Qt.Checked)
    assert window._alignment_select_all_button.text() == "Select All"
    second_item.setCheckState(Qt.Checked)
    assert window._alignment_select_all_button.text() == "Deselect All"
    second_item.setCheckState(Qt.Unchecked)
    assert window._alignment_select_all_button.text() == "Select All"

    epoch_table = window._alignment_epoch_table
    window._alignment_epoch_table = None
    window._on_alignment_select_all()
    window._alignment_epoch_table = epoch_table

    window._enable_plots = False
    window._on_alignment_preview()
    window._enable_plots = True

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_alignment_preview()
    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record

    window._alignment_epoch_metric_combo.clear()
    window._alignment_epoch_metric_combo.addItem("No metric", None)
    window._on_alignment_preview()

    window._alignment_epoch_metric_combo.clear()
    window._alignment_epoch_metric_combo.addItem("raw_power", "raw_power")
    window._on_alignment_preview()
    assert "missing tensor_warped.pkl" in window.statusBar().currentMessage()

    preview_path = main_window_module.alignment_metric_tensor_warped_path(
        resolver, slug, "raw_power"
    )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl({"placeholder": True}, preview_path)
    window._overrides["_load_pickle"] = lambda self, *_args, **_kwargs: state[
        "preview_payload"
    ]

    state["preview_payload"] = "bad-payload"
    window._on_alignment_preview()
    assert any("Preview failed" in item for item in warnings)

    state["preview_payload"] = {
        "tensor": np.zeros((1, 2, 3), dtype=float),
        "meta": {"axes": {"freq": np.array([4.0, 8.0]), "time": np.array([0.0, 1.0])}},
    }
    window._on_alignment_preview()

    state["preview_payload"] = {
        "tensor": np.zeros((2, 2, 3, 4), dtype=float),
        "meta": {"axes": {"freq": np.array([4.0, 8.0, 12.0]), "time": np.arange(4.0)}},
    }
    window._set_alignment_epoch_rows(
        [
            {
                "epoch_index": 0,
                "epoch_label": "event",
                "start_t": 0.1,
                "end_t": 0.9,
                "pick": False,
            }
        ]
    )
    window._on_alignment_preview()
    assert any("No picked epochs available for preview." in item for item in warnings)

    state["preview_payload"] = {
        "tensor": np.zeros((2, 2, 3, 4), dtype=float),
        "meta": {"axes": {}},
    }
    window._set_alignment_epoch_rows(
        [
            {
                "epoch_index": 0,
                "epoch_label": "event",
                "start_t": 0.1,
                "end_t": 0.9,
                "pick": True,
            }
        ]
    )
    window._on_alignment_preview()

    state["preview_payload"] = {
        "tensor": np.ones((2, 2, 3, 4), dtype=float),
        "meta": {
            "axes": {
                "freq": np.array([4.0, 8.0, 12.0], dtype=float),
                "time": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            }
        },
    }
    plotted: list[dict[str, object]] = []
    window._config_store.write_yaml(
        "alignment_preview.yml",
        {
            "metric_defaults": {
                "raw_power": {
                    "boxsize": [72.0, 58.0],
                    "font_size": 18,
                    "tick_label_size": 11,
                    "x_label_offset_mm": 8.0,
                    "y_label_offset_mm": 14.0,
                    "colorbar_pad_mm": 3.0,
                    "cbar_label_offset_mm": 12.0,
                    "colorbar_label": "Raw power",
                }
            }
        },
    )

    class _PreviewFigure:
        def show(self) -> None:
            plotted.append({"shown": True})

    def _fake_plot_single_effect_df(*args, **kwargs):  # noqa: ANN002, ANN003
        plot_df = args[0] if args else None
        plotted.append(
            {
                "plot_df": plot_df,
                "value_col": kwargs.get("value_col"),
                "x_label": kwargs.get("x_label"),
                "y_label": kwargs.get("y_label"),
                "y_log": kwargs.get("y_log"),
                "title": kwargs.get("title"),
                "cmap": kwargs.get("cmap"),
                "boxsize": kwargs.get("boxsize"),
                "axis_label_fontsize": kwargs.get("axis_label_fontsize"),
                "tick_label_fontsize": kwargs.get("tick_label_fontsize"),
                "x_label_offset_mm": kwargs.get("x_label_offset_mm"),
                "y_label_offset_mm": kwargs.get("y_label_offset_mm"),
                "colorbar_pad_mm": kwargs.get("colorbar_pad_mm"),
                "cbar_label_offset_mm": kwargs.get("cbar_label_offset_mm"),
                "colorbar_label": kwargs.get("colorbar_label"),
            }
        )
        return _PreviewFigure()

    window._overrides["_plot_single_effect_df"] = (
        lambda self, *args, **kwargs: _fake_plot_single_effect_df(*args, **kwargs)
    )
    window._overrides["_tighten_alignment_preview_figure"] = (
        lambda self, fig: plotted.append({"tightened": fig is not None})
    )
    window._on_alignment_preview()
    assert any(item.get("y_log") is True for item in plotted if "y_log" in item)
    assert any(item.get("shown") is True for item in plotted)
    assert any(item.get("tightened") is True for item in plotted)
    transformed_plot_df = next(
        item.get("plot_df")
        for item in plotted
        if isinstance(item.get("plot_df"), pd.DataFrame)
    )
    transformed_nested = transformed_plot_df.loc[0, "Value"]
    assert isinstance(transformed_nested, pd.DataFrame)
    np.testing.assert_allclose(
        transformed_nested.to_numpy(dtype=float),
        np.zeros((3, 4), dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )
    assert any(
        item.get("x_label") == "Percent (%)"
        and item.get("y_label") == "Frequency"
        and item.get("cmap") == "viridis"
        and item.get("title") is None
        and item.get("boxsize") == (72.0, 58.0)
        and item.get("axis_label_fontsize") == 18
        and item.get("tick_label_fontsize") == 11
        and item.get("x_label_offset_mm") == 8.0
        and item.get("y_label_offset_mm") == 14.0
        and item.get("colorbar_pad_mm") == 3.0
        and item.get("cbar_label_offset_mm") == 12.0
        and item.get("colorbar_label") == "Raw power"
        for item in plotted
        if "x_label" in item
    )

    state["preview_payload"] = {
        "tensor": np.ones((2, 2, 3, 4), dtype=float),
        "meta": {
            "axes": {
                "freq": np.array(["a", "b", "c"], dtype=object),
                "time": np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
            }
        },
    }
    window._on_alignment_preview()
    assert any(item.get("y_log") is False for item in plotted if "y_log" in item)

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_alignment_finish()
    assert (
        "Finish unavailable: select context and trial."
        in window.statusBar().currentMessage()
    )

    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record
    window._alignment_paradigm_list.setCurrentRow(0)
    window._on_alignment_finish()
    assert any(
        "Run Align Epochs successfully before Finish." in item for item in warnings
    )

    _seed_alignment_trial_finish_ready(resolver, slug, metrics=("raw_power", "psi"))
    state["method_state"] = "green"
    state["finish_result"] = (False, "finish failed")
    window._on_alignment_finish()
    assert "Finish failed: finish failed" in window.statusBar().currentMessage()

    state["finish_result"] = (True, "finish ok")
    window._on_alignment_finish()
    assert "Finish OK: finish ok" in window.statusBar().currentMessage()

    window.close()


def test_plot_tightening_helpers_reanchor_figure_legends() -> None:
    import matplotlib.pyplot as plt

    from lfptensorpipe.gui.shell.alignment_run import MainWindowAlignmentRunMixin
    from lfptensorpipe.gui.shell.features_plotting import (
        MainWindowFeaturesPlottingMixin,
    )

    def _exercise_helper(helper: Any) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=100)
        ax.plot([0.0, 1.0], [0.0, 1.0], label="line")
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
        fig.canvas.draw()
        old_size = tuple(float(v) for v in fig.get_size_inches())
        before_anchor = legend.get_bbox_to_anchor().transformed(
            fig.transFigure.inverted()
        )

        helper(fig)
        fig.canvas.draw()
        new_size = tuple(float(v) for v in fig.get_size_inches())
        after_anchor = legend.get_bbox_to_anchor().transformed(
            fig.transFigure.inverted()
        )

        assert len(fig.legends) == 1
        assert new_size != pytest.approx(old_size)
        assert after_anchor.x0 < before_anchor.x0
        plt.close(fig)

    _exercise_helper(MainWindowAlignmentRunMixin()._tighten_alignment_preview_figure)
    _exercise_helper(MainWindowFeaturesPlottingMixin()._tighten_features_plot_figure)


def test_mainwindow_alignment_paradigm_and_method_handler_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []
    reload_calls: list[str | None] = []
    state: dict[str, object] = {
        "input_text": ("New Paradigm", True),
        "create_result": (False, "create failed", None),
        "confirm_result": int(main_window_module.QMessageBox.No),
        "delete_result": (False, "delete failed"),
        "validate_result": (
            True,
            {
                "annotations": ["event"],
                "mode": "exact",
                "duration_range": [0.0, 100.0],
                "drop_bad": False,
                "pad_s": 0.0,
                "sample_rate": 0.4,
            },
            "",
        ),
        "update_result": (True, "update ok"),
        "dialog_exec_result": QDialog.Rejected,
        "dialog_params": None,
    }

    class _FakeParamsDialog:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            _ = (args, kwargs)
            params = state["dialog_params"]
            self.selected_params = dict(params) if isinstance(params, dict) else None

        def exec(self) -> int:
            return int(state["dialog_exec_result"])

    window, _context, _resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0,
            "_ask_question": lambda self, title, message, **kwargs: state[
                "confirm_result"
            ],
            "_prompt_text": lambda self, title, label: tuple(state["input_text"]),
            "_create_alignment_paradigm_runtime": (
                lambda self, *_args, **_kwargs: state["create_result"]
            ),
            "_delete_alignment_paradigm_runtime": (
                lambda self, *_args, **_kwargs: state["delete_result"]
            ),
            "_load_alignment_annotation_labels_runtime": (
                lambda self, *_args, **_kwargs: ["event"]
            ),
            "_validate_alignment_method_params_runtime": (
                lambda self, *_args, **_kwargs: state["validate_result"]
            ),
            "_update_alignment_paradigm_runtime": (
                lambda self, *_args, **_kwargs: state["update_result"]
            ),
            "_create_alignment_method_params_dialog": (
                lambda self, **kwargs: _FakeParamsDialog(**kwargs)
            ),
        },
    )

    def _tracked_reload_alignment_paradigms(
        self,
        preferred_slug: str | None = None,
    ) -> None:
        reload_calls.append(preferred_slug)
        MainWindow._reload_alignment_paradigms(self, preferred_slug)

    window._overrides["_reload_alignment_paradigms"] = (
        _tracked_reload_alignment_paradigms
    )
    assert window._alignment_paradigm_list is not None
    assert window._alignment_method_combo is not None
    window._alignment_paradigm_list.setCurrentRow(0)

    state["input_text"] = ("Skipped", False)
    window._on_alignment_paradigm_add()

    state["input_text"] = ("Fail", True)
    state["create_result"] = (False, "create failed", None)
    window._on_alignment_paradigm_add()
    assert any("create failed" in item for item in warnings)

    state["create_result"] = (
        True,
        "created ok",
        {"slug": "paradigm-created"},
    )
    window._on_alignment_paradigm_add()
    assert "created ok" in window.statusBar().currentMessage()
    assert reload_calls[-1] == "paradigm-created"

    window._alignment_paradigm_list.setCurrentRow(-1)
    window._on_alignment_paradigm_delete()
    window._alignment_paradigm_list.setCurrentRow(0)

    state["confirm_result"] = int(main_window_module.QMessageBox.No)
    window._on_alignment_paradigm_delete()

    state["confirm_result"] = int(main_window_module.QMessageBox.Yes)
    state["delete_result"] = (False, "delete failed")
    window._on_alignment_paradigm_delete()
    assert any("delete failed" in item for item in warnings)

    state["delete_result"] = (True, "delete ok")
    window._on_alignment_paradigm_delete()
    assert "delete ok" in window.statusBar().currentMessage()

    window._alignment_paradigm_list.setCurrentRow(-1)
    window._on_alignment_method_changed(0)
    window._alignment_paradigm_list.setCurrentRow(0)

    invalid_idx = window._alignment_method_combo.findData(None)
    if invalid_idx < 0:
        window._alignment_method_combo.addItem("Invalid", None)
        invalid_idx = window._alignment_method_combo.count() - 1
    window._alignment_method_combo.setCurrentIndex(invalid_idx)
    window._on_alignment_method_changed(invalid_idx)

    stack_idx = window._alignment_method_combo.findData("stack_warper")
    if stack_idx >= 0:
        window._alignment_method_combo.setCurrentIndex(stack_idx)
    state["validate_result"] = (False, {}, "validate failed")
    window._on_alignment_method_changed(0)
    assert any("validate failed" in item for item in warnings)

    state["validate_result"] = (
        True,
        {
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.4,
        },
        "",
    )
    state["update_result"] = (False, "update failed")
    window._on_alignment_method_changed(0)
    assert any("update failed" in item for item in warnings)

    state["update_result"] = (True, "update ok")
    window._on_alignment_method_changed(0)
    assert reload_calls[-1] == slug

    window._alignment_paradigm_list.setCurrentRow(-1)
    window._on_alignment_method_params()
    window._alignment_paradigm_list.setCurrentRow(0)

    method_combo = window._alignment_method_combo
    window._alignment_method_combo = None
    window._on_alignment_method_params()
    window._alignment_method_combo = method_combo

    window._alignment_method_combo.setCurrentIndex(invalid_idx)
    window._on_alignment_method_params()

    if stack_idx >= 0:
        window._alignment_method_combo.setCurrentIndex(stack_idx)

    state["dialog_exec_result"] = QDialog.Rejected
    state["dialog_params"] = {"sample_rate": 0.55}
    window._on_alignment_method_params()

    state["dialog_exec_result"] = QDialog.Accepted
    state["dialog_params"] = None
    window._on_alignment_method_params()

    state["dialog_exec_result"] = QDialog.Accepted
    state["dialog_params"] = {
        "annotations": ["event"],
        "mode": "exact",
        "duration_range": [0.0, 100.0],
        "drop_bad": False,
        "pad_s": 0.0,
        "sample_rate": 0.88,
    }
    state["update_result"] = (False, "params update failed")
    window._on_alignment_method_params()
    assert any("params update failed" in item for item in warnings)

    state["update_result"] = (True, "params update ok")
    window._on_alignment_method_params()
    assert reload_calls[-1] == slug

    window.close()


def test_mainwindow_alignment_delete_does_not_recreate_trial_directory(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_ask_question": lambda self, title, message, **kwargs: int(
                main_window_module.QMessageBox.Yes
            ),
        },
    )
    assert window._alignment_paradigm_list is not None
    assert window._alignment_paradigm_list.count() == 1
    trial_dir = resolver.alignment_paradigm_dir(slug, create=False)
    assert trial_dir.exists()

    persisted = window._persist_record_params_snapshot(reason="test_delete_trial")
    assert persisted

    window._on_alignment_paradigm_delete()
    window._refresh_stage_states_from_context()
    window._sync_record_params_from_logs(include_master=True, clear_dirty=True)

    assert context.record == "runA"
    assert window._alignment_paradigm_list.count() == 0
    assert not trial_dir.exists()
    window.close()


def test_mainwindow_alignment_config_import_export_handlers(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    warnings: list[str] = []
    infos: list[str] = []
    reload_calls: list[str | None] = []
    persist_reasons: list[str] = []
    update_calls: list[dict[str, object]] = []
    export_path = tmp_path / "alignment-export.json"
    import_path = tmp_path / "alignment-import.json"
    state: dict[str, object] = {
        "save_result": (str(export_path), "JSON files (*.json)"),
        "open_result": (str(import_path), "JSON files (*.json)"),
        "update_result": (True, "import ok"),
        "persist_ok": True,
    }

    window, _context, _resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0,
            "_show_information": (
                lambda self, title, message: infos.append(message) or 0
            ),
            "_save_file_name": lambda self, *_args, **_kwargs: state["save_result"],
            "_open_file_name": lambda self, *_args, **_kwargs: state["open_result"],
            "_update_alignment_paradigm_runtime": (
                lambda self, _config_store, **kwargs: update_calls.append(kwargs)
                or state["update_result"]
            ),
            "_persist_record_params_snapshot": (
                lambda self, *, reason: persist_reasons.append(reason)
                or bool(state["persist_ok"])
            ),
        },
    )
    window._overrides["_reload_alignment_paradigms"] = (
        lambda self, preferred_slug=None: reload_calls.append(preferred_slug)
    )

    assert window._alignment_paradigm_list is not None

    window._alignment_paradigm_list.setCurrentRow(-1)
    window._on_alignment_export_config()
    window._on_alignment_import_config()
    assert any(
        "Select project, subject, record, and one trial" in item for item in warnings
    )

    window._alignment_paradigm_list.setCurrentRow(0)
    window._on_alignment_export_config()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["schema"] == "lfptensorpipe.alignment-config"
    assert exported["version"] == 1
    assert exported["alignment"]["method"] == "stack_warper"
    assert exported["alignment"]["method_params"]["annotations"] == ["event"]
    assert "pad_s" not in exported["alignment"]["method_params"]
    assert "trial_slug" not in exported["alignment"]

    import_path.write_text("{bad json", encoding="utf-8")
    window._on_alignment_import_config()
    assert any("Import failed" in item for item in warnings)

    import_path.write_text(
        json.dumps(
            {
                "schema": "lfptensorpipe.alignment-config",
                "version": 1,
                "alignment": {
                    "method": "stack_warper",
                    "method_params": {
                        "annotations": ["event", "missing_event"],
                        "duration_range": [0.0, 100.0],
                        "drop_bad": False,
                        "sample_rate": 0.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    window._on_alignment_import_config()
    assert reload_calls[-1] == slug
    assert persist_reasons[-1] == "alignment_import_config"
    assert update_calls[-1]["slug"] == slug
    assert update_calls[-1]["method"] == "stack_warper"
    assert update_calls[-1]["method_params"] == {
        "annotations": ["event"],
        "duration_range": [0.0, 100.0],
        "drop_bad": False,
        "drop_fields": ["bad", "edge"],
        "sample_rate": 0.5,
    }
    assert any("warnings" in item for item in infos)

    import_path.write_text(
        json.dumps(
            {
                "schema": "lfptensorpipe.alignment-config",
                "version": 1,
                "alignment": {
                    "method": "stack_warper",
                    "method_params": {
                        "annotations": ["missing_event"],
                        "duration_range": [0.0, 100.0],
                        "drop_bad": False,
                        "sample_rate": 0.5,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    window._on_alignment_import_config()
    assert update_calls[-1]["method_params"] == {
        "annotations": [],
        "duration_range": [0.0, 100.0],
        "drop_bad": False,
        "drop_fields": ["bad", "edge"],
        "sample_rate": 0.5,
    }
    assert any("warnings" in item for item in infos)

    state["update_result"] = (False, "update failed")
    window._on_alignment_import_config()
    assert any("update failed" in item for item in warnings)

    window.close()


def test_mainwindow_features_config_import_export_handlers(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    warnings: list[str] = []
    infos: list[str] = []
    persist_reasons: list[str] = []
    export_path = tmp_path / "features-export.json"
    import_path = tmp_path / "features-import.json"
    state: dict[str, object] = {
        "save_result": (str(export_path), "JSON files (*.json)"),
        "open_result": (str(import_path), "JSON files (*.json)"),
        "persist_ok": True,
    }

    window, _context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0,
            "_show_information": (
                lambda self, title, message: infos.append(message) or 0
            ),
            "_save_file_name": lambda self, *_args, **_kwargs: state["save_result"],
            "_open_file_name": lambda self, *_args, **_kwargs: state["open_result"],
            "_persist_record_params_snapshot": (
                lambda self, *, reason: persist_reasons.append(reason)
                or bool(state["persist_ok"])
            ),
        },
    )

    raw_power_path = resolver.alignment_root / slug / "raw_power" / "na-raw.pkl"
    raw_power_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(
        pd.DataFrame(
            [
                {
                    "Value": pd.Series([1.0, 2.0], index=[13.0, 30.0]),
                }
            ]
        ),
        raw_power_path,
    )
    psi_path = resolver.alignment_root / slug / "psi" / "na-raw.pkl"
    psi_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(
        pd.DataFrame(
            [
                {
                    "Value": pd.Series([1.0, 2.0], index=["alpha", "beta"]),
                }
            ]
        ),
        psi_path,
    )
    _seed_alignment_trial_finish_ready(resolver, slug)

    window._reload_features_paradigms(preferred_slug=slug)
    assert window._features_paradigm_list is not None

    features_list = window._features_paradigm_list
    window._features_paradigm_list = None
    window._on_features_export_config()
    window._on_features_import_config()
    window._features_paradigm_list = features_list
    assert any(
        "Select project, subject, record, and one trial" in item for item in warnings
    )

    row = window._features_trial_row_for_slug(slug)
    assert row >= 0
    window._features_paradigm_list.setCurrentRow(row)
    assert window._features_axis_metric_combo is not None
    window._features_axes_by_metric = {
        "raw_power": {
            "bands": [
                {"name": "beta", "start": 13.0, "end": 30.0},
            ],
            "times": [
                {"name": "swing", "start": 0.0, "end": 25.0},
                {"name": "stance", "start": 25.0, "end": 75.0},
                {"name": "swing", "start": 75.0, "end": 100.0},
            ],
        },
        "psi": {
            "bands": [
                {"name": "ignored", "start": 1.0, "end": 2.0},
            ],
            "times": [
                {"name": "all", "start": 0.0, "end": 100.0},
            ],
        },
    }
    window._refresh_features_axis_metric_combo()
    raw_power_idx = window._features_axis_metric_combo.findData("raw_power")
    assert raw_power_idx >= 0
    window._features_axis_metric_combo.setCurrentIndex(raw_power_idx)

    window._on_features_export_config()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["schema"] == "lfptensorpipe.features-config"
    assert exported["version"] == 1
    assert exported["features"]["active_metric"] == "raw_power"
    assert exported["features"]["axes_by_metric"]["raw_power"]["bands"] == [
        {"name": "beta", "start": 13.0, "end": 30.0}
    ]
    assert exported["features"]["axes_by_metric"]["raw_power"]["times"] == [
        {"name": "swing", "start": 0.0, "end": 25.0},
        {"name": "stance", "start": 25.0, "end": 75.0},
        {"name": "swing", "start": 75.0, "end": 100.0},
    ]
    assert exported["features"]["axes_by_metric"]["psi"]["bands"] == []
    assert "trial_slug" not in exported["features"]

    import_path.write_text("{bad json", encoding="utf-8")
    window._on_features_import_config()
    assert any("Import failed" in item for item in warnings)

    import_path.write_text(
        json.dumps(
            {
                "schema": "lfptensorpipe.features-config",
                "version": 1,
                "features": {
                    "active_metric": "missing_metric",
                    "axes_by_metric": {
                        "raw_power": {
                            "bands": [
                                {"name": "alpha", "start": 8.0, "end": 12.0},
                            ],
                            "times": [
                                {"name": "stride", "start": 0.0, "end": 25.0},
                                {"name": "stance", "start": 25.0, "end": 75.0},
                                {"name": "stride", "start": 75.0, "end": 100.0},
                            ],
                        },
                        "psi": {
                            "bands": [
                                {"name": "discard", "start": 1.0, "end": 2.0},
                            ],
                            "times": [
                                {"name": "all", "start": 0.0, "end": 100.0},
                            ],
                        },
                        "missing_metric": {
                            "bands": [
                                {"name": "unused", "start": 1.0, "end": 2.0},
                            ],
                            "times": [
                                {"name": "unused", "start": 0.0, "end": 100.0},
                            ],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    window._on_features_import_config()
    assert persist_reasons[-1] == "features_import_config"
    assert window._features_axes_by_metric["raw_power"] == {
        "bands": [{"name": "alpha", "start": 8.0, "end": 12.0}],
        "times": [
            {"name": "stride", "start": 0.0, "end": 25.0},
            {"name": "stance", "start": 25.0, "end": 75.0},
            {"name": "stride", "start": 75.0, "end": 100.0},
        ],
    }
    assert window._features_axes_by_metric["psi"] == {
        "bands": [],
        "times": [{"name": "all", "start": 0.0, "end": 100.0}],
    }
    assert window._current_features_axis_metric() == "raw_power"
    assert any("warnings" in item.lower() for item in infos)

    import_path.write_text(
        json.dumps(
            {
                "schema": "lfptensorpipe.features-config",
                "version": 1,
                "features": {
                    "active_metric": "missing_metric",
                    "axes_by_metric": {
                        "missing_metric": {
                            "bands": [
                                {"name": "unused", "start": 1.0, "end": 2.0},
                            ],
                            "times": [
                                {"name": "unused", "start": 0.0, "end": 100.0},
                            ],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    window._on_features_import_config()
    assert any("no applicable metric config" in item for item in warnings)

    state["persist_ok"] = False
    import_path.write_text(
        json.dumps(
            {
                "schema": "lfptensorpipe.features-config",
                "version": 1,
                "features": {
                    "active_metric": "raw_power",
                    "axes_by_metric": {
                        "raw_power": {
                            "bands": [
                                {"name": "beta", "start": 13.0, "end": 30.0},
                            ],
                            "times": [
                                {"name": "all", "start": 0.0, "end": 100.0},
                            ],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    window._on_features_import_config()
    assert any("persisting record UI state failed" in item for item in warnings)

    window.close()


def test_mainwindow_alignment_panel_indicators_refresh_and_track_pick_changes(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    picked_calls: list[list[int]] = []
    state = {"method": "yellow", "epoch": "gray"}
    window, _context, _resolver, _slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_alignment_method_panel_state_runtime": (
                lambda self, resolver, *, paradigm: state["method"]
            ),
            "_alignment_epoch_inspector_state_runtime": (
                lambda self, resolver, *, paradigm, picked_epoch_indices: (
                    picked_calls.append(list(picked_epoch_indices or [])),
                    state["epoch"],
                )[1]
            ),
        },
    )

    assert window._alignment_method_indicator is not None
    assert window._alignment_epoch_inspector_indicator is not None

    window._refresh_alignment_controls()
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['yellow']}"
        in window._alignment_method_indicator.styleSheet()
    )
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['gray']}"
        in window._alignment_epoch_inspector_indicator.styleSheet()
    )

    state["method"] = "green"
    state["epoch"] = "yellow"
    window._set_alignment_epoch_rows(
        [
            {
                "epoch_index": 0,
                "epoch_label": "epoch_000",
                "start_t": 0.0,
                "end_t": 1.0,
                "pick": True,
            }
        ]
    )
    window._refresh_alignment_controls()
    assert picked_calls[-1] == [0]
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['green']}"
        in window._alignment_method_indicator.styleSheet()
    )
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['yellow']}"
        in window._alignment_epoch_inspector_indicator.styleSheet()
    )

    epoch_table = window._alignment_epoch_table
    assert epoch_table is not None
    pick_item = epoch_table.item(0, 0)
    assert pick_item is not None
    pick_item.setCheckState(Qt.Unchecked)
    assert picked_calls[-1] == []
    state["epoch"] = "green"
    pick_item.setCheckState(Qt.Checked)
    assert picked_calls[-1] == [0]
    assert (
        f"background-color: {main_window_module.INDICATOR_COLORS['green']}"
        in window._alignment_epoch_inspector_indicator.styleSheet()
    )

    window.close()


def test_mainwindow_alignment_epoch_pick_changes_sync_log_state(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    window, _context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
    )
    window._set_alignment_epoch_rows(
        [
            {
                "epoch_index": 0,
                "epoch_label": "epoch_000",
                "start_t": 0.0,
                "end_t": 1.0,
                "pick": True,
            },
            {
                "epoch_index": 1,
                "epoch_label": "epoch_001",
                "start_t": 1.0,
                "end_t": 2.0,
                "pick": True,
            },
        ]
    )

    log_path = alignment_paradigm_log_path(resolver, slug)
    before_payload = read_run_log(log_path)
    assert isinstance(before_payload, dict)

    epoch_table = window._alignment_epoch_table
    assert epoch_table is not None
    pick_item = epoch_table.item(1, 0)
    assert pick_item is not None
    pick_item.setCheckState(Qt.Unchecked)

    payload = read_run_log(log_path)
    assert isinstance(payload, dict)
    assert payload.get("step") == before_payload.get("step")
    assert payload.get("completed") == before_payload.get("completed")
    assert payload.get("state", {}).get("epoch_inspector", {}).get(
        "picked_epoch_indices"
    ) == [0]

    window.close()


def test_mainwindow_alignment_sync_prefers_log_picks_over_stale_ui_state(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    window, _context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
    )
    window._set_alignment_epoch_rows(
        [
            {
                "epoch_index": 0,
                "epoch_label": "epoch_000",
                "start_t": 0.0,
                "end_t": 1.0,
                "pick": True,
            },
            {
                "epoch_index": 1,
                "epoch_label": "epoch_001",
                "start_t": 1.0,
                "end_t": 2.0,
                "pick": True,
            },
        ]
    )

    append_run_log_event(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="build_raw_table",
            completed=True,
            params={
                "trial_slug": slug,
                "picked_epoch_indices": [0],
                "merge_location_info_ready": False,
            },
            input_path="in",
            output_path="out",
            message="finish ready",
        ),
    )
    write_ui_state(
        resolver.record_ui_state_path(create=True),
        {
            "alignment": {
                "trial_slug": slug,
                "picked_epoch_indices": [0, 1],
            }
        },
    )

    window._sync_record_params_from_logs(include_master=True, clear_dirty=True)

    assert window._selected_alignment_epoch_indices() == [0]

    window.close()


def test_mainwindow_alignment_merge_location_status_label_tracks_localize_and_layout(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    localize_state = {"value": "gray"}
    window, _context, _resolver, _slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_alignment_method_panel_state_runtime": (
                lambda self, resolver, *, paradigm: "green"
            ),
            "_alignment_epoch_inspector_state_runtime": (
                lambda self, resolver, *, paradigm, picked_epoch_indices: "yellow"
            ),
            "_localize_indicator_state_runtime": (
                lambda self, *args, **kwargs: localize_state["value"]
            ),
        },
    )
    window._set_alignment_epoch_rows(
        [
            {
                "epoch_index": 0,
                "epoch_label": "epoch_000",
                "start_t": 0.0,
                "end_t": 1.0,
                "pick": True,
            }
        ]
    )

    assert window._alignment_merge_location_status_label is not None
    assert window._alignment_select_all_button is not None
    assert window._alignment_preview_button is not None
    assert window._alignment_finish_button is not None

    window.route_to_stage("alignment")
    window.show()
    app.processEvents()
    window._refresh_alignment_controls()

    status_label = window._alignment_merge_location_status_label
    assert status_label.text() == "Merge Location Info: Not Ready"
    assert "color: #666666;" in status_label.styleSheet()

    localize_state["value"] = "green"
    window._refresh_alignment_controls()
    app.processEvents()
    assert status_label.text() == "Merge Location Info: Ready"
    assert "color: #1f7a1f;" in status_label.styleSheet()

    select_row = window._alignment_select_all_button.parentWidget()
    action_block = status_label.parentWidget()
    assert select_row is not None
    assert action_block is not None
    action_layout = action_block.layout()
    assert action_layout is not None
    assert action_layout.itemAt(0).widget() is select_row
    assert action_layout.itemAt(1).widget() is status_label

    select_layout = select_row.layout()
    assert select_layout is not None
    assert select_layout.itemAt(0).widget() is window._alignment_select_all_button
    assert select_layout.itemAt(1).widget() is window._alignment_preview_button
    assert select_layout.itemAt(2).widget() is window._alignment_finish_button
    assert select_layout.itemAt(3).spacerItem() is not None

    window.close()


def test_mainwindow_alignment_method_switch_restores_trial_method_cache(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    validate_calls: list[dict[str, object]] = []
    dialog_calls: list[dict[str, object]] = []

    class _FakeParamsDialog:
        def __init__(self, **kwargs) -> None:
            dialog_calls.append(kwargs)
            self.selected_params = None

        def exec(self) -> int:
            return int(QDialog.Rejected)

    window, _context, _resolver, _slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_validate_alignment_method_params_runtime": (
                lambda self, method_key, params, **kwargs: (
                    validate_calls.append(
                        {
                            "method_key": method_key,
                            "params": dict(params),
                            "annotation_labels": list(
                                kwargs.get("annotation_labels", [])
                            ),
                        }
                    )
                    or True,
                    dict(params),
                    "",
                )
            ),
            "_update_alignment_paradigm_runtime": (
                lambda self, *_args, **_kwargs: (True, "updated")
            ),
            "_reload_alignment_paradigms": (lambda self, preferred_slug=None: None),
            "_create_alignment_method_params_dialog": (
                lambda self, **kwargs: _FakeParamsDialog(**kwargs)
            ),
        },
    )

    assert window._alignment_paradigm_list is not None
    assert window._alignment_method_combo is not None

    cached_stack = {
        "annotations": ["event"],
        "duration_range": [0.0, 100.0],
        "drop_bad": False,
        "sample_rate": 0.4,
    }
    cached_concat = {
        "annotations": ["event"],
        "drop_bad": True,
        "sample_rate": 12.5,
    }
    window._alignment_paradigms = [
        {
            "name": "Gait",
            "slug": "paradigm-gait",
            "trial_slug": "paradigm-gait",
            "method": "stack_warper",
            "method_params": dict(cached_stack),
            "method_params_by_method": {
                "stack_warper": dict(cached_stack),
                "concat_warper": dict(cached_concat),
            },
            "annotation_filter": {},
        }
    ]
    window._alignment_paradigm_list.clear()
    item = main_window_module.QListWidgetItem("Gait")
    item.setData(Qt.UserRole, "paradigm-gait")
    window._alignment_paradigm_list.addItem(item)
    window._alignment_paradigm_list.setCurrentRow(0)

    window._alignment_paradigms[0]["method_params_by_method"] = {
        "stack_warper": dict(cached_stack),
        "concat_warper": dict(cached_concat),
    }

    concat_idx = window._alignment_method_combo.findData("concat_warper")
    assert concat_idx >= 0
    window._alignment_method_combo.setCurrentIndex(concat_idx)
    window._on_alignment_method_changed(concat_idx)

    assert validate_calls
    assert validate_calls[-1]["method_key"] == "concat_warper"
    assert validate_calls[-1]["params"] == cached_concat

    window._on_alignment_method_params()
    assert dialog_calls
    assert dialog_calls[-1]["method_key"] == "concat_warper"
    assert dialog_calls[-1]["session_params"] == cached_concat

    window.close()


def test_mainwindow_features_run_extract_writes_outputs(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)
    slug = "paradigm-gait"

    raw_table_path = resolver.alignment_root / slug / "raw_power" / "na-raw.pkl"
    nested = pd.DataFrame(
        np.ones((3, 6), dtype=float),
        index=np.array([4.0, 8.0, 12.0], dtype=float),
        columns=np.linspace(0.0, 100.0, 6, dtype=float),
    )
    frame = pd.DataFrame(
        [
            {
                "Subject": subject,
                "Record": record,
                "Trial": slug,
                "Metric": "raw_power",
                "Epoch": "epoch_000",
                "Channel": "CH1",
                "Value": nested,
            }
        ]
    )
    save_pkl(frame, raw_table_path)
    _seed_alignment_trial_finish_ready(resolver, slug)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    store.write_yaml(
        "alignment.yml",
        {
            "paradigms": [
                {
                    "name": "Gait",
                    "slug": slug,
                    "method": "stack_warper",
                    "method_params": {
                        "annotations": ["event"],
                        "mode": "exact",
                        "duration_range": [0.0, 100.0],
                        "drop_bad": False,
                        "pad_s": 0.0,
                        "sample_rate": 0.4,
                    },
                    "annotation_filter": {},
                }
            ]
        },
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)
    app.processEvents()

    assert window._stage_buttons["features"].isEnabled()
    assert window._features_paradigm_list is not None
    assert window._features_paradigm_list.count() == 1
    assert window._features_run_extract_button is not None
    window._features_axes_by_metric = {
        "raw_power": {
            "bands": [{"name": "theta", "start": 4.0, "end": 8.0}],
            "times": [{"name": "early", "start": 0.0, "end": 50.0}],
        }
    }
    window._refresh_features_axis_metric_combo()
    window._refresh_features_controls()
    assert window._features_run_extract_button.isEnabled()

    window._on_features_run_extract()

    deriv_path = resolver.features_root / slug / "raw_power" / "mean-spectral.pkl"
    deriv_log = resolver.features_root / slug / "lfptensorpipe_log.json"
    assert deriv_path.exists()
    assert indicator_from_log(deriv_log) == "green"

    assert window._features_available_table is not None
    assert window._features_available_table.rowCount() >= 1
    window._features_available_table.setCurrentCell(0, 0)
    assert window._features_plot_button is not None
    assert window._features_plot_button.isEnabled()
    window.close()


def test_mainwindow_features_trials_auto_discovered_from_alignment_dirs(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)

    slug = "cycle-l"
    (resolver.alignment_root / slug / "raw_power").mkdir(parents=True, exist_ok=True)
    _seed_alignment_trial_finish_ready(resolver, slug)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)
    app.processEvents()

    assert window._features_paradigm_list is not None
    assert window._features_paradigm_list.count() == 1
    assert window._current_features_paradigm_slug() == slug
    window.close()


def test_mainwindow_features_trials_share_current_trial_and_disable_yellow_entries(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)
    write_run_log(
        preproc_step_log_path(resolver, "finish"),
        RunLogRecord(
            step="finish",
            completed=True,
            params={},
            input_path="in",
            output_path=str(preproc_step_raw_path(resolver, "finish")),
            message="finish ready",
        ),
    )
    write_run_log(
        tensor_metric_log_path(resolver, "raw_power"),
        RunLogRecord(
            step="raw_power",
            completed=True,
            params={},
            input_path="in",
            output_path=str(tensor_metric_tensor_path(resolver, "raw_power")),
            message="raw_power tensor ready",
        ),
    )

    green_slug_a = "trial-a"
    yellow_slug = "trial-b"
    green_slug_c = "trial-c"
    for slug, completed in (
        (green_slug_a, True),
        (yellow_slug, False),
        (green_slug_c, True),
    ):
        trial_dir = resolver.alignment_paradigm_dir(slug)
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_cfg = {
            "name": slug,
            "slug": slug,
            "trial_slug": slug,
            "method": "stack_warper",
            "method_params": {
                "annotations": ["event"],
                "duration_range": [0.0, 10_000.0],
                "drop_bad": False,
                "pad_s": 0.0,
                "sample_rate": 0.8,
            },
            "annotation_filter": {},
        }
        append_run_log_event(
            alignment_paradigm_log_path(resolver, slug),
            RunLogRecord(
                step="run_align_epochs",
                completed=completed,
                params={
                    "trial_slug": slug,
                    "name": slug,
                    "method": "stack_warper",
                    "method_params": dict(trial_cfg["method_params"]),
                    "metrics": ["raw_power"],
                },
                input_path="in",
                output_path=str(trial_dir),
                message=f"{slug} state",
            ),
            state_patch={"trial_config": trial_cfg},
        )
        if completed:
            (trial_dir / "warp_fn.pkl").write_bytes(b"ok")
            (trial_dir / "warp_labels.pkl").write_bytes(b"ok")
            metric_dir = trial_dir / "raw_power"
            metric_dir.mkdir(parents=True, exist_ok=True)
            (metric_dir / "tensor_warped.pkl").write_bytes(b"ok")
            (metric_dir / "na-raw.pkl").write_bytes(b"ok")
            append_run_log_event(
                alignment_paradigm_log_path(resolver, slug),
                RunLogRecord(
                    step="build_raw_table",
                    completed=True,
                    params={
                        "trial_slug": slug,
                        "picked_epoch_indices": [0],
                        "merge_location_info_ready": False,
                    },
                    input_path=str(trial_dir),
                    output_path=str(trial_dir),
                    message=f"{slug} finish ready",
                ),
            )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        enable_plots=False,
    )
    _select_record_context(window, subject=subject, record=record)
    app.processEvents()

    assert window._current_alignment_paradigm_slug() == green_slug_a
    assert window._current_features_paradigm_slug() == green_slug_a
    assert window._stage_states["alignment"] == "green"
    assert window._stage_buttons["features"].isEnabled()

    features_list = window._features_paradigm_list
    alignment_list = window._alignment_paradigm_list
    assert features_list is not None
    assert alignment_list is not None
    assert features_list.count() == 3

    yellow_row = window._features_trial_row_for_slug(yellow_slug)
    green_row_c = window._features_trial_row_for_slug(green_slug_c)
    assert yellow_row >= 0
    assert green_row_c >= 0

    yellow_item = features_list.item(yellow_row)
    green_item_c = features_list.item(green_row_c)
    assert yellow_item is not None
    assert green_item_c is not None
    assert not bool(yellow_item.flags() & Qt.ItemIsEnabled)
    assert yellow_item.foreground().color().name().lower() == "#8a8a8a"
    assert bool(green_item_c.flags() & Qt.ItemIsEnabled)

    window.route_to_stage("features")
    assert window._active_stage_key == "features"
    features_list.setCurrentRow(green_row_c)
    app.processEvents()

    assert window._current_features_paradigm_slug() == green_slug_c
    assert window._current_alignment_paradigm_slug() == green_slug_c
    assert alignment_list.currentItem() is not None
    assert alignment_list.currentItem().data(Qt.UserRole) == green_slug_c

    yellow_alignment_row = window._alignment_row_for_slug(yellow_slug)
    assert yellow_alignment_row >= 0
    window.route_to_stage("alignment")
    alignment_list.setCurrentRow(yellow_alignment_row)
    app.processEvents()

    assert window._current_alignment_paradigm_slug() == yellow_slug
    assert window._current_features_paradigm_slug() is None
    assert window._stage_states["alignment"] == "yellow"
    assert window._stage_states["features"] == "gray"
    assert not window._stage_buttons["features"].isEnabled()

    active_before = window._active_stage_key
    window.route_to_stage("features")
    assert window._active_stage_key == active_before
    window.close()


def test_mainwindow_features_trials_restore_trial_scoped_drafts(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    window, context, resolver, slug_a = _build_alignment_window(
        tmp_path,
        enable_plots=False,
    )
    raw_a_path = resolver.alignment_root / slug_a / "raw_power" / "na-raw.pkl"
    raw_a_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(
        pd.DataFrame([{"Value": pd.Series([1.0, 2.0], index=[13.0, 30.0])}]),
        raw_a_path,
    )
    _seed_alignment_trial_finish_ready(resolver, slug_a)

    slug_b = _create_ready_alignment_trial(
        window._config_store,
        context,
        resolver,
        name="Stride",
    )

    window._refresh_stage_states_from_context()
    window._reload_alignment_paradigms(preferred_slug=slug_a)
    window._reload_features_paradigms(preferred_slug=slug_a)
    window.route_to_stage("features")
    app.processEvents()

    assert window._features_paradigm_list is not None
    row_a = window._features_trial_row_for_slug(slug_a)
    row_b = window._features_trial_row_for_slug(slug_b)
    assert row_a >= 0
    assert row_b >= 0

    window._features_paradigm_list.setCurrentRow(row_a)
    app.processEvents()
    window._features_axes_by_metric = {
        "raw_power": {
            "bands": [{"name": "alpha", "start": 8.0, "end": 12.0}],
            "times": [{"name": "swing", "start": 0.0, "end": 60.0}],
        }
    }
    window._refresh_features_axis_metric_combo()
    assert window._features_filter_feature_edit is not None
    assert window._features_x_label_edit is not None
    window._features_filter_feature_edit.setText("trial-a")
    window._features_x_label_edit.setText("Phase A")

    window._features_paradigm_list.setCurrentRow(row_b)
    app.processEvents()
    window._features_axes_by_metric = {
        "raw_power": {
            "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
            "times": [{"name": "stance", "start": 10.0, "end": 90.0}],
        }
    }
    window._refresh_features_axis_metric_combo()
    window._features_filter_feature_edit.setText("trial-b")
    window._features_x_label_edit.setText("Phase B")

    assert window._persist_record_params_snapshot(reason="test_features_trial_drafts")
    ui_state = json.loads(
        resolver.record_ui_state_path(create=False).read_text(encoding="utf-8")
    )
    assert ui_state["features"]["trial_params_by_slug"][slug_a]["filters"] == {
        "feature": "trial-a"
    }
    assert ui_state["features"]["trial_params_by_slug"][slug_b]["filters"] == {
        "feature": "trial-b"
    }

    window._features_paradigm_list.setCurrentRow(row_a)
    app.processEvents()
    assert window._features_axes_by_metric["raw_power"] == {
        "bands": [{"name": "alpha", "start": 8.0, "end": 12.0}],
        "times": [{"name": "swing", "start": 0.0, "end": 60.0}],
    }
    assert window._features_filter_feature_edit.text() == "trial-a"
    assert window._features_x_label_edit.text() == "Phase A"

    window._features_paradigm_list.setCurrentRow(row_b)
    app.processEvents()
    assert window._features_axes_by_metric["raw_power"] == {
        "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
        "times": [{"name": "stance", "start": 10.0, "end": 90.0}],
    }
    assert window._features_filter_feature_edit.text() == "trial-b"
    assert window._features_x_label_edit.text() == "Phase B"
    window.close()


def test_mainwindow_features_trials_restore_from_success_log_when_missing_draft(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    window, context, resolver, slug_a = _build_alignment_window(
        tmp_path,
        enable_plots=False,
    )
    raw_a_path = resolver.alignment_root / slug_a / "raw_power" / "na-raw.pkl"
    raw_a_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(
        pd.DataFrame([{"Value": pd.Series([1.0, 2.0], index=[13.0, 30.0])}]),
        raw_a_path,
    )
    _seed_alignment_trial_finish_ready(resolver, slug_a)

    slug_b = _create_ready_alignment_trial(
        window._config_store,
        context,
        resolver,
        name="Stride",
    )
    log_axes = {
        "raw_power": {
            "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
            "times": [{"name": "late", "start": 40.0, "end": 100.0}],
        }
    }
    feature_log_path = resolver.features_root / slug_b / "lfptensorpipe_log.json"
    feature_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_run_log(
        feature_log_path,
        RunLogRecord(
            step="run_extract_features",
            completed=True,
            params={
                "trial_slug": slug_b,
                "axes_by_metric": log_axes,
                "metrics": ["raw_power"],
                "target_outputs": 1,
                "saved_outputs": 1,
                "errors": [],
                "xlsx_warnings": [],
            },
            input_path="in",
            output_path=str(feature_log_path.parent),
            message="Extract Features completed.",
        ),
    )

    window._refresh_stage_states_from_context()
    window._reload_alignment_paradigms(preferred_slug=slug_a)
    window._reload_features_paradigms(preferred_slug=slug_a)
    window.route_to_stage("features")
    app.processEvents()

    assert window._features_paradigm_list is not None
    row_b = window._features_trial_row_for_slug(slug_b)
    assert row_b >= 0
    window._features_paradigm_list.setCurrentRow(row_b)
    app.processEvents()

    assert window._features_axes_by_metric["raw_power"] == log_axes["raw_power"]
    assert window._features_trial_params_by_slug[slug_b]["axes_by_metric"] == log_axes
    window.close()


def test_mainwindow_alignment_features_trials_panel_geometry_matches(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    window, _context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
    )

    _seed_alignment_trial_finish_ready(resolver, slug)
    window._refresh_stage_states_from_context()
    window._stage_states["tensor"] = "green"
    window._refresh_stage_controls()
    window.show()
    app.processEvents()

    alignment_block = window._alignment_trials_block
    features_block = window._features_trials_block
    alignment_list = window._alignment_paradigm_list
    features_list = window._features_paradigm_list
    alignment_row = window._alignment_trials_action_row
    features_row = window._features_trials_action_row
    assert alignment_block is not None
    assert features_block is not None
    assert alignment_list is not None
    assert features_list is not None
    assert alignment_row is not None
    assert features_row is not None

    window.route_to_stage("alignment")
    app.processEvents()
    alignment_block_size = alignment_block.size()
    alignment_list_size = alignment_list.viewport().size()
    alignment_row_geom = alignment_row.geometry()
    alignment_add_geom = window._alignment_paradigm_add_button.geometry()
    alignment_delete_geom = window._alignment_paradigm_delete_button.geometry()

    assert window._stage_buttons["features"].isEnabled()
    window.route_to_stage("features")
    app.processEvents()
    features_block_size = features_block.size()
    features_list_size = features_list.viewport().size()
    features_row_geom = features_row.geometry()
    features_add_geom = window._features_paradigm_add_button.geometry()
    features_delete_geom = window._features_paradigm_delete_button.geometry()

    assert abs(alignment_block_size.width() - features_block_size.width()) <= 2
    assert abs(alignment_block_size.height() - features_block_size.height()) <= 2
    assert abs(alignment_list_size.width() - features_list_size.width()) <= 2
    assert abs(alignment_list_size.height() - features_list_size.height()) <= 2
    assert abs(alignment_row_geom.y() - features_row_geom.y()) <= 2
    assert abs(alignment_row_geom.height() - features_row_geom.height()) <= 2
    assert abs(alignment_add_geom.x() - features_add_geom.x()) <= 2
    assert abs(alignment_add_geom.width() - features_add_geom.width()) <= 2
    assert abs(alignment_delete_geom.x() - features_delete_geom.x()) <= 2
    assert abs(alignment_delete_geom.width() - features_delete_geom.width()) <= 2
    window.close()


def test_mainwindow_features_axis_defaults_roundtrip(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    window, _context, _resolver, _slug = _build_alignment_window(
        tmp_path, enable_plots=False
    )
    band_rows = [{"name": "theta", "start": 4.0, "end": 8.0}]
    phase_rows = [{"name": "early", "start": 0.0, "end": 50.0}]

    window._save_features_axis_defaults(
        metric_key="raw_power",
        axis_key="bands",
        rows=band_rows,
    )
    window._save_features_axis_defaults(
        metric_key="raw_power",
        axis_key="times",
        rows=phase_rows,
    )
    loaded_bands = window._load_features_axis_defaults(
        metric_key="raw_power",
        axis_key="bands",
    )
    loaded_times = window._load_features_axis_defaults(
        metric_key="raw_power",
        axis_key="times",
    )
    assert loaded_bands == band_rows
    assert loaded_times == phase_rows

    window._features_axes_by_metric = {}
    axes = window._normalized_features_axes_for_metric("raw_power")
    assert axes["bands"] == band_rows
    assert axes["times"] == phase_rows
    window.close()


def test_mainwindow_features_auto_bands_disable_and_validate(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    window, _context, _resolver, _slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
    )
    window._features_axes_by_metric = {
        "psi": {
            "bands": [{"name": "legacy", "start": 1.0, "end": 2.0}],
            "times": [{"name": "all", "start": 0.0, "end": 100.0}],
        }
    }
    assert window._features_axis_metric_combo is not None
    window._features_axis_metric_combo.clear()
    window._features_axis_metric_combo.addItem("psi", "psi")
    window._features_axis_metric_combo.setCurrentIndex(0)

    window._overrides["_features_auto_band_names_from_alignment_raw"] = (
        lambda self, _metric: ["a", "b"]
    )
    window._refresh_features_axis_buttons()
    assert window._features_axis_bands_button is not None
    assert window._features_axis_bands_button.text() == "Bands Auto (2)"
    assert not window._features_axis_bands_button.isEnabled()

    ok, message = window._validate_features_axes_for_run(["psi"])
    assert ok
    assert message == ""

    window._overrides["_features_auto_band_names_from_alignment_raw"] = (
        lambda self, _metric: []
    )
    ok, message = window._validate_features_axes_for_run(["psi"])
    assert not ok
    assert "no band labels" in message.lower()
    window.close()


def test_mainwindow_features_plot_raw_vik_sets_vmode_sym(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    window, _context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=True,
        window_cls=OverrideMainWindow,
    )
    assert window._features_available_table is not None
    table = window._features_available_table

    payload = pd.DataFrame(
        {
            "Value": [
                pd.DataFrame(
                    np.ones((2, 2), dtype=float),
                    index=np.array([4.0, 8.0], dtype=float),
                    columns=np.array([0.0, 100.0], dtype=float),
                )
            ]
        }
    )
    feature_path = resolver.features_root / slug / "raw_power" / "mean-raw.pkl"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.touch(exist_ok=True)

    window._features_filtered_files = [
        {
            "path": feature_path,
            "derived_type": "raw",
            "display_feature": "raw",
            "feature": "raw",
            "property": "mean",
            "relative_stem": "raw_power/mean-raw",
            "stem": "mean-raw",
            "metric": "raw_power",
        }
    ]
    table.setRowCount(1)
    table.setItem(0, 0, QTableWidgetItem("raw_power"))
    table.setItem(0, 1, QTableWidgetItem("raw"))
    table.setItem(0, 2, QTableWidgetItem("mean"))
    table.setCurrentCell(0, 0)

    class _FakeFigure:
        def show(self) -> None:
            return

    captured: dict[str, object] = {}

    def _fake_plot_single_effect_df(*_args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return _FakeFigure()

    window._overrides["_load_pickle"] = lambda self, _path: payload
    window._overrides["_plot_single_effect_df"] = (
        lambda self, *_args, **kwargs: _fake_plot_single_effect_df(*_args, **kwargs)
    )
    window._overrides["_plot_single_effect_series"] = (
        lambda self, *_args, **_kwargs: _FakeFigure()
    )
    window._overrides["_plot_single_effect_scalar"] = (
        lambda self, *_args, **_kwargs: _FakeFigure()
    )
    window._overrides["_load_cmcrameri_vik"] = lambda self: "fake-vik"

    window._features_plot_advance_params["colormap"] = "cmcrameri.vik"
    window._on_features_plot()
    assert captured.get("vmode") == "sym"

    window._features_plot_advance_params["colormap"] = "viridis"
    captured.clear()
    window._on_features_plot()
    assert captured.get("vmode") == "auto"
    window.close()


@pytest.mark.parametrize(
    ("metric_key", "derived_type", "payload", "plotter_name", "expected"),
    [
        (
            "raw_power",
            "raw",
            pd.DataFrame(
                {
                    "Value": [
                        pd.DataFrame(
                            np.ones((2, 2), dtype=float),
                            index=[4.0, 8.0],
                            columns=[0.0, 50.0],
                        )
                    ]
                }
            ),
            "_plot_single_effect_df",
            {
                "boxsize": (72.0, 54.0),
                "axis_label_fontsize": 18.0,
                "tick_label_fontsize": 11.0,
                "x_label_offset_mm": 8.0,
                "y_label_offset_mm": 17.0,
                "colorbar_pad_mm": 5.0,
                "cbar_label_offset_mm": 14.0,
                "title": None,
            },
        ),
        (
            "psi",
            "spectral",
            pd.DataFrame({"Value": [pd.Series([1.0, 2.0], index=[10.0, 20.0])]}),
            "_plot_single_effect_series",
            {
                "boxsize": (66.0, 48.0),
                "axis_label_fontsize": 14.0,
                "tick_label_fontsize": 9.0,
                "x_label_offset_mm": 6.0,
                "y_label_offset_mm": 13.0,
                "legend_loc": "outside_left",
                "title": None,
            },
        ),
        (
            "burst",
            "scalar",
            pd.DataFrame({"Phase": ["Baseline"], "Value": [1.0]}),
            "_plot_single_effect_scalar",
            {
                "boxsize": (62.0, 44.0),
                "axis_label_fontsize": 12.0,
                "tick_label_fontsize": 7.0,
                "x_label_offset_mm": 8.0,
                "y_label_offset_mm": 11.0,
                "legend_loc": "outside_bottom",
                "title": None,
            },
        ),
    ],
)
def test_mainwindow_features_plot_loads_private_style_defaults(
    tmp_path: Path,
    metric_key: str,
    derived_type: str,
    payload: pd.DataFrame,
    plotter_name: str,
    expected: dict[str, object],
) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    window, _context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=True,
        window_cls=OverrideMainWindow,
    )
    assert window._features_available_table is not None
    table = window._features_available_table
    feature_path = (
        resolver.features_root / slug / metric_key / f"mean-{derived_type}.pkl"
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.touch(exist_ok=True)

    window._config_store.write_yaml(
        "features_plot.yml",
        {
            "plot_defaults": {
                "default": {
                    "boxsize": [60.0, 50.0],
                    "font_size": 16,
                    "tick_label_size": 12,
                    "x_label_offset_mm": 10.0,
                    "y_label_offset_mm": 18.0,
                    "colorbar_pad_mm": 4.0,
                    "cbar_label_offset_mm": 18.0,
                    "legend_position": "outside_right",
                },
                "by_derived_type": {
                    "raw": {
                        "font_size": 17,
                        "y_label_offset_mm": 16.0,
                        "colorbar_pad_mm": 3.0,
                        "legend_position": "outside_right",
                    },
                    "spectral": {
                        "boxsize": [64.0, 46.0],
                        "font_size": 15,
                        "y_label_offset_mm": 14.0,
                        "legend_position": "outside_top",
                    },
                    "scalar": {
                        "font_size": 13,
                        "x_label_offset_mm": 9.0,
                        "legend_position": "outside_right",
                    },
                },
                "by_metric": {
                    "raw_power": {
                        "boxsize": [70.0, 52.0],
                        "tick_label_size": 10,
                        "x_label_offset_mm": 9.0,
                    },
                    "psi": {
                        "boxsize": [66.0, 48.0],
                        "tick_label_size": 8,
                        "x_label_offset_mm": 7.0,
                        "legend_position": "outside_right",
                    },
                    "burst": {
                        "boxsize": [62.0, 44.0],
                        "y_label_offset_mm": 12.0,
                        "legend_position": "outside_left",
                    },
                },
                "by_metric_and_derived_type": {
                    "raw_power": {
                        "raw": {
                            "boxsize": [72.0, 54.0],
                            "font_size": 18,
                            "tick_label_size": 11,
                            "x_label_offset_mm": 8.0,
                            "y_label_offset_mm": 17.0,
                            "colorbar_pad_mm": 5.0,
                            "cbar_label_offset_mm": 14.0,
                        }
                    },
                    "psi": {
                        "spectral": {
                            "font_size": 14,
                            "tick_label_size": 9,
                            "x_label_offset_mm": 6.0,
                            "y_label_offset_mm": 13.0,
                            "legend_position": "outside_left",
                        }
                    },
                    "burst": {
                        "scalar": {
                            "font_size": 12,
                            "tick_label_size": 7,
                            "x_label_offset_mm": 8.0,
                            "y_label_offset_mm": 11.0,
                            "legend_position": "outside_bottom",
                        }
                    },
                },
            }
        },
    )

    window._features_filtered_files = [
        {
            "path": feature_path,
            "derived_type": derived_type,
            "display_feature": derived_type,
            "feature": derived_type,
            "property": "mean",
            "relative_stem": f"{metric_key}/mean-{derived_type}",
            "stem": f"mean-{derived_type}",
            "metric": metric_key,
        }
    ]
    table.setRowCount(1)
    table.setItem(0, 0, QTableWidgetItem(metric_key))
    table.setItem(0, 1, QTableWidgetItem(derived_type))
    table.setItem(0, 2, QTableWidgetItem("mean"))
    table.setCurrentCell(0, 0)

    captured: dict[str, object] = {}

    class _FakeFigure:
        def show(self) -> None:
            return

    def _capture_plot(*_args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return _FakeFigure()

    window._overrides["_load_pickle"] = lambda self, _path: payload
    window._overrides["_plot_single_effect_series"] = (
        (lambda self, *_args, **kwargs: _capture_plot(*_args, **kwargs))
        if plotter_name == "_plot_single_effect_series"
        else (lambda self, *_args, **_kwargs: _FakeFigure())
    )
    window._overrides["_plot_single_effect_df"] = (
        (lambda self, *_args, **kwargs: _capture_plot(*_args, **kwargs))
        if plotter_name == "_plot_single_effect_df"
        else (lambda self, *_args, **_kwargs: _FakeFigure())
    )
    window._overrides["_plot_single_effect_scalar"] = (
        (lambda self, *_args, **kwargs: _capture_plot(*_args, **kwargs))
        if plotter_name == "_plot_single_effect_scalar"
        else (lambda self, *_args, **_kwargs: _FakeFigure())
    )

    window._on_features_plot()

    for key, value in expected.items():
        assert captured.get(key) == value
    if plotter_name != "_plot_single_effect_df":
        assert "colorbar_pad_mm" not in captured
        assert "cbar_label_offset_mm" not in captured
        assert captured.get("legend_loc") == expected["legend_loc"]
    else:
        assert "legend_loc" not in captured
    window.close()


def test_mainwindow_features_plot_injects_all_phase_for_trace_and_raw(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    window, _context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=True,
        window_cls=OverrideMainWindow,
    )
    assert window._features_available_table is not None
    table = window._features_available_table
    feature_path = resolver.features_root / slug / "psi" / "mean-trace.pkl"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.touch(exist_ok=True)

    window._features_filtered_files = [
        {
            "path": feature_path,
            "derived_type": "trace",
            "display_feature": "trace",
            "feature": "trace",
            "property": "mean",
            "relative_stem": "psi/mean-trace",
            "stem": "mean-trace",
            "metric": "psi",
        }
    ]
    table.setRowCount(1)
    table.setItem(0, 0, QTableWidgetItem("psi"))
    table.setItem(0, 1, QTableWidgetItem("trace"))
    table.setItem(0, 2, QTableWidgetItem("mean"))
    table.setCurrentCell(0, 0)

    payload = pd.DataFrame({"Value": [pd.Series([1.0, 2.0, 3.0])]})

    class _FakeFigure:
        def show(self) -> None:
            return

    captured_trace: dict[str, object] = {}

    def _fake_plot_single_effect_series(df, *_args, **_kwargs):  # noqa: ANN001
        captured_trace["df"] = df
        return _FakeFigure()

    window._overrides["_load_pickle"] = lambda self, _path: payload
    window._overrides["_plot_single_effect_series"] = (
        lambda self, *_args, **_kwargs: _fake_plot_single_effect_series(
            *_args, **_kwargs
        )
    )
    window._overrides["_plot_single_effect_df"] = (
        lambda self, *_args, **_kwargs: _FakeFigure()
    )
    window._overrides["_plot_single_effect_scalar"] = (
        lambda self, *_args, **_kwargs: _FakeFigure()
    )

    window._on_features_plot()
    plotted_df = captured_trace.get("df")
    assert isinstance(plotted_df, pd.DataFrame)
    assert plotted_df["Phase"].tolist() == ["All"]
    window.close()


def test_feature_axis_configure_allow_all_name() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    dialog = main_window_module.FeatureAxisConfigureDialog(
        title="Bands",
        item_label="Band",
        current_rows=(),
        min_start=0.0,
        max_end=None,
        parent=None,
    )
    valid, _rows, message = dialog._validate_rows(
        [{"name": "All", "start": 1.0, "end": 2.0}]
    )
    assert valid
    assert message == ""
    dialog.close()


def test_feature_axis_configure_set_default_keeps_dialog_open() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app

    captured: list[list[dict[str, float | str]]] = []
    dialog = main_window_module.FeatureAxisConfigureDialog(
        title="Bands",
        item_label="Band",
        current_rows=({"name": "theta", "start": 4.0, "end": 8.0},),
        default_rows=(),
        set_default_callback=lambda rows: captured.append(
            [dict(item) for item in rows]
        ),
        min_start=0.0,
        max_end=None,
        parent=None,
    )
    dialog._on_submit("set_default")
    assert captured == [[{"name": "theta", "start": 4.0, "end": 8.0}]]
    assert dialog.result() == 0

    dialog._on_submit("save")
    assert dialog.result() == QDialog.Accepted
    dialog.close()


def test_mainwindow_features_colorbar_label_enabled_only_for_raw(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    window, _context, _resolver, _slug = _build_alignment_window(
        tmp_path, enable_plots=False
    )
    app.processEvents()

    table = window._features_available_table
    cbar = window._features_cbar_label_edit
    assert table is not None
    assert cbar is not None

    window._features_filtered_files = [
        {
            "path": str(tmp_path / "fake-raw.pkl"),
            "metric": "raw_power",
            "feature": "raw",
            "property": "na",
            "derived_type": "raw",
        },
        {
            "path": str(tmp_path / "fake-trace.pkl"),
            "metric": "raw_power",
            "feature": "trace",
            "property": "mean",
            "derived_type": "trace",
        },
    ]
    table.setRowCount(2)
    table.setItem(0, 0, QTableWidgetItem("raw_power"))
    table.setItem(0, 1, QTableWidgetItem("raw"))
    table.setItem(0, 2, QTableWidgetItem("na"))
    table.setItem(1, 0, QTableWidgetItem("raw_power"))
    table.setItem(1, 1, QTableWidgetItem("trace"))
    table.setItem(1, 2, QTableWidgetItem("mean"))

    table.setCurrentCell(0, 0)
    window._on_features_available_selection_changed()
    assert cbar.isEnabled()

    table.setCurrentCell(1, 0)
    window._on_features_available_selection_changed()
    assert not cbar.isEnabled()

    window.close()


def test_mainwindow_tensor_handler_deep_branches(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    warnings: list[str] = []
    infos: list[str] = []
    window, context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("finish",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0,
            "_show_information": (
                lambda self, title, message: infos.append(message) or 0
            ),
        },
    )
    app.processEvents()

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_tensor_run()
    assert "select project/subject/record" in window.statusBar().currentMessage()

    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record
    window._stage_states["preproc"] = "yellow"
    window._on_tensor_run()
    assert any("Preprocess Signal must be green" in item for item in warnings)
    window._stage_states["preproc"] = "green"

    def _raise_tensor_params(_context):  # noqa: ANN001
        raise ValueError("bad params")

    window._overrides["_collect_tensor_runtime_params"] = (
        lambda self, _context: _raise_tensor_params(_context)
    )
    window._on_tensor_run()
    assert any("Invalid tensor parameters" in item for item in warnings)
    assert "invalid parameters (bad params)" in window.statusBar().currentMessage()

    window._tensor_available_channels = ()
    window._on_tensor_channels_select()
    assert any("No channels available for selection." in item for item in warnings)

    window._tensor_available_channels = ("CH1", "CH2", "CH3")
    window._set_active_tensor_metric("raw_power")
    window._tensor_selected_channels_by_metric["raw_power"] = ("CH1",)
    channel_dialog_state: dict[str, object] = {
        "exec_result": QDialog.Rejected,
        "selected_channels": ("CH2",),
        "selected_action": "save",
    }
    channel_default_inputs: list[tuple[str, ...]] = []
    saved_default_channels: list[tuple[str, ...]] = []

    class _FakeTensorChannelDialog:
        def __init__(
            self,
            *,
            title,
            channels,
            session_selected,
            default_selected,
            set_default_callback=None,
            parent,
        ):  # noqa: ANN001
            _ = (title, channels, session_selected, parent)
            channel_default_inputs.append(tuple(default_selected))
            self.selected_channels = list(channel_dialog_state["selected_channels"])
            self.selected_action = str(channel_dialog_state["selected_action"])
            self._set_default_callback = set_default_callback

        def exec(self) -> int:
            if (
                int(channel_dialog_state["exec_result"]) == QDialog.Accepted
                and self.selected_action == "set_default"
                and self._set_default_callback is not None
            ):
                self._set_default_callback(tuple(self.selected_channels))
            return int(channel_dialog_state["exec_result"])

    window._overrides["_create_tensor_channel_select_dialog"] = (
        lambda self, **kwargs: _FakeTensorChannelDialog(**kwargs)
    )
    window._overrides["_tensor_load_default_channels"] = lambda self: None
    window._overrides["_tensor_save_default_channels"] = (
        lambda self, channels: saved_default_channels.append(tuple(channels))
    )

    window._on_tensor_channels_select()
    assert channel_default_inputs[-1] == tuple(window._tensor_available_channels)
    assert window._tensor_selected_channels_by_metric["raw_power"] == ("CH1",)

    channel_dialog_state["exec_result"] = QDialog.Accepted
    channel_dialog_state["selected_channels"] = ("CH2",)
    channel_dialog_state["selected_action"] = "set_default"
    window._on_tensor_channels_select()
    assert window._tensor_selected_channels_by_metric["raw_power"] == ("CH2",)
    assert saved_default_channels[-1] == ("CH2",)
    assert "defaults updated" in window.statusBar().currentMessage().lower()

    window._set_active_tensor_metric("unknown_metric")
    window._on_tensor_pairs_select()

    window._set_active_tensor_metric("raw_power")
    window._on_tensor_pairs_select()
    assert any("connectivity metrics" in item for item in infos)

    window._set_active_tensor_metric("coherence")
    window._tensor_available_channels = ()
    window._on_tensor_pairs_select()
    assert any("No channels available for pair selection." in item for item in warnings)

    window._tensor_available_channels = ("CH1", "CH2", "CH3")
    window._tensor_selected_pairs_by_metric["coherence"] = (("CH1", "CH3"),)
    pair_dialog_state: dict[str, object] = {
        "exec_result": QDialog.Rejected,
        "selected_pairs": [("CH1", "CH2")],
        "selected_action": "save",
    }
    saved_default_pairs: list[tuple[tuple[tuple[str, str], ...], bool]] = []

    class _FakeTensorPairDialog:
        def __init__(
            self,
            *,
            title,
            channel_names,
            session_pairs,
            default_pairs,
            directed,
            set_default_callback=None,
            parent,
        ):  # noqa: ANN001
            _ = (title, channel_names, session_pairs, default_pairs, directed, parent)
            self.selected_pairs = list(pair_dialog_state["selected_pairs"])
            self.selected_action = str(pair_dialog_state["selected_action"])
            self._set_default_callback = set_default_callback
            self._directed = bool(directed)

        def exec(self) -> int:
            if (
                int(pair_dialog_state["exec_result"]) == QDialog.Accepted
                and self.selected_action == "set_default"
                and self._set_default_callback is not None
            ):
                normalized = window._filter_tensor_pairs(
                    self.selected_pairs,
                    available_channels=window._tensor_available_channels,
                    directed=self._directed,
                )
                self._set_default_callback(normalized)
            return int(pair_dialog_state["exec_result"])

    window._overrides["_create_tensor_pair_select_dialog"] = (
        lambda self, **kwargs: _FakeTensorPairDialog(**kwargs)
    )
    window._overrides["_tensor_load_default_pairs"] = lambda self, directed=False: ()
    window._overrides["_tensor_save_default_pairs"] = (
        lambda self, pairs, directed=False: saved_default_pairs.append(
            (tuple(tuple(pair) for pair in pairs), bool(directed))
        )
    )

    window._on_tensor_pairs_select()
    assert window._tensor_selected_pairs_by_metric["coherence"] == (("CH1", "CH3"),)

    pair_dialog_state["exec_result"] = QDialog.Accepted
    pair_dialog_state["selected_pairs"] = [
        ("CH2", "CH1"),
        ("CH1", "CH1"),
        ("CH4", "CH2"),
    ]
    pair_dialog_state["selected_action"] = "set_default"
    window._on_tensor_pairs_select()
    assert window._tensor_selected_pairs_by_metric["coherence"] == (("CH1", "CH2"),)
    assert saved_default_pairs[-1] == ((("CH1", "CH2"),), False)

    window._set_active_tensor_metric("trgc")
    pair_dialog_state["selected_pairs"] = [
        ("CH2", "CH1"),
        ("CH1", "CH1"),
        ("CH4", "CH2"),
    ]
    window._on_tensor_pairs_select()
    assert window._tensor_selected_pairs_by_metric["trgc"] == (("CH2", "CH1"),)
    assert saved_default_pairs[-1] == ((("CH2", "CH1"),), True)

    window.close()


def test_mainwindow_tensor_non_handler_selector_and_filter_helpers(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("finish",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
    )

    assert MainWindow._parse_tensor_pair_token(("A", "B")) == ("A", "B")
    assert MainWindow._parse_tensor_pair_token(("A", " ")) is None
    assert MainWindow._parse_tensor_pair_token(123) is None
    assert MainWindow._parse_tensor_pair_token("  ") is None
    assert MainWindow._parse_tensor_pair_token("(A,B)") == ("A", "B")
    assert MainWindow._parse_tensor_pair_token("(A,)") is None
    assert MainWindow._parse_tensor_pair_token("A-B") == ("A", "B")
    assert MainWindow._parse_tensor_pair_token("A- ") is None
    with pytest.raises(ValueError, match="cannot be empty"):
        MainWindow._normalize_tensor_pair("", "B", directed=False)
    assert MainWindow._filter_tensor_pairs(
        [("A", "B"), ("B", "A"), ("A", "B")],
        available_channels=("A", "B"),
        directed=False,
    ) == (("A", "B"),)

    selector_key = main_window_module.TENSOR_SELECTOR_DEFAULTS_KEY
    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: ["bad_payload"]
    )
    assert window._tensor_read_selector_defaults_payload() == {}

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: {selector_key: "bad_selectors"}
    )
    assert window._tensor_read_selector_defaults_payload() == {}

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: {selector_key: {"channels": "bad_type"}}
    )
    assert window._tensor_load_default_channels() == ()

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: {
            selector_key: {"channels": [" CH1 ", "", "CH1", "CH2"]}
        }
    )
    assert window._tensor_load_default_channels() == ("CH1", "CH2")

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: {selector_key: {}}
    )
    assert window._tensor_load_default_pairs(directed=False) is None

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: {selector_key: {"undirected_pairs": "bad_type"}}
    )
    assert window._tensor_load_default_pairs(directed=False) == ()

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: {
            selector_key: {
                "undirected_pairs": [
                    ("A", "B"),
                    "B-A",
                    "(A,C)",
                    "bad",
                    ("A", "A"),
                ],
                "directed_pairs": [("B", "A"), "(A,C)", "A-A", " "],
            }
        }
    )
    assert window._tensor_load_default_pairs(directed=False) == (
        ("A", "B"),
        ("A", "B"),
        ("A", "C"),
    )
    assert window._tensor_load_default_pairs(directed=True) == (("B", "A"), ("A", "C"))

    writes: list[dict[str, object]] = []

    def _capture_write(path: str, payload: dict[str, object]):  # noqa: ARG001
        writes.append(dict(payload))
        return Path("/tmp/tensor.yml")

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: "bad_payload"
    )
    window._config_store.write_yaml = _capture_write  # type: ignore[method-assign]
    window._tensor_save_default_channels(("CH1", "CH2"))
    assert writes[-1][selector_key] == {"channels": ["CH1", "CH2"]}

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: {selector_key: "bad_selectors"}
    )
    window._tensor_save_default_pairs((("A", "B"),), directed=False)
    assert writes[-1][selector_key]["undirected_pairs"] == ["A-B"]  # type: ignore[index]
    window._tensor_save_default_pairs((("A", "B"),), directed=True)
    assert writes[-1][selector_key]["directed_pairs"] == ["(A,B)"]  # type: ignore[index]
    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda path, default=None: []
    )
    window._tensor_save_default_pairs((("A", "B"),), directed=False)
    assert writes[-1][selector_key]["undirected_pairs"] == ["A-B"]  # type: ignore[index]

    pairs_button = window._tensor_pairs_button
    assert pairs_button is not None
    window._tensor_pairs_button = None
    window._refresh_tensor_pair_button_text()
    window._tensor_pairs_button = pairs_button

    assert MainWindow._parse_filter_notches("") == []
    with pytest.raises(ValueError, match="positive"):
        MainWindow._parse_filter_notches("50,-1")

    assert window._preproc_filter_notches_edit is not None
    assert window._preproc_filter_low_freq_edit is not None
    assert window._preproc_filter_high_freq_edit is not None

    window._preproc_filter_notches_edit.setText("50,100")
    window._preproc_filter_low_freq_edit.setText("-1")
    window._preproc_filter_high_freq_edit.setText("100")
    with pytest.raises(ValueError, match="Low freq must be >= 0"):
        window._collect_filter_runtime_params()

    window._preproc_filter_low_freq_edit.setText("1")
    window._preproc_filter_high_freq_edit.setText("1")
    with pytest.raises(ValueError, match="greater than Low freq"):
        window._collect_filter_runtime_params()

    window._preproc_filter_notches_edit.setText("bad")
    window._preproc_filter_low_freq_edit.setText("1")
    window._preproc_filter_high_freq_edit.setText("100")
    with pytest.raises(ValueError):
        window._collect_filter_runtime_params()

    window._preproc_filter_notches_edit.setText("50,100")
    notches, low_freq, high_freq = window._collect_filter_runtime_params()
    assert notches == [50.0, 100.0]
    assert low_freq == 1.0
    assert high_freq == 100.0

    warnings: list[str] = []
    selector_state = {"exec_result": QDialog.Rejected, "selected_channels": ()}

    class _FakeChannelDialog:
        def __init__(
            self,
            *,
            title,
            channels,
            selected_channels,
            parent,
        ):  # noqa: ANN001
            _ = (title, channels, selected_channels, parent)
            self.selected_channels = tuple(selector_state["selected_channels"])

        def exec(self) -> int:
            return int(selector_state["exec_result"])

    def _fake_warning(parent, title, text, *args, **kwargs):  # noqa: ANN001, ARG001
        _ = (parent, title, args, kwargs)
        warnings.append(str(text))
        return 0

    window._overrides["_create_channel_select_dialog"] = (
        lambda self, **kwargs: _FakeChannelDialog(**kwargs)
    )
    window._overrides["_show_warning"] = lambda self, title, message: _fake_warning(
        None, title, message
    )

    result = window._run_channel_selector(
        title="Tensor Channels",
        available=("CH1", "CH2"),
        selected=("CH1",),
    )
    assert result is None

    selector_state["exec_result"] = QDialog.Accepted
    selector_state["selected_channels"] = ()
    result = window._run_channel_selector(
        title="Tensor Channels",
        available=("CH1", "CH2"),
        selected=("CH1",),
    )
    assert result is None
    assert any("At least one channel must be selected." in item for item in warnings)

    result = window._run_channel_selector(
        title="Tensor Channels",
        available=("CH1", "CH2"),
        selected=("CH1",),
        allow_empty=True,
    )
    assert result == ()

    selector_state["selected_channels"] = ("CH2",)
    result = window._run_channel_selector(
        title="Tensor Channels",
        available=("CH1", "CH2"),
        selected=("CH1",),
    )
    assert result == ("CH2",)

    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record
    window.close()


def test_mainwindow_tensor_metric_default_priority_prefers_metric_defaults(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("finish",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
    )

    window._tensor_available_channels = ("CH1", "CH2", "CH3")
    window._tensor_metric_params = {}
    window._tensor_selected_channels_by_metric = {}
    window._tensor_selected_pairs_by_metric = {}

    window._overrides["_tensor_load_default_channels"] = lambda self: ("CH2",)
    window._overrides["_tensor_load_default_pairs"] = lambda self, directed=False: (
        (("CH1", "CH2"),) if not directed else (("CH2", "CH1"),)
    )

    metric_overrides = {
        "raw_power": {"selected_channels": ["CH1"]},
        "coherence": {"selected_pairs": [["CH3", "CH2"]]},
        "trgc": {"selected_pairs": [["CH3", "CH1"]]},
    }
    window._overrides["_tensor_metric_default_override_node"] = (
        lambda self, metric_key: dict(metric_overrides.get(metric_key, {}))
    )

    window._ensure_tensor_metric_state_from_defaults(context)

    assert window._tensor_selected_channels_by_metric["raw_power"] == ("CH1",)
    assert window._tensor_selected_channels_by_metric["burst"] == ("CH2",)
    assert window._tensor_selected_pairs_by_metric["coherence"] == (("CH2", "CH3"),)
    assert window._tensor_selected_pairs_by_metric["trgc"] == (("CH3", "CH1"),)

    window.close()


def test_mainwindow_tensor_advance_persists_and_applies_full_metric_payload(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, _context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("finish",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
    )

    window._tensor_available_channels = ("CH1", "CH2", "CH3")
    window._set_active_tensor_metric("psi")
    window._tensor_selected_pairs_by_metric["psi"] = (("CH1", "CH2"),)
    window._tensor_metric_params["psi"] = {
        "time_resolution_s": 0.5,
        "hop_s": 0.025,
        "method": "morlet",
        "bands": [{"name": "alpha", "start": 8.0, "end": 12.0}],
        "selected_pairs": [["CH1", "CH2"]],
    }
    window._set_active_tensor_metric("burst")
    window._tensor_selected_channels_by_metric["burst"] = ("CH1",)
    window._tensor_metric_params["burst"] = {
        "percentile": 75.0,
        "min_cycles": 2.0,
        "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
        "selected_channels": ["CH1"],
    }

    saved_defaults: list[tuple[str, dict[str, object]]] = []
    window._overrides["_save_tensor_metric_default_params"] = (
        lambda self, metric_key, params: saved_defaults.append(
            (metric_key, dict(params))
        )
    )
    window._overrides["_load_burst_baseline_annotation_labels_runtime"] = (
        lambda self, _context: ["Rest", "Task"]
    )

    class _FakeTensorAdvanceDialog:
        def __init__(
            self,
            *,
            metric_key,
            metric_label,
            session_params,
            default_params,
            burst_baseline_annotations,
            set_default_callback,
            parent,
        ):  # noqa: ANN001
            _ = (metric_label, session_params, default_params, parent)
            if metric_key == "psi":
                set_default_callback(
                    {
                        "time_resolution_s": 0.4,
                        "hop_s": 0.02,
                        "method": "multitaper",
                        "mt_bandwidth": 4.0,
                        "bands": [{"name": "theta", "start": 4.0, "end": 7.0}],
                        "selected_pairs": [["CH2", "CH1"]],
                    }
                )
                self.selected_params = {
                    "time_resolution_s": 0.3,
                    "hop_s": 0.015,
                    "method": "multitaper",
                    "mt_bandwidth": 3.0,
                    "bands": [{"name": "alpha", "start": 8.0, "end": 12.0}],
                    "selected_pairs": [["CH2", "CH1"]],
                }
            else:
                assert burst_baseline_annotations == ("Rest", "Task")
                set_default_callback(
                    {
                        "percentile": 80.0,
                        "baseline_keep": ["Rest"],
                        "min_cycles": 3.0,
                        "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
                        "selected_channels": ["CH2"],
                    }
                )
                self.selected_params = {
                    "percentile": 82.0,
                    "baseline_keep": ["Task"],
                    "min_cycles": 2.5,
                    "bands": [{"name": "gamma", "start": 30.0, "end": 55.0}],
                    "selected_channels": ["CH2"],
                }

        def exec(self) -> int:
            return QDialog.Accepted

    window._overrides["_create_tensor_metric_advance_dialog"] = (
        lambda self, **kwargs: _FakeTensorAdvanceDialog(**kwargs)
    )

    window._set_active_tensor_metric("psi")
    window._on_tensor_metric_advance()
    assert saved_defaults[0][0] == "psi"
    assert saved_defaults[0][1]["selected_pairs"] == [["CH2", "CH1"]]
    assert saved_defaults[0][1]["bands"] == [
        {"name": "theta", "start": 4.0, "end": 7.0}
    ]
    assert window._tensor_selected_pairs_by_metric["psi"] == (("CH2", "CH1"),)
    assert window._tensor_metric_params["psi"]["method"] == "multitaper"

    window._set_active_tensor_metric("burst")
    window._on_tensor_metric_advance()
    assert saved_defaults[1][0] == "burst"
    assert saved_defaults[1][1]["selected_channels"] == ["CH2"]
    assert saved_defaults[1][1]["baseline_keep"] == ["Rest"]
    assert saved_defaults[1][1]["bands"] == [
        {"name": "beta", "start": 13.0, "end": 30.0}
    ]
    assert window._tensor_selected_channels_by_metric["burst"] == ("CH2",)
    assert window._tensor_metric_params["burst"]["baseline_keep"] == ["Task"]
    assert window._tensor_metric_params["burst"]["bands"] == [
        {"name": "gamma", "start": 30.0, "end": 55.0}
    ]

    window.close()


def test_mainwindow_tensor_prepare_metric_default_payload_contains_full_snapshot(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path, steps=("finish",), enable_plots=False
    )

    window._tensor_available_channels = ("CH1", "CH2", "CH3")
    window._tensor_selected_channels_by_metric["raw_power"] = ("CH2", "CH3")
    raw_payload = window._tensor_prepare_metric_default_payload(
        "raw_power",
        {"method": "multitaper", "time_bandwidth": 4.0},
    )
    assert raw_payload["selected_channels"] == ["CH2", "CH3"]

    window._tensor_selected_pairs_by_metric["psi"] = (("CH2", "CH1"),)
    window._tensor_metric_params["psi"] = {
        "bands": [{"name": "theta", "start": 4.0, "end": 7.0}],
        "time_resolution_s": 0.5,
        "hop_s": 0.025,
        "method": "morlet",
    }
    psi_payload = window._tensor_prepare_metric_default_payload(
        "psi",
        {"method": "multitaper", "mt_bandwidth": 3.0},
    )
    assert psi_payload["selected_pairs"] == [["CH2", "CH1"]]
    assert psi_payload["bands"] == [{"name": "theta", "start": 4.0, "end": 7.0}]

    window._set_active_tensor_metric("psi")
    default_bands = window._tensor_effective_metric_defaults(
        "psi",
        context=context,
        available_channels=window._tensor_available_channels,
    ).get("bands")
    assert isinstance(default_bands, list)
    window.close()


def test_mainwindow_finish_apply_inherits_tensor_metric_notches_from_filter(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("raw", "filter"),
        enable_plots=False,
    )

    filter_log = preproc_step_log_path(resolver, "filter")
    write_run_log(
        filter_log,
        RunLogRecord(
            step="filter",
            completed=True,
            params={
                "low_freq": 1.0,
                "high_freq": 120.0,
                "notches": [50.0, 100.0],
                "notch_widths": [1.5, 2.5],
            },
            input_path=str(step_paths["raw"]),
            output_path=str(step_paths["filter"]),
            message="filter ready",
        ),
    )

    window._tensor_metric_params["raw_power"]["notches"] = [60.0]
    window._tensor_metric_params["raw_power"]["notch_widths"] = 5.0
    window._tensor_metric_params["psi"]["notches"] = []
    window._tensor_metric_params["psi"]["notch_widths"] = 2.0

    window._on_preproc_finish_apply()

    for metric_key in ("raw_power", "psi", "burst"):
        params = window._tensor_metric_params[metric_key]
        assert params["notches"] == [50.0, 100.0]
        assert params["notch_widths"] == [1.5, 2.5]

    write_run_log(
        filter_log,
        RunLogRecord(
            step="filter",
            completed=False,
            params={
                "low_freq": 1.0,
                "high_freq": 120.0,
                "notches": [50.0],
                "notch_widths": 2.0,
            },
            input_path=str(step_paths["raw"]),
            output_path=str(step_paths["filter"]),
            message="filter incomplete",
        ),
    )
    assert window._inherit_tensor_metric_notches_from_filter(context) is True
    assert window._tensor_metric_params["raw_power"]["notches"] == []
    assert window._tensor_metric_params["raw_power"]["notch_widths"] == 2.0

    window.close()


def test_mainwindow_tensor_advance_warning_only_on_save(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    warning_messages: list[str] = []

    window, _context, resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("finish",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": (
                lambda self, title, message: warning_messages.append(message) or 0
            )
        },
    )
    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=True,
            params={
                "low_freq": 1.0,
                "high_freq": 120.0,
                "notches": [50.0],
                "notch_widths": 2.0,
            },
            input_path=str(step_paths["finish"]),
            output_path=str(preproc_step_raw_path(resolver, "filter")),
            message="filter ready",
        ),
    )

    action_state = {
        "selected_action": "save",
        "selected_params": {
            "low_freq_hz": 1.0,
            "high_freq_hz": 100.0,
            "freq_step_hz": 0.5,
            "time_resolution_s": 0.5,
            "hop_s": 0.025,
            "method": "morlet",
            "min_cycles": 3.0,
            "max_cycles": None,
            "time_bandwidth": 1.0,
            "notches": [],
            "notch_widths": 2.0,
        },
    }

    class _FakeTensorAdvanceDialog:
        def __init__(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            self.selected_action = str(action_state["selected_action"])
            self.selected_params = dict(action_state["selected_params"])

        def exec(self) -> int:
            return QDialog.Accepted

    window._overrides["_create_tensor_metric_advance_dialog"] = (
        lambda self, **kwargs: _FakeTensorAdvanceDialog(**kwargs)
    )

    window._set_active_tensor_metric("raw_power")
    window._on_tensor_metric_advance()
    assert any(
        "Missing preprocess filter notch 50 Hz." in item for item in warning_messages
    )

    warning_messages.clear()
    action_state["selected_action"] = "set_default"
    window._on_tensor_metric_advance()
    assert warning_messages == []

    window.close()


def test_mainwindow_tensor_run_preflight_warning_lists_selected_metrics(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    window, _context, resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("finish",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
    )
    write_run_log(
        preproc_step_log_path(resolver, "filter"),
        RunLogRecord(
            step="filter",
            completed=True,
            params={
                "low_freq": 1.0,
                "high_freq": 120.0,
                "notches": [50.0],
                "notch_widths": 2.0,
            },
            input_path=str(step_paths["finish"]),
            output_path=str(preproc_step_raw_path(resolver, "filter")),
            message="filter ready",
        ),
    )

    assert window._tensor_metric_checks["raw_power"].isEnabled()
    assert window._tensor_metric_checks["coherence"].isEnabled()
    window._tensor_metric_checks["raw_power"].setChecked(True)
    window._tensor_metric_checks["coherence"].setChecked(True)
    app.processEvents()

    window._tensor_metric_params["raw_power"]["notches"] = []
    window._tensor_metric_params["raw_power"]["notch_widths"] = 2.0
    window._tensor_metric_params["coherence"]["notches"] = []
    window._tensor_metric_params["coherence"]["notch_widths"] = 2.0
    window._tensor_metric_params["psi"]["notches"] = []
    window._tensor_metric_params["psi"]["notch_widths"] = 2.0

    captured: dict[str, object] = {}

    def _fake_confirm(warnings_by_metric):  # noqa: ANN001
        captured["warnings_by_metric"] = dict(warnings_by_metric)
        return False

    window._overrides["_confirm_tensor_preflight_notch_warnings"] = (
        lambda self, warnings_by_metric: _fake_confirm(warnings_by_metric)
    )
    window._overrides["_launch_tensor_run_process"] = (
        lambda self, **kwargs: captured.__setitem__("launched", kwargs)
    )

    window._on_tensor_run()

    assert (
        window.statusBar().currentMessage() == "Build Tensor cancelled before launch."
    )
    assert "launched" not in captured
    assert captured["warnings_by_metric"] == {
        "raw_power": ["Missing preprocess filter notch 50 Hz."],
        "coherence": ["Missing preprocess filter notch 50 Hz."],
    }

    window.close()


def test_mainwindow_tensor_non_handler_bands_and_runtime_helpers(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("finish",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
    )

    with pytest.raises(ValueError, match="At least one band is required"):
        window._collect_tensor_bands([])

    with pytest.raises(ValueError, match="At least one band is required"):
        window._collect_tensor_bands([{"name": "", "start": 1.0, "end": 2.0}])

    with pytest.raises(ValueError, match="At least one band is required"):
        window._collect_tensor_bands([{"name": "alpha", "start": "bad", "end": 2.0}])

    with pytest.raises(ValueError, match="At least one band is required"):
        window._collect_tensor_bands([{"name": "alpha", "start": 0.0, "end": 2.0}])

    deduped = window._collect_tensor_bands(
        [
            {"name": "alpha", "start": 1.0, "end": 4.0},
            {"name": "alpha", "start": 5.0, "end": 8.0},
        ]
    )
    assert deduped == [{"name": "alpha", "start": 1.0, "end": 4.0}]

    overlap_bands = window._collect_tensor_bands(
        [
            {"name": "alpha", "start": 1.0, "end": 4.0},
            {"name": "beta", "start": 3.0, "end": 6.0},
        ]
    )
    assert overlap_bands == [
        {"name": "alpha", "start": 1.0, "end": 4.0},
        {"name": "beta", "start": 3.0, "end": 6.0},
    ]

    bands = window._collect_tensor_bands(
        [
            {"name": "alpha", "start": 1.0, "end": 4.0},
            {"name": "beta", "start": 6.0, "end": 9.0},
        ]
    )
    assert bands == [
        {"name": "alpha", "start": 1.0, "end": 4.0},
        {"name": "beta", "start": 6.0, "end": 9.0},
    ]

    assert window._tensor_low_freq_edit is not None
    assert window._tensor_high_freq_edit is not None
    assert window._tensor_step_edit is not None

    low_edit = window._tensor_low_freq_edit
    high_edit = window._tensor_high_freq_edit
    step_edit = window._tensor_step_edit

    window._overrides["_selected_tensor_metrics"] = lambda self: []
    with pytest.raises(ValueError, match="Select at least one metric"):
        window._collect_tensor_runtime_params(context)

    window._tensor_selected_pairs_by_metric["trgc"] = (("CH2", "CH1"),)
    pairs_map = window._collect_tensor_pairs_for_metrics(["trgc"])
    assert pairs_map["trgc"] == [("CH2", "CH1")]

    metric_notice = window._tensor_metric_notice_label
    assert metric_notice is not None
    window._set_active_tensor_metric("unknown_metric")
    assert metric_notice.text() == "Unknown metric."

    pending_metric = SimpleNamespace(
        key="pending_metric",
        display_name="Pending Metric",
        group_name="Mock Group",
        supported=False,
    )
    window._overrides["_stage_tensor_metric_specs"] = lambda self: (pending_metric,)
    window._set_active_tensor_metric("pending_metric")
    assert "Pending implementation" in metric_notice.text()
    assert window._tensor_advance_button is not None
    assert not window._tensor_advance_button.isEnabled()
    assert window._tensor_low_freq_edit is not None
    assert window._tensor_high_freq_edit is not None
    assert window._tensor_step_edit is not None
    assert window._tensor_time_resolution_edit is not None
    assert window._tensor_hop_edit is not None
    assert window._tensor_method_combo is not None
    assert window._tensor_freq_range_edit is not None
    assert window._tensor_bands_configure_button is not None
    assert window._tensor_percentile_edit is not None
    assert window._tensor_min_cycles_basic_edit is not None
    assert window._tensor_low_freq_edit.isHidden()
    assert window._tensor_high_freq_edit.isHidden()
    assert window._tensor_step_edit.isHidden()
    assert window._tensor_time_resolution_edit.isHidden()
    assert window._tensor_hop_edit.isHidden()
    assert window._tensor_method_combo.isHidden()
    assert window._tensor_freq_range_edit.isHidden()
    assert window._tensor_bands_configure_button.isHidden()
    assert window._tensor_percentile_edit.isHidden()
    assert window._tensor_min_cycles_basic_edit.isHidden()

    window._set_active_tensor_metric("raw_power")
    window._tensor_selected_channels_by_metric["raw_power"] = ()
    window._tensor_available_channels = ("CH0",)
    window._overrides["_read_channel_names_from_raw"] = lambda self, _path: [
        "CH1",
        "CH2",
    ]
    window._overrides["_tensor_load_default_channels"] = lambda self: ("CH2",)
    window._overrides["_tensor_load_default_pairs"] = lambda self, directed=False: ()
    window._refresh_tensor_channel_state(context)
    assert window._tensor_selected_channels_by_metric["raw_power"] == ("CH2",)

    window._overrides["_selected_tensor_metrics"] = lambda self: ["raw_power"]
    window._overrides["_validate_tensor_frequency_params_runtime"] = (
        lambda self, *args, **kwargs: (True, "", {})
    )

    low_edit.setText("-1")
    high_edit.setText("20")
    step_edit.setText("0.5")
    with pytest.raises(ValueError, match="low freq must be > 0"):
        window._collect_tensor_runtime_params(context)

    low_edit.setText("5")
    high_edit.setText("5")
    with pytest.raises(ValueError, match="greater than low freq"):
        window._collect_tensor_runtime_params(context)

    low_edit.setText("5")
    high_edit.setText("10")
    step_edit.setText("0")
    with pytest.raises(ValueError, match="step must be > 0"):
        window._collect_tensor_runtime_params(context)

    low_edit.setText("5")
    high_edit.setText("10")
    step_edit.setText("0.5")
    window._overrides["_validate_tensor_frequency_params_runtime"] = (
        lambda self, *args, **kwargs: (False, "freq invalid", {})
    )
    with pytest.raises(ValueError, match="freq invalid"):
        window._collect_tensor_runtime_params(context)

    window._overrides["_validate_tensor_frequency_params_runtime"] = (
        lambda self, *args, **kwargs: (True, "", {})
    )
    window._tensor_mask_edge_checkbox = None
    selected_metrics, mask_edge, metric_params_map = (
        window._collect_tensor_runtime_params(context)
    )
    assert selected_metrics == ["raw_power"]
    assert mask_edge is True
    assert metric_params_map["raw_power"]["low_freq_hz"] == 5.0
    assert metric_params_map["raw_power"]["high_freq_hz"] == 10.0
    assert metric_params_map["raw_power"]["freq_step_hz"] == 0.5

    window.close()


def test_mainwindow_tensor_metric_panel_shows_only_active_basic_rows(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, _context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path, steps=("finish",), enable_plots=False
    )

    assert window._tensor_active_metric_key == "raw_power"
    assert window._tensor_low_freq_edit is not None
    assert window._tensor_high_freq_edit is not None
    assert window._tensor_step_edit is not None
    assert window._tensor_time_resolution_edit is not None
    assert window._tensor_hop_edit is not None
    assert window._tensor_method_combo is not None
    assert window._tensor_freq_range_edit is not None
    assert window._tensor_bands_configure_button is not None
    assert window._tensor_percentile_edit is not None
    assert window._tensor_min_cycles_basic_edit is not None
    assert window._tensor_channels_button is not None
    assert window._tensor_pairs_button is not None
    assert window._tensor_advance_button is not None
    assert window._tensor_metric_notice_label is not None

    row_widgets = {
        "low_freq_hz": window._tensor_low_freq_edit,
        "high_freq_hz": window._tensor_high_freq_edit,
        "freq_step_hz": window._tensor_step_edit,
        "time_resolution_s": window._tensor_time_resolution_edit,
        "hop_s": window._tensor_hop_edit,
        "method": window._tensor_method_combo,
        "freq_range_hz": window._tensor_freq_range_edit,
        "bands": window._tensor_bands_configure_button,
        "percentile": window._tensor_percentile_edit,
        "min_cycles": window._tensor_min_cycles_basic_edit,
    }

    def _assert_visible_rows(expected: set[str]) -> None:
        for key, widget in row_widgets.items():
            assert widget.isHidden() is (key not in expected)
        assert not window._tensor_advance_button.isHidden()
        assert not window._tensor_metric_notice_label.isHidden()

    def _assert_selector_visibility(
        *, channels_visible: bool, pairs_visible: bool
    ) -> None:
        assert window._tensor_channels_button.isHidden() is (not channels_visible)
        assert window._tensor_pairs_button.isHidden() is (not pairs_visible)

    _assert_visible_rows(
        {"low_freq_hz", "high_freq_hz", "freq_step_hz", "time_resolution_s", "hop_s"}
    )
    _assert_selector_visibility(channels_visible=True, pairs_visible=False)

    window._tensor_selected_channels_by_metric["raw_power"] = ("CH1",)
    window._tensor_selected_pairs_by_metric["coherence"] = (("CH1", "CH2"),)

    window._set_active_tensor_metric("periodic_aperiodic")
    _assert_visible_rows(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
            "freq_range_hz",
        }
    )
    _assert_selector_visibility(channels_visible=True, pairs_visible=False)

    for metric_key in ("coherence", "plv", "ciplv", "pli", "wpli", "trgc"):
        window._set_active_tensor_metric(metric_key)
        _assert_visible_rows(
            {
                "low_freq_hz",
                "high_freq_hz",
                "freq_step_hz",
                "time_resolution_s",
                "hop_s",
            }
        )
        _assert_selector_visibility(channels_visible=False, pairs_visible=True)

    window._set_active_tensor_metric("psi")
    _assert_visible_rows({"bands", "time_resolution_s", "hop_s"})
    _assert_selector_visibility(channels_visible=False, pairs_visible=True)

    window._set_active_tensor_metric("burst")
    _assert_visible_rows({"bands", "percentile"})
    _assert_selector_visibility(channels_visible=True, pairs_visible=False)

    window._set_active_tensor_metric("coherence")
    assert window._tensor_selected_pairs_by_metric["coherence"] == (("CH1", "CH2"),)

    window._set_active_tensor_metric("raw_power")
    assert window._tensor_selected_channels_by_metric["raw_power"] == ("CH1",)

    window.close()


def test_mainwindow_non_handler_ui_defaults_and_annotation_helpers(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, _resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("finish",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
    )

    active_before = window._active_stage_key
    window.route_to_stage("missing-stage-key")
    assert window._active_stage_key == active_before

    tensor_button = window._stage_buttons["tensor"]
    tensor_button.setEnabled(False)
    window.route_to_stage("tensor")
    assert window._active_stage_key == active_before
    tensor_button.setEnabled(True)

    left_widget = window._left_column_widget
    window._left_column_widget = None
    window._update_left_column_width()
    window._left_column_widget = left_widget

    class _Font:
        def __init__(self, value: float) -> None:
            self._value = value

        def pointSizeF(self) -> float:
            return self._value

    class _ProbeButton:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def font(self) -> _Font:
            return _Font(-1.0)

        def deleteLater(self) -> None:
            return

    window._overrides["font"] = lambda self: _Font(-1.0)
    main_window_layout_module.apply_panel_title_style(window, button_cls=_ProbeButton)
    assert "font-size: 10.0pt" in window.styleSheet()

    icon_button = main_window_module.QPushButton("Icon Button", window)
    icon_button.setIcon(window.style().standardIcon(QStyle.SP_DialogOpenButton))
    window._enforce_button_text_fit()

    window._busy_label = None
    window._on_busy_tick()
    window._render_busy_message()

    def _raise_busy() -> tuple[bool, str]:
        raise RuntimeError("busy-error")

    with pytest.raises(RuntimeError, match="busy-error"):
        window._run_with_busy("Busy", _raise_busy)

    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: {
            main_window_module.PREPROC_FILTER_DEFAULTS_KEY: {"epoch_dur": 0.0},
            main_window_module.PREPROC_FILTER_BASIC_DEFAULTS_KEY: {"h_freq": 0.0},
            main_window_module.PREPROC_VIZ_PSD_DEFAULTS_KEY: {"fmin": -1.0},
            main_window_module.PREPROC_VIZ_TFR_DEFAULTS_KEY: {"fmin": 0.0},
        }
    )
    filter_defaults = window._load_filter_advance_defaults()
    filter_basic_defaults = window._load_filter_basic_defaults()
    psd_defaults = window._load_preproc_viz_psd_defaults()
    tfr_defaults = window._load_preproc_viz_tfr_defaults()
    assert filter_defaults == main_window_module.default_filter_advance_params()
    assert (
        filter_basic_defaults
        == main_window_module.default_preproc_filter_basic_params()
    )
    assert psd_defaults == main_window_module.default_preproc_viz_psd_params()
    assert tfr_defaults == main_window_module.default_preproc_viz_tfr_params()

    writes: list[tuple[str, dict[str, object]]] = []
    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: []
    )
    window._config_store.write_yaml = (  # type: ignore[method-assign]
        lambda path, payload: writes.append((str(path), dict(payload)))
    )
    window._save_filter_advance_defaults(filter_defaults)
    window._save_filter_basic_defaults(filter_basic_defaults)
    window._save_preproc_viz_psd_defaults(psd_defaults)
    window._save_preproc_viz_tfr_defaults(tfr_defaults)
    assert writes and len(writes) == 4

    mock_metric = SimpleNamespace(
        key="mock_metric",
        display_name="Mock Metric",
        group_name="Mock Group",
        supported=False,
    )
    window._overrides["_stage_tensor_metric_specs"] = lambda self: (mock_metric,)
    block = window._build_tensor_metrics_block()
    _ = block
    mock_checkbox = window._tensor_metric_checks["mock_metric"]
    assert not mock_checkbox.isEnabled()
    assert "Pending implementation" in mock_checkbox.toolTip()

    original_table = window._preproc_annotations_table
    window._preproc_annotations_table = None
    assert window._annotations_table_rows() == ([], [])
    window._highlight_annotation_rows([0])
    window._append_annotation_rows(
        [{"description": "evt", "onset": 0.0, "duration": 1.0}]
    )

    window._preproc_annotations_table = original_table
    assert window._preproc_annotations_table is not None
    table = window._preproc_annotations_table
    table.setRowCount(1)
    rows, invalid_rows = window._annotations_table_rows()
    assert rows == []
    assert invalid_rows == []
    window._highlight_annotation_rows([0])
    assert table.item(0, 0) is not None
    assert table.item(0, 1) is not None
    assert table.item(0, 2) is not None

    table.setRowCount(0)
    window._append_annotation_rows(
        [{"description": "evt", "onset": 0.1, "duration": 1.0}]
    )
    assert table.rowCount() == 1

    window.close()


def test_mainwindow_alignment_features_and_stage_helper_guards(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    window, context, resolver, slug = _build_alignment_window(
        tmp_path, enable_plots=False
    )

    alignment_list = window._alignment_paradigm_list
    assert alignment_list is not None
    window._alignment_paradigm_list = None
    assert window._current_alignment_paradigm_slug() == slug
    window._set_shared_stage_trial_slug(None)
    assert window._current_alignment_paradigm_slug() is None
    window._reload_alignment_paradigms(preferred_slug=slug)
    window._alignment_paradigm_list = alignment_list

    item = alignment_list.currentItem()
    assert item is not None
    item.setData(Qt.UserRole, 123)
    assert window._current_alignment_paradigm_slug() is None
    item.setData(Qt.UserRole, slug)

    epoch_table = window._alignment_epoch_table
    assert epoch_table is not None
    window._alignment_epoch_table = None
    window._set_alignment_epoch_rows([{"epoch_index": 0, "pick": True}])
    assert window._selected_alignment_epoch_indices() == []
    window._alignment_epoch_table = epoch_table

    window._alignment_epoch_rows = [{"epoch_index": 7, "pick": True}]
    epoch_table.setRowCount(1)
    assert window._selected_alignment_epoch_indices() == []

    metric_combo = window._alignment_epoch_metric_combo
    assert metric_combo is not None
    window._alignment_epoch_metric_combo = None
    window._refresh_alignment_metric_combo()
    window._alignment_epoch_metric_combo = metric_combo

    paradigm_dir = resolver.alignment_paradigm_dir(slug)
    (paradigm_dir / "metric_a").mkdir(parents=True, exist_ok=True)
    (paradigm_dir / "metric_a" / "tensor_warped.pkl").write_text("x", encoding="utf-8")
    (paradigm_dir / "metric_b").mkdir(parents=True, exist_ok=True)
    (paradigm_dir / "metric_b" / "tensor_warped.pkl").write_text("x", encoding="utf-8")
    metric_combo.clear()
    metric_combo.addItem("metric_b", "metric_b")
    metric_combo.setCurrentIndex(0)
    window._refresh_alignment_metric_combo()
    assert metric_combo.currentData() == "metric_b"

    features_list = window._features_paradigm_list
    assert features_list is not None
    window._features_paradigm_list = None
    assert window._current_features_paradigm_slug() is None
    window._reload_features_paradigms(preferred_slug=slug)
    window._features_paradigm_list = features_list

    window._features_paradigms = [{"slug": 123}]
    features_list.setCurrentRow(0)
    assert window._current_features_paradigm_slug() is None
    window._reload_features_paradigms(preferred_slug=slug)

    edit = main_window_module.QLineEdit()
    edit.setText("Custom Label")
    assert MainWindow._resolve_plot_label(edit, "Default") == "Custom Label"

    features_table = window._features_available_table
    assert features_table is not None
    window._features_available_table = None
    window._refresh_features_available_files()
    window._features_available_table = features_table

    window._features_paradigms = [{"name": "Gait", "slug": slug}]
    features_list.clear()
    features_list.addItem("Gait")
    features_item = features_list.item(0)
    assert features_item is not None
    features_item.setData(Qt.UserRole, slug)
    features_list.setCurrentRow(0)
    window._set_shared_stage_trial_slug(slug)
    _seed_alignment_trial_finish_ready(resolver, slug)
    window._refresh_stage_states_from_context()
    window._refresh_features_available_files()

    feature_root = resolver.features_root / slug / "mock" / "sub"
    feature_root.mkdir(parents=True, exist_ok=True)
    feature_path = feature_root / "alpha-raw.pkl"
    save_pkl(pd.DataFrame({"Value": [pd.Series([1.0, 2.0])]}), feature_path)
    window._refresh_features_available_files()
    assert features_table.rowCount() >= 1
    features_table.setCurrentCell(0, 0)
    window._refresh_features_available_files()
    assert features_table.currentRow() >= 0

    window._refresh_stage_states_from_context()
    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._refresh_stage_states_from_context()

    warnings: list[str] = []
    viz_window, viz_context, _viz_resolver, viz_steps = (
        _build_window_with_preproc_steps(
            tmp_path / "viz",
            steps=("raw", "filter"),
            enable_plots=False,
            window_cls=OverrideMainWindow,
            overrides={
                "_show_warning": lambda self, title, message: warnings.append(message)
                or 0
            },
        )
    )
    viz_combo = viz_window._preproc_viz_step_combo
    assert viz_combo is not None
    raw_idx = viz_combo.findData("raw")
    assert raw_idx >= 0
    viz_window._preproc_viz_last_step = None
    viz_steps["raw"].unlink()

    viz_window._on_preproc_viz_step_changed(raw_idx)
    assert any("Falling back" in item for item in warnings)

    _ = viz_context
    viz_window.close()
    window.close()


def test_mainwindow_features_available_files_missing_root_branch(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    missing_context = RecordContext(
        project_root=tmp_path / "missing_project_root",
        subject="sub-001",
        record="runA",
    )
    window, context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_record_context": lambda self: missing_context,
            "_current_features_paradigm_slug": lambda self: slug,
        },
    )

    assert window._features_available_table is not None
    _ = context

    derivatives_slug_root = resolver.features_root / slug
    if derivatives_slug_root.exists():
        for child in derivatives_slug_root.rglob("*"):
            _ = child

    window._refresh_features_available_files()
    assert window._features_files == []
    assert window._features_filtered_files == []
    window.close()


def test_mainwindow_alignment_stage_handler_residual_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    normalized_params = {
        "annotations": ["event"],
        "mode": "exact",
        "duration_range": [0.0, 100.0],
        "drop_bad": False,
        "pad_s": 0.0,
        "sample_rate": 0.4,
    }
    window, context, _resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_validate_alignment_method_params_runtime": (
                lambda self, *_args, **_kwargs: (True, dict(normalized_params), "")
            ),
            "_update_alignment_paradigm_runtime": (
                lambda self, *_args, **_kwargs: (True, "updated")
            ),
        },
    )
    assert window._alignment_paradigm_list is not None
    assert window._alignment_method_combo is not None
    assert not hasattr(window, "_features_phases_table")

    window._alignment_paradigm_list.setCurrentRow(0)
    window._alignment_paradigms[0]["method_params"] = "bad"

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_alignment_paradigm_selected(0)
    assert window._alignment_epoch_rows == []

    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record

    method_idx = window._alignment_method_combo.findData("stack_warper")
    assert method_idx >= 0
    window._alignment_method_combo.setCurrentIndex(method_idx)
    window._alignment_paradigms[0]["method"] = "pad_warper"
    window._on_alignment_method_changed(method_idx)

    class _RejectedParamsDialog:
        selected_params = None

        def __init__(
            self,
            *,
            method_key: str,
            session_params: dict[str, object],
            annotation_labels: list[str],
            config_store: AppConfigStore,
            parent: MainWindow,
        ) -> None:
            _ = (method_key, session_params, annotation_labels, config_store, parent)

        def exec(self) -> int:
            return QDialog.Rejected

    window._overrides["_create_alignment_method_params_dialog"] = (
        lambda self, **kwargs: _RejectedParamsDialog(**kwargs)
    )
    window._alignment_paradigms[0]["method"] = "linear_warper"
    window._alignment_paradigms[0]["method_params"] = "bad"
    window._on_alignment_method_params()

    window._alignment_paradigms[0]["method"] = "stack_warper"
    window._alignment_paradigms[0]["method_params"] = "bad"
    window._stage_states["tensor"] = "green"
    window._overrides["_run_with_busy"] = lambda self, _label, _work: (
        True,
        "runtime ok",
        [],
    )
    window._on_alignment_run()
    assert "Align Epochs OK: runtime ok" in window.statusBar().currentMessage()

    window.close()


def test_mainwindow_preproc_stage_handler_residual_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    opened_plots: list[tuple[str, str, str | None]] = []
    window, context, _resolver, step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=(
            "filter",
            "annotations",
            "bad_segment_removal",
            "ecg_artifact_removal",
            "finish",
        ),
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_open_mne_raw_plot": (
                lambda self, raw_path, title_prefix, autosave_step=None: opened_plots.append(
                    (Path(raw_path).name, title_prefix, autosave_step)
                )
            )
        },
    )

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_preproc_raw_plot()
    assert "Raw Plot unavailable" in window.statusBar().currentMessage()

    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record
    window._overrides["_run_with_busy"] = lambda self, _label, _work: (
        False,
        "bootstrap failed",
    )
    window._on_preproc_raw_plot()
    assert "Raw Plot failed: bootstrap failed" in window.statusBar().currentMessage()

    window._on_preproc_filter_plot()
    window._on_preproc_annotations_plot()
    window._on_preproc_bad_segment_plot()
    window._on_preproc_ecg_plot()
    window._on_preproc_finish_plot()
    assert [item[1] for item in opened_plots] == [
        "Filter",
        "Annotations",
        "Bad Segment",
        "ECG",
        "Finish",
    ]

    window._preproc_ecg_available_channels = ("CH1", "CH2")
    window._preproc_ecg_selected_channels = ("CH1",)
    window._overrides["_run_channel_selector"] = lambda self, **_kwargs: None
    window._on_preproc_ecg_channels_select()
    assert window._preproc_ecg_selected_channels == ("CH1",)

    window._overrides["_current_preproc_viz_source"] = lambda self: (
        "filter",
        step_paths["filter"],
    )
    saved_tfr_defaults: list[dict[str, object]] = []
    window._overrides["_save_preproc_viz_tfr_defaults"] = (
        lambda self, params: saved_tfr_defaults.append(dict(params))
    )

    class _PsdSaveDialog:
        def __init__(
            self,
            *,
            mode: str,
            session_params: dict[str, object],
            default_params: dict[str, object],
            set_default_callback=None,  # noqa: ANN001
            parent: MainWindow,
        ) -> None:
            _ = (session_params, default_params, set_default_callback, parent)
            assert mode == "psd"
            self.selected_action = "save"
            self.selected_params = {
                "fmin": 2.0,
                "fmax": 40.0,
                "n_fft": 256,
                "average": True,
            }

        def exec(self) -> int:
            return QDialog.Accepted

    window._overrides["_create_qc_advance_dialog"] = (
        lambda self, **kwargs: _PsdSaveDialog(**kwargs)
    )
    window._on_preproc_viz_psd_advance()
    assert "session parameters updated" in window.statusBar().currentMessage().lower()

    class _TfrRejectedDialog:
        selected_params = None
        selected_action = "save"

        def __init__(
            self,
            *,
            mode: str,
            session_params: dict[str, object],
            default_params: dict[str, object],
            set_default_callback=None,  # noqa: ANN001
            parent: MainWindow,
        ) -> None:
            _ = (mode, session_params, default_params, set_default_callback, parent)

        def exec(self) -> int:
            return QDialog.Accepted

    window._overrides["_create_qc_advance_dialog"] = (
        lambda self, **kwargs: _TfrRejectedDialog(**kwargs)
    )
    tfr_before = dict(window._preproc_viz_tfr_params)
    window._on_preproc_viz_tfr_advance()
    assert window._preproc_viz_tfr_params == tfr_before

    class _TfrSetDefaultDialog:
        def __init__(
            self,
            *,
            mode: str,
            session_params: dict[str, object],
            default_params: dict[str, object],
            set_default_callback=None,  # noqa: ANN001
            parent: MainWindow,
        ) -> None:
            _ = (session_params, default_params, parent)
            self._set_default_callback = set_default_callback
            assert mode == "tfr"
            self.selected_action = "set_default"
            self.selected_params = {
                "fmin": 3.0,
                "fmax": 45.0,
                "n_freqs": 24,
                "decim": 3,
            }

        def exec(self) -> int:
            if self._set_default_callback is not None:
                self._set_default_callback(dict(self.selected_params))
            return QDialog.Accepted

    window._overrides["_create_qc_advance_dialog"] = (
        lambda self, **kwargs: _TfrSetDefaultDialog(**kwargs)
    )
    window._on_preproc_viz_tfr_advance()
    assert saved_tfr_defaults[-1]["decim"] == 3
    assert "session parameters updated" in window.statusBar().currentMessage().lower()

    window.close()


def test_mainwindow_features_handler_deep_branches(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []
    window, context, resolver, slug = _build_alignment_window(
        tmp_path,
        enable_plots=True,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0
        },
    )

    assert (
        MainWindow._default_plot_labels_for_derived_type("spectral")[0]
        == "Frequency (Hz)"
    )
    assert (
        MainWindow._default_plot_labels_for_derived_type("trace")[0] == "Percent / Time"
    )
    assert MainWindow._parse_derived_type_from_stem("nostem") == ""
    assert MainWindow._resolve_plot_label(None, "default") == "default"

    assert window._features_available_table is not None
    available_table = window._features_available_table
    window._features_available_table = None
    assert window._selected_features_file() is None
    window._features_available_table = available_table

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_features_run_extract()
    assert "select context and trial" in window.statusBar().currentMessage().lower()

    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record
    _seed_alignment_trial_finish_ready(resolver, slug)
    window._refresh_stage_states_from_context()

    deriv_root = resolver.features_root / slug / "metric_a"
    deriv_root.mkdir(parents=True, exist_ok=True)

    (deriv_root / "bad-load-raw.pkl").write_text("not a pickle", encoding="utf-8")
    save_pkl({"Value": 1}, deriv_root / "non-df-raw.pkl")
    save_pkl(pd.DataFrame({"Other": [1]}), deriv_root / "no-value-raw.pkl")
    save_pkl(pd.DataFrame({"Value": [1.0, 2.0]}), deriv_root / "scalar-only-raw.pkl")
    save_pkl(
        pd.DataFrame({"Value": [None, float("nan")]}),
        deriv_root / "empty-only-raw.pkl",
    )
    save_pkl(
        pd.DataFrame({"Value": [pd.Series([1.0, 2.0])]}), deriv_root / "dup-raw.pkl"
    )
    save_pkl(
        pd.DataFrame({"Value": [pd.Series([1.0, 2.0, 3.0])]}),
        deriv_root / "keep-trace.pkl",
    )

    window._refresh_features_available_files()
    names = {Path(row["path"]).name for row in window._features_files}
    assert "non-df-raw.pkl" not in names
    assert "no-value-raw.pkl" not in names
    assert "scalar-only-raw.pkl" in names
    assert "empty-only-raw.pkl" in names
    assert "dup-raw.pkl" in names
    assert "keep-trace.pkl" in names
    duplicate_rows = [
        row
        for row in window._features_files
        if row["feature"] == "raw" and row.get("property") == "dup"
    ]
    assert len(duplicate_rows) == 1

    assert window._features_filter_feature_edit is not None

    window._features_filter_feature_edit.setText("keep-trace")
    window._refresh_features_available_files()
    assert all(
        "keep-trace" in str(row.get("stem", "")).lower()
        for row in window._features_filtered_files
    )
    window._features_filter_feature_edit.setText("")
    window._on_features_refresh_files()
    assert "refreshed" in window.statusBar().currentMessage().lower()

    plot_payload_state: dict[str, object] = {
        "payload": pd.DataFrame({"Value": [pd.Series([1.0, 2.0])]}),
    }
    plot_path = resolver.features_root / slug / "plot" / "plot-raw.pkl"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.touch(exist_ok=True)
    window._features_filtered_files = [
        {
            "path": plot_path,
            "derived_type": "raw",
            "display_feature": "raw",
            "feature": "raw",
            "property": "plot",
            "relative_stem": "plot/plot-raw",
            "stem": "plot-raw",
            "metric": "plot",
        }
    ]
    available_table.setRowCount(1)
    available_table.setItem(0, 0, QTableWidgetItem("plot"))
    available_table.setItem(0, 1, QTableWidgetItem("raw"))
    available_table.setItem(0, 2, QTableWidgetItem("plot"))
    available_table.setCurrentCell(0, 0)

    shown_figures: list[str] = []
    tightened_figures: list[str] = []

    class _FakeFigure:
        def __init__(self, kind: str) -> None:
            self.kind = kind

        def show(self) -> None:
            shown_figures.append(self.kind)

        def savefig(self, path: Path | str, **_kwargs) -> None:
            Path(path).write_text("fig", encoding="utf-8")

    window._overrides["_load_pickle"] = lambda self, _path: plot_payload_state[
        "payload"
    ]
    window._overrides["_plot_single_effect_series"] = (
        lambda self, *_args, **_kwargs: _FakeFigure("series")
    )
    window._overrides["_plot_single_effect_df"] = (
        lambda self, *_args, **_kwargs: _FakeFigure("df")
    )
    window._overrides["_plot_single_effect_scalar"] = (
        lambda self, *_args, **_kwargs: _FakeFigure("scalar")
    )
    window._overrides["_tighten_features_plot_figure"] = (
        lambda self, fig: tightened_figures.append(fig.kind)
    )

    saved_filtered_files = list(window._features_filtered_files)
    window._features_filtered_files = []
    window._on_features_plot()
    assert any(
        "Select exactly one available feature file first." in item for item in warnings
    )
    window._features_filtered_files = saved_filtered_files
    available_table.setCurrentCell(0, 0)

    window._enable_plots = False
    window._on_features_plot()
    window._enable_plots = True
    assert shown_figures == []

    plot_payload_state["payload"] = "bad-payload"
    window._on_features_plot()
    plot_payload_state["payload"] = pd.DataFrame({"Other": [1]})
    window._on_features_plot()
    plot_payload_state["payload"] = pd.DataFrame({"Value": [1.0]})
    window._on_features_plot()
    plot_payload_state["payload"] = pd.DataFrame({"Value": [{"x": 1}]})
    window._on_features_plot()
    plot_payload_state["payload"] = pd.DataFrame(
        {"Value": [pd.Series([1.0]), pd.DataFrame(np.ones((2, 2), dtype=float))]}
    )
    window._on_features_plot()
    plot_payload_state["payload"] = pd.DataFrame({"Value": [None, np.nan]})
    window._on_features_plot()
    plot_payload_state["payload"] = pd.DataFrame({"Value": [pd.Series([1.0, 2.0])]})
    window._on_features_plot()
    plot_payload_state["payload"] = pd.DataFrame(
        {"Value": [pd.DataFrame(np.ones((2, 2), dtype=float))]}
    )
    window._on_features_plot()
    assert shown_figures == ["scalar", "series", "df"]
    assert tightened_figures == ["scalar", "series", "df"]
    assert any("Plot failed" in item for item in warnings)

    # plot export requires a plotted figure/data first
    assert window._features_plot_export_button is not None
    window._overrides["_save_file_name"] = lambda self, *_args, **_kwargs: (
        str((tmp_path / "plot-export.png").resolve()),
        "PNG (*.png)",
    )
    window._on_features_plot_export()
    assert (tmp_path / "plot-export.png").exists()
    assert (tmp_path / "plot-export.pkl").exists()
    assert (tmp_path / "plot-export.xlsx").exists() or any(
        "Export failed" in item for item in warnings
    )

    window.close()


def test_mainwindow_preproc_handler_guard_branches(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])
    warnings: list[str] = []
    window, context, resolver, _step_paths = _build_window_with_preproc_steps(
        tmp_path,
        steps=("raw",),
        enable_plots=False,
        window_cls=OverrideMainWindow,
        overrides={
            "_show_warning": lambda self, title, message: warnings.append(message) or 0
        },
    )

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_preproc_filter_advance()
    assert "Filter Advance unavailable" in window.statusBar().currentMessage()
    window._on_preproc_filter_apply()
    assert "Filter Apply unavailable" in window.statusBar().currentMessage()
    window._on_preproc_filter_plot()
    assert "Filter Plot unavailable" in window.statusBar().currentMessage()
    window._on_preproc_annotations_save()
    assert "Annotations Apply unavailable" in window.statusBar().currentMessage()
    window._on_preproc_annotations_plot()
    assert "Annotations Plot unavailable" in window.statusBar().currentMessage()
    window._on_preproc_bad_segment_apply()
    assert "Bad Segment Apply unavailable" in window.statusBar().currentMessage()
    window._on_preproc_bad_segment_plot()
    assert "Bad Segment Plot unavailable" in window.statusBar().currentMessage()
    window._on_preproc_ecg_apply()
    assert "ECG Apply unavailable" in window.statusBar().currentMessage()
    window._on_preproc_ecg_plot()
    assert "ECG Plot unavailable" in window.statusBar().currentMessage()
    window._on_preproc_finish_apply()
    assert "Finish Apply unavailable" in window.statusBar().currentMessage()
    window._on_preproc_finish_plot()
    assert "Finish Plot unavailable" in window.statusBar().currentMessage()

    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record

    filter_advance_state: dict[str, object] = {
        "exec_result": QDialog.Rejected,
        "selected_params": None,
        "selected_action": "save",
        "trigger_restore": False,
    }
    saved_defaults: list[dict[str, object]] = []
    saved_basic_defaults: list[dict[str, object]] = []

    class _FakeFilterAdvanceDialog:
        def __init__(
            self,
            *,
            session_params,  # noqa: ANN001
            default_params,  # noqa: ANN001
            set_default_callback=None,  # noqa: ANN001
            parent,  # noqa: ANN001
        ):
            _ = (session_params, default_params, parent)
            self._set_default_callback = set_default_callback
            payload = filter_advance_state["selected_params"]
            self.selected_params = (
                dict(payload) if isinstance(payload, dict) else payload
            )
            self.selected_action = str(filter_advance_state["selected_action"])
            self._restore_callback = None

        def set_restore_callback(self, callback):  # noqa: ANN001
            self._restore_callback = callback

        def exec(self) -> int:
            if filter_advance_state["trigger_restore"] and self._restore_callback:
                self._restore_callback()
            if (
                self.selected_action == "set_default"
                and self._set_default_callback is not None
                and isinstance(self.selected_params, dict)
            ):
                self._set_default_callback(dict(self.selected_params))
            return int(filter_advance_state["exec_result"])

    window._overrides["_create_filter_advance_dialog"] = (
        lambda self, **kwargs: _FakeFilterAdvanceDialog(**kwargs)
    )
    window._overrides["_save_filter_advance_defaults"] = (
        lambda self, params: saved_defaults.append(dict(params))
    )
    window._overrides["_save_filter_basic_defaults"] = (
        lambda self, params: saved_basic_defaults.append(dict(params))
    )
    window._overrides["_load_filter_basic_defaults"] = lambda self: {
        "notches": [50.0, 100.0],
        "l_freq": 1.0,
        "h_freq": 200.0,
    }

    assert window._preproc_filter_notches_edit is not None
    assert window._preproc_filter_low_freq_edit is not None
    assert window._preproc_filter_high_freq_edit is not None
    window._preproc_filter_notches_edit.setText("60")
    window._preproc_filter_low_freq_edit.setText("2")
    window._preproc_filter_high_freq_edit.setText("120")
    filter_advance_state["trigger_restore"] = True
    window._on_preproc_filter_advance()
    assert window._preproc_filter_notches_edit.text() == "50,100"
    assert window._preproc_filter_low_freq_edit.text() == "1"
    assert window._preproc_filter_high_freq_edit.text() == "200"
    filter_advance_state["trigger_restore"] = False
    filter_advance_state["exec_result"] = QDialog.Accepted
    window._on_preproc_filter_advance()
    filter_advance_state["selected_params"] = {
        "notch_widths": [2.0],
        "epoch_dur": 2.0,
        "p2p_thresh": 8.0,
        "autoreject_correct_factor": 1.5,
    }
    filter_advance_state["selected_action"] = "set_default"
    window._on_preproc_filter_advance()
    assert saved_defaults[-1]["epoch_dur"] == 2.0
    assert saved_basic_defaults[-1]["l_freq"] == 1.0
    assert saved_basic_defaults[-1]["h_freq"] == 200.0
    filter_advance_state["selected_action"] = "save"
    window._on_preproc_filter_advance()
    assert "session parameters updated" in window.statusBar().currentMessage().lower()

    def _raise_filter_params() -> tuple[list[float], float, float]:
        raise ValueError("bad filter")

    window._overrides["_collect_filter_runtime_params"] = (
        lambda self: _raise_filter_params()
    )
    window._on_preproc_filter_apply()
    assert any("Invalid filter parameters" in item for item in warnings)
    assert "invalid parameters (bad filter)" in window.statusBar().currentMessage()

    window._on_preproc_filter_plot()
    assert "filter/raw.fif is missing" in window.statusBar().currentMessage()

    assert window._preproc_annotations_table is not None
    annotations_table = window._preproc_annotations_table
    annotations_table.setRowCount(1)
    annotations_table.setItem(0, 0, QTableWidgetItem(""))
    annotations_table.setItem(0, 1, QTableWidgetItem("bad"))
    annotations_table.setItem(0, 2, QTableWidgetItem("1"))
    window._on_preproc_annotations_save()
    assert any("Invalid rows highlighted" in item for item in warnings)
    annotations_log = preproc_step_log_path(resolver, "annotations")
    assert indicator_from_log(annotations_log) == "yellow"

    configure_state: dict[str, object] = {
        "exec_result": QDialog.Rejected,
        "selected_rows": (),
    }

    class _FakeAnnotationConfigureDialog:
        def __init__(self, *, session_rows, project_root, parent):  # noqa: ANN001
            _ = (session_rows, project_root, parent)
            self._selected_rows = tuple(configure_state["selected_rows"])

        @property
        def selected_rows(self):  # noqa: ANN201
            return self._selected_rows

        def exec(self) -> int:
            return int(configure_state["exec_result"])

    window._overrides["_create_annotation_configure_dialog"] = (
        lambda self, **kwargs: _FakeAnnotationConfigureDialog(**kwargs)
    )
    window._on_preproc_annotations_edit()
    configure_state["exec_result"] = QDialog.Accepted
    configure_state["selected_rows"] = (
        {"description": "evt", "onset": 0.0, "duration": 1.0},
    )
    window._on_preproc_annotations_edit()
    assert annotations_table.rowCount() == 1
    assert annotations_table.item(0, 0).text() == "evt"

    annotations_table_ref = window._preproc_annotations_table
    window._preproc_annotations_table = None
    window._on_preproc_annotations_edit()
    window._preproc_annotations_table = annotations_table_ref
    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._mark_annotations_validation_failure(action="Save", invalid_rows=[0])
    window._current_project = context.project_root
    window._current_subject = context.subject
    window._current_record = context.record

    window._on_preproc_annotations_plot()
    assert "annotations/raw.fif is missing" in window.statusBar().currentMessage()
    window._on_preproc_bad_segment_plot()
    assert (
        "bad_segment_removal/raw.fif is missing" in window.statusBar().currentMessage()
    )
    window._preproc_ecg_selected_channels = ()
    window._on_preproc_ecg_apply()
    assert any("Select at least one ECG pick channel." in item for item in warnings)
    window._on_preproc_ecg_plot()
    assert (
        "ecg_artifact_removal/raw.fif is missing" in window.statusBar().currentMessage()
    )
    window._on_preproc_finish_plot()
    assert "finish/raw.fif is missing" in window.statusBar().currentMessage()

    window.close()
