"""Tests for Align-Epochs service helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app import alignment_service
from lfptensorpipe.app.alignment_service import (
    alignment_epoch_inspector_state,
    alignment_method_panel_state,
    alignment_trial_raw_table_path,
    create_alignment_paradigm,
    delete_alignment_paradigm,
    finish_alignment_epochs,
    load_alignment_annotation_labels,
    load_alignment_epoch_picks,
    load_alignment_epoch_rows,
    load_alignment_paradigms,
    persist_alignment_epoch_picks,
    run_align_epochs,
    save_alignment_paradigms,
    update_alignment_paradigm,
    validate_alignment_method_params,
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
    indicator_from_log,
    read_run_log,
    read_ui_state,
    write_ui_state,
    write_run_log,
)
from lfptensorpipe.app.tensor_service import (
    tensor_metric_log_path,
    tensor_metric_tensor_path,
)
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl


def _context(project: Path) -> RecordContext:
    return RecordContext(project_root=project, subject="sub-001", record="runA")


def _localize_log_path(context: RecordContext) -> Path:
    return (
        context.project_root
        / "derivatives"
        / "lfptensorpipe"
        / context.subject
        / context.record
        / "localize"
        / "lfptensorpipe_log.json"
    )


def _mark_localize_ready(context: RecordContext) -> None:
    log_path = _localize_log_path(context)
    write_run_log(
        log_path,
        RunLogRecord(
            step="run_localize_apply",
            completed=True,
            params={},
            input_path="in",
            output_path=str(log_path.parent),
            message="localize ready",
        ),
    )


def _localize_dir(context: RecordContext) -> Path:
    return (
        context.project_root
        / "derivatives"
        / "lfptensorpipe"
        / context.subject
        / context.record
        / "localize"
    )


def _write_localize_repcoord_artifact(
    context: RecordContext,
    filename: str,
    frame: pd.DataFrame,
) -> None:
    out_dir = _localize_dir(context)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pkl(frame, out_dir / f"{filename}.pkl")
    frame.to_csv(out_dir / f"{filename}.csv", index=False)


def _write_warped_tensor(
    resolver: PathResolver,
    slug: str,
    metric_key: str,
    *,
    channel_axis: list[object],
) -> None:
    path = resolver.alignment_paradigm_dir(slug) / metric_key / "tensor_warped.pkl"
    save_pkl(
        {
            "tensor": np.ones((1, len(channel_axis), 1, 4), dtype=float),
            "meta": {
                "axes": {
                    "epoch": ["epoch_000"],
                    "channel": channel_axis,
                    "freq": [10.0],
                    "time": [0.0, 0.5, 1.0, 1.5],
                }
            },
        },
        path,
    )


def _prepare_alignment_finish_fixture(
    tmp_path: Path,
) -> tuple[RecordContext, PathResolver, str]:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 1200), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.5, 2.0, 3.6],
            duration=[0.7, 0.8, 0.9],
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

    metric_path = tensor_metric_tensor_path(resolver, "raw_power")
    tensor = np.ones((1, 2, 6, 120), dtype=float)
    save_pkl(
        {
            "tensor": tensor,
            "meta": {
                "axes": {
                    "channel": np.array(["CH1", "CH2"], dtype=object),
                    "freq": np.linspace(2.0, 20.0, 6, dtype=float),
                    "time": np.linspace(0.0, 6.0, 120, dtype=float),
                    "shape": tensor.shape,
                },
                "params": {},
            },
        },
        metric_path,
    )
    write_run_log(
        tensor_metric_log_path(resolver, "raw_power"),
        RunLogRecord(
            step="raw_power",
            completed=True,
            params={},
            input_path="in",
            output_path=str(metric_path),
            message="tensor ready",
        ),
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    created, _, entry = create_alignment_paradigm(store, name="Gait", context=context)
    assert created
    assert isinstance(entry, dict)
    slug = str(entry["slug"])
    updated, _ = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
        method_params={
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 10_000.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.8,
        },
        context=context,
    )
    assert updated

    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
    )
    assert ok, message
    assert len(rows) >= 1
    return context, resolver, slug


def _prepare_run_align_epochs_fixture(
    tmp_path: Path,
) -> tuple[RecordContext, PathResolver, AppConfigStore]:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["CH1", "CH2"], sfreq=200.0, ch_types=["dbs", "dbs"])
    raw = mne.io.RawArray(np.zeros((2, 1200), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.5, 2.0, 3.6],
            duration=[0.7, 0.8, 0.9],
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

    metric_path = tensor_metric_tensor_path(resolver, "raw_power")
    tensor = np.ones((1, 2, 6, 120), dtype=float)
    save_pkl(
        {
            "tensor": tensor,
            "meta": {
                "axes": {
                    "channel": np.array(["CH1", "CH2"], dtype=object),
                    "freq": np.linspace(2.0, 20.0, 6, dtype=float),
                    "time": np.linspace(0.0, 6.0, 120, dtype=float),
                    "shape": tensor.shape,
                },
                "params": {},
            },
        },
        metric_path,
    )
    write_run_log(
        tensor_metric_log_path(resolver, "raw_power"),
        RunLogRecord(
            step="raw_power",
            completed=True,
            params={},
            input_path="in",
            output_path=str(metric_path),
            message="tensor ready",
        ),
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    return context, resolver, store


def test_run_align_epochs_and_finish_builds_raw_tables(tmp_path: Path) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)

    warped_path = (
        resolver.alignment_paradigm_dir(slug) / "raw_power" / "tensor_warped.pkl"
    )
    assert warped_path.exists()
    assert (
        indicator_from_log(
            resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
        )
        == "yellow"
    )
    ok_finish, finish_message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok_finish, finish_message
    raw_table_path = alignment_trial_raw_table_path(
        resolver,
        trial_slug=slug,
        metric_key="raw_power",
    )
    assert raw_table_path.exists()
    frame = load_pkl(raw_table_path)
    assert "Value" in frame.columns
    assert frame.shape[0] >= 1
    assert (
        indicator_from_log(
            resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
        )
        == "green"
    )
    payload = read_run_log(
        resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
    )
    assert isinstance(payload, dict)
    assert payload.get("step") == "build_raw_table"
    params = payload.get("params")
    assert isinstance(params, dict)
    assert params.get("merge_location_info_ready") is False
    assert params.get("merge_location_info_applied") is False
    history = payload.get("history")
    assert isinstance(history, list)
    assert any(
        str(item.get("step", "")) == "run_align_epochs"
        for item in history
        if isinstance(item, dict)
    )


def test_load_alignment_epoch_rows_restores_latest_finished_picks(
    tmp_path: Path,
) -> None:
    context, _resolver, slug = _prepare_alignment_finish_fixture(tmp_path)

    ok_finish, finish_message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok_finish, finish_message

    assert load_alignment_epoch_picks(context, paradigm_slug=slug) == [0]
    rows = load_alignment_epoch_rows(context, paradigm_slug=slug)
    assert [int(row["epoch_index"]) for row in rows if bool(row.get("pick"))] == [0]


def test_load_alignment_epoch_rows_prefers_draft_pick_state_over_finish_picks(
    tmp_path: Path,
) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)

    ok_finish, finish_message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok_finish, finish_message

    persisted = persist_alignment_epoch_picks(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[1],
    )
    assert persisted

    rows = load_alignment_epoch_rows(context, paradigm_slug=slug)
    assert [int(row["epoch_index"]) for row in rows if bool(row.get("pick"))] == [1]

    payload = read_run_log(
        resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
    )
    assert isinstance(payload, dict)
    assert payload.get("step") == "build_raw_table"
    assert bool(payload.get("completed")) is True
    assert payload.get("state", {}).get("epoch_inspector", {}).get(
        "picked_epoch_indices"
    ) == [1]


def test_run_align_epochs_invalidates_same_trial_features_log(tmp_path: Path) -> None:
    context, resolver, store = _prepare_run_align_epochs_fixture(tmp_path)
    created, _, entry = create_alignment_paradigm(store, name="Gait", context=context)
    assert created
    assert isinstance(entry, dict)
    slug = str(entry["slug"])
    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
        method_params={
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 10_000.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.8,
        },
        context=context,
    )
    assert updated, message

    features_log = resolver.features_root / slug / "lfptensorpipe_log.json"
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

    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
    )
    assert ok, message
    assert rows
    assert (
        indicator_from_log(
            resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
        )
        == "yellow"
    )
    assert indicator_from_log(features_log) == "yellow"


def test_finish_alignment_epochs_invalidates_same_trial_features_log(
    tmp_path: Path,
) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)
    features_log = resolver.features_root / slug / "lfptensorpipe_log.json"
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

    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok, message
    assert indicator_from_log(features_log) == "yellow"


def test_alignment_panel_indicator_states_track_run_finish_and_staleness(
    tmp_path: Path,
) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)
    paradigm = {
        "name": "Gait",
        "slug": slug,
        "trial_slug": slug,
        "method": "stack_warper",
        "method_params": {
            "annotations": ["event"],
            "duration_range": [0.0, 10000.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.8,
        },
        "annotation_filter": {},
    }

    assert alignment_method_panel_state(resolver, paradigm=paradigm) == "green"
    assert (
        alignment_epoch_inspector_state(
            resolver,
            paradigm=paradigm,
            picked_epoch_indices=[0],
        )
        == "yellow"
    )

    ok_finish, finish_message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok_finish, finish_message
    assert (
        alignment_epoch_inspector_state(
            resolver,
            paradigm=paradigm,
            picked_epoch_indices=[0],
        )
        == "green"
    )
    _mark_localize_ready(context)
    assert (
        alignment_epoch_inspector_state(
            resolver,
            paradigm=paradigm,
            picked_epoch_indices=[0],
        )
        == "yellow"
    )
    _localize_log_path(context).unlink()
    assert (
        alignment_epoch_inspector_state(
            resolver,
            paradigm=paradigm,
            picked_epoch_indices=[0],
        )
        == "green"
    )
    assert (
        alignment_epoch_inspector_state(
            resolver,
            paradigm=paradigm,
            picked_epoch_indices=[0, 1],
        )
        == "yellow"
    )

    stale_paradigm = dict(paradigm)
    stale_paradigm["method_params"] = dict(paradigm["method_params"])
    stale_paradigm["method_params"]["sample_rate"] = 0.9
    assert alignment_method_panel_state(resolver, paradigm=stale_paradigm) == "yellow"
    assert (
        alignment_epoch_inspector_state(
            resolver,
            paradigm=stale_paradigm,
            picked_epoch_indices=[0],
        )
        == "yellow"
    )
    assert alignment_method_panel_state(resolver, paradigm=paradigm) == "green"
    assert (
        alignment_epoch_inspector_state(
            resolver,
            paradigm=paradigm,
            picked_epoch_indices=[0],
        )
        == "green"
    )

    raw_table_path = alignment_trial_raw_table_path(
        resolver,
        trial_slug=slug,
        metric_key="raw_power",
    )
    raw_table_path.unlink()
    assert (
        alignment_epoch_inspector_state(
            resolver,
            paradigm=paradigm,
            picked_epoch_indices=[0],
        )
        == "yellow"
    )


def test_finish_alignment_epochs_merges_single_repcoord_pkl_with_conflict_suffix(
    tmp_path: Path,
) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)
    _mark_localize_ready(context)
    _write_localize_repcoord_artifact(
        context,
        "channel_representative_coords",
        pd.DataFrame(
            [
                {
                    "channel": "CH1",
                    "Metric": "rep-metric",
                    "rep_coord": "Mid",
                    "mni_x": 1.0,
                    "mni_y": 2.0,
                    "mni_z": 3.0,
                },
                {
                    "channel": "CH2",
                    "Metric": "rep-metric",
                    "rep_coord": "Anode",
                    "mni_x": 4.0,
                    "mni_y": 5.0,
                    "mni_z": 6.0,
                },
            ]
        ),
    )

    ok_finish, finish_message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok_finish, finish_message

    raw_table_path = alignment_trial_raw_table_path(
        resolver,
        trial_slug=slug,
        metric_key="raw_power",
    )
    frame = load_pkl(raw_table_path)
    assert "Metric_repcoord" in frame.columns
    assert "rep_coord" in frame.columns
    assert "mni_x" in frame.columns

    ch1 = frame.loc[frame["Channel"] == "CH1"]
    assert not ch1.empty
    assert set(ch1["Metric_repcoord"].dropna().astype(str)) == {"rep-metric"}
    assert set(ch1["rep_coord"].dropna().astype(str)) == {"Mid"}


def test_finish_alignment_epochs_merges_undirected_pair_repcoords_by_canonical_key(
    tmp_path: Path,
) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)
    _mark_localize_ready(context)
    _write_localize_repcoord_artifact(
        context,
        "channel_representative_coords",
        pd.DataFrame(
            [
                {
                    "channel": "CH1",
                    "rep_coord": "Mid",
                    "mni_x": 1.0,
                    "mni_y": 2.0,
                    "mni_z": 3.0,
                },
                {
                    "channel": "CH2",
                    "rep_coord": "Mid",
                    "mni_x": 4.0,
                    "mni_y": 5.0,
                    "mni_z": 6.0,
                },
            ]
        ),
    )
    _write_localize_repcoord_artifact(
        context,
        "channel_pair_undirected_representative_coords",
        pd.DataFrame(
            [
                {
                    "channel": ("CH1", "CH2"),
                    "channel_a": "CH1",
                    "channel_b": "CH2",
                    "pair_key": '["CH1", "CH2"]',
                    "pair_key_ordered": '["CH1", "CH2"]',
                    "pair_key_undirected": '["CH1", "CH2"]',
                    "mni_x": (1.0, 4.0),
                    "mni_y": (2.0, 5.0),
                    "mni_z": (3.0, 6.0),
                    "SNr-STN_in": True,
                }
            ]
        ),
    )
    _write_warped_tensor(
        resolver,
        slug,
        "coherence",
        channel_axis=[("CH2", "CH1")],
    )

    ok_finish, finish_message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok_finish, finish_message

    frame = load_pkl(
        alignment_trial_raw_table_path(
            resolver,
            trial_slug=slug,
            metric_key="coherence",
        )
    )
    assert frame.shape[0] == 1
    assert frame.iloc[0]["Channel"] == ("CH2", "CH1")
    assert frame.iloc[0]["channel"] == ("CH1", "CH2")
    assert frame.iloc[0]["pair_key_undirected"] == '["CH1", "CH2"]'
    assert frame.iloc[0]["mni_x"] == (1.0, 4.0)
    assert bool(frame.iloc[0]["SNr-STN_in"]) is True


def test_finish_alignment_epochs_merges_ordered_pair_repcoords_by_ordered_key(
    tmp_path: Path,
) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)
    _mark_localize_ready(context)
    _write_localize_repcoord_artifact(
        context,
        "channel_representative_coords",
        pd.DataFrame(
            [
                {
                    "channel": "CH1",
                    "rep_coord": "Mid",
                    "mni_x": 1.0,
                    "mni_y": 2.0,
                    "mni_z": 3.0,
                },
                {
                    "channel": "CH2",
                    "rep_coord": "Mid",
                    "mni_x": 4.0,
                    "mni_y": 5.0,
                    "mni_z": 6.0,
                },
            ]
        ),
    )
    _write_localize_repcoord_artifact(
        context,
        "channel_pair_ordered_representative_coords",
        pd.DataFrame(
            [
                {
                    "channel": ("CH2", "CH1"),
                    "channel_a": "CH2",
                    "channel_b": "CH1",
                    "pair_key": '["CH2", "CH1"]',
                    "pair_key_ordered": '["CH2", "CH1"]',
                    "pair_key_undirected": '["CH1", "CH2"]',
                    "mni_x": (4.0, 1.0),
                    "mni_y": (5.0, 2.0),
                    "mni_z": (6.0, 3.0),
                    "STN-SNr_in": True,
                }
            ]
        ),
    )
    _write_warped_tensor(
        resolver,
        slug,
        "psi",
        channel_axis=[("CH2", "CH1")],
    )

    ok_finish, finish_message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok_finish, finish_message

    frame = load_pkl(
        alignment_trial_raw_table_path(
            resolver,
            trial_slug=slug,
            metric_key="psi",
        )
    )
    assert frame.shape[0] == 1
    assert frame.iloc[0]["Channel"] == ("CH2", "CH1")
    assert frame.iloc[0]["channel"] == ("CH2", "CH1")
    assert frame.iloc[0]["pair_key_ordered"] == '["CH2", "CH1"]'
    assert frame.iloc[0]["mni_x"] == (4.0, 1.0)
    assert bool(frame.iloc[0]["STN-SNr_in"]) is True


def test_finish_alignment_epochs_skips_repcoord_merge_when_localize_not_ready(
    tmp_path: Path,
) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)
    repcoord_path = (
        context.project_root
        / "derivatives"
        / "lfptensorpipe"
        / context.subject
        / context.record
        / "localize"
        / "channel_representative_coords.csv"
    )
    repcoord_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "channel": "CH1",
                "rep_coord": "Mid",
                "mni_x": 1.0,
            }
        ]
    ).to_csv(repcoord_path, index=False)

    ok_finish, finish_message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok_finish, finish_message

    raw_table_path = alignment_trial_raw_table_path(
        resolver,
        trial_slug=slug,
        metric_key="raw_power",
    )
    frame = load_pkl(raw_table_path)
    assert "rep_coord" not in frame.columns
    assert "mni_x" not in frame.columns

    payload = read_run_log(
        resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
    )
    assert isinstance(payload, dict)
    params = payload.get("params")
    assert isinstance(params, dict)
    assert params.get("merge_location_info_ready") is False
    assert params.get("merge_location_info_applied") is False


def test_finish_alignment_epochs_time_axis_selection_by_method(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-time-axis"

    log_path = resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
    metric_path = (
        resolver.alignment_paradigm_dir(slug) / "raw_power" / "tensor_warped.pkl"
    )
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_trial_log(method: str) -> None:
        write_run_log(
            log_path,
            RunLogRecord(
                step="run_align_epochs",
                completed=True,
                params={
                    "trial_slug": slug,
                    "method": method,
                    "method_params": {},
                },
                input_path="in",
                output_path="out",
                message="ready",
            ),
        )

    def _write_metric(*, include_percent: bool, include_time: bool) -> None:
        axes: dict[str, object] = {
            "epoch": ["e0"],
            "channel": ["ch0"],
            "freq": [10.0],
        }
        if include_percent:
            axes["percent"] = [0.0, 50.0, 100.0]
        if include_time:
            axes["time"] = [1.5, 2.5, 3.5]
        save_pkl(
            {
                "tensor": np.zeros((1, 1, 1, 3), dtype=float),
                "meta": {"axes": axes},
            },
            metric_path,
        )

    _write_trial_log("linear_warper")
    _write_metric(include_percent=True, include_time=True)
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok, message
    frame = load_pkl(
        alignment_trial_raw_table_path(
            resolver,
            trial_slug=slug,
            metric_key="raw_power",
        )
    )
    value = frame.iloc[0]["Value"]
    assert isinstance(value, pd.DataFrame)
    assert list(value.columns) == [0.0, 50.0, 100.0]

    _write_trial_log("pad_warper")
    _write_metric(include_percent=True, include_time=True)
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok, message
    frame = load_pkl(
        alignment_trial_raw_table_path(
            resolver,
            trial_slug=slug,
            metric_key="raw_power",
        )
    )
    value = frame.iloc[0]["Value"]
    assert isinstance(value, pd.DataFrame)
    assert list(value.columns) == [1.5, 2.5, 3.5]

    _write_trial_log("linear_warper")
    _write_metric(include_percent=False, include_time=True)
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok, message
    frame = load_pkl(
        alignment_trial_raw_table_path(
            resolver,
            trial_slug=slug,
            metric_key="raw_power",
        )
    )
    value = frame.iloc[0]["Value"]
    assert isinstance(value, pd.DataFrame)
    assert list(value.columns) == [0.0, 1.0, 2.0]

    _write_trial_log("concat_warper")
    _write_metric(include_percent=True, include_time=False)
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok, message
    frame = load_pkl(
        alignment_trial_raw_table_path(
            resolver,
            trial_slug=slug,
            metric_key="raw_power",
        )
    )
    value = frame.iloc[0]["Value"]
    assert isinstance(value, pd.DataFrame)
    assert list(value.columns) == [0.0, 1.0, 2.0]


def test_finish_alignment_epochs_requires_green_alignment_log(tmp_path: Path) -> None:
    context = _context(tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    created, _, entry = create_alignment_paradigm(store, name="Task", context=context)
    assert created
    assert isinstance(entry, dict)
    slug = str(entry["slug"])

    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert not ok
    assert "Run Align Epochs successfully before Finish." in message


def test_finish_alignment_epochs_allows_restored_successful_run_config(
    tmp_path: Path,
) -> None:
    context, resolver, slug = _prepare_alignment_finish_fixture(tmp_path)
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()

    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="concat_warper",
        method_params={
            "annotations": ["event"],
            "drop_bad": False,
            "sample_rate": 20.0,
        },
        context=context,
    )
    assert updated, message

    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
        context=context,
    )
    assert updated, message

    log_path = resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
    assert indicator_from_log(log_path) == "yellow"

    paradigms = load_alignment_paradigms(store, context=context)
    paradigm = next(item for item in paradigms if item["slug"] == slug)
    assert alignment_method_panel_state(resolver, paradigm=paradigm) == "green"

    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert ok, message
    assert "Finish completed." in message


def test_validate_alignment_method_params_rejects_invalid_linear_anchors() -> None:
    ok, _, message = validate_alignment_method_params(
        "linear_warper",
        {
            "anchors_percent": {5.0: "start", 100.0: "end"},
            "epoch_duration_range": [None, None],
            "linear_warp": True,
            "percent_tolerance": 15.0,
            "drop_bad": True,
            "sample_rate": 5.0,
        },
        annotation_labels=[],
    )
    assert not ok
    assert "Anchors must start at 0 and end at 100." in message


def test_validate_alignment_method_params_ignores_mode_input() -> None:
    ok, normalized, message = validate_alignment_method_params(
        "stack_warper",
        {
            "annotations": ["event"],
            "mode": "contains",
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "sample_rate": 0.64,
        },
        annotation_labels=["event"],
    )
    assert ok, message
    assert "mode" not in normalized
    assert "drop_mode" not in normalized
    assert normalized["sample_rate"] == 0.64


def test_alignment_helper_normalizers_cover_error_paths() -> None:
    assert "annotations" in alignment_service.default_alignment_method_params(
        "pad_warper"
    )
    assert "pad_left" in alignment_service.default_alignment_method_params("pad_warper")
    assert "annotations" in alignment_service.default_alignment_method_params(
        "concat_warper"
    )
    assert "duration_range" not in alignment_service.default_alignment_method_params(
        "concat_warper"
    )
    assert "annotations" in alignment_service.default_alignment_method_params("unknown")
    assert "pad_s" not in alignment_service.default_alignment_method_params(
        "stack_warper"
    )
    assert "pad_s" not in alignment_service.default_alignment_method_params(
        "concat_warper"
    )
    assert "drop_mode" not in alignment_service.default_alignment_method_params(
        "linear_warper"
    )

    ok, _, message = alignment_service._normalize_sample_rate("x", fallback=5.0)
    assert not ok and "numeric" in message
    ok, _, message = alignment_service._normalize_sample_rate(0.0, fallback=5.0)
    assert not ok and "> 0" in message

    ok, _, message = alignment_service._normalize_duration_range(
        "bad", allow_none=False
    )
    assert not ok and "2 values" in message
    ok, _, message = alignment_service._normalize_duration_range(
        [None, 1.0], allow_none=False
    )
    assert not ok and "cannot be null" in message
    ok, _, message = alignment_service._normalize_duration_range(
        ["x", 1.0], allow_none=False
    )
    assert not ok and "must be numbers" in message
    ok, _, message = alignment_service._normalize_duration_range(
        [-1.0, 1.0], allow_none=False
    )
    assert not ok and "must be >= 0" in message
    ok, _, message = alignment_service._normalize_duration_range(
        [2.0, 1.0], allow_none=False
    )
    assert not ok and "max must be >= duration min" in message

    ok, _, message = alignment_service._normalize_annotations("bad")
    assert not ok and "must be a list" in message
    ok, annotations, _ = alignment_service._normalize_annotations([" evt ", "evt", " "])
    assert ok and annotations == ["evt"]
    ok, _, message = alignment_service._normalize_annotations(["", " "])
    assert not ok and "Select at least one annotation label" in message

    ok, _, message = alignment_service._normalize_anchors([])
    assert not ok and "must be a dict" in message
    ok, _, message = alignment_service._normalize_anchors({0.0: "", 100.0: "end"})
    assert not ok and "cannot be empty" in message
    ok, _, message = alignment_service._normalize_anchors({"x": "start", 100.0: "end"})
    assert not ok and "must be numeric" in message
    ok, _, message = alignment_service._normalize_anchors({-1.0: "start", 100.0: "end"})
    assert not ok and "within [0, 100]" in message
    ok, _, message = alignment_service._normalize_anchors({0.0: "start"})
    assert not ok and "At least 2 anchors" in message
    ok, _, message = alignment_service._normalize_anchors({5.0: "start", 100.0: "end"})
    assert not ok and "start at 0 and end at 100" in message
    ok, _, message = alignment_service._normalize_anchors(
        {"0": "start", "00": "middle", "100": "end"}
    )
    assert not ok and "strictly increasing" in message

    ok, _, message = alignment_service._normalize_nonnegative_float(
        "x",
        field_name="pad_left",
        fallback=0.5,
    )
    assert not ok and "numeric" in message
    ok, _, message = alignment_service._normalize_nonnegative_float(
        -1.0,
        field_name="pad_left",
        fallback=0.5,
    )
    assert not ok and ">= 0" in message


def test_validate_alignment_method_params_covers_fallback_and_error_branches() -> None:
    ok, _, message = validate_alignment_method_params(
        "bad_method", {}, annotation_labels=[]
    )
    assert not ok and "Unknown alignment method" in message

    ok, _, message = validate_alignment_method_params(
        "stack_warper",
        {
            "annotations": ["event"],
            "duration_range": [0.0, 1.0],
            "sample_rate": "x",
        },
        annotation_labels=[],
    )
    assert not ok and "sample_rate must be numeric" in message

    ok, normalized, message = validate_alignment_method_params(
        "linear_warper",
        {
            "anchors_percent": {5.0: "a", 100.0: "b"},
            "epoch_duration_range": [None, None],
            "percent_tolerance": 15.0,
            "sample_rate": 0.64,
        },
        annotation_labels=["a", "b"],
    )
    assert not ok and "Anchors must start at 0 and end at 100." in message

    ok, _, message = validate_alignment_method_params(
        "linear_warper",
        {
            "anchors_percent": {5.0: "only", 100.0: "only"},
            "epoch_duration_range": [None, None],
            "percent_tolerance": 15.0,
            "sample_rate": 0.64,
        },
        annotation_labels=["only"],
    )
    assert not ok and "Anchors must start at 0 and end at 100." in message

    ok, _, message = validate_alignment_method_params(
        "linear_warper",
        {
            "anchors_percent": {0.0: "start", 100.0: "end"},
            "epoch_duration_range": [None, None],
            "percent_tolerance": "x",
            "sample_rate": 0.64,
        },
        annotation_labels=["start", "end"],
    )
    assert not ok and "percent_tolerance must be numeric" in message

    ok, _, message = validate_alignment_method_params(
        "linear_warper",
        {
            "anchors_percent": {0.0: "start", 100.0: "end"},
            "epoch_duration_range": [None, None],
            "percent_tolerance": -1.0,
            "sample_rate": 0.64,
        },
        annotation_labels=["start", "end"],
    )
    assert not ok and "percent_tolerance must be >= 0" in message

    ok, normalized, message = validate_alignment_method_params(
        "pad_warper",
        {
            "annotations": ["event"],
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
            "duration_range": [0.0, 1.0],
            "sample_rate": 16.0,
        },
        annotation_labels=[],
    )
    assert ok, message
    assert normalized["annotations"] == ["event"]
    assert normalized["pad_left"] == 0.5

    ok, normalized, message = validate_alignment_method_params(
        "pad_warper",
        {
            "annotations": [],
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
            "duration_range": [0.0, 1.0],
            "sample_rate": 16.0,
        },
        annotation_labels=[],
    )
    assert ok and normalized["annotations"] == []

    ok, _, message = validate_alignment_method_params(
        "pad_warper",
        {
            "annotations": ["event"],
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
            "duration_range": [2.0, 1.0],
            "sample_rate": 16.0,
        },
        annotation_labels=[],
    )
    assert not ok and "max must be >= duration min" in message

    ok, normalized, message = validate_alignment_method_params(
        "stack_warper",
        {
            "annotations": [],
            "duration_range": [0.0, 1.0],
            "sample_rate": 0.16,
        },
        annotation_labels=["event"],
    )
    assert ok, message
    assert normalized["annotations"] == []

    ok, normalized, message = validate_alignment_method_params(
        "stack_warper",
        {
            "annotations": [],
            "duration_range": [0.0, 1.0],
            "sample_rate": 0.16,
        },
        annotation_labels=[],
    )
    assert ok, message
    assert normalized["annotations"] == []

    ok, normalized, message = validate_alignment_method_params(
        "concat_warper",
        {
            "annotations": ["event"],
            "pad_s": "x",
            "sample_rate": 16.0,
        },
        annotation_labels=["event"],
    )
    assert ok, message
    assert "pad_s" not in normalized
    assert "duration_range" not in normalized


def test_alignment_paradigm_crud_and_payload_normalization_branches(
    tmp_path: Path,
) -> None:
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()

    store.write_yaml("alignment.yml", ["bad"])
    assert load_alignment_paradigms(store) == []
    store.write_yaml("alignment.yml", {"paradigms": "bad"})
    assert load_alignment_paradigms(store) == []

    save_alignment_paradigms(
        store,
        paradigms=[
            "bad",
            {"name": "A", "slug": "a"},
            {"name": "A2", "slug": "a"},
            {"name": "B", "slug": "b", "method": "unknown"},
        ],
    )
    loaded = load_alignment_paradigms(store)
    assert [item["slug"] for item in loaded] == ["a", "b"]
    assert [item["trial_slug"] for item in loaded] == ["a", "b"]
    assert loaded[1]["method"] == "stack_warper"

    created, message, entry = create_alignment_paradigm(store, name="   ")
    assert not created and "cannot be empty" in message and entry is None
    created, message, entry = create_alignment_paradigm(store, name="!!!")
    assert not created and "Failed to generate trial slug" in message and entry is None

    created, _, entry = create_alignment_paradigm(store, name="Task")
    assert created and isinstance(entry, dict)
    created2, _, entry2 = create_alignment_paradigm(store, name="Task")
    assert created2 and isinstance(entry2, dict)
    assert str(entry2["slug"]).endswith("-2")

    deleted, message = delete_alignment_paradigm(store, slug="   ")
    assert not deleted and "slug is empty" in message
    deleted, message = delete_alignment_paradigm(store, slug="missing")
    assert not deleted and "not found" in message

    updated, message = update_alignment_paradigm(store, slug="missing")
    assert not updated and "not found" in message
    updated, message = update_alignment_paradigm(
        store,
        slug=str(entry["slug"]),
        method="unknown_method",
    )
    assert not updated and "Unknown method" in message
    updated, message = update_alignment_paradigm(
        store,
        slug=str(entry["slug"]),
        method="stack_warper",
        method_params={
            "annotations": [],
            "duration_range": [0.0, 1.0],
            "sample_rate": 0.1,
        },
    )
    assert updated and "updated" in message

    deleted, message = delete_alignment_paradigm(store, slug=str(entry["slug"]))
    assert deleted and "deleted" in message


def test_load_alignment_paradigms_with_context_ignores_legacy_config_trials(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()
    save_alignment_paradigms(
        store,
        paradigms=[
            {
                "name": "Legacy Trial",
                "trial_slug": "legacy-trial",
                "method": "stack_warper",
                "method_params": {
                    "annotations": [],
                    "duration_range": [0.0, 1.0],
                    "sample_rate": 0.1,
                },
                "annotation_filter": {},
            }
        ],
    )

    assert load_alignment_paradigms(store, context=context) == []


def test_alignment_log_path_read_does_not_recreate_trial_directory(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    trial_dir = resolver.alignment_paradigm_dir("ghost-trial", create=True)
    assert trial_dir.exists()

    trial_dir.rmdir()
    log_path = alignment_service.alignment_paradigm_log_path(resolver, "ghost-trial")

    assert log_path == trial_dir / "lfptensorpipe_log.json"
    assert not trial_dir.exists()


def test_delete_alignment_paradigm_with_context_cleans_artifacts_config_and_ui_state(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()
    save_alignment_paradigms(
        store,
        paradigms=[
            {"name": "Task", "trial_slug": "task"},
            {"name": "Keep", "trial_slug": "keep"},
        ],
    )
    trial_dir = resolver.alignment_paradigm_dir("task", create=True)
    features_dir = resolver.features_root / "task"
    legacy_raw_dir = resolver.features_root / "raw" / "task"
    legacy_derivatives_dir = resolver.features_root / "derivatives" / "task"
    legacy_derivatives_tx_dir = (
        resolver.features_root / "derivatives_transformed" / "task"
    )
    normalization_dir = resolver.features_root / "normalization" / "task"
    normalization_tx_dir = resolver.features_root / "normalization_transformed" / "task"
    for path in (
        features_dir,
        legacy_raw_dir,
        legacy_derivatives_dir,
        legacy_derivatives_tx_dir,
        normalization_dir,
        normalization_tx_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    ui_state_path = resolver.record_ui_state_path(create=True)
    write_ui_state(
        ui_state_path,
        {
            "alignment": {
                "trial_slug": "task",
                "method": "stack_warper",
                "sample_rate": 0.4,
                "epoch_metric": "raw_power",
                "epoch_channel": 0,
                "picked_epoch_indices": [0, 1],
            },
            "features": {
                "paradigm_slug": "task",
                "active_metric": "raw_power",
            },
        },
    )
    assert trial_dir.exists()

    deleted, message = delete_alignment_paradigm(store, slug="task", context=context)

    assert deleted, message
    for path in (
        trial_dir,
        features_dir,
        legacy_raw_dir,
        legacy_derivatives_dir,
        legacy_derivatives_tx_dir,
        normalization_dir,
        normalization_tx_dir,
    ):
        assert not path.exists()
    payload = read_ui_state(ui_state_path)
    assert isinstance(payload, dict)
    assert payload["alignment"]["trial_slug"] is None
    assert payload["alignment"]["method"] is None
    assert payload["alignment"]["picked_epoch_indices"] == []
    assert payload["features"]["paradigm_slug"] is None
    assert [item["slug"] for item in load_alignment_paradigms(store)] == ["keep"]


def test_delete_alignment_paradigm_with_context_does_not_resurrect_legacy_trials(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()
    save_alignment_paradigms(
        store,
        paradigms=[
            {"name": "Legacy Trial", "trial_slug": "legacy-trial"},
        ],
    )
    created, _, entry = create_alignment_paradigm(
        store,
        name="Active Trial",
        context=context,
    )
    assert created
    assert isinstance(entry, dict)
    slug = str(entry["slug"])
    assert [
        item["slug"] for item in load_alignment_paradigms(store, context=context)
    ] == [slug]

    deleted, message = delete_alignment_paradigm(store, slug=slug, context=context)

    assert deleted, message
    assert load_alignment_paradigms(store, context=context) == []


def test_update_alignment_paradigm_method_params_default_fallback_branch(
    tmp_path: Path,
) -> None:
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()
    captured: dict[str, object] = {}

    def _fake_save(
        config_store: AppConfigStore, paradigms: list[dict[str, object]]
    ) -> None:
        _ = config_store
        captured["paradigms"] = paradigms
        return None

    updated, message = update_alignment_paradigm(
        store,
        slug="paradigm-a",
        load_alignment_paradigms_fn=lambda config_store, **_kwargs: [
            {"slug": "paradigm-a", "method": "stack_warper", "method_params": []}
        ],
        save_alignment_paradigms_fn=_fake_save,
    )
    assert updated and "updated" in message

    paradigms = captured["paradigms"]
    assert isinstance(paradigms, list)
    assert paradigms
    method_params = paradigms[0]["method_params"]
    assert isinstance(method_params, dict)
    assert method_params["annotations"] == []
    method_params_by_method = paradigms[0]["method_params_by_method"]
    assert isinstance(method_params_by_method, dict)
    assert method_params_by_method["stack_warper"]["annotations"] == []


def test_update_alignment_paradigm_restores_trial_saved_method_params(
    tmp_path: Path,
) -> None:
    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()

    created, _, entry = create_alignment_paradigm(store, name="Gait")
    assert created and isinstance(entry, dict)
    slug = str(entry["slug"])

    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
        method_params={
            "annotations": ["event_a"],
            "duration_range": [0.0, 2.0],
            "drop_bad": False,
            "sample_rate": 0.4,
        },
    )
    assert updated and "updated" in message

    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="concat_warper",
        method_params={
            "annotations": ["event_b"],
            "drop_bad": True,
            "sample_rate": 12.5,
        },
    )
    assert updated and "updated" in message

    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
    )
    assert updated and "updated" in message

    paradigms = load_alignment_paradigms(store)
    assert len(paradigms) == 1
    paradigm = paradigms[0]
    assert paradigm["method"] == "stack_warper"
    assert paradigm["method_params"] == {
        "annotations": ["event_a"],
        "duration_range": [0.0, 2.0],
        "drop_bad": False,
        "drop_fields": ["bad", "edge"],
        "sample_rate": 0.4,
    }
    assert (
        paradigm["method_params_by_method"]["stack_warper"] == paradigm["method_params"]
    )
    assert paradigm["method_params_by_method"]["concat_warper"] == {
        "annotations": ["event_b"],
        "drop_bad": True,
        "drop_fields": ["bad", "edge"],
        "sample_rate": 12.5,
    }


def test_alignment_misc_helpers_and_preflight_branches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    assert load_alignment_annotation_labels(context) == []

    assert (
        alignment_service._load_unique_annotation_labels(
            tmp_path / "missing_raw.fif",
            read_raw_fif_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )
        == []
    )

    assert alignment_service._float_pair_list("bad", (1.0, 2.0)) == (1.0, 2.0)
    assert alignment_service._float_pair_list(["x", 2.0], (1.0, 2.0)) == (1.0, 2.0)
    assert alignment_service._float_pair_list([1.0, 2.0], (0.0, 0.0)) == (1.0, 2.0)

    with pytest.raises(ValueError, match="meta is missing"):
        alignment_service._coerce_alignment_tensor({"tensor": [1, 2, 3]})
    with pytest.raises(ValueError, match="expects"):
        alignment_service._coerce_alignment_tensor({"tensor": [1, 2, 3], "meta": {}})
    tensor_3d, meta = alignment_service._coerce_alignment_tensor(
        {"tensor": np.zeros((1, 2, 3, 4), dtype=float), "meta": {"axes": {}}}
    )
    assert tensor_3d.shape == (2, 3, 4)
    assert isinstance(meta, dict)

    info = mne.create_info(["CH1"], sfreq=1000.0, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, 100), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(onset=[0.1], duration=[0.2], description=["other"])
    )
    with pytest.raises(ValueError, match="No annotations remain"):
        alignment_service._filter_raw_annotations_by_duration(
            raw,
            keep=["event"],
            duration_range=(0.0, 1.0),
        )

    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()
    created, _, entry = create_alignment_paradigm(store, name="Gait", context=context)
    assert created and isinstance(entry, dict)
    slug = str(entry["slug"])

    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug="missing",
    )
    assert not ok and "Trial not found" in message and rows == []

    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
    )
    assert not ok and "Preprocess finish must be green" in message and rows == []

    write_run_log(
        preproc_step_log_path(resolver, "finish"),
        RunLogRecord(
            step="finish",
            completed=True,
            params={},
            input_path="in",
            output_path="out",
            message="finish ready",
        ),
    )
    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
    )
    assert not ok and "No completed tensor metrics" in message and rows == []

    metric_key = "raw_power"
    metric_tensor_path = tensor_metric_tensor_path(resolver, metric_key)
    metric_tensor_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(
        {"tensor": np.zeros((1, 1, 1, 1), dtype=float), "meta": {}}, metric_tensor_path
    )
    write_run_log(
        tensor_metric_log_path(resolver, metric_key),
        RunLogRecord(
            step=metric_key,
            completed=True,
            params={},
            input_path="in",
            output_path=str(metric_tensor_path),
            message="ready",
        ),
    )
    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
    )
    assert not ok and "Missing preproc finish raw.fif" in message and rows == []


def test_run_align_epochs_deep_try_runtime_branches(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "paradigm-task"

    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()

    finish_raw = preproc_step_raw_path(resolver, "finish")
    finish_raw.parent.mkdir(parents=True, exist_ok=True)
    finish_raw.write_bytes(b"RAW")
    write_run_log(
        preproc_step_log_path(resolver, "finish"),
        RunLogRecord(
            step="finish",
            completed=True,
            params={},
            input_path="in",
            output_path=str(finish_raw),
            message="finish ready",
        ),
    )

    metric_key = "raw_power"
    metric_path = tensor_metric_tensor_path(resolver, metric_key)
    write_run_log(
        tensor_metric_log_path(resolver, metric_key),
        RunLogRecord(
            step=metric_key,
            completed=True,
            params={},
            input_path="in",
            output_path=str(metric_path),
            message="tensor ready",
        ),
    )

    class _FakeRaw:
        def __init__(self, labels: list[str]) -> None:
            self.annotations = type("Anno", (), {"description": labels})()

        def close(self) -> None:
            return None

    class _Epoch:
        label = "event"
        start_t = 0.0
        end_t = 1.0

    def _valid_payload(include_time: bool = True) -> dict[str, object]:
        axes: dict[str, object] = {
            "channel": np.array(["CH1"], dtype=object),
            "freq": np.array([10.0], dtype=float),
        }
        if include_time:
            axes["time"] = np.array([0.0, 1.0, 2.0], dtype=float)
        return {
            "tensor": np.zeros((1, 1, 1, 3), dtype=float),
            "meta": {"axes": axes},
        }

    def _warp_success(tensor, **kw):
        _ = kw
        return (
            np.zeros((1, tensor.shape[0], tensor.shape[1], 8), dtype=float),
            np.linspace(0.0, 100.0, 8, dtype=float),
            [],
        )

    def build_success(*args, **kwargs):
        _ = (args, kwargs)
        return {"ALL": [_Epoch()]}, _warp_success

    normalized_stack_params = {
        "annotations": ["event"],
        "duration_range": [0.0, 10_000.0],
        "sample_rate": 0.08,
    }

    save_pkl(_valid_payload(include_time=True), metric_path)
    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
        load_raw_for_warp_fn=lambda raw_path: _FakeRaw(["event"]),
        build_warped_tensor_metadata_fn=lambda *args, **kwargs: {
            "axes": {"percent": [0.0, 100.0]}
        },
        update_alignment_paradigm_fn=lambda *args, **kwargs: (True, "ok"),
        save_pkl_fn=lambda *args, **kwargs: None,
        load_alignment_paradigms_fn=lambda config_store, **_kwargs: [
            {"slug": slug, "method": "stack_warper", "method_params": "bad"}
        ],
        validate_alignment_method_params_fn=lambda *args, **kwargs: (
            True,
            dict(normalized_stack_params),
            "",
        ),
        build_warper_fn=build_success,
    )
    assert ok and "completed" in message and rows

    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
        load_raw_for_warp_fn=lambda raw_path: _FakeRaw(["event"]),
        build_warped_tensor_metadata_fn=lambda *args, **kwargs: {
            "axes": {"percent": [0.0, 100.0]}
        },
        update_alignment_paradigm_fn=lambda *args, **kwargs: (True, "ok"),
        save_pkl_fn=lambda *args, **kwargs: None,
        load_alignment_paradigms_fn=lambda config_store, **_kwargs: [
            {"slug": slug, "method": "stack_warper", "method_params": {}}
        ],
        validate_alignment_method_params_fn=lambda *args, **kwargs: (
            False,
            {},
            "bad params",
        ),
        build_warper_fn=build_success,
    )
    assert not ok and "Invalid method params" in message and rows == []

    def _warp_no_epochs(tensor, **kw):
        _ = kw
        return (
            np.zeros((1, tensor.shape[0], tensor.shape[1], 8), dtype=float),
            np.linspace(0.0, 100.0, 8, dtype=float),
            [],
        )

    def build_no_epochs(*args, **kwargs):
        _ = (args, kwargs)
        return {"ALL": []}, _warp_no_epochs

    save_pkl(_valid_payload(include_time=True), metric_path)
    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
        load_raw_for_warp_fn=lambda raw_path: _FakeRaw(["event"]),
        build_warped_tensor_metadata_fn=lambda *args, **kwargs: {
            "axes": {"percent": [0.0, 100.0]}
        },
        update_alignment_paradigm_fn=lambda *args, **kwargs: (True, "ok"),
        save_pkl_fn=lambda *args, **kwargs: None,
        load_alignment_paradigms_fn=lambda config_store, **_kwargs: [
            {"slug": slug, "method": "stack_warper", "method_params": {}}
        ],
        validate_alignment_method_params_fn=lambda *args, **kwargs: (
            True,
            dict(normalized_stack_params),
            "",
        ),
        build_warper_fn=build_no_epochs,
    )
    assert not ok and "No valid epochs detected" in message and rows == []

    save_pkl(["bad"], metric_path)
    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
        load_raw_for_warp_fn=lambda raw_path: _FakeRaw(["event"]),
        build_warped_tensor_metadata_fn=lambda *args, **kwargs: {
            "axes": {"percent": [0.0, 100.0]}
        },
        update_alignment_paradigm_fn=lambda *args, **kwargs: (True, "ok"),
        save_pkl_fn=lambda *args, **kwargs: None,
        load_alignment_paradigms_fn=lambda config_store, **_kwargs: [
            {"slug": slug, "method": "stack_warper", "method_params": {}}
        ],
        validate_alignment_method_params_fn=lambda *args, **kwargs: (
            True,
            dict(normalized_stack_params),
            "",
        ),
        build_warper_fn=build_success,
    )
    assert not ok and "Invalid tensor payload for metric" in message and rows == []

    save_pkl(_valid_payload(include_time=False), metric_path)
    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
        load_raw_for_warp_fn=lambda raw_path: _FakeRaw(["event"]),
        build_warped_tensor_metadata_fn=lambda *args, **kwargs: {
            "axes": {"percent": [0.0, 100.0]}
        },
        update_alignment_paradigm_fn=lambda *args, **kwargs: (True, "ok"),
        save_pkl_fn=lambda *args, **kwargs: None,
        load_alignment_paradigms_fn=lambda config_store, **_kwargs: [
            {"slug": slug, "method": "stack_warper", "method_params": {}}
        ],
        validate_alignment_method_params_fn=lambda *args, **kwargs: (
            True,
            dict(normalized_stack_params),
            "",
        ),
        build_warper_fn=build_success,
    )
    assert not ok and "missing time axis" in message and rows == []

    def _warp_invalid_shape(tensor, **kw):
        _ = kw
        return (
            np.zeros((tensor.shape[0], tensor.shape[1], 8), dtype=float),
            np.linspace(0.0, 100.0, 8, dtype=float),
            [],
        )

    def build_invalid_shape(*args, **kwargs):
        _ = (args, kwargs)
        return {"ALL": [_Epoch()]}, _warp_invalid_shape

    save_pkl(_valid_payload(include_time=True), metric_path)
    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
        load_raw_for_warp_fn=lambda raw_path: _FakeRaw(["event"]),
        build_warped_tensor_metadata_fn=lambda *args, **kwargs: {
            "axes": {"percent": [0.0, 100.0]}
        },
        update_alignment_paradigm_fn=lambda *args, **kwargs: (True, "ok"),
        save_pkl_fn=lambda *args, **kwargs: None,
        load_alignment_paradigms_fn=lambda config_store, **_kwargs: [
            {"slug": slug, "method": "stack_warper", "method_params": {}}
        ],
        validate_alignment_method_params_fn=lambda *args, **kwargs: (
            True,
            dict(normalized_stack_params),
            "",
        ),
        build_warper_fn=build_invalid_shape,
    )
    assert not ok and "invalid shape" in message and rows == []


def test_load_alignment_annotation_labels_reads_finish_raw(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    finish_raw_path = preproc_step_raw_path(resolver, "finish")
    finish_raw_path.parent.mkdir(parents=True, exist_ok=True)

    info = mne.create_info(["CH1"], sfreq=200.0, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, 600), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.4, 0.8],
            duration=[0.1, 0.1, 0.1],
            description=["b", "a", "a"],
        )
    )
    raw.save(str(finish_raw_path), overwrite=True)

    labels = load_alignment_annotation_labels(context)
    assert labels == ["a", "b"]


def test_alignment_completed_metrics_and_annotation_duration_filter_success(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)

    assert alignment_service._completed_tensor_metrics(resolver) == []
    assert not resolver.tensor_root.exists()

    ready_key = "raw_power"
    ready_path = tensor_metric_tensor_path(resolver, ready_key)
    save_pkl({"tensor": np.zeros((1, 1, 1, 1), dtype=float), "meta": {}}, ready_path)
    write_run_log(
        tensor_metric_log_path(resolver, ready_key),
        RunLogRecord(
            step=ready_key,
            completed=True,
            params={},
            input_path="in",
            output_path=str(ready_path),
            message="ready",
        ),
    )

    missing_tensor_key = "plv"
    resolver.tensor_metric_dir(missing_tensor_key, create=True)
    write_run_log(
        tensor_metric_log_path(resolver, missing_tensor_key),
        RunLogRecord(
            step=missing_tensor_key,
            completed=True,
            params={},
            input_path="in",
            output_path="out",
            message="no tensor file",
        ),
    )

    yellow_key = "coherence"
    yellow_path = tensor_metric_tensor_path(resolver, yellow_key)
    save_pkl({"tensor": np.zeros((1, 1, 1, 1), dtype=float), "meta": {}}, yellow_path)
    write_run_log(
        tensor_metric_log_path(resolver, yellow_key),
        RunLogRecord(
            step=yellow_key,
            completed=False,
            params={},
            input_path="in",
            output_path=str(yellow_path),
            message="yellow",
        ),
    )

    assert alignment_service._completed_tensor_metrics(resolver) == [ready_key]

    periodic_path = tensor_metric_tensor_path(resolver, "periodic_aperiodic")
    aperiodic_path = tensor_metric_tensor_path(resolver, "aperiodic")
    save_pkl({"tensor": np.zeros((1, 1, 1, 1), dtype=float), "meta": {}}, periodic_path)
    save_pkl(
        {"tensor": np.zeros((1, 1, 1, 1), dtype=float), "meta": {}},
        aperiodic_path,
    )
    write_run_log(
        tensor_metric_log_path(resolver, "periodic_aperiodic"),
        RunLogRecord(
            step="periodic_aperiodic",
            completed=True,
            params={},
            input_path="in",
            output_path=str(periodic_path),
            message="periodic ready",
        ),
    )
    metrics = alignment_service._completed_tensor_metrics(resolver)
    assert "periodic" in metrics
    assert "aperiodic" in metrics
    assert metrics[metrics.index("periodic") + 1] == "aperiodic"

    info = mne.create_info(["CH1"], sfreq=1000.0, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, 3000), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.5, 0.9, 1.2],
            duration=[0.2, 0.7, 0.8, 1.8],
            description=["event", "event", "other", "event"],
        )
    )
    filtered = alignment_service._filter_raw_annotations_by_duration(
        raw,
        keep=["event"],
        duration_range=(0.5, 1.0),
    )
    assert list(filtered.annotations.description) == ["event"]
    assert np.allclose(filtered.annotations.duration, [0.7])


def test_alignment_build_warper_fallback_branches() -> None:
    info = mne.create_info(["CH1"], sfreq=1000.0, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, 3000), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.5],
            duration=[0.2, 0.3],
            description=["evt_a", "evt_b"],
        )
    )
    captured: dict[str, dict[str, object]] = {}

    def fake_linear(raw_obj, **kwargs):
        _ = raw_obj
        captured["linear"] = kwargs
        return {"ALL": [object()]}, (lambda tensor, **kw: (tensor, np.array([0.0]), []))

    def fake_pad(raw_obj, **kwargs):
        _ = raw_obj
        captured["pad"] = kwargs
        return {"ALL": [object()]}, (lambda tensor, **kw: (tensor, np.array([0.0]), []))

    def fake_concat(raw_obj, **kwargs):
        _ = raw_obj
        captured["concat"] = kwargs
        return {"ALL": [object()]}, (lambda tensor, **kw: (tensor, np.array([0.0]), []))

    def fake_stack(raw_obj, **kwargs):
        _ = raw_obj
        captured["stack"] = kwargs
        return {"ALL": [object()]}, (lambda tensor, **kw: (tensor, np.array([0.0]), []))

    alignment_service._build_warper(
        raw,
        method="linear_warper",
        method_params={"anchors_percent": {}, "drop_bad": True},
        linear_warper_fn=fake_linear,
        pad_warper_fn=fake_pad,
        concat_warper_fn=fake_concat,
        stack_warper_fn=fake_stack,
    )
    assert captured["linear"]["anchors_percent"] == {0.0: "evt_a", 100.0: "evt_b"}
    assert captured["linear"]["mode"] == "exact"
    assert captured["linear"]["drop_mode"] == "substring"

    alignment_service._build_warper(
        raw,
        method="pad_warper",
        method_params={
            "annotations": ["evt_a"],
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
        },
        linear_warper_fn=fake_linear,
        pad_warper_fn=fake_pad,
        concat_warper_fn=fake_concat,
        stack_warper_fn=fake_stack,
    )
    assert "evt_a" in captured["pad"]["anno_allowed"]
    assert captured["pad"]["mode"] == "exact"
    assert captured["pad"]["drop_mode"] == "substring"

    with pytest.raises(ValueError, match="concat warper"):
        alignment_service._build_warper(
            raw,
            method="concat_warper",
            method_params={"annotations": []},
            linear_warper_fn=fake_linear,
            pad_warper_fn=fake_pad,
            concat_warper_fn=fake_concat,
            stack_warper_fn=fake_stack,
        )

    alignment_service._build_warper(
        raw,
        method="stack_warper",
        method_params={"annotations": "bad"},
        linear_warper_fn=fake_linear,
        pad_warper_fn=fake_pad,
        concat_warper_fn=fake_concat,
        stack_warper_fn=fake_stack,
    )
    assert captured["stack"]["keep"] == ["evt_a", "evt_b"]
    assert captured["stack"]["mode"] == "exact"
    assert captured["stack"]["drop_mode"] == "substring"

    alignment_service._build_warper(
        raw,
        method="stack_warper",
        method_params={"annotations": ["evt_a"], "drop_mode": "exact"},
        linear_warper_fn=fake_linear,
        pad_warper_fn=fake_pad,
        concat_warper_fn=fake_concat,
        stack_warper_fn=fake_stack,
    )
    assert captured["stack"]["drop_mode"] == "exact"

    raw_empty = mne.io.RawArray(np.zeros((1, 200), dtype=float), info)
    with pytest.raises(ValueError, match="pad warper"):
        alignment_service._build_warper(
            raw_empty,
            method="pad_warper",
            method_params={
                "annotations": [],
                "pad_left": 0.5,
                "anno_left": 0.5,
                "anno_right": 0.5,
                "pad_right": 0.5,
            },
            linear_warper_fn=fake_linear,
            pad_warper_fn=fake_pad,
            concat_warper_fn=fake_concat,
            stack_warper_fn=fake_stack,
        )
    with pytest.raises(ValueError, match="stack warper"):
        alignment_service._build_warper(
            raw_empty,
            method="stack_warper",
            method_params={"annotations": []},
            linear_warper_fn=fake_linear,
            pad_warper_fn=fake_pad,
            concat_warper_fn=fake_concat,
            stack_warper_fn=fake_stack,
        )


def test_run_align_epochs_requires_nonempty_annotation_selection_for_event_methods(
    tmp_path: Path,
) -> None:
    context, resolver, store = _prepare_run_align_epochs_fixture(tmp_path)
    created, _, entry = create_alignment_paradigm(store, name="Task", context=context)
    assert created and isinstance(entry, dict)
    slug = str(entry["slug"])
    updated, message = update_alignment_paradigm(
        store,
        slug=slug,
        method="stack_warper",
        method_params={
            "annotations": [],
            "duration_range": [0.0, 10_000.0],
            "drop_bad": False,
            "sample_rate": 0.8,
        },
        context=context,
    )
    assert updated, message

    ok, message, rows = run_align_epochs(
        context,
        config_store=store,
        paradigm_slug=slug,
    )
    assert not ok
    assert rows == []
    assert "Select at least one annotation label" in message
    assert (
        indicator_from_log(
            resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json"
        )
        == "yellow"
    )


def test_alignment_build_warper_additional_deep_branches() -> None:
    info = mne.create_info(["CH1"], sfreq=1000.0, ch_types=["misc"])
    raw_single = mne.io.RawArray(np.zeros((1, 1000), dtype=float), info)
    raw_single.set_annotations(
        mne.Annotations(onset=[0.1], duration=[0.2], description=["only"])
    )

    captured: dict[str, dict[str, object]] = {}

    def fake_linear(raw_obj, **kwargs):
        _ = raw_obj
        captured["linear"] = kwargs
        return {"ALL": [object()]}, (lambda tensor, **kw: (tensor, np.array([0.0]), []))

    def fake_pad(raw_obj, **kwargs):
        _ = raw_obj
        captured["pad"] = kwargs
        return {"ALL": [object()]}, (lambda tensor, **kw: (tensor, np.array([0.0]), []))

    def fake_concat(raw_obj, **kwargs):
        _ = raw_obj
        captured["concat"] = kwargs
        return {"ALL": [object()]}, (lambda tensor, **kw: (tensor, np.array([0.0]), []))

    alignment_service._build_warper(
        raw_single,
        method="linear_warper",
        method_params={"anchors_percent": {}},
        linear_warper_fn=fake_linear,
        pad_warper_fn=fake_pad,
    )
    assert captured["linear"]["anchors_percent"] == {0.0: "only", 100.0: "only"}

    with pytest.raises(ValueError, match="linear warper anchors"):
        alignment_service._build_warper(
            mne.io.RawArray(np.zeros((1, 100), dtype=float), info),
            method="linear_warper",
            method_params={"anchors_percent": {}},
            linear_warper_fn=fake_linear,
            pad_warper_fn=fake_pad,
        )

    raw_two = mne.io.RawArray(np.zeros((1, 2000), dtype=float), info)
    raw_two.set_annotations(
        mne.Annotations(
            onset=[0.1, 0.5, 0.9],
            duration=[0.2, 0.7, 1.8],
            description=["evt_a", "evt_b", "evt_b"],
        )
    )
    alignment_service._build_warper(
        raw_two,
        method="pad_warper",
        method_params={
            "annotations": "bad",
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
        },
        linear_warper_fn=fake_linear,
        pad_warper_fn=fake_pad,
    )
    assert set(captured["pad"]["anno_allowed"]) == {"evt_a", "evt_b"}

    alignment_service._build_warper(
        raw_two,
        method="concat_warper",
        method_params={"annotations": ["evt_a", "evt_b"]},
        concat_warper_fn=fake_concat,
        linear_warper_fn=fake_linear,
        pad_warper_fn=fake_pad,
    )
    assert captured["concat"]["keep"] == ["evt_a", "evt_b"]

    with pytest.raises(ValueError, match="concat warper"):
        alignment_service._build_warper(
            mne.io.RawArray(np.zeros((1, 100), dtype=float), info),
            method="concat_warper",
            method_params={"annotations": []},
            linear_warper_fn=fake_linear,
            pad_warper_fn=fake_pad,
        )


def test_alignment_concat_builder_preserves_full_annotation_set_for_drop_matching() -> (
    None
):
    info = mne.create_info(["CH1"], sfreq=100.0, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, 300), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.0, 0.0, 1.0],
            duration=[0.0, 1.0, 0.0],
            description=["EDGE_concat", "sit", "EDGE_concat"],
        )
    )
    captured: dict[str, object] = {}

    def fake_concat(raw_obj, **kwargs):
        captured["descriptions"] = list(raw_obj.annotations.description)
        captured["kwargs"] = kwargs
        return {"ALL": [object()]}, (lambda tensor, **kw: (tensor, np.array([0.0]), []))

    alignment_service._build_warper(
        raw,
        method="concat_warper",
        method_params={
            "annotations": ["sit"],
            "drop_bad": True,
            "drop_fields": ["bad", "edge"],
        },
        concat_warper_fn=fake_concat,
    )
    assert captured["descriptions"] == ["EDGE_concat", "sit", "EDGE_concat"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["keep"] == ["sit"]
    assert kwargs["anno_drop"] == ("bad", "edge")


def test_stack_warper_keep_and_drop_modes_are_independent() -> None:
    info = mne.create_info(["CH1"], sfreq=100.0, ch_types=["misc"])
    raw = mne.io.RawArray(np.zeros((1, 200), dtype=float), info)
    raw.set_annotations(
        mne.Annotations(
            onset=[0.50, 0.55],
            duration=[0.20, 0.10],
            description=["event", "bad-zone"],
        )
    )

    epochs_exact_keep, _ = alignment_service.stack_warper(
        raw,
        keep=["eve"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
        require_match=False,
    )
    assert len(epochs_exact_keep["ALL"]) == 0

    epochs_sub_keep, _ = alignment_service.stack_warper(
        raw,
        keep=["eve"],
        mode="substring",
        anno_drop=["bad"],
        drop_mode="exact",
        require_match=False,
    )
    assert len(epochs_sub_keep["ALL"]) == 1

    epochs_drop_exact, _ = alignment_service.stack_warper(
        raw,
        keep=["event"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="exact",
        require_match=False,
    )
    assert len(epochs_drop_exact["ALL"]) == 1

    epochs_drop_sub, _ = alignment_service.stack_warper(
        raw,
        keep=["event"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
        require_match=False,
    )
    assert len(epochs_drop_sub["ALL"]) == 0


def test_other_alignment_warpers_use_drop_mode_for_blacklist_matching() -> None:
    info = mne.create_info(["CH1"], sfreq=100.0, ch_types=["misc"])

    raw_linear = mne.io.RawArray(np.zeros((1, 200), dtype=float), info)
    raw_linear.set_annotations(
        mne.Annotations(
            onset=[0.10, 0.20, 0.50],
            duration=[0.00, 0.10, 0.00],
            description=["start", "bad-zone", "end"],
        )
    )
    epochs_linear_exact, _ = alignment_service.linear_warper(
        raw_linear,
        anchors_percent={0.0: "start", 100.0: "end"},
        mode="exact",
        anno_drop=["bad"],
        drop_mode="exact",
    )
    assert len(epochs_linear_exact["ALL"]) == 1
    epochs_linear_sub, _ = alignment_service.linear_warper(
        raw_linear,
        anchors_percent={0.0: "start", 100.0: "end"},
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
    )
    assert len(epochs_linear_sub["ALL"]) == 0

    raw_pad = mne.io.RawArray(np.zeros((1, 200), dtype=float), info)
    raw_pad.set_annotations(
        mne.Annotations(
            onset=[0.50, 0.55],
            duration=[0.20, 0.10],
            description=["event", "bad-zone"],
        )
    )
    epochs_pad_exact, _ = alignment_service.pad_warper(
        raw_pad,
        anno_allowed={"event": (0.0, 0.05, 0.05, 0.0)},
        mode="exact",
        anno_drop=["bad"],
        drop_mode="exact",
    )
    assert len(epochs_pad_exact["ALL"]) == 1
    epochs_pad_sub, warp_fn_pad_sub = alignment_service.pad_warper(
        raw_pad,
        anno_allowed={"event": (0.0, 0.05, 0.05, 0.0)},
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
    )
    assert len(epochs_pad_sub["ALL"]) == 0
    assert warp_fn_pad_sub is None

    raw_concat = mne.io.RawArray(np.zeros((1, 200), dtype=float), info)
    raw_concat.set_annotations(
        mne.Annotations(
            onset=[0.50, 0.55],
            duration=[0.20, 0.10],
            description=["event", "bad-zone"],
        )
    )
    epochs_concat_exact, _ = alignment_service.concat_warper(
        raw_concat,
        keep=["event"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="exact",
        require_match=False,
    )
    assert len(epochs_concat_exact["ALL"][0].intervals_s) == 1
    epochs_concat_sub, _ = alignment_service.concat_warper(
        raw_concat,
        keep=["event"],
        mode="exact",
        anno_drop=["bad"],
        drop_mode="substring",
        require_match=False,
    )
    assert epochs_concat_sub["ALL"][0].intervals_s == []


def test_finish_alignment_and_epoch_row_loader_branch_paths(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "paradigm-task"

    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug="   ",
        picked_epoch_indices=[0],
    )
    assert not ok and "slug is empty" in message

    write_run_log(
        resolver.alignment_paradigm_dir(slug) / "lfptensorpipe_log.json",
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path="out",
            message="ready",
        ),
    )
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[],
    )
    assert not ok and "Select at least one epoch" in message
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[-1],
    )
    assert not ok and "Select at least one valid epoch" in message
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert not ok and "No warped tensor outputs found" in message

    metric_path = (
        resolver.alignment_paradigm_dir(slug) / "raw_power" / "tensor_warped.pkl"
    )
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(["bad"], metric_path)
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert not ok and "No raw-table outputs were generated" in message

    save_pkl({"tensor": np.zeros((2, 3, 4), dtype=float), "meta": {}}, metric_path)
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert not ok and "No raw-table outputs were generated" in message

    save_pkl(
        {"tensor": np.zeros((1, 1, 1, 2), dtype=float), "meta": {"axes": "bad"}},
        metric_path,
    )
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[0],
    )
    assert not ok and "No raw-table outputs were generated" in message

    save_pkl(
        {
            "tensor": np.zeros((1, 1, 1, 2), dtype=float),
            "meta": {
                "axes": {
                    "epoch": ["e0"],
                    "channel": ["ch0"],
                    "freq": [10.0],
                    "time": [0.0, 1.0],
                }
            },
        },
        metric_path,
    )
    ok, message = finish_alignment_epochs(
        context,
        paradigm_slug=slug,
        picked_epoch_indices=[5],
    )
    assert not ok and "No raw-table outputs were generated" in message

    assert load_alignment_annotation_labels(context) == []
    assert (
        alignment_service.load_alignment_epoch_rows(context, paradigm_slug=slug) == []
    )
    labels_path = resolver.alignment_paradigm_dir(slug) / "warp_labels.pkl"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("not a pickle", encoding="utf-8")
    assert (
        alignment_service.load_alignment_epoch_rows(context, paradigm_slug=slug) == []
    )
    save_pkl(["bad"], labels_path)
    assert (
        alignment_service.load_alignment_epoch_rows(context, paradigm_slug=slug) == []
    )

    class _Epoch:
        def __init__(self, label: str, start_t: float, end_t: float) -> None:
            self.label = label
            self.start_t = start_t
            self.end_t = end_t

    save_pkl({"ALL": [_Epoch("event", 0.1, 0.9)]}, labels_path)
    rows = alignment_service.load_alignment_epoch_rows(context, paradigm_slug=slug)
    assert len(rows) == 1
    assert rows[0]["epoch_label"] == "event"
    assert rows[0]["duration_s"] == pytest.approx(0.8)

    class _ConcatEpoch:
        def __init__(self, label: str, total_duration_s: float) -> None:
            self.label = label
            self.total_duration_s = total_duration_s

    save_pkl({"ALL": [_ConcatEpoch("stitched", 1.5)]}, labels_path)
    rows = alignment_service.load_alignment_epoch_rows(context, paradigm_slug=slug)
    assert len(rows) == 1
    assert rows[0]["epoch_label"] == "stitched"
    assert rows[0]["duration_s"] == pytest.approx(1.5)


def test_alignment_duration_validation_and_paradigm_normalization_branches(
    tmp_path: Path,
) -> None:
    ok, _, message = validate_alignment_method_params(
        "linear_warper",
        {
            "anchors_percent": {0.0: "start", 100.0: "end"},
            "epoch_duration_range": [2.0, 1.0],
            "sample_rate": 0.16,
        },
        annotation_labels=["start", "end"],
    )
    assert not ok and "max must be >= duration min" in message

    ok, _, message = validate_alignment_method_params(
        "stack_warper",
        {
            "annotations": ["event"],
            "duration_range": [2.0, 1.0],
            "sample_rate": 0.16,
        },
        annotation_labels=["event"],
    )
    assert not ok and "max must be >= duration min" in message

    normalized = alignment_service._normalize_paradigm(
        {
            "name": "",
            "slug": "",
            "method": "stack_warper",
            "method_params": {
                "annotations": [],
                "duration_range": [0.0, 1.0],
                "sample_rate": 0.16,
            },
            "annotation_filter": "bad",
        }
    )
    assert normalized["name"] == "Trial"
    assert normalized["slug"] == "trial"
    assert normalized["trial_slug"] == "trial"
    assert normalized["annotation_filter"] == {}
    assert normalized["method_params"]["annotations"] == []
    assert normalized["method_params"]["duration_range"] == [0.0, 1.0]
    assert normalized["method_params"]["sample_rate"] == 0.16
    assert normalized["method_params_by_method"]["stack_warper"]["annotations"] == []
    assert normalized["method_params_by_method"]["stack_warper"]["sample_rate"] == 0.16

    store = AppConfigStore(repo_root=tmp_path / "repo")
    store.ensure_core_files()
    store.write_yaml(
        "alignment.yml",
        {
            "paradigms": [
                "bad",
                {"name": "A", "slug": "a"},
                {"name": "A-dup", "slug": "a"},
            ]
        },
    )
    loaded = load_alignment_paradigms(store)
    assert [item["slug"] for item in loaded] == ["a"]
