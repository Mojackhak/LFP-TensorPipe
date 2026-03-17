"""Tests for Localize service (record-scoped contracts)."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app import localize_service
from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.localize_service import (
    _default_contact_viewer_launcher,
    LocalizePaths,
    can_open_contact_viewer,
    discover_atlases,
    discover_spaces,
    has_reconstruction_mat,
    infer_subject_space,
    infer_subject_spaces,
    launch_contact_viewer,
    load_localize_paths,
    localize_indicator_state,
    localize_log_path,
    localize_match_signature,
    localize_ordered_pair_representative_csv_path,
    localize_ordered_pair_representative_pkl_path,
    localize_panel_state,
    localize_representative_csv_path,
    localize_representative_pkl_path,
    localize_undirected_pair_representative_csv_path,
    localize_undirected_pair_representative_pkl_path,
    reconstruction_mat_path,
    run_localize_apply,
)
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    indicator_from_log,
    read_run_log,
    write_run_log,
    write_ui_state,
)
from lfptensorpipe.io.pkl_io import load_pkl


class _FakeLocalizeMatlabEngine:
    def __init__(self) -> None:
        self.addpath_calls: list[str] = []
        self.genpath_arg: str | None = None

    def genpath(self, path: str) -> str:
        self.genpath_arg = path
        return f"GENPATH::{path}"

    def addpath(self, path: str, nargout: int = 0) -> None:
        _ = nargout
        self.addpath_calls.append(path)

    def quit(self) -> None:
        return None


def _paths(tmp_path: Path) -> LocalizePaths:
    leaddbs = tmp_path / "leaddbs"
    matlab = tmp_path / "matlab_engine"
    leaddbs.mkdir(parents=True, exist_ok=True)
    matlab.mkdir(parents=True, exist_ok=True)
    return LocalizePaths(leaddbs_dir=leaddbs, matlab_engine_path=matlab)


def test_local_matlab_functions_dir_resolves_repo_helper_dir() -> None:
    helper_dir = localize_service._local_matlab_functions_dir()
    assert helper_dir.is_dir()
    assert helper_dir.as_posix().endswith("src/lfptensorpipe/anat/leaddbs")


def test_ensure_matlab_engine_ready_adds_repo_helper_dir(
    tmp_path: Path,
) -> None:
    fake_engine = _FakeLocalizeMatlabEngine()
    leaddbs_dir = tmp_path / "leaddbs"
    leaddbs_dir.mkdir()
    matlab_engine_path = tmp_path / "matlab_engine"
    matlab_engine_path.mkdir()
    (matlab_engine_path / "setup.py").write_text("x", encoding="utf-8")

    paths = LocalizePaths(
        leaddbs_dir=leaddbs_dir,
        matlab_engine_path=matlab_engine_path,
    )

    localize_service._drop_matlab_engine()
    try:
        engine = localize_service._ensure_matlab_engine_ready(
            paths,
            ensure_matlab_engine_fn=lambda _path: None,
            start_matlab_fn=lambda: fake_engine,
        )
        assert engine is fake_engine
        assert fake_engine.genpath_arg == str(leaddbs_dir)
        assert any(call.startswith("GENPATH::") for call in fake_engine.addpath_calls)
        assert any(
            call.endswith("src/lfptensorpipe/anat/leaddbs")
            for call in fake_engine.addpath_calls
        )
    finally:
        localize_service._drop_matlab_engine()


def _write_match_payload_to_record_ui_state(
    project_root: Path, subject: str, record: str, payload: dict[str, object]
) -> None:
    resolver = PathResolver(
        RecordContext(project_root=project_root, subject=subject, record=record)
    )
    write_ui_state(
        resolver.record_ui_state_path(create=True),
        {"localize": {"match": payload}},
    )


def _write_match_payload_to_legacy_record_log(
    project_root: Path, subject: str, record: str, payload: dict[str, object]
) -> None:
    resolver = PathResolver(
        RecordContext(project_root=project_root, subject=subject, record=record)
    )
    write_run_log(
        resolver.lfp_root / "lfptensorpipe_log.json",
        RunLogRecord(
            step="record_params_sync",
            completed=True,
            params={"localize": {"match": payload}},
        ),
    )


def test_localize_path_loading_and_discovery(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    leaddbs_dir = tmp_path / "leaddbs"
    matlab_root = tmp_path / "matlab"
    (leaddbs_dir / "templates" / "space" / "mni").mkdir(parents=True)
    (leaddbs_dir / "templates" / "space" / "native").mkdir(parents=True)
    (leaddbs_dir / "templates" / "space" / "mni" / "atlases" / "atlas_a").mkdir(
        parents=True
    )
    (leaddbs_dir / "templates" / "space" / "mni" / "atlases" / "atlas_b").mkdir(
        parents=True
    )
    matlab_root.mkdir(parents=True)

    store.write_yaml(
        "paths.yml",
        {
            "leaddbs_dir": str(leaddbs_dir),
            "matlab_root": str(matlab_root),
        },
    )
    paths = load_localize_paths(store)
    assert discover_spaces(paths.leaddbs_dir) == ["mni", "native"]
    assert discover_atlases(paths.leaddbs_dir, "mni") == ["atlas_a", "atlas_b"]
    ok, _ = can_open_contact_viewer(paths)
    assert ok
    migrated = store.read_yaml("paths.yml", default={})
    assert migrated["matlab_root"] == str(matlab_root)
    assert "matlab_engine_path" not in migrated


def test_load_localize_paths_handles_non_dict_payload(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.write_yaml("paths.yml", ["bad", "payload"])
    paths = load_localize_paths(store)
    assert paths.leaddbs_dir == Path("__missing_leaddbs_dir__")
    assert paths.matlab_root == Path("__missing_matlab_root__")


def test_discovery_helpers_return_empty_for_missing_paths(tmp_path: Path) -> None:
    assert discover_spaces(tmp_path / "missing_leaddbs") == []
    assert discover_atlases(tmp_path / "missing_leaddbs", "missing_space") == []


def test_infer_subject_space_prefers_transformations(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    trans_dir = (
        project
        / "derivatives"
        / "leaddbs"
        / subject
        / "normalization"
        / "transformations"
    )
    anat_dir = project / "derivatives" / "leaddbs" / subject / "normalization" / "anat"
    trans_dir.mkdir(parents=True)
    anat_dir.mkdir(parents=True)
    (trans_dir / f"{subject}_from-anchorNative_to-MNI_desc-ants.nii.gz").write_text(
        "x",
        encoding="utf-8",
    )
    (anat_dir / f"{subject}_acq-test_space-Native_desc.nii").write_text(
        "x",
        encoding="utf-8",
    )
    spaces = infer_subject_spaces(project, subject)
    assert spaces == ["MNI"]
    space, message = infer_subject_space(project, subject)
    assert space == "MNI"
    assert message == ""


def test_infer_subject_space_falls_back_to_anat(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    anat_dir = project / "derivatives" / "leaddbs" / subject / "normalization" / "anat"
    anat_dir.mkdir(parents=True)
    (anat_dir / f"{subject}_acq-test_space-fsaverage_desc.nii.gz").write_text(
        "x",
        encoding="utf-8",
    )
    space, message = infer_subject_space(project, subject)
    assert space == "fsaverage"
    assert message == ""


def test_infer_subject_space_reports_multiple_spaces(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    trans_dir = (
        project
        / "derivatives"
        / "leaddbs"
        / subject
        / "normalization"
        / "transformations"
    )
    trans_dir.mkdir(parents=True)
    (trans_dir / f"{subject}_from-anchorNative_to-MNI_desc-ants.nii.gz").write_text(
        "x",
        encoding="utf-8",
    )
    (trans_dir / f"{subject}_from-anchorNative_to-Native_desc-ants.nii.gz").write_text(
        "x",
        encoding="utf-8",
    )
    space, message = infer_subject_space(project, subject)
    assert space is None
    assert "Multiple spaces discovered" in message


def test_reconstruction_path_helpers(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    mat_path = reconstruction_mat_path(project, subject)
    assert not has_reconstruction_mat(project, subject)
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    mat_path.write_bytes(b"MAT")
    assert has_reconstruction_mat(project, subject)


def test_localize_panel_indicator_tracks_atlas_and_match_staleness(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    match_payload = {
        "completed": True,
        "mappings": [
            {
                "channel": "CH1",
                "anode": "L1",
                "cathode": "L2",
                "rep_coord": "Mid",
            }
        ],
    }
    write_run_log(
        localize_log_path(project, subject, record),
        RunLogRecord(
            step="localize_apply",
            completed=True,
            params={
                "atlas": "AtlasA",
                "selected_regions_signature": ["SNr", "STN"],
                "match_signature": localize_match_signature(match_payload),
            },
            input_path="in",
            output_path="out",
            message="localize ready",
        ),
    )

    assert (
        localize_panel_state(
            project,
            subject,
            record,
            atlas="AtlasA",
            selected_regions=["SNr", "STN"],
            match_payload=match_payload,
        )
        == "green"
    )
    assert (
        localize_panel_state(
            project,
            subject,
            record,
            atlas="AtlasB",
            selected_regions=["SNr", "STN"],
            match_payload=match_payload,
        )
        == "yellow"
    )
    assert (
        localize_panel_state(
            project,
            subject,
            record,
            atlas="AtlasA",
            selected_regions=["SNr"],
            match_payload=match_payload,
        )
        == "yellow"
    )
    assert (
        localize_panel_state(
            project,
            subject,
            record,
            atlas="AtlasA",
            selected_regions=["SNr", "STN"],
            match_payload={
                "completed": True,
                "mappings": [
                    {
                        "channel": "CH1",
                        "anode": "L1",
                        "cathode": "L3",
                        "rep_coord": "Mid",
                    }
                ],
            },
        )
        == "yellow"
    )
    assert (
        localize_panel_state(
            project,
            subject,
            record,
            atlas="AtlasA",
            selected_regions=["SNr", "STN"],
            match_payload=match_payload,
        )
        == "green"
    )
    write_run_log(
        localize_log_path(project, subject, record),
        RunLogRecord(
            step="localize_apply",
            completed=False,
            params={"atlas": "AtlasA"},
            input_path="in",
            output_path="out",
            message="localize failed",
        ),
    )
    assert (
        localize_panel_state(
            project,
            subject,
            record,
            atlas="AtlasA",
            selected_regions=["SNr", "STN"],
            match_payload=match_payload,
        )
        == "yellow"
    )


def test_run_localize_apply_blocks_read_only_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    ok, _ = run_localize_apply(
        project_root=project,
        subject=subject,
        record=record,
        space="MNI",
        atlas="AtlasA",
        selected_regions=("SNr",),
        read_only_project_root=project,
    )
    assert not ok
    assert not localize_log_path(project, subject, record).exists()


def test_run_localize_apply_fails_without_match_payload_and_marks_yellow(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    ok, message = run_localize_apply(
        project_root=project,
        subject=subject,
        record=record,
        space="MNI",
        atlas="AtlasA",
        selected_regions=("SNr",),
        paths=_paths(tmp_path),
    )
    assert not ok
    assert "Missing match payload in record UI state" in message
    assert localize_indicator_state(project, subject, record) == "yellow"


def test_run_localize_apply_rejects_incomplete_match_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    _write_match_payload_to_record_ui_state(
        project,
        subject,
        record,
        {
            "completed": False,
            "channels": ["0_1"],
            "mappings": [{"channel": "0_1", "anode": "R_K0 (R)", "cathode": "case"}],
        },
    )
    ok, message = run_localize_apply(
        project_root=project,
        subject=subject,
        record=record,
        space="MNI",
        atlas="AtlasA",
        selected_regions=("SNr",),
        paths=_paths(tmp_path),
    )
    assert not ok
    assert "not marked completed" in message
    assert localize_indicator_state(project, subject, record) == "yellow"


def test_run_localize_apply_success_writes_representative_artifacts(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    _write_match_payload_to_record_ui_state(
        project,
        subject,
        record,
        {
            "completed": True,
            "channels": ["0_1"],
            "mappings": [
                {
                    "channel": "0_1",
                    "anode": "R_K0 (R)",
                    "cathode": "R_K1 (R)",
                    "rep_coord": "Mid",
                }
            ],
        },
    )
    alignment_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "alignment"
        / "gait"
        / "lfptensorpipe_log.json"
    )
    write_run_log(
        alignment_log,
        RunLogRecord(
            step="build_raw_table",
            completed=True,
            params={},
            input_path="in",
            output_path="out",
            message="align finish ready",
        ),
    )
    features_log = (
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "features"
        / "gait"
        / "lfptensorpipe_log.json"
    )
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

    captured_region_names: list[str] = []

    def _build_repcoords_frame(**kwargs) -> pd.DataFrame:
        captured_region_names.extend(list(kwargs.get("region_names", ())))
        return pd.DataFrame(
            [
                {
                    "subject": subject,
                    "record": record,
                    "space": "MNI",
                    "atlas": "AtlasA",
                    "channel": "CH1",
                    "anode": "R_K0 (R)",
                    "cathode": "R_K1 (R)",
                    "rep_coord": "Mid",
                    "mni_x": 1.0,
                    "mni_y": 2.0,
                    "mni_z": 3.0,
                    "SNr_in": True,
                    "STN_in": False,
                },
                {
                    "subject": subject,
                    "record": record,
                    "space": "MNI",
                    "atlas": "AtlasA",
                    "channel": "CH2",
                    "anode": "R_K2 (R)",
                    "cathode": "R_K3 (R)",
                    "rep_coord": "Mid",
                    "mni_x": 4.0,
                    "mni_y": 5.0,
                    "mni_z": 6.0,
                    "SNr_in": False,
                    "STN_in": True,
                },
            ]
        )

    ok, message = run_localize_apply(
        project_root=project,
        subject=subject,
        record=record,
        space="MNI",
        atlas="AtlasA",
        selected_regions=("SNr", "STN"),
        paths=_paths(tmp_path),
        load_reconstruction_contacts_fn=lambda *_args, **_kwargs: (
            True,
            "",
            {"leads": []},
        ),
        build_repcoords_frame_fn=_build_repcoords_frame,
    )
    assert ok, message
    assert captured_region_names == ["SNr", "STN"]
    out_csv = localize_representative_csv_path(project, subject, record)
    out_pkl = localize_representative_pkl_path(project, subject, record)
    ordered_csv = localize_ordered_pair_representative_csv_path(
        project, subject, record
    )
    ordered_pkl = localize_ordered_pair_representative_pkl_path(
        project, subject, record
    )
    undirected_csv = localize_undirected_pair_representative_csv_path(
        project, subject, record
    )
    undirected_pkl = localize_undirected_pair_representative_pkl_path(
        project, subject, record
    )
    assert out_csv.is_file()
    assert out_pkl.is_file()
    assert ordered_csv.is_file()
    assert ordered_pkl.is_file()
    assert undirected_csv.is_file()
    assert undirected_pkl.is_file()

    frame = load_pkl(out_pkl)
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[0] == 2
    assert list(frame["channel"]) == ["CH1", "CH2"]

    ordered_frame = load_pkl(ordered_pkl)
    assert isinstance(ordered_frame, pd.DataFrame)
    assert ordered_frame.shape[0] == 2
    assert ordered_frame.iloc[0]["mni_x"] == (1.0, 4.0)
    assert "SNr-STN_in" in ordered_frame.columns
    assert "STN-SNr_in" in ordered_frame.columns

    undirected_frame = load_pkl(undirected_pkl)
    assert isinstance(undirected_frame, pd.DataFrame)
    assert undirected_frame.shape[0] == 1
    assert undirected_frame.iloc[0]["mni_x"] == (1.0, 4.0)
    assert "SNr-STN_in" in undirected_frame.columns
    assert "STN-SNr_in" not in undirected_frame.columns
    assert bool(undirected_frame.iloc[0]["SNr-STN_in"]) is True

    assert localize_indicator_state(project, subject, record) == "green"
    log = read_run_log(localize_log_path(project, subject, record))
    assert log is not None
    assert log["completed"] is True
    assert log["params"].get("record") == record
    assert log["params"].get("selected_regions_signature") == ["SNr", "STN"]
    assert log["params"].get("ordered_pair_rows") == 2
    assert log["params"].get("undirected_pair_rows") == 1
    assert indicator_from_log(alignment_log) == "yellow"
    assert indicator_from_log(features_log) == "yellow"


def test_run_localize_apply_reads_legacy_record_params_fallback(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    _write_match_payload_to_legacy_record_log(
        project,
        subject,
        record,
        {
            "completed": True,
            "channels": ["0_1"],
            "mappings": [
                {
                    "channel": "0_1",
                    "anode": "R_K0 (R)",
                    "cathode": "R_K1 (R)",
                    "rep_coord": "Mid",
                }
            ],
        },
    )

    ok, message = run_localize_apply(
        project_root=project,
        subject=subject,
        record=record,
        space="MNI",
        atlas="AtlasA",
        selected_regions=("SNr",),
        paths=_paths(tmp_path),
        load_reconstruction_contacts_fn=lambda *_args, **_kwargs: (
            True,
            "",
            {"leads": []},
        ),
        build_repcoords_frame_fn=lambda **_kwargs: pd.DataFrame(
            [
                {
                    "subject": subject,
                    "record": record,
                    "space": "MNI",
                    "atlas": "AtlasA",
                    "channel": "0_1",
                    "anode": "R_K0 (R)",
                    "cathode": "R_K1 (R)",
                    "rep_coord": "Mid",
                    "mni_x": 1.0,
                    "mni_y": 2.0,
                    "mni_z": 3.0,
                }
            ]
        ),
    )
    assert ok, message


def test_can_open_contact_viewer_reports_invalid_paths(tmp_path: Path) -> None:
    missing_leaddbs = LocalizePaths(
        leaddbs_dir=tmp_path / "missing_leaddbs",
        matlab_engine_path=tmp_path / "matlab_engine",
    )
    ok, message = can_open_contact_viewer(missing_leaddbs)
    assert not ok
    assert "Invalid Lead-DBS path" in message

    valid_leaddbs = tmp_path / "leaddbs"
    valid_leaddbs.mkdir(parents=True)
    missing_matlab = LocalizePaths(
        leaddbs_dir=valid_leaddbs,
        matlab_engine_path=tmp_path / "missing_matlab_engine",
    )
    ok, message = can_open_contact_viewer(missing_matlab)
    assert not ok
    assert "Invalid MATLAB installation path" in message


def test_launch_contact_viewer_requires_representative_csv(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    paths = _paths(tmp_path)
    ok, message = launch_contact_viewer(
        project_root=project,
        subject=subject,
        record=record,
        atlas="AtlasA",
        paths=paths,
    )
    assert not ok
    assert "Missing Localize representative CSV" in message


def test_launch_contact_viewer_uses_injected_launcher(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    paths = _paths(tmp_path)
    csv_path = localize_representative_csv_path(project, subject, record)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("channel,mni_x,mni_y,mni_z\n0_1,1,2,3\n", encoding="utf-8")

    captured: dict[str, str] = {}

    def fake_launcher(
        source_csv: Path, atlas: str, localize_paths: LocalizePaths
    ) -> None:
        captured["csv"] = str(source_csv)
        captured["atlas"] = atlas
        captured["leaddbs"] = str(localize_paths.leaddbs_dir)

    ok, _ = launch_contact_viewer(
        project_root=project,
        subject=subject,
        record=record,
        atlas="AtlasA",
        paths=paths,
        launcher=fake_launcher,
    )
    assert ok
    assert captured["csv"] == str(csv_path)
    assert captured["atlas"] == "AtlasA"


def test_launch_contact_viewer_reports_launcher_exception(tmp_path: Path) -> None:
    project = tmp_path / "project"
    subject = "sub-001"
    record = "runA"
    paths = _paths(tmp_path)
    csv_path = localize_representative_csv_path(project, subject, record)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("channel,mni_x,mni_y,mni_z\n0_1,1,2,3\n", encoding="utf-8")

    ok, message = launch_contact_viewer(
        project_root=project,
        subject=subject,
        record=record,
        atlas="AtlasA",
        paths=paths,
        launcher=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert not ok
    assert "launch failed" in message


def test_default_contact_viewer_launcher_invokes_worker_process(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "channel_representative_coords.csv"
    csv_path.write_text("channel,mni_x,mni_y,mni_z\n0_1,1,2,3\n", encoding="utf-8")
    paths = _paths(tmp_path)
    calls: dict[str, object] = {}

    def _fake_popen(cmd: list[str], start_new_session: bool = False) -> object:
        calls["cmd"] = cmd
        calls["start_new_session"] = start_new_session
        return object()

    _default_contact_viewer_launcher(
        csv_path,
        "AtlasA",
        paths,
        popen=_fake_popen,
    )
    assert calls["start_new_session"] is True
    cmd = calls["cmd"]
    assert isinstance(cmd, list)
    assert "lfptensorpipe.app.localize_viewer_worker" in cmd
    assert str(csv_path) in cmd
    assert "AtlasA" in cmd


def test_warmup_matlab_async_reuses_inflight_future(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    localize_service.reset_matlab_runtime(paths=None).result(timeout=2.0)

    started = threading.Event()
    release = threading.Event()
    calls = {"count": 0}

    def _fake_ensure(_paths: LocalizePaths) -> object:
        _ = _paths
        calls["count"] += 1
        started.set()
        release.wait(timeout=2.0)
        return object()

    future_a = localize_service.warmup_matlab_async(
        paths,
        ensure_matlab_engine_ready_fn=_fake_ensure,
    )
    assert started.wait(timeout=1.0)
    future_b = localize_service.warmup_matlab_async(
        paths,
        ensure_matlab_engine_ready_fn=_fake_ensure,
    )
    assert future_a is future_b

    release.set()
    ok, message = future_a.result(timeout=2.0)
    assert ok is True
    assert "ready" in message.lower()
    assert calls["count"] == 1

    localize_service.reset_matlab_runtime(paths=None).result(timeout=2.0)


def test_warmup_matlab_async_drops_stale_runtime_context(
    tmp_path: Path,
) -> None:
    localize_service.reset_matlab_runtime(paths=None).result(timeout=2.0)
    path_a = _paths(tmp_path / "a")
    path_b = _paths(tmp_path / "b")

    worker_started = threading.Event()
    worker_release = threading.Event()

    def _occupy_worker() -> None:
        worker_started.set()
        worker_release.wait(timeout=2.0)

    localize_service._MATLAB_TASK_EXECUTOR.submit(_occupy_worker)
    assert worker_started.wait(timeout=1.0)

    def fake_ensure(_paths: LocalizePaths) -> object:
        return object()

    stale_future = localize_service.warmup_matlab_async(
        path_a,
        ensure_matlab_engine_ready_fn=fake_ensure,
    )
    latest_future = localize_service.warmup_matlab_async(
        path_b,
        ensure_matlab_engine_ready_fn=fake_ensure,
    )

    worker_release.set()

    stale_ok, stale_message = stale_future.result(timeout=2.0)
    latest_ok, latest_message = latest_future.result(timeout=2.0)
    assert stale_ok is False
    assert localize_service.is_stale_context_message(stale_message)
    assert latest_ok is True
    assert "ready" in latest_message.lower()

    localize_service.reset_matlab_runtime(paths=None).result(timeout=2.0)
