"""Tests for dataset mutation services."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import mne
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfptensorpipe.app.dataset_actions import (
    _apply_bipolar_reference,
    _validate_bipolar_pairs,
    apply_reset_reference,
    create_subject,
    delete_record,
    import_record,
    load_import_channel_names,
    validate_record_name,
    validate_subject_name,
)
from lfptensorpipe.app.dataset_index import discover_records


def test_create_subject_and_duplicate_rejection(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    created, _ = create_subject(project, "sub-001")
    assert created
    duplicate, message = create_subject(project, "sub-001")
    assert not duplicate
    assert "already exists" in message


def test_import_csv_record_and_delete_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    created, _ = create_subject(project, "sub-001")
    assert created

    source_csv = tmp_path / "lfp.csv"
    pd.DataFrame(
        {
            "CH1": [1.0, 2.0, 3.0, 4.0],
            "CH2": [5.0, 6.0, 7.0, 8.0],
            "note": ["a", "b", "c", "d"],
        }
    ).to_csv(source_csv, index=False)

    channels = load_import_channel_names(source_csv, csv_sr=1000.0, csv_unit="uV")
    assert channels == ["CH1", "CH2"]

    imported = import_record(
        project_root=project,
        subject="sub-001",
        record="runA",
        source_path=source_csv,
        csv_sr=1000.0,
        csv_unit="uV",
    )
    assert imported.ok
    assert imported.raw_fif_path is not None and imported.raw_fif_path.exists()
    assert (
        imported.sourcedata_copy_path is not None
        and imported.sourcedata_copy_path.exists()
    )
    assert discover_records(project, "sub-001") == ["runA"]

    deleted = delete_record(project_root=project, subject="sub-001", record="runA")
    assert deleted.ok
    assert deleted.deleted_paths
    for deleted_path in deleted.deleted_paths:
        assert not deleted_path.exists()


def test_record_import_and_delete_block_on_read_only_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    created, _ = create_subject(project, "sub-001")
    assert created

    source_csv = tmp_path / "lfp.csv"
    pd.DataFrame({"CH1": [1.0, 2.0], "CH2": [3.0, 4.0]}).to_csv(source_csv, index=False)

    imported = import_record(
        project_root=project,
        subject="sub-001",
        record="runA",
        source_path=source_csv,
        csv_sr=1000.0,
        csv_unit="uV",
        read_only_project_root=project,
    )
    assert not imported.ok

    deleted = delete_record(
        project_root=project,
        subject="sub-001",
        record="runA",
        read_only_project_root=project,
    )
    assert not deleted.ok


def test_import_record_bipolar_keeps_only_new_named_channels(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    created, _ = create_subject(project, "sub-001")
    assert created

    source_csv = tmp_path / "lfp.csv"
    pd.DataFrame(
        {
            "LFP1": [1.0, 2.0, 3.0, 4.0],
            "LFP2": [0.5, 1.5, 2.5, 3.5],
            "LFP3": [2.0, 2.5, 3.0, 3.5],
        }
    ).to_csv(source_csv, index=False)

    imported = import_record(
        project_root=project,
        subject="sub-001",
        record="runB",
        source_path=source_csv,
        csv_sr=418.4,
        csv_unit="uV",
        bipolar_pairs=(("LFP1", "LFP2"), ("LFP3", "LFP2")),
        bipolar_names=("BIP_1", "BIP_2"),
    )
    assert imported.ok
    assert imported.raw_fif_path is not None and imported.raw_fif_path.exists()

    imported_raw = mne.io.read_raw_fif(
        str(imported.raw_fif_path), preload=False, verbose="ERROR"
    )
    assert imported_raw.ch_names == ["BIP_1", "BIP_2"]


def test_validate_name_contracts_and_create_subject_missing_project(
    tmp_path: Path,
) -> None:
    ok, msg = validate_subject_name("   ")
    assert not ok and "cannot be empty" in msg
    ok, msg = validate_subject_name("subject")
    assert not ok and "must match pattern" in msg

    ok, msg = validate_record_name("   ")
    assert not ok and "cannot be empty" in msg
    ok, msg = validate_record_name("bad name")
    assert not ok and "must match pattern" in msg

    created, msg = create_subject(tmp_path, "subject")
    assert not created and "must match pattern" in msg

    created, msg = create_subject(tmp_path / "missing_project", "sub-001")
    assert not created and "does not exist" in msg


def test_validate_bipolar_pairs_error_paths_and_default_names() -> None:
    class _Raw:
        ch_names = ["A", "B", "C"]

    with pytest.raises(ValueError, match="must match number"):
        _validate_bipolar_pairs(
            _Raw(),
            (("A", "B"),),
            bipolar_names=("only", "extra"),
        )
    with pytest.raises(ValueError, match="identical channels"):
        _validate_bipolar_pairs(_Raw(), (("A", "A"),))
    with pytest.raises(ValueError, match="anode channel not found"):
        _validate_bipolar_pairs(_Raw(), (("X", "B"),))
    with pytest.raises(ValueError, match="cathode channel not found"):
        _validate_bipolar_pairs(_Raw(), (("A", "X"),))

    names = _validate_bipolar_pairs(_Raw(), (("A", "B"), ("C", "B")))
    assert names == ("A-B", "C-B")

    with pytest.raises(ValueError, match="Empty bipolar channel name"):
        _validate_bipolar_pairs(_Raw(), (("A", "B"),), bipolar_names=("   ",))
    with pytest.raises(ValueError, match="Duplicate bipolar pairs"):
        _validate_bipolar_pairs(_Raw(), (("A", "B"), ("A", "B")))
    with pytest.raises(ValueError, match="Duplicate bipolar channel names"):
        _validate_bipolar_pairs(
            _Raw(),
            (("A", "B"), ("C", "B")),
            bipolar_names=("BP", "BP"),
        )


def test_apply_bipolar_reference_raises_when_expected_channels_missing() -> None:
    class _Raw:
        ch_names = ["A", "B"]

    class _BipolarRaw:
        ch_names = ["UNRELATED"]

        def pick_channels(self, channels: list[str], ordered: bool = True) -> None:
            _ = (channels, ordered)

    with pytest.raises(ValueError, match="Missing bipolar channels"):
        _apply_bipolar_reference(
            _Raw(),
            (("A", "B"),),
            set_bipolar_reference_fn=lambda *args, **kwargs: _BipolarRaw(),
        )


def test_apply_reset_reference_supports_unary_and_bipolar_rows() -> None:
    raw = mne.io.RawArray(
        np.asarray(
            [
                [1.0, 2.0, -3.0],
                [0.5, -1.0, 4.0],
            ],
            dtype=float,
        ),
        mne.create_info(
            ch_names=["A", "B"],
            sfreq=1000.0,
            ch_types=["misc", "misc"],
        ),
        verbose="ERROR",
    )
    out = apply_reset_reference(
        raw,
        (
            ("A", "", "A_keep"),
            ("", "B", "neg_B"),
            ("A", "B", "A_minus_B"),
        ),
    )
    assert out.ch_names == ["A_keep", "neg_B", "A_minus_B"]
    data = out.get_data()
    np.testing.assert_allclose(data[0], [1.0, 2.0, -3.0])
    np.testing.assert_allclose(data[1], [-0.5, 1.0, -4.0])
    np.testing.assert_allclose(data[2], [0.5, 3.0, -7.0])


def test_apply_reset_reference_rejects_empty_anode_and_cathode() -> None:
    raw = mne.io.RawArray(
        np.asarray([[1.0, 2.0]], dtype=float),
        mne.create_info(ch_names=["A"], sfreq=1000.0, ch_types=["misc"]),
        verbose="ERROR",
    )
    with pytest.raises(
        ValueError, match="At least one of anode or cathode is required."
    ):
        apply_reset_reference(raw, (("", "", "invalid"),))


def test_apply_reset_reference_allows_plus_and_minus_for_same_channel() -> None:
    raw = mne.io.RawArray(
        np.asarray([[1.5, -2.0]], dtype=float),
        mne.create_info(ch_names=["A"], sfreq=1000.0, ch_types=["misc"]),
        verbose="ERROR",
    )
    out = apply_reset_reference(
        raw,
        (
            ("A", "", "plus_A"),
            ("", "A", "minus_A"),
        ),
    )
    data = out.get_data()
    np.testing.assert_allclose(data[0], [1.5, -2.0])
    np.testing.assert_allclose(data[1], [-1.5, 2.0])


def test_load_import_channel_names_csv_requires_positive_sr(tmp_path: Path) -> None:
    source_csv = tmp_path / "lfp.csv"
    pd.DataFrame({"CH1": [1.0], "CH2": [2.0]}).to_csv(source_csv, index=False)

    with pytest.raises(ValueError, match="sr > 0"):
        load_import_channel_names(source_csv, csv_sr=None, csv_unit="uV")


def test_load_import_channel_names_reads_fif_via_mne_io(tmp_path: Path) -> None:
    raw = mne.io.RawArray(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        mne.create_info(ch_names=["L1", "L2"], sfreq=1000.0, ch_types=["misc", "misc"]),
    )
    source_fif = tmp_path / "raw.fif"
    raw.save(str(source_fif), overwrite=True)

    channels = load_import_channel_names(source_fif)
    assert channels == ["L1", "L2"]


def test_import_record_validation_error_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source_csv = tmp_path / "lfp.csv"
    pd.DataFrame({"CH1": [1.0], "CH2": [2.0]}).to_csv(source_csv, index=False)

    invalid_subject = import_record(
        project_root=project,
        subject="subject",
        record="runA",
        source_path=source_csv,
    )
    assert not invalid_subject.ok and "must match pattern" in invalid_subject.message

    invalid_record = import_record(
        project_root=project,
        subject="sub-001",
        record="bad name",
        source_path=source_csv,
    )
    assert not invalid_record.ok and "must match pattern" in invalid_record.message

    missing_project = import_record(
        project_root=tmp_path / "missing_project",
        subject="sub-001",
        record="runA",
        source_path=source_csv,
    )
    assert not missing_project.ok and "Missing project path" in missing_project.message

    missing_subject = import_record(
        project_root=project,
        subject="sub-001",
        record="runA",
        source_path=source_csv,
    )
    assert not missing_subject.ok and "Missing subject" in missing_subject.message

    created, _ = create_subject(project, "sub-001")
    assert created
    (project / "derivatives" / "lfptensorpipe" / "sub-001" / "runA").mkdir(parents=True)
    duplicate_record = import_record(
        project_root=project,
        subject="sub-001",
        record="runA",
        source_path=source_csv,
    )
    assert (
        not duplicate_record.ok and "Record already exists" in duplicate_record.message
    )

    missing_source = import_record(
        project_root=project,
        subject="sub-001",
        record="runB",
        source_path=tmp_path / "missing.csv",
    )
    assert not missing_source.ok and "Missing source file" in missing_source.message

    load_error = import_record(
        project_root=project,
        subject="sub-001",
        record="runC",
        source_path=source_csv,
        csv_sr=None,
    )
    assert not load_error.ok and "Failed to load source" in load_error.message


def test_import_record_reports_save_failure(tmp_path: Path) -> None:
    class _Raw:
        ch_names = ["A"]

        def save(self, path: str, overwrite: bool = True) -> None:
            _ = (path, overwrite)
            raise RuntimeError("save failed")

    project = tmp_path / "project"
    project.mkdir()
    created, _ = create_subject(project, "sub-001")
    assert created
    source_csv = tmp_path / "lfp.csv"
    pd.DataFrame({"CH1": [1.0]}).to_csv(source_csv, index=False)

    result = import_record(
        project_root=project,
        subject="sub-001",
        record="runZ",
        source_path=source_csv,
        load_raw_from_source_fn=lambda *args, **kwargs: (_Raw(), True),
        apply_bipolar_reference_fn=lambda raw, *_args: raw,
    )
    assert not result.ok
    assert "Failed to save raw.fif" in result.message


def test_delete_record_error_paths_and_file_unlink(tmp_path: Path) -> None:
    invalid_subject = delete_record(
        project_root=tmp_path,
        subject="subject",
        record="runA",
    )
    assert not invalid_subject.ok and "must match pattern" in invalid_subject.message

    invalid_record = delete_record(
        project_root=tmp_path,
        subject="sub-001",
        record="bad name",
    )
    assert not invalid_record.ok and "must match pattern" in invalid_record.message

    no_artifacts = delete_record(
        project_root=tmp_path,
        subject="sub-001",
        record="runA",
    )
    assert not no_artifacts.ok and "No record artifacts found" in no_artifacts.message

    file_target = tmp_path / "artifact.txt"
    file_target.write_text("data", encoding="utf-8")
    deleted = delete_record(
        project_root=tmp_path,
        subject="sub-001",
        record="runA",
        record_artifact_roots_fn=lambda *args, **kwargs: (file_target,),
    )
    assert deleted.ok
    assert file_target in deleted.deleted_paths
    assert not file_target.exists()


def test_delete_record_ignores_transient_missing_sidecar_and_reports_delete_error(
    tmp_path: Path,
) -> None:
    # Transient-missing entry (e.g. macOS AppleDouble like '._raw') should not fail delete.
    dir_target = tmp_path / "artifact_dir"
    dir_target.mkdir()

    def _fake_rmtree_ok(path: Path, onerror=None) -> None:  # noqa: ANN001
        assert onerror is not None
        try:
            raise FileNotFoundError(2, "No such file or directory", "._raw")
        except FileNotFoundError as exc:
            onerror(None, str(path / "._raw"), (FileNotFoundError, exc, None))
        path.rmdir()

    deleted = delete_record(
        project_root=tmp_path,
        subject="sub-001",
        record="runA",
        record_artifact_roots_fn=lambda *args, **kwargs: (dir_target,),
        rmtree_fn=_fake_rmtree_ok,
    )
    assert deleted.ok
    assert dir_target in deleted.deleted_paths
    assert not dir_target.exists()

    dir_target_2 = tmp_path / "artifact_dir_2"
    dir_target_2.mkdir()

    def _fake_rmtree_fail(path: Path, onerror=None) -> None:  # noqa: ANN001
        _ = (path, onerror)
        raise PermissionError("denied")

    failed = delete_record(
        project_root=tmp_path,
        subject="sub-001",
        record="runA",
        record_artifact_roots_fn=lambda *args, **kwargs: (dir_target_2,),
        rmtree_fn=_fake_rmtree_fail,
    )
    assert not failed.ok
    assert "Failed to delete record artifact" in failed.message
