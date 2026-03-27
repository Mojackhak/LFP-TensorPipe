"""Tests for PD anatomy table exports."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

for root in (REPO_ROOT, SRC_ROOT):
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from lfptensorpipe.io.pkl_io import save_pkl
from paper.pd.anat.anat import build_channel_coords_table, export_channel_coords_table


def _write_localize_table(
    project_root: Path,
    *,
    subject: str,
    frame: pd.DataFrame,
) -> Path:
    out_path = (
        project_root
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / "sit"
        / "localize"
        / "channel_representative_coords.pkl"
    )
    save_pkl(frame, out_path)
    return out_path


def _row(
    *,
    subject: str,
    channel: str,
    mni_x: float,
    mni_y: float,
    mni_z: float,
    snr_in: bool,
    stn_in: bool,
) -> dict[str, object]:
    return {
        "subject": subject,
        "channel": channel,
        "mni_x": mni_x,
        "mni_y": mni_y,
        "mni_z": mni_z,
        "SNr_in": snr_in,
        "STN_in": stn_in,
    }


def test_build_channel_coords_table_merges_sources_and_sorts_rows(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_localize_table(
        project_root,
        subject="sub-002",
        frame=pd.DataFrame(
            [
                _row(
                    subject="sub-002",
                    channel="10_11",
                    mni_x=10.0,
                    mni_y=11.0,
                    mni_z=12.0,
                    snr_in=False,
                    stn_in=False,
                ),
                _row(
                    subject="sub-002",
                    channel="8_9",
                    mni_x=8.0,
                    mni_y=9.0,
                    mni_z=10.0,
                    snr_in=False,
                    stn_in=True,
                ),
            ]
        ),
    )
    _write_localize_table(
        project_root,
        subject="sub-001",
        frame=pd.DataFrame(
            [
                _row(
                    subject="sub-001",
                    channel="2_3",
                    mni_x=2.0,
                    mni_y=3.0,
                    mni_z=4.0,
                    snr_in=True,
                    stn_in=True,
                ),
                _row(
                    subject="sub-001",
                    channel="0_1",
                    mni_x=0.0,
                    mni_y=1.0,
                    mni_z=2.0,
                    snr_in=True,
                    stn_in=False,
                ),
            ]
        ),
    )

    result = build_channel_coords_table(project_root=project_root)

    assert list(result.columns) == [
        "Subject",
        "Channel",
        "Region",
        "MNI_x",
        "MNI_y",
        "MNI_z",
    ]
    assert result["Subject"].tolist() == ["sub-001", "sub-001", "sub-002", "sub-002"]
    assert result["Channel"].tolist() == ["0_1", "2_3", "8_9", "10_11"]
    assert result["Region"].tolist() == ["SNr", "Mid", "STN", "EXT"]


def test_export_channel_coords_table_writes_csv_without_index(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_localize_table(
        project_root,
        subject="sub-001",
        frame=pd.DataFrame(
            [
                _row(
                    subject="sub-001",
                    channel="1_2",
                    mni_x=1.0,
                    mni_y=2.0,
                    mni_z=3.0,
                    snr_in=True,
                    stn_in=False,
                )
            ]
        ),
    )

    output_path = export_channel_coords_table(project_root=project_root)
    exported = pd.read_csv(output_path)

    assert output_path == project_root / "summary" / "cohort" / "channel_coords.csv"
    assert output_path.exists()
    assert "Unnamed: 0" not in exported.columns
    assert exported.to_dict(orient="records") == [
        {
            "Subject": "sub-001",
            "Channel": "1_2",
            "Region": "SNr",
            "MNI_x": 1.0,
            "MNI_y": 2.0,
            "MNI_z": 3.0,
        }
    ]


def test_build_channel_coords_table_raises_when_no_files_match(tmp_path: Path) -> None:
    project_root = tmp_path / "project"

    with pytest.raises(FileNotFoundError, match="No localization tables found"):
        build_channel_coords_table(project_root=project_root)


def test_build_channel_coords_table_raises_on_missing_columns(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_localize_table(
        project_root,
        subject="sub-001",
        frame=pd.DataFrame(
            [
                {
                    "subject": "sub-001",
                    "channel": "0_1",
                    "mni_x": 0.0,
                    "mni_y": 1.0,
                    "mni_z": 2.0,
                    "SNr_in": True,
                }
            ]
        ),
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        build_channel_coords_table(project_root=project_root)
