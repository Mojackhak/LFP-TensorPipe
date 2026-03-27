"""Tests for PD feature table merging."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

for root in (REPO_ROOT, SRC_ROOT):
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

from lfptensorpipe.io.pkl_io import load_pkl, save_pkl
from paper.pd.merge.core import export_merge_tables, export_named_tables


def _write_feature_table(
    project_root: Path,
    *,
    subject: str,
    record: str,
    trial: str,
    relative_path: str,
    frame: pd.DataFrame,
) -> None:
    out_path = (
        project_root
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "features"
        / trial
        / relative_path
    )
    save_pkl(frame, out_path)


def _non_connectivity_row(
    *,
    subject: str,
    channel: str,
    value: object,
    snr_in: bool,
    stn_in: bool,
    band: object = "theta",
    phase: object = "strike_l",
) -> dict[str, object]:
    return {
        "Subject": subject,
        "subject": subject,
        "Channel": channel,
        "mni_x": -1.0,
        "mni_y": 2.0,
        "mni_z": 3.0,
        "SNr_in": snr_in,
        "STN_in": stn_in,
        "Band": band,
        "Phase": phase,
        "Value": value,
    }


def _unordered_connectivity_row(
    *,
    subject: str,
    value: object,
    snr_snr_in: bool,
    snr_stn_in: bool,
    stn_stn_in: bool,
) -> dict[str, object]:
    return {
        "Subject": subject,
        "subject": subject,
        "Channel": ("0_1", "1_2"),
        "mni_x": (-1.0, 1.0),
        "mni_y": (2.0, 3.0),
        "mni_z": (4.0, 5.0),
        "SNr-SNr_in": snr_snr_in,
        "SNr-STN_in": snr_stn_in,
        "STN-STN_in": stn_stn_in,
        "Band": "theta",
        "Phase": "strike_l",
        "Value": value,
    }


def _ordered_connectivity_row(
    *,
    subject: str,
    value: object,
    snr_snr_in: bool,
    snr_stn_in: bool,
    stn_snr_in: bool,
    stn_stn_in: bool,
) -> dict[str, object]:
    return {
        "Subject": subject,
        "subject": subject,
        "Channel": ("0_1", "1_2"),
        "mni_x": (-1.0, 1.0),
        "mni_y": (2.0, 3.0),
        "mni_z": (4.0, 5.0),
        "SNr-SNr_in": snr_snr_in,
        "SNr-STN_in": snr_stn_in,
        "STN-SNr_in": stn_snr_in,
        "STN-STN_in": stn_stn_in,
        "Band": "theta",
        "Phase": "strike_l",
        "Value": value,
    }


def test_export_named_tables_merges_across_record_trial_groups_and_simplifies_output(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="sit",
        trial="off",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=1.0,
                    snr_in=True,
                    stn_in=False,
                )
            ]
        ),
    )
    _write_feature_table(
        project_root,
        subject="sub-002",
        record="sit",
        trial="off",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-002",
                    channel="8_9",
                    value=2.0,
                    snr_in=False,
                    stn_in=True,
                )
            ]
        ),
    )
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="sit_medon",
        trial="on",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="1_2",
                    value=3.0,
                    snr_in=True,
                    stn_in=False,
                )
            ]
        ),
    )
    _write_feature_table(
        project_root,
        subject="sub-002",
        record="sit_medon",
        trial="on",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-002",
                    channel="9_10",
                    value=4.0,
                    snr_in=False,
                    stn_in=True,
                )
            ]
        ),
    )

    report = export_named_tables(
        project_root,
        {"med": {"sit": "off", "sit_medon": "on"}},
    )

    out_path = project_root / "summary" / "table" / "med" / "periodic" / "mean-scalar.pkl"
    xlsx_path = out_path.with_suffix(".xlsx")
    merged = load_pkl(out_path)
    merged_xlsx = pd.read_excel(xlsx_path)

    assert out_path in report.named_outputs
    assert xlsx_path.exists()
    assert list(merged.columns) == [
        "Subject",
        "Channel",
        "Side",
        "MNI_x",
        "MNI_y",
        "MNI_z",
        "Region",
        "Band",
        "Phase",
        "Lat",
        "Value",
    ]
    assert list(merged.index) == [0, 1, 2, 3]
    assert merged["Subject"].tolist() == ["Sub-001", "Sub-002", "Sub-001", "Sub-002"]
    assert merged["Side"].tolist() == ["L", "R", "L", "R"]
    assert merged["Region"].tolist() == ["SNr", "STN", "SNr", "STN"]
    assert merged["Band"].tolist() == ["Theta", "Theta", "Theta", "Theta"]
    assert merged["Phase"].tolist() == ["Off", "Off", "On", "On"]
    assert merged["Lat"].isna().all()
    assert merged["Value"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert merged_xlsx["Phase"].tolist() == ["Off", "Off", "On", "On"]
    assert merged_xlsx["Value"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_export_merge_tables_accepts_underscore_trial_alias(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="turn-stack",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=1.0,
                    snr_in=True,
                    stn_in=False,
                )
            ]
        ),
    )
    _write_feature_table(
        project_root,
        subject="sub-002",
        record="gait",
        trial="turn-stack",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-002",
                    channel="8_9",
                    value=2.0,
                    snr_in=False,
                    stn_in=True,
                )
            ]
        ),
    )

    report = export_merge_tables(
        project_root,
        merge_spec={"turn_stack": {"gait": "turn_stack"}},
    )

    named_out = project_root / "summary" / "table" / "turn_stack" / "periodic" / "mean-scalar.pkl"
    record_out = (
        project_root
        / "summary"
        / "table"
        / "gait"
        / "features"
        / "turn-stack"
        / "periodic"
        / "mean-scalar.pkl"
    )

    named_merged = load_pkl(named_out)

    assert named_out in report.named_outputs
    assert not record_out.exists()
    assert named_merged["Side"].tolist() == ["L", "R"]
    assert named_merged["Region"].tolist() == ["SNr", "STN"]
    assert named_merged["Lat"].isna().all()
    assert named_merged["Value"].tolist() == [1.0, 2.0]


def test_export_named_tables_rewrites_motor_phase_from_trial(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="stand",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=1.0,
                    snr_in=True,
                    stn_in=False,
                    phase="all",
                )
            ]
        ),
    )
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="walk",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=2.0,
                    snr_in=True,
                    stn_in=False,
                    phase="all",
                )
            ]
        ),
    )

    export_named_tables(
        project_root,
        {"motor": {"gait": ["stand", "walk"]}},
    )

    merged = load_pkl(
        project_root / "summary" / "table" / "motor" / "periodic" / "mean-scalar.pkl"
    )

    assert merged["Phase"].tolist() == ["Stand", "Walk"]
    assert merged["Lat"].isna().all()
    assert merged["Value"].tolist() == [1.0, 2.0]


def test_export_named_tables_filters_unordered_connectivity_rows(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="cycle-l",
        relative_path="coherence/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _unordered_connectivity_row(
                    subject="sub-001",
                    value=0.1,
                    snr_snr_in=False,
                    snr_stn_in=True,
                    stn_stn_in=False,
                ),
                _unordered_connectivity_row(
                    subject="sub-001",
                    value=0.2,
                    snr_snr_in=True,
                    snr_stn_in=True,
                    stn_stn_in=False,
                ),
                _unordered_connectivity_row(
                    subject="sub-001",
                    value=pd.NA,
                    snr_snr_in=False,
                    snr_stn_in=True,
                    stn_stn_in=False,
                ),
            ]
        ),
    )

    export_named_tables(
        project_root,
        {"cycle": {"gait": "cycle_l"}},
    )

    merged = load_pkl(
        project_root / "summary" / "table" / "cycle" / "coherence" / "mean-scalar.pkl"
    )

    assert len(merged) == 1
    assert merged.iloc[0]["Region"] == "SNr-STN"
    assert merged.iloc[0]["Side"] == "L"
    assert merged.iloc[0]["Channel"] == ("0_1", "1_2")
    assert merged.iloc[0]["MNI_x"] == (-1.0, 1.0)
    assert merged.iloc[0]["Value"] == 0.1


def test_export_named_tables_filters_ordered_connectivity_rows(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="cycle-l",
        relative_path="trgc/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _ordered_connectivity_row(
                    subject="sub-001",
                    value=0.4,
                    snr_snr_in=False,
                    snr_stn_in=True,
                    stn_snr_in=False,
                    stn_stn_in=False,
                ),
                _ordered_connectivity_row(
                    subject="sub-001",
                    value=0.5,
                    snr_snr_in=False,
                    snr_stn_in=True,
                    stn_snr_in=True,
                    stn_stn_in=False,
                ),
            ]
        ),
    )

    export_named_tables(
        project_root,
        {"cycle": {"gait": "cycle_l"}},
    )

    merged = load_pkl(
        project_root / "summary" / "table" / "cycle" / "trgc" / "mean-scalar.pkl"
    )

    assert len(merged) == 1
    assert merged.iloc[0]["Region"] == "SNr->STN"
    assert merged.iloc[0]["Side"] == "L"
    assert merged.iloc[0]["Channel"] == ("0_1", "1_2")
    assert merged.iloc[0]["Value"] == 0.4


def test_export_named_tables_keeps_cycle_trace_rows_without_phase_filter(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="cycle-l",
        relative_path="periodic/mean-trace.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=pd.Series([1.0, np.nan, 2.0]),
                    snr_in=True,
                    stn_in=False,
                    phase=pd.NA,
                ),
                _non_connectivity_row(
                    subject="sub-001",
                    channel="8_9",
                    value=pd.Series([3.0, 4.0, 5.0]),
                    snr_in=False,
                    stn_in=True,
                    phase="all",
                ),
            ]
        ),
    )

    export_named_tables(
        project_root,
        {"cycle": {"gait": "cycle_l"}},
    )

    merged = load_pkl(
        project_root / "summary" / "table" / "cycle" / "periodic" / "mean-trace.pkl"
    )

    assert len(merged) == 2
    assert merged["Side"].tolist() == ["L", "R"]
    assert pd.isna(merged.iloc[0]["Phase"])
    assert merged["Phase"].tolist()[1] == "All"
    assert merged["Lat"].tolist() == ["Ipsi", "Contra"]
    pd.testing.assert_series_equal(
        merged.iloc[0]["Value"],
        pd.Series([1.0, np.nan, 2.0]),
    )
    pd.testing.assert_series_equal(
        merged.iloc[1]["Value"],
        pd.Series([3.0, 4.0, 5.0]),
    )


def test_export_named_tables_assigns_cycle_na_raw_lat_from_side(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="cycle-l",
        relative_path="periodic/na-raw.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=pd.DataFrame({"a": [1.0, np.nan]}),
                    snr_in=True,
                    stn_in=False,
                    phase=pd.NA,
                ),
                _non_connectivity_row(
                    subject="sub-001",
                    channel="8_9",
                    value=pd.DataFrame({"a": [3.0, 4.0]}),
                    snr_in=False,
                    stn_in=True,
                    phase="all",
                ),
            ]
        ),
    )

    export_named_tables(
        project_root,
        {"cycle": {"gait": "cycle_l"}},
    )

    merged = load_pkl(
        project_root / "summary" / "table" / "cycle" / "periodic" / "na-raw.pkl"
    )

    assert len(merged) == 2
    assert merged["Side"].tolist() == ["L", "R"]
    assert pd.isna(merged.iloc[0]["Phase"])
    assert merged["Phase"].tolist()[1] == "All"
    assert merged["Lat"].tolist() == ["Ipsi", "Contra"]
    pd.testing.assert_frame_equal(
        merged.iloc[0]["Value"],
        pd.DataFrame({"a": [1.0, np.nan]}),
    )
    pd.testing.assert_frame_equal(
        merged.iloc[1]["Value"],
        pd.DataFrame({"a": [3.0, 4.0]}),
    )


def test_export_named_tables_keeps_turn_trace_rows_without_phase_filter(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="turn",
        relative_path="periodic/mean-trace.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=pd.Series([3.0, 4.0]),
                    snr_in=True,
                    stn_in=False,
                    phase="all",
                )
            ]
        ),
    )

    export_named_tables(
        project_root,
        {"turn": {"gait": "turn"}},
    )

    merged = load_pkl(
        project_root / "summary" / "table" / "turn" / "periodic" / "mean-trace.pkl"
    )

    assert len(merged) == 1
    assert merged.iloc[0]["Phase"] == "All"
    assert pd.isna(merged.iloc[0]["Lat"])
    pd.testing.assert_series_equal(
        merged.iloc[0]["Value"],
        pd.Series([3.0, 4.0]),
    )


def test_export_named_tables_splits_cycle_phase_and_filters_turn_phase(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="cycle-l",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=1.0,
                    snr_in=True,
                    stn_in=False,
                    phase="strike_r",
                ),
                _non_connectivity_row(
                    subject="sub-001",
                    channel="8_9",
                    value=2.0,
                    snr_in=False,
                    stn_in=True,
                    phase="off_r",
                ),
            ]
        ),
    )
    _write_feature_table(
        project_root,
        subject="sub-001",
        record="gait",
        trial="turn",
        relative_path="periodic/mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _non_connectivity_row(
                    subject="sub-001",
                    channel="0_1",
                    value=3.0,
                    snr_in=True,
                    stn_in=False,
                    phase="pre",
                ),
                _non_connectivity_row(
                    subject="sub-001",
                    channel="8_9",
                    value=4.0,
                    snr_in=False,
                    stn_in=True,
                    phase="all",
                ),
            ]
        ),
    )

    export_named_tables(
        project_root,
        {"cycle": {"gait": "cycle_l"}, "turn": {"gait": "turn"}},
    )

    cycle = load_pkl(project_root / "summary" / "table" / "cycle" / "periodic" / "mean-scalar.pkl")
    turn = load_pkl(project_root / "summary" / "table" / "turn" / "periodic" / "mean-scalar.pkl")

    assert cycle["Phase"].tolist() == ["Strike", "Off"]
    assert cycle["Side"].tolist() == ["L", "R"]
    assert cycle["Lat"].tolist() == ["Contra", "Ipsi"]
    assert turn["Phase"].tolist() == ["Pre"]
    assert turn["Side"].tolist() == ["L"]
    assert turn["Lat"].isna().all()
    assert turn["Value"].tolist() == [3.0]
