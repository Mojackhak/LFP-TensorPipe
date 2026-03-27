"""Tests for PD paper table preprocessing."""

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
from paper.pd import specs as pd_specs
from paper.pd.preproc.core import export_preprocessed_tables
from paper.pd.specs import (
    DEFAULT_NORMALIZE_SPEC,
    DEFAULT_SCALAR_NORMALIZE_SPEC,
    DEFAULT_TRANSFORM_MODE_CFG,
)


def _write_summary_table(
    project_root: Path,
    *,
    name: str,
    metric: str,
    file_name: str,
    frame: pd.DataFrame,
) -> Path:
    out_path = project_root / "summary" / "table" / name / metric / file_name
    save_pkl(frame, out_path)
    return out_path


def _row(
    *,
    subject: str = "Sub-001",
    channel: object = "0_1",
    side: object = "L",
    phase: object = "Off",
    lat: object = "Ipsi",
    band: object = "Theta",
    region: object = "SNr",
    value: object,
) -> dict[str, object]:
    return {
        "Subject": subject,
        "Channel": channel,
        "Side": side,
        "MNI_x": -1.0,
        "MNI_y": 2.0,
        "MNI_z": 3.0,
        "Region": region,
        "Band": band,
        "Phase": phase,
        "Lat": lat,
        "Value": value,
    }


def test_default_transform_mode_cfg_uses_literal_coherence_key() -> None:
    assert "coherence" in DEFAULT_TRANSFORM_MODE_CFG
    assert "coh" not in DEFAULT_TRANSFORM_MODE_CFG
    assert not hasattr(pd_specs, "TRANSFORM_METRIC_ALIASES")


def test_export_preprocessed_tables_creates_summary_and_cycle_trace_normalization(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    trace_frame = pd.DataFrame(
        [
            _row(value=pd.Series([1.0, 2.0, 3.0, 4.0])),
            _row(value=pd.Series([2.0, 3.0, 4.0, 5.0])),
        ]
    )

    _write_summary_table(
        project_root,
        name="cycle",
        metric="periodic",
        file_name="mean-trace.pkl",
        frame=trace_frame,
    )

    report = export_preprocessed_tables(
        project_root,
        transform_mode_cfg=DEFAULT_TRANSFORM_MODE_CFG,
        normalize_spec=DEFAULT_NORMALIZE_SPEC,
        scalar_normalize_spec=DEFAULT_SCALAR_NORMALIZE_SPEC,
    )

    summary_path = (
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-trace_summary.pkl"
    )
    trans_path = (
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-trace_summary_trans.pkl"
    )
    norm_path = (
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-trace_summary_trans_normalized.pkl"
    )

    summarized = load_pkl(summary_path)
    transformed = load_pkl(trans_path)
    normalized = load_pkl(norm_path)

    pd.testing.assert_series_equal(
        summarized.iloc[0]["Value"],
        pd.Series([1.5, 2.5, 3.5, 4.5]),
    )
    pd.testing.assert_series_equal(
        transformed.iloc[0]["Value"],
        pd.Series([1.5, 2.5, 3.5, 4.5]),
    )
    pd.testing.assert_series_equal(
        normalized.iloc[0]["Value"],
        pd.Series([-1.5, -0.5, 0.5, 1.5]),
    )
    assert summary_path in report.summarized_outputs
    assert trans_path in report.transformed_outputs
    assert norm_path in report.normalized_outputs


def test_export_preprocessed_tables_uses_path_based_transform_rules_and_turn_stack_passthrough(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"

    _write_summary_table(
        project_root,
        name="turn_stack",
        metric="burst",
        file_name="rate-scalar.pkl",
        frame=pd.DataFrame(
            [
                _row(value=1.0, phase="All", lat=pd.NA),
                _row(value=2.0, phase="All", lat=pd.NA),
            ]
        ),
    )
    _write_summary_table(
        project_root,
        name="turn",
        metric="coherence",
        file_name="mean-trace.pkl",
        frame=pd.DataFrame(
            [
                _row(
                    value=pd.Series([0.25, 0.36, 0.49, 0.64]),
                    phase="Pre",
                    lat=pd.NA,
                )
            ]
        ),
    )

    report = export_preprocessed_tables(
        project_root,
        transform_mode_cfg=DEFAULT_TRANSFORM_MODE_CFG,
        normalize_spec=DEFAULT_NORMALIZE_SPEC,
        scalar_normalize_spec=DEFAULT_SCALAR_NORMALIZE_SPEC,
    )

    burst_trans = load_pkl(
        project_root
        / "summary"
        / "table"
        / "turn_stack"
        / "burst"
        / "rate-scalar_summary_trans.pkl"
    )
    burst_norm = load_pkl(
        project_root
        / "summary"
        / "table"
        / "turn_stack"
        / "burst"
        / "rate-scalar_summary_trans_normalized.pkl"
    )
    coherence_trans = load_pkl(
        project_root
        / "summary"
        / "table"
        / "turn"
        / "coherence"
        / "mean-trace_summary_trans.pkl"
    )
    coherence_norm = load_pkl(
        project_root
        / "summary"
        / "table"
        / "turn"
        / "coherence"
        / "mean-trace_summary_trans_normalized.pkl"
    )

    expected_burst = np.arcsinh(np.array([1.5]))
    np.testing.assert_allclose(burst_trans["Value"].to_numpy(), expected_burst)
    np.testing.assert_allclose(burst_norm["Value"].to_numpy(), expected_burst)

    transformed_values = np.arctanh(np.sqrt(np.array([0.25, 0.36, 0.49, 0.64])))
    pd.testing.assert_series_equal(
        coherence_trans.iloc[0]["Value"],
        pd.Series(transformed_values),
    )
    pd.testing.assert_series_equal(
        coherence_norm.iloc[0]["Value"],
        pd.Series(transformed_values - transformed_values[0]),
    )
    assert burst_norm.index.tolist() == [0]
    assert (
        project_root
        / "summary"
        / "table"
        / "turn_stack"
        / "burst"
        / "rate-scalar_summary_trans_normalized.pkl"
    ) in report.passthrough_outputs


def test_export_preprocessed_tables_shifts_cycle_trace_like_ipsi_rows(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"

    _write_summary_table(
        project_root,
        name="cycle",
        metric="periodic",
        file_name="mean-trace.pkl",
        frame=pd.DataFrame(
            [
                _row(
                    channel="0_1",
                    side="L",
                    lat="Ipsi",
                    value=pd.Series([1.0, 2.0, 3.0, 4.0]),
                ),
                _row(
                    channel="8_9",
                    side="R",
                    lat="Contra",
                    value=pd.Series([10.0, 20.0, 30.0, 40.0]),
                ),
            ]
        ),
    )
    _write_summary_table(
        project_root,
        name="cycle",
        metric="periodic",
        file_name="na-raw.pkl",
        frame=pd.DataFrame(
            [
                _row(
                    channel="0_1",
                    side="L",
                    lat="Ipsi",
                    value=pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]),
                ),
                _row(
                    channel="8_9",
                    side="R",
                    lat="Contra",
                    value=pd.DataFrame([[10.0, 20.0, 30.0, 40.0], [40.0, 30.0, 20.0, 10.0]]),
                ),
            ]
        ),
    )

    report = export_preprocessed_tables(
        project_root,
        transform_mode_cfg=DEFAULT_TRANSFORM_MODE_CFG,
        normalize_spec=DEFAULT_NORMALIZE_SPEC,
        scalar_normalize_spec=DEFAULT_SCALAR_NORMALIZE_SPEC,
    )

    trace_norm = load_pkl(
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-trace_summary_trans_normalized.pkl"
    )
    trace_shift = load_pkl(
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-trace_summary_trans_normalized_shift.pkl"
    )
    raw_norm = load_pkl(
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "na-raw_summary_trans_normalized.pkl"
    )
    raw_shift = load_pkl(
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "na-raw_summary_trans_normalized_shift.pkl"
    )

    expected_trace_ipsi = pd.Series(
        np.roll(trace_norm.iloc[0]["Value"].to_numpy(copy=True), -(len(trace_norm.iloc[0]["Value"]) // 2)),
        index=trace_norm.iloc[0]["Value"].index,
    )
    expected_raw_ipsi = pd.DataFrame(
        np.roll(
            raw_norm.iloc[0]["Value"].to_numpy(copy=True),
            -(raw_norm.iloc[0]["Value"].shape[1] // 2),
            axis=1,
        ),
        index=raw_norm.iloc[0]["Value"].index,
        columns=raw_norm.iloc[0]["Value"].columns,
    )

    assert trace_shift["Lat"].tolist() == ["Contra", "Contra"]
    assert raw_shift["Lat"].tolist() == ["Contra", "Contra"]
    pd.testing.assert_series_equal(trace_shift.iloc[0]["Value"], expected_trace_ipsi)
    pd.testing.assert_series_equal(trace_shift.iloc[1]["Value"], trace_norm.iloc[1]["Value"])
    pd.testing.assert_frame_equal(raw_shift.iloc[0]["Value"], expected_raw_ipsi)
    pd.testing.assert_frame_equal(raw_shift.iloc[1]["Value"], raw_norm.iloc[1]["Value"])
    assert (
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-trace_summary_trans_normalized_shift.pkl"
    ) in report.shifted_outputs
    assert (
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "na-raw_summary_trans_normalized_shift.pkl"
    ) in report.shifted_outputs


def test_export_preprocessed_tables_uses_scalar_normalize_df_for_selected_names(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"

    _write_summary_table(
        project_root,
        name="cycle",
        metric="periodic",
        file_name="mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _row(value=1.0, phase="Off", lat="Ipsi"),
                _row(value=3.0, phase="Strike", lat="Ipsi"),
            ]
        ),
    )
    _write_summary_table(
        project_root,
        name="turn",
        metric="periodic",
        file_name="mean-spectral.pkl",
        frame=pd.DataFrame(
            [
                _row(value=pd.Series([1.0, 2.0]), phase="Pre", lat=pd.NA),
                _row(value=pd.Series([4.0, 6.0]), phase="Onset", lat=pd.NA),
            ]
        ),
    )
    _write_summary_table(
        project_root,
        name="med",
        metric="periodic",
        file_name="mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _row(value=2.0, phase="Off", lat=pd.NA),
                _row(value=5.0, phase="On", lat=pd.NA),
            ]
        ),
    )
    _write_summary_table(
        project_root,
        name="motor",
        metric="periodic",
        file_name="mean-scalar.pkl",
        frame=pd.DataFrame(
            [
                _row(value=10.0, phase="Stand", lat=pd.NA),
                _row(value=13.0, phase="Walk", lat=pd.NA),
            ]
        ),
    )

    export_preprocessed_tables(
        project_root,
        transform_mode_cfg=DEFAULT_TRANSFORM_MODE_CFG,
        normalize_spec=DEFAULT_NORMALIZE_SPEC,
        scalar_normalize_spec=DEFAULT_SCALAR_NORMALIZE_SPEC,
    )

    cycle_norm = load_pkl(
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-scalar_summary_trans_normalized.pkl"
    )
    turn_norm = load_pkl(
        project_root
        / "summary"
        / "table"
        / "turn"
        / "periodic"
        / "mean-spectral_summary_trans_normalized.pkl"
    )
    med_norm = load_pkl(
        project_root
        / "summary"
        / "table"
        / "med"
        / "periodic"
        / "mean-scalar_summary_trans_normalized.pkl"
    )
    motor_norm = load_pkl(
        project_root
        / "summary"
        / "table"
        / "motor"
        / "periodic"
        / "mean-scalar_summary_trans_normalized.pkl"
    )

    cycle_summary_xlsx = (
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-scalar_summary.xlsx"
    )
    cycle_trans_xlsx = (
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-scalar_summary_trans.xlsx"
    )
    cycle_norm_xlsx = (
        project_root
        / "summary"
        / "table"
        / "cycle"
        / "periodic"
        / "mean-scalar_summary_trans_normalized.xlsx"
    )

    cycle_values = dict(zip(cycle_norm["Phase"], cycle_norm["Value"], strict=False))
    turn_values = dict(zip(turn_norm["Phase"], turn_norm["Value"], strict=False))
    med_values = dict(zip(med_norm["Phase"], med_norm["Value"], strict=False))
    motor_values = dict(zip(motor_norm["Phase"], motor_norm["Value"], strict=False))

    assert cycle_summary_xlsx.exists()
    assert cycle_trans_xlsx.exists()
    assert cycle_norm_xlsx.exists()
    assert cycle_values == {"Off": 0.0, "Strike": 2.0}
    pd.testing.assert_series_equal(turn_values["Pre"], pd.Series([0.0, 0.0]))
    pd.testing.assert_series_equal(turn_values["Onset"], pd.Series([3.0, 4.0]))
    assert med_values == {"Off": 0.0, "On": 3.0}
    assert motor_values == {"Stand": 0.0, "Walk": 3.0}

    cycle_norm_xlsx_df = pd.read_excel(cycle_norm_xlsx)
    assert cycle_norm_xlsx_df["Phase"].tolist() == ["Off", "Strike"]
    assert cycle_norm_xlsx_df["Value"].tolist() == [0.0, 2.0]
