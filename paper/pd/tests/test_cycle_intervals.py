"""Tests for cycle zero-baseline interval workflows."""

from __future__ import annotations

import shutil
import subprocess
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


def _subject_offsets(subject_index: int) -> float:
    offsets = np.linspace(-0.05, 0.05, 6)
    return float(offsets[subject_index])


def _trace_value(metric: str, band: str, region: str, t_pct: float, subject_index: int) -> float:
    offset = _subject_offsets(subject_index)
    value = offset

    if metric == "periodic" and band == "Theta" and region == "SNr" and 20.0 <= t_pct < 36.0:
        value += 1.2
    if metric == "periodic" and band == "Theta" and region == "STN" and 60.0 <= t_pct < 73.0:
        value -= 1.2
    if metric == "wpli" and band == "Alpha" and region == "SNr-STN" and 75.0 <= t_pct < 89.0:
        value += 1.0
    if metric == "coherence" and band == "Beta_low" and region == "SNr-STN" and (
        t_pct >= 97.0 or t_pct < 3.0
    ):
        value += 1.0
    if metric == "plv" and band == "Alpha" and region == "SNr-STN" and 40.0 <= t_pct < 43.0:
        value += 1.0
    if metric == "plv" and band == "Beta_low" and region == "SNr-STN" and 10.0 <= t_pct < 18.0:
        value += 1.0
    if metric == "psi" and band == "Beta_high" and region == "SNr->STN":
        if t_pct >= 94.0:
            value += 1.1
        elif t_pct < 6.0:
            value -= 1.1
    if metric == "aperiodic" and band == "Error_mae" and region == "SNr" and 10.0 <= t_pct < 20.0:
        value += 2.0
    if metric == "aperiodic" and band == "Offset" and region == "STN" and 30.0 <= t_pct < 40.0:
        value += 2.0

    return value


def _build_long_frame() -> pd.DataFrame:
    subjects = [f"Sub-{idx:03d}" for idx in range(1, 7)]
    timepoints = np.arange(0.5, 100.0, 1.0)
    channel_offsets = {"ch-a": -0.015, "ch-b": 0.015}
    rows: list[dict[str, object]] = []

    trace_specs = [
        ("periodic", "Theta", "SNr"),
        ("periodic", "Theta", "STN"),
        ("wpli", "Alpha", "SNr-STN"),
        ("coherence", "Beta_low", "SNr-STN"),
        ("plv", "Alpha", "SNr-STN"),
        ("plv", "Beta_low", "SNr-STN"),
        ("psi", "Beta_high", "SNr->STN"),
        ("aperiodic", "Error_mae", "SNr"),
        ("aperiodic", "Offset", "STN"),
    ]

    for subject_index, subject in enumerate(subjects):
        for channel, channel_offset in channel_offsets.items():
            for metric, band, region in trace_specs:
                for t_pct in timepoints:
                    rows.append(
                        {
                            "Subject": subject,
                            "Channel": f"{channel}-{metric}-{band}-{region}",
                            "Metric": metric,
                            "Band": band,
                            "Region": region,
                            "t_pct": t_pct,
                            "Value": _trace_value(metric, band, region, t_pct, subject_index)
                            + channel_offset,
                        }
                    )

    return pd.DataFrame(rows)


def _run_rscript(script: Path, *args: str) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required for cycle interval tests")

    completed = subprocess.run(
        [rscript, str(script), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"R script failed: {script}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def test_cycle_interval_workflow_detects_signed_and_circular_runs(tmp_path: Path) -> None:
    preprocessed_dir = tmp_path / "preprocessed"
    interval_dir = tmp_path / "interval"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    interval_dir.mkdir(parents=True, exist_ok=True)

    long_csv = preprocessed_dir / "cycle_trace_long.csv"
    _build_long_frame().to_csv(long_csv, index=False)

    run_intervals = REPO_ROOT / "paper" / "pd" / "stats" / "run_cycle_interval_fit.R"
    run_postprocess = REPO_ROOT / "paper" / "pd" / "stats" / "run_cycle_interval_postprocess.R"

    _run_rscript(
        run_intervals,
        "--input-csv",
        str(long_csv),
        "--output-dir",
        str(interval_dir),
        "--jobs",
        "2",
    )
    _run_rscript(
        run_postprocess,
        "--input-csv",
        str(interval_dir / "cycle_timepoint_deviation.csv"),
        "--output-dir",
        str(interval_dir),
    )

    deviation = pd.read_csv(interval_dir / "cycle_timepoint_deviation.csv")
    lmer_intervals_path = interval_dir / "cycle_timepoint_lmer_intervals.csv"
    lmer_candidates_path = interval_dir / "cycle_timepoint_lmer_interval_candidates.csv"
    rlmer_intervals_path = interval_dir / "cycle_timepoint_rlmer_intervals.csv"
    rlmer_candidates_path = interval_dir / "cycle_timepoint_rlmer_interval_candidates.csv"
    joint_intervals_path = interval_dir / "cycle_timepoint_joint_intervals.csv"
    joint_candidates_path = interval_dir / "cycle_timepoint_joint_interval_candidates.csv"

    for csv_path in (
        lmer_intervals_path,
        lmer_candidates_path,
        rlmer_intervals_path,
        rlmer_candidates_path,
        joint_intervals_path,
        joint_candidates_path,
    ):
        assert csv_path.exists(), csv_path

    assert not (interval_dir / "cycle_timepoint_intervals.csv").exists()
    assert not (interval_dir / "cycle_timepoint_interval_candidates.csv").exists()

    lmer_intervals = pd.read_csv(lmer_intervals_path)
    lmer_candidates = pd.read_csv(lmer_candidates_path)
    rlmer_intervals = pd.read_csv(rlmer_intervals_path)
    rlmer_candidates = pd.read_csv(rlmer_candidates_path)
    joint_intervals = pd.read_csv(joint_intervals_path)
    joint_candidates = pd.read_csv(joint_candidates_path)

    periodic_rows = deviation[(deviation["Metric"] == "periodic") & (deviation["Band"] == "Theta")]
    assert sorted(periodic_rows["Region"].unique().tolist()) == ["SNr", "STN"]
    assert set(lmer_intervals["interval_basis"].unique()) == {"lmer"}
    assert set(rlmer_intervals["interval_basis"].unique()) == {"rlmer"}
    assert set(joint_intervals["interval_basis"].unique()) == {"joint"}

    snr_interval = lmer_intervals[
        (lmer_intervals["Metric"] == "periodic")
        & (lmer_intervals["Band"] == "Theta")
        & (lmer_intervals["Region"] == "SNr")
        & (lmer_intervals["direction"] == "above_0")
    ]
    assert len(snr_interval) == 1
    assert np.isclose(snr_interval["start_pct"].iloc[0], 20.0)
    assert np.isclose(snr_interval["end_pct"].iloc[0], 36.0)

    stn_interval = lmer_intervals[
        (lmer_intervals["Metric"] == "periodic")
        & (lmer_intervals["Band"] == "Theta")
        & (lmer_intervals["Region"] == "STN")
        & (lmer_intervals["direction"] == "below_0")
    ]
    assert len(stn_interval) == 1
    assert np.isclose(stn_interval["start_pct"].iloc[0], 60.0)
    assert np.isclose(stn_interval["end_pct"].iloc[0], 73.0)

    synchrony_interval = lmer_intervals[
        (lmer_intervals["Metric"] == "wpli")
        & (lmer_intervals["Band"] == "Alpha")
        & (lmer_intervals["Region"] == "SNr-STN")
        & (lmer_intervals["direction"] == "above_0")
    ]
    assert len(synchrony_interval) == 1
    assert np.isclose(synchrony_interval["start_pct"].iloc[0], 75.0)
    assert np.isclose(synchrony_interval["end_pct"].iloc[0], 89.0)

    for frame in (lmer_intervals, rlmer_intervals, joint_intervals):
        wrap_interval = frame[
            (frame["Metric"] == "coherence")
            & (frame["Band"] == "Beta_low")
            & (frame["Region"] == "SNr-STN")
            & (frame["direction"] == "above_0")
        ]
        assert len(wrap_interval) == 1
        assert bool(wrap_interval["wraps_cycle"].iloc[0]) is True
        assert np.isclose(wrap_interval["start_pct"].iloc[0], 97.0)
        assert np.isclose(wrap_interval["end_pct"].iloc[0], 3.0)
        assert np.isclose(wrap_interval["span_pct"].iloc[0], 6.0)

    short_run = lmer_intervals[
        (lmer_intervals["Metric"] == "plv")
        & (lmer_intervals["Band"] == "Alpha")
        & (lmer_intervals["Region"] == "SNr-STN")
    ]
    assert short_run.empty

    opposite_sign = lmer_intervals[
        (lmer_intervals["Metric"] == "psi")
        & (lmer_intervals["Band"] == "Beta_high")
        & (lmer_intervals["Region"] == "SNr->STN")
    ].sort_values(["direction", "start_pct"])
    assert set(opposite_sign["direction"].tolist()) == {"above_0", "below_0"}
    assert not opposite_sign["wraps_cycle"].any()

    assert not lmer_candidates.empty
    assert {"candidate_rank", "interval_basis", "span_pct", "robust_fraction"}.issubset(
        lmer_candidates.columns
    )
    assert set(lmer_candidates["interval_basis"].unique()) == {"lmer"}
    assert set(rlmer_candidates["interval_basis"].unique()) == {"rlmer"}
    assert set(joint_candidates["interval_basis"].unique()) == {"joint"}
    assert not rlmer_candidates.empty
    assert not joint_candidates.empty

    assert "plv" not in set(lmer_candidates["Metric"])
    assert "burst" not in set(lmer_candidates["Metric"])
    assert "Error_mae" not in set(
        lmer_candidates.loc[lmer_candidates["Metric"] == "aperiodic", "Band"]
    )
    assert "periodic" in set(lmer_candidates["Metric"])
    assert "wpli" in set(lmer_candidates["Metric"])
    assert "coherence" in set(lmer_candidates["Metric"])
    assert "psi" in set(lmer_candidates["Metric"])
    assert set(
        lmer_candidates.loc[lmer_candidates["Metric"] == "aperiodic", "Band"]
    ).issubset({"Exponent", "Offset"})
    assert np.allclose(joint_intervals["robust_fraction"], 1.0)
