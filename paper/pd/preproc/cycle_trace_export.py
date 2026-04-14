"""Export shared smoothed cycle traces for downstream cycle interval analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from lfptensorpipe.io.pkl_io import load_pkl
from paper.pd.paths import resolve_project_root, summary_table_root

TRACE_FILE_NAME = "mean-trace_summary_trans_normalized_shift.pkl"
PREPROCESSED_DIR_NAME = "preprocessed"
RAW_CYCLE_POINTS = 500
DOWNSAMPLED_CYCLE_POINTS = 100
BLOCK_SIZE = 5
LOWESS_FRAC = 0.05
CYCLE_BIN_CENTERS = np.arange(0.5, 100.0, 1.0, dtype=float)


@dataclass(frozen=True)
class CycleTraceExportReport:
    """Report paths written by the shared cycle preprocessing step."""

    long_csv: Path
    parameters_csv: Path
    trace_sources: tuple[Path, ...]
    n_long_rows: int
    n_parameters: int


def preprocessed_output_root(
    project_root: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    create: bool = False,
) -> Path:
    """Return the shared cycle preprocessing output directory."""
    if output_root is not None:
        root = Path(output_root).expanduser().resolve()
    else:
        root = summary_table_root(project_root, create=create) / "cycle" / PREPROCESSED_DIR_NAME
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def discover_shift_trace_paths(project_root: str | Path | None = None) -> tuple[Path, ...]:
    """Return every shifted cycle trace input."""
    cycle_root = summary_table_root(project_root) / "cycle"
    return tuple(sorted(cycle_root.glob(f"*/{TRACE_FILE_NAME}")))


def export_cycle_trace_inputs(
    project_root: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
) -> CycleTraceExportReport:
    """Flatten shifted cycle traces into shared CSV inputs for cycle analysis."""
    project = resolve_project_root(project_root)
    trace_paths = discover_shift_trace_paths(project)
    if not trace_paths:
        raise FileNotFoundError(
            f"No shifted cycle trace tables found under {summary_table_root(project) / 'cycle'}"
        )

    long_frame = build_cycle_trace_long_frame(trace_paths)
    params_frame = build_cycle_trace_parameters_frame(long_frame)

    out_root = preprocessed_output_root(project, output_root=output_root, create=True)
    long_csv = out_root / "cycle_trace_long.csv"
    params_csv = out_root / "cycle_trace_parameters.csv"

    _write_table(long_frame, csv_path=long_csv)
    _write_table(params_frame, csv_path=params_csv)

    return CycleTraceExportReport(
        long_csv=long_csv,
        parameters_csv=params_csv,
        trace_sources=trace_paths,
        n_long_rows=int(len(long_frame)),
        n_parameters=int(len(params_frame)),
    )


def build_cycle_trace_long_frame(trace_paths: list[Path] | tuple[Path, ...]) -> pd.DataFrame:
    """Return one long frame spanning every shifted cycle trace path."""
    pieces = [_flatten_trace_path(path) for path in trace_paths]
    if not pieces:
        return pd.DataFrame(
            columns=[
                "Subject",
                "Channel",
                "Metric",
                "Band",
                "Region",
                "t_pct",
                "Value",
            ]
        )
    long_frame = pd.concat(pieces, ignore_index=True)
    return long_frame.sort_values(
        by=["Metric", "Band", "Region", "Subject", "Channel", "t_pct"],
        kind="stable",
        ignore_index=True,
    )


def build_cycle_trace_parameters_frame(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize one row per Metric/Band/Region combination."""
    if long_frame.empty:
        return pd.DataFrame(
            columns=[
                "Metric",
                "Band",
                "Region",
                "n_subjects",
                "n_channels",
                "n_rows",
                "n_timepoints",
                "n_trace_rows",
            ]
        )

    params = (
        long_frame.groupby(["Metric", "Band", "Region"], dropna=False)
        .agg(
            n_subjects=("Subject", "nunique"),
            n_channels=("Channel", "nunique"),
            n_rows=("Value", "size"),
            n_timepoints=("t_pct", "nunique"),
        )
        .reset_index()
    )
    params["n_trace_rows"] = params["n_rows"] / params["n_timepoints"]
    return params


def _flatten_trace_path(path: Path) -> pd.DataFrame:
    """Flatten one shifted trace pickle into long rows."""
    metric = path.parent.name
    frame = load_pkl(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(frame).__name__}")

    pieces: list[pd.DataFrame] = []
    for row in frame.to_dict(orient="records"):
        pieces.append(_flatten_trace_row(row=row, metric=metric, source_path=path))

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _flatten_trace_row(*, row: dict[str, Any], metric: str, source_path: Path) -> pd.DataFrame:
    """Flatten one trace cell into timepoint-level rows."""
    lat = row.get("Lat", pd.NA)
    if pd.notna(lat) and str(lat) != "Contra":
        raise ValueError(f"Expected Contra-only shifted rows in {source_path}, got Lat={lat!r}")

    series = row.get("Value")
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"Expected pd.Series in Value column for {source_path}, got {type(series).__name__}"
        )

    processed_series = preprocess_cycle_trace_series(series, source_path=source_path)
    t_pct = np.asarray(processed_series.index, dtype=float)

    return pd.DataFrame(
        {
            "Subject": [_required_text(row, "Subject", source_path)] * len(processed_series),
            "Channel": [_stringify_value(row.get("Channel"))] * len(processed_series),
            "Metric": [metric] * len(processed_series),
            "Band": [_required_text(row, "Band", source_path)] * len(processed_series),
            "Region": [_required_text(row, "Region", source_path)] * len(processed_series),
            "t_pct": t_pct,
            "Value": pd.to_numeric(processed_series.to_numpy(), errors="coerce"),
        }
    )


def preprocess_cycle_trace_series(series: pd.Series, *, source_path: Path | None = None) -> pd.Series:
    """Return one circularly smoothed and block-mean downsampled cycle trace."""
    values = pd.to_numeric(series.to_numpy(), errors="coerce")
    if len(values) != RAW_CYCLE_POINTS:
        source_label = f" in {source_path}" if source_path is not None else ""
        raise ValueError(
            f"Expected {RAW_CYCLE_POINTS} cycle points{source_label}, got {len(values)}"
        )
    values = fill_cycle_trace_nan_gaps(values, source_path=source_path)

    smoothed = circular_lowess_smooth(values, frac=LOWESS_FRAC)
    downsampled = block_mean_downsample(smoothed, block_size=BLOCK_SIZE)
    return pd.Series(downsampled, index=CYCLE_BIN_CENTERS, dtype=float)


def fill_cycle_trace_nan_gaps(
    values: np.ndarray, *, source_path: Path | None = None
) -> np.ndarray:
    """Fill missing cycle samples with circular linear interpolation."""
    values = np.asarray(values, dtype=float)
    if not np.isnan(values).any():
        return values

    finite_mask = np.isfinite(values)
    n_valid = int(finite_mask.sum())
    source_label = f" in {source_path}" if source_path is not None else ""
    if n_valid == 0:
        raise ValueError(f"Cycle trace contains only NaN values{source_label}")
    if n_valid == 1:
        fill_value = float(values[finite_mask][0])
        return np.full_like(values, fill_value, dtype=float)

    original_x = np.arange(len(values), dtype=float)
    valid_x = original_x[finite_mask]
    valid_y = values[finite_mask]
    wrapped_x = np.concatenate([valid_x - len(values), valid_x, valid_x + len(values)])
    wrapped_y = np.concatenate([valid_y, valid_y, valid_y])
    filled = np.interp(original_x, wrapped_x, wrapped_y)
    return np.asarray(filled, dtype=float)


def circular_lowess_smooth(values: np.ndarray, *, frac: float = LOWESS_FRAC) -> np.ndarray:
    """Apply LOWESS on a circularly padded copy of one cycle trace."""
    values = np.asarray(values, dtype=float)
    n_points = len(values)
    wrapped = np.concatenate([values, values, values])
    x = np.arange(len(wrapped), dtype=float)
    smoothed = lowess(wrapped, x, frac=frac, return_sorted=False)
    return np.asarray(smoothed[n_points : 2 * n_points], dtype=float)


def block_mean_downsample(values: np.ndarray, *, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Return block means for one fixed-length cycle trace."""
    values = np.asarray(values, dtype=float)
    if len(values) % block_size != 0:
        raise ValueError(
            f"Cycle trace length {len(values)} is not divisible by block size {block_size}"
        )
    return values.reshape(-1, block_size).mean(axis=1)


def _required_text(row: dict[str, Any], key: str, source_path: Path) -> str:
    """Return one required string-valued column."""
    value = row.get(key, pd.NA)
    if pd.isna(value):
        raise ValueError(f"Missing required column {key!r} in {source_path}")
    return str(value)


def _stringify_value(value: Any) -> str:
    """Serialize object values such as channel tuples for CSV export."""
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return str(value)


def _write_table(frame: pd.DataFrame, *, csv_path: Path) -> None:
    """Write one frame to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export shifted cycle traces to shared preprocessed CSV tables."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="PD project root. Defaults to paper.pd.specs.DEFAULT_PROJECT_ROOT.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional shared preprocessing output directory. "
            "Defaults to {project}/summary/table/cycle/preprocessed."
        ),
    )
    return parser


def main() -> None:
    """Run the shared cycle preprocessing export CLI."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    export_cycle_trace_inputs(
        project_root=args.project_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
