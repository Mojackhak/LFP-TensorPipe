"""Export a cohort-level channel coordinate table for the PD paper workspace."""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    SRC_ROOT = REPO_ROOT / "src"
    for root in (REPO_ROOT, SRC_ROOT):
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

from lfptensorpipe.io.pkl_io import load_pkl
from paper.pd.paths import derivatives_root, summary_root

DEFAULT_INPUT_GLOB = "sub-*/sit/localize/channel_representative_coords.pkl"
DEFAULT_OUTPUT_FILE_NAME = "channel_coords.csv"

REQUIRED_COLUMNS = (
    "subject",
    "channel",
    "mni_x",
    "mni_y",
    "mni_z",
    "SNr_in",
    "STN_in",
)

OUTPUT_COLUMNS = (
    "Subject",
    "Channel",
    "Region",
    "MNI_x",
    "MNI_y",
    "MNI_z",
)

REGION_BY_FLAGS = {
    (True, True): "Mid",
    (True, False): "SNr",
    (False, True): "STN",
    (False, False): "EXT",
}


def _resolve_input_pattern(
    project_root: str | Path | None = None,
    input_glob: str | None = None,
) -> str:
    """Return the absolute glob pattern for localization tables."""
    if input_glob is None:
        return str(derivatives_root(project_root) / DEFAULT_INPUT_GLOB)

    input_path = Path(input_glob).expanduser()
    if input_path.is_absolute():
        return str(input_path)
    return str(derivatives_root(project_root) / input_glob)


def _default_output_path(project_root: str | Path | None = None) -> Path:
    """Return the default cohort-level CSV output path."""
    return summary_root(project_root, create=False) / "cohort" / DEFAULT_OUTPUT_FILE_NAME


def _resolve_input_paths(
    project_root: str | Path | None = None,
    input_glob: str | None = None,
) -> list[Path]:
    """Return all matching localization pickle paths."""
    pattern = _resolve_input_pattern(project_root=project_root, input_glob=input_glob)
    return [Path(path).expanduser().resolve() for path in sorted(glob(pattern))]


def _validate_frame(frame: pd.DataFrame, source_path: Path) -> None:
    """Raise when a source table is missing required columns."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame in {source_path}, got {type(frame).__name__}."
        )

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {source_path}: {missing_text}")


def _compute_region(frame: pd.DataFrame) -> pd.Series:
    """Return region labels derived from SNr/STN inclusion flags."""
    flags = zip(frame["SNr_in"].astype(bool), frame["STN_in"].astype(bool))
    return pd.Series(
        [REGION_BY_FLAGS[flag_pair] for flag_pair in flags],
        index=frame.index,
        dtype="string",
    )


def _sort_channel_coords_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort the export by subject and channel order."""
    channel_parts = frame["Channel"].astype(str).str.extract(r"^(?P<start>\d+)_(?P<end>\d+)$")
    sorted_frame = frame.assign(
        _channel_start=pd.to_numeric(channel_parts["start"], errors="coerce"),
        _channel_end=pd.to_numeric(channel_parts["end"], errors="coerce"),
    ).sort_values(
        by=["Subject", "_channel_start", "_channel_end", "Channel"],
        kind="mergesort",
        na_position="last",
    )
    return sorted_frame.drop(columns=["_channel_start", "_channel_end"]).reset_index(
        drop=True
    )


def build_channel_coords_table(
    project_root: str | Path | None = None,
    input_glob: str | None = None,
) -> pd.DataFrame:
    """Build the cohort-level channel coordinate table."""
    input_paths = _resolve_input_paths(project_root=project_root, input_glob=input_glob)
    if not input_paths:
        pattern = _resolve_input_pattern(project_root=project_root, input_glob=input_glob)
        raise FileNotFoundError(f"No localization tables found for pattern: {pattern}")

    frames: list[pd.DataFrame] = []
    for path in input_paths:
        frame = load_pkl(path)
        _validate_frame(frame, path)
        frames.append(
            frame.loc[:, REQUIRED_COLUMNS]
            .copy()
            .assign(Region=_compute_region(frame))
            .rename(
                columns={
                    "subject": "Subject",
                    "channel": "Channel",
                    "mni_x": "MNI_x",
                    "mni_y": "MNI_y",
                    "mni_z": "MNI_z",
                }
            )
            .loc[:, OUTPUT_COLUMNS]
        )

    return _sort_channel_coords_table(pd.concat(frames, ignore_index=True))


def export_channel_coords_table(
    output_path: str | Path | None = None,
    project_root: str | Path | None = None,
    input_glob: str | None = None,
) -> Path:
    """Export the channel coordinate table to CSV and return the output path."""
    frame = build_channel_coords_table(project_root=project_root, input_glob=input_glob)
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else _default_output_path(project_root).resolve()
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved_output, index=False, encoding="utf-8")
    return resolved_output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for direct module execution."""
    parser = argparse.ArgumentParser(
        description="Export the PD cohort-level channel coordinate CSV."
    )
    parser.add_argument(
        "--project-root",
        help="PD project root. Defaults to paper.pd.specs.DEFAULT_PROJECT_ROOT.",
    )
    parser.add_argument(
        "--input-glob",
        help="Absolute glob or derivatives-relative glob for localization pickle files.",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to {project}/summary/cohort/channel_coords.csv.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Export the cohort-level channel coordinate CSV."""
    args = parse_args(argv)
    output_path = export_channel_coords_table(
        output_path=args.output,
        project_root=args.project_root,
        input_glob=args.input_glob,
    )
    print(f"Exported channel coordinate table to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
