"""Shared table export helpers for the PD paper workspace."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lfptensorpipe.app.features.table_io import _save_table_xlsx
from lfptensorpipe.io.pkl_io import save_pkl


def is_scalar_table_name(file_name: str) -> bool:
    """Return True when the file name denotes a scalar table payload."""
    return file_name.endswith("-scalar.pkl")


def xlsx_path_for_pickle(path: Path) -> Path:
    """Return the sibling xlsx path for a pickle output path."""
    return path.with_suffix(".xlsx")


def save_table_outputs(
    frame: pd.DataFrame,
    out_path: Path,
    *,
    export_xlsx: bool,
) -> Path | None:
    """Save one table as pickle, and optionally save a sibling xlsx file."""
    save_pkl(frame, out_path)
    if not export_xlsx:
        return None

    xlsx_path = xlsx_path_for_pickle(out_path)
    ok, message = _save_table_xlsx(frame, xlsx_path)
    if not ok:
        raise ValueError(f"Failed to export xlsx for {out_path.name}: {message}")
    return xlsx_path
