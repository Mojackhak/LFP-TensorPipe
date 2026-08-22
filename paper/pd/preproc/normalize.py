"""Scalar baseline normalization for maintained PD paper tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from lfptensorpipe.stats.preproc.normalize import baseline_normalize
from lfptensorpipe.tabular.nested_value import cell_is_empty_or_all_nan


def normalize_df(
    df: pd.DataFrame,
    group_cols: str | Sequence[str],
    baseline: Mapping[str, Any],
    value_col: str = "Value",
) -> pd.DataFrame:
    """Subtract each group's scalar baseline mean and drop baseline-less groups."""
    group_keys = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    work = df.reset_index(drop=True)
    normalized = np.full(len(work), np.nan, dtype=float)
    drop_mask = np.zeros(len(work), dtype=bool)

    for _, group in work.groupby(group_keys, observed=True, dropna=False):
        positions = group.index.to_numpy(dtype=int)
        baseline_mask = np.ones(len(group), dtype=bool)
        for key, value in baseline.items():
            baseline_mask &= group[key].to_numpy() == value

        scalar_values = pd.Series(
            [
                (
                    np.nan
                    if cell_is_empty_or_all_nan(cell)
                    else float(np.asarray(cell, dtype=float).item())
                )
                for cell in group[value_col]
            ],
            dtype=float,
        )
        usable_baseline = baseline_mask & scalar_values.notna().to_numpy()
        if not usable_baseline.any():
            drop_mask[positions] = True
            continue

        group_normalized = baseline_normalize(
            scalar_values,
            np.flatnonzero(usable_baseline),
            mode="mean",
        )
        normalized[positions] = group_normalized.to_numpy(dtype=float)

    keep_positions = np.flatnonzero(~drop_mask)
    out = df.iloc[keep_positions].copy()
    out[value_col] = normalized[keep_positions]
    return out
