# -*- coding: utf-8 -*-
"""

Core utilities for handling a "nested value column" in a pandas DataFrame.

In this project, summary tables often store numeric measurements in a single column
(e.g., `Value`), where each cell can be:
- a scalar number (float/int)
- a pandas.Series (e.g., PSD indexed by frequency)
- a pandas.DataFrame (e.g., time-frequency maps)

This module centralizes:
- missing-value definitions for nested cells (strict vs cell-level)
- a reusable template describing the expected nested structure
- conversion helpers: nested cell <-> numpy array, with optional alignment

All higher-level preprocessing (transform/normalize/aggregate/outlier filtering)
should call these functions instead of re-implementing nested handling.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

NestedKind = Literal["scalar", "series", "dataframe"]


def is_scalar_na(x: Any) -> bool:
    """Return True if x should be treated as scalar NA (None/NaN)."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def cell_is_empty_or_all_nan(cell: Any, *, drop_empty: bool = True) -> bool:
    """
    Cell-level missing definition (used by transform/normalize/explode):

    - scalar: pd.isna(cell)
    - Series: empty (optional) OR all elements NA
    - DataFrame: empty (optional) OR all elements NA
    """
    if isinstance(cell, pd.Series):
        if drop_empty and cell.size == 0:
            return True
        return bool(cell.isna().all())
    if isinstance(cell, pd.DataFrame):
        if drop_empty and (cell.shape[0] == 0 or cell.shape[1] == 0):
            return True
        return bool(cell.isna().to_numpy().all())
    return is_scalar_na(cell)
