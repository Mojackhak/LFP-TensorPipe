# -*- coding: utf-8 -*-
"""
Element-wise transforms for a nested value column in a summary table.

This module provides `transform_df`, which applies a transform (log, dB, logit,
Fisher-z, ...) to each nested cell in `value_col` while preserving the original
cell structure (scalar / Series / DataFrame).

Important note:
This transform does NOT align Series/DataFrame labels across rows. Each cell is
transformed independently, preserving its own index/columns.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ...tabular.nested_value import cell_is_empty_or_all_nan
from ...utils.numeric import (
    DEFAULT_REL_TOL,
    resolve_abs_tol,
    resolve_rel_tol,
    safe_divide,
)
from ...utils.transforms import TransformMode, apply_transform_array

NestedTransformMode = TransformMode | Literal["log10", "zscore"]


def transform_df(
    df: pd.DataFrame,
    *,
    value_col: str = "Value",
    mode: NestedTransformMode | None = None,
    out_col: Optional[str] = None,
    abs_tol: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """
    Apply an element-wise transform to `value_col` (scalar/Series/DataFrame).

    This function does NOT align labels (Series index / DataFrame index+columns)
    across rows. Each nested cell is transformed independently and its own
    index/columns are preserved.

    Transform modes:
      - "dB": 10*log10(x) for finite x > 0; otherwise NaN
      - "log10": log10(x) for finite x > 0; otherwise NaN
      - "log": log(x) for finite x > 0; otherwise NaN
      - "fisherz": atanh(x) for finite -1 < x < 1; otherwise NaN
      - "fisherz_sqrt": atanh(sqrt(C)) for finite 0 <= C < 1; otherwise NaN
      - "logit": log(x/(1-x)) for finite 0 < x < 1; otherwise NaN
      - "asinh": log(x + sqrt(x^2 + 1))
      - "zscore": z-score each Series over its full length, each DataFrame row
        over its columns, and leave scalars unchanged. Near-zero spreads return NaN.
      - "none": identity
    """
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found.")

    abs_tol_i = resolve_abs_tol(abs_tol)
    rel_tol_i = resolve_rel_tol(rel_tol)

    if mode is None:
        return df.copy()

    def coerce_series_to_float_array(s: pd.Series) -> np.ndarray:
        # Coerce non-numeric entries to NaN, preserving length and order.
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

    def coerce_df_to_float_array(m: pd.DataFrame) -> np.ndarray:
        # Flatten -> to_numeric -> reshape, keeping original shape.
        flat_obj = m.to_numpy().ravel(order="C")
        flat_num = pd.to_numeric(
            pd.Series(flat_obj, copy=False), errors="coerce"
        ).to_numpy(dtype=float)
        return flat_num.reshape(m.shape)

    def transform_array(arr: np.ndarray) -> np.ndarray:
        if mode == "log10":
            x = np.asarray(arr, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            valid = np.isfinite(x) & (x > 0.0)
            if np.any(valid):
                out[valid] = np.log10(x[valid])
            return out
        return apply_transform_array(arr, mode=mode)

    def zscore_series(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=float)
        out = np.full_like(x, np.nan, dtype=float)
        valid = np.isfinite(x)
        if not np.any(valid):
            return out
        center = float(np.nanmean(x))
        spread = float(np.nanstd(x))
        out[:] = safe_divide(
            x - center,
            np.full_like(x, spread, dtype=float),
            abs_tol=abs_tol_i,
            rel_tol=rel_tol_i,
        )
        return out

    def zscore_rows(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"Row-wise zscore expects 2D input, got shape {x.shape}.")
        means = np.nanmean(x, axis=1, keepdims=True)
        spreads = np.nanstd(x, axis=1, keepdims=True)
        out = np.full_like(x, np.nan, dtype=float)
        for row_idx in range(x.shape[0]):
            out[row_idx, :] = safe_divide(
                x[row_idx, :] - means[row_idx, 0],
                np.full((x.shape[1],), spreads[row_idx, 0], dtype=float),
                abs_tol=abs_tol_i,
                rel_tol=rel_tol_i,
            )
        out[~np.isfinite(x)] = np.nan
        return out

    def transform_cell(cell: Any) -> Any:
        if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
            return np.nan

        if isinstance(cell, pd.Series):
            arr = coerce_series_to_float_array(cell)
            out_arr = zscore_series(arr) if mode == "zscore" else transform_array(arr)
            # Preserve the original labels (index) and name.
            return pd.Series(out_arr, index=cell.index, name=cell.name)

        if isinstance(cell, pd.DataFrame):
            arr = coerce_df_to_float_array(cell)
            out_arr = zscore_rows(arr) if mode == "zscore" else transform_array(arr)
            # Preserve the original labels (index/columns).
            return pd.DataFrame(out_arr, index=cell.index, columns=cell.columns)

        # Scalar path (float/int/numpy scalar, etc.)
        arr0 = np.asarray(cell)
        if arr0.ndim != 0 and arr0.size != 1:
            raise TypeError(
                "Unsupported nested cell value. Expected scalar/pandas.Series/pandas.DataFrame; "
                f"got {type(cell)} with shape {getattr(arr0, 'shape', None)}."
            )
        if mode == "zscore":
            return cell
        arr = np.asarray(arr0, dtype=float).reshape(())
        out_arr = transform_array(arr)
        try:
            return float(np.asarray(out_arr).item())
        except Exception:
            return np.asarray(out_arr).item()

    out = df.copy()
    dest = value_col if out_col is None else out_col

    # IMPORTANT:
    # Do NOT use Series.apply here, because if the first transformed cell is a Series,
    # pandas may try to "expand" outputs into a DataFrame. We need an object column.
    transformed = [transform_cell(c) for c in out[value_col].tolist()]
    out[dest] = transformed
    return out
