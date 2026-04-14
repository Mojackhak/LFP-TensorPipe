# -*- coding: utf-8 -*-
"""
lfpscope.stats.preproc.normalize

Group-wise baseline normalization for a nested value column in a summary table.

This module contains:
- normalize_df: group-wise baseline normalization using metadata conditions
- baseline_normalize: baseline normalization for a single Series/DataFrame object
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union, overload

import numpy as np
import pandas as pd

from ...tabular.nested_value import (
    cell_is_empty_or_all_nan,
    coerce_cell_to_array,
    infer_nested_template,
    rebuild_cell_from_array,
)
from ...utils.numeric import DEFAULT_REL_TOL, resolve_abs_tol, resolve_rel_tol, safe_divide

BaselineStat = Literal["mean", "median", "max", "min"]
NormMode = Literal["mean", "ratio", "percent", "zscore"]
SliceMode = Literal["absolute", "percent"]


def normalize_df(
    df: pd.DataFrame,
    group_cols: Union[str, List[str]],
    baseline: Mapping[str, Any],
    value_col: str = "Value",
    mode: NormMode = "mean",
    mode_baseline: BaselineStat = "mean",
    out_col: Optional[str] = None,
    abs_tol: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    on_missing_baseline: Literal["error", "drop"] = "error",
    drop_empty: bool = True,
    align: Literal["strict", "reindex", "force"] = "strict",
) -> pd.DataFrame:
    """
    Normalize `value_col` relative to a group-wise baseline defined by column conditions.

    Within each group, rows matching `baseline` define the baseline set. A representative
    baseline value is computed element-wise using `mode_baseline`, and normalization
    is then applied.

    Normalization modes:
      - "mean":    value - baseline_rep
      - "ratio":   value / baseline_rep when the denominator is finite and not near zero; else NaN
      - "percent": (value - baseline_rep) / baseline_rep when the denominator is finite and not near zero; else NaN
      - "zscore":  (value - baseline_rep) / baseline_sd when the denominator is finite and not near zero; else NaN
    """
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found.")

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    abs_tol_i = resolve_abs_tol(abs_tol)
    rel_tol_i = resolve_rel_tol(rel_tol)

    valid_modes: set[str] = {"mean", "ratio", "percent", "zscore"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported normalization mode: {mode}. Valid: {sorted(valid_modes)}")

    valid_baseline: set[str] = {"mean", "median", "max", "min"}
    if mode_baseline not in valid_baseline:
        raise ValueError(f"Unsupported mode_baseline: {mode_baseline}. Valid: {sorted(valid_baseline)}")

    template = infer_nested_template(df[value_col], value_col=value_col, drop_empty=drop_empty)

    def _aggregate(arr_stack: np.ndarray, how: BaselineStat) -> np.ndarray:
        if how == "mean":
            return np.nanmean(arr_stack, axis=0)
        if how == "median":
            return np.nanmedian(arr_stack, axis=0)
        if how == "max":
            return np.nanmax(arr_stack, axis=0)
        if how == "min":
            return np.nanmin(arr_stack, axis=0)
        raise RuntimeError("Unexpected mode_baseline branch")

    normalized: Dict[Any, Any] = {}
    drop_indices: List[Any] = []

    for gkey, gdf in df.groupby(group_cols, observed=True, dropna=False):
        mask = np.ones(len(gdf), dtype=bool)
        for k, v in baseline.items():
            if k not in gdf.columns:
                raise KeyError(f"Baseline key '{k}' not found in DataFrame columns for group {gkey}.")
            mask &= (gdf[k].to_numpy() == v)

        base_df = gdf.loc[mask]

        base_cells = [c for c in base_df[value_col] if not cell_is_empty_or_all_nan(c, drop_empty=drop_empty)]
        if len(base_cells) == 0:
            if on_missing_baseline == "drop":
                drop_indices.extend(gdf.index.tolist())
                continue
            raise ValueError(f"Baseline is missing for group {gkey} with baseline {dict(baseline)}.")

        base_stack = np.stack(
            [coerce_cell_to_array(c, template, align=align, drop_empty=drop_empty) for c in base_cells],
            axis=0,
        )

        base_rep = _aggregate(base_stack, mode_baseline)
        base_sd = np.nanstd(base_stack, axis=0, ddof=0)

        for idx, row in gdf.iterrows():
            cell = row[value_col]
            if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
                normalized[idx] = np.nan
                continue

            arr = coerce_cell_to_array(cell, template, align=align, drop_empty=drop_empty)

            if mode == "mean":
                nrm = arr - base_rep
            elif mode == "ratio":
                nrm = safe_divide(arr, base_rep, abs_tol=abs_tol_i, rel_tol=rel_tol_i)
            elif mode == "percent":
                nrm = safe_divide(arr - base_rep, base_rep, abs_tol=abs_tol_i, rel_tol=rel_tol_i)
            elif mode == "zscore":
                nrm = safe_divide(arr - base_rep, base_sd, abs_tol=abs_tol_i, rel_tol=rel_tol_i)
            else:
                raise RuntimeError("Unexpected mode branch")

            normalized[idx] = rebuild_cell_from_array(nrm, template)

    out = df.copy()
    if on_missing_baseline == "drop" and len(drop_indices) > 0:
        out = out.drop(index=drop_indices)

    dest = value_col if out_col is None else out_col
    out[dest] = out.index.map(lambda i: normalized.get(i, np.nan))
    return out


def _normalize_indices(
    baseline: Union[slice, Sequence[int], np.ndarray],
    n_samples: int,
) -> list[int]:
    """
    Normalize the baseline input into a list of positional indices (iloc-friendly).
    Supports:
        - slice
        - integer array-like
        - boolean mask
    """
    if isinstance(baseline, slice):
        idx = np.arange(n_samples)[baseline]
        if idx.size == 0:
            raise ValueError("baseline slice selects no elements")
        return idx.tolist()

    idx = np.asarray(baseline)
    if idx.ndim > 1:
        raise TypeError("baseline indices must be 1D (slice / 1D integer indices / 1D boolean mask)")
    if idx.ndim == 0:
        idx = idx.reshape(1)

    if idx.dtype == bool:
        if idx.size != n_samples:
            raise ValueError(f"Boolean baseline mask must have length {n_samples}, got {idx.size}")
        out = np.flatnonzero(idx)
        if out.size == 0:
            raise ValueError("baseline boolean mask selects no elements")
        return out.tolist()

    if idx.size == 0:
        raise ValueError("baseline cannot be empty")

    if not np.issubdtype(idx.dtype, np.integer):
        try:
            idx = idx.astype(int)
        except Exception as e:
            raise TypeError("baseline must be int indices / slice / boolean mask") from e

    mn, mx = int(idx.min()), int(idx.max())
    if mn < -n_samples or mx >= n_samples:
        raise IndexError(
            f"baseline index out of range: allowed [-{n_samples}, {n_samples-1}], got [{mn}, {mx}]"
        )
    return idx.tolist()


def _normalize_percent_ranges(
    baseline: Union[Sequence[Sequence[float]], Sequence[float], np.ndarray],
    n_samples: int,
) -> list[int]:
    """
    Convert percent baseline ranges into a list of positional indices (iloc-friendly).

    Parameters
    ----------
    baseline:
        Percent ranges along the sample axis, expressed on a 0-100 scale.

        Supported forms:
            - sequence of [start_percent, stop_percent] pairs, e.g. [[0, 5], [8, 10]]
            - a single [start_percent, stop_percent] pair, e.g. [0, 5]

        Ranges are interpreted as half-open intervals [start, stop), similar to Python slicing.
        For example, [0, 5] selects the first 5% of samples.

    n_samples:
        Total number of samples along the sample axis.

    Notes
    -----
    Index mapping uses:
        start_idx = floor(start_percent / 100 * n_samples)
        stop_idx  = ceil(stop_percent  / 100 * n_samples)

    This tends to behave well for small n_samples where an exact fraction would otherwise
    round to an empty selection.

    Returns
    -------
    list[int]
        Unique, sorted positional indices.

    Raises
    ------
    ValueError
        If ranges are invalid or select no samples.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")

    arr = np.asarray(baseline, dtype=float)

    # Accept a single pair like [0, 5] as well as a list of pairs like [[0, 5], [8, 10]].
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(
                "percent baseline must be a (start, stop) pair or a sequence of (start, stop) pairs"
            )
        arr = arr.reshape(1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError("percent baseline must have shape (n_ranges, 2)")
        if arr.shape[0] == 0:
            raise ValueError("baseline cannot be empty")
    else:
        raise ValueError("percent baseline must be 1D or 2D array-like")

    parts: List[np.ndarray] = []
    for start_pct, stop_pct in arr:
        if not (np.isfinite(start_pct) and np.isfinite(stop_pct)):
            raise ValueError("percent baseline contains non-finite values (NaN/inf)")
        if start_pct < 0 or stop_pct > 100:
            raise ValueError(f"percent baseline must be within [0, 100], got [{start_pct}, {stop_pct}]")
        if stop_pct <= start_pct:
            raise ValueError(f"percent baseline requires stop > start, got [{start_pct}, {stop_pct}]")

        start_i = int(np.floor(start_pct / 100.0 * n_samples))
        stop_i = int(np.ceil(stop_pct / 100.0 * n_samples))

        # Clamp just in case floating point makes us drift outside.
        start_i = max(0, min(n_samples, start_i))
        stop_i = max(0, min(n_samples, stop_i))

        if stop_i <= start_i:
            raise ValueError(
                f"percent baseline range [{start_pct}, {stop_pct}] selects no elements for n_samples={n_samples}"
            )

        parts.append(np.arange(start_i, stop_i, dtype=int))

    out = np.unique(np.concatenate(parts))
    if out.size == 0:
        raise ValueError("percent baseline selects no elements")
    return out.tolist()



def _reduce(obj: Union[pd.Series, pd.DataFrame], how: BaselineStat, axis=None):
    """Compute baseline representative statistic."""
    if how == "mean":
        return obj.mean(axis=axis)
    if how == "median":
        return obj.median(axis=axis)
    if how == "max":
        return obj.max(axis=axis)
    if how == "min":
        return obj.min(axis=axis)
    raise ValueError(f"Unknown mode_baseline={how!r}")


def _safe_divide_frame_rows(
    numer: np.ndarray,
    denom_by_row: np.ndarray,
    *,
    abs_tol: float,
    rel_tol: float,
) -> np.ndarray:
    """Apply `safe_divide` row-wise so each row gets its own local tolerance."""

    out = np.full_like(numer, np.nan, dtype=float)
    for row_idx in range(numer.shape[0]):
        out[row_idx, :] = safe_divide(
            numer[row_idx, :],
            np.full((numer.shape[1],), denom_by_row[row_idx], dtype=float),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    return out


@overload
def baseline_normalize(
    x: pd.Series,
    baseline: Union[slice, Sequence[int], np.ndarray],
    mode_baseline: BaselineStat = "mean",
    mode: NormMode = "mean",
    abs_tol: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    slice_mode: Literal["absolute"] = "absolute",
) -> pd.Series: ...


@overload
def baseline_normalize(
    x: pd.Series,
    baseline: Union[Sequence[Sequence[float]], Sequence[float], np.ndarray],
    mode_baseline: BaselineStat = "mean",
    mode: NormMode = "mean",
    abs_tol: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    slice_mode: Literal["percent"] = "percent",
) -> pd.Series: ...


@overload
def baseline_normalize(
    x: pd.DataFrame,
    baseline: Union[slice, Sequence[int], np.ndarray],
    mode_baseline: BaselineStat = "mean",
    mode: NormMode = "mean",
    abs_tol: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    slice_mode: Literal["absolute"] = "absolute",
) -> pd.DataFrame: ...


@overload
def baseline_normalize(
    x: pd.DataFrame,
    baseline: Union[Sequence[Sequence[float]], Sequence[float], np.ndarray],
    mode_baseline: BaselineStat = "mean",
    mode: NormMode = "mean",
    abs_tol: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    slice_mode: Literal["percent"] = "percent",
) -> pd.DataFrame: ...


def baseline_normalize(
    x: Union[pd.Series, pd.DataFrame],
    baseline: Union[slice, Sequence[int], Sequence[Sequence[float]], Sequence[float], np.ndarray],
    mode_baseline: BaselineStat = "mean",
    mode: NormMode = "mean",
    abs_tol: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    slice_mode: SliceMode = "absolute",
) -> Union[pd.Series, pd.DataFrame]:
    """
    Baseline-normalize a pd.Series or pd.DataFrame.

    Inputs
    ------
    x:
        - pd.Series (n_samples)
        - pd.DataFrame (n_freqs, n_samples) where the sample axis is columns.

    baseline:
        Baseline selection along the sample axis.

        If slice_mode == "absolute":
            Positional indices (iloc-style) along the sample axis.
            Can be slice, integer indices, or boolean mask.

        If slice_mode == "percent":
            Percent ranges along the sample axis, expressed on a 0-100 scale.
            Pass a sequence of [start_percent, stop_percent] pairs, e.g. [[0, 5], [8, 10]].

            Ranges are interpreted as half-open intervals [start, stop), similar to Python slicing.

    slice_mode:
        How to interpret `baseline`:
            - "absolute": baseline is absolute iloc positions (original behavior)
            - "percent": baseline is specified in percent ranges (0-100) and converted to iloc positions

    mode_baseline:
        How to compute the baseline representative value:
            "mean" | "median" | "max" | "min"

    mode:
        Normalization method:
            mean:    arr - base_rep
            ratio:   arr / base_rep when the denominator is finite and not near zero; else NaN
            percent: (arr - base_rep) / base_rep when the denominator is finite and not near zero; else NaN
            zscore:  (arr - base_rep) / base_sd when the denominator is finite and not near zero; else NaN
    """
    abs_tol_i = resolve_abs_tol(abs_tol)
    rel_tol_i = resolve_rel_tol(rel_tol)

    if slice_mode not in {"absolute", "percent"}:
        raise ValueError(f"slice_mode must be 'absolute' or 'percent', got {slice_mode!r}")

    def _get_baseline_idx(n_samples: int) -> list[int]:
        if slice_mode == "absolute":
            return _normalize_indices(baseline, n_samples)
        return _normalize_percent_ranges(baseline, n_samples)

    if isinstance(x, pd.Series):
        idx = _get_baseline_idx(len(x))
        base = x.iloc[idx]
        base_rep = _reduce(base, mode_baseline)
        x_vals = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)

        if mode == "mean":
            return x - base_rep
        if mode == "ratio":
            out = safe_divide(
                x_vals,
                np.full_like(x_vals, float(base_rep), dtype=float),
                abs_tol=abs_tol_i,
                rel_tol=rel_tol_i,
            )
            return pd.Series(out, index=x.index, name=x.name)
        if mode == "percent":
            out = safe_divide(
                x_vals - float(base_rep),
                np.full_like(x_vals, float(base_rep), dtype=float),
                abs_tol=abs_tol_i,
                rel_tol=rel_tol_i,
            )
            return pd.Series(out, index=x.index, name=x.name)
        if mode == "zscore":
            base_sd = base.std(ddof=0)
            out = safe_divide(
                x_vals - float(base_rep),
                np.full_like(x_vals, float(base_sd), dtype=float),
                abs_tol=abs_tol_i,
                rel_tol=rel_tol_i,
            )
            return pd.Series(out, index=x.index, name=x.name)

        raise ValueError(f"Unknown mode={mode!r}")

    if isinstance(x, pd.DataFrame):
        idx = _get_baseline_idx(x.shape[1])
        base = x.iloc[:, idx]
        base_rep = _reduce(base, mode_baseline, axis=1)
        x_vals = x.to_numpy(dtype=float)
        base_rep_vals = pd.to_numeric(base_rep, errors="coerce").to_numpy(dtype=float)

        if mode == "mean":
            return x.sub(base_rep, axis=0)
        if mode == "ratio":
            out = _safe_divide_frame_rows(
                x_vals,
                base_rep_vals,
                abs_tol=abs_tol_i,
                rel_tol=rel_tol_i,
            )
            return pd.DataFrame(out, index=x.index, columns=x.columns)
        if mode == "percent":
            out = _safe_divide_frame_rows(
                x_vals - base_rep_vals[:, None],
                base_rep_vals,
                abs_tol=abs_tol_i,
                rel_tol=rel_tol_i,
            )
            return pd.DataFrame(out, index=x.index, columns=x.columns)
        if mode == "zscore":
            base_sd = base.std(axis=1, ddof=0)
            base_sd_vals = pd.to_numeric(base_sd, errors="coerce").to_numpy(dtype=float)
            out = _safe_divide_frame_rows(
                x_vals - base_rep_vals[:, None],
                base_sd_vals,
                abs_tol=abs_tol_i,
                rel_tol=rel_tol_i,
            )
            return pd.DataFrame(out, index=x.index, columns=x.columns)

        raise ValueError(f"Unknown mode={mode!r}")

    raise TypeError("x must be a pd.Series or pd.DataFrame")


def normalize_df_by_baseline(
    df: pd.DataFrame,
    baseline: Union[slice, Sequence[int], Sequence[Sequence[float]], Sequence[float], np.ndarray],
    value_col: str = "Value",
    mode_baseline: BaselineStat = "mean",
    mode: NormMode = "mean",
    abs_tol: float | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    slice_mode: SliceMode = "absolute",   
) -> pd.DataFrame:
    """
    Normalize `value_col`  using baseline_normalize.
    """
    out = df.copy()

    def norm_func(x: Any) -> pd.Series | pd.DataFrame:
        return baseline_normalize(
            x,
            baseline=baseline,
            mode_baseline=mode_baseline,
            mode=mode,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            slice_mode=slice_mode,
        )

    out[value_col] = out[value_col].map(norm_func)
    return out
