"""Aggregation helpers for PD paper preprocessing."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from lfptensorpipe.tabular.nested_value import (
    cell_is_empty_or_all_nan,
    coerce_cell_to_array,
    infer_nested_template,
    rebuild_cell_from_array,
)

AggMode = Literal["mean", "sum", "min", "max", "median", "std", "var"]


def summarize_df(
    df: pd.DataFrame,
    group_cols: str | list[str],
    value_col: str | list[str],
    mode: AggMode = "mean",
    *,
    drop_other_cols: bool = True,
    drop_empty: bool = True,
    align: Literal["strict", "reindex", "force"] = "strict",
) -> pd.DataFrame:
    """Summarize one or more value columns with nested-aware aggregation."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(value_col, str):
        value_cols = [value_col]
    else:
        value_cols = list(value_col)
    if not value_cols:
        raise ValueError("value_col must be a non-empty string or list of strings.")
    missing_cols = [column for column in value_cols if column not in df.columns]
    if missing_cols:
        raise KeyError(f"Column(s) not found: {missing_cols}")

    valid_modes: set[str] = {"mean", "sum", "min", "max", "median", "std", "var"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported aggregation mode: {mode}. Valid modes: {sorted(valid_modes)}")

    if df.empty:
        return df.copy()

    templates = {
        column: infer_nested_template(df[column], value_col=column, drop_empty=drop_empty)
        for column in value_cols
    }

    if all(template.kind == "scalar" for template in templates.values()):
        if drop_other_cols:
            out = (
                df.groupby(group_cols, observed=True, dropna=False)[value_cols]
                .agg(mode)
                .reset_index()
            )
            if group_cols:
                out = out.dropna(subset=group_cols)
            out = out.dropna(subset=value_cols, how="all")
            return out.reset_index(drop=True)

        extra_cols = [column for column in df.columns if column not in set(group_cols + value_cols)]
        agg_map: dict[str, Any] = {column: mode for column in value_cols}
        for column in extra_cols:
            agg_map[column] = "first"
        out = df.groupby(group_cols, observed=True, dropna=False).agg(agg_map).reset_index()
        return out.dropna(subset=value_cols, how="all").reset_index(drop=True)

    def _agg_stack(stack: np.ndarray, *, ddof: int) -> np.ndarray:
        if mode == "mean":
            return np.nanmean(stack, axis=0)
        if mode == "sum":
            return np.nansum(stack, axis=0)
        if mode == "min":
            return np.nanmin(stack, axis=0)
        if mode == "max":
            return np.nanmax(stack, axis=0)
        if mode == "median":
            return np.nanmedian(stack, axis=0)
        if mode == "std":
            return np.nanstd(stack, axis=0, ddof=ddof)
        if mode == "var":
            return np.nanvar(stack, axis=0, ddof=ddof)
        raise RuntimeError("Unexpected aggregation mode branch")

    records: list[dict[str, Any]] = []
    for group_key, group_df in df.groupby(group_cols, observed=True, dropna=False):
        if isinstance(group_key, tuple):
            row = dict(zip(group_cols, group_key, strict=False))
        else:
            row = {group_cols[0]: group_key}

        any_value = False
        for column in value_cols:
            template = templates[column]
            values = []
            for cell in group_df[column]:
                if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
                    continue
                values.append(coerce_cell_to_array(cell, template, align=align, drop_empty=drop_empty))

            if not values:
                aggregated = coerce_cell_to_array(np.nan, template, align=align, drop_empty=drop_empty)
                value = rebuild_cell_from_array(aggregated, template)
            else:
                stack = np.stack(values, axis=0)
                ddof = 1 if template.kind == "scalar" else 0
                aggregated = _agg_stack(stack, ddof=ddof)
                value = rebuild_cell_from_array(aggregated, template)
                any_value = True
            row[column] = value

        if not any_value:
            continue

        if not drop_other_cols:
            extra_cols = [column for column in df.columns if column not in set(group_cols + value_cols)]
            first_row = group_df.iloc[0]
            for column in extra_cols:
                row[column] = first_row[column]
        records.append(row)

    out = pd.DataFrame.from_records(records)
    if drop_other_cols:
        if group_cols:
            out = out.dropna(subset=group_cols)
        out = out.dropna(subset=value_cols, how="all")
        return out.reset_index(drop=True)
    return out.dropna(subset=value_cols, how="all").reset_index(drop=True)
