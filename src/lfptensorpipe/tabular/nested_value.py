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

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

NestedKind = Literal["scalar", "series", "dataframe"]


@dataclass(frozen=True)
class NestedTemplate:
    """Template describing the nested value structure to preserve in outputs."""

    kind: NestedKind
    index: pd.Index | None = None
    columns: pd.Index | None = None


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


def flatten_to_numeric_1d(arr_like: Any) -> np.ndarray:
    """
    Convert scalar/Series/DataFrame/ndarray to a 1D numeric array.

    Non-numeric entries are coerced to NaN.
    """
    if isinstance(arr_like, pd.Series):
        arr = arr_like.to_numpy()
    elif isinstance(arr_like, pd.DataFrame):
        arr = arr_like.to_numpy().ravel(order="C")
    elif isinstance(arr_like, np.ndarray):
        arr = arr_like.ravel(order="C")
    else:
        arr = np.array([arr_like], dtype=object)

    return pd.to_numeric(pd.Series(arr, copy=False), errors="coerce").to_numpy()


def cell_has_any_nonfinite(
    cell: Any,
    *,
    drop_empty: bool = True,
    mode: Literal["any", "all"] = "any",
) -> bool:
    """
    Strict missing definition (used by drop_missing_value_rows):

    - scalar: missing if NA or non-finite
    - Series/DataFrame: missing if empty (optional) OR meets `mode` condition:
        - mode="any": any element is NA/non-finite
        - mode="all": all elements are NA/non-finite
    """
    if mode not in ("any", "all"):
        raise ValueError("mode must be 'any' or 'all'")

    def _evaluate(flat: np.ndarray) -> bool:
        if flat.size == 0:
            return drop_empty
        nonfinite_mask = ~np.isfinite(flat)
        if mode == "any":
            return bool(nonfinite_mask.any())
        return bool(nonfinite_mask.all())

    if isinstance(cell, pd.Series):
        if drop_empty and cell.size == 0:
            return True
        flat = flatten_to_numeric_1d(cell)
        return _evaluate(flat)
    if isinstance(cell, pd.DataFrame):
        if drop_empty and (cell.shape[0] == 0 or cell.shape[1] == 0):
            return True
        flat = flatten_to_numeric_1d(cell)
        return _evaluate(flat)

    flat = flatten_to_numeric_1d(cell)
    return _evaluate(flat)


def infer_nested_template(
    values: pd.Series,
    *,
    value_col: str,
    drop_empty: bool = True,
) -> NestedTemplate:
    """
    Infer the nested-value template from the first non-missing element in `values`.

    If no non-missing element exists, returns a scalar template.
    """
    first = None
    for v in values:
        if cell_is_empty_or_all_nan(v, drop_empty=drop_empty):
            continue
        first = v
        break

    if first is None:
        return NestedTemplate(kind="scalar")

    if isinstance(first, pd.Series):
        return NestedTemplate(kind="series", index=first.index)
    if isinstance(first, pd.DataFrame):
        return NestedTemplate(
            kind="dataframe", index=first.index, columns=first.columns
        )

    return NestedTemplate(kind="scalar")


def coerce_cell_to_array(
    cell: Any,
    template: NestedTemplate,
    *,
    align: Literal["strict", "reindex", "force"] = "strict",
    drop_empty: bool = True,
) -> np.ndarray:
    """
    Convert a nested cell to a numpy array according to `template`.

    Parameters
    ----------
    align:
        - "strict": require identical labels (Series index / DataFrame index+columns)
        - "reindex": align to template labels (missing labels -> NaN)
        - "force": force labels to match template without any reordering/alignment.
            Requires equal length/shape; only relabels index/columns.

    Returns
    -------
    np.ndarray
        - scalar -> 0-d array (np.asarray(scalar, dtype=float))
        - Series -> 1-d array
        - DataFrame -> 2-d array
    """
    if template.kind == "scalar":
        if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
            return np.asarray(np.nan, dtype=float)
        return np.asarray(cell, dtype=float)

    if template.kind == "series":
        if template.index is None:
            raise ValueError("Series template requires `index`.")
        if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
            return np.full((len(template.index),), np.nan, dtype=float)
        if not isinstance(cell, pd.Series):
            raise TypeError(f"Expected pandas.Series, got {type(cell)}")

        if align == "strict":
            if not cell.index.equals(template.index):
                raise ValueError("Series index mismatch under align='strict'.")
            return cell.to_numpy(dtype=float)

        if align == "force":
            if len(cell) != len(template.index):
                raise ValueError(
                    "Series length mismatch under align='force': "
                    f"len(cell)={len(cell)} vs len(template.index)={len(template.index)}"
                )
            s = cell.copy()
            s.index = template.index
            return s.to_numpy(dtype=float)

        if align == "reindex":
            s = cell.reindex(template.index)
            return s.to_numpy(dtype=float)

        raise ValueError(f"Unsupported align={align!r} for template.kind='series'.")

    if template.kind == "dataframe":
        if template.index is None or template.columns is None:
            raise ValueError("DataFrame template requires `index` and `columns`.")
        if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
            return np.full(
                (len(template.index), len(template.columns)), np.nan, dtype=float
            )
        if not isinstance(cell, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(cell)}")

        if align == "strict":
            if not cell.index.equals(template.index) or not cell.columns.equals(
                template.columns
            ):
                raise ValueError(
                    "DataFrame index/columns mismatch under align='strict'."
                )
            return cell.to_numpy(dtype=float)

        if align == "force":
            expected_shape = (len(template.index), len(template.columns))
            if cell.shape != expected_shape:
                raise ValueError(
                    "DataFrame shape mismatch under align='force': "
                    f"cell.shape={cell.shape} vs expected={expected_shape}"
                )
            m = cell.copy()
            m.index = template.index
            m.columns = template.columns
            return m.to_numpy(dtype=float)

        if align == "reindex":
            m = cell.reindex(index=template.index, columns=template.columns)
            return m.to_numpy(dtype=float)

        raise ValueError(f"Unsupported align={align!r} for template.kind='dataframe'.")

    raise ValueError(f"Unsupported template.kind={template.kind!r}")


def rebuild_cell_from_array(arr: np.ndarray, template: NestedTemplate) -> Any:
    """Rebuild a nested cell from a numpy array according to `template`."""
    if template.kind == "scalar":
        try:
            return float(np.asarray(arr).item())
        except Exception:
            return np.asarray(arr).item()

    if template.kind == "series":
        if template.index is None:
            raise ValueError("Series template requires `index`.")
        return pd.Series(np.asarray(arr, dtype=float), index=template.index)

    if template.kind == "dataframe":
        if template.index is None or template.columns is None:
            raise ValueError("DataFrame template requires `index` and `columns`.")
        return pd.DataFrame(
            np.asarray(arr, dtype=float), index=template.index, columns=template.columns
        )

    raise ValueError(f"Unsupported template.kind={template.kind!r}")
