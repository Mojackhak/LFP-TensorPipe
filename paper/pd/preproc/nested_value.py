"""Nested-value shape preservation for maintained PD paper aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from lfptensorpipe.tabular.nested_value import cell_is_empty_or_all_nan

NestedKind = Literal["scalar", "series", "dataframe"]


@dataclass(frozen=True)
class NestedTemplate:
    """Describe the scalar or labeled nested structure to preserve."""

    kind: NestedKind
    index: pd.Index | None = None
    columns: pd.Index | None = None


def infer_nested_template(
    values: pd.Series,
    *,
    value_col: str,
    drop_empty: bool = True,
) -> NestedTemplate:
    """Infer a paper value column's structure from its first usable cell."""
    first = None
    for value in values:
        if cell_is_empty_or_all_nan(value, drop_empty=drop_empty):
            continue
        first = value
        break

    if first is None:
        return NestedTemplate(kind="scalar")
    if isinstance(first, pd.Series):
        return NestedTemplate(kind="series", index=first.index)
    if isinstance(first, pd.DataFrame):
        return NestedTemplate(
            kind="dataframe",
            index=first.index,
            columns=first.columns,
        )
    return NestedTemplate(kind="scalar")


def coerce_cell_to_array(
    cell: Any,
    template: NestedTemplate,
    *,
    align: Literal["strict", "reindex", "force"] = "strict",
    drop_empty: bool = True,
) -> np.ndarray:
    """Convert one paper value cell to the array described by ``template``."""
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
            series = cell.copy()
            series.index = template.index
            return series.to_numpy(dtype=float)
        if align == "reindex":
            return cell.reindex(template.index).to_numpy(dtype=float)
        raise ValueError(f"Unsupported align={align!r} for template.kind='series'.")

    if template.kind == "dataframe":
        if template.index is None or template.columns is None:
            raise ValueError("DataFrame template requires `index` and `columns`.")
        if cell_is_empty_or_all_nan(cell, drop_empty=drop_empty):
            return np.full(
                (len(template.index), len(template.columns)),
                np.nan,
                dtype=float,
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
            frame = cell.copy()
            frame.index = template.index
            frame.columns = template.columns
            return frame.to_numpy(dtype=float)
        if align == "reindex":
            return cell.reindex(
                index=template.index,
                columns=template.columns,
            ).to_numpy(dtype=float)
        raise ValueError(f"Unsupported align={align!r} for template.kind='dataframe'.")

    raise ValueError(f"Unsupported template.kind={template.kind!r}")


def rebuild_cell_from_array(array: np.ndarray, template: NestedTemplate) -> Any:
    """Rebuild one normalized or aggregated cell with its original labels."""
    if template.kind == "scalar":
        return np.asarray(array).item()
    if template.kind == "series":
        if template.index is None:
            raise ValueError("Series template requires `index`.")
        return pd.Series(np.asarray(array, dtype=float), index=template.index)
    if template.kind == "dataframe":
        if template.index is None or template.columns is None:
            raise ValueError("DataFrame template requires `index` and `columns`.")
        return pd.DataFrame(
            np.asarray(array, dtype=float),
            index=template.index,
            columns=template.columns,
        )
    raise ValueError(f"Unsupported template.kind={template.kind!r}")
