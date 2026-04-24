# -*- coding: utf-8 -*-
"""
Lightweight text / label utilities for pandas DataFrames.

This module is intentionally small and free of domain assumptions. It is used to
standardize string formatting (values / columns / index) before merging or reporting.
"""

from __future__ import annotations

import pandas as pd

from collections.abc import Sequence
from typing import Any, Union


def cap_first(x: Any) -> Any:
    if isinstance(x, str) and x and x[0].islower():
        return x[0].upper() + x[1:]
    return x


def capitalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capitalize the first character of strings (values, columns, index) only if it is lowercase.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    A new DataFrame with conditionally capitalized strings.
    """

    out = df.map(cap_first)
    out.columns = [cap_first(c) for c in out.columns]
    out.index = [cap_first(i) for i in out.index]
    return out


_LEFT_TOKENS = {"lh", "l", "left"}
_RIGHT_TOKENS = {"rh", "r", "right"}


def normalize_side(value: Any) -> str:
    """
    Normalize a side label into one of {"Left", "Right", "Bilat"}.

    Rules:
    - Convert input to lowercase string (after stripping whitespace).
    - If in {"lh","l","left"} -> "Left"
    - If in {"rh","r","right"} -> "Right"
    - Otherwise -> "Bilat"

    Notes:
    - Missing values (NaN/None/pd.NA) are treated as "Bilat".
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "Bilat"

    s = str(value).strip().lower()
    if s in _LEFT_TOKENS:
        return "Left"
    if s in _RIGHT_TOKENS:
        return "Right"
    return "Bilat"


def normalize_fog(fog: str) -> str:
    if fog.lower() in ["fog", "freeze", "freezing"]:
        return "FoG"
    else:
        return fog


def tuple_1st(side: Any | tuple[Any, Any]) -> Any:
    if isinstance(side, tuple):
        side = side[0]
    return side


def join_sequence(x: Union[str, Sequence[str]], sep: str = "-") -> str:
    """
    If `x` is a str, return it as-is.
    If `x` is a sequence of str, join with `sep` and return the joined str.
    """
    if isinstance(x, str):
        return x

    # Treat any non-str Sequence as a sequence of strings
    if isinstance(x, Sequence):
        # Optional: validate items are strings (fail fast, avoid silent nonsense)
        bad = [i for i, v in enumerate(x) if not isinstance(v, str)]
        if bad:
            raise TypeError(
                f"Expected Sequence[str], but got non-str at indices {bad}."
            )
        return sep.join(x)

    raise TypeError(f"Expected str or Sequence[str], got {type(x).__name__}.")
