"""Low-level masking utilities.

This module holds small, reusable helpers used by higher-level masking
pipelines.

Conventions
-----------
- Time axis is assumed to be the **last** dimension.
- A boolean mask of shape (n_times,) indicates which time bins to **keep**.
- Values outside the keep-mask are set to NaN.

The higher-level APIs live in :mod:`lfp.pipelines.masking`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Interval = Tuple[float, float]


def union_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """Merge overlapping (start, end) intervals into a union.

    Args:
        intervals: Sequence of (start_s, end_s) pairs.

    Returns:
        A sorted list of non-overlapping intervals.
    """
    cleaned = [
        (float(a), float(b))
        for a, b in intervals
        if np.isfinite(a) and np.isfinite(b) and float(b) > float(a)
    ]
    if not cleaned:
        return []

    cleaned.sort(key=lambda x: x[0])
    merged: List[Interval] = [cleaned[0]]

    for a, b in cleaned[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))

    return merged


def apply_time_mask_nan(tensor: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Set values to NaN where keep_mask is False.

    Args:
        tensor: Input array whose last axis is time.
        keep_mask: Boolean array of shape (n_times,). True means "keep".

    Returns:
        A masked copy of ``tensor``.

    Notes:
        - Integer/bool inputs are promoted to float64.
        - Complex inputs are promoted to complex64 and filled with ``nan + 1j*nan``.
    """
    x = np.asarray(tensor)
    m = np.asarray(keep_mask, dtype=bool)

    if x.ndim < 1:
        raise ValueError("`tensor` must have at least 1 dimension.")
    if m.ndim != 1:
        raise ValueError("`keep_mask` must be 1D.")
    if x.shape[-1] != m.shape[0]:
        raise ValueError(
            f"Keep-mask length {m.shape[0]} does not match tensor time axis {x.shape[-1]}."
        )

    if np.iscomplexobj(x):
        out = x.astype(np.complex64, copy=True)
        out[..., ~m] = np.nan + 1j * np.nan
        return out

    # Ensure dtype can represent NaN.
    if np.issubdtype(x.dtype, np.integer) or x.dtype == np.bool_:
        out = x.astype(np.float64, copy=True)
    else:
        out = x.astype(np.float64, copy=True)

    out[..., ~m] = np.nan
    return out
