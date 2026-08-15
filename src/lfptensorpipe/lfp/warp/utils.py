"""Utilities for warping/cropping tensors along their last (time) axis."""

from __future__ import annotations

from typing import Any

import numpy as np


def raw_sample_time_bounds(raw: Any) -> tuple[float, float]:
    """Return the half-open time support owned by one MNE Raw object."""
    start = float(raw.times[0])
    stop = float(raw.times[-1]) + 1.0 / float(raw.info["sfreq"])
    return start, stop


def intervals_overlap_half_open(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    """Return whether half-open intervals or point markers overlap."""
    left_start_f = float(left_start)
    left_end_f = float(left_end)
    right_start_f = float(right_start)
    right_end_f = float(right_end)
    left_is_point = left_end_f <= left_start_f
    right_is_point = right_end_f <= right_start_f

    if left_is_point and right_is_point:
        return left_start_f == right_start_f
    if left_is_point:
        return right_start_f <= left_start_f < right_end_f
    if right_is_point:
        return left_start_f <= right_start_f < left_end_f
    return left_start_f < right_end_f and right_start_f < left_end_f


def interp_along_last_axis(data: np.ndarray, idx_grid: np.ndarray) -> np.ndarray:
    """Linear interpolation along the last axis using floating point indices.

    Args:
        data: Array with time on the last axis.
        idx_grid: 1D array of floating indices in the original time axis.

    Returns:
        Array with the same leading dimensions as `data` and last axis length
        equal to idx_grid.size.
    """
    x = np.asarray(data)
    if x.ndim < 1:
        raise ValueError("`data` must have at least 1 dimension.")
    idx = np.asarray(idx_grid, dtype=float)
    if idx.ndim != 1:
        raise ValueError("`idx_grid` must be 1D.")

    T = x.shape[-1]
    if T < 2:
        # Degenerate: repeat the only sample.
        return np.repeat(x, idx.size, axis=-1)

    # Clip to valid [0, T-1] range and clamp i1 to stay in bounds even at the endpoint.
    idx = np.clip(idx, 0.0, T - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    lead = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    X = x.reshape(lead, T)
    v0 = np.take(X, i0, axis=1)  # (lead, M)
    v1 = np.take(X, i1, axis=1)  # (lead, M)
    exact = (idx == i0) | (i0 == i1)
    fractional = ~exact
    Y = np.empty(v0.shape, dtype=np.result_type(x.dtype, np.float64))
    if np.any(exact):
        Y[:, exact] = v0[:, exact]
    if np.any(fractional):
        alpha = (idx[fractional] - i0[fractional])[None, :]
        Y[:, fractional] = (1.0 - alpha) * v0[:, fractional] + alpha * v1[:, fractional]
    return Y.reshape((*x.shape[:-1], idx.size))


def time_s_to_sample_index(time_s: float, sr_hz: float) -> int:
    """Convert a time value (seconds) into an integer sample index.

    Args:
        time_s: Time in seconds.
        sr_hz: Sampling rate in Hz.

    Returns:
        Integer sample index (rounded).
    """
    if float(sr_hz) <= 0:
        raise ValueError("`sr_hz` must be > 0.")
    return int(np.round(float(time_s) * float(sr_hz)))
