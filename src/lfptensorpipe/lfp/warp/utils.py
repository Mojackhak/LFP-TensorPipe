"""Utilities for warping/cropping tensors along their last (time) axis."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def segment_lengths_from_anchors_percent(
    anchors_percent: Sequence[float], n_samples: int
) -> list[int]:
    """Compute integer sample counts per segment given anchor percentages.

    The last segment is adjusted to ensure the total sums exactly to `n_samples`.

    Args:
        anchors_percent: Anchor positions as percentages, e.g. [0, 12, 50, 62, 100].
        n_samples: Total number of samples in the warped time axis.

    Returns:
        List of segment lengths (integers), length = len(anchors_percent) - 1.
    """
    anchors = np.asarray(list(anchors_percent), dtype=float)
    if anchors.ndim != 1 or anchors.size < 2:
        raise ValueError("`anchors_percent` must contain at least 2 values.")
    if not np.all(np.isfinite(anchors)):
        raise ValueError("`anchors_percent` must be finite.")
    if anchors[0] != 0 or anchors[-1] != 100:
        raise ValueError("`anchors_percent` must start at 0 and end at 100.")
    if not np.all(np.diff(anchors) > 0):
        raise ValueError("`anchors_percent` must be strictly increasing.")
    if not (isinstance(n_samples, int) and n_samples >= 2):
        raise ValueError("`n_samples` must be an integer >= 2.")

    diffs = np.diff(anchors)  # segments in percent
    raw = diffs / 100.0 * float(n_samples)
    seg = np.rint(raw).astype(int)

    # Fix rounding to ensure exact total.
    delta = int(n_samples - int(seg.sum()))
    seg[-1] += delta

    # Guard against <=0 due to extreme rounding.
    seg[seg <= 0] = 1
    if int(seg.sum()) != int(n_samples):
        seg[-1] += int(n_samples - int(seg.sum()))

    return seg.tolist()


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
    alpha = (idx - i0)[None, :]  # (1, M)

    lead = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    X = x.reshape(lead, T)
    v0 = np.take(X, i0, axis=1)  # (lead, M)
    v1 = np.take(X, i1, axis=1)  # (lead, M)
    Y = (1.0 - alpha) * v0 + alpha * v1
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
