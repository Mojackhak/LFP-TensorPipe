"""Utilities for warping/cropping tensors along their last (time) axis."""

from __future__ import annotations

from typing import Any, Sequence

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


def resample_piecewise_segments(
    segments: Sequence[np.ndarray],
    *,
    n_samples: int,
    segment_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Resample concatenated segments without interpolating across their seams."""
    pieces = [np.asarray(segment) for segment in segments]
    if not pieces:
        raise ValueError("`segments` must contain at least one array.")
    if not isinstance(n_samples, (int, np.integer)) or isinstance(
        n_samples, (bool, np.bool_)
    ):
        raise ValueError("`n_samples` must be an integer >= 2.")
    n_out = int(n_samples)
    if n_out < 2:
        raise ValueError("`n_samples` must be an integer >= 2.")

    lead_shape = pieces[0].shape[:-1]
    lengths: list[int] = []
    for piece in pieces:
        if piece.ndim < 1 or piece.shape[:-1] != lead_shape:
            raise ValueError("All piecewise segments must share leading dimensions.")
        length = int(piece.shape[-1])
        if length < 1:
            raise ValueError("Piecewise segments must contain at least one sample.")
        lengths.append(length)

    weights = (
        np.asarray(lengths, dtype=float)
        if segment_weights is None
        else np.asarray(segment_weights, dtype=float)
    )
    if (
        weights.ndim != 1
        or weights.size != len(pieces)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("`segment_weights` must be finite and positive per segment.")
    cumulative = np.cumsum(weights)
    target = np.arange(n_out, dtype=float) * float(cumulative[-1]) / float(n_out)
    seam_tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(float(cumulative[-1])))
    segment_indices = np.searchsorted(
        cumulative,
        target + seam_tolerance,
        side="right",
    )
    segment_indices = np.minimum(segment_indices, len(pieces) - 1)
    starts = np.concatenate(([0.0], cumulative[:-1]))
    out = np.empty(
        lead_shape + (n_out,),
        dtype=np.result_type(*(piece.dtype for piece in pieces), np.float64),
    )
    for segment_index, piece in enumerate(pieces):
        output_mask = segment_indices == segment_index
        if not np.any(output_mask):
            continue
        local_fraction = (target[output_mask] - starts[segment_index]) / weights[
            segment_index
        ]
        local_fraction = np.clip(local_fraction, 0.0, np.nextafter(1.0, 0.0))
        local_indices = local_fraction * float(lengths[segment_index])
        out[..., output_mask] = interp_along_last_axis(piece, local_indices)
    return out


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
