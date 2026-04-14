"""Frequency helper utilities.

This module contains helpers to work with frequency grids in the presence of
notch-filtered (removed) frequency bins.

Public API
----------
- :func:`cut_notched_freqs`: remove bins within ±bandwidth around notch centers.
- :func:`split_bands_by_intervals`: split band definitions by arbitrary excluded intervals.
- :func:`split_bands_by_notches`: split band definitions to avoid notch holes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class NotchCutResult:
    """Result of cutting notched frequency bins."""

    freqs_kept: np.ndarray
    keep_mask: np.ndarray
    removed_mask: np.ndarray


def cut_notched_freqs(
    freqs: np.ndarray,
    notches: np.ndarray | Iterable[float],
    bandwidth_hz: float,
    *,
    include_edges: bool = True,
) -> NotchCutResult:
    """Remove frequency bins that fall within ±bandwidth_hz around notch centers.

    Args:
        freqs: 1D frequency grid.
        notches: Notch center frequencies (Hz).
        bandwidth_hz: Radius (Hz) around each notch center to remove.
        include_edges: If True, remove frequencies exactly at the boundaries.

    Returns:
        NotchCutResult with kept frequencies and masks on the original grid.
    """
    freqs_arr = np.asarray(freqs, dtype=float).ravel()
    notches_arr = np.asarray(list(notches), dtype=float).ravel()

    if freqs_arr.ndim != 1:
        raise ValueError("freqs must be a 1D array.")
    if float(bandwidth_hz) < 0:
        raise ValueError("bandwidth_hz must be >= 0.")
    if np.any(~np.isfinite(freqs_arr)):
        raise ValueError("freqs contains non-finite values.")
    if np.any(~np.isfinite(notches_arr)):
        raise ValueError("notches contains non-finite values.")

    if notches_arr.size == 0:
        keep_mask = np.ones(freqs_arr.shape[0], dtype=bool)
        return NotchCutResult(
            freqs_kept=freqs_arr.copy(), keep_mask=keep_mask, removed_mask=~keep_mask
        )

    removed_mask = np.zeros(freqs_arr.shape[0], dtype=bool)
    for notch in notches_arr:
        lo = float(notch) - float(bandwidth_hz)
        hi = float(notch) + float(bandwidth_hz)
        if include_edges:
            removed_mask |= (freqs_arr >= lo) & (freqs_arr <= hi)
        else:
            removed_mask |= (freqs_arr > lo) & (freqs_arr < hi)

    keep_mask = ~removed_mask
    return NotchCutResult(
        freqs_kept=freqs_arr[keep_mask], keep_mask=keep_mask, removed_mask=removed_mask
    )


BandValue = tuple[float, float]
BandValueOrSegments = BandValue | list[BandValue]


def split_bands_by_intervals(
    bands: Mapping[str, BandValue],
    intervals: Iterable[BandValue],
    *,
    include_edges: bool = True,
    drop_empty: bool = True,
) -> dict[str, BandValueOrSegments]:
    """Split band definitions by removing arbitrary excluded intervals.

    Args:
        bands: Mapping band_name -> (fmin, fmax).
        intervals: Excluded intervals `(fmin, fmax)` to cut out of each band.
        include_edges: If True, touching boundaries count as overlap.
        drop_empty: If True, drop bands that become empty after splitting.

    Returns:
        Mapping band_name -> `(fmin, fmax)` or `[(fmin, fmax), ...]`.
    """
    intervals_list = [
        (float(lo), float(hi))
        for lo, hi in intervals
        if np.isfinite(float(lo)) and np.isfinite(float(hi))
    ]

    out: dict[str, BandValueOrSegments] = {}

    for name, (band_lo, band_hi) in bands.items():
        band_lo_f = float(band_lo)
        band_hi_f = float(band_hi)
        if not np.isfinite(band_lo_f) or not np.isfinite(band_hi_f):
            raise ValueError(
                f"Band '{name}' has non-finite bounds: {(band_lo, band_hi)}"
            )
        if band_hi_f <= band_lo_f:
            raise ValueError(
                f"Band '{name}' must satisfy fmax > fmin, got {(band_lo, band_hi)}"
            )

        holes: list[tuple[float, float]] = []
        for hole_lo, hole_hi in intervals_list:
            overlaps = (
                hole_hi >= band_lo_f and hole_lo <= band_hi_f
                if include_edges
                else hole_hi > band_lo_f and hole_lo < band_hi_f
            )
            if not overlaps:
                continue
            holes.append((max(band_lo_f, hole_lo), min(band_hi_f, hole_hi)))

        if not holes:
            out[name] = (band_lo_f, band_hi_f)
            continue

        holes.sort(key=lambda item: item[0])
        merged: list[tuple[float, float]] = []
        cur_lo, cur_hi = holes[0]
        for lo, hi in holes[1:]:
            if lo <= cur_hi if include_edges else lo < cur_hi:
                cur_hi = max(cur_hi, hi)
            else:
                merged.append((cur_lo, cur_hi))
                cur_lo, cur_hi = lo, hi
        merged.append((cur_lo, cur_hi))

        segments: list[tuple[float, float]] = []
        cursor = band_lo_f
        for lo, hi in merged:
            if lo > cursor:
                segments.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < band_hi_f:
            segments.append((cursor, band_hi_f))

        segments = [(lo, hi) for lo, hi in segments if hi > lo]
        if not segments:
            if not drop_empty:
                out[name] = []
            continue

        out[name] = segments if len(segments) > 1 else segments[0]

    return out


def split_bands_by_notches(
    bands: Mapping[str, BandValue],
    notches: np.ndarray | Iterable[float],
    bandwidth_hz: float,
    *,
    include_edges: bool = True,
    drop_empty: bool = True,
) -> dict[str, BandValueOrSegments]:
    """Split band definitions by removing notch-hole intervals.

    If a band overlaps a notch hole (±bandwidth_hz around a notch center), the
    output for that band becomes a list of (fmin, fmax) segments after removing
    the hole(s).

    Args:
        bands: Mapping band_name -> (fmin,fmax).
        notches: Notch center frequencies (Hz).
        bandwidth_hz: Radius (Hz) around each notch center to exclude.
        include_edges: If True, treat boundaries as overlapping.
        drop_empty: If True, drop bands that become empty after splitting.

    Returns:
        A mapping band_name -> (fmin,fmax) or list[(fmin,fmax)].
    """
    notches_arr = np.asarray(list(notches), dtype=float).ravel()

    if float(bandwidth_hz) < 0:
        raise ValueError("bandwidth_hz must be >= 0.")
    if np.any(~np.isfinite(notches_arr)):
        raise ValueError("notches contains non-finite values.")

    intervals = [
        (float(notch) - float(bandwidth_hz), float(notch) + float(bandwidth_hz))
        for notch in notches_arr.tolist()
    ]
    return split_bands_by_intervals(
        bands,
        intervals,
        include_edges=include_edges,
        drop_empty=drop_empty,
    )
