"""Frequency helper utilities.

This module contains a helper to work with frequency grids in the presence of
excluded frequency intervals.

Public API
----------
- :func:`split_bands_by_intervals`: split band definitions by arbitrary excluded intervals.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

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
