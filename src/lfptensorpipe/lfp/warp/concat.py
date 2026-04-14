"""Concatenation warping for annotation-covered segments.

This module builds a warping function that **keeps** only selected Raw
annotation-covered intervals and **concatenates** them into a single, continuous
epoch.

Unlike the gait warper (piecewise-linear time normalization) and the pad warper
(crop + concat around each annotation), this warper removes gaps between
segments by stitching the kept intervals back-to-back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import mne

from ..mask.annotations import MatchMode
from .utils import interp_along_last_axis, time_s_to_sample_index


@dataclass(frozen=True)
class ConcatEpoch:
    """One epoch built by concatenating multiple [start, end] time intervals."""

    label: str
    intervals_s: List[Tuple[float, float]]
    total_duration_s: float


def _desc_matches(desc: str, keep: Sequence[str], *, mode: MatchMode) -> bool:
    d = str(desc).lower()
    if mode == "exact":
        return any(d == str(k).lower() for k in keep)
    return any(str(k).lower() in d for k in keep)


def _union_intervals(
    intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    cleaned: List[Tuple[float, float]] = []
    for a, b in intervals:
        a_f, b_f = float(a), float(b)
        if not (np.isfinite(a_f) and np.isfinite(b_f)):
            continue
        if b_f < a_f:
            a_f, b_f = b_f, a_f
        cleaned.append((a_f, b_f))
    if not cleaned:
        return []

    cleaned.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [cleaned[0]]
    for a, b in cleaned[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    return merged


def concat_warper(
    raw: mne.io.BaseRaw,
    *,
    keep: Sequence[str],
    mode: MatchMode = "exact",
    drop_mode: MatchMode = "substring",
    anno_drop: Sequence[str] | None = None,
    pad_s: float = 0.0,
    clip_to_raw: bool = True,
    require_match: bool = True,
) -> Tuple[Dict[str, List[ConcatEpoch]], Callable]:
    """Build a concat warper from Raw annotations.

    This warper selects annotation intervals whose description matches any entry
    in `keep`, then concatenates all matched intervals into a single continuous
    epoch.

    Args:
        raw: MNE Raw with annotations.
        keep: Annotation labels to keep (e.g., ["sit", "gait", "pain"]).
        mode: Matching mode: 'substring' (case-insensitive substring) or 'exact'.
        drop_mode: Matching mode for `anno_drop`: 'substring'
            (case-insensitive substring) or 'exact'.
        anno_drop: Optional blacklist of annotation description patterns.
            If provided, any candidate kept interval that overlaps a matching
            drop-annotation interval is discarded.
        pad_s: Optional padding (seconds) applied on both sides of each interval.
        clip_to_raw: Clip intervals to Raw time span.
        require_match: If True, raise if no annotation matches `keep`.

    Returns:
        epochs_by_label: Dict with one key 'ALL' and a single ConcatEpoch.
        warp_fn: Callable(data, sr, n_samples=None) -> (warped, percent_axis, meta_epochs)

    Notes:
        - `data` passed to warp_fn must be time-aligned with raw.
        - `sr` is the sampling rate of the *data time axis* (Hz), typically
          raw_sfreq / decim_eff for decimated tensors.
    """
    if keep is None or len(list(keep)) == 0:
        raise ValueError("`keep` must be a non-empty sequence of strings.")

    pad = float(pad_s)
    if pad < 0:
        raise ValueError("`pad_s` must be >= 0.")
    if drop_mode not in ("substring", "exact"):
        raise ValueError("`drop_mode` must be 'substring' or 'exact'.")

    drop_lower: list[str] = []
    if anno_drop is not None:
        drop_lower = [str(x).strip().lower() for x in anno_drop if str(x).strip()]

    drop_intervals: list[tuple[float, float]] = []
    if drop_lower:
        anns0 = raw.annotations
        for onset0, dur0, desc0 in zip(anns0.onset, anns0.duration, anns0.description):
            if _desc_matches(str(desc0), drop_lower, mode=drop_mode):
                start0 = float(onset0)
                end0 = float(onset0 + dur0)
                drop_intervals.append((start0, end0))

    def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
        return (float(a0) <= float(b1)) and (float(b0) <= float(a1))

    t_min = float(raw.times[0])
    t_max = float(raw.times[-1])

    intervals: List[Tuple[float, float]] = []
    for onset, dur, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        if not _desc_matches(str(desc), keep, mode=mode):
            continue
        start = float(onset) - pad
        end = float(onset + dur) + pad
        if clip_to_raw:
            start = max(start, t_min)
            end = min(end, t_max)
        if end < start:
            start, end = end, start

        if drop_intervals:
            has_drop = False
            for d0, d1 in drop_intervals:
                if _intervals_overlap(start, end, d0, d1):
                    has_drop = True
                    break
            if has_drop:
                continue

        intervals.append((start, end))

    intervals_u = _union_intervals(intervals)
    if require_match and len(intervals_u) == 0:
        raise RuntimeError("No matching annotation intervals found for concat warper.")

    total_duration = float(sum((b - a) for a, b in intervals_u))
    label = "+".join([str(k) for k in keep])
    epoch = ConcatEpoch(
        label=label, intervals_s=intervals_u, total_duration_s=total_duration
    )

    epochs_by_label: Dict[str, List[ConcatEpoch]] = {"ALL": [epoch]}

    def warp_fn(
        data: np.ndarray,
        *,
        sr: float,
        n_samples: Optional[int] = None,
        which: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[ConcatEpoch]]:
        """Concatenate the kept annotation segments (and optionally resample)."""
        if float(sr) <= 0:
            raise ValueError("`sr` must be > 0.")
        x = np.asarray(data)
        if x.ndim < 1:
            raise ValueError("`data` must have at least 1 dimension.")

        if which is not None:
            idx_list = [int(i) for i in list(which)]
            if idx_list != [0]:
                raise ValueError(
                    "concat_warper currently supports only one epoch (index 0)."
                )

        T_total = int(x.shape[-1])
        segs: List[np.ndarray] = []

        for start_s, end_s in epoch.intervals_s:
            i_start = max(time_s_to_sample_index(start_s, sr), 0)
            i_end = max(time_s_to_sample_index(end_s, sr), i_start + 1)

            i_start = min(i_start, T_total - 1)
            i_end = min(i_end, T_total)

            if i_end <= i_start:
                continue

            segs.append(x[..., i_start:i_end])

        if len(segs) == 0:
            raise RuntimeError(
                "All concat segments are empty after clipping. Check annotations and pad_s."
            )

        concat = np.concatenate(segs, axis=-1)
        L = int(concat.shape[-1])

        if n_samples is None:
            out = concat[np.newaxis, ...]
            percent = np.linspace(0.0, 100.0, L, dtype=float)
            return out.astype(np.result_type(out, np.float64)), percent, [epoch]

        if not (isinstance(n_samples, int) and int(n_samples) >= 2):
            raise ValueError("`n_samples` must be an integer >= 2 or None.")

        idx = np.linspace(0.0, max(L - 1, 0), int(n_samples), dtype=float)
        resamp = interp_along_last_axis(concat, idx)
        out = resamp[np.newaxis, ...]
        percent = np.linspace(0.0, 100.0, int(n_samples), dtype=float)
        return out.astype(np.result_type(out, np.float64)), percent, [epoch]

    return epochs_by_label, warp_fn
