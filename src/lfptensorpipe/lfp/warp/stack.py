"""Stack warping for annotation epochs.

This module selects annotation-covered intervals and treats each interval as an
independent epoch. Unlike ``concat_warper`` (which stitches all kept segments
into one long epoch), ``stack_warper`` linearly rescales each epoch to the same
number of samples and stacks them on an epoch axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import mne

from ..mask.annotations import MatchMode
from .utils import interp_along_last_axis, time_s_to_sample_index


@dataclass(frozen=True)
class StackEpoch:
    """One annotation interval treated as one stack epoch."""

    label: str
    start_t: float
    end_t: float
    intervals_s: List[Tuple[float, float]]
    total_duration_s: float


def _match_label(desc: str, keep: Sequence[str], *, mode: MatchMode) -> str | None:
    d = str(desc).lower()
    if mode == "exact":
        for k in keep:
            if d == str(k).lower():
                return str(k)
        return None
    for k in keep:
        if str(k).lower() in d:
            return str(k)
    return None


def stack_warper(
    raw: mne.io.BaseRaw,
    *,
    keep: Sequence[str],
    mode: MatchMode = "exact",
    drop_mode: MatchMode = "substring",
    duration_range: Tuple[float, float] = (0.0, np.inf),
    anno_drop: Sequence[str] | None = None,
    pad_s: float = 0.0,
    clip_to_raw: bool = True,
    require_match: bool = True,
) -> Tuple[Dict[str, List[StackEpoch]], Callable]:
    """Build a stack warper from Raw annotations.

    Args:
        raw: MNE Raw with annotations.
        keep: Annotation labels to keep.
        mode: Matching mode: 'substring' (case-insensitive substring) or 'exact'.
        drop_mode: Matching mode for `anno_drop`: 'substring'
            (case-insensitive substring) or 'exact'.
        duration_range: Allowed annotation duration range in seconds
            ``(min_dur, max_dur)``. Filtering uses the raw annotation duration
            (before applying ``pad_s`` / clipping).
        anno_drop: Optional blacklist of annotation description patterns.
            If provided, any candidate epoch that overlaps a matching
            drop-annotation interval is discarded.
        pad_s: Optional padding (seconds) applied on both sides of each interval.
        clip_to_raw: Clip intervals to Raw time span.
        require_match: If True, raise if no annotation matches `keep`.

    Returns:
        epochs_by_label: Dict with keys per matched label and 'ALL'.
        warp_fn: Callable(data, sr, n_samples=None, which=None)
            -> (warped, percent_axis, meta_epochs)
    """
    if keep is None or len(list(keep)) == 0:
        raise ValueError("`keep` must be a non-empty sequence of strings.")

    pad = float(pad_s)
    if pad < 0:
        raise ValueError("`pad_s` must be >= 0.")

    min_dur = float(duration_range[0])
    max_dur = float(duration_range[1])
    if min_dur < 0:
        raise ValueError("`duration_range[0]` must be >= 0.")
    if max_dur < min_dur:
        raise ValueError("`duration_range[1]` must be >= `duration_range[0]`.")
    if drop_mode not in ("substring", "exact"):
        raise ValueError("`drop_mode` must be 'substring' or 'exact'.")

    drop_lower: list[str] = []
    if anno_drop is not None:
        drop_lower = [str(x).strip().lower() for x in anno_drop if str(x).strip()]

    drop_intervals: list[tuple[float, float]] = []
    if drop_lower:
        anns0 = raw.annotations
        for onset0, dur0, desc0 in zip(anns0.onset, anns0.duration, anns0.description):
            if _match_label(str(desc0), drop_lower, mode=drop_mode) is not None:
                start0 = float(onset0)
                end0 = float(onset0 + dur0)
                drop_intervals.append((start0, end0))

    def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
        return (float(a0) <= float(b1)) and (float(b0) <= float(a1))

    t_min = float(raw.times[0])
    t_max = float(raw.times[-1])

    epochs_by_label: Dict[str, List[StackEpoch]] = {}
    epochs_all: List[StackEpoch] = []

    for onset, dur, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        matched_label = _match_label(str(desc), keep, mode=mode)
        if matched_label is None:
            continue

        anno_dur = float(dur)
        if anno_dur < min_dur or anno_dur > max_dur:
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

        ep = StackEpoch(
            label=matched_label,
            start_t=float(start),
            end_t=float(end),
            intervals_s=[(float(start), float(end))],
            total_duration_s=float(end - start),
        )
        epochs_all.append(ep)
        epochs_by_label.setdefault(matched_label, []).append(ep)

    epochs_all.sort(key=lambda ep: (ep.start_t, ep.end_t, ep.label))
    for k in list(epochs_by_label.keys()):
        epochs_by_label[k].sort(key=lambda ep: (ep.start_t, ep.end_t))
    epochs_by_label["ALL"] = epochs_all

    if require_match and len(epochs_all) == 0:
        raise RuntimeError("No matching annotation intervals found for stack warper.")

    def warp_fn(
        data: np.ndarray,
        *,
        sr: float,
        n_samples: Optional[int] = None,
        which: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[StackEpoch]]:
        """Resample each selected epoch to a shared timeline and stack them."""
        if float(sr) <= 0:
            raise ValueError("`sr` must be > 0.")

        x = np.asarray(data)
        if x.ndim < 1:
            raise ValueError("`data` must have at least 1 dimension.")

        idx_list = (
            list(range(len(epochs_all)))
            if which is None
            else [int(i) for i in list(which)]
        )
        if len(idx_list) == 0:
            raise ValueError("`which` selects zero epochs.")

        selected = [epochs_all[i] for i in idx_list]
        lead_shape = x.shape[:-1]
        T_total = int(x.shape[-1])

        native: List[np.ndarray] = []
        native_lengths: List[int] = []
        for ep in selected:
            i_start = max(time_s_to_sample_index(ep.start_t, sr), 0)
            i_end = max(time_s_to_sample_index(ep.end_t, sr), i_start + 1)

            i_start = min(i_start, T_total - 1)
            i_end = min(i_end, T_total)

            if i_end <= i_start:
                raise RuntimeError(
                    "A selected stack epoch became empty after clipping. "
                    f"epoch=({ep.start_t:.6f}, {ep.end_t:.6f})"
                )

            seg = x[..., i_start:i_end]
            native.append(seg)
            native_lengths.append(int(seg.shape[-1]))

        if n_samples is None:
            n_out = int(max(native_lengths))
        else:
            if not (isinstance(n_samples, int) and int(n_samples) >= 2):
                raise ValueError("`n_samples` must be an integer >= 2 or None.")
            n_out = int(n_samples)

        out = np.empty(
            (len(selected),) + lead_shape + (n_out,),
            dtype=np.result_type(x, np.float64),
        )

        for ei, seg in enumerate(native):
            L = int(seg.shape[-1])
            if L <= 1:
                out[ei, ...] = np.repeat(seg, n_out, axis=-1)
            elif L == n_out:
                out[ei, ...] = seg
            else:
                idx = np.linspace(
                    0.0, float(L - 1), num=n_out, endpoint=True, dtype=float
                )
                out[ei, ...] = interp_along_last_axis(seg, idx)

        percent = np.linspace(0.0, 100.0, n_out, endpoint=True, dtype=float)
        return out, percent, selected

    return epochs_by_label, warp_fn
