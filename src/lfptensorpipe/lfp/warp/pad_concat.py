"""Pad-and-concatenate warping for annotation-centered epochs.

This file builds epochs by **cropping** two windows around an annotation
(left around onset, right around offset), then **concatenating** them.

Unlike gait warping, this method does not linearly rescale time; it constructs
a new time axis by stitching segments together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import mne

from ..mask.annotations import MatchMode
from .utils import interp_along_last_axis, time_s_to_sample_index


@dataclass(frozen=True)
class PadEpoch:
    """One epoch built by padding & concatenation around a single annotation."""

    label: str
    start_t: float
    end_t: float
    events_t: Dict[str, float]


def pad_warper(
    raw: mne.io.BaseRaw,
    *,
    anno_allowed: Dict[str, Tuple[float, float, float, float]],
    mode: MatchMode = "exact",
    drop_mode: MatchMode = "substring",
    duration_range: Tuple[float, float] = (0.0, np.inf),
    anno_drop: Sequence[str] | None = None,
) -> Tuple[Dict[str, List[PadEpoch]], Callable | None]:
    """Label epochs from Raw annotations and build a pad+concat warper.

    For every annotation whose description contains a configured substring,
    we construct an epoch by concatenating two segments:
      - left segment  around annotation onset
      - right segment around annotation offset

    Segment definitions (seconds):
        anno_start = onset
        anno_end   = onset + duration

        left  segment: [anno_start - pad_left,   anno_start + anno_left]
        right segment: [anno_end   - anno_right, anno_end   + pad_right]

    Args:
        raw: MNE Raw with annotations.
        anno_allowed: Mapping label -> (pad_left, anno_left, anno_right, pad_right).
        mode: Matching mode for `anno_allowed` labels:
            - "exact": case-insensitive exact match
            - "substring": case-insensitive substring match
        drop_mode: Matching mode for `anno_drop` patterns:
            - "exact": case-insensitive exact match
            - "substring": case-insensitive substring match
        duration_range: Allowed (min_dur, max_dur) for annotation duration.
        anno_drop: Optional blacklist of annotation description patterns.
            If provided, any candidate epoch whose **concatenated segments** overlap
            an annotation matching one of these patterns will be discarded.

    Returns:
        epochs_by_label: dict with keys per label and 'ALL' (sorted union).
        warp_fn: Callable(data, sr, n_samples=None, which=None) -> (warped, percent_axis, meta_epochs)

    Notes:
        - This is **not** a time-scaling warp. It is a crop-and-concatenate construction.
        - If padding goes outside the Raw time range, boundaries are clipped.
    """
    t_min = float(raw.times[0])
    t_max = float(raw.times[-1])

    label_cfg = {str(k): tuple(map(float, v)) for k, v in anno_allowed.items()}
    labels_lower = {k.lower(): k for k in label_cfg.keys()}
    if mode not in ("substring", "exact"):
        raise ValueError("`mode` must be 'substring' or 'exact'.")
    if drop_mode not in ("substring", "exact"):
        raise ValueError("`drop_mode` must be 'substring' or 'exact'.")

    drop_lower: list[str] = []
    if anno_drop is not None:
        drop_lower = [str(x).strip().lower() for x in anno_drop if str(x).strip()]

    drop_intervals: list[tuple[float, float]] = []
    if drop_lower:
        anns0 = raw.annotations
        for onset0, dur0, desc0 in zip(anns0.onset, anns0.duration, anns0.description):
            desc_l0 = str(desc0).lower()
            if any(
                (desc_l0 == pat) if drop_mode == "exact" else (pat in desc_l0)
                for pat in drop_lower
            ):
                start0 = float(onset0)
                end0 = float(onset0 + dur0)
                drop_intervals.append((start0, end0))

    def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
        """Return True if [a0,a1] overlaps [b0,b1] (inclusive)."""
        return (float(a0) <= float(b1)) and (float(b0) <= float(a1))

    epochs_by_label: Dict[str, List[PadEpoch]] = {lbl: [] for lbl in label_cfg.keys()}

    anns = raw.annotations
    for onset, dur, desc in zip(anns.onset, anns.duration, anns.description):
        desc_l = str(desc).lower()
        anno_start = float(onset)
        anno_end = float(onset + dur)
        anno_dur = anno_end - anno_start

        if anno_dur < float(duration_range[0]) or anno_dur > float(duration_range[1]):
            continue

        matched_labels: List[str] = []
        for key_l, original_key in labels_lower.items():
            is_match = (desc_l == key_l) if mode == "exact" else (key_l in desc_l)
            if is_match:
                matched_labels.append(original_key)

        if not matched_labels:
            continue

        for label in matched_labels:
            pad_left, anno_left, anno_right, pad_right = label_cfg[label]

            pad_left_start = max(anno_start - pad_left, t_min)
            anno_left_end = min(anno_start + anno_left, t_max)
            anno_right_start = max(anno_end - anno_right, t_min)
            pad_right_end = min(anno_end + pad_right, t_max)

            if not (
                pad_left_start <= anno_left_end and anno_right_start <= pad_right_end
            ):
                continue

            # Optional blacklist: discard epochs whose stitched segments overlap
            # any forbidden annotation.
            if drop_intervals:
                left_seg = (pad_left_start, anno_left_end)
                right_seg = (anno_right_start, pad_right_end)
                has_drop = False
                for d0, d1 in drop_intervals:
                    if _intervals_overlap(
                        left_seg[0], left_seg[1], d0, d1
                    ) or _intervals_overlap(right_seg[0], right_seg[1], d0, d1):
                        has_drop = True
                        break
                if has_drop:
                    continue

            events_t = {
                "pad_left": pad_left_start,
                "anno_left": anno_left_end,
                "anno_right": anno_right_start,
                "pad_right": pad_right_end,
            }
            epochs_by_label[label].append(
                PadEpoch(
                    label=label, start_t=anno_start, end_t=anno_end, events_t=events_t
                )
            )

    all_epochs: List[PadEpoch] = []
    for eps in epochs_by_label.values():
        all_epochs.extend(eps)
    all_epochs.sort(key=lambda e: (e.start_t, e.label))
    epochs_by_label["ALL"] = all_epochs

    if len(all_epochs) == 0:
        Warning(
            "No pad+concat epochs detected; check annotations, anno_allowed, and duration_range."
        )
        return epochs_by_label, None

    def warp_fn(
        data: np.ndarray,
        *,
        sr: float,
        n_samples: Optional[int] = None,
        which: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[PadEpoch]]:
        """Extract and (optionally) resample pad+concat epochs.

        Args:
            data: Any tensor aligned with raw; time is on the last axis.
            sr: Sampling rate of `data` time axis (Hz).
            n_samples: If None, keep native concat length (then pad/crop to max length).
                       If int, resample each epoch to exactly `n_samples`.
            which: Optional indices into epochs_by_label['ALL'].

        Returns:
            warped: (n_epochs, *lead_dims, n_samples_out)
            percent_axis: (n_samples_out,) from 0..100
            meta: list of PadEpoch
        """
        if float(sr) <= 0:
            raise ValueError("`sr` must be > 0.")
        x = np.asarray(data)

        idx_list = (
            list(range(len(all_epochs)))
            if which is None
            else [int(i) for i in list(which)]
        )
        if len(idx_list) == 0:
            raise ValueError("`which` selects zero epochs.")
        selected_eps = [all_epochs[i] for i in idx_list]

        lead_shape = x.shape[:-1]
        warped_list: List[np.ndarray] = []
        native_lengths: List[int] = []

        T_total = x.shape[-1]

        for ep in selected_eps:
            pl = ep.events_t["pad_left"]
            al = ep.events_t["anno_left"]
            ar = ep.events_t["anno_right"]
            pr = ep.events_t["pad_right"]

            i1_start = max(time_s_to_sample_index(pl, sr), 0)
            i1_end = max(time_s_to_sample_index(al, sr), i1_start + 1)
            i2_start = max(time_s_to_sample_index(ar, sr), 0)
            i2_end = max(time_s_to_sample_index(pr, sr), i2_start + 1)

            i1_start = min(i1_start, T_total - 1)
            i1_end = min(i1_end, T_total)
            i2_start = min(i2_start, T_total - 1)
            i2_end = min(i2_end, T_total)

            if i1_end <= i1_start or i2_end <= i2_start:
                continue

            seg1 = x[..., i1_start:i1_end]
            seg2 = x[..., i2_start:i2_end]
            concat = np.concatenate([seg1, seg2], axis=-1)

            warped_list.append(concat)
            native_lengths.append(int(concat.shape[-1]))

        if len(warped_list) == 0:
            raise RuntimeError(
                "All selected epochs became empty after clipping. Check pad durations and raw time limits."
            )

        if n_samples is None:
            n_out = max(native_lengths)
            out = np.empty(
                (len(warped_list),) + lead_shape + (n_out,),
                dtype=np.result_type(x, np.float64),
            )
            for ei, ep_data in enumerate(warped_list):
                L = int(ep_data.shape[-1])
                if L == n_out:
                    out[ei, ...] = ep_data
                elif L > n_out:
                    out[ei, ...] = ep_data[..., :n_out]
                else:
                    pad_width = [(0, 0)] * ep_data.ndim
                    pad_width[-1] = (0, n_out - L)
                    out[ei, ...] = np.pad(ep_data, pad_width=pad_width, mode="edge")
        else:
            n_out = int(n_samples)
            if n_out < 2:
                raise ValueError("`n_samples` must be >= 2.")
            out = np.empty(
                (len(warped_list),) + lead_shape + (n_out,),
                dtype=np.result_type(x, np.float64),
            )
            for ei, ep_data in enumerate(warped_list):
                L = int(ep_data.shape[-1])
                if L <= 1:
                    out[ei, ...] = np.repeat(ep_data, n_out, axis=-1)
                elif L == n_out:
                    out[ei, ...] = ep_data
                else:
                    idx_local = np.linspace(0.0, float(L - 1), num=n_out, endpoint=True)
                    out[ei, ...] = interp_along_last_axis(ep_data, idx_local)

        percent_axis = np.linspace(0.0, 100.0, int(out.shape[-1]), endpoint=True)
        return out, percent_axis, selected_eps

    return epochs_by_label, warp_fn
