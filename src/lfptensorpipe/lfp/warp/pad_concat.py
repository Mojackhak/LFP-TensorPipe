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

from lfptensorpipe.io.timeline import raw_relative_onsets

from ..mask.annotations import MatchMode
from .utils import (
    intervals_overlap_half_open,
    raw_sample_time_bounds,
    resample_piecewise_segments,
    time_s_to_sample_index,
)


@dataclass(frozen=True)
class PadEpoch:
    """One epoch built by padding & concatenation around a single annotation."""

    label: str
    start_t: float
    end_t: float
    events_t: Dict[str, float]
    requested_intervals_s: tuple[tuple[float, float], ...] | None = None

    @property
    def nominal_duration_s(self) -> float:
        windows = self.requested_intervals_s or _pad_intervals(self.events_t)
        return sum(end - start for start, end in windows)


def _pad_intervals(events_t: dict[str, float]) -> tuple[tuple[float, float], ...]:
    return (
        (events_t["pad_left"], events_t["anno_left"]),
        (events_t["anno_right"], events_t["pad_right"]),
    )


def positive_pad_intervals(events_t: dict[str, float]) -> list[tuple[float, float]]:
    """Return the positive-width source fragments represented by one Clip epoch."""
    return [(a, b) for a, b in _pad_intervals(events_t) if b > a]


def pad_window_support(epoch: PadEpoch):
    """Yield requested and recorded bounds of each nonempty stitched window."""
    recorded = _pad_intervals(epoch.events_t)
    requested = getattr(epoch, "requested_intervals_s", None) or recorded
    for (start, end), (left, right) in zip(requested, recorded):
        if end > start:
            yield start, end, min(end, max(start, left)), max(start, min(end, right))


def pad_warper(
    raw: mne.io.BaseRaw,
    *,
    anno_allowed: Dict[str, Tuple[float, float, float, float]],
    mode: MatchMode = "exact",
    drop_mode: MatchMode = "substring",
    duration_range: Tuple[float | None, float | None] = (None, None),
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

        left  segment: [anno_start - pad_left,   anno_start + anno_left)
        right segment: [anno_end   - anno_right, anno_end   + pad_right)

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
        - Time outside the recording is retained as NaN on the stitched axis.
    """
    t_min, t_stop = raw_sample_time_bounds(raw)

    label_cfg = {str(k): tuple(map(float, v)) for k, v in anno_allowed.items()}
    for values in label_cfg.values():
        if not all(np.isfinite(value) for value in values):
            raise ValueError("`anno_allowed` window values must be finite.")
        if any(value < 0.0 for value in values):
            raise ValueError("`anno_allowed` window values must be non-negative.")
        if sum(values) <= 0.0:
            raise ValueError(
                "At least one Clip Around Event source window must have positive width."
            )
    min_dur_raw, max_dur_raw = duration_range
    min_dur = None if min_dur_raw is None else float(min_dur_raw)
    max_dur = None if max_dur_raw is None else float(max_dur_raw)
    if min_dur is not None and not np.isfinite(min_dur):
        raise ValueError("`duration_range[0]` must be finite or None.")
    if max_dur is not None and not np.isfinite(max_dur):
        raise ValueError("`duration_range[1]` must be finite or None.")
    if min_dur is not None and min_dur < 0:
        raise ValueError("`duration_range[0]` must be >= 0 or None.")
    if max_dur is not None and max_dur < 0:
        raise ValueError("`duration_range[1]` must be >= 0 or None.")
    if min_dur is not None and max_dur is not None and max_dur < min_dur:
        raise ValueError("`duration_range[1]` must be >= `duration_range[0]`.")
    labels_lower = {k.lower(): k for k in label_cfg.keys()}
    if mode not in ("substring", "exact"):
        raise ValueError("`mode` must be 'substring' or 'exact'.")
    if drop_mode not in ("substring", "exact"):
        raise ValueError("`drop_mode` must be 'substring' or 'exact'.")

    drop_lower: list[str] = []
    if anno_drop is not None:
        drop_lower = [str(x).strip().lower() for x in anno_drop if str(x).strip()]

    onsets = raw_relative_onsets(raw)
    drop_intervals: list[tuple[float, float]] = []
    if drop_lower:
        anns0 = raw.annotations
        for onset0, dur0, desc0 in zip(onsets, anns0.duration, anns0.description):
            desc_l0 = str(desc0).lower()
            if any(
                (desc_l0 == pat) if drop_mode == "exact" else (pat in desc_l0)
                for pat in drop_lower
            ):
                start0 = float(onset0)
                end0 = float(onset0 + dur0)
                drop_intervals.append((start0, end0))

    epochs_by_label: Dict[str, List[PadEpoch]] = {lbl: [] for lbl in label_cfg.keys()}

    anns = raw.annotations
    for onset, dur, desc in zip(onsets, anns.duration, anns.description):
        desc_l = str(desc).lower()
        anno_start = float(onset)
        anno_end = float(onset + dur)
        anno_dur = anno_end - anno_start

        if min_dur is not None and anno_dur < min_dur:
            continue
        if max_dur is not None and anno_dur > max_dur:
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
            requested = {
                "pad_left": anno_start - pad_left,
                "anno_left": anno_start + anno_left,
                "anno_right": anno_end - anno_right,
                "pad_right": anno_end + pad_right,
            }
            events_t = {
                key: min(t_stop, max(t_min, value)) for key, value in requested.items()
            }
            source_intervals = positive_pad_intervals(events_t)

            # Optional blacklist: discard epochs whose stitched segments overlap
            # any forbidden annotation.
            if drop_intervals:
                has_drop = False
                for d0, d1 in drop_intervals:
                    if any(
                        intervals_overlap_half_open(start, end, d0, d1)
                        for start, end in source_intervals
                    ):
                        has_drop = True
                        break
                if has_drop:
                    continue

            epochs_by_label[label].append(
                PadEpoch(
                    label=label,
                    start_t=anno_start,
                    end_t=anno_end,
                    events_t=events_t,
                    requested_intervals_s=(
                        _pad_intervals(requested) if events_t != requested else None
                    ),
                )
            )

    all_epochs: List[PadEpoch] = []
    for eps in epochs_by_label.values():
        all_epochs.extend(eps)
    all_epochs.sort(key=lambda e: (e.start_t, e.label))
    epochs_by_label["ALL"] = all_epochs

    if len(all_epochs) == 0:
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
            percent_axis: (n_samples_out,) on half-open support [0, 100).
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
        segment_list: List[list[np.ndarray]] = []
        segment_weight_list: List[list[float]] = []
        native_lengths: List[int] = []

        T_total = x.shape[-1]

        for ep in selected_eps:
            segments: list[np.ndarray] = []
            segment_weights: list[float] = []
            for start, end, left, right in pad_window_support(ep):
                for a, b, recorded in (
                    (start, left, False),
                    (left, right, True),
                    (right, end, False),
                ):
                    if b <= a:
                        continue
                    sample_start = time_s_to_sample_index(a, sr)
                    sample_stop = time_s_to_sample_index(b, sr)
                    first = max(sample_start, 0)
                    stop = min(sample_stop, T_total)
                    if recorded and stop > first:
                        piece = x[..., first:stop]
                    elif recorded and (start, end) == (left, right):
                        raise RuntimeError(
                            "A positive-width Clip Around Event source window contains "
                            "no samples at the supplied sampling rate. "
                            f"annotation='{ep.label}', interval=({a:.6f}, {b:.6f})"
                        )
                    else:
                        length = sample_stop - sample_start if n_samples is None else 1
                        piece = np.full(lead_shape + (length,), np.nan)
                    segments.append(piece)
                    segment_weights.append(b - a)

            if not segments:
                raise RuntimeError(
                    "A selected Clip Around Event epoch has no positive-width "
                    f"source window: annotation='{ep.label}'."
                )
            concat = np.concatenate(segments, axis=-1)

            warped_list.append(concat)
            segment_list.append(segments)
            segment_weight_list.append(segment_weights)
            native_lengths.append(int(concat.shape[-1]))

        if len(warped_list) == 0:
            raise RuntimeError(
                "All selected epochs became empty after clipping. Check pad durations and raw time limits."
            )

        if n_samples is None:
            n_out = max(native_lengths)
            if n_out == 0:
                raise RuntimeError(
                    "Selected Clip Around Event windows have no native samples "
                    "at the supplied sampling rate."
                )
            out = np.full(
                (len(warped_list),) + lead_shape + (n_out,),
                np.nan,
                dtype=np.result_type(x, np.float64),
            )
            for ei, ep_data in enumerate(warped_list):
                length = ep_data.shape[-1]
                out[ei, ..., :length] = ep_data
                if length < n_out and selected_eps[ei].requested_intervals_s is None:
                    out[ei, ..., length:] = ep_data[..., -1:]
        else:
            n_out = int(n_samples)
            if n_out < 2:
                raise ValueError("`n_samples` must be >= 2.")
            out = np.empty(
                (len(warped_list),) + lead_shape + (n_out,),
                dtype=np.result_type(x, np.float64),
            )
            for ei, (segments, segment_weights) in enumerate(
                zip(segment_list, segment_weight_list)
            ):
                out[ei, ...] = resample_piecewise_segments(
                    segments,
                    n_samples=n_out,
                    segment_weights=segment_weights,
                )

        percent_axis = np.linspace(0.0, 100.0, n_out, endpoint=False)
        return out, percent_axis, selected_eps

    return epochs_by_label, warp_fn
