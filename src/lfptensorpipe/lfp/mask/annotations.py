"""Mask tensors (and optionally Raw annotations) by MNE Raw annotations (pre-warp).

This module provides a small, explicit step that sits between:

    compute tensor  ->  mask by annotations  ->  warp/crop

The goal is to keep values only inside selected annotation-covered time ranges
(e.g., keep only ['sit', 'gait']) while setting all other time bins to NaN.

This module also supports **annotation masking** for interactive workflows:

- Keep only annotations that overlap the selected keep-intervals.
- Optionally split partially-overlapping annotations, or drop them entirely.

This is useful when downstream steps treat "irrelevant" annotations as
exclusion criteria (e.g., gait cycle detection).
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, TYPE_CHECKING

import numpy as np

from .mask import union_intervals

if TYPE_CHECKING:  # pragma: no cover
    import mne


MatchMode = Literal["substring", "exact"]
OverlapPolicy = Literal["split", "drop"]


def _is_fully_covered_by_union(
    start_s: float, end_s: float, union: Sequence[tuple[float, float]]
) -> bool:
    """Return True if an interval is fully covered by a union of intervals.

    This helper is used to implement the semantics required by the pipeline:
    for the **keep**-masking case, annotations that are *fully contained* inside
    the keep-interval union should be preserved even when ``overlap_policy='drop'``.

    Coverage is evaluated on continuous time in seconds.

    Args:
        start_s: Interval start time (seconds).
        end_s: Interval end time (seconds).
        union: Sorted, non-overlapping (start, end) intervals.

    Returns:
        True if [start_s, end_s] is fully covered by ``union``.
    """
    start = float(start_s)
    end = float(end_s)

    # Point event: consider it covered if the point lies within any interval.
    if end <= start:
        t = start
        for a, b in union:
            if (t >= float(a)) and (t <= float(b)):
                return True
        return False

    cur = start
    for a, b in union:
        a_f = float(a)
        b_f = float(b)
        if b_f < cur:
            continue
        if a_f > cur:
            return False  # gap
        cur = max(cur, b_f)
        if cur >= end:
            return True
    return False


def _normalize_keep_labels(keep: Sequence[str]) -> list[str]:
    labels = []
    for x in keep:
        s = str(x).strip()
        if s:
            labels.append(s.lower())
    return labels


def _desc_matches(desc: str, keep_lower: Sequence[str], *, mode: MatchMode) -> bool:
    d = str(desc).lower()
    if mode == "exact":
        return any(d == k for k in keep_lower)
    if mode == "substring":
        return any(k in d for k in keep_lower)
    raise ValueError("`mode` must be 'substring' or 'exact'.")


def _iter_matched_intervals(
    raw: "mne.io.BaseRaw",
    *,
    keep: Sequence[str],
    mode: MatchMode,
    pad_s: float,
    clip_to_raw: bool,
) -> list[dict[str, Any]]:
    """Collect matched annotation intervals as JSON-safe dicts."""
    keep_lower = _normalize_keep_labels(keep)
    if float(pad_s) < 0:
        raise ValueError("`pad_s` must be >= 0.")

    t_min = float(raw.times[0])
    t_max = float(raw.times[-1])

    matched: list[dict[str, Any]] = []
    anns = raw.annotations
    for onset, dur, desc in zip(anns.onset, anns.duration, anns.description):
        if not _desc_matches(str(desc), keep_lower, mode=mode):
            continue

        onset_f = float(onset)
        dur_f = float(dur)

        start = onset_f - float(pad_s)
        end = onset_f + dur_f + float(pad_s)

        if clip_to_raw:
            start = max(start, t_min)
            end = min(end, t_max)

        matched.append(
            dict(
                description=str(desc),
                onset_s=onset_f,
                duration_s=dur_f,
                start_s=float(start),
                end_s=float(end),
            )
        )

    return matched


def filter_raw_annotations(
    raw: "mne.io.BaseRaw",
    *,
    keep: Sequence[str],
    mode: MatchMode = "exact",
    pad_s: float = 0.0,
    clip_to_raw: bool = True,
    require_match: bool = False,
    overlap_policy: OverlapPolicy = "split",
) -> tuple["mne.io.BaseRaw", dict[str, Any]]:
    """Return a *shallow-copied* Raw with annotations filtered by keep-intervals.

    The keep-intervals are defined by annotations whose descriptions match
    any entry in ``keep`` (matching controlled by ``mode``). The returned
    Raw keeps:

    - All matching "keep" annotations (unchanged).
    - For all *other* annotations:
        * If they do not overlap the keep-interval union -> removed.
        * If they are fully covered by the keep-interval union -> kept unchanged.
        * If they partially overlap -> handled by ``overlap_policy``:
            - "split": keep only the overlapping portion(s).
            - "drop": remove the annotation entirely.

    Notes
    -----
    - This function does **not** modify the signal data.
    - The returned object is a *shallow copy* to avoid duplicating preloaded
      data buffers. Only the annotations are replaced.
    """
    if keep is None:
        raise ValueError("`keep` must be a sequence of strings, not None.")

    keep_lower = _normalize_keep_labels(keep)

    matched = _iter_matched_intervals(
        raw,
        keep=keep,
        mode=mode,
        pad_s=pad_s,
        clip_to_raw=clip_to_raw,
    )
    if require_match and len(matched) == 0:
        raise ValueError("No Raw annotations matched the requested `keep` labels.")

    keep_union = union_intervals([(itv["start_s"], itv["end_s"]) for itv in matched])

    # Prepare the filtered annotation lists.
    new_onset: list[float] = []
    new_duration: list[float] = []
    new_desc: list[str] = []

    n_total = int(len(raw.annotations))
    n_kept_exact = 0
    n_dropped_outside = 0
    n_dropped_overlap = 0
    n_kept_covered = 0
    n_split_segments = 0

    def _add(desc: str, start: float, end: float) -> None:
        # Keep MNE conventions: onset in seconds, duration >= 0.
        new_onset.append(float(start))
        new_duration.append(float(max(0.0, end - start)))
        new_desc.append(str(desc))

    for onset, dur, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        desc_s = str(desc)
        onset_f = float(onset)
        end_f = float(onset_f + float(dur))

        is_keep = _desc_matches(desc_s, keep_lower, mode=mode)
        if is_keep:
            _add(desc_s, onset_f, end_f)
            n_kept_exact += 1
            continue

        # If there is no keep interval, nothing overlaps -> drop.
        if len(keep_union) == 0:
            n_dropped_outside += 1
            continue

        # Compute intersections with keep union.
        overlaps: list[tuple[float, float]] = []
        for k0, k1 in keep_union:
            s = max(onset_f, float(k0))
            e = min(end_f, float(k1))
            if e < s:
                continue
            overlaps.append((s, e))

        if len(overlaps) == 0:
            n_dropped_outside += 1
            continue

        # If the annotation is fully covered by the keep-union, keep it unchanged
        # regardless of overlap_policy.
        if _is_fully_covered_by_union(onset_f, end_f, keep_union):
            _add(desc_s, onset_f, end_f)
            n_kept_covered += 1
            continue

        if overlap_policy == "drop":
            n_dropped_overlap += 1
            continue
        if overlap_policy != "split":
            raise ValueError("`overlap_policy` must be 'split' or 'drop'.")

        for s, e in overlaps:
            _add(desc_s, s, e)
            n_split_segments += 1

    # Sort by onset for sanity.
    if len(new_onset) > 1:
        order = np.argsort(np.asarray(new_onset, dtype=float), kind="mergesort")
        new_onset = [new_onset[i] for i in order]
        new_duration = [new_duration[i] for i in order]
        new_desc = [new_desc[i] for i in order]

    info: dict[str, Any] = dict(
        keep=[str(x) for x in keep],
        mode=str(mode),
        pad_s=float(pad_s),
        clip_to_raw=bool(clip_to_raw),
        require_match=bool(require_match),
        overlap_policy=str(overlap_policy),
        matched_intervals=matched,
        keep_union=keep_union,
        n_total=n_total,
        n_keep_exact=int(n_kept_exact),
        n_drop_outside=int(n_dropped_outside),
        n_drop_overlap=int(n_dropped_overlap),
        n_keep_covered=int(n_kept_covered),
        n_split_segments=int(n_split_segments),
        n_out=int(len(new_onset)),
    )

    # Create a shallow copy to avoid duplicating preloaded data.
    import copy

    try:
        import mne  # local import for optional dependency friendliness
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mne is required to filter Raw annotations.") from exc

    raw_masked = copy.copy(raw)
    new_ann = mne.Annotations(
        onset=np.asarray(new_onset, dtype=float),
        duration=np.asarray(new_duration, dtype=float),
        description=[str(d) for d in new_desc],
        orig_time=raw.annotations.orig_time,
    )
    raw_masked.set_annotations(new_ann)

    return raw_masked, info


def drop_raw_annotations(
    raw: "mne.io.BaseRaw",
    *,
    drop: Sequence[str],
    mode: MatchMode = "exact",
    pad_s: float = 0.0,
    clip_to_raw: bool = True,
    require_match: bool = False,
    overlap_policy: OverlapPolicy = "split",
) -> tuple["mne.io.BaseRaw", dict[str, Any]]:
    """Return a *shallow-copied* Raw with annotations removed in drop-intervals.

    The drop-intervals are defined by annotations whose descriptions match
    any entry in ``drop`` (matching controlled by ``mode``).

    The returned Raw keeps:

    - All annotations that do **not** overlap the drop-interval union.
    - For annotations that overlap the drop-interval union:
        * "split": remove only the overlapping portion(s) and keep the remainder.
        * "drop": remove the annotation entirely.

    Notes
    -----
    - This function does **not** modify the signal data.
    - The returned object is a *shallow copy* to avoid duplicating preloaded
      data buffers. Only the annotations are replaced.
    """
    if drop is None:
        raise ValueError("`drop` must be a sequence of strings, not None.")

    matched = _iter_matched_intervals(
        raw,
        keep=drop,
        mode=mode,
        pad_s=pad_s,
        clip_to_raw=clip_to_raw,
    )
    if require_match and len(matched) == 0:
        raise ValueError("No Raw annotations matched the requested `drop` labels.")

    drop_union = union_intervals([(itv["start_s"], itv["end_s"]) for itv in matched])

    new_onset: list[float] = []
    new_duration: list[float] = []
    new_desc: list[str] = []

    n_total = int(len(raw.annotations))
    n_kept_unchanged = 0
    n_dropped_full = 0
    n_dropped_overlap = 0
    n_split_segments = 0

    def _add(desc: str, start: float, end: float) -> None:
        new_onset.append(float(start))
        new_duration.append(float(max(0.0, end - start)))
        new_desc.append(str(desc))

    def _overlaps_any(t: float, intervals: Sequence[tuple[float, float]]) -> bool:
        for a, b in intervals:
            if t >= float(a) and t <= float(b):
                return True
        return False

    for onset, dur, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        desc_s = str(desc)
        onset_f = float(onset)
        dur_f = float(dur)
        end_f = float(onset_f + dur_f)

        # Fast-path: no drop interval -> keep everything.
        if len(drop_union) == 0:
            _add(desc_s, onset_f, end_f)
            n_kept_unchanged += 1
            continue

        # Point event.
        if dur_f == 0.0:
            if _overlaps_any(onset_f, drop_union):
                n_dropped_full += 1
            else:
                _add(desc_s, onset_f, end_f)
                n_kept_unchanged += 1
            continue

        # Compute intersection with the drop union.
        overlaps: list[tuple[float, float]] = []
        for d0, d1 in drop_union:
            s = max(onset_f, float(d0))
            e = min(end_f, float(d1))
            if e < s:
                continue
            overlaps.append((s, e))

        if len(overlaps) == 0:
            _add(desc_s, onset_f, end_f)
            n_kept_unchanged += 1
            continue

        if overlap_policy == "drop":
            n_dropped_overlap += 1
            continue
        if overlap_policy != "split":
            raise ValueError("`overlap_policy` must be 'split' or 'drop'.")

        # Remove the unioned drop intervals from this annotation interval.
        cur = onset_f
        kept_segments: list[tuple[float, float]] = []
        for d0, d1 in drop_union:
            if float(d1) < cur:
                continue
            if float(d0) > end_f:
                break
            s = max(cur, float(d0))
            if s > cur:
                kept_segments.append((cur, min(s, end_f)))
            cur = max(cur, float(d1))
            if cur >= end_f:
                break
        if cur < end_f:
            kept_segments.append((cur, end_f))

        # Add back only non-empty segments.
        kept_any = False
        for s, e in kept_segments:
            if e <= s:
                continue
            _add(desc_s, s, e)
            n_split_segments += 1
            kept_any = True

        if not kept_any:
            n_dropped_full += 1

    # Sort by onset for sanity.
    if len(new_onset) > 1:
        order = np.argsort(np.asarray(new_onset, dtype=float), kind="mergesort")
        new_onset = [new_onset[i] for i in order]
        new_duration = [new_duration[i] for i in order]
        new_desc = [new_desc[i] for i in order]

    info: dict[str, Any] = dict(
        drop=[str(x) for x in drop],
        mode=str(mode),
        pad_s=float(pad_s),
        clip_to_raw=bool(clip_to_raw),
        require_match=bool(require_match),
        overlap_policy=str(overlap_policy),
        matched_intervals=matched,
        drop_union=drop_union,
        n_total=n_total,
        n_kept_unchanged=int(n_kept_unchanged),
        n_drop_full=int(n_dropped_full),
        n_drop_overlap=int(n_dropped_overlap),
        n_split_segments=int(n_split_segments),
        n_out=int(len(new_onset)),
    )

    import copy

    try:
        import mne  # local import for optional dependency friendliness
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mne is required to filter Raw annotations.") from exc

    raw_masked = copy.copy(raw)
    new_ann = mne.Annotations(
        onset=np.asarray(new_onset, dtype=float),
        duration=np.asarray(new_duration, dtype=float),
        description=[str(d) for d in new_desc],
        orig_time=raw.annotations.orig_time,
    )
    raw_masked.set_annotations(new_ann)

    return raw_masked, info


def time_mask_by_annotations(
    raw: "mne.io.BaseRaw",
    times_s: Sequence[float],
    keep: Sequence[str],
    *,
    mode: MatchMode = "exact",
    pad_s: float = 0.0,
    clip_to_raw: bool = True,
    require_match: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a boolean keep-mask over a tensor time axis based on Raw annotations.

    Args:
        raw: MNE Raw whose annotations define time intervals.
        times_s: 1D time axis of the tensor (seconds).
        keep: List of annotation labels to keep. Matching is controlled by `mode`.
        mode: 'substring' (default) matches keep labels as substrings (case-insensitive);
            'exact' requires an exact string match after lowercasing.
        pad_s: Optional padding (seconds) applied on both sides of each matched interval.
        clip_to_raw: If True, clip intervals to [raw.times[0], raw.times[-1]].
        require_match: If True, raise if no annotations match `keep`.

    Returns:
        keep_mask: Boolean array with shape (n_times,) where True means "keep".
        info: Dict containing matched intervals and settings.
    """
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1:
        raise ValueError("`times_s` must be 1D.")

    if keep is None:
        raise ValueError("`keep` must be a sequence of strings, not None.")

    matched = _iter_matched_intervals(
        raw, keep=keep, mode=mode, pad_s=pad_s, clip_to_raw=clip_to_raw
    )
    if require_match and len(matched) == 0:
        raise ValueError("No Raw annotations matched the requested `keep` labels.")

    finite = np.isfinite(times)
    keep_mask = np.zeros(times.shape, dtype=bool)

    for itv in matched:
        start = float(itv["start_s"])
        end = float(itv["end_s"])
        keep_mask |= finite & (times >= start) & (times <= end)

    info: dict[str, Any] = dict(
        keep=[str(x) for x in keep],
        mode=str(mode),
        pad_s=float(pad_s),
        clip_to_raw=bool(clip_to_raw),
        matched_intervals=matched,
        n_times=int(times.size),
        n_keep=int(np.sum(keep_mask)),
    )
    return keep_mask, info


def mask_by_annotations(
    raw: "mne.io.BaseRaw",
    tensor: np.ndarray,
    times_s: Sequence[float],
    keep: Sequence[str],
    *,
    mode: MatchMode = "exact",
    pad_s: float = 0.0,
    clip_to_raw: bool = True,
    time_axis: int = -1,
    require_match: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Mask a time-aligned tensor by Raw annotations (set outside intervals to NaN).

    The mask is applied along `time_axis` (default: last axis).

    Args:
        raw: MNE Raw providing annotations.
        tensor: Input tensor (any shape) that is time-aligned with `raw`.
        times_s: Time axis for `tensor` along `time_axis` (seconds).
        keep: List of annotation labels to keep.
        mode: Matching mode ('substring' or 'exact').
        pad_s: Optional padding (seconds) applied to each matched interval.
        clip_to_raw: Clip matched intervals to Raw time span.
        time_axis: Axis index of `tensor` corresponding to time.
        require_match: If True, raise if no annotations match `keep`.

    Returns:
        tensor_masked: Copy of `tensor` with values outside kept intervals set to NaN.
        keep_mask: Boolean array of shape (n_times,) applied on the time axis.
        info: Dict describing the mask construction (intervals, settings, stats).
    """
    x = np.asarray(tensor)

    keep_mask, info = time_mask_by_annotations(
        raw,
        times_s=times_s,
        keep=keep,
        mode=mode,
        pad_s=pad_s,
        clip_to_raw=clip_to_raw,
        require_match=require_match,
    )

    # Validate axis and alignment.
    if x.ndim < 1:
        raise ValueError("`tensor` must have at least 1 dimension.")
    ax = int(time_axis)
    if ax < 0:
        ax = x.ndim + ax
    if ax < 0 or ax >= x.ndim:
        raise ValueError(
            f"`time_axis`={time_axis} is out of bounds for tensor ndim={x.ndim}."
        )

    x_moved = np.moveaxis(x, ax, -1)
    if x_moved.shape[-1] != keep_mask.shape[0]:
        raise ValueError(
            "Length mismatch: tensor time axis length does not match `times_s`."
            f" Got tensor_time={x_moved.shape[-1]} vs times={keep_mask.shape[0]}."
        )

    # Ensure dtype can represent NaN.
    if np.issubdtype(x_moved.dtype, np.integer) or x_moved.dtype == np.bool_:
        x_moved = x_moved.astype(np.float64, copy=True)
    else:
        x_moved = x_moved.copy()

    fill_value: Any
    if np.iscomplexobj(x_moved):
        fill_value = np.nan + 1j * np.nan
    else:
        fill_value = np.nan

    # Apply mask (False -> fill with NaN).
    x_moved[..., ~keep_mask] = fill_value

    x_out = np.moveaxis(x_moved, -1, ax)
    return x_out, keep_mask, info


def annotation_duration_sec(
    raw: mne.io.BaseRaw,
    annos: Sequence[str] | str,
    *,
    mode: MatchMode = "exact",
) -> float:
    """Compute the total duration (seconds) of selected annotations in an MNE Raw object.

    Matching is case-insensitive, performed against `raw.annotations.description`.

    Modes
    -----
    - "exact": normalized_description == normalized_anno
    - "substring": normalized_anno is a substring of normalized_description

    Parameters
    ----------
    raw:
        MNE Raw object.
    annos:
        Target annotation name(s).
    mode:
        Matching mode: "exact" or "substring".

    Returns
    -------
    float
        Total duration in seconds (sum of durations of all matched annotations).

    Notes
    -----
    This function sums durations as stored in `raw.annotations`. If matched annotations
    overlap in time, the overlapped time will be counted multiple times.
    """
    if isinstance(annos, str):
        anno_list = [annos]
    else:
        anno_list = list(annos)

    if mode not in ("exact", "substring"):
        raise ValueError(f"Invalid mode: {mode!r}. Expected 'exact' or 'substring'.")

    if not anno_list:
        return 0.0

    for a in anno_list:
        if not isinstance(a, str):
            raise TypeError(
                f"`annos` must be a string or a sequence of strings, got element: {type(a)}"
            )

    # Normalize targets
    targets = [a.strip().lower() for a in anno_list if a.strip()]
    if not targets:
        return 0.0

    anns = raw.annotations
    if anns is None or len(anns) == 0:
        return 0.0

    descs = np.asarray(anns.description, dtype=object)
    durs = np.asarray(anns.duration, dtype=float)

    descs_norm = np.array([str(d).strip().lower() for d in descs], dtype=object)

    if mode == "exact":
        target_set = set(targets)
        mask = np.array([d in target_set for d in descs_norm], dtype=bool)
    else:
        # substring
        mask = np.array(
            [any(t in d for t in targets) for d in descs_norm],
            dtype=bool,
        )

    return float(np.sum(durs[mask]))
