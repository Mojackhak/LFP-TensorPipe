"""Generic event-anchored, piecewise-linear warping from MNE annotations.

This module builds a warping function that maps detected event sequences onto a
normalized percent axis (0..100) using piecewise-linear interpolation between
anchor events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import mne
import numpy as np

from ..mask.annotations import MatchMode
from .utils import interp_along_last_axis, segment_lengths_from_anchors_percent


@dataclass(frozen=True)
class LinearEpoch:
    """Container for one generic event-anchored epoch."""

    label: str
    start_t: float
    end_t: float
    events_t: Dict[str, float]
    perc_actual: Dict[str, float]


def has_drop_annotations_between(
    raw: mne.io.BaseRaw,
    t0: float,
    t1: float,
    *,
    drop_substrings: Sequence[str],
    drop_mode: MatchMode = "substring",
) -> bool:
    """Return True if any drop-annotation overlaps [t0, t1]."""
    if drop_mode not in ("substring", "exact"):
        raise ValueError("`drop_mode` must be 'substring' or 'exact'.")
    drop = tuple(s.lower() for s in drop_substrings)
    if len(drop) == 0:
        return False
    for onset, dur, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        d = str(desc).lower()
        if not any((d == s) if drop_mode == "exact" else (s in d) for s in drop):
            continue
        a0, a1 = float(onset), float(onset) + float(dur)
        # Treat point events (dur=0) as [onset, onset].
        if not (a1 < t0 or a0 > t1):
            return True
    return False


def _safe_anchor_key(token: str, idx: int, perc: float) -> str:
    """Build a unique, stable key for one anchor in metadata dicts."""
    base = "".join(ch if ch.isalnum() else "_" for ch in str(token).lower()).strip("_")
    if not base:
        base = "event"
    perc_tag = f"{float(perc):g}".replace(".", "p").replace("-", "m")
    return f"{base}_{idx:02d}_{perc_tag}"


def _resolve_linear_anchor_config(
    anchors_percent: Dict[float, str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Resolve target percentages and ordered annotation tokens.

    The expected format is:
        {target_percent_float: annotation_name_str}

    Percent keys are sorted ascending and define the ordered anchor sequence.
    """
    if anchors_percent is None or len(anchors_percent) == 0:
        raise ValueError("`anchors_percent` must be a non-empty dict.")

    anchors: List[tuple[float, str]] = []
    for perc_raw, anno_raw in anchors_percent.items():
        try:
            perc = float(perc_raw)
        except Exception as exc:
            raise ValueError(f"Invalid anchor percent key: {perc_raw!r}") from exc
        if not np.isfinite(perc):
            raise ValueError(f"Anchor percent must be finite, got: {perc_raw!r}")

        anno = str(anno_raw).strip()
        if not anno:
            raise ValueError(f"Anchor annotation is empty for percent {perc_raw!r}.")

        anchors.append((perc, anno))

    anchors = sorted(anchors, key=lambda x: x[0])
    target_perc = np.asarray([p for p, _ in anchors], dtype=float)
    anchor_tokens = [a.lower() for _, a in anchors]
    anchor_keys = [_safe_anchor_key(tok, i, p) for i, (p, tok) in enumerate(anchors)]

    if target_perc.size < 2:
        raise ValueError("`anchors_percent` must contain at least 2 anchors.")
    if target_perc[0] != 0.0 or target_perc[-1] != 100.0:
        raise ValueError("Resolved anchors must start at 0 and end at 100.")
    if np.any(np.diff(target_perc) <= 0):
        raise ValueError("Resolved anchors must be strictly increasing.")

    return target_perc, anchor_tokens, anchor_keys


def _extract_named_events_from_annotations(
    raw: mne.io.BaseRaw,
    event_tokens: Sequence[str],
    *,
    mode: MatchMode = "exact",
) -> Dict[str, np.ndarray]:
    """Collect onset times (seconds) for each event token using match mode."""
    if mode not in ("substring", "exact"):
        raise ValueError("`mode` must be 'substring' or 'exact'.")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    keys: List[str] = []
    for token in event_tokens:
        t = str(token).strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        keys.append(t)

    out: Dict[str, List[float]] = {name: [] for name in keys}
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        d = str(desc).lower()
        for token in keys:
            is_match = (d == token) if mode == "exact" else (token in d)
            if is_match:
                out[token].append(float(onset))

    return {k: np.asarray(sorted(v), dtype=float) for k, v in out.items()}


def linear_warper(
    raw: mne.io.BaseRaw,
    *,
    anchors_percent: Dict[float, str],
    mode: MatchMode = "exact",
    drop_mode: MatchMode = "substring",
    epoch_duration_range: Tuple[float | None, float | None] = (None, None),
    linear_warp: bool = True,
    percent_tolerance: float = 5.0,
    anno_drop: Sequence[str] | None = None,
) -> Tuple[Dict[str, List[LinearEpoch]], Callable]:
    """Build a generic event-anchored epoch warper from annotation events.

    Args:
        anchors_percent: Mapping of ``target_percent -> annotation_token``.
            Example: ``{0.0: 'strike', 50.0: 'off', 100.0: 'strike'}``.
            Percent keys are sorted and define anchor order.
        mode: Match mode used to map annotation descriptions to anchor tokens:
            - "exact": case-insensitive exact match
            - "substring": case-insensitive substring match
        drop_mode: Match mode used to match `anno_drop` patterns:
            - "exact": case-insensitive exact match
            - "substring": case-insensitive substring match
        anno_drop: Optional blacklist of annotation description patterns.
            If provided, any candidate epoch that overlaps a matching
            drop-annotation interval is discarded.
    """
    if drop_mode not in ("substring", "exact"):
        raise ValueError("`drop_mode` must be 'substring' or 'exact'.")
    target_perc, event_order, anchor_keys = _resolve_linear_anchor_config(
        anchors_percent
    )
    events = _extract_named_events_from_annotations(raw, event_order, mode=mode)
    event_arrays = [events[name] for name in event_order]

    min_dur_raw, max_dur_raw = epoch_duration_range
    min_dur = None if min_dur_raw is None else float(min_dur_raw)
    max_dur = None if max_dur_raw is None else float(max_dur_raw)
    if min_dur is not None and min_dur < 0:
        raise ValueError("`epoch_duration_range[0]` must be >= 0 or None.")
    if max_dur is not None and max_dur <= 0:
        raise ValueError("`epoch_duration_range[1]` must be > 0 or None.")
    if min_dur is not None and max_dur is not None and max_dur < min_dur:
        raise ValueError("`epoch_duration_range` max must be >= min.")
    if float(percent_tolerance) < 0:
        raise ValueError("`percent_tolerance` must be >= 0.")

    drop_substrings_use: tuple[str, ...] | None
    if anno_drop is None:
        drop_substrings_use = None
    else:
        merged = [str(x).strip().lower() for x in anno_drop if str(x).strip()]
        seen_drop: set[str] = set()
        keep: List[str] = []
        for s in merged:
            if s in seen_drop:
                continue
            keep.append(s)
            seen_drop.add(s)
        drop_substrings_use = tuple(keep)

    selected_epochs: List[LinearEpoch] = []

    # Build ordered event sequences: choose one valid candidate per start event.
    if all(len(arr) > 0 for arr in event_arrays):
        starts = event_arrays[0]
        max_candidates = 200000
        candidate_count = 0
        seen_bounds: set[tuple[float, float]] = set()

        def _dfs(level: int, prev_t: float, chosen: List[float], t0: float):
            nonlocal candidate_count
            if candidate_count >= max_candidates:
                return
            arr = event_arrays[level]
            idx0 = int(np.searchsorted(arr, prev_t, side="right"))
            for ii in range(idx0, len(arr)):
                t_cur = float(arr[ii])
                if max_dur is not None and (t_cur - t0) > max_dur:
                    break
                chosen.append(t_cur)
                if level == len(event_arrays) - 1:
                    candidate_count += 1
                    yield tuple(chosen)
                else:
                    yield from _dfs(level + 1, t_cur, chosen, t0)
                chosen.pop()
                if candidate_count >= max_candidates:
                    return

        for t0 in starts:
            t0_f = float(t0)
            chosen0 = [t0_f]
            for seq in _dfs(1, t0_f, chosen0, t0_f):
                t_start = float(seq[0])
                t_end = float(seq[-1])
                dur = t_end - t_start
                if dur <= 0:
                    continue
                if min_dur is not None and dur < min_dur:
                    continue
                if max_dur is not None and dur > max_dur:
                    continue

                if drop_substrings_use is not None and has_drop_annotations_between(
                    raw,
                    t_start,
                    t_end,
                    drop_substrings=drop_substrings_use,
                    drop_mode=drop_mode,
                ):
                    continue

                perc_vec = (np.asarray(seq, dtype=float) - t_start) / dur * 100.0
                if np.any(np.abs(perc_vec - target_perc) > float(percent_tolerance)):
                    continue

                bounds_key = (round(t_start, 9), round(t_end, 9))
                if bounds_key in seen_bounds:
                    continue

                events_t = {name: float(t) for name, t in zip(anchor_keys, seq)}
                events_t["start"] = float(t_start)
                events_t["end"] = float(t_end)
                perc_actual = {name: float(p) for name, p in zip(anchor_keys, perc_vec)}

                selected_epochs.append(
                    LinearEpoch(
                        label="+".join(event_order),
                        start_t=float(t_start),
                        end_t=float(t_end),
                        events_t=events_t,
                        perc_actual=perc_actual,
                    )
                )
                seen_bounds.add(bounds_key)
                break

            if candidate_count >= max_candidates:
                break

    epochs_all: List[LinearEpoch] = list(selected_epochs)
    epochs_by_label: Dict[str, List[LinearEpoch]] = {"ALL": epochs_all}

    def warp_fn(
        data: np.ndarray,
        *,
        sr: float,
        n_samples: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[LinearEpoch]]:
        if len(epochs_all) == 0:
            raise RuntimeError(
                "No valid event-anchored epoch detected for `linear_warper`."
            )
        if float(sr) <= 0:
            raise ValueError("`sr` must be > 0.")

        ep0 = epochs_all[0]
        if n_samples is None:
            native_n = (
                int(np.round((float(ep0.end_t) - float(ep0.start_t)) * float(sr))) + 1
            )
            n_samples_use = max(native_n, 2)
        else:
            if not (isinstance(n_samples, int) and int(n_samples) >= 2):
                raise ValueError("`n_samples` must be an integer >= 2 or None.")
            n_samples_use = int(n_samples)

        x = np.asarray(data)
        lead_shape = x.shape[:-1]
        out = np.empty(
            (len(epochs_all),) + lead_shape + (n_samples_use,),
            dtype=np.result_type(x, np.float64),
        )
        percent_axis = np.linspace(0.0, 100.0, n_samples_use, endpoint=True)

        seg_lengths = (
            segment_lengths_from_anchors_percent(target_perc, n_samples_use)
            if linear_warp
            else None
        )

        for ei, ep in enumerate(epochs_all):
            idx_events = np.asarray(
                [float(ep.events_t[name]) * float(sr) for name in anchor_keys],
                dtype=float,
            )
            if np.any(np.diff(idx_events) <= 0):
                raise RuntimeError(
                    f"Non-monotonic anchors for linear epoch {ei}. idx_events={idx_events}."
                )

            if linear_warp:
                seg_grids = []
                n_seg = len(seg_lengths)
                for s in range(n_seg):
                    nseg = int(seg_lengths[s])
                    start_idx, end_idx = idx_events[s], idx_events[s + 1]
                    endpoint = s == (n_seg - 1)
                    seg_grids.append(
                        np.linspace(
                            start_idx, end_idx, num=nseg, endpoint=endpoint, dtype=float
                        )
                    )

                idx_grid = np.concatenate(seg_grids, axis=0)
                if idx_grid.size != n_samples_use:
                    if idx_grid.size > n_samples_use:
                        idx_grid = idx_grid[:n_samples_use]
                    else:
                        pad = np.full(
                            (n_samples_use - idx_grid.size,), idx_grid[-1], dtype=float
                        )
                        idx_grid = np.concatenate([idx_grid, pad], axis=0)
            else:
                start_idx = float(ep.start_t) * float(sr)
                end_idx = float(ep.end_t) * float(sr)
                if end_idx <= start_idx:
                    raise RuntimeError(
                        f"Invalid linear epoch bounds for epoch {ei}. "
                        f"start_idx={start_idx}, end_idx={end_idx}."
                    )
                idx_grid = np.linspace(
                    start_idx, end_idx, num=n_samples_use, endpoint=True, dtype=float
                )

            out[ei, ...] = interp_along_last_axis(x, idx_grid)

        return out, percent_axis, list(epochs_all)

    return epochs_by_label, warp_fn
