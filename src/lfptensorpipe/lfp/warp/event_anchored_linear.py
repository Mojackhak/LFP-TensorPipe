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
from .utils import (
    interp_along_last_axis,
    intervals_overlap_half_open,
)


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
    """Return True if a drop annotation overlaps the half-open epoch."""
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
        if intervals_overlap_half_open(t0, t1, a0, a1):
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


def _resolve_unique_intermediate_sequence(
    event_arrays: Sequence[np.ndarray],
    target_perc: np.ndarray,
    *,
    start_t: float,
    end_t: float,
    percent_tolerance: float | None,
) -> tuple[int, tuple[float, ...] | None]:
    """Return the clipped path count and the unique intermediate sequence."""
    intermediate_arrays = event_arrays[1:-1]
    if not intermediate_arrays:
        return 1, ()

    duration = float(end_t) - float(start_t)
    candidate_arrays: list[np.ndarray] = []
    for arr, target in zip(intermediate_arrays, target_perc[1:-1]):
        idx0 = int(np.searchsorted(arr, start_t, side="right"))
        idx1 = int(np.searchsorted(arr, end_t, side="left"))
        candidates = np.asarray(arr[idx0:idx1], dtype=float)
        if percent_tolerance is not None and candidates.size > 0:
            actual = (candidates - float(start_t)) / duration * 100.0
            candidates = candidates[
                np.abs(actual - float(target)) <= float(percent_tolerance)
            ]
        if candidates.size == 0:
            return 0, None
        candidate_arrays.append(candidates)

    previous_values = candidate_arrays[0]
    previous_counts = [1] * int(previous_values.size)
    previous_paths: list[tuple[float, ...] | None] = [
        (float(value),) for value in previous_values
    ]

    for current_values in candidate_arrays[1:]:
        current_counts: list[int] = []
        current_paths: list[tuple[float, ...] | None] = []
        previous_index = 0
        running_count = 0
        running_path: tuple[float, ...] | None = None

        for current_value_raw in current_values:
            current_value = float(current_value_raw)
            while (
                previous_index < int(previous_values.size)
                and float(previous_values[previous_index]) < current_value
            ):
                candidate_count = int(previous_counts[previous_index])
                if candidate_count > 0:
                    if running_count == 0 and candidate_count == 1:
                        running_count = 1
                        running_path = previous_paths[previous_index]
                    else:
                        running_count = 2
                        running_path = None
                previous_index += 1

            current_counts.append(running_count)
            current_paths.append(
                None
                if running_count != 1 or running_path is None
                else running_path + (current_value,)
            )

        previous_values = current_values
        previous_counts = current_counts
        previous_paths = current_paths

    total_count = 0
    unique_path: tuple[float, ...] | None = None
    for candidate_count, candidate_path in zip(previous_counts, previous_paths):
        if candidate_count == 0:
            continue
        if total_count == 0 and candidate_count == 1:
            total_count = 1
            unique_path = candidate_path
        else:
            return 2, None

    return total_count, unique_path


def linear_warper(
    raw: mne.io.BaseRaw,
    *,
    anchors_percent: Dict[float, str],
    mode: MatchMode = "exact",
    drop_mode: MatchMode = "substring",
    epoch_duration_range: Tuple[float | None, float | None] = (None, None),
    linear_warp: bool = True,
    percent_tolerance: float | None = None,
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
    if min_dur is not None and not np.isfinite(min_dur):
        raise ValueError("`epoch_duration_range[0]` must be finite or None.")
    if max_dur is not None and not np.isfinite(max_dur):
        raise ValueError("`epoch_duration_range[1]` must be finite or None.")
    if min_dur is not None and min_dur < 0:
        raise ValueError("`epoch_duration_range[0]` must be >= 0 or None.")
    if max_dur is not None and max_dur <= 0:
        raise ValueError("`epoch_duration_range[1]` must be > 0 or None.")
    if min_dur is not None and max_dur is not None and max_dur < min_dur:
        raise ValueError("`epoch_duration_range` max must be >= min.")
    percent_tolerance_f = (
        None if percent_tolerance is None else float(percent_tolerance)
    )
    if percent_tolerance_f is not None:
        if not np.isfinite(percent_tolerance_f):
            raise ValueError("`percent_tolerance` must be finite or None.")
        if percent_tolerance_f < 0:
            raise ValueError("`percent_tolerance` must be >= 0 or None.")

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
    starts = event_arrays[0]
    ends = event_arrays[-1]
    used_end_times: set[float] = set()
    pairing_diagnostics = {
        "n_starts": int(len(starts)),
        "n_accepted": 0,
        "excluded_no_end": 0,
        "excluded_duration": 0,
        "excluded_no_anchor_sequence": 0,
        "excluded_ambiguous_anchor_sequence": 0,
        "excluded_bad_edge": 0,
    }

    for start_raw in starts:
        start_t = float(start_raw)
        end_index = int(np.searchsorted(ends, start_t, side="right"))
        selected_end: float | None = None
        has_unused_end = False

        for end_raw in ends[end_index:]:
            end_t = float(end_raw)
            if end_t in used_end_times:
                continue
            has_unused_end = True
            duration = end_t - start_t
            if min_dur is not None and duration < min_dur:
                continue
            if max_dur is not None and duration > max_dur:
                break
            selected_end = end_t
            break

        if selected_end is None:
            diagnostic_key = (
                "excluded_duration" if has_unused_end else "excluded_no_end"
            )
            pairing_diagnostics[diagnostic_key] += 1
            continue

        duration = selected_end - start_t
        path_count, intermediate_sequence = _resolve_unique_intermediate_sequence(
            event_arrays,
            target_perc,
            start_t=start_t,
            end_t=selected_end,
            percent_tolerance=percent_tolerance_f,
        )
        if path_count > 1:
            pairing_diagnostics["excluded_ambiguous_anchor_sequence"] += 1
            continue
        if path_count == 0 or intermediate_sequence is None:
            pairing_diagnostics["excluded_no_anchor_sequence"] += 1
            continue

        if drop_substrings_use is not None and has_drop_annotations_between(
            raw,
            start_t,
            selected_end,
            drop_substrings=drop_substrings_use,
            drop_mode=drop_mode,
        ):
            pairing_diagnostics["excluded_bad_edge"] += 1
            continue

        sequence = (start_t, *intermediate_sequence, selected_end)
        perc_vec = (np.asarray(sequence, dtype=float) - start_t) / duration * 100.0
        events_t = {
            name: float(event_t) for name, event_t in zip(anchor_keys, sequence)
        }
        events_t["start"] = start_t
        events_t["end"] = selected_end
        perc_actual = {
            name: float(percent) for name, percent in zip(anchor_keys, perc_vec)
        }

        selected_epochs.append(
            LinearEpoch(
                label="+".join(event_order),
                start_t=start_t,
                end_t=selected_end,
                events_t=events_t,
                perc_actual=perc_actual,
            )
        )
        used_end_times.add(selected_end)
        pairing_diagnostics["n_accepted"] += 1

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
                if n_samples_use < target_perc.size:
                    raise ValueError(
                        "`n_samples` must be at least the number of target anchors."
                    )
                idx_grid = np.interp(percent_axis, target_perc, idx_events)
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

    warp_fn.alignment_diagnostics = {  # type: ignore[attr-defined]
        "linear_event_pairing_diagnostics": dict(pairing_diagnostics)
    }

    return epochs_by_label, warp_fn
