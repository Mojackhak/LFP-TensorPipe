"""Native-timeline mapping and reduction helpers for Burst data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

_OVERLAP_EPS_FACTOR = 64.0


@dataclass(frozen=True)
class WarpSegment:
    """One increasing target-percent to source-seconds mapping segment."""

    target_start: float
    target_end: float
    source_start: float
    source_end: float
    occurrence: int


@dataclass(frozen=True)
class MappedFragment:
    """One selected target fragment mapped back to original source time."""

    target_start: float
    target_end: float
    source_start: float
    source_end: float
    occurrence: int


@dataclass(frozen=True)
class BurstScalarResult:
    """Native-timeline Burst scalar values and their sufficient statistics."""

    mean: float
    rate: float
    duration: float
    occupancy: float
    valid_duration: float
    burst_duration: float
    event_count: int


def _positive_duration(start: float, end: float) -> bool:
    tolerance = (
        _OVERLAP_EPS_FACTOR
        * np.finfo(np.float64).eps
        * max(1.0, abs(float(start)), abs(float(end)))
    )
    return float(end) - float(start) > tolerance


def sample_cell_edges(times: Sequence[float]) -> np.ndarray:
    """Build midpoint cell edges for strictly increasing sample centers."""
    centers = np.asarray(times, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Burst source time axis must contain at least two samples.")
    if not np.all(np.isfinite(centers)) or np.any(np.diff(centers) <= 0.0):
        raise ValueError(
            "Burst source time axis must be finite and strictly increasing."
        )
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def _validated_segment(
    target_start: float,
    target_end: float,
    source_start: float,
    source_end: float,
    occurrence: int,
) -> WarpSegment:
    values = np.asarray(
        [target_start, target_end, source_start, source_end], dtype=float
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Burst alignment mapping contains non-finite bounds.")
    if not _positive_duration(target_start, target_end):
        raise ValueError("Burst alignment target bounds must be strictly increasing.")
    if not _positive_duration(source_start, source_end):
        raise ValueError("Burst alignment source bounds must be strictly increasing.")
    return WarpSegment(
        float(target_start),
        float(target_end),
        float(source_start),
        float(source_end),
        int(occurrence),
    )


def _cumulative_segments(
    intervals: Iterable[tuple[float, float]],
) -> list[WarpSegment]:
    source_intervals = [(float(start), float(end)) for start, end in intervals]
    durations = [end - start for start, end in source_intervals]
    if not source_intervals or any(
        not _positive_duration(0.0, dur) for dur in durations
    ):
        raise ValueError("Burst alignment contains an empty source interval.")
    total = float(sum(durations))
    cursor = 0.0
    segments: list[WarpSegment] = []
    for occurrence, ((start, end), duration) in enumerate(
        zip(source_intervals, durations)
    ):
        target_start = 100.0 * cursor / total
        cursor += duration
        target_end = 100.0 * cursor / total
        segments.append(
            _validated_segment(target_start, target_end, start, end, occurrence)
        )
    return segments


def build_warp_segments(
    method: str,
    method_params: dict[str, Any],
    epoch: Any,
) -> list[WarpSegment]:
    """Recover one epoch's persisted percent-to-source mapping."""
    method_key = str(method).strip().lower()
    if method_key == "stack_warper":
        return [
            _validated_segment(
                0.0,
                100.0,
                float(getattr(epoch, "start_t")),
                float(getattr(epoch, "end_t")),
                0,
            )
        ]
    if method_key == "linear_warper":
        from ..warp.event_anchored_linear import _resolve_linear_anchor_config

        if not bool(method_params.get("linear_warp", True)):
            return [
                _validated_segment(
                    0.0,
                    100.0,
                    float(getattr(epoch, "start_t")),
                    float(getattr(epoch, "end_t")),
                    0,
                )
            ]
        anchors_node = method_params.get("anchors_percent")
        events_t = getattr(epoch, "events_t", None)
        if not isinstance(anchors_node, dict) or not isinstance(events_t, dict):
            raise ValueError("Linear Burst mapping is missing persisted anchors.")
        target_array, _tokens, event_keys = _resolve_linear_anchor_config(anchors_node)
        target_anchors = target_array.tolist()
        if len(target_anchors) != len(event_keys) or len(target_anchors) < 2:
            raise ValueError("Linear Burst mapping anchor counts are inconsistent.")
        source_anchors = [float(events_t[key]) for key in event_keys]
        return [
            _validated_segment(
                target_anchors[index],
                target_anchors[index + 1],
                source_anchors[index],
                source_anchors[index + 1],
                0,
            )
            for index in range(len(target_anchors) - 1)
        ]
    if method_key == "pad_warper":
        events_t = getattr(epoch, "events_t", None)
        if not isinstance(events_t, dict):
            raise ValueError("Pad Burst mapping is missing persisted event bounds.")
        return _cumulative_segments(
            [
                (float(events_t["pad_left"]), float(events_t["anno_left"])),
                (float(events_t["anno_right"]), float(events_t["pad_right"])),
            ]
        )
    if method_key == "concat_warper":
        intervals = getattr(epoch, "intervals_s", None)
        if not isinstance(intervals, list):
            raise ValueError("Concat Burst mapping is missing persisted intervals.")
        return _cumulative_segments(intervals)
    raise ValueError(f"Unsupported Burst alignment method: {method}")


def merge_percent_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Validate, clip, sort, and union touching percent intervals."""
    cleaned: list[tuple[float, float]] = []
    for start, end in intervals:
        raw_start = float(start)
        raw_end = float(end)
        if not (np.isfinite(raw_start) and np.isfinite(raw_end)):
            raise ValueError("Burst Feature phase bounds must be finite.")
        start_f = max(0.0, raw_start)
        end_f = min(100.0, raw_end)
        if not _positive_duration(start_f, end_f):
            raise ValueError(
                "Burst Feature phase must have positive overlap with [0, 100]."
            )
        cleaned.append((start_f, end_f))
    cleaned.sort()
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def map_percent_intervals(
    segments: Sequence[WarpSegment],
    intervals: Iterable[tuple[float, float]],
) -> list[MappedFragment]:
    """Inverse-map percent intervals through ordered persisted warp segments."""
    fragments: list[MappedFragment] = []
    for phase_start, phase_end in merge_percent_intervals(intervals):
        for segment in segments:
            target_start = max(phase_start, segment.target_start)
            target_end = min(phase_end, segment.target_end)
            if not _positive_duration(target_start, target_end):
                continue
            scale = (segment.source_end - segment.source_start) / (
                segment.target_end - segment.target_start
            )
            source_start = (
                segment.source_start + (target_start - segment.target_start) * scale
            )
            source_end = (
                segment.source_start + (target_end - segment.target_start) * scale
            )
            fragments.append(
                MappedFragment(
                    target_start,
                    target_end,
                    source_start,
                    source_end,
                    segment.occurrence,
                )
            )
    fragments.sort(key=lambda item: (item.target_start, item.target_end))
    return fragments


def _cell_overlaps(
    edges: np.ndarray,
    fragment: MappedFragment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = max(
        0,
        int(np.searchsorted(edges, fragment.source_start, side="right")) - 1,
    )
    stop = min(
        edges.size - 1,
        int(np.searchsorted(edges, fragment.source_end, side="left")),
    )
    indices = np.arange(start, stop, dtype=np.int64)
    if indices.size == 0:
        empty = np.asarray([], dtype=float)
        return indices, empty, empty, empty
    overlap_starts = np.maximum(edges[indices], fragment.source_start)
    overlap_ends = np.minimum(edges[indices + 1], fragment.source_end)
    durations = overlap_ends - overlap_starts
    tolerances = (
        _OVERLAP_EPS_FACTOR
        * np.finfo(np.float64).eps
        * np.maximum.reduce(
            [
                np.ones(durations.size, dtype=float),
                np.abs(overlap_starts),
                np.abs(overlap_ends),
            ]
        )
    )
    keep = durations > tolerances
    return (
        indices[keep],
        durations[keep],
        overlap_starts[keep],
        overlap_ends[keep],
    )


def burst_event_ids(values: Sequence[float]) -> np.ndarray:
    """Label maximal finite-positive source-cell runs with stable integer IDs."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("Burst values must be one-dimensional.")
    positive = np.isfinite(data) & (data > 0.0)
    starts = positive.copy()
    starts[1:] &= ~positive[:-1]
    ids = np.cumsum(starts, dtype=np.int64) - 1
    ids[~positive] = -1
    return ids


def reduce_burst_scalars(
    values: Sequence[float],
    source_times: Sequence[float],
    fragments: Sequence[MappedFragment],
    *,
    cell_edges: np.ndarray | None = None,
    event_ids: np.ndarray | None = None,
) -> BurstScalarResult:
    """Compute Burst scalars directly from source cells selected by fragments."""
    data = np.asarray(values, dtype=float)
    edges = (
        sample_cell_edges(source_times)
        if cell_edges is None
        else np.asarray(cell_edges, dtype=float)
    )
    if data.ndim != 1 or data.size + 1 != edges.size:
        raise ValueError("Burst values and source time axis have inconsistent lengths.")
    ids = burst_event_ids(data) if event_ids is None else np.asarray(event_ids)
    if ids.ndim != 1 or ids.size != data.size:
        raise ValueError("Burst event IDs and values have inconsistent lengths.")

    valid_duration = 0.0
    burst_duration = 0.0
    amplitude_integral = 0.0
    identities: set[tuple[int, int]] = set()
    boundary_identities: list[
        tuple[float, float, tuple[int, int] | None, tuple[int, int] | None]
    ] = []
    for fragment in fragments:
        indices, durations, overlap_starts, overlap_ends = _cell_overlaps(
            edges, fragment
        )
        if indices.size == 0:
            boundary_identities.append(
                (fragment.target_start, fragment.target_end, None, None)
            )
            continue
        selected_values = data[indices]
        selected_ids = ids[indices]
        finite = np.isfinite(selected_values)
        positive = selected_ids >= 0
        valid_duration += float(np.sum(durations[finite]))
        burst_duration += float(np.sum(durations[positive]))
        amplitude_integral += float(
            np.sum(selected_values[positive] * durations[positive])
        )
        unique_ids = np.unique(selected_ids[positive])
        identities.update(
            (int(fragment.occurrence), int(event_id)) for event_id in unique_ids
        )

        positive_positions = np.flatnonzero(positive)
        first_identity = None
        last_identity = None
        if positive_positions.size:
            first_position = int(positive_positions[0])
            if not _positive_duration(
                fragment.source_start, overlap_starts[first_position]
            ):
                first_identity = (
                    int(fragment.occurrence),
                    int(selected_ids[first_position]),
                )
            last_position = int(positive_positions[-1])
            if not _positive_duration(overlap_ends[last_position], fragment.source_end):
                last_identity = (
                    int(fragment.occurrence),
                    int(selected_ids[last_position]),
                )
        boundary_identities.append(
            (
                fragment.target_start,
                fragment.target_end,
                first_identity,
                last_identity,
            )
        )

    if not _positive_duration(0.0, valid_duration):
        return BurstScalarResult(*(float("nan"),) * 4, 0.0, 0.0, 0)

    parent = {identity: identity for identity in identities}

    def find(identity: tuple[int, int]) -> tuple[int, int]:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for previous, current in zip(boundary_identities, boundary_identities[1:]):
        if (
            not _positive_duration(previous[1], current[0])
            and previous[3] is not None
            and current[2] is not None
        ):
            union(previous[3], current[2])

    event_count = len({find(identity) for identity in identities})
    if event_count == 0:
        return BurstScalarResult(
            float("nan"), 0.0, float("nan"), 0.0, valid_duration, 0.0, 0
        )
    return BurstScalarResult(
        amplitude_integral / burst_duration,
        event_count / valid_duration,
        burst_duration / event_count,
        100.0 * burst_duration / valid_duration,
        valid_duration,
        burst_duration,
        event_count,
    )


def display_bin_value(
    values: Sequence[float],
    source_times: Sequence[float],
    fragments: Sequence[MappedFragment],
    *,
    cell_edges: np.ndarray | None = None,
) -> float:
    """Aggregate one aligned Burst display bin from original source cells."""
    data = np.asarray(values, dtype=float)
    edges = (
        sample_cell_edges(source_times)
        if cell_edges is None
        else np.asarray(cell_edges, dtype=float)
    )
    if data.ndim != 1 or data.size + 1 != edges.size:
        raise ValueError("Burst values and source time axis have inconsistent lengths.")
    burst_duration = 0.0
    amplitude_integral = 0.0
    for fragment in fragments:
        if _positive_duration(fragment.source_start, edges[0]) or _positive_duration(
            edges[-1], fragment.source_end
        ):
            return float("nan")
        indices, durations, _starts, _ends = _cell_overlaps(edges, fragment)
        if indices.size == 0:
            continue
        selected_values = data[indices]
        if np.any(~np.isfinite(selected_values)):
            return float("nan")
        positive = selected_values > 0.0
        burst_duration += float(np.sum(durations[positive]))
        amplitude_integral += float(
            np.sum(selected_values[positive] * durations[positive])
        )
    if not _positive_duration(0.0, burst_duration):
        return float("nan")
    return amplitude_integral / burst_duration


def _display_bin_values(
    values: np.ndarray,
    source_edges: np.ndarray,
    fragments: Sequence[MappedFragment],
) -> np.ndarray:
    """Aggregate one display bin across flattened channel-band series."""
    output = np.full(values.shape[0], np.nan, dtype=float)
    index_parts: list[np.ndarray] = []
    duration_parts: list[np.ndarray] = []
    for fragment in fragments:
        if _positive_duration(
            fragment.source_start, source_edges[0]
        ) or _positive_duration(source_edges[-1], fragment.source_end):
            return output
        indices, durations, _starts, _ends = _cell_overlaps(source_edges, fragment)
        if indices.size:
            index_parts.append(indices)
            duration_parts.append(durations)
    if not index_parts:
        return output

    indices = np.concatenate(index_parts)
    durations = np.concatenate(duration_parts)
    selected = values[:, indices]
    finite = np.all(np.isfinite(selected), axis=1)
    positive = selected > 0.0
    burst_durations = np.sum(positive * durations[None, :], axis=1)
    amplitude_integrals = np.sum(
        np.where(positive, selected, 0.0) * durations[None, :],
        axis=1,
    )
    duration_tolerances = (
        _OVERLAP_EPS_FACTOR
        * np.finfo(np.float64).eps
        * np.maximum(1.0, np.abs(burst_durations))
    )
    accepted = finite & (burst_durations > duration_tolerances)
    output[accepted] = amplitude_integrals[accepted] / burst_durations[accepted]
    return output


def warp_burst_for_display(
    tensor: np.ndarray,
    source_times: Sequence[float],
    epochs: Sequence[Any],
    *,
    method: str,
    method_params: dict[str, Any],
    percent_axis: Sequence[float],
) -> np.ndarray:
    """Build a visualization-only aligned Burst tensor by target display bins."""
    values = np.asarray(tensor, dtype=float)
    if values.ndim != 3:
        raise ValueError("Burst source tensor must have shape (channel, band, time).")
    axis = np.asarray(percent_axis, dtype=float)
    bin_edges = sample_cell_edges(axis)
    bin_edges[0] = 0.0
    bin_edges[-1] = 100.0
    out = np.full(
        (len(epochs), values.shape[0], values.shape[1], axis.size),
        np.nan,
        dtype=float,
    )
    source_edges = sample_cell_edges(source_times)
    flattened = values.reshape(values.shape[0] * values.shape[1], values.shape[2])
    for epoch_index, epoch in enumerate(epochs):
        segments = build_warp_segments(method, method_params, epoch)
        for bin_index in range(axis.size):
            fragments = map_percent_intervals(
                segments, [(bin_edges[bin_index], bin_edges[bin_index + 1])]
            )
            out[epoch_index, :, :, bin_index] = _display_bin_values(
                flattened,
                source_edges,
                fragments,
            ).reshape(values.shape[0], values.shape[1])
    return out


__all__ = [
    "BurstScalarResult",
    "MappedFragment",
    "WarpSegment",
    "build_warp_segments",
    "burst_event_ids",
    "display_bin_value",
    "map_percent_intervals",
    "merge_percent_intervals",
    "reduce_burst_scalars",
    "sample_cell_edges",
    "warp_burst_for_display",
]
