"""Local, time-preserving gap and transient-peak interpolation."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
from scipy.interpolate import PchipInterpolator

from lfptensorpipe.lfp.mask.annotations import (
    annotation_sample_support_by_channel,
    valid_segments_from_annotation_support,
)
from lfptensorpipe.preproc.filter import _set_annotations_from_attached_frame


def default_signal_repair_params() -> dict[str, Any]:
    """Return independent, disabled-by-default repair settings."""
    return {
        "gaps": {
            "enabled": False,
            "method": "linear",
            "max_samples": 1,
            "context_samples": 2,
        },
        "peaks": {
            "enabled": False,
            "detection_method": "amplitude_mad",
            "background_window_s": 1.0,
            "guard_interval_ms": 10.0,
            "prediction_threshold": 6.0,
            "slope_threshold": 6.0,
            "method": "linear",
            "max_samples": 1,
            "context_samples": 2,
            "detection_window_s": 1.0,
            "baseline_window_s": 0.2,
            "mad_threshold": 8.0,
            "zscore_threshold": 3.0,
        },
    }


def normalize_signal_repair_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate consumed parameters once at the public/config boundary."""
    if not isinstance(params, dict):
        raise ValueError("Signal Repair parameters must be a mapping.")
    effective = {}
    for kind, defaults in default_signal_repair_params().items():
        supplied = params.get(kind, {})
        if not isinstance(supplied, dict):
            raise ValueError(f"{kind} parameters must be a mapping.")
        values = {**defaults, **supplied}
        if not isinstance(values["enabled"], bool):
            raise ValueError(f"{kind}.enabled must be boolean.")
        if not values["enabled"]:
            effective[kind] = {"enabled": False}
            continue
        method = values["method"]
        if method not in ("linear", "pchip"):
            raise ValueError(f"Unknown {kind} interpolation method: {method}")
        selected = {"enabled": True, "method": method}
        detector = values.get("detection_method", "amplitude_mad")
        if kind == "peaks":
            if detector not in ("amplitude_mad", "local_discontinuity", "local_zscore"):
                raise ValueError(f"Unknown peak detection method: {detector}")
            selected["detection_method"] = detector
        integer_keys = {"max_samples": 1}
        if kind == "peaks" and detector == "local_discontinuity":
            integer_keys = {}
            selected["max_samples"] = 1
        if method == "pchip":
            integer_keys["context_samples"] = 2
        for key, minimum in integer_keys.items():
            value = values[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{kind}.{key} must be an integer >= {minimum}.")
            selected[key] = int(value)
        if kind == "peaks":
            detector_keys = (
                (
                    "detection_window_s",
                    "baseline_window_s",
                    (
                        "zscore_threshold"
                        if detector == "local_zscore"
                        else "mad_threshold"
                    ),
                )
                if detector != "local_discontinuity"
                else (
                    "background_window_s",
                    "guard_interval_ms",
                    "prediction_threshold",
                    "slope_threshold",
                )
            )
            for key in detector_keys:
                value = values[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or (value < 0 if key == "guard_interval_ms" else value <= 0)
                ):
                    requirement = (
                        "nonnegative" if key == "guard_interval_ms" else "positive"
                    )
                    raise ValueError(f"{kind}.{key} must be finite and {requirement}.")
                selected[key] = float(value)
        effective[kind] = selected
    return effective


def _runs(mask: np.ndarray):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _peak_candidates(data, unavailable, sfreq, params, boundaries):
    # Bound temporary window arrays for long recordings; no cached signal state.
    width = max(3, int(round(params["detection_window_s"] * sfreq)))
    width += 1 - width % 2
    half = width // 2
    baseline_width = max(3, int(round(params["baseline_window_s"] * sfreq)))
    baseline_width += 1 - baseline_width % 2
    candidates = np.zeros(data.size, dtype=bool)
    zscore = params.get("detection_method") == "local_zscore"
    threshold = params["zscore_threshold" if zscore else "mad_threshold"]
    for left, right in valid_segments_from_annotation_support(unavailable, boundaries):
        if right - left < 3:
            continue
        values = data[left:right]
        windows_baseline = np.lib.stride_tricks.sliding_window_view(
            np.pad(values, baseline_width // 2, constant_values=np.nan), baseline_width
        )
        residual = np.empty(values.size)
        for offset in range(0, values.size, 4096):
            block = slice(offset, offset + 4096)
            residual[block] = values[block] - np.nanmedian(
                windows_baseline[block], axis=1
            )
        padded = np.pad(residual, (half, half), constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, width)
        for start in range(0, values.size, 4096):
            stop = min(start + 4096, values.size)
            chunk = windows[start:stop]
            if zscore:
                center = np.nanmean(chunk, axis=1)
                scale = np.nanstd(chunk, axis=1, ddof=1)
            else:
                # Only windows at the ends of a valid run contain NaN padding.
                median = (
                    np.median
                    if half <= start and stop <= values.size - half
                    else np.nanmedian
                )
                center = median(chunk, axis=1)
                scale = 1.4826 * median(np.abs(chunk - center[:, None]), axis=1)
            supported = (np.sum(np.isfinite(chunk), axis=1) >= 3) & (scale > 0)
            candidates[left + start : left + stop] = supported & (
                np.abs(residual[start:stop] - center) > threshold * scale
            )
    return candidates


def _discontinuity_candidates(data, unavailable, sfreq, params, boundaries):
    """Compare a sample's residual and two slopes to two independent backgrounds."""
    n = int(np.ceil(params["background_window_s"] * sfreq / 2))
    guard = int(np.ceil(params["guard_interval_ms"] * sfreq / 1000))
    candidates = np.zeros(data.size, dtype=bool)
    if n < 5:
        return candidates
    margin = n + guard
    shift = n + 2 * guard + 1
    cuts = sorted({0, data.size, *boundaries})
    for left, right in zip(cuts[:-1], cuts[1:]):
        for a, b in _runs(~unavailable[left:right]):
            start = left + a
            x = data[start : left + b]
            count = len(x) - 2 * margin
            if count <= 0:
                continue
            residual = x[1:-1] - (x[:-2] + x[2:]) / 2
            slope = np.diff(x) * sfreq
            error_windows = np.lib.stride_tricks.sliding_window_view(residual, n - 2)
            slope_windows = np.lib.stride_tricks.sliding_window_view(slope, n - 1)
            for offset in range(0, count, 4096):
                stop = min(offset + 4096, count)
                indices = np.arange(offset, stop) + margin
                entry, exit_slope = slope[indices - 1], slope[indices]
                eligible = entry * exit_slope < 0
                for displacement in (0, shift):
                    errors = error_windows[offset + displacement : stop + displacement]
                    slopes = slope_windows[offset + displacement : stop + displacement]
                    median_e = np.median(errors, axis=1)
                    median_d = np.median(slopes, axis=1)
                    scale_e = 1.4826 * np.median(
                        abs(errors - median_e[:, None]), axis=1
                    )
                    scale_d = 1.4826 * np.median(
                        abs(slopes - median_d[:, None]), axis=1
                    )
                    # Multiplication gives the same strict robust-z test without
                    # division by zero; zero scales explicitly reject the sample.
                    eligible &= (scale_e > 0) & (scale_d > 0)
                    eligible &= (
                        abs(residual[indices - 1] - median_e)
                        > params["prediction_threshold"] * scale_e
                    )
                    eligible &= (
                        abs(entry - median_d) > params["slope_threshold"] * scale_d
                    )
                    eligible &= (
                        abs(exit_slope - median_d) > params["slope_threshold"] * scale_d
                    )
                candidates[start + indices] = eligible
    return candidates


def _merge_gap_runs(
    gaps: list[tuple[int, int, int]],
) -> list[tuple[int, int, list[int]]]:
    """Merge sorted gap support and retain each contributing annotation index."""
    runs: list[tuple[int, int, list[int]]] = []
    for start, stop, annotation_index in gaps:
        if stop <= start:
            continue
        if runs and start <= runs[-1][1]:
            run_start, run_stop, members = runs[-1]
            members.append(annotation_index)
            runs[-1] = (run_start, max(run_stop, stop), members)
        else:
            runs.append((start, stop, [annotation_index]))
    return runs


def repair_signal(raw: mne.io.BaseRaw, params: dict[str, Any]):
    """Return a repaired copy and factual diagnostics; never mutate the input."""
    effective = normalize_signal_repair_params(params)
    if not any(group["enabled"] for group in effective.values()):
        raise ValueError("Enable gap or peak interpolation before Apply.")
    output = raw.copy().load_data()
    sfreq = float(raw.info["sfreq"])
    channels = tuple(raw.ch_names)
    bad_channels = set(raw.info["bads"])
    signal_types = {"eeg", "seeg", "ecog", "dbs"}
    eligible = {
        channel: index
        for index, (channel, channel_type) in enumerate(
            zip(channels, raw.get_channel_types())
        )
        if channel not in bad_channels and channel_type in signal_types
    }
    gaps_enabled = effective["gaps"]["enabled"]
    peaks_enabled = effective["peaks"]["enabled"]
    raw_annotations = raw.annotations
    labels = [label.strip().lower() for label in raw_annotations.description]
    gaps_by_channel: dict[str, list[tuple[int, int, int]]] = {
        channel: [] for channel in eligible
    }
    channels_by_gap = {}
    for index, (label, scope) in enumerate(zip(labels, raw_annotations.ch_names)):
        if label == "bad_gap":
            affected = eligible.keys() & scope if scope else eligible
            if affected:
                channels_by_gap[index] = affected
    if channels_by_gap:
        indices = list(channels_by_gap)
        onsets = raw_annotations.onset[indices] - raw.first_time
        times = np.column_stack((onsets, onsets + raw_annotations.duration[indices]))
        bounds = raw.time_as_index(times, use_rounding=True).clip(0, raw.n_times)
        for index, (start, stop) in sorted(zip(indices, bounds), key=lambda x: x[1][0]):
            gap = (int(start), int(stop), index)
            for channel in channels_by_gap[index]:
                gaps_by_channel[channel].append(gap)

    active_channels = tuple(
        channel
        for channel, gaps in gaps_by_channel.items()
        if peaks_enabled or any(start < stop for start, stop, _ in gaps)
    )
    mask_rows = {channel: row for row, channel in enumerate(active_channels)}
    protected_labels = {
        label
        for label in labels
        if label != "bad_gap" and label.startswith(("bad", "edge", "interpolated_"))
    }
    if active_channels and protected_labels:
        protected, protected_points, _ = annotation_sample_support_by_channel(
            raw, channels=active_channels, keep=tuple(protected_labels), mode="exact"
        )
    else:
        protected = np.broadcast_to(False, (len(active_channels), raw.n_times))
        protected_points = ((),) * len(active_channels)
    repaired_gap_channels: dict[int, set[str]] = {}
    annotations = []
    intervals = []
    for channel, channel_index in eligible.items():
        gaps = gaps_by_channel[channel]
        if gaps_enabled:
            for start, stop, _ in gaps:
                if start != stop:
                    continue
                intervals.append(
                    {
                        "channel": channel,
                        "kind": "gaps",
                        "start_sample": start,
                        "stop_sample": stop,
                        "method": effective["gaps"]["method"],
                        "source": "raw:BAD_gap",
                        "status": "skipped",
                        "reason": "empty_sample_support",
                    }
                )
        if channel not in mask_rows:
            continue
        x = output._data[channel_index]
        row = mask_rows[channel]
        protected_target = protected[row]
        invalid = protected_target.copy()
        gap_runs = _merge_gap_runs(gaps)
        for start, stop, _ in gap_runs:
            invalid[start:stop] = True
        points = sorted(
            set(protected_points[row]).union(
                start
                for start, _, index in gaps
                if raw_annotations.duration[index] == 0 and 0 < start < raw.n_times
            )
        )
        peak_runs = ()
        if peaks_enabled:
            valid = ~invalid
            np.isfinite(x, out=valid, where=valid)
            invalid = ~valid
            detector = (
                _discontinuity_candidates
                if effective["peaks"]["detection_method"] == "local_discontinuity"
                else _peak_candidates
            )
            peaks = detector(x, invalid, sfreq, effective["peaks"], points)
            invalid |= peaks
            peak_runs = ((int(start), int(stop), ()) for start, stop in _runs(peaks))
        for kind, runs in (
            ("gaps", gap_runs if gaps_enabled else ()),
            ("peaks", peak_runs),
        ):
            settings = effective[kind]
            for start, stop, gap_indices in runs:
                count = (
                    1 if settings["method"] == "linear" else settings["context_samples"]
                )
                left, right = start - count, stop + count
                record = {
                    "channel": channel,
                    "kind": kind,
                    "start_sample": start,
                    "stop_sample": stop,
                    "method": settings["method"],
                    "source": (
                        "raw:BAD_gap"
                        if kind == "gaps"
                        else (
                            "raw:local_discontinuity"
                            if settings["detection_method"] == "local_discontinuity"
                            else (
                                "raw:local_zscore"
                                if settings["detection_method"] == "local_zscore"
                                else "raw:local_mad"
                            )
                        )
                    ),
                    "support_start_sample": left,
                    "support_stop_sample": right,
                }
                reason = None
                if stop - start > settings["max_samples"]:
                    reason = "too_long"
                elif kind == "gaps" and protected_target[start:stop].any():
                    reason = "protected_target"
                elif left < 0 or right > raw.n_times:
                    reason = "missing_endpoint"
                elif (
                    invalid[left:start].any()
                    or invalid[stop:right].any()
                    or (
                        not peaks_enabled
                        and (
                            not np.isfinite(x[left:start]).all()
                            or not np.isfinite(x[stop:right]).all()
                        )
                    )
                ):
                    reason = "invalid_support"
                elif any(left < point < right for point in points):
                    reason = "point_boundary"
                elif (
                    kind == "peaks"
                    and (x[start] - x[start - 1]) * (x[stop] - x[stop - 1]) >= 0
                ):
                    reason = "not_transient"
                if reason is not None:
                    intervals.append({**record, "status": "skipped", "reason": reason})
                    continue
                anchors = np.r_[np.arange(left, start), np.arange(stop, right)]
                targets = np.arange(start, stop)
                if settings["method"] == "linear":
                    values = np.interp(targets, anchors, x[anchors])
                else:
                    values = PchipInterpolator(anchors, x[anchors], extrapolate=False)(
                        targets
                    )
                # Anchors exclude every candidate, and repair targets are disjoint.
                original_samples = x[start:stop].tolist()
                x[start:stop] = values
                label = "INTERPOLATED_gap" if kind == "gaps" else "INTERPOLATED_peak"
                annotations.append(
                    (
                        raw.first_time + start / sfreq,
                        (stop - start) / sfreq,
                        label,
                        (channel,),
                    )
                )
                gap_annotations = []
                for index in gap_indices:
                    repaired_gap_channels.setdefault(index, set()).add(channel)
                    gap_annotations.append(
                        [
                            float(raw_annotations.onset[index]),
                            float(raw_annotations.duration[index]),
                            str(raw_annotations.description[index]),
                            [channel],
                        ]
                    )
                intervals.append(
                    {
                        **record,
                        "status": "repaired",
                        "reason": None,
                        "accepted": True,
                        "original_samples": original_samples,
                        "interpolated_samples": values.tolist(),
                        "gap_annotations": gap_annotations,
                    }
                )

    if annotations:
        for index, (onset, duration, description, scope) in enumerate(
            zip(
                raw_annotations.onset,
                raw_annotations.duration,
                raw_annotations.description,
                raw_annotations.ch_names,
            )
        ):
            repaired = repaired_gap_channels.get(index)
            if repaired:
                annotations.extend(
                    (onset, duration, description, (channel,))
                    for channel in (scope or channels)
                    if channel not in repaired
                )
            else:
                annotations.append((onset, duration, description, scope))
        combined = mne.Annotations(
            onset=[a[0] for a in annotations],
            duration=[a[1] for a in annotations],
            description=[a[2] for a in annotations],
            ch_names=[a[3] for a in annotations],
            orig_time=raw_annotations.orig_time,
        )
        _set_annotations_from_attached_frame(output, combined)
    return output, {
        "effective_params": effective,
        "intervals": intervals,
        "summary": summarize_signal_repair(intervals),
    }


def summarize_signal_repair(intervals):
    """Count current accepted repairs and retained skipped candidates."""
    summary = {}
    for kind in ("gaps", "peaks"):
        repaired = [
            item
            for item in intervals
            if item["kind"] == kind
            and item["status"] == "repaired"
            and item.get("accepted", True)
        ]
        summary[kind] = {
            "repaired_intervals": len(repaired),
            "repaired_samples": sum(
                item["stop_sample"] - item["start_sample"] for item in repaired
            ),
            "skipped_intervals": sum(
                item["kind"] == kind and item["status"] == "skipped"
                for item in intervals
            ),
        }
    return summary
