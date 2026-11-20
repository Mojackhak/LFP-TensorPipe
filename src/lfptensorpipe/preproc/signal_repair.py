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


def repair_signal(raw: mne.io.BaseRaw, params: dict[str, Any]):
    """Return a repaired copy and factual diagnostics; never mutate the input."""
    effective = normalize_signal_repair_params(params)
    if not any(group["enabled"] for group in effective.values()):
        raise ValueError("Enable gap or peak interpolation before Apply.")
    output = raw.copy().load_data()
    sfreq = float(raw.info["sfreq"])
    channels = tuple(raw.ch_names)
    gaps, gap_points, _ = annotation_sample_support_by_channel(
        raw, channels=channels, keep=("BAD_gap",), mode="exact"
    )
    labels = [label.strip().lower() for label in raw.annotations.description]
    protected_labels = tuple(
        label
        for label in set(labels)
        if label != "bad_gap" and label.startswith(("bad", "edge", "interpolated_"))
    )
    if protected_labels:
        protected, protected_points, _ = annotation_sample_support_by_channel(
            raw, channels=channels, keep=protected_labels, mode="exact"
        )
    else:
        protected = np.zeros_like(gaps)
        protected_points = ((),) * len(channels)
    gap_bounds = {}
    if effective["gaps"]["enabled"] and "bad_gap" in labels:
        indices = [index for index, label in enumerate(labels) if label == "bad_gap"]
        onsets = raw.annotations.onset[indices] - raw.first_time
        times = np.column_stack((onsets, onsets + raw.annotations.duration[indices]))
        gap_bounds = dict(zip(indices, raw.time_as_index(times, use_rounding=True)))
    repaired_gaps = np.zeros_like(gaps)
    annotations = []
    intervals = []
    signal_types = {"eeg", "seeg", "ecog", "dbs"}
    for channel_index, (channel, channel_type) in enumerate(
        zip(channels, raw.get_channel_types())
    ):
        if channel in raw.info["bads"] or channel_type not in signal_types:
            continue
        x = output._data[channel_index].copy()
        protected_target = protected[channel_index]
        for annotation_index, (start, stop) in gap_bounds.items():
            scope = raw.annotations.ch_names[annotation_index]
            if start == stop and (not scope or channel in scope):
                intervals.append(
                    {
                        "channel": channel,
                        "kind": "gaps",
                        "start_sample": int(start),
                        "stop_sample": int(stop),
                        "method": effective["gaps"]["method"],
                        "source": "raw:BAD_gap",
                        "status": "skipped",
                        "reason": "empty_sample_support",
                    }
                )
        valid = ~(protected_target | gaps[channel_index])
        np.isfinite(x, out=valid, where=valid)
        invalid = ~valid
        points = sorted(
            set(gap_points[channel_index]).union(protected_points[channel_index])
        )
        peaks = np.zeros(raw.n_times, dtype=bool)
        if effective["peaks"]["enabled"]:
            detector = (
                _discontinuity_candidates
                if effective["peaks"]["detection_method"] == "local_discontinuity"
                else _peak_candidates
            )
            peaks = detector(x, invalid, sfreq, effective["peaks"], points)
        support_invalid = invalid | peaks
        for kind, mask in (("gaps", gaps[channel_index]), ("peaks", peaks)):
            settings = effective[kind]
            if not settings["enabled"]:
                continue
            for start, stop in _runs(mask):
                start, stop = int(start), int(stop)
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
                elif np.any(support_invalid[left:start]) or np.any(
                    support_invalid[stop:right]
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
                output._data[channel_index, start:stop] = values
                if kind == "gaps":
                    repaired_gaps[channel_index, start:stop] = True
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
                if kind == "gaps":
                    for ai in gap_bounds:
                        scope = raw.annotations.ch_names[ai]
                        if scope and channel not in scope:
                            continue
                        ann = raw.annotations[ai]
                        a = max(raw.first_time + start / sfreq, ann["onset"])
                        b = min(
                            raw.first_time + stop / sfreq,
                            ann["onset"] + ann["duration"],
                        )
                        if b > a:
                            gap_annotations.append(
                                [a, b - a, ann["description"], [channel]]
                            )
                intervals.append(
                    {
                        **record,
                        "status": "repaired",
                        "reason": None,
                        "accepted": True,
                        "original_samples": x[start:stop].tolist(),
                        "interpolated_samples": values.tolist(),
                        "gap_annotations": gap_annotations,
                    }
                )

    # Subtract only repaired time/channel support from the originating gap label.
    for ann_index, ann in enumerate(raw.annotations):
        onset, duration, description = ann["onset"], ann["duration"], ann["description"]
        scope = tuple(raw.annotations.ch_names[ann_index])
        if ann_index not in gap_bounds:
            annotations.append((onset, duration, description, scope))
            continue
        start, stop = np.clip(gap_bounds[ann_index], 0, raw.n_times)
        affected = [channels.index(ch) for ch in (scope or channels)]
        if not np.any(repaired_gaps[affected, start:stop]):
            annotations.append((onset, duration, description, scope))
            continue
        for index in affected:
            for a, b in _runs(~repaired_gaps[index, start:stop]):
                annotations.append(
                    (
                        raw.first_time + (start + int(a)) / sfreq,
                        (int(b) - int(a)) / sfreq,
                        description,
                        (channels[index],),
                    )
                )
    combined = mne.Annotations(
        onset=[a[0] for a in annotations],
        duration=[a[1] for a in annotations],
        description=[a[2] for a in annotations],
        ch_names=[a[3] for a in annotations],
        orig_time=raw.annotations.orig_time,
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
