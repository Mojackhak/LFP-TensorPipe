"""Resolve a Features phase table against the accepted aligned epoch axis."""

from __future__ import annotations

from bisect import bisect_left
from math import isfinite, ulp
from typing import Any

import pandas as pd

from lfptensorpipe.app.alignment.generation import latest_alignment_step_entry
from lfptensorpipe.app.alignment.paths import alignment_trial_raw_table_path, alignment_warp_labels_path
from lfptensorpipe.app.alignment.warper_builder import _resolve_target_duration_s
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.io.pkl_io import load_pkl


def trial_seconds_duration(resolver: PathResolver, slug: str) -> float | None:
    """Read the physical target duration only for methods with a seconds axis."""
    run = latest_alignment_step_entry(
        resolver, trial_slug=slug, step="run_align_epochs"
    )
    if run is None or not run[1].get("completed"):
        return None
    params = run[1]["params"]
    method = params.get("method")
    options = params.get("method_params")
    if not isinstance(options, dict):
        return None
    if method not in {"pad_warper", "concat_warper"} and not (
        method == "linear_warper" and "cue_response" in options
    ):
        return None
    if "target_duration_s" in params:
        return float(params["target_duration_s"])
    labels = {}
    if method == "concat_warper":
        path = alignment_warp_labels_path(resolver, slug)
        if not path.is_file():
            return None
        labels = load_pkl(path)
    return _resolve_target_duration_s(
        method=method, method_params=options, epochs_by_label=labels
    )


def phase_limit(unit: str, duration: float | None) -> float:
    if unit == "percent":
        return 100.0
    if unit != "s":
        raise ValueError(f"Unknown phase time unit: {unit!r}.")
    if duration is None or not isfinite(duration) or duration <= 0:
        raise ValueError(
            "Absolute phase time requires an accepted alignment with a seconds axis."
        )
    return duration


def validate_phase_bounds(rows, unit: str, duration: float | None) -> None:
    limit = phase_limit(unit, duration)
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Feature phase windows must be a list.")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("A phase must have a name, start and end.")
        try:
            lo, hi = float(row["start"]), float(row["end"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid phase bounds: {row!r}") from exc
        if not 0 <= lo < hi <= limit:
            raise ValueError(
                f"Phase {row.get('name', '')!r} [{lo:g}, {hi:g}] must be increasing within 0–{limit:g} {unit}."
            )


def convert_phase_rows(
    rows, source: str, target: str, duration: float | None
) -> list[dict]:
    if source == target:
        return [dict(row) for row in rows]
    old = phase_limit(source, duration)
    new = phase_limit(target, duration)
    return [
        {**row, "start": row["start"] / old * new, "end": row["end"] / old * new}
        for row in rows
    ]


def effective_phase_windows(axes, metric, interval_mode, support):
    """Interpret saved phase rows in the coordinate system used by extraction."""
    rows = axes["times"]
    unit = axes.get("time_unit")
    if unit is None:
        unit = "percent"
        if metric != "burst" and interval_mode == "absolute" and support is not None:
            unit = "s"
            origin = support[0]
            if origin < 0:
                rows = [
                    {**row, "start": row["start"] - origin, "end": row["end"] - origin}
                    for row in rows
                ]
    return {"time_unit": unit, "time_origin": "epoch_start", "times": rows}


def resolve_stored_phase_axes(resolver, slug, metric, axes, interval_mode):
    """Resolve unitless absolute windows using the Finish timeline that owned them."""
    if "time_unit" in axes or interval_mode != "absolute" or metric == "burst" or not slug:
        return axes
    duration = trial_seconds_duration(resolver, slug)
    if duration is None:
        return axes
    support = (0.0, duration)
    run = latest_alignment_step_entry(resolver, trial_slug=slug, step="run_align_epochs")[1]
    if "cue_response" in run["params"]["method_params"]:
        path = alignment_trial_raw_table_path(resolver, trial_slug=slug, metric_key=metric)
        if path.is_file():
            support = aligned_phase_support(load_pkl(path), "linear_warper") or support
    try:
        windows = effective_phase_windows(axes, metric, interval_mode, support)
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid stored phase window.") from exc
    return {**axes, "time_unit": windows["time_unit"], "times": windows["times"]}


def same_phase_boundary(a, b):
    return abs(a - b) <= 4 * max(ulp(a), ulp(b))


def same_phase_windows(a, b, duration: float | None) -> bool:
    left, right = a.get("times", []), b.get("times", [])
    left_unit, right_unit = a.get("time_unit", "percent"), b.get("time_unit", "percent")
    if len(left) != len(right):
        return False
    if left_unit != right_unit:
        if duration is None:
            return False
        left = convert_phase_rows(left, left_unit, right_unit, duration)
    return all(
        x["name"] == y["name"]
        and all(same_phase_boundary(x[k], y[k]) for k in ("start", "end"))
        for x, y in zip(left, right)
    )


def snap_phase_boundary(value, centers):
    """Use the exact sample coordinate when a converted window ends on a center."""
    index = bisect_left(centers, value)
    for center in centers[max(index - 1, 0) : index + 1]:
        if same_phase_boundary(float(center), value):
            return float(center)
    return value


def elapsed_intervals(intervals, duration, frame):
    """Keep half-open sample membership stable across percent/seconds round trips."""
    centers = next(
        (cell.columns for cell in frame["Value"] if isinstance(cell, pd.DataFrame)), ()
    )
    return {
        name: [
            [snap_phase_boundary(duration * p / 100.0, centers) for p in interval]
            for interval in ranges
        ]
        for name, ranges in intervals.items()
    }


def aligned_phase_support(frame, method):
    """Use the physical right edge, including one sample interval on older Finish tables."""
    if method not in {"pad_warper", "concat_warper", "linear_warper"}:
        return None
    support = frame.attrs.get("time_support_s")
    if support is not None:
        return tuple(map(float, support))
    for value in frame.get("Value", ()):
        if isinstance(value, pd.DataFrame) and len(value.columns) >= 2:
            start, last = float(value.columns[0]), float(value.columns[-1])
            if method == "linear_warper" and start >= 0:
                return None
            return start, last + (last - start) / (len(value.columns) - 1)
    return None


def elapsed_feature_frame(
    frame: pd.DataFrame, origin: float, duration: float
) -> pd.DataFrame:
    """Rebase nested time coordinates once when reading historical cue tables."""
    if origin >= 0:
        return frame

    def shifted(value: Any):
        if isinstance(value, pd.DataFrame):
            value = value.copy(deep=False)
            value.columns = value.columns - origin
        elif isinstance(value, pd.Series):
            value = value.copy(deep=False)
            value.index = value.index - origin
        return value

    result = frame.copy(deep=False)
    result["Value"] = frame["Value"].map(shifted)
    result.attrs["time_support_s"] = [0.0, duration]
    result.attrs["time_coordinate"] = "epoch_seconds"
    if "event_anchors_s" in frame.attrs:
        result.attrs["event_anchors_s"] = {
            name: value - origin
            for name, value in frame.attrs["event_anchors_s"].items()
        }
    return result


def elapsed_plot_frame(
    frame: pd.DataFrame, kind: str, resolver: PathResolver, slug: str
) -> pd.DataFrame:
    """Read previously extracted cue-relative raw and trace outputs for display."""
    if kind not in {"raw", "trace"} or "Value" not in frame:
        return frame
    if frame.attrs.get("time_coordinate") == "epoch_seconds":
        return frame
    support = frame.attrs.get("time_support_s")
    if support is not None and support[0] < 0:
        return elapsed_feature_frame(frame, support[0], support[1] - support[0])
    for cell in frame["Value"]:
        if isinstance(cell, (pd.DataFrame, pd.Series)):
            axis = cell.columns if isinstance(cell, pd.DataFrame) else cell.index
            if len(axis) < 1 or float(axis[0]) >= 0:
                return frame
            run = latest_alignment_step_entry(
                resolver, trial_slug=slug, step="run_align_epochs"
            )
            cue = (
                run[1]["params"].get("method_params", {}).get("cue_response")
                if run
                else None
            )
            if cue:
                duration = _resolve_target_duration_s(
                    method="linear_warper",
                    method_params=run[1]["params"]["method_params"],
                    epochs_by_label={},
                )
                return elapsed_feature_frame(frame, -cue["pre_cue_s"], duration)
            raise ValueError(
                "The source epoch for this historical time axis is unavailable. Recreate the selected trial's Features from its accepted alignment."
            )
    return frame
