"""grid.py

Gridding utilities for nested scalograms/traces stored inside a summary pandas DataFrame.

Typical input: a "summary" DataFrame where each row is one recording/epoch and the
column ``value_col`` (default: "Value") stores either:

- 2D scalogram: ``pd.DataFrame`` (index=freq, columns=time)
- 1D trace: ``pd.Series`` (index=time or freq)

The functions here build a user-defined grid (frequency wells and/or time wells)
and reduce each grid cell to a representative scalar.

Time/Frequency interval specifications
--------------------------------------

This module only supports numeric interval inputs in the following shape:

- list/array of (list/array/tuple of length-2 numeric), e.g.::

    [[0.0, 1.0]]
    [[1.5, 2.5], [3.0, 4.0]]

Whether the numeric interval endpoints represent absolute axis coordinates or percent-of-axis-span
is controlled by API parameters:

- ``time_interval_mode`` in :func:`grid_nested_values` / :func:`split_nested_values`
- ``freq_interval_mode`` in :func:`grid_nested_values` / :func:`split_nested_values`

For ``*_interval_mode="percent"``:
- If both endpoints are in [-1, 1], they are interpreted as fractions (0..1).
- Otherwise they are interpreted as percentages (0..100).
They are converted to axis coordinates using the span between axis[0] and axis[-1].

Label-group specifications
--------------------------

In addition to numeric intervals, you can specify wells by explicitly listing axis labels:

- For ``BandDefinition`` you may pass a list/array of (list/array of str), where each
  inner list defines one frequency well as a set of index labels. Example::

      bands = [["beta_low", "beta_high"], ["gamma"]]

  The first well selects rows whose index labels match "beta_low" or "beta_high".

- For ``TimeWells`` (leaf values under ``phase``) you may pass a list/array of
  (list/array of str) to select columns by label. All labels across inner lists
  are unioned to form the time selection mask.

Reducer kinds
-------------

- mean: trapezoid-integrated mean (only finite values are used)
- median: median of finite values
- count: number of contiguous finite segments inside the selection
- occupation: finite-time proportion (percent) inside the selection
- rate: count / total selection duration
- duration: mean duration of contiguous finite segments

Notes
-----

- If an axis is numeric-like (fully convertible to float), we treat it as numeric and
  assume it is increasing (input convention). We still sort for safety.
- If an axis is not numeric-like, we keep it as categorical and use positional
  coordinates (0..n-1) for distance-based computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

# -----------------------
# Public type aliases
# -----------------------
Interval = tuple[float, float]
IntervalsLike = Sequence[
    Sequence[float]
]  # outer sequence of length-2 numeric sequences

LabelGroup = Sequence[str]
LabelGroups = Sequence[LabelGroup]

TimeWellSpec = IntervalsLike | LabelGroups
BandWellSpec = IntervalsLike | LabelGroups

BandDefinition = Mapping[str, BandWellSpec] | LabelGroups
TimeWells = Mapping[str, TimeWellSpec]
# phase -> time well spec

SeriesAxis = Literal["time", "freq"]
AxisIntervalMode = Literal["absolute", "percent"]
OCCUPATION_SCALE = 1.0  # proportion; set to 100.0 for percent


class ReducerKind(str, Enum):
    """Supported reduction strategies for each grid cell."""

    MEAN = "mean"  # trapezoid-integrated mean
    MEDIAN = "median"

    COUNT = "count"
    OCCUPATION = "occupation"
    RATE = "rate"
    DURATION = "duration"  # mean duration of contiguous valid segments


@dataclass(frozen=True, slots=True)
class GridResultColumns:
    band: str = "Band"
    phase: str = "Phase"
    value: str = "Value_grid"


@dataclass(frozen=True, slots=True)
class AxisInfo:
    """Axis metadata.

    Attributes
    ----------
    labels:
        Original axis labels (after possible numeric coercion + sorting).
    coords:
        Numeric coordinate array used for distance-based computations.
        If the axis is not numeric-like, coords are positional (0..n-1).
    is_numeric:
        Whether the axis labels were fully convertible to float.
    """

    labels: pd.Index
    coords: np.ndarray
    is_numeric: bool


@dataclass(frozen=True, slots=True)
class AxisSelection:
    """Resolved axis selection as a boolean mask plus total requested length."""

    mask: np.ndarray
    total_length: float


def _trapezoid(y: np.ndarray, *, x: np.ndarray) -> float:
    """Trapezoidal integration via the NumPy 2.x API."""
    return float(np.trapezoid(y, x=x))


# -----------------------
# Axis preparation helpers
# -----------------------
def _try_numeric_axis(idx: pd.Index) -> np.ndarray | None:
    """Return numeric coordinates if the entire axis is convertible to float, else None."""
    arr = pd.to_numeric(idx, errors="coerce")
    # Treat axis as numeric only if *all* labels are finite numbers.
    if arr.isna().any():
        return None
    coords = arr.to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        return None
    return coords


def _make_axis_info(idx: pd.Index) -> AxisInfo:
    coords = _try_numeric_axis(idx)
    if coords is None:
        n = int(len(idx))
        return AxisInfo(labels=idx, coords=np.arange(n, dtype=float), is_numeric=False)
    return AxisInfo(
        labels=pd.Index(coords),
        coords=coords.astype(float, copy=False),
        is_numeric=True,
    )


def _sort_axis(
    df: pd.DataFrame, axis: int, info: AxisInfo
) -> tuple[pd.DataFrame, AxisInfo]:
    """Sort DataFrame along an axis if it is numeric; otherwise keep original order."""
    if not info.is_numeric:
        return df, info

    order = np.argsort(info.coords)
    if axis == 0:
        df_sorted = df.iloc[order, :]
        new_labels = df_sorted.index
    else:
        df_sorted = df.iloc[:, order]
        new_labels = df_sorted.columns

    new_info = _make_axis_info(new_labels)
    return df_sorted, new_info


def prepare_scalogram(df: pd.DataFrame) -> tuple[pd.DataFrame, AxisInfo, AxisInfo]:
    """Coerce scalogram values to float, preserve NaN gaps, and prepare axis metadata."""
    out = df.copy()

    # Values: coerce everything to float; non-numeric becomes NaN.
    out = out.apply(pd.to_numeric, errors="coerce")

    f_info = _make_axis_info(out.index)
    t_info = _make_axis_info(out.columns)

    # Sort only numeric axes; keep categorical order as-is.
    out, f_info = _sort_axis(out, axis=0, info=f_info)
    out, t_info = _sort_axis(out, axis=1, info=t_info)

    if out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError("Scalogram has empty frequency or time axis.")

    return out, f_info, t_info


def prepare_series(s: pd.Series) -> tuple[pd.Series, AxisInfo]:
    """Coerce Series values to float and prepare axis metadata."""
    out = s.copy()

    coords = _try_numeric_axis(out.index)
    if coords is not None:
        out.index = pd.Index(coords)
        out = out.sort_index()

    out = pd.to_numeric(out, errors="coerce")

    info = _make_axis_info(out.index)
    if info.is_numeric:
        order = np.argsort(info.coords)
        out = out.iloc[order]
        info = _make_axis_info(out.index)

    if out.shape[0] == 0:
        raise ValueError("Series has empty axis.")
    return out, info


# -----------------------
# Interval handling (numeric only)
# -----------------------
def _axis_span(coords: np.ndarray) -> tuple[float, float, float]:
    lo = float(coords[0])
    hi = float(coords[-1])
    return lo, hi, float(hi - lo)


def _percent_to_axis(p0: float, p1: float, coords: np.ndarray) -> tuple[float, float]:
    """Convert percent/fraction endpoints to axis coordinates."""
    lo, _, span = _axis_span(coords)
    if span == 0.0:
        return lo, lo

    # Support both 0-100 (%) and 0-1 (fraction) inputs.
    if max(abs(p0), abs(p1)) <= 1.0:
        f0, f1 = p0, p1
    else:
        f0, f1 = p0 / 100.0, p1 / 100.0

    a = lo + f0 * span
    b = lo + f1 * span
    return (a, b) if a <= b else (b, a)


def _clip_interval_to_axis(it: Interval, coords: np.ndarray) -> Interval | None:
    lo = float(coords[0])
    hi = float(coords[-1])
    a, b = it
    if a > b:
        a, b = b, a
    a = max(a, lo)
    b = min(b, hi)
    if b < a:
        return None
    return float(a), float(b)


def _merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    if not intervals:
        return []
    xs = sorted(
        (float(a), float(b)) if a <= b else (float(b), float(a)) for a, b in intervals
    )
    merged: list[Interval] = []
    cur_a, cur_b = xs[0]
    for a, b in xs[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    return merged


def _is_interval_pairs(obj: Any) -> bool:
    """True if obj looks like Sequence[Sequence[number]] where inner sequences have len==2."""
    if obj is None or isinstance(obj, str):
        return False
    if not isinstance(obj, Sequence):
        return False

    # We require the outer container to contain inner 2-lists/tuples/arrays.
    for it in obj:
        if it is None or isinstance(it, str):
            return False
        if not isinstance(it, Sequence):
            return False
        if len(it) != 2:
            return False
        a, b = it[0], it[1]
        if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
            return False
        if isinstance(a, Sequence) or isinstance(b, Sequence):
            # Disallow nested nesting like [[ [a], [b] ]].
            return False
        if not isinstance(a, (int, float, np.number)) or not isinstance(
            b, (int, float, np.number)
        ):
            return False
    return True


def resolve_intervals(
    intervals: IntervalsLike,
    coords: np.ndarray,
    *,
    mode: AxisIntervalMode,
) -> list[Interval]:
    """Validate, convert (percent->absolute if needed), clip, and merge intervals."""
    if not _is_interval_pairs(intervals):
        raise TypeError(
            "Intervals must be a list/array of length-2 numeric sequences, e.g. [[0, 1], [2, 3]]."
        )

    out: list[Interval] = []
    hi = float(coords[-1])
    for i, it in enumerate(intervals):
        if len(it) != 2:
            raise ValueError(f"Interval #{i} must have length 2, got: {it!r}")

        a = float(it[0])
        b = float(it[1])
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError(f"Interval endpoints must be finite numbers, got: {it!r}")

        if mode == "percent":
            a2, b2 = _percent_to_axis(a, b, coords)
        else:
            a2, b2 = (a, b) if a <= b else (b, a)

        clipped = _clip_interval_to_axis((a2, b2), coords)
        if clipped is not None:
            left, right = clipped
            # If the interval right edge was cropped by axis max, keep that endpoint
            # effectively inclusive under right-open masking (`x < end`) by nudging
            # the clipped upper bound to the next representable float.
            if b2 > hi and np.isclose(right, hi, rtol=0.0, atol=0.0):
                right = float(np.nextafter(hi, np.inf))
            # Left edge uses `x >= start` in both modes; no extra handling needed.
            out.append((left, right))

    return _merge_intervals(out)


def intervals_total_length(intervals: Sequence[Interval]) -> float:
    return float(sum(max(0.0, b - a) for a, b in intervals))


def mask_interval(
    coords: np.ndarray, start: float, end: float, inclusive: bool
) -> np.ndarray:
    return (
        (coords >= start) & (coords <= end)
        if inclusive
        else (coords >= start) & (coords < end)
    )


def mask_union_intervals(
    coords: np.ndarray, intervals: Sequence[Interval], inclusive: bool
) -> np.ndarray:
    mask = np.zeros(coords.shape[0], dtype=bool)
    for a, b in intervals:
        mask |= mask_interval(coords, a, b, inclusive=inclusive)
    return mask


# -----------------------
# Label-group parsing
# -----------------------
def _is_label_groups(obj: Any) -> bool:
    """True if obj looks like Sequence[Sequence[str]] (and not a single string)."""
    if obj is None or isinstance(obj, str):
        return False
    if not isinstance(obj, Sequence):
        return False
    for group in obj:
        if group is None or isinstance(group, str):
            return False
        if not isinstance(group, Sequence):
            return False
        for lab in group:
            if not isinstance(lab, (str, np.str_)):
                return False
    return True


def _flatten_label_groups(groups: LabelGroups) -> list[str]:
    out: list[str] = []
    for g in groups:
        for lab in g:
            out.append(str(lab))
    return out


def _mask_from_labels(axis: AxisInfo, labels: Sequence[str]) -> np.ndarray:
    """Build a boolean mask by matching provided string labels to axis labels."""
    n = int(len(axis.labels))
    if n == 0:
        return np.zeros(0, dtype=bool)

    axis_str = np.asarray([str(x) for x in axis.labels], dtype=object)
    mask = np.zeros(n, dtype=bool)

    for lab in labels:
        lab_s = str(lab)
        hit = axis_str == lab_s
        if hit.any():
            mask |= hit
            continue

        if axis.is_numeric:
            try:
                v = float(lab_s)
            except Exception:
                continue
            if np.isfinite(v):
                mask |= np.isclose(axis.coords, v, rtol=1e-9, atol=1e-12)

    return mask


def _mask_span_length(coords: np.ndarray, mask: np.ndarray) -> float:
    """Total span length covered by True-runs of mask on the coordinate axis."""
    runs = _true_runs(mask)
    total = 0.0
    for s, e in runs:
        if e > s:
            total += float(coords[e] - coords[s])
    return float(total)


def parse_axis_selection(
    spec: TimeWellSpec | BandWellSpec,
    axis: AxisInfo,
    *,
    inclusive: bool,
    interval_mode: AxisIntervalMode,
) -> AxisSelection:
    """Parse either an interval-based spec or label-group spec to a selection mask."""
    if _is_label_groups(spec):
        labels = _flatten_label_groups(spec)  # type: ignore[arg-type]
        mask = _mask_from_labels(axis, labels)
        total_len = _mask_span_length(axis.coords, mask)
        return AxisSelection(mask=mask, total_length=total_len)

    if not _is_interval_pairs(spec):
        raise TypeError(
            "Well spec must be either label-groups (list of list of str) "
            "or numeric intervals (list of [start, end])."
        )

    intervals = resolve_intervals(spec, axis.coords, mode=interval_mode)  # type: ignore[arg-type]
    mask = mask_union_intervals(axis.coords, intervals, inclusive=inclusive)
    total_len = intervals_total_length(intervals)
    return AxisSelection(mask=mask, total_length=total_len)


def _make_well_name(labels: Sequence[str]) -> str:
    if not labels:
        return "empty"
    if len(labels) == 1:
        return str(labels[0])
    return "+".join(str(x) for x in labels)


def parse_band_definitions(
    bands: BandDefinition,
    freq_axis: AxisInfo,
    *,
    inclusive: bool,
    interval_mode: AxisIntervalMode,
) -> list[tuple[str, AxisSelection]]:
    """Normalize the bands input into a list of (band_name, freq_selection)."""
    out: list[tuple[str, AxisSelection]] = []

    if isinstance(bands, Mapping):
        for name, spec in bands.items():
            sel = parse_axis_selection(
                spec, freq_axis, inclusive=inclusive, interval_mode=interval_mode
            )
            out.append((str(name), sel))
        return out

    if not _is_label_groups(bands):
        raise TypeError(
            "`bands` must be either a Mapping[str, spec] or a list/array of (list/array of str)."
        )

    used: dict[str, int] = {}
    for group in bands:  # type: ignore[assignment]
        group_labels = [str(x) for x in group]
        base = _make_well_name(group_labels)
        count = used.get(base, 0)
        used[base] = count + 1
        name = base if count == 0 else f"{base}_{count+1}"

        sel = parse_axis_selection(
            [group_labels], freq_axis, inclusive=inclusive, interval_mode=interval_mode
        )
        out.append((name, sel))

    return out


# -----------------------
# Segment helpers
# -----------------------
def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) inclusive for each contiguous True run."""
    if mask.size == 0:
        return []

    m = mask.astype(bool, copy=False)
    if not m.any():
        return []

    d = np.diff(m.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0]

    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size - 1]

    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=False)]


def _count_and_duration(
    coords: np.ndarray, valid_mask: np.ndarray
) -> tuple[int, float]:
    """Count contiguous valid runs and return their total duration on coordinate axis."""
    runs = _true_runs(valid_mask)
    total = 0.0
    for s, e in runs:
        if e > s:
            total += float(coords[e] - coords[s])
    return len(runs), float(total)


def _integrate_over_valid_segments(
    coords: np.ndarray, y: np.ndarray, valid_mask: np.ndarray
) -> tuple[float, float]:
    """Integrate y over coords within each contiguous valid segment.

    Returns
    -------
    integral_y, integral_one
        Approximations of ∫ y dx and ∫ 1 dx on the valid segments.

    Notes
    -----
    Integration is performed per contiguous valid segment to avoid bridging across NaN/Inf gaps.
    """
    runs = _true_runs(valid_mask)
    if not runs:
        return 0.0, 0.0

    num = 0.0
    den = 0.0
    for s, e in runs:
        xv = coords[s : e + 1]
        yv = y[s : e + 1]
        if xv.size == 1:
            num += float(yv[0])
            den += 1.0
        else:
            num += _trapezoid(yv, x=xv)
            den += float(xv[-1] - xv[0])
    return float(num), float(den)


# -----------------------
# Scalar reducers
# -----------------------
def _coerce_reducer_kind(reducer: ReducerKind | str) -> ReducerKind:
    if isinstance(reducer, ReducerKind):
        return reducer
    if not isinstance(reducer, str):
        raise TypeError("reducer must be a ReducerKind or str")
    return ReducerKind(reducer.strip().lower())


def _finite_values(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _reduce_1d_with_mask(
    axis: AxisInfo, y: np.ndarray, sel: AxisSelection, *, reducer: ReducerKind
) -> float:
    """Reduce a 1D vector y over a selection mask."""
    if y.shape[0] != sel.mask.shape[0]:
        raise ValueError("Shape mismatch between values and selection mask.")

    finite = np.isfinite(y)
    valid = sel.mask & finite

    if reducer is ReducerKind.MEDIAN:
        vals = _finite_values(y[sel.mask])
        return float(np.median(vals)) if vals.size else np.nan

    if reducer is ReducerKind.MEAN:
        if axis.is_numeric:
            num, den = _integrate_over_valid_segments(axis.coords, y, valid)
            return np.nan if den == 0.0 else float(num / den)
        # Categorical axis: fall back to arithmetic mean over finite points in selection.
        vals = y[valid]
        return float(np.mean(vals)) if vals.size else np.nan

    count, dur = _count_and_duration(axis.coords, valid)

    if reducer is ReducerKind.COUNT:
        return float(count)
    if reducer is ReducerKind.OCCUPATION:
        return (
            np.nan
            if sel.total_length == 0.0
            else float(dur / sel.total_length * OCCUPATION_SCALE)
        )  # as percent
    if reducer is ReducerKind.RATE:
        return np.nan if sel.total_length == 0.0 else float(count / sel.total_length)
    if reducer is ReducerKind.DURATION:
        return np.nan if count == 0 else float(dur / count)

    raise ValueError(f"Unsupported reducer: {reducer!r}")


def reduce_1d(
    s_sorted: pd.Series,
    axis: AxisInfo,
    sel: AxisSelection,
    *,
    reducer: ReducerKind,
) -> float:
    """Reduce 1D values over an axis selection."""
    y = s_sorted.to_numpy(dtype=float, copy=False)
    return _reduce_1d_with_mask(axis, y, sel, reducer=reducer)


def reduce_2d(
    df_sorted: pd.DataFrame,
    freq_axis: AxisInfo,
    time_axis: AxisInfo,
    f_sel: AxisSelection,
    t_sel: AxisSelection,
    *,
    reducer: ReducerKind,
) -> float:
    """Reduce 2D scalogram over a frequency selection and a time selection."""
    if not f_sel.mask.any() or not t_sel.mask.any():
        return np.nan

    Z = df_sorted.to_numpy(dtype=float, copy=False)

    if reducer is ReducerKind.MEDIAN:
        vals = Z[np.ix_(f_sel.mask, t_sel.mask)].reshape(-1)
        vals = _finite_values(vals)
        return float(np.median(vals)) if vals.size else np.nan

    if reducer in {
        ReducerKind.COUNT,
        ReducerKind.OCCUPATION,
        ReducerKind.RATE,
        ReducerKind.DURATION,
    }:
        any_finite = np.any(np.isfinite(Z[f_sel.mask, :]), axis=0)
        valid_time = t_sel.mask & any_finite

        count, dur = _count_and_duration(time_axis.coords, valid_time)

        if reducer is ReducerKind.COUNT:
            return float(count)
        if reducer is ReducerKind.OCCUPATION:
            return (
                np.nan
                if t_sel.total_length == 0.0
                else float(dur / t_sel.total_length * OCCUPATION_SCALE)
            )  # as percent
        if reducer is ReducerKind.RATE:
            return (
                np.nan
                if t_sel.total_length == 0.0
                else float(count / t_sel.total_length)
            )
        if reducer is ReducerKind.DURATION:
            return np.nan if count == 0 else float(dur / count)

    if reducer is not ReducerKind.MEAN:
        raise ValueError(f"Unsupported reducer: {reducer!r}")

    # Trapezoid-integrated mean over a 2D region:
    #   mean = (∫∫ Z df dt) / (∫∫ 1 df dt)
    # We compute:
    #   For each time point: integrate across frequency selection -> col_num, col_den
    #   Then integrate col_num and col_den across time selection.

    n_t = time_axis.coords.size
    col_num = np.full(n_t, np.nan, dtype=float)
    col_den = np.full(n_t, np.nan, dtype=float)

    t_idxs = np.where(t_sel.mask)[0]
    for tj in t_idxs:
        z_col = Z[:, tj]

        valid_f = f_sel.mask & np.isfinite(z_col)
        if not valid_f.any():
            continue

        if freq_axis.is_numeric:
            num_f, den_f = _integrate_over_valid_segments(
                freq_axis.coords, z_col, valid_f
            )
        else:
            num_f = float(np.sum(z_col[valid_f]))
            den_f = float(np.sum(valid_f))

        if den_f <= 0.0 or not np.isfinite(num_f) or not np.isfinite(den_f):
            continue

        col_num[tj] = num_f
        col_den[tj] = den_f

    valid_t = t_sel.mask & np.isfinite(col_den) & (col_den > 0.0)
    if not valid_t.any():
        return np.nan

    if time_axis.is_numeric:
        total_num, _ = _integrate_over_valid_segments(
            time_axis.coords, col_num, valid_t & np.isfinite(col_num)
        )
        total_den, _ = _integrate_over_valid_segments(
            time_axis.coords, col_den, valid_t
        )
    else:
        total_num = float(np.sum(col_num[valid_t & np.isfinite(col_num)]))
        total_den = float(np.sum(col_den[valid_t]))

    return np.nan if total_den == 0.0 else float(total_num / total_den)


# -----------------------
# Series reducers (for split)
# -----------------------
def reduce_time_selection_to_series_by_freq(
    df_sorted: pd.DataFrame,
    freq_axis: AxisInfo,
    time_axis: AxisInfo,
    t_sel: AxisSelection,
    *,
    reducer: ReducerKind,
) -> pd.Series:
    """For each frequency row, reduce across a time selection -> Series(index=freq labels)."""
    Z = df_sorted.to_numpy(dtype=float, copy=False)

    out = np.full(Z.shape[0], np.nan, dtype=float)
    for fi in range(Z.shape[0]):
        y = Z[fi, :]
        out[fi] = _reduce_1d_with_mask(time_axis, y, t_sel, reducer=reducer)

    return pd.Series(out, index=df_sorted.index, dtype=float)


def reduce_freq_selection_to_series_by_time(
    df_sorted: pd.DataFrame,
    freq_axis: AxisInfo,
    time_axis: AxisInfo,
    f_sel: AxisSelection,
    *,
    reducer: ReducerKind,
) -> pd.Series:
    """For each time point, reduce across a frequency selection -> Series(index=time labels)."""
    Z = df_sorted.to_numpy(dtype=float, copy=False)

    out = np.full(Z.shape[1], np.nan, dtype=float)
    for tj in range(Z.shape[1]):
        y = Z[:, tj]
        out[tj] = _reduce_1d_with_mask(freq_axis, y, f_sel, reducer=reducer)

    return pd.Series(out, index=df_sorted.columns, dtype=float)


# -----------------------
# Public API
# -----------------------
def grid_nested_values(
    summary_df: pd.DataFrame,
    *,
    value_col: str = "Value",
    bands: BandDefinition | None = None,
    times: TimeWells | None = None,
    axis: SeriesAxis = "time",
    reducer: ReducerKind | str = ReducerKind.MEAN,
    time_interval_mode: AxisIntervalMode = "absolute",
    freq_interval_mode: AxisIntervalMode = "absolute",
    inclusive: bool = True,
    drop_value: bool = True,
    keep_full_dim_cols: bool = False,
    on_missing: Literal["raise", "skip"] = "skip",
    out_cols: GridResultColumns = GridResultColumns(),
) -> pd.DataFrame:
    """Expand each row of ``summary_df`` into a long-form grid with a scalar per cell.

    2D payload (DataFrame): requires ``bands`` and ``times``.
    1D payload (Series):
      - axis="time": requires ``times``
      - axis="freq": requires ``bands``

    Parameters
    ----------
    time_interval_mode:
        How to interpret numeric time intervals in ``times``:
        - "absolute": endpoints are axis coordinates
        - "percent": endpoints are percent/fraction of axis span
    freq_interval_mode:
        How to interpret numeric frequency intervals in ``bands`` (when interval-based):
        - "absolute": endpoints are axis coordinates
        - "percent": endpoints are percent/fraction of axis span
    """
    if value_col not in summary_df.columns:
        raise ValueError(f"Input DataFrame must contain column {value_col!r}.")

    if time_interval_mode not in ("absolute", "percent"):
        raise ValueError("time_interval_mode must be 'absolute' or 'percent'.")
    if freq_interval_mode not in ("absolute", "percent"):
        raise ValueError("freq_interval_mode must be 'absolute' or 'percent'.")

    reducer_kind = _coerce_reducer_kind(reducer)
    meta_cols = [c for c in summary_df.columns if not (drop_value and c == value_col)]
    phase_order = tuple(times.keys()) if times is not None else tuple()

    records: list[dict[str, Any]] = []

    for _, row in summary_df.iterrows():
        payload = row[value_col]
        meta = {c: row[c] for c in meta_cols}

        # --------------------
        # 2D scalogram payload
        # --------------------
        if isinstance(payload, pd.DataFrame):
            if times is None or bands is None:
                raise ValueError(
                    "2D scalogram input requires both `times` and `bands`."
                )

            df_sorted, f_axis, t_axis = prepare_scalogram(payload)
            band_defs = parse_band_definitions(
                bands,
                f_axis,
                inclusive=inclusive,
                interval_mode=freq_interval_mode,
            )

            for phase in phase_order:
                t_sel = parse_axis_selection(
                    times[phase],
                    t_axis,
                    inclusive=inclusive,
                    interval_mode=time_interval_mode,
                )

                for band_name, f_sel in band_defs:
                    val = reduce_2d(
                        df_sorted,
                        f_axis,
                        t_axis,
                        f_sel,
                        t_sel,
                        reducer=reducer_kind,
                    )
                    rec = dict(meta)
                    rec[out_cols.value] = val
                    rec.update({out_cols.band: band_name, out_cols.phase: phase})
                    records.append(rec)

        # -----------------
        # 1D series payload
        # -----------------
        elif isinstance(payload, pd.Series):
            s_sorted, axis_info = prepare_series(payload)

            if axis == "time":
                if times is None:
                    raise ValueError("axis='time' requires `times`.")

                for phase in phase_order:
                    t_sel = parse_axis_selection(
                        times[phase],
                        axis_info,
                        inclusive=inclusive,
                        interval_mode=time_interval_mode,
                    )
                    val = reduce_1d(s_sorted, axis_info, t_sel, reducer=reducer_kind)

                    rec = dict(meta)
                    rec[out_cols.value] = val
                    if keep_full_dim_cols:
                        rec.update({out_cols.band: pd.NA, out_cols.phase: phase})
                    else:
                        rec.update({out_cols.phase: phase})
                    records.append(rec)

            elif axis == "freq":
                if bands is None:
                    raise ValueError("axis='freq' requires `bands`.")

                band_defs = parse_band_definitions(
                    bands,
                    axis_info,
                    inclusive=inclusive,
                    interval_mode=freq_interval_mode,
                )

                for band_name, f_sel in band_defs:
                    val = reduce_1d(s_sorted, axis_info, f_sel, reducer=reducer_kind)

                    rec = dict(meta)
                    rec[out_cols.value] = val
                    if keep_full_dim_cols:
                        rec.update({out_cols.band: band_name, out_cols.phase: pd.NA})
                    else:
                        rec.update({out_cols.band: band_name})
                    records.append(rec)

            else:
                raise ValueError("axis must be 'time' or 'freq'.")

        # -----------------------
        # Unsupported payload type
        # -----------------------
        else:
            if on_missing == "raise":
                raise TypeError(
                    f"Unsupported payload type in {value_col!r}: {type(payload)!r}. "
                    "Expected pd.DataFrame (2D) or pd.Series (1D)."
                )
            continue

    out = pd.DataFrame.from_records(records)

    # Ordering (optional)
    if not out.empty and keep_full_dim_cols:
        if bands is not None and out_cols.band in out.columns:
            if isinstance(bands, Mapping):
                band_names = [str(k) for k in bands.keys()]
            else:
                band_names = [
                    str(x) for x in out[out_cols.band].dropna().unique().tolist()
                ]
            if band_names:
                out[out_cols.band] = pd.Categorical(
                    out[out_cols.band], categories=band_names, ordered=True
                )

        if out_cols.phase in out.columns:
            out[out_cols.phase] = pd.Categorical(
                out[out_cols.phase], categories=list(phase_order), ordered=True
            )

        sort_cols = [c for c in (out_cols.band, out_cols.phase) if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def split_nested_values(
    summary_df: pd.DataFrame,
    *,
    value_col: str = "Value",
    bands: BandDefinition | None = None,
    times: TimeWells | None = None,
    axis: SeriesAxis = "time",
    reducer: ReducerKind | str = ReducerKind.MEAN,
    time_interval_mode: AxisIntervalMode = "absolute",
    freq_interval_mode: AxisIntervalMode = "absolute",
    inclusive: bool = True,
    drop_value: bool = True,
    keep_full_dim_cols: bool = False,
    on_missing: Literal["raise", "skip"] = "skip",
    out_cols: GridResultColumns = GridResultColumns(),
) -> pd.DataFrame:
    """Split 2D scalograms along one axis and return a Series on the other axis.

    axis="time": wells = Phase; each cell returns Series(index=freq labels)
    axis="freq": wells = Band; each cell returns Series(index=time labels)

    Parameters
    ----------
    time_interval_mode:
        See :func:`grid_nested_values`.
    freq_interval_mode:
        See :func:`grid_nested_values`.
    """
    if value_col not in summary_df.columns:
        raise ValueError(f"Input DataFrame must contain column {value_col!r}.")

    if time_interval_mode not in ("absolute", "percent"):
        raise ValueError("time_interval_mode must be 'absolute' or 'percent'.")
    if freq_interval_mode not in ("absolute", "percent"):
        raise ValueError("freq_interval_mode must be 'absolute' or 'percent'.")

    reducer_kind = _coerce_reducer_kind(reducer)
    meta_cols = [c for c in summary_df.columns if not (drop_value and c == value_col)]
    phase_order = tuple(times.keys()) if times is not None else tuple()

    records: list[dict[str, Any]] = []

    for _, row in summary_df.iterrows():
        payload = row[value_col]
        if not isinstance(payload, pd.DataFrame):
            if on_missing == "raise":
                raise TypeError(
                    f"split_nested_values expects pd.DataFrame payloads in {value_col!r}."
                )
            continue

        df_sorted, f_axis, t_axis = prepare_scalogram(payload)
        meta = {c: row[c] for c in meta_cols}

        if axis == "time":
            if times is None:
                raise ValueError("axis='time' requires `times`.")

            for phase in phase_order:
                t_sel = parse_axis_selection(
                    times[phase],
                    t_axis,
                    inclusive=inclusive,
                    interval_mode=time_interval_mode,
                )

                series_vals = reduce_time_selection_to_series_by_freq(
                    df_sorted,
                    f_axis,
                    t_axis,
                    t_sel,
                    reducer=reducer_kind,
                )

                rec = dict(meta)
                rec[out_cols.value] = series_vals
                if keep_full_dim_cols:
                    rec.update({out_cols.band: pd.NA, out_cols.phase: phase})
                else:
                    rec.update({out_cols.phase: phase})
                records.append(rec)

        elif axis == "freq":
            if bands is None:
                raise ValueError("axis='freq' requires `bands`.")

            band_defs = parse_band_definitions(
                bands,
                f_axis,
                inclusive=inclusive,
                interval_mode=freq_interval_mode,
            )

            for band_name, f_sel in band_defs:
                series_vals = reduce_freq_selection_to_series_by_time(
                    df_sorted,
                    f_axis,
                    t_axis,
                    f_sel,
                    reducer=reducer_kind,
                )

                rec = dict(meta)
                rec[out_cols.value] = series_vals
                if keep_full_dim_cols:
                    rec.update({out_cols.band: band_name, out_cols.phase: pd.NA})
                else:
                    rec.update({out_cols.band: band_name})
                records.append(rec)

        else:
            raise ValueError("axis must be 'time' or 'freq'.")

    out = pd.DataFrame.from_records(records)

    # Ordering (optional)
    if not out.empty and keep_full_dim_cols:
        if bands is not None and out_cols.band in out.columns:
            if isinstance(bands, Mapping):
                band_names = [str(k) for k in bands.keys()]
            else:
                band_names = [
                    str(x) for x in out[out_cols.band].dropna().unique().tolist()
                ]
            if band_names:
                out[out_cols.band] = pd.Categorical(
                    out[out_cols.band], categories=band_names, ordered=True
                )

        if out_cols.phase in out.columns:
            out[out_cols.phase] = pd.Categorical(
                out[out_cols.phase], categories=list(phase_order), ordered=True
            )

        sort_cols = [c for c in (out_cols.band, out_cols.phase) if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)

    return out
