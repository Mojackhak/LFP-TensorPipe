"""Tensor frequency, notch, and edge-mask helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import (
    preproc_step_log_path,
    preproc_step_raw_path,
)
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.lfp.runtime import (
    apply_dynamic_edge_mask_strict as _apply_dynamic_edge_mask_strict_runtime,
    build_frequency_grid as _build_frequency_grid_runtime,
    compute_mask_radii_seconds as _compute_mask_radii_seconds_runtime,
    compute_notch_intervals as _compute_notch_intervals_runtime,
    cut_frequency_grid_by_intervals as _cut_frequency_grid_by_intervals_runtime,
    cycles_from_time_resolution as _cycles_from_time_resolution_runtime,
    expand_notch_widths as _expand_notch_widths_runtime,
    parse_positive_float_tuple as _parse_positive_float_tuple_runtime,
    psi_band_radii_seconds as _psi_band_radii_seconds_runtime,
)

from .coercion import _as_float
from .params import DEFAULT_TENSOR_BANDS

DEFAULT_TENSOR_NOTCH_WIDTH = 2.0
TENSOR_NOTCH_TOLERANCE_HZ = 1e-6


@dataclass(frozen=True)
class TensorFilterInheritance:
    """Filter-derived defaults inherited by Build-Tensor metrics."""

    low_freq: float
    high_freq: float
    notches: tuple[float, ...]
    notch_widths: tuple[float, ...]


@dataclass(frozen=True)
class TensorFrequencyBounds:
    """Frequency constraints derived from preproc filter + finish raw Nyquist."""

    min_low_freq: float
    max_high_freq: float
    filter_low_freq: float
    filter_high_freq: float
    nyquist_freq: float | None


def default_tensor_metric_notch_params() -> dict[str, Any]:
    return {"notches": [], "notch_widths": float(DEFAULT_TENSOR_NOTCH_WIDTH)}


def _strict_positive_float_list(value: Any, *, field_name: str) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            return []
        items = [float(item) for item in parts]
    elif isinstance(value, (int, float)):
        items = [float(value)]
    elif isinstance(value, (list, tuple)):
        items = [float(item) for item in value]
    else:
        raise ValueError(
            f"{field_name} must be a number, list, or comma-separated string."
        )
    if any((not np.isfinite(float(item))) or float(item) <= 0.0 for item in items):
        raise ValueError(f"{field_name} must contain positive finite numbers.")
    return [float(item) for item in items]


def normalize_tensor_metric_notch_params(
    notches_value: Any,
    notch_widths_value: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    raw_notches = _strict_positive_float_list(notches_value, field_name="notches")
    if not raw_notches:
        return (), ()

    raw_widths = _strict_positive_float_list(
        notch_widths_value,
        field_name="notch_widths",
    )
    if not raw_widths:
        expanded_widths = [
            float(DEFAULT_TENSOR_NOTCH_WIDTH) for _ in range(len(raw_notches))
        ]
    elif len(raw_widths) == 1:
        expanded_widths = [float(raw_widths[0]) for _ in range(len(raw_notches))]
    elif len(raw_widths) == len(raw_notches):
        expanded_widths = [float(item) for item in raw_widths]
    else:
        expanded_widths = [float(raw_widths[0]) for _ in range(len(raw_notches))]

    paired = sorted(
        zip(raw_notches, expanded_widths, strict=False),
        key=lambda item: float(item[0]),
    )
    merged: list[list[float]] = []
    for notch, width in paired:
        notch_f = float(notch)
        width_f = float(width)
        if merged and abs(notch_f - merged[-1][0]) <= TENSOR_NOTCH_TOLERANCE_HZ:
            merged[-1][1] = max(float(merged[-1][1]), width_f)
            continue
        merged.append([notch_f, width_f])
    return (
        tuple(float(item[0]) for item in merged),
        tuple(float(item[1]) for item in merged),
    )


def build_tensor_metric_notch_payload(
    notches_value: Any,
    notch_widths_value: Any,
) -> dict[str, Any]:
    notches, notch_widths = normalize_tensor_metric_notch_params(
        notches_value,
        notch_widths_value,
    )
    if not notches:
        return default_tensor_metric_notch_params()
    widths_out: float | list[float]
    if all(
        abs(float(item) - float(notch_widths[0])) <= TENSOR_NOTCH_TOLERANCE_HZ
        for item in notch_widths
    ):
        widths_out = float(notch_widths[0])
    else:
        widths_out = [float(item) for item in notch_widths]
    return {
        "notches": [float(item) for item in notches],
        "notch_widths": widths_out,
    }


def _canonicalize_notch_intervals_on_grid(
    freqs: np.ndarray,
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sort and merge effective notch intervals on one frequency grid."""
    frequency_axis = np.asarray(freqs, dtype=float).ravel()
    effective: list[tuple[int, int, float, float]] = []
    for low, high in intervals:
        indices = np.flatnonzero(
            (frequency_axis >= float(low)) & (frequency_axis <= float(high))
        )
        if indices.size == 0:
            continue
        effective.append((int(indices[0]), int(indices[-1]), float(low), float(high)))

    effective.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    merged: list[tuple[int, int, float, float]] = []
    for first_index, last_index, low, high in effective:
        if merged and first_index <= merged[-1][1] + 1:
            previous_first, previous_last, previous_low, previous_high = merged[-1]
            merged[-1] = (
                previous_first,
                max(previous_last, last_index),
                min(previous_low, low),
                max(previous_high, high),
            )
            continue
        merged.append((first_index, last_index, low, high))
    return [(low, high) for _, _, low, high in merged]


def _clean_equal_width_notch_donor_bounds(
    freqs: np.ndarray,
    intervals: list[tuple[float, float]],
    target_interval: tuple[float, float],
) -> tuple[tuple[int, int], ...]:
    """Return clean in-grid donor bounds for one effective notch interval."""
    frequency_axis = np.asarray(freqs, dtype=float).ravel()
    low, high = target_interval
    target_indices = np.flatnonzero(
        (frequency_axis >= float(low)) & (frequency_axis <= float(high))
    )
    if target_indices.size == 0:
        return ()

    all_notch_mask = np.zeros(frequency_axis.size, dtype=bool)
    for interval_low, interval_high in intervals:
        all_notch_mask |= (frequency_axis >= float(interval_low)) & (
            frequency_axis <= float(interval_high)
        )

    first_index = int(target_indices[0])
    last_index = int(target_indices[-1])
    target_left = first_index - 1
    target_right = last_index + 1
    interval_size = int(target_indices.size)
    donor_stride = interval_size + 1
    candidates = (
        (target_left - donor_stride, target_left),
        (target_right, target_right + donor_stride),
    )
    donors: list[tuple[int, int]] = []
    for donor_left, donor_right in candidates:
        if donor_left < 0 or donor_right >= frequency_axis.size:
            continue
        donor_segment = np.arange(donor_left, donor_right + 1, dtype=int)
        if np.any(all_notch_mask[donor_segment]):
            continue
        donors.append((donor_left, donor_right))
    return tuple(donors)


def validate_periodic_aperiodic_notch_bounds(params: dict[str, Any]) -> None:
    """Require reconstructable SpecParam notch intervals on the model grid."""
    notches, notch_widths = normalize_tensor_metric_notch_params(
        params.get("notches"),
        params.get("notch_widths"),
    )
    if not notches:
        return

    freq_range = params.get("freq_range_hz")
    if freq_range is None:
        freq_range = (params.get("low_freq_hz"), params.get("high_freq_hz"))
    if not isinstance(freq_range, (list, tuple)) or len(freq_range) != 2:
        raise ValueError(
            "Periodic/Aperiodic requires SpecParam freq range as two numeric values."
        )
    try:
        spec_low = float(freq_range[0])
        spec_high = float(freq_range[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Periodic/Aperiodic requires SpecParam freq range as two numeric values."
        ) from exc
    if not np.isfinite(spec_low) or not np.isfinite(spec_high) or spec_high <= spec_low:
        raise ValueError(
            "Periodic/Aperiodic SpecParam freq range must be finite with high > low."
        )

    try:
        step_hz = float(params.get("freq_step_hz"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Periodic/Aperiodic requires a positive frequency step to validate notch intervals."
        ) from exc
    if not np.isfinite(step_hz) or step_hz <= 0.0:
        raise ValueError(
            "Periodic/Aperiodic requires a positive frequency step to validate notch intervals."
        )
    model_grid = _build_frequency_grid_runtime(spec_low, spec_high, step_hz)

    intervals = _compute_notch_intervals_runtime(
        low_freq=spec_low,
        high_freq=spec_high,
        notches=notches,
        notch_widths=notch_widths,
    )
    intervals = _canonicalize_notch_intervals_on_grid(model_grid, intervals)
    for interval_low, interval_high in intervals:
        interval_indices = np.flatnonzero(
            (model_grid >= interval_low) & (model_grid <= interval_high)
        )
        if (
            int(interval_indices[0]) == 0
            or int(interval_indices[-1]) == model_grid.size - 1
        ):
            raise ValueError(
                "Periodic/Aperiodic notch interval "
                f"[{interval_low:g}, {interval_high:g}] Hz touches or crosses "
                "the SpecParam model-grid boundary "
                f"[{model_grid[0]:g}, {model_grid[-1]:g}] Hz. "
                "Each intersecting interval must stay strictly inside the fitting range."
            )
        if not _clean_equal_width_notch_donor_bounds(
            model_grid,
            intervals,
            (interval_low, interval_high),
        ):
            raise ValueError(
                "Periodic/Aperiodic notch interval "
                f"[{interval_low:g}, {interval_high:g}] Hz has no clean "
                "equal-width donor segment on either side of the SpecParam "
                "model grid. Reduce the notch width, move the notch, or widen "
                "the fitting range."
            )


def load_tensor_filter_metric_notch_params(context: RecordContext) -> dict[str, Any]:
    inheritance = load_tensor_filter_inheritance(context)
    return build_tensor_metric_notch_payload(
        list(inheritance.notches),
        list(inheritance.notch_widths),
    )


def resolve_tensor_metric_interest_range(
    metric_key: str,
    metric_params: dict[str, Any],
) -> tuple[float, float] | None:
    if metric_key == "periodic_aperiodic":
        value = metric_params.get("freq_range_hz")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            low = float(value[0])
            high = float(value[1])
        except Exception:
            return None
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None
        return float(low), float(high)

    if metric_key in {"psi", "burst"}:
        raw_bands = metric_params.get("bands")
        if not isinstance(raw_bands, list) or not raw_bands:
            return None
        lows: list[float] = []
        highs: list[float] = []
        for band in raw_bands:
            if not isinstance(band, dict):
                continue
            try:
                start = float(band.get("start"))
                end = float(band.get("end"))
            except Exception:
                continue
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue
            lows.append(float(start))
            highs.append(float(end))
        if not lows or not highs:
            return None
        return min(lows), max(highs)

    try:
        low = float(metric_params.get("low_freq_hz"))
        high = float(metric_params.get("high_freq_hz"))
    except Exception:
        return None
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    return float(low), float(high)


def compute_tensor_metric_filter_notch_warnings(
    context: RecordContext,
    metric_key: str,
    metric_params: dict[str, Any],
) -> list[str]:
    inheritance = load_tensor_filter_inheritance(context)
    if not inheritance.notches:
        return []

    interest_range = resolve_tensor_metric_interest_range(metric_key, metric_params)
    if interest_range is None:
        return []
    metric_low, metric_high = interest_range

    metric_notches, metric_widths = normalize_tensor_metric_notch_params(
        metric_params.get("notches"),
        metric_params.get("notch_widths", DEFAULT_TENSOR_NOTCH_WIDTH),
    )
    warnings: list[str] = []

    for filter_notch, filter_width in zip(
        inheritance.notches,
        inheritance.notch_widths,
        strict=False,
    ):
        if not (
            float(metric_low) - float(filter_width)
            < float(filter_notch)
            < float(metric_high) + float(filter_width)
        ):
            continue

        matched_width: float | None = None
        for metric_notch, metric_width in zip(
            metric_notches,
            metric_widths,
            strict=False,
        ):
            if (
                abs(float(metric_notch) - float(filter_notch))
                <= TENSOR_NOTCH_TOLERANCE_HZ
            ):
                matched_width = float(metric_width)
                break

        if matched_width is None:
            warnings.append(
                f"Missing preprocess filter notch {float(filter_notch):g} Hz."
            )
            continue
        if float(matched_width) + TENSOR_NOTCH_TOLERANCE_HZ < float(filter_width):
            warnings.append(
                "Notch "
                f"{float(filter_notch):g} Hz width is too small "
                f"(metric={float(matched_width):g}, filter={float(filter_width):g})."
            )
    return warnings


def _cycles_from_time_resolution(
    freqs_hz: np.ndarray,
    *,
    method: str,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
    mt_time_bandwidth_product: float = 4.0,
    mt_min_cycles: float = 3.0,
) -> np.ndarray:
    return _cycles_from_time_resolution_runtime(
        freqs_hz,
        method=method,
        time_resolution_s=float(time_resolution_s),
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        mt_time_bandwidth_product=float(mt_time_bandwidth_product),
        mt_min_cycles=float(mt_min_cycles),
    )


def _compute_mask_radii_seconds(
    freqs_hz: np.ndarray,
    *,
    method: str,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
    mt_time_bandwidth_product: float = 4.0,
    mt_min_cycles: float = 3.0,
) -> np.ndarray:
    return _compute_mask_radii_seconds_runtime(
        freqs_hz,
        method=method,
        time_resolution_s=float(time_resolution_s),
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        mt_time_bandwidth_product=float(mt_time_bandwidth_product),
        mt_min_cycles=float(mt_min_cycles),
    )


def _apply_dynamic_edge_mask_strict(
    *,
    raw: Any,
    tensor: np.ndarray,
    metadata: dict[str, Any],
    metric_label: str,
    freqs_lookup: list[float | str],
    radii_s: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    return _apply_dynamic_edge_mask_strict_runtime(
        raw=raw,
        tensor=np.asarray(tensor),
        metadata=dict(metadata),
        metric_label=metric_label,
        freqs_lookup=list(freqs_lookup),
        radii_s=list(radii_s),
    )


def _psi_band_radii_seconds(
    *,
    metadata: dict[str, Any],
    method: str,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
) -> tuple[list[str], list[float]]:
    return _psi_band_radii_seconds_runtime(
        metadata=dict(metadata),
        method=method,
        time_resolution_s=float(time_resolution_s),
        min_cycles=min_cycles,
        max_cycles=max_cycles,
    )


def _parse_positive_float_tuple(value: Any) -> tuple[float, ...]:
    return _parse_positive_float_tuple_runtime(value)


def _expand_notch_widths(notch_widths: Any, n_notches: int) -> tuple[float, ...]:
    return _expand_notch_widths_runtime(notch_widths, int(n_notches))


def _compute_notch_intervals(
    *,
    low_freq: float,
    high_freq: float,
    notches: tuple[float, ...],
    notch_widths: tuple[float, ...],
) -> list[tuple[float, float]]:
    return _compute_notch_intervals_runtime(
        low_freq=float(low_freq),
        high_freq=float(high_freq),
        notches=tuple(notches),
        notch_widths=tuple(notch_widths),
    )


def _cut_frequency_grid_by_intervals(
    freqs: np.ndarray,
    intervals: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    return _cut_frequency_grid_by_intervals_runtime(
        np.asarray(freqs, dtype=float),
        list(intervals),
    )


def _build_frequency_grid(
    low_freq: float,
    high_freq: float,
    step_hz: float,
) -> np.ndarray:
    return _build_frequency_grid_runtime(
        float(low_freq),
        float(high_freq),
        float(step_hz),
    )


def _load_finish_nyquist_hz(
    context: RecordContext,
    *,
    read_raw_fif_fn=None,
) -> float | None:
    resolver = PathResolver(context)
    finish_path = preproc_step_raw_path(resolver, "finish")
    if not finish_path.exists():
        return None
    try:
        import mne

        reader = read_raw_fif_fn or mne.io.read_raw_fif
        raw = reader(str(finish_path), preload=False, verbose="ERROR")
        sfreq = float(raw.info.get("sfreq", 0.0))
        if hasattr(raw, "close"):
            raw.close()
        if not np.isfinite(sfreq) or sfreq <= 0.0:
            return None
        nyquist = sfreq / 2.0
        if not np.isfinite(nyquist) or nyquist <= 0.0:
            return None
        return nyquist
    except Exception:
        return None


def load_tensor_filter_inheritance(context: RecordContext) -> TensorFilterInheritance:
    """Load frequency/notch defaults inherited from preprocess filter log."""
    resolver = PathResolver(context)
    payload = read_run_log(preproc_step_log_path(resolver, "filter"))
    low_freq = 1.0
    high_freq = 200.0
    notches: tuple[float, ...] = ()
    notch_widths: tuple[float, ...] = ()

    if payload is not None and bool(payload.get("completed")):
        params = payload.get("params", {})
        if isinstance(params, dict):
            low_freq = _as_float(params.get("low_freq", low_freq), low_freq)
            high_freq = _as_float(params.get("high_freq", high_freq), high_freq)
            notches = _parse_positive_float_tuple(params.get("notches"))
            notch_widths = _expand_notch_widths(
                params.get("notch_widths", 2.0), len(notches)
            )

    if high_freq <= low_freq:
        high_freq = low_freq + 1.0
    return TensorFilterInheritance(
        low_freq=float(low_freq),
        high_freq=float(high_freq),
        notches=notches,
        notch_widths=notch_widths,
    )


def resolve_tensor_frequency_bounds(context: RecordContext) -> TensorFrequencyBounds:
    """Resolve metric frequency bounds from filter settings and Nyquist."""
    inheritance = load_tensor_filter_inheritance(context)
    min_low = float(inheritance.low_freq)
    max_high = float(inheritance.high_freq)
    nyquist = _load_finish_nyquist_hz(context)
    if nyquist is not None:
        max_high = min(max_high, float(nyquist))
    if max_high <= min_low:
        max_high = min_low + 1.0
    return TensorFrequencyBounds(
        min_low_freq=min_low,
        max_high_freq=max_high,
        filter_low_freq=float(inheritance.low_freq),
        filter_high_freq=float(inheritance.high_freq),
        nyquist_freq=nyquist,
    )


def validate_tensor_frequency_params(
    context: RecordContext,
    *,
    low_freq: float,
    high_freq: float,
    step_hz: float,
) -> tuple[bool, str, TensorFrequencyBounds]:
    """Validate Build-Tensor frequency params against preproc and data bounds."""
    bounds = resolve_tensor_frequency_bounds(context)
    low = float(low_freq)
    high = float(high_freq)
    step = float(step_hz)

    if low <= 0.0:
        return False, "Low freq must be > 0 Hz.", bounds
    if low < bounds.min_low_freq:
        return (
            False,
            (
                f"Low freq must be >= {bounds.min_low_freq:g} Hz "
                "(inherited from Preprocess Filter low freq)."
            ),
            bounds,
        )
    if high > bounds.max_high_freq:
        upper_source = (
            "min(preproc filter high freq, Nyquist)"
            if bounds.nyquist_freq is not None
            else "preproc filter high freq"
        )
        return (
            False,
            f"High freq must be <= {bounds.max_high_freq:g} Hz ({upper_source}).",
            bounds,
        )
    if high <= low:
        return False, "High freq must be greater than Low freq.", bounds
    if step <= 0.0:
        return False, "Step must be > 0.", bounds
    try:
        _build_frequency_grid(low, high, step)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc), bounds
    return True, "", bounds


def load_tensor_frequency_defaults(
    context: RecordContext,
) -> tuple[float, float, float]:
    """Return Build-Tensor frequency defaults from preproc filter output."""
    bounds = resolve_tensor_frequency_bounds(context)
    low_freq = max(float(bounds.min_low_freq), 0.5)
    high_freq = float(bounds.max_high_freq)
    step_hz = 0.5
    if high_freq <= low_freq:
        high_freq = low_freq + 1.0
    return low_freq, high_freq, step_hz


def _effective_n_jobs_payload(
    *,
    n_jobs: int,
    outer_n_jobs: int,
) -> dict[str, dict[str, int]]:
    return {
        "effective_n_jobs": {
            "n_jobs": int(n_jobs),
            "outer_n_jobs": int(outer_n_jobs),
        }
    }


__all__ = [
    "DEFAULT_TENSOR_BANDS",
    "DEFAULT_TENSOR_NOTCH_WIDTH",
    "TensorFilterInheritance",
    "TensorFrequencyBounds",
    "_apply_dynamic_edge_mask_strict",
    "_build_frequency_grid",
    "_compute_mask_radii_seconds",
    "_compute_notch_intervals",
    "_cut_frequency_grid_by_intervals",
    "_cycles_from_time_resolution",
    "_effective_n_jobs_payload",
    "_expand_notch_widths",
    "_load_finish_nyquist_hz",
    "_parse_positive_float_tuple",
    "_psi_band_radii_seconds",
    "build_tensor_metric_notch_payload",
    "compute_tensor_metric_filter_notch_warnings",
    "default_tensor_metric_notch_params",
    "load_tensor_filter_inheritance",
    "load_tensor_filter_metric_notch_params",
    "load_tensor_frequency_defaults",
    "normalize_tensor_metric_notch_params",
    "resolve_tensor_frequency_bounds",
    "resolve_tensor_metric_interest_range",
    "validate_tensor_frequency_params",
]
