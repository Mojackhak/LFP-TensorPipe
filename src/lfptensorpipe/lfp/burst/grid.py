"""
Burst detection on an MNE Raw object with output on the native sample grid.

Algorithm (per channel, per band):
1) Estimate a band magnitude with Hilbert, Morlet, or Multitaper.
2) Compute a percentile threshold (default 75th) on the magnitude.
   - If `baseline_keep` is provided, the threshold is computed ONLY from the samples
     covered by matching Raw annotations (e.g., baseline "sit"), and then applied
     to the full recording.
3) Detect supra-threshold contiguous segments and keep segments whose duration
   is at least `min_cycles` and, when provided, at most `max_cycles` periods of
   the band center frequency. Segments above the maximum are excluded in full.
4) Return a tensor on the native time grid. Accepted burst samples contain the
   threshold-normalized magnitude, valid non-burst samples are zero, and invalid
   support is NaN.

This native-rate three-state representation is required for later Burst feature
extraction. Alignment may derive a separate lower-rate visualization artifact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet, tfr_array_multitaper
from scipy.signal import hilbert

from ..common import (
    morlet_mask_radius_time_s_from_freqs_n_cycles,
    multitaper_mask_radius_time_s_from_freqs_n_cycles,
)
from ..mask.annotations import (
    ANNOTATION_SCOPE_SEMANTICS,
    annotation_sample_support_by_channel,
    output_time_mask_by_annotations,
    valid_segments_from_annotation_support,
)
from .semantics import (
    HILBERT_FIR_DESIGN,
    HILBERT_FIR_LENGTH_FACTOR,
    HILBERT_FIR_PAD,
    HILBERT_FIR_TRANSITION_POLICY,
    HILBERT_FIR_WINDOW,
    burst_estimator_signature,
    burst_value_semantics,
    normalize_burst_method,
    normalize_hilbert_filter_method,
)

Band = Tuple[float, float]
BandValueOrSegments = Band | list[Band]
BandSpec = Mapping[str, BandValueOrSegments]


MatchMode = Literal["substring", "exact"]


def _normalize_bands(bands: BandSpec) -> tuple[list[str], list[list[Band]], np.ndarray]:
    """Normalize a band spec mapping.

    Supports:
      - name -> (fmin,fmax)
      - name -> [(fmin,fmax), (fmin2,fmax2), ...]

    Returns:
        band_names: list[str]
        band_segments: list[list[(fmin,fmax)]] aligned with band_names
        union_edges: array (n_bands,2) where each row is (min_lo,max_hi)
    """
    if not isinstance(bands, Mapping) or len(bands) == 0:
        raise ValueError("bands must be a non-empty mapping.")

    band_names: list[str] = []
    band_segments: list[list[Band]] = []
    union_edges: list[tuple[float, float]] = []

    for name, spec in bands.items():
        bname = str(name)
        if (
            isinstance(spec, (tuple, list))
            and len(spec) == 2
            and not isinstance(spec[0], (tuple, list))
        ):
            segs: list[Band] = [(float(spec[0]), float(spec[1]))]  # type: ignore[arg-type]
        elif isinstance(spec, list):
            segs = [(float(a), float(b)) for (a, b) in spec]
        else:
            raise ValueError(
                f"Band '{bname}' must be (fmin,fmax) or list[(fmin,fmax)]."
            )

        if len(segs) == 0:
            raise ValueError(f"Band '{bname}' has no segments.")

        segs.sort(key=lambda item: (item[0], item[1]))

        for a, b in segs:
            if not (np.isfinite(a) and np.isfinite(b)):
                raise ValueError(f"Band '{bname}' has non-finite bounds: {(a, b)}")
            if a <= 0 or b <= 0:
                raise ValueError(f"Band '{bname}' bounds must be > 0 Hz, got {(a, b)}")
            if b <= a:
                raise ValueError(
                    f"Band '{bname}' must satisfy fmax > fmin, got {(a, b)}"
                )

        lo = min(a for a, _ in segs)
        hi = max(b for _, b in segs)
        band_names.append(bname)
        band_segments.append(segs)
        union_edges.append((lo, hi))

    return band_names, band_segments, np.asarray(union_edges, dtype=float)


def _compute_decim(sfreq_hz: float, hop_s: float | None, decim: int | None) -> int:
    if decim is not None:
        if isinstance(decim, (bool, np.bool_)):
            raise ValueError("decim must be a positive integer.")
        try:
            decim_value = float(decim)
        except (TypeError, ValueError) as exc:
            raise ValueError("decim must be a positive integer.") from exc
        if (
            not np.isfinite(decim_value)
            or decim_value <= 0.0
            or not decim_value.is_integer()
        ):
            raise ValueError("decim must be a positive integer.")
        return int(decim_value)
    if hop_s is None or float(hop_s) <= 0:
        raise ValueError(
            "Provide hop_s>0 or an explicit decim to define the time grid."
        )
    return max(1, int(round(float(sfreq_hz) * float(hop_s))))


def _annotation_intervals(
    raw: mne.io.BaseRaw,
    keep: Sequence[str],
    *,
    match: MatchMode = "substring",
) -> List[Tuple[float, float]]:
    """
    Collect [start, end) intervals (seconds) for annotations matching `keep`.

    match:
        - "substring": keep label matches if label is contained in description (case-insensitive)
        - "exact": keep label matches if equals description (case-insensitive, stripped)
    """
    if match not in {"substring", "exact"}:
        raise ValueError("match must be 'substring' or 'exact'.")

    keep_l = [k.lower() for k in keep]
    out: List[Tuple[float, float]] = []
    for onset, dur, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        d = str(desc).strip().lower()
        if match == "substring":
            hit = any(k in d for k in keep_l)
        else:
            hit = any(k == d for k in keep_l)
        if not hit:
            continue
        a0 = float(onset)
        a1 = float(onset + dur)
        out.append((a0, a1))
    return out


def _merge_intervals(
    intervals_s: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Merge overlapping or touching intervals.

    Parameters
    ----------
    intervals_s:
        Iterable of (start_s, end_s) intervals.

    Returns
    -------
    merged:
        List of merged (start_s, end_s) intervals, sorted by onset.
    """
    cleaned: List[Tuple[float, float]] = []
    for a0, a1 in intervals_s:
        a0 = float(a0)
        a1 = float(a1)
        if not (np.isfinite(a0) and np.isfinite(a1)):
            continue
        if a1 < a0:
            a0, a1 = a1, a0
        cleaned.append((a0, a1))

    if len(cleaned) == 0:
        return []

    cleaned.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [cleaned[0]]
    for a0, a1 in cleaned[1:]:
        b0, b1 = merged[-1]
        if a0 <= b1:
            merged[-1] = (b0, max(b1, a1))
        else:
            merged.append((a0, a1))
    return merged


def _expand_intervals(
    intervals_s: Sequence[Tuple[float, float]],
    *,
    guard_s: float,
    t_min_s: float,
    t_max_s: float,
) -> List[Tuple[float, float]]:
    """Expand each interval by `guard_s` seconds on both sides and merge overlaps."""
    g = float(guard_s)
    if g <= 0:
        return _merge_intervals(intervals_s)

    t_min = float(t_min_s)
    t_max = float(t_max_s)
    expanded: List[Tuple[float, float]] = []
    for a0, a1 in intervals_s:
        a0 = float(a0)
        a1 = float(a1)
        if not (np.isfinite(a0) and np.isfinite(a1)):
            continue
        if a1 < a0:
            a0, a1 = a1, a0
        expanded.append((max(t_min, a0 - g), min(t_max, a1 + g)))
    return _merge_intervals(expanded)


def _hilbert_fir_filter_specs(
    *,
    sfreq_hz: float,
    segments: Sequence[Band],
) -> list[dict[str, Any]]:
    """Return the fixed FIR specification for every surviving subband."""
    sfreq = float(sfreq_hz)
    nyquist = sfreq / 2.0
    ordered = [(float(low), float(high)) for low, high in segments]
    specifications: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(ordered):
        width = high - low
        lower_limits = [max(0.25 * low, 2.0), low, width / 2.0]
        upper_limits = [
            max(0.25 * high, 2.0),
            nyquist - high,
            width / 2.0,
        ]
        if index > 0:
            left_gap = low - ordered[index - 1][1]
            if left_gap <= 0.0:
                raise ValueError(
                    "Hilbert FIR subbands must be separated by a positive gap."
                )
            lower_limits.append(left_gap)
        else:
            left_gap = None
        if index + 1 < len(ordered):
            right_gap = ordered[index + 1][0] - high
            if right_gap <= 0.0:
                raise ValueError(
                    "Hilbert FIR subbands must be separated by a positive gap."
                )
            upper_limits.append(right_gap)
        else:
            right_gap = None

        lower_transition = float(min(lower_limits))
        upper_transition = float(min(upper_limits))
        if lower_transition <= 0.0 or upper_transition <= 0.0:
            raise ValueError(
                "Hilbert FIR transition bandwidths must be strictly positive."
            )
        lower_passband = low + lower_transition / 2.0
        upper_passband = high - upper_transition / 2.0
        if upper_passband <= lower_passband:
            raise ValueError("Hilbert FIR subband has no positive flat passband.")

        shortest_transition = min(lower_transition, upper_transition)
        filter_length = int(
            np.ceil(HILBERT_FIR_LENGTH_FACTOR * sfreq / shortest_transition)
        )
        if filter_length % 2 == 0:
            filter_length += 1
        specifications.append(
            {
                "segment_hz": [low, high],
                "cutoff_hz": [low, high],
                "passband_hz": [lower_passband, upper_passband],
                "stopband_hz": [
                    low - lower_transition / 2.0,
                    high + upper_transition / 2.0,
                ],
                "transition_bandwidth_hz": [
                    lower_transition,
                    upper_transition,
                ],
                "adjacent_excluded_gap_hz": [left_gap, right_gap],
                "filter_length_samples": filter_length,
                "filter_order": filter_length - 1,
                "fir_design": HILBERT_FIR_DESIGN,
                "fir_window": HILBERT_FIR_WINDOW,
                "phase": "zero",
                "pad": HILBERT_FIR_PAD,
                "transition_policy": HILBERT_FIR_TRANSITION_POLICY,
            }
        )
    if not specifications:  # pragma: no cover
        raise RuntimeError("Hilbert FIR estimator requires at least one subband.")
    return specifications


def _hilbert_magnitude(
    data: np.ndarray,
    *,
    sfreq_hz: float,
    segments: Sequence[Band],
    filter_order: int,
    filter_method: str = "iir",
) -> np.ndarray:
    """Return reconstructed-band Hilbert magnitude for one or more signals."""
    filter_method_eff = normalize_hilbert_filter_method(filter_method)
    filtered: np.ndarray | None = None
    if filter_method_eff == "iir":
        filter_specs: Sequence[dict[str, Any] | Band] = list(segments)
        iir_params = {
            "order": int(filter_order),
            "ftype": "butter",
            "output": "sos",
        }
    else:
        filter_specs = _hilbert_fir_filter_specs(
            sfreq_hz=float(sfreq_hz),
            segments=segments,
        )
        iir_params = None

    for specification in filter_specs:
        if filter_method_eff == "iir":
            low, high = specification
            part = mne.filter.filter_data(
                np.asarray(data, dtype=float),
                sfreq=float(sfreq_hz),
                l_freq=float(low),
                h_freq=float(high),
                method="iir",
                iir_params=iir_params,
                phase="zero",
                verbose=False,
            )
        else:
            fir_spec = specification
            lower_transition, upper_transition = fir_spec["transition_bandwidth_hz"]
            lower_passband, upper_passband = fir_spec["passband_hz"]
            part = mne.filter.filter_data(
                np.asarray(data, dtype=float),
                sfreq=float(sfreq_hz),
                l_freq=float(lower_passband),
                h_freq=float(upper_passband),
                filter_length=int(fir_spec["filter_length_samples"]),
                l_trans_bandwidth=float(lower_transition),
                h_trans_bandwidth=float(upper_transition),
                method="fir",
                phase="zero",
                fir_window=HILBERT_FIR_WINDOW,
                fir_design=HILBERT_FIR_DESIGN,
                pad=HILBERT_FIR_PAD,
                verbose=False,
            )
        filtered = (
            part.astype(np.float64, copy=True)
            if filtered is None
            else filtered + part.astype(np.float64, copy=False)
        )
    if filtered is None:  # pragma: no cover
        raise RuntimeError("Hilbert estimator requires at least one subband.")
    return np.abs(hilbert(filtered, axis=-1)).astype(np.float64, copy=False)


def _hilbert_probe_bank(
    *,
    n_times: int,
    sfreq_hz: float,
    segments: Sequence[Band],
) -> np.ndarray:
    """Build the deterministic probe bank declared by the Burst guard contract."""
    times = np.arange(int(n_times), dtype=float) / float(sfreq_hz)
    probes: list[np.ndarray] = []
    midpoint_frequencies: list[float] = []
    for low, high in segments:
        frequencies = (
            float(np.nextafter(float(low), float(high))),
            (float(low) + float(high)) / 2.0,
            float(np.nextafter(float(high), float(low))),
        )
        midpoint_frequencies.append(frequencies[1])
        for frequency in frequencies:
            for phase_index in range(8):
                phase = float(phase_index) * np.pi / 4.0
                probes.append(np.sin(2.0 * np.pi * frequency * times + phase))

    multitone = np.zeros(int(n_times), dtype=float)
    for index, frequency in enumerate(midpoint_frequencies):
        phase = 2.0 * np.pi * float(index) / float(len(midpoint_frequencies) + 1)
        multitone += np.sin(2.0 * np.pi * frequency * times + phase)
    multitone /= float(len(midpoint_frequencies))
    probes.append(multitone)
    return np.asarray(probes, dtype=np.float64)


def _hilbert_oracle_error(
    *,
    sfreq_hz: float,
    segments: Sequence[Band],
    filter_order: int,
    comparison_samples: int,
    filter_method: str = "iir",
) -> np.ndarray:
    """Return normalized isolated-versus-continuous errors for one duration."""
    comparison_samples_eff = int(comparison_samples)
    if comparison_samples_eff <= 0 or comparison_samples_eff % 2:
        raise ValueError("comparison_samples must be a positive even integer.")
    reference_samples = 5 * comparison_samples_eff
    reference_input = _hilbert_probe_bank(
        n_times=reference_samples,
        sfreq_hz=float(sfreq_hz),
        segments=segments,
    )
    start = 2 * comparison_samples_eff
    stop = 3 * comparison_samples_eff
    reference = _hilbert_magnitude(
        reference_input,
        sfreq_hz=float(sfreq_hz),
        segments=segments,
        filter_order=int(filter_order),
        filter_method=filter_method,
    )[:, start:stop]
    isolated = _hilbert_magnitude(
        reference_input[:, start:stop],
        sfreq_hz=float(sfreq_hz),
        segments=segments,
        filter_order=int(filter_order),
        filter_method=filter_method,
    )
    central = slice(
        comparison_samples_eff // 4,
        3 * comparison_samples_eff // 4,
    )
    reference_scale = np.median(reference[:, central], axis=1)
    if np.any(~np.isfinite(reference_scale)) or np.any(reference_scale <= 0.0):
        raise RuntimeError("Hilbert guard reference magnitude is not positive.")
    return np.abs(isolated - reference) / reference_scale[:, None]


def _guard_from_errors(errors: Sequence[np.ndarray], *, tolerance: float) -> int:
    """Return the smallest symmetric guard satisfying every error matrix."""
    required = 0
    for error in errors:
        over = np.any(np.asarray(error, dtype=float) > float(tolerance), axis=0)
        positions = np.flatnonzero(over)
        if positions.size == 0:
            continue
        n_times = int(over.size)
        nearest_edge_distance = np.minimum(positions, n_times - 1 - positions)
        required = max(required, int(np.max(nearest_edge_distance)) + 1)
    return int(required)


def _tail_error(errors: Sequence[np.ndarray], *, guard: int) -> float:
    """Return the pooled maximum error retained by one symmetric guard."""
    maximum = 0.0
    has_retained_support = False
    for error in errors:
        n_times = int(error.shape[-1])
        stop = n_times - int(guard) if guard else n_times
        if stop <= int(guard):
            continue
        has_retained_support = True
        maximum = max(maximum, float(np.max(error[:, int(guard) : stop])))
    return float(maximum) if has_retained_support else float("inf")


def _compute_hilbert_guard_samples(
    *,
    sfreq_hz: float,
    segments: Sequence[Band],
    filter_order: int,
    tolerance_pct: float,
    filter_method: str = "iir",
) -> tuple[int, dict[str, Any]]:
    """Calibrate one band-specific Hilbert guard against the declared oracle."""
    tolerance = float(tolerance_pct) / 100.0
    if not np.isfinite(tolerance) or not (0.0 < tolerance < 1.0):
        raise ValueError("tolerance_pct must be finite and in (0, 100).")
    filter_method_eff = normalize_hilbert_filter_method(filter_method)

    lowest_frequency = min(float(low) for low, _high in segments)
    base_samples = int(np.ceil(max(8.0, 16.0 / lowest_frequency) * float(sfreq_hz)))
    if base_samples % 2:
        base_samples += 1
    errors: list[np.ndarray] = []
    guards: list[int] = []
    comparison_sample_counts: list[int] = []
    usable_guards: list[int] = []
    converged = False
    for doubling_index in range(5):
        comparison_samples = int(base_samples * (2**doubling_index))
        error = _hilbert_oracle_error(
            sfreq_hz=float(sfreq_hz),
            segments=segments,
            filter_order=int(filter_order),
            comparison_samples=comparison_samples,
            filter_method=filter_method_eff,
        )
        errors.append(error)
        guard_estimate = _guard_from_errors([error], tolerance=tolerance)
        guards.append(guard_estimate)
        comparison_sample_counts.append(comparison_samples)
        if np.isfinite(_tail_error([error], guard=guard_estimate)):
            usable_guards.append(guard_estimate)
        if len(usable_guards) >= 2 and abs(usable_guards[-1] - usable_guards[-2]) <= 1:
            converged = True
            break
    guard = _guard_from_errors(errors, tolerance=tolerance)
    retained_error = _tail_error(errors, guard=guard)
    if not np.isfinite(retained_error) or retained_error > tolerance:
        raise RuntimeError(
            "Hilbert guard oracle has no retained support satisfying the tolerance."
        )
    previous_error = _tail_error(errors, guard=guard - 1) if guard > 0 else float("nan")
    if guard > 0 and not previous_error > tolerance:
        raise RuntimeError("Hilbert guard oracle did not return the minimum guard.")
    return int(guard), {
        "mode": "isolated_vs_five_length_continuous_reference",
        "filter_method": filter_method_eff,
        "fir_filter_specs": (
            _hilbert_fir_filter_specs(
                sfreq_hz=float(sfreq_hz),
                segments=segments,
            )
            if filter_method_eff == "fir"
            else None
        ),
        "tolerance_pct": float(tolerance_pct),
        "probe_count": int(errors[0].shape[0]),
        "comparison_sample_counts": comparison_sample_counts,
        "comparison_durations_s": [
            float(value) / float(sfreq_hz) for value in comparison_sample_counts
        ],
        "guard_estimates_samples": [int(value) for value in guards],
        "retained_tail_error": float(retained_error),
        "previous_tail_error": float(previous_error),
        "converged": bool(converged),
        "acceptance_mode": (
            "consecutive_guard_stability" if converged else "pooled_declared_durations"
        ),
    }


def _band_frequency_grid(
    segments: Sequence[Band],
    *,
    step_hz: float,
) -> np.ndarray:
    """Build one uniform notch-excluded frequency grid without interpolation."""
    step = float(step_hz)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("freq_step_hz must be finite and > 0.")
    low = min(float(start) for start, _stop in segments)
    high = max(float(stop) for _start, stop in segments)
    count = int(np.floor((high - low) / step)) + 1
    full = np.unique(np.round(low + np.arange(count, dtype=float) * step, 6))
    keep = np.zeros(full.size, dtype=bool)
    for index, (start, stop) in enumerate(segments):
        lower = full >= float(start) if index == 0 else full > float(start)
        upper = (
            full <= float(stop) if index == len(segments) - 1 else full < float(stop)
        )
        keep |= lower & upper
    frequencies = full[keep]
    if frequencies.size == 0:
        raise ValueError(
            "Burst frequency grid has no retained bins; reduce Step (Hz) or widen the band."
        )
    return frequencies


def _spectral_guard_samples(
    *,
    method: str,
    sfreq_hz: float,
    frequencies_hz: np.ndarray,
    n_cycles: float,
) -> int:
    """Return exact Morlet 5-sigma or Multitaper half-window support."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    cycles = np.full(frequencies.size, float(n_cycles), dtype=float)
    if method == "morlet":
        radius_s = morlet_mask_radius_time_s_from_freqs_n_cycles(
            frequencies,
            n_cycles=cycles,
        )
    elif method == "multitaper":
        radius_s = multitaper_mask_radius_time_s_from_freqs_n_cycles(
            frequencies,
            n_cycles=cycles,
        )
    else:  # pragma: no cover
        raise ValueError("Spectral guard requires morlet or multitaper.")
    return int(np.ceil(float(sfreq_hz) * float(np.max(radius_s))))


def _spectral_band_magnitude(
    data: np.ndarray,
    *,
    method: str,
    sfreq_hz: float,
    frequencies_hz: np.ndarray,
    n_cycles: float,
    mt_time_bandwidth_product: float,
    n_jobs: int,
) -> np.ndarray:
    """Return sqrt-mean-power magnitude on the native sample grid."""
    source = np.asarray(data, dtype=float)
    if source.ndim == 1:
        source = source[None, :]
    array = source[None, :, :]
    cycles = np.full(frequencies_hz.size, float(n_cycles), dtype=float)
    if method == "morlet":
        power = tfr_array_morlet(
            array,
            sfreq=float(sfreq_hz),
            freqs=np.asarray(frequencies_hz, dtype=float),
            n_cycles=cycles,
            output="power",
            decim=1,
            n_jobs=int(n_jobs),
            use_fft=True,
        )
    elif method == "multitaper":
        power = tfr_array_multitaper(
            array,
            sfreq=float(sfreq_hz),
            freqs=np.asarray(frequencies_hz, dtype=float),
            n_cycles=cycles,
            time_bandwidth=float(mt_time_bandwidth_product),
            output="power",
            decim=1,
            n_jobs=int(n_jobs),
            use_fft=True,
        )
    else:  # pragma: no cover
        raise ValueError("Spectral magnitude requires morlet or multitaper.")
    return np.sqrt(np.mean(np.asarray(power[0], dtype=float), axis=1))


def _band_magnitude(
    data: np.ndarray,
    *,
    method: str,
    sfreq_hz: float,
    segments: Sequence[Band],
    frequencies_hz: np.ndarray,
    filter_order: int,
    hilbert_filter_method: str,
    morlet_n_cycles: float,
    mt_n_cycles: float,
    mt_time_bandwidth_product: float,
    n_jobs: int,
) -> np.ndarray:
    """Dispatch one band magnitude while preserving a channel axis."""
    source = np.asarray(data, dtype=float)
    squeeze = source.ndim == 1
    if squeeze:
        source = source[None, :]
    if method == "hilbert":
        magnitude = _hilbert_magnitude(
            source,
            sfreq_hz=float(sfreq_hz),
            segments=segments,
            filter_order=int(filter_order),
            filter_method=hilbert_filter_method,
        )
    else:
        magnitude = _spectral_band_magnitude(
            source,
            method=method,
            sfreq_hz=float(sfreq_hz),
            frequencies_hz=frequencies_hz,
            n_cycles=(
                float(morlet_n_cycles) if method == "morlet" else float(mt_n_cycles)
            ),
            mt_time_bandwidth_product=mt_time_bandwidth_product,
            n_jobs=int(n_jobs),
        )
    return magnitude[0] if squeeze else magnitude


def _filter_runs_by_length(
    mask_1d: np.ndarray,
    min_len: int,
    max_len: int | None = None,
) -> np.ndarray:
    """Keep only True-runs within the inclusive sample-length bounds."""
    x = np.asarray(mask_1d, dtype=bool)
    if x.size == 0:
        return x.copy()

    # Find run starts/ends using diff on padded array
    padded = np.concatenate([[False], x, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)  # end indices in padded coords

    keep = np.zeros_like(x, dtype=bool)
    for s, e in zip(starts, ends):
        run_len = int(e - s)
        if run_len >= int(min_len) and (max_len is None or run_len <= int(max_len)):
            keep[s:e] = True
    return keep


def grid(
    raw: mne.io.BaseRaw,
    *,
    bands: BandSpec,
    thresholds: Sequence | None = None,
    percentile: float | None = 75.0,
    min_cycles: float = 2.0,
    max_cycles: float | None = None,
    hop_s: float | None = None,
    decim: int | None = None,
    target_n_times: int | None = None,
    picks: Sequence[str] | None = None,
    baseline_keep: Sequence[str] | None = None,
    baseline_match: MatchMode = "substring",
    baseline_fallback: str = "full",
    method: str = "hilbert",
    filter_order: int = 4,
    hilbert_filter_method: str = "iir",
    freq_step_hz: float = 1.0,
    morlet_n_cycles: float = 6.0,
    mt_n_cycles: float = 7.0,
    mt_time_bandwidth_product: float = 4.0,
    hilbert_edge_tolerance_pct: float = 10.0,
    edge_anno: Sequence[str] | None = ("bad", "edge"),
    mode: MatchMode = "substring",
    boundary_isolated_filter: bool = True,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute a Burst magnitude tensor on the native sample grid.

    Parameters
    ----------
    raw:
        MNE Raw with continuous data and annotations.
    bands:
        Dict mapping band name -> (l_freq, h_freq), in Hz.
    thresholds:
        Optional, user-provided envelope thresholds.

        If provided, this overrides percentile/baseline-based thresholding.
        The sequence order must match the order of the `bands` definition, and
        the length must equal the number of bands.

        Each element can be either:
          - a scalar (applied to all channels), or
          - an array-like of shape (n_channels,) providing a per-channel threshold
            for that band.
    percentile:
        Percentile used for envelope thresholding (per channel, per band).
    min_cycles:
        Minimum burst duration in cycles of the band center frequency.
    max_cycles:
        Optional maximum burst duration in cycles of the band center frequency.
        Supra-threshold segments longer than this limit are excluded in full.
    hop_s, decim:
        Time-grid controls retained for API compatibility. Their effective
        decimation must be one because Burst feature extraction uses native samples.
    target_n_times:
        If provided, enforce exact time-axis length (pad/trim with NaNs).
    picks:
        Optional channel selection.
    baseline_keep:
        Optional list of annotation labels used to compute the threshold. If provided,
        the threshold is computed only on envelope samples within matching annotation
        intervals, then applied to the full recording.
    baseline_match:
        "substring" (default) or "exact" matching for baseline_keep labels.
    baseline_fallback:
        "full" (default) uses each missing channel's own finite, non-edge
        recording support; "raise" raises if any selected channel lacks usable
        baseline support.
    method:
        Burst magnitude estimator: "hilbert", "morlet", or "multitaper".
    filter_order:
        Butterworth IIR order used only by the Hilbert estimator.
    hilbert_filter_method:
        Fixed Hilbert subband filter family: "iir" or "fir".
    freq_step_hz:
        Frequency spacing used only by Morlet and Multitaper.
    morlet_n_cycles:
        Fixed Morlet cycles per retained frequency.
    mt_n_cycles:
        Fixed Multitaper window cycles per retained frequency.
    mt_time_bandwidth_product:
        DPSS time-bandwidth product used only by Multitaper.
    hilbert_edge_tolerance_pct:
        Maximum declared Hilbert oracle error outside the per-band guard.

    edge_anno:
        Annotation labels whose intervals should be treated as edges/bad segments.
        Samples in these intervals are excluded from threshold estimation and burst
        detection, and will be NaN in the output.

        Default: ("bad", "edge"). Set to None or an empty sequence to disable.

    mode:
        Matching mode for edge_anno labels.

        - "substring": case-insensitive substring match (default)
        - "exact": case-insensitive exact match after stripping

    boundary_isolated_filter:
        When True and ``edge_anno`` is active, estimate each channel's continuous
        valid segment independently. When False, estimate the continuous channel
        and then apply annotation masking.

    Returns
    -------
    tensor:
        ndarray with shape (1, n_channels, n_bands, n_times), float64.
        Accepted burst samples contain threshold-normalized magnitude greater
        than 1.0, valid non-burst samples are 0.0, and invalid samples are NaN.
    metadata:
        Dict describing axes and parameters.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.BaseRaw.")
    if not bands:
        raise ValueError("bands must be a non-empty mapping.")
    percentile_eff: float | None = None
    if thresholds is None:
        if percentile is None:
            raise ValueError("percentile is required when thresholds are not provided.")
        percentile_eff = float(percentile)
        if not (0.0 < percentile_eff < 100.0):
            raise ValueError("percentile must be in (0, 100).")
    if float(min_cycles) <= 0:
        raise ValueError("min_cycles must be > 0.")
    max_cycles_eff: float | None = None
    if max_cycles is not None:
        max_cycles_eff = float(max_cycles)
        if not np.isfinite(max_cycles_eff) or max_cycles_eff <= 0:
            raise ValueError("max_cycles must be > 0 when provided.")
        if max_cycles_eff < float(min_cycles):
            raise ValueError("max_cycles must be >= min_cycles when provided.")
    if baseline_fallback not in {"full", "raise"}:
        raise ValueError("baseline_fallback must be 'full' or 'raise'.")
    if not isinstance(boundary_isolated_filter, (bool, np.bool_)):
        raise TypeError("boundary_isolated_filter must be true or false.")
    target_n_times_eff: int | None = None
    if target_n_times is not None:
        if isinstance(target_n_times, (bool, np.bool_)):
            raise ValueError("target_n_times must be a positive integer.")
        try:
            target_n_times_value = float(target_n_times)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_n_times must be a positive integer.") from exc
        if (
            not np.isfinite(target_n_times_value)
            or target_n_times_value <= 0.0
            or not target_n_times_value.is_integer()
        ):
            raise ValueError("target_n_times must be a positive integer.")
        target_n_times_eff = int(target_n_times_value)
    method_eff = normalize_burst_method(method)
    estimator_signature = burst_estimator_signature(
        method=method_eff,
        hilbert_filter_method=hilbert_filter_method,
        filter_order=filter_order,
        hilbert_edge_tolerance_pct=hilbert_edge_tolerance_pct,
        freq_step_hz=freq_step_hz,
        morlet_n_cycles=morlet_n_cycles,
        mt_n_cycles=mt_n_cycles,
        mt_time_bandwidth_product=mt_time_bandwidth_product,
    )
    if method_eff == "hilbert":
        hilbert_filter_method_eff = normalize_hilbert_filter_method(
            hilbert_filter_method
        )
        tolerance_pct_eff = float(hilbert_edge_tolerance_pct)
        if not np.isfinite(tolerance_pct_eff) or not (0.0 < tolerance_pct_eff < 100.0):
            raise ValueError("hilbert_edge_tolerance_pct must be in (0, 100).")
    else:
        hilbert_filter_method_eff = None
        tolerance_pct_eff = None

    sfreq = float(raw.info["sfreq"])
    decim_eff = _compute_decim(sfreq, hop_s, decim)
    if decim_eff != 1:
        raise ValueError(
            "Burst tensors must use the native sampling rate; effective decim must be 1."
        )

    # Pull data
    data = raw.get_data(picks=picks)  # (n_channels, n_times)
    ch_names = list(
        np.asarray(raw.ch_names, dtype=object)
        if picks is None
        else np.asarray(picks, dtype=object)
    )
    n_channels, n_times = data.shape
    raw_times = np.asarray(raw.times, dtype=float)

    # Edge intervals are derived from raw.annotations and then dilated by the
    # active method's per-band guard before being converted to a sample mask.
    edge_intervals: List[Tuple[float, float]] = []
    edge_anno_eff: Sequence[str] | None = edge_anno
    if edge_anno_eff is not None and len(edge_anno_eff) > 0:
        edge_intervals = _annotation_intervals(raw, edge_anno_eff, match=mode)
    boundary_isolated_requested = bool(boundary_isolated_filter)
    boundary_isolated_effective = bool(
        boundary_isolated_requested
        and edge_anno_eff is not None
        and len(edge_anno_eff) > 0
    )
    boundary_invalid = np.zeros((n_channels, n_times), dtype=bool)
    boundary_points: tuple[tuple[int, ...], ...] = tuple(() for _ in ch_names)
    boundary_support_info: Dict[str, Any] = {}
    valid_segments_by_channel: list[list[tuple[int, int]]] = [
        [(0, n_times)] for _ in ch_names
    ]
    if edge_anno_eff is not None and len(edge_anno_eff) > 0:
        boundary_invalid, boundary_points, boundary_support_info = (
            annotation_sample_support_by_channel(
                raw,
                channels=ch_names,
                keep=edge_anno_eff,
                mode=mode,
            )
        )
    if boundary_isolated_effective:
        valid_segments_by_channel = [
            valid_segments_from_annotation_support(
                boundary_invalid[channel_index],
                boundary_points[channel_index],
            )
            for channel_index in range(n_channels)
        ]

    # Baseline mask (sample domain)
    # NOTE: When user-provided thresholds are used, baseline_keep/baseline_* are
    # ignored for threshold computation.
    baseline_mask_by_channel = np.ones((n_channels, n_times), dtype=bool)
    baseline_support_info: Dict[str, Any] = {}
    baseline_intervals: List[Dict[str, Any]] = []
    baseline_keep_eff = baseline_keep if thresholds is None else None
    if baseline_keep_eff is not None:
        baseline_mask_by_channel, baseline_support_info = (
            output_time_mask_by_annotations(
                raw,
                times_s=raw_times,
                output_channels=[(str(channel),) for channel in ch_names],
                keep=baseline_keep_eff,
                mode=baseline_match,
                pad_s=0.0,
                clip_to_raw=True,
                require_match=False,
            )
        )
        baseline_intervals = [
            {
                "description": str(interval["description"]),
                "onset": float(interval["onset_s"]),
                "duration": float(interval["duration_s"]),
                "ch_names": list(interval["ch_names"]),
            }
            for interval in baseline_support_info["matched_intervals"]
        ]

    band_names, band_segments, band_union_edges = _normalize_bands(bands)
    nyquist = sfreq / 2.0
    if np.any(band_union_edges[:, 1] >= nyquist):
        raise ValueError("Burst band upper bounds must be below Nyquist.")
    band_centers = band_union_edges.mean(axis=1)
    frequency_grids_by_band: list[np.ndarray] = []
    hilbert_fir_filter_specs_by_band: list[list[dict[str, Any]] | None] = []
    for segments in band_segments:
        if method_eff == "hilbert":
            frequency_grids_by_band.append(np.asarray([], dtype=float))
            hilbert_fir_filter_specs_by_band.append(
                _hilbert_fir_filter_specs(sfreq_hz=sfreq, segments=segments)
                if hilbert_filter_method_eff == "fir"
                else None
            )
        else:
            frequency_grids_by_band.append(
                _band_frequency_grid(segments, step_hz=float(freq_step_hz))
            )
            hilbert_fir_filter_specs_by_band.append(None)
    min_run_samples_by_band = [
        int(np.ceil(float(min_cycles) * sfreq / float(f_center)))
        for f_center in band_centers
    ]
    max_run_samples_by_band: list[int | None] = (
        [
            int(np.floor(max_cycles_eff * sfreq / float(f_center)))
            for f_center in band_centers
        ]
        if max_cycles_eff is not None
        else [None] * len(band_names)
    )

    # Validate user-provided thresholds (if any) after we know n_channels.
    thresholds_by_band: list[np.ndarray] | None = None
    if thresholds is not None:
        # Numpy arrays are not instances of collections.abc.Sequence, so we accept
        # them explicitly as a common "array-like" input.
        if not isinstance(thresholds, (Sequence, np.ndarray)):
            raise TypeError("thresholds must be a Sequence/ndarray or None.")
        if len(thresholds) != len(band_names):
            raise ValueError(
                "thresholds length must match number of bands "
                f"(len(thresholds)={len(thresholds)} vs n_bands={len(band_names)}). "
                "The order must match the `bands` definition order."
            )

        thresholds_by_band = []
        for bi, thr_i in enumerate(thresholds):
            thr_arr = np.asarray(thr_i, dtype=float)
            if thr_arr.ndim == 0:
                thr_arr = np.full((n_channels,), float(thr_arr), dtype=np.float64)
            elif thr_arr.ndim == 1:
                if thr_arr.shape[0] != n_channels:
                    raise ValueError(
                        "Each thresholds element must be a scalar or have shape (n_channels,). "
                        f"Band index {bi} ('{band_names[bi]}') has shape {thr_arr.shape}, "
                        f"expected ({n_channels},)."
                    )
                thr_arr = thr_arr.astype(np.float64, copy=False)
            else:
                raise ValueError(
                    "Each thresholds element must be a scalar or a 1D array-like of shape (n_channels,). "
                    f"Band index {bi} ('{band_names[bi]}') has ndim={thr_arr.ndim}."
                )

            if not np.all(np.isfinite(thr_arr)):
                raise ValueError(
                    "thresholds contains non-finite values for band index "
                    f"{bi} ('{band_names[bi]}')."
                )
            if np.any(thr_arr <= 0.0):
                raise ValueError(
                    "thresholds must be strictly positive for band index "
                    f"{bi} ('{band_names[bi]}')."
                )
            thresholds_by_band.append(thr_arr)

    out_bands: List[np.ndarray] = []
    thresholds_used: List[np.ndarray] = []

    edge_guard_samples_by_band: List[int] = []
    edge_guard_seconds_by_band: List[float] = []
    edge_guard_details_by_band: List[Dict[str, Any]] = []
    edge_coverage_by_band: List[float] = []
    baseline_coverage_by_band: List[float] = []
    baseline_coverage_by_band_and_channel: List[List[float]] = []
    edge_intervals_dilated_by_band: List[List[Tuple[float, float]]] = []
    edge_mask_info_by_band: List[Dict[str, Any]] = []
    segment_count_by_band_channel: List[List[int]] = []
    short_segment_count_by_band_channel: List[List[int]] = []
    valid_fraction_by_band_channel: List[List[float]] = []
    guard_fraction_by_band_channel: List[List[float]] = []

    for bi, (band_name, segs, f_center, frequencies) in enumerate(
        zip(
            band_names,
            band_segments,
            band_centers,
            frequency_grids_by_band,
        )
    ):
        # Compute a per-band edge mask in the sample domain. This mask is derived
        # from raw.annotations using edge_anno/mode and dilated by the estimator's
        # declared boundary support.
        edge_guard_samples = 0
        guard_s = 0.0
        guard_details: Dict[str, Any] = {
            "mode": "not_applied",
            "method": method_eff,
        }
        edge_intervals_dilated: List[Tuple[float, float]] = []
        edge_mask = np.zeros((n_channels, n_times), dtype=bool)
        edge_mask_info: Dict[str, Any] = {}

        annotation_postmask_active = bool(
            len(edge_intervals) > 0
            and edge_anno_eff is not None
            and len(edge_anno_eff) > 0
        )
        if boundary_isolated_effective or annotation_postmask_active:
            if method_eff == "hilbert":
                edge_guard_samples, guard_details = _compute_hilbert_guard_samples(
                    sfreq_hz=sfreq,
                    segments=segs,
                    filter_order=int(filter_order),
                    tolerance_pct=float(tolerance_pct_eff),
                    filter_method=str(hilbert_filter_method_eff),
                )
            elif method_eff == "morlet":
                edge_guard_samples = _spectral_guard_samples(
                    method="morlet",
                    sfreq_hz=sfreq,
                    frequencies_hz=frequencies,
                    n_cycles=float(morlet_n_cycles),
                )
                guard_details = {
                    "mode": "morlet_5_sigma",
                    "method": "morlet",
                    "n_cycles": float(morlet_n_cycles),
                }
            else:
                edge_guard_samples = _spectral_guard_samples(
                    method="multitaper",
                    sfreq_hz=sfreq,
                    frequencies_hz=frequencies,
                    n_cycles=float(mt_n_cycles),
                )
                guard_details = {
                    "mode": "multitaper_half_window",
                    "method": "multitaper",
                    "n_cycles": float(mt_n_cycles),
                }
            guard_s = float(edge_guard_samples) / float(sfreq)

            if edge_intervals:
                edge_intervals_dilated = _expand_intervals(
                    edge_intervals,
                    guard_s=guard_s,
                    t_min_s=float(raw_times[0]),
                    t_max_s=float(raw_times[0] + n_times / sfreq),
                )

        if boundary_isolated_effective:
            env = np.full((n_channels, n_times), np.nan, dtype=np.float64)
            edge_mask = boundary_invalid.copy()
            analysis_interiors_by_channel: list[list[tuple[int, int]]] = [
                [] for _ in ch_names
            ]
            processed_counts = [0 for _ in ch_names]
            short_counts = [0 for _ in ch_names]
            for channel_index, segments_for_channel in enumerate(
                valid_segments_by_channel
            ):
                for start, stop in segments_for_channel:
                    if stop - start <= 2 * edge_guard_samples:
                        edge_mask[channel_index, start:stop] = True
                        short_counts[channel_index] += 1
                        continue
                    segment_data = data[channel_index, start:stop]
                    env[channel_index, start:stop] = _band_magnitude(
                        segment_data,
                        method=method_eff,
                        sfreq_hz=sfreq,
                        segments=segs,
                        frequencies_hz=frequencies,
                        filter_order=filter_order,
                        hilbert_filter_method=(hilbert_filter_method_eff or "iir"),
                        morlet_n_cycles=morlet_n_cycles,
                        mt_n_cycles=mt_n_cycles,
                        mt_time_bandwidth_product=mt_time_bandwidth_product,
                        n_jobs=n_jobs,
                    )
                    interior_start = start + edge_guard_samples
                    interior_stop = stop - edge_guard_samples
                    if edge_guard_samples > 0:
                        edge_mask[channel_index, start:interior_start] = True
                        edge_mask[channel_index, interior_stop:stop] = True
                    analysis_interiors_by_channel[channel_index].append(
                        (interior_start, interior_stop)
                    )
                    processed_counts[channel_index] += 1
            edge_mask_info = {
                **boundary_support_info,
                "boundary_processing": f"per_valid_segment_{method_eff}",
                "segment_count_by_channel": [int(value) for value in processed_counts],
                "short_segment_count_by_channel": [
                    int(value) for value in short_counts
                ],
            }
        else:
            if annotation_postmask_active:
                edge_mask, edge_mask_info = output_time_mask_by_annotations(
                    raw,
                    times_s=raw_times,
                    output_channels=[(str(channel),) for channel in ch_names],
                    keep=edge_anno_eff,
                    mode=mode,
                    pad_s=guard_s,
                    clip_to_raw=True,
                    require_match=False,
                )
            env = _band_magnitude(
                data,
                method=method_eff,
                sfreq_hz=sfreq,
                segments=segs,
                frequencies_hz=frequencies,
                filter_order=filter_order,
                hilbert_filter_method=(hilbert_filter_method_eff or "iir"),
                morlet_n_cycles=morlet_n_cycles,
                mt_n_cycles=mt_n_cycles,
                mt_time_bandwidth_product=mt_time_bandwidth_product,
                n_jobs=n_jobs,
            )
            analysis_interiors_by_channel = [[(0, n_times)] for _ in ch_names]
            processed_counts = [1 for _ in ch_names]
            short_counts = [0 for _ in ch_names]

        baseline_mask_band = baseline_mask_by_channel & ~edge_mask & np.isfinite(env)
        if thresholds_by_band is None:
            channels_with_baseline = np.any(baseline_mask_band, axis=1)
            if baseline_fallback == "full" and not np.all(channels_with_baseline):
                missing = ~channels_with_baseline
                baseline_mask_band[missing] = ~edge_mask[missing] & np.isfinite(
                    env[missing]
                )
                channels_with_baseline = np.any(baseline_mask_band, axis=1)
            if not np.all(channels_with_baseline):
                missing_indices = np.flatnonzero(~channels_with_baseline)
                missing_channels = [str(ch_names[index]) for index in missing_indices]
                baseline_was_present = all(
                    np.any(baseline_mask_by_channel[index]) for index in missing_indices
                )
                message_prefix = (
                    "No baseline samples remain for Burst channel(s): "
                    if baseline_was_present
                    else "No baseline samples found for Burst channel(s): "
                )
                raise ValueError(
                    message_prefix
                    + ", ".join(missing_channels)
                    + f" (band '{band_name}')."
                )

        edge_guard_samples_by_band.append(int(edge_guard_samples))
        edge_guard_seconds_by_band.append(float(guard_s))
        edge_guard_details_by_band.append(dict(guard_details))
        edge_coverage_by_band.append(float(np.mean(edge_mask)))
        baseline_coverage_by_band.append(float(np.mean(baseline_mask_band)))
        baseline_coverage_by_band_and_channel.append(
            [float(np.mean(baseline_mask_band[index])) for index in range(n_channels)]
        )
        edge_intervals_dilated_by_band.append(list(edge_intervals_dilated))
        edge_mask_info_by_band.append(edge_mask_info)
        segment_count_by_band_channel.append([int(value) for value in processed_counts])
        short_segment_count_by_band_channel.append(
            [int(value) for value in short_counts]
        )
        valid_fraction_by_band_channel.append(
            [
                float(np.mean(np.isfinite(env[index]) & ~edge_mask[index]))
                for index in range(n_channels)
            ]
        )
        guard_only_mask = edge_mask & ~boundary_invalid
        guard_fraction_by_band_channel.append(
            [float(np.mean(guard_only_mask[index])) for index in range(n_channels)]
        )

        # Threshold is either user-provided, or computed on baseline samples only
        # (if provided), per channel.
        if thresholds_by_band is not None:
            thr = thresholds_by_band[bi]
        else:
            thr = np.full(n_channels, np.nan, dtype=np.float64)
            for channel_index in range(n_channels):
                channel_baseline = baseline_mask_band[channel_index]
                if np.any(channel_baseline):
                    thr[channel_index] = float(
                        np.nanpercentile(
                            env[channel_index, channel_baseline],
                            percentile_eff,
                        )
                    )
        if np.any(~np.isfinite(thr)) or np.any(thr <= 0.0):
            invalid_channels = [
                str(ch_names[index])
                for index in np.flatnonzero(~np.isfinite(thr) | (thr <= 0.0))
            ]
            raise ValueError(
                "Burst thresholds must be finite and strictly positive for "
                f"band '{band_name}' and channel(s): "
                + ", ".join(invalid_channels)
                + "."
            )
        thresholds_used.append(thr)

        above = env > thr[:, None]

        # Edge samples are never considered bursts.
        if np.any(edge_mask):
            above[edge_mask] = False

        # Duration bounds in samples based on band *union* center frequency.
        min_len = min_run_samples_by_band[bi]
        max_len = max_run_samples_by_band[bi]

        burst_mask = np.zeros_like(above, dtype=bool)
        for ci in range(n_channels):
            if boundary_isolated_effective:
                for start, stop in analysis_interiors_by_channel[ci]:
                    burst_mask[ci, start:stop] = _filter_runs_by_length(
                        above[ci, start:stop],
                        min_len=min_len,
                        max_len=max_len,
                    )
            else:
                burst_mask[ci] = _filter_runs_by_length(
                    above[ci],
                    min_len=min_len,
                    max_len=max_len,
                )

        invalid_mask = ~np.isfinite(env)
        if np.any(edge_mask):
            invalid_mask |= edge_mask

        env_burst = np.zeros_like(env, dtype=np.float64)
        accepted_mask = burst_mask & ~invalid_mask
        normalized_magnitude = env / thr[:, None]
        env_burst[accepted_mask] = normalized_magnitude[accepted_mask]
        env_burst[invalid_mask] = np.nan

        # The native-rate contract requires decim_eff == 1 above.
        env_dec = env_burst[:, ::decim_eff]  # (n_channels, n_times_out)
        if target_n_times_eff is not None:
            if env_dec.shape[-1] > target_n_times_eff:
                env_dec = env_dec[..., :target_n_times_eff]
            elif env_dec.shape[-1] < target_n_times_eff:
                pad = np.full(
                    (n_channels, target_n_times_eff - env_dec.shape[-1]),
                    np.nan,
                    dtype=np.float64,
                )
                env_dec = np.concatenate([env_dec, pad], axis=-1)

        out_bands.append(env_dec)

    # Stack -> (n_channels, n_bands, n_times)
    out = np.stack(out_bands, axis=1).astype(np.float64, copy=False)
    out4d = out[None, ...]  # (1, ch, band, time)

    # Time axis (decimated)
    times_out = raw_times[::decim_eff]
    if target_n_times_eff is not None:
        if times_out.shape[0] > target_n_times_eff:
            times_out = times_out[:target_n_times_eff]
        elif times_out.shape[0] < target_n_times_eff:
            times_out = np.concatenate(
                [
                    times_out,
                    np.full(target_n_times_eff - times_out.shape[0], np.nan),
                ],
                axis=0,
            )

    thresholds_arr = np.stack(thresholds_used, axis=0)  # (n_bands, n_channels)

    metadata: Dict[str, Any] = dict(
        value_semantics=burst_value_semantics(),
        axes=dict(
            epoch=np.array([0], dtype=int),
            channel=np.array(ch_names, dtype=object),
            freq=list(band_names),
            time=np.asarray(times_out, dtype=float),
            shape=out4d.shape,
        ),
        params=dict(
            method=method_eff,
            estimator_signature=estimator_signature,
            bands_segments_hz={
                str(name): [[float(a), float(b)] for (a, b) in segs]
                for name, segs in zip(band_names, band_segments)
            },
            bands_union_hz={
                str(name): [
                    float(band_union_edges[i, 0]),
                    float(band_union_edges[i, 1]),
                ]
                for i, name in enumerate(band_names)
            },
            band_names=list(band_names),
            band_union_edges_hz=np.asarray(band_union_edges, dtype=float).tolist(),
            thresholds_provided=(thresholds is not None),
            percentile=percentile_eff,
            min_cycles=float(min_cycles),
            max_cycles=max_cycles_eff,
            min_run_samples_by_band=[int(x) for x in min_run_samples_by_band],
            max_run_samples_by_band=[
                int(x) if x is not None else None for x in max_run_samples_by_band
            ],
            hop_s=(float(hop_s) if hop_s is not None else None),
            decim_eff=int(decim_eff),
            target_n_times=target_n_times_eff,
            baseline_keep=(list(baseline_keep) if baseline_keep is not None else None),
            baseline_match=str(baseline_match),
            baseline_fallback=str(baseline_fallback),
            baseline_intervals=baseline_intervals,
            hilbert_filter_method=hilbert_filter_method_eff,
            filter_order=(
                int(filter_order)
                if method_eff == "hilbert" and hilbert_filter_method_eff == "iir"
                else None
            ),
            hilbert_fir_filter_specs_by_band={
                str(name): specifications
                for name, specifications in zip(
                    band_names,
                    hilbert_fir_filter_specs_by_band,
                )
                if specifications is not None
            },
            freq_step_hz=(
                float(freq_step_hz) if method_eff in {"morlet", "multitaper"} else None
            ),
            morlet_n_cycles=(
                float(morlet_n_cycles) if method_eff == "morlet" else None
            ),
            mt_n_cycles=(float(mt_n_cycles) if method_eff == "multitaper" else None),
            mt_time_bandwidth_product=(
                float(mt_time_bandwidth_product) if method_eff == "multitaper" else None
            ),
            hilbert_edge_tolerance_pct=tolerance_pct_eff,
            frequency_grid_hz_by_band={
                str(name): [float(value) for value in frequencies.tolist()]
                for name, frequencies in zip(band_names, frequency_grids_by_band)
            },
            interpolation_applied=False,
            edge_anno=(list(edge_anno_eff) if edge_anno_eff is not None else None),
            edge_match=str(mode),
            boundary_isolated_filter_requested=boundary_isolated_requested,
            boundary_isolated_filter_effective=boundary_isolated_effective,
            boundary_processing=(
                f"per_valid_segment_{method_eff}"
                if boundary_isolated_effective
                else f"continuous_{method_eff}_postmask"
            ),
            edge_intervals=[[float(a0), float(a1)] for (a0, a1) in edge_intervals],
            edge_intervals_dilated_by_band=[
                [[float(a0), float(a1)] for (a0, a1) in ints]
                for ints in edge_intervals_dilated_by_band
            ],
            edge_guard_samples_by_band=[int(x) for x in edge_guard_samples_by_band],
            edge_guard_seconds_by_band=[float(x) for x in edge_guard_seconds_by_band],
            edge_guard_details_by_band=edge_guard_details_by_band,
            annotation_scope_semantics=ANNOTATION_SCOPE_SEMANTICS,
            boundary_support_info=boundary_support_info,
            baseline_support_info=baseline_support_info,
            edge_mask_info_by_band=edge_mask_info_by_band,
        ),
        qc=dict(
            thresholds=thresholds_arr.astype(np.float64),
            baseline_coverage=(
                float(np.mean(baseline_mask_by_channel))
                if baseline_keep_eff is not None
                else None
            ),
            baseline_coverage_by_band=[float(x) for x in baseline_coverage_by_band],
            baseline_coverage_by_band_and_channel={
                "bands": list(band_names),
                "channels": [str(channel) for channel in ch_names],
                "values": baseline_coverage_by_band_and_channel,
            },
            edge_coverage_by_band=[float(x) for x in edge_coverage_by_band],
            bad_fraction_by_channel=[
                float(np.mean(boundary_invalid[index])) for index in range(n_channels)
            ],
            guard_fraction_by_band_channel=guard_fraction_by_band_channel,
            valid_fraction_by_band_channel=valid_fraction_by_band_channel,
            segment_count_by_band_channel=segment_count_by_band_channel,
            short_segment_count_by_band_channel=(short_segment_count_by_band_channel),
        ),
    )

    return out4d, metadata
