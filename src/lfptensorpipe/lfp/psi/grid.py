"""Compute Phase Slope Index (PSI) on a decimated time grid.

This module provides :func:`grid`, which returns a PSI tensor with shape:

    (1, n_pairs, n_bands, n_times)

The PSI is computed via :func:`mne_connectivity.phase_slope_index` using either:
  - ``method="morlet"`` -> ``mode="cwt_morlet"`` (native time-resolved PSI)
  - ``method="multitaper"`` -> one spectral PSI estimate per centered window

To match the TFR/connectivity time axis (typically decimated via
``hop_s``/``decim``):
  - Morlet path: decimate PSI output time axis.
  - Multitaper path: center one analysis window on each output time point and
    leave positions without a complete window as NaN.

Important: the input signal is **not** downsampled. This keeps the effective
sampling rate (and Nyquist frequency) unchanged, so high-frequency bands (e.g.
gamma up to 100 Hz) remain valid as long as ``raw.info['sfreq']`` supports them.

Notes:
  - PSI is inherently directed, so ordered pairs are often the right choice.
  - Frequency "bands" are passed via (fmin,fmax) tuples; the returned "freq" axis
    in metadata stores **band names** (strings) for compatibility with other
    non-numeric frequency-like axes (e.g., SpecParam parameter names).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple
import warnings

import numpy as np
import mne
from joblib import Parallel, delayed

from ..common.timefreq import (
    channel_names_after_picks,
    compute_decimation,
    decimated_times_from_raw,
    morlet_n_cycles_from_time_fwhm,
    multitaper_fixed_p_parameters,
    multitaper_window_geometry,
)
from ..connectivity.selection import resolve_pairs
from ..runtime.tensor_helpers import build_annotation_skip_time_mask

BandValue = Tuple[float, float]
BandValueOrSegments = BandValue | list[BandValue]
BandSpec = Mapping[str, BandValueOrSegments]


def _normalize_bands(
    bands: BandSpec,
) -> tuple[list[str], list[list[BandValue]], np.ndarray, np.ndarray, np.ndarray]:
    """Normalize band specs.

    Supports either:
      - name -> (fmin,fmax)
      - name -> [(fmin,fmax), (fmin2,fmax2), ...]

    Returns:
        band_names: list[str]
        band_segments: list[list[(fmin,fmax)]] aligned with band_names
        seg_edges: array (n_segments,2)
        seg_to_band: array (n_segments,) with band indices
        band_union_edges: array (n_bands,2) with (min_lo,max_hi) per band
    """
    if not isinstance(bands, Mapping) or len(bands) == 0:
        raise ValueError(
            "`bands` must be a non-empty mapping name -> (fmin,fmax) or list[(fmin,fmax)]."
        )

    band_names: list[str] = []
    band_segments: list[list[BandValue]] = []

    seg_edges: list[tuple[float, float]] = []
    seg_to_band: list[int] = []
    union_edges: list[tuple[float, float]] = []

    for band_i, (name, spec) in enumerate(bands.items()):
        bname = str(name)
        if (
            isinstance(spec, (tuple, list))
            and len(spec) == 2
            and not isinstance(spec[0], (tuple, list))
        ):
            segments: list[BandValue] = [(float(spec[0]), float(spec[1]))]  # type: ignore[arg-type]
        elif isinstance(spec, list):
            segments = [(float(a), float(b)) for (a, b) in spec]
        else:
            raise ValueError(
                f"Band '{bname}' must be (fmin,fmax) or list[(fmin,fmax)]."
            )

        if len(segments) == 0:
            raise ValueError(f"Band '{bname}' has no segments.")

        for a, b in segments:
            if not np.isfinite(a) or not np.isfinite(b):
                raise ValueError(f"Band '{bname}' has non-finite bounds: {(a, b)}")
            if a <= 0 or b <= 0:
                raise ValueError(f"Band '{bname}' bounds must be > 0 Hz, got {(a, b)}")
            if b <= a:
                raise ValueError(
                    f"Band '{bname}' must satisfy fmax > fmin, got {(a, b)}"
                )
            seg_edges.append((a, b))
            seg_to_band.append(band_i)

        lo = min(a for a, _ in segments)
        hi = max(b for _, b in segments)
        union_edges.append((lo, hi))

        band_names.append(bname)
        band_segments.append(segments)

    seg_edges_arr = np.asarray(seg_edges, dtype=float)
    seg_to_band_arr = np.asarray(seg_to_band, dtype=int)
    union_edges_arr = np.asarray(union_edges, dtype=float)
    return band_names, band_segments, seg_edges_arr, seg_to_band_arr, union_edges_arr


def _segment_frequency_support(
    seg_edges: np.ndarray,
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Count strict-internal frequencies and adjacent PSI terms per segment."""
    freqs = np.asarray(frequencies_hz, dtype=float)
    counts = np.asarray(
        [
            int(np.sum((freqs > float(low)) & (freqs < float(high))))
            for low, high in np.asarray(seg_edges, dtype=float)
        ],
        dtype=int,
    )
    return counts, np.maximum(counts - 1, 0)


def _validate_band_frequency_support(
    *,
    method: str,
    band_names: list[str],
    seg_edges: np.ndarray,
    seg_to_band: np.ndarray,
    n_internal_frequencies: np.ndarray,
    n_adjacent_pairs: np.ndarray,
) -> None:
    unsupported_bands: list[str] = []
    for band_i, band_name in enumerate(band_names):
        segment_indices = np.flatnonzero(seg_to_band == band_i)
        if bool(np.any(n_adjacent_pairs[segment_indices] > 0)):
            continue
        segment_details = ", ".join(
            f"[{float(seg_edges[index, 0]):g}, {float(seg_edges[index, 1]):g}] Hz: "
            f"{int(n_internal_frequencies[index])} internal frequencies, "
            f"{int(n_adjacent_pairs[index])} adjacent pairs"
            for index in segment_indices
        )
        unsupported_bands.append(f"'{band_name}' ({segment_details})")

    if not unsupported_bands:
        return
    advice = (
        "Decrease Step (Hz) or widen the band segment."
        if method == "morlet"
        else "Increase time_resolution_s or widen the band segment."
    )
    raise ValueError(
        f"{method.capitalize()} PSI requires at least two frequencies strictly "
        "inside at least one segment of every band. Unsupported band(s): "
        + "; ".join(unsupported_bands)
        + f". {advice}"
    )


def _partial_frequency_support_message(
    *,
    band_names: list[str],
    seg_edges: np.ndarray,
    seg_to_band: np.ndarray,
    n_internal_frequencies: np.ndarray,
    n_adjacent_pairs: np.ndarray,
) -> str | None:
    dropped: list[str] = []
    for segment_index in np.flatnonzero(n_adjacent_pairs == 0):
        band_name = band_names[int(seg_to_band[segment_index])]
        dropped.append(
            f"'{band_name}' [{float(seg_edges[segment_index, 0]):g}, "
            f"{float(seg_edges[segment_index, 1]):g}] Hz "
            f"({int(n_internal_frequencies[segment_index])} internal frequencies, "
            f"{int(n_adjacent_pairs[segment_index])} adjacent pairs)"
        )
    if not dropped:
        return None
    return (
        "PSI uses partial frequency support; dropped segment(s) without an "
        "adjacent frequency pair: " + "; ".join(dropped) + "."
    )


def _frequency_support_by_band(
    *,
    band_names: list[str],
    seg_edges: np.ndarray,
    seg_to_band: np.ndarray,
    n_internal_frequencies: np.ndarray,
    n_adjacent_pairs: np.ndarray,
) -> dict[str, dict[str, list[Any]]]:
    support: dict[str, dict[str, list[Any]]] = {}
    for band_i, band_name in enumerate(band_names):
        segment_indices = np.flatnonzero(seg_to_band == band_i)
        configured = [
            [float(seg_edges[index, 0]), float(seg_edges[index, 1])]
            for index in segment_indices
        ]
        used = [
            [float(seg_edges[index, 0]), float(seg_edges[index, 1])]
            for index in segment_indices
            if int(n_adjacent_pairs[index]) > 0
        ]
        dropped = [
            [float(seg_edges[index, 0]), float(seg_edges[index, 1])]
            for index in segment_indices
            if int(n_adjacent_pairs[index]) == 0
        ]
        support[str(band_name)] = {
            "configured_segments_hz": configured,
            "used_segments_hz": used,
            "dropped_segments_hz": dropped,
            "n_internal_frequencies_by_segment": [
                int(n_internal_frequencies[index]) for index in segment_indices
            ],
            "n_adjacent_pairs_by_segment": [
                int(n_adjacent_pairs[index]) for index in segment_indices
            ],
        }
    return support


def _default_cwt_freqs(band_edges: np.ndarray) -> np.ndarray:
    """Create a simple 1-Hz grid covering all bands."""
    fmin_all = float(np.min(band_edges[:, 0]))
    fmax_all = float(np.max(band_edges[:, 1]))

    # 1 Hz steps are usually fine for typical neuroscience bands (delta..gamma).
    start = max(0.1, np.floor(fmin_all))
    stop = np.ceil(fmax_all)
    freqs = np.arange(start, stop + 1.0, 1.0, dtype=float)

    # Ensure strictly positive.
    freqs = freqs[freqs > 0]
    if freqs.size < 2:
        raise ValueError(
            "Could not build a valid `cwt_freqs` grid (need at least 2 frequencies). "
            "Provide `cwt_freqs` explicitly."
        )
    return freqs


def _normalize_method(method: str) -> str:
    token = str(method).strip().lower()
    if token in {"morlet", "cwt_morlet", "cwt"}:
        return "morlet"
    if token in {"multitaper", "mt"}:
        return "multitaper"
    raise ValueError("`method` must be 'morlet' or 'multitaper'.")


def _run_multitaper_windows(
    *,
    data: np.ndarray,
    center_samples: np.ndarray,
    valid_time_indices: np.ndarray,
    half_window_samples: int,
    seeds_idx: np.ndarray,
    targets_idx: np.ndarray,
    seg_edges: np.ndarray,
    sfreq_hz: float,
    mt_bandwidth: float,
    block_size: int,
    outer_n_jobs: int,
    verbose: Any | None,
    phase_slope_index_fn: Any,
) -> np.ndarray:
    n_pairs = int(len(seeds_idx))
    n_segments = int(seg_edges.shape[0])
    out = np.full(
        (n_pairs, n_segments, int(center_samples.size)),
        np.nan,
        dtype=float,
    )
    fmin_tuple = tuple(float(item) for item in seg_edges[:, 0])
    fmax_tuple = tuple(float(item) for item in seg_edges[:, 1])

    def _run_one(time_index: int) -> tuple[int, np.ndarray]:
        center = int(center_samples[time_index])
        start = center - int(half_window_samples)
        stop = center + int(half_window_samples) + 1
        window = np.asarray(data[:, start:stop], dtype=float)[np.newaxis, ...]
        kwargs: dict[str, Any] = {
            "data": window,
            "indices": (seeds_idx, targets_idx),
            "sfreq": float(sfreq_hz),
            "mode": "multitaper",
            "fmin": fmin_tuple,
            "fmax": fmax_tuple,
            "block_size": int(block_size),
            "n_jobs": 1,
            "verbose": verbose,
        }
        kwargs["mt_bandwidth"] = float(mt_bandwidth)
        conn = phase_slope_index_fn(**kwargs)
        values = np.asarray(conn.get_data(), dtype=float)
        expected = (n_pairs, n_segments)
        if tuple(values.shape) != expected:
            raise RuntimeError(
                "Multitaper PSI window shape mismatch: "
                f"got {values.shape}, expected {expected}."
            )
        return int(time_index), values

    indices = [int(item) for item in np.asarray(valid_time_indices, dtype=int)]
    if int(outer_n_jobs) == 1:
        results = [_run_one(time_index) for time_index in indices]
    else:
        results = Parallel(
            n_jobs=int(outer_n_jobs),
            backend="loky",
            max_nbytes="100K",
        )(delayed(_run_one)(time_index) for time_index in indices)
    for time_index, values in results:
        out[:, :, time_index] = values
    return out


def grid(
    raw: mne.io.BaseRaw,
    *,
    bands: BandSpec,
    method: str = "morlet",
    time_resolution_s: float,
    pairs: Sequence[Tuple[str, str]] | None = None,
    groups: Dict[str, Sequence[str]] | None = None,
    ordered_pairs: bool = True,
    hop_s: float | None = 0.025,
    decim: int | None = None,
    target_n_times: int | None = None,
    picks: list[str] | None = None,
    cwt_freqs: np.ndarray | None = None,
    min_cycles: float | None = 1.0,
    max_cycles: float | None = None,
    mt_time_bandwidth_product: float = 4.0,
    mt_min_cycles: float = 3.0,
    mt_max_cycles: float | None = None,
    block_size: int = 1000,
    n_jobs: int = 1,
    outer_n_jobs: int | None = None,
    verbose: Any | None = None,
    mask_annotations: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Compute time-resolved PSI aligned to a TFR-like time grid.

    Args:
        raw: Continuous MNE Raw.
        bands: Mapping band_name -> (fmin,fmax) in Hz.
        time_resolution_s: Target Morlet time-domain FWHM, or minimum centered
            Multitaper analysis-window duration, in seconds.
        pairs: Optional explicit list of ordered pairs (seed, target).
        groups: Optional mapping group_name -> list[channel_name] to build pairs within groups.
        ordered_pairs: If True, build ordered pairs (A,B) and (B,A) where applicable.
        hop_s: Hop size used to derive the output decimation factor and
            Multitaper window-center spacing, in seconds.
        decim: Explicit decimation factor overriding `hop_s`.
        target_n_times: Optional fixed time-axis length (pad/trim with NaNs).
        picks: Optional list of channel names to include.
        cwt_freqs: Optional explicit wavelet frequency grid used internally by PSI.
        min_cycles: Lower bound for Morlet cycles when converting from time FWHM.
        max_cycles: Optional upper bound for Morlet cycles.
        mt_time_bandwidth_product: Dimensionless DPSS time-bandwidth product.
        mt_min_cycles: Minimum cycles in a Multitaper band window.
        mt_max_cycles: Optional maximum cycles in a Multitaper band window.
        block_size: Forwarded to :func:`mne_connectivity.phase_slope_index`.
        n_jobs: Parallel jobs forwarded to Morlet PSI. For Multitaper, this is
            the window-loop fallback when `outer_n_jobs` is None.
        outer_n_jobs: Parallel jobs across Multitaper windows. None uses
            `n_jobs`; each individual window keeps its inner PSI job count at 1.
        verbose: Verbosity forwarded to PSI function.

    Returns:
        psi_tensor: float array with shape (1, n_pairs, n_bands, n_times).
        metadata: Dict with axes + params.

    Raises:
        ModuleNotFoundError: If `mne_connectivity` is not installed.
    """
    if float(time_resolution_s) <= 0:
        raise ValueError("`time_resolution_s` must be > 0.")
    method_use = _normalize_method(method)
    mode_use = "cwt_morlet" if method_use == "morlet" else "multitaper"
    resolved_outer_n_jobs = int(n_jobs) if outer_n_jobs is None else int(outer_n_jobs)
    band_names, band_segments, seg_edges, seg_to_band, union_edges = _normalize_bands(
        bands
    )

    sfreq = float(raw.info["sfreq"])
    if sfreq <= 0:
        raise ValueError("Raw sampling rate must be > 0.")

    decim_eff, hop_s_eff = compute_decimation(sfreq, hop_s=hop_s, decim=decim)
    times = decimated_times_from_raw(
        raw, decim=decim_eff, target_n_times=target_n_times
    )

    # Pick channels (order must match the actual data extraction).
    ch_names = channel_names_after_picks(raw, picks)
    data = raw.get_data(picks=picks)
    if data.ndim != 2:
        raise RuntimeError(
            "Expected Raw.get_data() to return a 2D array (n_channels, n_times)."
        )

    # Resolve pairs from the picked channel list.
    seeds_idx, targets_idx, pair_names, pair_meta = resolve_pairs(
        ch_names,
        pairs=pairs,
        groups=groups,
        ordered_pairs=bool(ordered_pairs),
    )

    n_times_target = int(times.size)
    sfreq_use = sfreq

    # Nyquist guard.
    nyquist = sfreq_use / 2.0
    if float(np.max(seg_edges[:, 1])) > nyquist:
        raise ValueError(
            "Band upper edges exceed Nyquist. "
            f"max(fmax)={float(np.max(seg_edges[:, 1])):.3f} Hz, Nyquist={nyquist:.3f} Hz. "
            "Either lower your bands or use a Raw with a higher sampling rate."
        )
    cwt_freqs_use: np.ndarray | None = None
    cwt_n_cycles: np.ndarray | None = None
    mt_reference_frequencies_hz: np.ndarray | None = None
    mt_effective_window_s: np.ndarray | None = None
    mt_effective_bandwidth_hz: np.ndarray | None = None
    mt_mask_radius_s: np.ndarray | None = None
    mt_half_window_samples: list[int] | None = None
    mt_window_n_samples: list[int] | None = None
    mt_window_span_s: list[float] | None = None
    n_internal_frequencies = np.zeros(seg_edges.shape[0], dtype=int)
    n_adjacent_pairs = np.zeros(seg_edges.shape[0], dtype=int)
    if method_use == "morlet":
        cwt_freqs_use = (
            _default_cwt_freqs(seg_edges)
            if cwt_freqs is None
            else np.asarray(cwt_freqs, dtype=float)
        )
        if cwt_freqs_use.ndim != 1 or cwt_freqs_use.size < 2:
            raise ValueError(
                "`cwt_freqs` must be a 1D array with at least 2 frequencies."
            )
        if np.any(~np.isfinite(cwt_freqs_use)) or np.any(cwt_freqs_use <= 0):
            raise ValueError("`cwt_freqs` must be finite and > 0.")
        if float(np.max(cwt_freqs_use)) > nyquist:
            raise ValueError(
                "entries in cwt_freqs cannot be larger than Nyquist (sfreq / 2). "
                f"max(cwt_freqs)={float(np.max(cwt_freqs_use)):.3f} Hz, Nyquist={nyquist:.3f} Hz."
            )

        cwt_n_cycles = morlet_n_cycles_from_time_fwhm(
            cwt_freqs_use,
            time_fwhm_s=float(time_resolution_s),
            min_cycles=min_cycles,
            max_cycles=max_cycles,
        )
        n_internal_frequencies, n_adjacent_pairs = _segment_frequency_support(
            seg_edges,
            cwt_freqs_use,
        )
    else:
        mt_reference_frequencies_hz = np.asarray(
            [
                float(np.min(seg_edges[np.flatnonzero(seg_to_band == band_i), 0]))
                for band_i in range(len(band_names))
            ],
            dtype=float,
        )
        (
            _,
            mt_effective_window_s,
            mt_effective_bandwidth_hz,
            mt_mask_radius_s,
        ) = multitaper_fixed_p_parameters(
            mt_reference_frequencies_hz,
            time_resolution_s=float(time_resolution_s),
            mt_time_bandwidth_product=float(mt_time_bandwidth_product),
            mt_min_cycles=float(mt_min_cycles),
            mt_max_cycles=(float(mt_max_cycles) if mt_max_cycles is not None else None),
        )

        available_duration_s = float(max(0, data.shape[-1] - 1)) / float(sfreq_use)
        longest_band_index = int(np.argmax(mt_effective_window_s))
        if float(mt_effective_window_s[longest_band_index]) > available_duration_s:
            raise ValueError(
                "Effective Multitaper PSI window for band "
                f"'{band_names[longest_band_index]}' at "
                f"{float(mt_reference_frequencies_hz[longest_band_index]):g} Hz is "
                f"{float(mt_effective_window_s[longest_band_index]):g} s, longer "
                f"than the available duration {available_duration_s:g} s."
            )

        n_bands = len(band_names)
        mt_half_window_samples = [0] * n_bands
        mt_window_n_samples = [0] * n_bands
        mt_window_span_s = [0.0] * n_bands
        for band_i in range(n_bands):
            half_window_samples, window_n_samples, window_span_s = (
                multitaper_window_geometry(
                    sfreq_hz=float(sfreq_use),
                    time_resolution_s=float(mt_effective_window_s[band_i]),
                )
            )
            mt_half_window_samples[band_i] = int(half_window_samples)
            mt_window_n_samples[band_i] = int(window_n_samples)
            mt_window_span_s[band_i] = float(window_span_s)
            segment_indices = np.flatnonzero(seg_to_band == band_i)
            fourier_freqs = np.fft.rfftfreq(
                int(window_n_samples),
                d=1.0 / float(sfreq_use),
            )
            internal_counts, adjacent_counts = _segment_frequency_support(
                seg_edges[segment_indices],
                fourier_freqs,
            )
            n_internal_frequencies[segment_indices] = internal_counts
            n_adjacent_pairs[segment_indices] = adjacent_counts

    _validate_band_frequency_support(
        method=method_use,
        band_names=band_names,
        seg_edges=seg_edges,
        seg_to_band=seg_to_band,
        n_internal_frequencies=n_internal_frequencies,
        n_adjacent_pairs=n_adjacent_pairs,
    )
    used_segment_mask = n_adjacent_pairs > 0
    used_segment_indices = np.flatnonzero(used_segment_mask)
    partial_support_message = _partial_frequency_support_message(
        band_names=band_names,
        seg_edges=seg_edges,
        seg_to_band=seg_to_band,
        n_internal_frequencies=n_internal_frequencies,
        n_adjacent_pairs=n_adjacent_pairs,
    )

    # Compute PSI.
    try:
        from mne_connectivity.effective import phase_slope_index
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "mne_connectivity is required for PSI computation. "
            f"Import failed with {type(e).__name__}: {e}. "
            "Install with 'pip install mne-connectivity'."
        ) from e

    mt_n_valid_windows_by_band: list[int] | None = None
    mt_n_skipped_masked_by_band: list[int] | None = None
    mt_n_dropped_incomplete_by_band: list[int] | None = None
    n_valid_windows: int | None = None
    n_columns_total: int | None = None
    n_columns_skipped_masked: int | None = None
    n_columns_dropped_incomplete_window: int | None = None
    if method_use == "morlet":
        data_compute = data
        if target_n_times is not None and target_n_times > 0:
            max_sample = int((int(target_n_times) - 1) * int(decim_eff) + 1)
            max_sample = min(max_sample, int(data.shape[-1]))
            data_compute = data[:, :max_sample]

        used_segment_edges = seg_edges[used_segment_indices]
        fmin_tuple = tuple(float(x) for x in used_segment_edges[:, 0])
        fmax_tuple = tuple(float(x) for x in used_segment_edges[:, 1])
        conn_kwargs: dict[str, Any] = dict(
            data=data_compute[np.newaxis, :, :],
            indices=(seeds_idx, targets_idx),
            sfreq=float(sfreq_use),
            mode=mode_use,
            fmin=fmin_tuple,
            fmax=fmax_tuple,
            block_size=int(block_size),
            n_jobs=int(n_jobs),
            verbose=verbose,
        )
        conn_kwargs["cwt_freqs"] = cwt_freqs_use
        conn_kwargs["cwt_n_cycles"] = cwt_n_cycles
        conn = phase_slope_index(**conn_kwargs)
        psi_raw = np.asarray(conn.get_data(), dtype=float)
        expected_full = (
            len(pair_names),
            int(used_segment_edges.shape[0]),
            int(data_compute.shape[-1]),
        )
        if tuple(psi_raw.shape) != expected_full:
            raise RuntimeError(
                f"PSI data shape mismatch: got {psi_raw.shape}, expected {expected_full}."
            )
        psi_dec = np.full(
            (len(pair_names), int(seg_edges.shape[0]), n_times_target),
            np.nan,
            dtype=float,
        )
        psi_used_dec = psi_raw[:, :, :: int(decim_eff)]
        n_times_dec = int(psi_used_dec.shape[-1])
        n_times_compute = min(n_times_dec, n_times_target)
        psi_dec[:, used_segment_indices, :n_times_compute] = psi_used_dec[
            :, :, :n_times_compute
        ]
    else:
        assert mt_effective_window_s is not None
        assert mt_effective_bandwidth_hz is not None
        assert mt_mask_radius_s is not None
        assert mt_half_window_samples is not None
        assert mt_window_n_samples is not None
        assert mt_window_span_s is not None
        center_samples = np.arange(n_times_target, dtype=int) * int(decim_eff)
        finite_target_times = np.isfinite(times)
        n_columns_total = int(np.sum(finite_target_times))
        psi_dec = np.full(
            (len(pair_names), int(seg_edges.shape[0]), n_times_target),
            np.nan,
            dtype=float,
        )
        n_bands = len(band_names)
        mt_n_valid_windows_by_band = [0] * n_bands
        mt_n_skipped_masked_by_band = [0] * n_bands
        mt_n_dropped_incomplete_by_band = [0] * n_bands

        # Bands whose effective window is identical share one estimator geometry
        # (window samples, bandwidth, and mask radius all derive from it), so they
        # can be computed in one backend call per output center instead of one per
        # band. Grouping on the exact float is conservative: a spurious split only
        # costs extra calls, it never mixes different estimators.
        window_groups: dict[float, list[int]] = {}
        for band_i in range(n_bands):
            window_groups.setdefault(float(mt_effective_window_s[band_i]), []).append(
                band_i
            )

        for window_value, group_band_indices in window_groups.items():
            group_labels = ", ".join(
                f"'{band_names[band_i]}'" for band_i in group_band_indices
            )
            configured_segment_indices = np.flatnonzero(
                np.isin(seg_to_band, np.asarray(group_band_indices, dtype=int))
            )
            segment_indices = configured_segment_indices[
                used_segment_mask[configured_segment_indices]
            ]
            group_segment_edges = seg_edges[segment_indices]
            first_band_index = group_band_indices[0]
            half_window_samples = mt_half_window_samples[first_band_index]
            complete_windows = (
                finite_target_times
                & (center_samples - int(half_window_samples) >= 0)
                & (center_samples + int(half_window_samples) < int(data.shape[-1]))
            )
            if not bool(np.any(complete_windows)):
                raise ValueError(
                    "Multitaper PSI has no complete centered analysis window for "
                    f"band {group_labels} with required window {window_value:g} s."
                )

            if not mask_annotations:
                skip_time_mask = np.zeros(times.shape, dtype=bool)
            else:
                skip_time_mask = build_annotation_skip_time_mask(
                    raw,
                    times_s=times,
                    radius_s=float(mt_mask_radius_s[group_band_indices[0]]),
                    output_channels=pair_names,
                )
            valid_time_indices = np.flatnonzero(complete_windows & ~skip_time_mask)
            if valid_time_indices.size == 0:
                warnings.warn(
                    f"Multitaper PSI band {group_labels} has no usable output centers after BAD/EDGE masking; values remain NaN.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                group_values = _run_multitaper_windows(
                    data=data,
                    center_samples=center_samples,
                    valid_time_indices=valid_time_indices,
                    half_window_samples=int(half_window_samples),
                    seeds_idx=seeds_idx,
                    targets_idx=targets_idx,
                    seg_edges=group_segment_edges,
                    sfreq_hz=float(sfreq_use),
                    mt_bandwidth=float(
                        mt_effective_bandwidth_hz[group_band_indices[0]]
                    ),
                    block_size=int(block_size),
                    outer_n_jobs=int(resolved_outer_n_jobs),
                    verbose=verbose,
                    phase_slope_index_fn=phase_slope_index,
                )
                psi_dec[:, segment_indices, :] = group_values

            n_skipped_masked = int(np.sum(finite_target_times & skip_time_mask))
            n_dropped_incomplete = int(
                np.sum(finite_target_times & ~skip_time_mask & ~complete_windows)
            )
            for band_i in group_band_indices:
                mt_n_valid_windows_by_band[band_i] = int(valid_time_indices.size)
                mt_n_skipped_masked_by_band[band_i] = n_skipped_masked
                mt_n_dropped_incomplete_by_band[band_i] = n_dropped_incomplete

        n_valid_windows = int(min(mt_n_valid_windows_by_band))
        n_columns_skipped_masked = int(max(mt_n_skipped_masked_by_band))
        n_columns_dropped_incomplete_window = int(max(mt_n_dropped_incomplete_by_band))
        n_times_compute = int(n_times_target)

    n_pairs = len(pair_names)
    # Combine segments back to original band axis.
    psi_bands = np.full((n_pairs, len(band_names), n_times_target), np.nan, dtype=float)

    for bi in range(len(band_names)):
        configured_seg_idx = np.flatnonzero(seg_to_band == bi)
        seg_idx = configured_seg_idx[used_segment_mask[configured_seg_idx]]
        vals = psi_dec[:, seg_idx, :n_times_compute]
        finite_cells = np.all(np.isfinite(vals), axis=1)
        combined = np.sum(np.where(np.isfinite(vals), vals, 0.0), axis=1)
        combined[~finite_cells] = np.nan
        psi_bands[:, bi, :n_times_compute] = combined

    out = psi_bands[np.newaxis, :, :, :]
    frequency_support = _frequency_support_by_band(
        band_names=band_names,
        seg_edges=seg_edges,
        seg_to_band=seg_to_band,
        n_internal_frequencies=n_internal_frequencies,
        n_adjacent_pairs=n_adjacent_pairs,
    )

    metadata: Dict[str, Any] = dict(
        axes=dict(
            epoch=np.arange(1, dtype=int),
            channel=list(pair_names),
            freq=list(band_names),
            time=np.asarray(times, dtype=float),
            shape=out.shape,
        ),
        params=dict(
            bands_segments_hz={
                str(name): [[float(a), float(b)] for (a, b) in segs]
                for name, segs in zip(band_names, band_segments)
            },
            bands_union_hz={
                str(name): [float(union_edges[i, 0]), float(union_edges[i, 1])]
                for i, name in enumerate(band_names)
            },
            band_names=list(band_names),
            segments_flat_hz=np.asarray(seg_edges, dtype=float).tolist(),
            segments_to_band=np.asarray(seg_to_band, dtype=int).tolist(),
            partial_frequency_support=partial_support_message is not None,
            frequency_support_by_band=frequency_support,
            time_resolution_s=float(time_resolution_s),
            hop_s=hop_s,
            decim=int(decim_eff),
            hop_s_eff=hop_s_eff,
            target_n_times=target_n_times,
            picks=picks,
            ordered_pairs=bool(ordered_pairs),
            **pair_meta,
            method=str(method_use),
            spectral_mode=str(mode_use),
            mt_time_bandwidth_product=(
                float(mt_time_bandwidth_product) if method_use == "multitaper" else None
            ),
            mt_min_cycles=(
                float(mt_min_cycles) if method_use == "multitaper" else None
            ),
            mt_max_cycles=(
                float(mt_max_cycles)
                if method_use == "multitaper" and mt_max_cycles is not None
                else None
            ),
            mt_effective_window_s=mt_effective_window_s,
            mt_effective_bandwidth_hz=mt_effective_bandwidth_hz,
            mt_window_reference_frequency_hz=mt_reference_frequencies_hz,
            **(
                {
                    "time_axis_mode": "sliding_window",
                    "multitaper_half_window_samples_by_band": mt_half_window_samples,
                    "multitaper_window_n_samples_by_band": mt_window_n_samples,
                    "multitaper_window_span_s_by_band": mt_window_span_s,
                    "multitaper_n_valid_windows": n_valid_windows,
                    "multitaper_n_valid_windows_by_band": mt_n_valid_windows_by_band,
                    "annotation_skip_enabled": bool(mask_annotations),
                    "n_columns_total": n_columns_total,
                    "n_columns_skipped_masked": n_columns_skipped_masked,
                    "n_columns_skipped_masked_by_band": mt_n_skipped_masked_by_band,
                    "n_columns_dropped_incomplete_window": (
                        n_columns_dropped_incomplete_window
                    ),
                    "n_columns_dropped_incomplete_window_by_band": (
                        mt_n_dropped_incomplete_by_band
                    ),
                    "outer_n_jobs": int(resolved_outer_n_jobs),
                }
                if method_use == "multitaper"
                else {}
            ),
            cwt_freqs=(
                np.asarray(cwt_freqs_use, dtype=float)
                if cwt_freqs_use is not None
                else None
            ),
            cwt_n_cycles=(
                np.asarray(cwt_n_cycles, dtype=float)
                if cwt_n_cycles is not None
                else None
            ),
            block_size=int(block_size),
            n_jobs=int(n_jobs),
            sfreq_used_hz=float(sfreq_use),
        ),
    )

    if partial_support_message is not None:
        warnings.warn(
            partial_support_message,
            UserWarning,
            stacklevel=2,
        )

    return out, metadata
