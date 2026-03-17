"""Compute Phase Slope Index (PSI) on a decimated time grid.

This module provides :func:`grid`, which returns a PSI tensor with shape:

    (1, n_pairs, n_bands, n_times)

The PSI is computed via :func:`mne_connectivity.phase_slope_index` using either:
  - ``method="morlet"`` -> ``mode="cwt_morlet"`` (native time-resolved PSI)
  - ``method="multitaper"`` -> ``mode="multitaper"`` (spectral-only PSI)

To match the TFR/connectivity time axis (typically decimated via
``hop_s``/``decim``):
  - Morlet path: decimate PSI output time axis.
  - Multitaper path: repeat spectral-only PSI across the decimated time axis
    so the returned tensor shape stays consistent with morlet.

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

import numpy as np
import mne

from ..common.timefreq import (
    channel_names_after_picks,
    compute_decimation,
    decimated_times_from_raw,
    morlet_n_cycles_from_time_fwhm,
)
from ..connectivity.selection import resolve_pairs

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


def _weighted_nanmean(
    values: np.ndarray, weights: np.ndarray, *, axis: int
) -> np.ndarray:
    """Weighted mean that ignores NaNs.

    Args:
        values: Array with NaNs.
        weights: 1D weights aligned with `axis`.
        axis: Axis to reduce.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    w = w / np.sum(w) if np.sum(w) > 0 else w
    w_shape = [1] * values.ndim
    w_shape[axis] = w.size
    w_b = w.reshape(w_shape)

    finite = np.isfinite(values)
    num = np.nansum(values * w_b, axis=axis)
    den = np.nansum(w_b * finite, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out = np.where(den > 0, out, np.nan)
    return out


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
    mt_bandwidth: float | None = None,
    block_size: int = 1000,
    n_jobs: int = 1,
    verbose: Any | None = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Compute time-resolved PSI aligned to a TFR-like time grid.

    Args:
        raw: Continuous MNE Raw.
        bands: Mapping band_name -> (fmin,fmax) in Hz.
        time_resolution_s: Target time-domain Morlet FWHM (seconds). Used to set
            `cwt_n_cycles` so wavelet temporal resolution is approximately constant.
        pairs: Optional explicit list of ordered pairs (seed, target).
        groups: Optional mapping group_name -> list[channel_name] to build pairs within groups.
        ordered_pairs: If True, build ordered pairs (A,B) and (B,A) where applicable.
        hop_s: Hop size (seconds) used to derive a decimation factor (TFR-like).
        decim: Explicit decimation factor overriding `hop_s`.
        target_n_times: Optional fixed time-axis length (pad/trim with NaNs).
        picks: Optional list of channel names to include.
        cwt_freqs: Optional explicit wavelet frequency grid used internally by PSI.
        min_cycles: Lower bound for Morlet cycles when converting from time FWHM.
        max_cycles: Optional upper bound for Morlet cycles.
        block_size: Forwarded to :func:`mne_connectivity.phase_slope_index`.
        n_jobs: Parallel jobs forwarded to PSI function.
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
    if mt_bandwidth is not None and float(mt_bandwidth) <= 0:
        raise ValueError("`mt_bandwidth` must be > 0 when provided.")

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

    # Optional: If the caller requests a shorter *time axis* than the raw would
    # naturally yield, we can crop the input to avoid unnecessary computation.
    #
    # We still compute PSI at the *original* sampling rate. Only the output is
    # time-decimated to align with TFR/connectivity grids.
    n_times_target = int(times.size)
    if target_n_times is not None and target_n_times > 0:
        # We need samples up to (target_n_times-1)*decim_eff.
        max_sample = int((int(target_n_times) - 1) * int(decim_eff) + 1)
        max_sample = min(max_sample, int(data.shape[-1]))
        data = data[:, :max_sample]

    data_3d = data[np.newaxis, :, :]
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

    # Compute PSI.
    try:
        from mne_connectivity.effective import phase_slope_index
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "mne_connectivity is required for PSI computation. "
            f"Import failed with {type(e).__name__}: {e}. "
            "Install with 'pip install mne-connectivity'."
        ) from e

    fmin_tuple = tuple(float(x) for x in seg_edges[:, 0])
    fmax_tuple = tuple(float(x) for x in seg_edges[:, 1])

    conn_kwargs: dict[str, Any] = dict(
        data=data_3d,
        indices=(seeds_idx, targets_idx),
        sfreq=float(sfreq_use),
        mode=mode_use,
        fmin=fmin_tuple,
        fmax=fmax_tuple,
        block_size=int(block_size),
        n_jobs=int(n_jobs),
        verbose=verbose,
    )
    if method_use == "morlet":
        conn_kwargs["cwt_freqs"] = cwt_freqs_use
        conn_kwargs["cwt_n_cycles"] = cwt_n_cycles
    elif mt_bandwidth is not None:
        conn_kwargs["mt_bandwidth"] = float(mt_bandwidth)

    conn = phase_slope_index(**conn_kwargs)
    psi_raw = np.asarray(conn.get_data(), dtype=float)
    if method_use == "morlet":
        expected_full = (len(pair_names), int(seg_edges.shape[0]), int(data.shape[-1]))
        if tuple(psi_raw.shape) != expected_full:
            raise RuntimeError(
                f"PSI data shape mismatch: got {psi_raw.shape}, expected {expected_full}."
            )
        psi_dec = psi_raw[:, :, :: int(decim_eff)]
        n_times_dec = int(psi_dec.shape[-1])
        n_times_compute = min(n_times_dec, n_times_target)
    else:
        expected_full = (len(pair_names), int(seg_edges.shape[0]))
        if tuple(psi_raw.shape) != expected_full:
            raise RuntimeError(
                f"PSI data shape mismatch: got {psi_raw.shape}, expected {expected_full}."
            )
        psi_dec = np.repeat(psi_raw[:, :, np.newaxis], n_times_target, axis=-1)
        n_times_compute = int(n_times_target)

    n_pairs = len(pair_names)
    # Combine segments back to original band axis.
    psi_bands = np.full((n_pairs, len(band_names), n_times_target), np.nan, dtype=float)

    # Precompute segment weights using the internal cwt frequency grid.
    seg_weights = np.zeros(seg_edges.shape[0], dtype=float)
    for si, (lo, hi) in enumerate(seg_edges):
        if method_use == "morlet" and cwt_freqs_use is not None:
            seg_weights[si] = float(
                np.sum((cwt_freqs_use >= lo) & (cwt_freqs_use <= hi))
            )
            if seg_weights[si] <= 0:
                seg_weights[si] = float(max(hi - lo, 1.0))
        else:
            seg_weights[si] = float(max(hi - lo, 1e-6))

    for bi in range(len(band_names)):
        seg_idx = np.flatnonzero(seg_to_band == bi)
        if seg_idx.size == 0:
            continue
        vals = psi_dec[:, seg_idx, :n_times_compute]
        w = seg_weights[seg_idx]
        psi_bands[:, bi, :n_times_compute] = _weighted_nanmean(vals, w, axis=1)

    out = psi_bands[np.newaxis, :, :, :]

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
            mt_bandwidth=(float(mt_bandwidth) if mt_bandwidth is not None else None),
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

    return out, metadata
