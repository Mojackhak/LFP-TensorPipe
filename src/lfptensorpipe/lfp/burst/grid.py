"""
Burst detection on an MNE Raw object with output aligned to the TFR time grid.

Algorithm (per channel, per band):
1) Band-pass filter the Raw signal within the band.
2) Compute analytic signal via Hilbert transform and take the amplitude envelope.
3) Compute a percentile threshold (default 75th) on the envelope.
   - If `baseline_keep` is provided, the threshold is computed ONLY from the samples
     covered by matching Raw annotations (e.g., baseline "sit"), and then applied
     to the full recording.
4) Detect supra-threshold contiguous segments and prune segments shorter than
   `min_cycles` periods of the band center frequency.
5) Return a tensor on the decimated time grid (same decimation rule as `tfr.grid`):
   values are the envelope amplitude, and non-burst time bins are NaN.

The output is designed to have the same number of time bins (`target_n_times`) as
TFR/connectivity/PSI when the same `hop_s/decim/target_n_times` are used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import numpy as np
import mne
from scipy.signal import hilbert


Band = Tuple[float, float]
BandValueOrSegments = Band | list[Band]
BandSpec = Mapping[str, BandValueOrSegments]


MatchMode = Literal["substring", "exact"]
EdgeGuard = float | Literal["auto"]


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
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and not isinstance(spec[0], (tuple, list)):
            segs: list[Band] = [(float(spec[0]), float(spec[1]))]  # type: ignore[arg-type]
        elif isinstance(spec, list):
            segs = [(float(a), float(b)) for (a, b) in spec]
        else:
            raise ValueError(f"Band '{bname}' must be (fmin,fmax) or list[(fmin,fmax)].")

        if len(segs) == 0:
            raise ValueError(f"Band '{bname}' has no segments.")

        for (a, b) in segs:
            if not (np.isfinite(a) and np.isfinite(b)):
                raise ValueError(f"Band '{bname}' has non-finite bounds: {(a, b)}")
            if a <= 0 or b <= 0:
                raise ValueError(f"Band '{bname}' bounds must be > 0 Hz, got {(a, b)}")
            if b <= a:
                raise ValueError(f"Band '{bname}' must satisfy fmax > fmin, got {(a, b)}")

        lo = min(a for a, _ in segs)
        hi = max(b for _, b in segs)
        band_names.append(bname)
        band_segments.append(segs)
        union_edges.append((lo, hi))

    return band_names, band_segments, np.asarray(union_edges, dtype=float)


def _compute_decim(sfreq_hz: float, hop_s: float | None, decim: int | None) -> int:
    if decim is not None:
        if int(decim) <= 0:
            raise ValueError("decim must be a positive integer.")
        return int(decim)
    if hop_s is None or float(hop_s) <= 0:
        raise ValueError("Provide hop_s>0 or an explicit decim to define the time grid.")
    return max(1, int(round(float(sfreq_hz) * float(hop_s))))


def _time_to_sample_idx(t_s: float, sfreq_hz: float, t0_s: float) -> int:
    return int(np.round((float(t_s) - float(t0_s)) * float(sfreq_hz)))


def _annotation_intervals(
    raw: mne.io.BaseRaw,
    keep: Sequence[str],
    *,
    match: MatchMode = "substring",
) -> List[Tuple[float, float]]:
    """
    Collect [start, end] intervals (seconds) for annotations matching `keep`.

    match:
        - "substring": keep label matches if label is contained in description (case-insensitive)
        - "exact": keep label matches if equals description (case-insensitive, stripped)
    """
    if match not in {"substring", "exact"}:
        raise ValueError("match must be 'substring' or 'exact'.")

    keep_l = [k.lower() for k in keep]
    out: List[Tuple[float, float]] = []
    for onset, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
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


def _intervals_to_sample_mask(
    intervals_s: Sequence[Tuple[float, float]],
    *,
    raw_times_s: np.ndarray,
    sfreq_hz: float,
) -> np.ndarray:
    """
    Convert intervals in seconds to a boolean mask over raw_times_s.

    Notes:
        - Duration==0 events are expanded to one sample to avoid empty masks.
        - Interval bounds are inclusive in time, but sample mask uses index ranges.
    """
    t = np.asarray(raw_times_s, dtype=float)
    mask = np.zeros(t.shape[0], dtype=bool)
    if mask.size == 0:
        return mask

    t0 = float(t[0])
    for a0, a1 in intervals_s:
        a0 = float(a0)
        a1 = float(a1)
        if not (np.isfinite(a0) and np.isfinite(a1)):
            continue
        if a1 <= a0:
            a1 = a0 + (1.0 / float(sfreq_hz))

        i0 = _time_to_sample_idx(a0, sfreq_hz, t0)
        i1 = _time_to_sample_idx(a1, sfreq_hz, t0)
        i0 = max(0, min(i0, mask.size - 1))
        i1 = max(0, min(i1, mask.size - 1))
        if i1 < i0:
            i0, i1 = i1, i0

        # make it inclusive: [i0, i1]
        mask[i0 : i1 + 1] = True

    return mask


def _merge_intervals(intervals_s: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
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


def _compute_iir_guard_samples(
    *,
    sfreq_hz: float,
    l_freq_hz: float,
    h_freq_hz: float,
    filter_order: int,
    ftype: str = "butter",
    phase: str = "zero",
) -> int:
    """Compute a conservative guard size (in samples) for IIR filtering.

    The guard is meant for masking samples near boundaries of bad/edge segments.
    When phase='zero' (filtfilt), the filter is applied forward and backward,
    making the operation non-causal and increasing sensitivity to boundary
    discontinuities.

    This function tries to estimate the guard size using two quantities:
      - padlen: number of samples used for padding (estimated by MNE)
      - n_ring: estimated ringing length (samples)

    The returned guard is max(padlen, n_ring).
    """
    iir_params: Dict[str, Any] = {
        "order": int(filter_order),
        "ftype": str(ftype),
        # Explicitly request SOS for numerical stability across SciPy versions.
        "output": "sos",
    }

    try:
        iir_params = mne.filter.construct_iir_filter(
            iir_params,
            f_pass=[float(l_freq_hz), float(h_freq_hz)],
            f_stop=None,
            sfreq=float(sfreq_hz),
            btype="bandpass",
            phase=str(phase),
            return_copy=True,
            verbose=False,
        )
    except Exception:
        # If filter construction fails for any reason, fall back to no dilation.
        return 0

    padlen = int(iir_params.get("padlen", 0) or 0)

    system: Any | None
    if "sos" in iir_params and iir_params["sos"] is not None:
        system = iir_params["sos"]
    elif "b" in iir_params and "a" in iir_params and iir_params["b"] is not None and iir_params["a"] is not None:
        system = (iir_params["b"], iir_params["a"])
    else:
        system = None

    n_ring = 0
    if system is not None:
        try:
            n_ring = int(mne.filter.estimate_ringing_samples(system))
        except Exception:
            n_ring = 0

    return int(max(padlen, n_ring))


def _prune_short_runs(mask_1d: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True-runs with length >= min_len."""
    if min_len <= 1:
        return mask_1d.astype(bool, copy=True)

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
        if run_len >= int(min_len):
            keep[s:e] = True
    return keep


def grid(
    raw: mne.io.BaseRaw,
    *,
    bands: BandSpec,
    thresholds: Sequence | None = None,
    percentile: float = 75.0,
    min_cycles: float = 2.0,
    hop_s: float | None = None,
    decim: int | None = None,
    target_n_times: int | None = None,
    picks: Sequence[str] | None = None,
    baseline_keep: Sequence[str] | None = None,
    baseline_match: MatchMode = "substring",
    baseline_fallback: str = "full",
    filter_order: int = 4,
    edge_anno: Sequence[str] | None = ("bad", "edge"),
    mode: MatchMode = "substring",
    edge_guard_s: EdgeGuard = "auto",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute burst envelope tensor aligned to the TFR time grid.

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
    hop_s, decim:
        Time grid definition. Use the same settings as `tfr.grid` for alignment.
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
        "full" (default) uses the full recording if no baseline samples are found;
        "raise" raises an error.
    filter_order:
        Butterworth IIR order for band-pass filtering.

    edge_anno:
        Annotation labels whose intervals should be treated as edges/bad segments.
        Samples in these intervals are excluded from threshold estimation and burst
        detection, and will be NaN in the output.

        Default: ("bad", "edge"). Set to None or an empty sequence to disable.

    mode:
        Matching mode for edge_anno labels.

        - "substring": case-insensitive substring match (default)
        - "exact": case-insensitive exact match after stripping

    edge_guard_s:
        Amount of dilation to apply to edge_anno intervals, on each side.

        - "auto" (default): compute a conservative per-band guard using the
          band-pass IIR filter design (max(padlen, estimated ringing)).
        - float: fixed guard (seconds) applied to all bands.

    Returns
    -------
    tensor:
        ndarray with shape (1, n_channels, n_bands, n_times), float64.
        Non-burst bins are NaN.
    metadata:
        Dict describing axes and parameters.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.BaseRaw.")
    if not bands:
        raise ValueError("bands must be a non-empty mapping.")
    if thresholds is None and not (0.0 < float(percentile) < 100.0):
        raise ValueError("percentile must be in (0, 100).")
    if float(min_cycles) <= 0:
        raise ValueError("min_cycles must be > 0.")
    if baseline_fallback not in {"full", "raise"}:
        raise ValueError("baseline_fallback must be 'full' or 'raise'.")

    sfreq = float(raw.info["sfreq"])
    decim_eff = _compute_decim(sfreq, hop_s, decim)

    # Pull data
    data = raw.get_data(picks=picks)  # (n_channels, n_times)
    ch_names = list(np.asarray(raw.ch_names, dtype=object) if picks is None else np.asarray(picks, dtype=object))
    n_channels, n_times = data.shape
    raw_times = np.asarray(raw.times, dtype=float)

    # Edge intervals are derived from raw.annotations and then dilated by a guard
    # duration (edge_guard_s) before being converted to a sample mask.
    edge_intervals: List[Tuple[float, float]] = []
    edge_anno_eff: Sequence[str] | None = edge_anno
    if edge_anno_eff is not None and len(edge_anno_eff) > 0:
        edge_intervals = _annotation_intervals(raw, edge_anno_eff, match=mode)

    # Baseline mask (sample domain)
    # NOTE: When user-provided thresholds are used, baseline_keep/baseline_* are
    # ignored for threshold computation.
    baseline_mask_base = np.ones(n_times, dtype=bool)
    baseline_intervals: List[Tuple[float, float]] = []
    baseline_keep_eff = baseline_keep if thresholds is None else None
    if baseline_keep_eff is not None:
        baseline_intervals = _annotation_intervals(raw, baseline_keep_eff, match=baseline_match)
        baseline_mask_base = _intervals_to_sample_mask(
            baseline_intervals,
            raw_times_s=raw_times,
            sfreq_hz=sfreq,
        )
        if not np.any(baseline_mask_base):
            if baseline_fallback == "raise":
                raise ValueError(
                    "No baseline samples found for baseline_keep="
                    f"{list(baseline_keep_eff)} with match='{baseline_match}'."
                )
            baseline_mask_base = np.ones(n_times, dtype=bool)

    band_names, band_segments, band_union_edges = _normalize_bands(bands)
    band_centers = band_union_edges.mean(axis=1)

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
            thresholds_by_band.append(thr_arr)

    out_bands: List[np.ndarray] = []
    thresholds_used: List[np.ndarray] = []

    edge_guard_samples_by_band: List[int] = []
    edge_guard_seconds_by_band: List[float] = []
    edge_coverage_by_band: List[float] = []
    baseline_coverage_by_band: List[float] = []
    edge_intervals_dilated_by_band: List[List[Tuple[float, float]]] = []

    # Band-pass + Hilbert per band.
    # If a band is provided as multiple segments (e.g. split by notch holes),
    # we compute each segment envelope and combine them via RSS:
    #   env = sqrt(sum(env_segment**2)).
    iir_params = dict(order=int(filter_order), ftype="butter", output="sos")
    for bi, (band_name, segs, f_center) in enumerate(zip(band_names, band_segments, band_centers)):
        # Compute a per-band edge mask in the sample domain. This mask is derived
        # from raw.annotations using edge_anno/mode, and dilated by edge_guard_s
        # to account for boundary transients from band-pass filtering + Hilbert.
        edge_guard_samples = 0
        guard_s = 0.0
        edge_intervals_dilated: List[Tuple[float, float]] = []
        edge_mask = np.zeros(n_times, dtype=bool)

        if len(edge_intervals) > 0 and edge_anno_eff is not None and len(edge_anno_eff) > 0:
            if edge_guard_s == "auto":
                # If a band has multiple segments, take the maximum guard across
                # segments (conservative).
                edge_guard_samples = 0
                for (l_freq, h_freq) in segs:
                    edge_guard_samples = max(
                        edge_guard_samples,
                        _compute_iir_guard_samples(
                            sfreq_hz=sfreq,
                            l_freq_hz=float(l_freq),
                            h_freq_hz=float(h_freq),
                            filter_order=int(filter_order),
                            ftype="butter",
                            phase="zero",
                        ),
                    )
                guard_s = float(edge_guard_samples) / float(sfreq)
            else:
                guard_s = float(edge_guard_s)
                if not np.isfinite(guard_s) or guard_s < 0:
                    raise ValueError("edge_guard_s must be 'auto' or a non-negative finite float.")
                edge_guard_samples = int(np.ceil(guard_s * float(sfreq)))

            edge_intervals_dilated = _expand_intervals(
                edge_intervals,
                guard_s=guard_s,
                t_min_s=float(raw_times[0]),
                t_max_s=float(raw_times[-1]),
            )
            edge_mask = _intervals_to_sample_mask(
                edge_intervals_dilated,
                raw_times_s=raw_times,
                sfreq_hz=sfreq,
            )

        # Baseline samples used for percentile thresholding are restricted to
        # the baseline_keep intervals (if provided) and always exclude edges.
        baseline_mask_band = baseline_mask_base & ~edge_mask
        if thresholds_by_band is None and not np.any(baseline_mask_band):
            if baseline_keep_eff is not None and baseline_fallback == "raise":
                raise ValueError(
                    "No baseline samples remain after excluding edge segments. "
                    "Try changing baseline_keep/baseline_match, disabling edge masking, "
                    "or using baseline_fallback='full'."
                )
            baseline_mask_band = ~edge_mask
            if not np.any(baseline_mask_band):
                raise ValueError(
                    "All samples are marked as edge after dilation. "
                    "Disable edge masking (edge_anno=None) or reduce edge_guard_s."
                )

        edge_guard_samples_by_band.append(int(edge_guard_samples))
        edge_guard_seconds_by_band.append(float(guard_s))
        edge_coverage_by_band.append(float(np.mean(edge_mask)))
        baseline_coverage_by_band.append(float(np.mean(baseline_mask_band)))
        edge_intervals_dilated_by_band.append(list(edge_intervals_dilated))

        env_sum_sq: np.ndarray | None = None
        for (l_freq, h_freq) in segs:
            filt = mne.filter.filter_data(
                data,
                sfreq=sfreq,
                l_freq=float(l_freq),
                h_freq=float(h_freq),
                method="iir",
                iir_params=iir_params,
                verbose=False,
            )
            env_seg = np.abs(hilbert(filt, axis=-1)).astype(np.float64, copy=False)
            if env_sum_sq is None:
                env_sum_sq = env_seg.astype(np.float64, copy=True) ** 2
            else:
                env_sum_sq += env_seg.astype(np.float64, copy=False) ** 2

        if env_sum_sq is None:  # pragma: no cover
            raise RuntimeError(f"No valid segments for band '{band_name}'.")

        env = np.sqrt(env_sum_sq).astype(np.float64, copy=False)

        # Threshold is either user-provided, or computed on baseline samples only
        # (if provided), per channel.
        if thresholds_by_band is not None:
            thr = thresholds_by_band[bi]
        else:
            env_base = env[:, baseline_mask_band]
            thr = np.nanpercentile(env_base, float(percentile), axis=1).astype(np.float64, copy=False)
        thresholds_used.append(thr)

        above = env > thr[:, None]

        # Edge samples are never considered bursts.
        if np.any(edge_mask):
            above[:, edge_mask] = False

        # Minimum duration in samples based on band *union* center frequency.
        min_len = int(np.ceil(float(min_cycles) * sfreq / float(f_center)))
        min_len = max(1, min_len)

        burst_mask = np.zeros_like(above, dtype=bool)
        for ci in range(n_channels):
            burst_mask[ci] = _prune_short_runs(above[ci], min_len=min_len)

        env_burst = env.astype(np.float64, copy=True)
        env_burst[~burst_mask] = np.nan

        # Decimate to match TFR time grid
        env_dec = env_burst[:, ::decim_eff]  # (n_channels, n_times_out)
        if target_n_times is not None:
            target_n_times_i = int(target_n_times)
            if env_dec.shape[-1] > target_n_times_i:
                env_dec = env_dec[..., :target_n_times_i]
            elif env_dec.shape[-1] < target_n_times_i:
                pad = np.full((n_channels, target_n_times_i - env_dec.shape[-1]), np.nan, dtype=np.float64)
                env_dec = np.concatenate([env_dec, pad], axis=-1)

        out_bands.append(env_dec)

    # Stack -> (n_channels, n_bands, n_times)
    out = np.stack(out_bands, axis=1).astype(np.float64, copy=False)
    out4d = out[None, ...]  # (1, ch, band, time)

    # Time axis (decimated)
    times_out = raw_times[::decim_eff]
    if target_n_times is not None:
        target_n_times_i = int(target_n_times)
        if times_out.shape[0] > target_n_times_i:
            times_out = times_out[:target_n_times_i]
        elif times_out.shape[0] < target_n_times_i:
            times_out = np.concatenate([times_out, np.full(target_n_times_i - times_out.shape[0], np.nan)], axis=0)

    thresholds_arr = np.stack(thresholds_used, axis=0)  # (n_bands, n_channels)

    metadata: Dict[str, Any] = dict(
        axes=dict(
            epoch=np.array([0], dtype=int),
            channel=np.array(ch_names, dtype=object),
            freq=list(band_names),
            time=np.asarray(times_out, dtype=float),
            shape=out4d.shape,
        ),
        params=dict(
            bands_segments_hz={
                str(name): [[float(a), float(b)] for (a, b) in segs]
                for name, segs in zip(band_names, band_segments)
            },
            bands_union_hz={
                str(name): [float(band_union_edges[i, 0]), float(band_union_edges[i, 1])]
                for i, name in enumerate(band_names)
            },
            band_names=list(band_names),
            band_union_edges_hz=np.asarray(band_union_edges, dtype=float).tolist(),
            thresholds_provided=(thresholds is not None),
            percentile=float(percentile),
            min_cycles=float(min_cycles),
            hop_s=(float(hop_s) if hop_s is not None else None),
            decim_eff=int(decim_eff),
            target_n_times=(int(target_n_times) if target_n_times is not None else None),
            baseline_keep=(list(baseline_keep) if baseline_keep is not None else None),
            baseline_match=str(baseline_match),
            baseline_fallback=str(baseline_fallback),
            baseline_intervals=baseline_intervals,
            edge_anno=(list(edge_anno_eff) if edge_anno_eff is not None else None),
            edge_match=str(mode),
            edge_guard_s=("auto" if edge_guard_s == "auto" else float(edge_guard_s)),
            edge_intervals=[[float(a0), float(a1)] for (a0, a1) in edge_intervals],
            edge_intervals_dilated_by_band=[
                [[float(a0), float(a1)] for (a0, a1) in ints] for ints in edge_intervals_dilated_by_band
            ],
            edge_guard_samples_by_band=[int(x) for x in edge_guard_samples_by_band],
            edge_guard_seconds_by_band=[float(x) for x in edge_guard_seconds_by_band],
        ),
        qc=dict(
            thresholds=thresholds_arr.astype(np.float64),
            baseline_coverage=float(np.mean(baseline_mask_base)) if baseline_keep_eff is not None else None,
            baseline_coverage_by_band=[float(x) for x in baseline_coverage_by_band],
            edge_coverage_by_band=[float(x) for x in edge_coverage_by_band],
        ),
    )

    return out4d, metadata
