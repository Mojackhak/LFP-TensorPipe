"""Compute time-frequency connectivity aligned to a TFR time grid.

This module provides :func:`grid`, which computes a time-resolved connectivity
tensor (pairs x freqs x times) such that the **time centers** match the decimated
time grid produced by :func:`lfp.tfr.grid.grid`.

Design notes
------------
- This implementation always uses **frequency-dependent windowing** (formerly
  called "per_freq"): each frequency (or frequency group) is computed using a
  time window that safely contains either Morlet wavelet support
  (`spectral_mode='cwt_morlet'`) or the multitaper window
  (`spectral_mode='multitaper'`).
- When `min_cycles` imposes longer low-frequency windows, the effective
  temporal resolution becomes lower for those frequencies. We accept this (it is
  the intended trade-off for stable low-frequency estimates).
- For multivariate Granger causality (GC), MNE-Connectivity requires a minimum
  number of frequency bins per call relative to `gc_n_lags`. If `min_cycles`
  causes each low-frequency bin to become its own window-group, the GC estimator
  can fail with "frequency resolution (0)". We prevent this by **padding the
  frequency list per window-group** with neighboring bins (and then keeping only
  the original bins in the output).

All code comments and docstrings are in English (per project rules).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

import mne

from ..common.timefreq import (
    channel_names_after_picks,
    decimated_times_from_raw,
    morlet_n_cycles_from_time_fwhm,
)
from .selection import resolve_pairs


def _validate_freqs(freqs: np.ndarray) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1 or freqs.size < 1:
        raise ValueError("`freqs` must be a 1D array with at least one element.")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("`freqs` must be finite.")
    if np.any(freqs <= 0):
        raise ValueError("`freqs` must be > 0.")
    return freqs


def _centers_to_events(
    center_samps: np.ndarray,
    *,
    raw_first_samp: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Convert center sample indices into an MNE events array."""
    centers = np.round(center_samps[valid_mask]).astype("int64") + int(raw_first_samp)
    return np.column_stack(
        [
            centers,
            np.zeros(centers.size, dtype=int),
            np.ones(centers.size, dtype=int),
        ]
    )


def _compute_wavelet_support_seconds(
    freqs: np.ndarray, n_cycles: np.ndarray
) -> np.ndarray:
    """Morlet 10σ time support used as a safe window length.

    L_wave(f) = (5/pi) * n_cycles(f) / f
    """
    return (5.0 / np.pi) * (n_cycles / freqs)


def _compute_multitaper_window_seconds(
    freqs: np.ndarray, n_cycles: np.ndarray
) -> np.ndarray:
    """Multitaper time window length per frequency.

    For multitaper, window length is:
        T(f) = n_cycles(f) / f
    """
    return n_cycles / freqs


def _normalize_spectral_mode(spectral_mode: str) -> str:
    """Normalize user-facing spectral mode aliases to MNE-Connectivity names."""
    mode_in = str(spectral_mode).strip().lower()
    aliases = {
        "cwt_morlet": "cwt_morlet",
        "morlet": "cwt_morlet",
        "cwt": "cwt_morlet",
        "multitaper": "multitaper",
        "mt": "multitaper",
    }
    try:
        return aliases[mode_in]
    except KeyError as exc:
        raise ValueError(
            "`spectral_mode` must be one of "
            "{'cwt_morlet','morlet','cwt','multitaper','mt'}."
        ) from exc


def _gc_min_freqs(gc_n_lags: int) -> int:
    """Minimum number of frequency bins required for MNE-Connectivity GC.

    MNE-Connectivity uses:
        freq_res = 2 * (n_freqs - 1)
    and requires:
        n_lags < freq_res

    Therefore:
        n_freqs > (n_lags / 2) + 1
    The smallest integer satisfying this is:
        n_freqs_min = (n_lags // 2) + 2
    """
    n_lags = int(gc_n_lags)
    if n_lags < 0:
        raise ValueError("`gc_n_lags` must be >= 0.")
    return int((n_lags // 2) + 2)


def _gc_target_freqs(gc_n_lags: int, gc_pad_scale: float, n_total: int) -> int:
    """Return the padded GC target count capped by the available grid size."""
    n_total_i = int(n_total)
    if n_total_i <= 0:
        raise ValueError("`n_total` must be > 0.")

    scale = float(gc_pad_scale)
    if not np.isfinite(scale) or scale < 1.0:
        raise ValueError("`gc_pad_scale` must be a finite float >= 1.0.")

    base = _gc_min_freqs(int(gc_n_lags))
    if n_total_i < base:
        raise ValueError(
            f"Need at least {base} frequency bins for gc_n_lags={int(gc_n_lags)} "
            f"(got {n_total_i})."
        )
    target = int(np.ceil(base * scale))
    target = max(target, base)
    return min(target, n_total_i)


def _split_contiguous_runs(values: np.ndarray) -> List[np.ndarray]:
    """Split indices into contiguous runs where `values` is constant."""
    v = np.asarray(values)
    if v.size == 0:
        return []
    groups: List[np.ndarray] = []
    start = 0
    for i in range(1, v.size):
        if v[i] != v[i - 1]:
            groups.append(np.arange(start, i, dtype=int))
            start = i
    groups.append(np.arange(start, v.size, dtype=int))
    return groups


def _pad_freq_indices_for_gc(
    orig_idx: np.ndarray,
    *,
    n_total: int,
    min_size: int,
) -> np.ndarray:
    """Pad a frequency-index group with neighboring indices (symmetric-first).

    This is used for multivariate Granger causality (GC): MNE-Connectivity
    requires a minimum number of frequency bins per call relative to
    ``gc_n_lags``. If a window-group contains too few bins (often at low
    frequencies when ``min_cycles`` floors ``n_cycles``), we extend the
    frequency indices by including neighbors.

    Padding strategy (as requested)
    -------------------------------
    1) Expand *symmetrically* to both sides as much as possible.
    2) If still too small, expand to **higher frequencies** (right side).
    3) If still too small, expand to **lower frequencies** (left side).

    Notes
    -----
    - The returned indices are always a **contiguous** range.
    - The caller is responsible for discarding the padded bins in the final
      output (keeping only the original bins).

    Args:
        orig_idx: Original frequency indices for one window-group.
        n_total: Total number of frequencies in the full grid.
        min_size: Minimum required number of frequencies for a GC call.

    Returns:
        Padded contiguous indices to use in the GC call.
    """
    idx = np.asarray(orig_idx, dtype=int)
    if idx.size == 0:
        return idx
    idx = np.unique(idx)
    idx.sort()

    n_total_i = int(n_total)
    min_size_i = int(min_size)

    if n_total_i <= 0:
        raise ValueError("`n_total` must be > 0.")
    if min_size_i <= 0:
        raise ValueError("`min_size` must be > 0.")
    if n_total_i < min_size_i:
        raise ValueError(
            f"Cannot pad to min_size={min_size_i} with only n_total={n_total_i} frequencies."
        )

    left = int(idx[0])
    right = int(idx[-1])

    # If the current contiguous span already meets the minimum, return it.
    span_len = right - left + 1
    if span_len >= min_size_i:
        return np.arange(left, right + 1, dtype=int)

    need = min_size_i - span_len

    # ---- 1) symmetric padding first ----
    left_avail = left  # indices available on the left: [0 .. left-1]
    right_avail = (n_total_i - 1) - right  # indices available on the right
    sym_each = need // 2
    sym_take = min(sym_each, left_avail, right_avail)

    left -= sym_take
    right += sym_take
    need -= 2 * sym_take

    # ---- 2) then pad to higher frequencies (right) ----
    if need > 0:
        right_avail = (n_total_i - 1) - right
        take_r = min(need, right_avail)
        right += take_r
        need -= take_r

    # ---- 3) finally pad to lower frequencies (left) ----
    if need > 0:
        left_avail = left
        take_l = min(need, left_avail)
        left -= take_l
        need -= take_l

    if need > 0:
        raise ValueError(
            "Failed to pad frequency group to required minimum size. "
            f"need={need}, span=({left},{right}), n_total={n_total_i}, min_size={min_size_i}."
        )

    return np.arange(left, right + 1, dtype=int)


def grid(
    raw: mne.io.BaseRaw,
    *,
    freqs: np.ndarray,
    time_resolution_s: float,
    pairs: Sequence[Tuple[str, str]] | None = None,
    groups: Dict[str, Sequence[str]] | None = None,
    method: str = "coh",
    multivariate: bool = False,
    hop_s: float | None = None,
    decim: int | None = None,
    target_n_times: int | None = None,
    spectral_mode: str = "cwt_morlet",
    # Cycle constraints
    min_cycles: float | None = 1.0,
    max_cycles: float | None = None,
    # Multitaper options
    mt_bandwidth: float | None = None,
    mt_adaptive: bool | None = None,
    mt_low_bias: bool | None = None,
    window_multiple: float = 1.0,
    safety_margin: float = 1.0,
    round_ms: float = 10.0,
    duration_guard_samples: int = 1,
    group_by_samples: bool = True,
    # connectivity opts
    sm_times: float = 0.0,
    sm_freqs: int = 1,
    sm_kernel: str = "hanning",
    padding: float = 0.0,
    decim_internal: int = 1,
    gc_n_lags: int = 20,
    gc_pad_scale: float = 3.0,
    picks: list[str] | None = None,
    time_reversed: bool = False,
    # parallel (outer only)
    outer_n_jobs: int = -1,
    outer_backend: str = "loky",
    return_connectivity_objects: bool = False,
    ordered_pairs: bool = False,
) -> Any:
    """Compute time-frequency connectivity aligned to a TFR time grid.

    The returned tensor has shape (1, n_pairs, n_freqs, n_times).

    Pair selection:
      - Provide `pairs` for explicit ordered pairs.
      - Provide `groups` to build within-group pairs.
      - Provide neither to compute all pairs from the picked channels.

    Spectral estimation mode:
      - `spectral_mode="cwt_morlet"` uses CWT-Morlet kernels.
      - `spectral_mode="multitaper"` uses multitaper kernels
        (optionally with `mt_bandwidth`, `mt_adaptive`, `mt_low_bias`).

    Notes
    -----
    This implementation always uses frequency-dependent windowing (per-frequency
    or per-frequency-group). There is no "global" window mode.

    For multivariate Granger causality (GC), MNE-Connectivity requires a minimum
    number of frequency bins per call relative to ``gc_n_lags``. Window-grouping
    can produce very small frequency groups (especially at low frequencies when
    ``min_cycles`` floors ``n_cycles``). To avoid estimator failures and reduce
    numerical instability, GC window-groups are padded with neighboring frequency
    bins.

    The padding target is controlled by ``gc_pad_scale``:
    - Let ``gc_min_freqs = gc_n_lags//2 + 2`` (the smallest valid count).
    - We pad any GC frequency group to at least
      ``ceil(gc_pad_scale * gc_min_freqs)`` bins.
    - If the available full-grid size is smaller than `gc_min_freqs`, we still
      raise an error because the estimator hard minimum cannot be met.
    - Otherwise, if the scaled target exceeds the available full-grid size, it
      is capped to the maximum available frequency-bin count.
    - After computing GC on the padded bins, we keep only the original bins in
      the returned tensor.

    Returns:
        If return_connectivity_objects is False:
            (conn_tensor, metadata)
        else:
            (conn_tensor, metadata, con_objs)
    """
    freqs = _validate_freqs(freqs)
    if float(time_resolution_s) <= 0:
        raise ValueError("`time_resolution_s` must be > 0.")
    spectral_mode_use = _normalize_spectral_mode(spectral_mode)

    sfreq = float(raw.info["sfreq"])
    if sfreq <= 0:
        raise ValueError("Raw sampling rate must be > 0.")

    # Convenience alias:
    # Users often refer to "gc_tr" as time-reversed Granger causality.
    method_in = str(method)
    method_use = method_in
    time_reversed_use = bool(time_reversed)
    if method_in.lower() == "gc_tr":
        method_use = "gc"
        time_reversed_use = True

    is_gc_method = method_use.lower() == "gc"
    if is_gc_method and not bool(multivariate):
        raise ValueError(
            "For 'gc'/'gc_tr', set multivariate=True (directed connectivity)."
        )

    try:
        from mne_connectivity.spectral.time import spectral_connectivity_time
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "mne_connectivity is required for time-frequency connectivity. "
            f"Import failed with {type(exc).__name__}: {exc}. "
            "Install with 'pip install mne-connectivity'."
        ) from exc
    spectral_conn_params = set(inspect.signature(spectral_connectivity_time).parameters)
    if "mode" not in spectral_conn_params and spectral_mode_use != "cwt_morlet":
        raise RuntimeError(
            "Installed mne-connectivity does not expose a `mode` argument in "
            "`spectral_connectivity_time`, so multitaper mode is unavailable."
        )

    if spectral_mode_use == "cwt_morlet":
        if "n_cycles" in spectral_conn_params:
            cycles_kw = "n_cycles"
        elif "cwt_n_cycles" in spectral_conn_params:
            cycles_kw = "cwt_n_cycles"
        else:
            raise RuntimeError(
                "Installed mne-connectivity does not expose `n_cycles` or "
                "`cwt_n_cycles` in `spectral_connectivity_time`; cannot "
                "control cwt_morlet temporal resolution."
            )
    else:
        if "n_cycles" not in spectral_conn_params:
            raise RuntimeError(
                "Installed mne-connectivity does not expose `n_cycles` in "
                "`spectral_connectivity_time`; cannot control multitaper "
                "temporal resolution with `time_resolution_s`."
            )
        cycles_kw = "n_cycles"

    mt_param_flags = {
        "mt_bandwidth": mt_bandwidth,
        "mt_adaptive": mt_adaptive,
        "mt_low_bias": mt_low_bias,
    }
    if spectral_mode_use == "multitaper":
        if isinstance(mt_bandwidth, (int, float)) and float(mt_bandwidth) <= 0:
            raise ValueError("`mt_bandwidth` must be > 0 when provided.")
        unsupported_mt = [
            name
            for name, value in mt_param_flags.items()
            if value is not None and name not in spectral_conn_params
        ]
        if unsupported_mt:
            raise RuntimeError(
                "Installed mne-connectivity does not support multitaper arguments: "
                f"{unsupported_mt}."
            )

    # ----- (A) Recreate TFR time grid exactly -----
    if decim is not None and int(decim) > 0:
        decim_eff = int(decim)
    else:
        if hop_s is None or float(hop_s) <= 0:
            raise ValueError("Provide `hop_s` or explicit `decim` to align with TFR.")
        decim_eff = max(1, int(round(sfreq * float(hop_s))))

    times_tfr = decimated_times_from_raw(
        raw, decim=decim_eff, target_n_times=target_n_times
    )
    raw_times = np.asarray(raw.times, dtype=float)
    t0, t1 = float(raw_times[0]), float(raw_times[-1])

    center_samps_full = (times_tfr - raw_times[0]) * sfreq  # float sample indices

    # ----- (B) n_cycles + analysis window length per frequency -----
    if spectral_mode_use == "cwt_morlet":
        n_cycles = morlet_n_cycles_from_time_fwhm(
            freqs,
            time_fwhm_s=float(time_resolution_s),
            min_cycles=min_cycles,
            max_cycles=max_cycles,
        )
        window_len_s = _compute_wavelet_support_seconds(freqs, n_cycles)
        n_cycles_source = "morlet_fwhm_time_const"
    else:
        n_cycles = np.asarray(freqs, dtype=float) * float(time_resolution_s)
        n_cycles_source = "multitaper_T_const"
        if min_cycles is not None:
            n_cycles = np.maximum(n_cycles, float(min_cycles))
            n_cycles_source += f"+floor({float(min_cycles)})"
        if max_cycles is not None:
            if float(max_cycles) <= 0:
                raise ValueError("`max_cycles` must be > 0 when provided.")
            if min_cycles is not None and float(max_cycles) < float(min_cycles):
                raise ValueError("`max_cycles` must be >= `min_cycles`.")
            n_cycles = np.minimum(n_cycles, float(max_cycles))
            n_cycles_source += f"+ceil({float(max_cycles)})"
        window_len_s = _compute_multitaper_window_seconds(freqs, n_cycles)

    L_wave = np.maximum(
        window_len_s * float(window_multiple) * float(safety_margin), 1.0 / sfreq
    )

    # ----- (C) Pair indices -----
    ch_names = channel_names_after_picks(raw, picks)
    seeds_idx, targets_idx, pair_names, pair_meta = resolve_pairs(
        ch_names,
        pairs=pairs,
        groups=groups,
        ordered_pairs=bool(ordered_pairs),
    )
    n_pairs = len(pair_names)

    # MNE-Connectivity expects (n_pairs, 1) indices for multivariate methods.
    if multivariate:
        seeds_idx = seeds_idx.reshape(-1, 1)
        targets_idx = targets_idx.reshape(-1, 1)

    # ----- (D) Output accumulator aligned to TFR -----
    n_times_tfr = int(times_tfr.size)
    data_accum = np.full((n_times_tfr, n_pairs, freqs.size), np.nan, dtype=float)

    con_objs: list[Any] = []

    def _run_one_group(
        *,
        out_idx: np.ndarray,
        keep_pos: np.ndarray,
        half_s: float,
        valid_mask: np.ndarray,
        freqs_sub: np.ndarray,
        n_cycles_sub: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray]:
        """Compute connectivity for one frequency group."""
        if out_idx.size == 0:
            return out_idx, np.empty((0, n_pairs, 0), dtype=float), None, valid_mask

        if not np.any(valid_mask):
            return (
                out_idx,
                np.empty((0, n_pairs, out_idx.size), dtype=float),
                None,
                valid_mask,
            )

        events = _centers_to_events(
            center_samps_full, raw_first_samp=raw.first_samp, valid_mask=valid_mask
        )

        epochs = mne.Epochs(
            raw,
            events,
            tmin=-half_s,
            tmax=+half_s,
            baseline=None,
            preload=True,
            picks=picks,
            reject_by_annotation=False,
            verbose=False,
        )

        if time_reversed_use:
            # Reverse time within each epoch window (classic time-reversal control).
            # We keep the event centers the same so the output time grid remains aligned.
            ep_data = epochs.get_data(copy=True)
            epochs._data = ep_data[..., ::-1]

        conn_kwargs: Dict[str, Any] = dict(
            freqs=freqs_sub,
            method=method_use,
            indices=(seeds_idx, targets_idx),
            average=False,
            sm_times=sm_times,
            sm_freqs=sm_freqs,
            sm_kernel=sm_kernel,
            padding=padding,
            decim=int(decim_internal),
            n_jobs=1,
            verbose=False,
            **{cycles_kw: n_cycles_sub},
        )
        if "mode" in spectral_conn_params:
            conn_kwargs["mode"] = spectral_mode_use
        if "gc_n_lags" in spectral_conn_params:
            conn_kwargs["gc_n_lags"] = int(gc_n_lags)
        if spectral_mode_use == "multitaper":
            if mt_bandwidth is not None and "mt_bandwidth" in spectral_conn_params:
                conn_kwargs["mt_bandwidth"] = float(mt_bandwidth)
            if mt_adaptive is not None and "mt_adaptive" in spectral_conn_params:
                conn_kwargs["mt_adaptive"] = bool(mt_adaptive)
            if mt_low_bias is not None and "mt_low_bias" in spectral_conn_params:
                conn_kwargs["mt_low_bias"] = bool(mt_low_bias)

        con = spectral_connectivity_time(epochs, **conn_kwargs)
        D = con.get_data()  # (n_valid, n_pairs, n_freqs_sub)

        keep_pos_i = np.asarray(keep_pos, dtype=int)
        if keep_pos_i.ndim != 1 or keep_pos_i.size != out_idx.size:
            raise RuntimeError(
                "GC padding mapping error: keep_pos must be 1D and match out_idx size. "
                f"Got keep_pos shape={keep_pos_i.shape}, out_idx size={out_idx.size}."
            )

        D_keep = D[:, :, keep_pos_i]
        return out_idx, D_keep, con, valid_mask

    # ----- (E) Frequency-dependent grouping (always "per_freq") -----
    if group_by_samples:
        n_samp_per_freq = (np.ceil(L_wave * sfreq)).astype(int) + max(
            0, int(duration_guard_samples)
        )
        half_per_freq = (n_samp_per_freq - 1) / (2.0 * sfreq)
        group_key = n_samp_per_freq
    else:
        step = float(round_ms) / 1000.0
        if step <= 0:
            raise ValueError("`round_ms` must be > 0 when group_by_samples=False.")
        durations = np.ceil((L_wave + 1.0 / sfreq) / step) * step
        half_per_freq = durations / 2.0
        group_key = durations

    groups_idx = _split_contiguous_runs(group_key)

    # GC fix (neighbor padding):
    # MNE-Connectivity multivariate GC requires a minimum number of frequency bins per call:
    #   gc_n_lags < 2 * (n_freqs_sub - 1)
    # Singleton (or tiny) window-groups can happen at low frequencies when `min_cycles` floors n_cycles,
    # which may cause GC to fail with "frequency resolution (0)".
    #
    # We keep frequency-dependent windowing, but if a window-group has too few frequency bins for GC,
    # we *pad* its frequency index range with neighboring bins, compute GC, and then keep only the
    # original bins in the output tensor.
    is_gc_call = bool(is_gc_method) and bool(multivariate)
    gc_min_freqs_base: int | None = None
    gc_min_freqs_required: int | None = None
    if is_gc_call:
        gc_min_freqs_base = _gc_min_freqs(int(gc_n_lags))

        # The base minimum is the smallest count that satisfies the estimator's
        # hard constraint. `gc_pad_scale` lets users request a larger, more
        # stable frequency set for each window-group. If the full grid itself
        # cannot satisfy the estimator hard minimum, we still raise. Otherwise,
        # we cap any oversized scaled target to the available full-grid size.
        gc_min_freqs_required = _gc_target_freqs(
            int(gc_n_lags),
            float(gc_pad_scale),
            int(freqs.size),
        )

    def _make_group_call(
        out_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """Build a computation spec for one window-group."""
        out_idx = np.asarray(out_idx, dtype=int)
        call_idx = out_idx
        keep_pos = np.arange(out_idx.size, dtype=int)

        if (
            is_gc_call
            and gc_min_freqs_required is not None
            and out_idx.size < gc_min_freqs_required
        ):
            call_idx = _pad_freq_indices_for_gc(
                out_idx, n_total=freqs.size, min_size=gc_min_freqs_required
            )
            # Map output indices to their positions within the padded call indices.
            keep_pos = np.searchsorted(call_idx, out_idx)

        half = float(np.max(half_per_freq[call_idx]))
        valid_mask = np.isfinite(center_samps_full)
        if np.any(valid_mask):
            times_valid = times_tfr[valid_mask]
            feas = (times_valid - half >= t0) & (times_valid + half <= t1)
            valid_mask[valid_mask] &= feas

        return out_idx, call_idx, keep_pos, half, valid_mask

    groups_spec = [_make_group_call(g) for g in groups_idx]

    results = Parallel(n_jobs=int(outer_n_jobs), backend=str(outer_backend))(
        delayed(_run_one_group)(
            out_idx=out_idx,
            keep_pos=keep_pos,
            half_s=half,
            valid_mask=valid_mask,
            freqs_sub=freqs[call_idx],
            n_cycles_sub=n_cycles[call_idx],
        )
        for out_idx, call_idx, keep_pos, half, valid_mask in groups_spec
    )

    for out_idx, D, con, valid_mask in results:
        if D.size == 0:
            continue
        time_idx = np.flatnonzero(valid_mask)
        if D.shape != (time_idx.size, n_pairs, out_idx.size):
            raise RuntimeError(
                f"Connectivity data shape mismatch: got {D.shape}, expected "
                f"({time_idx.size}, {n_pairs}, {out_idx.size})."
            )
        data_accum[np.ix_(time_idx, np.arange(n_pairs), out_idx)] = D
        if return_connectivity_objects and con is not None:
            con_objs.append(con)

    conn = np.transpose(data_accum, (1, 2, 0))[
        np.newaxis, ...
    ]  # (1, n_pairs, n_freqs, n_times)
    metadata = dict(
        axes=dict(
            epoch=np.arange(conn.shape[0]),
            channel=list(pair_names),
            freq=freqs,
            time=times_tfr,
            shape=conn.shape,
        ),
        params=dict(
            method=str(method_in),
            method_internal=str(method_use),
            time_reversed=bool(time_reversed_use),
            multivariate=bool(multivariate),
            time_resolution_s=float(time_resolution_s),
            spectral_mode=str(spectral_mode_use),
            n_cycles_source=str(n_cycles_source),
            hop_s=hop_s,
            decim=int(decim_eff),
            target_n_times=target_n_times,
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            mt_bandwidth=(float(mt_bandwidth) if mt_bandwidth is not None else None),
            mt_adaptive=(bool(mt_adaptive) if mt_adaptive is not None else None),
            mt_low_bias=(bool(mt_low_bias) if mt_low_bias is not None else None),
            window_multiple=float(window_multiple),
            safety_margin=float(safety_margin),
            round_ms=float(round_ms),
            duration_guard_samples=int(duration_guard_samples),
            group_by_samples=bool(group_by_samples),
            sm_times=float(sm_times),
            sm_freqs=int(sm_freqs),
            sm_kernel=str(sm_kernel),
            padding=float(padding),
            decim_internal=int(decim_internal),
            gc_n_lags=int(gc_n_lags),
            gc_pad_scale=float(gc_pad_scale),
            picks=picks,
            outer_n_jobs=int(outer_n_jobs),
            outer_backend=str(outer_backend),
            ordered_pairs=bool(ordered_pairs),
            gc_min_freqs_base=gc_min_freqs_base,
            gc_min_freqs_required=gc_min_freqs_required,
            gc_freq_padding=("neighbors" if is_gc_call else None),
            **pair_meta,
        ),
    )

    if return_connectivity_objects:
        return conn, metadata, con_objs
    return conn, metadata


def n_samples_window_per_freq(
    freqs_hz: np.ndarray,
    sfreq_hz: float,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
    duration_guard_samples: int = 0,
) -> np.ndarray:
    """Compute per-frequency window length in samples for Morlet 10σ support."""
    freqs = np.asarray(freqs_hz, dtype=float)

    n_cycles0 = time_resolution_s * np.pi * freqs / np.sqrt(2.0 * np.log(2.0))
    n_cycles = n_cycles0.copy()

    if min_cycles is not None:
        n_cycles = np.maximum(n_cycles, float(min_cycles))
    if max_cycles is not None:
        n_cycles = np.minimum(n_cycles, float(max_cycles))

    L_wave_s = (5.0 / np.pi) * (n_cycles / freqs)  # 10σ support
    n_samp = np.ceil(L_wave_s * float(sfreq_hz)).astype(int) + int(
        duration_guard_samples
    )
    return n_samp
