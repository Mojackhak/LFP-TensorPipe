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
- Morlet cycle limits and the Multitaper minimum-cycle rule can impose longer
  low-frequency windows. The resulting lower temporal resolution is the
  intended trade-off for stable low-frequency estimates.
- Every input epoch contains the retained central analysis interval plus one
  half-kernel support margin on each side. The same margin is passed to
  MNE-Connectivity as ``padding`` so edge-contaminated coefficients are
  discarded before connectivity is aggregated.
- For multivariate Granger causality (GC), MNE-Connectivity requires a minimum
  number of frequency bins per call relative to `gc_n_lags`. If dynamic windows
  cause each low-frequency bin to become its own window-group, the GC estimator
  can fail with "frequency resolution (0)". We prevent this by **padding the
  frequency list per window-group** with neighboring bins (and then keeping only
  the original bins in the output).

All code comments and docstrings are in English (per project rules).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Sequence, Tuple
import warnings

import numpy as np
from joblib import Parallel, delayed

import mne

from ..common.timefreq import (
    channel_names_after_picks,
    decimated_times_from_raw,
    morlet_n_cycles_from_time_fwhm,
    multitaper_fixed_p_parameters,
)
from ..runtime.tensor_helpers import build_annotation_skip_time_mask
from . import CONNECTIVITY_PADDING_MODE
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


def _support_window_geometry(
    *,
    kernel_support_s: float,
    analysis_half_s: float,
    sfreq: float,
    decim_internal: int,
    requested_min_padding_s: float,
) -> dict[str, float | int]:
    """Resolve an edge-safe connectivity epoch on the raw sample grid.

    The central analysis interval is retained. One half of the longest kernel
    support used by the estimator call is added to each side, then removed by
    MNE-Connectivity's ``padding`` crop. Padding is rounded outward to an exact
    multiple of the internal decimation factor so the crop cannot retain a
    coefficient without full kernel support.
    """
    kernel_support = float(kernel_support_s)
    analysis_half = float(analysis_half_s)
    sfreq_use = float(sfreq)
    decim_use = int(decim_internal)
    requested_padding = float(requested_min_padding_s)

    if not np.isfinite(kernel_support) or kernel_support <= 0:
        raise ValueError("`kernel_support_s` must be finite and > 0.")
    if not np.isfinite(analysis_half) or analysis_half < 0:
        raise ValueError("`analysis_half_s` must be finite and >= 0.")
    analysis_radius_samples = int(np.ceil(analysis_half * sfreq_use))
    required_padding_samples = int(np.ceil(0.5 * kernel_support * sfreq_use))
    requested_padding_samples = int(np.ceil(requested_padding * sfreq_use))
    minimum_padding_samples = max(required_padding_samples, requested_padding_samples)
    padding_output_samples = int(np.ceil(minimum_padding_samples / float(decim_use)))
    padding_samples = padding_output_samples * decim_use
    input_epoch_radius_samples = analysis_radius_samples + padding_samples
    padding_s = padding_samples / sfreq_use

    return {
        "kernel_support_s": kernel_support,
        "analysis_radius_samples": analysis_radius_samples,
        "analysis_span_s": 2.0 * analysis_radius_samples / sfreq_use,
        "required_padding_samples": required_padding_samples,
        "padding_samples": padding_samples,
        "padding_output_samples": padding_output_samples,
        "padding_s": padding_s,
        "input_epoch_radius_samples": input_epoch_radius_samples,
        "input_epoch_span_s": 2.0 * input_epoch_radius_samples / sfreq_use,
        "input_epoch_n_samples": 2 * input_epoch_radius_samples + 1,
    }


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
    method: str | Sequence[str] = "coh",
    multivariate: bool = False,
    hop_s: float | None = None,
    decim: int | None = None,
    target_n_times: int | None = None,
    spectral_mode: str = "cwt_morlet",
    # Cycle constraints
    min_cycles: float | None = 1.0,
    max_cycles: float | None = None,
    # Multitaper options
    mt_time_bandwidth_product: float = 4.0,
    mt_min_cycles: float = 3.0,
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
    annotation_skip_radius_s: float | None = None,
    task_group_id: int | None = None,
    task_time_indices: Sequence[int] | None = None,
    return_task_description: bool = False,
    return_task_blocks: bool = False,
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
        (optionally with `mt_adaptive` and `mt_low_bias`).

    Notes
    -----
    This implementation always uses frequency-dependent windowing (per-frequency
    or per-frequency-group). There is no "global" window mode.

    ``padding`` is a programmatic minimum per-side crop. The effective padding
    is always at least one half of the longest spectral-kernel support used by
    the estimator call; matching raw samples are added to the input epoch before
    the crop is applied.

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
    if int(decim_internal) <= 0:
        raise ValueError("`decim_internal` must be > 0.")
    if not np.isfinite(float(padding)) or float(padding) < 0:
        raise ValueError("`padding` must be finite and >= 0.")

    # Convenience alias:
    # Users often refer to "gc_tr" as time-reversed Granger causality.
    method_inputs = (
        [str(method)] if isinstance(method, str) else [str(item) for item in method]
    )
    if not method_inputs:
        raise ValueError("`method` must contain at least one connectivity estimator.")
    if len(method_inputs) > 1 and any(
        item.lower() == "gc_tr" for item in method_inputs
    ):
        raise ValueError("`gc_tr` cannot be combined with another estimator call.")
    method_uses: list[str] = []
    method_use_by_input: dict[str, str] = {}
    output_components: dict[str, str | None] = {}
    time_reversed_use = bool(time_reversed)
    for method_input in method_inputs:
        method_use = method_input
        output_component: str | None = None
        if method_input.lower() == "imcoh_abs":
            method_use = "cohy"
            output_component = "absolute_imaginary"
        elif method_input.lower() == "gc_tr":
            method_use = "gc"
            time_reversed_use = True
        if method_use not in method_uses:
            method_uses.append(method_use)
        method_use_by_input[method_input] = method_use
        output_components[method_input] = output_component

    if time_reversed_use and any(item.lower() != "gc" for item in method_uses):
        raise ValueError("Time reversal is supported only for Granger causality.")
    is_gc_method = any(item.lower() == "gc" for item in method_uses)
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
        "mt_bandwidth": mt_time_bandwidth_product,
        "mt_adaptive": mt_adaptive,
        "mt_low_bias": mt_low_bias,
    }
    if spectral_mode_use == "multitaper":
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

    center_samps_full = (times_tfr - raw_times[0]) * sfreq  # float sample indices
    finite_time_mask = np.isfinite(times_tfr)
    if annotation_skip_radius_s is None:
        skip_time_mask = np.zeros(times_tfr.shape, dtype=bool)
    else:
        skip_time_mask = build_annotation_skip_time_mask(
            raw,
            times_s=times_tfr,
            radius_s=float(annotation_skip_radius_s),
        )
    n_columns_total = int(np.sum(finite_time_mask))
    n_columns_skipped_masked = int(np.sum(finite_time_mask & skip_time_mask))

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
        (
            n_cycles,
            window_len_s,
            mt_effective_bandwidth_hz,
            _,
        ) = multitaper_fixed_p_parameters(
            freqs,
            time_resolution_s=float(time_resolution_s),
            mt_time_bandwidth_product=float(mt_time_bandwidth_product),
            mt_min_cycles=float(mt_min_cycles),
        )
        n_cycles_source = "multitaper_fixed_P_adaptive_window"
        available_duration_s = float(max(0, raw.n_times - 1)) / float(sfreq)
        longest_index = int(np.argmax(window_len_s))
        if float(window_len_s[longest_index]) > available_duration_s:
            raise ValueError(
                "Effective Multitaper window at "
                f"{float(freqs[longest_index]):g} Hz is "
                f"{float(window_len_s[longest_index]):g} s, longer than the "
                f"available duration {available_duration_s:g} s."
            )

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

    n_times_tfr = int(times_tfr.size)
    con_objs: list[Any] = []

    def _run_one_group(
        *,
        out_idx: np.ndarray,
        keep_pos: np.ndarray,
        half_s: float,
        padding_s: float,
        valid_mask: np.ndarray,
        freqs_sub: np.ndarray,
        n_cycles_sub: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], Any, np.ndarray]:
        """Compute connectivity for one frequency group."""
        if out_idx.size == 0:
            return out_idx, {}, None, valid_mask

        if not np.any(valid_mask):
            return (
                out_idx,
                {},
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
            method=(method_uses[0] if len(method_uses) == 1 else list(method_uses)),
            indices=(seeds_idx, targets_idx),
            average=False,
            sm_times=sm_times,
            sm_freqs=sm_freqs,
            sm_kernel=sm_kernel,
            padding=np.nextafter(float(padding_s), np.inf),
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
            if "mt_bandwidth" in spectral_conn_params:
                conn_kwargs["mt_bandwidth"] = float(mt_time_bandwidth_product)
            if mt_adaptive is not None and "mt_adaptive" in spectral_conn_params:
                conn_kwargs["mt_adaptive"] = bool(mt_adaptive)
            if mt_low_bias is not None and "mt_low_bias" in spectral_conn_params:
                conn_kwargs["mt_low_bias"] = bool(mt_low_bias)

        con = spectral_connectivity_time(epochs, **conn_kwargs)
        con_by_method = (
            {method_uses[0]: con}
            if len(method_uses) == 1
            else dict(zip(method_uses, list(con)))
        )

        keep_pos_i = np.asarray(keep_pos, dtype=int)
        if keep_pos_i.ndim != 1 or keep_pos_i.size != out_idx.size:
            raise RuntimeError(
                "GC padding mapping error: keep_pos must be 1D and match out_idx size. "
                f"Got keep_pos shape={keep_pos_i.shape}, out_idx size={out_idx.size}."
            )

        data_keep: dict[str, np.ndarray] = {}
        for method_input in method_inputs:
            method_use = method_use_by_input[method_input]
            values = np.asarray(con_by_method[method_use].get_data())
            if output_components[method_input] == "absolute_imaginary":
                values = np.abs(np.imag(values))
            data_keep[method_input] = values[:, :, keep_pos_i]
        return out_idx, data_keep, con, valid_mask

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
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        np.ndarray,
        dict[str, Any],
    ]:
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

        analysis_half_s = float(np.max(half_per_freq[call_idx]))
        support_geometry = _support_window_geometry(
            kernel_support_s=float(np.max(L_wave[call_idx])),
            analysis_half_s=analysis_half_s,
            sfreq=sfreq,
            decim_internal=int(decim_internal),
            requested_min_padding_s=float(padding),
        )
        input_half_s = int(support_geometry["input_epoch_radius_samples"]) / sfreq
        padding_s = float(support_geometry["padding_s"])
        feasible_mask = np.isfinite(center_samps_full)
        if np.any(feasible_mask):
            centers_valid = np.round(center_samps_full[feasible_mask]).astype("int64")
            radius_samples = int(support_geometry["input_epoch_radius_samples"])
            feas = (centers_valid - radius_samples >= 0) & (
                centers_valid + radius_samples < raw.n_times
            )
            feasible_mask[feasible_mask] &= feas

        valid_mask = feasible_mask & ~skip_time_mask
        group_counts = {
            "output_frequency_indices": [int(item) for item in out_idx.tolist()],
            "n_columns_total": int(n_columns_total),
            "n_columns_skipped_masked": int(n_columns_skipped_masked),
            "n_columns_dropped_infeasible": int(
                np.sum(finite_time_mask & ~skip_time_mask & ~feasible_mask)
            ),
            **support_geometry,
        }

        return (
            out_idx,
            call_idx,
            keep_pos,
            input_half_s,
            padding_s,
            valid_mask,
            group_counts,
        )

    groups_spec = [
        (group_id, *_make_group_call(group_idx))
        for group_id, group_idx in enumerate(groups_idx)
    ]
    if task_group_id is not None:
        group_id_use = int(task_group_id)
        groups_spec = [spec for spec in groups_spec if int(spec[0]) == group_id_use]
        if not groups_spec:
            raise ValueError(f"Unknown connectivity task group id: {group_id_use}.")
    if task_time_indices is not None:
        task_time_idx = np.asarray(task_time_indices, dtype=int).ravel()
        if task_time_idx.size and (
            np.min(task_time_idx) < 0 or np.max(task_time_idx) >= n_times_tfr
        ):
            raise ValueError("Connectivity task time indices are out of range.")
        allowed_mask = np.zeros(n_times_tfr, dtype=bool)
        allowed_mask[task_time_idx] = True
        groups_spec = [
            (
                group_id,
                out_idx,
                call_idx,
                keep_pos,
                half,
                padding_s,
                valid_mask & allowed_mask,
                group_counts,
            )
            for (
                group_id,
                out_idx,
                call_idx,
                keep_pos,
                half,
                padding_s,
                valid_mask,
                group_counts,
            ) in groups_spec
        ]
    window_group_counts = [dict(spec[7]) for spec in groups_spec]

    task_description = {
        "shape": [1, int(n_pairs), int(freqs.size), int(n_times_tfr)],
        "pair_names": list(pair_names),
        "freqs": np.asarray(freqs, dtype=float),
        "times": np.asarray(times_tfr, dtype=float),
        "groups": [
            {
                "group_id": int(group_id),
                "output_frequency_indices": np.asarray(out_idx, dtype=int),
                "call_frequency_indices": np.asarray(call_idx, dtype=int),
                "valid_time_indices": np.flatnonzero(valid_mask),
                "support_geometry": {
                    key: value
                    for key, value in group_counts.items()
                    if key
                    not in {
                        "output_frequency_indices",
                        "n_columns_total",
                        "n_columns_skipped_masked",
                        "n_columns_dropped_infeasible",
                    }
                },
            }
            for (
                group_id,
                out_idx,
                call_idx,
                _keep_pos,
                _half,
                _padding_s,
                valid_mask,
                group_counts,
            ) in groups_spec
        ],
    }

    def _build_metadata(method_input: str) -> dict[str, Any]:
        method_use = method_use_by_input[method_input]
        output_component = output_components[method_input]
        shape = (1, n_pairs, freqs.size, n_times_tfr)
        return dict(
            axes=dict(
                epoch=np.arange(shape[0]),
                channel=list(pair_names),
                freq=freqs,
                time=times_tfr,
                shape=shape,
            ),
            params=dict(
                method=str(method_input),
                method_internal=str(method_use),
                **(
                    {"output_component": output_component}
                    if output_component is not None
                    else {}
                ),
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
                mt_time_bandwidth_product=(
                    float(mt_time_bandwidth_product)
                    if spectral_mode_use == "multitaper"
                    else None
                ),
                mt_min_cycles=(
                    float(mt_min_cycles) if spectral_mode_use == "multitaper" else None
                ),
                mt_effective_window_s=(
                    np.asarray(window_len_s, dtype=float)
                    if spectral_mode_use == "multitaper"
                    else None
                ),
                mt_effective_bandwidth_hz=(
                    np.asarray(mt_effective_bandwidth_hz, dtype=float)
                    if spectral_mode_use == "multitaper"
                    else None
                ),
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
                padding_mode=CONNECTIVITY_PADDING_MODE,
                requested_min_padding_s=float(padding),
                decim_internal=int(decim_internal),
                gc_n_lags=int(gc_n_lags),
                gc_pad_scale=float(gc_pad_scale),
                picks=picks,
                outer_n_jobs=int(outer_n_jobs),
                outer_backend=str(outer_backend),
                ordered_pairs=bool(ordered_pairs),
                annotation_skip_radius_s=(
                    float(annotation_skip_radius_s)
                    if annotation_skip_radius_s is not None
                    else None
                ),
                n_columns_total=int(n_columns_total),
                n_columns_skipped_masked=int(n_columns_skipped_masked),
                window_group_counts=window_group_counts,
                gc_min_freqs_base=gc_min_freqs_base,
                gc_min_freqs_required=gc_min_freqs_required,
                gc_freq_padding=("neighbors" if is_gc_call else None),
                **pair_meta,
            ),
        )

    metadata_by_method = {
        method_input: _build_metadata(method_input) for method_input in method_inputs
    }
    metadata_out: Any = (
        metadata_by_method[method_inputs[0]]
        if len(method_inputs) == 1
        else metadata_by_method
    )
    if return_task_description:
        return task_description, metadata_out

    # Pool tasks return chunk-sized blocks and never own a full output tensor.
    # The legacy complete-output path retains the accumulator contract.
    data_accum = None
    if not return_task_blocks:
        data_accum = {
            method_input: np.full(
                (n_times_tfr, n_pairs, freqs.size),
                np.nan,
                dtype=(
                    np.complex128
                    if method_use_by_input[method_input] == "cohy"
                    and output_components[method_input] is None
                    else float
                ),
            )
            for method_input in method_inputs
        }

    results = Parallel(n_jobs=int(outer_n_jobs), backend=str(outer_backend))(
        delayed(_run_one_group)(
            out_idx=out_idx,
            keep_pos=keep_pos,
            half_s=half,
            padding_s=padding_s,
            valid_mask=valid_mask,
            freqs_sub=freqs[call_idx],
            n_cycles_sub=n_cycles[call_idx],
        )
        for (
            _group_id,
            out_idx,
            call_idx,
            keep_pos,
            half,
            padding_s,
            valid_mask,
            _,
        ) in groups_spec
    )

    task_blocks: list[dict[str, Any]] = []
    for group_spec, result in zip(groups_spec, results):
        group_id = int(group_spec[0])
        out_idx, data_by_method, con, valid_mask = result
        if not data_by_method:
            continue
        time_idx = np.flatnonzero(valid_mask)
        for method_input, values in data_by_method.items():
            if values.shape != (time_idx.size, n_pairs, out_idx.size):
                raise RuntimeError(
                    f"Connectivity data shape mismatch: got {values.shape}, expected "
                    f"({time_idx.size}, {n_pairs}, {out_idx.size})."
                )
            if data_accum is not None:
                data_accum[method_input][
                    np.ix_(time_idx, np.arange(n_pairs), out_idx)
                ] = values
        task_blocks.append(
            {
                "group_id": group_id,
                "time_indices": time_idx,
                "frequency_indices": np.asarray(out_idx, dtype=int),
                "data": data_by_method,
            }
        )
        if return_connectivity_objects and con is not None:
            con_objs.append(con)

    if return_task_blocks:
        return task_blocks, metadata_out

    conn_by_method = {
        method_input: np.transpose(values, (1, 2, 0))[np.newaxis, ...]
        for method_input, values in data_accum.items()
    }
    unusable_counts = {
        method_input: int(np.sum(~np.any(np.isfinite(values), axis=(0, 1, 3))))
        for method_input, values in conn_by_method.items()
    }
    unusable_counts = {
        method_input: count
        for method_input, count in unusable_counts.items()
        if count > 0
    }
    if unusable_counts:
        if len(method_inputs) == 1:
            message = (
                "Connectivity contains "
                f"{next(iter(unusable_counts.values()))} frequency entries "
                "with no usable output centers; they remain NaN."
            )
        else:
            summary = ", ".join(
                f"{method_input}={count}"
                for method_input, count in unusable_counts.items()
            )
            message = (
                "Connectivity contains frequency entries with no usable output "
                f"centers by estimator ({summary}); they remain NaN."
            )
        warnings.warn(message, UserWarning, stacklevel=2)

    conn_out: Any = (
        conn_by_method[method_inputs[0]] if len(method_inputs) == 1 else conn_by_method
    )

    if return_connectivity_objects:
        return conn_out, metadata_out, con_objs
    return conn_out, metadata_out


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
