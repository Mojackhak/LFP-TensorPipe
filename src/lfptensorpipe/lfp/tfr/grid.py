"""Time-frequency representation (TFR) computation utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

import mne
from mne.time_frequency import (
    fwhm as mne_fwhm,
    tfr_array_morlet,
    tfr_array_multitaper,
    tfr_morlet,
    tfr_multitaper,
)

from ..common.timefreq import compute_decimation


def _compute_n_cycles(
    *,
    method: str,
    freqs: np.ndarray,
    time_resolution_s: Optional[float],
    cycles_strategy: str,
    n_cycles_constant: float,
    fwhm_power_hz: Optional[float],
    min_cycles: Optional[float],
    max_cycles: Optional[float],
) -> tuple[np.ndarray, str]:
    """Compute n_cycles for Morlet or multitaper with a clear provenance string."""
    method_l = method.lower()

    if method_l == "morlet":
        if time_resolution_s is not None:
            n_cycles_vec = (
                float(time_resolution_s) * np.pi * freqs / np.sqrt(2.0 * np.log(2.0))
            )
            cycles_source = "morlet_fwhm_time_const"
        else:
            if cycles_strategy == "fwhm_power":
                if not (
                    isinstance(fwhm_power_hz, (int, float)) and float(fwhm_power_hz) > 0
                ):
                    raise ValueError(
                        "For cycles_strategy='fwhm_power', provide positive `fwhm_power_hz`."
                    )
                # Power-domain Gaussian approx: FWHM_f(power) ≈ 1.665 * f / n_cycles
                n_cycles_vec = 1.665 * freqs / float(fwhm_power_hz)
                cycles_source = "morlet_fwhm_power_const"
            else:
                n_cycles_vec = np.full_like(
                    freqs, float(n_cycles_constant), dtype=float
                )
                cycles_source = "morlet_cycles_const"

    elif method_l == "multitaper":
        if time_resolution_s is not None:
            # Constant time window T across freqs: n_cycles(f) = f * T
            T = float(time_resolution_s)
            n_cycles_vec = freqs * T
            cycles_source = "multitaper_T_const"
        else:
            n_cycles_vec = np.full_like(freqs, float(n_cycles_constant), dtype=float)
            cycles_source = "multitaper_cycles_const"

    else:
        raise ValueError("`method` must be 'morlet' or 'multitaper'.")

    if min_cycles is not None:
        n_cycles_vec = np.maximum(n_cycles_vec, float(min_cycles))
        cycles_source += f"+floor({float(min_cycles)})"
    if (max_cycles is not None) and (float(max_cycles) > 0):
        if min_cycles is not None and float(max_cycles) < float(min_cycles):
            raise ValueError("`max_cycles` must be >= `min_cycles`.")
        n_cycles_vec = np.minimum(n_cycles_vec, float(max_cycles))
        cycles_source += f"+ceil({float(max_cycles)})"

    return n_cycles_vec, cycles_source


def grid(
    data: Union[mne.Epochs, mne.io.BaseRaw],
    *,
    method: str = "morlet",
    freqs: Sequence[float],
    # Resolution controls
    time_resolution_s: Optional[float] = None,
    # Morlet controls (used when time_resolution_s is None)
    cycles_strategy: str = "constant",
    n_cycles_constant: float = 9.0,
    fwhm_power_hz: Optional[float] = None,
    min_cycles: Optional[float] = 3.0,
    max_cycles: Optional[float] = None,
    # Multitaper controls
    time_bandwidth: float = 1.0,
    # Timing / compute controls
    hop_s: Optional[float] = 0.025,
    decim: Optional[int] = None,
    average: bool = False,
    return_itc: bool = False,
    picks: Sequence[str] | None = None,
    n_jobs: Optional[int] = None,
    **legacy_freq_grid_kwargs: Any,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Compute TFR on an explicit frequency grid.

    Returns:
        - power: mne.EpochsTFR / AverageTFR / ndarray
        - metadata: axes + params
    """
    legacy_keys = {"fmin", "fmax", "n_freqs", "log_grid"}
    unexpected_keys = sorted(set(legacy_freq_grid_kwargs).difference(legacy_keys))
    if unexpected_keys:
        raise TypeError(f"Unexpected keyword arguments: {unexpected_keys}.")

    freqs_use = np.asarray(freqs, dtype=float).ravel()
    if freqs_use.ndim != 1 or freqs_use.size < 2:
        raise ValueError("`freqs` must be a 1D array with at least 2 elements.")
    if np.any(~np.isfinite(freqs_use)) or np.any(freqs_use <= 0):
        raise ValueError("`freqs` must be finite and > 0.")
    if not np.all(np.diff(freqs_use) > 0):
        raise ValueError("`freqs` must be strictly increasing.")
    freqs_source = "explicit"
    fmin_req = float(freqs_use[0])
    fmax_req = float(freqs_use[-1])
    n_freqs_req = int(freqs_use.size)
    log_grid_req = None

    if isinstance(data, mne.io.BaseRaw):
        sfreq = float(data.info["sfreq"])
    elif isinstance(data, mne.Epochs):
        sfreq = float(data.info["sfreq"])
    else:
        raise TypeError("`data` must be mne.Epochs or mne.io.BaseRaw.")

    decim_eff, hop_s_eff = compute_decimation(sfreq, hop_s=hop_s, decim=decim)

    n_cycles_vec, cycles_source = _compute_n_cycles(
        method=method,
        freqs=freqs_use,
        time_resolution_s=time_resolution_s,
        cycles_strategy=cycles_strategy,
        n_cycles_constant=n_cycles_constant,
        fwhm_power_hz=fwhm_power_hz,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
    )

    method_l = method.lower()

    pick_names: list[str] | None = None
    pick_indices: np.ndarray | None = None
    if picks is not None:
        pick_names = [str(item) for item in picks if str(item).strip()]
        pick_names = [item for item in pick_names if item in data.ch_names]
        if not pick_names:
            raise ValueError("No valid picks selected for TFR grid.")
        pick_indices = mne.pick_channels(data.ch_names, pick_names, ordered=True)

    # --- compute TFR ---
    if isinstance(data, mne.Epochs):
        if method_l == "morlet":
            power = tfr_morlet(
                data,
                freqs=freqs_use,
                n_cycles=n_cycles_vec,
                return_itc=return_itc,
                decim=decim_eff,
                average=average,
                picks=pick_indices,
                n_jobs=n_jobs,
            )
            fwhm_time_est = np.array(
                [mne_fwhm(f, n) for f, n in zip(freqs_use, n_cycles_vec)]
            )
            fwhm_power_est = 1.665 * freqs_use / n_cycles_vec
            bandwidth_hz_est = None
            time_window_s = None
        else:
            power = tfr_multitaper(
                data,
                freqs=freqs_use,
                n_cycles=n_cycles_vec,
                time_bandwidth=time_bandwidth,
                return_itc=return_itc,
                decim=decim_eff,
                average=average,
                picks=pick_indices,
                n_jobs=n_jobs,
            )
            T = n_cycles_vec / freqs_use
            time_window_s = T
            bandwidth_hz_est = time_bandwidth / T
            fwhm_time_est = None
            fwhm_power_est = None

    else:
        X = data.get_data(picks=pick_names)[None, :, :]  # (1, n_channels, n_times)
        if method_l == "morlet":
            power = tfr_array_morlet(
                X,
                sfreq=sfreq,
                freqs=freqs_use,
                n_cycles=n_cycles_vec,
                output="power",
                decim=decim_eff,
                n_jobs=n_jobs,
                use_fft=True,
            )
            fwhm_time_est = np.array(
                [mne_fwhm(f, n) for f, n in zip(freqs_use, n_cycles_vec)]
            )
            fwhm_power_est = 1.665 * freqs_use / n_cycles_vec
            bandwidth_hz_est = None
            time_window_s = None
        else:
            power = tfr_array_multitaper(
                X,
                sfreq=sfreq,
                freqs=freqs_use,
                n_cycles=n_cycles_vec,
                time_bandwidth=time_bandwidth,
                output="power",
                decim=decim_eff,
                n_jobs=n_jobs,
                use_fft=True,
            )
            T = n_cycles_vec / freqs_use
            time_window_s = T
            bandwidth_hz_est = time_bandwidth / T
            fwhm_time_est = None
            fwhm_power_est = None

    # --- metadata ---
    if isinstance(power, np.ndarray):
        power_data = power
        ch_names = list(pick_names) if pick_names is not None else list(data.ch_names)
        times_axis = (
            np.asarray(data.times[::decim_eff])
            if decim_eff is not None
            else np.asarray(data.times)
        )
    else:
        power_data = power.data
        ch_names = list(getattr(power, "ch_names", getattr(data, "ch_names", [])))
        times_axis = np.asarray(power.times)

    if power_data.ndim == 4:
        n_epochs_meta, n_channels_meta, n_freqs_meta, n_times_meta = power_data.shape
    elif power_data.ndim == 3:
        n_epochs_meta, n_channels_meta, n_freqs_meta, n_times_meta = (
            1,
            *power_data.shape,
        )
    else:  # pragma: no cover
        n_epochs_meta = n_channels_meta = n_freqs_meta = n_times_meta = None

    metadata = dict(
        axes=dict(
            epoch=(
                np.arange(int(n_epochs_meta), dtype=int)
                if n_epochs_meta is not None
                else None
            ),
            channel=np.array(ch_names, dtype=object),
            freq=np.asarray(freqs_use, dtype=float),
            time=np.asarray(times_axis, dtype=float),
            shape=(
                (
                    int(n_epochs_meta),
                    int(n_channels_meta),
                    int(n_freqs_meta),
                    int(n_times_meta),
                )
                if n_epochs_meta is not None
                else None
            ),
        ),
        params=dict(
            method=method_l,
            fmin=float(freqs_use[0]),
            fmax=float(freqs_use[-1]),
            n_freqs=int(freqs_use.size),
            log_grid=None,
            freqs_source=str(freqs_source),
            fmin_requested=fmin_req,
            fmax_requested=fmax_req,
            n_freqs_requested=n_freqs_req,
            log_grid_requested=log_grid_req,
            time_resolution_s=time_resolution_s,
            cycles_strategy=cycles_strategy,
            n_cycles_constant=float(n_cycles_constant),
            fwhm_power_hz=fwhm_power_hz,
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            time_bandwidth=(
                float(time_bandwidth) if method_l == "multitaper" else None
            ),
            hop_s=hop_s,
            hop_s_eff=hop_s_eff,
            decim=decim,
            decim_eff=decim_eff,
            average=average,
            return_itc=return_itc,
            n_jobs=n_jobs,
            cycles_source=cycles_source,
            n_cycles=n_cycles_vec,
            freqs=freqs_use,
            fwhm_time_s_est=fwhm_time_est,
            fwhm_power_hz_est=fwhm_power_est,
            bandwidth_hz_est=bandwidth_hz_est,
            time_window_s=time_window_s,
        ),
    )

    return power, metadata
