"""Shared helpers for time-frequency analysis.

All code comments and docstrings are in English (per project rules).
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

try:
    import mne
except Exception:  # pragma: no cover
    mne = None  # type: ignore[assignment]


FreqGridKind = Literal["linear", "log"]


def make_frequency_grid(
    fmin: float,
    fmax: float,
    n_freqs: int,
    *,
    kind: FreqGridKind = "linear",
) -> np.ndarray:
    """Create a frequency grid.

    Args:
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz).
        n_freqs: Number of frequencies.
        kind: "linear" or "log".

    Returns:
        1D float array of frequencies (Hz).

    Raises:
        ValueError: If inputs are invalid.
    """
    if not (isinstance(n_freqs, int) and n_freqs >= 1):
        raise ValueError("`n_freqs` must be an integer >= 1.")
    if not (isinstance(fmin, (int, float)) and isinstance(fmax, (int, float))):
        raise ValueError("`fmin` and `fmax` must be numeric.")
    fmin_f = float(fmin)
    fmax_f = float(fmax)
    if fmin_f <= 0 or fmax_f <= 0:
        raise ValueError("`fmin` and `fmax` must be > 0.")
    if fmax_f <= fmin_f:
        raise ValueError("`fmax` must be > `fmin`.")

    if kind == "log":
        return np.logspace(
            np.log10(fmin_f), np.log10(fmax_f), num=n_freqs, endpoint=True
        )
    if kind == "linear":
        return np.linspace(fmin_f, fmax_f, num=n_freqs, endpoint=True)
    raise ValueError("`kind` must be 'linear' or 'log'.")


def compute_decimation(
    sfreq_hz: float,
    *,
    hop_s: float | None = 0.025,
    decim: int | None = None,
) -> tuple[int, float | None]:
    """Compute an effective decimation factor and its implied hop (seconds).

    This mirrors the behavior of the original implementation, but centralizes
    it for reuse across TFR and connectivity code.

    Args:
        sfreq_hz: Sampling rate in Hz.
        hop_s: Target hop size in seconds. Used only if `decim` is None.
        decim: Explicit decimation factor. If provided, overrides `hop_s`.

    Returns:
        (decim_eff, hop_s_eff) where hop_s_eff is None if decim_eff == 1.
    """
    sfreq = float(sfreq_hz)
    if sfreq <= 0:
        raise ValueError("`sfreq_hz` must be > 0.")

    if decim is not None:
        decim_eff = int(decim)
        if decim_eff <= 0:
            raise ValueError("`decim` must be a positive integer.")
    else:
        if hop_s is None or float(hop_s) <= 0:
            decim_eff = 1
        else:
            decim_eff = max(1, int(round(sfreq * float(hop_s))))

    hop_s_eff = None if decim_eff == 1 else (decim_eff / sfreq)
    return decim_eff, hop_s_eff


def infer_sfreq_from_times(
    times_s: Sequence[float], *, default: float | None = None
) -> float:
    """Infer a sampling rate from a time axis.

    Args:
        times_s: Time points in seconds (may include NaNs).
        default: Value to return if inference fails.

    Returns:
        Sampling rate (Hz) inferred from the median time step.

    Raises:
        ValueError: If inference fails and `default` is None.
    """
    t = np.asarray(times_s, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        if default is not None:
            return float(default)
        raise ValueError("Not enough finite time points to infer sampling rate.")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        if default is not None:
            return float(default)
        raise ValueError("Could not infer sampling rate from time axis.")
    return float(1.0 / np.median(dt))


def morlet_n_cycles_from_time_fwhm(
    freqs_hz: Sequence[float],
    *,
    time_fwhm_s: float,
    min_cycles: float | None = None,
    max_cycles: float | None = None,
) -> np.ndarray:
    """Convert a constant time-domain FWHM (seconds) into Morlet n_cycles(f).

    Uses the same formula as MNE's documentation for Morlet wavelets.

        n_cycles(f) = FWHM_t * pi * f / sqrt(2*ln(2))

    Args:
        freqs_hz: Frequency grid (Hz).
        time_fwhm_s: Desired time-domain FWHM in seconds.
        min_cycles: Optional lower bound applied after conversion.
        max_cycles: Optional upper bound applied after conversion.

    Returns:
        1D float array of n_cycles per frequency.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    if freqs.ndim != 1 or freqs.size < 1:
        raise ValueError("`freqs_hz` must be a 1D array with at least one element.")
    if float(time_fwhm_s) <= 0:
        raise ValueError("`time_fwhm_s` must be > 0.")

    n_cycles = float(time_fwhm_s) * np.pi * freqs / np.sqrt(2.0 * np.log(2.0))

    if min_cycles is not None:
        n_cycles = np.maximum(n_cycles, float(min_cycles))
    if max_cycles is not None:
        max_c = float(max_cycles)
        if max_c <= 0:
            raise ValueError("`max_cycles` must be > 0 if provided.")
        n_cycles = np.minimum(n_cycles, max_c)

    return n_cycles


def morlet_n_cycles_from_time_fwhm_vector(
    freqs_hz: Sequence[float],
    *,
    fwhm_time_s: float | Sequence[float],
    min_cycles: float | None = None,
    max_cycles: float | None = None,
) -> np.ndarray:
    """Convert a per-frequency time-domain FWHM (seconds) into Morlet n_cycles(f).

    This is the vector-valued counterpart of :func:`morlet_n_cycles_from_time_fwhm`.

    Uses the Morlet Gaussian-envelope relationship:

        n_cycles(f) = FWHM_t(f) * pi * f / sqrt(2*ln(2))

    Args:
        freqs_hz: Frequency grid (Hz).
        fwhm_time_s: Per-frequency time-domain FWHM (seconds), same length as freqs_hz,
            OR a single float (constant FWHM for all frequencies).
        min_cycles: Optional lower bound applied after conversion.
        max_cycles: Optional upper bound applied after conversion.

    Returns:
        1D float array of n_cycles per frequency.
    """
    freqs = np.asarray(freqs_hz, dtype=float).ravel()

    if freqs.ndim != 1 or freqs.size < 1:
        raise ValueError("`freqs_hz` must be a 1D array with at least one element.")
    if np.any(~np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError("`freqs_hz` must be finite and > 0.")

    # Allow scalar fwhm_time_s (float) or per-frequency vector
    if np.isscalar(fwhm_time_s):
        fwhm_scalar = float(fwhm_time_s)
        if not np.isfinite(fwhm_scalar) or fwhm_scalar <= 0:
            raise ValueError("`fwhm_time_s` must be finite and > 0.")
        fwhm = np.full(freqs.size, fwhm_scalar, dtype=float)
    else:
        fwhm = np.asarray(fwhm_time_s, dtype=float).ravel()
        if fwhm.ndim != 1 or fwhm.size != freqs.size:
            raise ValueError(
                "`fwhm_time_s` must be 1D and have the same length as `freqs_hz`, "
                "or be a single float."
            )
        if np.any(~np.isfinite(fwhm)) or np.any(fwhm <= 0):
            raise ValueError("`fwhm_time_s` must be finite and > 0.")

    n_cycles = fwhm * np.pi * freqs / np.sqrt(2.0 * np.log(2.0))

    if min_cycles is not None:
        n_cycles = np.maximum(n_cycles, float(min_cycles))
    if max_cycles is not None:
        max_c = float(max_cycles)
        if max_c <= 0:
            raise ValueError("`max_cycles` must be > 0 if provided.")
        n_cycles = np.minimum(n_cycles, max_c)

    return n_cycles


def multitaper_n_cycles_from_time_fwhm_vector(
    freqs_hz: Sequence[float],
    *,
    fwhm_time_s: float | Sequence[float],
    min_cycles: float | None = None,
    max_cycles: float | None = None,
) -> np.ndarray:
    """Convert a per-frequency time window (seconds) into multitaper n_cycles(f).

    For multitaper, MNE uses:

        T(f) = n_cycles(f) / f

    Therefore:

        n_cycles(f) = T(f) * f

    Here ``fwhm_time_s`` is interpreted as the desired multitaper window length
    ``T`` (seconds), and may be scalar or per-frequency.

    Args:
        freqs_hz: Frequency grid (Hz).
        fwhm_time_s: Per-frequency window length in seconds, same length as
            ``freqs_hz``, OR a single float (constant window for all frequencies).
        min_cycles: Optional lower bound applied after conversion.
        max_cycles: Optional upper bound applied after conversion.

    Returns:
        1D float array of n_cycles per frequency.
    """
    freqs = np.asarray(freqs_hz, dtype=float).ravel()

    if freqs.ndim != 1 or freqs.size < 1:
        raise ValueError("`freqs_hz` must be a 1D array with at least one element.")
    if np.any(~np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError("`freqs_hz` must be finite and > 0.")

    if np.isscalar(fwhm_time_s):
        t_scalar = float(fwhm_time_s)
        if not np.isfinite(t_scalar) or t_scalar <= 0:
            raise ValueError("`fwhm_time_s` must be finite and > 0.")
        t_win = np.full(freqs.size, t_scalar, dtype=float)
    else:
        t_win = np.asarray(fwhm_time_s, dtype=float).ravel()
        if t_win.ndim != 1 or t_win.size != freqs.size:
            raise ValueError(
                "`fwhm_time_s` must be 1D and have the same length as `freqs_hz`, "
                "or be a single float."
            )
        if np.any(~np.isfinite(t_win)) or np.any(t_win <= 0):
            raise ValueError("`fwhm_time_s` must be finite and > 0.")

    n_cycles = t_win * freqs

    if min_cycles is not None:
        n_cycles = np.maximum(n_cycles, float(min_cycles))
    if max_cycles is not None:
        max_c = float(max_cycles)
        if max_c <= 0:
            raise ValueError("`max_cycles` must be > 0 if provided.")
        n_cycles = np.minimum(n_cycles, max_c)

    return n_cycles


def morlet_mask_radius_time_s_from_freqs_n_cycles(
    freqs_hz: Sequence[float | str],
    *,
    n_cycles: Sequence[float],
    k_sigma: float = 5.0,
) -> np.ndarray:
    """Compute per-frequency time radii (seconds) for dynamic masking.

    The returned radius corresponds to the padding used by
    :func:`lfp.pipelines.masking.mask_tensor_dynamic` when expanding a matched
    annotation interval.

    For Morlet wavelets, we approximate the temporal standard deviation of the
    Gaussian envelope as:

        sigma_t(f) = n_cycles(f) / (2*pi*f)

    and define a conservative contamination radius as:

        time_radius_s(f) = k_sigma * sigma_t(f)

    Args:
        freqs_hz: Frequencies in Hz. Elements may be numeric, or strings that
            can be converted to float (e.g., '20', '20.0').
        n_cycles: Wavelet cycles per frequency, same length as ``freqs_hz``.
        k_sigma: Multiplier applied to sigma_t (default 5).

    Returns:
        1D float array of time radii in seconds, same length as ``freqs_hz``.

    Raises:
        ValueError: If inputs are invalid.
    """
    freqs = np.asarray([float(f) for f in freqs_hz], dtype=float).ravel()
    cycles = np.asarray(list(n_cycles), dtype=float).ravel()

    if freqs.ndim != 1 or freqs.size < 1:
        raise ValueError("`freqs_hz` must be a 1D array with at least one element.")
    if cycles.ndim != 1 or cycles.size != freqs.size:
        raise ValueError(
            "`n_cycles` must be 1D and have the same length as `freqs_hz`."
        )

    k = float(k_sigma)
    if not (k >= 0 and float("inf") > k):
        raise ValueError("`k_sigma` must be finite and >= 0.")

    if np.any(~np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError("`freqs_hz` must be finite and > 0.")
    if np.any(~np.isfinite(cycles)) or np.any(cycles <= 0):
        raise ValueError("`n_cycles` must be finite and > 0.")

    sigma_t_s = cycles / (2.0 * np.pi * freqs)
    return k * sigma_t_s


def multitaper_mask_radius_time_s_from_freqs_n_cycles(
    freqs_hz: Sequence[float | str],
    *,
    n_cycles: Sequence[float],
    k_window: float = 0.5,
) -> np.ndarray:
    """Compute per-frequency time radii (seconds) for dynamic masking.

    For multitaper, each time-frequency estimate uses a finite window:

        T(f) = n_cycles(f) / f

    The contamination radius around an annotation boundary is naturally one
    half-window by default:

        time_radius_s(f) = k_window * T(f)

    with ``k_window=0.5`` giving exactly the half-window radius.

    Args:
        freqs_hz: Frequencies in Hz. Elements may be numeric, or strings that
            can be converted to float (e.g., '20', '20.0').
        n_cycles: Multitaper cycles per frequency, same length as ``freqs_hz``.
        k_window: Multiplier applied to ``T(f)`` (default 0.5).

    Returns:
        1D float array of time radii in seconds, same length as ``freqs_hz``.

    Raises:
        ValueError: If inputs are invalid.
    """
    freqs = np.asarray([float(f) for f in freqs_hz], dtype=float).ravel()
    cycles = np.asarray(list(n_cycles), dtype=float).ravel()

    if freqs.ndim != 1 or freqs.size < 1:
        raise ValueError("`freqs_hz` must be a 1D array with at least one element.")
    if cycles.ndim != 1 or cycles.size != freqs.size:
        raise ValueError(
            "`n_cycles` must be 1D and have the same length as `freqs_hz`."
        )

    k = float(k_window)
    if not (k >= 0 and float("inf") > k):
        raise ValueError("`k_window` must be finite and >= 0.")

    if np.any(~np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError("`freqs_hz` must be finite and > 0.")
    if np.any(~np.isfinite(cycles)) or np.any(cycles <= 0):
        raise ValueError("`n_cycles` must be finite and > 0.")

    t_window_s = cycles / freqs
    return k * t_window_s


def decimated_times_from_raw(
    raw: "mne.io.BaseRaw",
    *,
    decim: int,
    target_n_times: int | None = None,
) -> np.ndarray:
    """Create a decimated time grid from Raw that matches MNE decim behavior.

    Args:
        raw: MNE Raw.
        decim: Decimation factor (positive integer).
        target_n_times: Optional fixed length. If longer than available, the
            output is right-padded with NaNs; if shorter, it is trimmed.

    Returns:
        1D array of times in seconds.
    """
    if mne is None:  # pragma: no cover
        raise ModuleNotFoundError("mne is required for `decimated_times_from_raw`.")
    if not isinstance(decim, int) or decim <= 0:
        raise ValueError("`decim` must be a positive integer.")

    times = np.asarray(raw.times[::decim], dtype=float)

    if target_n_times is not None:
        target = int(target_n_times)
        if target <= 0:
            raise ValueError("`target_n_times` must be > 0.")
        if times.size > target:
            times = times[:target]
        elif times.size < target:
            pad = np.full(target - times.size, np.nan, dtype=float)
            times = np.concatenate([times, pad], axis=0)

    return times


def channel_names_after_picks(
    raw: "mne.io.BaseRaw",
    picks: list[str] | None,
) -> list[str]:
    """Return channel names after applying an optional `picks` list.

    This helper mirrors MNE's channel picking behavior and returns channel names
    in the order they appear in `raw`.

    Args:
        raw: MNE Raw instance.
        picks: None (keep all channels) or a list of channel names to include.

    Returns:
        List of channel names in the picked order.

    Raises:
        ModuleNotFoundError: If MNE is not available.
        TypeError: If `picks` is not None or list[str].
    """
    if mne is None:  # pragma: no cover
        raise ModuleNotFoundError("mne is required for `channel_names_after_picks`.")

    if picks is None:
        return list(raw.ch_names)

    if not isinstance(picks, list) or not all(isinstance(p, str) for p in picks):
        raise TypeError("`picks` must be None or a list[str] of channel names.")

    pick_inds = mne.pick_channels(raw.ch_names, include=picks, exclude=[])
    return [raw.ch_names[i] for i in pick_inds]
