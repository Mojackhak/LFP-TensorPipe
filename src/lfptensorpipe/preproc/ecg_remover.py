"""
ECG artifact removal utilities for LFP signals.

This module provides multiple strategies to reduce ECG contamination in DBS-LFP
(or similar LFP) recordings:

1) Template fitting + subtraction ("template")
   - Detect QRS-like peaks in a baseline-stabilized signal.
   - Build an average QRS (or PQRST) template.
   - Fit (scale, offset) per-beat and subtract from the raw signal.

2) Correlation-based detection + mirror replacement ("perceive")
   - Build an initial template by segmenting and cross-correlation alignment.
   - Refine/crop the template to the QRS region.
   - Detect ECG peaks via correlation with adaptive thresholding.
   - Remove artifacts by replacing the contaminated segment with mirrored samples.

3) Epoch-matrix SVD reconstruction + subtraction ("svd")
   - Extract epochs around detected ECG peaks.
   - Apply SVD across the epoch matrix.
   - Reconstruct the ECG artifact using a small number of components.
   - Subtract the reconstructed artifact (with offset fit) from each epoch.

Reference (inspiration): https://doi.org/10.1016/j.clinph.2022.11.011

Notes
-----
- All algorithms expect a 1D array (single channel) and the sampling rate in Hz.
- Units: most steps are unit-agnostic (z-scored detection), but some amplitude
  thresholds (e.g., threshold_v) depend on the input units.
- Plotting and MNE integration are optional and imported lazily.

This file is intentionally self-contained so it can be dropped into a project
without additional package scaffolding.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import correlate, find_peaks
from scipy.stats import zscore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Orientation = Literal["positive", "negative"]
MethodName = Literal["template", "perceive", "svd"]


class ECGRemovalError(ValueError):
    """Raised when ECG removal cannot proceed due to invalid inputs."""


@dataclass(frozen=True, slots=True)
class TemplateFitConfig:
    """Configuration for the template fitting ECG remover."""
    window_ms: float = 200.0
    peak_height_range: tuple[float, float] = (2.5, float("inf"))
    min_interpeak_ms: float = 300.0
    force_orientation: Orientation | None = None
    pre_ms: float = 150.0
    post_ms: float = 150.0
    tail_ms: float = 60.0
    qrs_duration_ms: float = 120.0
    pqrst: bool = False


@dataclass(frozen=True, slots=True)
class PerceiveConfig:
    """Configuration for the correlation + mirror replacement ECG remover."""
    epoch_length_ms: float = 1000.0
    window_ms: float = 200.0
    threshold_v: float = 200e-6
    pad_ms: float = 15.0
    min_bpm: int = 40
    max_bpm: int = 180
    threshold_start: float | None = None
    threshold_step: float | None = None
    max_threshold_tries: int = 100
    pass_rate: float = 0.95
    before_ms: float = 50.0
    after_ms: float = 100.0
    enforce_max_interval: bool = True


@dataclass(frozen=True, slots=True)
class SvdConfig:
    """Configuration for the SVD-based ECG remover."""
    components: int = 2
    window_ms: float = 200.0
    peak_height_range: tuple[float, float] = (2.5, float("inf"))
    min_interpeak_ms: float = 300.0
    force_orientation: Orientation | None = None
    pre_ms: float = 150.0
    post_ms: float = 150.0
    tail_ms: float = 60.0
    qrs_duration_ms: float = 120.0
    pqrst: bool = False


@dataclass(slots=True)
class ECGRemovalDiagnostics:
    """Diagnostics returned by ECG removal methods."""
    method: str
    fs: float
    r_peaks: NDArray[np.int_] | None = None
    orientation: Orientation | None = None
    template: NDArray[np.float64] | None = None
    epoch_start: int | None = None
    epoch_end: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Validation and small utilities
# -----------------------------------------------------------------------------
def _as_1d_float(x: np.ndarray | Sequence[float]) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ECGRemovalError(f"lfp_signal must be 1D, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ECGRemovalError("lfp_signal contains NaN/Inf values.")
    return arr


def _validate_fs(fs: float) -> float:
    fs_f = float(fs)
    if not np.isfinite(fs_f) or fs_f <= 0:
        raise ECGRemovalError(f"fs must be a positive finite number, got {fs!r}")
    return fs_f


def _ms_to_samples(ms: float, fs: float) -> int:
    # Use rounding to reduce systematic bias when ms does not map to an integer.
    n = int(round((ms / 1000.0) * fs))
    return max(n, 1)


def _robust_scale_1d(
    x: NDArray[np.float64],
    *,
    q_low: float,
    q_high: float,
    eps: float = 1e-120,
) -> NDArray[np.float64]:
    """
    Robustly scale a 1D array using quantiles in [0, 1] (fractions, not percents).

    Returns (x - median) / (q_high - q_low).
    """
    q_low_c = float(np.clip(q_low, 0.0, 0.49))
    q_high_c = float(np.clip(q_high, 0.51, 1.0))
    if q_low_c >= q_high_c:
        raise ECGRemovalError("q_low must be smaller than q_high for robust scaling.")

    center = float(np.median(x))
    lo = float(np.quantile(x, q_low_c))
    hi = float(np.quantile(x, q_high_c))
    scale = max(hi - lo, eps)
    return (x - center) / scale


# -----------------------------------------------------------------------------
# Baseline / feature extraction
# -----------------------------------------------------------------------------
def lfp_baseline(lfp_signal: np.ndarray, fs: float, window_ms: float = 200.0) -> NDArray[np.float64]:
    """
    Estimate a slow baseline using a median filter.

    Parameters
    ----------
    lfp_signal
        1D LFP signal.
    fs
        Sampling rate (Hz).
    window_ms
        Median filter window length in milliseconds.

    Returns
    -------
    baseline
        Baseline estimate (same shape as input).
    """
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)
    size = _ms_to_samples(window_ms, fs_f)
    return median_filter(x, size=size)


def lfp_feature(lfp_signal: np.ndarray, fs: float, window_ms: float = 200.0) -> NDArray[np.float64]:
    """
    Compute a feature signal by subtracting a median-filter baseline.

    This is used to make QRS-like transients stand out for peak detection.
    """
    x = _as_1d_float(lfp_signal)
    base = lfp_baseline(x, fs, window_ms)
    return x - base


# -----------------------------------------------------------------------------
# Peak detection and template creation (template-fit / SVD)
# -----------------------------------------------------------------------------
def detect_qrs_peaks(
    lfp_signal: np.ndarray,
    fs: float,
    force_orientation: Orientation | None = None,
    peak_height_range: tuple[float, float] = (2.5, float("inf")),
    min_interpeak_ms: float = 300.0,
) -> tuple[NDArray[np.int_], Orientation | None]:
    """
    Detect putative QRS-like peaks in an LFP signal.

    The input is z-scored, then peaks are detected in both positive and negative
    orientation. The final orientation is selected based on the number of peaks
    and their mean height (unless force_orientation is provided).

    Returns
    -------
    peaks
        Peak indices in samples.
    orientation
        "positive", "negative", or None (if no peaks are found).
    """
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)

    mean_val = float(np.mean(x))
    std_val = float(np.std(x))
    if std_val < 1e-12:
        logger.warning("Signal variance too small for peak detection.")
        return np.array([], dtype=int), None

    z_signal = (x - mean_val) / std_val

    min_distance_samples = _ms_to_samples(min_interpeak_ms, fs_f)

    pos_peaks, _ = find_peaks(z_signal, height=peak_height_range, distance=min_distance_samples)
    pos_peak_heights = z_signal[pos_peaks] if len(pos_peaks) else np.array([], dtype=float)

    neg_signal = -z_signal
    neg_peaks, _ = find_peaks(neg_signal, height=peak_height_range, distance=min_distance_samples)
    neg_peak_heights = neg_signal[neg_peaks] if len(neg_peaks) else np.array([], dtype=float)

    if len(pos_peaks) == 0 and len(neg_peaks) == 0:
        logger.info("No QRS-like peaks detected.")
        return np.array([], dtype=int), None

    # Default selection: more peaks, then larger mean peak height
    num_pos = len(pos_peaks)
    num_neg = len(neg_peaks)
    mean_pos = float(np.mean(pos_peak_heights)) if num_pos else 0.0
    mean_neg = float(np.mean(neg_peak_heights)) if num_neg else 0.0

    chosen_peaks = pos_peaks
    orientation: Orientation = "positive"
    if (num_neg > num_pos) or ((num_neg == num_pos) and (mean_neg > mean_pos)):
        chosen_peaks = neg_peaks
        orientation = "negative"

    if force_orientation is not None:
        if force_orientation == "positive":
            chosen_peaks = pos_peaks
            orientation = "positive"
        elif force_orientation == "negative":
            chosen_peaks = neg_peaks
            orientation = "negative"
        else:
            raise ECGRemovalError(f"Invalid force_orientation={force_orientation!r}")

    logger.info("Detected %d QRS-like peaks (orientation=%s).", len(chosen_peaks), orientation)
    return np.asarray(chosen_peaks, dtype=int), orientation


def generate_qrs_template(
    lfp_signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray,
    input_orientation: Orientation | None,
    force_orientation: Orientation | None = None,
    pre_ms: float = 150.0,
    post_ms: float = 150.0,
    tail_ms: float = 60.0,
    qrs_duration_ms: float = 120.0,
    pqrst: bool = False,
) -> tuple[NDArray[np.float64], Orientation, int, int] | tuple[None, None, None, None]:
    """
    Generate an averaged QRS (or PQRST) template from an LFP signal.

    Returns
    -------
    template
        Cropped template (1D).
    orientation
        Chosen orientation of the main peak in the averaged template.
    epoch_start
        Start offset (samples) relative to each R-peak index.
    epoch_end
        End offset (samples, exclusive) relative to each R-peak index.

    Notes
    -----
    The template is first built by averaging fixed-length epochs around r_peaks,
    then cropped to minimize boundary discontinuity around the QRS region.
    """
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)
    r = np.asarray(r_peaks, dtype=int)

    if r.size == 0:
        warnings.warn("No R-peaks provided; cannot generate template.", RuntimeWarning)
        return None, None, None, None

    pre_samples = _ms_to_samples(pre_ms, fs_f)
    post_samples = _ms_to_samples(post_ms, fs_f)
    epoch_length = pre_samples + post_samples

    # Extract epochs
    epochs: list[NDArray[np.float64]] = []
    for peak_idx in r:
        start_idx = peak_idx - pre_samples
        end_idx = peak_idx + post_samples
        if start_idx < 0 or end_idx > x.size:
            continue
        epoch = x[start_idx:end_idx]
        if epoch.size == epoch_length:
            epochs.append(epoch)

    if not epochs:
        warnings.warn("No valid epochs extracted; cannot generate template.", RuntimeWarning)
        return None, None, None, None

    template_raw = np.mean(np.stack(epochs, axis=0), axis=0)

    # Z-score template for orientation + peak finding
    t_mean = float(np.mean(template_raw))
    t_std = float(np.std(template_raw))
    if t_std < 1e-12:
        warnings.warn("Template variance too small; cannot determine orientation.", RuntimeWarning)
        return None, None, None, None
    z_template = (template_raw - t_mean) / t_std

    pos_peak_idx = int(np.argmax(z_template))
    pos_peak_val = float(z_template[pos_peak_idx])
    neg_peak_idx = int(np.argmin(z_template))
    neg_peak_val = float(-z_template[neg_peak_idx])

    if force_orientation is None:
        if pos_peak_val >= neg_peak_val:
            final_orientation: Orientation = "positive"
            r_idx = pos_peak_idx
        else:
            final_orientation = "negative"
            r_idx = neg_peak_idx

        if input_orientation is not None and final_orientation != input_orientation:
            warnings.warn(
                "Template peak orientation differs from detect_qrs_peaks orientation "
                f"(template={final_orientation}, detect={input_orientation}). Using template orientation.",
                RuntimeWarning,
            )
    else:
        final_orientation = force_orientation
        r_idx = pos_peak_idx if force_orientation == "positive" else neg_peak_idx

    # Find Q and S candidates on the chosen orientation
    # If final_orientation == "positive", Q/S are likely negative deflections -> look for peaks in -z_template.
    template_for_qs = (-z_template) if final_orientation == "positive" else z_template

    # Convert to samples; divide by 1.5 to allow closer peaks than full QRS duration.
    qrs_duration_samples = max(int(round((qrs_duration_ms / 1500.0) * fs_f)), 1)

    all_peaks, _ = find_peaks(template_for_qs)
    if all_peaks.size < 1:
        warnings.warn("No Q or S peaks found in the averaged template.", RuntimeWarning)
        return None, None, None, None

    largest_peak_idx = int(np.argmax(template_for_qs))
    if largest_peak_idx < r_idx:
        q_idx = largest_peak_idx
        threshold_idx = max(q_idx + qrs_duration_samples, r_idx)
        right_peaks = all_peaks[all_peaks > threshold_idx]
        if right_peaks.size == 0:
            warnings.warn("No S peak found in the averaged template.", RuntimeWarning)
            return None, None, None, None
        s_idx = int(right_peaks[0])
    else:
        s_idx = largest_peak_idx
        threshold_idx = min(s_idx - qrs_duration_samples, r_idx)
        left_peaks = all_peaks[all_peaks < threshold_idx]
        if left_peaks.size == 0:
            warnings.warn("No Q peak found in the averaged template.", RuntimeWarning)
            return None, None, None, None
        q_idx = int(left_peaks[-1])

    tail_samples = _ms_to_samples(tail_ms, fs_f)

    if pqrst:
        left_start = 0
        left_end = min(tail_samples, template_raw.size)
        right_start = max(0, template_raw.size - tail_samples)
        right_end = template_raw.size
    else:
        left_start = max(0, q_idx - tail_samples)
        left_end = min(q_idx, template_raw.size)
        right_start = max(0, s_idx)
        right_end = min(s_idx + tail_samples, template_raw.size)

    left_block = template_raw[left_start:left_end]
    right_block = template_raw[right_start:right_end]
    if left_block.size == 0 or right_block.size == 0:
        # Fall back to the full epoch if we cannot crop safely.
        start_template_idx = 0
        end_template_idx = template_raw.size
    else:
        # Find the pair (i, j) that minimizes |left[i] - right[j]|.
        diff = np.abs(left_block[:, None] - right_block[None, :])
        i_min, j_min = np.unravel_index(int(np.argmin(diff)), diff.shape)
        start_template_idx = int(left_start + i_min)
        end_template_idx = int(right_start + j_min + 1)  # exclusive

    template_final = template_raw[start_template_idx:end_template_idx]
    if template_final.size < 2:
        warnings.warn("Cropped template is too short; using the full epoch template.", RuntimeWarning)
        template_final = template_raw.copy()
        start_template_idx = 0
        end_template_idx = template_raw.size

    # Make endpoints equal-ish to reduce discontinuity.
    if not np.isclose(template_final[0], template_final[-1]):
        pad_val = float(max(template_final[0], template_final[-1]))
        template_final = template_final.copy()
        template_final[0] = pad_val
        template_final[-1] = pad_val

    # Offsets relative to each R-peak index.
    epoch_start = start_template_idx - pre_samples
    epoch_end = epoch_start + int(template_final.size)  # exclusive

    return template_final.astype(float), final_orientation, epoch_start, epoch_end


def optimize_template(template: np.ndarray, lfp_epoch: np.ndarray) -> tuple[NDArray[np.float64], float, float, float]:
    """
    Fit (scale, offset) so that scale * template + offset best matches lfp_epoch.

    Uses a closed-form least-squares solution.
    """
    t = _as_1d_float(template)
    y = _as_1d_float(lfp_epoch)
    if t.size != y.size:
        raise ECGRemovalError(f"template and lfp_epoch must have same length, got {t.size} vs {y.size}")

    A = np.vstack([t, np.ones_like(t)]).T  # shape (N, 2)
    params, *_ = np.linalg.lstsq(A, y, rcond=None)
    scale_opt = float(params[0])
    offset_opt = float(params[1])

    fitted = scale_opt * t + offset_opt
    sse = float(np.sum((y - fitted) ** 2))
    return fitted, scale_opt, offset_opt, sse


# -----------------------------------------------------------------------------
# Removal methods
# -----------------------------------------------------------------------------
def template_ecg_remover(
    lfp_signal: np.ndarray,
    fs: float,
    window_ms: float = 200.0,
    peak_height_range: tuple[float, float] = (2.5, float("inf")),
    min_interpeak_ms: float = 300.0,
    force_orientation: Orientation | None = None,
    pre_ms: float = 150.0,
    post_ms: float = 150.0,
    tail_ms: float = 60.0,
    qrs_duration_ms: float = 120.0,
    pqrst: bool = False,
    return_figure: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], Any]:
    """
    ECG removal via per-beat template fitting and subtraction.

    Returns the cleaned signal. If return_figure=True, also returns a matplotlib
    figure of the estimated template.
    """
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)

    # Baseline-stabilized signal for detection
    feat = lfp_feature(x, fs_f, window_ms)

    r_peaks, input_orientation = detect_qrs_peaks(
        feat,
        fs_f,
        force_orientation=force_orientation,
        peak_height_range=peak_height_range,
        min_interpeak_ms=min_interpeak_ms,
    )

    if r_peaks.size == 0:
        logger.warning("No peaks detected; returning original signal.")
        cleaned = x.copy()
        if return_figure:
            return cleaned, None
        return cleaned

    template, orientation, epoch_start, epoch_end = generate_qrs_template(
        x,
        fs_f,
        r_peaks,
        input_orientation=input_orientation,
        force_orientation=force_orientation,
        pre_ms=pre_ms,
        post_ms=post_ms,
        tail_ms=tail_ms,
        qrs_duration_ms=qrs_duration_ms,
        pqrst=pqrst,
    )

    if template is None:
        logger.warning("Template generation failed; returning original signal.")
        cleaned = x.copy()
        if return_figure:
            return cleaned, None
        return cleaned

    cleaned = x.copy()

    # Fit + subtract per beat
    for peak in r_peaks:
        start = int(peak + epoch_start)
        end = int(peak + epoch_end)
        if start < 0 or end > x.size:
            continue
        lfp_epoch = x[start:end]
        fitted, _, _, _ = optimize_template(template, lfp_epoch)
        cleaned[start:end] = lfp_epoch - fitted

    fig = None
    if return_figure:
        fig = plot_vt(template, fs_f)
        return cleaned, fig
    return cleaned


def segment_signal(signal: np.ndarray, fs: float, epoch_length_ms: float = 1000.0) -> NDArray[np.float64]:
    """
    Break a 1D signal into non-overlapping epochs of fixed length.

    Returns
    -------
    epochs
        Array of shape (n_epochs, samples_per_epoch).
    """
    x = _as_1d_float(signal)
    fs_f = _validate_fs(fs)
    samples_per_epoch = _ms_to_samples(epoch_length_ms, fs_f)

    n_full_epochs = x.size // samples_per_epoch
    if n_full_epochs < 1:
        raise ECGRemovalError("Signal too short for the requested epoch_length_ms.")

    trimmed = x[: n_full_epochs * samples_per_epoch]
    return trimmed.reshape(n_full_epochs, samples_per_epoch)


def cross_correlation_align(epochs: NDArray[np.float64], fs: float, window_ms: float = 200.0) -> NDArray[np.float64]:
    """
    Align epochs via cross-correlation (on feature signal) and average to form a template.
    """
    fs_f = _validate_fs(fs)
    if epochs.ndim != 2:
        raise ECGRemovalError(f"epochs must be 2D, got shape={epochs.shape}")

    # Pick a reference epoch that does not have extreme outliers in its feature signal.
    ref_idx = 0
    for i, ep in enumerate(epochs):
        ref_features = lfp_feature(ep, fs_f, window_ms)
        ref_features_z = zscore(ref_features)
        if np.all(np.abs(ref_features_z) < 10):
            ref_idx = i
            break

    ref_epoch = epochs[ref_idx]
    ref_features = lfp_feature(ref_epoch, fs_f, window_ms)

    aligned_epochs: list[NDArray[np.float64]] = []
    for ep in epochs:
        ep_features = lfp_feature(ep, fs_f, window_ms)
        corr = correlate(ep_features, ref_features, mode="full")
        shift = int(np.argmax(corr) - (len(ep) - 1))

        if shift > 0:
            ep_shifted = np.concatenate([ep[shift:], np.zeros(shift)])
        elif shift < 0:
            s = abs(shift)
            ep_shifted = np.concatenate([np.zeros(s), ep[:-s]])
        else:
            ep_shifted = ep

        aligned_epochs.append(ep_shifted.astype(float))

    return np.mean(np.stack(aligned_epochs, axis=0), axis=0)


def find_ecg_template1(template: np.ndarray, fs: float, threshold_v: float = 200e-6, pad_ms: float = 15.0) -> NDArray[np.float64]:
    """
    Crop a template to the putative QRS region based on local extrema.

    The difference between the main extreme and flanking extrema must exceed
    threshold_v (in the same units as 'template').

    If cropping fails, returns the input template unchanged.
    """
    t = _as_1d_float(template)
    fs_f = _validate_fs(fs)

    abs_template = np.abs(t)
    max_idx = int(np.argmax(abs_template))
    max_value = float(t[max_idx])

    primary_extreme = "max" if max_value > 0 else "min"

    if primary_extreme == "max":
        minima, _ = find_peaks(-t)
        valid_minima = minima[np.abs(minima - max_idx).argsort()]
        flank_extrema = [int(i) for i in valid_minima if (max_value - float(t[i])) >= threshold_v]
    else:
        maxima, _ = find_peaks(t)
        valid_maxima = maxima[np.abs(maxima - max_idx).argsort()]
        flank_extrema = [int(i) for i in valid_maxima if (float(t[i]) - max_value) >= threshold_v]

    if len(flank_extrema) < 2:
        warnings.warn("Unable to find flank extrema satisfying threshold; returning original template.", RuntimeWarning)
        return t

    left_candidates = [i for i in flank_extrema if i < max_idx]
    right_candidates = [i for i in flank_extrema if i > max_idx]
    if not left_candidates or not right_candidates:
        warnings.warn("Unable to bracket QRS region; returning original template.", RuntimeWarning)
        return t

    left_extreme = left_candidates[0]
    right_extreme = right_candidates[0]

    def find_turning_point(signal_1d: NDArray[np.float64], peak_idx: int, direction: int) -> int:
        idx = int(peak_idx)
        while 1 <= idx < (signal_1d.size - 1):
            if direction == -1 and signal_1d[idx] < signal_1d[idx - 1]:
                break
            if direction == 1 and signal_1d[idx] < signal_1d[idx + 1]:
                break
            idx += int(direction)
        return idx

    if primary_extreme == "max":
        left_turning = find_turning_point(-t, left_extreme, -1)
        right_turning = find_turning_point(-t, right_extreme, 1)
    else:
        left_turning = find_turning_point(t, left_extreme, -1)
        right_turning = find_turning_point(t, right_extreme, 1)

    pad_samples = _ms_to_samples(pad_ms, fs_f)
    start_idx = max(0, left_turning - pad_samples)
    end_idx = min(t.size, right_turning + pad_samples + 1)
    return t[start_idx:end_idx]


def adaptive_threshold_peak_detection(
    lfp_signal: np.ndarray,
    template1: np.ndarray,
    fs: float,
    min_bpm: int = 40,
    max_bpm: int = 180,
    threshold_start: float | None = None,
    threshold_step: float | None = None,
    max_threshold_tries: int = 100,
    pass_rate: float = 0.95,
    *,
    enforce_max_interval: bool = True,
) -> tuple[NDArray[np.int_], float, NDArray[np.float64]] | tuple[None, None, None]:
    """
    Detect correlation peaks with adaptive thresholding and plausible inter-peak intervals.

    Returns (peaks, best_threshold, corr_values_scaled). If no suitable peaks are found,
    returns (None, None, None).
    """
    x = _as_1d_float(lfp_signal)
    t = _as_1d_float(template1)
    fs_f = _validate_fs(fs)

    if min_bpm <= 0 or max_bpm <= 0 or min_bpm >= max_bpm:
        raise ECGRemovalError("min_bpm and max_bpm must be positive with min_bpm < max_bpm.")
    if max_threshold_tries < 1:
        raise ECGRemovalError("max_threshold_tries must be >= 1.")
    if not (0.0 < pass_rate <= 1.0):
        raise ECGRemovalError("pass_rate must be in (0, 1].")

    correlation_values = correlate(x, t, mode="full").astype(float)

    # Heuristic robust scaling: use very low/high quantiles derived from expected beat rate.
    q_low = float(np.clip((min_bpm / 60.0) / fs_f, 1e-6, 0.1))
    corr_values_norm = _robust_scale_1d(correlation_values, q_low=q_low, q_high=1.0 - q_low)

    # Inter-peak constraints in samples
    min_peak_distance = int(fs_f / (max_bpm / 60.0))
    max_peak_distance = int(fs_f / (min_bpm / 60.0))

    # Threshold search range
    threshold_end = float(np.quantile(corr_values_norm, 1.0 - (q_low * 0.5)))

    if threshold_start is None:
        threshold_start = float(np.quantile(corr_values_norm, 0.6))
    if threshold_step is None:
        threshold_step = (threshold_end - threshold_start) / float(max_threshold_tries)

    best_peaks: NDArray[np.int_] | None = None
    best_threshold: float | None = None

    threshold = float(threshold_start)
    for _ in range(max_threshold_tries):
        peaks, _ = find_peaks(corr_values_norm, height=threshold)

        if peaks.size > 1:
            peak_distances = np.diff(peaks)
            if enforce_max_interval:
                valid = (peak_distances >= min_peak_distance) & (peak_distances <= max_peak_distance)
            else:
                valid = peak_distances >= min_peak_distance

            if float(np.sum(valid)) > float(valid.size) * float(pass_rate):
                best_peaks = peaks.astype(int)
                best_threshold = float(threshold)
                break

        threshold += float(threshold_step)
        if (threshold_step >= 0 and threshold >= threshold_end) or (threshold_step < 0 and threshold <= threshold_end):
            break

    if best_peaks is None or best_threshold is None:
        logger.warning("No suitable peaks found after adaptive thresholding.")
        return None, None, None

    logger.info("Adaptive threshold success: threshold=%.4f, n_peaks=%d", best_threshold, best_peaks.size)
    return best_peaks, best_threshold, corr_values_norm


def create_ecg_template(
    lfp_signal: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    before_ms: float = 50.0,
    after_ms: float = 100.0,
) -> NDArray[np.float64] | None:
    """
    Average epochs around detected peaks to build an ECG template.
    """
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)
    p = np.asarray(peaks, dtype=int)

    if p.size == 0:
        raise ECGRemovalError("peaks array cannot be empty.")

    before_samples = _ms_to_samples(before_ms, fs_f)
    after_samples = _ms_to_samples(after_ms, fs_f)

    epochs: list[NDArray[np.float64]] = []
    for peak in p:
        start = int(peak - before_samples)
        end = int(peak + after_samples + 1)
        if start >= 0 and end <= x.size:
            epochs.append(x[start:end])

    if not epochs:
        logger.warning("No valid epochs extracted for template creation.")
        return None

    return np.mean(np.stack(epochs, axis=0), axis=0)


def perceive_ecg_remover(
    lfp_signal: np.ndarray,
    fs: float,
    epoch_length_ms: float = 1000.0,
    window_ms: float = 200.0,
    threshold_v: float = 200e-6,
    pad_ms: float = 15.0,
    min_bpm: int = 40,
    max_bpm: int = 180,
    threshold_start: float | None = None,
    threshold_step: float | None = None,
    max_threshold_tries: int = 100,
    pass_rate: float = 0.95,
    before_ms: float = 50.0,
    after_ms: float = 100.0,
    enforce_max_interval: bool = True,
    return_figure: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], Any]:
    """
    ECG removal via correlation-based detection and mirror replacement.

    This is inspired by Perceive-like workflows. It is conservative: it does not
    attempt to estimate an ECG waveform to subtract, but replaces the artifact
    segment with mirrored samples from surrounding data.

    Returns the cleaned signal. If return_figure=True, also returns a figure of
    the refined template used for detection.
    """
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)

    epochs = segment_signal(x, fs_f, epoch_length_ms)
    template0 = cross_correlation_align(epochs, fs_f, window_ms)
    template1 = find_ecg_template1(template0, fs_f, threshold_v, pad_ms)

    peaks_corr, _, _ = adaptive_threshold_peak_detection(
        x,
        template1,
        fs_f,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        threshold_start=threshold_start,
        threshold_step=threshold_step,
        max_threshold_tries=max_threshold_tries,
        pass_rate=pass_rate,
        enforce_max_interval=enforce_max_interval,
    )
    if peaks_corr is None:
        cleaned = x.copy()
        if return_figure:
            return cleaned, None
        return cleaned

    peak_template1_idx = int(np.argmax(np.abs(template1)))
    peaks = peaks_corr - int(template1.size) + 1 + peak_template1_idx
    peaks = peaks[(peaks >= 0) & (peaks < x.size)]

    template_mean = create_ecg_template(x, peaks, fs_f, before_ms, after_ms)
    if template_mean is None:
        cleaned = x.copy()
        if return_figure:
            return cleaned, None
        return cleaned

    template_mean1 = find_ecg_template1(template_mean, fs_f, threshold_v, pad_ms)

    peaks_mean_corr, _, _ = adaptive_threshold_peak_detection(
        x,
        template_mean1,
        fs_f,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        threshold_start=threshold_start,
        threshold_step=threshold_step,
        max_threshold_tries=max_threshold_tries,
        pass_rate=pass_rate,
        enforce_max_interval=enforce_max_interval,
    )
    if peaks_mean_corr is None:
        cleaned = x.copy()
        if return_figure:
            return cleaned, None
        return cleaned

    peak_template_mean1_idx = int(np.argmax(np.abs(template_mean1)))
    peaks_mean = peaks_mean_corr - int(template_mean1.size) + 1 + peak_template_mean1_idx
    peaks_mean = peaks_mean[(peaks_mean >= -int(template_mean1.size)) & (peaks_mean < x.size + int(template_mean1.size))]

    template_len = int(template_mean1.size)
    before_samples = peak_template_mean1_idx
    after_samples = template_len - peak_template_mean1_idx - 1

    cleaned = x.copy()
    n = x.size

    # Mirror replacement uses the original signal x for sampling, writes into cleaned.
    for peak in peaks_mean.astype(int):
        start = peak - before_samples
        end = peak + after_samples + 1  # exclusive

        start_in = max(0, start)
        end_in = min(n, end)
        if start_in >= end_in:
            continue

        # Build full replacement (length template_len) in the "virtual" index space [start, end).
        interp_start = start - before_samples - 1
        left_slice = x[max(0, interp_start) : max(0, start)]
        mirror_before = left_slice[::-1]
        if mirror_before.size < (before_samples + 1):
            pad_len = (before_samples + 1) - int(mirror_before.size)
            mirror_before = np.pad(
                mirror_before,
                (pad_len, 0),
                mode="constant",
                constant_values=float(x[0]),
            )

        interp_end = end + after_samples
        right_slice = x[min(n, end) : min(n, interp_end)]
        mirror_after = right_slice[::-1]
        if mirror_after.size < after_samples:
            pad_len = after_samples - int(mirror_after.size)
            mirror_after = np.pad(
                mirror_after,
                (0, pad_len),
                mode="constant",
                constant_values=float(x[-1]),
            )
            mirror_after = mirror_after[::-1]

        replacement_full = np.concatenate([mirror_before, mirror_after])
        if replacement_full.size != template_len:
            # Extremely defensive: should never happen.
            replacement_full = replacement_full[:template_len]

        left_missing = start_in - start
        right_missing = end - end_in
        replacement_in = replacement_full[left_missing : template_len - right_missing]

        if replacement_in.size != (end_in - start_in):
            # Defensive: mismatch due to extreme edge cases.
            replacement_in = replacement_in[: (end_in - start_in)]

        cleaned[start_in:end_in] = replacement_in

    fig = None
    if return_figure:
        fig = plot_vt(template_mean1, fs_f)
        return cleaned, fig
    return cleaned


def svd_ecg_remover(
    lfp_signal: np.ndarray,
    fs: float,
    components: int = 2,
    window_ms: float = 200.0,
    peak_height_range: tuple[float, float] = (2.5, float("inf")),
    min_interpeak_ms: float = 300.0,
    force_orientation: Orientation | None = None,
    pre_ms: float = 150.0,
    post_ms: float = 150.0,
    tail_ms: float = 60.0,
    qrs_duration_ms: float = 120.0,
    pqrst: bool = False,
    return_figure: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], Any]:
    """
    Remove ECG contamination using an epoch-matrix SVD reconstruction.

    Returns the cleaned signal. If return_figure=True, also returns a figure of the
    estimated template.
    """
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)
    if components < 1:
        raise ECGRemovalError("components must be >= 1")

    # Detect peaks on baseline-stabilized signal
    feat = lfp_feature(x, fs_f, window_ms)
    r_peaks, input_orientation = detect_qrs_peaks(
        feat,
        fs_f,
        force_orientation=force_orientation,
        peak_height_range=peak_height_range,
        min_interpeak_ms=min_interpeak_ms,
    )
    if r_peaks.size == 0:
        cleaned = x.copy()
        if return_figure:
            return cleaned, None
        return cleaned

    template, _, epoch_start, epoch_end = generate_qrs_template(
        x,
        fs_f,
        r_peaks,
        input_orientation=input_orientation,
        force_orientation=force_orientation,
        pre_ms=pre_ms,
        post_ms=post_ms,
        tail_ms=tail_ms,
        qrs_duration_ms=qrs_duration_ms,
        pqrst=pqrst,
    )
    if template is None:
        cleaned = x.copy()
        if return_figure:
            return cleaned, None
        return cleaned

    # Remove peaks that would generate out-of-bounds epochs.
    valid_mask = (r_peaks + epoch_start >= 0) & (r_peaks + epoch_end <= x.size)
    r_peaks = r_peaks[valid_mask]
    if r_peaks.size < 2:
        cleaned = x.copy()
        if return_figure:
            fig = plot_vt(template, fs_f)
            return cleaned, fig
        return cleaned

    epoch_len = int(template.size)
    epochs = np.zeros((epoch_len, int(r_peaks.size)), dtype=float)
    for k, peak in enumerate(r_peaks):
        start_idx = int(peak + epoch_start)
        end_idx = int(peak + epoch_end)
        epochs[:, k] = x[start_idx:end_idx]

    # SVD: epochs = U * diag(S) * Vt
    U, S, Vt = svd(epochs, full_matrices=False)

    n_comp = min(int(components), int(S.size))
    reconstructed = np.zeros_like(epochs)
    for i in range(n_comp):
        reconstructed += float(S[i]) * np.outer(U[:, i], Vt[i, :])

    # Boundary handling inside each epoch
    tail_samples = min(_ms_to_samples(tail_ms, fs_f), max(1, epoch_len // 5))

    cleaned = x.copy()

    for k in range(reconstructed.shape[1]):
        ecg = reconstructed[:, k]

        # Find epoch-local crop boundaries by minimizing boundary mismatch.
        ecg_left = ecg[:tail_samples]
        ecg_right = ecg[epoch_len - tail_samples :]
        diff = np.abs(ecg_left[:, None] - ecg_right[None, :])
        i_min, j_min = np.unravel_index(int(np.argmin(diff)), diff.shape)
        start_idx_local = int(i_min)
        end_idx_local = int(epoch_len - tail_samples + j_min)  # index of last sample in right block
        end_idx_local = max(end_idx_local, start_idx_local + 1)

        # Convert to Python slice end (exclusive)
        start_crop = start_idx_local
        end_crop = min(end_idx_local + 1, epoch_len)

        # Pad endpoints to reduce discontinuity
        pad_val = float((ecg[start_crop] + ecg[end_crop - 1]) / 2.0)
        ecg_fit = ecg.copy()
        ecg_fit[start_crop] = pad_val
        ecg_fit[end_crop - 1] = pad_val

        # Fit only an offset (scale already captured by SVD) on the cropped part
        start_global = int(r_peaks[k] + epoch_start)
        end_global = int(r_peaks[k] + epoch_end)
        epoch = epochs[:, k]
        epoch_crop = epoch[start_crop:end_crop]
        ecg_crop = ecg_fit[start_crop:end_crop]

        def residual(offset: NDArray[np.float64]) -> NDArray[np.float64]:
            return epoch_crop - (ecg_crop + float(offset[0]))

        res = least_squares(residual, x0=np.array([0.0], dtype=float))
        offset_opt = float(res.x[0])

        ecg_fit = ecg_fit + offset_opt
        ecg_fit[start_crop] = pad_val
        ecg_fit[end_crop - 1] = pad_val
        ecg_fit[:start_crop] = 0.0
        ecg_fit[end_crop:] = 0.0

        corrected_epoch = epoch - ecg_fit
        cleaned[start_global:end_global] = corrected_epoch

    fig = None
    if return_figure:
        fig = plot_vt(template, fs_f)
        return cleaned, fig
    return cleaned


# -----------------------------------------------------------------------------
# High-level dispatch + wrappers
# -----------------------------------------------------------------------------
def remove_ecg_artifact(
    lfp_signal: np.ndarray,
    fs: float,
    method: MethodName = "template",
    *,
    template_config: TemplateFitConfig | None = None,
    perceive_config: PerceiveConfig | None = None,
    svd_config: SvdConfig | None = None,
    return_diagnostics: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], ECGRemovalDiagnostics]:
    """
    Unified entry point for ECG artifact removal on a single-channel LFP signal.

    Parameters
    ----------
    method
        One of: "template", "perceive", "svd".
    return_diagnostics
        If True, returns (cleaned, diagnostics).
    """
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)

    if method == "template":
        cfg = template_config or TemplateFitConfig()
        cleaned = template_ecg_remover(
            x,
            fs_f,
            window_ms=cfg.window_ms,
            peak_height_range=cfg.peak_height_range,
            min_interpeak_ms=cfg.min_interpeak_ms,
            force_orientation=cfg.force_orientation,
            pre_ms=cfg.pre_ms,
            post_ms=cfg.post_ms,
            tail_ms=cfg.tail_ms,
            qrs_duration_ms=cfg.qrs_duration_ms,
            pqrst=cfg.pqrst,
            return_figure=False,
        )
        diag = ECGRemovalDiagnostics(method="template", fs=fs_f, extra={"config": cfg})
    elif method == "perceive":
        cfg = perceive_config or PerceiveConfig()
        cleaned = perceive_ecg_remover(
            x,
            fs_f,
            epoch_length_ms=cfg.epoch_length_ms,
            window_ms=cfg.window_ms,
            threshold_v=cfg.threshold_v,
            pad_ms=cfg.pad_ms,
            min_bpm=cfg.min_bpm,
            max_bpm=cfg.max_bpm,
            threshold_start=cfg.threshold_start,
            threshold_step=cfg.threshold_step,
            max_threshold_tries=cfg.max_threshold_tries,
            pass_rate=cfg.pass_rate,
            before_ms=cfg.before_ms,
            after_ms=cfg.after_ms,
            enforce_max_interval=cfg.enforce_max_interval,
            return_figure=False,
        )
        diag = ECGRemovalDiagnostics(method="perceive", fs=fs_f, extra={"config": cfg})
    elif method == "svd":
        cfg = svd_config or SvdConfig()
        cleaned = svd_ecg_remover(
            x,
            fs_f,
            components=cfg.components,
            window_ms=cfg.window_ms,
            peak_height_range=cfg.peak_height_range,
            min_interpeak_ms=cfg.min_interpeak_ms,
            force_orientation=cfg.force_orientation,
            pre_ms=cfg.pre_ms,
            post_ms=cfg.post_ms,
            tail_ms=cfg.tail_ms,
            qrs_duration_ms=cfg.qrs_duration_ms,
            pqrst=cfg.pqrst,
            return_figure=False,
        )
        diag = ECGRemovalDiagnostics(method="svd", fs=fs_f, extra={"config": cfg})
    else:
        raise ECGRemovalError(f"Unknown method={method!r}")

    cleaned_arr = np.asarray(cleaned, dtype=float)
    if return_diagnostics:
        return cleaned_arr, diag
    return cleaned_arr


def call_ecgremover(
    method_func: Callable[..., Any],
    df: Any,
    fs: float | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """
    Backwards-compatible DataFrame API.

    Parameters
    ----------
    method_func
        Function with signature f(lfp_signal, fs, *args, **kwargs).
    df
        pandas.DataFrame with a time column + one or more signal columns.
    fs
        Sampling rate in Hz. If None, caller must ensure method_func can handle it.

    Returns
    -------
    df_clean, figs
        Cleaned DataFrame and optional figure dict (if method returns figures).
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("pandas is required for call_ecgremover().") from e

    if not isinstance(df, pd.DataFrame):
        raise ECGRemovalError("df must be a pandas.DataFrame")

    df_clean = df.copy()
    time_col = df.columns[0]
    ch_cols = [c for c in df.columns if c != time_col]

    figs: dict[str, Any] = {}
    for ch_col in ch_cols:
        lfp = df[ch_col].to_numpy(dtype=float)
        result = method_func(lfp, fs, *args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 2:
            df_clean[ch_col] = np.asarray(result[0], dtype=float)
            figs[ch_col] = result[1]
        else:
            df_clean[ch_col] = np.asarray(result, dtype=float)

    return df_clean, figs


def raw_call_ecgremover(
    raw: Any,
    method: str | Callable[..., Any],
    picks: Sequence[str] | str,
    *,
    inplace: bool = False,
    method_map: dict[str, Callable[..., Any]] | None = None,
    time_col_name: str = "Time",
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """
    Backwards-compatible MNE Raw adapter.

    This function is imported lazily to avoid a hard dependency on MNE.
    """
    try:
        import mne  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("mne is required for raw_call_ecgremover().") from e

    if not hasattr(raw, "info") or not hasattr(raw, "get_data"):
        raise ECGRemovalError("raw must be an MNE Raw-like object.")

    if isinstance(picks, str):
        picks = [picks]
    picks_list = list(picks)

    missing = sorted(set(picks_list) - set(raw.ch_names))
    if missing:
        raise ECGRemovalError(f"Channels not found in raw: {missing}")

    if method_map is None:
        method_map = {
            "template": template_ecg_remover,
            "perceive": perceive_ecg_remover,
            "svd": svd_ecg_remover,
        }

    if isinstance(method, str):
        if method not in method_map:
            raise ECGRemovalError(f"Unknown method '{method}'. Available: {list(method_map.keys())}")
        method_func = method_map[method]
        method_name = method
    elif callable(method):
        method_func = method
        method_name = getattr(method, "__name__", "custom_method")
    else:
        raise ECGRemovalError("method must be a string or a callable.")

    def _safe_method(lfp: np.ndarray, fs_local: float, **kw: Any) -> tuple[NDArray[np.float64], Any]:
        out = method_func(lfp, fs_local, **kw)
        if isinstance(out, tuple) and len(out) >= 2:
            return np.asarray(out[0], dtype=float), out[1]
        return np.asarray(out, dtype=float), None

    fs_local = float(raw.info["sfreq"])
    data = raw.get_data(picks=picks_list)  # (n_sel, n_times)
    t = raw.times

    import pandas as pd  # type: ignore

    df = pd.DataFrame({time_col_name: t})
    for ch, x in zip(picks_list, data):
        df[ch] = x

    if verbose:
        logger.info("Running ECG remover '%s' on channels: %s", method_name, picks_list)

    df_clean, figs = call_ecgremover(_safe_method, df, fs=fs_local, **kwargs)

    raw_out = raw if inplace else raw.copy()
    raw_out.load_data()
    cleaned_block = df_clean[picks_list].to_numpy(dtype=float).T

    idxs = mne.pick_channels(raw_out.ch_names, include=picks_list)
    if cleaned_block.shape != raw_out._data[idxs, :].shape:  # noqa: SLF001 (MNE uses _data internally)
        raise RuntimeError("Shape mismatch when writing back cleaned data.")
    raw_out._data[idxs, :] = cleaned_block

    if verbose:
        logger.info("ECG removal done for %d channel(s) with '%s'. inplace=%s", len(picks_list), method_name, inplace)

    return raw_out, figs


# -----------------------------------------------------------------------------
# Plotting (optional)
# -----------------------------------------------------------------------------
def plot_vt(arr: np.ndarray, fs: float) -> Any:
    """
    Plot a template waveform and return the matplotlib Figure.

    If matplotlib is not available, returns None.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return None

    x = _as_1d_float(arr)
    fs_f = _validate_fs(fs)

    t_ms = (np.arange(x.size) / fs_f) * 1000.0
    fig = plt.figure(figsize=(8, 6))
    plt.plot(t_ms, x)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (input units)")
    plt.title("ECG artifact template")
    plt.grid(True)
    return fig
