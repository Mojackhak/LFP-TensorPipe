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
- The MNE Raw adapter automatically excludes `BAD*` local supports for the three
  built-in methods while preserving BAD samples and annotations in its output.
- Plotting and MNE integration are optional and imported lazily.

This file is intentionally self-contained so it can be dropped into a project
without additional package scaffolding.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
from scipy.ndimage import maximum_filter, median_filter
from scipy.optimize import least_squares
from scipy.signal import correlate, find_peaks
from scipy.stats import zscore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Orientation = Literal["positive", "negative"]
MethodName = Literal["template", "perceive", "svd"]
ECG_FILTER_EDGE_DESCRIPTION = "EDGE_filter_post_ecg"
ANNOTATIONS_FILTER_EDGE_DESCRIPTION = "EDGE_filter_post_annotations"


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
class _ECGChannelRun:
    """Internal cleaned-channel result with JSON-safe run diagnostics."""

    cleaned: NDArray[np.float64]
    figure: Any = None
    candidate_beats: int = 0
    eligible_beats: int = 0
    bad_overlap_skipped_beats: int = 0
    corrected_beats: int = 0
    unchanged_reason: str = ""

    def diagnostics(self, channel: str) -> dict[str, Any]:
        return {
            "channel": str(channel),
            "candidate_beats": int(self.candidate_beats),
            "eligible_beats": int(self.eligible_beats),
            "bad_overlap_skipped_beats": int(self.bad_overlap_skipped_beats),
            "corrected_beats": int(self.corrected_beats),
            "unchanged_reason": str(self.unchanged_reason),
        }


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


def _bad_prefix_sum(bad_sample_mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Return a leading-zero prefix sum for interval BAD-overlap queries."""
    return np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(bad_sample_mask, dtype=np.int64),
        )
    )


def _interval_has_bad(
    bad_prefix: NDArray[np.int64],
    start: int,
    end: int,
) -> bool:
    """Return whether the in-bounds half-open interval contains BAD samples."""
    return bool(bad_prefix[int(end)] - bad_prefix[int(start)])


def _eligible_peak_mask(
    peaks: NDArray[np.int_],
    *,
    start_offset: int,
    end_offset: int,
    n_samples: int,
    bad_prefix: NDArray[np.int64],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return full-support eligibility and BAD-overlap masks for peak epochs."""
    eligible = np.zeros(peaks.size, dtype=bool)
    overlaps_bad = np.zeros(peaks.size, dtype=bool)
    for index, peak in enumerate(peaks.astype(int)):
        start = int(peak + start_offset)
        end = int(peak + end_offset)
        if start < 0 or end > n_samples or start >= end:
            continue
        overlaps_bad[index] = _interval_has_bad(bad_prefix, start, end)
        eligible[index] = not overlaps_bad[index]
    return eligible, overlaps_bad


def _feature_support_mask(
    bad_sample_mask: NDArray[np.bool_],
    *,
    window_samples: int,
) -> NDArray[np.bool_]:
    """Return positions whose median-filter support contains no BAD sample."""
    touches_bad = maximum_filter(
        bad_sample_mask,
        size=int(window_samples),
        mode="reflect",
    )
    return ~np.asarray(touches_bad, dtype=bool)


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
def lfp_baseline(
    lfp_signal: np.ndarray, fs: float, window_ms: float = 200.0
) -> NDArray[np.float64]:
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


def lfp_feature(
    lfp_signal: np.ndarray, fs: float, window_ms: float = 200.0
) -> NDArray[np.float64]:
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

    pos_peaks, _ = find_peaks(
        z_signal, height=peak_height_range, distance=min_distance_samples
    )
    pos_peak_heights = (
        z_signal[pos_peaks] if len(pos_peaks) else np.array([], dtype=float)
    )

    neg_signal = -z_signal
    neg_peaks, _ = find_peaks(
        neg_signal, height=peak_height_range, distance=min_distance_samples
    )
    neg_peak_heights = (
        neg_signal[neg_peaks] if len(neg_peaks) else np.array([], dtype=float)
    )

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

    logger.info(
        "Detected %d QRS-like peaks (orientation=%s).", len(chosen_peaks), orientation
    )
    return np.asarray(chosen_peaks, dtype=int), orientation


def _detect_qrs_peaks_masked(
    lfp_signal: NDArray[np.float64],
    fs: float,
    *,
    valid_sample_mask: NDArray[np.bool_],
    force_orientation: Orientation | None,
    peak_height_range: tuple[float, float],
    min_interpeak_ms: float,
) -> tuple[NDArray[np.int_], Orientation | None]:
    """Detect peaks globally while excluding invalid local-support positions."""
    valid_values = lfp_signal[valid_sample_mask]
    if valid_values.size == 0:
        return np.array([], dtype=int), None

    mean_val = float(np.mean(valid_values))
    std_val = float(np.std(valid_values))
    if std_val < 1e-12:
        logger.warning("Valid signal variance too small for peak detection.")
        return np.array([], dtype=int), None

    z_valid = (valid_values - mean_val) / std_val
    positive_score = np.full(lfp_signal.size, -np.inf, dtype=float)
    negative_score = np.full(lfp_signal.size, -np.inf, dtype=float)
    positive_score[valid_sample_mask] = z_valid
    negative_score[valid_sample_mask] = -z_valid

    min_distance_samples = _ms_to_samples(min_interpeak_ms, fs)
    pos_peaks, _ = find_peaks(
        positive_score,
        height=peak_height_range,
        distance=min_distance_samples,
    )
    neg_peaks, _ = find_peaks(
        negative_score,
        height=peak_height_range,
        distance=min_distance_samples,
    )

    if pos_peaks.size == 0 and neg_peaks.size == 0:
        logger.info("No QRS-like peaks detected outside BAD support.")
        return np.array([], dtype=int), None

    pos_heights = positive_score[pos_peaks]
    neg_heights = negative_score[neg_peaks]
    mean_pos = float(np.mean(pos_heights)) if pos_peaks.size else 0.0
    mean_neg = float(np.mean(neg_heights)) if neg_peaks.size else 0.0

    chosen_peaks = pos_peaks
    orientation: Orientation = "positive"
    if (neg_peaks.size > pos_peaks.size) or (
        neg_peaks.size == pos_peaks.size and mean_neg > mean_pos
    ):
        chosen_peaks = neg_peaks
        orientation = "negative"

    if force_orientation == "positive":
        chosen_peaks = pos_peaks
        orientation = "positive"
    elif force_orientation == "negative":
        chosen_peaks = neg_peaks
        orientation = "negative"
    elif force_orientation is not None:
        raise ECGRemovalError(f"Invalid force_orientation={force_orientation!r}")

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
        warnings.warn(
            "No valid epochs extracted; cannot generate template.", RuntimeWarning
        )
        return None, None, None, None

    template_raw = np.mean(np.stack(epochs, axis=0), axis=0)

    # Z-score template for orientation + peak finding
    t_mean = float(np.mean(template_raw))
    t_std = float(np.std(template_raw))
    if t_std < 1e-12:
        warnings.warn(
            "Template variance too small; cannot determine orientation.", RuntimeWarning
        )
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
        warnings.warn(
            "Cropped template is too short; using the full epoch template.",
            RuntimeWarning,
        )
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


def optimize_template(
    template: np.ndarray, lfp_epoch: np.ndarray
) -> tuple[NDArray[np.float64], float, float, float]:
    """
    Fit (scale, offset) so that scale * template + offset best matches lfp_epoch.

    Uses a closed-form least-squares solution.
    """
    t = _as_1d_float(template)
    y = _as_1d_float(lfp_epoch)
    if t.size != y.size:
        raise ECGRemovalError(
            f"template and lfp_epoch must have same length, got {t.size} vs {y.size}"
        )

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
def _template_ecg_remover_core(
    lfp_signal: np.ndarray,
    fs: float,
    *,
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
    bad_sample_mask: NDArray[np.bool_] | None = None,
) -> _ECGChannelRun:
    """Run template subtraction and collect internal channel diagnostics."""
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)

    # Baseline-stabilized signal for detection
    feat = lfp_feature(x, fs_f, window_ms)
    if bad_sample_mask is None:
        r_peaks, input_orientation = detect_qrs_peaks(
            feat,
            fs_f,
            force_orientation=force_orientation,
            peak_height_range=peak_height_range,
            min_interpeak_ms=min_interpeak_ms,
        )
    else:
        valid_feature_mask = _feature_support_mask(
            bad_sample_mask,
            window_samples=_ms_to_samples(window_ms, fs_f),
        )
        r_peaks, input_orientation = _detect_qrs_peaks_masked(
            feat,
            fs_f,
            valid_sample_mask=valid_feature_mask,
            force_orientation=force_orientation,
            peak_height_range=peak_height_range,
            min_interpeak_ms=min_interpeak_ms,
        )

    run = _ECGChannelRun(cleaned=x.copy(), candidate_beats=int(r_peaks.size))

    if r_peaks.size == 0:
        logger.warning("No peaks detected; returning original signal.")
        run.unchanged_reason = "No reliable ECG peaks were detected."
        return run

    template_peaks = r_peaks
    if bad_sample_mask is not None:
        bad_prefix = _bad_prefix_sum(bad_sample_mask)
        pre_samples = _ms_to_samples(pre_ms, fs_f)
        post_samples = _ms_to_samples(post_ms, fs_f)
        eligible_mask, overlaps_bad = _eligible_peak_mask(
            r_peaks,
            start_offset=-pre_samples,
            end_offset=post_samples,
            n_samples=x.size,
            bad_prefix=bad_prefix,
        )
        run.bad_overlap_skipped_beats = int(np.sum(overlaps_bad))
        template_peaks = r_peaks[eligible_mask]
        run.eligible_beats = int(template_peaks.size)
        if template_peaks.size == 0:
            run.unchanged_reason = "No full clean heartbeat epochs were available."
            return run

    template, orientation, epoch_start, epoch_end = generate_qrs_template(
        x,
        fs_f,
        template_peaks,
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
        run.unchanged_reason = "ECG template generation failed."
        return run

    cleaned = x.copy()

    # Fit + subtract per beat
    eligible_count = 0
    for peak in template_peaks:
        start = int(peak + epoch_start)
        end = int(peak + epoch_end)
        if start < 0 or end > x.size:
            continue
        eligible_count += 1
        lfp_epoch = x[start:end]
        fitted, _, _, _ = optimize_template(template, lfp_epoch)
        cleaned[start:end] = lfp_epoch - fitted
        run.corrected_beats += 1

    if bad_sample_mask is None:
        run.eligible_beats = int(eligible_count)
    else:
        cleaned[bad_sample_mask] = x[bad_sample_mask]

    if run.corrected_beats == 0:
        run.unchanged_reason = "No heartbeat epoch could be corrected."

    fig = None
    if return_figure:
        fig = plot_vt(template, fs_f)
    run.cleaned = cleaned
    run.figure = fig
    return run


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
    run = _template_ecg_remover_core(
        lfp_signal,
        fs,
        window_ms=window_ms,
        peak_height_range=peak_height_range,
        min_interpeak_ms=min_interpeak_ms,
        force_orientation=force_orientation,
        pre_ms=pre_ms,
        post_ms=post_ms,
        tail_ms=tail_ms,
        qrs_duration_ms=qrs_duration_ms,
        pqrst=pqrst,
        return_figure=return_figure,
        bad_sample_mask=None,
    )
    if return_figure:
        return run.cleaned, run.figure
    return run.cleaned


def segment_signal(
    signal: np.ndarray, fs: float, epoch_length_ms: float = 1000.0
) -> NDArray[np.float64]:
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


def cross_correlation_align(
    epochs: NDArray[np.float64], fs: float, window_ms: float = 200.0
) -> NDArray[np.float64]:
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


def find_ecg_template1(
    template: np.ndarray, fs: float, threshold_v: float = 200e-6, pad_ms: float = 15.0
) -> NDArray[np.float64]:
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
        flank_extrema = [
            int(i) for i in valid_minima if (max_value - float(t[i])) >= threshold_v
        ]
    else:
        maxima, _ = find_peaks(t)
        valid_maxima = maxima[np.abs(maxima - max_idx).argsort()]
        flank_extrema = [
            int(i) for i in valid_maxima if (float(t[i]) - max_value) >= threshold_v
        ]

    if len(flank_extrema) < 2:
        warnings.warn(
            "Unable to find flank extrema satisfying threshold; returning original template.",
            RuntimeWarning,
        )
        return t

    left_candidates = [i for i in flank_extrema if i < max_idx]
    right_candidates = [i for i in flank_extrema if i > max_idx]
    if not left_candidates or not right_candidates:
        warnings.warn(
            "Unable to bracket QRS region; returning original template.", RuntimeWarning
        )
        return t

    left_extreme = left_candidates[0]
    right_extreme = right_candidates[0]

    def find_turning_point(
        signal_1d: NDArray[np.float64], peak_idx: int, direction: int
    ) -> int:
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
        raise ECGRemovalError(
            "min_bpm and max_bpm must be positive with min_bpm < max_bpm."
        )
    if max_threshold_tries < 1:
        raise ECGRemovalError("max_threshold_tries must be >= 1.")
    if not (0.0 < pass_rate <= 1.0):
        raise ECGRemovalError("pass_rate must be in (0, 1].")

    correlation_values = correlate(x, t, mode="full").astype(float)

    # Heuristic robust scaling: use very low/high quantiles derived from expected beat rate.
    q_low = float(np.clip((min_bpm / 60.0) / fs_f, 1e-6, 0.1))
    corr_values_norm = _robust_scale_1d(
        correlation_values, q_low=q_low, q_high=1.0 - q_low
    )

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
                valid = (peak_distances >= min_peak_distance) & (
                    peak_distances <= max_peak_distance
                )
            else:
                valid = peak_distances >= min_peak_distance

            if float(np.sum(valid)) >= float(valid.size) * float(pass_rate):
                best_peaks = peaks.astype(int)
                best_threshold = float(threshold)
                break

        threshold += float(threshold_step)
        if (threshold_step >= 0 and threshold >= threshold_end) or (
            threshold_step < 0 and threshold <= threshold_end
        ):
            break

    if best_peaks is None or best_threshold is None:
        logger.warning("No suitable peaks found after adaptive thresholding.")
        return None, None, None

    logger.info(
        "Adaptive threshold success: threshold=%.4f, n_peaks=%d",
        best_threshold,
        best_peaks.size,
    )
    return best_peaks, best_threshold, corr_values_norm


def _adaptive_threshold_peak_detection_masked(
    lfp_signal: NDArray[np.float64],
    template1: NDArray[np.float64],
    fs: float,
    *,
    bad_sample_mask: NDArray[np.bool_],
    min_bpm: int,
    max_bpm: int,
    threshold_start: float | None,
    threshold_step: float | None,
    max_threshold_tries: int,
    pass_rate: float,
    enforce_max_interval: bool,
) -> tuple[NDArray[np.int_], float, NDArray[np.float64]] | tuple[None, None, None]:
    """Detect Perceive peaks globally from correlation supports outside BAD."""
    x = _as_1d_float(lfp_signal)
    template = _as_1d_float(template1)
    fs_f = _validate_fs(fs)

    if min_bpm <= 0 or max_bpm <= 0 or min_bpm >= max_bpm:
        raise ECGRemovalError(
            "min_bpm and max_bpm must be positive with min_bpm < max_bpm."
        )
    if max_threshold_tries < 1:
        raise ECGRemovalError("max_threshold_tries must be >= 1.")
    if not (0.0 < pass_rate <= 1.0):
        raise ECGRemovalError("pass_rate must be in (0, 1].")

    correlation_input = x.copy()
    correlation_input[bad_sample_mask] = 0.0
    correlation_values = correlate(correlation_input, template, mode="full").astype(
        float
    )
    template_size = int(template.size)
    starts = np.arange(correlation_values.size, dtype=int) - template_size + 1
    ends = starts + template_size
    valid_correlation = (starts >= 0) & (ends <= x.size)

    bad_prefix = _bad_prefix_sum(bad_sample_mask)
    valid_indices = np.flatnonzero(valid_correlation)
    valid_correlation[valid_indices] = (
        bad_prefix[ends[valid_indices]] - bad_prefix[starts[valid_indices]]
    ) == 0

    valid_values = correlation_values[valid_correlation]
    if valid_values.size == 0:
        logger.warning("No full clean correlation supports are available.")
        return None, None, None

    q_low = float(np.clip((min_bpm / 60.0) / fs_f, 1e-6, 0.1))
    center = float(np.median(valid_values))
    low = float(np.quantile(valid_values, q_low))
    high = float(np.quantile(valid_values, 1.0 - q_low))
    scale = max(high - low, 1e-120)
    corr_values_norm = np.full(correlation_values.size, -np.inf, dtype=float)
    corr_values_norm[valid_correlation] = (
        correlation_values[valid_correlation] - center
    ) / scale
    valid_normalized = corr_values_norm[valid_correlation]

    min_peak_distance = int(fs_f / (max_bpm / 60.0))
    max_peak_distance = int(fs_f / (min_bpm / 60.0))
    threshold_end = float(np.quantile(valid_normalized, 1.0 - (q_low * 0.5)))

    if threshold_start is None:
        threshold_start = float(np.quantile(valid_normalized, 0.6))
    if threshold_step is None:
        threshold_step = (threshold_end - threshold_start) / float(max_threshold_tries)

    template_peak_index = int(np.argmax(np.abs(template)))
    best_peaks: NDArray[np.int_] | None = None
    best_threshold: float | None = None
    threshold = float(threshold_start)
    for _ in range(max_threshold_tries):
        peaks, _ = find_peaks(corr_values_norm, height=threshold)

        if peaks.size > 1:
            peak_distances = np.diff(peaks)
            signal_peaks = peaks - template_size + 1 + template_peak_index
            interval_is_observed = np.ones(peak_distances.size, dtype=bool)
            for index, (left, right) in enumerate(
                zip(signal_peaks[:-1], signal_peaks[1:])
            ):
                start = int(min(left, right))
                end = int(max(left, right) + 1)
                interval_is_observed[index] = not _interval_has_bad(
                    bad_prefix,
                    start,
                    end,
                )

            if enforce_max_interval:
                plausible = (peak_distances >= min_peak_distance) & (
                    peak_distances <= max_peak_distance
                )
            else:
                plausible = peak_distances >= min_peak_distance

            n_observed = int(np.sum(interval_is_observed))
            n_plausible = int(np.sum(plausible & interval_is_observed))
            if n_observed > 0 and float(n_plausible) >= float(n_observed) * float(
                pass_rate
            ):
                best_peaks = peaks.astype(int)
                best_threshold = float(threshold)
                break

        threshold += float(threshold_step)
        if (threshold_step >= 0 and threshold >= threshold_end) or (
            threshold_step < 0 and threshold <= threshold_end
        ):
            break

    if best_peaks is None or best_threshold is None:
        logger.warning("No suitable clean peaks found after adaptive thresholding.")
        return None, None, None

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


def _perceive_ecg_remover_core(
    lfp_signal: np.ndarray,
    fs: float,
    *,
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
    bad_sample_mask: NDArray[np.bool_] | None = None,
) -> _ECGChannelRun:
    """Run Perceive removal and collect internal channel diagnostics."""
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)
    run = _ECGChannelRun(cleaned=x.copy())
    bad_skipped_peaks: set[int] = set()

    if bad_sample_mask is None:
        epochs = segment_signal(x, fs_f, epoch_length_ms)
    else:
        samples_per_epoch = _ms_to_samples(epoch_length_ms, fs_f)
        n_full_epochs = x.size // samples_per_epoch
        if n_full_epochs < 1:
            raise ECGRemovalError("Signal too short for the requested epoch_length_ms.")
        trimmed_samples = n_full_epochs * samples_per_epoch
        epoch_matrix = x[:trimmed_samples].reshape(n_full_epochs, samples_per_epoch)
        bad_epoch_matrix = bad_sample_mask[:trimmed_samples].reshape(
            n_full_epochs,
            samples_per_epoch,
        )
        epochs = epoch_matrix[~np.any(bad_epoch_matrix, axis=1)]
        if epochs.shape[0] == 0:
            run.unchanged_reason = (
                "No full clean initial Perceive epochs were available."
            )
            return run

    template0 = cross_correlation_align(epochs, fs_f, window_ms)
    template1 = find_ecg_template1(template0, fs_f, threshold_v, pad_ms)

    if bad_sample_mask is None:
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
    else:
        peaks_corr, _, _ = _adaptive_threshold_peak_detection_masked(
            x,
            template1,
            fs_f,
            bad_sample_mask=bad_sample_mask,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            threshold_start=threshold_start,
            threshold_step=threshold_step,
            max_threshold_tries=max_threshold_tries,
            pass_rate=pass_rate,
            enforce_max_interval=enforce_max_interval,
        )
    if peaks_corr is None:
        run.unchanged_reason = "Perceive initial peak detection failed."
        return run

    peak_template1_idx = int(np.argmax(np.abs(template1)))
    peaks = peaks_corr - int(template1.size) + 1 + peak_template1_idx
    peaks = peaks[(peaks >= 0) & (peaks < x.size)]

    template_peaks = peaks
    if bad_sample_mask is not None:
        bad_prefix = _bad_prefix_sum(bad_sample_mask)
        before_template = _ms_to_samples(before_ms, fs_f)
        after_template = _ms_to_samples(after_ms, fs_f)
        eligible_template, overlaps_bad = _eligible_peak_mask(
            peaks,
            start_offset=-before_template,
            end_offset=after_template + 1,
            n_samples=x.size,
            bad_prefix=bad_prefix,
        )
        bad_skipped_peaks.update(peaks[overlaps_bad].astype(int).tolist())
        template_peaks = peaks[eligible_template]
        if template_peaks.size == 0:
            run.candidate_beats = int(peaks.size)
            run.bad_overlap_skipped_beats = len(bad_skipped_peaks)
            run.unchanged_reason = (
                "No full clean Perceive template epochs were available."
            )
            return run

    template_mean = create_ecg_template(
        x,
        template_peaks,
        fs_f,
        before_ms,
        after_ms,
    )
    if template_mean is None:
        run.candidate_beats = int(peaks.size)
        run.bad_overlap_skipped_beats = len(bad_skipped_peaks)
        run.unchanged_reason = "Perceive template generation failed."
        return run

    template_mean1 = find_ecg_template1(template_mean, fs_f, threshold_v, pad_ms)

    if bad_sample_mask is None:
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
    else:
        peaks_mean_corr, _, _ = _adaptive_threshold_peak_detection_masked(
            x,
            template_mean1,
            fs_f,
            bad_sample_mask=bad_sample_mask,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            threshold_start=threshold_start,
            threshold_step=threshold_step,
            max_threshold_tries=max_threshold_tries,
            pass_rate=pass_rate,
            enforce_max_interval=enforce_max_interval,
        )
    if peaks_mean_corr is None:
        run.candidate_beats = int(peaks.size)
        run.bad_overlap_skipped_beats = len(bad_skipped_peaks)
        run.unchanged_reason = "Perceive refined peak detection failed."
        return run

    peak_template_mean1_idx = int(np.argmax(np.abs(template_mean1)))
    peaks_mean = (
        peaks_mean_corr - int(template_mean1.size) + 1 + peak_template_mean1_idx
    )
    peaks_mean = peaks_mean[
        (peaks_mean >= -int(template_mean1.size))
        & (peaks_mean < x.size + int(template_mean1.size))
    ]

    template_len = int(template_mean1.size)
    before_samples = peak_template_mean1_idx
    after_samples = template_len - peak_template_mean1_idx - 1

    run.candidate_beats = int(peaks_mean.size)
    replacement_peaks = peaks_mean
    if bad_sample_mask is not None:
        eligible_replacement, overlaps_bad = _eligible_peak_mask(
            peaks_mean,
            start_offset=-(2 * before_samples + 1),
            end_offset=(2 * after_samples + 1),
            n_samples=x.size,
            bad_prefix=bad_prefix,
        )
        bad_skipped_peaks.update(peaks_mean[overlaps_bad].astype(int).tolist())
        replacement_peaks = peaks_mean[eligible_replacement]
        run.eligible_beats = int(replacement_peaks.size)
        run.bad_overlap_skipped_beats = len(bad_skipped_peaks)
        if replacement_peaks.size == 0:
            run.unchanged_reason = (
                "No full clean Perceive replacement supports were available."
            )
            return run

    cleaned = x.copy()
    n = x.size

    # Mirror replacement uses the original signal x for sampling, writes into cleaned.
    eligible_count = 0
    for peak in replacement_peaks.astype(int):
        start = peak - before_samples
        end = peak + after_samples + 1  # exclusive

        start_in = max(0, start)
        end_in = min(n, end)
        if start_in >= end_in:
            continue
        eligible_count += 1

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
        if right_slice.size < after_samples:
            pad_len = after_samples - int(right_slice.size)
            right_slice = np.pad(
                right_slice,
                (0, pad_len),
                mode="constant",
                constant_values=float(x[-1]),
            )
        mirror_after = right_slice[::-1]

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

        run.corrected_beats += 1

    if bad_sample_mask is None:
        run.eligible_beats = int(eligible_count)
    else:
        cleaned[bad_sample_mask] = x[bad_sample_mask]

    if run.corrected_beats == 0:
        run.unchanged_reason = "No Perceive heartbeat replacement was completed."

    fig = None
    if return_figure:
        fig = plot_vt(template_mean1, fs_f)
    run.cleaned = cleaned
    run.figure = fig
    return run


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
    run = _perceive_ecg_remover_core(
        lfp_signal,
        fs,
        epoch_length_ms=epoch_length_ms,
        window_ms=window_ms,
        threshold_v=threshold_v,
        pad_ms=pad_ms,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        threshold_start=threshold_start,
        threshold_step=threshold_step,
        max_threshold_tries=max_threshold_tries,
        pass_rate=pass_rate,
        before_ms=before_ms,
        after_ms=after_ms,
        enforce_max_interval=enforce_max_interval,
        return_figure=return_figure,
        bad_sample_mask=None,
    )
    if return_figure:
        return run.cleaned, run.figure
    return run.cleaned


def _svd_ecg_remover_core(
    lfp_signal: np.ndarray,
    fs: float,
    *,
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
    bad_sample_mask: NDArray[np.bool_] | None = None,
) -> _ECGChannelRun:
    """Run SVD removal and collect internal channel diagnostics."""
    x = _as_1d_float(lfp_signal)
    fs_f = _validate_fs(fs)
    if components < 1:
        raise ECGRemovalError("components must be >= 1")

    # Detect peaks on baseline-stabilized signal
    feat = lfp_feature(x, fs_f, window_ms)
    if bad_sample_mask is None:
        r_peaks, input_orientation = detect_qrs_peaks(
            feat,
            fs_f,
            force_orientation=force_orientation,
            peak_height_range=peak_height_range,
            min_interpeak_ms=min_interpeak_ms,
        )
    else:
        valid_feature_mask = _feature_support_mask(
            bad_sample_mask,
            window_samples=_ms_to_samples(window_ms, fs_f),
        )
        r_peaks, input_orientation = _detect_qrs_peaks_masked(
            feat,
            fs_f,
            valid_sample_mask=valid_feature_mask,
            force_orientation=force_orientation,
            peak_height_range=peak_height_range,
            min_interpeak_ms=min_interpeak_ms,
        )

    run = _ECGChannelRun(cleaned=x.copy(), candidate_beats=int(r_peaks.size))
    if r_peaks.size == 0:
        run.unchanged_reason = "No reliable ECG peaks were detected."
        return run

    model_peaks = r_peaks
    if bad_sample_mask is not None:
        bad_prefix = _bad_prefix_sum(bad_sample_mask)
        pre_samples = _ms_to_samples(pre_ms, fs_f)
        post_samples = _ms_to_samples(post_ms, fs_f)
        eligible_mask, overlaps_bad = _eligible_peak_mask(
            r_peaks,
            start_offset=-pre_samples,
            end_offset=post_samples,
            n_samples=x.size,
            bad_prefix=bad_prefix,
        )
        run.bad_overlap_skipped_beats = int(np.sum(overlaps_bad))
        model_peaks = r_peaks[eligible_mask]
        run.eligible_beats = int(model_peaks.size)
        if model_peaks.size < 2:
            run.unchanged_reason = (
                "Fewer than two full clean heartbeat epochs were available."
            )
            return run

    template, _, epoch_start, epoch_end = generate_qrs_template(
        x,
        fs_f,
        model_peaks,
        input_orientation=input_orientation,
        force_orientation=force_orientation,
        pre_ms=pre_ms,
        post_ms=post_ms,
        tail_ms=tail_ms,
        qrs_duration_ms=qrs_duration_ms,
        pqrst=pqrst,
    )
    if template is None:
        run.unchanged_reason = "ECG template generation failed."
        return run

    # Remove peaks that would generate out-of-bounds epochs.
    valid_mask = (model_peaks + epoch_start >= 0) & (model_peaks + epoch_end <= x.size)
    correction_peaks = model_peaks[valid_mask]
    run.eligible_beats = int(correction_peaks.size)
    if correction_peaks.size < 2:
        run.unchanged_reason = "Fewer than two heartbeat epochs could enter SVD."
        if return_figure:
            run.figure = plot_vt(template, fs_f)
        return run

    epoch_len = int(template.size)
    epochs = np.zeros((epoch_len, int(correction_peaks.size)), dtype=float)
    for k, peak in enumerate(correction_peaks):
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
        end_idx_local = int(
            epoch_len - tail_samples + j_min
        )  # index of last sample in right block
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
        start_global = int(correction_peaks[k] + epoch_start)
        end_global = int(correction_peaks[k] + epoch_end)
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
        run.corrected_beats += 1

    if bad_sample_mask is not None:
        cleaned[bad_sample_mask] = x[bad_sample_mask]

    if run.corrected_beats == 0:
        run.unchanged_reason = "No SVD heartbeat correction was completed."

    fig = None
    if return_figure:
        fig = plot_vt(template, fs_f)
    run.cleaned = cleaned
    run.figure = fig
    return run


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
    run = _svd_ecg_remover_core(
        lfp_signal,
        fs,
        components=components,
        window_ms=window_ms,
        peak_height_range=peak_height_range,
        min_interpeak_ms=min_interpeak_ms,
        force_orientation=force_orientation,
        pre_ms=pre_ms,
        post_ms=post_ms,
        tail_ms=tail_ms,
        qrs_duration_ms=qrs_duration_ms,
        pqrst=pqrst,
        return_figure=return_figure,
        bad_sample_mask=None,
    )
    if return_figure:
        return run.cleaned, run.figure
    return run.cleaned


# -----------------------------------------------------------------------------
# High-level dispatch + wrappers
# -----------------------------------------------------------------------------


def _run_supported_ecg_method(
    method: str,
    lfp_signal: NDArray[np.float64],
    fs: float,
    *,
    bad_sample_mask: NDArray[np.bool_] | None,
    method_kwargs: dict[str, Any],
) -> _ECGChannelRun:
    """Run one built-in ECG method without changing its public return contract."""
    if method == "template":
        params: dict[str, Any] = asdict(TemplateFitConfig())
        params["return_figure"] = False
        params.update(method_kwargs)
        return _template_ecg_remover_core(
            lfp_signal,
            fs,
            bad_sample_mask=bad_sample_mask,
            **params,
        )

    if method == "perceive":
        params = asdict(PerceiveConfig())
        params["return_figure"] = False
        params.update(method_kwargs)
        if lfp_signal.size < _ms_to_samples(float(params["epoch_length_ms"]), fs):
            return _ECGChannelRun(
                cleaned=np.asarray(lfp_signal, dtype=float).copy(),
                unchanged_reason=(
                    "The signal is shorter than one full Perceive initial epoch."
                ),
            )
        return _perceive_ecg_remover_core(
            lfp_signal,
            fs,
            bad_sample_mask=bad_sample_mask,
            **params,
        )

    if method == "svd":
        params = asdict(SvdConfig())
        params["return_figure"] = False
        params.update(method_kwargs)
        return _svd_ecg_remover_core(
            lfp_signal,
            fs,
            bad_sample_mask=bad_sample_mask,
            **params,
        )

    raise ECGRemovalError(f"Unknown built-in ECG method: {method}")


def _ecg_bad_support_by_channel(
    raw: Any,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return ECG-compatible BAD support and zero-duration points by channel."""
    from lfptensorpipe.io.timeline import raw_relative_onsets

    channels = tuple(str(name) for name in raw.ch_names)
    channel_indices = {name: index for index, name in enumerate(channels)}
    support = np.zeros((len(channels), int(raw.n_times)), dtype=bool)
    points: list[set[int]] = [set() for _ in channels]
    relative_onsets = raw_relative_onsets(raw)

    for onset, duration, description, annotation_channels in zip(
        relative_onsets,
        raw.annotations.duration,
        raw.annotations.description,
        raw.annotations.ch_names,
    ):
        if not str(description).casefold().startswith("bad"):
            continue
        scope = tuple(str(name) for name in annotation_channels)
        target_indices = (
            tuple(range(len(channels)))
            if not scope
            else tuple(
                channel_indices[name] for name in scope if name in channel_indices
            )
        )
        duration_f = float(duration)
        start, stop = raw.time_as_index(
            [float(onset), float(onset) + duration_f],
            use_rounding=True,
        )
        start = int(np.clip(start, 0, int(raw.n_times)))
        stop = int(np.clip(stop, 0, int(raw.n_times)))
        for channel_index in target_indices:
            if duration_f == 0.0:
                points[channel_index].add(start)
            elif stop > start:
                support[channel_index, start:stop] = True

    return support, tuple(tuple(sorted(values)) for values in points)


def _ecg_mask_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return []
    changes = np.diff(values.astype(np.int8))
    starts = list(np.flatnonzero(changes == 1) + 1)
    stops = list(np.flatnonzero(changes == -1) + 1)
    if values[0]:
        starts.insert(0, 0)
    if values[-1]:
        stops.append(int(values.size))
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _ecg_edge_rows(
    edge_masks: NDArray[np.bool_],
    *,
    channels: Sequence[str],
    sfreq: float,
    first_samp: int,
) -> list[tuple[float, float, tuple[str, ...]]]:
    """Convert per-channel masks to compact global/channel annotation rows."""
    if edge_masks.size == 0:
        return []
    common = np.logical_and.reduce(edge_masks)
    scoped_masks = [((), common)]
    scoped_masks.extend(
        ((str(channel),), edge_masks[index] & ~common)
        for index, channel in enumerate(channels)
    )
    rows: list[tuple[float, float, tuple[str, ...]]] = []
    for scope, mask in scoped_masks:
        rows.extend(
            (
                float((start + first_samp) / sfreq),
                float((stop - start) / sfreq),
                scope,
            )
            for start, stop in _ecg_mask_runs(mask)
        )
    return rows


def finalize_reviewed_bad_annotations(
    source_raw: Any,
    reviewed_raw: Any,
    *,
    mark_filter_edges: bool,
    edge_description: str,
    review_label: str,
    filter_support_radius_samples: int | None = None,
) -> Any:
    """Validate reviewed BAD support and rebuild one owned edge label."""
    try:
        import mne  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "mne is required for finalize_reviewed_bad_annotations()."
        ) from exc
    from lfptensorpipe.preproc.filter import _set_annotations_from_attached_frame

    if not isinstance(mark_filter_edges, bool):
        raise ValueError("mark_filter_edges must be a boolean.")
    normalized_review_label = str(review_label).strip()
    normalized_edge_description = str(edge_description).strip()
    if not normalized_review_label or not normalized_edge_description:
        raise ValueError("review_label and edge_description must be non-empty.")
    if tuple(source_raw.ch_names) != tuple(reviewed_raw.ch_names):
        raise ValueError(
            f"{normalized_review_label} review channels no longer match the accepted input."
        )
    if int(source_raw.n_times) != int(reviewed_raw.n_times):
        raise ValueError(
            f"{normalized_review_label} review length no longer matches the accepted input."
        )
    if float(source_raw.info["sfreq"]) != float(reviewed_raw.info["sfreq"]):
        raise ValueError(
            f"{normalized_review_label} review sampling rate no longer matches the input."
        )
    if int(source_raw.first_samp) != int(reviewed_raw.first_samp):
        raise ValueError(
            f"{normalized_review_label} review sample origin no longer matches the input."
        )

    annotations = reviewed_raw.annotations
    keep = np.asarray(
        [
            str(description) != normalized_edge_description
            for description in annotations.description
        ],
        dtype=bool,
    )
    if np.any(keep):
        reviewed_annotations = annotations[keep]
    else:
        reviewed_annotations = mne.Annotations(
            [],
            [],
            [],
            orig_time=annotations.orig_time,
        )
    out = reviewed_raw.copy()
    _set_annotations_from_attached_frame(out, reviewed_annotations)

    source_support, source_points = _ecg_bad_support_by_channel(source_raw)
    reviewed_support, reviewed_points = _ecg_bad_support_by_channel(out)
    if np.any(source_support & ~reviewed_support):
        raise ValueError(
            f"{normalized_review_label} review cannot shorten or remove BAD support "
            "from its accepted input."
        )
    for channel_index, points in enumerate(source_points):
        missing_points = set(points).difference(reviewed_points[channel_index])
        for point in missing_points:
            left = max(0, min(int(point) - 1, int(out.n_times) - 1))
            right = max(0, min(int(point), int(out.n_times) - 1))
            if int(out.n_times) == 0 or not (
                reviewed_support[channel_index, left]
                or reviewed_support[channel_index, right]
            ):
                raise ValueError(
                    f"{normalized_review_label} review cannot remove a zero-duration "
                    "BAD boundary from "
                    "its accepted input."
                )

    if not mark_filter_edges:
        return out
    if (
        isinstance(filter_support_radius_samples, bool)
        or not isinstance(filter_support_radius_samples, int)
        or filter_support_radius_samples < 0
    ):
        raise ValueError(
            "A non-negative integer Filter support radius is required to mark "
            f"{normalized_review_label} review edges."
        )

    radius = int(filter_support_radius_samples)
    edge_masks = np.zeros_like(reviewed_support)
    if radius > 0:
        n_times = int(out.n_times)
        for channel_index in range(len(out.ch_names)):
            new_support = (
                reviewed_support[channel_index] & ~source_support[channel_index]
            )
            for start, stop in _ecg_mask_runs(new_support):
                edge_masks[
                    channel_index,
                    max(0, start - radius) : min(n_times, stop + radius),
                ] = True
            new_points = set(reviewed_points[channel_index]).difference(
                source_points[channel_index]
            )
            for point in new_points:
                edge_masks[
                    channel_index,
                    max(0, int(point) - radius) : min(
                        n_times,
                        int(point) + radius,
                    ),
                ] = True
            edge_masks[channel_index] &= ~reviewed_support[channel_index]

    rows = _ecg_edge_rows(
        edge_masks,
        channels=out.ch_names,
        sfreq=float(out.info["sfreq"]),
        first_samp=int(out.first_samp),
    )
    combined = out.annotations.copy()
    if rows:
        edge_annotations = mne.Annotations(
            onset=[row[0] for row in rows],
            duration=[row[1] for row in rows],
            description=[normalized_edge_description] * len(rows),
            orig_time=combined.orig_time,
            ch_names=[row[2] for row in rows],
        )
        combined = combined + edge_annotations
        order = np.argsort(np.asarray(combined.onset, dtype=float), kind="stable")
        combined = mne.Annotations(
            onset=np.asarray(combined.onset, dtype=float)[order].tolist(),
            duration=np.asarray(combined.duration, dtype=float)[order].tolist(),
            description=np.asarray(combined.description, dtype=object)[order].tolist(),
            orig_time=combined.orig_time,
            ch_names=[combined.ch_names[index] for index in order],
        )
    _set_annotations_from_attached_frame(out, combined)
    return out


def finalize_reviewed_ecg_annotations(
    source_raw: Any,
    reviewed_raw: Any,
    *,
    mark_filter_edges: bool,
    filter_support_radius_samples: int | None = None,
) -> Any:
    """Validate reviewed ECG BAD support and rebuild ECG-owned edge marks."""
    return finalize_reviewed_bad_annotations(
        source_raw,
        reviewed_raw,
        mark_filter_edges=mark_filter_edges,
        edge_description=ECG_FILTER_EDGE_DESCRIPTION,
        review_label="ECG",
        filter_support_radius_samples=filter_support_radius_samples,
    )


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

    The built-in string methods automatically derive a BAD sample mask from Raw
    annotations. Custom callables and caller-supplied method maps retain their
    legacy behavior. MNE is imported lazily to avoid a hard dependency.
    """
    try:
        import mne  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("mne is required for raw_call_ecgremover().") from e

    if not hasattr(raw, "info") or not hasattr(raw, "get_data"):
        raise ECGRemovalError("raw must be an MNE Raw-like object.")

    _diagnostics_out = kwargs.pop("_diagnostics_out", None)

    if isinstance(picks, str):
        picks = [picks]
    picks_list = list(picks)

    missing = sorted(set(picks_list) - set(raw.ch_names))
    if missing:
        raise ECGRemovalError(f"Channels not found in raw: {missing}")

    use_builtin_runner = method_map is None
    if method_map is None:
        method_map = {
            "template": template_ecg_remover,
            "perceive": perceive_ecg_remover,
            "svd": svd_ecg_remover,
        }

    if isinstance(method, str):
        if method not in method_map:
            raise ECGRemovalError(
                f"Unknown method '{method}'. Available: {list(method_map.keys())}"
            )
        method_func = method_map[method]
        method_name = method
    elif callable(method):
        method_func = method
        method_name = getattr(method, "__name__", "custom_method")
    else:
        raise ECGRemovalError("method must be a string or a callable.")

    fs_local = float(raw.info["sfreq"])
    data = raw.get_data(picks=picks_list)  # (n_sel, n_times)

    from lfptensorpipe.preproc.filter import (
        _build_bad_sample_mask,
        _startswith_any,
    )

    if verbose:
        logger.info("Running ECG remover '%s' on channels: %s", method_name, picks_list)

    cleaned_channels: list[NDArray[np.float64]] = []
    figs: dict[str, Any] = {}
    channel_diagnostics: list[dict[str, Any]] = []
    bad_masks_by_channel: list[np.ndarray] = []
    for channel, lfp in zip(picks_list, data):
        bad_sample_mask = np.asarray(
            _build_bad_sample_mask(
                raw,
                bad_prefixes=("BAD",),
                channel=channel,
            ),
            dtype=bool,
        )
        active_bad_mask = bad_sample_mask if bool(np.any(bad_sample_mask)) else None
        bad_masks_by_channel.append(bad_sample_mask)
        if use_builtin_runner and isinstance(method, str):
            run = _run_supported_ecg_method(
                method,
                np.asarray(lfp, dtype=float),
                fs_local,
                bad_sample_mask=active_bad_mask,
                method_kwargs=dict(kwargs),
            )
        else:
            out = method_func(lfp, fs_local, **kwargs)
            if isinstance(out, tuple) and len(out) >= 2:
                cleaned = np.asarray(out[0], dtype=float)
                figure = out[1]
            else:
                cleaned = np.asarray(out, dtype=float)
                figure = None
            changed = not np.array_equal(cleaned, np.asarray(lfp, dtype=float))
            run = _ECGChannelRun(
                cleaned=cleaned,
                figure=figure,
                eligible_beats=int(changed),
                corrected_beats=int(changed),
                unchanged_reason=(
                    "The custom ECG method produced no sample changes."
                    if not changed
                    else ""
                ),
            )

        cleaned_channels.append(run.cleaned)
        figs[channel] = run.figure
        channel_diagnostic = run.diagnostics(channel)
        channel_diagnostic["n_bad_samples"] = int(np.sum(bad_sample_mask))
        channel_diagnostics.append(channel_diagnostic)

    raw_out = raw if inplace else raw.copy()
    raw_out.load_data()
    cleaned_block = (
        np.stack(cleaned_channels, axis=0) if cleaned_channels else np.empty_like(data)
    )

    idxs = mne.pick_channels(raw_out.ch_names, include=picks_list)
    if (
        cleaned_block.shape != raw_out._data[idxs, :].shape
    ):  # noqa: SLF001 (MNE uses _data internally)
        raise RuntimeError("Shape mismatch when writing back cleaned data.")
    raw_out._data[idxs, :] = cleaned_block

    if _diagnostics_out is not None:
        unchanged_channels = [
            {
                "channel": item["channel"],
                "reason": item["unchanged_reason"]
                or "No ECG heartbeat correction was completed.",
            }
            for item in channel_diagnostics
            if int(item["corrected_beats"]) == 0
        ]
        n_bad_annotations = sum(
            _startswith_any(str(description), ("BAD",))
            for description in raw.annotations.description
        )
        union_bad_sample_mask = (
            np.logical_or.reduce(bad_masks_by_channel)
            if bad_masks_by_channel
            else np.zeros(int(raw.n_times), dtype=bool)
        )
        _diagnostics_out.clear()
        _diagnostics_out.update(
            {
                "n_bad_annotations": int(n_bad_annotations),
                "n_bad_samples": int(np.sum(union_bad_sample_mask)),
                "bad_duration_s": float(np.sum(union_bad_sample_mask) / fs_local),
                "n_channels_selected": int(len(picks_list)),
                "n_channels_processed": int(
                    sum(
                        int(item["corrected_beats"]) > 0 for item in channel_diagnostics
                    )
                ),
                "n_channels_unchanged": int(len(unchanged_channels)),
                "channel_diagnostics": channel_diagnostics,
                "unchanged_channels": unchanged_channels,
            }
        )

    if verbose:
        logger.info(
            "ECG removal done for %d channel(s) with '%s'. inplace=%s",
            len(picks_list),
            method_name,
            inplace,
        )

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
