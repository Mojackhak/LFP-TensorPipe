"""Core helpers for import-time synchronization."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .models import (
    MarkerPair,
    MarkerPoint,
    PeakDetectConfig,
    SyncEstimate,
    SyncFigureData,
)


def _normalize_markers(
    markers: list[MarkerPoint],
    *,
    default_label: str,
) -> list[MarkerPoint]:
    ordered = sorted(markers, key=lambda item: (float(item.time_s), str(item.label)))
    return [
        MarkerPoint(
            marker_index=index,
            time_s=float(item.time_s),
            label=str(item.label).strip() or f"{default_label}_{index}",
            source=str(item.source).strip() or default_label,
        )
        for index, item in enumerate(ordered)
    ]


def _float_tuple(values: np.ndarray | list[float]) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return tuple(float(value) for value in arr.tolist())


def _resolve_search_range(
    n_samples: int,
    sfreq_hz: float,
    search_range_s: tuple[float, float] | None,
) -> tuple[int, int]:
    if search_range_s is None:
        return 0, int(n_samples)
    start_s, stop_s = search_range_s
    if stop_s <= start_s:
        raise ValueError("search_range_s must satisfy stop > start.")
    start = max(0, int(np.floor(start_s * sfreq_hz)))
    stop = min(int(n_samples), int(np.ceil(stop_s * sfreq_hz)))
    if stop <= start:
        raise ValueError("search_range_s does not overlap the signal.")
    return start, stop


def _decimate_signal(
    signal_times_s: np.ndarray,
    signal_values: np.ndarray,
    *,
    max_points: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    if signal_times_s.size <= max_points:
        return signal_times_s, signal_values
    step = max(1, int(np.ceil(signal_times_s.size / max_points)))
    return signal_times_s[::step], signal_values[::step]


def _detect_peaks(
    signal: np.ndarray,
    sfreq_hz: float,
    config: PeakDetectConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, stop = _resolve_search_range(
        signal.shape[0], sfreq_hz, config.search_range_s
    )
    signal_slice = np.asarray(signal[start:stop], dtype=float)
    detect_signal = np.abs(signal_slice) if config.use_abs else signal_slice
    distance = max(1, int(round(config.min_distance_s * sfreq_hz)))
    peaks, _ = find_peaks(
        detect_signal,
        distance=distance,
        height=config.height,
        prominence=config.prominence,
    )
    if peaks.size == 0:
        raise ValueError("No peaks detected with the current parameters.")
    if config.max_peaks is not None and peaks.size > int(config.max_peaks):
        scores = detect_signal[peaks]
        keep = np.argsort(scores)[::-1][: int(config.max_peaks)]
        peaks = np.sort(peaks[keep])
    peaks_abs = peaks + start
    times_s = np.arange(start, stop, dtype=float) / float(sfreq_hz)
    return peaks_abs.astype(int), times_s, signal_slice


def _load_audio_mono(audio_path: str) -> tuple[np.ndarray, int]:
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), always_2d=False)
        data = np.asarray(audio, dtype=float)
    except Exception:
        from scipy.io import wavfile

        sr, audio = wavfile.read(str(path))
        data = np.asarray(audio)
        if np.issubdtype(data.dtype, np.integer):
            max_val = float(np.iinfo(data.dtype).max)
            data = data.astype(float) / max_val
        else:
            data = data.astype(float)

    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=float).reshape(-1), int(sr)


def seed_lfp_markers_from_raw(raw: Any) -> list[MarkerPoint]:
    """Seed LFP markers from non-BAD annotations."""
    annotations = getattr(raw, "annotations", None)
    if annotations is None or len(annotations) == 0:
        return []
    markers: list[MarkerPoint] = []
    for onset, description in zip(annotations.onset, annotations.description):
        label = str(description).strip()
        if not label or label.upper().startswith("BAD_"):
            continue
        markers.append(
            MarkerPoint(
                marker_index=len(markers),
                time_s=float(onset),
                label=label,
                source="parsed_annotations",
            )
        )
    return _normalize_markers(markers, default_label="lfp")


def detect_raw_channel_markers(
    raw: Any,
    channel_name: str,
    config: PeakDetectConfig,
) -> tuple[list[MarkerPoint], SyncFigureData | None]:
    """Detect waveform markers from one raw channel."""
    sfreq_hz = float(raw.info["sfreq"])
    signal = np.asarray(raw.get_data(picks=[channel_name])[0], dtype=float)
    peaks_abs, signal_times_s, signal_slice = _detect_peaks(signal, sfreq_hz, config)
    markers = _normalize_markers(
        [
            MarkerPoint(
                marker_index=index,
                time_s=float(sample) / sfreq_hz,
                label=f"lfp_peak_{index}",
                source="lfp_channel_peaks",
            )
            for index, sample in enumerate(peaks_abs.tolist())
        ],
        default_label="lfp_peak",
    )
    dec_times, dec_values = _decimate_signal(signal_times_s, signal_slice)
    figure_data = SyncFigureData(
        kind="waveform",
        source_label=channel_name,
        signal_times_s=_float_tuple(dec_times),
        signal_values=_float_tuple(dec_values),
        peak_times_s=tuple(marker.time_s for marker in markers),
        marker_times_s=tuple(marker.time_s for marker in markers),
        search_range_s=config.search_range_s,
        title=f"LFP Marker Source ({channel_name})",
    )
    return markers, figure_data


def load_external_markers_from_audio(
    audio_path: str,
    config: PeakDetectConfig,
) -> tuple[list[MarkerPoint], SyncFigureData]:
    """Load audio and detect peaks as external markers."""
    audio, sr = _load_audio_mono(audio_path)
    peaks_abs, signal_times_s, signal_slice = _detect_peaks(audio, float(sr), config)
    markers = _normalize_markers(
        [
            MarkerPoint(
                marker_index=index,
                time_s=float(sample) / float(sr),
                label=f"external_peak_{index}",
                source="external_audio_peaks",
            )
            for index, sample in enumerate(peaks_abs.tolist())
        ],
        default_label="external_peak",
    )
    dec_times, dec_values = _decimate_signal(signal_times_s, signal_slice)
    return markers, SyncFigureData(
        kind="waveform",
        source_label=Path(audio_path).name,
        signal_times_s=_float_tuple(dec_times),
        signal_values=_float_tuple(dec_values),
        peak_times_s=tuple(marker.time_s for marker in markers),
        marker_times_s=tuple(marker.time_s for marker in markers),
        search_range_s=config.search_range_s,
        title=f"External Marker Source ({Path(audio_path).name})",
    )


def load_external_markers_from_csv(csv_path: str) -> list[MarkerPoint]:
    """Load one-column marker times in seconds."""
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV marker file not found: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 1:
        raise ValueError(
            "CSV marker file must contain exactly one column of times in seconds."
        )
    times = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
    if times.size == 0:
        raise ValueError("CSV marker file does not contain any numeric marker times.")
    if np.any(times < 0):
        raise ValueError("CSV marker times must be non-negative.")
    unique = np.unique(times)
    return _normalize_markers(
        [
            MarkerPoint(
                marker_index=index,
                time_s=float(value),
                label=f"external_csv_{index}",
                source="external_csv_times",
            )
            for index, value in enumerate(unique.tolist())
        ],
        default_label="external_csv",
    )


def estimate_sync_from_pairs(
    *,
    lfp_markers: list[MarkerPoint] | tuple[MarkerPoint, ...],
    external_markers: list[MarkerPoint] | tuple[MarkerPoint, ...],
    pairs: list[MarkerPair] | tuple[MarkerPair, ...],
    sfreq_before_hz: float,
    correct_sfreq: bool,
) -> SyncEstimate:
    """Estimate lag and optional sampling-frequency correction from marker pairs."""
    if sfreq_before_hz <= 0:
        raise ValueError("sfreq_before_hz must be > 0.")
    if len(pairs) == 0:
        raise ValueError("At least one marker pair is required.")
    lfp_by_index = {item.marker_index: item for item in lfp_markers}
    ext_by_index = {item.marker_index: item for item in external_markers}
    lfp_times: list[float] = []
    ext_times: list[float] = []
    deltas: list[float] = []
    for pair in pairs:
        lfp_marker = lfp_by_index.get(pair.lfp_marker_index)
        ext_marker = ext_by_index.get(pair.external_marker_index)
        if lfp_marker is None or ext_marker is None:
            raise ValueError("Marker pair references a missing marker.")
        lfp_times.append(float(lfp_marker.time_s))
        ext_times.append(float(ext_marker.time_s))
        deltas.append(float(lfp_marker.time_s - ext_marker.time_s))

    lfp_arr = np.asarray(lfp_times, dtype=float)
    ext_arr = np.asarray(ext_times, dtype=float)
    delta_arr = np.asarray(deltas, dtype=float)
    if correct_sfreq:
        if len(pairs) < 2:
            raise ValueError("Correct sfreq requires at least two marker pairs.")
        y = lfp_arr * float(sfreq_before_hz)
        slope, intercept = np.polyfit(ext_arr, y, 1)
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("Estimated sync sampling frequency is invalid.")
        pred = slope * ext_arr + intercept
        residual_s = (pred - y) / float(slope)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = None if ss_tot <= 0 else max(0.0, 1.0 - (ss_res / ss_tot))
        return SyncEstimate(
            lag_s=float(intercept / slope),
            sfreq_before_hz=float(sfreq_before_hz),
            sfreq_after_hz=float(slope),
            pair_count=len(pairs),
            correct_sfreq=True,
            rmse_ms=float(np.sqrt(np.mean(residual_s**2)) * 1000.0),
            r2=r2,
            intercept_samples=float(intercept),
            deltas_before_sync_s=_float_tuple(delta_arr),
        )

    lag_s = float(np.median(delta_arr))
    residual_s = delta_arr - lag_s
    return SyncEstimate(
        lag_s=lag_s,
        sfreq_before_hz=float(sfreq_before_hz),
        sfreq_after_hz=float(sfreq_before_hz),
        pair_count=len(pairs),
        correct_sfreq=False,
        rmse_ms=float(np.sqrt(np.mean(residual_s**2)) * 1000.0),
        r2=None,
        intercept_samples=None,
        deltas_before_sync_s=_float_tuple(delta_arr),
    )


def build_synced_raw(raw: Any, estimate: SyncEstimate) -> Any:
    """Create a new synced raw object without mutating the parse result."""
    import mne

    sfreq_before_hz = float(raw.info["sfreq"])
    sfreq_after_hz = float(estimate.sfreq_after_hz)
    if sfreq_after_hz <= 0:
        raise ValueError("sfreq_after_hz must be > 0.")

    data = np.asarray(raw.get_data(), dtype=float)
    shift_samples = int(round(float(estimate.lag_s) * sfreq_after_hz))
    if shift_samples >= 0:
        if shift_samples >= data.shape[1]:
            raise ValueError("Positive lag trims all samples from the recording.")
        synced_data = data[:, shift_samples:].copy()
    else:
        pad = np.zeros((data.shape[0], abs(shift_samples)), dtype=float)
        synced_data = np.concatenate([pad, data], axis=1)

    info = mne.create_info(
        ch_names=list(raw.ch_names),
        sfreq=sfreq_after_hz,
        ch_types=raw.get_channel_types(),
    )
    info["bads"] = list(raw.info.get("bads", []))
    synced_raw = mne.io.RawArray(synced_data, info, verbose="ERROR")

    meas_date = raw.info.get("meas_date")
    if meas_date is not None:
        synced_raw.set_meas_date(meas_date + timedelta(seconds=float(estimate.lag_s)))

    annotations = getattr(raw, "annotations", None)
    if annotations is not None and len(annotations) > 0:
        onsets = np.asarray(annotations.onset, dtype=float)
        durations = np.asarray(annotations.duration, dtype=float)
        desc = list(annotations.description)
        if sfreq_after_hz != sfreq_before_hz:
            scale = sfreq_before_hz / sfreq_after_hz
            onsets = onsets * scale
            durations = durations * scale
        onsets = onsets - float(estimate.lag_s)
        max_time_s = (
            float(synced_data.shape[1] - 1) / sfreq_after_hz
            if synced_data.shape[1]
            else 0.0
        )
        keep = (onsets >= 0.0) & (onsets <= max_time_s)
        if np.any(keep):
            synced_raw.set_annotations(
                mne.Annotations(
                    onset=onsets[keep],
                    duration=durations[keep],
                    description=[
                        desc[index] for index, flag in enumerate(keep.tolist()) if flag
                    ],
                )
            )
    return synced_raw
