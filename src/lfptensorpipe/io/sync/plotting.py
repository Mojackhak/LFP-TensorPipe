"""Visualization helpers for import sync."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import MarkerPair, MarkerPoint, SyncEstimate, SyncFigureData

LFP_COLOR = "tab:blue"
EXTERNAL_COLOR = "tab:orange"


def _normalize_signal(values: np.ndarray) -> np.ndarray:
    arr = np.abs(np.asarray(values, dtype=float).reshape(-1))
    if arr.size == 0:
        return arr
    max_val = float(np.nanmax(arr))
    if not np.isfinite(max_val) or max_val <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / max_val


def _overlay_events(
    ax,
    *,
    markers: tuple[MarkerPoint, ...],
    color: str,
    stem_top: float,
    label: str,
) -> None:
    if not markers:
        return
    times = np.asarray([marker.time_s for marker in markers], dtype=float)
    ax.vlines(times, 0.0, stem_top, color=color, linewidth=1.0, alpha=0.7)
    ax.scatter(
        times,
        np.full(times.shape, stem_top, dtype=float),
        color=color,
        s=18,
        label=label,
        zorder=3,
    )


def _overlay_waveform(
    ax,
    *,
    figure_data: SyncFigureData,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    times = np.asarray(figure_data.signal_times_s, dtype=float)
    values = _normalize_signal(np.asarray(figure_data.signal_values, dtype=float))
    if times.size == 0 or values.size == 0:
        return
    ax.plot(times, values, color=color, linewidth=1.0, linestyle=linestyle, label=label)
    if figure_data.search_range_s is not None:
        ax.axvspan(
            figure_data.search_range_s[0],
            figure_data.search_range_s[1],
            color=color,
            alpha=0.06,
        )


def _marker_overlay_axis(
    ax,
    *,
    lfp_markers: tuple[MarkerPoint, ...],
    external_markers: tuple[MarkerPoint, ...],
    lfp_figure_data: SyncFigureData | None,
    external_figure_data: SyncFigureData | None,
) -> None:
    ax.set_title("Marker Overlay")
    if lfp_figure_data is not None and lfp_figure_data.kind == "waveform":
        _overlay_waveform(
            ax,
            figure_data=lfp_figure_data,
            color=LFP_COLOR,
            label="LFP waveform",
        )
    if external_figure_data is not None and external_figure_data.kind == "waveform":
        _overlay_waveform(
            ax,
            figure_data=external_figure_data,
            color=EXTERNAL_COLOR,
            label="External waveform",
            linestyle="--",
        )
    _overlay_events(
        ax,
        markers=lfp_markers,
        color=LFP_COLOR,
        stem_top=1.0,
        label="LFP markers",
    )
    _overlay_events(
        ax,
        markers=external_markers,
        color=EXTERNAL_COLOR,
        stem_top=0.9,
        label="External markers",
    )
    if not lfp_markers and not external_markers:
        ax.text(
            0.5,
            0.5,
            "No markers",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized peak/marker")
    ax.set_ylim(0.0, 1.05)
    if ax.has_data():
        ax.legend(loc="upper right", fontsize=8)


def _fit_line_external_from_lfp(
    x_values: np.ndarray,
    estimate: SyncEstimate,
) -> np.ndarray:
    if estimate.correct_sfreq and estimate.sfreq_after_hz > 0:
        slope = float(estimate.sfreq_before_hz) / float(estimate.sfreq_after_hz)
        intercept = -float(estimate.intercept_samples or 0.0) / float(
            estimate.sfreq_after_hz
        )
        return slope * x_values + intercept
    return x_values - float(estimate.lag_s)


def _fit_scatter_axis(
    ax,
    *,
    lfp_markers: tuple[MarkerPoint, ...],
    external_markers: tuple[MarkerPoint, ...],
    pairs: tuple[MarkerPair, ...],
    estimate: SyncEstimate,
) -> None:
    ax.set_title("Fit Scatter")
    ax.set_xlabel("LFP time (s)")
    ax.set_ylabel("External time (s)")
    lfp_by_index = {marker.marker_index: marker for marker in lfp_markers}
    external_by_index = {marker.marker_index: marker for marker in external_markers}
    x_points: list[float] = []
    y_points: list[float] = []
    for pair in pairs:
        lfp_marker = lfp_by_index.get(pair.lfp_marker_index)
        external_marker = external_by_index.get(pair.external_marker_index)
        if lfp_marker is None or external_marker is None:
            continue
        x_points.append(float(lfp_marker.time_s))
        y_points.append(float(external_marker.time_s))
    x_arr = np.asarray(x_points, dtype=float)
    y_arr = np.asarray(y_points, dtype=float)
    if x_arr.size == 0:
        ax.text(
            0.5,
            0.5,
            "No paired markers",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.scatter(x_arr, y_arr, color=LFP_COLOR, s=22, zorder=3)
        if x_arr.size >= 2:
            x_line = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), 100)
            y_line = _fit_line_external_from_lfp(x_line, estimate)
            ax.plot(x_line, y_line, color="tab:red", linewidth=1.0)
    metrics = [
        f"pairs={estimate.pair_count}",
        f"lag={estimate.lag_s:.6f} s",
        f"sfreq={estimate.sfreq_after_hz:.6f} Hz",
    ]
    if estimate.rmse_ms is not None:
        metrics.append(f"rmse={estimate.rmse_ms:.3f} ms")
    if estimate.r2 is not None:
        metrics.append(f"r2={estimate.r2:.4f}")
    ax.text(
        0.02,
        0.98,
        "\n".join(metrics),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def build_sync_summary_figure(
    *,
    lfp_markers: tuple[MarkerPoint, ...],
    external_markers: tuple[MarkerPoint, ...],
    pairs: tuple[MarkerPair, ...],
    estimate: SyncEstimate,
    lfp_figure_data: SyncFigureData | None = None,
    external_figure_data: SyncFigureData | None = None,
) -> plt.Figure:
    """Build the shared two-panel sync preview/export figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _marker_overlay_axis(
        axes[0],
        lfp_markers=lfp_markers,
        external_markers=external_markers,
        lfp_figure_data=lfp_figure_data,
        external_figure_data=external_figure_data,
    )
    _fit_scatter_axis(
        axes[1],
        lfp_markers=lfp_markers,
        external_markers=external_markers,
        pairs=pairs,
        estimate=estimate,
    )
    fig.tight_layout()
    return fig


def save_sync_summary_figure(path: str | Path, **kwargs) -> Path:
    """Build and save one sync summary figure."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_sync_summary_figure(**kwargs)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_sync_detection_figure(
    path: str | Path,
    figure_data: SyncFigureData,
    title: str,
) -> Path:
    """Save one detection detail figure from waveform figure data."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = np.asarray(figure_data.signal_times_s, dtype=float)
    y = np.asarray(figure_data.signal_values, dtype=float)
    if x.size > 0:
        ax.plot(x, y, color="0.25", linewidth=0.8)
    peak_times = np.asarray(figure_data.peak_times_s, dtype=float)
    if peak_times.size > 0:
        ax.vlines(
            peak_times,
            np.nanmin(y) if y.size else 0.0,
            np.nanmax(y) if y.size else 1.0,
            color="tab:red",
            alpha=0.6,
            linewidth=1.0,
        )
    if figure_data.search_range_s is not None:
        ax.axvspan(
            figure_data.search_range_s[0],
            figure_data.search_range_s[1],
            color="tab:blue",
            alpha=0.08,
        )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
