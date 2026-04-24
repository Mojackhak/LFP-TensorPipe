"""Data models for import-time synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MarkerPoint:
    """One synchronization marker point in seconds."""

    marker_index: int
    time_s: float
    label: str
    source: str


@dataclass(frozen=True, slots=True)
class MarkerPair:
    """One explicit LFP/external marker pairing."""

    pair_id: int
    lfp_marker_index: int
    external_marker_index: int


@dataclass(frozen=True, slots=True)
class PeakDetectConfig:
    """Peak-detection parameters for waveform marker discovery."""

    search_range_s: tuple[float, float] | None = None
    min_distance_s: float = 1.0
    height: float | None = None
    prominence: float | None = None
    max_peaks: int | None = None
    use_abs: bool = True


@dataclass(frozen=True, slots=True)
class SyncEstimate:
    """Estimated synchronization parameters."""

    lag_s: float
    sfreq_before_hz: float
    sfreq_after_hz: float
    pair_count: int
    correct_sfreq: bool
    rmse_ms: float | None = None
    r2: float | None = None
    intercept_samples: float | None = None
    deltas_before_sync_s: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class SyncFigureData:
    """Data needed to build a sync preview or export figure."""

    kind: str
    source_label: str
    signal_times_s: tuple[float, ...] = ()
    signal_values: tuple[float, ...] = ()
    peak_times_s: tuple[float, ...] = ()
    marker_times_s: tuple[float, ...] = ()
    search_range_s: tuple[float, float] | None = None
    title: str = ""


@dataclass(frozen=True, slots=True)
class PersistedSyncArtifacts:
    """Paths written for one synced import."""

    summary_path: Path
    pairs_path: Path
    lfp_markers_path: Path
    external_markers_path: Path
    config_path: Path
    log_path: Path
    lfp_detection_path: Path | None = None
    external_detection_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ImportSyncState:
    """Saved import-dialog sync state."""

    lfp_markers: tuple[MarkerPoint, ...]
    external_markers: tuple[MarkerPoint, ...]
    pairs: tuple[MarkerPair, ...]
    estimate: SyncEstimate
    lfp_source_kind: str
    external_source_kind: str
    lfp_source_path: str = ""
    external_source_path: str = ""
    lfp_detect_config: PeakDetectConfig | None = None
    external_detect_config: PeakDetectConfig | None = None
    lfp_figure_data: SyncFigureData | None = None
    external_figure_data: SyncFigureData | None = None
    notes: dict[str, str] = field(default_factory=dict)
