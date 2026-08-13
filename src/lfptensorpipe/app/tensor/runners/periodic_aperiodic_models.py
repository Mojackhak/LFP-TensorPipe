"""Shared state objects for the periodic/aperiodic tensor runner."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.utils.transforms import TransformMode

from ..frequency import TensorFilterInheritance

NOTCH_INTERPOLATION_METHOD = "empirical_residual.local_stratified"
NOTCH_INTERPOLATION_SEED = 42
MASK_SUPPORT_SEMANTICS = "joint_fit_consumed_tfr_max_plus_time_smoothing"


def derive_notch_interpolation_seed(context: RecordContext) -> int:
    """Derive a stable seed from the configured base seed and record identity."""
    payload = (
        f"{NOTCH_INTERPOLATION_SEED}\0{context.subject}\0{context.record}"
    ).encode("utf-8")
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")


@dataclass(frozen=True)
class PeriodicAperiodicOptions:
    context: RecordContext
    low_freq: float
    high_freq: float
    step_hz: float
    mask_edge_effects: bool
    bands: list[dict[str, Any]]
    selected_channels: list[str] | None
    method: str
    time_resolution_s: float
    hop_s: float
    min_cycles: float | None
    max_cycles: float | None
    mt_time_bandwidth_product: float
    mt_min_cycles: float
    freq_range_hz: tuple[float, float] | None
    freq_smooth_enabled: bool
    freq_smooth_sigma: float | None
    time_smooth_enabled: bool
    time_smooth_kernel_size: int | None
    aperiodic_mode: str
    peak_width_limits_hz: tuple[float, float]
    max_n_peaks: float
    min_peak_height: float
    peak_threshold: float
    fit_qc_threshold: float
    notches: Any
    notch_radii: Any
    n_jobs: int
    outer_n_jobs: int
    value_transform_mode: TransformMode


@dataclass(frozen=True)
class PeriodicAperiodicPaths:
    resolver: Any
    input_path: Path
    output_path: Path
    aperiodic_output_path: Path
    report_dir: Path
    config_path: Path
    log_path: Path


@dataclass
class PeriodicAperiodicPreparedInput:
    raw: Any
    picks: list[str]
    inheritance: TensorFilterInheritance
    runtime_notches: tuple[float, ...]
    runtime_notch_radii: tuple[float, ...]
    spec_low: float
    spec_high: float
    freqs_model: np.ndarray
    freqs_final: np.ndarray
    freqs_compute: np.ndarray
    notch_intervals: list[tuple[float, float]]
    interpolation_applied: bool
    method_norm: str


@dataclass
class PeriodicAperiodicOutputs:
    tensor: np.ndarray
    metadata: dict[str, Any]
    params_tensor: np.ndarray
    params_meta: dict[str, Any]
