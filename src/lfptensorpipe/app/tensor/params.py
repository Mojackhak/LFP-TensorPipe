"""Tensor metric catalog and static parameter metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorMetricSpec:
    """Display/runtime definition for one tensor metric row."""

    key: str
    display_name: str
    group_name: str
    supported: bool = False


TENSOR_METRICS: tuple[TensorMetricSpec, ...] = (
    TensorMetricSpec("raw_power", "Raw power", "Power", supported=True),
    TensorMetricSpec(
        "periodic_aperiodic",
        "Periodic/APeriodic (SpecParam)",
        "Power",
        supported=True,
    ),
    TensorMetricSpec(
        "coherence", "Coherence", "Undirected Connectivity", supported=True
    ),
    TensorMetricSpec("plv", "PLV", "Undirected Connectivity", supported=True),
    TensorMetricSpec("ciplv", "ciPLV", "Undirected Connectivity", supported=True),
    TensorMetricSpec("pli", "PLI", "Undirected Connectivity", supported=True),
    TensorMetricSpec("wpli", "wPLI", "Undirected Connectivity", supported=True),
    TensorMetricSpec("trgc", "TRGC", "Directed Connectivity", supported=True),
    TensorMetricSpec("psi", "PSI", "Directed Connectivity", supported=True),
    TensorMetricSpec("burst", "Burst", "Temporal Events", supported=True),
)
TENSOR_METRICS_BY_KEY = {metric.key: metric for metric in TENSOR_METRICS}

TENSOR_CHANNEL_SELECTOR_KEYS = {"raw_power", "periodic_aperiodic", "burst"}
TENSOR_UNDIRECTED_SELECTOR_KEYS = {"coherence", "plv", "ciplv", "pli", "wpli"}
TENSOR_DIRECTED_SELECTOR_KEYS = {"trgc", "psi"}
TENSOR_COMMON_BASIC_KEYS = {
    "raw_power",
    "periodic_aperiodic",
    "coherence",
    "plv",
    "ciplv",
    "pli",
    "wpli",
    "trgc",
}
TENSOR_BAND_REQUIRED_KEYS = {"psi", "burst"}

DEFAULT_TENSOR_BANDS: tuple[dict[str, float | str], ...] = (
    {"name": "delta", "start": 1.0, "end": 4.0},
    {"name": "theta", "start": 4.0, "end": 8.0},
    {"name": "alpha", "start": 8.0, "end": 13.0},
    {"name": "beta_low", "start": 13.0, "end": 20.0},
    {"name": "beta_high", "start": 20.0, "end": 35.0},
    {"name": "gamma", "start": 35.0, "end": 100.0},
)

__all__ = [
    "DEFAULT_TENSOR_BANDS",
    "TENSOR_BAND_REQUIRED_KEYS",
    "TENSOR_CHANNEL_SELECTOR_KEYS",
    "TENSOR_COMMON_BASIC_KEYS",
    "TENSOR_DIRECTED_SELECTOR_KEYS",
    "TENSOR_METRICS",
    "TENSOR_METRICS_BY_KEY",
    "TENSOR_UNDIRECTED_SELECTOR_KEYS",
    "TensorMetricSpec",
]
