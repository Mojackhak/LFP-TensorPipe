"""Data models for tensor runtime-plan construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .orchestration_execution import RuntimePlan


@dataclass
class TensorPlanBuildResult:
    overall_ok: bool
    messages: list[str]
    metric_statuses: dict[str, str]
    effective_n_jobs_map: dict[str, dict[str, int]]
    runtime_plans: dict[str, RuntimePlan]


@dataclass(frozen=True)
class MetricPlanInputs:
    metric_key: str
    metric_params: dict[str, Any]
    metric_channels: list[str] | None
    metric_pairs: list[tuple[str, str]] | None
    metric_low: float
    metric_high: float
    metric_step: float
    metric_bands: list[dict[str, Any]]
    parsed_freq_range: tuple[float, float] | None = None
    parsed_peak_width_limits: tuple[float, float] = (2.0, 12.0)
    max_n_peaks: float = float("inf")
