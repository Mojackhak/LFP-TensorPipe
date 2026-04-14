"""Dispatch helpers for tensor runtime-plan builders."""

from __future__ import annotations

from typing import Any

from .orchestration_execution import RuntimePlan
from .orchestration_plan_burst import plan_burst
from .orchestration_plan_connectivity import plan_psi, plan_trgc, plan_undirected
from .orchestration_plan_models import MetricPlanInputs
from .orchestration_plan_power import plan_periodic_aperiodic, plan_raw_power


def build_runtime_plan(
    svc: Any,
    context: Any,
    *,
    prepared: MetricPlanInputs,
    mask_edge_effects: bool,
) -> dict[str, RuntimePlan]:
    metric_key = prepared.metric_key
    if metric_key == "raw_power":
        return {
            metric_key: plan_raw_power(
                svc,
                context,
                metric_low=prepared.metric_low,
                metric_high=prepared.metric_high,
                metric_step=prepared.metric_step,
                metric_bands=prepared.metric_bands,
                metric_channels=prepared.metric_channels,
                metric_params=prepared.metric_params,
                mask_edge_effects=mask_edge_effects,
            )
        }

    if metric_key == "periodic_aperiodic":
        return {
            metric_key: plan_periodic_aperiodic(
                svc,
                context,
                metric_low=prepared.metric_low,
                metric_high=prepared.metric_high,
                metric_step=prepared.metric_step,
                metric_bands=prepared.metric_bands,
                metric_channels=prepared.metric_channels,
                metric_params=prepared.metric_params,
                mask_edge_effects=mask_edge_effects,
                parsed_freq_range=prepared.parsed_freq_range,
                parsed_peak_width_limits=prepared.parsed_peak_width_limits,
                max_n_peaks=prepared.max_n_peaks,
            )
        }

    if metric_key in svc.TENSOR_UNDIRECTED_SELECTOR_KEYS:
        return {
            metric_key: plan_undirected(
                svc,
                context,
                metric_key=metric_key,
                metric_low=prepared.metric_low,
                metric_high=prepared.metric_high,
                metric_step=prepared.metric_step,
                metric_bands=prepared.metric_bands,
                metric_channels=prepared.metric_channels,
                metric_pairs=prepared.metric_pairs,
                metric_params=prepared.metric_params,
                mask_edge_effects=mask_edge_effects,
            )
        }

    if metric_key == "trgc":
        return plan_trgc(
            svc,
            context,
            metric_low=prepared.metric_low,
            metric_high=prepared.metric_high,
            metric_step=prepared.metric_step,
            metric_bands=prepared.metric_bands,
            metric_channels=prepared.metric_channels,
            metric_pairs=prepared.metric_pairs,
            metric_params=prepared.metric_params,
            mask_edge_effects=mask_edge_effects,
        )

    if metric_key == "psi":
        return {
            metric_key: plan_psi(
                svc,
                context,
                metric_low=prepared.metric_low,
                metric_high=prepared.metric_high,
                metric_step=prepared.metric_step,
                metric_bands=prepared.metric_bands,
                metric_channels=prepared.metric_channels,
                metric_pairs=prepared.metric_pairs,
                metric_params=prepared.metric_params,
                mask_edge_effects=mask_edge_effects,
            )
        }

    if metric_key == "burst":
        return {
            metric_key: plan_burst(
                svc,
                context,
                metric_low=prepared.metric_low,
                metric_high=prepared.metric_high,
                metric_step=prepared.metric_step,
                metric_bands=prepared.metric_bands,
                metric_channels=prepared.metric_channels,
                metric_params=prepared.metric_params,
                mask_edge_effects=mask_edge_effects,
            )
        }

    raise KeyError(metric_key)
