"""Legacy argument merge helpers for Build Tensor orchestration."""

from __future__ import annotations

from typing import Any

from lfptensorpipe.app.path_resolver import RecordContext


def merge_metric_params_map(
    svc: Any,
    context: RecordContext,
    *,
    metrics: list[str],
    metric_params_map: dict[str, dict[str, Any]] | None,
    low_freq: float | None,
    high_freq: float | None,
    step_hz: float | None,
    bands: list[dict[str, Any]] | None,
    selected_channels: list[str] | None,
    selected_pairs: dict[str, list[tuple[str, str]]] | None,
) -> dict[str, dict[str, Any]]:
    legacy_low, legacy_high, legacy_step = svc.load_tensor_frequency_defaults(context)
    legacy_low = float(low_freq) if low_freq is not None else float(legacy_low)
    legacy_high = float(high_freq) if high_freq is not None else float(legacy_high)
    legacy_step = float(step_hz) if step_hz is not None else float(legacy_step)
    legacy_bands = (
        [dict(item) for item in bands]
        if isinstance(bands, list)
        else [dict(item) for item in svc.DEFAULT_TENSOR_BANDS]
    )
    legacy_channels = svc._normalize_metric_channels(selected_channels)
    legacy_pairs_raw = selected_pairs if isinstance(selected_pairs, dict) else {}

    merged_metric_params_map: dict[str, dict[str, Any]] = {}
    provided_map = metric_params_map if isinstance(metric_params_map, dict) else {}
    directed_or_undirected = (
        svc.TENSOR_UNDIRECTED_SELECTOR_KEYS | svc.TENSOR_DIRECTED_SELECTOR_KEYS
    )

    for metric_key in metrics:
        merged = dict(provided_map.get(metric_key, {}))
        if metric_key in svc.TENSOR_COMMON_BASIC_KEYS:
            merged.setdefault("low_freq_hz", float(legacy_low))
            merged.setdefault("high_freq_hz", float(legacy_high))
            merged.setdefault("freq_step_hz", float(legacy_step))
        if metric_key == "periodic_aperiodic":
            merged.setdefault(
                "freq_range_hz",
                [
                    float(merged.get("low_freq_hz", legacy_low)),
                    float(merged.get("high_freq_hz", legacy_high)),
                ],
            )
        if metric_key in svc.TENSOR_BAND_REQUIRED_KEYS:
            merged.setdefault("bands", [dict(item) for item in legacy_bands])
        if (
            metric_key in svc.TENSOR_CHANNEL_SELECTOR_KEYS
            and "selected_channels" not in merged
            and legacy_channels is not None
        ):
            merged["selected_channels"] = list(legacy_channels)
        if metric_key in directed_or_undirected and "selected_pairs" not in merged:
            pairs_source = legacy_pairs_raw.get(metric_key)
            if pairs_source is None and metric_key in svc.TENSOR_UNDIRECTED_SELECTOR_KEYS:
                pairs_source = legacy_pairs_raw.get("undirected")
            if pairs_source is None and metric_key in svc.TENSOR_DIRECTED_SELECTOR_KEYS:
                pairs_source = legacy_pairs_raw.get("directed")
            if pairs_source is not None:
                merged["selected_pairs"] = list(pairs_source)
            elif selected_pairs is not None:
                merged["selected_pairs"] = []
        merged_metric_params_map[metric_key] = merged

    return merged_metric_params_map


__all__ = ["merge_metric_params_map"]
