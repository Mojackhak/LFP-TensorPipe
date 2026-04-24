"""Draft-aware indicator helpers for Build Tensor metric rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import read_run_log

from .coercion import _as_float, _as_int, _as_optional_float, _as_optional_int
from .frequency import (
    DEFAULT_TENSOR_NOTCH_WIDTH,
    _compute_notch_intervals,
    load_tensor_frequency_defaults,
    normalize_tensor_metric_notch_params,
    validate_tensor_frequency_params,
)
from .orchestration_plan_validation import prepare_metric_plan_inputs
from .params import (
    TENSOR_BAND_REQUIRED_KEYS,
    TENSOR_CHANNEL_SELECTOR_KEYS,
    TENSOR_COMMON_BASIC_KEYS,
    TENSOR_DIRECTED_SELECTOR_KEYS,
    TENSOR_METRICS_BY_KEY,
    TENSOR_UNDIRECTED_SELECTOR_KEYS,
)
from .paths import tensor_metric_log_path
from .runners.burst import (
    _build_runtime_bands as _build_burst_runtime_bands,
    _resolve_burst_time_grid,
    _serialize_runtime_bands as _serialize_burst_runtime_bands,
)
from .runners.connectivity_psi import (
    _build_runtime_bands as _build_psi_runtime_bands,
    _serialize_runtime_bands as _serialize_psi_runtime_bands,
)
from .selectors import (
    normalize_metric_bands,
    normalize_metric_channels,
    normalize_metric_pairs,
)
from .validators import validate_bands as _validate_bands


class _IndicatorValidationSvc:
    TENSOR_BAND_REQUIRED_KEYS = TENSOR_BAND_REQUIRED_KEYS
    TENSOR_CHANNEL_SELECTOR_KEYS = TENSOR_CHANNEL_SELECTOR_KEYS
    TENSOR_COMMON_BASIC_KEYS = TENSOR_COMMON_BASIC_KEYS
    TENSOR_DIRECTED_SELECTOR_KEYS = TENSOR_DIRECTED_SELECTOR_KEYS
    TENSOR_UNDIRECTED_SELECTOR_KEYS = TENSOR_UNDIRECTED_SELECTOR_KEYS

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        return _as_float(value, default)

    @staticmethod
    def _as_optional_float(value: Any, default: float | None = None) -> float | None:
        return _as_optional_float(value, default)

    @staticmethod
    def _normalize_metric_channels(value: Any) -> list[str] | None:
        return normalize_metric_channels(value)

    @staticmethod
    def _normalize_metric_pairs(value: Any) -> list[tuple[str, str]] | None:
        return normalize_metric_pairs(value)

    @staticmethod
    def _normalize_metric_bands(value: Any) -> list[dict[str, Any]]:
        return normalize_metric_bands(value)

    @staticmethod
    def _validate_bands(value: list[dict[str, Any]]) -> tuple[bool, str]:
        return _validate_bands(value)

    @staticmethod
    def load_tensor_frequency_defaults(
        context: RecordContext,
    ) -> tuple[float, float, float]:
        return load_tensor_frequency_defaults(context)

    @staticmethod
    def validate_tensor_frequency_params(
        context: RecordContext,
        *,
        low_freq: float,
        high_freq: float,
        step_hz: float,
    ) -> tuple[bool, str, Any]:
        return validate_tensor_frequency_params(
            context,
            low_freq=low_freq,
            high_freq=high_freq,
            step_hz=step_hz,
        )


_VALIDATION_SVC = _IndicatorValidationSvc()


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_run_log(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _notch_payload(params: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    try:
        notches, notch_widths = normalize_tensor_metric_notch_params(
            params.get("notches"),
            params.get("notch_widths", DEFAULT_TENSOR_NOTCH_WIDTH),
        )
    except Exception:
        return None
    return list(notches), list(notch_widths)


def _normalize_channels(value: Any) -> list[str] | None:
    try:
        return normalize_metric_channels(value)
    except Exception:
        return None


def _normalize_pairs(value: Any, *, directed: bool) -> list[list[str]] | None:
    try:
        pairs = normalize_metric_pairs(value)
    except Exception:
        return None
    if pairs is None:
        return None
    normalized: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for source, target in pairs:
        pair = (source, target) if directed else tuple(sorted((source, target)))
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append([pair[0], pair[1]])
    return normalized


def _normalize_runtime_bands_signature(
    value: Any,
) -> dict[str, list[float] | list[list[float]]] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, list[float] | list[list[float]]] = {}
    for name in sorted(str(key).strip() for key in value.keys() if str(key).strip()):
        item = value.get(name)
        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and all(isinstance(part, (int, float)) for part in item)
        ):
            normalized[name] = [float(item[0]), float(item[1])]
            continue
        if not isinstance(item, (list, tuple)):
            return None
        segments: list[list[float]] = []
        for segment in item:
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                return None
            segments.append([float(segment[0]), float(segment[1])])
        normalized[name] = segments
    return normalized


def _max_n_peaks_signature(value: Any) -> str | float:
    if isinstance(value, str) and value.strip().lower() == "inf":
        return "inf"
    parsed = float(value)
    return "inf" if np.isinf(parsed) else float(parsed)


def _periodic_freq_range(params: dict[str, Any]) -> list[float] | None:
    freq_range = params.get("freq_range_hz")
    if isinstance(freq_range, (list, tuple)) and len(freq_range) == 2:
        low = _as_optional_float(freq_range[0])
        high = _as_optional_float(freq_range[1])
        if low is not None and high is not None and high > low:
            return [float(low), float(high)]
    spec_low = _as_optional_float(params.get("specparam_low_freq"))
    spec_high = _as_optional_float(params.get("specparam_high_freq"))
    if spec_low is None or spec_high is None or spec_high <= spec_low:
        return None
    return [float(spec_low), float(spec_high)]


def _metric_log_signature(
    metric_key: str, params: dict[str, Any]
) -> dict[str, Any] | None:
    notch_payload = _notch_payload(params)
    if notch_payload is None:
        return None
    notches, notch_widths = notch_payload
    if metric_key == "raw_power":
        channels = _normalize_channels(params.get("selected_channels"))
        if channels is None:
            return None
        return {
            "low_freq": float(params.get("low_freq")),
            "high_freq": float(params.get("high_freq")),
            "step_hz": float(params.get("step_hz")),
            "method": str(params.get("method", "morlet")),
            "time_resolution_s": float(params.get("time_resolution_s")),
            "hop_s": float(params.get("hop_s")),
            "min_cycles": _as_optional_float(params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(params.get("max_cycles")),
            "time_bandwidth": _as_float(params.get("time_bandwidth"), 1.0),
            "mask_edge_effects": bool(params.get("mask_edge_effects", True)),
            "notches": notches,
            "notch_widths": notch_widths,
            "selected_channels": channels,
        }
    if metric_key == "periodic_aperiodic":
        channels = _normalize_channels(params.get("selected_channels"))
        freq_range = _periodic_freq_range(params)
        if channels is None or freq_range is None:
            return None
        peak_width_limits = params.get("peak_width_limits_hz", [2.0, 12.0])
        if (
            not isinstance(peak_width_limits, (list, tuple))
            or len(peak_width_limits) != 2
        ):
            return None
        return {
            "low_freq": float(params.get("low_freq")),
            "high_freq": float(params.get("high_freq")),
            "step_hz": float(params.get("step_hz")),
            "method": str(params.get("method", "morlet")),
            "time_resolution_s": float(params.get("time_resolution_s")),
            "hop_s": float(params.get("hop_s")),
            "min_cycles": _as_optional_float(params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(params.get("max_cycles")),
            "time_bandwidth": _as_float(params.get("time_bandwidth"), 1.0),
            "freq_range_hz": freq_range,
            "freq_smooth_enabled": bool(params.get("freq_smooth_enabled", True)),
            "freq_smooth_sigma": _as_optional_float(
                params.get("freq_smooth_sigma"), 1.5
            ),
            "time_smooth_enabled": bool(params.get("time_smooth_enabled", True)),
            "time_smooth_kernel_size": _as_optional_int(
                params.get("time_smooth_kernel_size")
            ),
            "aperiodic_mode": str(params.get("aperiodic_mode", "fixed")),
            "peak_width_limits_hz": [
                float(peak_width_limits[0]),
                float(peak_width_limits[1]),
            ],
            "max_n_peaks": _max_n_peaks_signature(params.get("max_n_peaks", "inf")),
            "min_peak_height": _as_float(params.get("min_peak_height"), 0.0),
            "peak_threshold": _as_float(params.get("peak_threshold"), 2.0),
            "fit_qc_threshold": _as_float(params.get("fit_qc_threshold"), 0.6),
            "mask_edge_effects": bool(params.get("mask_edge_effects", True)),
            "notches": notches,
            "notch_widths": notch_widths,
            "selected_channels": channels,
        }
    if metric_key in {"coherence", "plv", "ciplv", "pli", "wpli"}:
        directed = False
        pairs = _normalize_pairs(params.get("selected_pairs"), directed=directed)
        if pairs is None:
            return None
        connectivity_metric_map = {
            "coherence": "coh",
            "plv": "plv",
            "ciplv": "ciplv",
            "pli": "pli",
            "wpli": "wpli",
        }
        return {
            "low_freq": float(params.get("low_freq")),
            "high_freq": float(params.get("high_freq")),
            "step_hz": float(params.get("step_hz")),
            "time_resolution_s": float(params.get("time_resolution_s")),
            "hop_s": float(params.get("hop_s")),
            "connectivity_metric": connectivity_metric_map[metric_key],
            "method": str(params.get("method", "morlet")),
            "mt_bandwidth": _as_optional_float(params.get("mt_bandwidth")),
            "min_cycles": _as_optional_float(params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(params.get("max_cycles")),
            "mask_edge_effects": bool(params.get("mask_edge_effects", True)),
            "notches": notches,
            "notch_widths": notch_widths,
            "selected_pairs": pairs,
        }
    if metric_key == "trgc":
        pairs = _normalize_pairs(params.get("selected_pairs"), directed=True)
        if pairs is None:
            return None
        return {
            "low_freq": float(params.get("low_freq")),
            "high_freq": float(params.get("high_freq")),
            "step_hz": float(params.get("step_hz")),
            "time_resolution_s": float(params.get("time_resolution_s")),
            "hop_s": float(params.get("hop_s")),
            "connectivity_metric": "trgc",
            "method": str(params.get("method", "morlet")),
            "mt_bandwidth": _as_optional_float(params.get("mt_bandwidth")),
            "min_cycles": _as_optional_float(params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(params.get("max_cycles")),
            "gc_n_lags": _as_int(params.get("gc_n_lags"), 20),
            "group_by_samples": bool(params.get("group_by_samples", False)),
            "round_ms": _as_float(params.get("round_ms"), 50.0),
            "mask_edge_effects": bool(params.get("mask_edge_effects", True)),
            "notches": notches,
            "notch_widths": notch_widths,
            "selected_pairs": pairs,
        }
    if metric_key == "psi":
        pairs = _normalize_pairs(params.get("selected_pairs"), directed=True)
        bands_used = _normalize_runtime_bands_signature(params.get("bands_used"))
        if pairs is None or bands_used is None:
            return None
        return {
            "low_freq": float(params.get("low_freq")),
            "high_freq": float(params.get("high_freq")),
            "step_hz": float(params.get("step_hz")),
            "method": str(params.get("method", "morlet")),
            "time_resolution_s": float(params.get("time_resolution_s")),
            "hop_s": float(params.get("hop_s")),
            "mt_bandwidth": _as_optional_float(params.get("mt_bandwidth")),
            "min_cycles": _as_optional_float(params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(params.get("max_cycles")),
            "mask_edge_effects": bool(params.get("mask_edge_effects", True)),
            "notches": notches,
            "notch_widths": notch_widths,
            "bands_used": bands_used,
            "selected_pairs": pairs,
        }
    if metric_key == "burst":
        channels = _normalize_channels(params.get("selected_channels"))
        bands_used = _normalize_runtime_bands_signature(params.get("bands_used"))
        if channels is None or bands_used is None:
            return None
        return {
            "low_freq": float(params.get("low_freq")),
            "high_freq": float(params.get("high_freq")),
            "step_hz": float(params.get("step_hz")),
            "percentile": _as_float(params.get("percentile"), 75.0),
            "baseline_keep": sorted(
                {
                    str(item).strip()
                    for item in (params.get("baseline_keep") or [])
                    if str(item).strip()
                }
            )
            or None,
            "baseline_match": "exact",
            "min_cycles": _as_float(params.get("min_cycles"), 2.0),
            "max_cycles": _as_optional_float(params.get("max_cycles")),
            "hop_s": _as_optional_float(params.get("hop_s")),
            "decim": _as_optional_int(params.get("decim"), 1),
            "mask_edge_effects": bool(params.get("mask_edge_effects", True)),
            "thresholds_source_path": (
                str(params.get("thresholds_source_path")).strip()
                if str(params.get("thresholds_source_path", "")).strip()
                else None
            ),
            "notches": notches,
            "notch_widths": notch_widths,
            "bands_used": bands_used,
            "selected_channels": channels,
        }
    return None


def _psi_or_burst_bands_signature(
    *,
    metric_key: str,
    metric_low: float,
    metric_high: float,
    bands: list[dict[str, Any]],
    notches: list[float],
    notch_widths: list[float],
) -> dict[str, list[float] | list[list[float]]] | None:
    notch_intervals = _compute_notch_intervals(
        low_freq=metric_low,
        high_freq=metric_high,
        notches=tuple(notches),
        notch_widths=tuple(notch_widths),
    )
    if metric_key == "psi":
        runtime_bands = _build_psi_runtime_bands(
            bands=bands,
            low_freq=metric_low,
            high_freq=metric_high,
            notch_intervals=notch_intervals,
        )
        return _normalize_runtime_bands_signature(
            _serialize_psi_runtime_bands(runtime_bands)
        )
    runtime_bands = _build_burst_runtime_bands(
        bands=bands,
        low_freq=metric_low,
        high_freq=metric_high,
        notch_intervals=notch_intervals,
    )
    return _normalize_runtime_bands_signature(
        _serialize_burst_runtime_bands(runtime_bands)
    )


def _current_metric_signature(
    context: RecordContext,
    *,
    metric_key: str,
    metric_params: dict[str, Any],
    mask_edge_effects: bool,
) -> dict[str, Any] | None:
    spec = TENSOR_METRICS_BY_KEY.get(metric_key)
    if spec is None or not spec.supported:
        return None
    notch_payload = _notch_payload(metric_params)
    if notch_payload is None:
        return None
    notches, notch_widths = notch_payload
    prepared = prepare_metric_plan_inputs(
        _VALIDATION_SVC,
        context,
        metric_key=metric_key,
        metric_label=spec.display_name,
        metric_params=dict(metric_params),
    )
    if metric_key == "raw_power":
        channels = _normalize_channels(prepared.metric_channels)
        if channels is None:
            return None
        return {
            "low_freq": prepared.metric_low,
            "high_freq": prepared.metric_high,
            "step_hz": prepared.metric_step,
            "method": str(metric_params.get("method", "morlet")),
            "time_resolution_s": _as_float(metric_params.get("time_resolution_s"), 0.5),
            "hop_s": _as_float(metric_params.get("hop_s"), 0.025),
            "min_cycles": _as_optional_float(metric_params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(metric_params.get("max_cycles")),
            "time_bandwidth": _as_float(metric_params.get("time_bandwidth"), 1.0),
            "mask_edge_effects": bool(mask_edge_effects),
            "notches": notches,
            "notch_widths": notch_widths,
            "selected_channels": channels,
        }
    if metric_key == "periodic_aperiodic":
        channels = _normalize_channels(prepared.metric_channels)
        if channels is None or prepared.parsed_freq_range is None:
            return None
        peak_width_limits = prepared.parsed_peak_width_limits
        return {
            "low_freq": prepared.metric_low,
            "high_freq": prepared.metric_high,
            "step_hz": prepared.metric_step,
            "method": str(metric_params.get("method", "morlet")),
            "time_resolution_s": _as_float(metric_params.get("time_resolution_s"), 0.5),
            "hop_s": _as_float(metric_params.get("hop_s"), 0.025),
            "min_cycles": _as_optional_float(metric_params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(metric_params.get("max_cycles")),
            "time_bandwidth": _as_float(metric_params.get("time_bandwidth"), 1.0),
            "freq_range_hz": [
                float(prepared.parsed_freq_range[0]),
                float(prepared.parsed_freq_range[1]),
            ],
            "freq_smooth_enabled": bool(metric_params.get("freq_smooth_enabled", True)),
            "freq_smooth_sigma": _as_optional_float(
                metric_params.get("freq_smooth_sigma"), 1.5
            ),
            "time_smooth_enabled": bool(metric_params.get("time_smooth_enabled", True)),
            "time_smooth_kernel_size": _as_optional_int(
                metric_params.get("time_smooth_kernel_size")
            ),
            "aperiodic_mode": str(metric_params.get("aperiodic_mode", "fixed")),
            "peak_width_limits_hz": [
                float(peak_width_limits[0]),
                float(peak_width_limits[1]),
            ],
            "max_n_peaks": _max_n_peaks_signature(
                metric_params.get("max_n_peaks", float(np.inf))
            ),
            "min_peak_height": _as_float(metric_params.get("min_peak_height"), 0.0),
            "peak_threshold": _as_float(metric_params.get("peak_threshold"), 2.0),
            "fit_qc_threshold": _as_float(metric_params.get("fit_qc_threshold"), 0.6),
            "mask_edge_effects": bool(mask_edge_effects),
            "notches": notches,
            "notch_widths": notch_widths,
            "selected_channels": channels,
        }
    if metric_key in {"coherence", "plv", "ciplv", "pli", "wpli"}:
        pairs = _normalize_pairs(prepared.metric_pairs, directed=False)
        if pairs is None:
            return None
        connectivity_metric_map = {
            "coherence": "coh",
            "plv": "plv",
            "ciplv": "ciplv",
            "pli": "pli",
            "wpli": "wpli",
        }
        return {
            "low_freq": prepared.metric_low,
            "high_freq": prepared.metric_high,
            "step_hz": prepared.metric_step,
            "time_resolution_s": _as_float(metric_params.get("time_resolution_s"), 0.5),
            "hop_s": _as_float(metric_params.get("hop_s"), 0.025),
            "connectivity_metric": connectivity_metric_map[metric_key],
            "method": str(metric_params.get("method", "morlet")),
            "mt_bandwidth": _as_optional_float(metric_params.get("mt_bandwidth")),
            "min_cycles": _as_optional_float(metric_params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(metric_params.get("max_cycles")),
            "mask_edge_effects": bool(mask_edge_effects),
            "notches": notches,
            "notch_widths": notch_widths,
            "selected_pairs": pairs,
        }
    if metric_key == "trgc":
        pairs = _normalize_pairs(prepared.metric_pairs, directed=True)
        if pairs is None:
            return None
        return {
            "low_freq": prepared.metric_low,
            "high_freq": prepared.metric_high,
            "step_hz": prepared.metric_step,
            "time_resolution_s": _as_float(metric_params.get("time_resolution_s"), 0.5),
            "hop_s": _as_float(metric_params.get("hop_s"), 0.025),
            "connectivity_metric": "trgc",
            "method": str(metric_params.get("method", "morlet")),
            "mt_bandwidth": _as_optional_float(metric_params.get("mt_bandwidth")),
            "min_cycles": _as_optional_float(metric_params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(metric_params.get("max_cycles")),
            "gc_n_lags": _as_int(metric_params.get("gc_n_lags"), 20),
            "group_by_samples": bool(metric_params.get("group_by_samples", False)),
            "round_ms": _as_float(metric_params.get("round_ms"), 50.0),
            "mask_edge_effects": bool(mask_edge_effects),
            "notches": notches,
            "notch_widths": notch_widths,
            "selected_pairs": pairs,
        }
    if metric_key == "psi":
        pairs = _normalize_pairs(prepared.metric_pairs, directed=True)
        bands = normalize_metric_bands(metric_params.get("bands"))
        bands_used = _psi_or_burst_bands_signature(
            metric_key=metric_key,
            metric_low=prepared.metric_low,
            metric_high=prepared.metric_high,
            bands=bands,
            notches=notches,
            notch_widths=notch_widths,
        )
        if pairs is None or bands_used is None:
            return None
        return {
            "low_freq": prepared.metric_low,
            "high_freq": prepared.metric_high,
            "step_hz": prepared.metric_step,
            "method": str(metric_params.get("method", "morlet")),
            "time_resolution_s": _as_float(metric_params.get("time_resolution_s"), 0.5),
            "hop_s": _as_float(metric_params.get("hop_s"), 0.025),
            "mt_bandwidth": _as_optional_float(metric_params.get("mt_bandwidth")),
            "min_cycles": _as_optional_float(metric_params.get("min_cycles"), 3.0),
            "max_cycles": _as_optional_float(metric_params.get("max_cycles")),
            "mask_edge_effects": bool(mask_edge_effects),
            "notches": notches,
            "notch_widths": notch_widths,
            "bands_used": bands_used,
            "selected_pairs": pairs,
        }
    if metric_key == "burst":
        channels = _normalize_channels(prepared.metric_channels)
        bands = normalize_metric_bands(metric_params.get("bands"))
        bands_used = _psi_or_burst_bands_signature(
            metric_key=metric_key,
            metric_low=prepared.metric_low,
            metric_high=prepared.metric_high,
            bands=bands,
            notches=notches,
            notch_widths=notch_widths,
        )
        if channels is None or bands_used is None:
            return None
        hop_s, decim = _resolve_burst_time_grid(
            hop_s=_as_optional_float(metric_params.get("hop_s")),
            decim=_as_optional_int(metric_params.get("decim")),
        )
        baseline_keep = (
            sorted(
                {
                    str(item).strip()
                    for item in (metric_params.get("baseline_keep") or [])
                    if str(item).strip()
                }
            )
            or None
        )
        thresholds_path = (
            str(metric_params.get("thresholds_path")).strip()
            if str(metric_params.get("thresholds_path", "")).strip()
            else None
        )
        return {
            "low_freq": prepared.metric_low,
            "high_freq": prepared.metric_high,
            "step_hz": prepared.metric_step,
            "percentile": _as_float(metric_params.get("percentile"), 75.0),
            "baseline_keep": baseline_keep,
            "baseline_match": "exact",
            "min_cycles": _as_float(metric_params.get("min_cycles"), 2.0),
            "max_cycles": _as_optional_float(metric_params.get("max_cycles")),
            "hop_s": hop_s,
            "decim": decim,
            "mask_edge_effects": bool(mask_edge_effects),
            "thresholds_source_path": thresholds_path,
            "notches": notches,
            "notch_widths": notch_widths,
            "bands_used": bands_used,
            "selected_channels": channels,
        }
    return None


def tensor_metric_panel_state(
    context: RecordContext | None,
    *,
    metric_key: str,
    metric_params: dict[str, Any] | None,
    mask_edge_effects: bool,
) -> str:
    """Return `gray|yellow|green` for one Build Tensor metric row."""
    if context is None:
        return "gray"
    resolver = PathResolver(context)
    payload = _read_payload(tensor_metric_log_path(resolver, metric_key))
    if payload is None:
        return "gray"
    completed = payload.get("completed")
    if completed is False:
        return "yellow"
    if completed is not True:
        return "gray"
    params = payload.get("params")
    completed_signature = (
        _metric_log_signature(metric_key, params) if isinstance(params, dict) else None
    )
    if completed_signature is None:
        return "green"
    current_params = dict(metric_params) if isinstance(metric_params, dict) else {}
    try:
        current_signature = _current_metric_signature(
            context,
            metric_key=metric_key,
            metric_params=current_params,
            mask_edge_effects=mask_edge_effects,
        )
    except Exception:
        return "yellow"
    if current_signature is None:
        return "yellow"
    return "green" if current_signature == completed_signature else "yellow"


__all__ = ["tensor_metric_panel_state"]
