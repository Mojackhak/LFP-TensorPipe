"""Shared Tensor page JSON parameter normalization."""

from __future__ import annotations

import math
from typing import Any

from lfptensorpipe.app.shared.page_config import json_value
from lfptensorpipe.io.burst_thresholds import normalize_burst_threshold_payload

from .frequency import build_tensor_metric_notch_payload
from .params import TENSOR_METRICS_BY_KEY
from .selectors import (
    coerce_tensor_channels,
    coerce_tensor_pairs,
    filter_tensor_pairs,
    normalize_tensor_bands_rows,
    tensor_metric_pair_mode,
)

TENSOR_CONFIG_SCHEMA = "lfptensorpipe.tensor-config"
TENSOR_CONFIG_VERSION = 4
TENSOR_CONFIG_LEGACY_VERSION = 3
TENSOR_CONFIG_FILE_NAME = "lfptensorpipe_tensor_config.json"
TENSOR_MULTITAPER_METRIC_KEYS = frozenset(
    {
        "raw_power",
        "periodic_aperiodic",
        "coherence",
        "imcoh_abs",
        "plv",
        "ciplv",
        "pli",
        "wpli",
        "trgc",
        "psi",
    }
)
TENSOR_DIRTY_KEYS = {
    "tensor.active_metric",
    "tensor.mask_edge_effects",
    "tensor.cpu_percent",
    "tensor.metric_params",
    "tensor.selected_metrics",
    "tensor.selectors",
}
TENSOR_CONFIG_FIELDS_BY_METRIC: dict[str, tuple[str, ...]] = {
    "raw_power": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "min_cycles",
        "max_cycles",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "notches",
        "notch_radii",
        "selected_channels",
    ),
    "periodic_aperiodic": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "freq_range_hz",
        "min_cycles",
        "max_cycles",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "freq_smooth_enabled",
        "freq_smooth_sigma",
        "time_smooth_enabled",
        "time_smooth_kernel_size",
        "aperiodic_mode",
        "peak_width_limits_hz",
        "max_n_peaks",
        "min_peak_height",
        "peak_threshold",
        "fit_qc_threshold",
        "notches",
        "notch_radii",
        "selected_channels",
    ),
    "coherence": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "imcoh_abs": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "plv": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "ciplv": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "pli": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "wpli": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "trgc": (
        "low_freq_hz",
        "high_freq_hz",
        "freq_step_hz",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "min_cycles",
        "max_cycles",
        "gc_n_lags",
        "group_by_samples",
        "round_ms",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "psi": (
        "freq_step_hz",
        "bands",
        "time_resolution_s",
        "hop_s",
        "method",
        "mt_time_bandwidth_product",
        "mt_min_cycles",
        "mt_max_cycles",
        "min_cycles",
        "max_cycles",
        "notches",
        "notch_radii",
        "selected_pairs",
    ),
    "burst": (
        "bands",
        "percentile",
        "baseline_keep",
        "boundary_isolated_filter",
        "method",
        "hilbert_filter_method",
        "freq_step_hz",
        "morlet_n_cycles",
        "mt_n_cycles",
        "mt_time_bandwidth_product",
        "hilbert_edge_tolerance_pct",
        "min_cycles",
        "max_cycles",
        "thresholds",
        "notches",
        "notch_radii",
        "selected_channels",
    ),
}


def convert_legacy_tensor_metric(
    metric_key: str, node: dict[str, Any], *, keep_current_notches: bool = False
) -> dict[str, Any]:
    """Apply the existing GUI conversion for a supported legacy Tensor export."""
    out = dict(node)
    radius_present = "notch_widths" in out
    radius = out.pop("notch_widths", None)
    if not keep_current_notches:
        out.pop("notch_radii", None)
    if radius_present:
        out["notch_radii"] = radius
    if metric_key in TENSOR_MULTITAPER_METRIC_KEYS:
        out.pop("time_bandwidth", None)
        out.pop("mt_bandwidth", None)
        out.update(mt_time_bandwidth_product=4.0, mt_min_cycles=3.0, mt_max_cycles=None)
    return out


def normalize_tensor_config_metric_params(
    metric_key: str,
    node: dict[str, Any],
    *,
    available_channels: tuple[str, ...],
    legacy_notch_fields: bool = False,
    strict: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(node, dict):
        raise ValueError(f"tensor.metric_params.{metric_key} must be an object.")
    if "notch_widths" in node:
        raise ValueError(
            f"tensor.metric_params.{metric_key}.notch_widths was removed "
            "from Build Tensor schema 4. Use notch_radii instead."
        )
    if metric_key == "periodic_aperiodic":
        removed_keys = sorted(
            key for key in ("smooth_enabled", "kernel_size") if key in node
        )
        if removed_keys:
            raise ValueError(
                "tensor.metric_params.periodic_aperiodic contains removed keys: "
                + ", ".join(removed_keys)
                + "."
            )

    whitelist = TENSOR_CONFIG_FIELDS_BY_METRIC.get(metric_key, ())
    if strict:
        unknown = sorted(set(node) - set(whitelist))
        if unknown:
            raise ValueError(
                f"{metric_key} contains unsupported config fields: {', '.join(unknown)}."
            )
    pair_mode = tensor_metric_pair_mode(metric_key)
    warnings: list[str] = []
    out: dict[str, Any] = {}

    for key in whitelist:
        if key not in node:
            continue
        value = node.get(key)
        if key == "selected_channels":
            if not isinstance(value, list):
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.selected_channels must be a list."
                )
            if strict and any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(
                    f"{metric_key}.selected_channels contains an invalid channel name."
                )
            normalized = coerce_tensor_channels(value)
            filtered = tuple(
                channel for channel in normalized if channel in set(available_channels)
            )
            dropped = len(normalized) - len(filtered)
            if dropped > 0:
                warnings.append(
                    f"{TENSOR_METRICS_BY_KEY[metric_key].display_name} ignored "
                    f"{dropped} unavailable channel(s), including any channel(s) "
                    "marked bad in Preprocess Finish."
                )
            out[key] = [str(item) for item in filtered]
            continue
        if key == "selected_pairs":
            if not isinstance(value, list):
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.selected_pairs must be a list."
                )
            if strict and any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(item, str) or not item.strip() for item in pair)
                or pair[0].strip() == pair[1].strip()
                for pair in value
            ):
                raise ValueError(
                    f"{metric_key}.selected_pairs contains an invalid or self pair."
                )
            normalized_pairs = coerce_tensor_pairs(
                value,
                directed=(pair_mode == "directed"),
            )
            filtered_pairs = filter_tensor_pairs(
                normalized_pairs,
                available_channels=available_channels,
                directed=(pair_mode == "directed"),
            )
            dropped = len(normalized_pairs) - len(filtered_pairs)
            if dropped > 0:
                warnings.append(
                    f"{TENSOR_METRICS_BY_KEY[metric_key].display_name} ignored "
                    f"{dropped} unavailable pair(s), including any pair(s) "
                    "containing a channel marked bad in Preprocess Finish."
                )
            out[key] = [[source, target] for source, target in filtered_pairs]
            continue
        if key == "bands":
            if not isinstance(value, list):
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.bands must be a list."
                )
            out[key] = [dict(item) for item in normalize_tensor_bands_rows(value)]
            if strict and len(out[key]) != len(value):
                raise ValueError(
                    f"{metric_key}.bands contains invalid or duplicate rows."
                )
            continue
        if key == "notches":
            out[key] = build_tensor_metric_notch_payload(
                value,
                node.get("notch_radii"),
                legacy_mismatched_list_broadcast=legacy_notch_fields,
            )["notches"]
            continue
        if key == "notch_radii":
            out[key] = build_tensor_metric_notch_payload(
                node.get("notches"),
                value,
                legacy_mismatched_list_broadcast=legacy_notch_fields,
            )["notch_radii"]
            continue
        if key == "thresholds":
            if value is None:
                out[key] = None
            else:
                try:
                    out[key] = normalize_burst_threshold_payload(value)
                except ValueError as exc:
                    raise ValueError(
                        "tensor.metric_params.burst.thresholds is invalid: " + str(exc)
                    ) from exc
            continue
        if key == "baseline_keep":
            if value is None:
                out[key] = None
                continue
            if not isinstance(value, list):
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.baseline_keep must be a list or null."
                )
            if strict and any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(
                    f"{metric_key}.baseline_keep contains an invalid annotation name."
                )
            labels: list[str] = []
            seen: set[str] = set()
            for item in value:
                label = str(item).strip()
                if not label or label in seen:
                    continue
                seen.add(label)
                labels.append(label)
            out[key] = labels or None
            continue
        if key == "boundary_isolated_filter":
            if not isinstance(value, bool):
                raise ValueError(
                    "tensor.metric_params.burst.boundary_isolated_filter "
                    "must be true or false."
                )
            out[key] = bool(value)
            continue
        if key == "mt_time_bandwidth_product":
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.{key} must be a number."
                ) from exc
            if not math.isfinite(parsed) or parsed < 2.0:
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.{key} must be finite and >= 2."
                )
            out[key] = parsed
            continue
        if key == "mt_min_cycles":
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.{key} must be a number."
                ) from exc
            if not math.isfinite(parsed) or parsed <= 0.0:
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.{key} must be finite and > 0."
                )
            out[key] = parsed
            continue
        if key == "mt_max_cycles":
            if value is None:
                out[key] = None
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.{key} must be a number or null."
                ) from exc
            if not math.isfinite(parsed) or parsed <= 0.0:
                raise ValueError(
                    f"tensor.metric_params.{metric_key}.{key} must be finite and > 0 when provided."
                )
            out[key] = parsed
            continue
        out[key] = json_value(value)

    out.update(
        build_tensor_metric_notch_payload(
            out.get("notches"),
            out.get("notch_radii"),
            legacy_mismatched_list_broadcast=legacy_notch_fields,
        )
    )
    mt_min_cycles = out.get("mt_min_cycles")
    mt_max_cycles = out.get("mt_max_cycles")
    if (
        mt_max_cycles is not None
        and mt_min_cycles is not None
        and float(mt_max_cycles) < float(mt_min_cycles)
    ):
        raise ValueError(
            f"tensor.metric_params.{metric_key}.mt_max_cycles must be >= mt_min_cycles."
        )
    if strict and warnings:
        raise ValueError(" ".join(warnings))
    return out, warnings
