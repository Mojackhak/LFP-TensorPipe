"""Derive/plot default configuration loaders and reducers."""

from __future__ import annotations

from typing import Any

import numpy as np

from lfptensorpipe.app.config_store import AppConfigStore, FEATURES_CONFIG_FILENAME
from lfptensorpipe.tabular.grid import GridResultColumns

DEFAULT_DERIVE_PARAM_CFG: dict[str, dict[str, bool]] = {
    "aperiodic": {"raw": False, "spectral": False, "trace": True, "scalar": True},
    "psi": {"raw": True, "spectral": False, "trace": True, "scalar": True},
    "burst": {"raw": True, "spectral": False, "trace": False, "scalar": True},
    "default": {"raw": True, "spectral": True, "trace": True, "scalar": True},
}

DEFAULT_REDUCER_CFG: dict[str, dict[str, list[str]]] = {
    "burst": {"burst": ["mean", "occupation", "rate", "duration"]},
    "default": {"default": ["mean"]},
}

DEFAULT_REDUCER_RULE_BY_METHOD: dict[str, dict[str, list[str]]] = {
    "burst": {
        "linear_warper": ["mean", "occupation"],
        "stack_warper": ["mean", "occupation"],
    }
}

DEFAULT_COLLAPSE_BASE_CFG: dict[str, Any] = {
    "value_col": "Value",
    "time_interval_mode": "percent",
    "freq_interval_mode": "absolute",
    "inclusive": False,
    "drop_value": True,
    "keep_full_dim_cols": False,
    "on_missing": "skip",
}

DEFAULT_PLOT_ADVANCE_CFG: dict[str, Any] = {
    "transform_mode": "none",
    "normalize_mode": "none",
    "baseline_mode": "mean",
    "baseline_percent_ranges": [[0.0, 20.0]],
    "colormap": "viridis",
    "x_log": False,
    "y_log": False,
}

AUTO_BAND_METRIC_KEYS = frozenset({"aperiodic", "periodic_aperiodic", "psi", "burst"})


def _read_derive_payload(config_store: AppConfigStore | None) -> dict[str, Any]:
    if config_store is None:
        return {}
    payload = config_store.read_yaml(FEATURES_CONFIG_FILENAME, default={})
    return payload if isinstance(payload, dict) else {}


def _load_post_transform_modes(config_store: AppConfigStore | None) -> dict[str, str]:
    defaults = {
        "raw": "none",
        "trace": "none",
        "scalar": "none",
        "spectral": "none",
    }
    payload = _read_derive_payload(config_store)
    post = payload.get("post_transform_mode", {})
    if not isinstance(post, dict):
        return defaults
    out = dict(defaults)
    for key, value in post.items():
        key_parsed = str(key).strip().lower()
        value_parsed = str(value).strip().lower()
        if key_parsed in out and value_parsed:
            out[key_parsed] = value_parsed
    return out


def _load_plot_advance_defaults(config_store: AppConfigStore | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "transform_mode": str(DEFAULT_PLOT_ADVANCE_CFG["transform_mode"]),
        "normalize_mode": str(DEFAULT_PLOT_ADVANCE_CFG["normalize_mode"]),
        "baseline_mode": str(DEFAULT_PLOT_ADVANCE_CFG["baseline_mode"]),
        "baseline_percent_ranges": [
            [float(pair[0]), float(pair[1])]
            for pair in DEFAULT_PLOT_ADVANCE_CFG["baseline_percent_ranges"]
        ],
        "colormap": str(DEFAULT_PLOT_ADVANCE_CFG["colormap"]),
        "x_log": bool(DEFAULT_PLOT_ADVANCE_CFG["x_log"]),
        "y_log": bool(DEFAULT_PLOT_ADVANCE_CFG["y_log"]),
    }
    payload = _read_derive_payload(config_store)
    node = payload.get("plot_advance_defaults", {})
    if not isinstance(node, dict):
        return out
    transform_mode = str(node.get("transform_mode", out["transform_mode"])).strip()
    normalize_mode = str(node.get("normalize_mode", out["normalize_mode"])).strip()
    baseline_mode = str(node.get("baseline_mode", out["baseline_mode"])).strip()
    colormap = str(node.get("colormap", out["colormap"])).strip()
    x_log = bool(node.get("x_log", out["x_log"]))
    y_log = bool(node.get("y_log", out["y_log"]))
    ranges = node.get("baseline_percent_ranges", out["baseline_percent_ranges"])
    parsed_ranges: list[list[float]] = []
    if isinstance(ranges, list):
        for item in ranges:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                start = float(item[0])
                end = float(item[1])
            except Exception:
                continue
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue
            parsed_ranges.append([start, end])
    out["transform_mode"] = transform_mode or str(out["transform_mode"])
    out["normalize_mode"] = normalize_mode or str(out["normalize_mode"])
    out["baseline_mode"] = baseline_mode or str(out["baseline_mode"])
    out["colormap"] = colormap or str(out["colormap"])
    out["x_log"] = x_log
    out["y_log"] = y_log
    if parsed_ranges:
        out["baseline_percent_ranges"] = parsed_ranges
    return out


def _normalize_enabled_outputs_map(value: Any) -> dict[str, bool]:
    defaults = {"raw": True, "spectral": True, "trace": True, "scalar": True}
    if not isinstance(value, dict):
        return defaults
    out = dict(defaults)
    for key in ("raw", "spectral", "trace", "scalar"):
        if key in value:
            out[key] = bool(value.get(key))
    return out


def _load_derive_param_cfg(
    config_store: AppConfigStore | None,
) -> dict[str, dict[str, bool]]:
    payload = _read_derive_payload(config_store)
    candidate = payload.get("derive_param_cfg", {})
    out = {key: dict(value) for key, value in DEFAULT_DERIVE_PARAM_CFG.items()}
    if not isinstance(candidate, dict):
        return out
    for raw_key, raw_value in candidate.items():
        metric_key = str(raw_key).strip()
        if not metric_key:
            continue
        out[metric_key] = _normalize_enabled_outputs_map(raw_value)
    if "default" not in out:
        out["default"] = dict(DEFAULT_DERIVE_PARAM_CFG["default"])
    return out


def _normalize_reducer_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["mean"]
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item).strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out or ["mean"]


def _load_reducer_cfg(
    config_store: AppConfigStore | None,
) -> dict[str, dict[str, list[str]]]:
    payload = _read_derive_payload(config_store)
    candidate = payload.get("reducer_cfg", {})
    out = {
        key: {sub: list(items) for sub, items in value.items()}
        for key, value in DEFAULT_REDUCER_CFG.items()
    }
    if not isinstance(candidate, dict):
        return out
    for raw_key, raw_value in candidate.items():
        metric_key = str(raw_key).strip()
        if not metric_key or not isinstance(raw_value, dict):
            continue
        parsed_entry: dict[str, list[str]] = {}
        for raw_sub, raw_reducers in raw_value.items():
            subkey = str(raw_sub).strip()
            if not subkey:
                continue
            parsed_entry[subkey] = _normalize_reducer_list(raw_reducers)
        if parsed_entry:
            out[metric_key] = parsed_entry
    if "default" not in out:
        out["default"] = {"default": ["mean"]}
    return out


def _load_reducer_rule_by_method(
    config_store: AppConfigStore | None,
) -> dict[str, dict[str, list[str]]]:
    payload = _read_derive_payload(config_store)
    candidate = payload.get("reducer_rule_by_method", {})
    out = {
        metric_key: {
            method_key: list(reducers) for method_key, reducers in node.items()
        }
        for metric_key, node in DEFAULT_REDUCER_RULE_BY_METHOD.items()
    }
    if not isinstance(candidate, dict):
        return out
    for raw_metric, raw_node in candidate.items():
        metric_key = str(raw_metric).strip()
        if not metric_key or not isinstance(raw_node, dict):
            continue
        parsed_node: dict[str, list[str]] = {}
        for raw_method, raw_reducers in raw_node.items():
            method_key = str(raw_method).strip()
            if not method_key:
                continue
            parsed_node[method_key] = _normalize_reducer_list(raw_reducers)
        if parsed_node:
            out[metric_key] = parsed_node
    return out


def _load_collapse_base_cfg(config_store: AppConfigStore | None) -> dict[str, Any]:
    out = dict(DEFAULT_COLLAPSE_BASE_CFG)
    payload = _read_derive_payload(config_store)
    candidate = payload.get("collapse_base_cfg", {})
    if not isinstance(candidate, dict):
        return out
    for key in (
        "value_col",
        "time_interval_mode",
        "freq_interval_mode",
        "inclusive",
        "drop_value",
        "keep_full_dim_cols",
        "on_missing",
    ):
        if key in candidate:
            out[key] = candidate.get(key)
    out["out_cols"] = GridResultColumns(band="Band", phase="Phase", value="Value")
    return out


def _resolve_enabled_outputs(
    derive_param_cfg: dict[str, dict[str, bool]],
    metric_key: str,
) -> dict[str, bool]:
    node = derive_param_cfg.get(metric_key, derive_param_cfg.get("default", {}))
    return _normalize_enabled_outputs_map(node)


def _resolve_reducers(
    reducer_cfg: dict[str, dict[str, list[str]]],
    metric_key: str,
) -> list[str]:
    node = reducer_cfg.get(metric_key, reducer_cfg.get("default", {}))
    if not isinstance(node, dict) or not node:
        return ["mean"]
    if metric_key in node:
        return _normalize_reducer_list(node.get(metric_key))
    if "default" in node:
        return _normalize_reducer_list(node.get("default"))
    first_key = next(iter(node.keys()))
    return _normalize_reducer_list(node.get(first_key))


def _metric_uses_auto_bands(metric_key: str) -> bool:
    return metric_key.strip().lower() in AUTO_BAND_METRIC_KEYS


def load_derive_defaults(config_store: AppConfigStore | None) -> dict[str, Any]:
    return {
        "derive_param_cfg": _load_derive_param_cfg(config_store),
        "reducer_cfg": _load_reducer_cfg(config_store),
        "reducer_rule_by_method": _load_reducer_rule_by_method(config_store),
        "collapse_base_cfg": _load_collapse_base_cfg(config_store),
        "post_transform_mode": _load_post_transform_modes(config_store),
        "plot_advance_defaults": _load_plot_advance_defaults(config_store),
    }
