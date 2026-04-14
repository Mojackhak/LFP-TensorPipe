"""Compatibility facade for feature derive configuration helpers."""

from __future__ import annotations

from .derive_alignment import _extract_alignment_method_from_log
from .derive_axes import (
    _bands_from_raw_value_index,
    _normalize_axis_rows,
    _rows_to_interval_mapping,
)
from .derive_defaults import (
    AUTO_BAND_METRIC_KEYS,
    DEFAULT_COLLAPSE_BASE_CFG,
    DEFAULT_DERIVE_PARAM_CFG,
    DEFAULT_PLOT_ADVANCE_CFG,
    DEFAULT_REDUCER_CFG,
    DEFAULT_REDUCER_RULE_BY_METHOD,
    _load_collapse_base_cfg,
    _load_derive_param_cfg,
    _load_plot_advance_defaults,
    _load_post_transform_modes,
    _load_reducer_cfg,
    _load_reducer_rule_by_method,
    _metric_uses_auto_bands,
    _normalize_enabled_outputs_map,
    _normalize_reducer_list,
    _read_derive_payload,
    _resolve_enabled_outputs,
    _resolve_reducers,
    load_derive_defaults,
)

__all__ = [
    "AUTO_BAND_METRIC_KEYS",
    "DEFAULT_COLLAPSE_BASE_CFG",
    "DEFAULT_DERIVE_PARAM_CFG",
    "DEFAULT_PLOT_ADVANCE_CFG",
    "DEFAULT_REDUCER_CFG",
    "DEFAULT_REDUCER_RULE_BY_METHOD",
    "_bands_from_raw_value_index",
    "_extract_alignment_method_from_log",
    "_load_collapse_base_cfg",
    "_load_derive_param_cfg",
    "_load_plot_advance_defaults",
    "_load_post_transform_modes",
    "_load_reducer_cfg",
    "_load_reducer_rule_by_method",
    "_metric_uses_auto_bands",
    "_normalize_axis_rows",
    "_normalize_enabled_outputs_map",
    "_normalize_reducer_list",
    "_read_derive_payload",
    "_resolve_enabled_outputs",
    "_resolve_reducers",
    "_rows_to_interval_mapping",
    "load_derive_defaults",
]
