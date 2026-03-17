"""Alignment configuration persistence facade."""

from __future__ import annotations

from .config_payload import (
    _normalize_method_defaults,
    _read_alignment_payload,
    _resolve_alignment_method_key,
)
from .trial_config import (
    _append_alignment_history,
    _extract_trial_config_from_history,
    _finish_time_axis_values,
    _load_trial_config_from_log,
    _normalize_paradigm,
    _trial_config_from_payload,
)
from .trial_store import (
    create_alignment_paradigm,
    delete_alignment_paradigm,
    load_alignment_method_default_params,
    load_alignment_method_defaults,
    load_alignment_paradigms,
    save_alignment_method_default_params,
    save_alignment_paradigms,
    update_alignment_paradigm,
)

__all__ = [
    "_append_alignment_history",
    "_extract_trial_config_from_history",
    "_finish_time_axis_values",
    "_load_trial_config_from_log",
    "_normalize_method_defaults",
    "_normalize_paradigm",
    "_read_alignment_payload",
    "_resolve_alignment_method_key",
    "_trial_config_from_payload",
    "create_alignment_paradigm",
    "delete_alignment_paradigm",
    "load_alignment_method_default_params",
    "load_alignment_method_defaults",
    "load_alignment_paradigms",
    "save_alignment_method_default_params",
    "save_alignment_paradigms",
    "update_alignment_paradigm",
]
