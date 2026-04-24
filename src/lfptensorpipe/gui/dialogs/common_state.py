"""Shared dialog state normalization wrappers."""

from __future__ import annotations

from typing import Any

from lfptensorpipe.gui.state import (
    deep_merge_dict as _state_deep_merge_dict,
    nested_get as _state_nested_get,
    nested_set as _state_nested_set,
    default_preproc_filter_basic_params as _state_default_preproc_filter_basic_params,
    default_preproc_viz_psd_params as _state_default_preproc_viz_psd_params,
    default_preproc_viz_tfr_params as _state_default_preproc_viz_tfr_params,
    normalize_filter_notches_config as _state_normalize_filter_notches_config,
    normalize_preproc_filter_basic_params as _state_normalize_preproc_filter_basic_params,
    normalize_preproc_viz_psd_params as _state_normalize_preproc_viz_psd_params,
    normalize_preproc_viz_tfr_params as _state_normalize_preproc_viz_tfr_params,
)


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    return _state_deep_merge_dict(base, overlay)


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    return _state_nested_get(payload, path)


def _nested_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    _state_nested_set(payload, path, value)


def default_preproc_viz_psd_params() -> dict[str, Any]:
    return _state_default_preproc_viz_psd_params()


def default_preproc_filter_basic_params() -> dict[str, Any]:
    return _state_default_preproc_filter_basic_params()


def _normalize_filter_notches_config(value: Any) -> list[float]:
    return _state_normalize_filter_notches_config(value)


def normalize_preproc_filter_basic_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], str]:
    return _state_normalize_preproc_filter_basic_params(params)


def normalize_preproc_viz_psd_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], str]:
    return _state_normalize_preproc_viz_psd_params(params)


def default_preproc_viz_tfr_params() -> dict[str, Any]:
    return _state_default_preproc_viz_tfr_params()


def normalize_preproc_viz_tfr_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], str]:
    return _state_normalize_preproc_viz_tfr_params(params)


__all__ = [
    name for name in globals() if not (name.startswith("__") and name.endswith("__"))
]
