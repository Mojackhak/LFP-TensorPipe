"""GUI state normalization helpers.

This package is reserved for small, UI-scoped state transforms and default
normalizers. MainWindow orchestration should stay in `gui.shell`.
"""

from __future__ import annotations

from .normalizers import (
    deep_merge_dict,
    nested_get,
    nested_set,
    default_preproc_filter_basic_params,
    default_preproc_viz_psd_params,
    default_preproc_viz_tfr_params,
    normalize_filter_notches_config,
    normalize_preproc_filter_basic_params,
    normalize_preproc_viz_psd_params,
    normalize_preproc_viz_tfr_params,
)

__all__ = [
    "deep_merge_dict",
    "nested_get",
    "nested_set",
    "default_preproc_filter_basic_params",
    "default_preproc_viz_psd_params",
    "default_preproc_viz_tfr_params",
    "normalize_filter_notches_config",
    "normalize_preproc_filter_basic_params",
    "normalize_preproc_viz_psd_params",
    "normalize_preproc_viz_tfr_params",
]
