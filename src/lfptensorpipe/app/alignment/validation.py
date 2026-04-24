"""Alignment method validation facade."""

from __future__ import annotations

from .method_params import (
    default_alignment_method_params,
    validate_alignment_method_params,
)
from .method_specs import (
    ALIGNMENT_METHODS,
    ALIGNMENT_METHODS_BY_KEY,
    ALIGNMENT_METHODS_BY_LABEL,
    DEFAULT_DROP_FIELDS,
    DEFAULT_DROP_MODE,
    AlignmentMethodSpec,
)
from .param_normalizers import (
    _normalize_anchors,
    _normalize_annotations,
    _normalize_drop_fields,
    _normalize_duration_range,
    _normalize_nonnegative_float,
    _normalize_sample_rate,
    _normalize_slug,
)

__all__ = [
    "AlignmentMethodSpec",
    "ALIGNMENT_METHODS",
    "ALIGNMENT_METHODS_BY_KEY",
    "ALIGNMENT_METHODS_BY_LABEL",
    "DEFAULT_DROP_MODE",
    "DEFAULT_DROP_FIELDS",
    "default_alignment_method_params",
    "validate_alignment_method_params",
    "_normalize_sample_rate",
    "_normalize_nonnegative_float",
    "_normalize_duration_range",
    "_normalize_annotations",
    "_normalize_drop_fields",
    "_normalize_anchors",
    "_normalize_slug",
]
