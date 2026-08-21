"""Backend runtime package for LFP-centric core computations.

Target home for Build Tensor / Align Epochs / feature-related numeric kernels
migrated out of app-layer services.
"""

from __future__ import annotations

from .tensor_helpers import (
    ESTIMATOR_MASK_SUPPORT_SEMANTICS,
    apply_dynamic_edge_mask_strict,
    build_annotation_skip_time_mask,
    build_frequency_grid,
    compute_mask_radii_seconds,
    compute_notch_intervals,
    connectivity_consumed_support_radii_seconds,
    cut_frequency_grid_by_intervals,
    cycles_from_time_resolution,
    expand_notch_radii,
    interpolated_support_radii_seconds,
    parse_positive_float_tuple,
    psi_band_radii_seconds,
)

__all__ = [
    "ESTIMATOR_MASK_SUPPORT_SEMANTICS",
    "apply_dynamic_edge_mask_strict",
    "build_annotation_skip_time_mask",
    "build_frequency_grid",
    "compute_mask_radii_seconds",
    "compute_notch_intervals",
    "connectivity_consumed_support_radii_seconds",
    "cut_frequency_grid_by_intervals",
    "cycles_from_time_resolution",
    "expand_notch_radii",
    "interpolated_support_radii_seconds",
    "parse_positive_float_tuple",
    "psi_band_radii_seconds",
]
