"""Backend runtime package for LFP-centric core computations.

Target home for Build Tensor / Align Epochs / feature-related numeric kernels
migrated out of app-layer services.
"""

from __future__ import annotations

from .tensor_helpers import (
    apply_dynamic_edge_mask_strict,
    build_frequency_grid,
    compute_mask_radii_seconds,
    compute_notch_intervals,
    cut_frequency_grid_by_intervals,
    cycles_from_time_resolution,
    expand_notch_widths,
    parse_positive_float_tuple,
    psi_band_radii_seconds,
)

__all__ = [
    "apply_dynamic_edge_mask_strict",
    "build_frequency_grid",
    "compute_mask_radii_seconds",
    "compute_notch_intervals",
    "cut_frequency_grid_by_intervals",
    "cycles_from_time_resolution",
    "expand_notch_widths",
    "parse_positive_float_tuple",
    "psi_band_radii_seconds",
]
