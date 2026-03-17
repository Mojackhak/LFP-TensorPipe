"""Numeric safety helpers for transform and normalization paths.

These helpers implement the project policy that invalid values should remain
explicitly invalid (`NaN`) instead of being forced to finite outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_ABS_TOL = float(np.finfo(float).tiny)
DEFAULT_REL_TOL = 1e-8


def _validate_non_negative_finite(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be a finite number >= 0")
    return out


def resolve_abs_tol(abs_tol: float | None = None) -> float:
    """Resolve the absolute tolerance used for near-zero denominator checks."""

    if abs_tol is not None:
        return _validate_non_negative_finite(abs_tol, name="abs_tol")

    return DEFAULT_ABS_TOL


def resolve_rel_tol(rel_tol: float = DEFAULT_REL_TOL) -> float:
    """Validate the relative tolerance used for near-zero denominator checks."""

    return _validate_non_negative_finite(rel_tol, name="rel_tol")


def dynamic_denominator_tolerance(
    denominator: Any,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> float:
    """Compute the near-zero tolerance for a denominator context.

    The scale term follows the project rule:
        tol = max(abs_tol, rel_tol * P95(abs(denominator)))
    using only finite denominator values.
    """

    abs_tol_f = _validate_non_negative_finite(abs_tol, name="abs_tol")
    rel_tol_f = _validate_non_negative_finite(rel_tol, name="rel_tol")

    denom = np.asarray(denominator, dtype=float)
    finite_abs = np.abs(denom[np.isfinite(denom)])
    scale = float(np.percentile(finite_abs, 95.0)) if finite_abs.size else 0.0
    return max(abs_tol_f, rel_tol_f * scale)


def safe_divide(
    numerator: Any,
    denominator: Any,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> np.ndarray:
    """Divide with `NaN` output when the denominator is invalid or near zero."""

    numer = np.asarray(numerator, dtype=float)
    denom = np.asarray(denominator, dtype=float)
    numer_b, denom_b = np.broadcast_arrays(numer, denom)

    out = np.full(numer_b.shape, np.nan, dtype=float)
    tol = dynamic_denominator_tolerance(
        denom_b,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    valid = np.isfinite(numer_b) & np.isfinite(denom_b) & (np.abs(denom_b) > tol)
    if np.any(valid):
        out[valid] = numer_b[valid] / denom_b[valid]
    return out
