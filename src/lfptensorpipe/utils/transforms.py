"""Element-wise numeric transforms for tensors and arrays.

This module centralizes small, reusable transform primitives used across the
project (stats, visualization, and tensor post-processing).

The key design goal is to keep transforms:
  - **pure** (no side effects),
  - **vectorized** (NumPy arrays in / arrays out),
  - **explicitly invertible** when possible.

Notes
-----
These transforms are intentionally *element-wise*.
They do not do any alignment / broadcasting based on metadata.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np


TransformMode = Literal[
    "dB",
    "log",
    "fisherz",
    "fisherz_sqrt",
    "logit",
    "asinh",
    "none",
    None,
]


def get_transform_pair(
    mode: TransformMode,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return (forward, inverse) callables for a transform mode.

    Notes
    -----
    - Forward transform first checks the mathematical domain.
      Values outside the domain are set to NaN.
    """
    if mode is None or mode == "none":
        return (lambda x: x, lambda x: x)

    if mode == "dB":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            valid = np.isfinite(x) & (x > 0.0)
            if np.any(valid):
                out[valid] = 10.0 * np.log10(x[valid])
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.power(10.0, y / 10.0)

        return forward, inverse

    if mode == "log":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            valid = np.isfinite(x) & (x > 0.0)
            if np.any(valid):
                out[valid] = np.log(x[valid])
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.exp(y)

        return forward, inverse

    if mode == "fisherz":
        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            valid = np.isfinite(x) & (x > -1.0) & (x < 1.0)
            if np.any(valid):
                out[valid] = np.arctanh(x[valid])
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.tanh(y)

        return forward, inverse

    if mode == "fisherz_sqrt":
        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            # Domain: 0 <= x < 1
            valid = np.isfinite(x) & (x >= 0.0) & (x < 1.0)
            if np.any(valid):
                out[valid] = np.arctanh(np.sqrt(x[valid]))
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.square(np.tanh(y))

        return forward, inverse

    if mode == "logit":
        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            # Domain: 0 < x < 1
            valid = np.isfinite(x) & (x > 0.0) & (x < 1.0)
            if np.any(valid):
                out[valid] = np.log(x[valid] / (1.0 - x[valid]))
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            # Stable sigmoid (kept as-is; may warn on extreme values but returns finite 0/1)
            return 1.0 / (1.0 + np.exp(-y))

        return forward, inverse

    if mode == "asinh":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return np.arcsinh(x)

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.sinh(y)

        return forward, inverse

    raise ValueError(f"Unsupported transform mode: {mode}")


def apply_transform_array(
    arr: np.ndarray,
    *,
    mode: TransformMode,
) -> np.ndarray:
    """Apply a forward transform to a NumPy array.

    This function always returns a floating array.
    """
    forward, _ = get_transform_pair(mode)
    x = np.asarray(arr, dtype=float)
    return forward(x)


def apply_inverse_transform_array(
    arr: np.ndarray,
    *,
    mode: TransformMode,
) -> np.ndarray:
    """Apply an inverse transform to a NumPy array.

    This function always returns a floating array.
    """
    _, inverse = get_transform_pair(mode)
    y = np.asarray(arr, dtype=float)
    return inverse(y)
