"""Axis-wise smoothing utilities for LFP feature tensors.

This module provides a focused API to smooth a tensor along one chosen axis
while optionally applying an element-wise transform prior to smoothing.

Why this exists
--------------
Many pipelines compute time-frequency features using grids that are denser than
the effective support of the estimator. The resulting tensors can be noisy
along either time or frequency, which destabilizes downstream steps such as
spectral parameterization.

Smoothing is therefore treated as an explicit post-processing step:
  1) (optional) transform -> 2) smooth -> 3) (optional) inverse transform

All code comments and docstrings are in English (per project rules).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from scipy.ndimage import convolve1d, gaussian_filter1d, median_filter

from ...utils.transforms import (
    TransformMode,
    apply_inverse_transform_array,
    apply_transform_array,
)

SmoothMethod = Literal["median", "mean", "gaussian"]
NanPolicy = Literal["omit", "propagate"]
PadMode = Literal["nearest", "reflect", "mirror", "constant", "wrap"]


def _normalize_kernel_size(kernel_size: int, *, enforce_odd: bool = True) -> int:
    k = int(kernel_size)
    if k < 1:
        raise ValueError("kernel_size must be >= 1")
    if enforce_odd and (k % 2 == 0):
        k += 1
    return k


def _axis_to_positive(axis: int, ndim: int) -> int:
    ax = int(axis)
    if ax < 0:
        ax += int(ndim)
    if ax < 0 or ax >= int(ndim):
        raise ValueError("axis is out of bounds for tensor")
    return ax


def _validate_sigma(sigma: float | None) -> float:
    if sigma is None:
        raise ValueError("sigma must be provided when method='gaussian'")
    sigma_f = float(sigma)
    if not np.isfinite(sigma_f) or sigma_f <= 0.0:
        raise ValueError("sigma must be a positive finite number")
    return sigma_f


def _validate_truncate(truncate: float | None) -> float | None:
    if truncate is None:
        return None
    truncate_f = float(truncate)
    if not np.isfinite(truncate_f) or truncate_f <= 0.0:
        raise ValueError("truncate must be a positive finite number")
    return truncate_f


def smooth_axis(
    tensor: np.ndarray,
    *,
    kernel_size: int | None = None,
    method: SmoothMethod = "median",
    axis: int = -1,
    sigma: float | None = None,
    truncate: float | None = None,
    transform_mode: TransformMode | None = None,
    return_in_original_domain: bool = True,
    pad_mode: PadMode = "nearest",
    cval: float = 0.0,
    nan_policy: NanPolicy = "omit",
    enforce_odd_kernel: bool = True,
) -> np.ndarray:
    """Smooth a tensor along one axis.

    Args:
        tensor: Input tensor (commonly 4D: epochs x channels x freqs x time).
        kernel_size: Smoothing window length in samples along `axis` for
            `method='median'` and `method='mean'`.
        method: `median`, `mean`, or `gaussian`.
        axis: Axis index to smooth along.
        sigma: Gaussian sigma in bins along `axis` for `method='gaussian'`.
        truncate: Optional Gaussian support truncation in sigma units.
        transform_mode: Optional element-wise transform to apply before
            smoothing (for example `dB` or `fisherz`).
        return_in_original_domain: If True and `transform_mode` is not None,
            apply the inverse transform after smoothing.
        pad_mode: Boundary handling mode passed to SciPy.
        cval: Constant value used when `pad_mode == 'constant'`.
        nan_policy:
            - `omit`: ignore NaNs using weighted normalization for `mean` and
              `gaussian`
            - `propagate`: keep SciPy default NaN propagation behavior
        enforce_odd_kernel: If True, even kernel sizes are rounded up to the
            next odd integer for methods that use `kernel_size`.

    Returns:
        Smoothed tensor with the same shape as the input.
    """
    x_in = np.asarray(tensor)
    if x_in.ndim < 1:
        raise ValueError("tensor must have at least 1 dimension")

    ax = _axis_to_positive(axis, x_in.ndim)
    method_l = str(method).lower()

    if method_l in {"median", "mean"}:
        if kernel_size is None:
            raise ValueError(
                "kernel_size must be provided for method='median' and method='mean'"
            )
        k = _normalize_kernel_size(kernel_size, enforce_odd=enforce_odd_kernel)
        if k == 1:
            return np.array(x_in, copy=True)
    elif method_l == "gaussian":
        sigma_f = _validate_sigma(sigma)
        truncate_f = _validate_truncate(truncate)
    else:
        raise ValueError("method must be 'median', 'mean', or 'gaussian'")

    out_dtype = x_in.dtype if np.issubdtype(x_in.dtype, np.floating) else np.float64
    x = np.asarray(x_in, dtype=float)

    if transform_mode is not None:
        x = apply_transform_array(x, mode=transform_mode)

    if method_l == "median":
        size = [1] * x.ndim
        size[ax] = k
        x_smooth = median_filter(x, size=tuple(size), mode=pad_mode, cval=float(cval))

    elif method_l == "mean":
        if nan_policy == "omit":
            mask = np.isfinite(x)
            x0 = np.where(mask, x, 0.0)
            w = np.ones(k, dtype=float) / float(k)
            num = convolve1d(x0, w, axis=ax, mode=pad_mode, cval=float(cval))
            den = convolve1d(
                mask.astype(float),
                w,
                axis=ax,
                mode=pad_mode,
                cval=float(cval),
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                x_smooth = num / den
            x_smooth = np.where(den > 0, x_smooth, np.nan)
        elif nan_policy == "propagate":
            w = np.ones(k, dtype=float) / float(k)
            x_smooth = convolve1d(x, w, axis=ax, mode=pad_mode, cval=float(cval))
        else:
            raise ValueError("nan_policy must be 'omit' or 'propagate'")

    else:
        gaussian_kwargs = {
            "sigma": sigma_f,
            "axis": ax,
            "mode": pad_mode,
            "cval": float(cval),
        }
        if truncate_f is not None:
            gaussian_kwargs["truncate"] = truncate_f

        if nan_policy == "omit":
            mask = np.isfinite(x)
            x0 = np.where(mask, x, 0.0)
            num = gaussian_filter1d(x0, **gaussian_kwargs)
            den = gaussian_filter1d(mask.astype(float), **gaussian_kwargs)
            with np.errstate(invalid="ignore", divide="ignore"):
                x_smooth = num / den
            x_smooth = np.where(den > 0, x_smooth, np.nan)
        elif nan_policy == "propagate":
            x_smooth = gaussian_filter1d(x, **gaussian_kwargs)
        else:
            raise ValueError("nan_policy must be 'omit' or 'propagate'")

    if transform_mode is not None and return_in_original_domain:
        x_smooth = apply_inverse_transform_array(x_smooth, mode=transform_mode)

    return np.asarray(x_smooth, dtype=out_dtype)


__all__ = ["smooth_axis"]
