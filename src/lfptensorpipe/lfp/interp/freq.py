"""Frequency-axis interpolation for LFP tensors.

This module fills "holes" on a full frequency grid (freqs_out) using values
computed on a reduced grid (freqs_in), typically after removing notch-overlapping
frequency bins.

Design principles (practical and boring, as it should be):
- Missing bins are derived ONLY from (freqs_in vs freqs_out). No notch metadata
  is required for the interpolation itself.
- If freqs_in is a strict subset of freqs_out (within a tight tolerance), we can
  identify missing bins and optionally apply smoothing only on those missing bins.
- If freqs_in is NOT a subset of freqs_out, we either:
    (a) raise, or
    (b) fall back to a pure resampling interpolation onto freqs_out.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Literal

import numpy as np
from scipy.signal import savgol_filter
from lfptensorpipe.utils.transforms import TransformMode, get_transform_pair

InterpMethod = Literal["linear", "savgol"]
OnMismatch = Literal["raise", "interpolate"]


def _validate_freq_grid(freqs: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a 1D, strictly increasing, finite frequency grid."""
    f = np.asarray(freqs, dtype=float).ravel()
    if f.ndim != 1 or f.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 elements.")
    if np.any(~np.isfinite(f)):
        raise ValueError(f"{name} must be finite (no NaN/inf).")
    if not np.all(np.diff(f) > 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return f


def _default_match_tol_hz(freqs_out: np.ndarray) -> float:
    """A tight default tolerance for matching freqs_in to freqs_out (Hz)."""
    fout = _validate_freq_grid(freqs_out, name="freqs_out")
    step = float(np.median(np.diff(fout)))
    # Tight tolerance: enough for floating-point round-off, not enough for "half a bin".
    return max(1e-12, 1e-6 * step)


def _subset_indices_or_none(
    freqs_in: np.ndarray,
    freqs_out: np.ndarray,
    *,
    tol_hz: float | None = None,
) -> np.ndarray | None:
    """Return indices mapping freqs_in -> freqs_out if subset; else None.

    The mapping is 1-to-1 and strictly increasing.

    This is intentionally strict: if the grids do not align, any attempt to
    "copy back" or compute a set difference will be ambiguous and may create
    spurious bumps/steps.
    """
    fin = _validate_freq_grid(freqs_in, name="freqs_in")
    fout = _validate_freq_grid(freqs_out, name="freqs_out")

    if fin[0] < fout[0] or fin[-1] > fout[-1]:
        return None

    if tol_hz is None:
        tol_hz = _default_match_tol_hz(fout)
    tol_hz = float(tol_hz)
    if tol_hz <= 0:
        raise ValueError("tol_hz must be > 0.")

    idx = np.empty(fin.size, dtype=int)
    last = -1

    for i, f in enumerate(fin):
        j = int(np.searchsorted(fout, f, side="left"))

        # Candidate indices: j-1 and j (if in range)
        cand: list[int] = []
        if 0 <= j - 1 < fout.size:
            cand.append(j - 1)
        if 0 <= j < fout.size:
            cand.append(j)

        if not cand:
            return None

        k = min(cand, key=lambda kk: abs(float(fout[kk] - f)))
        if abs(float(fout[k] - f)) > tol_hz:
            return None

        if k <= last:
            return None

        idx[i] = k
        last = k

    return idx


def diff_freq_grids(
    freqs_in: np.ndarray,
    freqs_out: np.ndarray,
    *,
    tol_hz: float | None = None,
) -> Dict[str, Any]:
    """Compare two frequency grids and compute missing bins (freqs_out \\ freqs_in).

    Returns a dict with:
      - is_subset: bool
      - keep_out_idx: np.ndarray | None
      - missing_out_idx: np.ndarray | None
      - missing_freqs: np.ndarray | None
      - tol_hz: float
    """
    x_in = _validate_freq_grid(freqs_in, name="freqs_in")
    x_out = _validate_freq_grid(freqs_out, name="freqs_out")

    if tol_hz is None:
        tol_hz = _default_match_tol_hz(x_out)

    keep_out_idx = _subset_indices_or_none(x_in, x_out, tol_hz=tol_hz)
    if keep_out_idx is None:
        return {
            "is_subset": False,
            "keep_out_idx": None,
            "missing_out_idx": None,
            "missing_freqs": None,
            "tol_hz": float(tol_hz),
        }

    keep_mask = np.zeros(x_out.size, dtype=bool)
    keep_mask[keep_out_idx] = True
    missing_out_idx = np.flatnonzero(~keep_mask)
    return {
        "is_subset": True,
        "keep_out_idx": keep_out_idx,
        "missing_out_idx": missing_out_idx,
        "missing_freqs": x_out[missing_out_idx],
        "tol_hz": float(tol_hz),
    }


def _normalize_savgol_params(
    n: int,
    *,
    window_length: int,
    polyorder: int,
) -> tuple[int, int]:
    """Auto-adjust Savitzky-Golay parameters to valid values."""
    if n < 3:
        raise ValueError("Need at least 3 frequency bins for Savitzky-Golay.")

    w = int(window_length)
    p = int(polyorder)

    if p < 0:
        raise ValueError("savgol polyorder must be >= 0.")
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if (n % 2 == 1) else (n - 1)
    if w < (p + 2):
        p = max(0, w - 2)

    return w, p


def interpolate_freq_axis(
    tensor: np.ndarray,
    *,
    freqs_in: np.ndarray,
    freqs_out: np.ndarray,
    axis: int,
    method: InterpMethod = "linear",
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
    freq_match_tol_hz: float | None = None,
    on_mismatch: OnMismatch = "raise",
) -> np.ndarray:
    """Interpolate a tensor from freqs_in to freqs_out along one axis.

    Semantics:
      - method="linear": pure piecewise-linear interpolation onto freqs_out.
      - method="savgol": linear interpolation first; then apply Savitzky-Golay
        smoothing ONLY to bins that are missing in freqs_in (requires subset match).

    If freqs_in is NOT a strict subset of freqs_out:
      - on_mismatch="raise": raise ValueError.
      - on_mismatch="interpolate": fall back to pure linear interpolation. If
        method="savgol", we warn and still fall back to pure linear interpolation,
        because "missing-bin-only smoothing" is undefined without subset matching.
    """
    x_in = _validate_freq_grid(freqs_in, name="freqs_in")
    x_out = _validate_freq_grid(freqs_out, name="freqs_out")

    axis_i = int(axis)
    if axis_i < 0:
        axis_i += tensor.ndim
    if axis_i < 0 or axis_i >= tensor.ndim:
        raise ValueError("axis is out of bounds for tensor.")
    if tensor.shape[axis_i] != x_in.size:
        raise ValueError("tensor frequency axis length must match freqs_in.")

    info = diff_freq_grids(x_in, x_out, tol_hz=freq_match_tol_hz)
    is_subset = bool(info["is_subset"])

    if method == "savgol" and not is_subset:
        if on_mismatch == "raise":
            raise ValueError(
                "method='savgol' requires freqs_in to be a strict subset of freqs_out. "
                "Set on_mismatch='interpolate' to fall back to pure linear interpolation."
            )
        warnings.warn(
            "freqs_in is not a subset of freqs_out; falling back to pure linear interpolation "
            "(no missing-bin-only Savitzky-Golay smoothing).",
            category=RuntimeWarning,
            stacklevel=2,
        )
        method_eff: InterpMethod = "linear"
    else:
        method_eff = method

    # Bring frequency axis to last dimension and flatten all other dimensions.
    data = np.moveaxis(np.asarray(tensor), axis_i, -1)
    orig_shape = data.shape
    n_freq_in = orig_shape[-1]
    n_series = int(np.prod(orig_shape[:-1]))
    y_in = data.reshape(n_series, n_freq_in)

    # Interpolate each series independently.
    y_out = np.full((n_series, x_out.size), np.nan, dtype=float)
    for i in range(n_series):
        yi = y_in[i]
        finite = np.isfinite(yi)
        if int(np.sum(finite)) < 2:
            continue
        y_out[i] = np.interp(x_out, x_in[finite], yi[finite])

    if method_eff == "savgol":
        missing_idx = np.asarray(info["missing_out_idx"], dtype=int)
        if missing_idx.size > 0:
            w, p = _normalize_savgol_params(
                x_out.size, window_length=savgol_window, polyorder=savgol_polyorder
            )
            # Savgol cannot handle NaNs; only smooth fully-finite rows.
            finite_rows = np.all(np.isfinite(y_out), axis=1)
            if np.any(finite_rows):
                rows = np.flatnonzero(finite_rows)
                y_sg = savgol_filter(
                    y_out[rows],
                    window_length=w,
                    polyorder=p,
                    axis=-1,
                    mode="interp",
                )
                # Replace ONLY missing bins with the smoothed estimate.
                y_out[rows[:, None], missing_idx[None, :]] = y_sg[:, missing_idx]
    elif method_eff != "linear":
        raise ValueError("method must be 'linear' or 'savgol'.")

    out = y_out.reshape(*orig_shape[:-1], x_out.size)
    out = np.moveaxis(out, -1, axis_i)
    return out.astype(tensor.dtype, copy=False)


def interpolate_tensor_with_metadata(
    tensor: np.ndarray,
    metadata: Dict[str, Any],
    *,
    freqs_out: np.ndarray,
    axis: int,
    method: InterpMethod = "linear",
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
    freq_match_tol_hz: float | None = None,
    on_mismatch: OnMismatch = "raise",
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Interpolate tensor and update metadata frequency axis.

    Requirements:
      metadata["axes"]["freq"] must exist and correspond to tensor along `axis`.
    """
    if "axes" not in metadata or "freq" not in metadata["axes"]:
        raise ValueError("metadata must contain metadata['axes']['freq'].")

    freqs_in = np.asarray(metadata["axes"]["freq"], dtype=float)
    tensor_i = interpolate_freq_axis(
        tensor,
        freqs_in=freqs_in,
        freqs_out=freqs_out,
        axis=axis,
        method=method,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        freq_match_tol_hz=freq_match_tol_hz,
        on_mismatch=on_mismatch,
    )

    meta_i: Dict[str, Any] = copy.deepcopy(metadata)
    meta_i.setdefault("axes", {})
    meta_i["axes"]["freq"] = np.asarray(freqs_out, dtype=float)

    # Update stored shape, if present.
    if "shape" in meta_i["axes"] and meta_i["axes"]["shape"] is not None:
        shp = list(meta_i["axes"]["shape"])
        ax = int(axis)
        if ax < 0:
            ax += len(shp)
        if 0 <= ax < len(shp):
            shp[ax] = int(np.asarray(freqs_out).size)
            meta_i["axes"]["shape"] = tuple(int(x) for x in shp)

    # Store interpolation bookkeeping for downstream QC/debug.
    info = diff_freq_grids(freqs_in, freqs_out, tol_hz=freq_match_tol_hz)
    meta_i.setdefault("params", {})
    meta_i["params"]["freqs_compute"] = np.asarray(freqs_in, dtype=float)
    meta_i["params"]["freqs_full"] = np.asarray(freqs_out, dtype=float)
    meta_i["params"]["n_freqs_compute"] = int(np.asarray(freqs_in).size)
    meta_i["params"]["n_freqs_full"] = int(np.asarray(freqs_out).size)
    meta_i["params"]["interp_method"] = str(method)
    meta_i["params"]["freq_match_tol_hz"] = float(info["tol_hz"])
    meta_i["params"]["freqs_in_is_subset_of_freqs_out"] = bool(info["is_subset"])
    meta_i["params"]["missing_out_idx"] = (
        None if info["missing_out_idx"] is None else np.asarray(info["missing_out_idx"], dtype=int)
    )
    meta_i["params"]["missing_freqs_hz"] = (
        None if info["missing_freqs"] is None else np.asarray(info["missing_freqs"], dtype=float)
    )
    if method == "savgol":
        meta_i["params"]["savgol_window"] = int(savgol_window)
        meta_i["params"]["savgol_polyorder"] = int(savgol_polyorder)

    return tensor_i, meta_i


def interpolate_freq_axis_transformed(
    tensor: np.ndarray,
    *,
    freqs_in: np.ndarray,
    freqs_out: np.ndarray,
    axis: int,
    method: InterpMethod = "linear",
    transform_mode: TransformMode = "none",
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
    freq_match_tol_hz: float | None = None,
    on_mismatch: OnMismatch = "raise",
) -> np.ndarray:
    """Transform -> interpolate -> inverse-transform along the frequency axis.

    This is the safe way to do "interpolate in dB/log domain but return to linear".

    Args:
        tensor: N-D array.
        freqs_in: Frequency grid corresponding to tensor on `axis`.
        freqs_out: Target full frequency grid.
        axis: Frequency axis index (can be negative, e.g., -2).
        method: "linear" or "savgol" (see interpolate_freq_axis).
        transform_mode: Transform mode applied BEFORE interpolation and inverted AFTER.
        savgol_window/polyorder: used only if method="savgol".
        freq_match_tol_hz/on_mismatch: passed through to interpolate_freq_axis.

    Returns:
        Tensor interpolated onto freqs_out, returned in the ORIGINAL domain.
    """
    if transform_mode == "none":
        return interpolate_freq_axis(
            tensor,
            freqs_in=freqs_in,
            freqs_out=freqs_out,
            axis=axis,
            method=method,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
            freq_match_tol_hz=freq_match_tol_hz,
            on_mismatch=on_mismatch,
        )

    forward, inverse = get_transform_pair(transform_mode)

    # Transform in float to avoid dtype traps; preserve NaNs naturally.
    x = np.asarray(tensor, dtype=float)
    x_t = forward(x)

    y_t = interpolate_freq_axis(
        x_t,
        freqs_in=freqs_in,
        freqs_out=freqs_out,
        axis=axis,
        method=method,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        freq_match_tol_hz=freq_match_tol_hz,
        on_mismatch=on_mismatch,
    )

    y = inverse(np.asarray(y_t, dtype=float))

    # Return float in general; if you really want original dtype, cast outside explicitly.
    return y


def interpolate_tensor_with_metadata_transformed(
    tensor: np.ndarray,
    metadata: Dict[str, Any],
    *,
    freqs_out: np.ndarray,
    axis: int,
    method: InterpMethod = "linear",
    transform_mode: TransformMode = "none",
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
    freq_match_tol_hz: float | None = None,
    on_mismatch: OnMismatch = "raise",
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Transform -> interpolate -> inverse-transform, and update metadata freq axis.

    Requirements:
        metadata["axes"]["freq"] must exist.
    """
    if "axes" not in metadata or "freq" not in metadata["axes"]:
        raise ValueError("metadata must contain metadata['axes']['freq'].")

    if transform_mode is None:
        transform_mode = "none"
        
    freqs_in = np.asarray(metadata["axes"]["freq"], dtype=float)

    tensor_i = interpolate_freq_axis_transformed(
        tensor,
        freqs_in=freqs_in,
        freqs_out=freqs_out,
        axis=axis,
        method=method,
        transform_mode=transform_mode,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        freq_match_tol_hz=freq_match_tol_hz,
        on_mismatch=on_mismatch,
    )

    meta_i: Dict[str, Any] = copy.deepcopy(metadata)
    meta_i.setdefault("axes", {})
    meta_i["axes"]["freq"] = np.asarray(freqs_out, dtype=float)

    # Update stored shape if present.
    if "shape" in meta_i["axes"] and meta_i["axes"]["shape"] is not None:
        shp = list(meta_i["axes"]["shape"])
        ax = int(axis)
        if ax < 0:
            ax += len(shp)
        if 0 <= ax < len(shp):
            shp[ax] = int(np.asarray(freqs_out).size)
            meta_i["axes"]["shape"] = tuple(int(x) for x in shp)

    # Bookkeeping for QC/debug.
    info = diff_freq_grids(freqs_in, freqs_out, tol_hz=freq_match_tol_hz)
    meta_i.setdefault("params", {})
    meta_i["params"]["freqs_compute"] = np.asarray(freqs_in, dtype=float)
    meta_i["params"]["freqs_full"] = np.asarray(freqs_out, dtype=float)
    meta_i["params"]["interp_method"] = str(method)
    meta_i["params"]["freq_match_tol_hz"] = float(info["tol_hz"])
    meta_i["params"]["freqs_in_is_subset_of_freqs_out"] = bool(info["is_subset"])
    meta_i["params"]["missing_out_idx"] = (
        None if info["missing_out_idx"] is None else np.asarray(info["missing_out_idx"], dtype=int)
    )
    meta_i["params"]["missing_freqs_hz"] = (
        None if info["missing_freqs"] is None else np.asarray(info["missing_freqs"], dtype=float)
    )

    meta_i["params"]["interp_transform_mode"] = str(transform_mode)
    if method == "savgol":
        meta_i["params"]["savgol_window"] = int(savgol_window)
        meta_i["params"]["savgol_polyorder"] = int(savgol_polyorder)

    return tensor_i, meta_i
