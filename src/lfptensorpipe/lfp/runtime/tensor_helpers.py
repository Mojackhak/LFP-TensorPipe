"""Core tensor helper kernels extracted from app-layer orchestration.

These helpers implement reusable numeric/masking/frequency utilities used by
Build Tensor metric runners.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cycles_from_time_resolution(
    freqs_hz: np.ndarray,
    *,
    method: str,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
) -> np.ndarray:
    if method == "morlet":
        cycles = (
            np.asarray(freqs_hz, dtype=float)
            * float(time_resolution_s)
            * np.pi
            / np.sqrt(2.0 * np.log(2.0))
        )
    else:
        cycles = np.asarray(freqs_hz, dtype=float) * float(time_resolution_s)
    if min_cycles is not None:
        cycles = np.maximum(cycles, float(min_cycles))
    if max_cycles is not None:
        cycles = np.minimum(cycles, float(max_cycles))
    return np.asarray(cycles, dtype=float)


def compute_mask_radii_seconds(
    freqs_hz: np.ndarray,
    *,
    method: str,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
) -> np.ndarray:
    from lfptensorpipe.lfp.common import (
        morlet_mask_radius_time_s_from_freqs_n_cycles,
        multitaper_mask_radius_time_s_from_freqs_n_cycles,
    )

    cycles = cycles_from_time_resolution(
        freqs_hz,
        method=method,
        time_resolution_s=float(time_resolution_s),
        min_cycles=min_cycles,
        max_cycles=max_cycles,
    )
    if method == "morlet":
        return np.asarray(
            morlet_mask_radius_time_s_from_freqs_n_cycles(
                np.asarray(freqs_hz, dtype=float),
                n_cycles=np.asarray(cycles, dtype=float),
            ),
            dtype=float,
        )
    return np.asarray(
        multitaper_mask_radius_time_s_from_freqs_n_cycles(
            np.asarray(freqs_hz, dtype=float),
            n_cycles=np.asarray(cycles, dtype=float),
        ),
        dtype=float,
    )


def apply_dynamic_edge_mask_strict(
    *,
    raw: Any,
    tensor: np.ndarray,
    metadata: dict[str, Any],
    metric_label: str,
    freqs_lookup: list[float | str],
    radii_s: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    from lfptensorpipe.lfp.pipelines.masking import mask_tensor_dynamic

    _, masked = mask_tensor_dynamic(
        raw,
        {"tensor": np.asarray(tensor), "meta": dict(metadata)},
        drop=("bad", "edge"),
        freqs=freqs_lookup,
        time=radii_s,
        mode="substring",
        clip_to_raw=True,
        require_match=False,
    )
    masked_meta = masked.get("meta", {})
    if not isinstance(masked_meta, dict):
        raise ValueError(f"{metric_label} edge mask returned invalid metadata.")
    mask_info = masked_meta.get("mask", {})
    if not isinstance(mask_info, dict):
        raise ValueError(f"{metric_label} edge mask returned invalid mask info.")
    kind = str(mask_info.get("kind", "")).strip()
    if kind in {"drop_dynamic_time_only", "drop_dynamic_time_only_fallback"}:
        raise ValueError(
            f"{metric_label} edge mask failed: frequency-axis mapping fallback is not allowed."
        )
    if mask_info.get("freq_axis_mapped") is False:
        raise ValueError(
            f"{metric_label} edge mask failed: frequency-axis mapping is incomplete."
        )
    masked_tensor = np.asarray(masked.get("tensor"), dtype=float)
    if masked_tensor.shape != np.asarray(tensor).shape:
        raise ValueError(
            f"{metric_label} edge mask changed tensor shape unexpectedly: "
            f"{np.asarray(tensor).shape} -> {masked_tensor.shape}"
        )
    return masked_tensor, masked_meta


def psi_band_radii_seconds(
    *,
    metadata: dict[str, Any],
    method: str,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
) -> tuple[list[str], list[float]]:
    axes = metadata.get("axes", {}) if isinstance(metadata, dict) else {}
    params = metadata.get("params", {}) if isinstance(metadata, dict) else {}
    if not isinstance(axes, dict) or not isinstance(params, dict):
        raise ValueError("PSI metadata is missing required axes/params.")
    freq_axis = list(np.asarray(axes.get("freq", []), dtype=object).ravel())
    band_names = [str(item) for item in freq_axis]
    if not band_names:
        raise ValueError("PSI metadata has empty band axis.")

    union = params.get("bands_union_hz")
    if not isinstance(union, dict):
        raise ValueError("PSI metadata missing bands_union_hz.")

    if method == "morlet":
        cwt_freqs = params.get("cwt_freqs")
        cwt_cycles = params.get("cwt_n_cycles")
        if cwt_freqs is not None and cwt_cycles is not None:
            freqs_arr = np.asarray(cwt_freqs, dtype=float).ravel()
            cycles_arr = np.asarray(cwt_cycles, dtype=float).ravel()
            if freqs_arr.size == cycles_arr.size and freqs_arr.size > 0:
                from lfptensorpipe.lfp.common import (
                    morlet_mask_radius_time_s_from_freqs_n_cycles,
                )

                radii_dense = np.asarray(
                    morlet_mask_radius_time_s_from_freqs_n_cycles(
                        freqs_arr,
                        n_cycles=cycles_arr,
                    ),
                    dtype=float,
                )
                out: list[float] = []
                for name in band_names:
                    bounds = union.get(name)
                    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                        raise ValueError(
                            f"PSI metadata missing valid union edges for band: {name}"
                        )
                    lo = float(bounds[0])
                    hi = float(bounds[1])
                    mask = (freqs_arr >= lo) & (freqs_arr <= hi)
                    if bool(np.any(mask)):
                        out.append(float(np.max(radii_dense[mask])))
                    else:
                        center = np.asarray([(lo + hi) / 2.0], dtype=float)
                        out.append(
                            float(
                                compute_mask_radii_seconds(
                                    center,
                                    method=method,
                                    time_resolution_s=float(time_resolution_s),
                                    min_cycles=min_cycles,
                                    max_cycles=max_cycles,
                                )[0]
                            )
                        )
                return band_names, out

    out = []
    for name in band_names:
        bounds = union.get(name)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"PSI metadata missing valid union edges for band: {name}")
        center = np.asarray([(float(bounds[0]) + float(bounds[1])) / 2.0], dtype=float)
        out.append(
            float(
                compute_mask_radii_seconds(
                    center,
                    method=method,
                    time_resolution_s=float(time_resolution_s),
                    min_cycles=min_cycles,
                    max_cycles=max_cycles,
                )[0]
            )
        )
    return band_names, out


def parse_positive_float_tuple(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    parsed: list[float] = []
    for item in items:
        try:
            val = float(item)
        except Exception:
            continue
        if not np.isfinite(val) or val <= 0.0:
            continue
        parsed.append(val)
    return tuple(parsed)


def expand_notch_widths(notch_widths: Any, n_notches: int) -> tuple[float, ...]:
    if n_notches <= 0:
        return ()
    widths = parse_positive_float_tuple(notch_widths)
    if not widths:
        return tuple(2.0 for _ in range(n_notches))
    if len(widths) == 1:
        return tuple(float(widths[0]) for _ in range(n_notches))
    if len(widths) == n_notches:
        return tuple(float(item) for item in widths)
    return tuple(float(widths[0]) for _ in range(n_notches))


def compute_notch_intervals(
    *,
    low_freq: float,
    high_freq: float,
    notches: tuple[float, ...],
    notch_widths: tuple[float, ...],
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for notch, width in zip(notches, notch_widths, strict=False):
        lo = float(notch) - float(width)
        hi = float(notch) + float(width)
        if hi < low_freq or lo > high_freq:
            continue
        intervals.append((lo, hi))
    return intervals


def cut_frequency_grid_by_intervals(
    freqs: np.ndarray,
    intervals: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    removed_mask = np.zeros(freqs.shape[0], dtype=bool)
    for lo, hi in intervals:
        removed_mask |= (freqs >= float(lo)) & (freqs <= float(hi))
    kept = freqs[~removed_mask]
    return kept, removed_mask


def build_frequency_grid(low_freq: float, high_freq: float, step_hz: float) -> np.ndarray:
    if low_freq <= 0.0:
        raise ValueError("Low frequency must be > 0.")
    if high_freq <= low_freq:
        raise ValueError("High frequency must be greater than low frequency.")
    if step_hz <= 0.0:
        raise ValueError("Step must be > 0.")

    count = int(np.floor((high_freq - low_freq) / step_hz)) + 1
    if count < 2:
        raise ValueError("Frequency grid requires at least two bins.")
    freqs = low_freq + np.arange(count, dtype=float) * step_hz
    freqs = np.unique(np.round(freqs, decimals=6))
    if freqs.size < 2:
        raise ValueError("Frequency grid requires at least two unique bins.")
    return freqs
