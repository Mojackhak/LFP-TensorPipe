"""Core tensor helper kernels extracted from app-layer orchestration.

These helpers implement reusable numeric/masking/frequency utilities used by
Build Tensor metric runners.
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np

ESTIMATOR_MASK_SUPPORT_SEMANTICS = "full_estimator_input_and_notch_donor_support"


def build_annotation_skip_time_mask(
    raw: Any,
    *,
    times_s: np.ndarray,
    radius_s: float,
    output_channels: Sequence[Sequence[str]] | None = None,
) -> np.ndarray:
    """Return centers where BAD/EDGE masks every requested output."""
    from lfptensorpipe.lfp.mask.annotations import (
        output_time_mask_by_annotations,
        time_mask_by_annotations,
    )

    if output_channels is None:
        skip_mask, _ = time_mask_by_annotations(
            raw,
            times_s=np.asarray(times_s, dtype=float),
            keep=("bad", "edge"),
            mode="substring",
            pad_s=float(radius_s),
            clip_to_raw=True,
            require_match=False,
        )
    else:
        output_mask, _ = output_time_mask_by_annotations(
            raw,
            times_s=np.asarray(times_s, dtype=float),
            output_channels=output_channels,
            keep=("bad", "edge"),
            mode="substring",
            pad_s=float(radius_s),
            clip_to_raw=True,
            require_match=False,
        )
        skip_mask = (
            np.all(output_mask, axis=0)
            if output_mask.shape[0] > 0
            else np.zeros(np.asarray(times_s).shape, dtype=bool)
        )
    return np.asarray(skip_mask, dtype=bool)


def cycles_from_time_resolution(
    freqs_hz: np.ndarray,
    *,
    method: str,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
    mt_time_bandwidth_product: float = 4.0,
    mt_min_cycles: float = 3.0,
) -> np.ndarray:
    if method == "morlet":
        cycles = (
            np.asarray(freqs_hz, dtype=float)
            * float(time_resolution_s)
            * np.pi
            / np.sqrt(2.0 * np.log(2.0))
        )
        if min_cycles is not None:
            cycles = np.maximum(cycles, float(min_cycles))
        if max_cycles is not None:
            cycles = np.minimum(cycles, float(max_cycles))
    else:
        from lfptensorpipe.lfp.common import multitaper_fixed_p_parameters

        cycles, _, _, _ = multitaper_fixed_p_parameters(
            np.asarray(freqs_hz, dtype=float),
            time_resolution_s=float(time_resolution_s),
            mt_time_bandwidth_product=float(mt_time_bandwidth_product),
            mt_min_cycles=float(mt_min_cycles),
        )
    return np.asarray(cycles, dtype=float)


def compute_mask_radii_seconds(
    freqs_hz: np.ndarray,
    *,
    method: str,
    time_resolution_s: float,
    min_cycles: float | None,
    max_cycles: float | None,
    mt_time_bandwidth_product: float = 4.0,
    mt_min_cycles: float = 3.0,
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
        mt_time_bandwidth_product=float(mt_time_bandwidth_product),
        mt_min_cycles=float(mt_min_cycles),
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
    warn_fully_masked: bool = True,
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
    axes = masked_meta.get("axes", {})
    freq_values = axes.get("freq", []) if isinstance(axes, dict) else []
    if masked_tensor.ndim >= 2 and masked_tensor.shape[-2] == len(freq_values):
        reduction_axes = tuple(
            axis for axis in range(masked_tensor.ndim) if axis != masked_tensor.ndim - 2
        )
        fully_masked = ~np.any(np.isfinite(masked_tensor), axis=reduction_axes)
        if warn_fully_masked and bool(np.any(fully_masked)):
            warnings.warn(
                f"{metric_label} contains {int(np.sum(fully_masked))} frequency or band "
                "entries with no usable values after BAD/EDGE masking; they remain NaN.",
                UserWarning,
                stacklevel=2,
            )
    return masked_tensor, masked_meta


def connectivity_consumed_support_radii_seconds(
    metadata: dict[str, Any],
    *,
    sfreq: float,
    n_freqs: int,
) -> np.ndarray:
    """Map Connectivity group geometry to consumed raw-support radii."""
    sfreq_use = float(sfreq)
    n_freqs_use = int(n_freqs)
    if not np.isfinite(sfreq_use) or sfreq_use <= 0.0:
        raise ValueError(
            "Connectivity support mapping requires a positive sampling rate."
        )
    if n_freqs_use <= 0:
        raise ValueError(
            "Connectivity support mapping requires a non-empty frequency axis."
        )
    params = metadata.get("params") if isinstance(metadata, dict) else None
    groups = params.get("window_group_counts") if isinstance(params, dict) else None
    if not isinstance(groups, list) or not groups:
        raise ValueError("Connectivity metadata is missing window_group_counts.")

    radii = np.full(n_freqs_use, np.nan, dtype=float)
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError(
                "Connectivity window_group_counts contains an invalid group."
            )
        indices = group.get("output_frequency_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("Connectivity support group has no output frequencies.")
        try:
            analysis_radius = int(group["analysis_radius_samples"])
            required_padding = int(group["required_padding_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Connectivity support group is missing valid analysis/padding geometry."
            ) from exc
        if analysis_radius < 0 or required_padding < 0:
            raise ValueError("Connectivity support geometry cannot be negative.")
        radius_s = (analysis_radius + required_padding) / sfreq_use
        for raw_index in indices:
            index = int(raw_index)
            if index < 0 or index >= n_freqs_use:
                raise ValueError(
                    "Connectivity support group frequency index is out of range."
                )
            if np.isfinite(radii[index]):
                raise ValueError(
                    "Connectivity support groups contain a duplicate frequency index."
                )
            radii[index] = radius_s
    if np.any(~np.isfinite(radii)):
        raise ValueError("Connectivity support groups do not cover every frequency.")
    return radii


def interpolated_support_radii_seconds(
    freqs_compute: np.ndarray,
    radii_compute_s: np.ndarray,
    freqs_full: np.ndarray,
) -> np.ndarray:
    """Propagate retained-frequency support to notch-reconstructed cells."""
    compute = np.asarray(freqs_compute, dtype=float).ravel()
    radii = np.asarray(radii_compute_s, dtype=float).ravel()
    full = np.asarray(freqs_full, dtype=float).ravel()
    if compute.size < 2 or full.size < compute.size or radii.size != compute.size:
        raise ValueError("Invalid compute/full frequency support mapping.")
    if (
        np.any(~np.isfinite(compute))
        or np.any(~np.isfinite(full))
        or np.any(~np.isfinite(radii))
        or np.any(radii < 0.0)
        or np.any(np.diff(compute) <= 0.0)
        or np.any(np.diff(full) <= 0.0)
    ):
        raise ValueError("Frequency support mapping requires finite ordered values.")

    compute_positions: list[int] = []
    for frequency in compute:
        matches = np.flatnonzero(np.isclose(full, frequency, rtol=0.0, atol=1e-9))
        if matches.size != 1:
            raise ValueError(
                "Compute frequencies must be a unique subset of the full grid."
            )
        compute_positions.append(int(matches[0]))
    if any(
        left >= right for left, right in zip(compute_positions, compute_positions[1:])
    ):
        raise ValueError("Compute frequency order does not match the full grid.")

    result = np.full(full.size, float(np.max(radii)), dtype=float)
    for compute_index, full_index in enumerate(compute_positions):
        result[full_index] = radii[compute_index]
    for left_index, right_index in zip(compute_positions, compute_positions[1:]):
        if right_index - left_index <= 1:
            continue
        donor_radius = max(result[left_index], result[right_index])
        result[left_index + 1 : right_index] = donor_radius
    return result


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

    if method == "multitaper":
        windows = np.asarray(
            params.get("mt_effective_window_s", []), dtype=float
        ).ravel()
        if windows.size != len(band_names):
            raise ValueError(
                "PSI Multitaper metadata window count does not match the band axis."
            )
        if np.any(~np.isfinite(windows)) or np.any(windows <= 0.0):
            raise ValueError("PSI Multitaper metadata has invalid effective windows.")
        return band_names, [float(item) / 2.0 for item in windows.tolist()]

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


def expand_notch_radii(
    notch_radii: Any,
    n_notches: int,
    *,
    legacy_mismatched_list_broadcast: bool = False,
) -> tuple[float, ...]:
    if n_notches <= 0:
        return ()
    radii = parse_positive_float_tuple(notch_radii)
    if not radii:
        return tuple(2.0 for _ in range(n_notches))
    if len(radii) == 1:
        return tuple(float(radii[0]) for _ in range(n_notches))
    if len(radii) == n_notches:
        return tuple(float(item) for item in radii)
    if legacy_mismatched_list_broadcast:
        return tuple(float(radii[0]) for _ in range(n_notches))
    raise ValueError(
        "notch_radii must contain one value or match the number of notches."
    )


def compute_notch_intervals(
    *,
    low_freq: float,
    high_freq: float,
    notches: tuple[float, ...],
    notch_radii: tuple[float, ...],
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for notch, radius in zip(notches, notch_radii, strict=False):
        lo = float(notch) - float(radius)
        hi = float(notch) + float(radius)
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


def build_frequency_grid(
    low_freq: float, high_freq: float, step_hz: float
) -> np.ndarray:
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
