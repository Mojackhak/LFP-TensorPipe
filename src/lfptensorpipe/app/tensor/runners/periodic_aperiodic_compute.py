"""Compute helpers for the periodic/aperiodic tensor runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import service as svc
from .periodic_aperiodic_models import (
    PeriodicAperiodicOptions,
    PeriodicAperiodicOutputs,
    PeriodicAperiodicPreparedInput,
)


def _normalize_power_tensor(power: Any) -> np.ndarray:
    power_tensor = np.asarray(power, dtype=float)
    if power_tensor.ndim == 3:
        power_tensor = power_tensor[None, ...]
    if power_tensor.ndim != 4:
        raise ValueError(f"Unexpected TFR tensor shape: {power_tensor.shape}")
    return power_tensor


def _axes_from_metadata(
    metadata: Any, power_tensor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    axes = metadata.get("axes", {}) if isinstance(metadata, dict) else {}
    freqs_meta = np.asarray(axes.get("freq"), dtype=float)
    if freqs_meta.ndim != 1 or freqs_meta.size != power_tensor.shape[2]:
        raise ValueError("TFR metadata is missing/invalid frequency axis.")
    times_meta = np.asarray(axes.get("time"), dtype=float)
    channel_meta = [
        str(item) for item in np.asarray(axes.get("channel"), dtype=object).tolist()
    ]
    return freqs_meta, times_meta, channel_meta


def _minimum_positive_finite_value(power_tensor: np.ndarray) -> float:
    positive_values = np.asarray(power_tensor, dtype=float)
    positive_values = positive_values[
        np.isfinite(positive_values) & (positive_values > 0)
    ]
    if positive_values.size < 1:
        raise ValueError(
            "Periodic/APeriodic interpolation clip requires at least one positive finite TFR value."
        )
    return float(np.min(positive_values))


def _run_tfr_grid(
    prepared: PeriodicAperiodicPreparedInput,
    options: PeriodicAperiodicOptions,
    *,
    tfr_grid_fn=None,
    interpolate_freq_tensor_fn=None,
    smooth_axis_fn=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if interpolate_freq_tensor_fn is None:
        from lfptensorpipe.lfp.interp.freq import (
            interpolate_tensor_with_metadata_transformed as interpolate_freq_tensor,
        )
    else:
        interpolate_freq_tensor = interpolate_freq_tensor_fn
    if smooth_axis_fn is None:
        from lfptensorpipe.lfp.smooth.smooth import smooth_axis
    else:
        smooth_axis = smooth_axis_fn
    if tfr_grid_fn is None:
        from lfptensorpipe.lfp.tfr.grid import grid as tfr_grid
    else:
        tfr_grid = tfr_grid_fn

    power, metadata = tfr_grid(
        prepared.raw,
        method=prepared.method_norm,
        freqs=prepared.freqs_compute,
        picks=prepared.picks,
        time_resolution_s=float(options.time_resolution_s),
        hop_s=float(options.hop_s),
        min_cycles=options.min_cycles,
        max_cycles=options.max_cycles,
        time_bandwidth=float(options.time_bandwidth),
        n_jobs=int(options.n_jobs),
    )
    power_tensor = _normalize_power_tensor(power)

    if prepared.interpolation_applied:
        positive_floor = _minimum_positive_finite_value(power_tensor)
        power_tensor, metadata = interpolate_freq_tensor(
            power_tensor,
            metadata,
            freqs_out=prepared.freqs_model,
            axis=-2,
            method="linear",
            transform_mode="dB",
        )
        power_tensor = np.asarray(power_tensor, dtype=float)
        power_tensor = np.clip(power_tensor, a_min=positive_floor, a_max=None)

    if bool(options.freq_smooth_enabled):
        power_tensor = np.asarray(
            smooth_axis(
                power_tensor,
                method="gaussian",
                axis=-2,
                sigma=(
                    float(options.freq_smooth_sigma)
                    if options.freq_smooth_sigma is not None
                    else 1.5
                ),
                transform_mode="dB",
                nan_policy="omit",
            ),
            dtype=float,
        )

    if bool(options.time_smooth_enabled):
        kernel = (
            int(options.time_smooth_kernel_size)
            if options.time_smooth_kernel_size is not None
            else max(
                1,
                int(round(float(options.time_resolution_s) / float(options.hop_s))),
            )
        )
        power_tensor = np.asarray(
            smooth_axis(
                power_tensor,
                kernel_size=max(1, kernel),
                method="median",
                axis=-1,
                transform_mode="dB",
                nan_policy="omit",
            ),
            dtype=float,
        )

    return power_tensor, dict(metadata)


def _run_decomposition(
    power_tensor: np.ndarray,
    metadata: dict[str, Any],
    prepared: PeriodicAperiodicPreparedInput,
    options: PeriodicAperiodicOptions,
    report_dir: Path,
    *,
    decompose_fn=None,
    make_gof_rsquared_masker_fn=None,
) -> PeriodicAperiodicOutputs:
    from lfptensorpipe.lfp.interp.freq import diff_freq_grids

    if decompose_fn is None or make_gof_rsquared_masker_fn is None:
        from lfptensorpipe.lfp.tfr.decompose import (
            decompose as tfr_decompose,
            make_gof_rsquared_masker,
        )

        if decompose_fn is None:
            decompose_fn = tfr_decompose
        if make_gof_rsquared_masker_fn is None:
            make_gof_rsquared_masker_fn = make_gof_rsquared_masker

    freqs_meta, times_meta, channel_meta = _axes_from_metadata(metadata, power_tensor)
    _, tfr_periodic, _, params_tensor, params_meta = decompose_fn(
        power_tensor,
        freqs_meta,
        times=times_meta,
        ch_names=channel_meta,
        freq_range=(float(prepared.spec_low), float(prepared.spec_high)),
        aperiodic_mode=str(options.aperiodic_mode),
        peak_width_limits=options.peak_width_limits_hz,
        max_n_peaks=options.max_n_peaks,
        min_peak_height=float(options.min_peak_height),
        peak_threshold=float(options.peak_threshold),
        n_jobs=int(options.n_jobs),
        report_dir=report_dir,
        verbose=False,
    )
    tensor = np.asarray(tfr_periodic, dtype=float)
    if tensor.ndim != 4:
        raise ValueError(f"Unexpected periodic tensor shape: {tensor.shape}")
    params_tensor_arr = np.asarray(params_tensor, dtype=float)
    if params_tensor_arr.ndim != 4:
        raise ValueError(
            f"Unexpected periodic params tensor shape: {params_tensor_arr.shape}"
        )
    params_meta_dict = (
        dict(params_meta)
        if isinstance(params_meta, dict)
        else {"axes": {"shape": tuple(params_tensor_arr.shape)}}
    )

    try:
        gof_masker = make_gof_rsquared_masker_fn(
            params_tensor_arr,
            params_meta_dict,
            threshold=float(options.fit_qc_threshold),
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Periodic/APeriodic QC mask failed: missing/invalid gof_rsquared in params tensor."
        ) from exc
    tensor = np.asarray(gof_masker(tensor), dtype=float)
    params_tensor_arr = np.asarray(gof_masker(params_tensor_arr), dtype=float)

    final_subset = diff_freq_grids(prepared.freqs_final, freqs_meta)
    if not bool(final_subset.get("is_subset")):
        raise ValueError(
            "Final frequency grid is not aligned with SpecParam/model frequency grid."
        )
    keep_idx = np.asarray(final_subset.get("keep_out_idx"), dtype=int).ravel()
    if keep_idx.size < 2:
        raise ValueError("Final frequency grid requires at least two bins.")
    tensor = tensor[:, :, keep_idx, :]
    if isinstance(metadata, dict):
        axes_out = dict(metadata.get("axes", {}) or {})
        axes_out["freq"] = np.asarray(prepared.freqs_final, dtype=float)
        axes_out["shape"] = tuple(tensor.shape)
        metadata["axes"] = axes_out

    if options.mask_edge_effects:
        tensor, metadata, params_tensor_arr, params_meta_dict = _apply_edge_masks(
            prepared,
            options,
            tensor,
            metadata,
            params_tensor_arr,
            params_meta_dict,
        )

    return PeriodicAperiodicOutputs(
        tensor=np.asarray(tensor, dtype=float),
        metadata=dict(metadata),
        params_tensor=np.asarray(params_tensor_arr, dtype=float),
        params_meta=dict(params_meta_dict),
    )


def _apply_edge_masks(
    prepared: PeriodicAperiodicPreparedInput,
    options: PeriodicAperiodicOptions,
    tensor: np.ndarray,
    metadata: dict[str, Any],
    params_tensor: np.ndarray,
    params_meta: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:
    freq_axis = np.asarray(
        (metadata.get("axes", {}) or {}).get("freq", []), dtype=float
    ).ravel()
    if freq_axis.size != tensor.shape[2]:
        raise ValueError(
            "Periodic/APeriodic edge mask failed: metadata frequency axis does not match tensor."
        )
    radii = svc._compute_mask_radii_seconds(
        freq_axis,
        method=prepared.method_norm,
        time_resolution_s=float(options.time_resolution_s),
        min_cycles=options.min_cycles,
        max_cycles=options.max_cycles,
    )
    tensor, metadata = svc._apply_dynamic_edge_mask_strict(
        raw=prepared.raw,
        tensor=tensor,
        metadata=metadata,
        metric_label="Periodic/APeriodic",
        freqs_lookup=[float(item) for item in freq_axis.tolist()],
        radii_s=[float(item) for item in radii.tolist()],
    )

    params_freq_axis = [
        str(item)
        for item in np.asarray(
            (params_meta.get("axes", {}) or {}).get("freq", []),
            dtype=object,
        ).ravel()
    ]
    if len(params_freq_axis) != int(params_tensor.shape[2]):
        raise ValueError(
            "Periodic/APeriodic params edge mask failed: params metadata frequency axis does not match tensor."
        )
    max_radius = float(np.max(radii)) if radii.size > 0 else 0.0
    params_tensor, params_meta = svc._apply_dynamic_edge_mask_strict(
        raw=prepared.raw,
        tensor=params_tensor,
        metadata=params_meta,
        metric_label="Periodic/APeriodic params",
        freqs_lookup=params_freq_axis,
        radii_s=[max_radius for _ in params_freq_axis],
    )
    return (
        np.asarray(tensor, dtype=float),
        dict(metadata),
        np.asarray(params_tensor, dtype=float),
        dict(params_meta),
    )


def compute_periodic_aperiodic_outputs(
    prepared: PeriodicAperiodicPreparedInput,
    options: PeriodicAperiodicOptions,
    report_dir: Path,
    *,
    tfr_grid_fn=None,
    interpolate_freq_tensor_fn=None,
    smooth_axis_fn=None,
    decompose_fn=None,
    make_gof_rsquared_masker_fn=None,
) -> PeriodicAperiodicOutputs:
    power_tensor, metadata = _run_tfr_grid(
        prepared,
        options,
        tfr_grid_fn=tfr_grid_fn,
        interpolate_freq_tensor_fn=interpolate_freq_tensor_fn,
        smooth_axis_fn=smooth_axis_fn,
    )
    return _run_decomposition(
        power_tensor,
        metadata,
        prepared,
        options,
        report_dir,
        decompose_fn=decompose_fn,
        make_gof_rsquared_masker_fn=make_gof_rsquared_masker_fn,
    )
