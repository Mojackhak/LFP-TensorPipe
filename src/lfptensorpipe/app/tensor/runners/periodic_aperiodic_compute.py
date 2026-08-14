"""Compute helpers for the periodic/aperiodic tensor runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lfptensorpipe.lfp.mask.annotations import (
    has_channel_specific_mask_annotations,
    output_time_mask_by_annotations,
)
from lfptensorpipe.lfp.runtime.tensor_helpers import build_annotation_skip_time_mask
from lfptensorpipe.utils.transforms import (
    attach_transform_policy,
    get_transform_policy,
)

from .. import service as svc
from .periodic_aperiodic_models import (
    MASK_SUPPORT_SEMANTICS,
    NOTCH_INTERPOLATION_METHOD,
    NOTCH_INTERPOLATION_SEED,
    PeriodicAperiodicOptions,
    PeriodicAperiodicOutputs,
    PeriodicAperiodicPreparedInput,
    derive_notch_interpolation_seed,
)


def _effective_time_smoothing_kernel_size(
    options: PeriodicAperiodicOptions,
) -> int | None:
    if not bool(options.time_smooth_enabled):
        return None
    kernel = (
        int(options.time_smooth_kernel_size)
        if options.time_smooth_kernel_size is not None
        else max(
            1,
            int(round(float(options.time_resolution_s) / float(options.hop_s))),
        )
    )
    kernel = max(1, kernel)
    return kernel + 1 if kernel % 2 == 0 else kernel


def _effective_tfr_hop_seconds(
    metadata: dict[str, Any],
    prepared: PeriodicAperiodicPreparedInput,
) -> float:
    params = metadata.get("params", {}) if isinstance(metadata, dict) else {}
    if not isinstance(params, dict):
        params = {}
    hop_s_eff = params.get("hop_s_eff")
    try:
        hop_s_eff_float = float(hop_s_eff)
    except (TypeError, ValueError):
        hop_s_eff_float = float("nan")
    if np.isfinite(hop_s_eff_float) and hop_s_eff_float > 0.0:
        return hop_s_eff_float

    try:
        decim_eff = int(params.get("decim_eff"))
        sfreq = float(prepared.raw.info["sfreq"])
    except (KeyError, TypeError, ValueError):
        decim_eff = 0
        sfreq = float("nan")
    if decim_eff > 0 and np.isfinite(sfreq) and sfreq > 0.0:
        return float(decim_eff / sfreq)

    axes = metadata.get("axes", {}) if isinstance(metadata, dict) else {}
    times = np.asarray(axes.get("time", []), dtype=float).ravel()
    finite_diffs = np.diff(times)
    finite_diffs = finite_diffs[np.isfinite(finite_diffs) & (finite_diffs > 0.0)]
    if finite_diffs.size > 0:
        return float(np.median(finite_diffs))
    raise ValueError("Periodic/Aperiodic TFR metadata is missing its effective hop.")


def _periodic_annotation_support(
    prepared: PeriodicAperiodicPreparedInput,
    options: PeriodicAperiodicOptions,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    consumed_freqs = np.asarray(prepared.freqs_compute, dtype=float).ravel()
    if consumed_freqs.size < 1 or not np.all(np.isfinite(consumed_freqs)):
        raise ValueError(
            "Periodic/Aperiodic consumed TFR frequencies must be finite and non-empty."
        )
    hop_s_eff = _effective_tfr_hop_seconds(metadata, prepared)
    kernel_eff = _effective_time_smoothing_kernel_size(options)
    support: dict[str, Any] = {
        "mask_support_semantics": MASK_SUPPORT_SEMANTICS,
        "consumed_tfr_frequency_min_hz": float(np.min(consumed_freqs)),
        "consumed_tfr_frequency_max_hz": float(np.max(consumed_freqs)),
        "consumed_tfr_frequency_count": int(consumed_freqs.size),
        "time_smoothing_kernel_size_eff": kernel_eff,
        "hop_s_eff": float(hop_s_eff),
    }
    if not bool(options.mask_edge_effects):
        support.update(
            {
                "annotation_estimator_support_radius_s": None,
                "annotation_time_smoothing_radius_s": None,
                "annotation_skip_radius_s": None,
            }
        )
        return support

    estimator_radii = svc._compute_mask_radii_seconds(
        consumed_freqs,
        method=prepared.method_norm,
        time_resolution_s=float(options.time_resolution_s),
        min_cycles=options.min_cycles,
        max_cycles=options.max_cycles,
        mt_time_bandwidth_product=float(options.mt_time_bandwidth_product),
        mt_min_cycles=float(options.mt_min_cycles),
    )
    estimator_radius_s = float(np.max(estimator_radii))
    time_smoothing_radius_s = (
        float((kernel_eff // 2) * hop_s_eff) if kernel_eff is not None else 0.0
    )
    support.update(
        {
            "annotation_estimator_support_radius_s": estimator_radius_s,
            "annotation_time_smoothing_radius_s": time_smoothing_radius_s,
            "annotation_skip_radius_s": (estimator_radius_s + time_smoothing_radius_s),
        }
    )
    return support


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


def _impute_notch_intervals_local_residual(
    power_tensor: np.ndarray,
    freqs: np.ndarray,
    intervals: list[tuple[float, float]],
    *,
    seed: int = NOTCH_INTERPOLATION_SEED,
) -> np.ndarray:
    """Impute notch bins with reproducibly sampled local spectral residuals."""
    if not intervals:
        return np.asarray(power_tensor, dtype=float)

    frequency_axis = np.asarray(freqs, dtype=float).ravel()
    values = np.asarray(power_tensor, dtype=float)
    if values.shape[-2] != frequency_axis.size:
        raise ValueError(
            "Periodic/APeriodic tensor frequency axis does not match the model grid."
        )

    moved = np.moveaxis(values, -2, -1)
    original_shape = moved.shape
    rows = moved.reshape(-1, frequency_axis.size)
    supported = np.isfinite(rows) & (rows > 0.0)
    source_log_power = np.full(rows.shape, np.nan, dtype=float)
    source_log_power[supported] = np.log10(rows[supported])
    filled = np.where(supported, rows, np.nan)
    log_frequency = np.log10(frequency_axis)
    rng = np.random.default_rng(int(seed))

    def _endpoint_chord(
        left_index: int,
        right_index: int,
        target_indices: np.ndarray,
    ) -> np.ndarray:
        weights = (log_frequency[target_indices] - log_frequency[left_index]) / (
            log_frequency[right_index] - log_frequency[left_index]
        )
        return (
            source_log_power[:, left_index, None]
            + (
                source_log_power[:, right_index, None]
                - source_log_power[:, left_index, None]
            )
            * weights[None, :]
        )

    for low, high in intervals:
        interval_mask = (frequency_axis >= float(low)) & (frequency_axis <= float(high))
        interval_indices = np.flatnonzero(interval_mask)
        if interval_indices.size == 0:
            continue
        first_index = int(interval_indices[0])
        last_index = int(interval_indices[-1])
        target_left = first_index - 1
        target_right = last_index + 1
        if target_left < 0 or target_right >= frequency_axis.size:
            raise ValueError(
                "Notch imputation requires one model bin outside each interval."
            )

        donor_bounds = svc._clean_equal_width_notch_donor_bounds(
            frequency_axis,
            intervals,
            (low, high),
        )
        donor_residuals: list[np.ndarray] = []
        for donor_left, donor_right in donor_bounds:
            donor_indices = np.arange(donor_left + 1, donor_right, dtype=int)
            donor_residuals.append(
                source_log_power[:, donor_indices]
                - _endpoint_chord(donor_left, donor_right, donor_indices)
            )

        if not donor_residuals:
            raise ValueError(
                "Notch imputation requires a nearest equal-width non-notch donor "
                "segment on at least one side of the interval."
            )

        residual_stack = np.stack(donor_residuals, axis=1)
        time_count = int(original_shape[-2])
        series_count = int(rows.shape[0] // time_count)
        combination_count = 2 * len(donor_residuals)
        assignments = np.empty(rows.shape[0], dtype=int)
        for series_index in range(series_count):
            series_assignments = np.arange(time_count) % combination_count
            rng.shuffle(series_assignments)
            start = series_index * time_count
            assignments[start : start + time_count] = series_assignments
        donor_choice = assignments // 2
        selected_residual = residual_stack[
            np.arange(rows.shape[0]), donor_choice
        ].copy()
        reflect_rows = assignments % 2 == 1
        selected_residual[reflect_rows] = selected_residual[reflect_rows, ::-1]

        target_chord = _endpoint_chord(
            target_left,
            target_right,
            interval_indices,
        )
        filled[:, interval_indices] = np.power(
            10.0,
            target_chord + selected_residual,
        )
    restored = filled.reshape(original_shape)
    return np.moveaxis(restored, -1, -2)


def _run_tfr_grid(
    prepared: PeriodicAperiodicPreparedInput,
    options: PeriodicAperiodicOptions,
    *,
    tfr_grid_fn=None,
    interpolate_freq_tensor_fn=None,
    smooth_axis_fn=None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    if interpolate_freq_tensor_fn is None:
        from lfptensorpipe.lfp.interp.freq import (
            interpolate_tensor_with_metadata_policy as interpolate_freq_tensor,
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
        mt_time_bandwidth_product=float(options.mt_time_bandwidth_product),
        mt_min_cycles=float(options.mt_min_cycles),
        n_jobs=int(options.n_jobs),
    )
    power_tensor = _normalize_power_tensor(power)
    transform_policy = get_transform_policy(options.value_transform_mode)

    if prepared.interpolation_applied:
        interpolation_seed = derive_notch_interpolation_seed(options.context)
        positive_floor = _minimum_positive_finite_value(power_tensor)
        power_tensor, metadata = interpolate_freq_tensor(
            power_tensor,
            metadata,
            freqs_out=prepared.freqs_model,
            axis=-2,
            method="linear",
            policy=transform_policy,
        )
        power_tensor = np.asarray(power_tensor, dtype=float)
        power_tensor = np.clip(power_tensor, a_min=positive_floor, a_max=None)
        power_tensor = _impute_notch_intervals_local_residual(
            power_tensor,
            prepared.freqs_model,
            prepared.notch_intervals,
            seed=interpolation_seed,
        )
        metadata = dict(metadata or {})
        metadata["notch_interpolation"] = {
            "method": NOTCH_INTERPOLATION_METHOD,
            "domain": "log_frequency_log_power",
            "seed": interpolation_seed,
            "donor_scope": "same_spectrum_nearest_equal_width_non_notch",
            "assignment": "balanced_within_epoch_channel_over_time",
            "orientations": ["native", "reflected"],
            "intervals_hz": [
                [float(low), float(high)] for low, high in prepared.notch_intervals
            ],
        }

    unsupported_cells = ~np.isfinite(power_tensor) | (power_tensor <= 0.0)
    unsupported_spectrum_mask = np.any(unsupported_cells, axis=-2)
    unsupported_first_freq_index = np.argmax(unsupported_cells, axis=-2)

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
                transform_mode=transform_policy.mode,
                nan_policy="omit",
            ),
            dtype=float,
        )

    if bool(options.time_smooth_enabled):
        kernel = _effective_time_smoothing_kernel_size(options)
        if kernel is None:
            raise RuntimeError("Time smoothing kernel resolution failed.")
        power_tensor = np.asarray(
            smooth_axis(
                power_tensor,
                kernel_size=kernel,
                method="median",
                axis=-1,
                transform_mode=transform_policy.mode,
                nan_policy="omit",
            ),
            dtype=float,
        )

    return (
        power_tensor,
        attach_transform_policy(metadata, transform_policy),
        unsupported_spectrum_mask,
        unsupported_first_freq_index,
    )


def _run_decomposition(
    power_tensor: np.ndarray,
    metadata: dict[str, Any],
    unsupported_spectrum_mask: np.ndarray,
    unsupported_first_freq_index: np.ndarray,
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
    annotation_support = _periodic_annotation_support(prepared, options, metadata)
    if options.mask_edge_effects:
        annotation_skip_radius_s = float(annotation_support["annotation_skip_radius_s"])
        has_channel_specific_mask = has_channel_specific_mask_annotations(prepared.raw)
        if has_channel_specific_mask:
            skip_time_mask, _ = output_time_mask_by_annotations(
                prepared.raw,
                times_s=times_meta,
                output_channels=[(str(channel),) for channel in channel_meta],
                keep=("bad", "edge"),
                mode="substring",
                pad_s=annotation_skip_radius_s,
                clip_to_raw=True,
                require_match=False,
            )
        else:
            shared_skip_time_mask = build_annotation_skip_time_mask(
                prepared.raw,
                times_s=times_meta,
                radius_s=annotation_skip_radius_s,
            )
            skip_time_mask = np.broadcast_to(
                shared_skip_time_mask[None, :],
                (len(channel_meta), times_meta.size),
            ).copy()
    else:
        annotation_skip_radius_s = None
        skip_time_mask = np.zeros(
            (len(channel_meta), times_meta.size),
            dtype=bool,
        )
    finite_times = np.isfinite(times_meta)
    valid_time_mask = finite_times[None, :] & ~skip_time_mask
    unsupported_retained = unsupported_spectrum_mask & valid_time_mask[None, :, :]
    if np.any(unsupported_retained):
        unsupported_count = int(np.sum(unsupported_retained))
        epoch_index, channel_index, time_index = (
            int(item) for item in np.argwhere(unsupported_retained)[0]
        )
        frequency_index = int(
            unsupported_first_freq_index[epoch_index, channel_index, time_index]
        )
        raise ValueError(
            "Periodic/Aperiodic encountered "
            f"{unsupported_count} unsupported spectra outside the BAD/EDGE mask; "
            f"first at epoch={epoch_index}, channel={channel_meta[channel_index]!r}, "
            f"frequency={float(freqs_meta[frequency_index]):g} Hz, "
            f"time_index={time_index}, time={float(times_meta[time_index]):g} s."
        )
    n_spectra_unsupported_masked = int(np.sum(unsupported_spectrum_mask))
    n_columns_total = int(np.sum(finite_times))
    n_columns_skipped_masked = int(
        np.sum(finite_times & np.all(skip_time_mask, axis=0))
    )
    n_channel_columns_total = int(len(channel_meta) * np.sum(finite_times))
    n_channel_columns_skipped_masked = int(
        np.sum(skip_time_mask & finite_times[None, :])
    )
    valid_time_mask_for_decompose = (
        valid_time_mask[0]
        if valid_time_mask.shape[0] > 0
        and np.all(valid_time_mask == valid_time_mask[0])
        else valid_time_mask
    )
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
        valid_time_mask=valid_time_mask_for_decompose,
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

    transform_policy = get_transform_policy(options.value_transform_mode)
    metadata = attach_transform_policy(metadata, transform_policy)
    params_meta_dict = attach_transform_policy(
        params_meta_dict,
        get_transform_policy("none"),
    )
    if "notch_interpolation" in metadata:
        params_meta_dict["notch_interpolation"] = dict(metadata["notch_interpolation"])
    metadata_params = dict(metadata.get("params", {}) or {})
    metadata_params.update(
        {
            **annotation_support,
            "n_columns_total": n_columns_total,
            "n_columns_skipped_masked": n_columns_skipped_masked,
            "n_channel_columns_total": n_channel_columns_total,
            "n_channel_columns_skipped_masked": n_channel_columns_skipped_masked,
            "n_spectra_unsupported_masked": n_spectra_unsupported_masked,
        }
    )
    metadata["params"] = metadata_params
    params_runtime = dict(params_meta_dict.get("params", {}) or {})
    params_runtime.update(
        {
            **annotation_support,
            "n_columns_total": n_columns_total,
            "n_columns_skipped_masked": n_columns_skipped_masked,
            "n_channel_columns_total": n_channel_columns_total,
            "n_channel_columns_skipped_masked": n_channel_columns_skipped_masked,
            "n_spectra_unsupported_masked": n_spectra_unsupported_masked,
        }
    )
    params_meta_dict["params"] = params_runtime

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
        mt_time_bandwidth_product=float(options.mt_time_bandwidth_product),
        mt_min_cycles=float(options.mt_min_cycles),
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
    (
        power_tensor,
        metadata,
        unsupported_spectrum_mask,
        unsupported_first_freq_index,
    ) = _run_tfr_grid(
        prepared,
        options,
        tfr_grid_fn=tfr_grid_fn,
        interpolate_freq_tensor_fn=interpolate_freq_tensor_fn,
        smooth_axis_fn=smooth_axis_fn,
    )
    return _run_decomposition(
        power_tensor,
        metadata,
        unsupported_spectrum_mask,
        unsupported_first_freq_index,
        prepared,
        options,
        report_dir,
        decompose_fn=decompose_fn,
        make_gof_rsquared_masker_fn=make_gof_rsquared_masker_fn,
    )
