"""Persistence and logging helpers for the periodic/aperiodic tensor runner."""

from __future__ import annotations

import numpy as np

from .. import service as svc
from .periodic_aperiodic_models import (
    NOTCH_INTERPOLATION_METHOD,
    PeriodicAperiodicOptions,
    PeriodicAperiodicOutputs,
    PeriodicAperiodicPaths,
    PeriodicAperiodicPreparedInput,
    derive_notch_interpolation_seed,
)
from .periodic_aperiodic_prepare import METRIC_KEY


def _max_n_peaks_payload(value: float) -> str | float:
    return "inf" if np.isinf(float(value)) else float(value)


def write_periodic_aperiodic_missing_input(
    paths: PeriodicAperiodicPaths,
    *,
    message: str,
) -> tuple[bool, str]:
    svc._write_metric_log(
        paths.resolver,
        METRIC_KEY,
        completed=False,
        params={},
        input_path=str(paths.input_path),
        output_path=str(paths.output_path),
        message=message,
    )
    return False, message


def write_periodic_aperiodic_failure(
    paths: PeriodicAperiodicPaths,
    options: PeriodicAperiodicOptions,
    *,
    notches: list[float],
    notch_radii: list[float],
    inherited_filter_notches: list[float],
    inherited_filter_notch_widths: list[float],
    message: str,
) -> tuple[bool, str]:
    svc._write_metric_log(
        paths.resolver,
        METRIC_KEY,
        completed=False,
        params={
            "low_freq": float(options.low_freq),
            "high_freq": float(options.high_freq),
            "step_hz": float(options.step_hz),
            "method": str(options.method),
            "time_resolution_s": float(options.time_resolution_s),
            "hop_s": float(options.hop_s),
            "min_cycles": (
                float(options.min_cycles) if options.min_cycles is not None else None
            ),
            "max_cycles": (
                float(options.max_cycles) if options.max_cycles is not None else None
            ),
            "mt_time_bandwidth_product": float(options.mt_time_bandwidth_product),
            "mt_min_cycles": float(options.mt_min_cycles),
            "mt_max_cycles": (
                float(options.mt_max_cycles)
                if options.mt_max_cycles is not None
                else None
            ),
            "freq_range_hz": (
                [float(options.freq_range_hz[0]), float(options.freq_range_hz[1])]
                if options.freq_range_hz is not None
                else None
            ),
            "freq_smooth_enabled": bool(options.freq_smooth_enabled),
            "freq_smooth_sigma": (
                float(options.freq_smooth_sigma)
                if options.freq_smooth_sigma is not None
                else None
            ),
            "time_smooth_enabled": bool(options.time_smooth_enabled),
            "time_smooth_kernel_size": (
                int(options.time_smooth_kernel_size)
                if options.time_smooth_kernel_size is not None
                else None
            ),
            "aperiodic_mode": str(options.aperiodic_mode),
            "peak_width_limits_hz": [
                float(options.peak_width_limits_hz[0]),
                float(options.peak_width_limits_hz[1]),
            ],
            "max_n_peaks": _max_n_peaks_payload(options.max_n_peaks),
            "min_peak_height": float(options.min_peak_height),
            "peak_threshold": float(options.peak_threshold),
            "fit_qc_threshold": float(options.fit_qc_threshold),
            "mask_edge_effects": bool(options.mask_edge_effects),
            "notches": list(notches),
            "notch_radii": list(notch_radii),
            "inherited_filter_notches": list(inherited_filter_notches),
            "inherited_filter_notch_widths": list(inherited_filter_notch_widths),
            "specparam_report_dir": str(paths.report_dir),
            "selected_channels": [
                str(item) for item in (options.selected_channels or [])
            ],
            **svc._effective_n_jobs_payload(
                n_jobs=int(options.n_jobs),
                outer_n_jobs=int(options.outer_n_jobs),
            ),
        },
        input_path=str(paths.input_path),
        output_path=str(paths.output_path),
        message=message,
    )
    return False, message


def write_periodic_aperiodic_success(
    paths: PeriodicAperiodicPaths,
    options: PeriodicAperiodicOptions,
    prepared: PeriodicAperiodicPreparedInput,
    outputs: PeriodicAperiodicOutputs,
) -> tuple[bool, str]:
    interpolation_seed = derive_notch_interpolation_seed(options.context)
    notch_metadata = {
        "notches": [float(item) for item in prepared.runtime_notches],
        "notch_radii": [float(item) for item in prepared.runtime_notch_radii],
        "notch_intervals_hz": [
            [float(low), float(high)] for low, high in prepared.notch_intervals
        ],
        "inherited_filter_notches": [
            float(item) for item in prepared.inheritance.notches
        ],
        "inherited_filter_notch_widths": [
            float(item) for item in prepared.inheritance.notch_widths
        ],
    }
    tensor_metadata = dict(outputs.metadata)
    tensor_metadata.update(notch_metadata)
    params_metadata = dict(outputs.params_meta)
    params_metadata.update(notch_metadata)
    runtime_params = (
        outputs.metadata.get("params", {}) if isinstance(outputs.metadata, dict) else {}
    )
    if not isinstance(runtime_params, dict):
        runtime_params = {}
    count_payload = {
        "n_columns_total": int(runtime_params.get("n_columns_total", 0)),
        "n_columns_skipped_masked": int(
            runtime_params.get("n_columns_skipped_masked", 0)
        ),
        "n_spectra_unsupported_masked": int(
            runtime_params.get("n_spectra_unsupported_masked", 0)
        ),
    }
    support_payload = {
        "mask_support_semantics": str(runtime_params["mask_support_semantics"]),
        "annotation_estimator_support_radius_s": runtime_params.get(
            "annotation_estimator_support_radius_s"
        ),
        "annotation_time_smoothing_radius_s": runtime_params.get(
            "annotation_time_smoothing_radius_s"
        ),
        "annotation_skip_radius_s": runtime_params.get("annotation_skip_radius_s"),
        "consumed_tfr_frequency_min_hz": runtime_params.get(
            "consumed_tfr_frequency_min_hz"
        ),
        "consumed_tfr_frequency_max_hz": runtime_params.get(
            "consumed_tfr_frequency_max_hz"
        ),
        "consumed_tfr_frequency_count": runtime_params.get(
            "consumed_tfr_frequency_count"
        ),
        "time_smoothing_kernel_size_eff": runtime_params.get(
            "time_smoothing_kernel_size_eff"
        ),
        "hop_s_eff": runtime_params.get("hop_s_eff"),
    }
    config_payload = {
        "metric_key": METRIC_KEY,
        "metric_label": svc.TENSOR_METRICS_BY_KEY[METRIC_KEY].display_name,
        "output_component": "periodic",
        "periodic_tensor_path": str(paths.output_path),
        "aperiodic_tensor_path": str(paths.aperiodic_output_path),
        "specparam_report_dir": str(paths.report_dir),
        "method": prepared.method_norm,
        "low_freq": float(options.low_freq),
        "high_freq": float(options.high_freq),
        "step_hz": float(options.step_hz),
        "specparam_low_freq": float(prepared.spec_low),
        "specparam_high_freq": float(prepared.spec_high),
        "time_resolution_s": float(options.time_resolution_s),
        "hop_s": float(options.hop_s),
        "min_cycles": (
            float(options.min_cycles) if options.min_cycles is not None else None
        ),
        "max_cycles": (
            float(options.max_cycles) if options.max_cycles is not None else None
        ),
        "mt_time_bandwidth_product": float(options.mt_time_bandwidth_product),
        "mt_min_cycles": float(options.mt_min_cycles),
        "mt_max_cycles": (
            float(options.mt_max_cycles) if options.mt_max_cycles is not None else None
        ),
        "freq_range_hz": [float(prepared.spec_low), float(prepared.spec_high)],
        "freq_smooth_enabled": bool(options.freq_smooth_enabled),
        "freq_smooth_sigma": (
            float(options.freq_smooth_sigma)
            if options.freq_smooth_sigma is not None
            else None
        ),
        "time_smooth_enabled": bool(options.time_smooth_enabled),
        "time_smooth_kernel_size": (
            int(options.time_smooth_kernel_size)
            if options.time_smooth_kernel_size is not None
            else None
        ),
        "aperiodic_mode": str(options.aperiodic_mode),
        "peak_width_limits_hz": [
            float(options.peak_width_limits_hz[0]),
            float(options.peak_width_limits_hz[1]),
        ],
        "max_n_peaks": _max_n_peaks_payload(options.max_n_peaks),
        "min_peak_height": float(options.min_peak_height),
        "peak_threshold": float(options.peak_threshold),
        "fit_qc_threshold": float(options.fit_qc_threshold),
        "mask_edge_effects": bool(options.mask_edge_effects),
        "bands": options.bands,
        "channels": prepared.picks,
        "selected_channels": prepared.picks,
        "freqs_compute": [float(item) for item in prepared.freqs_compute.tolist()],
        "freqs_full": [float(item) for item in prepared.freqs_model.tolist()],
        "freqs_final": [float(item) for item in prepared.freqs_final.tolist()],
        "notches": [float(item) for item in prepared.runtime_notches],
        "notch_radii": [float(item) for item in prepared.runtime_notch_radii],
        "inherited_filter_notches": [
            float(item) for item in prepared.inheritance.notches
        ],
        "inherited_filter_notch_widths": [
            float(item) for item in prepared.inheritance.notch_widths
        ],
        "notch_intervals_hz": [
            [float(lo), float(hi)] for lo, hi in prepared.notch_intervals
        ],
        "interpolation_applied": bool(prepared.interpolation_applied),
        "notch_interpolation_method": (
            NOTCH_INTERPOLATION_METHOD if prepared.interpolation_applied else "none"
        ),
        "notch_interpolation_seed": (
            interpolation_seed if prepared.interpolation_applied else None
        ),
        "notch_interpolation_assignment": (
            "balanced_within_epoch_channel_over_time"
            if prepared.interpolation_applied
            else None
        ),
        "tensor_shape": [int(item) for item in outputs.tensor.shape],
        "params_tensor_shape": [int(item) for item in outputs.params_tensor.shape],
        **support_payload,
        **count_payload,
        **svc._effective_n_jobs_payload(
            n_jobs=int(options.n_jobs),
            outer_n_jobs=int(options.outer_n_jobs),
        ),
    }
    svc._write_outputs_atomically(
        [
            (
                paths.output_path,
                lambda path: svc.save_pkl(
                    {"tensor": outputs.tensor, "meta": tensor_metadata},
                    path,
                ),
            ),
            (
                paths.aperiodic_output_path,
                lambda path: svc.save_pkl(
                    {"tensor": outputs.params_tensor, "meta": params_metadata},
                    path,
                ),
            ),
            (
                paths.config_path,
                lambda path: svc._write_metric_config(path, config_payload),
            ),
            (
                paths.log_path,
                lambda path: svc._write_metric_log_to_path(
                    path,
                    METRIC_KEY,
                    completed=True,
                    params={
                        "low_freq": float(options.low_freq),
                        "high_freq": float(options.high_freq),
                        "specparam_low_freq": float(prepared.spec_low),
                        "specparam_high_freq": float(prepared.spec_high),
                        "step_hz": float(options.step_hz),
                        "method": prepared.method_norm,
                        "time_resolution_s": float(options.time_resolution_s),
                        "hop_s": float(options.hop_s),
                        "min_cycles": (
                            float(options.min_cycles)
                            if options.min_cycles is not None
                            else None
                        ),
                        "max_cycles": (
                            float(options.max_cycles)
                            if options.max_cycles is not None
                            else None
                        ),
                        "mt_time_bandwidth_product": float(
                            options.mt_time_bandwidth_product
                        ),
                        "mt_min_cycles": float(options.mt_min_cycles),
                        "mt_max_cycles": (
                            float(options.mt_max_cycles)
                            if options.mt_max_cycles is not None
                            else None
                        ),
                        "freq_range_hz": [
                            float(prepared.spec_low),
                            float(prepared.spec_high),
                        ],
                        "freq_smooth_enabled": bool(options.freq_smooth_enabled),
                        "freq_smooth_sigma": (
                            float(options.freq_smooth_sigma)
                            if options.freq_smooth_sigma is not None
                            else None
                        ),
                        "time_smooth_enabled": bool(options.time_smooth_enabled),
                        "time_smooth_kernel_size": (
                            int(options.time_smooth_kernel_size)
                            if options.time_smooth_kernel_size is not None
                            else None
                        ),
                        "aperiodic_mode": str(options.aperiodic_mode),
                        "peak_width_limits_hz": [
                            float(options.peak_width_limits_hz[0]),
                            float(options.peak_width_limits_hz[1]),
                        ],
                        "max_n_peaks": _max_n_peaks_payload(options.max_n_peaks),
                        "min_peak_height": float(options.min_peak_height),
                        "peak_threshold": float(options.peak_threshold),
                        "fit_qc_threshold": float(options.fit_qc_threshold),
                        "mask_edge_effects": bool(options.mask_edge_effects),
                        "notches": [float(item) for item in prepared.runtime_notches],
                        "notch_radii": [
                            float(item) for item in prepared.runtime_notch_radii
                        ],
                        "inherited_filter_notches": [
                            float(item) for item in prepared.inheritance.notches
                        ],
                        "inherited_filter_notch_widths": [
                            float(item) for item in prepared.inheritance.notch_widths
                        ],
                        "interpolation_applied": bool(prepared.interpolation_applied),
                        "notch_interpolation_method": (
                            NOTCH_INTERPOLATION_METHOD
                            if prepared.interpolation_applied
                            else "none"
                        ),
                        "notch_interpolation_seed": (
                            interpolation_seed
                            if prepared.interpolation_applied
                            else None
                        ),
                        "notch_interpolation_assignment": (
                            "balanced_within_epoch_channel_over_time"
                            if prepared.interpolation_applied
                            else None
                        ),
                        "n_channels": len(prepared.picks),
                        "selected_channels": prepared.picks,
                        "n_freqs": int(outputs.tensor.shape[2]),
                        "n_times": int(outputs.tensor.shape[3]),
                        "component": "periodic",
                        "aperiodic_tensor_path": str(paths.aperiodic_output_path),
                        "specparam_report_dir": str(paths.report_dir),
                        "n_params": int(outputs.params_tensor.shape[2]),
                        **support_payload,
                        **count_payload,
                        **svc._effective_n_jobs_payload(
                            n_jobs=int(options.n_jobs),
                            outer_n_jobs=int(options.outer_n_jobs),
                        ),
                    },
                    input_path=str(paths.input_path),
                    output_path=str(paths.output_path),
                    message=(
                        "Periodic/APeriodic tensor computed with reproducible local residual notch imputation."
                        if prepared.interpolation_applied
                        else "Periodic/APeriodic tensor computed."
                    ),
                ),
            ),
        ]
    )
    success_message = (
        "Periodic/APeriodic tensor computed with reproducible local residual notch imputation."
        if prepared.interpolation_applied
        else "Periodic/APeriodic tensor computed."
    )
    return True, success_message
