"""Preparation helpers for the periodic/aperiodic tensor runner."""

from __future__ import annotations

import numpy as np

from .. import service as svc
from .periodic_aperiodic_models import (
    PeriodicAperiodicOptions,
    PeriodicAperiodicPaths,
    PeriodicAperiodicPreparedInput,
)

METRIC_KEY = "periodic_aperiodic"
METRIC_LABEL = "Periodic/APeriodic"


def build_periodic_aperiodic_paths(
    options: PeriodicAperiodicOptions,
) -> PeriodicAperiodicPaths:
    resolver = svc.PathResolver(options.context)
    output_path = svc.tensor_metric_tensor_path(resolver, METRIC_KEY, create=True)
    return PeriodicAperiodicPaths(
        resolver=resolver,
        input_path=svc.preproc_step_raw_path(resolver, "finish"),
        output_path=output_path,
        aperiodic_output_path=svc.tensor_metric_tensor_path(
            resolver,
            "aperiodic",
            create=True,
        ),
        report_dir=output_path.parent / "specparam_report",
        config_path=svc.tensor_metric_config_path(resolver, METRIC_KEY, create=True),
        log_path=svc.tensor_metric_log_path(resolver, METRIC_KEY, create=True),
    )


def validate_preproc_input(paths: PeriodicAperiodicPaths) -> str | None:
    if (
        svc.indicator_from_log(svc.preproc_step_log_path(paths.resolver, "finish"))
        != "green"
    ):
        return "Missing green preproc finish log."
    if not paths.input_path.exists():
        return "Missing preproc finish raw input."
    return None


def prepare_periodic_aperiodic_runtime(
    options: PeriodicAperiodicOptions,
    paths: PeriodicAperiodicPaths,
    *,
    read_raw_fif_fn=None,
    load_tensor_filter_inheritance_fn=None,
    compute_notch_intervals_fn=None,
) -> PeriodicAperiodicPreparedInput:
    if read_raw_fif_fn is None:
        import mne

        read_raw_fif = mne.io.read_raw_fif
    else:
        read_raw_fif = read_raw_fif_fn

    raw = read_raw_fif(str(paths.input_path), preload=False, verbose="ERROR")
    available_channels = set(raw.ch_names)
    picks = [
        name
        for name in (options.selected_channels or raw.ch_names)
        if name in available_channels
    ]
    if not picks:
        raise ValueError("No valid channels selected for Periodic/APeriodic.")

    spec_range = options.freq_range_hz
    if spec_range is None:
        spec_range = (float(options.low_freq), float(options.high_freq))
    spec_low = float(spec_range[0])
    spec_high = float(spec_range[1])
    if spec_high <= spec_low:
        raise ValueError("SpecParam freq range must satisfy high > low.")
    if float(options.low_freq) < spec_low or float(options.high_freq) > spec_high:
        raise ValueError("Low/high frequency must stay within SpecParam freq range.")

    nyquist = float(raw.info["sfreq"]) / 2.0
    if spec_high > nyquist:
        raise ValueError("SpecParam high frequency exceeds Nyquist frequency.")

    inheritance = (
        load_tensor_filter_inheritance_fn or svc.load_tensor_filter_inheritance
    )(options.context)
    runtime_notch_payload = svc.build_tensor_metric_notch_payload(
        options.notches,
        options.notch_widths,
    )
    runtime_notches = tuple(float(item) for item in runtime_notch_payload["notches"])
    runtime_notch_widths = svc._expand_notch_widths(
        runtime_notch_payload["notch_widths"],
        len(runtime_notches),
    )
    freqs_model = svc._build_frequency_grid(spec_low, spec_high, options.step_hz)
    freqs_final = svc._build_frequency_grid(
        options.low_freq,
        options.high_freq,
        options.step_hz,
    )
    notch_intervals = (compute_notch_intervals_fn or svc._compute_notch_intervals)(
        low_freq=spec_low,
        high_freq=spec_high,
        notches=runtime_notches,
        notch_widths=runtime_notch_widths,
    )
    freqs_compute = freqs_model
    interpolation_applied = False
    if notch_intervals:
        freqs_compute, removed_mask = svc._cut_frequency_grid_by_intervals(
            freqs_model,
            notch_intervals,
        )
        if bool(np.any(removed_mask)):
            if freqs_compute.size < 2:
                raise ValueError(
                    "Notch exclusion removed too many bins; relax notch widths or frequency range."
                )
            interpolation_applied = True

    return PeriodicAperiodicPreparedInput(
        raw=raw,
        picks=picks,
        inheritance=inheritance,
        runtime_notches=runtime_notches,
        runtime_notch_widths=runtime_notch_widths,
        spec_low=spec_low,
        spec_high=spec_high,
        freqs_model=np.asarray(freqs_model, dtype=float),
        freqs_final=np.asarray(freqs_final, dtype=float),
        freqs_compute=np.asarray(freqs_compute, dtype=float),
        notch_intervals=list(notch_intervals),
        interpolation_applied=bool(interpolation_applied),
        method_norm=svc._normalize_metric_method(
            options.method,
            metric_label=METRIC_LABEL,
        ),
    )
