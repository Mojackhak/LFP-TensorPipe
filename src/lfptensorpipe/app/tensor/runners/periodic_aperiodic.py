"""Periodic/APeriodic metric runner facade."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import numpy as np

from .. import service as svc
from .periodic_aperiodic_compute import compute_periodic_aperiodic_outputs
from .periodic_aperiodic_models import PeriodicAperiodicOptions
from .periodic_aperiodic_persist import (
    write_periodic_aperiodic_failure,
    write_periodic_aperiodic_missing_input,
    write_periodic_aperiodic_success,
)
from .periodic_aperiodic_prepare import (
    METRIC_KEY,
    build_periodic_aperiodic_paths,
    prepare_periodic_aperiodic_runtime,
    validate_preproc_input,
)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


@contextmanager
def _prepare_specparam_report_dir(report_dir: Path):
    backup_dir: Path | None = None
    if report_dir.exists():
        backup_dir = report_dir.parent / f".{report_dir.name}.bak-{uuid4().hex}"
        report_dir.replace(backup_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield report_dir
    except Exception:
        _remove_path(report_dir)
        if backup_dir is not None and backup_dir.exists():
            backup_dir.replace(report_dir)
        raise
    else:
        if backup_dir is not None:
            _remove_path(backup_dir)


def run_periodic_aperiodic_metric(
    context,
    *,
    low_freq: float,
    high_freq: float,
    step_hz: float,
    mask_edge_effects: bool,
    bands: list[dict[str, Any]],
    selected_channels: list[str] | None,
    method: str = "morlet",
    time_resolution_s: float = 0.5,
    hop_s: float = 0.025,
    min_cycles: float | None = 3.0,
    max_cycles: float | None = None,
    time_bandwidth: float = 1.0,
    freq_range_hz: tuple[float, float] | None = None,
    freq_smooth_enabled: bool = True,
    freq_smooth_sigma: float | None = 1.5,
    time_smooth_enabled: bool = True,
    time_smooth_kernel_size: int | None = None,
    aperiodic_mode: str = "fixed",
    peak_width_limits_hz: tuple[float, float] = (2.0, 12.0),
    max_n_peaks: float = np.inf,
    min_peak_height: float = 0.0,
    peak_threshold: float = 2.0,
    fit_qc_threshold: float = 0.6,
    notches: Any = None,
    notch_widths: Any = 2.0,
    n_jobs: int = 1,
    outer_n_jobs: int = 1,
    read_raw_fif_fn=None,
    load_tensor_filter_inheritance_fn=None,
    compute_notch_intervals_fn=None,
    tfr_grid_fn=None,
    interpolate_freq_tensor_fn=None,
    smooth_axis_fn=None,
    decompose_fn=None,
    make_gof_rsquared_masker_fn=None,
) -> tuple[bool, str]:
    options = PeriodicAperiodicOptions(
        context=context,
        low_freq=low_freq,
        high_freq=high_freq,
        step_hz=step_hz,
        mask_edge_effects=mask_edge_effects,
        bands=bands,
        selected_channels=selected_channels,
        method=method,
        time_resolution_s=time_resolution_s,
        hop_s=hop_s,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        time_bandwidth=time_bandwidth,
        freq_range_hz=freq_range_hz,
        freq_smooth_enabled=freq_smooth_enabled,
        freq_smooth_sigma=freq_smooth_sigma,
        time_smooth_enabled=time_smooth_enabled,
        time_smooth_kernel_size=time_smooth_kernel_size,
        aperiodic_mode=aperiodic_mode,
        peak_width_limits_hz=peak_width_limits_hz,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        fit_qc_threshold=fit_qc_threshold,
        notches=notches,
        notch_widths=notch_widths,
        n_jobs=n_jobs,
        outer_n_jobs=outer_n_jobs,
    )
    paths = build_periodic_aperiodic_paths(options)
    input_failure = validate_preproc_input(paths)
    if input_failure is not None:
        return write_periodic_aperiodic_missing_input(paths, message=input_failure)

    prepared = None
    try:
        prepared = prepare_periodic_aperiodic_runtime(
            options,
            paths,
            read_raw_fif_fn=read_raw_fif_fn,
            load_tensor_filter_inheritance_fn=load_tensor_filter_inheritance_fn,
            compute_notch_intervals_fn=compute_notch_intervals_fn,
        )
        with _prepare_specparam_report_dir(paths.report_dir):
            outputs = compute_periodic_aperiodic_outputs(
                prepared,
                options,
                paths.report_dir,
                tfr_grid_fn=tfr_grid_fn,
                interpolate_freq_tensor_fn=interpolate_freq_tensor_fn,
                smooth_axis_fn=smooth_axis_fn,
                decompose_fn=decompose_fn,
                make_gof_rsquared_masker_fn=make_gof_rsquared_masker_fn,
            )
            return write_periodic_aperiodic_success(paths, options, prepared, outputs)
    except Exception as exc:  # noqa: BLE001
        runtime_notch_payload = svc.build_tensor_metric_notch_payload(
            options.notches,
            options.notch_widths,
        )
        inheritance = (
            prepared.inheritance
            if prepared is not None
            else svc.load_tensor_filter_inheritance(options.context)
        )
        return write_periodic_aperiodic_failure(
            paths,
            options,
            notches=[float(item) for item in runtime_notch_payload["notches"]],
            notch_widths=list(
                svc._expand_notch_widths(
                    runtime_notch_payload["notch_widths"],
                    len(runtime_notch_payload["notches"]),
                )
            ),
            inherited_filter_notches=[float(item) for item in inheritance.notches],
            inherited_filter_notch_widths=[
                float(item) for item in inheritance.notch_widths
            ],
            message=f"Periodic/APeriodic failed: {exc}",
        )
    finally:
        if prepared is not None and hasattr(prepared.raw, "close"):
            prepared.raw.close()


__all__ = ["run_periodic_aperiodic_metric", "METRIC_KEY"]
