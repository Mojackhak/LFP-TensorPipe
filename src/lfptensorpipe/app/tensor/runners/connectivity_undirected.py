"""Undirected connectivity tensor runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.utils.transforms import attach_transform_policy, get_transform_policy

from .. import service as svc


def run_undirected_connectivity_metric(
    context: RecordContext,
    *,
    metric_key: str,
    connectivity_metric: str,
    low_freq: float,
    high_freq: float,
    step_hz: float,
    mask_edge_effects: bool,
    bands: list[dict[str, Any]],
    selected_channels: list[str] | None,
    selected_pairs: list[tuple[str, str]] | None,
    time_resolution_s: float = 0.5,
    hop_s: float = 0.025,
    method: str = "morlet",
    mt_time_bandwidth_product: float = 4.0,
    mt_min_cycles: float = 3.0,
    min_cycles: float | None = 3.0,
    max_cycles: float | None = None,
    notches: Any = None,
    notch_radii: Any = 2.0,
    n_jobs: int = 1,
    outer_n_jobs: int = 1,
    read_raw_fif_fn=None,
    conn_grid_fn=None,
    interpolate_freq_tensor_fn=None,
    normalize_selected_pairs_fn=None,
    compute_notch_intervals_fn=None,
) -> tuple[bool, str]:
    if conn_grid_fn is None:
        from lfptensorpipe.lfp.connectivity.grid import grid as conn_grid
    else:
        conn_grid = conn_grid_fn
    if interpolate_freq_tensor_fn is None:
        from lfptensorpipe.lfp.interp.freq import (
            interpolate_tensor_with_metadata_transformed as interpolate_freq_tensor,
        )
    else:
        interpolate_freq_tensor = interpolate_freq_tensor_fn

    PathResolver = svc.PathResolver
    preproc_step_raw_path = svc.preproc_step_raw_path
    preproc_step_log_path = svc.preproc_step_log_path
    tensor_metric_tensor_path = svc.tensor_metric_tensor_path
    tensor_metric_config_path = svc.tensor_metric_config_path
    tensor_metric_log_path = svc.tensor_metric_log_path
    load_tensor_filter_inheritance = svc.load_tensor_filter_inheritance
    indicator_from_log = svc.indicator_from_log
    _write_metric_log = svc._write_metric_log
    _write_metric_log_to_path = svc._write_metric_log_to_path
    _write_metric_config = svc._write_metric_config
    _write_outputs_atomically = svc._write_outputs_atomically
    _build_frequency_grid = svc._build_frequency_grid
    _compute_notch_intervals = (
        compute_notch_intervals_fn or svc._compute_notch_intervals
    )
    _cut_frequency_grid_by_intervals = svc._cut_frequency_grid_by_intervals
    _normalize_metric_method = svc._normalize_metric_method
    _compute_mask_radii_seconds = svc._compute_mask_radii_seconds
    _apply_dynamic_edge_mask_strict = svc._apply_dynamic_edge_mask_strict
    _effective_n_jobs_payload = svc._effective_n_jobs_payload
    _normalize_selected_pairs = (
        normalize_selected_pairs_fn or svc._normalize_selected_pairs
    )
    save_pkl = svc.save_pkl
    TENSOR_METRICS_BY_KEY = svc.TENSOR_METRICS_BY_KEY
    combinations = svc.combinations

    resolver = PathResolver(context)
    metric_spec = TENSOR_METRICS_BY_KEY.get(metric_key)
    metric_label = metric_spec.display_name if metric_spec is not None else metric_key
    transform_policy = get_transform_policy(
        metric_spec.value_transform_mode if metric_spec is not None else "none"
    )
    input_path = preproc_step_raw_path(resolver, "finish")
    output_path = tensor_metric_tensor_path(resolver, metric_key, create=True)
    config_path = tensor_metric_config_path(resolver, metric_key, create=True)
    log_path = tensor_metric_log_path(resolver, metric_key, create=True)
    inheritance = load_tensor_filter_inheritance(context)
    runtime_notch_payload = svc.build_tensor_metric_notch_payload(notches, notch_radii)
    runtime_notches = tuple(float(item) for item in runtime_notch_payload["notches"])
    runtime_notch_radii = svc._expand_notch_radii(
        runtime_notch_payload["notch_radii"],
        len(runtime_notches),
    )

    if indicator_from_log(preproc_step_log_path(resolver, "finish")) != "green":
        message = "Missing green preproc finish log."
        _write_metric_log(
            resolver,
            metric_key,
            completed=False,
            params={},
            input_path=str(input_path),
            output_path=str(output_path),
            message=message,
        )
        return False, message
    if not input_path.exists():
        message = "Missing preproc finish raw input."
        _write_metric_log(
            resolver,
            metric_key,
            completed=False,
            params={},
            input_path=str(input_path),
            output_path=str(output_path),
            message=message,
        )
        return False, message

    try:
        if read_raw_fif_fn is None:
            import mne

            read_raw_fif = mne.io.read_raw_fif
        else:
            read_raw_fif = read_raw_fif_fn

        raw = read_raw_fif(str(input_path), preload=False, verbose="ERROR")
        available_channels = set(raw.ch_names)
        if selected_pairs is None:
            picks = [
                name
                for name in (selected_channels or raw.ch_names)
                if name in available_channels
            ]
            if len(picks) < 2:
                raise ValueError(f"{metric_label} requires at least 2 valid channels.")
            pairs = list(combinations(picks, 2))
        else:
            pairs = _normalize_selected_pairs(
                selected_pairs,
                available_channels=available_channels,
                directed=False,
            )
            if not pairs:
                raise ValueError(
                    f"No valid selected pairs available for {metric_label}."
                )
            picks = [
                name for name in raw.ch_names if any(name in pair for pair in pairs)
            ]
            if len(picks) < 2:
                raise ValueError(f"{metric_label} requires at least 2 valid channels.")

        nyquist = float(raw.info["sfreq"]) / 2.0
        applied_high = min(high_freq, nyquist)
        if applied_high <= low_freq:
            raise ValueError("High frequency is invalid for current Nyquist frequency.")
        freqs_full = _build_frequency_grid(low_freq, applied_high, step_hz)
        notch_intervals = _compute_notch_intervals(
            low_freq=low_freq,
            high_freq=applied_high,
            notches=runtime_notches,
            notch_radii=runtime_notch_radii,
        )
        freqs_compute = freqs_full
        interpolation_applied = False
        if notch_intervals:
            freqs_compute, removed_mask = _cut_frequency_grid_by_intervals(
                freqs_full, notch_intervals
            )
            if bool(np.any(removed_mask)):
                if freqs_compute.size < 2:
                    raise ValueError(
                        "Notch exclusion removed too many bins; reduce notch radii or widen the frequency range."
                    )
                interpolation_applied = True

        method_norm = _normalize_metric_method(method, metric_label=metric_label)
        spectral_mode_use = "cwt_morlet" if method_norm == "morlet" else "multitaper"
        final_mask_radii = _compute_mask_radii_seconds(
            freqs_full,
            method=method_norm,
            time_resolution_s=float(time_resolution_s),
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            mt_time_bandwidth_product=float(mt_time_bandwidth_product),
            mt_min_cycles=float(mt_min_cycles),
        )
        annotation_skip_radius_s = (
            float(np.min(final_mask_radii)) if mask_edge_effects else None
        )

        tensor, metadata = conn_grid(
            raw,
            freqs=freqs_compute,
            time_resolution_s=float(time_resolution_s),
            hop_s=float(hop_s),
            pairs=pairs,
            method=str(connectivity_metric),
            multivariate=False,
            ordered_pairs=False,
            picks=picks,
            spectral_mode=str(spectral_mode_use),
            mt_time_bandwidth_product=float(mt_time_bandwidth_product),
            mt_min_cycles=float(mt_min_cycles),
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            outer_n_jobs=int(outer_n_jobs),
            annotation_skip_radius_s=annotation_skip_radius_s,
        )

        tensor4d = np.asarray(tensor, dtype=float)
        if tensor4d.ndim == 3:
            tensor4d = tensor4d[None, ...]
        if tensor4d.ndim != 4:
            raise ValueError(
                f"Unexpected {metric_label} tensor shape: {tensor4d.shape}"
            )

        if interpolation_applied:
            tensor4d, metadata = interpolate_freq_tensor(
                tensor4d,
                metadata,
                freqs_out=freqs_full,
                axis=-2,
                method="linear",
                transform_mode=(
                    transform_policy.mode
                    if transform_policy.interpolation_domain == "transformed"
                    else "none"
                ),
            )
            tensor4d = np.asarray(tensor4d, dtype=float)
        metadata = attach_transform_policy(metadata, transform_policy)
        if mask_edge_effects:
            freq_axis = np.asarray(
                (metadata.get("axes", {}) or {}).get("freq", []), dtype=float
            ).ravel()
            if freq_axis.size != tensor4d.shape[2]:
                raise ValueError(
                    f"{metric_label} edge mask failed: metadata frequency axis does not match tensor."
                )
            tensor4d, metadata = _apply_dynamic_edge_mask_strict(
                raw=raw,
                tensor=tensor4d,
                metadata=metadata,
                metric_label=metric_label,
                freqs_lookup=[float(item) for item in freq_axis.tolist()],
                radii_s=[float(item) for item in final_mask_radii.tolist()],
            )
        metadata = dict(metadata)
        metadata.update(
            {
                "notches": [float(item) for item in runtime_notches],
                "notch_radii": [float(item) for item in runtime_notch_radii],
                "notch_intervals_hz": [
                    [float(low), float(high)] for low, high in notch_intervals
                ],
                "inherited_filter_notches": [
                    float(item) for item in inheritance.notches
                ],
                "inherited_filter_notch_widths": [
                    float(item) for item in inheritance.notch_widths
                ],
            }
        )
        grid_params = metadata.get("params", {}) if isinstance(metadata, dict) else {}
        if not isinstance(grid_params, dict):
            grid_params = {}
        count_payload = {
            "n_columns_total": int(grid_params.get("n_columns_total", 0)),
            "n_columns_skipped_masked": int(
                grid_params.get("n_columns_skipped_masked", 0)
            ),
            "window_group_counts": list(grid_params.get("window_group_counts", [])),
        }
        if hasattr(raw, "close"):
            raw.close()

        config_payload = {
            "metric_key": metric_key,
            "metric_label": TENSOR_METRICS_BY_KEY[metric_key].display_name,
            "connectivity_metric": str(connectivity_metric),
            "method": str(method_norm),
            "low_freq": float(low_freq),
            "high_freq": float(applied_high),
            "step_hz": float(step_hz),
            "time_resolution_s": float(time_resolution_s),
            "hop_s": float(hop_s),
            "mt_time_bandwidth_product": float(mt_time_bandwidth_product),
            "mt_min_cycles": float(mt_min_cycles),
            "min_cycles": (float(min_cycles) if min_cycles is not None else None),
            "max_cycles": (float(max_cycles) if max_cycles is not None else None),
            "mask_edge_effects": bool(mask_edge_effects),
            "bands": bands,
            "channels": picks,
            "selected_channels": picks,
            "selected_pairs": [[str(a), str(b)] for a, b in pairs],
            "pairs": [[str(a), str(b)] for a, b in pairs],
            "freqs_compute": [float(item) for item in freqs_compute.tolist()],
            "freqs_full": [float(item) for item in freqs_full.tolist()],
            "notches": [float(item) for item in runtime_notches],
            "notch_radii": [float(item) for item in runtime_notch_radii],
            "inherited_filter_notches": [float(item) for item in inheritance.notches],
            "inherited_filter_notch_widths": [
                float(item) for item in inheritance.notch_widths
            ],
            "notch_intervals_hz": [
                [float(lo), float(hi)] for lo, hi in notch_intervals
            ],
            "interpolation_applied": bool(interpolation_applied),
            "tensor_shape": [int(item) for item in tensor4d.shape],
            **count_payload,
            **_effective_n_jobs_payload(
                n_jobs=int(n_jobs),
                outer_n_jobs=int(outer_n_jobs),
            ),
        }
        success_message = (
            f"{metric_label} tensor computed (with notch interpolation)."
            if interpolation_applied
            else f"{metric_label} tensor computed."
        )
        log_params = {
            "low_freq": float(low_freq),
            "high_freq": float(applied_high),
            "step_hz": float(step_hz),
            "time_resolution_s": float(time_resolution_s),
            "hop_s": float(hop_s),
            "connectivity_metric": str(connectivity_metric),
            "method": str(method_norm),
            "mt_time_bandwidth_product": float(mt_time_bandwidth_product),
            "mt_min_cycles": float(mt_min_cycles),
            "min_cycles": (float(min_cycles) if min_cycles is not None else None),
            "max_cycles": (float(max_cycles) if max_cycles is not None else None),
            "mask_edge_effects": bool(mask_edge_effects),
            "notches": [float(item) for item in runtime_notches],
            "notch_radii": [float(item) for item in runtime_notch_radii],
            "inherited_filter_notches": [float(item) for item in inheritance.notches],
            "inherited_filter_notch_widths": [
                float(item) for item in inheritance.notch_widths
            ],
            "interpolation_applied": bool(interpolation_applied),
            "n_channels": len(picks),
            "n_pairs": len(pairs),
            "selected_channels": picks,
            "selected_pairs": [[str(a), str(b)] for a, b in pairs],
            "n_freqs": int(tensor4d.shape[2]),
            "n_times": int(tensor4d.shape[3]),
            **count_payload,
            **_effective_n_jobs_payload(
                n_jobs=int(n_jobs),
                outer_n_jobs=int(outer_n_jobs),
            ),
        }
        _write_outputs_atomically(
            [
                (
                    output_path,
                    lambda path: save_pkl(
                        {"tensor": tensor4d, "meta": metadata},
                        path,
                    ),
                ),
                (config_path, lambda path: _write_metric_config(path, config_payload)),
                (
                    log_path,
                    lambda path: _write_metric_log_to_path(
                        path,
                        metric_key,
                        completed=True,
                        params=log_params,
                        input_path=str(input_path),
                        output_path=str(output_path),
                        message=success_message,
                    ),
                ),
            ]
        )
        return (
            True,
            (
                f"{metric_label} tensor computed with notch interpolation."
                if interpolation_applied
                else f"{metric_label} tensor computed."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _write_metric_log(
            resolver,
            metric_key,
            completed=False,
            params={
                "low_freq": float(low_freq),
                "high_freq": float(high_freq),
                "step_hz": float(step_hz),
                "time_resolution_s": float(time_resolution_s),
                "hop_s": float(hop_s),
                "connectivity_metric": str(connectivity_metric),
                "method": str(method),
                "mt_time_bandwidth_product": float(mt_time_bandwidth_product),
                "mt_min_cycles": float(mt_min_cycles),
                "min_cycles": (float(min_cycles) if min_cycles is not None else None),
                "max_cycles": (float(max_cycles) if max_cycles is not None else None),
                "mask_edge_effects": bool(mask_edge_effects),
                "notches": [float(item) for item in runtime_notches],
                "notch_radii": [float(item) for item in runtime_notch_radii],
                "inherited_filter_notches": [
                    float(item) for item in inheritance.notches
                ],
                "inherited_filter_notch_widths": [
                    float(item) for item in inheritance.notch_widths
                ],
                "selected_channels": [str(item) for item in (selected_channels or [])],
                "selected_pairs": (
                    [[str(a), str(b)] for a, b in selected_pairs]
                    if selected_pairs is not None
                    else []
                ),
                **_effective_n_jobs_payload(
                    n_jobs=int(n_jobs),
                    outer_n_jobs=int(outer_n_jobs),
                ),
            },
            input_path=str(input_path),
            output_path=str(output_path),
            message=f"{metric_label} failed: {exc}",
        )
        return False, f"{metric_label} failed: {exc}"
