"""PSI connectivity tensor runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.utils.freqs import split_bands_by_intervals

from .. import service as svc


def _build_runtime_bands(
    *,
    bands: list[dict[str, Any]],
    low_freq: float,
    high_freq: float,
    notch_intervals: list[tuple[float, float]],
) -> dict[str, Any]:
    runtime_bands: dict[str, tuple[float, float]] = {}
    for band in bands:
        name = str(band.get("name", "")).strip()
        if not name:
            continue
        start = float(band.get("start"))
        end = float(band.get("end"))
        lo = max(start, float(low_freq))
        hi = min(end, float(high_freq))
        if hi <= lo:
            continue
        runtime_bands[name] = (float(lo), float(hi))
    if not runtime_bands:
        return {}
    return split_bands_by_intervals(
        runtime_bands,
        notch_intervals,
        include_edges=True,
        drop_empty=True,
    )


def _serialize_runtime_bands(
    bands: dict[str, Any],
) -> dict[str, list[float] | list[list[float]]]:
    serialized: dict[str, list[float] | list[list[float]]] = {}
    for name, value in bands.items():
        if isinstance(value, tuple) and len(value) == 2:
            serialized[name] = [float(value[0]), float(value[1])]
            continue
        segments = [[float(segment[0]), float(segment[1])] for segment in list(value)]
        serialized[name] = segments
    return serialized


def run_psi_metric(
    context: RecordContext,
    *,
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
    mt_bandwidth: float | None = None,
    min_cycles: float | None = 3.0,
    max_cycles: float | None = None,
    notches: Any = None,
    notch_widths: Any = 2.0,
    n_jobs: int = 1,
    outer_n_jobs: int = 1,
    read_raw_fif_fn=None,
    psi_grid_fn=None,
    normalize_selected_pairs_fn=None,
    compute_notch_intervals_fn=None,
) -> tuple[bool, str]:
    if psi_grid_fn is None:
        from lfptensorpipe.lfp.psi.grid import grid as psi_grid
    else:
        psi_grid = psi_grid_fn

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
    _compute_notch_intervals = (
        compute_notch_intervals_fn or svc._compute_notch_intervals
    )
    _normalize_metric_method = svc._normalize_metric_method
    _apply_dynamic_edge_mask_strict = svc._apply_dynamic_edge_mask_strict
    _psi_band_radii_seconds = svc._psi_band_radii_seconds
    _effective_n_jobs_payload = svc._effective_n_jobs_payload
    _normalize_selected_pairs = (
        normalize_selected_pairs_fn or svc._normalize_selected_pairs
    )
    save_pkl = svc.save_pkl
    TENSOR_METRICS_BY_KEY = svc.TENSOR_METRICS_BY_KEY
    permutations = svc.permutations

    _ = step_hz
    resolver = PathResolver(context)
    metric_key = "psi"
    metric_label = TENSOR_METRICS_BY_KEY[metric_key].display_name
    method_norm = _normalize_metric_method(method, metric_label=metric_label)
    input_path = preproc_step_raw_path(resolver, "finish")
    output_path = tensor_metric_tensor_path(resolver, metric_key, create=True)
    config_path = tensor_metric_config_path(resolver, metric_key, create=True)
    log_path = tensor_metric_log_path(resolver, metric_key, create=True)
    inheritance = load_tensor_filter_inheritance(context)
    runtime_notch_payload = svc.build_tensor_metric_notch_payload(notches, notch_widths)
    runtime_notches = tuple(float(item) for item in runtime_notch_payload["notches"])
    runtime_notch_widths = svc._expand_notch_widths(
        runtime_notch_payload["notch_widths"],
        len(runtime_notches),
    )
    notch_intervals = _compute_notch_intervals(
        low_freq=float(low_freq),
        high_freq=float(high_freq),
        notches=runtime_notches,
        notch_widths=runtime_notch_widths,
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

    raw = None
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
            pairs = list(permutations(picks, 2))
        else:
            pairs = _normalize_selected_pairs(
                selected_pairs,
                available_channels=available_channels,
                directed=True,
            )
            if not pairs:
                raise ValueError(
                    f"No valid selected directed pairs available for {metric_label}."
                )
            picks = [
                name for name in raw.ch_names if any(name in pair for pair in pairs)
            ]
            if len(picks) < 2:
                raise ValueError(f"{metric_label} requires at least 2 valid channels.")

        psi_bands = _build_runtime_bands(
            bands=bands,
            low_freq=float(low_freq),
            high_freq=float(high_freq),
            notch_intervals=notch_intervals,
        )
        if not psi_bands:
            raise ValueError(
                "No valid band intersects current PSI frequency range. "
                "Adjust bands or low/high frequency limits."
            )

        tensor, metadata = psi_grid(
            raw,
            bands=psi_bands,
            method=method_norm,
            time_resolution_s=float(time_resolution_s),
            hop_s=float(hop_s),
            pairs=pairs,
            ordered_pairs=True,
            mt_bandwidth=mt_bandwidth,
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            picks=picks,
            n_jobs=int(n_jobs),
        )

        tensor4d = np.asarray(tensor, dtype=float)
        if tensor4d.ndim == 3:
            tensor4d = tensor4d[None, ...]
        if tensor4d.ndim != 4:
            raise ValueError(
                f"Unexpected {metric_label} tensor shape: {tensor4d.shape}"
            )
        if mask_edge_effects:
            band_names, band_radii = _psi_band_radii_seconds(
                metadata=metadata,
                method=method_norm,
                time_resolution_s=float(time_resolution_s),
                min_cycles=min_cycles,
                max_cycles=max_cycles,
            )
            tensor4d, metadata = _apply_dynamic_edge_mask_strict(
                raw=raw,
                tensor=tensor4d,
                metadata=metadata,
                metric_label=metric_label,
                freqs_lookup=[str(item) for item in band_names],
                radii_s=[float(item) for item in band_radii],
            )
        config_payload = {
            "metric_key": metric_key,
            "metric_label": metric_label,
            "connectivity_metric": "psi",
            "method": str(method_norm),
            "low_freq": float(low_freq),
            "high_freq": float(high_freq),
            "step_hz": float(step_hz),
            "time_resolution_s": float(time_resolution_s),
            "hop_s": float(hop_s),
            "mt_bandwidth": (float(mt_bandwidth) if mt_bandwidth is not None else None),
            "min_cycles": (float(min_cycles) if min_cycles is not None else None),
            "max_cycles": (float(max_cycles) if max_cycles is not None else None),
            "mask_edge_effects": bool(mask_edge_effects),
            "bands": bands,
            "bands_used": _serialize_runtime_bands(psi_bands),
            "channels": picks,
            "selected_channels": picks,
            "selected_pairs": [[str(a), str(b)] for a, b in pairs],
            "pairs": [[str(a), str(b)] for a, b in pairs],
            "notches": [float(item) for item in runtime_notches],
            "notch_widths": [float(item) for item in runtime_notch_widths],
            "inherited_filter_notches": [float(item) for item in inheritance.notches],
            "inherited_filter_notch_widths": [
                float(item) for item in inheritance.notch_widths
            ],
            "notch_intervals_hz": [
                [float(lo), float(hi)] for lo, hi in notch_intervals
            ],
            "interpolation_applied": False,
            "tensor_shape": [int(item) for item in tensor4d.shape],
            **_effective_n_jobs_payload(
                n_jobs=int(n_jobs),
                outer_n_jobs=int(outer_n_jobs),
            ),
        }
        success_message = f"{metric_label} tensor computed."
        log_params = {
            "low_freq": float(low_freq),
            "high_freq": float(high_freq),
            "step_hz": float(step_hz),
            "method": str(method_norm),
            "time_resolution_s": float(time_resolution_s),
            "hop_s": float(hop_s),
            "mt_bandwidth": (float(mt_bandwidth) if mt_bandwidth is not None else None),
            "min_cycles": (float(min_cycles) if min_cycles is not None else None),
            "max_cycles": (float(max_cycles) if max_cycles is not None else None),
            "mask_edge_effects": bool(mask_edge_effects),
            "notches": [float(item) for item in runtime_notches],
            "notch_widths": [float(item) for item in runtime_notch_widths],
            "inherited_filter_notches": [float(item) for item in inheritance.notches],
            "inherited_filter_notch_widths": [
                float(item) for item in inheritance.notch_widths
            ],
            "bands_used": _serialize_runtime_bands(psi_bands),
            "interpolation_applied": False,
            "n_channels": len(picks),
            "n_pairs": len(pairs),
            "selected_channels": picks,
            "selected_pairs": [[str(a), str(b)] for a, b in pairs],
            "n_bands": int(tensor4d.shape[2]),
            "n_times": int(tensor4d.shape[3]),
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
        return True, f"{metric_label} tensor computed."
    except Exception as exc:  # noqa: BLE001
        _write_metric_log(
            resolver,
            metric_key,
            completed=False,
            params={
                "low_freq": float(low_freq),
                "high_freq": float(high_freq),
                "step_hz": float(step_hz),
                "method": str(method),
                "time_resolution_s": float(time_resolution_s),
                "hop_s": float(hop_s),
                "mt_bandwidth": (
                    float(mt_bandwidth) if mt_bandwidth is not None else None
                ),
                "min_cycles": (float(min_cycles) if min_cycles is not None else None),
                "max_cycles": (float(max_cycles) if max_cycles is not None else None),
                "mask_edge_effects": bool(mask_edge_effects),
                "notches": [float(item) for item in runtime_notches],
                "notch_widths": [float(item) for item in runtime_notch_widths],
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
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()


__all__ = ["run_psi_metric"]
