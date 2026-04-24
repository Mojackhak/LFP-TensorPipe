"""Burst metric runner."""

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
        serialized[name] = [
            [float(segment[0]), float(segment[1])] for segment in list(value)
        ]
    return serialized


def _resolve_burst_time_grid(
    *,
    hop_s: float | None,
    decim: int | None,
) -> tuple[float | None, int | None]:
    hop_s_use = float(hop_s) if hop_s is not None else None
    if decim is not None:
        return hop_s_use, int(decim)
    if hop_s_use is None:
        return None, 1
    return hop_s_use, None


def run_burst_metric(
    context: RecordContext,
    *,
    low_freq: float,
    high_freq: float,
    step_hz: float,
    mask_edge_effects: bool,
    bands: list[dict[str, Any]],
    selected_channels: list[str] | None,
    percentile: float = 75.0,
    baseline_keep: list[str] | None = None,
    min_cycles: float = 2.0,
    max_cycles: float | None = None,
    hop_s: float | None = None,
    decim: int | None = None,
    thresholds: Any = None,
    thresholds_source_path: str | None = None,
    notches: Any = None,
    notch_widths: Any = 2.0,
    n_jobs: int = 1,
    outer_n_jobs: int = 1,
    read_raw_fif_fn=None,
    burst_grid_fn=None,
    compute_notch_intervals_fn=None,
) -> tuple[bool, str]:
    if burst_grid_fn is None:
        from lfptensorpipe.lfp.burst.grid import grid as burst_grid
    else:
        burst_grid = burst_grid_fn

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
    _effective_n_jobs_payload = svc._effective_n_jobs_payload
    save_pkl = svc.save_pkl
    TENSOR_METRICS_BY_KEY = svc.TENSOR_METRICS_BY_KEY

    resolver = PathResolver(context)
    metric_key = "burst"
    metric_label = TENSOR_METRICS_BY_KEY[metric_key].display_name
    metric_dir = resolver.tensor_metric_dir(metric_key, create=True)
    input_path = preproc_step_raw_path(resolver, "finish")
    output_path = tensor_metric_tensor_path(resolver, metric_key, create=True)
    config_path = tensor_metric_config_path(resolver, metric_key, create=True)
    log_path = tensor_metric_log_path(resolver, metric_key, create=True)
    thresholds_artifact_path = metric_dir / "thresholds.pkl"
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
    hop_s_use, decim_use = _resolve_burst_time_grid(hop_s=hop_s, decim=decim)

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
        picks = [
            name
            for name in (selected_channels or raw.ch_names)
            if name in available_channels
        ]
        if not picks:
            raise ValueError(f"{metric_label} requires at least 1 valid channel.")

        burst_bands = _build_runtime_bands(
            bands=bands,
            low_freq=float(low_freq),
            high_freq=float(high_freq),
            notch_intervals=notch_intervals,
        )
        if not burst_bands:
            raise ValueError(
                "No valid burst band intersects current frequency range. "
                "Adjust bands or low/high frequency limits."
            )

        tensor, metadata = burst_grid(
            raw,
            bands=burst_bands,
            thresholds=thresholds,
            percentile=float(percentile),
            baseline_keep=baseline_keep,
            baseline_match="exact",
            min_cycles=float(min_cycles),
            hop_s=hop_s_use,
            decim=decim_use,
            picks=picks,
        )
        tensor4d = np.asarray(tensor, dtype=float)
        if tensor4d.ndim == 3:
            tensor4d = tensor4d[None, ...]
        if tensor4d.ndim != 4:
            raise ValueError(
                f"Unexpected {metric_label} tensor shape: {tensor4d.shape}"
            )

        written_thresholds = None
        if isinstance(metadata, dict):
            qc = metadata.get("qc", {})
            if isinstance(qc, dict):
                written_thresholds = qc.get("thresholds")

        thresholds_payload_path: str | None = None
        if written_thresholds is not None:
            thresholds_payload_path = str(thresholds_artifact_path)
        elif isinstance(thresholds_source_path, str) and thresholds_source_path.strip():
            thresholds_payload_path = thresholds_source_path.strip()

        config_payload = {
            "metric_key": metric_key,
            "metric_label": metric_label,
            "method": "burst_grid",
            "low_freq": float(low_freq),
            "high_freq": float(high_freq),
            "step_hz": float(step_hz),
            "percentile": float(percentile),
            "baseline_keep": list(baseline_keep) if baseline_keep is not None else None,
            "baseline_match": "exact",
            "min_cycles": float(min_cycles),
            "max_cycles": (float(max_cycles) if max_cycles is not None else None),
            "hop_s": hop_s_use,
            "decim": decim_use,
            "mask_edge_effects": bool(mask_edge_effects),
            "bands": bands,
            "bands_used": _serialize_runtime_bands(burst_bands),
            "channels": picks,
            "selected_channels": picks,
            "thresholds_source_path": (
                str(thresholds_source_path).strip()
                if isinstance(thresholds_source_path, str)
                and thresholds_source_path.strip()
                else None
            ),
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
            "thresholds_path": thresholds_payload_path,
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
            "percentile": float(percentile),
            "baseline_keep": list(baseline_keep) if baseline_keep is not None else None,
            "baseline_match": "exact",
            "min_cycles": float(min_cycles),
            "max_cycles": (float(max_cycles) if max_cycles is not None else None),
            "hop_s": hop_s_use,
            "decim": decim_use,
            "mask_edge_effects": bool(mask_edge_effects),
            "thresholds_source_path": (
                str(thresholds_source_path).strip()
                if isinstance(thresholds_source_path, str)
                and thresholds_source_path.strip()
                else None
            ),
            "notches": [float(item) for item in runtime_notches],
            "notch_widths": [float(item) for item in runtime_notch_widths],
            "inherited_filter_notches": [float(item) for item in inheritance.notches],
            "inherited_filter_notch_widths": [
                float(item) for item in inheritance.notch_widths
            ],
            "bands_used": _serialize_runtime_bands(burst_bands),
            "interpolation_applied": False,
            "n_channels": len(picks),
            "selected_channels": picks,
            "n_bands": int(tensor4d.shape[2]),
            "n_times": int(tensor4d.shape[3]),
            "thresholds_written": bool(written_thresholds is not None),
            **_effective_n_jobs_payload(
                n_jobs=int(n_jobs),
                outer_n_jobs=int(outer_n_jobs),
            ),
        }
        outputs = [
            (
                output_path,
                lambda path: save_pkl({"tensor": tensor4d, "meta": metadata}, path),
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
        if written_thresholds is not None:
            outputs.append(
                (
                    thresholds_artifact_path,
                    lambda path: save_pkl(
                        np.asarray(written_thresholds, dtype=float),
                        path,
                    ),
                )
            )
        _write_outputs_atomically(outputs)
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
                "percentile": float(percentile),
                "baseline_keep": (
                    list(baseline_keep) if baseline_keep is not None else None
                ),
                "baseline_match": "exact",
                "min_cycles": float(min_cycles),
                "max_cycles": (float(max_cycles) if max_cycles is not None else None),
                "hop_s": hop_s_use,
                "decim": decim_use,
                "mask_edge_effects": bool(mask_edge_effects),
                "thresholds_source_path": (
                    str(thresholds_source_path).strip()
                    if isinstance(thresholds_source_path, str)
                    and thresholds_source_path.strip()
                    else None
                ),
                "notches": [float(item) for item in runtime_notches],
                "notch_widths": [float(item) for item in runtime_notch_widths],
                "inherited_filter_notches": [
                    float(item) for item in inheritance.notches
                ],
                "inherited_filter_notch_widths": [
                    float(item) for item in inheritance.notch_widths
                ],
                "selected_channels": [str(item) for item in (selected_channels or [])],
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


__all__ = ["run_burst_metric"]
