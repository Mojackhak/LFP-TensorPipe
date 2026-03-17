"""TRGC connectivity tensor runners."""

from __future__ import annotations

from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import (
    preproc_step_log_path,
    preproc_step_raw_path,
)
from lfptensorpipe.app.runlog_store import indicator_from_log
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl

from ..atomic_io import write_outputs_atomically as _write_outputs_atomically
from ..coercion import _normalize_metric_method
from ..frequency import (
    _apply_dynamic_edge_mask_strict,
    _build_frequency_grid,
    _compute_mask_radii_seconds,
    _compute_notch_intervals,
    _cut_frequency_grid_by_intervals,
    _effective_n_jobs_payload,
    _expand_notch_widths,
    load_tensor_filter_inheritance,
    build_tensor_metric_notch_payload,
)
from ..logging import (
    write_metric_config as _write_metric_config,
    write_metric_log as _write_metric_log,
    write_metric_log_to_path as _write_metric_log_to_path,
)
from ..params import TENSOR_METRICS_BY_KEY
from ..paths import (
    tensor_metric_config_path,
    tensor_metric_log_path,
    tensor_metric_tensor_path,
)
from ..selectors import normalize_selected_pairs

TRGC_GC_BACKEND_PLAN_KEY = "trgc_gc_backend"
TRGC_GC_TR_BACKEND_PLAN_KEY = "trgc_gc_tr_backend"
TRGC_FINALIZE_PLAN_KEY = "trgc"
TRGC_BACKEND_METHODS = ("gc", "gc_tr")
_TRGC_NUMPY_DET_WARNING_MESSAGE = r"invalid value encountered in det"
_TRGC_NUMPY_DET_WARNING_MODULE = r"numpy\.linalg\._linalg"


def _normalize_backend_tensor(
    tensor: np.ndarray,
    *,
    metric_label: str,
    backend_method: str,
) -> np.ndarray:
    tensor4d = np.asarray(tensor, dtype=float)
    if tensor4d.ndim == 3:
        tensor4d = tensor4d[None, ...]
    if tensor4d.ndim != 4:
        raise ValueError(
            f"Unexpected {metric_label} tensor shape from backend '{backend_method}': {tensor4d.shape}"
        )
    return tensor4d


def _normalize_pair_axis(
    metadata: dict[str, Any],
    *,
    expected_size: int,
    metric_label: str,
    backend_method: str,
) -> list[tuple[str, str]]:
    pair_axis = (metadata.get("axes", {}) or {}).get("channel", [])
    pair_names: list[tuple[str, str]] = []
    for item in list(pair_axis):
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"{metric_label} backend '{backend_method}' returned invalid directed pair metadata: {item!r}"
            )
        pair_names.append((str(item[0]), str(item[1])))
    if len(pair_names) != int(expected_size):
        raise ValueError(
            f"{metric_label} backend '{backend_method}' pair axis length mismatch: "
            f"{len(pair_names)} != {int(expected_size)}."
        )
    return pair_names


def _close_directed_pairs(
    requested_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    pairs_compute: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for seed, target in requested_pairs:
        pair = (str(seed), str(target))
        if pair not in seen:
            seen.add(pair)
            pairs_compute.append(pair)
    for seed, target in requested_pairs:
        recip = (str(target), str(seed))
        if recip not in seen:
            seen.add(recip)
            pairs_compute.append(recip)
    return pairs_compute


def _serialize_pairs(pairs: list[tuple[str, str]]) -> list[list[str]]:
    return [[str(seed), str(target)] for seed, target in pairs]


def _select_pair_axis(
    tensor: np.ndarray,
    metadata: dict[str, Any],
    *,
    pair_names_full: list[tuple[str, str]],
    pair_names_keep: list[tuple[str, str]],
) -> tuple[np.ndarray, dict[str, Any]]:
    idx_of = {pair: idx for idx, pair in enumerate(pair_names_full)}
    missing = [pair for pair in pair_names_keep if pair not in idx_of]
    if missing:
        raise ValueError(f"Missing directed pairs in TRGC backend output: {missing}")
    keep_idx = np.asarray([idx_of[pair] for pair in pair_names_keep], dtype=int)
    tensor_keep = np.take(np.asarray(tensor, dtype=float), keep_idx, axis=1)

    metadata_keep = deepcopy(metadata)
    axes = dict(metadata_keep.get("axes", {}) or {})
    axes["channel"] = list(pair_names_keep)
    axes["shape"] = tensor_keep.shape
    metadata_keep["axes"] = axes
    return tensor_keep, metadata_keep


def _derive_trgc_tensor(
    gc_tensor: np.ndarray,
    gc_metadata: dict[str, Any],
    gc_tr_tensor: np.ndarray,
    gc_tr_metadata: dict[str, Any],
    *,
    requested_pairs: list[tuple[str, str]],
    compute_pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, dict[str, Any]]:
    from lfptensorpipe.lfp.connectivity.utils import (
        swap_reciprocal_pairs_on_channel_axis,
    )

    pair_names_gc = _normalize_pair_axis(
        gc_metadata,
        expected_size=gc_tensor.shape[1],
        metric_label="TRGC",
        backend_method="gc",
    )
    pair_names_gc_tr = _normalize_pair_axis(
        gc_tr_metadata,
        expected_size=gc_tr_tensor.shape[1],
        metric_label="TRGC",
        backend_method="gc_tr",
    )
    if pair_names_gc != pair_names_gc_tr:
        raise ValueError(
            "TRGC backend pair axes do not match between 'gc' and 'gc_tr'."
        )
    if pair_names_gc != compute_pairs:
        raise ValueError(
            "TRGC backend pair axis does not match the requested compute pair order."
        )
    if gc_tensor.shape != gc_tr_tensor.shape:
        raise ValueError(
            f"TRGC backend tensor shape mismatch: {gc_tensor.shape} != {gc_tr_tensor.shape}."
        )

    net_gc = gc_tensor - swap_reciprocal_pairs_on_channel_axis(gc_tensor, pair_names_gc)
    net_gc_tr = gc_tr_tensor - swap_reciprocal_pairs_on_channel_axis(
        gc_tr_tensor, pair_names_gc
    )
    trgc_tensor = np.asarray(net_gc - net_gc_tr, dtype=float)

    trgc_tensor, trgc_metadata = _select_pair_axis(
        trgc_tensor,
        gc_metadata,
        pair_names_full=pair_names_gc,
        pair_names_keep=requested_pairs,
    )
    params = dict(trgc_metadata.get("params", {}) or {})
    params.update(
        {
            "method": "trgc",
            "method_internal": "trgc",
            "time_reversed": False,
            "connectivity_metric": "trgc",
            "backend_methods": ["gc", "gc_tr"],
            "pairs_requested": list(requested_pairs),
            "pairs_compute": list(compute_pairs),
        }
    )
    trgc_metadata["params"] = params
    return trgc_tensor, trgc_metadata


def _validate_backend_method(backend_method: str) -> str:
    normalized = str(backend_method)
    if normalized not in TRGC_BACKEND_METHODS:
        raise ValueError(f"Unsupported TRGC backend method: {backend_method!r}")
    return normalized


def _trgc_metric_dir(resolver: Any, *, create: bool = False) -> Path:
    return tensor_metric_tensor_path(resolver, "trgc", create=create).parent


def _trgc_backend_tensor_path(
    resolver: Any, backend_method: str, *, create: bool = False
) -> Path:
    normalized = _validate_backend_method(backend_method)
    metric_dir = _trgc_metric_dir(resolver, create=create)
    return metric_dir / f"_{normalized}_backend.pkl"


def _trgc_backend_log_path(
    resolver: Any, backend_method: str, *, create: bool = False
) -> Path:
    normalized = _validate_backend_method(backend_method)
    metric_dir = _trgc_metric_dir(resolver, create=create)
    return metric_dir / f"_{normalized}_backend_log.json"


def _prepare_trgc_backend_inputs(
    context: RecordContext,
    *,
    low_freq: float,
    high_freq: float,
    step_hz: float,
    bands: list[dict[str, Any]],
    selected_channels: list[str] | None,
    selected_pairs: list[tuple[str, str]] | None,
    time_resolution_s: float,
    hop_s: float,
    method: str,
    mt_bandwidth: float | None,
    min_cycles: float | None,
    max_cycles: float | None,
    gc_n_lags: int,
    group_by_samples: bool,
    round_ms: float,
    notches: Any = None,
    notch_widths: Any = 2.0,
    read_raw_fif_fn=None,
    normalize_selected_pairs_fn=None,
    compute_notch_intervals_fn=None,
) -> dict[str, Any]:
    resolver = PathResolver(context)
    metric_label = TENSOR_METRICS_BY_KEY["trgc"].display_name
    input_path = preproc_step_raw_path(resolver, "finish")
    inheritance = load_tensor_filter_inheritance(context)
    runtime_notch_payload = build_tensor_metric_notch_payload(
        notches,
        notch_widths,
    )
    runtime_notches = tuple(float(item) for item in runtime_notch_payload["notches"])
    runtime_notch_widths = _expand_notch_widths(
        runtime_notch_payload["notch_widths"],
        len(runtime_notches),
    )

    if indicator_from_log(preproc_step_log_path(resolver, "finish")) != "green":
        raise ValueError("Missing green preproc finish log.")
    if not input_path.exists():
        raise ValueError("Missing preproc finish raw input.")

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
        pairs_requested = list(permutations(picks, 2))
    else:
        pairs_requested = (normalize_selected_pairs_fn or normalize_selected_pairs)(
            selected_pairs,
            available_channels=available_channels,
            directed=True,
        )
        if not pairs_requested:
            raise ValueError(
                f"No valid selected directed pairs available for {metric_label}."
            )
        picks = [
            name
            for name in raw.ch_names
            if any(name in pair for pair in pairs_requested)
        ]
        if len(picks) < 2:
            raise ValueError(f"{metric_label} requires at least 2 valid channels.")

    pairs_compute = _close_directed_pairs(pairs_requested)
    nyquist = float(raw.info["sfreq"]) / 2.0
    applied_high = min(float(high_freq), nyquist)
    if applied_high <= float(low_freq):
        raise ValueError("High frequency is invalid for current Nyquist frequency.")

    freqs_full = _build_frequency_grid(low_freq, applied_high, step_hz)
    notch_intervals = (compute_notch_intervals_fn or _compute_notch_intervals)(
        low_freq=low_freq,
        high_freq=applied_high,
        notches=runtime_notches,
        notch_widths=runtime_notch_widths,
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
                    "Notch exclusion removed too many bins; relax notch widths or frequency range."
                )
            interpolation_applied = True

    method_norm = _normalize_metric_method(method, metric_label=metric_label)
    spectral_mode_use = "cwt_morlet" if method_norm == "morlet" else "multitaper"
    return {
        "resolver": resolver,
        "metric_label": metric_label,
        "input_path": input_path,
        "raw": raw,
        "bands": list(bands),
        "picks": picks,
        "pairs_requested": pairs_requested,
        "pairs_compute": pairs_compute,
        "applied_high": float(applied_high),
        "freqs_full": np.asarray(freqs_full, dtype=float),
        "freqs_compute": np.asarray(freqs_compute, dtype=float),
        "notch_intervals": [(float(lo), float(hi)) for lo, hi in notch_intervals],
        "interpolation_applied": bool(interpolation_applied),
        "inheritance": inheritance,
        "runtime_notches": runtime_notches,
        "runtime_notch_widths": runtime_notch_widths,
        "method_norm": str(method_norm),
        "spectral_mode_use": str(spectral_mode_use),
        "time_resolution_s": float(time_resolution_s),
        "hop_s": float(hop_s),
        "mt_bandwidth": mt_bandwidth,
        "min_cycles": min_cycles,
        "max_cycles": max_cycles,
        "gc_n_lags": int(gc_n_lags),
        "group_by_samples": bool(group_by_samples),
        "round_ms": float(round_ms),
        "low_freq": float(low_freq),
        "step_hz": float(step_hz),
        "selected_channels_input": [str(item) for item in (selected_channels or [])],
    }


def _compute_trgc_backend_tensor(
    prepared: dict[str, Any],
    *,
    backend_method: str,
    conn_grid_fn=None,
    interpolate_freq_tensor_fn=None,
) -> tuple[np.ndarray, dict[str, Any]]:
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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_TRGC_NUMPY_DET_WARNING_MESSAGE,
            category=RuntimeWarning,
            module=_TRGC_NUMPY_DET_WARNING_MODULE,
        )
        backend_tensor, backend_metadata = conn_grid(
            prepared["raw"],
            freqs=prepared["freqs_compute"],
            time_resolution_s=float(prepared["time_resolution_s"]),
            hop_s=float(prepared["hop_s"]),
            pairs=prepared["pairs_compute"],
            method=str(backend_method),
            multivariate=True,
            ordered_pairs=True,
            spectral_mode=str(prepared["spectral_mode_use"]),
            mt_bandwidth=prepared["mt_bandwidth"],
            min_cycles=prepared["min_cycles"],
            max_cycles=prepared["max_cycles"],
            gc_n_lags=int(prepared["gc_n_lags"]),
            group_by_samples=bool(prepared["group_by_samples"]),
            round_ms=float(prepared["round_ms"]),
            picks=prepared["picks"],
            outer_n_jobs=1,
        )
    backend_tensor4d = _normalize_backend_tensor(
        backend_tensor,
        metric_label=prepared["metric_label"],
        backend_method=str(backend_method),
    )
    if prepared["interpolation_applied"]:
        backend_tensor4d, backend_metadata = interpolate_freq_tensor(
            backend_tensor4d,
            backend_metadata,
            freqs_out=prepared["freqs_full"],
            axis=-2,
            method="linear",
            transform_mode=None,
        )
        backend_tensor4d = np.asarray(backend_tensor4d, dtype=float)
    return backend_tensor4d, backend_metadata


def _build_trgc_backend_state(
    prepared: dict[str, Any],
    *,
    backend_method: str,
    n_jobs: int,
) -> dict[str, Any]:
    inheritance = prepared["inheritance"]
    runtime_notches = prepared["runtime_notches"]
    runtime_notch_widths = prepared["runtime_notch_widths"]
    return {
        "metric_key": "trgc",
        "metric_label": prepared["metric_label"],
        "backend_method": str(backend_method),
        "connectivity_metric": "trgc",
        "method": str(prepared["method_norm"]),
        "backend_methods": list(TRGC_BACKEND_METHODS),
        "low_freq": float(prepared["low_freq"]),
        "high_freq": float(prepared["applied_high"]),
        "step_hz": float(prepared["step_hz"]),
        "time_resolution_s": float(prepared["time_resolution_s"]),
        "hop_s": float(prepared["hop_s"]),
        "mt_bandwidth": (
            float(prepared["mt_bandwidth"])
            if prepared["mt_bandwidth"] is not None
            else None
        ),
        "min_cycles": (
            float(prepared["min_cycles"])
            if prepared["min_cycles"] is not None
            else None
        ),
        "max_cycles": (
            float(prepared["max_cycles"])
            if prepared["max_cycles"] is not None
            else None
        ),
        "group_by_samples": bool(prepared["group_by_samples"]),
        "round_ms": float(prepared["round_ms"]),
        "bands": list(prepared["bands"]),
        "channels": list(prepared["picks"]),
        "selected_channels": list(prepared["picks"]),
        "selected_pairs": _serialize_pairs(prepared["pairs_requested"]),
        "pairs": _serialize_pairs(prepared["pairs_requested"]),
        "pairs_compute": _serialize_pairs(prepared["pairs_compute"]),
        "gc_n_lags": int(prepared["gc_n_lags"]),
        "freqs_compute": [float(item) for item in prepared["freqs_compute"].tolist()],
        "freqs_full": [float(item) for item in prepared["freqs_full"].tolist()],
        "notches": [float(item) for item in runtime_notches],
        "notch_widths": [float(item) for item in runtime_notch_widths],
        "inherited_filter_notches": [float(item) for item in inheritance.notches],
        "inherited_filter_notch_widths": [
            float(item) for item in inheritance.notch_widths
        ],
        "notch_intervals_hz": [
            [float(lo), float(hi)] for lo, hi in prepared["notch_intervals"]
        ],
        "interpolation_applied": bool(prepared["interpolation_applied"]),
        "n_pairs": len(prepared["pairs_requested"]),
        "n_pairs_compute": len(prepared["pairs_compute"]),
        "execution_model": "trgc_backend_plan",
        **_effective_n_jobs_payload(
            n_jobs=int(n_jobs),
            outer_n_jobs=1,
        ),
    }


def _load_trgc_backend_payload(resolver: Any, backend_method: str) -> dict[str, Any]:
    backend_path = _trgc_backend_tensor_path(resolver, backend_method)
    if not backend_path.exists():
        raise FileNotFoundError(f"Missing TRGC backend artifact: {backend_path}")
    payload = load_pkl(backend_path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid TRGC backend artifact payload type for {backend_method}: {type(payload)!r}"
        )
    if "tensor" not in payload or "meta" not in payload or "state" not in payload:
        raise ValueError(
            f"TRGC backend artifact for {backend_method} is missing required keys."
        )
    return payload


def _write_trgc_public_failure_log(
    context: RecordContext,
    *,
    message: str,
    mask_edge_effects: bool,
    group_by_samples: bool,
    round_ms: float,
    n_jobs: int,
    outer_n_jobs: int,
) -> None:
    resolver = PathResolver(context)
    _write_metric_log(
        resolver,
        "trgc",
        completed=False,
        params={
            "connectivity_metric": "trgc",
            "mask_edge_effects": bool(mask_edge_effects),
            "group_by_samples": bool(group_by_samples),
            "round_ms": float(round_ms),
            "execution_model": "trgc_one_shot_wrapper",
            **_effective_n_jobs_payload(
                n_jobs=int(n_jobs),
                outer_n_jobs=int(outer_n_jobs),
            ),
        },
        input_path=str(preproc_step_raw_path(resolver, "finish")),
        output_path=str(tensor_metric_tensor_path(resolver, "trgc", create=True)),
        message=str(message),
    )


def run_trgc_backend_metric(
    context: RecordContext,
    *,
    backend_method: str,
    low_freq: float,
    high_freq: float,
    step_hz: float,
    bands: list[dict[str, Any]],
    selected_channels: list[str] | None,
    selected_pairs: list[tuple[str, str]] | None,
    time_resolution_s: float = 0.5,
    hop_s: float = 0.025,
    method: str = "morlet",
    mt_bandwidth: float | None = None,
    min_cycles: float | None = 3.0,
    max_cycles: float | None = None,
    gc_n_lags: int = 20,
    group_by_samples: bool = False,
    round_ms: float = 50.0,
    notches: Any = None,
    notch_widths: Any = 2.0,
    n_jobs: int = 1,
    outer_n_jobs: int = 1,
    read_raw_fif_fn=None,
    normalize_selected_pairs_fn=None,
    compute_notch_intervals_fn=None,
    conn_grid_fn=None,
    interpolate_freq_tensor_fn=None,
) -> tuple[bool, str]:
    normalized_backend = _validate_backend_method(backend_method)
    plan_key = (
        TRGC_GC_BACKEND_PLAN_KEY
        if normalized_backend == "gc"
        else TRGC_GC_TR_BACKEND_PLAN_KEY
    )
    prepared: dict[str, Any] | None = None
    output_path = None
    log_path = None
    runtime_notch_payload = build_tensor_metric_notch_payload(notches, notch_widths)
    runtime_notches = [float(item) for item in runtime_notch_payload["notches"]]
    runtime_notch_widths = list(
        _expand_notch_widths(
            runtime_notch_payload["notch_widths"],
            len(runtime_notches),
        )
    )
    inheritance = load_tensor_filter_inheritance(context)
    try:
        prepared = _prepare_trgc_backend_inputs(
            context,
            low_freq=low_freq,
            high_freq=high_freq,
            step_hz=step_hz,
            bands=bands,
            selected_channels=selected_channels,
            selected_pairs=selected_pairs,
            time_resolution_s=time_resolution_s,
            hop_s=hop_s,
            method=method,
            mt_bandwidth=mt_bandwidth,
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            gc_n_lags=gc_n_lags,
            group_by_samples=group_by_samples,
            round_ms=round_ms,
            notches=notches,
            notch_widths=notch_widths,
            read_raw_fif_fn=read_raw_fif_fn,
            normalize_selected_pairs_fn=normalize_selected_pairs_fn,
            compute_notch_intervals_fn=compute_notch_intervals_fn,
        )
        resolver = prepared["resolver"]
        output_path = _trgc_backend_tensor_path(
            resolver, normalized_backend, create=True
        )
        log_path = _trgc_backend_log_path(resolver, normalized_backend, create=True)
        tensor4d, metadata = _compute_trgc_backend_tensor(
            prepared,
            backend_method=normalized_backend,
            conn_grid_fn=conn_grid_fn,
            interpolate_freq_tensor_fn=interpolate_freq_tensor_fn,
        )
        artifact_payload = {
            "tensor": tensor4d,
            "meta": metadata,
            "state": _build_trgc_backend_state(
                prepared,
                backend_method=normalized_backend,
                n_jobs=int(n_jobs),
            ),
        }
        success_message = (
            f"{prepared['metric_label']} {normalized_backend} backend computed."
        )
        log_params = dict(artifact_payload["state"])
        log_params["tensor_shape"] = [int(item) for item in tensor4d.shape]
        log_params["outer_n_jobs_requested"] = int(outer_n_jobs)
        _write_outputs_atomically(
            [
                (output_path, lambda path: save_pkl(artifact_payload, path)),
                (
                    log_path,
                    lambda path: _write_metric_log_to_path(
                        path,
                        plan_key,
                        completed=True,
                        params=log_params,
                        input_path=str(prepared["input_path"]),
                        output_path=str(output_path),
                        message=success_message,
                    ),
                ),
            ]
        )
        return True, success_message
    except Exception as exc:  # noqa: BLE001
        if prepared is None:
            resolver = PathResolver(context)
            input_path = preproc_step_raw_path(resolver, "finish")
            output_path = _trgc_backend_tensor_path(
                resolver, normalized_backend, create=True
            )
            log_path = _trgc_backend_log_path(resolver, normalized_backend, create=True)
            metric_label = TENSOR_METRICS_BY_KEY["trgc"].display_name
        else:
            resolver = prepared["resolver"]
            input_path = prepared["input_path"]
            metric_label = prepared["metric_label"]
        _write_metric_log_to_path(
            log_path,
            plan_key,
            completed=False,
            params={
                "backend_method": normalized_backend,
                "connectivity_metric": "trgc",
                "group_by_samples": bool(group_by_samples),
                "round_ms": float(round_ms),
                "notches": list(runtime_notches),
                "notch_widths": list(runtime_notch_widths),
                "inherited_filter_notches": [
                    float(item) for item in inheritance.notches
                ],
                "inherited_filter_notch_widths": [
                    float(item) for item in inheritance.notch_widths
                ],
                "n_jobs": int(n_jobs),
                "outer_n_jobs": 1,
                "outer_n_jobs_requested": int(outer_n_jobs),
            },
            input_path=str(input_path),
            output_path=str(output_path),
            message=f"{metric_label} {normalized_backend} backend failed: {exc}",
        )
        return False, f"{metric_label} {normalized_backend} backend failed: {exc}"
    finally:
        if prepared is not None:
            raw = prepared.get("raw")
            if raw is not None and hasattr(raw, "close"):
                raw.close()


def run_trgc_finalize_metric(
    context: RecordContext,
    *,
    mask_edge_effects: bool,
    n_jobs: int = 1,
    outer_n_jobs: int = 1,
) -> tuple[bool, str]:
    resolver = PathResolver(context)
    metric_key = "trgc"
    metric_label = TENSOR_METRICS_BY_KEY[metric_key].display_name
    input_path = preproc_step_raw_path(resolver, "finish")
    output_path = tensor_metric_tensor_path(resolver, metric_key, create=True)
    config_path = tensor_metric_config_path(resolver, metric_key, create=True)
    log_path = tensor_metric_log_path(resolver, metric_key, create=True)
    raw = None
    gc_state: dict[str, Any] = {}
    try:
        gc_payload = _load_trgc_backend_payload(resolver, "gc")
        gc_tr_payload = _load_trgc_backend_payload(resolver, "gc_tr")

        gc_state = dict(gc_payload["state"])
        gc_tr_state = dict(gc_tr_payload["state"])
        if gc_state.get("pairs_compute") != gc_tr_state.get("pairs_compute"):
            raise ValueError(
                "TRGC backend pair metadata mismatch between gc and gc_tr."
            )
        if gc_state.get("selected_pairs") != gc_tr_state.get("selected_pairs"):
            raise ValueError(
                "TRGC selected-pair metadata mismatch between gc and gc_tr."
            )
        if gc_state.get("freqs_full") != gc_tr_state.get("freqs_full"):
            raise ValueError("TRGC frequency metadata mismatch between gc and gc_tr.")

        requested_pairs = [
            (str(seed), str(target))
            for seed, target in list(gc_state.get("selected_pairs", []))
        ]
        compute_pairs = [
            (str(seed), str(target))
            for seed, target in list(gc_state.get("pairs_compute", []))
        ]
        tensor4d, metadata = _derive_trgc_tensor(
            np.asarray(gc_payload["tensor"], dtype=float),
            dict(gc_payload["meta"]),
            np.asarray(gc_tr_payload["tensor"], dtype=float),
            dict(gc_tr_payload["meta"]),
            requested_pairs=requested_pairs,
            compute_pairs=compute_pairs,
        )

        if mask_edge_effects:
            import mne

            if indicator_from_log(preproc_step_log_path(resolver, "finish")) != "green":
                raise ValueError("Missing green preproc finish log.")
            if not input_path.exists():
                raise ValueError("Missing preproc finish raw input.")

            raw = mne.io.read_raw_fif(str(input_path), preload=False, verbose="ERROR")
            freq_axis = np.asarray(
                (metadata.get("axes", {}) or {}).get("freq", []), dtype=float
            ).ravel()
            if freq_axis.size != tensor4d.shape[2]:
                raise ValueError(
                    f"{metric_label} edge mask failed: metadata frequency axis does not match tensor."
                )
            radii = _compute_mask_radii_seconds(
                freq_axis,
                method=str(gc_state["method"]),
                time_resolution_s=float(gc_state["time_resolution_s"]),
                min_cycles=gc_state.get("min_cycles"),
                max_cycles=gc_state.get("max_cycles"),
            )
            tensor4d, metadata = _apply_dynamic_edge_mask_strict(
                raw=raw,
                tensor=tensor4d,
                metadata=metadata,
                metric_label=metric_label,
                freqs_lookup=[float(item) for item in freq_axis.tolist()],
                radii_s=[float(item) for item in radii.tolist()],
            )

        config_payload = {
            "metric_key": metric_key,
            "metric_label": metric_label,
            "connectivity_metric": "trgc",
            "method": str(gc_state["method"]),
            "backend_methods": list(TRGC_BACKEND_METHODS),
            "low_freq": float(gc_state["low_freq"]),
            "high_freq": float(gc_state["high_freq"]),
            "step_hz": float(gc_state["step_hz"]),
            "time_resolution_s": float(gc_state["time_resolution_s"]),
            "hop_s": float(gc_state["hop_s"]),
            "mt_bandwidth": gc_state.get("mt_bandwidth"),
            "min_cycles": gc_state.get("min_cycles"),
            "max_cycles": gc_state.get("max_cycles"),
            "group_by_samples": bool(gc_state.get("group_by_samples", False)),
            "round_ms": float(gc_state.get("round_ms", 50.0)),
            "mask_edge_effects": bool(mask_edge_effects),
            "bands": list(gc_state.get("bands", [])),
            "channels": list(gc_state.get("channels", [])),
            "selected_channels": list(gc_state.get("selected_channels", [])),
            "selected_pairs": list(gc_state.get("selected_pairs", [])),
            "pairs": list(gc_state.get("selected_pairs", [])),
            "pairs_compute": list(gc_state.get("pairs_compute", [])),
            "gc_n_lags": int(gc_state["gc_n_lags"]),
            "freqs_compute": list(gc_state.get("freqs_compute", [])),
            "freqs_full": list(gc_state.get("freqs_full", [])),
            "notches": list(gc_state.get("notches", [])),
            "notch_widths": list(gc_state.get("notch_widths", [])),
            "inherited_filter_notches": list(
                gc_state.get("inherited_filter_notches", [])
            ),
            "inherited_filter_notch_widths": list(
                gc_state.get("inherited_filter_notch_widths", [])
            ),
            "notch_intervals_hz": list(gc_state.get("notch_intervals_hz", [])),
            "interpolation_applied": bool(gc_state.get("interpolation_applied", False)),
            "tensor_shape": [int(item) for item in tensor4d.shape],
            "n_pairs": int(gc_state.get("n_pairs", len(requested_pairs))),
            "n_pairs_compute": int(gc_state.get("n_pairs_compute", len(compute_pairs))),
            "execution_model": "trgc_backend_finalize",
            "runtime_plans": [
                TRGC_GC_BACKEND_PLAN_KEY,
                TRGC_GC_TR_BACKEND_PLAN_KEY,
                TRGC_FINALIZE_PLAN_KEY,
            ],
            **_effective_n_jobs_payload(
                n_jobs=int(n_jobs),
                outer_n_jobs=int(outer_n_jobs),
            ),
        }
        success_message = (
            f"{metric_label} tensor computed (with notch interpolation)."
            if bool(gc_state.get("interpolation_applied", False))
            else f"{metric_label} tensor computed."
        )
        log_params = {
            "low_freq": float(gc_state["low_freq"]),
            "high_freq": float(gc_state["high_freq"]),
            "step_hz": float(gc_state["step_hz"]),
            "time_resolution_s": float(gc_state["time_resolution_s"]),
            "hop_s": float(gc_state["hop_s"]),
            "connectivity_metric": "trgc",
            "method": str(gc_state["method"]),
            "backend_methods": list(TRGC_BACKEND_METHODS),
            "mt_bandwidth": gc_state.get("mt_bandwidth"),
            "min_cycles": gc_state.get("min_cycles"),
            "max_cycles": gc_state.get("max_cycles"),
            "group_by_samples": bool(gc_state.get("group_by_samples", False)),
            "round_ms": float(gc_state.get("round_ms", 50.0)),
            "mask_edge_effects": bool(mask_edge_effects),
            "notches": list(gc_state.get("notches", [])),
            "notch_widths": list(gc_state.get("notch_widths", [])),
            "inherited_filter_notches": list(
                gc_state.get("inherited_filter_notches", [])
            ),
            "inherited_filter_notch_widths": list(
                gc_state.get("inherited_filter_notch_widths", [])
            ),
            "interpolation_applied": bool(gc_state.get("interpolation_applied", False)),
            "n_channels": len(gc_state.get("channels", [])),
            "n_pairs": int(gc_state.get("n_pairs", len(requested_pairs))),
            "n_pairs_compute": int(gc_state.get("n_pairs_compute", len(compute_pairs))),
            "selected_channels": list(gc_state.get("selected_channels", [])),
            "selected_pairs": list(gc_state.get("selected_pairs", [])),
            "pairs_compute": list(gc_state.get("pairs_compute", [])),
            "n_freqs": int(tensor4d.shape[2]),
            "n_times": int(tensor4d.shape[3]),
            "gc_n_lags": int(gc_state["gc_n_lags"]),
            "execution_model": "trgc_backend_finalize",
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
                (
                    config_path,
                    lambda path: _write_metric_config(path, config_payload),
                ),
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
                if bool(gc_state.get("interpolation_applied", False))
                else f"{metric_label} tensor computed."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _write_metric_log(
            resolver,
            metric_key,
            completed=False,
            params={
                "connectivity_metric": "trgc",
                "mask_edge_effects": bool(mask_edge_effects),
                "group_by_samples": bool(gc_state.get("group_by_samples", False)),
                "round_ms": float(gc_state.get("round_ms", 50.0)),
                "execution_model": "trgc_backend_finalize",
                "notches": list(gc_state.get("notches", [])),
                "notch_widths": list(gc_state.get("notch_widths", [])),
                "inherited_filter_notches": list(
                    gc_state.get("inherited_filter_notches", [])
                ),
                "inherited_filter_notch_widths": list(
                    gc_state.get("inherited_filter_notch_widths", [])
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


def run_trgc_metric(
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
    gc_n_lags: int = 20,
    group_by_samples: bool = False,
    round_ms: float = 50.0,
    notches: Any = None,
    notch_widths: Any = 2.0,
    n_jobs: int = 1,
    outer_n_jobs: int = 1,
    read_raw_fif_fn=None,
    normalize_selected_pairs_fn=None,
    compute_notch_intervals_fn=None,
    conn_grid_fn=None,
    interpolate_freq_tensor_fn=None,
) -> tuple[bool, str]:
    ok_gc, message_gc = run_trgc_backend_metric(
        context,
        backend_method="gc",
        low_freq=low_freq,
        high_freq=high_freq,
        step_hz=step_hz,
        bands=bands,
        selected_channels=selected_channels,
        selected_pairs=selected_pairs,
        time_resolution_s=time_resolution_s,
        hop_s=hop_s,
        method=method,
        mt_bandwidth=mt_bandwidth,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        gc_n_lags=gc_n_lags,
        group_by_samples=group_by_samples,
        round_ms=round_ms,
        notches=notches,
        notch_widths=notch_widths,
        n_jobs=n_jobs,
        outer_n_jobs=1,
        read_raw_fif_fn=read_raw_fif_fn,
        normalize_selected_pairs_fn=normalize_selected_pairs_fn,
        compute_notch_intervals_fn=compute_notch_intervals_fn,
        conn_grid_fn=conn_grid_fn,
        interpolate_freq_tensor_fn=interpolate_freq_tensor_fn,
    )
    if not ok_gc:
        _write_trgc_public_failure_log(
            context,
            message=message_gc,
            mask_edge_effects=mask_edge_effects,
            group_by_samples=group_by_samples,
            round_ms=round_ms,
            n_jobs=n_jobs,
            outer_n_jobs=outer_n_jobs,
        )
        return False, message_gc

    ok_gc_tr, message_gc_tr = run_trgc_backend_metric(
        context,
        backend_method="gc_tr",
        low_freq=low_freq,
        high_freq=high_freq,
        step_hz=step_hz,
        bands=bands,
        selected_channels=selected_channels,
        selected_pairs=selected_pairs,
        time_resolution_s=time_resolution_s,
        hop_s=hop_s,
        method=method,
        mt_bandwidth=mt_bandwidth,
        min_cycles=min_cycles,
        max_cycles=max_cycles,
        gc_n_lags=gc_n_lags,
        group_by_samples=group_by_samples,
        round_ms=round_ms,
        notches=notches,
        notch_widths=notch_widths,
        n_jobs=n_jobs,
        outer_n_jobs=1,
        read_raw_fif_fn=read_raw_fif_fn,
        normalize_selected_pairs_fn=normalize_selected_pairs_fn,
        compute_notch_intervals_fn=compute_notch_intervals_fn,
        conn_grid_fn=conn_grid_fn,
        interpolate_freq_tensor_fn=interpolate_freq_tensor_fn,
    )
    if not ok_gc_tr:
        _write_trgc_public_failure_log(
            context,
            message=message_gc_tr,
            mask_edge_effects=mask_edge_effects,
            group_by_samples=group_by_samples,
            round_ms=round_ms,
            n_jobs=n_jobs,
            outer_n_jobs=outer_n_jobs,
        )
        return False, message_gc_tr

    return run_trgc_finalize_metric(
        context,
        mask_edge_effects=mask_edge_effects,
        n_jobs=n_jobs,
        outer_n_jobs=outer_n_jobs,
    )


__all__ = [
    "TRGC_BACKEND_METHODS",
    "TRGC_FINALIZE_PLAN_KEY",
    "TRGC_GC_BACKEND_PLAN_KEY",
    "TRGC_GC_TR_BACKEND_PLAN_KEY",
    "run_trgc_backend_metric",
    "run_trgc_finalize_metric",
    "run_trgc_metric",
]
