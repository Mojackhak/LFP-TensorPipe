"""ECG-artifact-removal step runtime helper."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import shutil
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

from ..paths import preproc_step_raw_path, write_preproc_step_config

MarkStepFn = Callable[..., Any]
InvalidateFn = Callable[[RecordContext, str], list[Any]]


def _ecg_config_for_method(method: str) -> Any:
    from lfptensorpipe.preproc.ecg_remover import (
        PerceiveConfig,
        SvdConfig,
        TemplateFitConfig,
    )

    config_types = {
        "template": TemplateFitConfig,
        "perceive": PerceiveConfig,
        "svd": SvdConfig,
    }
    config_type = config_types.get(str(method).strip().lower())
    if config_type is None:
        raise ValueError(f"Unknown ECG method: {method}")
    return config_type()


def default_ecg_method_params(method: str) -> dict[str, Any]:
    """Return JSON-safe defaults for one ECG-removal method."""
    normalized_method = str(method).strip().lower()
    params = asdict(_ecg_config_for_method(normalized_method))
    if "peak_height_range" in params:
        lower, upper = params["peak_height_range"]
        params["peak_height_range"] = [
            float(lower),
            None if math.isinf(float(upper)) else float(upper),
        ]
    return params


def default_ecg_params_by_method() -> dict[str, dict[str, Any]]:
    """Return JSON-safe defaults for all supported ECG methods."""
    return {
        method: default_ecg_method_params(method)
        for method in ("template", "perceive", "svd")
    }


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _positive_float(name: str, value: Any) -> float:
    number = _finite_float(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be greater than 0.")
    return number


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1.")
    return int(value)


def _strict_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def normalize_ecg_method_params(
    method: str,
    params: dict[str, Any] | None,
    *,
    base_params: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any], str]:
    """Validate and normalize JSON-safe parameters for one ECG method."""
    normalized_method = str(method).strip().lower()
    try:
        defaults = default_ecg_method_params(normalized_method)
    except ValueError as exc:
        return False, {}, str(exc)

    base = dict(defaults)
    if base_params is not None:
        if not isinstance(base_params, dict):
            return False, defaults, "Base ECG parameters must be a dictionary."
        unknown_base = sorted(set(base_params) - set(defaults))
        if unknown_base:
            return (
                False,
                defaults,
                f"Unknown ECG parameter(s): {', '.join(unknown_base)}",
            )
        base.update(base_params)

    if params is None:
        candidate = base
    elif isinstance(params, dict):
        unknown = sorted(set(params) - set(defaults))
        if unknown:
            return (
                False,
                defaults,
                f"Unknown ECG parameter(s): {', '.join(unknown)}",
            )
        candidate = dict(base)
        candidate.update(params)
    else:
        return False, defaults, "ECG method parameters must be a dictionary."

    try:
        if normalized_method in {"template", "svd"}:
            peak_range = candidate["peak_height_range"]
            if not isinstance(peak_range, (list, tuple)) or len(peak_range) != 2:
                raise ValueError("peak_height_range must contain minimum and maximum.")
            peak_min = _finite_float("peak_height_range minimum", peak_range[0])
            if peak_min < 0.0:
                raise ValueError("peak_height_range minimum must be at least 0.")
            raw_peak_max = peak_range[1]
            if raw_peak_max is None:
                peak_max: float | None = None
            elif (
                isinstance(raw_peak_max, (int, float))
                and not isinstance(raw_peak_max, bool)
                and math.isinf(float(raw_peak_max))
                and float(raw_peak_max) > 0.0
            ):
                peak_max = None
            else:
                peak_max = _finite_float("peak_height_range maximum", raw_peak_max)
                if peak_max <= peak_min:
                    raise ValueError(
                        "peak_height_range maximum must exceed the minimum."
                    )

            force_orientation = candidate["force_orientation"]
            if force_orientation not in {None, "positive", "negative"}:
                raise ValueError(
                    "force_orientation must be null, positive, or negative."
                )

            normalized = {
                "window_ms": _positive_float("window_ms", candidate["window_ms"]),
                "peak_height_range": [peak_min, peak_max],
                "min_interpeak_ms": _positive_float(
                    "min_interpeak_ms", candidate["min_interpeak_ms"]
                ),
                "force_orientation": force_orientation,
                "pre_ms": _positive_float("pre_ms", candidate["pre_ms"]),
                "post_ms": _positive_float("post_ms", candidate["post_ms"]),
                "tail_ms": _positive_float("tail_ms", candidate["tail_ms"]),
                "qrs_duration_ms": _positive_float(
                    "qrs_duration_ms", candidate["qrs_duration_ms"]
                ),
                "pqrst": _strict_bool("pqrst", candidate["pqrst"]),
            }
            if normalized_method == "svd":
                normalized = {
                    "components": _positive_int("components", candidate["components"]),
                    **normalized,
                }
            return True, normalized, ""

        threshold_start_raw = candidate["threshold_start"]
        threshold_step_raw = candidate["threshold_step"]
        if (threshold_start_raw is None) != (threshold_step_raw is None):
            raise ValueError(
                "threshold_start and threshold_step must both be null or numeric."
            )
        if threshold_start_raw is None:
            threshold_start = None
            threshold_step = None
        else:
            threshold_start = _finite_float("threshold_start", threshold_start_raw)
            threshold_step = _finite_float("threshold_step", threshold_step_raw)
            if threshold_step == 0.0:
                raise ValueError("threshold_step cannot be 0.")

        min_bpm = _positive_int("min_bpm", candidate["min_bpm"])
        max_bpm = _positive_int("max_bpm", candidate["max_bpm"])
        if min_bpm >= max_bpm:
            raise ValueError("min_bpm must be smaller than max_bpm.")
        pass_rate = _finite_float("pass_rate", candidate["pass_rate"])
        if not 0.0 < pass_rate < 1.0:
            raise ValueError("pass_rate must be greater than 0 and smaller than 1.")

        normalized = {
            "epoch_length_ms": _positive_float(
                "epoch_length_ms", candidate["epoch_length_ms"]
            ),
            "window_ms": _positive_float("window_ms", candidate["window_ms"]),
            "threshold_v": _positive_float("threshold_v", candidate["threshold_v"]),
            "pad_ms": _positive_float("pad_ms", candidate["pad_ms"]),
            "min_bpm": min_bpm,
            "max_bpm": max_bpm,
            "threshold_start": threshold_start,
            "threshold_step": threshold_step,
            "max_threshold_tries": _positive_int(
                "max_threshold_tries", candidate["max_threshold_tries"]
            ),
            "pass_rate": pass_rate,
            "before_ms": _positive_float("before_ms", candidate["before_ms"]),
            "after_ms": _positive_float("after_ms", candidate["after_ms"]),
            "enforce_max_interval": _strict_bool(
                "enforce_max_interval", candidate["enforce_max_interval"]
            ),
        }
        return True, normalized, ""
    except (KeyError, TypeError, ValueError) as exc:
        return False, defaults, str(exc)


def normalize_ecg_params_by_method(
    params_by_method: Any,
    *,
    base_by_method: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, dict[str, dict[str, Any]], str]:
    """Normalize all ECG method maps, falling back one method at a time."""
    methods = ("template", "perceive", "svd")
    defaults = default_ecg_params_by_method()
    messages: list[str] = []
    if params_by_method is None:
        params_by_method = {}
    elif not isinstance(params_by_method, dict):
        params_by_method = {}
        messages.append("ECG params_by_method must be a dictionary.")

    unknown_methods = sorted(set(params_by_method) - set(methods))
    if unknown_methods:
        messages.append(f"Unknown ECG method(s): {', '.join(unknown_methods)}")

    normalized: dict[str, dict[str, Any]] = {}
    for method in methods:
        base = (
            base_by_method.get(method)
            if isinstance(base_by_method, dict)
            and isinstance(base_by_method.get(method), dict)
            else defaults[method]
        )
        ok, method_params, message = normalize_ecg_method_params(
            method,
            params_by_method.get(method),
            base_params=base,
        )
        if ok:
            normalized[method] = method_params
        else:
            fallback_ok, fallback, _ = normalize_ecg_method_params(method, base)
            normalized[method] = fallback if fallback_ok else defaults[method]
            messages.append(f"{method}: {message}")

    return not messages, normalized, "; ".join(messages)


def ecg_method_runtime_kwargs(
    method: str,
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], dict[str, Any], str]:
    """Return normalized persisted params and runtime kwargs for one method."""
    ok, normalized, message = normalize_ecg_method_params(method, params)
    if not ok:
        return False, normalized, {}, message
    runtime = dict(normalized)
    if "peak_height_range" in runtime:
        peak_min, peak_max = runtime["peak_height_range"]
        runtime["peak_height_range"] = (
            float(peak_min),
            float("inf") if peak_max is None else float(peak_max),
        )
    return True, normalized, runtime, ""


def apply_ecg_step(
    context: RecordContext,
    *,
    source: tuple[str, Path] | None,
    method: str,
    picks: list[str] | tuple[str, ...] | None,
    method_kwargs: dict[str, Any] | None = None,
    ecg_methods: tuple[str, ...],
    mark_preproc_step_fn: MarkStepFn,
    invalidate_downstream_fn: InvalidateFn,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    raw_call_ecgremover_fn: Callable[..., Any] | None = None,
    copy2_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Apply ECG-artifact removal with normalized method parameters."""
    from lfptensorpipe.preproc.ecg_remover import (
        raw_call_ecgremover,
    )

    resolver = PathResolver(context)
    dst = preproc_step_raw_path(resolver, "ecg_artifact_removal")

    if method not in ecg_methods:
        mark_preproc_step_fn(
            resolver=resolver,
            step="ecg_artifact_removal",
            completed=False,
            input_path=str(source[1]) if source is not None else "",
            output_path=str(dst),
            message=f"Unknown ECG method: {method}",
        )
        return False, f"Unknown ECG method: {method}"

    if source is None:
        mark_preproc_step_fn(
            resolver=resolver,
            step="ecg_artifact_removal",
            completed=False,
            input_path="",
            output_path=str(dst),
            message="No valid preprocess input for ECG step.",
        )
        return False, "No valid preprocess input for ECG step."

    source_step, src = source

    try:
        runtime_copy2 = copy2_fn or shutil.copy2
        if read_raw_fif_fn is None:
            import mne

            read_raw_fif_fn = mne.io.read_raw_fif
        runtime_raw_call = raw_call_ecgremover_fn or raw_call_ecgremover

        valid_params, persisted_kwargs, runtime_kwargs, params_message = (
            ecg_method_runtime_kwargs(method, method_kwargs)
        )
        if not valid_params:
            raise ValueError(f"Invalid ECG parameters: {params_message}")

        selected_picks = list(picks) if picks is not None else None
        if selected_picks is not None and not selected_picks:
            dst.parent.mkdir(parents=True, exist_ok=True)
            runtime_copy2(src, dst)
            write_preproc_step_config(
                resolver=resolver,
                step="ecg_artifact_removal",
                config={
                    "method": method,
                    "mode": "passthrough_copy",
                    "picks": [],
                    "method_kwargs": runtime_kwargs,
                    "figure_channels": [],
                },
            )
            mark_preproc_step_fn(
                resolver=resolver,
                step="ecg_artifact_removal",
                completed=True,
                params={
                    "method": method,
                    "picks": [],
                    "method_kwargs": persisted_kwargs,
                },
                input_path=str(src),
                output_path=str(dst),
                message=(
                    "ECG step completed without channel picks; "
                    f"copied {source_step} output unchanged."
                ),
            )
            invalidate_downstream_fn(context, "ecg_artifact_removal")
            return True, "ECG step completed."

        raw = read_raw_fif_fn(str(src), preload=True, verbose="ERROR")
        available = list(raw.ch_names)
        if not available:
            raise ValueError("No channels available for ECG removal.")
        selected_picks = selected_picks if selected_picks is not None else available
        missing = [name for name in selected_picks if name not in available]
        if missing:
            raise ValueError(f"Unknown ECG picks: {missing}")

        raw_clean, figs = runtime_raw_call(
            raw,
            method=method,
            picks=selected_picks,
            inplace=False,
            verbose=False,
            **runtime_kwargs,
        )

        dst.parent.mkdir(parents=True, exist_ok=True)
        raw_clean.save(str(dst), overwrite=True)
        write_preproc_step_config(
            resolver=resolver,
            step="ecg_artifact_removal",
            config={
                "method": method,
                "picks": selected_picks,
                "method_kwargs": runtime_kwargs,
                "figure_channels": sorted(figs.keys()),
            },
        )
        mark_preproc_step_fn(
            resolver=resolver,
            step="ecg_artifact_removal",
            completed=True,
            params={
                "method": method,
                "picks": selected_picks,
                "method_kwargs": persisted_kwargs,
            },
            input_path=str(src),
            output_path=str(dst),
            message=(
                f"ECG step completed with method {method} using source: {source_step}."
            ),
        )
        invalidate_downstream_fn(context, "ecg_artifact_removal")
    except Exception as exc:
        mark_preproc_step_fn(
            resolver=resolver,
            step="ecg_artifact_removal",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"ECG step failed: {exc}",
        )
        return False, f"ECG step failed: {exc}"

    return True, "ECG step completed."
