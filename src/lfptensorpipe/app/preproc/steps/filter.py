"""Filter-step runtime helpers for preprocess stage."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import threading
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet

from ..paths import (
    preproc_step_config_path,
    preproc_step_log_path,
    preproc_step_raw_path,
    write_preproc_step_config,
)

MarkStepFn = Callable[..., Any]
InvalidateFn = Callable[[RecordContext, str], list[Any]]


def default_filter_advance_params() -> dict[str, Any]:
    """Return default Filter-Advance params derived from BadAnnotationConfig."""
    from lfptensorpipe.preproc.filter import BadAnnotationConfig

    cfg = BadAnnotationConfig()

    return {
        "notch_widths": 2.0,
        "epoch_dur": float(cfg.epoch_dur),
        "p2p_thresh": [float(cfg.p2p_thresh[0]), float(cfg.p2p_thresh[1])],
        "autoreject_correct_factor": float(cfg.autoreject_correct_factor),
    }


def _normalize_notch_widths(value: Any) -> float | list[float]:
    if isinstance(value, (int, float)):
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError("notch_widths must contain positive finite values.")
        return parsed

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("notch_widths cannot be empty.")
        parsed_list = [float(item) for item in value]
        if any(not math.isfinite(item) or item <= 0.0 for item in parsed_list):
            raise ValueError("notch_widths must contain positive finite values.")
        return parsed_list if len(parsed_list) > 1 else parsed_list[0]

    raise ValueError("notch_widths must be a number or a numeric list.")


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite when provided.")
    return parsed


def normalize_filter_runtime_params(
    *,
    notches: Any,
    l_freq: Any,
    h_freq: Any,
) -> tuple[bool, dict[str, Any], str]:
    """Normalize nullable basic Filter parameters at the runtime boundary."""
    try:
        low_freq = _optional_float(l_freq, field_name="l_freq")
        high_freq = _optional_float(h_freq, field_name="h_freq")
        if notches is None or (isinstance(notches, str) and not notches.strip()):
            parsed_notches: list[float] = []
        elif isinstance(notches, str):
            parsed_notches = [
                float(item.strip()) for item in notches.split(",") if item.strip()
            ]
        elif isinstance(notches, (list, tuple)):
            parsed_notches = [float(item) for item in notches]
        else:
            raise ValueError("notches must be empty or a number list when provided.")
    except Exception as exc:  # noqa: BLE001
        return False, {}, str(exc)

    if low_freq is not None and low_freq < 0.0:
        return False, {}, "l_freq must be >= 0 when provided."
    if high_freq is not None and high_freq <= 0.0:
        return False, {}, "h_freq must be > 0 when provided."
    if low_freq is not None and high_freq is not None and high_freq <= low_freq:
        return False, {}, "h_freq must be greater than l_freq when both are provided."
    if any(not math.isfinite(value) or value <= 0.0 for value in parsed_notches):
        return False, {}, "notches must contain positive finite values."

    return (
        True,
        {
            "notches": parsed_notches,
            "l_freq": low_freq,
            "h_freq": high_freq,
        },
        "",
    )


def filter_nyquist_warning(
    *,
    sfreq_hz: Any,
    notches: list[float] | tuple[float, ...],
    h_freq: float | None,
) -> str:
    """Return a blocking warning for a requested frequency at/above Nyquist."""
    sfreq = float(sfreq_hz)
    if not math.isfinite(sfreq) or sfreq <= 0.0:
        return "Input sampling frequency must be positive and finite."
    nyquist = sfreq / 2.0
    if h_freq is not None and float(h_freq) >= nyquist:
        return (
            f"High freq {float(h_freq):g} Hz must be below the Nyquist frequency "
            f"of {nyquist:g} Hz.\n\nEnter a value below {nyquist:g} Hz, or "
            "leave High freq empty to disable low-pass filtering."
        )
    for notch in notches:
        if float(notch) >= nyquist:
            return (
                f"Notch frequency {float(notch):g} Hz must be below the Nyquist "
                f"frequency of {nyquist:g} Hz.\n\nRemove it or leave Notches empty."
            )
    return ""


def normalize_filter_advance_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], str]:
    """Validate and normalize filter-advance params for runtime/config usage."""
    defaults = default_filter_advance_params()
    if params is None:
        return True, defaults, ""
    if not isinstance(params, dict):
        return False, defaults, "Filter Advance params must be a dictionary."

    merged: dict[str, Any] = dict(defaults)
    for key in (
        "notch_widths",
        "epoch_dur",
        "p2p_thresh",
        "autoreject_correct_factor",
    ):
        if key in params:
            merged[key] = params[key]

    try:
        notch_widths = _normalize_notch_widths(merged["notch_widths"])
        epoch_dur = float(merged["epoch_dur"])
        p2p_raw = merged["p2p_thresh"]
        autoreject_correct_factor = float(merged["autoreject_correct_factor"])
    except Exception as exc:  # noqa: BLE001
        return False, defaults, str(exc)

    if not math.isfinite(epoch_dur) or epoch_dur <= 0.0:
        return False, defaults, "epoch_dur must be finite and > 0."
    if not math.isfinite(autoreject_correct_factor) or autoreject_correct_factor <= 0.0:
        return (
            False,
            defaults,
            "autoreject_correct_factor must be finite and > 0.",
        )

    p2p_thresh: list[float] | None
    if p2p_raw is None or p2p_raw == "" or p2p_raw == [] or p2p_raw == ():
        p2p_thresh = None
    else:
        if not isinstance(p2p_raw, (list, tuple)) or len(p2p_raw) != 2:
            return (
                False,
                defaults,
                "p2p_thresh must be empty or contain exactly two numbers.",
            )
        try:
            p2p_min = float(p2p_raw[0])
            p2p_max = float(p2p_raw[1])
        except Exception:  # noqa: BLE001
            return False, defaults, "p2p_thresh must contain valid numbers."
        if not math.isfinite(p2p_min):
            return False, defaults, "p2p_thresh minimum must be finite."
        if not math.isfinite(p2p_max):
            return False, defaults, "p2p_thresh maximum must be finite."
        if p2p_min < 0.0 or p2p_max <= 0.0 or p2p_min >= p2p_max:
            return False, defaults, "p2p_thresh must satisfy 0 <= min < max."
        p2p_thresh = [p2p_min, p2p_max]

    return (
        True,
        {
            "notch_widths": notch_widths,
            "epoch_dur": epoch_dur,
            "p2p_thresh": p2p_thresh,
            "autoreject_correct_factor": autoreject_correct_factor,
        },
        "",
    )


def apply_filter_step(
    context: RecordContext,
    *,
    advance_params: dict[str, Any] | None,
    notches: list[float] | tuple[float, ...] | None,
    l_freq: float | None,
    h_freq: float | None,
    mark_preproc_step_fn: MarkStepFn,
    invalidate_downstream_fn: InvalidateFn,
    thread_module: Any = threading,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    mark_lfp_bad_segments_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Apply preprocess filter step using BAD-segment marker defaults."""
    from lfptensorpipe.preproc.filter import BadAnnotationConfig, mark_lfp_bad_segments

    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "raw")
    dst = preproc_step_raw_path(resolver, "filter")
    reject_plot_path = (
        resolver.preproc_step_dir("filter", create=True) / "qc" / "reject.png"
    )
    valid_params, normalized_params, message = normalize_filter_advance_params(
        advance_params
    )
    if not valid_params:
        return False, f"Invalid Filter Advance params: {message}"

    valid_runtime, runtime_params, runtime_message = normalize_filter_runtime_params(
        notches=notches,
        l_freq=l_freq,
        h_freq=h_freq,
    )
    if not valid_runtime:
        return False, f"Invalid Filter params: {runtime_message}"
    runtime_l_freq = runtime_params["l_freq"]
    runtime_h_freq = runtime_params["h_freq"]
    runtime_notches = list(runtime_params["notches"])

    if not src.exists():
        mark_preproc_step_fn(
            resolver=resolver,
            step="filter",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message="Missing preprocess raw input for filter step.",
        )
        invalidate_downstream_fn(context, "filter")
        return False, "Missing preprocess raw input for filter step."

    try:
        if read_raw_fif_fn is None:
            import mne

            read_raw_fif_fn = mne.io.read_raw_fif
        runtime_mark_lfp_bad_segments = (
            mark_lfp_bad_segments_fn or mark_lfp_bad_segments
        )

        raw = read_raw_fif_fn(str(src), preload=True, verbose="ERROR")
        nyquist = float(raw.info["sfreq"]) / 2.0
        nyquist_message = filter_nyquist_warning(
            sfreq_hz=raw.info["sfreq"],
            notches=runtime_notches,
            h_freq=runtime_h_freq,
        )
        if nyquist_message:
            if hasattr(raw, "close"):
                raw.close()
            return False, nyquist_message
        cfg = replace(
            BadAnnotationConfig(),
            l_freq=runtime_l_freq,
            h_freq=runtime_h_freq,
            notches=tuple(runtime_notches) if runtime_notches else None,
            notch_widths=normalized_params["notch_widths"],
            epoch_dur=normalized_params["epoch_dur"],
            p2p_thresh=(
                None
                if normalized_params["p2p_thresh"] is None
                else (
                    float(normalized_params["p2p_thresh"][0]),
                    float(normalized_params["p2p_thresh"][1]),
                )
            ),
            autoreject_correct_factor=normalized_params["autoreject_correct_factor"],
        )
        runtime_reject_plot_path = (
            reject_plot_path
            if thread_module.current_thread() is thread_module.main_thread()
            else None
        )
        raw_marked, _, summary = runtime_mark_lfp_bad_segments(
            raw,
            cfg,
            reject_plot_path=runtime_reject_plot_path,
        )
        config_path = preproc_step_config_path(resolver, "filter")
        log_path = preproc_step_log_path(resolver, "filter")
        with AtomicOutputSet(
            [dst, config_path, log_path],
            cleanup_stale_residues=True,
        ) as output_set:
            raw_marked.save(
                str(output_set.staged_path(dst)),
                overwrite=True,
            )
            write_preproc_step_config(
                resolver=resolver,
                step="filter",
                path=output_set.staged_path(config_path),
                config={
                    "low_freq": cfg.l_freq,
                    "high_freq": cfg.h_freq,
                    "notches": list(cfg.notches or []),
                    "nyquist_freq": nyquist,
                    "bad_annotation_config": asdict(cfg),
                    "summary": summary,
                    "reject_plot_path": str(reject_plot_path),
                },
            )
            mark_preproc_step_fn(
                resolver=resolver,
                step="filter",
                completed=True,
                params={
                    "low_freq": cfg.l_freq,
                    "high_freq": cfg.h_freq,
                    "notches": list(cfg.notches or []),
                    "nyquist_freq": nyquist,
                    "notch_widths": cfg.notch_widths,
                    "epoch_dur": cfg.epoch_dur,
                    "p2p_thresh": (
                        None if cfg.p2p_thresh is None else list(cfg.p2p_thresh)
                    ),
                    "autoreject_correct_factor": cfg.autoreject_correct_factor,
                    "reject_plot_path": str(reject_plot_path),
                },
                input_path=str(src),
                output_path=str(dst),
                message="Filter step completed with mark_lfp_bad_segments defaults.",
                log_path=output_set.staged_path(log_path),
            )
            output_set.commit()
        invalidate_downstream_fn(context, "filter")
    except Exception as exc:
        mark_preproc_step_fn(
            resolver=resolver,
            step="filter",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Filter step failed: {exc}",
        )
        invalidate_downstream_fn(context, "filter")
        return False, f"Filter step failed: {exc}"

    return True, "Filter step completed."
