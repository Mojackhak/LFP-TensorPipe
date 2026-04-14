"""Filter-step runtime helpers for preprocess stage."""

from __future__ import annotations

from dataclasses import asdict, replace
import threading
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

from ..paths import preproc_step_raw_path, write_preproc_step_config

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
        if parsed <= 0.0:
            raise ValueError("notch_widths must be > 0.")
        return parsed

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("notch_widths cannot be empty.")
        parsed_list = [float(item) for item in value]
        if any(item <= 0.0 for item in parsed_list):
            raise ValueError("notch_widths values must be > 0.")
        return parsed_list if len(parsed_list) > 1 else parsed_list[0]

    raise ValueError("notch_widths must be a number or a numeric list.")


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

    if epoch_dur <= 0.0:
        return False, defaults, "epoch_dur must be > 0."
    if autoreject_correct_factor <= 0.0:
        return False, defaults, "autoreject_correct_factor must be > 0."
    if not isinstance(p2p_raw, (list, tuple)) or len(p2p_raw) != 2:
        return False, defaults, "p2p_thresh must contain exactly two numbers."

    try:
        p2p_min = float(p2p_raw[0])
        p2p_max = float(p2p_raw[1])
    except Exception:  # noqa: BLE001
        return False, defaults, "p2p_thresh must contain valid numbers."

    if p2p_min < 0.0 or p2p_max <= 0.0 or p2p_min >= p2p_max:
        return False, defaults, "p2p_thresh must satisfy 0 <= min < max."

    return (
        True,
        {
            "notch_widths": notch_widths,
            "epoch_dur": epoch_dur,
            "p2p_thresh": [p2p_min, p2p_max],
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
        mark_preproc_step_fn(
            resolver=resolver,
            step="filter",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Invalid Filter Advance params: {message}",
        )
        return False, f"Invalid Filter Advance params: {message}"

    try:
        runtime_l_freq = float(1.0 if l_freq is None else l_freq)
        runtime_h_freq = float(200.0 if h_freq is None else h_freq)
    except Exception as exc:  # noqa: BLE001
        mark_preproc_step_fn(
            resolver=resolver,
            step="filter",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Invalid Filter freq params: {exc}",
        )
        return False, f"Invalid Filter freq params: {exc}"
    if runtime_l_freq < 0.0 or runtime_h_freq <= runtime_l_freq:
        mark_preproc_step_fn(
            resolver=resolver,
            step="filter",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message="Invalid Filter freq params: require 0 <= low < high.",
        )
        return False, "Invalid Filter freq params: require 0 <= low < high."

    runtime_notches: list[float] = []
    if notches is not None:
        try:
            runtime_notches = [float(value) for value in notches]
        except Exception as exc:  # noqa: BLE001
            mark_preproc_step_fn(
                resolver=resolver,
                step="filter",
                completed=False,
                input_path=str(src),
                output_path=str(dst),
                message=f"Invalid Filter notches: {exc}",
            )
            return False, f"Invalid Filter notches: {exc}"
        if any(value <= 0.0 for value in runtime_notches):
            mark_preproc_step_fn(
                resolver=resolver,
                step="filter",
                completed=False,
                input_path=str(src),
                output_path=str(dst),
                message="Invalid Filter notches: values must be > 0.",
            )
            return False, "Invalid Filter notches: values must be > 0."
    else:
        runtime_notches = [50.0, 100.0]

    if not src.exists():
        mark_preproc_step_fn(
            resolver=resolver,
            step="filter",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message="Missing preprocess raw input for filter step.",
        )
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
        max_h_freq = nyquist - 1e-3
        applied_h_freq = min(runtime_h_freq, max_h_freq)
        if applied_h_freq <= runtime_l_freq:
            raise ValueError(
                "Filter high freq is too high for current data Nyquist or not greater than low freq."
            )
        applied_notches = [value for value in runtime_notches if value < max_h_freq]
        dropped_notches = [value for value in runtime_notches if value >= max_h_freq]
        cfg = replace(
            BadAnnotationConfig(),
            l_freq=runtime_l_freq,
            h_freq=applied_h_freq,
            notches=tuple(applied_notches) if applied_notches else None,
            notch_widths=normalized_params["notch_widths"],
            epoch_dur=normalized_params["epoch_dur"],
            p2p_thresh=(
                float(normalized_params["p2p_thresh"][0]),
                float(normalized_params["p2p_thresh"][1]),
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw_marked.save(str(dst), overwrite=True)
        write_preproc_step_config(
            resolver=resolver,
            step="filter",
            config={
                "low_freq": cfg.l_freq,
                "high_freq": cfg.h_freq,
                "notches": list(cfg.notches or []),
                "dropped_notches": dropped_notches,
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
                "dropped_notches": dropped_notches,
                "notch_widths": cfg.notch_widths,
                "epoch_dur": cfg.epoch_dur,
                "p2p_thresh": list(cfg.p2p_thresh),
                "autoreject_correct_factor": cfg.autoreject_correct_factor,
                "reject_plot_path": str(reject_plot_path),
            },
            input_path=str(src),
            output_path=str(dst),
            message="Filter step completed with mark_lfp_bad_segments defaults.",
        )
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
        return False, f"Filter step failed: {exc}"

    return True, "Filter step completed."
