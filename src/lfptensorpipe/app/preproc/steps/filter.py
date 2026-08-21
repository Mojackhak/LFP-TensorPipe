"""Filter-step runtime helpers for preprocess stage."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import threading
from typing import Any, Callable

import yaml

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet
from lfptensorpipe.app.shared.generation_lineage import (
    new_result_generation_id,
    params_with_generation_lineage,
    params_with_input_generation_receipts,
)

from ..lineage import (
    PreprocInputGenerationChanged,
    capture_preproc_input_generation,
    filter_preview_lineage_is_current,
    preproc_input_generation_matches,
)
from ..paths import (
    preproc_filter_preview_config_path,
    preproc_filter_preview_log_path,
    preproc_filter_preview_raw_path,
    preproc_step_config_path,
    preproc_step_log_path,
    preproc_step_raw_path,
    write_preproc_step_config,
)

MarkStepFn = Callable[..., Any]
InvalidateFn = Callable[[RecordContext, str], list[Any]]
FILTER_EPOCH_COVERAGE_SEMANTICS = "grid_plus_end_aligned_tail"
FILTER_BAD_CHANNEL_DETECTION_SEMANTICS = "exclude_info_bads_from_p2p_and_autoreject"


def filter_log_has_current_bad_channel_detection_semantics(payload: Any) -> bool:
    """Return whether one Filter log declares the current channel contract."""
    if not isinstance(payload, dict):
        return False
    params = payload.get("params")
    return bool(
        isinstance(params, dict)
        and params.get("bad_channel_detection_semantics")
        == FILTER_BAD_CHANNEL_DETECTION_SEMANTICS
    )


def default_filter_advance_params() -> dict[str, Any]:
    """Return default Filter-Advance params derived from BadAnnotationConfig."""
    from lfptensorpipe.preproc.filter import BadAnnotationConfig

    cfg = BadAnnotationConfig()

    return {
        "notch_widths": 2.0,
        "epoch_dur": float(cfg.epoch_dur),
        "p2p_thresh": [float(cfg.p2p_thresh[0]), float(cfg.p2p_thresh[1])],
        "autoreject_correct_factor": float(cfg.autoreject_correct_factor),
        "isolate_bad_boundaries": True,
        "mark_filter_edges": False,
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
        "isolate_bad_boundaries",
        "mark_filter_edges",
    ):
        if key in params:
            merged[key] = params[key]

    if not isinstance(merged["isolate_bad_boundaries"], bool):
        return False, defaults, "isolate_bad_boundaries must be true or false."
    if not isinstance(merged["mark_filter_edges"], bool):
        return False, defaults, "mark_filter_edges must be true or false."
    if merged["mark_filter_edges"] and not merged["isolate_bad_boundaries"]:
        return (
            False,
            defaults,
            "mark_filter_edges requires isolate_bad_boundaries to be true.",
        )

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
            "isolate_bad_boundaries": merged["isolate_bad_boundaries"],
            "mark_filter_edges": merged["mark_filter_edges"],
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
    thread_module: Any = threading,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    mark_lfp_bad_segments_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Create a detection-filtered Preview for manual BAD review."""
    from lfptensorpipe.preproc.filter import BadAnnotationConfig, mark_lfp_bad_segments

    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "raw")
    preview = preproc_filter_preview_raw_path(resolver)
    preview_config_path = preproc_filter_preview_config_path(resolver)
    preview_log_path = preproc_filter_preview_log_path(resolver)
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
        return False, "Missing preprocess raw input for filter step."
    captured = capture_preproc_input_generation(resolver, "filter")
    if captured is None or captured[0] != "raw":
        return False, "Filter source changed before input read."
    _, input_generations = captured

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
        with AtomicOutputSet(
            [preview, preview_config_path, preview_log_path],
            cleanup_stale_residues=True,
        ) as output_set:
            raw_marked.save(
                str(output_set.staged_path(preview)),
                overwrite=True,
            )
            write_preproc_step_config(
                resolver=resolver,
                step="filter",
                path=output_set.staged_path(preview_config_path),
                config={
                    "low_freq": cfg.l_freq,
                    "high_freq": cfg.h_freq,
                    "notches": list(cfg.notches or []),
                    "nyquist_freq": nyquist,
                    "bad_annotation_config": asdict(cfg),
                    "summary": summary,
                    "reject_plot_path": str(reject_plot_path),
                    "isolate_bad_boundaries": normalized_params[
                        "isolate_bad_boundaries"
                    ],
                    "mark_filter_edges": normalized_params["mark_filter_edges"],
                    "epoch_coverage_semantics": FILTER_EPOCH_COVERAGE_SEMANTICS,
                    "bad_channel_detection_semantics": (
                        FILTER_BAD_CHANNEL_DETECTION_SEMANTICS
                    ),
                    "review_status": "required",
                    "filter_output_role": "preview",
                },
            )
            mark_preproc_step_fn(
                resolver=resolver,
                step="filter",
                completed=False,
                params=params_with_input_generation_receipts(
                    {
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
                        "isolate_bad_boundaries": normalized_params[
                            "isolate_bad_boundaries"
                        ],
                        "mark_filter_edges": normalized_params["mark_filter_edges"],
                        "epoch_coverage_semantics": (FILTER_EPOCH_COVERAGE_SEMANTICS),
                        "bad_channel_detection_semantics": (
                            FILTER_BAD_CHANNEL_DETECTION_SEMANTICS
                        ),
                        "review_status": "required",
                        "filter_output_role": "preview",
                    },
                    input_generations=input_generations,
                ),
                input_path=str(src),
                output_path=str(preview),
                message="Filter preview ready; manual review is required.",
                log_path=output_set.staged_path(preview_log_path),
            )
            if not preproc_input_generation_matches(
                resolver,
                "filter",
                source_step="raw",
                input_generations=input_generations,
            ):
                raise PreprocInputGenerationChanged(
                    "Filter source changed during execution."
                )
            output_set.commit()
    except Exception as exc:
        return False, f"Filter step failed: {exc}"

    return True, "Filter preview ready; close Plot to finalize."


def finalize_filter_review(
    context: RecordContext,
    *,
    reviewed_annotations: Any,
    reviewed_bads: list[str] | tuple[str, ...],
    mark_preproc_step_fn: MarkStepFn,
    invalidate_downstream_fn: InvalidateFn,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    finalize_reviewed_filter_fn: Callable[..., Any] | None = None,
    review_source_is_current_fn: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
    """Regenerate and atomically accept Filter output from reviewed annotations."""
    from lfptensorpipe.preproc.filter import finalize_reviewed_lfp_filter

    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "raw")
    dst = preproc_step_raw_path(resolver, "filter")
    preview = preproc_filter_preview_raw_path(resolver)
    preview_config_path = preproc_filter_preview_config_path(resolver)
    preview_log_path = preproc_filter_preview_log_path(resolver)
    config_path = preproc_step_config_path(resolver, "filter")
    log_path = preproc_step_log_path(resolver, "filter")
    if not src.exists():
        return False, "Missing preprocess raw input for Filter finalization."
    captured = capture_preproc_input_generation(resolver, "filter")
    if captured is None or captured[0] != "raw":
        return False, "Filter source changed before input read."
    _, input_generations = captured
    result_generation_id = new_result_generation_id()
    if review_source_is_current_fn is not None and not review_source_is_current_fn():
        return False, "Filter review source changed before input read."

    preview_payload = read_run_log(preview_log_path)
    preview_params = (
        preview_payload.get("params") if isinstance(preview_payload, dict) else None
    )
    pending_preview = bool(
        preview.exists()
        and isinstance(preview_params, dict)
        and preview_payload.get("completed") is False
        and preview_params.get("review_status") == "required"
        and preview_params.get("filter_output_role") == "preview"
        and filter_log_has_current_bad_channel_detection_semantics(preview_payload)
        and filter_preview_lineage_is_current(resolver, preview_payload)
    )
    state_log_path = preview_log_path if pending_preview else log_path
    state_config_path = preview_config_path if pending_preview else config_path
    payload = read_run_log(state_log_path)
    params = payload.get("params") if isinstance(payload, dict) else None
    if not isinstance(params, dict):
        params = {}
    config: dict[str, Any] = {}
    if state_config_path.exists():
        loaded = yaml.safe_load(state_config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config = loaded
    detection_config = config.get("bad_annotation_config")
    if not isinstance(detection_config, dict):
        detection_config = {}

    valid_runtime, runtime_params, runtime_message = normalize_filter_runtime_params(
        notches=params.get("notches", config.get("notches", [])),
        l_freq=params.get("low_freq", config.get("low_freq")),
        h_freq=params.get("high_freq", config.get("high_freq")),
    )
    persisted_advance: dict[str, Any] = {}
    for key in (
        "notch_widths",
        "epoch_dur",
        "p2p_thresh",
        "autoreject_correct_factor",
        "isolate_bad_boundaries",
        "mark_filter_edges",
    ):
        if key in params:
            persisted_advance[key] = params[key]
        elif key in config:
            persisted_advance[key] = config[key]
        elif key in detection_config:
            persisted_advance[key] = detection_config[key]
    valid_advance, advance, advance_message = normalize_filter_advance_params(
        persisted_advance
    )
    if not valid_runtime:
        return False, f"Invalid persisted Filter params: {runtime_message}"
    if not valid_advance:
        return False, f"Invalid persisted Filter Advance params: {advance_message}"

    cleanup_warning = ""
    try:
        if read_raw_fif_fn is None:
            import mne

            read_raw_fif_fn = mne.io.read_raw_fif
        runtime_finalize = finalize_reviewed_filter_fn or finalize_reviewed_lfp_filter
        raw = read_raw_fif_fn(str(src), preload=True, verbose="ERROR")
        finalized, support_report = runtime_finalize(
            raw,
            reviewed_annotations=reviewed_annotations,
            reviewed_bads=reviewed_bads,
            l_freq=runtime_params["l_freq"],
            h_freq=runtime_params["h_freq"],
            notches=runtime_params["notches"],
            notch_widths=advance["notch_widths"],
            isolate_bad_boundaries=advance["isolate_bad_boundaries"],
            mark_filter_edges=advance["mark_filter_edges"],
        )
        final_params = {
            **params,
            "low_freq": runtime_params["l_freq"],
            "high_freq": runtime_params["h_freq"],
            "notches": runtime_params["notches"],
            "notch_widths": advance["notch_widths"],
            "epoch_dur": advance["epoch_dur"],
            "p2p_thresh": advance["p2p_thresh"],
            "autoreject_correct_factor": advance["autoreject_correct_factor"],
            "isolate_bad_boundaries": advance["isolate_bad_boundaries"],
            "mark_filter_edges": advance["mark_filter_edges"],
            "review_status": "finalized",
            "filter_output_role": "scientific",
            "filter_support_radius_samples": support_report["support_radius_samples"],
        }
        final_config = {
            **config,
            "low_freq": runtime_params["l_freq"],
            "high_freq": runtime_params["h_freq"],
            "notches": runtime_params["notches"],
            "isolate_bad_boundaries": advance["isolate_bad_boundaries"],
            "mark_filter_edges": advance["mark_filter_edges"],
            "review_status": "finalized",
            "filter_output_role": "scientific",
        }
        coverage_semantics = params.get(
            "epoch_coverage_semantics",
            config.get("epoch_coverage_semantics"),
        )
        if coverage_semantics == FILTER_EPOCH_COVERAGE_SEMANTICS:
            final_params["epoch_coverage_semantics"] = FILTER_EPOCH_COVERAGE_SEMANTICS
            final_config["epoch_coverage_semantics"] = FILTER_EPOCH_COVERAGE_SEMANTICS
        else:
            final_params.pop("epoch_coverage_semantics", None)
            final_config.pop("epoch_coverage_semantics", None)
        detection_semantics = params.get(
            "bad_channel_detection_semantics",
            config.get("bad_channel_detection_semantics"),
        )
        if detection_semantics == FILTER_BAD_CHANNEL_DETECTION_SEMANTICS:
            final_params["bad_channel_detection_semantics"] = (
                FILTER_BAD_CHANNEL_DETECTION_SEMANTICS
            )
            final_config["bad_channel_detection_semantics"] = (
                FILTER_BAD_CHANNEL_DETECTION_SEMANTICS
            )
        else:
            final_params.pop("bad_channel_detection_semantics", None)
            final_config.pop("bad_channel_detection_semantics", None)
        with AtomicOutputSet(
            [dst, config_path, log_path],
            cleanup_stale_residues=True,
        ) as output_set:
            finalized.save(str(output_set.staged_path(dst)), overwrite=True)
            write_preproc_step_config(
                resolver=resolver,
                step="filter",
                path=output_set.staged_path(config_path),
                config=final_config,
            )
            mark_preproc_step_fn(
                resolver=resolver,
                step="filter",
                completed=True,
                params=params_with_generation_lineage(
                    {
                        **final_params,
                        "source_step": "raw",
                    },
                    result_generation_id=result_generation_id,
                    input_generations=input_generations,
                ),
                input_path=str(src),
                output_path=str(dst),
                message="Filter review finalized from the original Raw.",
                log_path=output_set.staged_path(log_path),
            )
            if not preproc_input_generation_matches(
                resolver,
                "filter",
                source_step="raw",
                input_generations=input_generations,
            ):
                raise PreprocInputGenerationChanged(
                    "Filter source changed during execution."
                )
            if (
                review_source_is_current_fn is not None
                and not review_source_is_current_fn()
            ):
                raise PreprocInputGenerationChanged(
                    "Filter review source changed during execution."
                )
            output_set.commit()
        cleanup_errors: list[str] = []
        for preview_artifact in (preview, preview_config_path, preview_log_path):
            try:
                preview_artifact.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append(f"{preview_artifact.name}: {exc}")
        if cleanup_errors:
            cleanup_warning = " Preview cleanup warning: " + "; ".join(cleanup_errors)
        invalidate_downstream_fn(context, "filter")
    except Exception as exc:
        return False, f"Filter finalization failed: {exc}"

    return True, "Filter review finalized." + cleanup_warning
