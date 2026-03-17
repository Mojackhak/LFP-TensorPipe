"""ECG-artifact-removal step runtime helper."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

from ..paths import preproc_step_raw_path, write_preproc_step_config

MarkStepFn = Callable[..., Any]
InvalidateFn = Callable[[RecordContext, str], list[Any]]


def apply_ecg_step(
    context: RecordContext,
    *,
    method: str,
    picks: list[str] | tuple[str, ...] | None,
    ecg_methods: tuple[str, ...],
    mark_preproc_step_fn: MarkStepFn,
    invalidate_downstream_fn: InvalidateFn,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    raw_call_ecgremover_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Apply ECG-artifact-removal step with selected method defaults."""
    from lfptensorpipe.preproc.ecg_remover import (
        PerceiveConfig,
        SvdConfig,
        TemplateFitConfig,
        raw_call_ecgremover,
    )

    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "bad_segment_removal")
    dst = preproc_step_raw_path(resolver, "ecg_artifact_removal")

    if method not in ecg_methods:
        mark_preproc_step_fn(
            resolver=resolver,
            step="ecg_artifact_removal",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Unknown ECG method: {method}",
        )
        return False, f"Unknown ECG method: {method}"

    if not src.exists():
        mark_preproc_step_fn(
            resolver=resolver,
            step="ecg_artifact_removal",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message="Missing bad-segment raw input for ECG step.",
        )
        return False, "Missing bad-segment raw input for ECG step."

    try:
        if read_raw_fif_fn is None:
            import mne

            read_raw_fif_fn = mne.io.read_raw_fif
        runtime_raw_call = raw_call_ecgremover_fn or raw_call_ecgremover

        raw = read_raw_fif_fn(str(src), preload=True, verbose="ERROR")
        available = list(raw.ch_names)
        if not available:
            raise ValueError("No channels available for ECG removal.")
        selected_picks = list(picks) if picks is not None else available
        if not selected_picks:
            raise ValueError("No picks selected for ECG removal.")
        missing = [name for name in selected_picks if name not in available]
        if missing:
            raise ValueError(f"Unknown ECG picks: {missing}")

        method_kwargs: dict[str, Any]
        if method == "template":
            method_kwargs = asdict(TemplateFitConfig())
        elif method == "perceive":
            method_kwargs = asdict(PerceiveConfig())
        else:
            method_kwargs = asdict(SvdConfig())

        raw_clean, figs = runtime_raw_call(
            raw,
            method=method,
            picks=selected_picks,
            inplace=False,
            verbose=False,
            **method_kwargs,
        )

        dst.parent.mkdir(parents=True, exist_ok=True)
        raw_clean.save(str(dst), overwrite=True)
        write_preproc_step_config(
            resolver=resolver,
            step="ecg_artifact_removal",
            config={
                "method": method,
                "picks": selected_picks,
                "method_kwargs": method_kwargs,
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
            },
            input_path=str(src),
            output_path=str(dst),
            message=f"ECG step completed with method: {method}.",
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
