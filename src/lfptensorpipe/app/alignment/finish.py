"""Alignment finish runner."""

from __future__ import annotations

import numpy as np

from lfptensorpipe.app.localize_service import localize_indicator_state
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_alignment_finish,
)

from . import service as svc

def finish_alignment_epochs(
    context: RecordContext,
    *,
    paradigm_slug: str,
    picked_epoch_indices: list[int],
) -> tuple[bool, str]:
    """Build raw-table outputs for picked epochs."""
    _normalize_slug = svc._normalize_slug
    indicator_from_log = svc.indicator_from_log
    alignment_paradigm_log_path = svc.alignment_paradigm_log_path
    alignment_method_panel_state = svc.alignment_method_panel_state
    _load_trial_config_from_log = svc._load_trial_config_from_log
    _resolve_alignment_method_key = svc._resolve_alignment_method_key
    alignment_paradigm_dir = svc.alignment_paradigm_dir
    load_pkl = svc.load_pkl
    _finish_time_axis_values = svc._finish_time_axis_values
    split_tensor4d_to_nested_df = svc.split_tensor4d_to_nested_df
    _merge_representative_coords_for_metric = svc._merge_representative_coords_for_metric
    alignment_trial_raw_table_path = svc.alignment_trial_raw_table_path
    save_pkl = svc.save_pkl
    _append_alignment_history = svc._append_alignment_history
    RunLogRecord = svc.RunLogRecord

    resolver = PathResolver(context)
    slug = _normalize_slug(paradigm_slug)
    if not slug:
        return False, "Trial slug is empty."
    merge_location_info_ready = (
        localize_indicator_state(
            context.project_root,
            context.subject,
            context.record,
        )
        == "green"
    )
    log_path = alignment_paradigm_log_path(resolver, slug)
    trial_cfg = _load_trial_config_from_log(resolver, slug=slug)
    run_ready = False
    if isinstance(trial_cfg, dict):
        run_ready = alignment_method_panel_state(resolver, paradigm=trial_cfg) == "green"
    if not run_ready:
        run_ready = indicator_from_log(log_path) == "green"
    if not run_ready:
        return False, "Run Align Epochs successfully before Finish."
    if not picked_epoch_indices:
        return False, "Select at least one epoch before Finish."
    picked = sorted({int(item) for item in picked_epoch_indices if int(item) >= 0})
    if not picked:
        return False, "Select at least one valid epoch before Finish."
    finish_method = _resolve_alignment_method_key(
        trial_cfg.get("method", "") if isinstance(trial_cfg, dict) else ""
    )

    paradigm_dir = alignment_paradigm_dir(resolver, slug)
    metric_paths = sorted(paradigm_dir.glob("*/tensor_warped.pkl"))
    if not metric_paths:
        return False, "No warped tensor outputs found for selected trial."

    saved = 0
    repcoord_warnings: list[str] = []
    for metric_path in metric_paths:
        metric_key = metric_path.parent.name
        payload = load_pkl(metric_path)
        if not isinstance(payload, dict):
            continue
        tensor = np.asarray(payload.get("tensor"), dtype=float)
        meta = payload.get("meta", {})
        if tensor.ndim != 4:
            continue
        axes = meta.get("axes", {}) if isinstance(meta, dict) else {}
        if not isinstance(axes, dict):
            continue
        epoch_axis_raw = axes.get("epoch")
        channel_axis_raw = axes.get("channel")
        freq_axis_raw = axes.get("freq")

        epoch_axis = (
            list(epoch_axis_raw)
            if epoch_axis_raw is not None
            else [f"epoch_{idx:03d}" for idx in range(tensor.shape[0])]
        )
        channel_axis = (
            list(channel_axis_raw)
            if channel_axis_raw is not None
            else [f"ch_{idx:03d}" for idx in range(tensor.shape[1])]
        )
        freq_axis = (
            list(freq_axis_raw)
            if freq_axis_raw is not None
            else list(np.arange(tensor.shape[2], dtype=float))
        )
        time_axis = _finish_time_axis_values(
            axes,
            method_key=finish_method,
            n_time=tensor.shape[3],
        )

        keep = [idx for idx in picked if idx < tensor.shape[0]]
        if not keep:
            continue
        tensor_keep = tensor[keep, ...]
        epochs_keep = [epoch_axis[idx] for idx in keep]

        frame = split_tensor4d_to_nested_df(
            tensor_keep,
            epoch=epochs_keep,
            channel=channel_axis,
            freq=freq_axis,
            time=time_axis,
        )
        frame = frame.rename(
            columns={"epoch": "Epoch", "channel": "Channel", "value": "Value"}
        )
        frame["Subject"] = context.subject
        frame["Record"] = context.record
        frame["Trial"] = slug
        frame["Metric"] = metric_key
        if merge_location_info_ready:
            frame, merge_warning = _merge_representative_coords_for_metric(
                frame,
                context,
                metric_key=metric_key,
            )
            if merge_warning:
                repcoord_warnings.append(f"{metric_key}:{merge_warning}")

        out_path = alignment_trial_raw_table_path(
            resolver,
            trial_slug=slug,
            metric_key=metric_key,
        )
        save_pkl(frame, out_path)
        saved += 1

    if saved == 0:
        return False, "No raw-table outputs were generated."

    _append_alignment_history(
        log_path,
        entry=RunLogRecord(
            step="build_raw_table",
            completed=True,
            params={
                "trial_slug": slug,
                "picked_epoch_indices": picked,
                "merge_location_info_ready": merge_location_info_ready,
                "merge_location_info_applied": (
                    merge_location_info_ready and not repcoord_warnings
                ),
                "saved_tables": saved,
                "repcoord_merge_warnings": repcoord_warnings,
            },
            input_path=str(paradigm_dir),
            output_path=str(paradigm_dir),
            message=(
                "Raw tables generated from warped tensors and merged representative coords."
                if merge_location_info_ready and not repcoord_warnings
                else (
                    "Raw tables generated from warped tensors; representative-coordinate merge completed with warnings."
                    if merge_location_info_ready
                    else "Raw tables generated from warped tensors without representative-coordinate merge."
                )
            ),
        ).to_dict(),
        keep_top_level=True,
    )
    invalidate_after_alignment_finish(context, paradigm_slug=slug)
    return (
        True,
        (
            f"Finish completed. Saved {saved} raw table(s). "
            f"Merge Location Info: {'Ready' if merge_location_info_ready else 'Not Ready'}."
        ),
    )
