"""Alignment finish runner."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lfptensorpipe.app.alignment.generation import (
    AlignmentInputGenerationChangedError,
    accepted_alignment_metrics,
    alignment_finish_input_generations_match,
    alignment_generation_rerun_message,
    capture_alignment_finish_input_generations,
)
from lfptensorpipe.app.localize_service import localize_indicator_state
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_alignment_finish,
)
from lfptensorpipe.app.shared.generation_lineage import (
    new_result_generation_id,
    params_with_generation_lineage,
)
from lfptensorpipe.utils.transforms import VALUE_TRANSFORM_POLICY_KEY

from . import service as svc


def finish_alignment_epochs(
    context: RecordContext,
    *,
    paradigm_slug: str,
    picked_epoch_indices: Iterable[int | np.integer],
    selected_metrics: list[str] | tuple[str, ...] | None = None,
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
    _merge_representative_coords_for_metric = (
        svc._merge_representative_coords_for_metric
    )
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
        run_ready = (
            alignment_method_panel_state(resolver, paradigm=trial_cfg) == "green"
        )
    else:
        run_ready = indicator_from_log(log_path) == "green"
    if not run_ready:
        return False, "Run Align Epochs successfully before Finish."
    run_metrics = accepted_alignment_metrics(
        resolver,
        trial_slug=slug,
        stage="run",
    )
    if run_metrics is None:
        return False, (
            alignment_generation_rerun_message(
                resolver,
                trial_slug=slug,
                stage="run",
            )
            or "Latest Align Run has no accepted metric generation."
        )
    picked_items = list(picked_epoch_indices)
    if not picked_items:
        return False, "Select at least one epoch before Finish."
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, np.integer))
        or int(item) < 0
        for item in picked_items
    ):
        return False, "Epoch indices must be non-negative integers."
    picked = sorted({int(item) for item in picked_items})
    finish_method = _resolve_alignment_method_key(
        trial_cfg.get("method", "") if isinstance(trial_cfg, dict) else ""
    )

    required_metrics = list(run_metrics)
    if selected_metrics is not None:
        requested_metrics = [
            str(metric).strip() for metric in selected_metrics if str(metric).strip()
        ]
        requested_set = set(requested_metrics)
        unknown = sorted(requested_set.difference(run_metrics))
        if unknown:
            return (
                False,
                "Selected metrics are not in the accepted Align Run: "
                + ", ".join(unknown),
            )
        required_metrics = [metric for metric in run_metrics if metric in requested_set]
    if not required_metrics:
        return False, "No accepted warped tensor metrics selected for Finish."

    input_generations = capture_alignment_finish_input_generations(
        resolver,
        trial_slug=slug,
        include_localize=merge_location_info_ready,
    )
    if input_generations is None:
        return False, "Align Finish inputs are not current; rerun their producers."

    paradigm_dir = alignment_paradigm_dir(resolver, slug)
    repcoord_warnings: list[str] = []
    frames: dict[str, object] = {}
    output_paths = {
        metric: alignment_trial_raw_table_path(
            resolver,
            trial_slug=slug,
            metric_key=metric,
        )
        for metric in required_metrics
    }
    try:
        for metric_key in required_metrics:
            metric_path = paradigm_dir / metric_key / "tensor_warped.pkl"
            if not metric_path.exists():
                raise FileNotFoundError(
                    f"Missing warped tensor for required metric: {metric_key}"
                )
            payload = load_pkl(metric_path)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid warped tensor payload for metric: {metric_key}"
                )
            tensor = np.asarray(payload.get("tensor"), dtype=float)
            meta = payload.get("meta")
            if tensor.ndim != 4:
                raise ValueError(
                    f"Warped tensor must be 4D for {metric_key}: {tensor.shape}"
                )
            if not isinstance(meta, dict):
                raise ValueError(f"Warped metadata is invalid for: {metric_key}")
            axes = meta.get("axes")
            if not isinstance(axes, dict):
                raise ValueError(f"Warped axes are invalid for: {metric_key}")

            axis_values: dict[str, list[object]] = {}
            for axis_name, expected_length in zip(
                ("epoch", "channel", "freq", "time"),
                tensor.shape,
            ):
                raw_axis = axes.get(axis_name)
                if raw_axis is None:
                    raise ValueError(
                        f"Warped {axis_name} axis is missing for: {metric_key}"
                    )
                values = list(raw_axis)
                if len(values) != expected_length:
                    raise ValueError(
                        f"Warped {axis_name} axis length mismatch for {metric_key}: "
                        f"{len(values)} != {expected_length}"
                    )
                axis_values[axis_name] = values

            invalid_picks = [idx for idx in picked if idx >= tensor.shape[0]]
            if invalid_picks:
                raise ValueError(
                    f"Picked epoch indices exceed {metric_key} epoch support: "
                    + ", ".join(str(index) for index in invalid_picks)
                )
            tensor_keep = tensor[picked, ...]
            epochs_keep = [axis_values["epoch"][idx] for idx in picked]
            time_axis = _finish_time_axis_values(
                axes,
                method_key=finish_method,
                n_time=tensor.shape[3],
            )
            if len(time_axis) != tensor.shape[3]:
                raise ValueError(f"Finish time axis length mismatch for: {metric_key}")

            frame = split_tensor4d_to_nested_df(
                tensor_keep,
                epoch=epochs_keep,
                channel=axis_values["channel"],
                freq=axis_values["freq"],
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
            transform_policy = meta.get(VALUE_TRANSFORM_POLICY_KEY)
            if isinstance(transform_policy, dict):
                frame.attrs[VALUE_TRANSFORM_POLICY_KEY] = dict(transform_policy)
            frames[metric_key] = frame

        with AtomicOutputSet(
            [*output_paths.values(), log_path],
            cleanup_stale_residues=True,
        ) as output_set:
            for metric_key in required_metrics:
                save_pkl(
                    frames[metric_key],
                    output_set.staged_path(output_paths[metric_key]),
                )
            _append_alignment_history(
                output_set.staged_path(log_path),
                entry=RunLogRecord(
                    step="build_raw_table",
                    completed=True,
                    params=params_with_generation_lineage(
                        {
                            "trial_slug": slug,
                            "picked_epoch_indices": picked,
                            "metrics": required_metrics,
                            "n_metrics": len(required_metrics),
                            "merge_location_info_ready": merge_location_info_ready,
                            "merge_location_info_applied": (
                                merge_location_info_ready and not repcoord_warnings
                            ),
                            "saved_tables": len(frames),
                            "repcoord_merge_warnings": repcoord_warnings,
                        },
                        result_generation_id=new_result_generation_id(),
                        input_generations=input_generations,
                    ),
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
                source_path=log_path,
            )
            if not alignment_finish_input_generations_match(
                resolver,
                trial_slug=slug,
                include_localize=merge_location_info_ready,
                input_generations=input_generations,
            ):
                raise AlignmentInputGenerationChangedError(
                    "Alignment inputs changed while Finish was running; "
                    "candidate outputs were not accepted."
                )
            output_set.commit()
    except AlignmentInputGenerationChangedError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        _append_alignment_history(
            log_path,
            entry=RunLogRecord(
                step="build_raw_table",
                completed=False,
                params={
                    "trial_slug": slug,
                    "picked_epoch_indices": picked,
                    "metrics": required_metrics,
                    "n_metrics": len(required_metrics),
                },
                input_path=str(paradigm_dir),
                output_path=str(paradigm_dir),
                message=f"Finish failed: {exc}",
            ).to_dict(),
            keep_top_level=True,
        )
        return False, f"Finish failed: {exc}"

    invalidate_after_alignment_finish(context, paradigm_slug=slug)
    return True, (
        f"Finish completed. Saved {len(frames)} raw table(s). "
        f"Merge Location Info: {'Ready' if merge_location_info_ready else 'Not Ready'}."
    )
