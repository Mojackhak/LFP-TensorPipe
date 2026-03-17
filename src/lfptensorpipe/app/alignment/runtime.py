"""Alignment runtime runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_alignment_run,
)

from . import service as svc
from .epoch_view import _epoch_duration_s


def run_align_epochs(
    context: RecordContext,
    *,
    config_store: AppConfigStore,
    paradigm_slug: str,
    load_alignment_paradigms_fn: Any | None = None,
    load_raw_for_warp_fn: Any | None = None,
    validate_alignment_method_params_fn: Any | None = None,
    update_alignment_paradigm_fn: Any | None = None,
    build_warper_fn: Any | None = None,
    save_pkl_fn: Any | None = None,
    load_pkl_fn: Any | None = None,
    build_warped_tensor_metadata_fn: Any | None = None,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Run Align Epochs for one selected trial."""
    _normalize_slug = svc._normalize_slug
    load_alignment_paradigms = load_alignment_paradigms_fn or svc.load_alignment_paradigms
    indicator_from_log = svc.indicator_from_log
    preproc_step_log_path = svc.preproc_step_log_path
    _completed_tensor_metrics = svc._completed_tensor_metrics
    preproc_step_raw_path = svc.preproc_step_raw_path
    _load_raw_for_warp = load_raw_for_warp_fn or svc._load_raw_for_warp
    validate_alignment_method_params = (
        validate_alignment_method_params_fn or svc.validate_alignment_method_params
    )
    default_alignment_method_params = svc.default_alignment_method_params
    update_alignment_paradigm = update_alignment_paradigm_fn or svc.update_alignment_paradigm
    _build_warper = build_warper_fn or svc._build_warper
    _resolve_target_n_samples = svc._resolve_target_n_samples
    save_pkl = save_pkl_fn or svc.save_pkl
    alignment_warp_fn_path = svc.alignment_warp_fn_path
    alignment_warp_labels_path = svc.alignment_warp_labels_path
    tensor_metric_tensor_path = svc.tensor_metric_tensor_path
    load_pkl = load_pkl_fn or svc.load_pkl
    _coerce_alignment_tensor = svc._coerce_alignment_tensor
    infer_sfreq_from_times = svc.infer_sfreq_from_times
    build_warped_tensor_metadata = (
        build_warped_tensor_metadata_fn or svc.build_warped_tensor_metadata
    )
    alignment_metric_tensor_warped_path = svc.alignment_metric_tensor_warped_path
    _normalize_paradigm = svc._normalize_paradigm
    _append_alignment_history = svc._append_alignment_history
    alignment_paradigm_log_path = svc.alignment_paradigm_log_path
    RunLogRecord = svc.RunLogRecord
    alignment_paradigm_dir = svc.alignment_paradigm_dir

    resolver = PathResolver(context)
    resolver.ensure_record_roots()
    slug = _normalize_slug(paradigm_slug)
    paradigms = load_alignment_paradigms(config_store, context=context)
    paradigm = next(
        (
            item
            for item in paradigms
            if str(item.get("trial_slug", item.get("slug", ""))) == slug
        ),
        None,
    )
    if paradigm is None:
        return False, f"Trial not found: {slug}", []

    if indicator_from_log(preproc_step_log_path(resolver, "finish")) != "green":
        return False, "Preprocess finish must be green before Align Epochs.", []

    metrics = _completed_tensor_metrics(resolver)
    if not metrics:
        return False, "No completed tensor metrics available.", []

    finish_raw = preproc_step_raw_path(resolver, "finish")
    if not finish_raw.exists():
        return False, "Missing preproc finish raw.fif.", []

    method = str(paradigm.get("method", "stack_warper")).strip()
    method_params = paradigm.get("method_params", {})
    if not isinstance(method_params, dict):
        method_params = {}

    try:
        raw = _load_raw_for_warp(finish_raw)
        labels = sorted(
            set(str(item) for item in raw.annotations.description if str(item).strip())
        )
        valid_params, normalized_params, message = validate_alignment_method_params(
            method,
            method_params,
            annotation_labels=labels,
        )
        if not valid_params:
            raise ValueError(f"Invalid method params: {message}")
        method_params = normalized_params
        selected_annotations = method_params.get("annotations", [])
        if not isinstance(selected_annotations, list):
            selected_annotations = []
        selected_annotations = [
            str(item).strip() for item in selected_annotations if str(item).strip()
        ]
        if method in {"pad_warper", "stack_warper", "concat_warper"} and not selected_annotations:
            raise ValueError("Select at least one annotation label.")
        sample_rate = float(
            method_params.get(
                "sample_rate",
                default_alignment_method_params(method).get("sample_rate", 5.0),
            )
        )

        _ = update_alignment_paradigm(
            config_store,
            slug=slug,
            method=method,
            method_params=method_params,
            context=context,
        )
        epochs_by_label, warp_fn = _build_warper(
            raw,
            method=method,
            method_params=method_params,
        )
        n_samples = _resolve_target_n_samples(
            method=method,
            method_params=method_params,
            epochs_by_label=epochs_by_label,
        )
        if hasattr(raw, "close"):
            raw.close()
        save_pkl(warp_fn, alignment_warp_fn_path(resolver, slug))
        save_pkl(epochs_by_label, alignment_warp_labels_path(resolver, slug))

        epoch_rows: list[dict[str, Any]] = []
        for idx, epoch in enumerate(epochs_by_label.get("ALL", [])):
            start_t = float(getattr(epoch, "start_t", np.nan))
            end_t = float(getattr(epoch, "end_t", np.nan))
            epoch_rows.append(
                {
                    "epoch_index": idx,
                    "epoch_label": str(getattr(epoch, "label", f"epoch_{idx:03d}")),
                    "duration_s": _epoch_duration_s(epoch),
                    "start_t": start_t,
                    "end_t": end_t,
                    "pick": True,
                }
            )
        if not epoch_rows:
            raise ValueError("No valid epochs detected for selected trial.")

        for metric_key in metrics:
            tensor_path = tensor_metric_tensor_path(resolver, metric_key)
            payload = load_pkl(tensor_path)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid tensor payload for metric: {metric_key}")
            tensor_3d, meta_in = _coerce_alignment_tensor(payload)
            axes = meta_in.get("axes", {})
            if not isinstance(axes, dict) or "time" not in axes:
                raise ValueError(
                    f"Tensor metadata missing time axis for metric: {metric_key}"
                )
            sr = infer_sfreq_from_times(axes.get("time"), default=40.0)
            warped, percent_axis, meta_epochs = warp_fn(
                tensor_3d,
                sr=float(sr),
                n_samples=n_samples,
            )
            warped_arr = np.asarray(warped, dtype=float)
            if warped_arr.ndim != 4:
                raise ValueError(
                    f"Warped tensor has invalid shape for {metric_key}: {warped_arr.shape}"
                )
            meta_warped = build_warped_tensor_metadata(
                axes,
                np.asarray(percent_axis, dtype=float),
                meta_epochs,
                source_meta=meta_in,
            )
            out_path = alignment_metric_tensor_warped_path(resolver, slug, metric_key)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_pkl({"tensor": warped_arr, "meta": meta_warped}, out_path)

        trial_config_payload = _normalize_paradigm(
            {
                "name": paradigm.get("name", slug),
                "trial_slug": slug,
                "slug": slug,
                "method": method,
                "method_params": method_params,
                "annotation_filter": paradigm.get("annotation_filter", {}),
            }
        )
        _append_alignment_history(
            alignment_paradigm_log_path(resolver, slug),
            entry=RunLogRecord(
                step="run_align_epochs",
                completed=True,
                params={
                    "trial_slug": slug,
                    "name": trial_config_payload.get("name", slug),
                    "method": method,
                    "method_params": method_params,
                    "sample_rate": sample_rate,
                    "warped_n_samples": int(n_samples),
                    "n_metrics": len(metrics),
                    "n_epochs": len(epoch_rows),
                    "metrics": metrics,
                },
                input_path=str(resolver.tensor_root),
                output_path=str(alignment_paradigm_dir(resolver, slug)),
                message="Align Epochs completed.",
            ).to_dict(),
            keep_top_level=False,
            trial_config=trial_config_payload,
        )
        invalidate_after_alignment_run(context, paradigm_slug=slug)
        return True, "Align Epochs completed.", epoch_rows
    except Exception as exc:  # noqa: BLE001
        trial_config_payload = _normalize_paradigm(
            {
                "name": (
                    paradigm.get("name", slug) if isinstance(paradigm, dict) else slug
                ),
                "trial_slug": slug,
                "slug": slug,
                "method": method,
                "method_params": method_params,
                "annotation_filter": (
                    paradigm.get("annotation_filter", {})
                    if isinstance(paradigm, dict)
                    else {}
                ),
            }
        )
        _append_alignment_history(
            alignment_paradigm_log_path(resolver, slug),
            entry=RunLogRecord(
                step="run_align_epochs",
                completed=False,
                params={
                    "trial_slug": slug,
                    "name": trial_config_payload.get("name", slug),
                    "method": method,
                    "method_params": method_params,
                },
                input_path=str(resolver.tensor_root),
                output_path=str(alignment_paradigm_dir(resolver, slug)),
                message=f"Align Epochs failed: {exc}",
            ).to_dict(),
            keep_top_level=False,
            trial_config=trial_config_payload,
        )
        return False, f"Align Epochs failed: {exc}", []
