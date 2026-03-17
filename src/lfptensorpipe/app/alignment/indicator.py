"""Indicator derivation helpers for Align-Epochs panel-level status lights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app.localize_service import localize_indicator_state
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log

from .trial_config import _load_trial_config_from_log, _normalize_paradigm


def _trial_root(resolver: PathResolver, slug: str) -> Path:
    return resolver.alignment_paradigm_dir(slug, create=False)


def _alignment_log_path(resolver: PathResolver, slug: str) -> Path:
    return _trial_root(resolver, slug) / "lfptensorpipe_log.json"


def _history_entries(
    payload: dict[str, Any] | None,
) -> list[tuple[int, dict[str, Any]]]:
    if not isinstance(payload, dict):
        return []

    history = payload.get("history")
    if isinstance(history, list):
        entries: list[tuple[int, dict[str, Any]]] = []
        for idx, item in enumerate(history):
            if not isinstance(item, dict):
                continue
            step = item.get("step")
            completed = item.get("completed")
            if not isinstance(step, str) or not isinstance(completed, bool):
                continue
            entries.append((idx, item))
        return entries

    step = payload.get("step")
    completed = payload.get("completed")
    if isinstance(step, str) and isinstance(completed, bool):
        return [(0, payload)]
    return []


def _latest_step_entry(
    entries: list[tuple[int, dict[str, Any]]],
    step: str,
) -> tuple[int, dict[str, Any]] | None:
    target = str(step).strip()
    if not target:
        return None
    for idx, item in reversed(entries):
        if str(item.get("step", "")).strip() == target:
            return idx, item
    return None


def _latest_successful_step_entry(
    entries: list[tuple[int, dict[str, Any]]],
    step: str,
) -> tuple[int, dict[str, Any]] | None:
    target = str(step).strip()
    if not target:
        return None
    for idx, item in reversed(entries):
        if str(item.get("step", "")).strip() != target:
            continue
        if bool(item.get("completed", False)):
            return idx, item
    return None


def _normalize_run_signature(
    slug: str,
    entry: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    params = entry.get("params")
    if not isinstance(params, dict):
        return None
    method = params.get("method")
    method_params = params.get("method_params")
    if not isinstance(method, str) or not isinstance(method_params, dict):
        return None
    normalized = _normalize_paradigm(
        {
            "name": str(params.get("name", slug)).strip() or slug,
            "trial_slug": slug,
            "slug": slug,
            "method": method,
            "method_params": method_params,
            "annotation_filter": {},
        }
    )
    return (
        str(normalized.get("method", "")),
        dict(normalized.get("method_params", {})),
    )


def _normalize_current_signature(
    paradigm: dict[str, Any],
) -> tuple[str, str, dict[str, Any]] | None:
    if not isinstance(paradigm, dict):
        return None
    normalized = _normalize_paradigm(dict(paradigm))
    slug = str(normalized.get("slug", "")).strip()
    method = str(normalized.get("method", "")).strip()
    method_params = normalized.get("method_params", {})
    if not slug or not method or not isinstance(method_params, dict):
        return None
    return slug, method, dict(method_params)


def _metric_keys_from_run_entry(
    resolver: PathResolver,
    slug: str,
    entry: dict[str, Any],
) -> list[str]:
    params = entry.get("params")
    metrics: list[str] = []
    if isinstance(params, dict):
        raw_metrics = params.get("metrics")
        if isinstance(raw_metrics, list):
            seen: set[str] = set()
            for item in raw_metrics:
                metric_key = str(item).strip()
                if not metric_key or metric_key in seen:
                    continue
                seen.add(metric_key)
                metrics.append(metric_key)
    if metrics:
        return metrics

    trial_root = _trial_root(resolver, slug)
    discovered: list[str] = []
    seen = set()
    if trial_root.exists():
        for path in sorted(trial_root.glob("*/tensor_warped.pkl")):
            metric_key = path.parent.name.strip()
            if not metric_key or metric_key in seen:
                continue
            seen.add(metric_key)
            discovered.append(metric_key)
    return discovered


def _run_artifacts_exist(
    resolver: PathResolver,
    slug: str,
    entry: dict[str, Any],
) -> bool:
    trial_root = _trial_root(resolver, slug)
    if not trial_root.exists():
        return False
    if not (trial_root / "warp_fn.pkl").exists():
        return False
    if not (trial_root / "warp_labels.pkl").exists():
        return False

    metric_keys = _metric_keys_from_run_entry(resolver, slug, entry)
    if metric_keys:
        return all(
            (trial_root / metric_key / "tensor_warped.pkl").exists()
            for metric_key in metric_keys
        )
    return any(trial_root.glob("*/tensor_warped.pkl"))


def _raw_table_artifacts_exist(
    resolver: PathResolver,
    slug: str,
    run_entry: dict[str, Any],
) -> bool:
    trial_root = _trial_root(resolver, slug)
    if not trial_root.exists():
        return False

    metric_keys = _metric_keys_from_run_entry(resolver, slug, run_entry)
    if metric_keys:
        return all(
            (trial_root / metric_key / "na-raw.pkl").exists()
            for metric_key in metric_keys
        )
    return any(trial_root.glob("*/na-raw.pkl"))


def _normalize_picks(picked_epoch_indices: list[int] | None) -> list[int]:
    if not isinstance(picked_epoch_indices, list):
        return []
    normalized = {
        int(item)
        for item in picked_epoch_indices
        if isinstance(item, (int, float)) and int(item) >= 0
    }
    return sorted(normalized)


def _current_merge_location_info_ready(resolver: PathResolver) -> bool:
    context = resolver.context
    return (
        localize_indicator_state(
            context.project_root,
            context.subject,
            context.record,
        )
        == "green"
    )


def _finished_merge_location_info_ready(
    entry: dict[str, Any],
    *,
    fallback: bool,
) -> bool:
    params = entry.get("params")
    if not isinstance(params, dict):
        return fallback
    value = params.get("merge_location_info_ready")
    if isinstance(value, bool):
        return value
    return fallback


def _latest_finished_picks(
    payload: dict[str, Any] | None,
) -> list[int]:
    entries = _history_entries(payload)
    latest_successful_finish = _latest_successful_step_entry(entries, "build_raw_table")
    if latest_successful_finish is None:
        return []
    finish_params = latest_successful_finish[1].get("params")
    if not isinstance(finish_params, dict):
        return []
    return _normalize_picks(finish_params.get("picked_epoch_indices"))


def alignment_method_panel_state(
    resolver: PathResolver,
    *,
    paradigm: dict[str, Any] | None,
) -> str:
    """Return `gray|yellow|green` for the Align `Method + Params` panel."""
    current = _normalize_current_signature(paradigm or {})
    if current is None:
        return "gray"
    slug, current_method, current_params = current

    try:
        payload = read_run_log(_alignment_log_path(resolver, slug))
    except Exception:
        payload = None

    entries = _history_entries(payload)
    latest_run = _latest_step_entry(entries, "run_align_epochs")
    latest_successful_run = _latest_successful_step_entry(entries, "run_align_epochs")

    if latest_run is None and latest_successful_run is None:
        return "gray"
    if latest_run is not None and not bool(latest_run[1].get("completed", False)):
        return "yellow"
    if latest_successful_run is None:
        return "yellow"

    run_signature = _normalize_run_signature(slug, latest_successful_run[1])
    if run_signature is None:
        return "yellow"
    run_method, run_params = run_signature
    if run_method != current_method or run_params != current_params:
        return "yellow"
    if not _run_artifacts_exist(resolver, slug, latest_successful_run[1]):
        return "yellow"
    return "green"


def alignment_epoch_inspector_state(
    resolver: PathResolver,
    *,
    paradigm: dict[str, Any] | None,
    picked_epoch_indices: list[int] | None,
) -> str:
    """Return `gray|yellow|green` for the Align `Epoch Inspector` panel."""
    current = _normalize_current_signature(paradigm or {})
    if current is None:
        return "gray"
    slug, _method, _params = current

    try:
        payload = read_run_log(_alignment_log_path(resolver, slug))
    except Exception:
        payload = None

    entries = _history_entries(payload)
    latest_run = _latest_step_entry(entries, "run_align_epochs")
    latest_successful_run = _latest_successful_step_entry(entries, "run_align_epochs")
    latest_finish = _latest_step_entry(entries, "build_raw_table")
    latest_successful_finish = _latest_successful_step_entry(entries, "build_raw_table")

    if latest_run is None and latest_finish is None and latest_successful_run is None:
        return "gray"
    if latest_finish is not None and not bool(latest_finish[1].get("completed", False)):
        return "yellow"

    method_state = alignment_method_panel_state(resolver, paradigm=paradigm)
    if method_state != "green":
        return (
            "yellow" if latest_run is not None or latest_finish is not None else "gray"
        )

    if latest_successful_run is None:
        return "gray"
    if latest_successful_finish is None:
        return "yellow"
    if latest_successful_finish[0] < latest_successful_run[0]:
        return "yellow"

    finish_params = latest_successful_finish[1].get("params")
    finished_picks = (
        _normalize_picks(finish_params.get("picked_epoch_indices"))
        if isinstance(finish_params, dict)
        else []
    )
    current_merge_ready = _current_merge_location_info_ready(resolver)
    finished_merge_ready = _finished_merge_location_info_ready(
        latest_successful_finish[1],
        fallback=current_merge_ready,
    )
    if _normalize_picks(picked_epoch_indices) != finished_picks:
        return "yellow"
    if current_merge_ready != finished_merge_ready:
        return "yellow"
    if not _raw_table_artifacts_exist(resolver, slug, latest_successful_run[1]):
        return "yellow"
    return "green"


def alignment_trial_stage_state(
    resolver: PathResolver,
    *,
    paradigm_slug: str,
) -> str:
    """Return result readiness for one trial without relying on current UI picks."""
    slug = str(paradigm_slug).strip()
    if not slug:
        return "gray"
    try:
        payload = read_run_log(_alignment_log_path(resolver, slug))
    except Exception:
        payload = None
    paradigm = _load_trial_config_from_log(resolver, slug=slug)
    if paradigm is None:
        return "gray" if payload is None else "yellow"
    return alignment_epoch_inspector_state(
        resolver,
        paradigm=paradigm,
        picked_epoch_indices=_latest_finished_picks(payload),
    )


__all__ = [
    "alignment_method_panel_state",
    "alignment_epoch_inspector_state",
    "alignment_trial_stage_state",
]
