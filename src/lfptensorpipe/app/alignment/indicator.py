"""Indicator derivation helpers for Align-Epochs panel-level status lights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app.localize_service import localize_indicator_state
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log

from .generation import (
    alignment_generation_requires_burst_sample_support_rerun,
    alignment_stage_lineage_is_current,
    metrics_from_alignment_entry,
)
from .method_params import validate_alignment_method_params
from .method_specs import (
    CLIP_STITCH_GEOMETRY,
    CLIP_STITCH_GEOMETRY_KEY,
    LINEAR_EVENT_PAIRING,
    LINEAR_EVENT_PAIRING_KEY,
    LINEAR_WARP_GEOMETRY,
    LINEAR_WARP_GEOMETRY_KEY,
    ZERO_DURATION_ALIGNMENT,
    ZERO_DURATION_ALIGNMENT_KEY,
    ZERO_DURATION_ALIGNMENT_METHODS,
)
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
    slug = str(paradigm.get("trial_slug", paradigm.get("slug", ""))).strip()
    method = str(paradigm.get("method", "")).strip()
    raw_params = paradigm.get("method_params", {})
    if not slug or not method or not isinstance(raw_params, dict):
        return None
    ok, method_params, _ = validate_alignment_method_params(method, raw_params)
    if not ok:
        return None
    if method != "linear_warper" and not method_params.get("annotations"):
        return None
    return slug, method, dict(method_params)


def _has_current_linear_warp_geometry(
    entry: dict[str, Any],
    *,
    method: str,
    method_params: dict[str, Any],
) -> bool:
    if method != "linear_warper" or not bool(method_params.get("linear_warp", True)):
        return True
    metrics = metrics_from_alignment_entry(entry)
    if metrics is None:
        return False
    if all(metric == "burst" for metric in metrics):
        return True
    params = entry.get("params")
    return (
        isinstance(params, dict)
        and params.get(LINEAR_WARP_GEOMETRY_KEY) == LINEAR_WARP_GEOMETRY
    )


def _has_current_linear_event_pairing(
    entry: dict[str, Any],
    *,
    method: str,
) -> bool:
    if method != "linear_warper":
        return True
    params = entry.get("params")
    return (
        isinstance(params, dict)
        and params.get(LINEAR_EVENT_PAIRING_KEY) == LINEAR_EVENT_PAIRING
    )


def _has_current_clip_stitch_geometry(
    entry: dict[str, Any],
    *,
    method: str,
) -> bool:
    if method not in {"pad_warper", "concat_warper"}:
        return True
    params = entry.get("params")
    return (
        isinstance(params, dict)
        and params.get(CLIP_STITCH_GEOMETRY_KEY) == CLIP_STITCH_GEOMETRY
    )


def _has_current_zero_duration_alignment(
    entry: dict[str, Any],
    *,
    method: str,
) -> bool:
    if method not in ZERO_DURATION_ALIGNMENT_METHODS:
        return True
    params = entry.get("params")
    return (
        isinstance(params, dict)
        and params.get(ZERO_DURATION_ALIGNMENT_KEY) == ZERO_DURATION_ALIGNMENT
    )


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
    latest_finish = _latest_step_entry(entries, "build_raw_table")
    if latest_finish is None or latest_finish[1].get("completed") is not True:
        return []
    finish_params = latest_finish[1].get("params")
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

    if latest_run is None:
        return "gray"
    if not bool(latest_run[1].get("completed", False)):
        return "yellow"

    run_signature = _normalize_run_signature(slug, latest_run[1])
    if run_signature is None:
        return "yellow"
    run_method, run_params = run_signature
    if run_method != current_method or run_params != current_params:
        return "yellow"
    if not _has_current_linear_warp_geometry(
        latest_run[1],
        method=run_method,
        method_params=run_params,
    ):
        return "yellow"
    if not _has_current_linear_event_pairing(
        latest_run[1],
        method=run_method,
    ):
        return "yellow"
    if not _has_current_clip_stitch_geometry(
        latest_run[1],
        method=run_method,
    ):
        return "yellow"
    if not _has_current_zero_duration_alignment(
        latest_run[1],
        method=run_method,
    ):
        return "yellow"
    if alignment_generation_requires_burst_sample_support_rerun(latest_run[1]):
        return "yellow"
    if not alignment_stage_lineage_is_current(
        resolver,
        trial_slug=slug,
        stage="run",
    ):
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
    latest_finish = _latest_step_entry(entries, "build_raw_table")

    if latest_run is None and latest_finish is None:
        return "gray"
    if latest_finish is not None and not bool(latest_finish[1].get("completed", False)):
        return "yellow"

    method_state = alignment_method_panel_state(resolver, paradigm=paradigm)
    if method_state != "green":
        return (
            "yellow" if latest_run is not None or latest_finish is not None else "gray"
        )

    if latest_run is None:
        return "gray"
    if latest_finish is None:
        return "yellow"
    if latest_finish[0] < latest_run[0]:
        return "yellow"

    finish_params = latest_finish[1].get("params")
    finished_picks = (
        _normalize_picks(finish_params.get("picked_epoch_indices"))
        if isinstance(finish_params, dict)
        else []
    )
    current_merge_ready = _current_merge_location_info_ready(resolver)
    finished_merge_ready = _finished_merge_location_info_ready(
        latest_finish[1],
        fallback=current_merge_ready,
    )
    if _normalize_picks(picked_epoch_indices) != finished_picks:
        return "yellow"
    if current_merge_ready != finished_merge_ready:
        return "yellow"
    if not alignment_stage_lineage_is_current(
        resolver,
        trial_slug=slug,
        stage="finish",
    ):
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
