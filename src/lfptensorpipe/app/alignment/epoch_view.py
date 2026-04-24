"""Alignment epoch-row view helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

from . import service as svc


def _normalize_pick_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    normalized = {
        int(item) for item in value if isinstance(item, (int, float)) and int(item) >= 0
    }
    return sorted(normalized)


def _alignment_log_payload(
    resolver: PathResolver,
    *,
    slug: str,
) -> dict[str, Any] | None:
    alignment_paradigm_log_path = svc.alignment_paradigm_log_path
    read_run_log = svc.read_run_log

    path = alignment_paradigm_log_path(resolver, slug)
    try:
        payload = read_run_log(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _draft_alignment_epoch_picks(payload: dict[str, Any] | None) -> list[int] | None:
    if not isinstance(payload, dict):
        return None
    state = payload.get("state")
    if not isinstance(state, dict):
        return None
    epoch_inspector = state.get("epoch_inspector")
    if not isinstance(epoch_inspector, dict):
        return None
    return _normalize_pick_list(epoch_inspector.get("picked_epoch_indices"))


def _latest_finished_alignment_epoch_picks(
    payload: dict[str, Any] | None,
) -> list[int] | None:
    if not isinstance(payload, dict):
        return None
    history = payload.get("history")
    if not isinstance(history, list):
        return None
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("step", "")).strip() != "build_raw_table":
            continue
        if not bool(item.get("completed", False)):
            continue
        params = item.get("params")
        if not isinstance(params, dict):
            return None
        return _normalize_pick_list(params.get("picked_epoch_indices"))
    return None


def _epoch_duration_s(epoch: Any) -> float:
    """Return the best-effort epoch duration in seconds."""
    for attr in ("total_duration_s", "nominal_duration_s", "duration_s"):
        try:
            value = float(getattr(epoch, attr))
        except Exception:
            continue
        if np.isfinite(value):
            return value
    try:
        start_t = float(getattr(epoch, "start_t"))
        end_t = float(getattr(epoch, "end_t"))
    except Exception:
        return float(np.nan)
    if not np.all(np.isfinite([start_t, end_t])):
        return float(np.nan)
    return float(end_t - start_t)


def load_alignment_epoch_picks(
    context: RecordContext,
    *,
    paradigm_slug: str,
) -> list[int] | None:
    """Load persisted epoch picks for one trial from alignment log state/history."""
    _normalize_slug = svc._normalize_slug

    resolver = PathResolver(context)
    slug = _normalize_slug(paradigm_slug)
    if not slug:
        return None
    payload = _alignment_log_payload(resolver, slug=slug)
    draft_picks = _draft_alignment_epoch_picks(payload)
    if draft_picks is not None:
        return draft_picks
    return _latest_finished_alignment_epoch_picks(payload)


def load_alignment_epoch_rows(
    context: RecordContext, *, paradigm_slug: str
) -> list[dict[str, Any]]:
    """Load saved epoch rows from `warp_labels.pkl` for one trial."""
    _normalize_slug = svc._normalize_slug
    alignment_warp_labels_path = svc.alignment_warp_labels_path
    load_pkl = svc.load_pkl

    resolver = PathResolver(context)
    slug = _normalize_slug(paradigm_slug)
    persisted_picks = load_alignment_epoch_picks(context, paradigm_slug=slug)
    selected_epoch_indices = (
        set(persisted_picks) if persisted_picks is not None else None
    )
    labels_path = alignment_warp_labels_path(resolver, slug)
    if not labels_path.exists():
        return []
    try:
        payload = load_pkl(labels_path)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for idx, epoch in enumerate(payload.get("ALL", [])):
        rows.append(
            {
                "epoch_index": idx,
                "epoch_label": str(getattr(epoch, "label", f"epoch_{idx:03d}")),
                "duration_s": _epoch_duration_s(epoch),
                "start_t": float(getattr(epoch, "start_t", np.nan)),
                "end_t": float(getattr(epoch, "end_t", np.nan)),
                "pick": (
                    True
                    if selected_epoch_indices is None
                    else idx in selected_epoch_indices
                ),
            }
        )
    return rows


def persist_alignment_epoch_picks(
    context: RecordContext,
    *,
    paradigm_slug: str,
    picked_epoch_indices: list[int] | None,
) -> bool:
    """Persist current epoch-inspector draft picks into alignment log state."""
    _normalize_slug = svc._normalize_slug
    alignment_paradigm_log_path = svc.alignment_paradigm_log_path
    update_run_log_state = svc.update_run_log_state

    resolver = PathResolver(context)
    slug = _normalize_slug(paradigm_slug)
    if not slug:
        return False
    log_path = alignment_paradigm_log_path(resolver, slug)
    if not log_path.exists():
        return False
    normalized_picks = _normalize_pick_list(picked_epoch_indices)
    update_run_log_state(
        log_path,
        state_patch={
            "epoch_inspector": {
                "picked_epoch_indices": (
                    list(normalized_picks) if normalized_picks is not None else []
                )
            }
        },
    )
    return True
