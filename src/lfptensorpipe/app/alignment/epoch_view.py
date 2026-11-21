"""Alignment epoch-row view helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.runlog_store import _write_updated_run_log_state

from . import service as svc
from .generation import _history_entries


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


def _successful_run_indices(history: list[dict[str, Any]]) -> Iterator[int]:
    """Yield positions in the trial's append-only history, newest Run first."""
    for index in range(len(history) - 1, -1, -1):
        event = history[index]
        if event.get("step") == "run_align_epochs" and event.get("completed") is True:
            yield index


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
    if payload is None:
        return None
    history = _history_entries(payload)
    runs = _successful_run_indices(history)
    run_index = next(runs, -1)
    state = payload.get("state")
    inspector = state.get("epoch_inspector") if isinstance(state, dict) else None
    if isinstance(inspector, dict):
        draft = _normalize_pick_list(inspector.get("picked_epoch_indices"))
        # Older drafts are unambiguous only until the first successful rerun.
        owner = inspector.get("run_history_index")
        if draft is not None and (
            owner == run_index or (owner is None and next(runs, None) is None)
        ):
            return draft

    for event in reversed(history[run_index + 1 :]):
        if event.get("step") == "build_raw_table" and event.get("completed") is True:
            params = event.get("params")
            return (
                _normalize_pick_list(params.get("picked_epoch_indices"))
                if isinstance(params, dict)
                else None
            )
    if run_index >= 0:
        params = history[run_index].get("params")
        count = params.get("n_epochs") if isinstance(params, dict) else None
        if type(count) is int and count >= 0:
            return list(range(count))
    return None


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

    resolver = PathResolver(context)
    slug = _normalize_slug(paradigm_slug)
    if not slug:
        return False
    log_path = alignment_paradigm_log_path(resolver, slug)
    payload = svc.read_run_log(log_path)
    if payload is None:
        return False
    runs = _successful_run_indices(_history_entries(payload))
    _write_updated_run_log_state(
        log_path,
        payload,
        {
            "epoch_inspector": {
                "run_history_index": next(runs, -1),
                "picked_epoch_indices": _normalize_pick_list(picked_epoch_indices)
                or [],
            }
        },
    )
    return True
