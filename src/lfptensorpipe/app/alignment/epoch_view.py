"""Alignment epoch-row view helpers."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.runlog_store import _write_updated_run_log_state

from . import service as svc
from .generation import _history_entries
from .inspector_config import epoch_start_rule, epoch_start_scheme


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
    slug = svc._normalize_slug(paradigm_slug)
    if not slug:
        return False
    return save_epoch_review(
        context,
        slug,
        {
            "picked_epoch_indices": _normalize_pick_list(picked_epoch_indices) or [],
        },
    )


def _epoch_start_report(report, pre):
    """Read both Run reports and exclusion snapshots on the current epoch axis."""
    legacy = [
        key
        for key, node in report.get("resolved", {}).items()
        if node.get("axis") == "cue_seconds"
    ]
    if not legacy:
        return report
    report = deepcopy(report)
    report["config"] = epoch_start_scheme(report["config"], pre)
    for key in legacy:
        node = report["resolved"][key]
        node["axis"] = "epoch_seconds"
        node["time"] = [t + pre for t in node["time"]]
        node["rule"] = epoch_start_rule(node["rule"], pre)
        for epoch in report.get("epochs", [report.get("result")]):
            for hit in epoch["rules"][key]["evidence"]:
                hit["time"] = [t + pre for t in hit["time"]]
                if "missing_time" in hit:
                    hit["missing_time"] = [
                        [lo + pre, hi + pre] for lo, hi in hit["missing_time"]
                    ]
    return report


def _current_review_state(payload):
    review = (payload.get("state") or {}).get("epoch_inspector") or {}
    history = _history_entries(payload)
    owner = next(_successful_run_indices(history), None)
    if owner is not None:
        params = history[owner].get("params", {}).get("method_params", {})
        pre = params.get("cue_response", {}).get("pre_cue_s")
        if pre is not None:
            review = dict(review)
            if review.get("qc_config"):
                review["qc_config"] = epoch_start_scheme(review["qc_config"], pre)
            if review.get("qc_result"):
                review["qc_result"] = _epoch_start_report(review["qc_result"], pre)
            if review.get("exclusions"):
                review["exclusions"] = {
                    k: _epoch_start_report(v, pre)
                    for k, v in review["exclusions"].items()
                }
    report = review.get("qc_result") or {}
    for resolved in report.get("resolved", {}).values():
        bins = resolved.get("frequency_bins", [])
        if (
            resolved["rule"]["method"] == "MAD"
            and bins
            and isinstance(bins[0], (int, float))
            and (len(bins) > 1 or bins[0] <= 0)
            and resolved.get("cluster_frequency_scale") != "log"
        ):
            # Original exclusion evidence stays attached to its manual decision.
            return {**review, "qc_result": None}
    return review


def load_epoch_review(context, slug):
    """Read current Run decisions and trial-wide QC/Preview preferences."""
    payload = _alignment_log_payload(PathResolver(context), slug=slug) or {}
    review = _current_review_state(payload)
    run_index = next(_successful_run_indices(_history_entries(payload)), -1)
    return {**review, "run_history_index": run_index}


def save_epoch_review(context, slug, patch, *, run_index=None):
    """Replace edited inspector fields without merging an old rule or deleted note."""
    path = svc.alignment_paradigm_log_path(PathResolver(context), slug)
    payload = svc.read_run_log(path)
    if not payload:
        return False
    runs = _successful_run_indices(_history_entries(payload))
    owner = next(runs, -1)
    if run_index is not None and owner != run_index:
        raise ValueError("The Align Run changed. Reopen the current trial.")
    state = dict(payload.get("state", {}))
    inspector = dict(state.get("epoch_inspector", {}))
    previous_owner = inspector.get("run_history_index")
    if previous_owner != owner and (
        previous_owner is not None or next(runs, None) is not None
    ):
        inspector["picked_epoch_indices"] = None
    inspector.update(patch, run_history_index=owner)
    state["epoch_inspector"] = inspector
    _write_updated_run_log_state(path, {**payload, "state": state}, {})
    return True


def finish_review_snapshot(context, slug, picked, epochs):
    """Attach Run-owned decisions to the source geometry already read by Finish."""
    from lfptensorpipe.lfp.warp.pad_concat import positive_pad_intervals

    payload = _alignment_log_payload(PathResolver(context), slug=slug) or {}
    owner = next(_successful_run_indices(_history_entries(payload)), None)
    review = _current_review_state(payload)
    decisions = []
    accepted = set(picked)
    for i, epoch in enumerate(epochs):
        events = epoch.get("events_t") or {}
        intervals = epoch.get("intervals_s")
        if intervals is None and "pad_left" in events:
            intervals = positive_pad_intervals(events)
        if intervals is None and np.all(
            np.isfinite([epoch.get("start_t", np.nan), epoch.get("end_t", np.nan)])
        ):
            intervals = [[epoch["start_t"], epoch["end_t"]]]
        key = str(i)
        decisions.append(
            {
                "epoch_index": i,
                "epoch": epoch["label"],
                "source_intervals_s": intervals,
                "source_events": events,
                "include": i in accepted,
                "notes": (review.get("notes") or {}).get(key, ""),
                "automatic_exclusion": (
                    (review.get("exclusions") or {}).get(key)
                    if i not in accepted
                    else None
                ),
            }
        )
    return {
        "run_history_index": owner,
        "qc_result": review.get("qc_result"),
        "decisions": decisions,
    }
