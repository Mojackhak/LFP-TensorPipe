"""Draft-aware indicator helpers for editable Preprocess panels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log

from .lineage import (
    filter_preview_lineage_is_current,
    preproc_step_lineage_is_current,
)
from .steps.annotations import (
    _normalize_annotation_rows,
    annotation_log_has_current_support_semantics,
)
from .steps.bad_segment import (
    bad_segment_log_has_current_match_semantics,
)
from .steps.ecg import normalize_ecg_method_params, normalize_ecg_review_params
from .steps.filter import (
    FILTER_EPOCH_COVERAGE_SEMANTICS,
    filter_log_has_current_bad_channel_detection_semantics,
    normalize_filter_advance_params,
)

_UPSTREAM_INVALIDATION_PREFIX = "Invalidated by upstream step re-apply:"


def _step_log_path(resolver: PathResolver, step: str) -> Path:
    return resolver.preproc_step_dir(step, create=False) / "lfptensorpipe_log.json"


def _step_config_path(resolver: PathResolver, step: str) -> Path:
    return resolver.preproc_step_dir(step, create=False) / "config.yml"


def _filter_preview_log_path(resolver: PathResolver) -> Path:
    return resolver.preproc_step_dir("filter", create=False) / "qc" / "preview_log.json"


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_run_log(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _log_state(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    completed = payload.get("completed")
    if isinstance(completed, bool):
        return "green" if completed else "yellow"
    return None


def _filter_preview_payload_is_pending(
    resolver: PathResolver,
    payload: dict[str, Any] | None,
) -> bool:
    if not isinstance(payload, dict) or payload.get("completed") is not False:
        return False
    params = payload.get("params")
    return bool(
        isinstance(params, dict)
        and params.get("review_status") == "required"
        and params.get("filter_output_role") == "preview"
        and (
            resolver.preproc_step_dir("filter", create=False) / "qc" / "preview_raw.fif"
        ).exists()
        and filter_preview_lineage_is_current(resolver, payload)
    )


def _preproc_log_state(
    resolver: PathResolver,
    step: str,
    payload: dict[str, Any] | None,
) -> str:
    state = _log_state(payload)
    if state is None:
        return "gray"
    raw_path = resolver.preproc_step_dir(step, create=False) / "raw.fif"
    if state == "green" and not raw_path.exists():
        return "yellow"
    if state == "yellow" and isinstance(payload, dict):
        message = str(payload.get("message", ""))
        if message.startswith(_UPSTREAM_INVALIDATION_PREFIX) and not raw_path.exists():
            return "gray"
    return state


def preproc_step_indicator_state(resolver: PathResolver, step: str) -> str:
    """Return the effective `gray|yellow|green` state for one preproc step."""
    log_path = _step_log_path(resolver, step)
    payload = _read_payload(log_path)
    state = _preproc_log_state(resolver, step, payload)
    if state != "green":
        return state
    assert payload is not None
    if not preproc_step_lineage_is_current(resolver, step):
        return "yellow"
    if step == "bad_segment_removal" and not (
        bad_segment_log_has_current_match_semantics(payload)
    ):
        return "yellow"
    return "green"


def _normalize_notches(value: Any) -> list[float] | None:
    if value is None:
        return []
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, (list, tuple)):
        return None
    normalized: set[float] = set()
    for item in value:
        try:
            parsed = float(item)
        except Exception:
            return None
        if not math.isfinite(parsed) or parsed <= 0.0:
            return None
        normalized.add(float(parsed))
    return sorted(normalized)


def _filter_signature(
    *,
    notches: Any,
    l_freq: Any,
    h_freq: Any,
    advance_params: dict[str, Any] | None,
    epoch_coverage_semantics: Any = FILTER_EPOCH_COVERAGE_SEMANTICS,
) -> dict[str, Any] | None:
    ok_advance, normalized_advance, _ = normalize_filter_advance_params(advance_params)
    if not ok_advance:
        return None
    normalized_notches = _normalize_notches(notches)
    if normalized_notches is None:
        return None
    try:
        low_freq = (
            None
            if l_freq is None or (isinstance(l_freq, str) and not l_freq.strip())
            else float(l_freq)
        )
        high_freq = (
            None
            if h_freq is None or (isinstance(h_freq, str) and not h_freq.strip())
            else float(h_freq)
        )
    except Exception:
        return None
    if low_freq is not None and (not math.isfinite(low_freq) or low_freq < 0.0):
        return None
    if high_freq is not None and (not math.isfinite(high_freq) or high_freq <= 0.0):
        return None
    if low_freq is not None and high_freq is not None and high_freq <= low_freq:
        return None
    return {
        "low_freq": low_freq,
        "high_freq": high_freq,
        "notches": normalized_notches,
        "notch_widths": (
            normalized_advance["notch_widths"] if normalized_notches else None
        ),
        "epoch_dur": normalized_advance["epoch_dur"],
        "p2p_thresh": (
            None
            if normalized_advance["p2p_thresh"] is None
            else list(normalized_advance["p2p_thresh"])
        ),
        "autoreject_correct_factor": normalized_advance["autoreject_correct_factor"],
        "isolate_bad_boundaries": normalized_advance["isolate_bad_boundaries"],
        "mark_filter_edges": normalized_advance["mark_filter_edges"],
        "epoch_coverage_semantics": epoch_coverage_semantics,
    }


def _filter_signature_from_log(payload: dict[str, Any]) -> dict[str, Any] | None:
    params = payload.get("params")
    if not isinstance(params, dict):
        return None
    return _filter_signature(
        notches=params.get("notches"),
        l_freq=params.get("low_freq"),
        h_freq=params.get("high_freq"),
        advance_params={
            "notch_widths": params.get("notch_widths"),
            "epoch_dur": params.get("epoch_dur"),
            "p2p_thresh": params.get("p2p_thresh"),
            "autoreject_correct_factor": params.get("autoreject_correct_factor"),
            "isolate_bad_boundaries": params.get("isolate_bad_boundaries"),
            "mark_filter_edges": params.get("mark_filter_edges"),
        },
        epoch_coverage_semantics=params.get("epoch_coverage_semantics"),
    )


def preproc_filter_panel_state(
    resolver: PathResolver,
    *,
    notches: Any,
    l_freq: Any,
    h_freq: Any,
    advance_params: dict[str, Any] | None,
) -> str:
    """Return `gray|yellow|green` for the editable Filter panel."""
    preview_payload = _read_payload(_filter_preview_log_path(resolver))
    if _filter_preview_payload_is_pending(resolver, preview_payload):
        return "yellow"
    payload = _read_payload(_step_log_path(resolver, "filter"))
    state = _preproc_log_state(resolver, "filter", payload)
    if state == "gray":
        return "gray"
    if state == "yellow":
        return "yellow"
    assert payload is not None
    if not preproc_step_lineage_is_current(resolver, "filter"):
        return "yellow"
    if not filter_log_has_current_bad_channel_detection_semantics(payload):
        return "yellow"
    completed_signature = _filter_signature_from_log(payload)
    current_signature = _filter_signature(
        notches=notches,
        l_freq=l_freq,
        h_freq=h_freq,
        advance_params=advance_params,
    )
    if completed_signature is None:
        return "yellow"
    if current_signature is None:
        return "yellow"
    return "green" if current_signature == completed_signature else "yellow"


def preproc_filter_review_required(resolver: PathResolver) -> bool:
    """Return whether one valid Filter detection Preview awaits review."""
    payload = _read_payload(_filter_preview_log_path(resolver))
    return bool(
        _filter_preview_payload_is_pending(resolver, payload)
        and filter_log_has_current_bad_channel_detection_semantics(payload)
    )


def _annotations_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    normalized_rows, invalid_rows = _normalize_annotation_rows(rows)
    if invalid_rows:
        return None
    return normalized_rows


def _read_annotations_csv_signature(
    resolver: PathResolver,
) -> list[dict[str, Any]] | None:
    csv_path = resolver.preproc_root / "annotations" / "annotations.csv"
    if not csv_path.exists():
        return None
    import csv

    rows: list[dict[str, Any]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "description": row.get("description", ""),
                        "onset": row.get("onset", ""),
                        "duration": row.get("duration", ""),
                    }
                )
    except Exception:
        return None
    return _annotations_signature(rows)


def preproc_annotations_panel_state(
    resolver: PathResolver,
    *,
    rows: list[dict[str, Any]],
) -> str:
    """Return `gray|yellow|green` for the editable Annotations panel."""
    payload = _read_payload(_step_log_path(resolver, "annotations"))
    state = _preproc_log_state(resolver, "annotations", payload)
    if state == "gray":
        return "gray"
    if state == "yellow":
        return "yellow"
    assert payload is not None
    if not preproc_step_lineage_is_current(resolver, "annotations"):
        return "yellow"
    if not annotation_log_has_current_support_semantics(payload):
        return "yellow"
    completed_signature = _read_annotations_csv_signature(resolver)
    if completed_signature is None:
        return "yellow"
    current_signature = _annotations_signature(rows)
    if current_signature is None:
        return "yellow"
    return "green" if current_signature == completed_signature else "yellow"


def _normalize_ecg_signature(
    *,
    method: Any,
    picks: Any,
    method_kwargs: dict[str, Any] | None,
    mark_filter_edges: Any,
) -> dict[str, Any] | None:
    method_name = str(method).strip().lower()
    if not method_name:
        return None
    if picks is None:
        normalized_picks: list[str] = []
    elif isinstance(picks, (list, tuple)):
        normalized_picks = sorted(
            {str(item).strip() for item in picks if str(item).strip()}
        )
    else:
        return None
    ok_params, normalized_params, _ = normalize_ecg_method_params(
        method_name,
        method_kwargs,
    )
    if not ok_params:
        return None
    ok_review, review_params, _ = normalize_ecg_review_params(
        {"mark_filter_edges": mark_filter_edges}
    )
    if not ok_review:
        return None
    return {
        "method": method_name,
        "picks": normalized_picks,
        "method_kwargs": normalized_params,
        **review_params,
    }


def _read_ecg_config_method_kwargs(
    resolver: PathResolver,
) -> dict[str, Any] | None:
    path = _step_config_path(resolver, "ecg_artifact_removal")
    if not path.exists():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    method_kwargs = payload.get("method_kwargs")
    return method_kwargs if isinstance(method_kwargs, dict) else None


def _ecg_signature_from_log(
    resolver: PathResolver,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    params = payload.get("params")
    if not isinstance(params, dict):
        return None
    method_kwargs = params.get("method_kwargs")
    if not isinstance(method_kwargs, dict):
        method_kwargs = _read_ecg_config_method_kwargs(resolver)
    return _normalize_ecg_signature(
        method=params.get("method"),
        picks=params.get("picks"),
        method_kwargs=method_kwargs,
        mark_filter_edges=params.get("mark_filter_edges", False),
    )


def preproc_ecg_panel_state(
    resolver: PathResolver,
    *,
    method: Any,
    picks: Any,
    method_kwargs: dict[str, Any] | None = None,
    mark_filter_edges: Any = False,
) -> str:
    """Return `gray|yellow|green` for the editable ECG panel."""
    payload = _read_payload(_step_log_path(resolver, "ecg_artifact_removal"))
    state = _preproc_log_state(resolver, "ecg_artifact_removal", payload)
    if state == "gray":
        return "gray"
    if state == "yellow":
        return "yellow"
    assert payload is not None
    if not preproc_step_lineage_is_current(resolver, "ecg_artifact_removal"):
        return "yellow"
    completed_signature = _ecg_signature_from_log(resolver, payload)
    current_signature = _normalize_ecg_signature(
        method=method,
        picks=picks,
        method_kwargs=method_kwargs,
        mark_filter_edges=mark_filter_edges,
    )
    if completed_signature is None:
        return "green"
    if current_signature is None:
        return "yellow"
    return "green" if current_signature == completed_signature else "yellow"


__all__ = [
    "preproc_annotations_panel_state",
    "preproc_ecg_panel_state",
    "preproc_filter_panel_state",
    "preproc_step_indicator_state",
]
