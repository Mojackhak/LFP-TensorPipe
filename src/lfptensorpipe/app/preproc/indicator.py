"""Draft-aware indicator helpers for editable Preprocess panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log

from .paths import preproc_step_log_path
from .steps.annotations import _normalize_annotation_rows
from .steps.filter import normalize_filter_advance_params


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
        if parsed <= 0.0:
            return None
        normalized.add(float(parsed))
    return sorted(normalized)


def _filter_signature(
    *,
    notches: Any,
    l_freq: Any,
    h_freq: Any,
    advance_params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    ok_advance, normalized_advance, _ = normalize_filter_advance_params(advance_params)
    if not ok_advance:
        return None
    normalized_notches = _normalize_notches(notches)
    if normalized_notches is None:
        return None
    try:
        low_freq = float(l_freq)
        high_freq = float(h_freq)
    except Exception:
        return None
    if low_freq < 0.0 or high_freq <= low_freq:
        return None
    return {
        "low_freq": float(low_freq),
        "high_freq": float(high_freq),
        "notches": normalized_notches,
        "notch_widths": normalized_advance["notch_widths"],
        "epoch_dur": normalized_advance["epoch_dur"],
        "p2p_thresh": list(normalized_advance["p2p_thresh"]),
        "autoreject_correct_factor": normalized_advance["autoreject_correct_factor"],
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
        },
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
    payload = _read_payload(preproc_step_log_path(resolver, "filter"))
    state = _log_state(payload)
    if state is None:
        return "gray"
    if state == "yellow":
        return "yellow"
    assert payload is not None
    completed_signature = _filter_signature_from_log(payload)
    current_signature = _filter_signature(
        notches=notches,
        l_freq=l_freq,
        h_freq=h_freq,
        advance_params=advance_params,
    )
    if completed_signature is None:
        return "green"
    if current_signature is None:
        return "yellow"
    return "green" if current_signature == completed_signature else "yellow"


def _annotations_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    normalized_rows, invalid_rows = _normalize_annotation_rows(rows)
    if invalid_rows:
        return None
    return normalized_rows


def _read_annotations_csv_signature(resolver: PathResolver) -> list[dict[str, Any]] | None:
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
    payload = _read_payload(preproc_step_log_path(resolver, "annotations"))
    state = _log_state(payload)
    if state is None:
        return "gray"
    if state == "yellow":
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
) -> dict[str, Any] | None:
    method_name = str(method).strip().lower()
    if not method_name:
        return None
    if picks is None:
        normalized_picks: list[str] = []
    elif isinstance(picks, (list, tuple)):
        normalized_picks = sorted(
            {
                str(item).strip()
                for item in picks
                if str(item).strip()
            }
        )
    else:
        return None
    return {
        "method": method_name,
        "picks": normalized_picks,
    }


def _ecg_signature_from_log(payload: dict[str, Any]) -> dict[str, Any] | None:
    params = payload.get("params")
    if not isinstance(params, dict):
        return None
    return _normalize_ecg_signature(
        method=params.get("method"),
        picks=params.get("picks"),
    )


def preproc_ecg_panel_state(
    resolver: PathResolver,
    *,
    method: Any,
    picks: Any,
) -> str:
    """Return `gray|yellow|green` for the editable ECG panel."""
    payload = _read_payload(preproc_step_log_path(resolver, "ecg_artifact_removal"))
    state = _log_state(payload)
    if state is None:
        return "gray"
    if state == "yellow":
        return "yellow"
    assert payload is not None
    completed_signature = _ecg_signature_from_log(payload)
    current_signature = _normalize_ecg_signature(method=method, picks=picks)
    if completed_signature is None:
        return "green"
    if current_signature is None:
        return "yellow"
    return "green" if current_signature == completed_signature else "yellow"


__all__ = [
    "preproc_annotations_panel_state",
    "preproc_ecg_panel_state",
    "preproc_filter_panel_state",
]
