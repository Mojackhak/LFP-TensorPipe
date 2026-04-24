"""Indicator and path helpers for Extract-Features stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import indicator_from_log, read_run_log

from .derive_axes import _normalize_axis_rows

_AUTO_BAND_METRICS = {"psi", "burst"}


def _normalize_slug(value: str) -> str:
    token = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.strip().lower()
    )
    token = token.strip("-_")
    token = token.replace("_", "-")
    return token


def _aggregate_states(log_paths: list[Path]) -> str:
    if not log_paths:
        return "gray"
    states = [indicator_from_log(path) for path in log_paths]
    if any(state == "yellow" for state in states):
        return "yellow"
    if all(state == "green" for state in states):
        return "green"
    if any(state == "green" for state in states):
        return "yellow"
    return "gray"


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_run_log(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_axes_node(
    metric_key: str, value: Any
) -> dict[str, list[dict[str, float | str]]] | None:
    if not isinstance(value, dict):
        return None
    bands = (
        []
        if metric_key in _AUTO_BAND_METRICS
        else _normalize_axis_rows(
            value.get("bands"),
            allow_duplicate_names=False,
        )
    )
    times = _normalize_axis_rows(
        value.get("times"),
        allow_duplicate_names=True,
    )
    return {
        "bands": [dict(item) for item in bands],
        "times": [dict(item) for item in times],
    }


def _normalize_axes_by_metric(
    value: Any,
) -> dict[str, dict[str, list[dict[str, float | str]]]] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    for raw_metric_key in sorted(value.keys(), key=lambda item: str(item).strip()):
        metric_key = str(raw_metric_key).strip()
        if not metric_key:
            continue
        axis_node = _normalize_axes_node(metric_key, value.get(raw_metric_key))
        if axis_node is None:
            return None
        normalized[metric_key] = axis_node
    return normalized


def features_derivatives_root(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    transformed: bool | None = None,
    create: bool = False,
) -> Path:
    """Return feature-output root for one trial."""
    _ = transformed
    slug = _normalize_slug(str(trial_slug if trial_slug is not None else paradigm_slug))
    out = resolver.features_root / slug
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def features_derivatives_log_path(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    transformed: bool | None = None,
    create: bool = False,
) -> Path:
    """Return Extract-Features trial log path."""
    return (
        features_derivatives_root(
            resolver,
            trial_slug=trial_slug,
            paradigm_slug=paradigm_slug,
            transformed=transformed,
            create=create,
        )
        / "lfptensorpipe_log.json"
    )


def features_normalization_root(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    transformed: bool | None = None,
    create: bool = False,
) -> Path:
    """Deprecated: kept for compatibility; no new files should be written."""
    slug = _normalize_slug(str(trial_slug if trial_slug is not None else paradigm_slug))
    root_name = "normalization_transformed" if transformed else "normalization"
    out = resolver.features_root / root_name / slug
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def features_normalization_log_path(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    transformed: bool | None = None,
    create: bool = False,
) -> Path:
    """Deprecated: kept for compatibility; no new logs should be written."""
    return (
        features_normalization_root(
            resolver,
            trial_slug=trial_slug,
            paradigm_slug=paradigm_slug,
            transformed=transformed,
            create=create,
        )
        / "lfptensorpipe_log.json"
    )


def extract_features_indicator_state(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
) -> str:
    """Derive Extract-Features indicator for one trial."""
    log_path = features_derivatives_log_path(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
    )
    return _aggregate_states([log_path])


def features_panel_state(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    axes_by_metric: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Return `gray|yellow|green` for the inline editable Features panel light."""
    log_path = features_derivatives_log_path(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
    )
    payload = _read_payload(log_path)
    if payload is None:
        return "gray"
    completed = payload.get("completed")
    if completed is False:
        return "yellow"
    if completed is not True:
        return "gray"
    params = payload.get("params")
    logged_axes = None
    if isinstance(params, dict):
        logged_axes = _normalize_axes_by_metric(params.get("axes_by_metric"))
    if logged_axes is None:
        return "green"
    current_axes = _normalize_axes_by_metric(axes_by_metric)
    if current_axes is None:
        return "yellow"
    return "green" if current_axes == logged_axes else "yellow"


def normalization_indicator_state(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
) -> str:
    """Deprecated normalization indicator (always gray in new workflow)."""
    _ = (resolver, trial_slug, paradigm_slug)
    return "gray"
