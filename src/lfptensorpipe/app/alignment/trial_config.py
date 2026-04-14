"""Alignment trial-config normalization and log-history helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import append_run_log_event, read_run_log

from .paths import alignment_paradigm_log_path
from .validation import ALIGNMENT_METHODS_BY_KEY


def _svc():
    from . import service as svc

    return svc


def _normalize_method_params_by_method(
    item: dict[str, Any],
    *,
    active_method: str,
    active_method_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    svc = _svc()
    out: dict[str, dict[str, Any]] = {}
    raw_cache = item.get("method_params_by_method", {})
    if isinstance(raw_cache, dict):
        for raw_key, raw_params in raw_cache.items():
            method_key = str(raw_key).strip()
            if method_key not in ALIGNMENT_METHODS_BY_KEY:
                continue
            if not isinstance(raw_params, dict):
                continue
            ok_params, normalized_params, _ = svc.validate_alignment_method_params(
                method_key,
                raw_params,
            )
            if ok_params:
                out[method_key] = normalized_params
    out[active_method] = dict(active_method_params)
    return out


def _normalize_paradigm(item: dict[str, Any]) -> dict[str, Any]:
    svc = _svc()
    name = str(item.get("name", "")).strip()
    slug = svc._normalize_slug(str(item.get("trial_slug", item.get("slug", ""))))
    method = str(item.get("method", "stack_warper")).strip()
    if method not in ALIGNMENT_METHODS_BY_KEY:
        method = "stack_warper"
    method_params = item.get("method_params", {})
    ok_params, normalized_params, _ = svc.validate_alignment_method_params(
        method,
        method_params,
    )
    if not ok_params:
        normalized_params = svc.default_alignment_method_params(method)
    annotation_filter = item.get("annotation_filter", {})
    if not isinstance(annotation_filter, dict):
        annotation_filter = {}
    if not name:
        name = slug or "Trial"
    if not slug:
        slug = svc._normalize_slug(name)
    method_params_by_method = _normalize_method_params_by_method(
        item,
        active_method=method,
        active_method_params=normalized_params,
    )
    return {
        "name": name,
        "trial_slug": slug,
        "slug": slug,
        "method": method,
        "method_params": normalized_params,
        "method_params_by_method": method_params_by_method,
        "annotation_filter": annotation_filter,
    }


def _trial_config_from_payload(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    state_node = payload.get("state")
    raw_cfg = state_node.get("trial_config") if isinstance(state_node, dict) else None
    if not isinstance(raw_cfg, dict):
        raw_cfg = payload.get("trial_config")
    if not isinstance(raw_cfg, dict):
        return None
    return _normalize_paradigm(raw_cfg)


def _extract_trial_config_from_history(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    history = payload.get("history")
    if not isinstance(history, list):
        return None
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        cfg = _trial_config_from_payload(item)
        if cfg is not None:
            return cfg
        params = item.get("params")
        if not isinstance(params, dict):
            continue
        method = params.get("method")
        method_params = params.get("method_params")
        if isinstance(method, str) and isinstance(method_params, dict):
            name = str(params.get("name", params.get("trial_slug", "Trial"))).strip()
            slug = str(params.get("trial_slug", "")).strip()
            return _normalize_paradigm(
                {
                    "name": name or slug or "Trial",
                    "trial_slug": slug,
                    "slug": slug,
                    "method": method,
                    "method_params": method_params,
                    "method_params_by_method": params.get("method_params_by_method"),
                    "annotation_filter": {},
                }
            )
    return None


def _load_trial_config_from_log(
    resolver: PathResolver,
    *,
    slug: str,
) -> dict[str, Any] | None:
    path = alignment_paradigm_log_path(resolver, slug)
    try:
        payload = read_run_log(path)
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return None
    cfg = _trial_config_from_payload(payload)
    if cfg is not None:
        return cfg
    params = payload.get("params")
    if isinstance(params, dict):
        method = params.get("method")
        method_params = params.get("method_params")
        if isinstance(method, str) and isinstance(method_params, dict):
            return _normalize_paradigm(
                {
                    "name": str(params.get("name", slug)).strip() or slug,
                    "trial_slug": slug,
                    "slug": slug,
                    "method": method,
                    "method_params": method_params,
                    "method_params_by_method": params.get("method_params_by_method"),
                    "annotation_filter": {},
                }
            )
    return _extract_trial_config_from_history(payload)


def _finish_time_axis_values(
    axes: dict[str, Any],
    *,
    method_key: str,
    n_time: int,
) -> list[Any]:
    if method_key in {"linear_warper", "stack_warper"}:
        candidate = axes.get("percent")
    elif method_key in {"pad_warper", "concat_warper"}:
        candidate = axes.get("time")
    else:
        candidate = None
    if candidate is None:
        return list(range(n_time))
    return list(candidate)


def _append_alignment_history(
    path: Path,
    *,
    entry: dict[str, Any],
    keep_top_level: bool,
    trial_config: dict[str, Any] | None = None,
) -> None:
    _ = keep_top_level
    state_patch = (
        {"trial_config": dict(trial_config)} if isinstance(trial_config, dict) else None
    )
    append_run_log_event(path, entry, state_patch=state_patch)


__all__ = [
    "_append_alignment_history",
    "_extract_trial_config_from_history",
    "_finish_time_axis_values",
    "_load_trial_config_from_log",
    "_normalize_paradigm",
    "_trial_config_from_payload",
]
