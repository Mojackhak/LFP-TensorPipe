"""Alignment-log helpers for feature derivation defaults."""

from __future__ import annotations

from lfptensorpipe.app.alignment_service import alignment_paradigm_log_path
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log


def _extract_alignment_method_from_log(
    resolver: PathResolver,
    *,
    trial_slug: str,
) -> str:
    path = alignment_paradigm_log_path(resolver, trial_slug)
    try:
        payload = read_run_log(path)
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return ""
    state_node = payload.get("state")
    trial_cfg = state_node.get("trial_config") if isinstance(state_node, dict) else None
    if not isinstance(trial_cfg, dict):
        trial_cfg = payload.get("trial_config")
    if isinstance(trial_cfg, dict):
        method = str(trial_cfg.get("method", "")).strip()
        if method:
            return method
    params = payload.get("params")
    if isinstance(params, dict):
        method = str(params.get("method", "")).strip()
        if method:
            return method
    history = payload.get("history")
    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict):
                continue
            cfg = item.get("trial_config")
            if isinstance(cfg, dict):
                method = str(cfg.get("method", "")).strip()
                if method:
                    return method
            params_item = item.get("params")
            if isinstance(params_item, dict):
                method = str(params_item.get("method", "")).strip()
                if method:
                    return method
    return ""
