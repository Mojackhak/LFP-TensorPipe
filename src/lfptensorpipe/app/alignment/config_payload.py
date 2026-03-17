"""Alignment config payload and method-default normalization helpers."""

from __future__ import annotations

from typing import Any

from lfptensorpipe.app.config_store import AppConfigStore

from .validation import ALIGNMENT_METHODS_BY_KEY, ALIGNMENT_METHODS_BY_LABEL


def _svc():
    from . import service as svc

    return svc


def _read_alignment_payload(config_store: AppConfigStore) -> dict[str, Any]:
    payload = config_store.read_yaml(
        "alignment.yml",
        default={
            "defaults": {"method_params_by_method": {}},
        },
    )
    if not isinstance(payload, dict):
        return {"trials": [], "defaults": {"method_params_by_method": {}}}
    trials = payload.get("trials", payload.get("paradigms", []))
    if not isinstance(trials, list):
        trials = []
    defaults_payload = payload.get("defaults", {})
    method_defaults_raw: dict[str, dict[str, Any]] = {}
    if isinstance(defaults_payload, dict):
        method_defaults_candidate = defaults_payload.get("method_params_by_method", {})
        if isinstance(method_defaults_candidate, dict):
            for raw_key, raw_params in method_defaults_candidate.items():
                key = str(raw_key).strip()
                if key not in ALIGNMENT_METHODS_BY_KEY or not isinstance(raw_params, dict):
                    continue
                method_defaults_raw[key] = dict(raw_params)
    return {
        "trials": trials,
        "defaults": {"method_params_by_method": method_defaults_raw},
    }


def _normalize_method_defaults(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    svc = _svc()
    defaults_payload = payload.get("defaults", {})
    if not isinstance(defaults_payload, dict):
        return {}
    method_defaults_candidate = defaults_payload.get("method_params_by_method", {})
    if not isinstance(method_defaults_candidate, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw_key, raw_params in method_defaults_candidate.items():
        method_key = str(raw_key).strip()
        if method_key not in ALIGNMENT_METHODS_BY_KEY:
            continue
        ok_params, normalized_params, _ = svc.validate_alignment_method_params(
            method_key,
            raw_params if isinstance(raw_params, dict) else {},
        )
        if ok_params:
            out[method_key] = normalized_params
    return out


def _resolve_alignment_method_key(raw_method: Any) -> str:
    token = str(raw_method).strip()
    if token in ALIGNMENT_METHODS_BY_KEY:
        return token
    if token in ALIGNMENT_METHODS_BY_LABEL:
        return ALIGNMENT_METHODS_BY_LABEL[token].key
    return ""


__all__ = [
    "_normalize_method_defaults",
    "_read_alignment_payload",
    "_resolve_alignment_method_key",
]
