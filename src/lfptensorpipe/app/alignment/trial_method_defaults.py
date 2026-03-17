"""Alignment method default-param persistence helpers."""

from __future__ import annotations

from typing import Any

from lfptensorpipe.app.config_store import AppConfigStore

from .config_payload import _normalize_method_defaults, _read_alignment_payload
from .validation import ALIGNMENT_METHODS_BY_KEY


def _svc():
    from . import service as svc

    return svc


def load_alignment_method_defaults(
    config_store: AppConfigStore,
) -> dict[str, dict[str, Any]]:
    payload = _read_alignment_payload(config_store)
    return _normalize_method_defaults(payload)


def load_alignment_method_default_params(
    config_store: AppConfigStore,
    *,
    method_key: str,
) -> dict[str, Any]:
    svc = _svc()
    key = str(method_key).strip()
    if key not in ALIGNMENT_METHODS_BY_KEY:
        return svc.default_alignment_method_params("stack_warper")
    defaults = svc.load_alignment_method_defaults(config_store)
    params = defaults.get(key)
    if isinstance(params, dict):
        return dict(params)
    return svc.default_alignment_method_params(key)


def save_alignment_method_default_params(
    config_store: AppConfigStore,
    *,
    method_key: str,
    method_params: dict[str, Any],
    annotation_labels: list[str] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    svc = _svc()
    key = str(method_key).strip()
    if key not in ALIGNMENT_METHODS_BY_KEY:
        return False, f"Unknown method: {key}", None
    ok_params, normalized_params, message = svc.validate_alignment_method_params(
        key,
        method_params,
        annotation_labels=annotation_labels,
    )
    if not ok_params:
        return False, message, None
    paradigms = svc.load_alignment_paradigms(config_store)
    defaults = svc.load_alignment_method_defaults(config_store)
    defaults[key] = dict(normalized_params)
    config_store.write_yaml(
        "alignment.yml",
        {
            "trials": [
                {
                    "name": item.get("name"),
                    "trial_slug": item.get("trial_slug", item.get("slug")),
                    "method": item.get("method"),
                    "method_params": item.get("method_params"),
                    "method_params_by_method": item.get("method_params_by_method", {}),
                    "annotation_filter": item.get("annotation_filter", {}),
                }
                for item in paradigms
            ],
            "defaults": {"method_params_by_method": defaults},
        },
    )
    return True, "Default params saved.", normalized_params
