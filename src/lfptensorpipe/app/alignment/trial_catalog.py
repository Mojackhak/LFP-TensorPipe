"""Alignment trial catalog loading and legacy config persistence."""

from __future__ import annotations

from typing import Any

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

from .config_payload import _read_alignment_payload


def _svc():
    from . import service as svc

    return svc


def load_alignment_paradigms(
    config_store: AppConfigStore,
    *,
    context: RecordContext | None = None,
) -> list[dict[str, Any]]:
    svc = _svc()
    paradigms: list[dict[str, Any]] = []
    seen: set[str] = set()

    if context is not None:
        resolver = PathResolver(context)
        root = resolver.alignment_root
        if not root.exists():
            return []
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            slug = svc._normalize_slug(path.name)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            cfg = svc._load_trial_config_from_log(resolver, slug=slug)
            if cfg is None:
                cfg = svc._normalize_paradigm(
                    {
                        "name": slug,
                        "trial_slug": slug,
                        "slug": slug,
                        "method": "stack_warper",
                        "method_params": svc.default_alignment_method_params(
                            "stack_warper"
                        ),
                        "annotation_filter": {},
                    }
                )
            paradigms.append(cfg)
        return paradigms

    payload = _read_alignment_payload(config_store)
    for item in payload["trials"]:
        if not isinstance(item, dict):
            continue
        normalized = svc._normalize_paradigm(item)
        slug = normalized["slug"]
        if not slug or slug in seen:
            continue
        seen.add(slug)
        paradigms.append(normalized)
    return paradigms


def save_alignment_paradigms(
    config_store: AppConfigStore,
    paradigms: list[dict[str, Any]],
):
    svc = _svc()
    payload = _read_alignment_payload(config_store)
    method_defaults = svc._normalize_method_defaults(payload)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in paradigms:
        if not isinstance(item, dict):
            continue
        parsed = svc._normalize_paradigm(item)
        slug = parsed["slug"]
        if not slug or slug in seen:
            continue
        seen.add(slug)
        normalized.append(
            {
                "name": parsed["name"],
                "trial_slug": parsed["trial_slug"],
                "method": parsed["method"],
                "method_params": parsed["method_params"],
                "method_params_by_method": parsed["method_params_by_method"],
                "annotation_filter": parsed["annotation_filter"],
            }
        )
    return config_store.write_yaml(
        "alignment.yml",
        {
            "trials": normalized,
            "defaults": {"method_params_by_method": method_defaults},
        },
    )
