"""Alignment trial CRUD helpers."""

from __future__ import annotations

import shutil
from typing import Any

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import RunLogRecord, read_ui_state, write_ui_state

from .paths import alignment_paradigm_dir, alignment_paradigm_log_path
from .validation import ALIGNMENT_METHODS_BY_KEY


def _svc():
    from . import service as svc

    return svc


def _default_trial_entry(svc: Any, slug: str) -> dict[str, Any]:
    default_params = svc.default_alignment_method_params("stack_warper")
    return {
        "name": slug or "Trial",
        "trial_slug": slug,
        "slug": slug,
        "method": "stack_warper",
        "method_params": default_params,
        "method_params_by_method": {"stack_warper": dict(default_params)},
        "annotation_filter": {},
    }


def _stored_method_params_for_method(
    svc: Any,
    *,
    existing: dict[str, Any],
    method_key: str,
) -> dict[str, Any]:
    raw_cache = existing.get("method_params_by_method", {})
    if isinstance(raw_cache, dict):
        raw_cached = raw_cache.get(method_key)
        if isinstance(raw_cached, dict):
            return dict(raw_cached)
    if str(existing.get("method", "")).strip() == method_key:
        raw_existing = existing.get("method_params")
        if isinstance(raw_existing, dict):
            return dict(raw_existing)
    return svc.default_alignment_method_params(method_key)


def _merge_method_params_by_method(
    svc: Any,
    *,
    existing: dict[str, Any],
    active_method: str,
    active_method_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    raw_cache = existing.get("method_params_by_method", {})
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
                merged[method_key] = normalized_params
    merged[active_method] = dict(active_method_params)
    return merged


def _resolve_update_method_params(
    svc: Any,
    *,
    active_method: str,
    existing: dict[str, Any],
    method_params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any] | None, str]:
    if method_params is not None:
        params_candidate = dict(method_params)
    else:
        params_candidate = _stored_method_params_for_method(
            svc,
            existing=existing,
            method_key=active_method,
        )
    ok_params, normalized_params, message = svc.validate_alignment_method_params(
        active_method,
        params_candidate,
    )
    if not ok_params:
        return False, None, message
    return True, normalized_params, ""


def _remove_legacy_trial_entry(
    svc: Any,
    *,
    config_store: AppConfigStore,
    slug: str,
) -> bool:
    paradigms = svc.load_alignment_paradigms(config_store)
    kept = [
        item
        for item in paradigms
        if str(item.get("trial_slug", item.get("slug", ""))) != slug
    ]
    if len(kept) == len(paradigms):
        return False
    svc.save_alignment_paradigms(config_store, kept)
    return True


def _trial_artifact_dirs(
    *,
    resolver: PathResolver,
    slug: str,
) -> list[Any]:
    return [
        resolver.alignment_paradigm_dir(slug, create=False),
        resolver.features_root / slug,
        resolver.features_root / "raw" / slug,
        resolver.features_root / "derivatives" / slug,
        resolver.features_root / "derivatives_transformed" / slug,
        resolver.features_root / "normalization" / slug,
        resolver.features_root / "normalization_transformed" / slug,
    ]


def _remove_trial_artifacts(
    *,
    resolver: PathResolver,
    slug: str,
) -> bool:
    removed_any = False
    for path in _trial_artifact_dirs(resolver=resolver, slug=slug):
        if not path.exists():
            continue
        shutil.rmtree(path, ignore_errors=False)
        removed_any = True
    return removed_any


def _clear_deleted_trial_from_ui_state(
    *,
    context: RecordContext,
    slug: str,
) -> bool:
    resolver = PathResolver(context)
    path = resolver.record_ui_state_path(create=False)
    if not path.is_file():
        return False
    payload = read_ui_state(path)
    if not isinstance(payload, dict):
        return False

    changed = False
    alignment_node = payload.get("alignment")
    if isinstance(alignment_node, dict):
        if str(alignment_node.get("trial_slug", "")).strip() == slug:
            alignment_node["trial_slug"] = None
            alignment_node["paradigm_slug"] = None
            alignment_node["method"] = None
            alignment_node["sample_rate"] = None
            alignment_node["epoch_metric"] = None
            alignment_node["epoch_channel"] = None
            alignment_node["picked_epoch_indices"] = []
            changed = True
        elif str(alignment_node.get("paradigm_slug", "")).strip() == slug:
            alignment_node["paradigm_slug"] = None
            changed = True

    features_node = payload.get("features")
    if isinstance(features_node, dict):
        if str(features_node.get("paradigm_slug", "")).strip() == slug:
            features_node["paradigm_slug"] = None
            changed = True
        if str(features_node.get("trial_slug", "")).strip() == slug:
            features_node["trial_slug"] = None
            changed = True

    if changed:
        write_ui_state(path, payload)
    return changed


def create_alignment_paradigm(
    config_store: AppConfigStore,
    *,
    name: str,
    context: RecordContext | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    svc = _svc()
    parsed_name = str(name).strip()
    if not parsed_name:
        return False, "Trial name cannot be empty.", None
    slug_base = svc._normalize_slug(parsed_name)
    if not slug_base:
        return False, "Failed to generate trial slug.", None

    paradigms = svc.load_alignment_paradigms(config_store, context=context)
    existing = {str(item.get("trial_slug", item.get("slug", ""))) for item in paradigms}
    slug = slug_base
    idx = 2
    while slug in existing:
        slug = f"{slug_base}-{idx}"
        idx += 1

    entry = _default_trial_entry(svc, slug)
    entry["name"] = parsed_name
    if context is None:
        paradigms.append(entry)
        svc.save_alignment_paradigms(config_store, paradigms)
        return True, f"Trial created: {slug}", entry

    resolver = PathResolver(context)
    out_dir = alignment_paradigm_dir(resolver, slug, create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_payload = {
        "trial_slug": slug,
        "name": parsed_name,
        "method": "stack_warper",
        "method_params": dict(entry["method_params"]),
        "method_params_by_method": {
            key: dict(value)
            for key, value in entry["method_params_by_method"].items()
        },
    }
    svc._append_alignment_history(
        alignment_paradigm_log_path(resolver, slug),
        entry=RunLogRecord(
            step="trial_config",
            completed=False,
            params=params_payload,
            input_path=str(out_dir),
            output_path=str(out_dir),
            message="Trial configuration initialized.",
        ).to_dict(),
        keep_top_level=False,
        trial_config=entry,
    )
    return True, f"Trial created: {slug}", entry


def delete_alignment_paradigm(
    config_store: AppConfigStore,
    *,
    slug: str,
    context: RecordContext | None = None,
) -> tuple[bool, str]:
    svc = _svc()
    target = svc._normalize_slug(slug)
    if not target:
        return False, "Trial slug is empty."
    if context is None:
        if not _remove_legacy_trial_entry(svc, config_store=config_store, slug=target):
            return False, f"Trial not found: {target}"
        return True, f"Trial deleted: {target}"

    resolver = PathResolver(context)
    removed_artifacts = False
    try:
        removed_artifacts = _remove_trial_artifacts(
            resolver=resolver,
            slug=target,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to delete trial: {exc}"
    try:
        removed_ui_state = _clear_deleted_trial_from_ui_state(
            context=context,
            slug=target,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to delete trial UI state: {exc}"
    removed_legacy = _remove_legacy_trial_entry(
        svc,
        config_store=config_store,
        slug=target,
    )
    if not removed_artifacts and not removed_legacy and not removed_ui_state:
        return False, f"Trial not found: {target}"
    return True, f"Trial deleted: {target}"


def update_alignment_paradigm(
    config_store: AppConfigStore,
    *,
    slug: str,
    method: str | None = None,
    method_params: dict[str, Any] | None = None,
    context: RecordContext | None = None,
    load_alignment_paradigms_fn: Any | None = None,
    save_alignment_paradigms_fn: Any | None = None,
) -> tuple[bool, str]:
    svc = _svc()
    load_paradigms = load_alignment_paradigms_fn or svc.load_alignment_paradigms
    save_paradigms = save_alignment_paradigms_fn or svc.save_alignment_paradigms
    target = svc._normalize_slug(slug)
    if context is not None:
        resolver = PathResolver(context)
        trial_dir = resolver.alignment_paradigm_dir(target, create=False)
        if not trial_dir.exists():
            return False, f"Trial not found: {target}"
        current_cfg = svc._load_trial_config_from_log(
            resolver,
            slug=target,
        ) or svc._normalize_paradigm(_default_trial_entry(svc, target))
        active_method = str(current_cfg.get("method", "stack_warper")).strip()
        if method is not None:
            candidate = str(method).strip()
            if candidate not in ALIGNMENT_METHODS_BY_KEY:
                return False, f"Unknown method: {candidate}"
            active_method = candidate
        ok_params, normalized_params, message = _resolve_update_method_params(
            svc,
            active_method=active_method,
            existing=current_cfg,
            method_params=method_params,
        )
        if not ok_params or normalized_params is None:
            return False, message
        method_params_by_method = _merge_method_params_by_method(
            svc,
            existing=current_cfg,
            active_method=active_method,
            active_method_params=normalized_params,
        )
        current_cfg["method"] = active_method
        current_cfg["method_params"] = normalized_params
        current_cfg["method_params_by_method"] = method_params_by_method
        params_payload = {
            "trial_slug": target,
            "name": str(current_cfg.get("name", target)).strip() or target,
            "method": active_method,
            "method_params": normalized_params,
            "method_params_by_method": {
                key: dict(value) for key, value in method_params_by_method.items()
            },
        }
        svc._append_alignment_history(
            alignment_paradigm_log_path(resolver, target),
            entry=RunLogRecord(
                step="trial_config",
                completed=False,
                params=params_payload,
                input_path=str(trial_dir),
                output_path=str(trial_dir),
                message="Trial configuration updated.",
            ).to_dict(),
            keep_top_level=False,
            trial_config=current_cfg,
        )
        return True, "Trial updated."

    paradigms = load_paradigms(config_store)
    changed = False
    for item in paradigms:
        item_slug = str(item.get("trial_slug", item.get("slug", "")))
        if item_slug != target:
            continue
        active_method = str(item.get("method", "stack_warper")).strip()
        if method is not None:
            candidate = str(method).strip()
            if candidate not in ALIGNMENT_METHODS_BY_KEY:
                return False, f"Unknown method: {candidate}"
            active_method = candidate
            item["method"] = candidate
        ok_params, normalized_params, message = _resolve_update_method_params(
            svc,
            active_method=active_method,
            existing=item,
            method_params=method_params,
        )
        if not ok_params or normalized_params is None:
            return False, message
        method_params_by_method = _merge_method_params_by_method(
            svc,
            existing=item,
            active_method=active_method,
            active_method_params=normalized_params,
        )
        item["method_params"] = normalized_params
        item["method_params_by_method"] = method_params_by_method
        changed = True
        break
    if not changed:
        return False, f"Trial not found: {target}"
    save_paradigms(config_store, paradigms)
    return True, "Trial updated."
