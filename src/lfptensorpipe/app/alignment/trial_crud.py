"""Alignment trial CRUD helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import RunLogRecord, read_ui_state, write_ui_state
from lfptensorpipe.app.shared.runlog_store import _write_json_payload

from .paths import alignment_paradigm_dir, alignment_paradigm_log_path
from .validation import ALIGNMENT_METHODS_BY_KEY

_TRIAL_DELETE_MANIFEST_FILENAME = "trial_delete_manifest.json"


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
    _, kept, changed = _prepare_legacy_trial_update(
        svc,
        config_store=config_store,
        slug=slug,
    )
    if not changed:
        return False
    svc.save_alignment_paradigms(config_store, kept)
    return True


def _prepare_legacy_trial_update(
    svc: Any,
    *,
    config_store: AppConfigStore,
    slug: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    paradigms = svc.load_alignment_paradigms(config_store)
    kept = [
        item
        for item in paradigms
        if str(item.get("trial_slug", item.get("slug", ""))) != slug
    ]
    return paradigms, kept, len(kept) != len(paradigms)


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


def _restore_staged_trial_artifacts(
    *,
    staged_paths: list[tuple[Path, Path]],
) -> list[str]:
    errors: list[str] = []
    for public_path, staged_path in reversed(staged_paths):
        if not staged_path.exists():
            continue
        try:
            if public_path.exists():
                raise FileExistsError(f"Restore target already exists: {public_path}")
            staged_path.rename(public_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{staged_path} -> {public_path}: {exc}")
    return errors


def _trial_delete_manifest_path(quarantine_root: Path) -> Path:
    return quarantine_root / _TRIAL_DELETE_MANIFEST_FILENAME


def _stage_trial_artifacts(
    *,
    resolver: PathResolver,
    slug: str,
    manifest: dict[str, Any],
) -> tuple[list[tuple[Path, Path]], Path | None]:
    indexed_public_paths = [
        (index, path)
        for index, path in enumerate(_trial_artifact_dirs(resolver=resolver, slug=slug))
        if path.exists()
    ]
    if not indexed_public_paths:
        return [], None

    quarantine_root = resolver.lfp_root / f".trial-delete-{slug}-{uuid4().hex}"
    quarantine_root.mkdir(parents=False, exist_ok=False)
    staged_paths: list[tuple[Path, Path]] = []
    try:
        manifest["artifact_indices"] = [
            index for index, _public_path in indexed_public_paths
        ]
        _write_json_payload(_trial_delete_manifest_path(quarantine_root), manifest)
        for index, public_path in indexed_public_paths:
            staged_path = quarantine_root / f"{index:02d}-{public_path.name}"
            public_path.rename(staged_path)
            staged_paths.append((public_path, staged_path))
    except Exception as exc:
        restore_errors = _restore_staged_trial_artifacts(staged_paths=staged_paths)
        if not restore_errors:
            try:
                shutil.rmtree(quarantine_root)
            except OSError as cleanup_exc:
                raise RuntimeError(
                    "Failed to stage trial artifacts; artifacts were restored but "
                    f"quarantine cleanup failed at {quarantine_root}: {exc}; "
                    f"cleanup: {cleanup_exc}"
                ) from exc
            raise RuntimeError(f"Failed to stage trial artifacts: {exc}") from exc
        raise RuntimeError(
            "Failed to stage trial artifacts and rollback was incomplete; "
            f"recovery retained at {quarantine_root}: {exc}; "
            f"{'; '.join(restore_errors)}"
        ) from exc
    return staged_paths, quarantine_root


def _load_trial_delete_manifest(
    *,
    resolver: PathResolver,
    quarantine_root: Path,
) -> dict[str, Any] | None:
    manifest_path = _trial_delete_manifest_path(quarantine_root)
    payload = read_ui_state(manifest_path)
    if payload is None:
        return None
    if payload.get("operation") != "delete_alignment_trial":
        raise ValueError(f"Invalid trial-delete operation in {manifest_path}")

    phase = payload.get("phase")
    if phase not in {"prepared", "committed"}:
        raise ValueError(f"Invalid trial-delete phase in {manifest_path}")

    slug = payload.get("slug")
    if not isinstance(slug, str) or _svc()._normalize_slug(slug) != slug:
        raise ValueError(f"Invalid trial-delete slug in {manifest_path}")
    if not quarantine_root.name.startswith(f".trial-delete-{slug}-"):
        raise ValueError(f"Trial-delete slug does not match {quarantine_root}")

    artifact_indices = payload.get("artifact_indices")
    if not isinstance(artifact_indices, list):
        raise ValueError(f"Invalid trial-delete artifact indices in {manifest_path}")
    artifact_count = len(_trial_artifact_dirs(resolver=resolver, slug=slug))
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= artifact_count
        for index in artifact_indices
    ) or len(set(artifact_indices)) != len(artifact_indices):
        raise ValueError(f"Invalid trial-delete artifact indices in {manifest_path}")

    legacy_changed = payload.get("legacy_changed")
    ui_changed = payload.get("ui_changed")
    if not isinstance(legacy_changed, bool) or not isinstance(ui_changed, bool):
        raise ValueError(f"Invalid trial-delete metadata flags in {manifest_path}")
    if legacy_changed and not isinstance(payload.get("original_legacy"), list):
        raise ValueError(f"Invalid original legacy state in {manifest_path}")
    if ui_changed and not isinstance(payload.get("original_ui"), dict):
        raise ValueError(f"Invalid original UI state in {manifest_path}")
    return payload


def _restore_manifest_artifacts(
    *,
    resolver: PathResolver,
    quarantine_root: Path,
    slug: str,
    artifact_indices: list[int],
) -> list[str]:
    public_paths = _trial_artifact_dirs(resolver=resolver, slug=slug)
    errors: list[str] = []
    for index in reversed(artifact_indices):
        public_path = public_paths[index]
        staged_path = quarantine_root / f"{index:02d}-{public_path.name}"
        if staged_path.exists():
            try:
                if public_path.exists():
                    raise FileExistsError(
                        f"Restore target already exists: {public_path}"
                    )
                staged_path.rename(public_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{staged_path} -> {public_path}: {exc}")
        elif not public_path.exists():
            errors.append(f"Missing both staged and public artifact: {public_path}")
    return errors


def recover_trial_delete_transactions(
    config_store: AppConfigStore,
    *,
    context: RecordContext,
) -> list[str]:
    """Recover one record's interrupted deletions and return pending warnings."""
    resolver = PathResolver(context)
    recovery_warnings: list[str] = []
    if not resolver.lfp_root.exists():
        return recovery_warnings

    for quarantine_root in sorted(resolver.lfp_root.glob(".trial-delete-*")):
        if not quarantine_root.is_dir():
            continue
        try:
            manifest = _load_trial_delete_manifest(
                resolver=resolver,
                quarantine_root=quarantine_root,
            )
            if manifest is None:
                if any(quarantine_root.iterdir()):
                    raise RuntimeError("missing trial-delete manifest")
                quarantine_root.rmdir()
                continue
            if manifest["phase"] == "committed":
                shutil.rmtree(quarantine_root)
                continue

            svc = _svc()
            if manifest["legacy_changed"]:
                svc.save_alignment_paradigms(
                    config_store,
                    manifest["original_legacy"],
                )
            if manifest["ui_changed"]:
                write_ui_state(
                    resolver.record_ui_state_path(create=True),
                    manifest["original_ui"],
                )
            restore_errors = _restore_manifest_artifacts(
                resolver=resolver,
                quarantine_root=quarantine_root,
                slug=manifest["slug"],
                artifact_indices=manifest["artifact_indices"],
            )
            if restore_errors:
                raise RuntimeError("; ".join(restore_errors))
            shutil.rmtree(quarantine_root)
        except Exception as exc:
            recovery_warnings.append(
                f"Trial deletion recovery pending at {quarantine_root}: {exc}"
            )
    return recovery_warnings


def _prepare_deleted_trial_ui_state(
    *,
    context: RecordContext,
    slug: str,
) -> tuple[Path, dict[str, Any] | None, dict[str, Any] | None, bool]:
    resolver = PathResolver(context)
    path = resolver.record_ui_state_path(create=False)
    if not path.is_file():
        return path, None, None, False
    original_payload = read_ui_state(path)
    if not isinstance(original_payload, dict):
        return path, None, None, False
    payload = deepcopy(original_payload)

    changed = False
    alignment_node = payload.get("alignment")
    if isinstance(alignment_node, dict):
        if str(alignment_node.get("trial_slug", "")).strip() == slug:
            alignment_node["trial_slug"] = None
            alignment_node["method"] = None
            alignment_node["method_params"] = {}
            alignment_node["method_params_by_method"] = {}
            alignment_node["epoch_metric"] = None
            alignment_node["epoch_channel"] = None
            alignment_node["picked_epoch_indices"] = []
            changed = True

    features_node = payload.get("features")
    if isinstance(features_node, dict):
        trial_params_by_slug = features_node.get("trial_params_by_slug")
        if isinstance(trial_params_by_slug, dict) and slug in trial_params_by_slug:
            trial_params_by_slug.pop(slug)
            changed = True

    return path, original_payload, payload, changed


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

    recovery_warnings: list[str] = []
    paradigms = svc.load_alignment_paradigms(
        config_store,
        context=context,
        recovery_warnings=recovery_warnings,
    )
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
            key: dict(value) for key, value in entry["method_params_by_method"].items()
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
    message = f"Trial created: {slug}"
    if recovery_warnings:
        message += f" | {' | '.join(recovery_warnings)}"
    return True, message, entry


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
    try:
        (
            ui_path,
            original_ui,
            updated_ui,
            removed_ui_state,
        ) = _prepare_deleted_trial_ui_state(context=context, slug=target)
        original_legacy, updated_legacy, removed_legacy = _prepare_legacy_trial_update(
            svc,
            config_store=config_store,
            slug=target,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to prepare trial deletion: {exc}"

    manifest: dict[str, Any] = {
        "operation": "delete_alignment_trial",
        "phase": "prepared",
        "slug": target,
        "legacy_changed": removed_legacy,
        "original_legacy": original_legacy,
        "ui_changed": removed_ui_state,
        "original_ui": original_ui,
    }
    try:
        staged_paths, quarantine_root = _stage_trial_artifacts(
            resolver=resolver,
            slug=target,
            manifest=manifest,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to delete trial: {exc}"

    legacy_committed = False
    ui_committed = False
    try:
        if removed_legacy:
            svc.save_alignment_paradigms(config_store, updated_legacy)
            legacy_committed = True
        if removed_ui_state and updated_ui is not None:
            write_ui_state(ui_path, updated_ui)
            ui_committed = True
        if quarantine_root is not None:
            manifest["phase"] = "committed"
            _write_json_payload(
                _trial_delete_manifest_path(quarantine_root),
                manifest,
            )
    except Exception as exc:  # noqa: BLE001
        rollback_errors: list[str] = []
        if ui_committed and original_ui is not None:
            try:
                write_ui_state(ui_path, original_ui)
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_errors.append(f"UI state: {rollback_exc}")
        if legacy_committed:
            try:
                svc.save_alignment_paradigms(config_store, original_legacy)
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_errors.append(f"legacy config: {rollback_exc}")
        rollback_errors.extend(
            _restore_staged_trial_artifacts(staged_paths=staged_paths)
        )
        if not any(staged_path.exists() for _, staged_path in staged_paths):
            try:
                if quarantine_root is not None:
                    shutil.rmtree(quarantine_root)
            except OSError as rollback_exc:
                rollback_errors.append(f"quarantine cleanup: {rollback_exc}")
        if rollback_errors:
            recovery_path = (
                f"; recovery retained at {quarantine_root}"
                if quarantine_root is not None and quarantine_root.exists()
                else ""
            )
            return (
                False,
                "Failed to commit trial deletion and rollback was incomplete"
                f"{recovery_path}: {exc}; {'; '.join(rollback_errors)}",
            )
        return False, f"Failed to commit trial deletion; changes rolled back: {exc}"

    removed_artifacts = bool(staged_paths)
    if not removed_artifacts and not removed_legacy and not removed_ui_state:
        return False, f"Trial not found: {target}"

    if quarantine_root is not None:
        try:
            shutil.rmtree(quarantine_root)
        except Exception as exc:  # noqa: BLE001
            return (
                True,
                f"Trial deleted: {target}; staged cleanup pending at "
                f"{quarantine_root}: {exc}",
            )
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
