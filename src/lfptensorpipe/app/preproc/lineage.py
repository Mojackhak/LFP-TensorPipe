"""Accepted-generation lineage for Preprocess artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.shared.runlog_store import cache_in_run_log_read_snapshot
from lfptensorpipe.app.shared.generation_lineage import (
    accepted_result_generation_id,
    input_generation_receipts_match,
    parse_generation_lineage,
    preproc_generation_ref,
)

from .paths import read_preproc_step_routing

_PREPROC_STEPS = (
    "raw",
    "filter",
    "ecg_artifact_removal",
    "annotations",
    "finish",
)
_UPSTREAM_INVALIDATION_PREFIX = "Invalidated by upstream step re-apply:"


class PreprocInputGenerationChanged(RuntimeError):
    """Raised when a staged Preprocess result used an obsolete source."""


def _step_log_path(resolver: PathResolver, step: str) -> Path:
    return resolver.preproc_step_dir(step, create=False) / "lfptensorpipe_log.json"


def _step_raw_path(resolver: PathResolver, step: str) -> Path:
    return resolver.preproc_step_dir(step, create=False) / "raw.fif"


def _read_step_payload(resolver: PathResolver, step: str) -> dict[str, Any] | None:
    try:
        payload = read_run_log(_step_log_path(resolver, step))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _step_semantics_are_current(step: str, payload: dict[str, Any]) -> bool:
    if step == "filter":
        from .steps.filter import filter_log_has_current_bad_channel_detection_semantics

        return filter_log_has_current_bad_channel_detection_semantics(payload)
    if step == "annotations":
        from .steps.annotations import annotation_log_has_current_support_semantics

        return annotation_log_has_current_support_semantics(payload)
    return True


def preproc_step_is_skipped(resolver: PathResolver, step: str) -> bool:
    """Return the independent routing choice for one optional step."""
    return read_preproc_step_routing(resolver, step)["skipped"]


def _failed_step_is_gray(
    resolver: PathResolver,
    step: str,
    payload: dict[str, Any],
) -> bool:
    message = str(payload.get("message", ""))
    return (
        message.startswith(_UPSTREAM_INVALIDATION_PREFIX)
        and not _step_raw_path(resolver, step).exists()
    )


def _filter_preview_blocks_routing(resolver: PathResolver) -> bool:
    if preproc_step_is_skipped(resolver, "filter"):
        return False
    preview_path = (
        resolver.preproc_step_dir("filter", create=False) / "qc" / "preview_log.json"
    )
    try:
        preview_payload = read_run_log(preview_path)
    except Exception:
        return False
    if (
        not isinstance(preview_payload, dict)
        or preview_payload.get("completed") is not False
    ):
        return False
    params = preview_payload.get("params")
    return bool(
        isinstance(params, dict)
        and params.get("review_status") == "required"
        and params.get("filter_output_role") == "preview"
        and (preview_path.parent / "preview_raw.fif").exists()
        and filter_preview_lineage_is_current(resolver, preview_payload)
    )


def _prior_steps(target_step: str) -> tuple[str, ...]:
    try:
        index = _PREPROC_STEPS.index(target_step)
    except ValueError:
        return ()
    return tuple(reversed(_PREPROC_STEPS[:index]))


def _current_source_step(
    resolver: PathResolver,
    target_step: str,
    *,
    active: set[str],
) -> str | None:
    prior_steps = _prior_steps(target_step)
    if "filter" in prior_steps and _filter_preview_blocks_routing(resolver):
        return None
    for candidate in prior_steps:
        if preproc_step_is_skipped(resolver, candidate):
            continue
        payload = _read_step_payload(resolver, candidate)
        if payload is None:
            continue
        if payload.get("completed") is not True:
            if _failed_step_is_gray(resolver, candidate, payload):
                continue
            return None
        if preproc_step_lineage_is_current(resolver, candidate, _active=active):
            return candidate
        return None
    return None


@cache_in_run_log_read_snapshot(
    lambda resolver, step, *, _active=None: (
        None if _active is not None else (resolver.context, step)
    )
)
def preproc_step_lineage_is_current(
    resolver: PathResolver,
    step: str,
    *,
    _active: set[str] | None = None,
) -> bool:
    """Return whether one Preprocess artifact has current accepted lineage."""
    if step not in _PREPROC_STEPS:
        return False
    active = set(_active or ())
    if step in active:
        return False
    active.add(step)

    payload = _read_step_payload(resolver, step)
    if (
        payload is None
        or payload.get("completed") is not True
        or not _step_raw_path(resolver, step).is_file()
        or not _step_semantics_are_current(step, payload)
    ):
        return False
    lineage = parse_generation_lineage(payload)
    if lineage is None:
        return False
    if step == "raw":
        return input_generation_receipts_match(payload, expected={})

    source_step = _current_source_step(resolver, step, active=active)
    if source_step is None:
        return False
    params = payload.get("params")
    if not isinstance(params, dict) or params.get("source_step") != source_step:
        return False
    source_payload = _read_step_payload(resolver, source_step)
    expected = {
        preproc_generation_ref(source_step): accepted_result_generation_id(
            source_payload
        )
    }
    return input_generation_receipts_match(payload, expected=expected)


def capture_preproc_input_generation(
    resolver: PathResolver,
    target_step: str,
) -> tuple[str, dict[str, str | None]] | None:
    """Capture the current dynamic source and direct generation receipt."""
    if target_step not in _PREPROC_STEPS or target_step == "raw":
        return None
    source_step = _current_source_step(resolver, target_step, active={target_step})
    if source_step is None:
        return None
    source_payload = _read_step_payload(resolver, source_step)
    return (
        source_step,
        {
            preproc_generation_ref(source_step): accepted_result_generation_id(
                source_payload
            )
        },
    )


def preproc_input_generation_matches(
    resolver: PathResolver,
    target_step: str,
    *,
    source_step: str,
    input_generations: dict[str, str | None],
) -> bool:
    """Return whether a captured Preprocess input is still current."""
    current = capture_preproc_input_generation(resolver, target_step)
    return current == (source_step, input_generations)


def preproc_accepted_ancestor_payload(
    resolver: PathResolver,
    source_step: str,
    ancestor_step: str,
) -> dict[str, Any] | None:
    """Return one exact accepted ancestor payload from persisted source links."""
    if source_step not in _PREPROC_STEPS or ancestor_step not in _PREPROC_STEPS:
        return None
    if not preproc_step_lineage_is_current(resolver, source_step):
        return None
    current = source_step
    visited: set[str] = set()
    while current not in visited:
        visited.add(current)
        payload = _read_step_payload(resolver, current)
        if payload is None or payload.get("completed") is not True:
            return None
        if current == ancestor_step:
            return payload
        if current == "raw":
            return None
        params = payload.get("params")
        next_step = params.get("source_step") if isinstance(params, dict) else None
        if not isinstance(next_step, str) or next_step not in _PREPROC_STEPS:
            return None
        current = next_step
    return None


def filter_preview_lineage_is_current(
    resolver: PathResolver,
    payload: dict[str, Any] | None,
) -> bool:
    """Return whether a pending Filter Preview still matches Raw."""
    if not isinstance(payload, dict):
        return False
    if not preproc_step_lineage_is_current(resolver, "raw"):
        return False
    raw_payload = _read_step_payload(resolver, "raw")
    return input_generation_receipts_match(
        payload,
        expected={
            preproc_generation_ref("raw"): accepted_result_generation_id(raw_payload)
        },
        require_result_generation=False,
    )


__all__ = [
    "PreprocInputGenerationChanged",
    "capture_preproc_input_generation",
    "filter_preview_lineage_is_current",
    "preproc_step_is_skipped",
    "preproc_input_generation_matches",
    "preproc_accepted_ancestor_payload",
    "preproc_step_lineage_is_current",
]
