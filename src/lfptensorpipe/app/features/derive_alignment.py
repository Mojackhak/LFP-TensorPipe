"""Accepted Alignment helpers for feature derivation."""

from __future__ import annotations

from lfptensorpipe.app.alignment.generation import (
    alignment_stage_lineage_is_current,
    latest_alignment_step_entry,
)
from lfptensorpipe.app.path_resolver import PathResolver


def _extract_alignment_method_from_log(
    resolver: PathResolver,
    *,
    trial_slug: str,
) -> str:
    """Return the accepted Run method only when its Finish remains current."""
    slug = str(trial_slug).strip()
    if not slug or not alignment_stage_lineage_is_current(
        resolver,
        trial_slug=slug,
        stage="finish",
    ):
        return ""
    accepted_run = latest_alignment_step_entry(
        resolver,
        trial_slug=slug,
        step="run_align_epochs",
    )
    if accepted_run is None:
        return ""
    params = accepted_run[1].get("params")
    return str(params.get("method", "")).strip() if isinstance(params, dict) else ""
