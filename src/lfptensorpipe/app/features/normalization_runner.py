"""Normalization runner placeholder for Extract-Features stage."""

from __future__ import annotations

from lfptensorpipe.app.path_resolver import RecordContext


def run_normalization(
    context: RecordContext,
    *,
    paradigm_slug: str,
    baseline_phase: str,
    mode: str,
) -> tuple[bool, str]:
    """Deprecated in new workflow: normalization is handled in plot advance only."""
    _ = (context, paradigm_slug, baseline_phase, mode)
    return False, "Run Normalization is disabled. Use Plot Advance parameters instead."
