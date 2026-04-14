"""Path and config helpers for preprocess stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext


def rawdata_input_fif_path(context: RecordContext) -> Path:
    """Return canonical raw FIF input path under `rawdata`."""
    return (
        context.project_root
        / "rawdata"
        / context.subject
        / "ses-postop"
        / "lfp"
        / context.record
        / "raw"
        / "raw.fif"
    )


def preproc_step_raw_path(resolver: PathResolver, step: str) -> Path:
    """Return `{step}/raw.fif` path inside preproc root."""
    return resolver.preproc_step_dir(step, create=True) / "raw.fif"


def preproc_step_log_path(resolver: PathResolver, step: str) -> Path:
    """Return `{step}/lfptensorpipe_log.json` path inside preproc root."""
    return resolver.preproc_step_dir(step, create=True) / "lfptensorpipe_log.json"


def preproc_step_config_path(resolver: PathResolver, step: str) -> Path:
    """Return `{step}/config.yml` path inside preproc root."""
    return resolver.preproc_step_dir(step, create=True) / "config.yml"


def write_preproc_step_config(
    *,
    resolver: PathResolver,
    step: str,
    config: dict[str, Any],
) -> Path:
    """Persist one preprocess step config YAML."""
    path = preproc_step_config_path(resolver, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
    return path
