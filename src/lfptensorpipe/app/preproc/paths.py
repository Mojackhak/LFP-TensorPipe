"""Path and config helpers for preprocess stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet


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


def preproc_filter_preview_raw_path(resolver: PathResolver) -> Path:
    """Return the private detection-preview FIF path for Filter review."""
    return resolver.preproc_step_dir("filter") / "qc" / "preview_raw.fif"


def preproc_filter_preview_config_path(resolver: PathResolver) -> Path:
    """Return the private detection-preview config path."""
    return resolver.preproc_step_dir("filter") / "qc" / "preview_config.yml"


def preproc_filter_preview_log_path(resolver: PathResolver) -> Path:
    """Return the private detection-preview run-log path."""
    return resolver.preproc_step_dir("filter") / "qc" / "preview_log.json"


def preproc_step_log_path(resolver: PathResolver, step: str) -> Path:
    """Return `{step}/lfptensorpipe_log.json` path inside preproc root."""
    return resolver.preproc_step_dir(step, create=True) / "lfptensorpipe_log.json"


def preproc_step_config_path(resolver: PathResolver, step: str) -> Path:
    """Return `{step}/config.yml` path inside preproc root."""
    return resolver.preproc_step_dir(step, create=True) / "config.yml"


def preproc_step_routing_path(resolver: PathResolver, step: str) -> Path:
    """Return the independent routing-state path for one optional step."""
    return resolver.preproc_step_dir(step, create=False) / "routing.yml"


def read_preproc_step_routing(
    resolver: PathResolver,
    step: str,
) -> dict[str, bool]:
    """Read and validate normalized routing state at its file boundary."""
    path = preproc_step_routing_path(resolver, step)
    if not path.exists():
        return {"skipped": False}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("skipped"), bool):
        raise ValueError(f"Invalid preprocess routing state: {path}")
    return {"skipped": payload["skipped"]}


def write_preproc_step_routing(
    *,
    resolver: PathResolver,
    step: str,
    skipped: bool,
) -> Path:
    """Atomically persist normalized routing state for one optional step."""
    output_path = preproc_step_routing_path(resolver, step)
    with AtomicOutputSet(
        [output_path],
        cleanup_stale_residues=True,
    ) as output_set:
        staged_path = output_set.staged_path(output_path)
        staged_path.write_text(
            yaml.safe_dump(
                {"skipped": skipped},
                sort_keys=False,
                allow_unicode=False,
            ),
            encoding="utf-8",
        )
        output_set.commit()
    return output_path


def write_preproc_step_config(
    *,
    resolver: PathResolver,
    step: str,
    config: dict[str, Any],
    path: Path | None = None,
) -> Path:
    """Persist one preprocess step config YAML."""
    output_path = path or preproc_step_config_path(resolver, step)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)
    return output_path
