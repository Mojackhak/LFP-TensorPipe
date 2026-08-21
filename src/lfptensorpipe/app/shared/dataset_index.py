"""Dataset discovery and record-context log scanning."""

from __future__ import annotations

import os
from pathlib import Path

from .runlog_store import indicator_from_log

DEFAULT_DEMO_DATA_ROOT = Path(__file__).resolve().parents[4] / "demo"
DEMO_DATA_ROOT_KEY = "DEMO_DATA_ROOT"
DEMO_DATA_SOURCE_READONLY_KEY = "DEMO_DATA_SOURCE_READONLY"


def _read_override_value(override_file: Path | None, key: str) -> str | None:
    path = override_file
    if path is None or not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith(key):
            continue
        _, raw_value = line.split("=", maxsplit=1)
        value = raw_value.strip()
        if value:
            return value
        return None
    return None


def _read_env_value(key: str) -> str | None:
    raw_value = os.environ.get(key, "").strip()
    return raw_value or None


def _resolve_override_value(override_file: Path | None, key: str) -> str | None:
    value = _read_override_value(override_file, key)
    if value:
        return value
    if override_file is not None:
        return None
    return _read_env_value(key)


def resolve_demo_data_root(
    override_file: Path | None = None,
    fallback_root: Path = DEFAULT_DEMO_DATA_ROOT,
) -> Path:
    """Resolve demo data root from override file or fallback path."""
    value = _resolve_override_value(override_file, DEMO_DATA_ROOT_KEY)
    candidate = Path(value) if value else fallback_root
    return candidate.expanduser().resolve()


def resolve_demo_data_source_readonly(
    override_file: Path | None = None,
) -> Path | None:
    """Resolve optional read-only source demo root from override file."""
    value = _resolve_override_value(override_file, DEMO_DATA_SOURCE_READONLY_KEY)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def discover_subjects(project_root: Path) -> list[str]:
    """Discover and sort `sub-*` subject names from all configured scan roots."""
    candidates: list[str] = []
    for subject_dir in (project_root / "derivatives").glob("*/sub-*"):
        if subject_dir.is_dir():
            candidates.append(subject_dir.name)
    for subject_dir in (project_root / "sourcedata").glob("sub-*"):
        if subject_dir.is_dir():
            candidates.append(subject_dir.name)
    for subject_dir in (project_root / "rawdata").glob("sub-*"):
        if subject_dir.is_dir():
            candidates.append(subject_dir.name)

    return sorted(set(candidates))


def standard_record_scope_roots(project_root: Path, subject: str) -> dict[str, Path]:
    """Return the subject-level parent of each standard record root, by scope.

    This is the single definition of the current standard record layout. Record
    discovery and the import-occupancy check must both derive from it, otherwise
    a layout change could make a record block imports without being listed, or
    the reverse.
    """
    return {
        "derivatives": project_root / "derivatives" / "lfptensorpipe" / subject,
        "rawdata": project_root / "rawdata" / subject / "ses-postop" / "lfp",
        "sourcedata": project_root / "sourcedata" / subject / "lfp",
    }


STANDARD_RECORD_SCOPES: tuple[str, ...] = tuple(standard_record_scope_roots(Path(), ""))


def discover_records(project_root: Path, subject: str) -> list[str]:
    """Discover records from all current standard record roots."""
    candidates: list[str] = []
    for records_root in standard_record_scope_roots(project_root, subject).values():
        if not records_root.exists():
            continue
        candidates.extend(path.name for path in records_root.iterdir() if path.is_dir())
    return sorted(set(candidates))


def _aggregate_state_values(states: list[str]) -> str:
    if not states:
        return "gray"
    if any(state == "yellow" for state in states):
        return "yellow"
    if any(state == "green" for state in states):
        return "green"
    return "gray"


def _aggregate_tensor_stage_state(
    resolver, tensor_stage_log: Path, tensor_logs: list[Path]
) -> str:
    """Build Tensor stage is green once any current metric result is green.

    The aggregate stage log is status-only and cannot restore result readiness when
    no metric artifact/log generation is current.
    """
    from lfptensorpipe.app.tensor.lineage import tensor_metric_lineage_is_current

    metric_states = []
    for path in tensor_logs:
        state = indicator_from_log(path)
        if state == "green" and not tensor_metric_lineage_is_current(
            resolver,
            path.parent.name,
        ):
            state = "yellow"
        metric_states.append(state)
    if any(state == "green" for state in metric_states):
        return "green"
    if any(state == "yellow" for state in metric_states):
        return "yellow"
    stage_state = indicator_from_log(tensor_stage_log)
    if stage_state in {"green", "yellow"}:
        return "yellow"
    return "gray"


def scan_stage_states(project_root: Path, subject: str, record: str) -> dict[str, str]:
    """Scan record-scoped logs and derive stage indicator states."""
    from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
    from lfptensorpipe.app.alignment.indicator import alignment_trial_stage_state
    from lfptensorpipe.app.features.indicator import extract_features_indicator_state
    from lfptensorpipe.app.preproc.indicator import preproc_step_indicator_state

    base = project_root / "derivatives" / "lfptensorpipe" / subject / record

    tensor_stage_log = base / "tensor" / "lfptensorpipe_log.json"
    tensor_logs = list((base / "tensor").glob("*/lfptensorpipe_log.json"))
    alignment_logs = list((base / "alignment").glob("*/lfptensorpipe_log.json"))
    features_logs = list((base / "features").glob("*/lfptensorpipe_log.json"))
    resolver = PathResolver(
        RecordContext(
            project_root=project_root,
            subject=subject,
            record=record,
        )
    )
    tensor_state = _aggregate_tensor_stage_state(
        resolver,
        tensor_stage_log,
        tensor_logs,
    )
    preproc_state = preproc_step_indicator_state(resolver, "finish")
    alignment_state = _aggregate_state_values(
        [
            alignment_trial_stage_state(resolver, paradigm_slug=path.parent.name)
            for path in alignment_logs
        ]
    )
    features_state = _aggregate_state_values(
        [
            extract_features_indicator_state(
                resolver,
                trial_slug=path.parent.name,
            )
            for path in features_logs
        ]
    )

    return {
        "preproc": preproc_state,
        "tensor": tensor_state,
        "alignment": alignment_state,
        "features": features_state,
    }
