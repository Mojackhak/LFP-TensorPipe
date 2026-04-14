"""Dataset discovery and record-context log scanning."""

from __future__ import annotations

import os
from pathlib import Path

from lfptensorpipe.app.runlog_store import indicator_from_log

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


def discover_records(project_root: Path, subject: str) -> list[str]:
    """Discover records from derivatives root only."""
    records_root = project_root / "derivatives" / "lfptensorpipe" / subject
    if not records_root.exists():
        return []
    candidates = [path.name for path in records_root.iterdir() if path.is_dir()]
    return sorted(set(candidates))


def _aggregate_states(log_paths: list[Path]) -> str:
    if not log_paths:
        return "gray"
    states = [indicator_from_log(path) for path in log_paths]
    if any(state == "yellow" for state in states):
        return "yellow"
    if any(state == "green" for state in states):
        return "green"
    return "gray"


def _aggregate_tensor_stage_state(tensor_stage_log: Path, tensor_logs: list[Path]) -> str:
    """Build Tensor stage is green once any metric log is green.

    This avoids stale yellow stage logs blocking downstream pages when at least one
    metric has a valid tensor artifact/log result.
    """
    metric_states = [indicator_from_log(path) for path in tensor_logs]
    if any(state == "green" for state in metric_states):
        return "green"
    if any(state == "yellow" for state in metric_states):
        return "yellow"
    stage_state = indicator_from_log(tensor_stage_log)
    if stage_state in {"green", "yellow"}:
        return stage_state
    return "gray"


def scan_stage_states(project_root: Path, subject: str, record: str) -> dict[str, str]:
    """Scan record-scoped logs and derive stage indicator states."""
    base = project_root / "derivatives" / "lfptensorpipe" / subject / record

    preproc_log = base / "preproc" / "finish" / "lfptensorpipe_log.json"
    tensor_stage_log = base / "tensor" / "lfptensorpipe_log.json"
    tensor_logs = list((base / "tensor").glob("*/lfptensorpipe_log.json"))
    alignment_logs = list((base / "alignment").glob("*/lfptensorpipe_log.json"))
    features_logs = list((base / "features").glob("*/lfptensorpipe_log.json"))
    tensor_state = _aggregate_tensor_stage_state(tensor_stage_log, tensor_logs)

    return {
        "preproc": indicator_from_log(preproc_log),
        "tensor": tensor_state,
        "alignment": _aggregate_states(alignment_logs),
        "features": _aggregate_states(features_logs),
    }
