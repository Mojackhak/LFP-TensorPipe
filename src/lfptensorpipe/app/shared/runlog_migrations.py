"""Versioned run-log schema registry and deterministic upgrade helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

RUNLOG_SCHEMA_NAME = "lfptensorpipe.runlog"
RUNLOG_SCHEMA_KEY = "log_schema"
RUNLOG_VERSION_KEY = "log_version"
RUNLOG_MIGRATION_META_KEY = "migration_meta"
RUNLOG_UPGRADES_KEY = "applied_upgrades"
RUNLOG_ALL_STEPS = "*"
RUNLOG_BASE_VERSION = 1

RunLogMigrationFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class RunLogMigrationSpec:
    """One deterministic `vN -> vN+1` run-log migration."""

    step: str
    from_version: int
    to_version: int
    migrate: RunLogMigrationFn


_CURRENT_RUN_LOG_VERSIONS: dict[str, int] = {RUNLOG_ALL_STEPS: RUNLOG_BASE_VERSION}
_RUN_LOG_MIGRATIONS: dict[tuple[str, int], RunLogMigrationSpec] = {}


def _normalize_step_key(step: str | None) -> str:
    if step is None:
        return RUNLOG_ALL_STEPS
    text = str(step).strip()
    return text or RUNLOG_ALL_STEPS


def register_run_log_version(step: str | None, version: int) -> None:
    """Register the latest supported schema version for one step."""
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Run-log version must be an integer >= 1.")
    _CURRENT_RUN_LOG_VERSIONS[_normalize_step_key(step)] = version


def current_run_log_version(step: str | None) -> int:
    """Return the latest supported schema version for one step."""
    step_key = _normalize_step_key(step)
    if step_key in _CURRENT_RUN_LOG_VERSIONS:
        return _CURRENT_RUN_LOG_VERSIONS[step_key]
    return _CURRENT_RUN_LOG_VERSIONS[RUNLOG_ALL_STEPS]


def register_run_log_migration(
    step: str | None,
    from_version: int,
    to_version: int,
    migrate: RunLogMigrationFn,
) -> None:
    """Register one deterministic migration for a step or all steps."""
    if (
        isinstance(from_version, bool)
        or not isinstance(from_version, int)
        or from_version < 1
    ):
        raise ValueError("from_version must be an integer >= 1.")
    if (
        isinstance(to_version, bool)
        or not isinstance(to_version, int)
        or to_version <= from_version
    ):
        raise ValueError("to_version must be greater than from_version.")
    if not callable(migrate):
        raise ValueError("migrate must be callable.")

    step_key = _normalize_step_key(step)
    key = (step_key, from_version)
    if key in _RUN_LOG_MIGRATIONS:
        raise ValueError(
            f"Run-log migration already registered for step={step_key!r} "
            f"from_version={from_version}."
        )
    _RUN_LOG_MIGRATIONS[key] = RunLogMigrationSpec(
        step=step_key,
        from_version=from_version,
        to_version=to_version,
        migrate=migrate,
    )
    register_run_log_version(
        step_key, max(current_run_log_version(step_key), to_version)
    )


def infer_run_log_version(payload: dict[str, Any]) -> int:
    """Infer schema version from a payload written by current/future code."""
    raw_schema = payload.get(RUNLOG_SCHEMA_KEY)
    raw_version = payload.get(RUNLOG_VERSION_KEY)

    if raw_schema is None and raw_version is None:
        return RUNLOG_BASE_VERSION
    if raw_schema != RUNLOG_SCHEMA_NAME:
        raise ValueError(
            f"Unsupported run-log schema {raw_schema!r}; expected {RUNLOG_SCHEMA_NAME!r}."
        )
    if (
        isinstance(raw_version, bool)
        or not isinstance(raw_version, int)
        or raw_version < 1
    ):
        raise ValueError("Run-log version must be an integer >= 1.")
    return raw_version


def stamp_run_log_metadata(
    payload: dict[str, Any], *, step: str | None = None
) -> dict[str, Any]:
    """Attach the latest supported schema metadata for a payload's step."""
    out = deepcopy(payload)
    target_step = step if step is not None else str(out.get("step", "")).strip()
    out[RUNLOG_SCHEMA_KEY] = RUNLOG_SCHEMA_NAME
    out[RUNLOG_VERSION_KEY] = current_run_log_version(target_step)
    return out


def _resolve_run_log_migration(
    step: str,
    version: int,
) -> RunLogMigrationSpec | None:
    exact = _RUN_LOG_MIGRATIONS.get((step, version))
    if exact is not None:
        return exact
    return _RUN_LOG_MIGRATIONS.get((RUNLOG_ALL_STEPS, version))


def _append_migration_meta(
    payload: dict[str, Any],
    upgrades: list[tuple[int, int]],
) -> dict[str, Any]:
    out = deepcopy(payload)
    existing = out.get(RUNLOG_MIGRATION_META_KEY)
    meta = deepcopy(existing) if isinstance(existing, dict) else {}
    raw_applied = meta.get(RUNLOG_UPGRADES_KEY)
    applied = deepcopy(raw_applied) if isinstance(raw_applied, list) else []
    for from_version, to_version in upgrades:
        applied.append(
            {
                "from_version": from_version,
                "to_version": to_version,
            }
        )
    meta[RUNLOG_UPGRADES_KEY] = applied
    out[RUNLOG_MIGRATION_META_KEY] = meta
    return out


def upgrade_run_log_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Upgrade one run-log payload to the latest supported step schema."""
    if not isinstance(payload, dict):
        raise TypeError("Run-log payload must be a dict.")

    source_version = infer_run_log_version(payload)
    step = str(payload.get("step", "")).strip()
    target_version = current_run_log_version(step)
    if source_version > target_version:
        raise ValueError(
            f"Run log for step {step!r} has unsupported newer version "
            f"{source_version}; current supported version is {target_version}."
        )

    upgraded = deepcopy(payload)
    changed = False
    if upgraded.get(RUNLOG_SCHEMA_KEY) != RUNLOG_SCHEMA_NAME:
        upgraded[RUNLOG_SCHEMA_KEY] = RUNLOG_SCHEMA_NAME
        changed = True
    if upgraded.get(RUNLOG_VERSION_KEY) != source_version:
        upgraded[RUNLOG_VERSION_KEY] = source_version
        changed = True

    upgrades: list[tuple[int, int]] = []
    current_version = source_version
    while current_version < target_version:
        spec = _resolve_run_log_migration(step, current_version)
        if spec is None:
            raise ValueError(
                f"No run-log migration registered for step {step!r} "
                f"from version {current_version} to {target_version}."
            )
        migrated = spec.migrate(deepcopy(upgraded))
        if not isinstance(migrated, dict):
            raise ValueError("Run-log migration must return a dict payload.")
        upgraded = migrated
        current_version = spec.to_version
        upgrades.append((spec.from_version, spec.to_version))
        changed = True

    upgraded[RUNLOG_SCHEMA_KEY] = RUNLOG_SCHEMA_NAME
    upgraded[RUNLOG_VERSION_KEY] = current_version
    if upgrades:
        upgraded = _append_migration_meta(upgraded, upgrades)
    return upgraded, changed


__all__ = [
    "RUNLOG_ALL_STEPS",
    "RUNLOG_BASE_VERSION",
    "RUNLOG_MIGRATION_META_KEY",
    "RUNLOG_SCHEMA_KEY",
    "RUNLOG_SCHEMA_NAME",
    "RUNLOG_UPGRADES_KEY",
    "RUNLOG_VERSION_KEY",
    "RunLogMigrationFn",
    "RunLogMigrationSpec",
    "current_run_log_version",
    "infer_run_log_version",
    "register_run_log_migration",
    "register_run_log_version",
    "stamp_run_log_metadata",
    "upgrade_run_log_payload",
]
