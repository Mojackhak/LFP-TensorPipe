"""Shared atomic promotion for fixed multi-file output sets."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shutil
import stat
from types import TracebackType
from uuid import uuid4

OUTPUT_TRANSACTION_RUN_ID_ENV = "LFPTENSORPIPE_TENSOR_RUN_ID"

OutputWriter = Callable[[Path], None]
ReplaceFn = Callable[[Path, Path], Path]
PrecommitCheck = Callable[[], None]

logger = logging.getLogger(__name__)

_TRANSACTION_TOKEN_PATTERN = r"[0-9a-f]{32}"


@dataclass(frozen=True)
class _PreparedOutput:
    target: Path
    temporary: Path
    backup: Path
    had_original: bool
    target_mode: int | None
    staging_directory: Path | None = None


@dataclass(frozen=True)
class _StaleResidueRule:
    pattern: re.Pattern[str]
    expected_directory: bool
    target: Path


class AtomicOutputSet:
    """Stage and collectively promote one fixed set of public output files."""

    def __init__(
        self,
        targets: Iterable[Path],
        *,
        replace_fn: ReplaceFn | None = None,
        cleanup_stale_residues: bool = False,
    ) -> None:
        token = uuid4().hex
        self._token = token
        prepared: list[_PreparedOutput] = []
        seen: set[Path] = set()
        for raw_target in targets:
            target = Path(raw_target).expanduser().resolve()
            if target in seen:
                raise ValueError(f"Duplicate atomic output target: {target}")
            seen.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.name.endswith((".fif", ".fif.gz")):
                staging_directory = target.parent / f".{target.name}.tmp-{token}"
                temporary = staging_directory / target.name
            else:
                staging_directory = None
                temporary = target.parent / (
                    f".{target.name}.tmp-{token}{target.suffix}"
                )
            backup = target.parent / f".{target.name}.bak-{token}"
            had_original = target.exists()
            prepared.append(
                _PreparedOutput(
                    target=target,
                    temporary=temporary,
                    backup=backup,
                    had_original=had_original,
                    target_mode=(
                        stat.S_IMODE(target.stat().st_mode) if had_original else None
                    ),
                    staging_directory=staging_directory,
                )
            )

        self._prepared = tuple(prepared)
        self._declared_targets = tuple(item.target for item in prepared)
        self._by_target = {item.target: item for item in self._prepared}
        self._replace = replace_fn or (lambda source, target: source.replace(target))
        self._cleanup_stale_residues_enabled = cleanup_stale_residues
        self._manifest_path: Path | None = None
        self._manifest_payload: dict[str, object] | None = None
        self._obsolete_fif_splits: tuple[Path, ...] = ()
        self._entered = False
        self._committed = False
        self._recovery_retained = False

    def __enter__(self) -> AtomicOutputSet:
        if self._entered:
            raise RuntimeError("Atomic output set has already been entered.")
        self._entered = True
        try:
            for directory in self._staging_directories():
                directory.mkdir(parents=False, exist_ok=False)
            self._prepare_manifest()
        except Exception:
            self._cleanup_staging_directories()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        _ = exc_type, exc, traceback
        if not self._committed and not self._recovery_retained:
            self._cleanup_rejected_candidate()
        return False

    def staged_path(self, target: Path) -> Path:
        """Return the private staged path assigned to one declared target."""
        if not self._entered:
            raise RuntimeError("Enter the atomic output set before requesting paths.")
        resolved = Path(target).expanduser().resolve()
        try:
            return self._by_target[resolved].temporary
        except KeyError as exc:
            raise KeyError(f"Undeclared atomic output target: {resolved}") from exc

    def commit(self) -> None:
        """Promote every staged output or restore the complete prior set."""
        if not self._entered:
            raise RuntimeError("Enter the atomic output set before commit.")
        if self._committed:
            raise RuntimeError("Atomic output set has already been committed.")
        if not self._prepared:
            self._committed = True
            return

        self._expand_fif_output_sets()
        missing = [
            item.temporary for item in self._prepared if not item.temporary.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing staged atomic output(s): "
                + ", ".join(str(path) for path in missing)
            )
        for item in self._prepared:
            if item.target_mode is not None:
                item.temporary.chmod(item.target_mode)

        committed: list[Path] = []
        promotion_complete = False
        try:
            for item in self._prepared:
                if item.had_original:
                    self._replace(item.target, item.backup)
            for item in self._prepared:
                self._replace(item.temporary, item.target)
                committed.append(item.target)
            self._mark_manifest_committed()
            promotion_complete = True
        except Exception:
            if promotion_complete:
                raise
            rollback_complete = False
            try:
                for target in committed:
                    target.unlink(missing_ok=True)
                for item in self._prepared:
                    if item.backup.exists():
                        self._replace(item.backup, item.target)
                rollback_complete = True
            except Exception:
                self._recovery_retained = True
                raise
            finally:
                if rollback_complete:
                    self._cleanup_rejected_candidate()
            raise

        self._committed = True
        self._cleanup_accepted_candidate()
        self._cleanup_obsolete_fif_splits()
        if self._cleanup_stale_residues_enabled:
            self._cleanup_stale_residues()

    def _staging_directories(self) -> tuple[Path, ...]:
        return tuple(
            dict.fromkeys(
                item.staging_directory
                for item in self._prepared
                if item.staging_directory is not None
            )
        )

    def _expand_fif_output_sets(self) -> None:
        expanded = list(self._prepared)
        known_targets = {item.target for item in expanded}
        obsolete_candidates: set[Path] = set()
        for item in tuple(self._prepared):
            directory = item.staging_directory
            if directory is None:
                continue
            if not item.temporary.is_file():
                raise FileNotFoundError(f"Missing staged FIF output: {item.temporary}")
            for temporary in sorted(directory.iterdir()):
                if temporary == item.temporary or temporary.name.startswith("._"):
                    continue
                if not temporary.is_file() or not temporary.name.endswith(
                    (".fif", ".fif.gz")
                ):
                    raise ValueError(
                        f"Unexpected file in staged FIF output set: {temporary}"
                    )
                target = item.target.parent / temporary.name
                if target in known_targets:
                    raise ValueError(f"Duplicate atomic output target: {target}")
                known_targets.add(target)
                had_original = target.exists()
                expanded.append(
                    _PreparedOutput(
                        target=target,
                        temporary=temporary,
                        backup=target.parent / f".{target.name}.bak-{self._token}",
                        had_original=had_original,
                        target_mode=(
                            stat.S_IMODE(target.stat().st_mode)
                            if had_original
                            else None
                        ),
                        staging_directory=directory,
                    )
                )
            obsolete_candidates.update(self._existing_fif_split_siblings(item.target))
        self._prepared = tuple(expanded)
        self._by_target = {item.target: item for item in self._prepared}
        self._obsolete_fif_splits = tuple(
            sorted(obsolete_candidates.difference(known_targets))
        )
        self._refresh_prepared_manifest()

    @staticmethod
    def _existing_fif_split_siblings(target: Path) -> set[Path]:
        if target.name.endswith(".fif.gz"):
            stem = target.name[: -len(".fif.gz")]
            suffix = ".fif.gz"
        else:
            stem = target.name[: -len(".fif")]
            suffix = ".fif"
        pattern = re.compile(rf"{re.escape(stem)}-[0-9]+{re.escape(suffix)}")
        return {
            candidate
            for candidate in target.parent.iterdir()
            if candidate.is_file() and pattern.fullmatch(candidate.name)
        }

    def _cleanup_obsolete_fif_splits(self) -> None:
        for path in self._obsolete_fif_splits:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    def _cleanup_stale_residues(self) -> None:
        rules_by_parent: dict[Path, list[_StaleResidueRule]] = {}
        for target in self._declared_targets:
            rules_by_parent.setdefault(target.parent, []).extend(
                self._stale_residue_rules(target)
            )

        for parent, rules in rules_by_parent.items():
            try:
                candidates = tuple(parent.iterdir())
            except OSError as exc:
                logger.warning(
                    "Atomic output stale-residue scan failed for targets %s "
                    "in %s: %s",
                    sorted({str(rule.target) for rule in rules}),
                    parent,
                    exc,
                )
                continue
            for candidate in candidates:
                rule = next(
                    (item for item in rules if item.pattern.fullmatch(candidate.name)),
                    None,
                )
                if rule is None:
                    continue
                self._remove_stale_residue(candidate, rule)

    @staticmethod
    def _stale_residue_rules(target: Path) -> tuple[_StaleResidueRule, ...]:
        target_name = re.escape(target.name)
        backup_pattern = re.compile(
            rf"\.{target_name}\.bak-{_TRANSACTION_TOKEN_PATTERN}"
        )
        if not target.name.endswith((".fif", ".fif.gz")):
            temporary_pattern = re.compile(
                rf"\.{target_name}\.tmp-{_TRANSACTION_TOKEN_PATTERN}"
                rf"{re.escape(target.suffix)}"
            )
            return (
                _StaleResidueRule(backup_pattern, False, target),
                _StaleResidueRule(temporary_pattern, False, target),
            )

        if target.name.endswith(".fif.gz"):
            stem = target.name[: -len(".fif.gz")]
            suffix = ".fif.gz"
        else:
            stem = target.name[: -len(".fif")]
            suffix = ".fif"
        split_backup_pattern = re.compile(
            rf"\.{re.escape(stem)}-[0-9]+{re.escape(suffix)}"
            rf"\.bak-{_TRANSACTION_TOKEN_PATTERN}"
        )
        staging_pattern = re.compile(
            rf"\.{target_name}\.tmp-{_TRANSACTION_TOKEN_PATTERN}"
        )
        return (
            _StaleResidueRule(backup_pattern, False, target),
            _StaleResidueRule(split_backup_pattern, False, target),
            _StaleResidueRule(staging_pattern, True, target),
        )

    @staticmethod
    def _remove_stale_residue(candidate: Path, rule: _StaleResidueRule) -> None:
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning(
                "Atomic output stale-residue inspection failed for target %s "
                "at %s: %s",
                rule.target,
                candidate,
                exc,
            )
            return
        try:
            if stat.S_ISLNK(mode):
                candidate.unlink()
                logger.warning(
                    "Removed symbolic-link atomic output residue for target %s: %s",
                    rule.target,
                    candidate,
                )
                return
            if rule.expected_directory:
                if not stat.S_ISDIR(mode):
                    logger.warning(
                        "Skipped atomic output residue with unexpected type for "
                        "target %s: %s (expected directory, mode=%s)",
                        rule.target,
                        candidate,
                        oct(mode),
                    )
                    return
                shutil.rmtree(candidate)
                return
            if not stat.S_ISREG(mode):
                logger.warning(
                    "Skipped atomic output residue with unexpected type for "
                    "target %s: %s (expected regular file, mode=%s)",
                    rule.target,
                    candidate,
                    oct(mode),
                )
                return
            candidate.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning(
                "Atomic output stale-residue cleanup failed for target %s at %s: %s",
                rule.target,
                candidate,
                exc,
            )

    def _prepare_manifest(self) -> None:
        run_id = os.environ.get(OUTPUT_TRANSACTION_RUN_ID_ENV, "").strip()
        if not self._prepared or not run_id:
            return

        from lfptensorpipe.app.tensor.transaction_manifest import (
            TRANSACTION_MANIFEST_SCHEMA,
            transaction_manifest_path,
            write_transaction_manifest,
        )

        self._manifest_path = transaction_manifest_path(
            self._prepared[0].target,
            run_id=run_id,
            token=uuid4().hex,
        )
        self._manifest_payload = {
            "schema": TRANSACTION_MANIFEST_SCHEMA,
            "run_id": run_id,
            "phase": "prepared",
            "entries": self._manifest_entries(),
        }
        try:
            write_transaction_manifest(
                self._manifest_path,
                self._manifest_payload,
            )
        except Exception:
            self._manifest_path.with_name(f"{self._manifest_path.name}.writing").unlink(
                missing_ok=True
            )
            raise

    def _manifest_entries(self) -> list[dict[str, object]]:
        return [
            {
                "target": str(item.target),
                "temporary": str(item.temporary),
                "backup": str(item.backup),
                "had_original": item.had_original,
            }
            for item in self._prepared
        ]

    def _refresh_prepared_manifest(self) -> None:
        if self._manifest_path is None or self._manifest_payload is None:
            return
        from lfptensorpipe.app.tensor.transaction_manifest import (
            write_transaction_manifest,
        )

        self._manifest_payload["entries"] = self._manifest_entries()
        write_transaction_manifest(self._manifest_path, self._manifest_payload)

    def _mark_manifest_committed(self) -> None:
        if self._manifest_path is None or self._manifest_payload is None:
            return
        from lfptensorpipe.app.tensor.transaction_manifest import (
            write_transaction_manifest,
        )

        self._manifest_payload["phase"] = "committed"
        write_transaction_manifest(self._manifest_path, self._manifest_payload)

    def _cleanup_paths(self) -> None:
        for item in self._prepared:
            item.temporary.unlink(missing_ok=True)
            item.backup.unlink(missing_ok=True)
        self._cleanup_staging_directories()

    def _cleanup_staging_directories(self) -> None:
        for directory in self._staging_directories():
            if directory.exists():
                shutil.rmtree(directory)

    def _cleanup_manifest(self) -> None:
        if self._manifest_path is None:
            return
        self._manifest_path.unlink(missing_ok=True)
        self._manifest_path.with_name(f"{self._manifest_path.name}.writing").unlink(
            missing_ok=True
        )

    def _cleanup_rejected_candidate(self) -> None:
        self._cleanup_paths()
        self._cleanup_manifest()

    def _cleanup_accepted_candidate(self) -> None:
        if self._manifest_path is None:
            self._cleanup_paths()
            return
        try:
            self._cleanup_paths()
            self._cleanup_manifest()
        except OSError:
            # The committed Tensor manifest owns terminal cleanup.
            return


def write_outputs_atomically(
    outputs: list[tuple[Path, OutputWriter]],
    *,
    replace_fn: ReplaceFn | None = None,
    cleanup_stale_residues: bool = False,
    precommit_check: PrecommitCheck | None = None,
) -> None:
    """Write and promote a complete fixed output set."""
    if not outputs:
        return
    with AtomicOutputSet(
        [target for target, _writer in outputs],
        replace_fn=replace_fn,
        cleanup_stale_residues=cleanup_stale_residues,
    ) as output_set:
        for target, writer in outputs:
            writer(output_set.staged_path(target))
        if precommit_check is not None:
            precommit_check()
        output_set.commit()


__all__ = [
    "AtomicOutputSet",
    "OUTPUT_TRANSACTION_RUN_ID_ENV",
    "write_outputs_atomically",
]
