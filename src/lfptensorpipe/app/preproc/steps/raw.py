"""Raw-step acceptance and restoration helpers for preprocess stage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet
from lfptensorpipe.app.shared.generation_lineage import (
    new_result_generation_id,
    params_with_generation_lineage,
)

from ..paths import preproc_step_log_path, preproc_step_raw_path, rawdata_input_fif_path

MarkStepFn = Callable[..., Any]


def _copy_raw_fif_set(
    source: Path, staged: Path, *, copy2_fn: Callable[..., Any] | None = None
) -> None:
    """Stage the primary FIF and every continuation referenced by its Raw."""
    import mne

    copy_file = copy2_fn or shutil.copy2
    copy_file(source, staged)
    with mne.io.read_raw_fif(source, preload=False, verbose="ERROR") as raw:
        for filename in raw.filenames[1:]:
            part = Path(filename)
            copy_file(part, staged.with_name(part.name))


def restore_raw_step_from_rawdata(
    context: RecordContext, *, mark_preproc_step_fn: MarkStepFn
) -> tuple[bool, str]:
    """Restore the canonical FIF set without losing a previously accepted Raw."""
    from send2trash import send2trash

    resolver = PathResolver(context)
    source = rawdata_input_fif_path(context)
    target = preproc_step_raw_path(resolver, "raw")
    log_path = preproc_step_log_path(resolver, "raw")
    if not source.is_file() or not target.is_file():
        return False, "Restore requires canonical rawdata and an existing preproc Raw."
    retained = []

    def retire(path: Path) -> None:
        try:
            send2trash(str(path))
        except OSError as exc:
            # Promotion succeeded; retain the original backup if Trash is unavailable.
            retained.append(f"{path} ({exc})")

    try:
        with AtomicOutputSet(
            [target, log_path], retire_existing_fn=retire
        ) as output_set:
            _copy_raw_fif_set(source, output_set.staged_path(target))
            mark_preproc_step_fn(
                resolver=resolver,
                step="raw",
                completed=True,
                params=params_with_generation_lineage(
                    {"source_step": "rawdata", "action": "restore"},
                    result_generation_id=new_result_generation_id(),
                    input_generations={},
                ),
                input_path=str(source),
                output_path=str(target),
                message="Restored Raw from canonical rawdata.",
                log_path=output_set.staged_path(log_path),
            )
            output_set.commit()
    except (OSError, ValueError, RuntimeError) as exc:
        return False, f"Raw Restore failed: {exc}"
    message = "Raw restored from rawdata; dependent results are stale."
    if retained:
        message += (
            " Previous files could not be moved to Trash and remain at: "
            + "; ".join(retained)
        )
    return True, message


def bootstrap_raw_step_from_rawdata(
    context: RecordContext,
    *,
    rawdata_input_fif_path_fn: Callable[[RecordContext], Path],
    preproc_step_raw_path_fn: Callable[[PathResolver, str], Path],
    mark_preproc_step_fn: MarkStepFn,
    reviewed_raw: Any = None,
    review_is_current_fn: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
    """Accept canonical Raw, including optional browser edits, into preproc."""
    resolver = PathResolver(context)
    src = rawdata_input_fif_path_fn(context)
    dst = preproc_step_raw_path_fn(resolver, "raw")
    if not src.exists():
        mark_preproc_step_fn(
            resolver=resolver,
            step="raw",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message="Missing canonical rawdata raw.fif input.",
        )
        return False, "Missing canonical rawdata input raw.fif."

    log_path = preproc_step_log_path(resolver, "raw")
    success_params = params_with_generation_lineage(
        {},
        result_generation_id=new_result_generation_id(),
        input_generations={},
    )
    try:
        with AtomicOutputSet(
            [dst, log_path],
            cleanup_stale_residues=True,
        ) as output_set:
            if reviewed_raw is None:
                _copy_raw_fif_set(src, output_set.staged_path(dst))
            else:
                reviewed_raw.save(
                    str(output_set.staged_path(dst)), fmt="double", overwrite=True
                )
            mark_preproc_step_fn(
                resolver=resolver,
                step="raw",
                completed=True,
                params=success_params,
                input_path=str(src),
                output_path=str(dst),
                message="Accepted canonical rawdata input into preproc raw step.",
                log_path=output_set.staged_path(log_path),
            )
            if review_is_current_fn is not None and not review_is_current_fn():
                return (
                    False,
                    "Raw review source changed before acceptance; edits discarded.",
                )
            output_set.commit()
    except Exception as exc:  # noqa: BLE001
        mark_preproc_step_fn(
            resolver=resolver,
            step="raw",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Raw step failed: {exc}",
        )
        return False, f"Raw step failed: {exc}"
    return True, "Raw step bootstrapped from rawdata input."
