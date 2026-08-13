"""Raw-step bootstrap helpers for preprocess stage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet

from ..paths import preproc_step_log_path

MarkStepFn = Callable[..., Any]


def bootstrap_raw_step_from_rawdata(
    context: RecordContext,
    *,
    rawdata_input_fif_path_fn: Callable[[RecordContext], Path],
    preproc_step_raw_path_fn: Callable[[PathResolver, str], Path],
    mark_preproc_step_fn: MarkStepFn,
) -> tuple[bool, str]:
    """Copy canonical rawdata FIF into preproc `raw/` and mark step complete."""
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
    try:
        with AtomicOutputSet([dst, log_path]) as output_set:
            shutil.copy2(src, output_set.staged_path(dst))
            mark_preproc_step_fn(
                resolver=resolver,
                step="raw",
                completed=True,
                input_path=str(src),
                output_path=str(dst),
                message="Copied canonical rawdata input into preproc raw step.",
                log_path=output_set.staged_path(log_path),
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
