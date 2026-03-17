"""Raw-step bootstrap helpers for preprocess stage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

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

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    mark_preproc_step_fn(
        resolver=resolver,
        step="raw",
        completed=True,
        input_path=str(src),
        output_path=str(dst),
        message="Copied canonical rawdata input into preproc raw step.",
    )
    return True, "Raw step bootstrapped from rawdata input."
