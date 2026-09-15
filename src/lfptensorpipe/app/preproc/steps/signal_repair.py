"""Accepted Signal Repair output using the existing Preprocess lifecycle."""

from __future__ import annotations

from typing import Any

import mne

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet
from lfptensorpipe.app.shared.generation_lineage import (
    new_result_generation_id,
    params_with_generation_lineage,
)
from lfptensorpipe.preproc.signal_repair import repair_signal

from ..lineage import (
    PreprocInputGenerationChanged,
    capture_preproc_input_generation,
    preproc_input_generation_matches,
)
from ..paths import (
    preproc_step_raw_path,
    preproc_step_log_path,
    preproc_step_config_path,
    write_preproc_step_config,
)


def apply_signal_repair_step(
    context: RecordContext, *, params: dict[str, Any], mark_preproc_step_fn
):
    """Repair from Raw and immediately accept the result without review gating."""
    resolver = PathResolver(context)
    captured = capture_preproc_input_generation(resolver, "signal_repair")
    if captured is None:
        return False, "Signal Repair requires a current Raw result."
    source_step, input_generations = captured
    src = preproc_step_raw_path(resolver, source_step)
    dst = preproc_step_raw_path(resolver, "signal_repair")
    config = preproc_step_config_path(resolver, "signal_repair")
    log = preproc_step_log_path(resolver, "signal_repair")
    raw = mne.io.read_raw_fif(str(src), preload=True, verbose="ERROR")
    output = None
    try:
        output, report = repair_signal(raw, params)
        with AtomicOutputSet(
            [dst, config, log], cleanup_stale_residues=True
        ) as outputs:
            output.save(str(outputs.staged_path(dst)), fmt="double", overwrite=True)
            write_preproc_step_config(
                resolver=resolver,
                step="signal_repair",
                config=params,
                path=outputs.staged_path(config),
            )
            mark_preproc_step_fn(
                resolver=resolver,
                step="signal_repair",
                completed=True,
                params=params_with_generation_lineage(
                    {"source_step": source_step, "settings": params, **report},
                    result_generation_id=new_result_generation_id(),
                    input_generations=input_generations,
                ),
                input_path=str(src),
                output_path=str(dst),
                message="Signal Repair completed.",
                log_path=outputs.staged_path(log),
            )
            if not preproc_input_generation_matches(
                resolver,
                "signal_repair",
                source_step=source_step,
                input_generations=input_generations,
            ):
                raise PreprocInputGenerationChanged(
                    "Signal Repair input changed during execution."
                )
            outputs.commit()
        detail = "; ".join(
            f"{kind}: {counts['repaired_intervals']} repaired intervals, "
            f"{counts['repaired_samples']} samples, {counts['skipped_intervals']} skipped"
            for kind, counts in report["summary"].items()
        )
        return True, detail
    finally:
        raw.close()
        if output is not None:
            output.close()
