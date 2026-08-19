"""Shared accepted Alignment mapping state for native-timeline Features."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from lfptensorpipe.app.alignment.generation import (
    accepted_alignment_metrics,
    alignment_generation_rerun_message,
)
from lfptensorpipe.app.alignment.paths import (
    alignment_paradigm_log_path,
    alignment_warp_labels_path,
)
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.io.pkl_io import load_pkl


def _latest_successful_entry(
    payload: dict[str, Any],
    step: str,
) -> tuple[int, dict[str, Any]] | None:
    history = payload.get("history")
    if not isinstance(history, list):
        return None
    for index in range(len(history) - 1, -1, -1):
        entry = history[index]
        if (
            isinstance(entry, dict)
            and str(entry.get("step", "")) == step
            and entry.get("completed") is True
        ):
            return index, entry
    return None


def load_alignment_mapping_state(
    resolver: PathResolver,
    trial_slug: str,
    *,
    metric_key: str,
) -> tuple[str, dict[str, Any], list[Any], list[int]]:
    """Load the accepted method, epoch mappings, and Finish picks for a metric."""
    rerun_message = alignment_generation_rerun_message(
        resolver,
        trial_slug=trial_slug,
        stage="finish",
    )
    if rerun_message is not None:
        raise ValueError(rerun_message)
    run_metrics = accepted_alignment_metrics(
        resolver,
        trial_slug=trial_slug,
        stage="run",
    )
    if run_metrics is None:
        raise ValueError(
            alignment_generation_rerun_message(
                resolver,
                trial_slug=trial_slug,
                stage="run",
            )
            or "Alignment has no accepted Run metric generation."
        )
    finish_metrics = accepted_alignment_metrics(
        resolver,
        trial_slug=trial_slug,
        stage="finish",
    )
    if finish_metrics is None:
        raise ValueError(
            alignment_generation_rerun_message(
                resolver,
                trial_slug=trial_slug,
                stage="finish",
            )
            or "Latest Alignment Run has not been finished."
        )
    if metric_key not in finish_metrics:
        raise ValueError(f"Accepted Alignment Finish does not include {metric_key}.")

    log_payload = read_run_log(alignment_paradigm_log_path(resolver, trial_slug))
    if not isinstance(log_payload, dict):
        raise ValueError("Alignment log payload is invalid.")
    run_match = _latest_successful_entry(log_payload, "run_align_epochs")
    finish_match = _latest_successful_entry(log_payload, "build_raw_table")
    if run_match is None or finish_match is None:
        raise ValueError(
            "Native-timeline features require successful Align Run and Finish entries."
        )
    run_index, run_entry = run_match
    finish_index, finish_entry = finish_match
    if finish_index <= run_index:
        raise ValueError("Latest successful Alignment Run has not been finished.")
    run_params = run_entry.get("params")
    finish_params = finish_entry.get("params")
    if not isinstance(run_params, dict) or not isinstance(finish_params, dict):
        raise ValueError("Alignment mapping log parameters are invalid.")
    method = str(run_params.get("method", "")).strip()
    method_params = run_params.get("method_params")
    picks_raw = finish_params.get("picked_epoch_indices")
    if (
        not method
        or not isinstance(method_params, dict)
        or not isinstance(picks_raw, list)
    ):
        raise ValueError("Alignment mapping state is incomplete.")
    picks = sorted({int(value) for value in picks_raw if int(value) >= 0})
    labels_payload = load_pkl(alignment_warp_labels_path(resolver, trial_slug))
    epochs = labels_payload.get("ALL") if isinstance(labels_payload, dict) else None
    if not isinstance(epochs, list) or not picks:
        raise ValueError("Persisted Alignment epochs or Finish picks are missing.")
    if any(index >= len(epochs) for index in picks):
        raise ValueError(
            "Finish picked epoch indices exceed persisted Alignment epochs."
        )
    return method, method_params, epochs, picks


def aligned_row_metadata(
    aligned_payload: pd.DataFrame,
    *,
    selected_count: int,
    channels: Sequence[Any],
) -> list[dict[str, Any]]:
    """Return aligned row metadata in accepted epoch-major/channel-major order."""
    expected_rows = selected_count * len(channels)
    if len(aligned_payload) != expected_rows:
        raise ValueError(
            "Alignment raw-table rows do not match Finish picks and channels."
        )
    metadata_rows: list[dict[str, Any]] = []
    for epoch_position in range(selected_count):
        for channel_index, channel in enumerate(channels):
            row = aligned_payload.iloc[epoch_position * len(channels) + channel_index]
            row_channel = row.get("Channel", "")
            if str(row_channel) != str(channel):
                raise ValueError("Alignment raw-table channel order is inconsistent.")
            metadata_rows.append(
                {
                    column: row[column]
                    for column in aligned_payload.columns
                    if column != "Value"
                }
            )
    return metadata_rows


__all__ = ["aligned_row_metadata", "load_alignment_mapping_state"]
