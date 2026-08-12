"""Native-timeline Burst scalar extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lfptensorpipe.app.alignment.paths import (
    alignment_paradigm_log_path,
    alignment_warp_labels_path,
)
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.tensor.paths import tensor_metric_tensor_path
from lfptensorpipe.io.pkl_io import load_pkl
from lfptensorpipe.lfp.burst.semantics import has_current_burst_value_semantics
from lfptensorpipe.lfp.burst.timeline import (
    MappedFragment,
    build_warp_segments,
    burst_event_ids,
    map_percent_intervals,
    reduce_burst_scalars,
    sample_cell_edges,
)
from lfptensorpipe.utils.transforms import (
    VALUE_TRANSFORM_POLICY_KEY,
    get_transform_policy,
    transform_policy_metadata,
)

BURST_REDUCERS = ("mean", "rate", "duration", "occupancy")
LEGACY_BURST_REDUCER_WARNING = (
    "Burst reducer 'occupation' is deprecated; using 'occupancy'."
)
BURST_UNITS = {
    "mean": "V",
    "rate": "bursts/s",
    "duration": "s",
    "occupancy": "%",
}


def normalize_burst_reducers(reducers: Iterable[str]) -> list[str]:
    """Normalize Burst reducers and accept the legacy occupation spelling."""
    normalized: list[str] = []
    for raw_value in reducers:
        value = str(raw_value).strip().lower()
        if value == "occupation":
            value = "occupancy"
        if value not in BURST_REDUCERS:
            raise ValueError(f"Unsupported Burst reducer: {raw_value}")
        if value not in normalized:
            normalized.append(value)
    return normalized or list(BURST_REDUCERS)


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


def _load_alignment_mapping_state(
    resolver: PathResolver,
    trial_slug: str,
) -> tuple[str, dict[str, Any], list[Any], list[int]]:
    log_payload = read_run_log(alignment_paradigm_log_path(resolver, trial_slug))
    if not isinstance(log_payload, dict):
        raise ValueError("Alignment log payload is invalid.")
    run_match = _latest_successful_entry(log_payload, "run_align_epochs")
    finish_match = _latest_successful_entry(log_payload, "build_raw_table")
    if run_match is None or finish_match is None:
        raise ValueError(
            "Burst features require successful Align Run and Finish entries."
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


def _load_native_burst_tensor(
    resolver: PathResolver,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    source_payload = load_pkl(tensor_metric_tensor_path(resolver, "burst"))
    if not isinstance(source_payload, dict):
        raise ValueError("Burst tensor payload is invalid.")
    metadata = source_payload.get("meta")
    if not isinstance(metadata, dict) or not has_current_burst_value_semantics(
        metadata.get("value_semantics")
    ):
        raise ValueError(
            "Legacy Burst tensor detected. Rerun Burst, Align Run, Finish, and Extract Features."
        )
    params = metadata.get("params")
    if not isinstance(params, dict) or int(params.get("decim_eff", 0)) != 1:
        raise ValueError("Burst feature extraction requires a native-rate tensor.")
    tensor = np.asarray(source_payload.get("tensor"), dtype=float)
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Burst tensor must have shape (1, channel, band, time).")
    if np.any(np.isfinite(tensor) & (tensor < 0.0)):
        raise ValueError(
            "Burst tensor contains negative values outside its value contract."
        )
    axes = metadata.get("axes")
    if not isinstance(axes, dict):
        raise ValueError("Burst tensor metadata is missing axes.")
    channels = [str(value) for value in list(axes.get("channel", []))]
    bands = [str(value) for value in list(axes.get("freq", []))]
    times = np.asarray(axes.get("time"), dtype=float)
    if (
        len(channels) != tensor.shape[1]
        or len(bands) != tensor.shape[2]
        or times.size != tensor.shape[3]
    ):
        raise ValueError("Burst tensor axes do not match its array shape.")
    return tensor[0], channels, bands, times


def _aligned_row_metadata(
    aligned_payload: pd.DataFrame,
    *,
    selected_count: int,
    channels: list[str],
) -> list[dict[str, Any]]:
    expected_rows = selected_count * len(channels)
    if len(aligned_payload) != expected_rows:
        raise ValueError(
            "Alignment Burst raw-table rows do not match Finish picks and channels."
        )
    metadata_rows: list[dict[str, Any]] = []
    for epoch_position in range(selected_count):
        for channel_index, channel in enumerate(channels):
            row = aligned_payload.iloc[epoch_position * len(channels) + channel_index]
            if str(row.get("Channel", "")) != channel:
                raise ValueError(
                    "Alignment Burst raw-table channel order is inconsistent."
                )
            metadata_rows.append(
                {
                    column: row[column]
                    for column in aligned_payload.columns
                    if column != "Value"
                }
            )
    return metadata_rows


def build_burst_scalar_tables(
    resolver: PathResolver,
    *,
    trial_slug: str,
    aligned_payload: pd.DataFrame,
    phases: dict[str, list[list[float]]],
    reducers: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Build normalized Burst scalar tables from the original Burst tensor."""
    reducer_names = list(reducers) or list(BURST_REDUCERS)
    if not phases:
        raise ValueError("Burst scalar extraction requires at least one Feature phase.")
    method, method_params, all_epochs, picks = _load_alignment_mapping_state(
        resolver, trial_slug
    )
    tensor, channels, bands, source_times = _load_native_burst_tensor(resolver)
    metadata_rows = _aligned_row_metadata(
        aligned_payload,
        selected_count=len(picks),
        channels=channels,
    )
    records_by_reducer: dict[str, list[dict[str, Any]]] = {
        reducer: [] for reducer in reducer_names
    }
    cell_edges = sample_cell_edges(source_times)
    fragments_by_epoch: list[dict[str, list[MappedFragment]]] = []
    for epoch_index in picks:
        segments = build_warp_segments(method, method_params, all_epochs[epoch_index])
        fragments_by_epoch.append(
            {
                phase_name: map_percent_intervals(
                    segments,
                    [(float(item[0]), float(item[1])) for item in phase_intervals],
                )
                for phase_name, phase_intervals in phases.items()
            }
        )

    results: dict[tuple[int, int, str, int], Any] = {}
    for channel_index, _channel in enumerate(channels):
        for band_index, _band_name in enumerate(bands):
            series = tensor[channel_index, band_index]
            event_ids = burst_event_ids(series)
            for epoch_position, fragments_by_phase in enumerate(fragments_by_epoch):
                for phase_name, fragments in fragments_by_phase.items():
                    results[(epoch_position, channel_index, phase_name, band_index)] = (
                        reduce_burst_scalars(
                            series,
                            source_times,
                            fragments,
                            cell_edges=cell_edges,
                            event_ids=event_ids,
                        )
                    )

    for epoch_position, _epoch_index in enumerate(picks):
        for channel_index, _channel in enumerate(channels):
            base = metadata_rows[epoch_position * len(channels) + channel_index]
            for phase_name in phases:
                for band_index, band_name in enumerate(bands):
                    result = results[
                        (epoch_position, channel_index, phase_name, band_index)
                    ]
                    for reducer in reducer_names:
                        record = dict(base)
                        record["Value"] = float(getattr(result, reducer))
                        record["Band"] = band_name
                        record["Phase"] = phase_name
                        record["Unit"] = BURST_UNITS[reducer]
                        records_by_reducer[reducer].append(record)
    policy = transform_policy_metadata(get_transform_policy("none"))
    tables: dict[str, pd.DataFrame] = {}
    for reducer, records in records_by_reducer.items():
        table = pd.DataFrame.from_records(records)
        table.attrs[VALUE_TRANSFORM_POLICY_KEY] = policy
        tables[reducer] = table
    return tables


def cleanup_legacy_occupation_outputs(metric_output_dir: Path) -> None:
    """Remove deprecated occupation artifacts after accepted replacement outputs."""
    for path in metric_output_dir.glob("occupation-*"):
        if path.is_file():
            path.unlink()


__all__ = [
    "BURST_REDUCERS",
    "BURST_UNITS",
    "LEGACY_BURST_REDUCER_WARNING",
    "build_burst_scalar_tables",
    "cleanup_legacy_occupation_outputs",
    "normalize_burst_reducers",
]
