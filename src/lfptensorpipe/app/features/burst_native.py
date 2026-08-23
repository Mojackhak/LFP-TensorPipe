"""Native-timeline Burst scalar extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.tensor.paths import tensor_metric_tensor_path
from lfptensorpipe.io.pkl_io import load_pkl
from lfptensorpipe.lfp.burst.semantics import (
    has_compatible_burst_value_semantics,
)
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
    transform_policy_from_metadata,
)

from .native_mapping import aligned_row_metadata, load_alignment_mapping_state

BURST_REDUCERS = ("mean", "rate", "duration", "occupancy")
LEGACY_BURST_REDUCER_WARNING = (
    "Burst reducer 'occupation' is deprecated; using 'occupancy'."
)
BURST_UNITS = {
    "mean": "threshold multiple",
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


def _load_native_burst_tensor(
    resolver: PathResolver,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    source_payload = load_pkl(tensor_metric_tensor_path(resolver, "burst"))
    if not isinstance(source_payload, dict):
        raise ValueError("Burst tensor payload is invalid.")
    metadata = source_payload.get("meta")
    params = metadata.get("params") if isinstance(metadata, dict) else None
    bands_segments_hz = (
        params.get("bands_segments_hz") if isinstance(params, dict) else None
    )
    if not isinstance(metadata, dict) or not has_compatible_burst_value_semantics(
        metadata.get("value_semantics"),
        bands_segments_hz=bands_segments_hz,
    ):
        raise ValueError(
            "Legacy Burst tensor detected. Rerun Burst, Align Run, Finish, and Extract Features."
        )
    if transform_policy_from_metadata(metadata) != get_transform_policy("log10"):
        raise ValueError(
            "Legacy Burst transform policy detected. Rerun Burst, Align Run, Finish, and Extract Features."
        )
    if not isinstance(params, dict) or int(params.get("decim_eff", 0)) != 1:
        raise ValueError("Burst feature extraction requires a native-rate tensor.")
    tensor = np.asarray(source_payload.get("tensor"), dtype=float)
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Burst tensor must have shape (1, channel, band, time).")
    if np.any(np.isfinite(tensor) & (tensor < 0.0)):
        raise ValueError(
            "Burst tensor contains negative values outside its value contract."
        )
    if np.any(np.isinf(tensor)):
        raise ValueError(
            "Burst tensor contains infinite values outside its value contract."
        )
    if np.any(np.isfinite(tensor) & (tensor > 0.0) & (tensor <= 1.0)):
        raise ValueError(
            "Burst tensor contains positive values that are not greater than one."
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
    method, method_params, all_epochs, picks = load_alignment_mapping_state(
        resolver,
        trial_slug,
        metric_key="burst",
    )
    tensor, channels, bands, source_times = _load_native_burst_tensor(resolver)
    metadata_rows = aligned_row_metadata(
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
    tables: dict[str, pd.DataFrame] = {}
    for reducer, records in records_by_reducer.items():
        table = pd.DataFrame.from_records(records)
        policy_mode = "log10" if reducer == "mean" else "none"
        table.attrs[VALUE_TRANSFORM_POLICY_KEY] = transform_policy_metadata(
            get_transform_policy(policy_mode)
        )
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
