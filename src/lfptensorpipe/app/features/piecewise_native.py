"""Native-timeline mean reduction for discontinuous Clip/Stitch fragments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lfptensorpipe.app.alignment.tensor_inventory import _coerce_alignment_tensor
from lfptensorpipe.app.alignment.warper_builder import _resolve_target_duration_s
from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.tensor.paths import tensor_metric_tensor_path
from lfptensorpipe.io.pkl_io import load_pkl
from lfptensorpipe.lfp.burst.timeline import (
    MappedFragment,
    WarpSegment,
    build_warp_segments,
    map_percent_intervals,
)
from lfptensorpipe.lfp.warp.utils import interp_along_last_axis
from lfptensorpipe.tabular.grid import (
    AxisInfo,
    AxisSelection,
    GridResultColumns,
    mean_sufficient_statistics,
    parse_band_definitions,
    prepare_scalogram,
)
from lfptensorpipe.utils.transforms import (
    TransformPolicy,
    convert_transform_domain_array,
)

from .native_mapping import aligned_row_metadata, load_alignment_mapping_state


def _load_native_metric_tensor(
    resolver: PathResolver,
    *,
    metric_key: str,
    transform_policy: TransformPolicy,
) -> tuple[np.ndarray, list[Any], list[Any], np.ndarray]:
    payload = load_pkl(tensor_metric_tensor_path(resolver, metric_key))
    if not isinstance(payload, dict):
        raise ValueError(f"Native Tensor payload is invalid for {metric_key}.")
    tensor, metadata = _coerce_alignment_tensor(payload)
    axes = metadata.get("axes")
    if not isinstance(axes, dict):
        raise ValueError(f"Native Tensor axes are missing for {metric_key}.")
    channels = list(axes.get("channel", []))
    freqs = list(axes.get("freq", []))
    times = np.asarray(axes.get("time"), dtype=float)
    if (
        len(channels) != tensor.shape[0]
        or len(freqs) != tensor.shape[1]
        or times.size != tensor.shape[2]
    ):
        raise ValueError(f"Native Tensor axes do not match {metric_key} shape.")
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(
            f"Native Tensor time axis must be finite and increasing for {metric_key}."
        )
    reduction_tensor = convert_transform_domain_array(
        tensor,
        mode=transform_policy.mode,
        source_domain=transform_policy.tensor_storage_domain,
        target_domain=transform_policy.reduction_domain,
    )
    return np.asarray(reduction_tensor, dtype=float), channels, freqs, times


def _phase_percent_intervals(
    phases: dict[str, list[list[float]]],
    *,
    interval_mode: str,
    target_duration_s: float,
) -> dict[str, list[tuple[float, float]]]:
    if interval_mode not in {"percent", "absolute"}:
        raise ValueError("time_interval_mode must be 'absolute' or 'percent'.")
    scale = 1.0 if interval_mode == "percent" else 100.0 / target_duration_s
    return {
        phase_name: [
            (float(bounds[0]) * scale, float(bounds[1]) * scale) for bounds in intervals
        ]
        for phase_name, intervals in phases.items()
    }


def _fragment_values(
    values: np.ndarray,
    source_times: np.ndarray,
    fragment: MappedFragment,
    segment: WarpSegment,
) -> tuple[np.ndarray, np.ndarray]:
    start = float(fragment.source_start)
    end = float(fragment.source_end)
    source_stop = float(source_times[-1] + (source_times[-1] - source_times[-2]))
    tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            1.0,
            abs(start),
            abs(end),
            abs(float(source_times[0])),
            abs(source_stop),
        )
    )
    if start < float(source_times[0]) - tolerance or end > source_stop + tolerance:
        raise ValueError("Alignment fragment exceeds native Tensor time support.")
    start = max(start, float(source_times[0]))
    reaches_segment_end = abs(end - float(segment.source_end)) <= tolerance
    if reaches_segment_end:
        eligible = source_times[source_times < float(segment.source_end) - tolerance]
        if eligible.size == 0:
            end = start
        else:
            end = min(end, float(eligible[-1]))
    else:
        end = min(end, float(source_times[-1]))
    if end < start:
        return np.asarray([], dtype=float), values[..., :0]
    if end == start:
        nodes = np.asarray([start], dtype=float)
        source_indices = np.interp(
            nodes,
            source_times,
            np.arange(source_times.size, dtype=float),
        )
        return nodes, interp_along_last_axis(values, source_indices)

    interior = source_times[(source_times > start) & (source_times < end)]
    nodes = np.concatenate(([start], interior, [end])).astype(float, copy=False)
    source_indices = np.interp(
        nodes,
        source_times,
        np.arange(source_times.size, dtype=float),
    )
    return nodes, interp_along_last_axis(values, source_indices)


def _numeric_selection(start: float, end: float, size: int) -> AxisSelection:
    return AxisSelection(
        mask=np.ones(size, dtype=bool),
        total_length=float(end - start),
        intervals=((float(start), float(end)),),
    )


def _spectral_mean(
    prepared_fragments: list[tuple[np.ndarray, np.ndarray]],
    freq_labels: pd.Index,
) -> pd.Series:
    numerators = np.zeros(len(freq_labels), dtype=float)
    denominators = np.zeros(len(freq_labels), dtype=float)
    for nodes, fragment_values in prepared_fragments:
        time_axis = AxisInfo(pd.Index(nodes), nodes, True)
        selection = _numeric_selection(nodes[0], nodes[-1], nodes.size)
        for freq_index in range(len(freq_labels)):
            numerator, denominator = mean_sufficient_statistics(
                time_axis,
                fragment_values[freq_index],
                selection,
            )
            numerators[freq_index] += numerator
            denominators[freq_index] += denominator
    means = np.divide(
        numerators,
        denominators,
        out=np.full(numerators.shape, np.nan, dtype=float),
        where=denominators > 0.0,
    )
    return pd.Series(means, index=freq_labels, dtype=float)


def _scalar_mean(
    prepared_fragments: list[tuple[np.ndarray, np.ndarray]],
    freq_axis: AxisInfo,
    freq_selection: AxisSelection,
) -> float:
    total_numerator = 0.0
    total_denominator = 0.0
    for nodes, fragment_values in prepared_fragments:
        column_numerators = np.full(nodes.size, np.nan, dtype=float)
        column_denominators = np.full(nodes.size, np.nan, dtype=float)
        for time_index in range(nodes.size):
            numerator, denominator = mean_sufficient_statistics(
                freq_axis,
                fragment_values[:, time_index],
                freq_selection,
            )
            if denominator > 0.0:
                column_numerators[time_index] = numerator
                column_denominators[time_index] = denominator

        time_axis = AxisInfo(pd.Index(nodes), nodes, True)
        time_selection = _numeric_selection(nodes[0], nodes[-1], nodes.size)
        numerator, _ = mean_sufficient_statistics(
            time_axis,
            column_numerators,
            time_selection,
        )
        denominator, _ = mean_sufficient_statistics(
            time_axis,
            column_denominators,
            time_selection,
        )
        total_numerator += numerator
        total_denominator += denominator
    return (
        float(total_numerator / total_denominator)
        if total_denominator > 0.0
        else float("nan")
    )


def build_piecewise_mean_outputs(
    resolver: PathResolver,
    *,
    trial_slug: str,
    metric_key: str,
    aligned_payload: pd.DataFrame,
    phases: dict[str, list[list[float]]],
    bands: dict[str, Any],
    transform_policy: TransformPolicy,
    include_spectral: bool,
    include_scalar: bool,
    time_interval_mode: str,
    freq_interval_mode: str,
    inclusive: bool,
    keep_full_dim_cols: bool,
    out_cols: GridResultColumns,
) -> dict[str, pd.DataFrame]:
    """Build Clip/Stitch mean outputs from independently integrated fragments."""
    method, method_params, all_epochs, picks = load_alignment_mapping_state(
        resolver,
        trial_slug,
        metric_key=metric_key,
    )
    if method not in {"pad_warper", "concat_warper"}:
        raise ValueError("Piecewise native means require Clip or Stitch alignment.")
    tensor, channels, freqs, source_times = _load_native_metric_tensor(
        resolver,
        metric_key=metric_key,
        transform_policy=transform_policy,
    )
    metadata_rows = aligned_row_metadata(
        aligned_payload,
        selected_count=len(picks),
        channels=channels,
    )
    target_duration_s = _resolve_target_duration_s(
        method=method,
        method_params=method_params,
        epochs_by_label={"ALL": all_epochs},
    )
    phase_intervals = _phase_percent_intervals(
        phases,
        interval_mode=time_interval_mode,
        target_duration_s=target_duration_s,
    )
    fragments_by_epoch: list[dict[str, list[MappedFragment]]] = []
    segments_by_epoch: list[dict[int, WarpSegment]] = []
    for epoch_index in picks:
        segments = build_warp_segments(method, method_params, all_epochs[epoch_index])
        segments_by_epoch.append({segment.occurrence: segment for segment in segments})
        fragments_by_epoch.append(
            {
                phase_name: map_percent_intervals(segments, intervals)
                for phase_name, intervals in phase_intervals.items()
            }
        )

    spectral_records: list[dict[str, Any]] = []
    scalar_records: list[dict[str, Any]] = []
    for epoch_position, fragments_by_phase in enumerate(fragments_by_epoch):
        segments_by_occurrence = segments_by_epoch[epoch_position]
        for channel_index, _channel in enumerate(channels):
            base = metadata_rows[epoch_position * len(channels) + channel_index]
            source_frame = pd.DataFrame(
                tensor[channel_index],
                index=pd.Index(freqs, name="freq"),
                columns=pd.Index(source_times, name="time"),
            )
            sorted_frame, freq_axis, _time_axis = prepare_scalogram(source_frame)
            sorted_values = sorted_frame.to_numpy(dtype=float, copy=False)
            band_definitions = (
                parse_band_definitions(
                    bands,
                    freq_axis,
                    inclusive=inclusive,
                    interval_mode=freq_interval_mode,
                )
                if include_scalar
                else []
            )
            for phase_name, fragments in fragments_by_phase.items():
                prepared_fragments = []
                for fragment in fragments:
                    nodes, fragment_values = _fragment_values(
                        sorted_values,
                        source_times,
                        fragment,
                        segments_by_occurrence[fragment.occurrence],
                    )
                    if nodes.size >= 2:
                        prepared_fragments.append((nodes, fragment_values))
                if include_spectral:
                    record = dict(base)
                    record[out_cols.value] = _spectral_mean(
                        prepared_fragments,
                        sorted_frame.index,
                    )
                    record[out_cols.phase] = phase_name
                    if keep_full_dim_cols:
                        record[out_cols.band] = pd.NA
                    spectral_records.append(record)
                if include_scalar:
                    for band_name, freq_selection in band_definitions:
                        record = dict(base)
                        record[out_cols.value] = _scalar_mean(
                            prepared_fragments,
                            freq_axis,
                            freq_selection,
                        )
                        record[out_cols.band] = band_name
                        record[out_cols.phase] = phase_name
                        scalar_records.append(record)

    outputs: dict[str, pd.DataFrame] = {}
    if include_spectral:
        outputs["spectral"] = pd.DataFrame.from_records(spectral_records)
    if include_scalar:
        outputs["scalar"] = pd.DataFrame.from_records(scalar_records)
    return outputs


__all__ = ["build_piecewise_mean_outputs"]
