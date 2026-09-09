# filter.py
"""
Preprocessing utilities for STN+SNr DBS LFP recordings.

This module centralizes two reusable preprocessing steps that are shared across
multiple paradigms (e.g., rest/standing/gait/pain):

1) BAD-segment annotation:
   Detects artifact-contaminated time windows on continuous MNE Raw objects using
   fixed-length epochs, peak-to-peak (p2p) amplitude criteria, and autoreject-style
   channel-wise thresholds. It also supports marking short "good gaps" as BAD and
   merging contiguous/near-contiguous BAD intervals into consolidated blocks.

2) Filtering and good-segment extraction:
   Transfers BAD annotations onto a target Raw, optionally applies a pre-filter to
   reduce edge artifacts, removes BAD intervals to extract continuous clean segments,
   and applies final notch and band-pass filtering to generate analysis-ready signals.

Design notes:
- BAD masking is restricted to annotations whose description starts with 'BAD' to avoid
  accidentally treating task/event annotations (e.g., gait events or pain markers) as artifacts.
- Functions are config-driven and pipeline-friendly (no working-directory side effects),
  enabling reproducible batch processing and QC figure/log generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import mne
from lfptensorpipe.io.timeline import raw_relative_onsets
from ..lfp.mask.annotations import (
    BoundaryMatchMode,
    MatchMode,
    annotation_sample_support_by_channel,
    valid_segments_from_annotation_support,
)

FILTER_EDGE_DESCRIPTION = "EDGE_filter"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _append_info_description(raw: mne.io.BaseRaw, text: str) -> None:
    """Append a preprocessing note to raw.info['description'] without overwriting."""
    old = raw.info.get("description", "") or ""
    raw.info["description"] = (old + " | " + text).strip(" | ")


def _startswith_any(s: str, prefixes: Sequence[str]) -> bool:
    normalized = s.casefold()
    return any(normalized.startswith(prefix.casefold()) for prefix in prefixes)


def _safe_makedirs(p: Optional[Union[str, Path]]) -> None:
    if p is None:
        return
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def _ensure_mne_exports_for_autoreject() -> None:
    """Warm up MNE exports that `autoreject` expects in frozen builds."""
    export_map: dict[str, Any] = {}

    try:
        from mne._fiff.meas_info import create_info
        from mne._fiff.pick import channel_type, pick_info, pick_types

        export_map.update(
            {
                "channel_type": channel_type,
                "create_info": create_info,
                "pick_info": pick_info,
                "pick_types": pick_types,
            }
        )
    except Exception:
        pass

    try:
        from mne.epochs import BaseEpochs, EpochsArray, make_fixed_length_epochs

        export_map.update(
            {
                "BaseEpochs": BaseEpochs,
                "EpochsArray": EpochsArray,
                "make_fixed_length_epochs": make_fixed_length_epochs,
            }
        )
    except Exception:
        pass

    for name, value in export_map.items():
        if not hasattr(mne, name):
            setattr(mne, name, value)

    try:
        from mne.io.array._array import RawArray

        if not hasattr(mne.io, "RawArray"):
            mne.io.RawArray = RawArray
    except Exception:
        pass

    try:
        from mne.viz.epochs import plot_epochs

        if not hasattr(mne.viz, "plot_epochs"):
            mne.viz.plot_epochs = plot_epochs
    except Exception:
        pass


def _save_reject_log_plot_agg(
    reject_log: Any, reject_plot_path: Union[str, Path]
) -> None:
    """Save reject-log visualization using Agg backend only (thread-safe)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    labels = np.asarray(getattr(reject_log, "labels", []), dtype=float)
    bad_epochs = np.asarray(getattr(reject_log, "bad_epochs", []), dtype=bool)
    ch_names = list(getattr(reject_log, "ch_names", []) or [])

    if labels.ndim == 1:
        labels = labels[np.newaxis, :]
    if labels.ndim != 2:
        labels = np.zeros((1, 1), dtype=float)
    if labels.size == 0:
        labels = np.zeros((1, 1), dtype=float)

    n_epochs = int(labels.shape[0])
    n_channels = int(labels.shape[1])
    width = min(20.0, max(6.0, 0.18 * n_epochs + 3.0))
    height = min(12.0, max(2.4, 0.22 * n_channels + 1.2))

    fig = Figure(figsize=(width, height), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    image = ax.imshow(
        labels.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Channel")
    ax.set_title("Reject Log")

    if n_channels > 0 and ch_names:
        max_ticks = min(16, n_channels)
        tick_idx = np.linspace(0, n_channels - 1, num=max_ticks, dtype=int)
        ax.set_yticks(tick_idx.tolist())
        ax.set_yticklabels(
            [ch_names[idx] if idx < len(ch_names) else str(idx) for idx in tick_idx],
            fontsize=8,
        )

    if bad_epochs.size:
        for bad_idx in np.where(bad_epochs)[0]:
            ax.axvline(float(bad_idx), color="#ff3b30", linewidth=0.6, alpha=0.3)

    fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02, label="Rejected")
    fig.tight_layout()
    fig.savefig(str(reject_plot_path), dpi=300, bbox_inches="tight")


def _restore_channel_types(raw: mne.io.BaseRaw, original_types: Dict[str, str]) -> None:
    """Restore channel types from a mapping of ch_name -> type."""
    # Only restore channels that still exist in the Raw object.
    restore_map = {ch: tp for ch, tp in original_types.items() if ch in raw.ch_names}
    if restore_map:
        raw.set_channel_types(restore_map)


def _get_channel_types_map(raw: mne.io.BaseRaw) -> Dict[str, str]:
    """Return channel type mapping ch_name -> type."""
    types = raw.get_channel_types(picks=None, unique=False)
    return {ch: tp for ch, tp in zip(raw.ch_names, types)}


def _build_bad_sample_mask(
    raw: mne.io.BaseRaw,
    bad_prefixes: Sequence[str] = ("BAD",),
    *,
    channel: str | None = None,
) -> np.ndarray:
    """
    Build a boolean mask of bad samples (length = raw.n_times) from annotations.

    IMPORTANT:
        Only annotations with a description starting with any of ``bad_prefixes``
        are treated as bad. Prefix matching is case-insensitive, matching MNE's
        BAD-annotation semantics.
        This prevents accidental masking of task/event annotations (gait/pain).
        With ``channel=None``, only global annotations are included. With a
        channel name, global annotations plus annotations assigned to that
        channel are included.
    """
    n_times = int(raw.n_times)
    bad_mask = np.zeros(n_times, dtype=bool)

    ann = raw.annotations
    if ann is None or len(ann) == 0:
        return bad_mask

    relative_onsets = raw_relative_onsets(raw)

    for onset_relative, duration, desc, ch_names in zip(
        relative_onsets,
        ann.duration,
        ann.description,
        ann.ch_names,
    ):
        desc = str(desc)
        if not _startswith_any(desc, bad_prefixes):
            continue
        annotation_scope = tuple(str(name) for name in ch_names)
        if annotation_scope and (
            channel is None or str(channel) not in annotation_scope
        ):
            continue

        start_samp, stop_samp = raw.time_as_index(
            [
                float(onset_relative),
                float(onset_relative) + float(duration),
            ],
            use_rounding=True,
        )

        # Clip to valid data range
        start_samp = max(start_samp, 0)
        stop_samp = min(stop_samp, n_times)

        if stop_samp > start_samp:
            bad_mask[start_samp:stop_samp] = True

    return bad_mask


def _set_annotations_from_attached_frame(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations,
) -> None:
    """Attach annotations whose onsets are already in Raw's exposed frame."""
    prepared = annotations
    if annotations.orig_time is None:
        prepared = mne.Annotations(
            onset=np.asarray(annotations.onset, dtype=float) - float(raw.first_time),
            duration=np.asarray(annotations.duration, dtype=float),
            description=np.asarray(annotations.description, dtype=object).tolist(),
            orig_time=None,
            ch_names=list(annotations.ch_names),
        )
    raw.set_annotations(prepared)


def _without_filter_edges(annotations: mne.Annotations) -> mne.Annotations:
    """Remove the exact system-owned filter-support annotation."""
    if len(annotations) == 0:
        return annotations.copy()
    keep = np.asarray(
        [str(desc) != FILTER_EDGE_DESCRIPTION for desc in annotations.description],
        dtype=bool,
    )
    if np.any(keep):
        return annotations[keep]
    return mne.Annotations([], [], [], orig_time=annotations.orig_time)


def _filter_boundary_support(
    raw: mne.io.BaseRaw,
    *,
    channel: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return invalid samples and point boundaries for one channel."""
    invalid, point_boundaries, _ = annotation_sample_support_by_channel(
        raw,
        channels=(channel,),
        keep=("bad", "edge"),
        mode="prefix",
    )
    return invalid[0], point_boundaries[0]


def _valid_filter_segments(
    invalid: np.ndarray,
    point_boundaries: Sequence[int],
) -> list[tuple[int, int]]:
    """Return half-open valid runs split at zero-duration EDGE/BAD points."""
    return valid_segments_from_annotation_support(invalid, point_boundaries)


def _resolved_notch_widths(
    freqs: np.ndarray,
    notch_widths: Union[float, Sequence[float]],
) -> np.ndarray:
    widths = np.atleast_1d(np.asarray(notch_widths, dtype=float))
    if widths.size == 1:
        return np.full(freqs.shape, float(widths[0]), dtype=float)
    if widths.size != freqs.size:
        raise ValueError("notch_widths must be scalar or match the notch count.")
    return widths


def _filter_support_geometry(
    *,
    n_times: int,
    sfreq: float,
    l_freq: float | None,
    h_freq: float | None,
    notches: Sequence[float] | None,
    notch_widths: Union[float, Sequence[float]],
    notch_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine FIR kernel support with the active estimator dependency bound."""
    probe = np.zeros(max(1, int(n_times)), dtype=float)
    kernel_lengths: dict[str, int] = {}

    if l_freq is not None or h_freq is not None:
        band_kernel = mne.filter.create_filter(
            probe,
            sfreq,
            l_freq,
            h_freq,
            phase="zero",
            fir_design="firwin",
            verbose="ERROR",
        )
        kernel_lengths["bandpass"] = int(np.asarray(band_kernel).size)

    resolved_notches = (
        np.asarray(tuple(notches), dtype=float)
        if notches is not None
        else np.empty(0, dtype=float)
    )
    model_radius = 0
    if resolved_notches.size and notch_model and notch_model["enabled"]:
        from .notch import model_support_samples

        model_radius = model_support_samples(notch_model, sfreq, n_times)
    elif resolved_notches.size:
        widths = _resolved_notch_widths(resolved_notches, notch_widths)
        transition_half_width = 0.5
        lows = resolved_notches - widths / 2.0 - transition_half_width
        highs = resolved_notches + widths / 2.0 + transition_half_width
        notch_kernel = mne.filter.create_filter(
            probe,
            sfreq,
            highs,
            lows,
            l_trans_bandwidth=transition_half_width,
            h_trans_bandwidth=transition_half_width,
            phase="zero",
            fir_design="firwin",
            verbose="ERROR",
        )
        kernel_lengths["notch"] = int(np.asarray(notch_kernel).size)

    radius_samples = (
        int(sum((length - 1) // 2 for length in kernel_lengths.values())) + model_radius
    )
    return {
        "kernel_lengths": kernel_lengths,
        "model_support_radius_samples": model_radius,
        "support_radius_samples": radius_samples,
        "support_radius_sec": float(radius_samples / sfreq),
    }


def _filter_valid_segment(
    data: np.ndarray,
    *,
    sfreq: float,
    l_freq: float | None,
    h_freq: float | None,
    notches: Sequence[float] | None,
    notch_widths: Union[float, Sequence[float]],
    significance_thresholds: Sequence[float] | None = None,
    notch_model: dict[str, Any] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Apply the detection-order filters to one independent valid segment."""
    filtered = np.asarray(data, dtype=float).copy()
    if l_freq is not None or h_freq is not None:
        filtered = mne.filter.filter_data(
            filtered,
            sfreq,
            l_freq,
            h_freq,
            phase="zero",
            fir_design="firwin",
            verbose="ERROR",
        )
    resolved_notches = (
        np.asarray(tuple(notches), dtype=float)
        if notches is not None
        else np.empty(0, dtype=float)
    )
    if resolved_notches.size and notch_model and notch_model["enabled"]:
        from .notch import subtract_notch_model

        filtered = subtract_notch_model(
            filtered,
            sfreq,
            resolved_notches,
            notch_model,
            significance_thresholds=significance_thresholds,
            diagnostics=diagnostics,
            background_bounds=(l_freq, h_freq),
        )
    elif resolved_notches.size:
        filtered = mne.filter.notch_filter(
            filtered,
            sfreq,
            resolved_notches,
            notch_widths=notch_widths,
            phase="zero",
            fir_design="firwin",
            verbose="ERROR",
        )
    return np.asarray(filtered, dtype=float)


def _mask_annotation_rows(
    mask: np.ndarray,
    *,
    sfreq: float,
    first_samp: int,
    scope: tuple[str, ...],
) -> list[tuple[float, float, str, tuple[str, ...]]]:
    """Convert a sample mask into half-open annotation rows."""
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return []
    changes = np.diff(values.astype(np.int8))
    starts = list(np.where(changes == 1)[0] + 1)
    stops = list(np.where(changes == -1)[0] + 1)
    if values[0]:
        starts.insert(0, 0)
    if values[-1]:
        stops.append(int(values.size))
    return [
        (
            float((start + first_samp) / sfreq),
            float((stop - start) / sfreq),
            FILTER_EDGE_DESCRIPTION,
            scope,
        )
        for start, stop in zip(starts, stops)
        if stop > start
    ]


def finalize_reviewed_lfp_filter(
    raw: mne.io.BaseRaw,
    *,
    reviewed_annotations: mne.Annotations,
    reviewed_bads: Sequence[str],
    l_freq: float | None,
    h_freq: float | None,
    notches: Sequence[float] | None,
    notch_widths: Union[float, Sequence[float]],
    isolate_bad_boundaries: bool = True,
    mark_filter_edges: bool = False,
    notch_model: dict[str, Any] | None = None,
    n_jobs: int | None = None,
) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    """Build the accepted Filter result from original data and reviewed BAD/EDGE.

    With ``isolate_bad_boundaries=True`` every global/channel-specific valid
    interval is filtered independently, so reviewed BAD values cannot enter an
    adjacent filter input. ``mark_filter_edges`` independently controls whether
    the sequential FIR support (or model-window support bound) at each endpoint is marked
    as `EDGE_filter`. With isolation disabled, the reviewed Raw is filtered
    continuously exactly like MNE whole-Raw filtering. ``n_jobs`` limits isolated
    CleanLine channel workers; None selects up to four CPUs and 1 runs serially.
    """
    if n_jobs is not None and (
        isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs < 1
    ):
        raise ValueError("n_jobs must be None or a positive integer.")
    if mark_filter_edges and not isolate_bad_boundaries:
        raise ValueError(
            "mark_filter_edges requires isolate_bad_boundaries to be true."
        )
    if notch_model is not None:
        from .notch import (
            normalize_notch_model,
            effective_notch_model,
            model_notch_frequencies,
        )

        notch_model = effective_notch_model(normalize_notch_model(notch_model))
        notches = model_notch_frequencies(notches, notch_model)
    from .notch import cleanline_channel_thresholds

    thresholds = cleanline_channel_thresholds(notch_model, raw.ch_names)
    out = raw.copy()
    out.load_data()
    reviewed = _without_filter_edges(reviewed_annotations)
    _set_annotations_from_attached_frame(out, reviewed)
    out.info["bads"] = [name for name in reviewed_bads if name in out.ch_names]

    sfreq = float(out.info["sfreq"])
    support = _filter_support_geometry(
        n_times=out.n_times,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        notches=notches,
        notch_widths=notch_widths,
        notch_model=notch_model,
    )
    radius = int(support["support_radius_samples"])
    data = out.get_data()
    edge_masks: list[np.ndarray] = []
    segment_counts: dict[str, int] = {}
    adaptive_reports: list[dict[str, Any]] = []
    bad_reference_reports: list[dict[str, Any]] = []

    if not isolate_bad_boundaries:
        segment_reports = []
        data = _filter_valid_segment(
            data,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            notches=notches,
            notch_widths=notch_widths,
            notch_model=notch_model,
            significance_thresholds=list(thresholds.values()) if thresholds else None,
            diagnostics=segment_reports,
        )
        adaptive_reports.extend(
            {
                **entry,
                "channel": out.ch_names[entry["channel_index"]],
                "segment_start_sample": 0,
                "segment_stop_sample": out.n_times,
            }
            for entry in segment_reports
        )
        segment_counts = {channel: 1 for channel in out.ch_names}
    else:
        from joblib import Parallel, delayed, cpu_count, parallel_config

        def filter_channel(channel_index, channel, channel_data, segments, invalid):
            channel_data = channel_data.copy()
            channel_reports = []
            reference_reports = []
            if invalid.any():
                reference = _filter_valid_segment(
                    channel_data,
                    sfreq=sfreq,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    notches=notches,
                    notch_widths=notch_widths,
                    notch_model=notch_model,
                    significance_thresholds=(
                        [thresholds[channel]] if thresholds else None
                    ),
                    diagnostics=reference_reports,
                )
                channel_data[invalid] = reference[invalid]
                reference_reports = [
                    {
                        **entry,
                        "channel_index": channel_index,
                        "channel": channel,
                        "segment_start_sample": 0,
                        "segment_stop_sample": len(channel_data),
                    }
                    for entry in reference_reports
                ]
            channel_edges = np.zeros(channel_data.size, dtype=bool)
            for start, stop in segments:
                length = stop - start
                segment_reports = []
                channel_data[start:stop] = _filter_valid_segment(
                    channel_data[start:stop],
                    sfreq=sfreq,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    notches=notches,
                    notch_widths=notch_widths,
                    notch_model=notch_model,
                    significance_thresholds=(
                        [thresholds[channel]] if thresholds else None
                    ),
                    diagnostics=segment_reports,
                )
                channel_reports.extend(
                    {
                        **entry,
                        "channel_index": channel_index,
                        "channel": channel,
                        "segment_start_sample": int(start),
                        "segment_stop_sample": int(stop),
                    }
                    for entry in segment_reports
                )
                if mark_filter_edges and radius > 0:
                    if length <= 2 * radius:
                        channel_edges[start:stop] = True
                    else:
                        channel_edges[start : start + radius] = True
                        channel_edges[stop - radius : stop] = True
            return channel_data, channel_reports, channel_edges, reference_reports

        tasks = []
        for channel_index, channel in enumerate(out.ch_names):
            invalid, point_boundaries = _filter_boundary_support(out, channel=channel)
            segments = _valid_filter_segments(invalid, point_boundaries)
            segment_counts[channel] = len(segments)
            tasks.append(
                (channel_index, channel, data[channel_index], segments, invalid)
            )
        workers = 1
        if (
            notch_model
            and notch_model["enabled"]
            and notch_model["method"] == "cleanline"
        ):
            workers = min(len(tasks), n_jobs or min(4, cpu_count()))
        if workers > 1:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=workers)(
                    delayed(filter_channel)(*task) for task in tasks
                )
        else:
            results = [filter_channel(*task) for task in tasks]
        for channel_index, (
            channel_data,
            reports,
            channel_edges,
            reference_reports,
        ) in enumerate(results):
            data[channel_index] = channel_data
            adaptive_reports.extend(reports)
            bad_reference_reports.extend(reference_reports)
            if mark_filter_edges:
                edge_masks.append(channel_edges)

    out._data[...] = data
    annotation_rows: list[tuple[float, float, str, tuple[str, ...]]] = []
    if edge_masks:
        common_edges = np.logical_and.reduce(edge_masks)
        annotation_rows.extend(
            _mask_annotation_rows(
                common_edges,
                sfreq=sfreq,
                first_samp=int(out.first_samp),
                scope=(),
            )
        )
        for channel, channel_edges in zip(out.ch_names, edge_masks):
            annotation_rows.extend(
                _mask_annotation_rows(
                    channel_edges & ~common_edges,
                    sfreq=sfreq,
                    first_samp=int(out.first_samp),
                    scope=(channel,),
                )
            )

    combined = out.annotations.copy()
    if annotation_rows:
        edge_annotations = mne.Annotations(
            onset=[row[0] for row in annotation_rows],
            duration=[row[1] for row in annotation_rows],
            description=[row[2] for row in annotation_rows],
            orig_time=combined.orig_time,
            ch_names=[row[3] for row in annotation_rows],
        )
        combined = combined + edge_annotations
        order = np.argsort(np.asarray(combined.onset, dtype=float))
        combined = mne.Annotations(
            onset=np.asarray(combined.onset, dtype=float)[order].tolist(),
            duration=np.asarray(combined.duration, dtype=float)[order].tolist(),
            description=np.asarray(combined.description, dtype=object)[order].tolist(),
            orig_time=combined.orig_time,
            ch_names=[combined.ch_names[index] for index in order],
        )
    _set_annotations_from_attached_frame(out, combined)

    _append_info_description(
        out,
        (
            "reviewed_segment_filter: "
            if isolate_bad_boundaries
            else "reviewed_continuous_filter: "
        )
        + f"{l_freq}-{h_freq} Hz; notches={list(notches or [])}; "
        f"support_radius_sec={support['support_radius_sec']}; "
        f"mark_filter_edges={bool(mark_filter_edges)}",
    )
    report = {
        **support,
        "isolate_bad_boundaries": bool(isolate_bad_boundaries),
        "mark_filter_edges": bool(mark_filter_edges),
        "edge_description": FILTER_EDGE_DESCRIPTION if mark_filter_edges else None,
        "n_edge_annotations": len(annotation_rows),
        "segments_by_channel": segment_counts,
        "filter_order": ["bandpass", "notch"],
        "bad_samples_policy": (
            "filtered_reference" if isolate_bad_boundaries else "continuous"
        ),
    }
    if adaptive_reports:
        report["cleanline_adaptive"] = adaptive_reports
    if bad_reference_reports:
        report["bad_reference"] = bad_reference_reports
    return out, report


def merge_contiguous_bad_annotations(
    raw: mne.io.BaseRaw,
    gap: float = 0.5,
    bad_tag: Union[str, Iterable[str]] = "BAD",
    *,
    match_mode: MatchMode = "exact",
    case_sensitive: bool = False,
    merged_description: Optional[str] = None,
    keep_original_bad: bool = False,
    merged_first: bool = True,
    sort_by_onset: bool = False,
) -> mne.io.BaseRaw:
    """
    Merge overlapping / near-contiguous "bad" annotations in an MNE Raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw instance carrying raw.annotations.
    gap : float
        Merge two bad intervals if the gap between them is <= gap (seconds).
    bad_tag : str or iterable of str
        Tag(s) used to identify bad annotations.
    match_mode : {"substring", "exact"}
        How to match bad_tag(s) against annotation descriptions.
        - "substring": tag in description (optionally case-insensitive)
        - "exact": description == tag (optionally case-insensitive)
    case_sensitive : bool
        Whether matching is case-sensitive.
    merged_description : str or None
        Description used for merged intervals. If None:
        - if bad_tag is str: f"{bad_tag}_merged"
        - if iterable: f"{first_tag}_merged"
    keep_original_bad : bool
        If True, keep original bad annotations and append merged intervals.
        If False (default), remove original bad annotations and keep only merged ones.
    merged_first : bool
        If True, output annotations as merged + other. Otherwise other + merged.
        Ignored if sort_by_onset=True.
    sort_by_onset : bool
        If True, sort output annotations by onset time.

    Returns
    -------
    out : mne.io.BaseRaw
        A copy of raw with merged bad annotations.
    """
    tags: Tuple[str, ...] = (bad_tag,) if isinstance(bad_tag, str) else tuple(bad_tag)
    if len(tags) == 0:
        return raw

    def _norm(s: str) -> str:
        return s if case_sensitive else s.upper()

    tags_norm = tuple(_norm(t) for t in tags)

    if match_mode not in ("substring", "exact"):
        raise ValueError(
            f"match_mode must be 'substring' or 'exact', got: {match_mode}"
        )

    def is_bad_desc(desc: str) -> bool:
        d = _norm(str(desc))
        if match_mode == "substring":
            return any(t in d for t in tags_norm)
        else:  # exact
            return any(d == t for t in tags_norm)

    ann = raw.annotations
    if ann is None or len(ann) == 0:
        return raw

    # Collect bad spans separately for each MNE annotation channel scope.
    spans_by_scope: Dict[tuple[str, ...], List[tuple[float, float]]] = {}
    for onset, duration, desc, ch_names in zip(
        ann.onset,
        ann.duration,
        ann.description,
        ann.ch_names,
    ):
        if is_bad_desc(desc):
            start = float(onset)
            end = float(onset) + float(duration)
            scope = tuple(str(name) for name in ch_names)
            spans_by_scope.setdefault(scope, []).append((start, end))

    if not spans_by_scope:
        return raw

    # Greedy union with tolerance gap, without merging different scopes.
    merged: List[tuple[float, float, tuple[str, ...]]] = []
    for scope, spans in spans_by_scope.items():
        spans.sort(key=lambda item: item[0])
        cur_s, cur_e = spans[0]
        for start, end in spans[1:]:
            if start <= cur_e + float(gap):
                cur_e = max(cur_e, end)
            else:
                merged.append((cur_s, cur_e, scope))
                cur_s, cur_e = start, end
        merged.append((cur_s, cur_e, scope))
    merged.sort(key=lambda item: (item[0], item[1], item[2]))

    # Default merged description compatible with your original version
    if merged_description is None:
        base = tags[0]
        merged_description = f"{base}_merged"

    merged_ann = mne.Annotations(
        onset=[start for start, _end, _scope in merged],
        duration=[end - start for start, end, _scope in merged],
        description=[merged_description] * len(merged),
        orig_time=ann.orig_time,
        ch_names=[scope for _start, _end, scope in merged],
    )

    # Build "other" annotations
    if keep_original_bad:
        other_ann = ann.copy()
    else:
        keep_mask = np.array([not is_bad_desc(d) for d in ann.description], dtype=bool)
        other_ann = (
            ann[keep_mask]
            if np.any(keep_mask)
            else mne.Annotations([], [], [], orig_time=ann.orig_time)
        )

    # Combine
    if sort_by_onset:
        combined = merged_ann + other_ann
        order = np.argsort(np.asarray(combined.onset, dtype=float))
        combined_sorted = mne.Annotations(
            onset=np.asarray(combined.onset, dtype=float)[order],
            duration=np.asarray(combined.duration, dtype=float)[order],
            description=np.asarray(combined.description, dtype=object)[order].tolist(),
            orig_time=combined.orig_time,
            ch_names=[combined.ch_names[index] for index in order],
        )
        out = raw.copy()
        _set_annotations_from_attached_frame(out, combined_sorted)
        return out

    combined = (merged_ann + other_ann) if merged_first else (other_ann + merged_ann)
    out = raw.copy()
    _set_annotations_from_attached_frame(out, combined)
    return out


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BadAnnotationConfig:
    """
    Configuration for marking BAD segments.
    """

    l_freq: Optional[float] = 1.0
    h_freq: Optional[float] = 200.0
    epoch_dur: float = 1.0
    overlap: float = 0.0
    p2p_thresh: Optional[Tuple[float, float]] = (
        1e-6,
        1e-3,
    )  # (min, max) in Volts; None disables fixed P2P rejection
    autoreject_correct_factor: float = 1.5
    notches: Optional[Sequence[float]] = None
    notch_widths: Union[float, Sequence[float]] = 1.0
    notch_model: dict[str, Any] | None = None
    isolate_bad_boundaries: bool = True
    min_good_len_sec: float = 3.0
    merge_gap_sec: float = 0.5

    # Annotation labels
    bad_prefix: str = "BAD"
    desc_p2p: str = "BAD"
    desc_autoreject: str = "BAD"
    desc_gap: str = "BAD"
    merged_bad_desc: str = "BAD"
    match_mode: MatchMode = "exact"

    # Autoreject params
    random_state: int = 42
    autoreject_method: str = "random_search"
    verbose: bool = True


# -----------------------------------------------------------------------------
# Filtering utilities (standalone)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# 1) BAD annotation function
# -----------------------------------------------------------------------------


def mark_lfp_bad_segments(
    raw: mne.io.BaseRaw,
    cfg: BadAnnotationConfig,
    *,
    eeg_like_channels: Optional[Sequence[str]] = None,
    reject_plot_path: Optional[Union[str, Path]] = None,
) -> Tuple[mne.io.BaseRaw, Optional[Any], Dict[str, Any]]:
    """
    Mark BAD segments on a copy of raw using:
      (1) fixed-length epochs + hard p2p range threshold
      (2) autoreject-like per-channel thresholds (compute_thresholds) on remaining epochs
      (3) mark short "good gaps" as BAD_gap
      (4) merge all BAD_* into merged 'BAD' blocks

    Returns
    -------
    raw_marked:
        A filtered copy of raw containing BAD annotations.
        Use its annotations to label the original raw (recommended).
    reject_log:
        autoreject.RejectLog if autoreject succeeded, else None.
    summary:
        dict of counts and parameters.
    """
    if cfg.p2p_thresh is not None and cfg.p2p_thresh[0] >= cfg.p2p_thresh[1]:
        raise ValueError(f"Invalid p2p_thresh: {cfg.p2p_thresh}. Expected (min, max).")

    # Import autoreject lazily to keep module import lightweight
    _ensure_mne_exports_for_autoreject()

    try:
        from autoreject import RejectLog, compute_thresholds
    except Exception as e:
        raise ImportError(
            "This function requires 'autoreject' to be installed (compute_thresholds, RejectLog)."
        ) from e

    raw_mark = raw.copy()
    raw_mark.load_data()

    sfreq = float(raw_mark.info["sfreq"])
    epoch_n_times = int(np.round(sfreq * float(cfg.epoch_dur)))
    if raw_mark.n_times < epoch_n_times:
        raise ValueError(
            "Recording duration is shorter than Filter epoch duration; "
            "reduce Epoch duration."
        )
    from .notch import (
        normalize_notch_model,
        effective_notch_model,
        model_notch_frequencies,
    )

    active_model = effective_notch_model(normalize_notch_model(cfg.notch_model))
    notches = np.asarray(
        model_notch_frequencies(cfg.notches, active_model), dtype=float
    )
    notch_widths = cfg.notch_widths

    requested_channels = (
        list(raw_mark.ch_names)
        if eeg_like_channels is None
        else [str(channel) for channel in eeg_like_channels]
    )
    requested_set = set(requested_channels)
    listed_bad_set = set(raw_mark.info["bads"])
    detection_channels = [
        channel
        for channel in raw_mark.ch_names
        if channel in requested_set and channel not in listed_bad_set
    ]
    excluded_bad_channels = [
        channel
        for channel in raw_mark.ch_names
        if channel in requested_set and channel in listed_bad_set
    ]
    if not detection_channels:
        raise ValueError(
            "No usable channels remain for Filter artifact detection after "
            "excluding raw.info['bads']."
        )

    # 1) Temporarily set channel types to EEG for autoreject picks='eeg'
    original_types = _get_channel_types_map(raw_mark)
    eeg_type_map = {channel: "eeg" for channel in detection_channels}
    if eeg_type_map:
        raw_mark.set_channel_types(eeg_type_map)

    # 2) Filter for detection
    use_model = bool(active_model["enabled"] and notches is not None and notches.size)
    model_report = {}
    if use_model or cfg.isolate_bad_boundaries:
        raw_mark, model_report = finalize_reviewed_lfp_filter(
            raw_mark,
            reviewed_annotations=raw_mark.annotations,
            reviewed_bads=raw_mark.info["bads"],
            l_freq=cfg.l_freq,
            h_freq=cfg.h_freq,
            notches=notches,
            notch_widths=notch_widths,
            notch_model=active_model,
            isolate_bad_boundaries=cfg.isolate_bad_boundaries,
        )
    elif cfg.l_freq is not None or cfg.h_freq is not None:
        raw_mark.filter(
            l_freq=cfg.l_freq,
            h_freq=cfg.h_freq,
            fir_design="firwin",
            phase="zero",
        )
    if (
        (notches is not None)
        and (notches.size > 0)
        and not use_model
        and not cfg.isolate_bad_boundaries
    ):
        raw_mark.notch_filter(freqs=notches, notch_widths=notch_widths)

    # 3) Fixed-length epochs
    epochs_regular = mne.make_fixed_length_epochs(
        raw_mark,
        duration=cfg.epoch_dur,
        overlap=cfg.overlap,
        preload=True,
        reject_by_annotation=False,
    )

    n_regular_epochs = len(epochs_regular)
    epoch_n_times = int(epochs_regular.get_data(copy=False).shape[-1])
    last_regular_start = int(epochs_regular.events[-1, 0]) - int(raw_mark.first_samp)
    tail_epoch_added = last_regular_start + epoch_n_times < raw_mark.n_times
    if tail_epoch_added:
        tail_event = np.array(
            [
                [
                    int(raw_mark.first_samp) + raw_mark.n_times - epoch_n_times,
                    0,
                    int(epochs_regular.events[-1, 2]),
                ]
            ],
            dtype=int,
        )
        evaluation_events = np.vstack([epochs_regular.events, tail_event])
        epochs_evaluation = mne.Epochs(
            raw_mark,
            evaluation_events,
            event_id=epochs_regular.event_id,
            tmin=0.0,
            tmax=(epoch_n_times - 1) / sfreq,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            proj=True,
        )
    else:
        epochs_evaluation = epochs_regular
    epochs_detection = epochs_evaluation.copy().pick(detection_channels)

    n_epochs = len(epochs_evaluation)
    win_len = float(cfg.epoch_dur)

    # 4) Hard p2p threshold (range)
    data = epochs_detection.get_data()  # (n_epochs, n_detection_ch, n_times)
    p2p = np.ptp(data, axis=2)
    if cfg.p2p_thresh is None:
        bad_p2p = np.zeros(n_epochs, dtype=bool)
    else:
        bad_p2p = ((p2p < cfg.p2p_thresh[0]) | (p2p > cfg.p2p_thresh[1])).any(axis=1)

    evaluation_keep_idx = np.where(~bad_p2p)[0]
    training_keep_idx = np.where(~bad_p2p[:n_regular_epochs])[0]

    # Annotate BAD_p2p
    on_p2p = epochs_evaluation.events[bad_p2p, 0] / sfreq
    ann_p2p = mne.Annotations(
        onset=on_p2p.tolist(),
        duration=[win_len] * int(bad_p2p.sum()),
        description=[cfg.desc_p2p] * int(bad_p2p.sum()),
        orig_time=raw_mark.annotations.orig_time,
    )
    _set_annotations_from_attached_frame(raw_mark, raw_mark.annotations + ann_p2p)

    # 5) AutoReject thresholds on remaining epochs
    reject_log = None
    n_bad_ar = 0
    autoreject_error: str | None = None
    autoreject_plot_error: str | None = None

    epochs_training = epochs_detection[training_keep_idx]
    if len(epochs_training) >= 1:
        try:
            threshes = compute_thresholds(
                epochs_training,
                picks="eeg",
                method=cfg.autoreject_method,
                random_state=cfg.random_state,
                augment=False,
                verbose=cfg.verbose,
            )
            threshes = {
                ch: float(t) * float(cfg.autoreject_correct_factor)
                for ch, t in threshes.items()
            }

            epochs_threshold_evaluation = epochs_detection[evaluation_keep_idx]
            p2p_thr = np.ptp(
                epochs_threshold_evaluation.get_data(), axis=2
            )  # (n_epochs_remain, n_ch)
            thr_vec = np.array(
                [threshes[ch] for ch in epochs_threshold_evaluation.ch_names],
                dtype=float,
            )

            bad_epochs_mask = (p2p_thr > thr_vec).any(axis=1)
            labels_int = (p2p_thr > thr_vec).astype(int)

            reject_log = RejectLog(
                bad_epochs=bad_epochs_mask,
                labels=labels_int,
                ch_names=epochs_threshold_evaluation.ch_names,
            )

            # Map back to original epoch indices
            bad_ar_orig = evaluation_keep_idx[bad_epochs_mask]
            n_bad_ar = int(np.sum(bad_epochs_mask))

            # Annotate BAD_autoreject
            on_ar = epochs_evaluation.events[bad_ar_orig, 0] / sfreq
            ann_ar = mne.Annotations(
                onset=on_ar.tolist(),
                duration=[win_len] * len(on_ar),
                description=[cfg.desc_autoreject] * len(on_ar),
                orig_time=raw_mark.annotations.orig_time,
            )
            _set_annotations_from_attached_frame(
                raw_mark, raw_mark.annotations + ann_ar
            )

            if reject_plot_path is not None:
                try:
                    _safe_makedirs(reject_plot_path)
                    _save_reject_log_plot_agg(reject_log, reject_plot_path)
                except Exception as plot_exc:
                    autoreject_plot_error = f"{type(plot_exc).__name__}: {plot_exc}"
                    _append_info_description(
                        raw_mark,
                        f"autoreject_plot_failed: {autoreject_plot_error}",
                    )

        except Exception as e:
            # If autoreject fails, proceed with p2p-only annotations
            reject_log = None
            n_bad_ar = 0
            autoreject_error = f"{type(e).__name__}: {e}"
            _append_info_description(raw_mark, f"autoreject_failed: {autoreject_error}")

    # 6) Mark short good gaps as BAD_gap (only based on BAD* annotations)
    bad_mask = _build_bad_sample_mask(raw_mark, bad_prefixes=(cfg.bad_prefix,))
    good_indices = np.flatnonzero(~bad_mask)

    if good_indices.size > 0:
        breaks = np.where(np.diff(good_indices) > 1)[0]
        runs = np.split(good_indices, breaks + 1)

        min_samps = int(round(float(cfg.min_good_len_sec) * sfreq))
        gap_onsets: List[float] = []
        gap_durs: List[float] = []

        sample_shift = int(raw_mark.first_samp)
        for run in runs:
            run_len = int(run.size)
            if 0 < run_len < min_samps:
                # Convert relative sample index -> absolute onset(sec) consistent with other annotations
                onset_abs = (int(run[0]) + sample_shift) / sfreq
                dur_sec = run_len / sfreq
                gap_onsets.append(float(onset_abs))
                gap_durs.append(float(dur_sec))

        if gap_onsets:
            ann_gap = mne.Annotations(
                onset=gap_onsets,
                duration=gap_durs,
                description=[cfg.desc_gap] * len(gap_onsets),
                orig_time=raw_mark.annotations.orig_time,
            )
            _set_annotations_from_attached_frame(
                raw_mark,
                raw_mark.annotations + ann_gap,
            )

    # 7) Restore channel types
    _restore_channel_types(raw_mark, original_types)

    # 8) Merge BAD* into 'BAD' blocks (replace originals)
    raw_mark = merge_contiguous_bad_annotations(
        raw_mark,
        gap=cfg.merge_gap_sec,
        bad_tag=(cfg.bad_prefix,),
        match_mode=cfg.match_mode,
        case_sensitive=False,
        merged_description=cfg.merged_bad_desc,
        keep_original_bad=False,
        merged_first=True,
        sort_by_onset=True,
    )

    summary = {
        "sfreq": sfreq,
        "l_freq": cfg.l_freq,
        "h_freq": cfg.h_freq,
        "epoch_dur": cfg.epoch_dur,
        "overlap": cfg.overlap,
        "p2p_thresh": cfg.p2p_thresh,
        "n_epochs": int(n_epochs),
        "n_regular_epochs": int(n_regular_epochs),
        "n_evaluation_epochs": int(n_epochs),
        "tail_epoch_added": bool(tail_epoch_added),
        "n_bad_p2p": int(np.sum(bad_p2p)),
        "n_bad_autoreject": int(n_bad_ar),
        "detection_channels": list(detection_channels),
        "excluded_bad_channels": list(excluded_bad_channels),
        "notches": notches.tolist() if notches is not None else None,
        "notch_widths": notch_widths,
        "min_good_len_sec": cfg.min_good_len_sec,
        "merge_gap_sec": cfg.merge_gap_sec,
    }
    if "cleanline_adaptive" in model_report:
        summary["cleanline_adaptive"] = model_report["cleanline_adaptive"]
    summary["bad_samples_policy"] = model_report.get("bad_samples_policy", "continuous")
    if "bad_reference" in model_report:
        summary["bad_reference"] = model_report["bad_reference"]
    if autoreject_error is not None:
        summary["autoreject_error"] = autoreject_error
    if autoreject_plot_error is not None:
        summary["autoreject_plot_error"] = autoreject_plot_error
    return raw_mark, reject_log, summary


# -----------------------------------------------------------------------------
# 2) Filtering + extract-good function
# -----------------------------------------------------------------------------
def filter_lfp_with_bad_annotations(
    raw: mne.io.BaseRaw,
    bad_annotations: Optional[mne.Annotations] = None,
    *,
    bad_descs: Union[str, Sequence[str]] = ("BAD",),
    # -------------------- Filtering params --------------------
    l_freq: float = 1.0,
    h_freq: float = 200.0,
    do_pre_filter: bool = False,
    pre_filter_kwargs: Optional[Dict[str, Any]] = None,
    # Optional: notch on continuous data BEFORE extraction (reduces join transients)
    do_pre_notch: bool = False,
    notches: Optional[Sequence[float]] = None,
    notch_widths: Union[float, Sequence[float]] = 1.0,
    # Post filtering (applied after extraction on compressed timeline)
    do_post_notch: bool = False,
    post_notches: Optional[Sequence[float]] = None,
    post_notch_widths: Union[float, Sequence[float]] = 1.0,
    do_post_filter: bool = False,
    post_filter_kwargs: Optional[Dict[str, Any]] = None,
    # -------------------- Annotation handling --------------------
    overlap_policy: str = "split",  # "split", "drop", or "compress"
    match_mode: BoundaryMatchMode = "exact",
    case_sensitive: bool = False,
    # -------------------- ADDED: report + concat markers --------------------
    verbose: bool = True,
    add_concat_annotations: bool = True,
    concat_desc: str = "EDGE",
    concat_duration: float = 0.0,
    # ----------------------------------------------------------
) -> Tuple[mne.io.RawArray, Dict[str, Any]]:
    """
    End-to-end helper to:
      1) attach BAD annotations onto raw (optional)
      2) apply pre-filtering on the continuous recording (optional)
      3) remove globally scoped BAD samples and rebuild a compressed RawArray
         (no raw.crop used); channel-specific BAD annotations remain scoped on
         the shared timeline for downstream masks
      4) add annotations at concatenation boundaries (optional)
      5) apply notch and band-pass after extraction (optional)

    Returns
    -------
    raw_good : mne.io.RawArray
        Compressed raw containing only non-BAD samples with re-mapped annotations.
    report : dict
        Summary including percent removed and concat boundary info.
    """
    # Validate args
    if overlap_policy not in ("split", "drop", "compress"):
        raise ValueError(
            "overlap_policy must be 'split', 'drop', or 'compress', "
            f"got: {overlap_policy}"
        )
    if match_mode not in ("substring", "exact", "prefix"):
        raise ValueError(
            "match_mode must be 'substring', 'exact', or 'prefix', "
            f"got: {match_mode}"
        )

    patterns: Tuple[str, ...] = (
        (bad_descs,) if isinstance(bad_descs, str) else tuple(bad_descs)
    )
    if len(patterns) == 0:
        raise ValueError(
            "bad_descs cannot be empty; provide at least one BAD identifier."
        )

    def _norm(s: str) -> str:
        return s if case_sensitive else s.upper()

    patterns_norm = tuple(_norm(p) for p in patterns)

    def is_bad_desc(desc: str) -> bool:
        d = _norm(str(desc))
        if match_mode == "substring":
            return any(p in d for p in patterns_norm)
        if match_mode == "prefix":
            return any(d.startswith(p) for p in patterns_norm)
        return any(d == p for p in patterns_norm)

    # Work on a copy
    raw_labeled = raw.copy()
    raw_labeled.load_data()

    # 0) Attach BAD annotations (optional)
    if bad_annotations is not None and len(bad_annotations) > 0:
        target_orig_time = raw_labeled.annotations.orig_time
        filtered_bad_onsets: List[float] = []
        filtered_bad_durs: List[float] = []
        filtered_bad_descs: List[str] = []
        filtered_bad_ch_names: List[tuple[str, ...]] = []
        for onset, duration, desc, ch_names in zip(
            bad_annotations.onset,
            bad_annotations.duration,
            bad_annotations.description,
            bad_annotations.ch_names,
        ):
            if not is_bad_desc(str(desc)):
                continue
            filtered_bad_onsets.append(float(onset))
            filtered_bad_durs.append(float(duration))
            filtered_bad_descs.append(str(desc))
            filtered_bad_ch_names.append(tuple(str(name) for name in ch_names))
        if filtered_bad_descs:
            bad_annotations = mne.Annotations(
                onset=filtered_bad_onsets,
                duration=filtered_bad_durs,
                description=filtered_bad_descs,
                orig_time=target_orig_time,
                ch_names=filtered_bad_ch_names,
            )
            raw_labeled.set_annotations(bad_annotations + raw_labeled.annotations)

    sfreq = float(raw_labeled.info["sfreq"])
    n_times = int(raw_labeled.n_times)

    # Defaults mirroring your pipelines
    if pre_filter_kwargs is None:
        pre_filter_kwargs = dict(
            l_trans_bandwidth="auto",
            h_trans_bandwidth=4,
            phase="zero",
            fir_design="firwin",
        )
    if post_filter_kwargs is None:
        post_filter_kwargs = dict(
            l_trans_bandwidth="auto",
            h_trans_bandwidth=4,
            phase="zero",
            fir_design="firwin",
        )

    # 1) Pre-filter on continuous signal (recommended to reduce edge artifacts after cutting)
    if do_pre_filter:
        raw_labeled.filter(l_freq=l_freq, h_freq=h_freq, **pre_filter_kwargs)
        _append_info_description(raw_labeled, f"pre_filter: {l_freq}-{h_freq} Hz")

    # Optional pre-notch (continuous) to reduce join transients later
    # If notches is None, no notch filtering is applied.
    if do_pre_notch and notches is not None:
        resolved_notches = np.asarray(notches, dtype=float)
        if resolved_notches.size > 0:
            raw_labeled.notch_filter(freqs=resolved_notches, notch_widths=notch_widths)
            _append_info_description(
                raw_labeled, f"pre_notch: {resolved_notches.tolist()} Hz"
            )
    ann = raw_labeled.annotations
    if ann is None:
        ann = mne.Annotations([], [], [], orig_time=None)

    # 2) Build bad sample mask (sample-accurate; consistent with your first_samp logic)
    bad_mask = np.zeros(n_times, dtype=bool)
    sample_shift = int(getattr(raw_labeled, "first_samp", 0))
    relative_onsets = raw_relative_onsets(raw_labeled)

    for onset_relative, duration, desc, ch_names in zip(
        relative_onsets,
        ann.duration,
        ann.description,
        ann.ch_names,
    ):
        if not is_bad_desc(desc):
            continue
        if tuple(ch_names):
            continue
        start_samp, stop_samp = raw_labeled.time_as_index(
            [
                float(onset_relative),
                float(onset_relative) + float(duration),
            ],
            use_rounding=True,
        )
        start_samp = max(start_samp, 0)
        stop_samp = min(stop_samp, n_times)
        if stop_samp > start_samp:
            bad_mask[start_samp:stop_samp] = True

    good_mask = ~bad_mask

    # -------------------- ADDED: crop percent report --------------------
    good_samples = int(np.sum(good_mask))
    removed_samples = int(n_times - good_samples)
    removed_pct = (removed_samples / n_times * 100.0) if n_times > 0 else 0.0
    # ----------------------------------------------------------

    # 3) Find contiguous good segments (start, end_exclusive)
    good_int = good_mask.astype(np.int8)
    changes = np.diff(good_int)

    starts = list(np.where(changes == 1)[0] + 1)
    ends = list(np.where(changes == -1)[0] + 1)
    if good_mask[0]:
        starts = [0] + starts
    if good_mask[-1]:
        ends = ends + [n_times]

    segments = list(zip(starts, ends))
    if len(segments) == 0:
        raise RuntimeError("No good samples remain after removing BAD annotations.")

    # 4) Concatenate good data into a single RawArray
    data_parts: List[np.ndarray] = []
    seg_new_starts: List[int] = []
    cum = 0
    for s, e in segments:
        seg_new_starts.append(cum)
        data_parts.append(raw_labeled.get_data(start=s, stop=e))
        cum += e - s

    new_data = np.concatenate(data_parts, axis=1)

    # 5) Re-map NON-BAD annotations onto the compressed timeline
    new_onsets: List[float] = []
    new_durs: List[float] = []
    new_descs: List[str] = []
    new_ch_names: List[tuple[str, ...]] = []

    def find_containing_segment(sample_idx: int) -> int:
        for i, (s, e) in enumerate(segments):
            if s <= sample_idx < e:
                return i
        return -1

    for onset, duration, desc, ch_names in zip(
        ann.onset,
        ann.duration,
        ann.description,
        ann.ch_names,
    ):
        annotation_scope = tuple(str(name) for name in ch_names)
        if is_bad_desc(desc) and not annotation_scope:
            continue

        a_start = int(round(float(onset) * sfreq)) - sample_shift
        a_end = int(round((float(onset) + float(duration)) * sfreq)) - sample_shift
        a_start = max(a_start, 0)
        a_end = min(a_end, n_times)

        # Handle point events (duration == 0)
        if a_end <= a_start:
            seg_i = find_containing_segment(a_start)
            if seg_i < 0:
                continue
            s, _ = segments[seg_i]
            new_start_samp = seg_new_starts[seg_i] + (a_start - s)
            new_onsets.append(new_start_samp / sfreq)
            new_durs.append(0.0)
            new_descs.append(str(desc))
            new_ch_names.append(annotation_scope)
            continue

        if overlap_policy == "drop":
            kept = False
            for seg_i, (s, e) in enumerate(segments):
                if (s <= a_start) and (a_end <= e):
                    new_start_samp = seg_new_starts[seg_i] + (a_start - s)
                    new_onsets.append(new_start_samp / sfreq)
                    new_durs.append((a_end - a_start) / sfreq)
                    new_descs.append(str(desc))
                    new_ch_names.append(annotation_scope)
                    kept = True
                    break
            if not kept:
                pass  # overlaps BAD -> drop whole annotation

        else:
            mapped_chunks: List[tuple[int, int]] = []
            for seg_i, (s, e) in enumerate(segments):
                ov_s = max(s, a_start)
                ov_e = min(e, a_end)
                if ov_e <= ov_s:
                    continue
                new_start_samp = seg_new_starts[seg_i] + (ov_s - s)
                mapped_chunks.append((new_start_samp, ov_e - ov_s))

            if overlap_policy == "split":
                for chunk_start, chunk_len in mapped_chunks:
                    new_onsets.append(chunk_start / sfreq)
                    new_durs.append(chunk_len / sfreq)
                    new_descs.append(str(desc))
                    new_ch_names.append(annotation_scope)
            else:  # overlap_policy == "compress"
                if not mapped_chunks:
                    continue
                merged_start = mapped_chunks[0][0]
                merged_len = sum(chunk_len for _, chunk_len in mapped_chunks)
                new_onsets.append(merged_start / sfreq)
                new_durs.append(merged_len / sfreq)
                new_descs.append(str(desc))
                new_ch_names.append(annotation_scope)

    # 6) Build RawArray (compressed timeline is not absolute time)
    # Note: In recent MNE versions, info['meas_date'] cannot be set directly.
    # Use Raw.set_meas_date(None) instead to clear absolute timing.
    info = raw_labeled.info.copy()
    raw_good = mne.io.RawArray(new_data, info)
    raw_good.set_meas_date(None)

    # Set re-mapped non-BAD annotations
    if len(new_onsets) > 0:
        raw_good.set_annotations(
            mne.Annotations(
                new_onsets,
                new_durs,
                new_descs,
                orig_time=None,
                ch_names=new_ch_names,
            )
        )
    else:
        raw_good.set_annotations(mne.Annotations([], [], [], orig_time=None))

    # -------------------- ADDED: concat boundary annotations --------------------
    concat_onsets_sec: List[float] = (
        []
    )  # boundary times (centers) in the compressed timeline
    concat_ann_onsets_sec: List[float] = []  # actual annotation start times
    concat_ann_durations_sec: List[float] = (
        []
    )  # actual annotation durations (may be clipped)
    if add_concat_annotations and len(segments) > 1:
        if not np.isfinite(float(concat_duration)):
            raise ValueError(f"concat_duration must be finite, got: {concat_duration}")
        if float(concat_duration) < 0:
            raise ValueError(f"concat_duration must be >= 0, got: {concat_duration}")

        # Total length of the compressed timeline (seconds)
        total_length_sec = float(raw_good.n_times) / sfreq
        half = float(concat_duration) / 2.0

        # Mark the boundary between segments and build a symmetric window around each boundary.
        # Desired window (in seconds):
        #   [clip(t_concat - concat_duration/2, 0, total_length),
        #    clip(t_concat + concat_duration/2, 0, total_length)]
        for start_samp in seg_new_starts[1:]:
            t_concat = float(start_samp) / sfreq
            concat_onsets_sec.append(t_concat)

            t0 = float(np.clip(t_concat - half, 0.0, total_length_sec))
            t1 = float(np.clip(t_concat + half, 0.0, total_length_sec))
            if t1 < t0:
                t0, t1 = t1, t0

            concat_ann_onsets_sec.append(t0)
            concat_ann_durations_sec.append(t1 - t0)

        if concat_ann_onsets_sec:
            concat_ann = mne.Annotations(
                onset=concat_ann_onsets_sec,
                duration=concat_ann_durations_sec,
                description=[str(concat_desc)] * len(concat_ann_onsets_sec),
                orig_time=None,
            )
            raw_good.set_annotations(raw_good.annotations + concat_ann)

    # ----------------------------------------------------------

    # 7) Post notch + band-pass (optional)

    # If notches is None, no notch filtering is applied.
    if do_post_notch and post_notches is not None:
        resolved_notches = np.asarray(post_notches, dtype=float)
        if resolved_notches.size > 0:
            raw_good.notch_filter(
                freqs=resolved_notches, notch_widths=post_notch_widths
            )
            _append_info_description(
                raw_good,
                f"post_notch: {resolved_notches.tolist()} Hz; width={post_notch_widths}",
            )
    if do_post_filter:
        raw_good.filter(l_freq=l_freq, h_freq=h_freq, **post_filter_kwargs)
        _append_info_description(raw_good, f"post_filter: {l_freq}-{h_freq} Hz")

    # 8) Report
    report: Dict[str, Any] = {
        "sfreq": sfreq,
        "orig_n_times": n_times,
        "good_n_times": int(raw_good.n_times),
        "removed_samples": removed_samples,
        "removed_pct": float(removed_pct),
        "orig_duration_sec": n_times / sfreq,
        "good_duration_sec": float(raw_good.n_times) / sfreq,
        "n_segments": len(segments),
        "n_concat_points": max(0, len(segments) - 1),
        "concat_onsets_sec": concat_onsets_sec,
        "overlap_policy": overlap_policy,
        "match_mode": match_mode,
    }

    if verbose:
        print(
            f"[filter_lfp_with_bad_annotations] Removed {removed_samples}/{n_times} samples "
            f"({removed_pct:.2f}%). Segments: {len(segments)}; Concat points: {report['n_concat_points']}."
        )

    return raw_good, report


def add_head_tail_annotations(
    raw: mne.io.BaseRaw,
    *,
    head_duration_sec: float = 0.0,
    tail_duration_sec: float = 0.0,
    description: str = "EDGE",
    head_description: str | None = None,
    tail_description: str | None = None,
    replace_existing_same_desc: bool = False,
    sort_by_onset: bool = True,
    copy: bool = True,
) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Add annotations at the beginning and end of an MNE Raw object.

    This helper is designed to mark edge regions (e.g., filter transients) so that
    downstream code can avoid them by excluding a shared label such as "EDGE".

    The function adds up to two annotations:
      - a head annotation covering [start, start + head_duration_sec]
      - a tail annotation covering [end - tail_duration_sec, end]

    Durations are clipped to the recording bounds.

    Notes on 0-second durations:
      - If head_duration_sec == 0, we still add a 0-duration annotation at recording start.
      - If tail_duration_sec == 0, we still add a 0-duration annotation at the last sample time
        (to ensure it lies within the data range).

    The annotation time reference frame follows existing `raw.annotations.orig_time`:
      - If orig_time is None, MNE re-adds `raw.first_samp / sfreq` inside
        `set_annotations`, so head/tail onsets are built in the raw-relative frame and
        existing onsets (which already carry that offset) are converted back before
        they are merged. Mixing the two frames would push existing annotations outside
        the crop window applied by `set_annotations` and silently drop them.
      - If orig_time is not None, onsets are absolute seconds from orig_time, aligned
        using raw.first_samp, and no conversion is needed.

    Returns
    -------
    raw_out : mne.io.BaseRaw
        Raw with appended head/tail annotations.
    report : dict
        Summary including onsets/durations and reference-frame details. All onset-like
        report fields are expressed in the frame of the returned
        `raw_out.annotations.onset`, so they can be compared against the saved file
        directly; `first_time_sec` is the offset that converts them back to the
        raw-relative frame.
    """
    raw_out = raw.copy() if copy else raw

    if not np.isfinite(float(head_duration_sec)) or float(head_duration_sec) < 0:
        raise ValueError(
            f"head_duration_sec must be finite and >= 0, got: {head_duration_sec}"
        )

    tail_val = head_duration_sec if tail_duration_sec is None else tail_duration_sec
    if not np.isfinite(float(tail_val)) or float(tail_val) < 0:
        raise ValueError(f"tail_duration_sec must be finite and >= 0, got: {tail_val}")

    ann = raw_out.annotations
    if ann is None:
        ann = mne.Annotations([], [], [], orig_time=None)

    sfreq = float(raw_out.info["sfreq"])
    n_times = int(raw_out.n_times)
    if n_times <= 0:
        raise ValueError("raw has no samples (n_times <= 0).")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid sfreq: {sfreq}")

    total_len_sec = float(n_times) / sfreq
    orig_time = ann.orig_time
    first_time_sec = float(raw_out.first_samp) / sfreq
    n_annotations_in = int(len(ann))

    # Determine recording bounds in the annotation reference frame.
    # `report_offset` converts that frame into the frame of the annotations that
    # `set_annotations` will finally expose on raw_out.
    if orig_time is None:
        # set_annotations() re-adds first_time for orig_time=None, so build the new
        # onsets raw-relative and pull the existing ones back into the same frame.
        rec_start_sec = 0.0
        rec_end_sec = total_len_sec
        last_sample_sec = max(rec_start_sec, rec_end_sec - 1.0 / sfreq)
        report_offset = first_time_sec
        ann = mne.Annotations(
            onset=np.asarray(ann.onset, dtype=float) - first_time_sec,
            duration=np.asarray(ann.duration, dtype=float),
            description=np.asarray(ann.description, dtype=object),
            orig_time=None,
            ch_names=list(ann.ch_names),
        )
    else:
        rec_start_sec = first_time_sec
        rec_end_sec = float(raw_out.first_samp + n_times) / sfreq  # exclusive end
        last_sample_sec = float(raw_out.first_samp + n_times - 1) / sfreq
        report_offset = 0.0

    head_desc = str(description) if head_description is None else str(head_description)
    tail_desc = str(description) if tail_description is None else str(tail_description)

    new_onsets: List[float] = []
    new_durs: List[float] = []
    new_descs: List[str] = []

    head_d = float(min(float(head_duration_sec), rec_end_sec - rec_start_sec))
    tail_d = float(min(float(tail_val), rec_end_sec - rec_start_sec))

    # Head window: add even if head_d == 0
    head_start = rec_start_sec
    if head_d == 0.0:
        new_onsets.append(float(head_start))
        new_durs.append(0.0)
        new_descs.append(head_desc)
    else:
        head_end = min(rec_end_sec, rec_start_sec + head_d)
        new_onsets.append(float(head_start))
        new_durs.append(float(head_end - head_start))
        new_descs.append(head_desc)

    # Tail window: add even if tail_d == 0
    if tail_d == 0.0:
        # Put the point marker at the last sample to stay within the data range.
        new_onsets.append(float(last_sample_sec))
        new_durs.append(0.0)
        new_descs.append(tail_desc)
    else:
        tail_end = rec_end_sec
        tail_start = max(rec_start_sec, rec_end_sec - tail_d)
        new_onsets.append(float(tail_start))
        new_durs.append(float(tail_end - tail_start))
        new_descs.append(tail_desc)

    existing = ann.copy()
    if replace_existing_same_desc and len(existing) > 0:
        remove_set = {str(description), head_desc, tail_desc}
        keep_mask = np.array(
            [str(d) not in remove_set for d in existing.description], dtype=bool
        )
        existing = (
            existing[keep_mask]
            if np.any(keep_mask)
            else mne.Annotations([], [], [], orig_time=orig_time)
        )

    edge_ann = mne.Annotations(
        onset=new_onsets,
        duration=new_durs,
        description=new_descs,
        orig_time=orig_time,
    )
    combined = existing + edge_ann

    if sort_by_onset and len(combined) > 1:
        order = np.argsort(np.asarray(combined.onset, dtype=float))
        combined = mne.Annotations(
            onset=np.asarray(combined.onset, dtype=float)[order].tolist(),
            duration=np.asarray(combined.duration, dtype=float)[order].tolist(),
            description=np.asarray(combined.description, dtype=object)[order].tolist(),
            orig_time=combined.orig_time,
            ch_names=[combined.ch_names[index] for index in order],
        )

    raw_out.set_annotations(combined)

    report: Dict[str, Any] = {
        "description": str(description),
        "head_duration_sec": float(head_duration_sec),
        "tail_duration_sec": float(tail_val),
        "head_description": head_desc,
        "tail_description": tail_desc,
        "n_added": int(len(new_onsets)),
        "added_onsets_sec": [float(x) + report_offset for x in new_onsets],
        "added_durations_sec": [float(x) for x in new_durs],
        "orig_time_is_none": bool(orig_time is None),
        "first_time_sec": float(first_time_sec),
        "recording_start_sec": float(rec_start_sec) + report_offset,
        "recording_end_sec": float(rec_end_sec) + report_offset,
        "last_sample_sec": float(last_sample_sec) + report_offset,
        "total_duration_sec": float(total_len_sec),
        "n_annotations_in": n_annotations_in,
        "n_annotations_out": int(len(raw_out.annotations)),
    }

    return raw_out, report
