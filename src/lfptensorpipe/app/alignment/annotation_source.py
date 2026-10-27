"""Alignment raw/annotation source helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import preproc_step_raw_path


def _load_unique_annotation_labels(
    raw_path: Path,
    *,
    read_raw_fif_fn: Any | None = None,
) -> list[str]:
    try:
        if read_raw_fif_fn is None:
            import mne

            read_raw_fif_fn = mne.io.read_raw_fif
        raw = read_raw_fif_fn(str(raw_path), preload=False, verbose="ERROR")
        labels = sorted(
            set(str(item) for item in raw.annotations.description if str(item).strip())
        )
        if hasattr(raw, "close"):
            raw.close()
        return labels
    except Exception:
        return []


def load_alignment_annotation_labels(context: RecordContext) -> list[str]:
    resolver = PathResolver(context)
    finish_raw = preproc_step_raw_path(resolver, "finish")
    if not finish_raw.exists():
        return []
    return _load_unique_annotation_labels(finish_raw)


def load_alignment_annotation_events(
    context: RecordContext,
) -> dict[str, tuple[float, ...]]:
    """Read Finish annotation identities and onsets without loading samples."""
    import mne

    path = preproc_step_raw_path(PathResolver(context), "finish")
    if not path.exists():
        return {}
    raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
    try:
        events: dict[str, list[float]] = {}
        for label, onset in zip(raw.annotations.description, raw.annotations.onset):
            events.setdefault(str(label), []).append(float(onset))
        return {label: tuple(onsets) for label, onsets in events.items()}
    finally:
        raw.close()


def cue_response_annotation_error(
    config: dict[str, Any], events: dict[str, tuple[float, ...]]
) -> str | None:
    """Check normalized pairs against the current record's annotation identities."""
    for pair in config["event_pairs"]:
        cue = events.get(pair["cue_label"], ())
        response = events.get(pair["response_label"], ())
        if len(cue) != 1 or len(response) != 1:
            return (
                "cue_response.event_pairs requires exactly one annotation for each "
                f"label: {pair['cue_label']} / {pair['response_label']}."
            )
        if response[0] <= cue[0]:
            return (
                "cue_response.event_pairs requires the response after its cue: "
                f"{pair['cue_label']} / {pair['response_label']}."
            )
    return None


def _load_raw_for_warp(raw_path: Path):
    import mne

    return mne.io.read_raw_fif(str(raw_path), preload=False, verbose="ERROR")


def _float_pair_list(
    value: Any,
    default: tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    try:
        left = float(value[0]) if value[0] is not None else None
        right = float(value[1]) if value[1] is not None else None
        return left, right
    except Exception:
        return default


def _filter_raw_annotations_by_duration(
    raw: Any,
    *,
    keep: list[str],
    duration_range: tuple[float | None, float | None],
):
    import mne

    min_duration, max_duration = duration_range
    annotations = raw.annotations
    selected_onset: list[float] = []
    selected_duration: list[float] = []
    selected_description: list[str] = []
    selected_ch_names: list[tuple[str, ...]] = []
    for onset, duration, description, ch_names in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
        annotations.ch_names,
        strict=False,
    ):
        label = str(description).strip()
        if label not in keep:
            continue
        duration_f = float(duration)
        if min_duration is not None and duration_f < min_duration:
            continue
        if max_duration is not None and duration_f > max_duration:
            continue
        selected_onset.append(float(onset))
        selected_duration.append(duration_f)
        selected_description.append(label)
        selected_ch_names.append(tuple(str(name) for name in ch_names))
    if not selected_description:
        raise ValueError("No annotations remain after duration-range filtering.")
    filtered = raw.copy()
    filtered.set_annotations(
        mne.Annotations(
            onset=np.asarray(selected_onset, dtype=float),
            duration=np.asarray(selected_duration, dtype=float),
            description=np.asarray(selected_description, dtype=object),
            orig_time=annotations.orig_time,
            ch_names=selected_ch_names,
        )
    )
    return filtered


__all__ = [
    "_filter_raw_annotations_by_duration",
    "_float_pair_list",
    "_load_raw_for_warp",
    "_load_unique_annotation_labels",
    "load_alignment_annotation_labels",
]
