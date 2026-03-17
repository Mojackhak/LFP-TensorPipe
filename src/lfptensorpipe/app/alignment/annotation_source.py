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
    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
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
    if not selected_description:
        raise ValueError("No annotations remain after duration-range filtering.")
    filtered = raw.copy()
    filtered.set_annotations(
        mne.Annotations(
            onset=np.asarray(selected_onset, dtype=float),
            duration=np.asarray(selected_duration, dtype=float),
            description=np.asarray(selected_description, dtype=object),
            orig_time=annotations.orig_time,
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
