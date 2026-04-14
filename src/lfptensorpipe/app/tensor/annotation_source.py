"""Burst baseline-annotation source helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import preproc_step_raw_path


def _load_positive_duration_annotation_labels(
    raw_path: Path,
    *,
    read_raw_fif_fn: Any | None = None,
) -> list[str]:
    raw = None
    try:
        if read_raw_fif_fn is None:
            import mne

            read_raw_fif_fn = mne.io.read_raw_fif
        raw = read_raw_fif_fn(str(raw_path), preload=False, verbose="ERROR")
        labels: set[str] = set()
        annotations = raw.annotations
        for duration, description in zip(
            annotations.duration,
            annotations.description,
            strict=False,
        ):
            label = str(description).strip()
            if not label:
                continue
            try:
                duration_value = float(duration)
            except Exception:
                continue
            if duration_value <= 0.0:
                continue
            labels.add(label)
        return sorted(labels)
    except Exception:
        return []
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()


def load_burst_baseline_annotation_labels(
    context: RecordContext,
    *,
    read_raw_fif_fn: Any | None = None,
) -> list[str]:
    resolver = PathResolver(context)
    finish_raw = preproc_step_raw_path(resolver, "finish")
    if not finish_raw.exists():
        return []
    return _load_positive_duration_annotation_labels(
        finish_raw,
        read_raw_fif_fn=read_raw_fif_fn,
    )


__all__ = [
    "_load_positive_duration_annotation_labels",
    "load_burst_baseline_annotation_labels",
]
