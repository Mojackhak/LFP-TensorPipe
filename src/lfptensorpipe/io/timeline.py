"""Raw timeline helpers shared by the import, re-reference, and sync paths.

MNE keeps `raw.annotations.onset` in a frame that already carries
`raw.first_samp / sfreq`. Any code that rebuilds a Raw (which resets `first_samp`
to 0) must therefore pull the onsets back into the record-relative frame first,
otherwise `set_annotations` crops them away and the annotations are lost without
an error. `raw_relative_onsets` is that conversion; `normalize_raw_timeline`
applies it together with a `meas_date` compensation so absolute time is
preserved while `first_samp` becomes 0.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np


def raw_relative_onsets(raw: Any) -> np.ndarray:
    """Return `raw.annotations.onset` converted to the record-relative frame.

    The returned onsets are measured from the first retained sample, i.e. the
    frame a freshly built `RawArray` (with `first_samp == 0`) expects.
    """
    annotations = getattr(raw, "annotations", None)
    if annotations is None or len(annotations) == 0:
        return np.zeros(0, dtype=float)
    sfreq = float(raw.info["sfreq"])
    first_time = float(raw.first_samp) / sfreq if sfreq > 0 else 0.0
    return np.asarray(annotations.onset, dtype=float) - first_time


def normalize_raw_timeline(raw: Any) -> tuple[Any, dict[str, Any]]:
    """Rebase one raw onto `first_samp == 0` without moving anything in time.

    Absolute time is preserved: `meas_date` is pushed forward by
    `first_samp / sfreq` and every annotation onset is pulled back by the same
    amount, so `meas_date + onset` is invariant. Sample data is untouched.

    Returns the raw unchanged (same object) when `first_samp` is already 0.
    """
    import mne

    sfreq = float(raw.info["sfreq"])
    first_samp = int(getattr(raw, "first_samp", 0) or 0)
    if first_samp == 0 or sfreq <= 0:
        return raw, {"normalized": False}

    first_time = float(first_samp) / sfreq
    meas_date_before = raw.info.get("meas_date")
    annotations = getattr(raw, "annotations", None)
    n_before = int(len(annotations)) if annotations is not None else 0

    info = raw.info.copy()
    out = mne.io.RawArray(
        np.asarray(raw.get_data(), dtype=float), info, verbose="ERROR"
    )

    meas_date_after = meas_date_before
    if meas_date_before is not None:
        meas_date_after = meas_date_before + timedelta(seconds=first_time)
        out.set_meas_date(meas_date_after)

    if n_before > 0:
        out.set_annotations(
            mne.Annotations(
                onset=raw_relative_onsets(raw),
                duration=np.asarray(annotations.duration, dtype=float),
                description=np.asarray(annotations.description, dtype=object),
                orig_time=meas_date_after,
                ch_names=list(annotations.ch_names),
            )
        )

    report = {
        "normalized": True,
        "first_samp_before": first_samp,
        "first_time_sec": first_time,
        "meas_date_before": meas_date_before,
        "meas_date_after": meas_date_after,
        "n_annotations_before": n_before,
        "n_annotations_after": int(len(out.annotations)),
    }
    return out, report


def format_timeline_normalization(report: dict[str, Any] | None) -> str:
    """Return a one-line human summary, or an empty string when it was a no-op."""
    if not isinstance(report, dict) or not report.get("normalized"):
        return ""
    return (
        f"Normalized timeline: first_samp {int(report['first_samp_before'])} -> 0 "
        f"(shifted meas_date by {float(report['first_time_sec']):.6f} s)"
    )


__all__ = [
    "format_timeline_normalization",
    "normalize_raw_timeline",
    "raw_relative_onsets",
]
