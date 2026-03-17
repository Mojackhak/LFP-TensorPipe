"""Build metadata for warped/cropped tensors.

The goal is to standardize axis naming for tensors after a warp/crop step
(e.g., gait cycle normalization or pad/concat around task annotations).

Conventions used across the LFP pipeline:

- Unwarped tensors (pre-warp) should use:
    axes = {'epoch', 'channel', 'freq', 'time'}

- Warped tensors (post-warp) extend this with:
    axes = {'annotation', 'epoch', 'channel', 'freq', 'time', 'percent'}

Notes
-----
- ``axes['freq']`` may be numeric (Hz) **or** a list of strings (e.g., band names,
  SpecParam parameter names). Downstream code must not assume it is float.
- For directed connectivity, ``axes['channel']`` typically stores channel pairs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def _epoch_label(ep: Any, fallback: str) -> str:
    """Extract a label for an epoch from warp metadata."""
    if hasattr(ep, "label"):
        return str(getattr(ep, "label"))
    if hasattr(ep, "side"):
        return str(getattr(ep, "side"))
    return fallback


def _infer_epoch_duration_s(ep: Any) -> Tuple[str, float]:
    """Infer the *warped* epoch duration in seconds from an epoch meta object.

    Returns:
        kind: One of {'gait', 'pad', 'concat', 'unknown'}.
        duration_s: Best-effort duration in seconds (may be NaN).
    """
    # Concat epoch: explicitly stores total concatenated duration.
    if hasattr(ep, "total_duration_s"):
        try:
            return "concat", float(getattr(ep, "total_duration_s"))
        except Exception:
            return "concat", float(np.nan)

    # Gait epoch: has side + start/end timestamps.
    if hasattr(ep, "side") and hasattr(ep, "start_t") and hasattr(ep, "end_t"):
        try:
            dur = float(getattr(ep, "end_t")) - float(getattr(ep, "start_t"))
            return "gait", dur
        except Exception:
            return "gait", float(np.nan)

    # Pad epoch: duration is defined by the pad config (preferred) or inferred from events.
    if hasattr(ep, "nominal_duration_s"):
        try:
            return "pad", float(getattr(ep, "nominal_duration_s"))
        except Exception:
            return "pad", float(np.nan)

    events_t = getattr(ep, "events_t", None)
    if isinstance(events_t, dict) and {
        "pad_left",
        "anno_left",
        "anno_right",
        "pad_right",
    }.issubset(events_t.keys()):
        try:
            left = float(events_t["anno_left"]) - float(events_t["pad_left"])
            right = float(events_t["pad_right"]) - float(events_t["anno_right"])
            return "pad", left + right
        except Exception:
            return "pad", float(np.nan)

    return "unknown", float(np.nan)


def _robust_duration_s(meta_epochs: Sequence[Any]) -> Tuple[str, float, np.ndarray]:
    """Compute a robust representative duration for the warped time axis."""
    kinds = []
    durs = []
    for ep in meta_epochs:
        kind, dur = _infer_epoch_duration_s(ep)
        kinds.append(kind)
        durs.append(dur)
    kinds_arr = np.asarray(kinds, dtype=object)
    durs_arr = np.asarray(durs, dtype=float)

    valid = np.isfinite(durs_arr) & (durs_arr > 0)
    if not np.any(valid):
        return ("unknown", float(np.nan), durs_arr)

    # Prefer the most common kind among valid epochs.
    kinds_valid = kinds_arr[valid]
    uniq, counts = np.unique(kinds_valid, return_counts=True)
    kind_major = str(uniq[int(np.argmax(counts))])

    if kind_major == "gait":
        dur = float(np.mean(durs_arr[valid]))
    else:
        dur = float(np.median(durs_arr[valid]))

    return (kind_major, dur, durs_arr)


def build_warped_tensor_metadata(
    base_axes: Dict[str, Any] | None,
    percent_axis: np.ndarray,
    meta_epochs: Sequence[Any],
    *,
    ch_names: Optional[Sequence[Any]] = None,
    freqs: Optional[Sequence[float]] = None,
    source_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a metadata dict for warped tensor outputs.

    The function is intentionally generic: `meta_epochs` can be a list of
    objects from either:
      - the event-anchored (gait) piecewise-linear warper, or
      - the pad+concat warper.

    Args:
        base_axes: Optional base axes metadata (e.g., from `tfr_grid()['metadata']['axes']`).
        percent_axis: 0..100 normalized time axis returned by the warp function.
        meta_epochs: A list of epoch metadata objects returned by the warp function.
        ch_names: Optional channel/pair labels overriding base axes.
        freqs: Optional frequency axis overriding base axes.
        source_meta: Optional dict to embed as the "source" (unwarped) metadata.

    Returns:
        A dict with:
            - axes: epoch/channel/freq/time/percent/shape
            - source: embedded unwarped meta (optional)
            - warp_epochs: list of per-epoch metadata dicts (safe JSON types)
    """
    axes_base = base_axes or {}

    # Explicit args override base metadata if provided.
    freq_axis = freqs if freqs is not None else axes_base.get("freqs", axes_base.get("freq", None))
    chan_axis = ch_names if ch_names is not None else axes_base.get("channels", axes_base.get("channel", None))

    # Infer shape from inputs.
    n_epochs = len(meta_epochs)
    n_channels = len(chan_axis) if chan_axis is not None else None
    n_freqs = len(freq_axis) if freq_axis is not None else None
    n_times = len(percent_axis) if percent_axis is not None else None
    shape = (n_epochs, n_channels, n_freqs, n_times)

    kind_major, duration_s, durations_per_epoch = _robust_duration_s(meta_epochs)

    if n_times is None:
        time_axis = None
    elif n_times <= 1:
        time_axis = np.array([0.0], dtype=float)
    elif np.isfinite(duration_s) and duration_s > 0:
        # "Warped time" axis in seconds (0..duration_s).
        time_axis = np.linspace(0.0, duration_s, int(n_times), endpoint=True, dtype=float)
    else:
        # Fallback: preserve a unit interval so downstream code always has a `time` axis.
        time_axis = np.linspace(0.0, 1.0, int(n_times), endpoint=True, dtype=float)

    epoch_labels = np.array(
        [_epoch_label(ep, fallback=f"epoch_{i:04d}") for i, ep in enumerate(meta_epochs)],
        dtype=object,
    )

    # Serialize per-epoch metadata with a conservative schema.
    warp_events = []
    for i, ep in enumerate(meta_epochs):
        events_t = getattr(ep, "events_t", None)
        kind, dur_s = _infer_epoch_duration_s(ep)
        item = {
            "epoch_index": int(i),
            "label": _epoch_label(ep, fallback=f"epoch_{i:04d}"),
            "start_t": float(getattr(ep, "start_t", np.nan)),
            "end_t": float(getattr(ep, "end_t", np.nan)),
            "events_t": events_t if isinstance(events_t, dict) else None,
            "kind": kind,
            "duration_s": float(dur_s) if np.isfinite(dur_s) else np.nan,
        }
        if hasattr(ep, "side"):
            item["side"] = str(getattr(ep, "side"))
        if hasattr(ep, "intervals_s"):
            item["intervals_s"] = list(getattr(ep, "intervals_s"))
        warp_events.append(item)

    metadata: Dict[str, Any] = dict(
        axes=dict(
            epoch=epoch_labels,
            channel=np.fromiter(chan_axis, dtype=object, count=len(chan_axis)) if chan_axis is not None else None,
            freq=(
                list(freq_axis)
                if isinstance(freq_axis, list)
                else (
                    np.asarray(freq_axis, dtype=float)
                    if (freq_axis is not None and np.issubdtype(np.asarray(freq_axis).dtype, np.number))
                    else (np.asarray(freq_axis, dtype=object) if freq_axis is not None else None)
                )
            ),            
            time=time_axis,
            percent=np.array(percent_axis, dtype=float),
            shape=shape,
            kind=str(kind_major),
            duration_s=float(duration_s) if np.isfinite(duration_s) else np.nan,
            duration_s_per_epoch=durations_per_epoch,
        ),
        source=source_meta,
        warp_epochs=warp_events,
    )
    return metadata
