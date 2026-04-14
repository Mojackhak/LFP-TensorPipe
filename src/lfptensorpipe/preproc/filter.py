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

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import mne
from ..lfp.mask.annotations import MatchMode

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _append_info_description(raw: mne.io.BaseRaw, text: str) -> None:
    """Append a preprocessing note to raw.info['description'] without overwriting."""
    old = raw.info.get("description", "") or ""
    raw.info["description"] = (old + " | " + text).strip(" | ")


def _startswith_any(s: str, prefixes: Sequence[str]) -> bool:
    return any(s.startswith(p) for p in prefixes)


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


def _save_reject_log_plot_agg(reject_log: Any, reject_plot_path: Union[str, Path]) -> None:
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
) -> np.ndarray:
    """
    Build a boolean mask of bad samples (length = raw.n_times) from annotations.

    IMPORTANT:
        Only annotations with description starting with any of bad_prefixes are treated as bad.
        This prevents accidental masking of task/event annotations (gait/pain).
    """
    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    bad_mask = np.zeros(n_times, dtype=bool)

    ann = raw.annotations
    if ann is None or len(ann) == 0:
        return bad_mask

    sample_shift = int(raw.first_samp)

    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        desc = str(desc)
        if not _startswith_any(desc, bad_prefixes):
            continue

        # Convert absolute onset(sec) -> sample index relative to raw data array
        start_samp = int(round(float(onset) * sfreq)) - sample_shift
        stop_samp = int(round((float(onset) + float(duration)) * sfreq)) - sample_shift

        # Clip to valid data range
        start_samp = max(start_samp, 0)
        stop_samp = min(stop_samp, n_times)

        if stop_samp > start_samp:
            bad_mask[start_samp:stop_samp] = True

    return bad_mask


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
        raise ValueError(f"match_mode must be 'substring' or 'exact', got: {match_mode}")

    def is_bad_desc(desc: str) -> bool:
        d = _norm(str(desc))
        if match_mode == "substring":
            return any(t in d for t in tags_norm)
        else:  # exact
            return any(d == t for t in tags_norm)

    ann = raw.annotations
    if ann is None or len(ann) == 0:
        return raw

    # Collect bad spans: (start, end)
    spans = []
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        if is_bad_desc(desc):
            start = float(onset)
            end = float(onset) + float(duration)
            spans.append((start, end))

    if not spans:
        return raw

    spans.sort(key=lambda x: x[0])

    # Greedy union with tolerance gap
    merged = []
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e + float(gap):
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Default merged description compatible with your original version
    if merged_description is None:
        base = tags[0]
        merged_description = f"{base}_merged"

    merged_ann = mne.Annotations(
        onset=[s for s, _ in merged],
        duration=[e - s for s, e in merged],
        description=[merged_description] * len(merged),
        orig_time=ann.orig_time,
    )

    # Build "other" annotations
    if keep_original_bad:
        other_ann = ann.copy()
    else:
        keep_mask = np.array([not is_bad_desc(d) for d in ann.description], dtype=bool)
        other_ann = ann[keep_mask] if np.any(keep_mask) else mne.Annotations([], [], [], orig_time=ann.orig_time)

    # Combine
    if sort_by_onset:
        combined = merged_ann + other_ann
        order = np.argsort(np.asarray(combined.onset, dtype=float))
        combined_sorted = mne.Annotations(
            onset=np.asarray(combined.onset, dtype=float)[order],
            duration=np.asarray(combined.duration, dtype=float)[order],
            description=np.asarray(combined.description, dtype=object)[order].tolist(),
            orig_time=combined.orig_time,
        )
        out = raw.copy()
        out.set_annotations(combined_sorted)
        return out

    combined = (merged_ann + other_ann) if merged_first else (other_ann + merged_ann)
    out = raw.copy()
    out.set_annotations(combined)
    return out


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BadAnnotationConfig:
    """
    Configuration for marking BAD segments.
    """
    l_freq: float = 1.0
    h_freq: float = 200.0
    epoch_dur: float = 1.0
    overlap: float = 0.0
    p2p_thresh: Tuple[float, float] = (1e-6, 1e-3)  # (min, max) in Volts
    autoreject_correct_factor: float = 1.5
    notches: Optional[Sequence[float]] = None
    notch_widths: Union[float, Sequence[float]] = 1.0
    min_good_len_sec: float = 3.0
    merge_gap_sec: float = 0.5

    # Annotation labels
    bad_prefix: str = "BAD"
    desc_p2p: str = "BAD"
    desc_autoreject: str = "BAD"
    desc_gap: str = "BAD"
    merged_bad_desc: str = "BAD"
    match_mode: MatchMode = 'exact'

    # Autoreject params
    random_state: int = 42
    autoreject_method: str = "random_search"
    verbose: bool = True


@dataclass(frozen=True)
class LfpFilterConfig:
    """
    Configuration for final filtering + extracting good segments.
    """
    l_freq: float = 1.0
    h_freq: float = 200.0
    notches: Optional[Sequence[float]] = None
    notch_widths: Union[float, Sequence[float]] = 1.0

    # Filter kwargs (match your scripts defaults, but configurable)
    pre_filter_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        l_trans_bandwidth="auto",
        h_trans_bandwidth=4,
        phase="zero",
        fir_design="firwin",
    ))
    post_filter_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        l_trans_bandwidth="auto",
        h_trans_bandwidth=4,
        phase="zero",
        fir_design="firwin",
    ))

    # Behavior switches
    do_pre_filter_before_extract: bool = True
    do_post_notch: bool = True
    do_post_filter: bool = True

    # Which annotations count as "bad"
    bad_prefixes: Tuple[str, ...] = ("BAD",)

    # Optional: use your existing helper for extraction if available
    prefer_project_helper: bool = True



# -----------------------------------------------------------------------------
# Filtering utilities (standalone)
# -----------------------------------------------------------------------------

def apply_lfp_filter_config(
    raw: mne.io.BaseRaw,
    cfg: LfpFilterConfig,
    *,
    copy: bool = True,
    picks: Optional[Union[str, Sequence[str]]] = None,
    append_description: bool = True,
    description_prefix: str = "LFP_filter",
    use_post_filter_kwargs: bool = True,
) -> mne.io.BaseRaw:
    """
    Apply notch filtering (optional) and band-pass filtering (optional) to an MNE
    Raw object using an :class:`LfpFilterConfig` instance.

    This helper mirrors the common pattern used throughout your preprocessing
    scripts: start from a clean/"good" Raw, apply notch_filter() to suppress
    line-related components, then apply band-pass filtering to standardize the
    final frequency range. The applied steps are recorded in raw.info['description']
    via an appended text entry, preserving any existing history.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Input Raw object. Data must be preloadable (this function calls load_data()).
    cfg : LfpFilterConfig
        Filtering configuration (notches may be None).
    copy : bool
        If True, operate on a copy and return it. If False, modify raw in-place.
    picks : str | sequence of str | None
        Channels to filter. If None, all channels are processed.
    append_description : bool
        If True (default), append a human-readable summary to info['description'].
        If False, overwrite info['description'] with the summary for this step.
    description_prefix : str
        Prefix used in the description entry.
    use_post_filter_kwargs : bool
        If True, use cfg.post_filter_kwargs; otherwise use cfg.pre_filter_kwargs.

    Returns
    -------
    out : mne.io.BaseRaw
        Filtered Raw object.
    """
    out = raw.copy() if copy else raw
    out.load_data()

    steps: List[str] = []

    # 1) Notch filter (explicit-only: cfg.notches must be provided)
    if cfg.do_post_notch and (cfg.notches is not None):
        freqs = np.asarray(cfg.notches, dtype=float)
        if freqs.size > 0:
            out.notch_filter(freqs=freqs, notch_widths=cfg.notch_widths, picks=picks)
            steps.append(
                f"notch={freqs.tolist()} Hz; notch_widths={cfg.notch_widths}"
            )

    # 2) Band-pass filter
    if cfg.do_post_filter:
        fkwargs = cfg.post_filter_kwargs if use_post_filter_kwargs else cfg.pre_filter_kwargs
        out.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks=picks, **fkwargs)

        # Keep the description compact: only include the most informative kwargs
        key_order = ("l_trans_bandwidth", "h_trans_bandwidth", "phase", "fir_design")
        kv = [f"{k}={fkwargs[k]}" for k in key_order if k in fkwargs]
        kw_text = ", ".join(kv)
        if kw_text:
            steps.append(f"bandpass={cfg.l_freq}-{cfg.h_freq} Hz; {kw_text}")
        else:
            steps.append(f"bandpass={cfg.l_freq}-{cfg.h_freq} Hz")

    # 3) Record description
    if steps:
        info_text = f"{description_prefix}: " + " | ".join(steps)
        if append_description:
            _append_info_description(out, info_text)
        else:
            out.info["description"] = info_text

    return out

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
    if cfg.p2p_thresh[0] >= cfg.p2p_thresh[1]:
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
    notches = np.asarray(cfg.notches, dtype=float) if cfg.notches is not None else None
    notch_widths = cfg.notch_widths

    # 1) Temporarily set channel types to EEG for autoreject picks='eeg'
    original_types = _get_channel_types_map(raw_mark)
    if eeg_like_channels is None:
        eeg_like_channels = list(raw_mark.ch_names)
    eeg_type_map = {ch: "eeg" for ch in eeg_like_channels if ch in raw_mark.ch_names}
    if eeg_type_map:
        raw_mark.set_channel_types(eeg_type_map)

    # 2) Filter for detection
    raw_mark.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, fir_design="firwin", phase="zero")
    if (notches is not None) and (notches.size > 0):
        raw_mark.notch_filter(freqs=notches, notch_widths=notch_widths)

    # 3) Fixed-length epochs
    epochs_all = mne.make_fixed_length_epochs(
        raw_mark,
        duration=cfg.epoch_dur,
        overlap=cfg.overlap,
        preload=True,
        reject_by_annotation=False,
    )

    n_epochs = len(epochs_all)
    win_len = float(cfg.epoch_dur)

    if n_epochs == 0:
        _restore_channel_types(raw_mark, original_types)
        return raw_mark, None, {
            "n_epochs": 0,
            "n_bad_p2p": 0,
            "n_bad_autoreject": 0,
            "note": "No epochs created; skipping marking.",
        }

    # 4) Hard p2p threshold (range)
    data = epochs_all.get_data()  # (n_epochs, n_ch, n_times)
    p2p = np.ptp(data, axis=2)
    bad_p2p = ((p2p < cfg.p2p_thresh[0]) | (p2p > cfg.p2p_thresh[1])).any(axis=1)

    keep_idx = np.where(~bad_p2p)[0]

    # Annotate BAD_p2p
    on_p2p = epochs_all.events[bad_p2p, 0] / sfreq
    ann_p2p = mne.Annotations(
        onset=on_p2p.tolist(),
        duration=[win_len] * int(bad_p2p.sum()),
        description=[cfg.desc_p2p] * int(bad_p2p.sum()),
        orig_time=raw_mark.annotations.orig_time,
    )
    raw_mark.set_annotations(raw_mark.annotations + ann_p2p)

    # 5) AutoReject thresholds on remaining epochs
    reject_log = None
    n_bad_ar = 0
    autoreject_error: str | None = None
    autoreject_plot_error: str | None = None

    epochs_thresh = epochs_all.copy().drop(np.where(bad_p2p)[0])
    if len(epochs_thresh) >= 1:
        try:
            threshes = compute_thresholds(
                epochs_thresh,
                picks="eeg",
                method=cfg.autoreject_method,
                random_state=cfg.random_state,
                augment=False,
                verbose=cfg.verbose,
            )
            threshes = {ch: float(t) * float(cfg.autoreject_correct_factor) for ch, t in threshes.items()}

            p2p_thr = np.ptp(epochs_thresh.get_data(), axis=2)  # (n_epochs_remain, n_ch)
            thr_vec = np.array([threshes[ch] for ch in epochs_thresh.ch_names], dtype=float)

            bad_epochs_mask = (p2p_thr > thr_vec).any(axis=1)
            labels_int = (p2p_thr > thr_vec).astype(int)

            reject_log = RejectLog(
                bad_epochs=bad_epochs_mask,
                labels=labels_int,
                ch_names=epochs_thresh.ch_names,
            )

            # Map back to original epoch indices
            bad_ar_orig = keep_idx[bad_epochs_mask]
            n_bad_ar = int(np.sum(bad_epochs_mask))

            # Annotate BAD_autoreject
            on_ar = epochs_all.events[bad_ar_orig, 0] / sfreq
            ann_ar = mne.Annotations(
                onset=on_ar.tolist(),
                duration=[win_len] * len(on_ar),
                description=[cfg.desc_autoreject] * len(on_ar),
                orig_time=raw_mark.annotations.orig_time,
            )
            raw_mark.set_annotations(raw_mark.annotations + ann_ar)

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
            raw_mark.set_annotations(raw_mark.annotations + ann_gap)

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
        "n_bad_p2p": int(np.sum(bad_p2p)),
        "n_bad_autoreject": int(n_bad_ar),
        "notches": notches.tolist() if notches is not None else None,
        "notch_widths": notch_widths,
        "min_good_len_sec": cfg.min_good_len_sec,
        "merge_gap_sec": cfg.merge_gap_sec,
    }
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
    overlap_policy: str = "split",  # "split" or "drop"
    match_mode: MatchMode = "exact",  # "substring" or "exact"
    case_sensitive: bool = False,

    # -------------------- ADDED: report + concat markers --------------------
    verbose: bool = True,
    add_concat_annotations: bool = True,
    concat_desc: str = "EDGE",
    concat_duration: float = 0.0,
    # ----------------------------------------------------------

) -> Tuple[mne.io.RawArray, mne.io.RawArray, Dict[str, Any]]:
    """
    End-to-end helper to:
      1) attach BAD annotations onto raw (optional)
      2) apply pre-filtering on the continuous recording (optional)
      3) remove BAD samples and rebuild a compressed RawArray (no raw.crop used)
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
    if overlap_policy not in ("split", "drop"):
        raise ValueError(f"overlap_policy must be 'split' or 'drop', got: {overlap_policy}")
    if match_mode not in ("substring", "exact"):
        raise ValueError(f"match_mode must be 'substring' or 'exact', got: {match_mode}")

    patterns: Tuple[str, ...] = (bad_descs,) if isinstance(bad_descs, str) else tuple(bad_descs)
    if len(patterns) == 0:
        raise ValueError("bad_descs cannot be empty; provide at least one BAD identifier.")

    def _norm(s: str) -> str:
        return s if case_sensitive else s.upper()

    patterns_norm = tuple(_norm(p) for p in patterns)

    def is_bad_desc(desc: str) -> bool:
        d = _norm(str(desc))
        if match_mode == "substring":
            return any(p in d for p in patterns_norm)
        else:  # exact
            return any(d == p for p in patterns_norm)

    # Work on a copy
    raw_labeled = raw.copy()
    raw_labeled.load_data()

    # 0) Attach BAD annotations (optional)
    if bad_annotations is not None and len(bad_annotations) > 0:
        # Normalize orig_time to avoid MNE errors when adding annotations
        target_orig_time = raw_labeled.annotations.orig_time
        if bad_annotations.orig_time != target_orig_time:
            bad_annotations = mne.Annotations(
                onset=list(bad_annotations.onset),
                duration=list(bad_annotations.duration),
                description=list(bad_annotations.description),
                orig_time=target_orig_time,
            )
        raw_labeled.set_annotations(bad_annotations + raw_labeled.annotations)

    sfreq = float(raw_labeled.info["sfreq"])
    n_times = int(raw_labeled.n_times)

    # Defaults mirroring your pipelines
    if pre_filter_kwargs is None:
        pre_filter_kwargs = dict(l_trans_bandwidth="auto", h_trans_bandwidth=4, phase="zero", fir_design="firwin")
    if post_filter_kwargs is None:
        post_filter_kwargs = dict(l_trans_bandwidth="auto", h_trans_bandwidth=4, phase="zero", fir_design="firwin")

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
            _append_info_description(raw_labeled, f"pre_notch: {resolved_notches.tolist()} Hz")
    ann = raw_labeled.annotations
    if ann is None:
        ann = mne.Annotations([], [], [], orig_time=None)

    # 2) Build bad sample mask (sample-accurate; consistent with your first_samp logic)
    bad_mask = np.zeros(n_times, dtype=bool)
    sample_shift = int(getattr(raw_labeled, "first_samp", 0))

    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        if not is_bad_desc(desc):
            continue
        start_samp = int(round(float(onset) * sfreq)) - sample_shift
        stop_samp = int(round((float(onset) + float(duration)) * sfreq)) - sample_shift
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
        cum += (e - s)

    new_data = np.concatenate(data_parts, axis=1)

    # 5) Re-map NON-BAD annotations onto the compressed timeline
    new_onsets: List[float] = []
    new_durs: List[float] = []
    new_descs: List[str] = []

    def find_containing_segment(sample_idx: int) -> int:
        for i, (s, e) in enumerate(segments):
            if s <= sample_idx < e:
                return i
        return -1

    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        if is_bad_desc(desc):
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
            continue

        if overlap_policy == "drop":
            kept = False
            for seg_i, (s, e) in enumerate(segments):
                if (s <= a_start) and (a_end <= e):
                    new_start_samp = seg_new_starts[seg_i] + (a_start - s)
                    new_onsets.append(new_start_samp / sfreq)
                    new_durs.append((a_end - a_start) / sfreq)
                    new_descs.append(str(desc))
                    kept = True
                    break
            if not kept:
                pass  # overlaps BAD -> drop whole annotation

        else:  # overlap_policy == "split"
            for seg_i, (s, e) in enumerate(segments):
                ov_s = max(s, a_start)
                ov_e = min(e, a_end)
                if ov_e <= ov_s:
                    continue
                new_start_samp = seg_new_starts[seg_i] + (ov_s - s)
                new_onsets.append(new_start_samp / sfreq)
                new_durs.append((ov_e - ov_s) / sfreq)
                new_descs.append(str(desc))

    # 6) Build RawArray (compressed timeline is not absolute time)
    # Note: In recent MNE versions, info['meas_date'] cannot be set directly.
    # Use Raw.set_meas_date(None) instead to clear absolute timing.
    info = raw_labeled.info.copy()
    raw_good = mne.io.RawArray(new_data, info)
    raw_good.set_meas_date(None)

    # Set re-mapped non-BAD annotations
    if len(new_onsets) > 0:
        raw_good.set_annotations(mne.Annotations(new_onsets, new_durs, new_descs, orig_time=None))
    else:
        raw_good.set_annotations(mne.Annotations([], [], [], orig_time=None))

    # -------------------- ADDED: concat boundary annotations --------------------
    concat_onsets_sec: List[float] = []  # boundary times (centers) in the compressed timeline
    concat_ann_onsets_sec: List[float] = []  # actual annotation start times
    concat_ann_durations_sec: List[float] = []  # actual annotation durations (may be clipped)
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
            raw_good.notch_filter(freqs=resolved_notches, notch_widths=post_notch_widths)
            _append_info_description(raw_good, f"post_notch: {resolved_notches.tolist()} Hz; width={post_notch_widths}")
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
      - If orig_time is None, onsets are relative to the current raw start (0 sec).
      - If orig_time is not None, onsets are absolute seconds from orig_time, aligned
        using raw.first_samp.

    Returns
    -------
    raw_out : mne.io.BaseRaw
        Raw with appended head/tail annotations.
    report : dict
        Summary including onsets/durations and reference-frame details.
    """
    raw_out = raw.copy() if copy else raw

    if not np.isfinite(float(head_duration_sec)) or float(head_duration_sec) < 0:
        raise ValueError(f"head_duration_sec must be finite and >= 0, got: {head_duration_sec}")

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

    # Determine recording bounds in the annotation reference frame
    if orig_time is None:
        rec_start_sec = 0.0
        rec_end_sec = total_len_sec
        last_sample_sec = max(rec_start_sec, rec_end_sec - 1.0 / sfreq)
    else:
        rec_start_sec = float(raw_out.first_samp) / sfreq
        rec_end_sec = float(raw_out.first_samp + n_times) / sfreq  # exclusive end
        last_sample_sec = float(raw_out.first_samp + n_times - 1) / sfreq

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
        keep_mask = np.array([str(d) not in remove_set for d in existing.description], dtype=bool)
        existing = existing[keep_mask] if np.any(keep_mask) else mne.Annotations([], [], [], orig_time=orig_time)

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
        )

    raw_out.set_annotations(combined)

    report: Dict[str, Any] = {
        "description": str(description),
        "head_duration_sec": float(head_duration_sec),
        "tail_duration_sec": float(tail_val),
        "head_description": head_desc,
        "tail_description": tail_desc,
        "n_added": int(len(new_onsets)),
        "added_onsets_sec": [float(x) for x in new_onsets],
        "added_durations_sec": [float(x) for x in new_durs],
        "orig_time_is_none": bool(orig_time is None),
        "recording_start_sec": float(rec_start_sec),
        "recording_end_sec": float(rec_end_sec),
        "last_sample_sec": float(last_sample_sec),
        "total_duration_sec": float(total_len_sec),
    }

    return raw_out, report
