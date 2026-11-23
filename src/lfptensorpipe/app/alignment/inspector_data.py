"""Read the accepted Align Run geometry, warped metrics and native signal on demand."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from lfptensorpipe.app.alignment.generation import (
    accepted_alignment_artifact_paths,
    latest_alignment_step_entry,
)
from lfptensorpipe.app.alignment.paths import alignment_warp_labels_path
from lfptensorpipe.app.alignment.warper_builder import _resolve_target_duration_s
from lfptensorpipe.app.preproc_service import preproc_step_raw_path
from lfptensorpipe.io.pkl_io import load_pkl
from lfptensorpipe.lfp.mask.annotations import annotation_sample_support_by_channel
from lfptensorpipe.lfp.burst.timeline import WarpSegment, build_warp_segments
from lfptensorpipe.lfp.warp.event_anchored_linear import cue_response_anchors

from .inspector_config import SIGNAL


def runs(mask):
    """Contiguous half-open slices of a one-dimensional Boolean mask."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


@dataclass
class SignalSpan:
    time: np.ndarray
    end: np.ndarray
    values: np.ndarray
    seconds_per_unit: float
    segment: WarpSegment
    boundaries: tuple = ()
    sample_time: np.ndarray | None = None


@dataclass
class MetricGrid:
    values: np.ndarray
    channels: list[str]
    freq: np.ndarray
    time: np.ndarray
    notches: list
    bands: dict

    @property
    def numeric(self):
        return np.issubdtype(self.freq.dtype, np.number)

    def frequency_mask(self, selection):
        mask = np.ones(len(self.freq), dtype=bool)
        if self.numeric:
            for lo, hi in self.notches:
                mask &= (self.freq < lo) | (self.freq > hi)
        if selection == "All":
            return mask
        if self.numeric and "hz" in selection:
            lo, hi = selection["hz"]
            return mask & (self.freq >= lo) & (self.freq <= hi)
        if not self.numeric and "label" in selection:
            if "segments_hz" in selection and selection[
                "segments_hz"
            ] != self.bands.get(selection["label"]):
                return np.zeros_like(mask)
            return mask & (self.freq.astype(str) == selection["label"])
        return np.zeros(len(self.freq), dtype=bool)


class InspectorData:
    def __init__(self, resolver, slug):
        self.resolver = resolver
        self.slug = slug
        self.paths = dict(
            accepted_alignment_artifact_paths(resolver, trial_slug=slug, stage="run")
        )
        if not self.paths:
            raise ValueError("Run Align Epochs with the current trial settings first.")
        self.run_index, entry = latest_alignment_step_entry(
            resolver, trial_slug=slug, step="run_align_epochs"
        )
        self.method = entry["params"]["method"]
        self.params = entry["params"]["method_params"]
        labels = load_pkl(alignment_warp_labels_path(resolver, slug))
        self.epochs = labels["ALL"]
        self.raw_path = preproc_step_raw_path(resolver, "finish")
        self.metrics = ([SIGNAL] if self.raw_path.is_file() else []) + list(self.paths)
        self.axis = "percent"
        self.bounds = (0.0, 100.0)
        params = self.params
        if "cue_response" in params:
            cue = params["cue_response"]
            self.axis = "cue_seconds"
            self.bounds = (
                -cue["pre_cue_s"],
                cue["response_target_s"] + cue["post_response_s"],
            )
            params = {
                **params,
                "linear_warp": True,
                "anchors_percent": cue_response_anchors(cue),
            }
        elif self.method in ("pad_warper", "concat_warper"):
            self.axis = "epoch_seconds"
            self.bounds = (
                0.0,
                _resolve_target_duration_s(
                    method=self.method, method_params=params, epochs_by_label=labels
                ),
            )
        self.unit = "%" if self.axis == "percent" else "s"
        lo, hi = self.bounds
        self.mappings = []
        for epoch in self.epochs:
            segments = build_warp_segments(self.method, params, epoch)
            self.mappings.append(
                [
                    WarpSegment(
                        lo + s.target_start * (hi - lo) / 100,
                        lo + s.target_end * (hi - lo) / 100,
                        s.source_start,
                        s.source_end,
                        s.occurrence,
                    )
                    for s in segments
                ]
            )
        self._grid = None
        self._metric = None

    @cached_property
    def signal_channels(self):
        import mne

        with mne.io.read_raw_fif(self.raw_path, preload=False, verbose="ERROR") as raw:
            return list(raw.ch_names)

    def channels(self, metric):
        return self.signal_channels if metric == SIGNAL else self.grid(metric).channels

    def grid(self, metric):
        if self._metric == metric:
            return self._grid
        if metric not in self.paths:
            raise ValueError(f"Metric unavailable in this Run: {metric}")
        payload = load_pkl(self.paths[metric])
        meta = payload["meta"]
        axes = meta["axes"]
        source = meta.get("source") or {}
        src = source.get("params", {})
        freq = np.asarray(axes["freq"])
        self._grid = MetricGrid(
            np.asarray(payload["tensor"], dtype=float),
            [str(c) for c in axes["channel"]],
            freq,
            np.asarray(
                axes["percent" if self.axis == "percent" else "time"], dtype=float
            ),
            source.get("notch_intervals_hz", src.get("notch_intervals_hz", [])),
            src.get("bands_segments_hz", {}),
        )
        self._metric = metric
        return self._grid

    def validate_rule(self, rule):
        metric = rule["metric"]
        if metric not in self.metrics:
            raise ValueError(f"Metric unavailable in this trial: {metric}")
        if "axis" in rule and rule["axis"] != self.axis:
            raise ValueError(
                f"Review the time settings: {rule['axis']} differs from {self.axis}."
            )
        channels = rule["channels"]
        if channels != "All":
            missing = set(channels) - set(self.channels(metric))
            if missing:
                raise ValueError(
                    "Channels unavailable in this trial: " + ", ".join(sorted(missing))
                )
        if (
            rule["frequency"] != "All"
            and not self.grid(metric).frequency_mask(rule["frequency"]).any()
        ):
            raise ValueError("The selected frequency range or bin is unavailable.")

    def read_signal(self, indices=None):
        """Read only the source fragments of the requested epochs, with scoped BAD/EDGE masks."""
        import mne

        result = []
        with mne.io.read_raw_fif(self.raw_path, preload=False, verbose="ERROR") as raw:
            sf = float(raw.info["sfreq"])
            invalid, boundaries, _ = annotation_sample_support_by_channel(
                raw, channels=raw.ch_names, keep=("bad", "edge")
            )
            for name in raw.info["bads"]:
                invalid[raw.ch_names.index(name)] = True
            for index in range(len(self.epochs)) if indices is None else indices:
                spans = []
                for part, s in enumerate(self.mappings[index]):
                    start = int(
                        np.clip(np.floor(s.source_start * sf + 1e-9), 0, raw.n_times)
                    )
                    stop = int(
                        np.clip(np.ceil(s.source_end * sf - 1e-9), 0, raw.n_times)
                    )
                    endpoint = (
                        self.method == "linear_warper"
                        and self.axis == "percent"
                        and part == len(self.mappings[index]) - 1
                    )
                    if (
                        endpoint
                        and stop < raw.n_times
                        and abs(stop / sf - s.source_end) < 1e-10
                    ):
                        stop += 1
                    times = np.arange(start, stop) / sf
                    source = np.maximum(times, s.source_start)
                    end = np.minimum(times + 1 / sf, s.source_end)
                    values = (
                        raw.get_data(start=start, stop=stop) * 1e6
                        if stop > start
                        else np.empty((len(raw.ch_names), 0))
                    )
                    values[invalid[:, start:stop]] = np.nan
                    values[~np.isfinite(values)] = np.nan
                    cuts = tuple(
                        tuple(p - start for p in points if start <= p < stop)
                        for points in boundaries
                    )
                    scale = (s.source_end - s.source_start) / (
                        s.target_end - s.target_start
                    )
                    sample_time = s.target_start + (times - s.source_start) / scale
                    if self.method == "stack_warper":
                        first = max(0, round(s.source_start * sf))
                        last = min(raw.n_times, round(s.source_end * sf)) - 1
                        indices = np.arange(start, stop)
                        sample_time = s.target_start + (indices - first) * (
                            s.target_end - s.target_start
                        ) / max(1, last - first)
                        sample_time[(indices < first) | (indices > last)] = np.nan
                    else:
                        sample_time[times < s.source_start - 1e-10] = np.nan
                    spans.append(
                        SignalSpan(
                            s.target_start + (source - s.source_start) / scale,
                            s.target_start + (end - s.source_start) / scale,
                            values,
                            scale,
                            s,
                            cuts,
                            sample_time,
                        )
                    )
                result.append(spans)
        return result

    @cached_property
    def native_burst(self):
        from lfptensorpipe.app.features.burst_native import _load_native_burst_tensor

        try:
            return _load_native_burst_tensor(self.resolver)
        except (FileNotFoundError, ValueError):
            return None

    def source_intervals(self, epoch, lo, hi):
        result = []
        for s in self.mappings[epoch]:
            a, b = max(lo, s.target_start), min(hi, s.target_end)
            if a < b or (lo == hi and s.target_start <= lo <= s.target_end):
                ratio = (s.source_end - s.source_start) / (
                    s.target_end - s.target_start
                )
                result.append(
                    [
                        s.source_start + (a - s.target_start) * ratio,
                        s.source_start + (b - s.target_start) * ratio,
                    ]
                )
        return result
