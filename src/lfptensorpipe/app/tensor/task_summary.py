"""Read-only planning for the Build Tensor status bar and its Details window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.lfp.common.timefreq import compute_decimation

from . import service as svc
from .connectivity_coordinator import (
    _prepare_psi,
    _prepare_trgc,
    _prepare_undirected_cohort,
    shared_compute_signature,
)
from .cpu_budget import derive_global_compute_slots
from .m6_reference import Point, make_profile, reference
from .orchestration_plan_dispatch import build_runtime_plan
from .orchestration_plans import prepare_effective_metric_inputs
from .params import TENSOR_DIRECTED_SELECTOR_KEYS, TENSOR_UNDIRECTED_SELECTOR_KEYS
from .selectors import coerce_tensor_channels, coerce_tensor_pairs


@dataclass
class MetricTask:
    key: str
    channels: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()
    output_grid: str = "—"
    planned_work: str = "—"
    work_details: str = ""
    array_bytes: int | None = None
    point: Point | None = None
    group: int = 0
    weight: float | None = None
    units: float | None = None
    error: str = ""
    frequency_groups: int | None = None


@dataclass
class TaskSummary:
    metrics: dict[str, MetricTask]
    cpu_percent: float | None
    slots: int | None = None
    duration: float | None = None
    sfreq: float | None = None
    input_channels: int | None = None
    runtime: float | None = None
    basis: str = "Unavailable"
    error: str = ""

    @property
    def array_bytes(self):
        sizes = [m.array_bytes for m in self.metrics.values()]
        return sum(sizes) if sizes and None not in sizes else None

    def counts(self):
        channels = set()
        directed = set()
        undirected = set()
        for m in self.metrics.values():
            channels.update(m.channels)
            if m.key in TENSOR_DIRECTED_SELECTOR_KEYS:
                directed.update(m.pairs)
            elif m.key in TENSOR_UNDIRECTED_SELECTOR_KEYS:
                undirected.update(m.pairs)
        return len(channels), len(undirected), len(directed)


def selection_summary(params, cpu_percent):
    """Fields available immediately, including while a FIF header is being read."""
    rows = {}
    for key, p in params.items():
        pairs = coerce_tensor_pairs(
            p.get("selected_pairs"), directed=key in TENSOR_DIRECTED_SELECTOR_KEYS
        )
        channels = (
            tuple(dict.fromkeys(ch for pair in pairs for ch in pair))
            if pairs
            else coerce_tensor_channels(p.get("selected_channels"))
        )
        rows[key] = MetricTask(key, channels, pairs)
    summary = TaskSummary(rows, cpu_percent)
    if cpu_percent is not None:
        summary.slots = derive_global_compute_slots(cpu_percent, detected_cpu_count=12)[
            2
        ]
    return summary


def _specparam_size(context, plan, n_times, decim, sfreq):
    from .runners.periodic_aperiodic_models import PeriodicAperiodicOptions
    from .runners.periodic_aperiodic_prepare import (
        build_periodic_aperiodic_paths,
        prepare_periodic_aperiodic_runtime,
    )
    from .runners.periodic_aperiodic_compute import (
        _periodic_annotation_support,
        spectrum_skip_mask,
    )

    kwargs = {
        "time_smooth_kernel_size": None,
        "freq_smooth_sigma": None,
        **plan.runner_kwargs,
    }
    opt = PeriodicAperiodicOptions(
        context=context, **kwargs, n_jobs=1, outer_n_jobs=1, value_transform_mode="dB"
    )
    prepared = prepare_periodic_aperiodic_runtime(
        opt, build_periodic_aperiodic_paths(opt, create=False)
    )
    try:
        fits = n_times * len(prepared.picks)
        if opt.mask_edge_effects and len(prepared.raw.annotations):
            support = _periodic_annotation_support(
                prepared, opt, {"params": {"decim_eff": decim}}
            )
            mask = spectrum_skip_mask(
                prepared.raw,
                prepared.picks,
                np.arange(n_times) * decim / sfreq,
                support["annotation_skip_radius_s"],
            )
            fits -= int(mask.sum())
        # The benchmark parameter axis contains the aperiodic fields and fit statistics.
        num_params = 4 if opt.aperiodic_mode == "fixed" else 5
        return fits, len(prepared.freqs_final), len(prepared.freqs_compute), num_params
    finally:
        prepared.raw.close()


def _notch_intervals(kw):
    ns = tuple(kw["notches"])
    return svc._compute_notch_intervals(
        low_freq=kw["low_freq"],
        high_freq=kw["high_freq"],
        notches=ns,
        notch_radii=svc._expand_notch_radii(kw["notch_radii"], len(ns)),
    )


def _connection_work_details(descriptors):
    return "\n".join(
        f"{group['owner_id']} · group {group['group_id']}: {len(group['valid_time_indices']):,} valid windows"
        for group in descriptors
    )


def build_task_summary(context: RecordContext, params, mask_edge_effects, cpu_percent):
    """Use the numerical runners' task descriptions. Never create products or run a fit."""
    import mne
    from .runners.burst import _build_runtime_bands
    from lfptensorpipe.utils.freqs import decimal_notch_intervals

    summary = selection_summary(params, cpu_percent)
    resolver = PathResolver(context)
    path = svc.preproc_step_raw_path(resolver, "finish", create=False)
    with mne.io.read_raw_fif(path, preload=False, verbose="ERROR") as raw:
        inventory = svc._tensor_channel_inventory_from_raw(raw)
        sfreq = summary.sfreq = float(raw.info["sfreq"])
        n_samples = int(raw.n_times)
        summary.duration = n_samples / sfreq
        summary.input_channels = len(raw.ch_names)
        has_mask_labels = any(
            any(token in desc.lower() for token in ("bad", "edge"))
            for desc in raw.annotations.description
        )
    undirected = {}
    group_ids = {}
    for key in sorted(params):
        row = summary.metrics[key]
        metric_params = dict(params[key])
        try:
            prepared, _ = prepare_effective_metric_inputs(
                svc,
                context,
                key,
                svc.TENSOR_METRICS_BY_KEY[key].display_name,
                metric_params,
                inventory,
            )
            if key in svc.TENSOR_CHANNEL_SELECTOR_KEYS:
                row.channels = tuple(prepared.metric_channels)
            else:
                row.pairs = tuple(
                    svc._normalize_selected_pairs(
                        prepared.metric_pairs,
                        available_channels=set(inventory.usable_channels),
                        directed=key in TENSOR_DIRECTED_SELECTOR_KEYS,
                    )
                )
                row.channels = tuple(
                    dict.fromkeys(channel for pair in row.pairs for channel in pair)
                )
            plans = build_runtime_plan(
                svc, context, prepared=prepared, mask_edge_effects=mask_edge_effects
            )
            metric_params = prepared.metric_params
            plan = plans[key]
            decim = (
                1
                if key == "burst"
                else compute_decimation(sfreq, hop_s=metric_params["hop_s"])[0]
            )
            output_times = len(range(0, n_samples, decim))
            bands = []
            compute_bins = 1
            parameter_bins = 0
            window_groups = 1
            sharing: Any = key
            if key in TENSOR_UNDIRECTED_SELECTOR_KEYS:
                sharing = shared_compute_signature(plan)
                if sharing not in undirected:
                    undirected[sharing] = _prepare_undirected_cohort(
                        svc, context, [plan], cohort_id=key
                    )
                state = undirected[sharing]
                compute_bins = len(state["grid_kwargs"]["freqs"])
                output_bins = len(
                    svc._build_frequency_grid(
                        prepared.metric_low,
                        min(prepared.metric_high, sfreq / 2),
                        prepared.metric_step,
                    )
                )
                counts = [
                    len(g["valid_time_indices"]) for g in state["group_descriptors"]
                ]
                row.work_details = _connection_work_details(state["group_descriptors"])
                windows = state["description"]["groups"]
                window_groups = len(windows)
                work = sum(
                    len(g["valid_time_indices"]) * len(g["call_frequency_indices"])
                    for g in windows
                )
            elif key == "trgc":
                state = _prepare_trgc(
                    context,
                    {
                        backend.runner_kwargs["backend_method"]: backend
                        for backend in plans.values()
                        if backend.runner_key == "trgc_backend"
                    },
                )
                compute_bins = len(state["prepared"]["freqs_compute"])
                output_bins = len(state["prepared"]["freqs_full"])
                counts = [
                    len(g["valid_time_indices"]) for g in state["group_descriptors"]
                ]
                row.work_details = _connection_work_details(state["group_descriptors"])
                row.frequency_groups = len(state["descriptions"]["gc"]["groups"])
                work = sum(counts)
            elif key == "psi":
                state = _prepare_psi(svc, context, plan)["prepared"]["metadata"]
                bands = [
                    b
                    for b in prepared.metric_bands
                    if b["name"] in state["axes"]["freq"]
                ]
                output_bins = len(bands)
                compute_bins = sum(
                    map(len, state["params"]["frequencies_by_band"].values())
                )
                counts = list(state["params"]["n_valid_windows_by_band"])
                row.work_details = "\n".join(
                    f"{name}: {count:,} valid windows, {len(state['params']['frequencies_by_band'][name])} frequencies"
                    for name, count in zip(state["axes"]["freq"], counts)
                )
                work = sum(counts)
            elif key == "periodic_aperiodic":
                work, output_bins, compute_bins, parameter_bins = _specparam_size(
                    context, plan, output_times, decim, sfreq
                )
                row.planned_work = f"{work:,} spectra"
            elif key == "burst":
                runner_args = plan.runner_kwargs
                spectral = runner_args["method"] != "hilbert"
                segments, grids = _build_runtime_bands(
                    bands=prepared.metric_bands,
                    low_freq=runner_args["low_freq"],
                    high_freq=runner_args["high_freq"],
                    notch_intervals=_notch_intervals(runner_args),
                    spectral_step=runner_args.get("step_hz") if spectral else None,
                    spectral_notches=(
                        decimal_notch_intervals(
                            runner_args["notches"],
                            svc._expand_notch_radii(
                                runner_args["notch_radii"], len(runner_args["notches"])
                            ),
                        )
                        if spectral
                        else None
                    ),
                )
                bands = [b for b in prepared.metric_bands if b["name"] in segments]
                if not bands:
                    raise ValueError("No valid Burst bands.")
                if runner_args.get("max_cycles") is not None:
                    from lfptensorpipe.lfp.burst.grid import (
                        _normalize_bands,
                        burst_run_sample_limits,
                    )

                    names, _, edges = _normalize_bands(segments)
                    for name, center in zip(names, edges.mean(axis=1)):
                        burst_run_sample_limits(
                            name,
                            center,
                            sfreq,
                            runner_args["min_cycles"],
                            runner_args["max_cycles"],
                        )
                output_bins = len(bands)
                compute_bins = sum(map(len, grids.values())) if grids else 1
                work = n_samples * len(row.channels) * output_bins
                row.planned_work = f"{output_bins} {'band' if output_bins == 1 else 'bands'} × {len(row.channels)} channels"
            else:
                runner_args = plan.runner_kwargs
                grid = svc._build_frequency_grid(
                    prepared.metric_low,
                    min(prepared.metric_high, sfreq / 2),
                    prepared.metric_step,
                )
                output_bins = len(grid)
                remaining, removed = svc._cut_frequency_grid_by_intervals(
                    grid, _notch_intervals(runner_args)
                )
                compute_bins = len(remaining)
                if np.any(removed) and compute_bins < 2:
                    raise ValueError(
                        "Notch exclusion removed too many bins; reduce notch radii or widen the frequency range."
                    )
                work = n_samples * len(row.channels)
                row.planned_work = (
                    f"{len(row.channels)} channels × {compute_bins} frequencies"
                )
            if row.pairs:
                row.planned_work = f"{sum(counts):,} valid windows"
                if len(counts) > 1:
                    row.planned_work += f" ({len(counts)} groups)"
            row.group = group_ids.setdefault(sharing, len(group_ids))
            profile = make_profile(
                key,
                metric_params,
                sfreq=sfreq,
                frequencies=compute_bins,
                channels=row.channels,
                pairs=row.pairs,
                bands=bands,
            )
            row.point = Point(profile, summary.duration, sfreq, work, window_groups)
            size = len(row.pairs) or len(row.channels)
            row.array_bytes = size * (output_bins + parameter_bins) * output_times * 8
            row.output_grid = (
                f"{output_bins} {'band' if output_bins == 1 else 'bands'}"
                if bands
                else f"{output_bins} freq"
            )
            row.output_grid += f" × {output_times:,} times"
        except (ValueError, OSError, RuntimeError) as exc:
            row.error = str(exc)
    rows = [summary.metrics[k] for k in sorted(params)]
    if cpu_percent is not None and rows and all(row.point for row in rows):
        units, summary.slots, summary.runtime, summary.basis = reference(
            tuple(row.point for row in rows),
            tuple(row.group for row in rows),
            cpu_percent,
            mask_matches=mask_edge_effects and not has_mask_labels,
        )
        if units and sum(units) > 0:
            for row, u in zip(rows, units):
                row.units = u
                row.weight = 100 * u / sum(units)
    return summary
