"""Machine-independent Build Tensor workload and wall-time references."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from statistics import median
from typing import NamedTuple

from .connectivity_coordinator import _UNDIRECTED_ESTIMATOR_BY_METRIC
from .cpu_budget import derive_global_compute_slots


class Profile(NamedTuple):
    metric: str
    method: str
    mode: str
    resolution: float
    hop: float
    frequencies: int
    series: int
    bands: int
    lags: int
    details: str


def make_profile(key, params, *, sfreq, frequencies, channels, pairs, bands):
    """Describe effective work without embedding subject, channel, or atlas names."""
    method = params.get("method", "hilbert" if key == "burst" else "morlet")
    mode = params.get("aperiodic_mode", "fixed") if key == "periodic_aperiodic" else ""
    hop = params.get("hop_s", 0.025) if key != "burst" else None
    decim = max(1, round(sfreq * hop)) if hop is not None else 1
    # Keep the remaining normalized estimator settings for exact benchmark
    # matching. Timing depends on selector sizes and sharing, not their names.
    dimensions = {
        "method",
        "aperiodic_mode",
        "hop_s",
        "time_resolution_s",
        "bands",
        "selected_channels",
        "selected_pairs",
        "gc_n_lags",
        "thresholds_source_path",
        "thresholds",
    }
    details = {k: v for k, v in params.items() if k not in dimensions and v is not None}
    if key == "burst":
        method_fields = {
            "hilbert": {"hilbert_filter_method", "hilbert_edge_tolerance_pct"},
            "morlet": {"freq_step_hz", "morlet_n_cycles"},
            "multitaper": {"freq_step_hz", "mt_n_cycles", "mt_time_bandwidth_product"},
        }
        for inactive in set.union(*method_fields.values()) - method_fields[method]:
            details.pop(inactive, None)
        if not details.get("baseline_keep"):
            details.pop("baseline_keep", None)
    if not details.get("notches"):
        details.pop("notch_radii", None)
    if bands:
        details["bands"] = [(float(b["start"]), float(b["end"])) for b in bands]
    if pairs:
        details["channels"] = len(channels)
    if key == "trgc":
        details["internal_pairs"] = len({tuple(sorted(pair)) for pair in pairs})
    if params.get("thresholds") is not None:
        details["thresholds"] = "provided"
    return Profile(
        key,
        method,
        mode,
        params.get("time_resolution_s", 0.0) if hop else 0.0,
        round(decim / sfreq, 9),
        frequencies,
        len(pairs) if pairs else len(channels),
        len(bands),
        params.get("gc_n_lags", 20) if key == "trgc" else 0,
        json.dumps(details, sort_keys=True, separators=(",", ":")),
    )


@dataclass(frozen=True)
class Point:
    profile: Profile
    duration: float
    sfreq: float
    work: int
    window_groups: int = 1

    def coordinates(self):
        p = self.profile
        return (
            self.duration,
            self.sfreq,
            p.resolution,
            1 / p.hop,
            p.frequencies,
            p.series,
            p.bands,
            p.lags,
        )


class Measurement(NamedTuple):
    points: tuple[Point, ...]
    slots: int
    seconds: float
    compute: float
    parts: tuple[float, ...]
    groups: tuple[int, ...]


@lru_cache(maxsize=1)
def _table():
    from lfptensorpipe.resources.m6_tensor import PROFILES, RUNS

    profiles = [Profile(*p) for p in PROFILES]
    by_selection = defaultdict(list)
    for duration, sfreq, slots, ids, work, parts, groups, seconds, compute in RUNS:
        pts = tuple(
            (
                Point(profiles[i], duration, sfreq, *n)
                if isinstance(n, tuple)
                else Point(profiles[i], duration, sfreq, n)
            )
            for i, n in zip(ids, work)
        )
        by_selection[tuple(p.profile.metric for p in pts)].append(
            Measurement(pts, slots, seconds, compute, parts, groups)
        )
    return by_selection


def _blend(values, x):
    """Local power interpolation on measured positive workload coordinates."""
    if x in values:
        return values[x], False
    keys = sorted(values)
    if len(keys) == 1 or x <= 0 or keys[0] <= 0:
        return values[keys[0]], True
    i = min(max(bisect_left(keys, x) - 1, 0), len(keys) - 2)
    a, b = keys[i : i + 2]
    exponent = math.log(values[b] / values[a]) / math.log(b / a)
    return values[a] * (x / a) ** exponent, x < a or x > b


def _coordinate_distance(left, right):
    return sum(abs(math.log(x / y)) for x, y in zip(left, right) if x > 0 and y > 0)


def _distance(a: Point, b: Point) -> float:
    return _coordinate_distance(a.coordinates(), b.coordinates())


@lru_cache(maxsize=512)
def _single_cost(point):
    """Single-slot equivalent seconds. Scale cost per unit so work is counted once."""
    p = point.profile
    if point.work == 0 and p.metric in {
        *_UNDIRECTED_ESTIMATOR_BY_METRIC,
        "psi",
        "trgc",
    }:
        return 0.0, True
    rows = [
        r
        for r in _table()[(p.metric,)]
        if r.points[0].profile.method == p.method and r.points[0].profile.mode == p.mode
    ]
    one = [r for r in rows if r.slots == 1]
    exact = next((r for r in one if r.points[0] == point), None)
    if exact:
        return exact.parts[0], False
    nine = [r for r in rows if r.slots == 9]
    # The resolution sweeps used nine reference slots. Anchor their ratios to
    # single-slot work through the same workload measured at both budgets.
    anchors = [(a, b) for a in one for b in nine if a.points == b.points]
    if not anchors:
        # Multitaper single-slot scans used the same single-slot / nine-slot
        # baseline conversion as the matching Morlet metric.
        all_rows = [
            r for r in _table()[(p.metric,)] if r.points[0].profile.mode == p.mode
        ]
        anchors = [
            (a, b)
            for a in all_rows
            if a.slots == 1
            for b in all_rows
            if b.slots == 9 and a.points == b.points
        ]
    if not nine or not anchors:
        return None, True
    serial, parallel = min(anchors, key=lambda ab: _distance(ab[0].points[0], point))
    scale = serial.parts[0] / parallel.parts[0]
    nearest = min(
        nine,
        key=lambda r: (
            _distance(r.points[0], point),
            r.points[0].profile.details != p.details,
        ),
    )
    coords = nearest.points[0].coordinates()
    rate = nearest.parts[0] / max(1, nearest.points[0].work)
    outside = point.window_groups > max(r.points[0].window_groups for r in nine)
    for axis, (start, target) in enumerate(zip(coords, point.coordinates())):
        if target == start:
            continue
        # Prefer a true one-factor sweep. With two changed settings the
        # nearest record is often itself an off-baseline measurement.
        curves = defaultdict(dict)
        for measurement in nine:
            source = measurement.points[0]
            position = source.coordinates()
            fixed_axes = position[:axis] + position[axis + 1 :]
            curves[fixed_axes][position[axis]] = measurement.parts[0] / max(
                1, source.work
            )
        sweeps = [
            fixed_axes for fixed_axes, samples in curves.items() if len(samples) >= 2
        ]
        if not sweeps:
            outside = True
            continue
        fixed_target = coords[:axis] + coords[axis + 1 :]
        closest = min(
            sweeps, key=lambda fixed: _coordinate_distance(fixed, fixed_target)
        )
        samples = curves[closest]
        initial_rate, _ = _blend(samples, start)
        target_rate, extrapolated = _blend(samples, target)
        rate *= target_rate / initial_rate
        outside |= extrapolated
    return rate * max(1, point.work) * scale, outside


def _split(costs, points, groups):
    """Divide shared work among unique estimators, then among their outputs."""
    out = list(costs)
    for group in set(groups):
        members = [i for i, g in enumerate(groups) if g == group]
        if len(members) < 2:
            continue
        est = [
            _UNDIRECTED_ESTIMATOR_BY_METRIC[points[i].profile.metric] for i in members
        ]
        shared = costs[members[0]]
        for i, e in zip(members, est):
            out[i] = shared / (len(set(est)) * est.count(e))
    return tuple(out)


def _estimate_costs(points, groups, records):
    one = [r for r in records if r.slots == 1 and tuple(r.groups) == tuple(groups)]
    anchor = min(
        one,
        key=lambda r: (
            sum(
                (a.profile.method, a.profile.mode) != (b.profile.method, b.profile.mode)
                for a, b in zip(points, r.points)
            ),
            sum(_distance(a, b) for a, b in zip(points, r.points)),
            sum(a != b for a, b in zip(points, r.points)),
        ),
        default=None,
    )
    if anchor and anchor.points == points:
        return _split(anchor.parts, points, groups)
    single = [_single_cost(p)[0] for p in points]
    if any(x is None for x in single):
        return None
    if anchor:
        # Matched nine-slot combinations capture shared computation and the
        # measured interaction between parameters. Transfer their per-owner
        # ratios to a common single-slot budget.
        nine = [r for r in records if r.slots == 9 and r.groups == anchor.groups]
        target_nine = next((r for r in nine if r.points == points), None)
        base_nine = next((r for r in nine if r.points == anchor.points), None)
        if target_nine and base_nine and all(base_nine.parts):
            projected = [
                base * new / old
                for base, new, old in zip(
                    anchor.parts, target_nine.parts, base_nine.parts
                )
            ]
            return _split(projected, points, groups)
        ref = [_single_cost(p)[0] for p in anchor.points]
        if all(ref):
            parts = _split(anchor.parts, points, groups)
            return tuple(s * t / u for s, t, u in zip(parts, single, ref))
    # For an unmeasured combination use measured cohorts as a reference when
    # available. Otherwise count the common spectral calculation once.
    costs = list(single)
    for group in set(groups):
        ix = [i for i, g in enumerate(groups) if g == group]
        if len(ix) < 2:
            continue
        subset = tuple(points[i] for i in ix)
        named = _table().get(tuple(p.profile.metric for p in subset), ())
        if len(ix) < len(points) and named:
            values = _estimate_costs(subset, (0,) * len(ix), named)
            for i, v in zip(ix, values):
                costs[i] = v
        else:
            full = tuple(sorted(_UNDIRECTED_ESTIMATOR_BY_METRIC))
            base = next(
                (
                    r
                    for r in _table().get(full, ())
                    if r.slots == 1 and r.points[0].duration == 60
                ),
                None,
            )
            avg = sum(single[i] for i in ix) / len(ix)
            if base:
                solo = [_single_cost(p)[0] for p in base.points]
                n = len(
                    {_UNDIRECTED_ESTIMATOR_BY_METRIC[p.profile.metric] for p in subset}
                )
                avg *= 1 + (base.parts[0] / (sum(solo) / len(solo)) - 1) * (n - 1) / 4
            shares = _split([avg] * len(ix), subset, (0,) * len(ix))
            for i, v in zip(ix, shares):
                costs[i] = v
    return tuple(costs)


def _pooled(point):
    return (
        point.profile.metric in _UNDIRECTED_ESTIMATOR_BY_METRIC
        or point.profile.metric in ("trgc", "psi")
    )


@lru_cache(maxsize=1)
def _component_factors():
    """Measure overlap and fixed overhead using jointly and individually timed work."""
    singles = {
        (r.points[0], r.slots): r.compute
        for selection, runs in _table().items()
        if len(selection) == 1
        for r in runs
    }
    observations = defaultdict(list)
    for selection, runs in _table().items():
        if len(selection) < 2:
            continue
        for r in runs:
            # Standalone timings can be summed for calibration only when every
            # component was timed at these slots and has a separate compute owner.
            if len(set(r.groups)) != len(selection):
                continue
            separate = [singles.get((p, r.slots)) for p in r.points]
            if all(separate):
                pooled = sum(_pooled(p) for p in r.points)
                key = r.slots, r.points[0].duration, len(selection) - pooled, pooled
                observations[key].append(
                    (r.seconds - r.compute, r.compute / sum(separate))
                )
    return {
        key: (median(h for h, _ in rows), median(ratio for _, ratio in rows))
        for key, rows in observations.items()
    }


def _component_runtime(points, groups, costs, slots):
    """Estimate an unmeasured selection from unique work and M6 scheduling data."""
    independent = len({g for g, p in zip(groups, points) if not _pooled(p)})
    pooled = len(set(groups)) - independent
    factors = _component_factors()
    schedule = {}
    for budget in {k[0] for k in factors}:
        # Match the independent/pool mix first, then the number of groups and
        # the input duration. Use aggregate measurements for overhead and overlap.
        key = min(
            (k for k in factors if k[0] == budget),
            key=lambda k: (
                (bool(k[2]) != bool(independent)) + (bool(k[3]) != bool(pooled)),
                abs(k[2] - independent) + abs(k[3] - pooled),
                abs(k[1] - points[0].duration),
                k,
            ),
        )
        overhead, ratio = factors[key]
        compute = 0.0
        for point, cost in zip(points, costs):
            if budget != 1 and _pooled(point) and cost:
                runs = (
                    r for r in _table()[(point.profile.metric,)] if r.slots == budget
                )
                matched = min(
                    runs,
                    key=lambda r: (
                        r.points[0].profile.method != point.profile.method,
                        _distance(r.points[0], point),
                    ),
                )
                cost *= matched.compute / _single_cost(matched.points[0])[0]
            compute += cost
        schedule[budget] = max(0.0, overhead) + ratio * compute
    return _blend(schedule, slots)


def reference(points, groups, cpu_percent, *, mask_matches=True):
    """Return metric work units, M6 slots, wall seconds and the timing basis."""
    from lfptensorpipe.resources.m6_tensor import UNIT_SECONDS

    slots = derive_global_compute_slots(cpu_percent, detected_cpu_count=12)[2]
    records = _table().get(tuple(p.profile.metric for p in points), ())
    compatible = [r for r in records if tuple(r.groups) == tuple(groups)]
    costs = _estimate_costs(points, groups, records)
    units = tuple(v / UNIT_SECONDS for v in costs) if costs else None
    matched = [r for r in compatible if r.points == points]
    same = next((r for r in matched if r.slots == slots), None)
    if same and mask_matches:
        return units, slots, same.seconds, "Measured median"
    if not costs:
        return units, slots, None, "Unavailable"
    # Prefer the closest measured combination with these algorithms. Interpolate
    # its slot curve and transfer only the compute portion to altered workloads.
    candidates = [
        r
        for r in compatible
        if all(
            a.profile.method == b.profile.method and a.profile.mode == b.profile.mode
            for a, b in zip(r.points, points)
        )
    ]
    schedule = {}
    target_cost = sum(costs)
    for budget in {r.slots for r in candidates}:
        nearest = min(
            (r for r in candidates if r.slots == budget),
            key=lambda r: (
                sum(_distance(a, b) for a, b in zip(r.points, points)),
                sum(
                    a.profile.details != b.profile.details
                    for a, b in zip(r.points, points)
                ),
                sum(abs(a.work - b.work) for a, b in zip(r.points, points)),
            ),
        )
        prior = _estimate_costs(nearest.points, groups, records)
        if prior and sum(prior) > 0:
            # Keep measured launch and publication overhead at this slot budget.
            schedule[budget] = max(
                0, nearest.seconds - nearest.compute
            ) + nearest.compute * target_cost / sum(prior)
    if schedule:
        seconds, outside = _blend(schedule, slots)
    else:
        seconds, outside = _component_runtime(points, groups, costs, slots)
    outside |= any(_single_cost(p)[1] for p in points)
    return (
        units,
        slots,
        seconds,
        ("Extrapolated" if outside else "Estimated from M6 benchmarks"),
    )
