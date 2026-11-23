"""Cross-epoch MAD, valid coverage and review evidence in aligned coordinates."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import label, sum_labels

from .inspector_config import LETTERS, SIGNAL, evaluate, expression_program, logic
from .inspector_data import runs
from lfptensorpipe.lfp.burst.timeline import sample_cell_edges
from lfptensorpipe.lfp.mask.mask import union_intervals


def nan_stat(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="All-NaN|Mean of empty|Degrees of freedom",
        )
        return fn(*args, **kwargs)


def reference(values, minimum):
    values = np.where(np.isfinite(values), values, np.nan)
    median = nan_stat(np.nanmedian, values, axis=0)
    scale = 1.4826 * nan_stat(np.nanmedian, np.abs(values - median), axis=0)
    usable = (np.isfinite(values).sum(axis=0) >= minimum) & (scale > 0)
    return median, np.where(usable, scale, np.nan)


def signal_blocks(spans, channel, baseline=0.0):
    """Join continuous mapping slopes; keep distinct source occurrences separate."""
    import pandas as pd

    blocks = []
    for span in spans:
        if not len(span.time):
            continue
        occurrence = span.segment.occurrence
        points = span.boundaries[channel] if span.boundaries else ()
        cuts = sorted({0, *points, len(span.time)})
        for start, stop in zip(cuts, cuts[1:]):
            if not blocks or blocks[-1][0] != occurrence or start in points:
                blocks.append((occurrence, [], [], []))
            observed = span.time if span.sample_time is None else span.sample_time
            keep = np.isfinite(observed[start:stop])
            blocks[-1][1].append(observed[start:stop][keep])
            blocks[-1][2].append(span.end[start:stop][keep])
            blocks[-1][3].append(span.values[channel, start:stop][keep])
    result = []
    for _, xs, ends, ys in blocks:
        x, end, y = np.concatenate(xs), np.concatenate(ends), np.concatenate(ys)
        if baseline:
            for a, b in runs(np.isfinite(y)):
                series = pd.Series(y[a:b], index=pd.to_timedelta(x[a:b], unit="s"))
                med = series.rolling(
                    pd.Timedelta(seconds=baseline), center=True, closed="both"
                ).median()
                y[a:b] -= med.to_numpy()
        result.append((x, end, y))
    return result


def interpolate(blocks, x):
    out = np.full(len(x), np.nan)
    for times, _, values in blocks:
        for a, b in runs(np.isfinite(values)):
            tolerance = (
                32 * np.finfo(float).eps * max(1.0, abs(times[a]), abs(times[b - 1]))
            )
            start = np.searchsorted(x, times[a] - tolerance)
            stop = np.searchsorted(x, times[b - 1] + tolerance, side="right")
            if stop > start:
                out[start:stop] = np.interp(x[start:stop], times[a:b], values[a:b])
    return out


def evidence(data, epoch, channel, start, end, **extra):
    return {
        "channel": channel,
        "time": [float(start), float(end)],
        "unit": data.unit,
        "source_intervals_s": data.source_intervals(epoch, start, end),
        **extra,
    }


def missing_intervals(left, right, invalid, window):
    low, high = window
    return [
        [float(max(low, left[a])), float(min(high, right[b - 1]))]
        for a, b in runs(invalid & (right > low) & (left < high))
    ]


def coverage_result(percent, rule, data, epoch, channel, window, **extra):
    if not np.isfinite(percent):
        return None, []
    percent = float(np.clip(round(percent, 12), 0, 100))
    hit = percent < rule["coverage"]
    if not hit:
        return False, []
    return True, [
        evidence(
            data,
            epoch,
            channel,
            *window,
            coverage_percent=percent,
            interval_kind="evaluated window",
            **extra,
        )
    ]


def signal_scores(data, signal, channel, rule, window):
    lo, hi = window
    name = data.signal_channels[channel]
    if rule["method"] == "Valid coverage":
        scores = []
        for i, spans in enumerate(signal):
            expected = sum(b - a for a, b in data.source_intervals(i, lo, hi))
            valid = sum(
                np.sum(
                    np.maximum(0, np.minimum(s.end, hi) - np.maximum(s.time, lo))
                    * np.isfinite(s.values[channel])
                )
                * s.seconds_per_unit
                for s in spans
            )
            outcome, details = coverage_result(
                100 * valid / expected if expected else np.nan,
                rule,
                data,
                i,
                name,
                window,
            )
            if outcome:
                details[0]["missing_time"] = [
                    gap
                    for span in spans
                    for gap in missing_intervals(
                        span.time, span.end, ~np.isfinite(span.values[channel]), window
                    )
                ]
            scores.append((outcome, details))
        return scores

    blocks = [signal_blocks(s, channel, rule["baseline"]) for s in signal]
    scores = []
    common = {}
    if blocks and all(len(parts) == len(blocks[0]) for parts in blocks):
        for k, (times, _, _) in enumerate(blocks[0]):
            if all(
                len(parts[k][0]) == len(times)
                and np.allclose(parts[k][0], times, rtol=0, atol=1e-12)
                for parts in blocks
            ):
                # Fixed-length epochs share a template; variable warps retain their native samples.
                common[k] = reference(
                    np.stack([parts[k][2] for parts in blocks]), rule["minimum_epochs"]
                )
    for i, target in enumerate(blocks):
        details, assessed = [], False
        for k, (x, end, y) in enumerate(target):
            selected = ((x < hi) | ((x == hi) & (end == x))) & ((end > lo) | (x == lo))
            times, ends, values = x[selected], end[selected], y[selected]
            if not times.size:
                continue
            if k in common:
                median, scale = (arr[selected] for arr in common[k])
            else:
                others = np.array(
                    [
                        interpolate(b, times) if j != i else values
                        for j, b in enumerate(blocks)
                    ]
                )
                median, scale = reference(others, rule["minimum_epochs"])
            z = (values - median) / scale
            assessed |= bool(np.isfinite(z).any())
            for a, b in runs(np.abs(z) > rule["threshold"]):
                start, stop = max(lo, times[a]), min(hi, ends[b - 1])
                if stop - start + 1e-12 >= (rule["duration"] or 0.0):
                    details.append(
                        evidence(
                            data,
                            i,
                            name,
                            start,
                            stop,
                            peak_abs_z=float(np.max(np.abs(z[a:b]))),
                        )
                    )
        scores.append((True if details else False if assessed else None, details))
    return scores


def cell_bounds(centers, *, support=None, left_edges=False):
    centers = np.asarray(centers, float)
    if len(centers) == 1:
        edges = np.array(support or [centers[0] - 0.5, centers[0] + 0.5])
    elif left_edges:
        edges = np.r_[centers, support[1]]
    else:
        mids = (centers[1:] + centers[:-1]) / 2
        edges = np.r_[
            centers[0] - (mids[0] - centers[0]),
            mids,
            centers[-1] + (centers[-1] - mids[-1]),
        ]
    if support is not None:
        edges = np.clip(edges, *support)
    return edges


def log_frequency_cells(grid, selection):
    """Return selected positive centers and their band-clipped cell edges in Hz."""
    centers = np.asarray(grid.freq, dtype=float)
    first = np.searchsorted(centers, 0.0, side="right")
    selected = grid.frequency_mask(selection)
    selected[:first] = False
    edges = np.full(centers.size + 1, np.nan)
    positive = centers[first:]
    if positive.size:
        edges[first:] = np.r_[
            positive[0], np.sqrt(positive[:-1] * positive[1:]), positive[-1]
        ]
        if isinstance(selection, dict) and "hz" in selection:
            np.clip(edges, *selection["hz"], out=edges)
    return selected, edges


def _frequency_width(lo, hi, log_scale):
    hi = np.maximum(lo, hi)
    return np.log2(hi / lo) if log_scale else hi - lo


def frequency_support(grid, selection, *, log_scale=False):
    """Measure numeric MAD cells in octaves and numeric coverage cells in Hz."""
    if not grid.numeric:
        return grid.frequency_mask(selection).astype(float), np.arange(len(grid.freq))
    groups = np.zeros(len(grid.freq), int)
    if log_scale:
        chosen, edges = log_frequency_cells(grid, selection)
    else:
        chosen = grid.frequency_mask(selection)
        edges = cell_bounds(grid.freq)
        if isinstance(selection, dict) and "hz" in selection:
            np.clip(edges, *selection["hz"], out=edges)

    lo, hi = edges[:-1][chosen], edges[1:][chosen]
    # A lone measured center with no bandwidth has a time-only cluster fraction.
    if log_scale and lo.size == 1 and lo[0] == hi[0]:
        return chosen.astype(float), groups
    widths = _frequency_width(lo, hi, log_scale)
    intervals = union_intervals(grid.notches) if log_scale else grid.notches
    for a, b in intervals:
        widths -= _frequency_width(np.maximum(lo, a), np.minimum(hi, b), log_scale)
    for _, end in grid.notches:
        # A point notch also separates otherwise adjacent centers.
        groups += grid.freq > end
    weights = np.zeros(len(grid.freq))
    weights[chosen] = np.maximum(0, widths)
    return weights, groups


def burst_coverage(data, epoch, channel, grid, freq_weights, window, missing=None):
    if data.native_burst is None:
        return np.nan
    tensor, names, bands, times = data.native_burst
    edges = sample_cell_edges(times)
    c = names.index(channel)
    valid = total = 0.0
    for s in data.mappings[epoch]:
        a, b = max(window[0], s.target_start), min(window[1], s.target_end)
        if b <= a:
            continue
        scale = (s.source_end - s.source_start) / (s.target_end - s.target_start)
        src_lo = s.source_start + (a - s.target_start) * scale
        src_hi = s.source_start + (b - s.target_start) * scale
        i = max(0, np.searchsorted(edges, src_lo, side="right") - 1)
        j = min(len(times), np.searchsorted(edges, src_hi))
        durations = np.maximum(
            0, np.minimum(edges[i + 1 : j + 1], src_hi) - np.maximum(edges[i:j], src_lo)
        )
        if missing is not None:
            bands_selected = [
                bands.index(str(grid.freq[row])) for row in np.flatnonzero(freq_weights)
            ]
            invalid = (~np.isfinite(tensor[c, bands_selected, i:j])).any(axis=0)
            gap_bounds = missing_intervals(
                edges[i:j], edges[i + 1 : j + 1], invalid, (src_lo, src_hi)
            )
            missing.extend(
                [
                    [
                        s.target_start + (a - s.source_start) / scale,
                        s.target_start + (b - s.source_start) / scale,
                    ]
                    for a, b in gap_bounds
                ]
            )
        for row in np.flatnonzero(freq_weights):
            mask = np.isfinite(tensor[c, bands.index(str(grid.freq[row])), i:j])
            valid += float(durations[mask].sum() / scale) * freq_weights[row]
            total += (b - a) * freq_weights[row]
    return 100 * valid / total if total else np.nan


def metric_scores(data, metric, channel, rule, window):
    grid = data.grid(metric)
    values = grid.values[:, channel]
    name = grid.channels[channel]
    fw, fgroups = frequency_support(
        grid, rule["frequency"], log_scale=rule["method"] == "MAD"
    )
    edges = cell_bounds(
        grid.time, support=data.bounds, left_edges=data.axis != "percent"
    )
    lo, hi = np.maximum(edges[:-1], window[0]), np.minimum(edges[1:], window[1])
    area = fw[:, None] * np.maximum(0, hi - lo)
    expected = float(area.sum())
    if not expected:
        return [(None, []) for _ in values]
    if rule["method"] == "Valid coverage":
        rows = []
        for i, plane in enumerate(values):
            missing = []
            percent = (
                burst_coverage(data, i, name, grid, fw, window, missing)
                if metric == "burst"
                else float(100 * area[np.isfinite(plane)].sum() / expected)
            )
            active = grid.freq[fw > 0]
            detail = (
                {"frequency": [float(active[0]), float(active[-1])]}
                if grid.numeric
                else {"bins": list(map(str, active))}
            )
            outcome, details = coverage_result(
                percent, rule, data, i, name, window, **detail
            )
            if outcome:
                if metric != "burst":
                    invalid = (~np.isfinite(plane[fw > 0])).any(axis=0)
                    for seg in data.mappings[i]:
                        bounds = (
                            max(window[0], seg.target_start),
                            min(window[1], seg.target_end),
                        )
                        if bounds[0] < bounds[1]:
                            missing.extend(missing_intervals(lo, hi, invalid, bounds))
                details[0]["missing_time"] = missing
            rows.append((outcome, details))
        return rows
    median, scale = reference(
        np.where(np.isfinite(values), values, np.nan), rule["minimum_epochs"]
    )
    scores = []
    for i, plane in enumerate(values):
        z = (plane - median) / scale
        valid = np.isfinite(z) & (area > 0)
        details = []
        time_groups = np.zeros(len(grid.time), int)
        for seg in data.mappings[i][1:]:
            if seg.occurrence != data.mappings[i][0].occurrence:
                time_groups += grid.time >= seg.target_start
        for group in np.unique(fgroups[fw > 0]):
            for part in np.unique(time_groups):
                domain = valid & (fgroups == group)[:, None] & (time_groups == part)
                for direction in (1, -1):
                    regions, count = label(domain & (direction * z > rule["threshold"]))
                    ids = np.arange(1, count + 1)
                    sizes = sum_labels(area, regions, index=ids)
                    for k in ids[sizes + 1e-12 >= expected * rule["area"] / 100]:
                        f, t = np.nonzero(regions == k)
                        detail = (
                            {
                                "frequency": [
                                    float(grid.freq[f.min()]),
                                    float(grid.freq[f.max()]),
                                ]
                            }
                            if grid.numeric
                            else {"bins": [str(grid.freq[f[0]])]}
                        )
                        details.append(
                            evidence(
                                data,
                                i,
                                name,
                                lo[t.min()],
                                hi[t.max()],
                                direction="high" if direction == 1 else "low",
                                cluster_area_percent=float(
                                    100 * sizes[k - 1] / expected
                                ),
                                **detail,
                            )
                        )
        scores.append((True if details else False if valid.any() else None, details))
    return scores


def run_qc(data, config):
    program = expression_program(config["expression"], len(config["rules"]))
    outcomes = [{} for _ in data.epochs]
    resolved = {}
    signal = None
    for idx, rule in enumerate(config["rules"]):
        letter = LETTERS[idx]
        if letter not in program:
            continue
        data.validate_rule(rule)
        metric = rule["metric"]
        all_channels = data.channels(metric)
        selected = all_channels if rule["channels"] == "All" else rule["channels"]
        lo, hi = data.bounds
        if rule["time"] != "All":
            lo, hi = max(lo, rule["time"][0]), min(hi, rule["time"][1])
        resolved[letter] = {
            "channels": selected,
            "time": [lo, hi],
            "axis": data.axis,
            "rule": rule,
        }
        if metric != SIGNAL:
            grid = data.grid(metric)
            if grid.numeric and rule["method"] == "MAD":
                selected_bins, _ = log_frequency_cells(grid, rule["frequency"])
                resolved[letter]["cluster_frequency_scale"] = "log"
            else:
                selected_bins = grid.frequency_mask(rule["frequency"])
            bins = grid.freq[selected_bins]
            resolved[letter]["frequency_bins"] = bins.tolist()
            if grid.bands:
                resolved[letter]["band_segments_hz"] = {
                    str(b): grid.bands[str(b)] for b in bins if str(b) in grid.bands
                }
        per_channel = []
        for channel in selected:
            c = all_channels.index(channel)
            if hi <= lo:
                series = [(None, []) for _ in outcomes]
            elif metric == SIGNAL:
                if signal is None:
                    signal = data.read_signal()
                series = signal_scores(data, signal, c, rule, (lo, hi))
            else:
                series = metric_scores(data, metric, c, rule, (lo, hi))
            per_channel.append(series)
        for i, row in enumerate(outcomes):
            hits = [scores[i][0] for scores in per_channel]
            row[letter] = {
                "hit": (
                    logic("AND" if rule["match"] == "All" else "OR", hits)
                    if hits
                    else None
                ),
                "evidence": [e for scores in per_channel for e in scores[i][1]],
            }
    return {
        "config": config,
        "resolved": resolved,
        "epochs": [
            {
                "flag": evaluate(program, {k: v["hit"] for k, v in row.items()}),
                "rules": row,
            }
            for row in outcomes
        ],
    }
