"""Pair accepted feature values and aligned signal coverage on original events."""

from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from lfptensorpipe.cli import check_trial, record_context
from lfptensorpipe.io.atomic_file import write_json_atomically
from lfptensorpipe.io.pkl_io import load_pkl
from lfptensorpipe.utils.transforms import VALUE_TRANSFORM_POLICY_KEY
from .alignment.epoch_view import epoch_key
from .alignment.paths import alignment_trial_raw_table_path
from .cli import RunFailed
from .features.generation import (
    accepted_feature_artifact_paths,
    features_derivatives_log_path,
)
from .features.phase_time import aligned_phase_support
from .path_resolver import PathResolver
from .shared.atomic_outputs import AtomicOutputSet
from .shared.generation_lineage import accepted_result_generation_id
from .shared.runlog_store import read_run_log

KEY = (
    "subject",
    "record",
    "trial",
    "source_epoch",
    "channel",
    "metric",
    "output",
    "reducer",
    "band",
    "phase",
    "frequency",
    "time",
)
COVER_KEY = (
    "subject",
    "record",
    "trial",
    "source_epoch",
    "channel",
    "metric",
    "band",
    "phase",
    "time_unit",
)


def encoded(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _open_run(folder):
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if not isinstance(run, dict) or run.get("status") != "completed":
        raise RunFailed(f"Only completed runs have comparable Features: {folder}")
    if any(
        not isinstance(run.get(k), str) or Path(run[k]).name != run[k]
        for k in ("subject", "record", "trial")
    ):
        raise ValueError(f"Invalid record identity in {folder}/run.json")
    ctx = record_context(
        SimpleNamespace(
            subject_path=folder / "derivatives" / "lfptensorpipe" / run["subject"],
            record=run["record"],
        )
    )
    check_trial(ctx, "features", run["trial"])
    if ctx.project_root != folder:
        raise ValueError("The selected record must belong to this run.")
    resolver = PathResolver(ctx)
    paths = accepted_feature_artifact_paths(resolver, trial_slug=run["trial"])
    entry = read_run_log(
        features_derivatives_log_path(resolver, trial_slug=run["trial"])
    )
    if (
        not paths
        or accepted_result_generation_id(entry)
        != run["stages"]["features"]["generation"]
    ):
        raise RunFailed(f"Features have changed since the run was published: {folder}")
    return run, resolver, paths


def _text(v):
    if isinstance(v, (tuple, list, np.ndarray)):
        return encoded(list(v))
    return "" if pd.isna(v) else str(v)


def _cells(v, kind):
    if isinstance(v, pd.DataFrame):
        for f, row in v.iterrows():
            for t, n in row.items():
                yield _text(f), _text(t), n
    elif isinstance(v, pd.Series):
        for coord, n in v.items():
            yield (_text(coord) if kind == "spectral" else ""), (
                _text(coord) if kind == "trace" else ""
            ), n
    else:
        yield "", "", v


def _window(axes, field, name):
    return sorted((r["start"], r["end"]) for r in axes[field] if r["name"] == name)


def _common_definition(effective, metric):
    tensor = effective["tensor"]
    owner = metric if metric in tensor["metric_params"] else "periodic_aperiodic"
    params = tensor["metric_params"].get(owner, {})
    return {
        "tensor": {
            k: v
            for k, v in params.items()
            if k not in {"selected_channels", "selected_pairs"}
        },
        "alignment": {
            "method": effective["alignment"]["method"],
            "method_params": {
                k: v
                for k, v in effective["alignment"]["method_params"].items()
                if k not in {"annotations", "drop_bad", "drop_fields"}
            },
        },
    }


def _feature_values(run, paths, seconds):
    effective = run["effective"]
    feature = effective["features"]
    columns = feature["execution"]["collapse"].get(
        "out_cols", {"band": "Band", "phase": "Phase", "value": "Value"}
    )
    values = defaultdict(list)
    definitions = {}
    for metric, path in paths:
        frame = load_pkl(path)
        reducer, kind = path.stem.rsplit("-", 1)
        axes = feature["axes_by_metric"][metric]
        policy = frame.attrs.get(
            VALUE_TRANSFORM_POLICY_KEY,
            feature["execution"]["metrics"][metric]["value_policy"],
        )
        unit_map = feature["execution"]["metrics"][metric]["units"]
        common = _common_definition(effective, metric)
        for row in frame.to_dict("records"):
            source = row.get("SourceEpoch")
            if not isinstance(source, str):
                raise RunFailed(
                    f"An accepted feature lacks its original event identity: {path}"
                )
            band, phase = (_text(row.get(columns[k])) for k in ("band", "phase"))
            unit = _text(row.get("Unit")) or unit_map.get(reducer) or "unspecified"
            spec_key = (metric, kind, reducer, band, phase, unit)
            if spec_key not in definitions:
                definition = {**common, "value_policy": policy, "unit": unit}
                if kind != "raw":
                    definition["reduction"] = {
                        k: v
                        for k, v in feature["execution"]["collapse"].items()
                        if k
                        in ("inclusive", "freq_interval_mode", "time_interval_mode")
                    }
                if band:
                    definition["band_hz"] = _window(axes, "bands", band)
                if phase:
                    windows = frame.attrs.get("phase_windows", axes)
                    unit_time = windows.get("time_unit", "percent")
                    bounds = _window(windows, "times", phase)
                    if seconds is not None and unit_time == "percent":
                        bounds = [
                            (a * seconds / 100, b * seconds / 100) for a, b in bounds
                        ]
                        unit_time = "s"
                    definition.update(phase_windows=bounds, time_unit=unit_time)
                definitions[spec_key] = encoded(definition)
            prefix = (
                run["subject"],
                run["record"],
                run["trial"],
                source,
                _text(row["Channel"]),
                metric,
                kind,
                reducer,
                band,
                phase,
            )
            v = row.get(columns["value"], row.get("Value"))
            values[prefix].append(
                (v, definitions[spec_key], unit, policy["feature_storage_domain"])
            )
    return values, definitions


def _row_values(rows, kind):
    samples = defaultdict(list)
    for value, definition, unit, domain in rows:
        for freq, time, number in _cells(value, kind):
            n = float(number) if number is not None and number is not pd.NA else np.nan
            samples[(freq, time)].append((n, definition, unit, domain))
    return samples


def _paired_coordinates(
    left, right, paths_left, paths_right, defs_left, defs_right, durations
):
    sides = [
        {(metric, p.stem): (metric, p) for metric, p in paths}
        for paths in (paths_left, paths_right)
    ]
    for artifact in sorted(sides[0].keys() | sides[1].keys()):
        tables = []
        for run, side, definitions, seconds in zip(
            (left, right), sides, (defs_left, defs_right), durations
        ):
            rows, defs = _feature_values(
                run, [side[artifact]] if artifact in side else [], seconds
            )
            definitions.update(defs)
            tables.append(rows)
        for prefix in sorted(tables[0].keys() | tables[1].keys()):
            a, b = [_row_values(table.get(prefix, []), prefix[6]) for table in tables]
            duplicate = any(len(table.get(prefix, [])) > 1 for table in tables)
            for coords in sorted(a.keys() | b.keys()):
                yield (*prefix, *coords), a.get(coords, []), b.get(
                    coords, []
                ), duplicate


def _phase_ranges(axes, duration, unit):
    windows = defaultdict(list)
    for row in axes["times"]:
        start, end = row["start"], row["end"]
        if unit == "s" and axes.get("time_unit", "percent") == "percent":
            start, end = start * duration / 100, end * duration / 100
        windows[row["name"]].append((start, end))
    from lfptensorpipe.tabular.grid import _merge_intervals

    return {name: _merge_intervals(spans) for name, spans in windows.items()}


def _coverage(run, resolver, metric):
    """Count aligned cells, including masked cells, in each configured band/phase."""
    result = {}
    config = run["effective"]["features"]
    method = run["effective"]["alignment"]["method"]
    axes = config["axes_by_metric"].get(metric)
    if axes is None:
        return result
    frame = load_pkl(
        alignment_trial_raw_table_path(
            resolver, trial_slug=run["trial"], metric_key=metric
        )
    )
    support = aligned_phase_support(frame, method)
    unit = "s" if support is not None else "percent"
    duration = support[1] - support[0] if support else 100.0
    windows = _phase_ranges(axes, duration, unit)
    collapse = config["execution"]["collapse"]
    inclusive = collapse["inclusive"]
    for row in frame.to_dict("records"):
        v = row["Value"]
        ts = np.asarray(v.columns, dtype=float)
        if support:
            ts = ts - support[0]
        band_coords = {}
        if axes["bands"]:
            from lfptensorpipe.tabular.grid import resolve_intervals

            freq = np.asarray(v.index, dtype=float)
            for band in axes["bands"]:
                spans = resolve_intervals(
                    [[band["start"], band["end"]]],
                    freq,
                    mode=collapse["freq_interval_mode"],
                )
                band_coords[band["name"]] = [
                    i
                    for i, x in enumerate(freq)
                    if any(
                        lo <= x and (x <= hi if inclusive else x < hi)
                        for lo, hi in spans
                    )
                ]
        else:
            band_coords = {str(f): [i] for i, f in enumerate(v.index)}
        for band, indices in band_coords.items():
            for phase, ranges in windows.items():
                key = (
                    run["subject"],
                    run["record"],
                    run["trial"],
                    row["SourceEpoch"],
                    _text(row["Channel"]),
                    metric,
                    band,
                    phase,
                    unit,
                )
                cells = {
                    (str(v.index[i]), float(t)): bool(np.isfinite(v.iat[i, j]))
                    for j, t in enumerate(ts)
                    if any(
                        lo <= t and (t <= hi if inclusive else t < hi)
                        for lo, hi in ranges
                    )
                    for i in indices
                }
                edges = np.append(ts[1:], duration)
                widths = {
                    float(t): sum(
                        max(0.0, min(end, hi) - max(t, lo)) for lo, hi in ranges
                    )
                    for t, end in zip(ts, edges)
                    if any(lo <= t <= hi for lo, hi in ranges)
                }
                result[key] = None if key in result else (cells, widths)
    return result


def _csv_writer(stack, path, columns):
    stream = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    return writer


def _write_coverage(left, right, left_resolver, right_resolver, writer):
    metrics = (
        left["effective"]["features"]["axes_by_metric"].keys()
        | right["effective"]["features"]["axes_by_metric"].keys()
    )
    same = True
    for metric in sorted(metrics):
        a, b = _coverage(left, left_resolver, metric), _coverage(
            right, right_resolver, metric
        )
        same &= a == b
        for key in sorted(a.keys() | b.keys()):
            row = dict(zip(COVER_KEY, key))
            if (key in a and a[key] is None) or (key in b and b[key] is None):
                writer.writerow({**row, "status": "ambiguous"})
                same = False
                continue
            (x, x_width), (y, y_width) = a.get(key, ({}, {})), b.get(key, ({}, {}))
            common = x.keys() & y.keys()
            valid_common = {k for k in common if x[k] and y[k]}
            xt = {t for (_, t), finite in x.items() if finite}
            yt = {t for (_, t), finite in y.items() if finite}
            ct = {t for _, t in valid_common}
            writer.writerow(
                {
                    **row,
                    "status": "paired" if key in a and key in b else "one_run_only",
                    "reference_cells": len(x),
                    "candidate_cells": len(y),
                    "reference_valid": sum(x.values()),
                    "candidate_valid": sum(y.values()),
                    "common_cells": len(common),
                    "common_valid": len(valid_common),
                    "reference_valid_times": len(xt),
                    "candidate_valid_times": len(yt),
                    "common_valid_times": len(ct),
                    "reference_valid_duration": sum(x_width[t] for t in xt),
                    "candidate_valid_duration": sum(y_width[t] for t in yt),
                    "common_valid_duration": (
                        sum(x_width[t] for t in ct) if x_width == y_width else None
                    ),
                }
            )
    return same


def _decision_set(run):
    rows = run["effective"]["alignment"]["epoch_selection"]["decisions"]
    return Counter((epoch_key(r), r["include"]) for r in rows)


def _numerical_settings(run):
    eff = run["effective"]
    return {
        "preprocess": eff["preprocess"],
        "tensor": eff["tensor"],
        "alignment": {
            k: v for k, v in eff["alignment"].items() if k != "epoch_selection"
        },
        "features": {k: v for k, v in eff["features"].items() if k != "active_metric"},
    }


def compare_runs(reference, candidate, output, *, mode, atol, rtol):
    reference, candidate, output = (
        p.expanduser().resolve() for p in (reference, candidate, output)
    )
    if reference == candidate:
        raise ValueError("Select two independently published runs.")
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError("Comparison tolerances must be finite and nonnegative.")
    if any(output.is_relative_to(p) for p in (reference, candidate)):
        raise ValueError("Choose a report directory outside both input runs.")
    left, lr, lp = _open_run(reference)
    right, rr, rp = _open_run(candidate)
    same_input = all(
        left["input"].get(k) == right["input"].get(k)
        for k in ("project", "subject", "record", "raw_generation", "localization")
    )
    if not same_input or left["trial"] != right["trial"]:
        raise ValueError(
            "A paired comparison requires the same accepted Raw, localization and trial."
        )
    cols = [
        (
            *KEY,
            "unit",
            "storage_domain",
            "reference",
            "candidate",
            "difference",
            "both_finite",
            "same_missingness",
            "within_tolerance",
        ),
        (*KEY, "reason", "reference_rows", "candidate_rows"),
        (
            *COVER_KEY,
            "status",
            "reference_cells",
            "candidate_cells",
            "reference_valid",
            "candidate_valid",
            "common_cells",
            "common_valid",
            "reference_valid_times",
            "candidate_valid_times",
            "common_valid_times",
            "reference_valid_duration",
            "candidate_valid_duration",
            "common_valid_duration",
        ),
    ]
    paths = [
        output / name
        for name in ("paired.csv", "unmatched.csv", "coverage.csv", "summary.json")
    ]
    if any(p.exists() for p in paths):
        raise ValueError("Choose a new comparison output directory.")
    with AtomicOutputSet(paths) as files:
        with ExitStack() as stack:
            writer, missing_writer, coverage_writer = [
                _csv_writer(stack, files.staged_path(path), columns)
                for path, columns in zip(paths, cols)
            ]
            report = _report(
                left,
                right,
                lr,
                rr,
                lp,
                rp,
                writer,
                missing_writer,
                coverage_writer,
                mode,
                atol,
                rtol,
            )
        report.update(reference=str(reference), candidate=str(candidate))
        write_json_atomically(files.staged_path(paths[-1]), report)
        files.commit()
    return report["repeat_pass"] if mode == "repeat" else bool(report["paired_finite"])


def _report(
    left,
    right,
    lr,
    rr,
    lp,
    rp,
    writer,
    missing_writer,
    coverage_writer,
    mode,
    atol,
    rtol,
):
    from .features.phase_time import trial_seconds_duration

    durations = [
        trial_seconds_duration(resolver, run["trial"])
        for resolver, run in ((lr, left), (rr, right))
    ]
    defs_a, defs_b = {}, {}
    stats = defaultdict(list)
    n_paired = n_unmatched = n_valid = n_changed_missing = n_outside = 0
    for key, av, bv, duplicate in _paired_coordinates(
        left, right, lp, rp, defs_a, defs_b, durations
    ):
        record = dict(zip(KEY, key))
        if duplicate or len(av) != 1 or len(bv) != 1:
            reason = (
                "ambiguous"
                if duplicate or len(av) > 1 or len(bv) > 1
                else ("reference_only" if av else "candidate_only")
            )
        elif av[0][1] != bv[0][1]:
            reason = "different_definition"
        else:
            reason = None
        if reason:
            n_unmatched += 1
            missing_writer.writerow(
                {
                    **record,
                    "reason": reason,
                    "reference_rows": len(av),
                    "candidate_rows": len(bv),
                }
            )
            continue
        x, _, unit, domain = av[0]
        y = bv[0][0]
        valid = np.isfinite(x) and np.isfinite(y)
        delta = within = None
        if valid:
            n_valid += 1
            delta = y - x
            within = abs(delta) <= atol + rtol * abs(x)
            n_outside += int(not within)
            stats[(key[5], key[6], key[7], unit)].append(delta)
        same_missingness = bool(valid or (np.isnan(x) and np.isnan(y)))
        n_changed_missing += int(not same_missingness)
        n_paired += 1
        writer.writerow(
            {
                **record,
                "unit": unit,
                "storage_domain": domain,
                "reference": x,
                "candidate": y,
                "difference": delta,
                "both_finite": bool(valid),
                "same_missingness": same_missingness,
                "within_tolerance": within,
            }
        )
    same_coverage = _write_coverage(left, right, lr, rr, coverage_writer)
    da, db = _decision_set(left), _decision_set(right)
    left_settings, right_settings = (_numerical_settings(run) for run in (left, right))
    from .features.phase_time import same_phase_windows

    a_axes = left_settings["features"]["axes_by_metric"]
    b_axes = right_settings["features"]["axes_by_metric"]
    same_settings = (
        all(
            left_settings[k] == right_settings[k]
            for k in ("preprocess", "tensor", "alignment")
        )
        and left_settings["features"]["execution"]
        == right_settings["features"]["execution"]
        and durations[0] == durations[1]
        and a_axes.keys() == b_axes.keys()
        and all(
            a_axes[m]["bands"] == b_axes[m]["bands"]
            and same_phase_windows(a_axes[m], b_axes[m], durations[0])
            for m in a_axes
        )
    )
    independent = all(
        left["stages"][s]["generation"] != right["stages"][s]["generation"]
        for s in left["stages"].keys() & right["stages"].keys()
        if s != "raw"
        and left["stages"][s].get("generation")
        and right["stages"][s].get("generation")
    )

    def retention(d):
        return {
            "candidates": sum(d.values()),
            "included": sum(n for (_, picked), n in d.items() if picked),
            "unique_candidates": len({key for key, _ in d}),
            "unique_included": len({key for key, pick in d if pick}),
        }

    repeat_pass = bool(
        n_valid
        and not n_unmatched
        and not n_changed_missing
        and not n_outside
        and same_settings
        and da == db
        and same_coverage
        and independent
    )
    summary = {
        "mode": mode,
        "software": {"reference": left["software"], "candidate": right["software"]},
        "paired_values": n_paired,
        "paired_finite": n_valid,
        "unmatched_keys": n_unmatched,
        "missingness_differences": n_changed_missing,
        "outside_tolerance": n_outside,
        "atol": atol,
        "rtol": rtol,
        "same_effective_settings": same_settings,
        "same_decisions": da == db,
        "same_valid_coverage": same_coverage,
        "independent_generations": independent,
        "reference_epochs": retention(da),
        "candidate_only_epochs": sorted({k for k, _ in db} - {k for k, _ in da}),
        "reference_only_epochs": sorted({k for k, _ in da} - {k for k, _ in db}),
        "excluded_selections": {
            side: record.get("excluded_selections", {})
            for side, record in (("reference", left), ("candidate", right))
        },
        "candidate_epochs": retention(db),
        "common_included_epochs": len(
            {k for k, p in da if p} & {k for k, p in db if p}
        ),
        "repeat_pass": repeat_pass if mode == "repeat" else None,
        "differences_by_output": [
            {
                "metric": m,
                "output": o,
                "reducer": r,
                "unit": u,
                "finite_pairs": len(nums),
                "median_difference": float(np.median(nums)),
                "max_absolute_difference": float(np.max(np.abs(nums))),
            }
            for (m, o, r, u), nums in stats.items()
        ],
        "effective_settings": {
            "reference": _numerical_settings(left),
            "candidate": _numerical_settings(right),
        },
        "definitions": {
            side: [
                {
                    "metric": k[0],
                    "output": k[1],
                    "reducer": k[2],
                    "band": k[3],
                    "phase": k[4],
                    **json.loads(val),
                }
                for k, val in defs.items()
            ]
            for side, defs in (("reference", defs_a), ("candidate", defs_b))
        },
    }
    return summary
