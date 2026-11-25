"""Portable changes made during manual review of a preprocessing result."""

from collections import Counter
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def review_state(annotations, bads, first_time, *, precise=False):
    rows = []
    for ann in annotations:
        onset = float(ann["onset"]) - first_time
        duration = float(ann["duration"])
        rows.append(
            [
                onset if precise else round(onset, 6),
                duration if precise else round(duration, 6),
                str(ann["description"]),
                list(ann.get("ch_names", ())),
                dict(ann.get("extras") or {}),
            ]
        )
    return {"annotations": rows, "bads": sorted(bads)}


def current_state(raw, *, precise=False):
    return review_state(
        raw.annotations, raw.info["bads"], raw.first_time, precise=precise
    )


def _without_owned_edges(rows, step):
    from lfptensorpipe.preproc.filter import FILTER_EDGE_DESCRIPTION
    from lfptensorpipe.preproc.ecg_remover import (
        ANNOTATIONS_FILTER_EDGE_DESCRIPTION,
        ECG_FILTER_EDGE_DESCRIPTION,
    )

    owned = {
        "filter": FILTER_EDGE_DESCRIPTION,
        "annotations": ANNOTATIONS_FILTER_EDGE_DESCRIPTION,
        "ecg_artifact_removal": ECG_FILTER_EDGE_DESCRIPTION,
    }.get(step)
    return [row for row in rows if row[2] != owned]


def _review_counts(state, field, step):
    rows = state[field]
    if field == "annotations" and step:
        rows = _without_owned_edges(rows, step)
    return Counter(json.dumps(row, sort_keys=True) for row in rows)


def review_delta(before, after, *, step=None, persisted=None):
    change = {}
    for field in ("annotations", "bads"):
        left = _review_counts(before, field, step)
        right = _review_counts(after, field, step)
        unchanged = left & right
        if persisted is not None:
            # Even a retained row can move on its next FIF save. Reapply its
            # exact writer inputs if the saved observation changed.
            unchanged &= _review_counts(persisted, field, step)
        for action, difference in (
            ("add", right - unchanged),
            ("remove", left - unchanged),
        ):
            if difference:
                change[f"{action}_{field}"] = [
                    json.loads(s) for s in sorted(difference.elements())
                ]
    return change


def _fif_grid_bounds(low, high):
    """First and last representable float32 within a closed interval."""
    first, last = np.float32(low), np.float32(high)
    if float(first) < low:
        first = np.nextafter(first, np.float32(np.inf))
    if float(last) > high:
        last = np.nextafter(last, np.float32(-np.inf))
    return first, last


def _matches_fif_removal(candidate, observed, first_time):
    """Project a candidate onto the precision of a later FIF observation."""
    if candidate[2:] != observed[2:]:
        return False
    onset, duration = candidate[:2]
    wanted_onset, wanted_duration = observed[:2]
    if (round(onset, 6), round(duration, 6)) == (wanted_onset, wanted_duration):
        return True

    # Old change lists rounded each original value to six decimal places.
    half_us = 0.5e-6
    onset_error = half_us if onset == round(onset, 6) else 0.0
    duration_error = half_us if duration == round(duration, 6) else 0.0
    start = first_time + onset
    start_low = float(np.float32(start - onset_error))
    start_high = float(np.float32(start + onset_error))
    end_low = float(
        np.float32(start - onset_error + max(0.0, duration - duration_error))
    )
    end_high = float(np.float32(start + onset_error + duration + duration_error))

    # MNE dates the onset to a microsecond on reopening. A legacy observer
    # may also round it after subtracting a fractional first-sample time.
    observed_error = half_us + (
        half_us if wanted_onset == round(wanted_onset, 6) else 0.0
    )
    start_low, start_high = _fif_grid_bounds(
        max(start_low, first_time + wanted_onset - observed_error),
        min(start_high, first_time + wanted_onset + observed_error),
    )
    if start_low > start_high:
        return False
    # The reader subtracts two stored float32 endpoints in float32. Invert
    # that rounding to check whether an admissible end lies on the FIF grid.
    observed_error = half_us if wanted_duration == round(wanted_duration, 6) else 0.0
    dmin, dmax = _fif_grid_bounds(
        wanted_duration - observed_error, wanted_duration + observed_error
    )
    if dmin > dmax:
        return False
    min_diff = (float(np.nextafter(dmin, np.float32(-np.inf))) + float(dmin)) / 2
    max_diff = (float(np.nextafter(dmax, np.float32(np.inf))) + float(dmax)) / 2
    end_low, end_high = _fif_grid_bounds(
        max(end_low, float(start_low) + min_diff),
        min(end_high, float(start_high) + max_diff),
    )
    return end_low <= end_high


def has_unrecorded_plot_edits(entry):
    """A plot cannot reconstruct edits missing from an earlier manual review."""
    for event in reversed(entry.get("history") or [entry]):
        if (
            not event.get("completed")
            or "plot" not in str(event.get("message", "")).lower()
        ):
            break
        if "review_edits" not in event.get("params", {}):
            return True
    return False


def validate_review_changes(edits):
    """Validate portable change-list structure before any preprocessing step runs."""
    if not isinstance(edits, list):
        raise ValueError("Review edits must be a list of changes.")
    for change in edits:
        if not isinstance(change, dict) or set(change) - {
            "add_annotations",
            "remove_annotations",
            "add_bads",
            "remove_bads",
        }:
            raise ValueError("Invalid review change.")
        for key, entries in change.items():
            if not isinstance(entries, list):
                raise ValueError(f"{key} must be a list.")
            for item in entries:
                if key.endswith("_bads"):
                    if not isinstance(item, str):
                        raise ValueError(f"Invalid review channel: {item!r}")
                elif (
                    not isinstance(item, list)
                    or len(item) != 5
                    or not all(
                        type(x) in (int, float) and isfinite(x) for x in item[:2]
                    )
                    or item[1] < 0
                    or not isinstance(item[2], str)
                    or not isinstance(item[3], list)
                    or not all(isinstance(ch, str) for ch in item[3])
                    or not isinstance(item[4], dict)
                ):
                    raise ValueError(f"Invalid review annotation: {item!r}")


def user_review_changes(edits, step):
    """Retain the portable actions taken by the reviewer, in review order."""
    validate_review_changes(edits)
    result = []
    for change in edits:
        kept = {}
        for key, rows in change.items():
            if key.endswith("_annotations") and step:
                rows = _without_owned_edges(rows, step)
            if rows:
                kept[key] = rows
        if kept:
            result.append(kept)
    return result


def apply_edits(raw, edits, *, step=None):
    import mne
    from lfptensorpipe.preproc.filter import _set_annotations_from_attached_frame

    state = current_state(raw, precise=True)
    if isinstance(edits, dict):
        edits = [review_delta(current_state(raw), edits["accepted_state"], step=step)]
    edits = user_review_changes(edits, step)
    for change in edits:
        for key, entries in change.items():
            for item in entries:
                names = [item] if key.endswith("_bads") else item[3]
                if any(ch not in raw.ch_names for ch in names):
                    raise ValueError(f"Unknown review channel: {item!r}")
        for field in ("annotations", "bads"):
            removed, added = (
                change.get(action + field, []) for action in ("remove_", "add_")
            )
            for row in removed:
                if field == "annotations" and (
                    row not in state[field] or all(t == round(t, 6) for t in row[:2])
                ):
                    matches = [
                        r
                        for r in state[field]
                        if r == row or _matches_fif_removal(r, row, raw.first_time)
                    ]
                    if matches:
                        if any(r != matches[0] for r in matches):
                            raise ValueError(
                                f"Ambiguous reviewed annotation removal: {row}"
                            )
                        row = matches[0]
                if row not in state[field]:
                    raise ValueError(
                        f"A reviewed {field} entry no longer matches: {row}"
                    )
                state[field].remove(row)
            state[field].extend(added)
    rows = state["annotations"]
    annotations = mne.Annotations(
        onset=[r[0] + raw.first_time for r in rows],
        duration=[r[1] for r in rows],
        description=[r[2] for r in rows],
        ch_names=[r[3] for r in rows],
        extras=[r[4] for r in rows],
        orig_time=raw.annotations.orig_time,
    )
    stop = raw.n_times / raw.info["sfreq"]
    if all(0 <= row[0] and row[0] + row[1] <= stop for row in rows):
        # Raw.set_annotations crops via datetime (microseconds). In-bounds
        # reviewed rows are already in the attached frame, as in the GUI.
        raw.annotations.delete(np.arange(len(raw.annotations)))
        if len(annotations):
            raw.annotations.append(
                annotations.onset,
                annotations.duration,
                annotations.description,
                ch_names=annotations.ch_names,
                extras=annotations.extras,
            )
    else:
        _set_annotations_from_attached_frame(raw, annotations)
    raw.info["bads"] = sorted(set(state["bads"]))
    return edits


class ReviewReplayError(ValueError):
    """A saved review cannot be applied to the generated candidate."""


def write_reviewed_fif(
    path,
    write_candidate,
    review,
    *,
    resolver,
    step,
    source_step,
    mark_filter_edges=False,
):
    """Replay exported edits on the FIF as the reviewer saw it, before publication."""
    if not review:
        write_candidate(path)
        return {}

    import mne
    from ..shared.generation_lineage import accepted_result_generation_id
    from ..shared.runlog_store import read_run_log
    from .lineage import PreprocInputGenerationChanged
    from .paths import preproc_step_log_path
    from .steps.annotations import prepare_annotations_plot_review
    from .steps.ecg import prepare_ecg_plot_review

    log_path = preproc_step_log_path(resolver, step, create=False)
    previous = accepted_result_generation_id(read_run_log(log_path))
    # FIF round-trips annotation timing. Keep the candidate's split files
    # separate from the reviewed set that the step will promote.
    with TemporaryDirectory(dir=path.parent) as folder:
        candidate = Path(folder) / path.name
        write_candidate(candidate)
        with mne.io.read_raw_fif(candidate, preload=True, verbose="ERROR") as raw:
            try:
                edits = apply_edits(raw, review, step=step)
                prepare = {
                    "ecg_artifact_removal": prepare_ecg_plot_review,
                    "annotations": prepare_annotations_plot_review,
                }.get(step)
                output = (
                    prepare(
                        resolver,
                        raw,
                        source_step=source_step,
                        mark_filter_edges=mark_filter_edges,
                    )
                    if prepare
                    else raw
                )
            except ValueError as exc:
                raise ReviewReplayError(str(exc)) from exc
            try:
                output.save(
                    path,
                    fmt="double" if step == "signal_repair" else "single",
                    overwrite=True,
                    verbose="ERROR",
                )
            finally:
                if output is not raw:
                    output.close()
    if previous != accepted_result_generation_id(read_run_log(log_path)):
        raise PreprocInputGenerationChanged(
            f"The reviewed {step} changed during replay."
        )
    return {"review_edits": edits}


def plot_edits(raw_path, reviewed, step, prior, updates, *, persisted=None):
    """Isolate annotation edits from reversible repair-interval decisions."""
    import mne
    from lfptensorpipe.preproc.repair_review import repair_decisions, replay_repair

    with mne.io.read_raw_fif(
        raw_path, preload=step == "signal_repair", verbose="ERROR"
    ) as original:
        if step == "signal_repair" and "intervals" in updates:
            replay_repair(
                original, prior["intervals"], repair_decisions(updates["intervals"])
            )
        delta = review_delta(
            current_state(original, precise=True),
            current_state(reviewed, precise=True),
            step=step,
            persisted=persisted,
        )
    return [*prior.get("review_edits", []), *([delta] if delta else [])]


def stored_plot_edits(original_path, staged_path, reviewed, step, prior, updates):
    """Capture writer inputs and account for the exact observations on reopening."""
    import mne

    with mne.io.read_raw_fif(staged_path, preload=False, verbose="ERROR") as accepted:
        return plot_edits(
            original_path,
            reviewed,
            step,
            prior,
            updates,
            persisted=current_state(accepted, precise=True),
        )
