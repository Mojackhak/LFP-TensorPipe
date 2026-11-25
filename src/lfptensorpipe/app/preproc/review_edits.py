"""Portable changes made during manual review of a preprocessing result."""

from collections import Counter
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

def review_state(annotations, bads, first_time):
    rows = []
    for ann in annotations:
        rows.append(
            [
                round(float(ann["onset"]) - first_time, 6),
                round(float(ann["duration"]), 6),
                str(ann["description"]),
                list(ann.get("ch_names", ())),
                dict(ann.get("extras") or {}),
            ]
        )
    return {"annotations": rows, "bads": sorted(bads)}


def current_state(raw):
    return review_state(raw.annotations, raw.info["bads"], raw.first_time)


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


def review_delta(before, after, *, step=None):
    change = {}
    for field in ("annotations", "bads"):
        left, right = [
            Counter(
                json.dumps(x, sort_keys=True)
                for x in (
                    _without_owned_edges(state[field], step)
                    if field == "annotations" and step
                    else state[field]
                )
            )
            for state in (before, after)
        ]
        for action, counts in (("add", right - left), ("remove", left - right)):
            if counts:
                change[f"{action}_{field}"] = [
                    json.loads(s) for s in sorted(counts.elements())
                ]
    return change


def _fif_times(row, first_time):
    onset = np.float32(row[0] + first_time)
    end = np.float32(row[0] + first_time + row[1])
    return round(float(onset) - first_time, 6), round(float(end - onset), 6)


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

    state = current_state(raw)
    if isinstance(edits, dict):
        edits = [review_delta(state, edits["accepted_state"], step=step)]
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
                if field == "annotations" and row not in state[field]:
                    # A later GUI review opens FIF, which stores float32 onset and end.
                    matches = [
                        r
                        for r in state[field]
                        if r[2:] == row[2:]
                        and _fif_times(r, raw.first_time)
                        == _fif_times(row, raw.first_time)
                    ]
                    if matches and all(r == matches[0] for r in matches):
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


def plot_edits(raw_path, reviewed, step, prior, updates):
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
            current_state(original), current_state(reviewed), step=step
        )
    return [*prior.get("review_edits", []), *([delta] if delta else [])]
