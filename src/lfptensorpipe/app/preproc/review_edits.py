"""Portable changes made during manual review of a preprocessing result."""

from collections import Counter
import json
from math import isfinite


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


def review_delta(before, after):
    change = {}
    for field in ("annotations", "bads"):
        left, right = [
            Counter(json.dumps(x, sort_keys=True) for x in state[field])
            for state in (before, after)
        ]
        for action, counts in (("add", right - left), ("remove", left - right)):
            if counts:
                change[f"{action}_{field}"] = [
                    json.loads(s) for s in sorted(counts.elements())
                ]
    return change


def apply_edits(raw, edits):
    import mne
    from lfptensorpipe.preproc.filter import _set_annotations_from_attached_frame

    state = current_state(raw)
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
                    if not isinstance(item, str) or item not in raw.ch_names:
                        raise ValueError(f"Unknown review channel: {item!r}")
                elif (
                    not isinstance(item, list)
                    or len(item) != 5
                    or not all(
                        type(x) in (int, float) and isfinite(x) for x in item[:2]
                    )
                    or item[1] < 0
                    or not isinstance(item[2], str)
                    or not isinstance(item[3], list)
                    or any(ch not in raw.ch_names for ch in item[3])
                    or not isinstance(item[4], dict)
                ):
                    raise ValueError(f"Invalid review annotation: {item!r}")
        for field in ("annotations", "bads"):
            for row in change.get("remove_" + field, []):
                if row not in state[field]:
                    raise ValueError(
                        f"A reviewed {field} entry no longer matches: {row}"
                    )
                state[field].remove(row)
            state[field].extend(change.get("add_" + field, []))
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
        delta = review_delta(current_state(original), current_state(reviewed))
    return [*prior.get("review_edits", []), *([delta] if delta else [])]
