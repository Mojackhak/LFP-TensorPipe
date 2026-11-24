"""Review and replay per-interval Signal Repair decisions."""

from collections import Counter, defaultdict, deque
from copy import deepcopy

from lfptensorpipe.preproc.filter import _set_annotations_from_attached_frame


class SignalRepairReview:
    """Keep the complete interval ledger while changing only managed support."""

    def __init__(self, raw, intervals):
        self.raw = raw
        self.intervals = deepcopy(intervals)
        self.rows = sorted(
            (item for item in self.intervals if "accepted" in item),
            key=lambda item: item["start_sample"],
        )
        self.opened_selection = self.selection
        self._opened_managed = {}
        by_support = defaultdict(deque)
        for ann in raw.annotations:
            record = self._annotation_record(ann)
            by_support[self.annotation_key(record)].append(record)
        for item in self.rows:
            exact = []
            for target in self.annotations_for(item):
                matches = by_support.get(self.annotation_key(target))
                exact.append(matches.popleft() if matches else target)
            self._opened_managed[id(item)] = item["accepted"], exact

    @staticmethod
    def _annotation_record(annotation):
        record = [
            annotation["onset"],
            annotation["duration"],
            annotation["description"],
            list(annotation.get("ch_names", ())),
        ]
        if annotation.get("extras"):
            record.append(dict(annotation["extras"]))
        return record

    @property
    def selection(self):
        return tuple(item["accepted"] for item in self.rows)

    @property
    def changed(self):
        return self.selection != self.opened_selection

    @property
    def summary(self):
        from lfptensorpipe.preproc.signal_repair import summarize_signal_repair

        return summarize_signal_repair(self.intervals)

    def annotations_for(self, item):
        opened = self._opened_managed.get(id(item))
        if opened is not None and item["accepted"] == opened[0]:
            return opened[1]
        if not item["accepted"]:
            return item["gap_annotations"]
        sfreq = self.raw.info["sfreq"]
        return [
            [
                self.raw.first_time + item["start_sample"] / sfreq,
                (item["stop_sample"] - item["start_sample"]) / sfreq,
                "INTERPOLATED_gap" if item["kind"] == "gaps" else "INTERPOLATED_peak",
                [item["channel"]],
            ]
        ]

    def annotation_key(self, annotation):
        # Identify managed support by samples, including after FIF time rounding.
        onset, duration = annotation[:2]
        start = onset - self.raw.first_time
        bounds = self.raw.time_as_index(
            [start, start + duration],
            use_rounding=True,
        )
        return annotation[2], tuple(annotation[3]), tuple(bounds)

    def set_accepted(self, row, accepted):
        item = self.rows[row]
        if item["accepted"] == accepted:
            return
        annotations = self.raw.annotations
        managed = Counter(
            self.annotation_key(row) for row in self.annotations_for(item)
        )
        removed = []
        for index, ann in enumerate(annotations):
            key = self.annotation_key(self._annotation_record(ann))
            if managed[key]:
                removed.append(index)
                managed[key] -= 1
        item["accepted"] = accepted
        index = self.raw.ch_names.index(item["channel"])
        self.raw._data[index, item["start_sample"] : item["stop_sample"]] = item[
            "interpolated_samples" if accepted else "original_samples"
        ]
        updated = annotations.copy()
        updated.delete(removed)
        records = self.annotations_for(item)
        if records:
            extras = (
                {"extras": [row[4] if len(row) > 4 else None for row in records]}
                if hasattr(annotations, "extras")
                else {}
            )
            updated.append(
                onset=[row[0] for row in records],
                duration=[row[1] for row in records],
                description=[row[2] for row in records],
                ch_names=[row[3] for row in records],
                **extras,
            )
        _set_annotations_from_attached_frame(self.raw, updated)


def interval_id(row):
    return tuple(row[key] for key in ("channel", "kind", "start_sample", "stop_sample"))


def repair_decisions(intervals):
    return [
        {
            key: row[key]
            for key in ("channel", "kind", "start_sample", "stop_sample", "accepted")
        }
        for row in intervals
        if "accepted" in row
    ]


def replay_repair(raw, intervals, decisions):
    review = SignalRepairReview(raw, intervals)
    if not isinstance(decisions, list) or any(
        not isinstance(row, dict) or type(row.get("accepted")) is not bool
        for row in decisions
    ):
        raise ValueError(
            "Repair decisions require a Boolean accepted value for each candidate."
        )
    try:
        by_id = {interval_id(row): row["accepted"] for row in decisions}
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid Signal Repair interval identity.") from exc
    current = [interval_id(row) for row in review.rows]
    if (
        len(by_id) != len(decisions)
        or len(current) != len(set(current))
        or set(current) != set(by_id)
    ):
        raise ValueError("Signal Repair candidates differ from the reviewed intervals.")
    for i, key in enumerate(current):
        review.set_accepted(i, by_id[key])
    return review
