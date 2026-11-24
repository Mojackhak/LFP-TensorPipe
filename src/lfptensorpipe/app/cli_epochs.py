"""Read and replay human epoch decisions on the original recording timeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .alignment.epoch_view import epoch_key

from .alignment.generation import (
    latest_alignment_step_entry,
    alignment_generation_rerun_message,
)
from .alignment.paths import alignment_warp_labels_path
from .cli import ConfigError, RunFailed
from .path_resolver import PathResolver
from lfptensorpipe.io.atomic_file import write_json_atomically
from lfptensorpipe.io.pkl_io import load_pkl


def accepted_selection(context, trial):
    resolver = PathResolver(context)
    why = alignment_generation_rerun_message(resolver, trial_slug=trial, stage="finish")
    if why:
        raise RunFailed(why)
    match = latest_alignment_step_entry(
        resolver, trial_slug=trial, step="build_raw_table"
    )
    decisions = (
        (match[1]["params"].get("epoch_review") or {}).get("decisions")
        if match
        else None
    )
    if not decisions:
        raise RunFailed("The accepted Finish has no source-event decisions to export.")
    selection = {
        "subject": context.subject,
        "record": context.record,
        "trial": trial,
        "decisions": decisions,
    }
    evidence = match[1]["params"]["epoch_review"].get("qc_result")
    if evidence:
        selection["source_qc"] = evidence
    return selection


def selection_reference(value, context, trial):
    if value == "accepted":
        value = accepted_selection(context, trial)
    elif isinstance(value, str):
        try:
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Cannot read the epoch selection: {exc}") from exc
    if not isinstance(value, dict) or any(
        value.get(k) != v
        for k, v in (
            ("subject", context.subject),
            ("record", context.record),
            ("trial", trial),
        )
    ):
        raise ConfigError(
            "The epoch selection must name the same subject, record and trial."
        )
    rows = value.get("decisions")
    if not isinstance(rows, list) or not rows:
        raise ConfigError("The epoch selection has no decisions.")
    for row in rows:
        if not isinstance(row, dict) or type(row.get("include")) is not bool:
            raise ConfigError("Each epoch decision needs a Boolean include value.")
        try:
            epoch_key(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    return value


def epoch_match_mode(node, reference):
    mode = node.get("epoch_match", "exact")
    if mode not in ("exact", "common") or ("epoch_match" in node and reference is None):
        raise ConfigError(
            "epoch_match requires an epoch selection and exact or common matching."
        )
    return mode


def _missing_evidence(resolver, trial, missing):
    """Report observed annotation support without attributing an unexplained loss."""
    import mne
    from lfptensorpipe.io.timeline import raw_relative_onsets
    from lfptensorpipe.lfp.warp.stack import _match_label
    from lfptensorpipe.lfp.warp.utils import intervals_overlap_half_open
    from .alignment.method_specs import DEFAULT_DROP_FIELDS, DEFAULT_DROP_MODE
    from .preproc.paths import preproc_step_raw_path

    latest = latest_alignment_step_entry(
        resolver, trial_slug=trial, step="run_align_epochs"
    )
    options = latest[1]["params"].get("method_params", {}) if latest else {}
    fields = (
        options.get("drop_fields", DEFAULT_DROP_FIELDS)
        if options.get("drop_bad")
        else []
    )
    path = preproc_step_raw_path(resolver, "finish", create=False)
    if not path.is_file():
        return {}
    with mne.io.read_raw_fif(path, preload=False, verbose="ERROR") as raw:
        labels = set(raw.annotations.description)
        excluded = [
            (float(t), float(t + d), str(label))
            for t, d, label in zip(
                raw_relative_onsets(raw),
                raw.annotations.duration,
                raw.annotations.description,
            )
            if _match_label(
                str(label), fields, mode=options.get("drop_mode", DEFAULT_DROP_MODE)
            )
            is not None
        ]
    evidence = {}
    for key in missing:
        label, spans, _ = json.loads(key)
        overlapping = [
            {"start": t, "end": end, "label": name}
            for t, end, name in excluded
            if any(intervals_overlap_half_open(a, b, t, end) for a, b in spans)
        ]
        evidence[key] = {
            "overlapping_drop_annotations": overlapping,
            "source_label_in_finish": label in labels,
            "observation": (
                "overlapping_drop_annotation"
                if overlapping
                else "no_matching_candidate"
            ),
        }
    return evidence


def replay_picks(context, trial, reference, mode):
    from .alignment.epoch_view import finish_review_snapshot, save_epoch_review

    resolver = PathResolver(context)
    epochs = load_pkl(alignment_warp_labels_path(resolver, trial))["ALL"]
    from lfptensorpipe.lfp.warp.metadata import source_epoch_metadata

    rows = source_epoch_metadata(epochs)
    candidates = finish_review_snapshot(context, trial, [], rows)["decisions"]
    requested = reference["decisions"]
    wanted = [epoch_key(r) for r in requested]
    actual = [epoch_key(r) for r in candidates]
    ambiguous = sorted(
        {
            k
            for counts in (Counter(wanted), Counter(actual))
            for k, n in counts.items()
            if n > 1
        }
    )
    missing = sorted(set(wanted) - set(actual))
    added = sorted(set(actual) - set(wanted))
    included = {k for k, row in zip(wanted, requested) if row["include"]}
    picks = [i for i, k in enumerate(actual) if k in included]
    report = {
        "mode": mode,
        "missing": missing,
        "additional": added,
        "ambiguous": ambiguous,
        "included_indices": picks,
        "reference_decisions": requested,
        "reference_qc": reference.get("source_qc"),
        "reference_candidates": len(wanted),
        "generated_candidates": len(actual),
    }
    if missing:
        report["missing_evidence"] = _missing_evidence(resolver, trial, missing)
    path = resolver.alignment_root / trial / "epoch_replay.json"
    write_json_atomically(path, report)
    if ambiguous or (mode == "exact" and (missing or added)) or not picks:
        raise RunFailed(
            f"Epoch replay could not confirm the selection. Details: {path}"
        )
    by_key = dict(zip(wanted, requested))
    notes = {}
    exclusions = {}
    for i, key in enumerate(actual):
        row = by_key.get(key)
        if row is None:
            notes[str(i)] = "Excluded from replay: no decision in the reference."
        elif row.get("notes"):
            notes[str(i)] = row["notes"]
        if row and not row["include"] and row.get("automatic_exclusion"):
            exclusions[str(i)] = row["automatic_exclusion"]
    if not save_epoch_review(
        context, trial, {"notes": notes, "exclusions": exclusions}
    ):
        raise RunFailed("The replayed epoch review could not be saved.")
    return picks
