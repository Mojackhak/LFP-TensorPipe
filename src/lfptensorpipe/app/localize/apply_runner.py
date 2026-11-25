"""Publish one record's applied label source and representative tables."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from lfptensorpipe.anat.repcoords import (
    build_ordered_pair_representative_frame,
    build_undirected_pair_representative_frame,
)
from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    append_run_log_event,
    read_run_log,
)
from lfptensorpipe.app.shared.atomic_outputs import write_outputs_atomically
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_localize_result_change,
)
from lfptensorpipe.app.shared.generation_lineage import (
    new_result_generation_id,
    params_with_generation_lineage,
)
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl

from .atlas_lookup import (
    assign_atlas_regions,
    build_repcoords_frame as build_atlas_frame,
)
from .lead_inputs import cached_lead_signature, compare_lead_inputs
from .config import build_lead_signature, moved_representative_channels
from .paths import (
    LEAD_DBS,
    USER_LABELS,
    TABLE_STEMS,
    applied_source,
    localize_match_signature,
    localize_record_dir,
    localize_selected_regions_signature,
    reconstruction_mat_path,
    record_channel_fif,
    source_directory,
    source_files_ready,
    source_results,
)
from .user_labels import UserLabels, read_labels, write_labels_csv

logger = logging.getLogger(__name__)


def _table_writers(directory, index, frame, *, missing_only=False):
    stem = directory / TABLE_STEMS[index]
    writers = []
    for ext in (".pkl", ".csv"):
        path = stem.with_suffix(ext)
        if missing_only and path.is_file():
            continue
        if ext == ".pkl":
            writers.append((path, lambda dest: save_pkl(frame, dest)))
        else:
            writers.append((path, lambda dest: frame.to_csv(dest, index=False)))
    return writers


def _repair_tables(directory, ordered, undirected):
    writers = []
    primary = directory / (TABLE_STEMS[0] + ".pkl")
    channels = load_pkl(primary)
    for i, build in enumerate((None, ordered, undirected)):
        stem = directory / TABLE_STEMS[i]
        if stem.with_suffix(".pkl").is_file() and stem.with_suffix(".csv").is_file():
            continue
        if build is None:
            frame = channels
        elif stem.with_suffix(".pkl").is_file():
            frame = load_pkl(stem.with_suffix(".pkl"))
        else:
            frame = build(channels)
        writers.extend(_table_writers(directory, i, frame, missing_only=True))
    return writers


def _prepare_lead_table(
    *,
    project_root,
    subject,
    record,
    space,
    atlas,
    selected_regions,
    paths,
    channels,
    saved,
    primary,
    load_match_payload,
    load_reconstruction_contacts,
    build_repcoords_frame,
):
    """Return a new/updated frame, or None when the retained table is current."""
    frame = None
    if paths is None:
        raise ValueError(
            "Missing Localize runtime paths; configure the app path settings."
        )
    match = load_match_payload(project_root, subject, record)
    mapping = localize_match_signature(match)
    regions = localize_selected_regions_signature(selected_regions)
    if not mapping or not regions or not space or not atlas:
        raise ValueError("Complete Match and Atlas configuration before Apply.")
    if channels and {row["channel"] for row in mapping} != set(channels):
        raise ValueError("Match does not cover the current record channels.")
    input_path = reconstruction_mat_path(project_root, subject)
    params = {
        "record": record,
        "space": space,
        "atlas": atlas,
        "match_signature": mapping,
        "selected_regions_signature": regions,
    }
    same_geometry = False
    same_regions = False
    same_endpoints = False
    reconstruction = None
    if hasattr(paths, "leaddbs_dir"):
        geometry, masks, same_geometry, same_regions = compare_lead_inputs(
            saved, project_root, subject, space, atlas, regions, paths, mapping
        )
        params.update(geometry_inputs=geometry, region_inputs=masks)
        reco_stamp = geometry["files"].get(str(input_path))
        same_endpoints = (
            reco_stamp is not None
            and saved.get("space") == space
            and saved.get("geometry_inputs", {}).get("files", {}).get(str(input_path))
            == reco_stamp
        )
        signature = cached_lead_signature(saved, input_path, paths)
        if signature is not None:
            params["lead_signature"] = signature
            params["lead_signature_elspec"] = saved["lead_signature_elspec"]

    def load_contacts():
        ok, message, summary = load_reconstruction_contacts(
            project_root, subject, paths
        )
        if not ok:
            raise RuntimeError(message)
        if "leads" in summary:
            params["lead_signature"] = build_lead_signature(summary["leads"])
            params["lead_signature_elspec"] = summary.get("lead_signature_elspec")
        return summary

    expected = match.get("lead_signature")
    if (expected is not None or saved.get("lead_signature")) and "lead_signature" not in params:
        reconstruction = load_contacts()
    if expected is not None and expected != params.get("lead_signature"):
        raise ValueError(
            "Localize match lead_signature differs from the current reconstruction."
        )
    moved = moved_representative_channels(
        mapping, saved.get("lead_signature"), params.get("lead_signature")
    )
    old_match = saved.get("match_signature", [])
    old_frame = None
    if same_geometry and primary.is_file() and "geometry_inputs" not in saved:
        # v0.3 has an output ID but no individual input receipts. Check the
        # artifact itself against the historical channel/atlas assignment.
        old_frame = load_pkl(primary)
        required = {
            "subject",
            "record",
            "space",
            "atlas",
            "channel",
            "anode",
            "cathode",
            "rep_coord",
            "mni_x",
            "mni_y",
            "mni_z",
        }
        same_geometry = (
            isinstance(old_frame, pd.DataFrame)
            and required.issubset(old_frame)
            and old_frame["subject"].eq(subject).all()
            and old_frame["record"].eq(record).all()
            and old_frame["space"].eq(space).all()
            and localize_match_signature(
                {"completed": True, "mappings": old_frame.to_dict("records")}
            )
            == old_match
        )
        same_regions = bool(
            same_geometry
            and same_regions
            and old_frame["atlas"].eq(atlas).all()
            and all(r + "_in" in old_frame for r in regions)
        )
    reuse = (
        primary.is_file()
        and same_geometry
        and same_regions
        and mapping == old_match
        and not moved
    )
    if not reuse:
        parts = []
        changed = list(match["mappings"])
        if (same_geometry or same_endpoints) and primary.is_file():
            if old_frame is None:
                old_frame = load_pkl(primary)
            kept = {
                row["channel"]
                for row in mapping
                if row in old_match
                and row["channel"] not in moved
                and (same_geometry or row["rep_coord"] != "Mid")
            }
            base = old_frame.loc[old_frame["channel"].isin(kept)].copy()
            if not base.empty:
                if not same_regions:
                    base = base.drop(
                        columns=[col for col in base if col.endswith("_in")]
                    )
                    base["atlas"] = atlas
                    base = assign_atlas_regions(
                        base,
                        paths.leaddbs_dir
                        / "templates"
                        / "space"
                        / space
                        / "atlases"
                        / atlas,
                        regions,
                        threshold=masks["threshold"],
                    )
                parts.append(base)
            changed = [row for row in changed if row["channel"] not in kept]
        if changed:
            if reconstruction is None:
                reconstruction = load_contacts()
            build = build_repcoords_frame or partial(
                build_atlas_frame,
                threshold=params["region_inputs"]["threshold"],
                capture_anchor=(
                    "fsl" in geometry["method"] or "fnirt" in geometry["method"]
                ),
            )
            parts.append(
                build(
                    project_root=project_root,
                    subject=subject,
                    record=record,
                    space=space,
                    atlas=atlas,
                    region_names=regions,
                    paths=paths,
                    reconstruction=reconstruction,
                    mappings=changed,
                )
            )
            anchor_file = parts[-1].attrs.get("anchor_file")
            if anchor_file:
                previous_anchor = geometry["anchor_file"]
                if previous_anchor and previous_anchor != anchor_file:
                    geometry["files"].pop(previous_anchor, None)
                geometry["anchor_file"] = anchor_file
                geometry["files"][anchor_file] = Path(anchor_file).stat().st_mtime_ns
        if len(parts) > 1:
            order = [row["channel"] for row in match["mappings"]]
            frame = (
                pd.concat(parts)
                .set_index("channel", drop=False)
                .loc[order]
                .reset_index(drop=True)
            )
        else:
            frame = parts[0]

    return frame, params, input_path


def run_localize_apply(
    *,
    project_root: Path,
    subject: str,
    record: str,
    space: str,
    atlas: str,
    selected_regions: list[str] | tuple[str, ...],
    paths: Any | None,
    read_only_project_root: Path | None,
    load_match_payload: Callable,
    load_reconstruction_contacts: Callable,
    build_repcoords_frame: Callable | None,
    build_ordered_pair_repcoords_frame: Callable = build_ordered_pair_representative_frame,
    build_undirected_pair_repcoords_frame: Callable = build_undirected_pair_representative_frame,
    source: str = LEAD_DBS,
    channels: tuple[str, ...] = (),
    user_labels: UserLabels | None = None,
) -> tuple[bool, str]:
    """Apply one source, preserving both accepted receipts and the other source's tables."""
    if (
        read_only_project_root is not None
        and project_root.resolve() == read_only_project_root.resolve()
    ):
        return False, "Read-only demo project: Localize Apply is disabled."

    root = localize_record_dir(project_root, subject, record)
    log_path = root / "lfptensorpipe_log.json"
    try:
        directory = source_directory(root, source)
        previous = read_run_log(log_path)
    except (OSError, ValueError) as exc:
        return False, f"Localize apply failed: {exc}"
    sources = source_results(previous).copy()
    active = applied_source(previous)
    saved = sources.get(source, {})
    primary = directory / (TABLE_STEMS[0] + ".pkl")
    writers = []
    frame = None
    context = RecordContext(project_root, subject, record)
    try:
        if source == USER_LABELS:
            if not channels:
                from lfptensorpipe.app.tensor.selectors import (
                    load_tensor_channel_inventory,
                )

                channels = load_tensor_channel_inventory(
                    record_channel_fif(context)
                ).all_channels
            labels_path = directory / "input.csv"
            labels = (
                user_labels
                if user_labels is not None
                else read_labels(labels_path, channels)
            )
            if not labels.regions or not labels.assessed:
                raise ValueError(
                    "Save at least one assessed channel and one region first."
                )
            signature = labels.signature(channels)
            params = {"record": record, "label_signature": signature}
            reuse = saved.get("label_signature") == signature and primary.is_file()
            if not reuse:
                frame = labels.frame(subject, record, channels)
            if user_labels is not None:
                writers.append(
                    (labels_path, lambda path: write_labels_csv(path, labels, channels))
                )
            input_path = labels_path
        else:
            frame, params, input_path = _prepare_lead_table(
                project_root=project_root,
                subject=subject,
                record=record,
                space=space,
                atlas=atlas,
                selected_regions=selected_regions,
                paths=paths,
                channels=channels,
                saved=saved,
                primary=primary,
                load_match_payload=load_match_payload,
                load_reconstruction_contacts=load_reconstruction_contacts,
                build_repcoords_frame=build_repcoords_frame,
            )
            reuse = frame is None

        if reuse:
            if source_files_ready(root, source):
                identity_updated = "lead_signature" in params and any(
                    params.get(key) != saved.get(key)
                    for key in ("lead_signature", "lead_signature_elspec")
                )
                if active == source and not identity_updated and not writers:
                    return True, "Up to date."
            else:
                writers.extend(
                    _repair_tables(
                        directory,
                        build_ordered_pair_repcoords_frame,
                        build_undirected_pair_repcoords_frame,
                    )
                )
            params = {**saved, **params}
            message = "Using existing Localize result."
        else:
            if frame.empty:
                raise ValueError("No representative channel rows generated.")
            ordered = build_ordered_pair_repcoords_frame(frame)
            undirected = build_undirected_pair_repcoords_frame(frame)
            for i, table in enumerate((frame, ordered, undirected)):
                writers.extend(_table_writers(directory, i, table))
            params.update(
                channel_rows=len(frame),
                channel_columns=len(frame.columns),
                ordered_pair_rows=len(ordered),
                ordered_pair_columns=len(ordered.columns),
                undirected_pair_rows=len(undirected),
                undirected_pair_columns=len(undirected.columns),
                computed_at=datetime.now(timezone.utc).isoformat(),
            )
            params = params_with_generation_lineage(
                params,
                result_generation_id=new_result_generation_id(),
                input_generations={},
            )
            message = f"Saved {len(frame)} channel row(s), {len(ordered)} ordered pair row(s), and {len(undirected)} undirected pair row(s)."

        params["label_source"] = source
        sources[source] = params
        event = RunLogRecord(
            step="localize_apply",
            completed=True,
            params=params,
            input_path=str(input_path),
            output_path=str(directory),
            message=message,
        )
        writers.append(
            (
                log_path,
                lambda dest: append_run_log_event(
                    dest,
                    event,
                    source_path=log_path,
                    state_replacements={
                        "active_source": source,
                        "source_results": sources,
                    },
                ),
            )
        )

        def check_binding() -> None:
            if read_run_log(log_path) != previous:
                raise RuntimeError(
                    "Localize changed during Apply. Apply again to use the current result."
                )

        write_outputs_atomically(
            writers, cleanup_stale_residues=True, precommit_check=check_binding
        )
    except Exception as exc:
        # Report a failed scientific or filesystem operation while retaining
        # the last accepted binding (including a v0.3 single-source result).
        message = f"Localize apply failed: {exc}"
        current = read_run_log(log_path)
        append_run_log_event(
            log_path,
            RunLogRecord(
                step="localize_apply",
                completed=False,
                params={"label_source": source},
                message=message,
            ),
            state_replacements={
                "active_source": applied_source(current),
                "source_results": source_results(current),
            },
        )
        return False, message

    if not reuse or active != source:
        try:
            invalidate_after_localize_result_change(context)
        except Exception as exc:
            logger.warning("Could not invalidate Localize downstream results: %s", exc)
    return True, "Localize completed. " + message
