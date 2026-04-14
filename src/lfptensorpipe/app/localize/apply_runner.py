"""Orchestration runner for Localize Apply."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.app.runlog_store import RunLogRecord, write_run_log
from lfptensorpipe.app.shared.downstream_invalidation import (
    invalidate_after_localize_result_change,
)
from lfptensorpipe.io.pkl_io import save_pkl

from .paths import (
    localize_match_signature,
    localize_log_path,
    localize_ordered_pair_representative_csv_path,
    localize_ordered_pair_representative_pkl_path,
    localize_selected_regions_signature,
    localize_record_dir,
    localize_representative_csv_path,
    localize_representative_pkl_path,
    localize_undirected_pair_representative_csv_path,
    localize_undirected_pair_representative_pkl_path,
    reconstruction_mat_path,
)

LoadMatchPayloadFn = Callable[[Path, str, str], dict[str, Any] | None]
LoadReconstructionContactsFn = Callable[
    [Path, str, Any], tuple[bool, str, dict[str, Any]]
]
BuildRepcoordsFrameFn = Callable[..., Any]
BuildPairRepcoordsFrameFn = Callable[[Any], Any]


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
    load_match_payload: LoadMatchPayloadFn,
    load_reconstruction_contacts: LoadReconstructionContactsFn,
    build_repcoords_frame: BuildRepcoordsFrameFn,
    build_ordered_pair_repcoords_frame: BuildPairRepcoordsFrameFn,
    build_undirected_pair_repcoords_frame: BuildPairRepcoordsFrameFn,
) -> tuple[bool, str]:
    """Build record-scoped representative-channel localize artifacts."""
    if (
        read_only_project_root is not None
        and project_root.resolve() == read_only_project_root.resolve()
    ):
        return False, "Read-only demo project: Localize Apply is disabled."

    out_dir = localize_record_dir(project_root, subject, record)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = localize_representative_csv_path(project_root, subject, record)
    out_pkl = localize_representative_pkl_path(project_root, subject, record)
    ordered_csv = localize_ordered_pair_representative_csv_path(
        project_root, subject, record
    )
    ordered_pkl = localize_ordered_pair_representative_pkl_path(
        project_root, subject, record
    )
    undirected_csv = localize_undirected_pair_representative_csv_path(
        project_root, subject, record
    )
    undirected_pkl = localize_undirected_pair_representative_pkl_path(
        project_root, subject, record
    )
    out_log = localize_log_path(project_root, subject, record)

    try:
        if paths is None:
            raise RuntimeError(
                "Missing Localize runtime paths; configure the app path settings."
            )
        match_payload = load_match_payload(project_root, subject, record)
        if not isinstance(match_payload, dict):
            raise FileNotFoundError(
                "Missing match payload in record UI state (localize.match)."
            )
        mappings_raw = match_payload.get("mappings")
        if not isinstance(mappings_raw, list) or not mappings_raw:
            raise ValueError("Match payload contains no mappings.")
        if not bool(match_payload.get("completed", False)):
            raise ValueError("Match file is not marked completed.")
        match_signature = localize_match_signature(match_payload)
        if match_signature is None:
            raise ValueError("Match payload is incomplete.")

        ok_reco, msg_reco, reconstruction = load_reconstruction_contacts(
            project_root, subject, paths
        )
        if not ok_reco:
            raise RuntimeError(msg_reco)

        frame = build_repcoords_frame(
            project_root=project_root,
            subject=subject,
            record=record,
            space=space,
            atlas=atlas,
            region_names=selected_regions,
            paths=paths,
            reconstruction=reconstruction,
            mappings=[item for item in mappings_raw if isinstance(item, dict)],
        )
        if frame.empty:
            raise ValueError("No representative channel rows generated.")
        ordered_frame = build_ordered_pair_repcoords_frame(frame)
        undirected_frame = build_undirected_pair_repcoords_frame(frame)

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        save_pkl(frame, out_pkl)
        frame.to_csv(out_csv, index=False)
        save_pkl(ordered_frame, ordered_pkl)
        ordered_frame.to_csv(ordered_csv, index=False)
        save_pkl(undirected_frame, undirected_pkl)
        undirected_frame.to_csv(undirected_csv, index=False)

        write_run_log(
            out_log,
            RunLogRecord(
                step="localize_apply",
                completed=True,
                params={
                    "space": space,
                    "atlas": atlas,
                    "record": record,
                    "selected_regions_signature": localize_selected_regions_signature(
                        list(selected_regions)
                    ),
                    "channel_rows": int(frame.shape[0]),
                    "channel_columns": int(frame.shape[1]),
                    "ordered_pair_rows": int(ordered_frame.shape[0]),
                    "ordered_pair_columns": int(ordered_frame.shape[1]),
                    "undirected_pair_rows": int(undirected_frame.shape[0]),
                    "undirected_pair_columns": int(undirected_frame.shape[1]),
                    "match_signature": match_signature,
                },
                input_path=str(reconstruction_mat_path(project_root, subject)),
                output_path=str(out_dir),
                message=(
                    "Record-scoped representative localize artifacts generated "
                    "(channel, ordered pair, undirected pair)."
                ),
            ),
        )
        invalidate_after_localize_result_change(
            RecordContext(project_root=project_root, subject=subject, record=record)
        )
        return (
            True,
            "Localize completed. "
            f"Saved {frame.shape[0]} channel row(s), "
            f"{ordered_frame.shape[0]} ordered pair row(s), and "
            f"{undirected_frame.shape[0]} undirected pair row(s).",
        )
    except Exception as exc:
        message = f"Localize apply failed: {exc}"
        write_run_log(
            out_log,
            RunLogRecord(
                step="localize_apply",
                completed=False,
                params={
                    "space": space,
                    "atlas": atlas,
                    "record": record,
                    "selected_regions_signature": localize_selected_regions_signature(
                        list(selected_regions)
                    ),
                    "match_signature": (
                        localize_match_signature(match_payload)
                        if "match_payload" in locals()
                        else None
                    ),
                },
                input_path=str(reconstruction_mat_path(project_root, subject)),
                output_path=str(out_dir),
                message=message,
            ),
        )
        return False, message
