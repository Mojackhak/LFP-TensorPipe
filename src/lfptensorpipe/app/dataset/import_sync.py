"""Import-time synchronization orchestration and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import RunLogRecord, write_run_log
from lfptensorpipe.io.sync import (
    ImportSyncState,
    PersistedSyncArtifacts,
    SyncEstimate,
    build_synced_raw,
    estimate_sync_from_pairs,
    save_sync_detection_figure,
    save_sync_summary_figure,
    seed_lfp_markers_from_raw,
)


def build_import_sync_seed(raw: Any) -> list[Any]:
    """Return default parsed-annotation markers for the sync dialog."""
    return seed_lfp_markers_from_raw(raw)


def estimate_import_sync(
    *,
    lfp_markers,
    external_markers,
    pairs,
    sfreq_before_hz: float,
    correct_sfreq: bool,
) -> SyncEstimate:
    """Estimate import sync parameters from explicit marker pairs."""
    return estimate_sync_from_pairs(
        lfp_markers=lfp_markers,
        external_markers=external_markers,
        pairs=pairs,
        sfreq_before_hz=sfreq_before_hz,
        correct_sfreq=correct_sfreq,
    )


def build_import_synced_raw(raw: Any, estimate: SyncEstimate) -> Any:
    """Return a new synced raw object for import."""
    return build_synced_raw(raw, estimate)


def _sync_dir(context: RecordContext, *, create: bool) -> Path:
    resolver = PathResolver(context)
    return resolver.import_sync_dir(create=create)


def _markers_to_frame(markers, paired_indices: set[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "marker_index": marker.marker_index,
                "time_s": marker.time_s,
                "label": marker.label,
                "source": marker.source,
                "paired": marker.marker_index in paired_indices,
            }
            for marker in markers
        ]
    )


def _pairs_to_frame(sync_state: ImportSyncState) -> pd.DataFrame:
    lfp_by_index = {marker.marker_index: marker for marker in sync_state.lfp_markers}
    ext_by_index = {
        marker.marker_index: marker for marker in sync_state.external_markers
    }
    deltas = list(sync_state.estimate.deltas_before_sync_s)
    return pd.DataFrame(
        [
            {
                "pair_id": pair.pair_id,
                "lfp_marker_index": pair.lfp_marker_index,
                "lfp_time_s": lfp_by_index[pair.lfp_marker_index].time_s,
                "external_marker_index": pair.external_marker_index,
                "external_time_s": ext_by_index[pair.external_marker_index].time_s,
                "delta_time_s_before_sync": (
                    deltas[index] if index < len(deltas) else None
                ),
            }
            for index, pair in enumerate(sync_state.pairs)
        ]
    )


def _config_payload(
    *,
    sync_state: ImportSyncState,
    raw_fif_path: Path,
    artifacts: PersistedSyncArtifacts,
) -> dict[str, Any]:
    estimate = sync_state.estimate
    return {
        "raw_fif_path": str(raw_fif_path),
        "lfp_source": {
            "kind": sync_state.lfp_source_kind,
            "path": sync_state.lfp_source_path,
            "detect_config": (
                None
                if sync_state.lfp_detect_config is None
                else {
                    "search_range_s": sync_state.lfp_detect_config.search_range_s,
                    "min_distance_s": sync_state.lfp_detect_config.min_distance_s,
                    "height": sync_state.lfp_detect_config.height,
                    "prominence": sync_state.lfp_detect_config.prominence,
                    "max_peaks": sync_state.lfp_detect_config.max_peaks,
                }
            ),
        },
        "external_source": {
            "kind": sync_state.external_source_kind,
            "path": sync_state.external_source_path,
            "detect_config": (
                None
                if sync_state.external_detect_config is None
                else {
                    "search_range_s": sync_state.external_detect_config.search_range_s,
                    "min_distance_s": sync_state.external_detect_config.min_distance_s,
                    "height": sync_state.external_detect_config.height,
                    "prominence": sync_state.external_detect_config.prominence,
                    "max_peaks": sync_state.external_detect_config.max_peaks,
                }
            ),
        },
        "pair_ids": [pair.pair_id for pair in sync_state.pairs],
        "lfp_markers": [
            {
                "marker_index": marker.marker_index,
                "time_s": marker.time_s,
                "label": marker.label,
                "source": marker.source,
            }
            for marker in sync_state.lfp_markers
        ],
        "external_markers": [
            {
                "marker_index": marker.marker_index,
                "time_s": marker.time_s,
                "label": marker.label,
                "source": marker.source,
            }
            for marker in sync_state.external_markers
        ],
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "lfp_marker_index": pair.lfp_marker_index,
                "external_marker_index": pair.external_marker_index,
            }
            for pair in sync_state.pairs
        ],
        "correct_sfreq": estimate.correct_sfreq,
        "lag_s": estimate.lag_s,
        "sfreq_before_hz": estimate.sfreq_before_hz,
        "sfreq_after_hz": estimate.sfreq_after_hz,
        "rmse_ms": estimate.rmse_ms,
        "r2": estimate.r2,
        "figure_paths": {
            "summary": str(artifacts.summary_path),
            "lfp_detection": (
                str(artifacts.lfp_detection_path)
                if artifacts.lfp_detection_path
                else ""
            ),
            "external_detection": (
                str(artifacts.external_detection_path)
                if artifacts.external_detection_path
                else ""
            ),
        },
        "notes": dict(sync_state.notes),
    }


def persist_import_sync_artifacts(
    *,
    project_root: Path,
    subject: str,
    record: str,
    raw_fif_path: Path,
    sync_state: ImportSyncState,
) -> PersistedSyncArtifacts:
    """Persist record-scoped sync artifacts after confirm-import succeeds."""
    context = RecordContext(project_root=project_root, subject=subject, record=record)
    out_dir = _sync_dir(context, create=True)
    artifacts = PersistedSyncArtifacts(
        summary_path=out_dir / "summary.png",
        pairs_path=out_dir / "pairs.csv",
        lfp_markers_path=out_dir / "lfp_markers.csv",
        external_markers_path=out_dir / "external_markers.csv",
        config_path=out_dir / "config.yml",
        log_path=out_dir / "lfptensorpipe_log.json",
        lfp_detection_path=(
            out_dir / "lfp_detection.png"
            if sync_state.lfp_figure_data is not None
            and sync_state.lfp_figure_data.kind == "waveform"
            else None
        ),
        external_detection_path=(
            out_dir / "external_detection.png"
            if sync_state.external_figure_data is not None
            and sync_state.external_figure_data.kind == "waveform"
            else None
        ),
    )
    paired_lfp = {pair.lfp_marker_index for pair in sync_state.pairs}
    paired_ext = {pair.external_marker_index for pair in sync_state.pairs}
    _markers_to_frame(sync_state.lfp_markers, paired_lfp).to_csv(
        artifacts.lfp_markers_path,
        index=False,
    )
    _markers_to_frame(sync_state.external_markers, paired_ext).to_csv(
        artifacts.external_markers_path,
        index=False,
    )
    _pairs_to_frame(sync_state).to_csv(artifacts.pairs_path, index=False)
    save_sync_summary_figure(
        artifacts.summary_path,
        lfp_markers=sync_state.lfp_markers,
        external_markers=sync_state.external_markers,
        pairs=sync_state.pairs,
        estimate=sync_state.estimate,
        lfp_figure_data=sync_state.lfp_figure_data,
        external_figure_data=sync_state.external_figure_data,
    )
    if artifacts.lfp_detection_path and sync_state.lfp_figure_data is not None:
        save_sync_detection_figure(
            artifacts.lfp_detection_path,
            sync_state.lfp_figure_data,
            "LFP Detection",
        )
    if (
        artifacts.external_detection_path
        and sync_state.external_figure_data is not None
    ):
        save_sync_detection_figure(
            artifacts.external_detection_path,
            sync_state.external_figure_data,
            "External Detection",
        )
    with artifacts.config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _config_payload(
                sync_state=sync_state,
                raw_fif_path=raw_fif_path,
                artifacts=artifacts,
            ),
            handle,
            sort_keys=False,
            allow_unicode=False,
        )
    write_run_log(
        artifacts.log_path,
        RunLogRecord(
            step="import_sync",
            completed=True,
            params={
                "pair_count": len(sync_state.pairs),
                "lag_s": sync_state.estimate.lag_s,
                "sfreq_before_hz": sync_state.estimate.sfreq_before_hz,
                "sfreq_after_hz": sync_state.estimate.sfreq_after_hz,
                "rmse_ms": sync_state.estimate.rmse_ms,
                "r2": sync_state.estimate.r2,
                "summary_path": str(artifacts.summary_path),
                "pairs_path": str(artifacts.pairs_path),
                "lfp_markers_path": str(artifacts.lfp_markers_path),
                "external_markers_path": str(artifacts.external_markers_path),
            },
            input_path=str(raw_fif_path),
            output_path=str(artifacts.summary_path),
            message="Import sync artifacts exported.",
        ),
    )
    return artifacts
