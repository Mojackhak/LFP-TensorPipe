"""Atlas lookup and representative-coordinate builders for Localize."""

from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfptensorpipe.anat.atlas_membership import (
    append_region_membership_columns,
    collect_region_paths as _collect_region_paths_impl,
)
from lfptensorpipe.anat.lead_config import discover_regions, read_atlas_threshold
from lfptensorpipe.anat.repcoords import build_channel_representative_frame


def default_map_native_to_mni(
    project_root: Path,
    subject: str,
    native_points: np.ndarray,
    paths: Any,
) -> np.ndarray:
    if native_points.size == 0:
        return np.zeros((0, 3), dtype=float)

    from . import service as svc

    subject_dir = project_root / "derivatives" / "leaddbs" / subject

    def _task(eng: Any) -> Any:
        import matlab

        points = matlab.double(native_points.astype(float).tolist())
        resolved = str(eng.which("map_native_to_mni"))
        if not resolved:
            raise RuntimeError(
                "MATLAB function `map_native_to_mni` not found on path. "
                "Ensure `src/lfptensorpipe/anat/leaddbs` is available."
            )
        return eng.map_native_to_mni(
            str(subject_dir),
            points,
            nargout=1,
        )

    future = svc.submit_matlab_task(paths, _task)
    try:
        mapped = future.result(timeout=svc._MATLAB_TASK_TIMEOUT_S)
    except FutureTimeout as exc:
        message = f"MATLAB request timed out after {int(svc._MATLAB_TASK_TIMEOUT_S)}s."
        svc._set_matlab_runtime_status("failed", message)
        raise RuntimeError(message) from exc

    arr = np.asarray(mapped, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Invalid mapped point shape: {arr.shape}")
    return arr


def collect_region_paths(atlas_path: Path) -> dict[str, list[Path]]:
    return _collect_region_paths_impl(atlas_path)


def build_repcoords_frame(
    *,
    project_root: Path,
    subject: str,
    record: str,
    space: str,
    atlas: str,
    region_names: list[str] | tuple[str, ...] | None = None,
    paths: Any,
    reconstruction: dict[str, Any],
    mappings: list[dict[str, Any]],
) -> pd.DataFrame:
    atlas_path = paths.leaddbs_dir / "templates" / "space" / space / "atlases" / atlas
    atlas_index = atlas_path / "atlas_index.mat"
    if not atlas_path.is_dir():
        raise FileNotFoundError(f"Atlas path not found: {atlas_path}")
    if not atlas_index.is_file():
        raise FileNotFoundError(f"atlas_index.mat not found: {atlas_index}")

    frame = build_channel_representative_frame(
        subject=subject,
        record=record,
        space=space,
        atlas=atlas,
        reconstruction=reconstruction,
        mappings=mappings,
        map_native_to_mni_fn=lambda native_points: default_map_native_to_mni(
            project_root,
            subject,
            native_points,
            paths,
        ),
    )
    target_region_names = (
        [str(item).strip() for item in region_names if str(item).strip()]
        if region_names is not None
        else discover_regions(atlas_path)
    )
    if not target_region_names:
        return frame

    threshold = float(read_atlas_threshold(atlas_index))
    return append_region_membership_columns(
        frame,
        atlas_path=atlas_path,
        threshold=threshold,
        region_names=target_region_names,
    )
