"""Atlas region-membership helpers for representative-coordinate frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import mapper
from .lead_config import discover_regions


def collect_region_paths(atlas_path: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for side_dir in atlas_path.iterdir():
        if not side_dir.is_dir():
            continue
        for region_file in side_dir.iterdir():
            if not region_file.is_file():
                continue
            name = region_file.name
            region = ""
            if name.endswith(".nii.gz"):
                region = name[:-7]
            elif name.endswith(".nii"):
                region = name[:-4]
            if not region:
                continue
            out.setdefault(region, []).append(region_file)
    return out


def append_region_membership_columns(
    frame: pd.DataFrame,
    *,
    atlas_path: Path,
    threshold: float,
    region_names: list[str] | None = None,
    coord_cols: tuple[str, str, str] = ("mni_x", "mni_y", "mni_z"),
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out

    names = (
        list(region_names) if region_names is not None else discover_regions(atlas_path)
    )
    if not names:
        return out

    region_paths = collect_region_paths(atlas_path)
    coords = out.loc[:, list(coord_cols)].to_numpy(dtype=float)
    valid_mask = np.isfinite(coords).all(axis=1)
    valid_indices = np.where(valid_mask)[0]
    valid_coords = coords[valid_mask]

    for region in names:
        inside = np.zeros(out.shape[0], dtype=bool)
        if valid_coords.size != 0:
            for region_file in region_paths.get(region, []):
                hit_subset = mapper.points_in_region(
                    valid_coords,
                    str(region_file),
                    float(threshold),
                )
                inside[valid_indices] = inside[valid_indices] | hit_subset
        out[f"{region}_in"] = inside
    return out


__all__ = [
    "append_region_membership_columns",
    "collect_region_paths",
]
