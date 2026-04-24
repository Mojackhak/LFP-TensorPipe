"""mapper.py

Map DBS contact coordinates (MNI space) to atlas regions.

This module takes contact coordinates (typically exported by Lead-DBS) and maps each contact
to user-provided atlas regions stored as NIfTI masks. For every contact and every region, it
computes:
- whether the contact falls inside the region mask (boolean),
- relative (x, y, z) offset from the region's center of gravity (CoG),

Design constraints (kept intentionally simple):
- Lead-DBS `.mat` structure is treated as fixed (hard-coded indexing).
- During a single mapping call, each atlas NIfTI file is loaded at most once, and each
  region CoG is computed at most once (cached) to avoid repeated IO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, Literal, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io


# -----------------------------------------------------------------------------
# Public low-level helpers (no caching across calls)
# -----------------------------------------------------------------------------
def points_in_region(
    coords: np.ndarray, region_nii_path: str, threshold: float
) -> np.ndarray:
    """Check whether points (MNI) fall inside a region mask.

    Note: This function loads the NIfTI on every call. For batch mapping, prefer
    `get_points_loc()` / `get_points_loc_df()` which cache NIfTI volumes per call.
    """
    img = nib.load(region_nii_path)
    data = img.get_fdata()
    inv_affine = np.linalg.inv(img.affine)

    coords_arr = np.asarray(coords, dtype=float)
    hom = np.hstack([coords_arr, np.ones((coords_arr.shape[0], 1))])
    vox = (inv_affine @ hom.T).T[:, :3]
    ijk = np.round(vox).astype(int)

    x_max, y_max, z_max = data.shape
    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < x_max)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < y_max)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < z_max)
    )

    inside = np.zeros(coords_arr.shape[0], dtype=bool)
    if np.any(valid):
        vi, vj, vk = ijk[valid].T
        inside[valid] = data[vi, vj, vk] > threshold
    return inside


def get_region_mid(region_nii_path: str, threshold: float = 0.0) -> np.ndarray:
    """Compute the center of gravity (CoG) of a region in MNI space.

    Note: This function loads the NIfTI on every call. For batch mapping, prefer
    `get_points_loc()` / `get_points_loc_df()` which cache NIfTI volumes per call.
    """
    img = nib.load(region_nii_path)
    data = img.get_fdata()
    affine = np.asarray(img.affine, dtype=float)

    mask = data > threshold
    if not np.any(mask):
        return np.zeros(3, dtype=float)

    ijk = np.column_stack(np.nonzero(mask))
    weights = data[mask].astype(float)
    total = float(np.sum(weights))
    if total == 0.0:
        return np.zeros(3, dtype=float)

    hom = np.hstack([ijk, np.ones((ijk.shape[0], 1))])
    xyz = (affine @ hom.T).T[:, :3]
    cog = np.sum(xyz * weights[:, None], axis=0) / total
    return np.asarray(cog, dtype=float)


def get_mid_coords(coords: np.ndarray) -> np.ndarray:
    """Compute midpoints between successive points."""
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape[0] < 2:
        return np.empty((0, 3), dtype=float)
    return (coords_arr[:-1, :] + coords_arr[1:, :]) / 2.0


# -----------------------------------------------------------------------------
# Internal per-call caching utilities
# -----------------------------------------------------------------------------
_NiftiCacheValue = tuple[np.ndarray, np.ndarray, np.ndarray]  # data, affine, inv_affine


def _load_region_volume(
    region_path: str, nifti_cache: dict[str, _NiftiCacheValue]
) -> _NiftiCacheValue:
    """Load a NIfTI region volume with per-call caching."""
    cached = nifti_cache.get(region_path)
    if cached is not None:
        return cached

    img = nib.load(region_path)
    data = img.get_fdata()  # keep dtype as-is; atlas files are typically small masks
    affine = np.asarray(img.affine, dtype=float)
    inv_affine = np.linalg.inv(affine)

    nifti_cache[region_path] = (data, affine, inv_affine)
    return nifti_cache[region_path]


def _points_in_region_cached(
    coords: np.ndarray,
    data: np.ndarray,
    inv_affine: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Vectorized point-in-region check using a preloaded volume."""
    coords_arr = np.asarray(coords, dtype=float)

    hom = np.hstack([coords_arr, np.ones((coords_arr.shape[0], 1))])
    vox = (inv_affine @ hom.T).T[:, :3]
    ijk = np.round(vox).astype(int)

    x_max, y_max, z_max = data.shape
    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < x_max)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < y_max)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < z_max)
    )

    inside = np.zeros(coords_arr.shape[0], dtype=bool)
    if np.any(valid):
        vi, vj, vk = ijk[valid].T
        inside[valid] = data[vi, vj, vk] > threshold
    return inside


def _region_cog_cached(
    region_path: str,
    threshold: float,
    nifti_cache: dict[str, _NiftiCacheValue],
    cog_cache: dict[tuple[str, float], np.ndarray],
) -> np.ndarray:
    """Compute (and cache) the region CoG for a given threshold."""
    key = (region_path, float(threshold))
    cached = cog_cache.get(key)
    if cached is not None:
        return cached

    data, affine, _ = _load_region_volume(region_path, nifti_cache)
    mask = data > threshold
    if not np.any(mask):
        cog = np.zeros(3, dtype=float)
        cog_cache[key] = cog
        return cog

    ijk = np.column_stack(np.nonzero(mask))
    weights = data[mask].astype(float)
    total = float(np.sum(weights))
    if total == 0.0:
        cog = np.zeros(3, dtype=float)
        cog_cache[key] = cog
        return cog

    hom = np.hstack([ijk, np.ones((ijk.shape[0], 1))])
    xyz = (affine @ hom.T).T[:, :3]
    cog = np.sum(xyz * weights[:, None], axis=0) / total

    cog_cache[key] = np.asarray(cog, dtype=float)
    return cog_cache[key]


_DEFAULT_REGION_THRESHOLD = 0.5


def _resolve_threshold(threshold: float | Mapping[str, float], region: str) -> float:
    if isinstance(threshold, Mapping):
        if region in threshold:
            val = threshold[region]
            if isinstance(val, Mapping):
                for key in ("threshold", "thr", "value", "val"):
                    if key in val:
                        val = val[key]
                        break
            try:
                return float(val)
            except Exception as exc:
                raise TypeError(
                    f'Threshold for region "{region}" must be a number, got {type(val).__name__}.'
                ) from exc
        if "default" in threshold:
            return float(threshold["default"])
        if "_default" in threshold:
            return float(threshold["_default"])
        return _DEFAULT_REGION_THRESHOLD
    return float(threshold)


def _map_coords_to_regions(
    coords: np.ndarray,
    index: pd.Index,
    side: str,
    region_list: list[str],
    atlas_path: str,
    threshold: float | Mapping[str, float],
    nifti_cache: dict[str, _NiftiCacheValue],
    cog_cache: dict[tuple[str, float], np.ndarray],
) -> pd.DataFrame:
    """Map a coordinate array to all regions for a single hemisphere/side."""
    coords_arr = np.asarray(coords, dtype=float)
    out_frames: list[pd.DataFrame] = []

    for region in region_list:
        # Keep hard-coded atlas layout: {atlas_path}/{side}/{region}.nii.gz
        region_path = os.path.join(atlas_path, side, f"{region}.nii.gz")

        region_thr = _resolve_threshold(threshold, region)
        data, _, inv_affine = _load_region_volume(region_path, nifti_cache)
        region_mid = _region_cog_cached(region_path, region_thr, nifti_cache, cog_cache)

        inside = _points_in_region_cached(coords_arr, data, inv_affine, region_thr)
        rel = coords_arr - region_mid.reshape(1, 3)

        out_frames.append(
            pd.DataFrame(
                {
                    f"{region}_in": inside,
                    f"{region}_x": rel[:, 0],
                    f"{region}_y": rel[:, 1],
                    f"{region}_z": rel[:, 2],
                },
                index=index,
            )
        )

    if not out_frames:
        return pd.DataFrame(index=index)

    return pd.concat(out_frames, axis=1)


# -----------------------------------------------------------------------------
# Public mapping APIs
# -----------------------------------------------------------------------------
def get_mni_coords(d):
    return d["coords"]["mni"][0][0][0]


def get_mni_reco(d):
    return d["reco"]["mni"][0][0][0][0][0][0]


COORDS_MODE = Literal["coords", "reco"]
POINT_NAMES_MODE = Literal["single", "pair"]


def get_points_loc(
    coordsFile: str | Path,
    region_list: list[str],
    atlas_path: str | Path,
    threshold: float | Mapping[str, float],
    side_dict: dict[str, int | Sequence[int]],
    coords_mode: COORDS_MODE = "coords",
    point_names_mode: POINT_NAMES_MODE = "pair",
) -> pd.DataFrame:
    """Map Lead-DBS exported contact coordinates to atlas regions.

    Parameters
    ----------
    coordsFile:
        Path to Lead-DBS .mat export (hard-coded structure).
    region_list:
        List of atlas region names (NIfTI basenames).
    atlas_path:
        Atlas root folder. Expected layout: {atlas_path}/{side}/{region}.nii.gz
    threshold:
        Either a scalar threshold (applied to all regions) or a dict
        mapping region -> threshold. Missing keys default to 0.5.
    side_dict:
        Mapping from side folder name (e.g. 'lh', 'rh') to Lead-DBS index (e.g. 0/1).
        If multiple leads per side, pass a sequence of indices. E.g. {'lh': [0, 2], 'rh': 1}

    Returns
    -------
    DataFrame with a `contact` column plus:
        side, MNI_x, MNI_y, MNI_z,
        {region}_in, {region}_x/{region}_y/{region}_z for each region
    """
    coords_raw: dict[str, Any] = scipy.io.loadmat(str(coordsFile))

    # Per-call caches (avoid repeated disk IO within this call)
    nifti_cache: dict[str, _NiftiCacheValue] = {}
    cog_cache: dict[tuple[str, float], np.ndarray] = {}

    points_num = 0
    side_frames: list[pd.DataFrame] = []

    for side, idxs in side_dict.items():
        # Allow multiple electrodes/leads per hemisphere by passing a sequence of indices.
        if isinstance(idxs, (list, tuple, np.ndarray)):
            idx_list = [int(i) for i in idxs]
        else:
            idx_list = [int(idxs)]

        for idx in idx_list:
            # Keep hard-coded Lead-DBS structure access.
            if coords_mode == "coords":
                coords = get_mni_coords(coords_raw)[idx]
            else:  # coords_mode == 'reco'
                coords = get_mni_reco(coords_raw)[idx]
            if point_names_mode == "pair":
                point_names = [
                    f"{i + points_num}_{i + points_num + 1}"
                    for i in range(coords.shape[0])
                ]
                points_num += coords.shape[0] + 1
            else:  # point_names_mode == 'single'
                point_names = [f"{i + points_num}" for i in range(coords.shape[0])]
                points_num += coords.shape[0]

            points_index = pd.RangeIndex(coords.shape[0])

            basic_df = pd.DataFrame(
                {
                    "contact": point_names,
                    "side": np.array([side] * coords.shape[0]),
                    "MNI_x": coords[:, 0],
                    "MNI_y": coords[:, 1],
                    "MNI_z": coords[:, 2],
                },
                index=points_index,
            )

            mapped_df = _map_coords_to_regions(
                coords=coords,
                index=points_index,
                side=side,
                region_list=region_list,
                atlas_path=atlas_path,
                threshold=threshold,
                nifti_cache=nifti_cache,
                cog_cache=cog_cache,
            )

            side_frames.append(pd.concat([basic_df, mapped_df], axis=1))

    if not side_frames:
        return pd.DataFrame()

    points_loc_df = pd.concat(side_frames, axis=0)
    return points_loc_df
