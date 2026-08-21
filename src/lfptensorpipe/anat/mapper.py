"""Map MNI-space DBS contact coordinates to atlas region masks."""

from __future__ import annotations

import nibabel as nib
import numpy as np


def points_in_region(
    coords: np.ndarray, region_nii_path: str, threshold: float
) -> np.ndarray:
    """Return whether each MNI coordinate falls inside one atlas mask."""
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
