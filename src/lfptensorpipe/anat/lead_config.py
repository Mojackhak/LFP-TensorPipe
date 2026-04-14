"""Lead-DBS atlas configuration helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io

from lfptensorpipe.app.config_store import AppConfigStore


def load_localization_config() -> dict:
    store = AppConfigStore()
    store.ensure_core_files()
    payload = store.read_yaml("localization.yml", default={})
    return payload if isinstance(payload, dict) else {}


def discover_regions(atlas_path: Path) -> list[str]:
    patterns = ("*.nii", "*.nii.gz")
    region_files = []
    # Only scan one-level subfolders, not files in atlas_path itself.
    for subdir in atlas_path.iterdir():
        if not subdir.is_dir():
            continue
        for pat in patterns:
            region_files.extend(subdir.glob(pat))

    names = []
    for f in region_files:
        name = f.name
        if name.endswith(".nii.gz"):
            names.append(name[:-7])
        elif name.endswith(".nii"):
            names.append(name[:-4])
    return sorted(set(names))


def _scalar_from_any(value: object) -> float:
    arr = np.array(value, dtype=float)
    return float(arr.squeeze())


def read_atlas_threshold(atlas_index_path: Path) -> float:
    if not atlas_index_path.is_file():
        raise FileNotFoundError(f"atlas_index.mat not found: {atlas_index_path}")

    # Try MATLAB v7 (scipy)
    try:
        mat = scipy.io.loadmat(
            str(atlas_index_path), squeeze_me=True, struct_as_record=False
        )
        atl = mat.get("atlases")
        if atl is not None and hasattr(atl, "threshold"):
            thr = atl.threshold
            if hasattr(thr, "value"):
                return _scalar_from_any(thr.value)
    except NotImplementedError:
        pass

    # MATLAB v7.3 (HDF5)
    with h5py.File(atlas_index_path, "r") as f:
        if "atlases" not in f:
            raise RuntimeError(f'"atlases" not found in {atlas_index_path}')
        atl = f["atlases"]
        if "threshold" not in atl or "value" not in atl["threshold"]:
            raise RuntimeError(
                f'"atlases.threshold.value" not found in {atlas_index_path}'
            )
        val = atl["threshold"]["value"][()]
        return _scalar_from_any(val)
