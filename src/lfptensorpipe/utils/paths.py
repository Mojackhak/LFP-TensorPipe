"""paths.py

Path and filesystem structure utilities.

This module contains helpers for:
- Discovering files (optionally recursive)
- Filtering path lists by regex/suffix

All functions are written to be lightweight and cross-platform.
"""

from __future__ import annotations

from pathlib import Path
import glob
import os
import re
from typing import Sequence

import numpy as np


def _normalize_filetypes(
    filetypes: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Normalize user-provided file types into a tuple of suffix strings.

    Examples
    --------
    - "csv"   -> (".csv",)
    - ".csv"  -> (".csv",)
    - ("csv", "h5") -> (".csv", ".h5")

    Parameters
    ----------
    filetypes:
        A file suffix (with or without leading dot) or a sequence of suffixes.

    Returns
    -------
    tuple[str, ...] | None
        Normalized suffixes, or None if no filtering is requested.
    """
    if filetypes is None:
        return None

    if isinstance(filetypes, str):
        candidates = [filetypes]
    else:
        candidates = list(filetypes)

    normalized: list[str] = []
    for ft in candidates:
        if not ft:
            continue
        ft = ft.strip()
        if not ft:
            continue
        if not ft.startswith("."):
            ft = "." + ft
        normalized.append(ft.lower())

    return tuple(normalized) if normalized else None


def get_files_absolute_path(
    folder_path: str | Path,
    filetypes: str | Sequence[str] | None = None,
    recursive: bool = False,
    contain: str | None = None,
    exclude: str | None = None,
) -> np.ndarray:
    """Collect absolute file paths from a directory with filtering options.

    Parameters
    ----------
    folder_path:
        Directory to scan.
    filetypes:
        File suffix or suffixes to include (e.g., ".txt" or (".jpg", ".png")).
    recursive:
        If True, recursively scan subfolders.
    contain:
        Regex pattern that must match the full path string for a file to be included.
    exclude:
        Regex pattern that, if matched on the full path string, excludes the file.

    Returns
    -------
    numpy.ndarray
        Array of absolute file paths as strings.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"The specified folder does not exist: {folder}")
        return np.array([])

    suffixes = _normalize_filetypes(filetypes)
    results: list[str] = []

    if recursive:
        iterator = folder.rglob("*")
    else:
        iterator = folder.glob("*")

    for p in iterator:
        if not p.is_file():
            continue

        if suffixes is not None and p.suffix.lower() not in suffixes:
            continue

        p_str = str(p)
        if contain and re.search(contain, p_str) is None:
            continue
        if exclude and re.search(exclude, p_str) is not None:
            continue

        results.append(str(p.resolve()))

    return np.array(results)


def check_and_collect_files(
    path: str | Path,
    filetypes: str | Sequence[str] | None = None,
    recursive: bool = False,
    contain: str | None = None,
    exclude: str | None = None,
) -> list[str]:
    """Collect files from a path (file or directory) with optional filters.

    This is an internal helper used by `get_files_list()`.

    Parameters
    ----------
    path:
        A directory path or file path.
    filetypes:
        File suffix (or suffixes) to include.
    recursive:
        Only used when `path` is a directory.
    contain:
        Regex pattern applied to the basename when `path` is a file, or full path when directory scan.
    exclude:
        Regex pattern applied similarly to `contain`.

    Returns
    -------
    list[str]
        A list of file paths (strings). Empty list if nothing matches.
    """
    p = Path(path)

    if p.is_dir():
        files = get_files_absolute_path(p, filetypes, recursive, contain, exclude)
        return list(files)

    if p.is_file():
        suffixes = _normalize_filetypes(filetypes)
        if suffixes is not None and p.suffix.lower() not in suffixes:
            return []

        base = p.name
        if contain and re.search(contain, base) is None:
            return []
        if exclude and re.search(exclude, base) is not None:
            return []

        return [str(p.resolve())]

    print("Error: The provided path is not valid.")
    return []


def get_files_list(
    var: str | Path | Sequence[str | Path] | np.ndarray,
    filetypes: str | Sequence[str] | None = None,
    recursive: bool = False,
    contain: str | None = None,
    exclude: str | None = None,
    drop_metafiles: bool = True,
) -> np.ndarray:
    """Normalize input paths and return a NumPy array of `Path` objects.

    Parameters
    ----------
    var:
        A single path (file/folder) or a sequence of paths.
    filetypes:
        File suffix or suffixes to include.
    recursive:
        Recursively scan folders when True.
    contain:
        Regex pattern to include.
    exclude:
        Regex pattern to exclude.
    drop_metafiles:
        If True, exclude common metafiles whose filenames start with dot,
        like ".DS_Store".

    Returns
    -------
    numpy.ndarray
        Array of `pathlib.Path` objects.
    """
    results: list[str | Path] = []

    if isinstance(var, (str, Path)):
        results.extend(
            check_and_collect_files(str(var), filetypes, recursive, contain, exclude)
        )
    elif isinstance(var, (list, tuple, np.ndarray)):
        for item in var:
            results.extend(
                check_and_collect_files(
                    str(item), filetypes, recursive, contain, exclude
                )
            )
    else:
        print("Error: Input must be a string/path or a list/array of strings/paths.")
        return np.array([])

    results = [
        Path(f)
        for f in results
        if not (drop_metafiles and Path(f).name.startswith("."))
    ]

    return np.array(results)


def filter_files_by_re(
    files: Sequence[str | Path] | np.ndarray, regex_pattern: str
) -> np.ndarray:
    """Filter file paths by regex on the *basename without extension*.

    Parameters
    ----------
    files:
        Sequence/array of file paths.
    regex_pattern:
        Regular expression applied to file stem (basename without extension).

    Returns
    -------
    numpy.ndarray
        Filtered NumPy array (same element type as input after `np.array()`).
    """
    arr = np.array(files)
    mask = []
    for p in arr:
        stem = Path(p).stem
        mask.append(re.search(regex_pattern, stem) is not None)
    return arr[np.array(mask, dtype=bool)]


def get_files_in_structure(
    wd_folder: str | Path, search_pattern: str, drop_metafiles: bool = True
) -> list[Path]:
    """Search for files matching a directory structure pattern.

    Parameters
    ----------
    wd_folder:
        Root folder.
    search_pattern:
        Glob pattern relative to `wd_folder`, e.g. "*/*/surgery/waveform/*.csv".
    drop_metafiles:
        If True, exclude common metafiles whose filenames start with dot,
        like ".DS_Store".

    Returns
    -------
    list[pathlib.Path]
        Absolute paths to matching files.
    """
    wd = Path(wd_folder)
    pattern = str(wd / search_pattern)
    files = glob.glob(pattern, recursive=True)
    if drop_metafiles:
        files = [f for f in files if not Path(f).name.startswith(".")]
    return [Path(os.path.abspath(f)) for f in files]


def find_repo_root(
    start: str | Path | None = None,
) -> Path:
    """Find the project root with a few graceful fallbacks.

    Order:
    1) LFP_TENSORPIPE_ROOT env var (if set)
    2) Use the fixed relative location of this file:
       src/lfptensorpipe/utils/paths.py -> project root is parents[3]
    """
    env_root = os.environ.get("LFP_TENSORPIPE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return Path(__file__).resolve().parents[3]
