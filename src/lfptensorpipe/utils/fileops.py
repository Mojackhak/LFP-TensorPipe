"""fileops.py

File-level operations that are not purely path-listing utilities:
- Creating folders
- Moving files/folders to a new location
- Copying files to a new folder
- Moving files/folders to the recycle bin (via send2trash)

These functions are intentionally small and dependency-light.
"""

from __future__ import annotations

import numpy as np
import shutil
from pathlib import Path
from typing import Sequence


def create_subfolder(parent_folder: str | Path, subfolder_name: str) -> Path:
    """Create a subfolder under a parent directory.

    Parameters
    ----------
    parent_folder:
        Existing folder path.
    subfolder_name:
        Name of the subfolder to create.

    Returns
    -------
    pathlib.Path
        The created (or already-existing) subfolder path.
    """
    parent = Path(parent_folder)
    subfolder_path = parent / subfolder_name

    if subfolder_path.exists():
        print(f"The subfolder already exists: {subfolder_path}")
    else:
        try:
            subfolder_path.mkdir(parents=True, exist_ok=True)
            print(f"Subfolder '{subfolder_path}' created successfully.")
        except Exception as exc:  # pragma: no cover (OS-specific)
            print(f"Failed to create subfolder '{subfolder_path}'. Error: {exc}")

    return subfolder_path


def move_files_to_folder(
    files: Sequence[str | Path] | np.ndarray, folder: str | Path
) -> list[str]:
    """Move files into a folder (creating it if needed).

    Parameters
    ----------
    files:
        Paths to move.
    folder:
        Destination folder.

    Returns
    -------
    list[str]
        Destination file paths as strings.
    """
    dest_folder = Path(folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    destfiles: list[str] = []
    for file in files:
        src = Path(file)
        if not src.exists():
            print(f"Error: The file '{src}' does not exist.")
            continue

        dst = dest_folder / src.name
        shutil.move(str(src), str(dst))
        destfiles.append(str(dst))
        print(f"Moved file '{src}' to '{dest_folder}'.")

    return destfiles


def rename_file_or_folder(
    file_path: str | Path, new_name: str, overwrite: bool = False
) -> str | None:
    """Rename a file or folder within its current directory.

    Parameters
    ----------
    file_path:
        File/folder path to rename.
    new_name:
        New base name (no directory).
    overwrite:
        If True, delete an existing target with the same name first.

    Returns
    -------
    str | None
        The new path as a string, or None if the input does not exist.
    """
    src = Path(file_path)
    if not src.exists():
        print(f"The path '{src}' does not exist.")
        return None

    dst = src.parent / new_name
    if dst.exists():
        if not overwrite:
            print(
                f"The new name '{new_name}' already exists in the directory. Skipping rename."
            )
            return str(dst)

        # Overwrite: remove existing target
        try:
            if dst.is_file():
                dst.unlink()
                print(f"Removed existing file '{dst}'.")
            elif dst.is_dir():
                shutil.rmtree(dst)
                print(f"Removed existing folder '{dst}'.")
        except Exception as exc:  # pragma: no cover (OS-specific)
            print(f"Failed to remove existing path '{dst}'. Error: {exc}")
            return str(dst)

    src.rename(dst)
    print(f"'{src}' has been renamed to '{new_name}' successfully.")
    return str(dst)


def copy_file_to_new_folder(
    source_file: str | Path, destination_folder: str | Path, overwrite: bool = False
) -> Path:
    """Copy a file into a destination folder.

    Parameters
    ----------
    source_file:
        Path to the source file.
    destination_folder:
        Destination directory.
    overwrite:
        If True, an existing destination file with the same name will be removed first.

    Returns
    -------
    pathlib.Path
        Destination file path (even if the copy was skipped because it already existed).
    """
    src = Path(source_file)
    dst_folder = Path(destination_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    dst_file = dst_folder / src.name

    if not src.exists():
        print(f"The source file does not exist: {src}")
        return dst_file

    if dst_file.exists():
        print(f"The file already exists in the destination folder: {dst_file}")
        if overwrite:
            dst_file.unlink()
            print(f"Removed existing file: {dst_file}")
        else:
            print("Skipping copy.")
            return dst_file

    try:
        shutil.copy2(str(src), str(dst_file))
        print(f"File copied successfully to {dst_folder}")
    except Exception as exc:  # pragma: no cover (OS-specific)
        print(f"Failed to copy the file. Error: {exc}")

    return dst_file


def _send_to_trash(path: str | Path) -> None:
    # Lazy import avoids loading platform-specific bindings at module import time.
    from send2trash import send2trash

    send2trash(str(path))


def remove_folder_to_recycle_bin(folder_path: str | Path) -> None:
    """Move a folder to the recycle bin.

    Parameters
    ----------
    folder_path:
        Path to the folder.
    """
    try:
        _send_to_trash(folder_path)
        print(f"Folder {folder_path} has been moved to the recycle bin.")
    except Exception as exc:  # pragma: no cover (OS-specific)
        print(f"An error occurred while moving {folder_path} to the recycle bin: {exc}")


def move_files_to_bin(files: Sequence[str | Path]) -> None:
    """Move a list of files to the recycle bin.

    Parameters
    ----------
    files:
        Iterable of file paths.
    """
    for filename in files:
        p = Path(filename)
        if p.is_file():
            try:
                _send_to_trash(p)
                print(f"Moved to bin: {p}")
            except Exception as exc:  # pragma: no cover (OS-specific)
                print(f"Error moving {p} to bin: {exc}")
