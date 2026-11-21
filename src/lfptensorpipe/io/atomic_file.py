"""Single-file replacement for exported files."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
from typing import Any, BinaryIO
from uuid import uuid4


@contextmanager
def atomic_binary_file(path: Path, *, create_mode: int) -> Iterator[BinaryIO]:
    """Publish a sibling temporary file on success, preserving an existing mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    mode = create_mode if old_mode is None else create_mode & old_mode
    fd = os.open(temporary, flags, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
        if old_mode is not None:
            temporary.chmod(old_mode)
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def strict_json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def write_json_atomically(path: Path, payload: Any) -> None:
    """Replace the chosen JSON only after the complete UTF-8 content is ready."""
    data = strict_json_text(payload).encode("utf-8")
    with atomic_binary_file(path.expanduser().resolve(), create_mode=0o666) as stream:
        if stream.write(data) != len(data):
            raise OSError(f"Incomplete JSON write for {path}")
