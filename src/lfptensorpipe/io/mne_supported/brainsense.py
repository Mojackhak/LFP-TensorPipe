from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

VENDOR_NAME = "Legacy (MNE supported)"


@dataclass(frozen=True, slots=True)
class ParseError(Exception):
    code: str
    message: str
    vendor: str = VENDOR_NAME
    version: str = "unknown"
    status: str = "error"

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    @property
    def report(self) -> dict[str, str]:
        return {
            "vendor": self.vendor,
            "version": self.version,
            "status": self.status,
        }


def _require_file_path(paths: dict[str, str], key: str) -> Path:
    raw = paths.get(key)
    if raw is None or not str(raw).strip():
        raise ParseError(
            code="PARSE_INPUT_MISSING_KEY",
            message=f"Missing required key in paths: {key}",
        )
    path = Path(str(raw))
    if not path.exists() or not path.is_file():
        raise ParseError(
            code="PARSE_INPUT_FILE_NOT_FOUND",
            message=f"Input file not found: {path}",
        )
    return path


def _validate_options(options: dict[str, Any] | None) -> None:
    if options is None:
        return
    if len(options) == 0:
        return
    raise ParseError(
        code="PARSE_SCHEMA_INVALID",
        message=(
            "Legacy (MNE supported) parser does not accept options. "
            f"Got keys: {sorted(options.keys())}"
        ),
    )


def parse(
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str]]:
    version = "unknown"

    try:
        _validate_options(options)
        file_path = _require_file_path(paths, "file_path")

        import mne

        try:
            # Keep original meas_date/annotations as provided by MNE reader.
            raw = mne.io.read_raw(str(file_path), preload=True, verbose="error")
        except Exception as exc:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"Failed to read MNE-supported file: {file_path}",
                version=version,
            ) from exc

        report = {"vendor": VENDOR_NAME, "version": version, "status": "ok"}
        return raw, report
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(
            code="PARSE_INTERNAL_ERROR",
            message=f"Failed to parse legacy MNE-supported record: {exc}",
            version=version,
        ) from exc
