from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lfptensorpipe.io.converter import df2mne

VENDOR_NAME = "Legacy (CSV)"


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


def _resolve_sr_and_unit(options: dict[str, Any] | None) -> tuple[float, str]:
    if options is None:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="Legacy (CSV) requires options with keys: 'sr' and 'unit'.",
        )

    if options.get("sr") is None:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="Missing required options['sr'] for Legacy (CSV).",
        )
    if options.get("unit") is None or not str(options.get("unit")).strip():
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="Missing required options['unit'] for Legacy (CSV).",
        )

    try:
        sr = float(options.get("sr"))
    except Exception as exc:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Invalid options['sr']: {options.get('sr')!r}",
        ) from exc

    if sr <= 0:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"options['sr'] must be > 0, got {sr}",
        )

    unit = str(options.get("unit")).strip()
    return sr, unit


def _read_channel_df(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Failed to read CSV: {csv_path}",
        ) from exc

    if df.empty:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"CSV has no rows: {csv_path}",
        )

    if df.columns.duplicated().any():
        dups = list(df.columns[df.columns.duplicated()])
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=f"Duplicated channel columns in CSV: {dups}",
        )

    if any(str(col).strip().lower() == "time" for col in df.columns):
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="Legacy (CSV) expects channel-only CSV. 'Time' column is not allowed.",
        )

    numeric_cols: list[str] = []
    for col in df.columns:
        series_raw = df[col]
        series_num = pd.to_numeric(series_raw, errors="coerce")
        if int(series_num.notna().sum()) == int(series_raw.notna().sum()):
            numeric_cols.append(col)

    if not numeric_cols:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="No numeric channel columns found in CSV.",
        )

    channel_df = df.loc[:, numeric_cols].apply(pd.to_numeric, errors="raise")
    return channel_df


def parse(
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str]]:
    version = "unknown"

    try:
        csv_path = _require_file_path(paths, "file_path")
        if csv_path.suffix.lower() != ".csv":
            raise ParseError(
                code="PARSE_INPUT_FILE_TYPE_MISMATCH",
                message=f"Legacy (CSV) file must be .csv: {csv_path.name}",
                version=version,
            )

        sr, unit = _resolve_sr_and_unit(options)
        channel_df = _read_channel_df(csv_path)

        try:
            raw = df2mne(channel_df, sr=sr, unit=unit)
        except Exception as exc:
            code = (
                "PARSE_UNIT_NORMALIZATION_FAILED"
                if "Unsupported voltage unit" in str(exc)
                else "PARSE_SCHEMA_INVALID"
            )
            raise ParseError(
                code=code,
                message=f"Failed to convert CSV to MNE Raw: {exc}",
                version=version,
            ) from exc

        # Legacy CSV contract: meas_date is always unset.
        raw.set_meas_date(None)
        report = {"vendor": VENDOR_NAME, "version": version, "status": "ok"}
        return raw, report
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(
            code="PARSE_INTERNAL_ERROR",
            message=f"Failed to parse legacy CSV record: {exc}",
            version=version,
        ) from exc
