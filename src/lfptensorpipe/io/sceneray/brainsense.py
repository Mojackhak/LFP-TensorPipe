from __future__ import annotations

import csv
import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from lfptensorpipe.io.converter import df2mne

VENDOR_NAME = "Sceneray"
CANONICAL_CHANNEL_RE = re.compile(r"^\d+[A-Za-z]*_\d+[A-Za-z]*(_[LR])?$")


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


@dataclass(frozen=True, slots=True)
class _TxtMeta:
    app_version: str | None
    sample_frequency_hz: float | None
    start_time_bjs: dt.datetime | None


def _require_file(paths: dict[str, str], key: str) -> Path:
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


def _normalize_channel_token(token: str) -> str:
    t = token.strip()
    if "~" in t:
        left, right = t.split("~", 1)
        t = f"{right.strip()}_{left.strip()}"
    return t


def _parse_channel_names(channel_tokens: Sequence[str], n_channels: int) -> list[str]:
    names: list[str] = []
    for token in channel_tokens:
        normalized = _normalize_channel_token(str(token))
        if not normalized:
            continue
        names.append(normalized)

    if len(names) != n_channels:
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=f"Channel count mismatch: channel row has {len(names)}, but CSV data has {n_channels}.",
        )

    invalid = [name for name in names if CANONICAL_CHANNEL_RE.fullmatch(name) is None]
    if invalid:
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=(
                f"Invalid channel names: {invalid}. "
                "Expected pattern: ^\\d+[A-Za-z]*_\\d+[A-Za-z]*(_[LR])?$"
            ),
        )
    return names


def _try_parse_float(text: str) -> float | None:
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if m is None:
            return None
        return float(m.group(0))


def _try_parse_datetime_bjs(text: str) -> dt.datetime | None:
    s = str(text).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _read_txt_meta(txt_path: Path) -> _TxtMeta:
    app_version: str | None = None
    sample_frequency_hz: float | None = None
    start_time_bjs: dt.datetime | None = None

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    for i, line in enumerate(lines):
        line_s = line.strip()
        line_lower = line_s.lower()

        if line_lower.startswith("app version"):
            app_version = line_s[len("App Version") :].strip() or None

        if "sample frequency" in line_lower:
            maybe = _try_parse_float(line_s)
            if maybe is not None and maybe > 0:
                sample_frequency_hz = float(maybe)

        if "start time" in line_lower and i + 1 < len(lines):
            maybe_dt = _try_parse_datetime_bjs(lines[i + 1])
            if maybe_dt is not None:
                start_time_bjs = maybe_dt

    return _TxtMeta(
        app_version=app_version,
        sample_frequency_hz=sample_frequency_hz,
        start_time_bjs=start_time_bjs,
    )


def _find_header_and_meta_rows(csv_path: Path) -> tuple[int, float | None, list[str]]:
    header_row = -1
    sfreq_hz: float | None = None
    channel_tokens: list[str] = []

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            first = str(row[0]).strip().lower()
            if first == "sampling rate":
                for cell in row[1:]:
                    maybe = _try_parse_float(cell)
                    if maybe is not None and maybe > 0:
                        sfreq_hz = float(maybe)
                        break
            if first == "channel":
                channel_tokens = [
                    str(cell).strip() for cell in row[1:] if str(cell).strip()
                ]
            if first == "packet index":
                header_row = row_idx
                break

    if header_row < 0:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="CSV header row not found: first column 'Packet Index' is missing.",
        )
    return header_row, sfreq_hz, channel_tokens


def _infer_n_channels_from_columns(df: pd.DataFrame) -> int:
    nums: list[int] = []
    for c in df.columns:
        m = re.match(r"^CH(\d+)", str(c).strip())
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="Could not infer channels: no columns matching ^CH\\d+",
        )
    return max(nums)


def _compress_int_ranges(values: Sequence[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = int(values[0])
    prev = int(values[0])
    for value in values[1:]:
        v = int(value)
        if v == prev + 1:
            prev = v
            continue
        ranges.append((start, prev))
        start = v
        prev = v
    ranges.append((start, prev))
    return ranges


def _drop_dups_and_fill_packet_index(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[int, int]], int]:
    packet_col = "Packet Index"
    if packet_col not in df.columns:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Missing required CSV column: {packet_col}",
        )

    df = df.copy()
    df[packet_col] = pd.to_numeric(df[packet_col], errors="coerce").astype("Int64")
    if df[packet_col].isna().any():
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="Packet Index contains non-numeric values.",
        )

    dup_mask = df.duplicated(subset=packet_col, keep=False)
    if dup_mask.any():
        df = df.loc[~dup_mask].copy()

    min_idx = int(df[packet_col].min())
    max_idx = int(df[packet_col].max())
    full_idx = np.arange(min_idx, max_idx + 1, dtype=int)
    present = set(int(v) for v in df[packet_col].to_list())
    missing_values = [int(i) for i in full_idx if int(i) not in present]

    # Reindexed loss-packet rows are zero-filled.
    df2 = (
        df.set_index(packet_col)
        .reindex(full_idx)
        .fillna(0.0)
        .reset_index()
        .rename(columns={"index": packet_col})
    )
    df2[packet_col] = df2[packet_col].astype(int)
    return df2, _compress_int_ranges(missing_values), min_idx


def _block_sort_key(col_name: str) -> tuple[int, str]:
    s = str(col_name).strip()
    if re.fullmatch(r"CH\d+", s):
        return (0, s)
    m = re.fullmatch(r"CH\d+\.(\d+)", s)
    if m:
        return (int(m.group(1)), s)
    m = re.fullmatch(r"CH\d+_(\d+)", s)
    if m:
        return (int(m.group(1)), s)
    m = re.search(r"[._](\d+)$", s)
    return (int(m.group(1)) if m else 0, s)


def _collect_channel_block_columns(
    df: pd.DataFrame, n_channels: int
) -> list[list[str]]:
    cols_by_ch: list[list[str]] = []
    for ch_i in range(1, n_channels + 1):
        pattern = re.compile(rf"^CH{ch_i}(?!\d)")
        cols = [c for c in df.columns if pattern.search(str(c))]
        if not cols:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"No block columns found for channel CH{ch_i}.",
            )
        cols_by_ch.append(sorted(cols, key=_block_sort_key))

    n_blocks = len(cols_by_ch[0])
    for ch_i, cols in enumerate(cols_by_ch, start=1):
        if len(cols) != n_blocks:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=(
                    f"Inconsistent block count: CH1 has {n_blocks} blocks, "
                    f"CH{ch_i} has {len(cols)} blocks."
                ),
            )
    return cols_by_ch


def _parse_tag_code(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip()
    if not s:
        return 0
    s = s.split(".")[0]
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return 0


def _parse_tag_index_ms(value: Any) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        return int(float(value))
    except Exception:
        return 0


def _build_tag_code_series(
    df_packets: pd.DataFrame, n_blocks: int, sfreq_hz: float
) -> np.ndarray:
    n_packets = len(df_packets)
    total_samples = int(n_packets * n_blocks)
    out = np.zeros(total_samples, dtype=int)

    if ("Tag Code" not in df_packets.columns) or (
        "Tag Index" not in df_packets.columns
    ):
        return out

    packet_ms = int(math.ceil(n_blocks * 1000.0 / sfreq_hz))
    for i in range(n_packets):
        code = _parse_tag_code(df_packets.loc[i, "Tag Code"])
        if code == 0:
            continue
        idx_ms = _parse_tag_index_ms(df_packets.loc[i, "Tag Index"])
        idx_ms_safe = max(1, idx_ms)
        if 1 <= idx_ms_safe <= packet_ms:
            block_1based = int(math.ceil((idx_ms_safe / 1000.0) * sfreq_hz))
            block0 = max(0, min(n_blocks - 1, block_1based - 1))
        else:
            block0 = 0
        out[i * n_blocks + block0] = code
    return out


def _append_annotations(
    raw: Any,
    *,
    sfreq_hz: float,
    tag_code_ts: np.ndarray,
    gap_specs: list[tuple[float, float]],
) -> None:
    import mne

    ann_list: list[mne.Annotations] = []

    idx = np.flatnonzero(tag_code_ts != 0)
    if idx.size > 0:
        onsets = idx.astype(float) / float(sfreq_hz)
        durations = np.zeros_like(onsets)
        descriptions = [f"tag_{int(tag_code_ts[i]):X}" for i in idx]
        ann_list.append(
            mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
            )
        )

    if gap_specs:
        gap_onsets = np.asarray([spec[0] for spec in gap_specs], dtype=float)
        gap_durations = np.asarray([spec[1] for spec in gap_specs], dtype=float)
        gap_desc = ["BAD_gap"] * len(gap_specs)
        ann_list.append(
            mne.Annotations(
                onset=gap_onsets,
                duration=gap_durations,
                description=gap_desc,
            )
        )

    if ann_list:
        merged = raw.annotations
        for ann in ann_list:
            merged = merged + ann
        raw.set_annotations(merged)


def _set_meas_date_from_bjs(raw: Any, start_time_bjs: dt.datetime | None) -> None:
    if start_time_bjs is None:
        return
    bjs_tz = dt.timezone(dt.timedelta(hours=8))
    dt_local = start_time_bjs
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=bjs_tz)
    dt_utc = dt_local.astimezone(dt.timezone.utc)
    raw.set_meas_date(dt_utc)


def parse(
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str]]:
    version = "unknown"

    try:
        csv_path = _require_file(paths, "file_path")
        if not csv_path.name.endswith("_uv.csv"):
            raise ParseError(
                code="PARSE_INPUT_FILE_TYPE_MISMATCH",
                message=f"Sceneray file must end with '_uv.csv': {csv_path.name}",
            )

        metadata_raw = paths.get("metadata_path")
        txt_path: Path | None = None
        if metadata_raw is not None and str(metadata_raw).strip():
            txt_candidate = Path(str(metadata_raw))
            if not txt_candidate.exists() or not txt_candidate.is_file():
                raise ParseError(
                    code="PARSE_SIDECAR_NOT_FOUND",
                    message=f"metadata_path not found: {txt_candidate}",
                )
            txt_path = txt_candidate
        else:
            inferred = Path(str(csv_path).replace("_uv.csv", ".txt"))
            if inferred.exists() and inferred.is_file():
                txt_path = inferred

        txt_meta: _TxtMeta | None = None
        if txt_path is not None:
            txt_meta = _read_txt_meta(txt_path)
            if txt_meta.app_version:
                version = txt_meta.app_version

        sr_from_options: float | None = None
        if options is not None and "sr" in options and options.get("sr") is not None:
            try:
                sr_from_options = float(options["sr"])
            except Exception as exc:
                raise ParseError(
                    code="PARSE_SCHEMA_INVALID",
                    message=f"Invalid options['sr']: {options['sr']!r}",
                    version=version,
                ) from exc
            if sr_from_options <= 0:
                raise ParseError(
                    code="PARSE_SCHEMA_INVALID",
                    message=f"Invalid options['sr'] <= 0: {sr_from_options}",
                    version=version,
                )

        header_row, sfreq_from_csv, channel_tokens = _find_header_and_meta_rows(
            csv_path
        )
        sfreq_hz = sr_from_options if sr_from_options is not None else sfreq_from_csv
        if (
            sfreq_hz is None
            and txt_meta is not None
            and txt_meta.sample_frequency_hz is not None
        ):
            sfreq_hz = float(txt_meta.sample_frequency_hz)
        if sfreq_hz is None or float(sfreq_hz) <= 0:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message="Sampling Rate unavailable from options['sr'], CSV, and txt metadata.",
                version=version,
            )

        df_packets = pd.read_csv(csv_path, skiprows=header_row, skipinitialspace=True)
        df_packets.columns = [str(c).strip() for c in df_packets.columns]
        df_packets, missing_ranges, min_packet_idx = _drop_dups_and_fill_packet_index(
            df_packets
        )

        n_channels = _infer_n_channels_from_columns(df_packets)
        channel_names = _parse_channel_names(channel_tokens, n_channels)
        cols_by_ch = _collect_channel_block_columns(df_packets, n_channels)
        n_packets = len(df_packets)
        n_blocks = len(cols_by_ch[0])

        cube = np.empty((n_packets, n_blocks, n_channels), dtype=float)
        for ch0, cols in enumerate(cols_by_ch):
            cube[:, :, ch0] = df_packets[cols].astype(float).to_numpy()
        data2d = cube.reshape(n_packets * n_blocks, n_channels)

        df_channels = pd.DataFrame(data2d, columns=channel_names)
        raw = df2mne(df_channels, sr=float(sfreq_hz), unit="uV")

        tag_code_ts = _build_tag_code_series(
            df_packets, n_blocks=n_blocks, sfreq_hz=float(sfreq_hz)
        )
        gap_specs: list[tuple[float, float]] = []
        for start_idx, end_idx in missing_ranges:
            onset = ((start_idx - min_packet_idx) * n_blocks) / float(sfreq_hz)
            duration = ((end_idx - start_idx + 1) * n_blocks) / float(sfreq_hz)
            gap_specs.append((float(onset), float(duration)))
        _append_annotations(
            raw,
            sfreq_hz=float(sfreq_hz),
            tag_code_ts=tag_code_ts,
            gap_specs=gap_specs,
        )

        _set_meas_date_from_bjs(
            raw,
            txt_meta.start_time_bjs if txt_meta is not None else None,
        )

        report = {"vendor": VENDOR_NAME, "version": version, "status": "ok"}
        return raw, report
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(
            code="PARSE_INTERNAL_ERROR",
            message=f"Failed to parse Sceneray record: {exc}",
            version=version,
        ) from exc
