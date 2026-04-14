from __future__ import annotations

import csv
import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfptensorpipe.io.converter import df2mne

VENDOR_NAME = "PINS"
CANONICAL_CHANNEL_RE = re.compile(r"^\d+[A-Za-z]*_\d+[A-Za-z]*(_[LR])?$")
PINS_RAW_CHANNEL_RE = re.compile(
    r"CH\d+_([0-9A-Za-z]+)\+([0-9A-Za-z]+)-(?:_([LR]))?$",
    flags=re.IGNORECASE,
)


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
class _MetaInfo:
    sampling_rate_hz: float | None
    sampling_start_bjs: dt.datetime | None
    sampling_device_version: str | None


@dataclass(frozen=True, slots=True)
class _Packet:
    num: int
    declared_len: int
    data: np.ndarray  # shape: (n_samples, n_channels)


def _parse_float(value: Any) -> float:
    s = str(value).strip()
    if not s:
        raise ValueError("empty numeric field")
    return float(s)


def _parse_int(value: Any) -> int:
    return int(float(str(value).strip()))


def _extract_first_float(text: str) -> float | None:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if m is None:
        return None
    return float(m.group(0))


def _parse_datetime(value: str) -> dt.datetime | None:
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


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


def _validate_record_filename(path: Path) -> None:
    name = path.name
    if (
        not name.startswith("EEGRealTime_")
        or name.endswith("_Para.txt")
        or path.suffix.lower() != ".txt"
    ):
        raise ParseError(
            code="PARSE_INPUT_FILE_TYPE_MISMATCH",
            message=(
                "PINS record file must start with 'EEGRealTime_', end with '.txt', "
                "and must not end with '_Para.txt'."
            ),
        )


def _resolve_metadata_path(paths: dict[str, str], file_path: Path) -> Path:
    explicit = paths.get("metadata_path")
    if explicit is not None and str(explicit).strip():
        meta = Path(str(explicit))
    else:
        meta = file_path.with_name(f"{file_path.stem}_Para.txt")
    if not meta.exists() or not meta.is_file():
        raise ParseError(
            code="PARSE_SIDECAR_NOT_FOUND",
            message=f"Metadata file not found: {meta}",
        )
    return meta


def _resolve_marker_path(paths: dict[str, str], file_path: Path) -> Path | None:
    explicit = paths.get("marker_path")
    if explicit is not None and str(explicit).strip():
        marker = Path(str(explicit))
        if not marker.exists() or not marker.is_file():
            raise ParseError(
                code="PARSE_INPUT_FILE_NOT_FOUND",
                message=f"marker_path file not found: {marker}",
            )
        return marker

    timestamp = file_path.stem.split("_")[-1]
    candidates = sorted(file_path.parent.glob(f"RealTimeMarker_*_{timestamp}.txt"))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ParseError(
            code="PARSE_SIDECAR_AMBIGUOUS",
            message=f"Multiple marker files matched timestamp {timestamp}: {[p.name for p in candidates]}",
        )
    return candidates[0]


def _read_metadata(meta_path: Path) -> _MetaInfo:
    sampling_rate_hz: float | None = None
    sampling_start_bjs: dt.datetime | None = None
    sampling_device_version: str | None = None

    with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            s_lower = s.lower()

            if s_lower.startswith("sampling rate:"):
                value = _extract_first_float(s)
                if value is not None and value > 0:
                    sampling_rate_hz = float(value)
                continue

            if s_lower.startswith("sampling start time:"):
                raw = s.split(":", 1)[1] if ":" in s else ""
                maybe_dt = _parse_datetime(raw)
                if maybe_dt is not None:
                    sampling_start_bjs = maybe_dt
                continue

            if s_lower.startswith("sampling device version:"):
                raw = s.split(":", 1)[1] if ":" in s else ""
                sampling_device_version = raw.strip() or None
                continue

    return _MetaInfo(
        sampling_rate_hz=sampling_rate_hz,
        sampling_start_bjs=sampling_start_bjs,
        sampling_device_version=sampling_device_version,
    )


def _resolve_sampling_rate(
    options: dict[str, Any] | None, meta: _MetaInfo, *, version: str
) -> float:
    if options is not None and options.get("sr") is not None:
        try:
            sr = float(options["sr"])
        except Exception as exc:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"Invalid options['sr']: {options['sr']!r}",
                version=version,
            ) from exc
        if sr <= 0:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"Invalid options['sr'] <= 0: {sr}",
                version=version,
            )
        return float(sr)

    if meta.sampling_rate_hz is not None and meta.sampling_rate_hz > 0:
        return float(meta.sampling_rate_hz)

    raise ParseError(
        code="PARSE_SCHEMA_INVALID",
        message="Sampling rate unavailable from options['sr'] and metadata.",
        version=version,
    )


def _normalize_channel_name(raw_name: str) -> str:
    # Example: CH1_8+1- -> 1_8 (cathode first, then anode)
    # Relaxed canonical rule also allows contact suffix letters and optional _L/_R side.
    m = PINS_RAW_CHANNEL_RE.fullmatch(str(raw_name).strip())
    if m is None:
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=(
                f"Unsupported PINS channel format: {raw_name!r}. "
                "Expected form like CH1_8+1- or CH1_8A+1B-_L."
            ),
        )
    anode = m.group(1)
    cathode = m.group(2)
    side = m.group(3)
    out = f"{cathode}_{anode}"
    if side:
        out = f"{out}_{side.upper()}"
    if CANONICAL_CHANNEL_RE.fullmatch(out) is None:
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=(
                f"Invalid normalized channel name: {out!r}. "
                "Expected pattern: ^\\d+[A-Za-z]*_\\d+[A-Za-z]*(_[LR])?$"
            ),
        )
    return out


def _finalize_packet(
    *,
    packet_num: int,
    declared_len: int,
    rows: list[list[float]],
    n_channels: int,
) -> _Packet:
    if declared_len <= 0:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Packet {packet_num} has non-positive declared length: {declared_len}",
        )
    if len(rows) != declared_len:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=(
                f"Packet {packet_num} row count mismatch: observed {len(rows)} vs declared {declared_len}"
            ),
        )
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != n_channels:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Packet {packet_num} data shape invalid: {arr.shape}",
        )
    return _Packet(num=packet_num, declared_len=declared_len, data=arr)


def _read_signal_packets(file_path: Path) -> tuple[list[str], list[_Packet]]:
    with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"Empty record file: {file_path}",
            ) from exc

        header_trim = [str(h).strip() for h in header]
        try:
            idx_packet_num = header_trim.index("Packet num")
            idx_packet_len = header_trim.index("Packet length")
        except ValueError as exc:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message="Missing required columns: 'Packet num' and/or 'Packet length'.",
            ) from exc

        channel_cols = [
            i
            for i, name in enumerate(header_trim)
            if i not in (idx_packet_num, idx_packet_len)
        ]
        if not channel_cols:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message="No channel columns found in PINS record file.",
            )

        ch_names = [_normalize_channel_name(header_trim[i]) for i in channel_cols]
        if len(ch_names) != len(set(ch_names)):
            raise ParseError(
                code="PARSE_CHANNEL_MAP_INVALID",
                message=f"Duplicate normalized channel names: {ch_names}",
            )

        packets: list[_Packet] = []
        current_packet_num: int | None = None
        current_packet_declared_len: int | None = None
        current_rows: list[list[float]] = []
        prev_packet_num: int | None = None

        for row_idx, row in enumerate(reader, start=2):
            if not row or all(not str(cell).strip() for cell in row):
                continue
            if len(row) < len(header_trim):
                raise ParseError(
                    code="PARSE_SCHEMA_INVALID",
                    message=f"Row {row_idx} has too few columns: {len(row)} < {len(header_trim)}",
                )

            try:
                packet_num = _parse_int(row[idx_packet_num])
                packet_len = _parse_int(row[idx_packet_len])
                vals = [_parse_float(row[i]) for i in channel_cols]
            except Exception as exc:
                raise ParseError(
                    code="PARSE_SCHEMA_INVALID",
                    message=f"Failed to parse row {row_idx}.",
                ) from exc

            if prev_packet_num is not None and packet_num < prev_packet_num:
                raise ParseError(
                    code="PARSE_TIMELINE_INVALID",
                    message=(
                        f"Packet num decreased at row {row_idx}: "
                        f"{packet_num} < {prev_packet_num}"
                    ),
                )

            if current_packet_num is None:
                current_packet_num = packet_num
                current_packet_declared_len = packet_len

            if packet_num != current_packet_num:
                packets.append(
                    _finalize_packet(
                        packet_num=current_packet_num,
                        declared_len=int(current_packet_declared_len),
                        rows=current_rows,
                        n_channels=len(ch_names),
                    )
                )
                current_rows = []
                current_packet_num = packet_num
                current_packet_declared_len = packet_len
            else:
                if (
                    current_packet_declared_len is not None
                    and packet_len != current_packet_declared_len
                ):
                    raise ParseError(
                        code="PARSE_SCHEMA_INVALID",
                        message=(
                            f"Packet {packet_num} has inconsistent Packet length values: "
                            f"{current_packet_declared_len} vs {packet_len}"
                        ),
                    )

            current_rows.append(vals)
            prev_packet_num = packet_num

        if current_packet_num is not None and current_packet_declared_len is not None:
            packets.append(
                _finalize_packet(
                    packet_num=current_packet_num,
                    declared_len=int(current_packet_declared_len),
                    rows=current_rows,
                    n_channels=len(ch_names),
                )
            )

    if not packets:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"No packet samples found in file: {file_path}",
        )

    return ch_names, packets


def _mode_packet_length(packets: list[_Packet]) -> int:
    counter = Counter(pkt.declared_len for pkt in packets)
    # Deterministic tie-break: longer packet length first when counts are equal.
    length, _count = sorted(
        counter.items(), key=lambda kv: (kv[1], kv[0]), reverse=True
    )[0]
    return int(length)


def _reconstruct_with_gaps(
    packets: list[_Packet],
    *,
    n_channels: int,
    mode_packet_len: int,
    sfreq_hz: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    if mode_packet_len <= 0:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Invalid mode packet length: {mode_packet_len}",
        )

    chunks: list[np.ndarray] = []
    gap_specs: list[tuple[float, float]] = []
    sample_cursor = 0

    expected_packet = 0
    for pkt in packets:
        if pkt.num > expected_packet:
            missing_packets = int(pkt.num - expected_packet)
            missing_rows = int(missing_packets * mode_packet_len)
            if missing_rows > 0:
                chunks.append(np.zeros((missing_rows, n_channels), dtype=float))
                gap_specs.append((sample_cursor / sfreq_hz, missing_rows / sfreq_hz))
                sample_cursor += missing_rows

        chunks.append(pkt.data)
        sample_cursor += int(pkt.data.shape[0])
        expected_packet = int(pkt.num + 1)

    if not chunks:
        return np.empty((0, n_channels), dtype=float), []
    return np.vstack(chunks), gap_specs


def _parse_duration_to_seconds(value: str) -> float:
    s = str(value).strip()
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid duration format: {value!r}")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = float(parts[2])
    return float(hh * 3600 + mm * 60 + ss)


def _normalize_marker_source(source: str) -> str:
    s = str(source).strip().lower()
    if not s:
        return "source"
    if s in {"mannual", "manual"}:
        return "manual"

    m = re.fullmatch(r"外部打标源(\d+)", str(source).strip())
    if m is not None:
        return f"external_{m.group(1)}"

    token = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if token:
        return token

    digits = re.findall(r"\d+", s)
    if digits:
        return f"source_{'_'.join(digits)}"
    return "source"


def _read_marker_events(
    marker_path: Path | None,
    *,
    eeg_start_bjs: dt.datetime | None,
) -> list[tuple[float, str]]:
    if marker_path is None:
        return []

    marker_start_bjs: dt.datetime | None = None
    rows: list[list[str]] = []
    with marker_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        rows = [list(r) for r in reader if r]

    header_idx = None
    for i, row in enumerate(rows):
        first = str(row[0]).strip().lower()
        if first.startswith("marker start time"):
            raw = row[0].split(":", 1)[1] if ":" in row[0] else ""
            marker_start_bjs = _parse_datetime(raw)
            continue
        if first == "duration":
            header_idx = i
            break

    if header_idx is None:
        return []

    base_offset = 0.0
    if eeg_start_bjs is not None and marker_start_bjs is not None:
        base_offset = float((marker_start_bjs - eeg_start_bjs).total_seconds())

    out: list[tuple[float, str]] = []
    for row in rows[header_idx + 1 :]:
        if len(row) < 3:
            continue
        try:
            duration_sec = _parse_duration_to_seconds(row[0])
            source = _normalize_marker_source(row[1])
            sn = _parse_int(row[2])
        except Exception:
            continue
        onset = float(base_offset + duration_sec)
        out.append((onset, f"tag_{source}_{sn}"))
    return out


def _set_meas_date_from_bjs(raw: Any, start_bjs: dt.datetime | None) -> None:
    if start_bjs is None:
        return
    bjs_tz = dt.timezone(dt.timedelta(hours=8))
    local_dt = (
        start_bjs if start_bjs.tzinfo is not None else start_bjs.replace(tzinfo=bjs_tz)
    )
    utc_dt = local_dt.astimezone(dt.timezone.utc)
    raw.set_meas_date(utc_dt)


def _append_annotations(
    raw: Any,
    *,
    marker_events: list[tuple[float, str]],
    gap_specs: list[tuple[float, float]],
) -> None:
    import mne

    ann = raw.annotations
    if marker_events:
        onsets = np.asarray([x[0] for x in marker_events], dtype=float)
        durations = np.zeros_like(onsets)
        desc = [x[1] for x in marker_events]
        ann = ann + mne.Annotations(onset=onsets, duration=durations, description=desc)

    if gap_specs:
        onsets = np.asarray([x[0] for x in gap_specs], dtype=float)
        durations = np.asarray([x[1] for x in gap_specs], dtype=float)
        desc = ["BAD_gap"] * len(gap_specs)
        ann = ann + mne.Annotations(onset=onsets, duration=durations, description=desc)

    raw.set_annotations(ann)


def parse(
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str]]:
    version = "unknown"

    try:
        file_path = _require_file_path(paths, "file_path")
        _validate_record_filename(file_path)

        metadata_path = _resolve_metadata_path(paths, file_path)
        marker_path = _resolve_marker_path(paths, file_path)

        meta = _read_metadata(metadata_path)
        if meta.sampling_device_version:
            version = meta.sampling_device_version

        sfreq_hz = _resolve_sampling_rate(options, meta, version=version)
        ch_names, packets = _read_signal_packets(file_path)
        mode_packet_len = _mode_packet_length(packets)
        data, gap_specs = _reconstruct_with_gaps(
            packets,
            n_channels=len(ch_names),
            mode_packet_len=mode_packet_len,
            sfreq_hz=sfreq_hz,
        )

        df = pd.DataFrame(data, columns=ch_names)
        raw = df2mne(df, sr=sfreq_hz, unit="uV")

        marker_events = _read_marker_events(
            marker_path, eeg_start_bjs=meta.sampling_start_bjs
        )
        _append_annotations(raw, marker_events=marker_events, gap_specs=gap_specs)
        _set_meas_date_from_bjs(raw, meta.sampling_start_bjs)

        report = {"vendor": VENDOR_NAME, "version": version, "status": "ok"}
        return raw, report
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(
            code="PARSE_INTERNAL_ERROR",
            message=f"Failed to parse PINS record: {exc}",
            version=version,
        ) from exc
