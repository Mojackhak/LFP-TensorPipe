from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VENDOR_NAME = "Medtronic"
SECTION_NAME = "BrainSenseTimeDomain"
MICROVOLTS_TO_VOLTS = 1e-6
CANONICAL_CHANNEL_RE = re.compile(r"^\d+[A-Za-z]*_\d+[A-Za-z]*(_[LR])?$")
_CONTACT_TOKEN_RE = re.compile(r"^\d+[A-Z]*$")
_LETTER_TOKEN_RE = re.compile(r"^[A-Z]+$")

_NUMBER_WORDS: dict[str, str] = {
    "ZERO": "0",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
    "TEN": "10",
    "ELEVEN": "11",
    "TWELVE": "12",
    "THIRTEEN": "13",
    "FOURTEEN": "14",
    "FIFTEEN": "15",
}


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
class _Entry:
    order: int
    run_start_utc: dt.datetime
    sfreq_hz: float
    channel: str
    data_v: np.ndarray


@dataclass(frozen=True, slots=True)
class _Run:
    run_start_utc: dt.datetime
    start_sample: int
    n_times: int
    channel_data: dict[str, np.ndarray]


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


def _load_session_json(file_path: Path) -> dict[str, Any]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as exc:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Invalid JSON: {file_path}",
        ) from exc

    if not isinstance(obj, dict):
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message="Top-level JSON must be an object.",
        )
    return obj


def _extract_version(session: dict[str, Any]) -> str:
    raw = session.get("ProgrammerVersion")
    if raw is None:
        return "unknown"
    text = str(raw).strip()
    return text if text else "unknown"


def _parse_iso8601_to_utc(value: Any, *, version: str, field: str) -> dt.datetime:
    raw = str(value).strip()
    if not raw:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Missing {field}.",
            version=version,
        )
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except Exception as exc:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"Invalid ISO datetime in {field}: {raw!r}",
            version=version,
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _consume_contact(
    tokens: list[str], start: int, *, version: str, original: str
) -> tuple[str, int]:
    if start >= len(tokens):
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=f"Incomplete channel token in {original!r}.",
            version=version,
        )

    token = tokens[start]
    if _CONTACT_TOKEN_RE.fullmatch(token) is None:
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=f"Invalid contact token {token!r} in channel {original!r}.",
            version=version,
        )

    contact = token
    idx = start + 1
    while idx < len(tokens) and _LETTER_TOKEN_RE.fullmatch(tokens[idx]) is not None:
        contact = f"{contact}{tokens[idx]}"
        idx += 1
    return contact, idx


def _normalize_channel_name(value: Any, *, version: str) -> str:
    original = str(value).strip()
    if not original:
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message="Missing Channel value.",
            version=version,
        )

    text = original.upper()
    side: str | None = None
    for suffix, mapped in (("_LEFT", "L"), ("_RIGHT", "R"), ("_L", "L"), ("_R", "R")):
        if text.endswith(suffix):
            side = mapped
            text = text[: -len(suffix)]
            break

    text = text.replace("_AND_", "_")
    raw_tokens = [tok for tok in re.split(r"[_\s]+", text) if tok and tok != "AND"]
    mapped_tokens = [_NUMBER_WORDS.get(tok, tok) for tok in raw_tokens]

    contact_1, idx = _consume_contact(
        mapped_tokens, 0, version=version, original=original
    )
    contact_2, idx = _consume_contact(
        mapped_tokens, idx, version=version, original=original
    )
    if idx != len(mapped_tokens):
        extra = "_".join(mapped_tokens[idx:])
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=f"Unsupported extra channel tokens in {original!r}: {extra!r}",
            version=version,
        )

    out = f"{contact_1}_{contact_2}"
    if side is not None:
        out = f"{out}_{side}"
    if CANONICAL_CHANNEL_RE.fullmatch(out) is None:
        raise ParseError(
            code="PARSE_CHANNEL_MAP_INVALID",
            message=(
                f"Invalid normalized channel name: {out!r}. "
                "Expected pattern ^\\d+[A-Za-z]*_\\d+[A-Za-z]*(_[LR])?$"
            ),
            version=version,
        )
    return out


def _parse_entries(session: dict[str, Any], *, version: str) -> list[_Entry]:
    section = session.get(SECTION_NAME)
    if not isinstance(section, list) or not section:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"{SECTION_NAME} must be a non-empty list.",
            version=version,
        )

    entries: list[_Entry] = []
    for order, entry in enumerate(section):
        if not isinstance(entry, dict):
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"{SECTION_NAME}[{order}] must be an object.",
                version=version,
            )

        run_start_utc = _parse_iso8601_to_utc(
            entry.get("FirstPacketDateTime"),
            version=version,
            field="FirstPacketDateTime",
        )

        try:
            sfreq_hz = float(entry.get("SampleRateInHz"))
        except Exception as exc:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"Invalid SampleRateInHz at {SECTION_NAME}[{order}].",
                version=version,
            ) from exc
        if sfreq_hz <= 0:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"SampleRateInHz must be > 0 at {SECTION_NAME}[{order}].",
                version=version,
            )

        if "Gain" not in entry:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"Missing Gain at {SECTION_NAME}[{order}].",
                version=version,
            )
        try:
            float(entry.get("Gain"))
        except Exception as exc:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"Invalid Gain at {SECTION_NAME}[{order}].",
                version=version,
            ) from exc

        channel = _normalize_channel_name(entry.get("Channel"), version=version)
        td = entry.get("TimeDomainData")
        if not isinstance(td, list) or len(td) == 0:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"TimeDomainData must be a non-empty list at {SECTION_NAME}[{order}].",
                version=version,
            )
        try:
            data_uV = np.asarray(td, dtype=float).ravel()
        except Exception as exc:
            raise ParseError(
                code="PARSE_SCHEMA_INVALID",
                message=f"Non-numeric TimeDomainData at {SECTION_NAME}[{order}].",
                version=version,
            ) from exc

        entries.append(
            _Entry(
                order=order,
                run_start_utc=run_start_utc,
                sfreq_hz=sfreq_hz,
                channel=channel,
                data_v=data_uV * MICROVOLTS_TO_VOLTS,
            )
        )

    if not entries:
        raise ParseError(
            code="PARSE_SCHEMA_INVALID",
            message=f"No valid entries in {SECTION_NAME}.",
            version=version,
        )
    return entries


def _collect_channel_order(entries: list[_Entry]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for entry in sorted(entries, key=lambda item: item.order):
        if entry.channel in seen:
            continue
        seen.add(entry.channel)
        out.append(entry.channel)
    return out


def _build_runs(
    entries: list[_Entry], *, sfreq_hz: float, version: str
) -> tuple[list[_Run], list[tuple[float, float]]]:
    grouped: dict[dt.datetime, dict[str, np.ndarray]] = {}
    for entry in entries:
        channel_map = grouped.setdefault(entry.run_start_utc, {})
        if entry.channel in channel_map:
            raise ParseError(
                code="PARSE_CHANNEL_MAP_INVALID",
                message=f"Duplicate channel {entry.channel!r} within run {entry.run_start_utc.isoformat()}.",
                version=version,
            )
        channel_map[entry.channel] = entry.data_v

    sorted_starts = sorted(grouped.keys())
    global_start_utc = sorted_starts[0]
    runs: list[_Run] = []
    gap_specs: list[tuple[float, float]] = []

    prev_end_sample = 0
    for idx, run_start_utc in enumerate(sorted_starts):
        offset_sec = (run_start_utc - global_start_utc).total_seconds()
        start_sample = int(round(offset_sec * sfreq_hz))
        if start_sample < 0:
            raise ParseError(
                code="PARSE_TIMELINE_INVALID",
                message="Run start time is earlier than global start after sorting.",
                version=version,
            )

        channel_data = grouped[run_start_utc]
        run_len = max(int(arr.size) for arr in channel_data.values())

        if idx > 0:
            if start_sample < prev_end_sample:
                raise ParseError(
                    code="PARSE_TIMELINE_INVALID",
                    message=(
                        "Run overlap detected: "
                        f"{run_start_utc.isoformat()} starts before previous run ends."
                    ),
                    version=version,
                )
            if start_sample > prev_end_sample:
                gap_specs.append(
                    (
                        prev_end_sample / sfreq_hz,
                        (start_sample - prev_end_sample) / sfreq_hz,
                    )
                )

        runs.append(
            _Run(
                run_start_utc=run_start_utc,
                start_sample=start_sample,
                n_times=run_len,
                channel_data=channel_data,
            )
        )
        prev_end_sample = start_sample + run_len

    return runs, gap_specs


def _build_raw(
    runs: list[_Run],
    channel_order: list[str],
    *,
    sfreq_hz: float,
    ch_type: str,
) -> Any:
    import mne

    n_channels = len(channel_order)
    total_n_times = max(run.start_sample + run.n_times for run in runs)
    data = np.zeros((n_channels, total_n_times), dtype=float)
    ch_index = {name: idx for idx, name in enumerate(channel_order)}

    for run in runs:
        for name, series in run.channel_data.items():
            dst = ch_index[name]
            off = run.start_sample
            end = off + int(series.size)
            data[dst, off:end] = series

    try:
        info = mne.create_info(
            ch_names=channel_order, sfreq=sfreq_hz, ch_types=[ch_type] * n_channels
        )
    except Exception:
        info = mne.create_info(
            ch_names=channel_order, sfreq=sfreq_hz, ch_types=["seeg"] * n_channels
        )

    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_meas_date(runs[0].run_start_utc)
    return raw


def _append_gap_annotations(raw: Any, gap_specs: list[tuple[float, float]]) -> None:
    if not gap_specs:
        return
    import mne

    onsets = np.asarray([spec[0] for spec in gap_specs], dtype=float)
    durations = np.asarray([spec[1] for spec in gap_specs], dtype=float)
    descriptions = ["BAD_gap"] * len(gap_specs)
    raw.set_annotations(
        raw.annotations
        + mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    )


def parse(
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str]]:
    version = "unknown"

    try:
        file_path = _require_file_path(paths, "file_path")
        if file_path.suffix.lower() != ".json":
            raise ParseError(
                code="PARSE_INPUT_FILE_TYPE_MISMATCH",
                message=f"Medtronic file must be .json: {file_path.name}",
            )

        session = _load_session_json(file_path)
        version = _extract_version(session)

        entries = _parse_entries(session, version=version)
        sfreq_set = {round(entry.sfreq_hz, 8) for entry in entries}
        if len(sfreq_set) != 1:
            raise ParseError(
                code="PARSE_TIMELINE_INVALID",
                message=f"Mixed SampleRateInHz values found: {sorted(sfreq_set)}",
                version=version,
            )
        sfreq_hz = float(next(iter(sfreq_set)))

        channel_order = _collect_channel_order(entries)
        runs, gap_specs = _build_runs(entries, sfreq_hz=sfreq_hz, version=version)

        ch_type = "dbs"
        if options is not None and options.get("ch_type") is not None:
            ch_type = str(options.get("ch_type")).strip() or "dbs"

        raw = _build_raw(runs, channel_order, sfreq_hz=sfreq_hz, ch_type=ch_type)
        _append_gap_annotations(raw, gap_specs)
        report = {"vendor": VENDOR_NAME, "version": version, "status": "ok"}
        return raw, report
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(
            code="PARSE_INTERNAL_ERROR",
            message=f"Failed to parse Medtronic record: {exc}",
            version=version,
        ) from exc
