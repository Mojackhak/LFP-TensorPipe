"""Read one record's information without modifying the study project."""

from __future__ import annotations

from contextlib import ExitStack
from itertools import chain, permutations
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import yaml

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.io.source_metadata import SourceMetadata
from .record_info import (
    InputChannel,
    OutputChannel,
    RecordInformation,
    RecordProvenance,
    acquisition_channels,
    channel_tables,
)
from .validation import rawdata_record_fif_path, sourcedata_record_raw_dir


def _vendor_header(path: Path, vendor: str) -> bool:
    """Distinguish vendor exports from user-named Legacy files before recovery."""
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
        rows = csv.reader(stream)
        if vendor == "PINS":
            header = next(rows, [])
            return {"Packet num", "Packet length"}.issubset(c.strip() for c in header)
        channel_header = False
        for row in rows:
            if not row or not row[0].strip():
                continue
            key = row[0].strip().lower()
            if key == "packet index":
                return channel_header
            if key == "channel":
                channel_header = True
            # The body of a numeric Legacy CSV starts without packet metadata.
            if key[:1] in "0123456789-.":
                return False
    return False


def _historical_source(
    folder: Path, known_file: str | None = None
) -> tuple[str | None, Path | None, dict[str, Any] | None]:
    if not folder.is_dir():
        return None, None, None
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if known_file:
        files = [path for path in files if path.name == known_file]
    identified: list[tuple[str, Path, dict[str, Any] | None]] = []
    for f in files:
        if f.name.endswith("_uv.csv") and _vendor_header(f, "Sceneray"):
            identified.append(("Sceneray", f, None))
        elif (
            f.name.startswith("EEGRealTime_")
            and f.suffix.lower() == ".txt"
            and not f.name.endswith("_Para.txt")
            and _vendor_header(f, "PINS")
        ):
            identified.append(("PINS", f, None))
        elif f.suffix.lower() == ".json":
            try:
                with f.open(encoding="utf-8-sig") as stream:
                    session = json.load(stream)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(session, dict) and "BrainSenseTimeDomain" in session:
                identified.append(("Medtronic", f, session))
    if identified:
        return identified[0] if len(identified) == 1 else (None, None, None)
    # MNE selects the reader by the primary extension. A BrainVision header
    # or an EEGLAB SET can have separate data and marker files.
    primaries = [f for f in files if f.suffix.lower() in {".vhdr", ".set", ".lay"}]
    if len(primaries) == 1:
        return "Legacy (MNE supported)", primaries[0], None
    candidates = [
        f
        for f in files
        if f.suffix.lower()
        not in {
            ".txt",
            ".json",
            ".png",
            ".wav",
            ".vmrk",
            ".fdt",
            ".pkf",
            ".pkl",
            ".yaml",
            ".yml",
            ".md",
        }
        and not f.name.endswith(".annotations.csv")
    ]
    if len(candidates) == 1 and candidates[0].suffix.lower() == ".csv":
        return "Legacy (CSV)", candidates[0], None
    if len(candidates) == 1:
        return "Legacy (MNE supported)", candidates[0], None
    return None, None, None


def _verified_outputs(
    inputs: list[InputChannel], values: NDArray[Any], raw: Any, shift: int
) -> list[OutputChannel]:
    """Return only operations with a unique full-trace match at FIF precision."""
    count = values.shape[1] - shift
    if count != raw.n_times or not inputs:
        return []
    # Compare the shared original sample axis. Verify the sync prefix too.
    observed = raw.get_data()
    if shift > 0:
        values = values[:, shift:]
    elif shift < 0:
        if np.any(observed[:, :-shift] != 0):
            return []
        observed = observed[:, -shift:]
    if not values.shape[1]:
        return []
    dtype = np.float32 if raw.orig_format == "single" else np.float64
    observed = observed.astype(dtype, copy=False)
    sample = np.linspace(0, values.shape[1] - 1, min(257, values.shape[1]), dtype=int)
    preview = values[:, sample]
    viable = []
    for index, row in enumerate(observed):
        finite = row[np.isfinite(row)]
        if finite.size and finite.min() != finite.max():
            viable.append(index)
    found: dict[int, list[OutputChannel]] = {idx: [] for idx in viable}
    ids = range(len(inputs))
    operations = chain(
        ((i, None) for i in ids), ((None, i) for i in ids), permutations(ids, 2)
    )

    def difference(data: NDArray[Any], i: int | None, j: int | None) -> NDArray[Any]:
        waveform: NDArray[Any]
        if i is None:
            waveform = -data[j]
        elif j is None:
            waveform = data[i]
        else:
            waveform = data[i] - data[j]
        return waveform

    for i, j in operations:
        trial = difference(preview, i, j).astype(dtype, copy=False)
        candidates = [
            k
            for k in viable
            if np.array_equal(trial, observed[k, sample], equal_nan=True)
        ]
        if candidates:
            trace = difference(values, i, j).astype(dtype, copy=False)
            for k in candidates:
                if np.array_equal(trace, observed[k], equal_nan=True):
                    found[k].append(
                        OutputChannel(
                            raw.ch_names[k],
                            inputs[i].name if i is not None else None,
                            inputs[j].name if j is not None else None,
                            "waveform match",
                        )
                    )
    return [options[0] for options in found.values() if len(options) == 1]


def _recover(
    info: RecordProvenance,
    source: Path,
    session: dict[str, Any] | None,
    raw: Any,
    sync: dict[str, Any],
) -> None:
    import mne
    from lfptensorpipe.io.csv import brainsense as legacy_csv
    from lfptensorpipe.io.medtronic import brainsense as med
    from lfptensorpipe.io.pins import brainsense as pins
    from lfptensorpipe.io.sceneray import brainsense as scene

    kind = info.import_type or ""
    metadata = SourceMetadata()
    if kind.startswith("Legacy"):
        # Legacy contact roles AND signal expressions are always unknown.
        # Only enumerate the available inputs; never match their waveforms.
        if kind == "Legacy (CSV)":
            names = legacy_csv._read_channel_df(source).columns
        else:
            with mne.io.read_raw(source, preload=False, verbose="ERROR") as original:
                names = original.ch_names
                info.source_sampling_rate_hz = float(original.info["sfreq"])
        info.inputs = acquisition_channels(kind, names, metadata)
        return

    decode_rate = sync.get("sfreq_before_hz") or (
        raw.info["sfreq"] if raw is not None else None
    )
    values = None
    if kind == "Sceneray":
        try:
            original, _ = scene.parse(
                {"file_path": str(source)},
                {"sr": decode_rate} if decode_rate else None,
                source_metadata=metadata,
            )
        except scene.ParseError as exc:
            if not metadata.labels or decode_rate or metadata.sampling_rate_hz:
                raise ValueError(str(exc)) from exc
        else:
            with original:
                if raw is not None:
                    values = original.get_data()
    elif kind == "PINS":
        meta = source.with_name(source.stem + "_Para.txt")
        if meta.is_file():
            metadata.metadata_file = meta.name
            metadata.sampling_rate_hz = pins._read_metadata(meta).sampling_rate_hz
        try:
            marker = pins._resolve_marker_path({}, source)
        except pins.ParseError as exc:
            if exc.code != "PARSE_SIDECAR_AMBIGUOUS":
                raise
            marker = None
        metadata.marker_file = marker.name if marker else None
        names, packets = pins._read_signal_packets(source, metadata)
        if raw is not None:
            data, _ = pins._reconstruct_with_gaps(
                packets,
                n_channels=len(names),
                mode_packet_len=pins._mode_packet_length(packets),
                sfreq_hz=metadata.sampling_rate_hz
                or sync.get("sfreq_before_hz")
                or float(raw.info["sfreq"]),
            )
            values = data.T * 1e-6
    elif kind == "Medtronic" and session is not None:
        original, _ = med.parse_session(session, source_metadata=metadata)
        with original:
            metadata.labels = {n: metadata.labels[n] for n in original.ch_names}
            if raw is not None:
                values = original.get_data()
    info.inputs = acquisition_channels(kind, metadata.labels, metadata)
    info.metadata_file = metadata.metadata_file
    info.marker_file = metadata.marker_file
    info.source_sampling_rate_hz = metadata.sampling_rate_hz
    if raw is not None and values is not None:
        try:
            if sync and "shift_samples" not in sync:
                shift = round(float(sync["lag_s"]) * float(sync["sfreq_after_hz"]))
            else:
                shift = int(sync.get("shift_samples", 0))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "The Sync configuration does not identify the applied sample shift."
            ) from exc
        info.outputs = _verified_outputs(info.inputs, values, raw, shift)


def read_record_information(context: RecordContext) -> RecordInformation:
    """Inspect only the selected record, including partial or source-only records."""
    import mne

    root, subject, record = context.project_root, context.subject, context.record
    fif = rawdata_record_fif_path(root, subject, record)
    sidecar = fif.with_name("record_info.json")
    sync_path = PathResolver(context).import_sync_dir() / "config.yml"
    notices = []
    sync = {}
    known_timebase = True
    if sync_path.is_file():
        try:
            sync = yaml.safe_load(sync_path.read_text(encoding="utf-8"))
            if not isinstance(sync, dict):
                raise ValueError("Expected a Sync configuration object.")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            notices.append(f"Sync details unavailable: {exc}")
            sync = {}
            known_timebase = False
    before = sync.get("sfreq_before_hz")
    if before is not None and (
        type(before) not in (int, float) or not np.isfinite(before) or before <= 0
    ):
        notices.append("Invalid pre-Sync sampling rate.")
        sync.pop("sfreq_before_hz")
    external_source = sync.get("external_source")
    external = (
        external_source.get("path") if isinstance(external_source, dict) else None
    )
    sync_name = (
        Path(external.replace("\\", "/")).name
        if isinstance(external, str) and external
        else None
    )
    with ExitStack() as files:
        raw = None
        if fif.is_file():
            try:
                raw = files.enter_context(
                    mne.io.read_raw_fif(fif, preload=False, verbose="ERROR")
                )
            except (OSError, ValueError, RuntimeError) as exc:
                notices.append(f"Imported FIF unavailable: {exc}")
        info = RecordProvenance(None, None)
        historical = not sidecar.is_file()
        if not historical:
            try:
                info = RecordProvenance.read(sidecar)
                historical = info.import_type is None and not info.inputs
            except (OSError, ValueError) as exc:
                notices.append(f"Saved source details unavailable: {exc}")
        if historical:
            from lfptensorpipe.io.medtronic import ParseError as MedtronicError
            from lfptensorpipe.io.pins import ParseError as PinsError
            from lfptensorpipe.io.csv import ParseError as LegacyCSVError

            kind, source, session = _historical_source(
                sourcedata_record_raw_dir(root, subject, record), info.source_file
            )
            info = RecordProvenance(kind, source.name if source else info.source_file)
            if source:
                try:
                    _recover(
                        info, source, session, raw if known_timebase else None, sync
                    )
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    RuntimeError,
                    PinsError,
                    LegacyCSVError,
                    MedtronicError,
                ) as exc:
                    notices.append(f"Original recording details unavailable: {exc}")
        names = raw.ch_names if raw is not None else [r.name for r in info.outputs]
        legacy = historical and (info.import_type or "").startswith("Legacy")
        ins, outs = channel_tables(info, names, historical_legacy=legacy)
        return RecordInformation(
            subject=subject,
            record=record,
            provenance=info,
            sampling_rate=float(raw.info["sfreq"]) if raw is not None else None,
            samples=int(raw.n_times) if raw is not None else None,
            input_rows=ins,
            output_rows=outs,
            pre_sync_rate=(
                sync.get("sfreq_before_hz")
                if sync.get("correct_sfreq") is True
                else None
            ),
            sync_source=sync_name,
            has_fif=raw is not None,
            notice="\n".join(notices) or None,
        )
