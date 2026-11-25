"""Recording provenance and read-only presentation of imported channels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any

from lfptensorpipe.io.atomic_file import write_json_atomically
from lfptensorpipe.io.source_metadata import SourceMetadata

UNKNOWN = "Unknown"
VENDORS = {"Sceneray", "PINS", "Medtronic"}
CONTACT_PAIR = re.compile(r"(\d+[A-Za-z]*)_(\d+[A-Za-z]*)(?:_([LR]))?")


@dataclass
class InputChannel:
    name: str
    source_label: str
    recording_contact: str | None = None
    reference_contact: str | None = None
    recording_contact_source: str | None = None
    reference_contact_source: str | None = None
    device_positive: str | None = None
    device_negative: str | None = None


@dataclass
class OutputChannel:
    name: str
    signal_channel: str | None = None
    reference_channel: str | None = None
    operation_source: str | None = None


def contact_roles(name: str) -> tuple[str | None, str | None]:
    match = CONTACT_PAIR.fullmatch(name)
    if not match:
        return None, None
    recording, reference, side = match.groups()
    suffix = f" ({side})" if side else ""
    return recording + suffix, reference + suffix


def _apply_sceneray_polarity(channel: InputChannel) -> None:
    """Resolve the device terminals from the original reference~recording label."""
    if "~" in channel.source_label:
        from lfptensorpipe.io.sceneray.brainsense import _normalize_channel_token

        positive, negative = contact_roles(
            _normalize_channel_token(channel.source_label)
        )
        if positive is not None:
            channel.device_positive = positive
            channel.device_negative = negative


def acquisition_channels(
    import_type: str, names: Iterable[str], metadata: SourceMetadata
) -> list[InputChannel]:
    channels = []
    for name in names:
        label = metadata.labels.get(name, name)
        row = InputChannel(name, label)
        if import_type in VENDORS:
            row.recording_contact, row.reference_contact = contact_roles(name)
            if row.recording_contact is not None:
                row.recording_contact_source = row.reference_contact_source = "device"
                if import_type == "Sceneray":
                    _apply_sceneray_polarity(row)
                elif import_type == "PINS" and "+" in label and "-" in label:
                    row.device_positive = row.reference_contact
                    row.device_negative = row.recording_contact
        channels.append(row)
    return channels


@dataclass
class RecordProvenance:
    import_type: str | None
    source_file: str | None
    inputs: list[InputChannel] = field(default_factory=list)
    outputs: list[OutputChannel] = field(default_factory=list)
    metadata_file: str | None = None
    marker_file: str | None = None
    source_sampling_rate_hz: float | None = None

    @classmethod
    def from_parsed(
        cls, import_type: str, source: Path, raw: Any, metadata: SourceMetadata
    ) -> RecordProvenance:
        """Capture acquisition identity before synchronization or channel operations."""
        return cls(
            import_type,
            source.name,
            acquisition_channels(import_type, getattr(raw, "ch_names", ()), metadata),
            metadata_file=metadata.metadata_file,
            marker_file=metadata.marker_file,
            source_sampling_rate_hz=metadata.sampling_rate_hz
            or getattr(raw, "info", {}).get("sfreq"),
        )

    def save(self, fif_path: Path) -> None:
        write_json_atomically(fif_path.with_name("record_info.json"), asdict(self))

    @classmethod
    def read(cls, path: Path) -> RecordProvenance:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            inputs = [InputChannel(**row) for row in data["inputs"]]
            outputs = [OutputChannel(**row) for row in data["outputs"]]
            channels: list[InputChannel | OutputChannel] = [*inputs, *outputs]
            for row in channels:
                if not isinstance(row.name, str) or not row.name:
                    raise ValueError("Invalid channel name.")
                if any(
                    v is not None and not isinstance(v, str) for v in vars(row).values()
                ):
                    raise ValueError("Invalid channel metadata.")
            if len({r.name for r in inputs}) != len(inputs) or len(
                {r.name for r in outputs}
            ) != len(outputs):
                raise ValueError("Repeated channel name.")
            for key in ("import_type", "source_file", "metadata_file", "marker_file"):
                if data.get(key) is not None and not isinstance(data[key], str):
                    raise ValueError(f"Invalid {key}.")
            rate = data.get("source_sampling_rate_hz")
            if rate is not None and (
                type(rate) not in (float, int) or not math.isfinite(rate) or rate <= 0
            ):
                raise ValueError("Invalid source sampling rate.")
            if any(
                not isinstance(row.source_label, str) or not row.source_label
                for row in inputs
            ):
                raise ValueError("Invalid original input label.")
            names = {row.name for row in inputs}
            if any(
                key and key not in names
                for row in outputs
                for key in (row.signal_channel, row.reference_channel)
            ):
                raise ValueError("Unknown input channel in an output definition.")
            if data["import_type"] == "Sceneray":
                # Earlier sidecars reversed the polarity encoded in the source label.
                for channel in inputs:
                    _apply_sceneray_polarity(channel)
            return cls(**{**data, "inputs": inputs, "outputs": outputs})
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid recording information in {path.name}: {exc}"
            ) from exc


def imported_outputs(
    input_names: Iterable[str], rows: Iterable[tuple[str, str, str]]
) -> list[OutputChannel]:
    """Use exactly the rows applied to the Raw, including unary operations."""
    configured = [
        OutputChannel(name, signal or None, reference or None, "import")
        for signal, reference, name in rows
    ]
    return configured or [
        OutputChannel(name, signal_channel=name, operation_source="import")
        for name in input_names
    ]


@dataclass(frozen=True)
class ChannelDisplay:
    name: str
    recording_contact: str = UNKNOWN
    reference_contact: str = UNKNOWN
    recorded_signal: str = UNKNOWN


def _contact_expression(row: InputChannel) -> str | None:
    if row.device_positive and row.device_negative:
        return f"{row.device_positive} − {row.device_negative}"
    return None


def _difference_roles(
    signal: InputChannel, reference: InputChannel
) -> tuple[str | None, str | None]:
    if (
        signal.reference_contact
        and signal.reference_contact == reference.reference_contact
        and signal.recording_contact != reference.recording_contact
    ):
        return signal.recording_contact, reference.recording_contact
    if (
        signal.recording_contact
        and signal.recording_contact == reference.recording_contact
        and reference.reference_contact != signal.reference_contact
    ):
        return reference.reference_contact, signal.reference_contact
    return None, None


def channel_tables(
    info: RecordProvenance,
    output_names: Iterable[str],
    *,
    historical_legacy: bool = False,
) -> tuple[list[ChannelDisplay], list[ChannelDisplay]]:
    if historical_legacy:
        return [ChannelDisplay(r.name) for r in info.inputs], [
            ChannelDisplay(n) for n in output_names
        ]
    inputs = {r.name: r for r in info.inputs}
    in_rows = [
        ChannelDisplay(
            r.name,
            r.recording_contact or UNKNOWN,
            r.reference_contact or UNKNOWN,
            _contact_expression(r) or r.source_label,
        )
        for r in info.inputs
    ]
    outputs = {r.name: r for r in info.outputs}
    out_rows = []
    for name in output_names:
        op = outputs.get(name)
        signal = inputs.get(op.signal_channel or "") if op else None
        reference = inputs.get(op.reference_channel or "") if op else None
        recording = ref = None
        expression = UNKNOWN
        if signal and reference:
            recording, ref = _difference_roles(signal, reference)
            signal_expr, ref_expr = _contact_expression(signal), _contact_expression(
                reference
            )
            expression = (
                f"({signal_expr}) − ({ref_expr})"
                if signal_expr and ref_expr
                else f"{signal.name} − {reference.name}"
            )
        elif signal:
            recording, ref = signal.recording_contact, signal.reference_contact
            expression = _contact_expression(signal) or signal.source_label
        elif reference:
            recording, ref = reference.reference_contact, reference.recording_contact
            operand = _contact_expression(reference)
            expression = f"−({operand})" if operand else f"−{reference.name}"
        out_rows.append(
            ChannelDisplay(name, recording or UNKNOWN, ref or UNKNOWN, expression)
        )
    return in_rows, out_rows


@dataclass
class RecordInformation:
    subject: str
    record: str
    provenance: RecordProvenance
    sampling_rate: float | None
    samples: int | None
    input_rows: list[ChannelDisplay]
    output_rows: list[ChannelDisplay]
    pre_sync_rate: float | None
    sync_source: str | None
    has_fif: bool
    notice: str | None = None
