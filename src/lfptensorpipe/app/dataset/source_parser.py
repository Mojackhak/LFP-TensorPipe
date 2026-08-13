"""Source parsing and reference transform helpers for dataset imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lfptensorpipe.io.converter import df2mne
from lfptensorpipe.io.timeline import (
    format_timeline_normalization,
    normalize_raw_timeline,
    raw_relative_onsets,
)


def _is_fif_like_path(path: Path) -> bool:
    name = str(path.name).lower()
    return name.endswith(".fif") or name.endswith(".fif.gz")


def _validate_bipolar_pairs(
    raw: Any,
    bipolar_pairs: tuple[tuple[str, str], ...],
    bipolar_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    names = set(str(name) for name in raw.ch_names)
    pair_names: list[str] = []
    resolved_names: list[str] = []
    if bipolar_names is not None and len(bipolar_names) != len(bipolar_pairs):
        raise ValueError("Bipolar channel names must match number of bipolar pairs.")
    for anode, cathode in bipolar_pairs:
        if anode == cathode:
            raise ValueError(f"Bipolar pair has identical channels: {anode}")
        if anode not in names:
            raise ValueError(f"Bipolar anode channel not found: {anode}")
        if cathode not in names:
            raise ValueError(f"Bipolar cathode channel not found: {cathode}")
        pair_names.append(f"{anode}-{cathode}")
    for idx, pair in enumerate(bipolar_pairs):
        if bipolar_names is None:
            resolved_names.append(f"{pair[0]}-{pair[1]}")
            continue
        channel_name = str(bipolar_names[idx]).strip()
        if not channel_name:
            raise ValueError(f"Empty bipolar channel name at row {idx + 1}.")
        resolved_names.append(channel_name)
    if len(pair_names) != len(set(pair_names)):
        raise ValueError("Duplicate bipolar pairs are not allowed.")
    if len(resolved_names) != len(set(resolved_names)):
        raise ValueError("Duplicate bipolar channel names are not allowed.")
    return tuple(resolved_names)


def _apply_bipolar_reference(
    raw: Any,
    bipolar_pairs: tuple[tuple[str, str], ...],
    bipolar_names: tuple[str, ...] | None = None,
    *,
    set_bipolar_reference_fn: Any | None = None,
) -> Any:
    if not bipolar_pairs:
        return raw
    names = _validate_bipolar_pairs(raw, bipolar_pairs, bipolar_names)
    if set_bipolar_reference_fn is None:
        import mne

        set_bipolar_reference_fn = mne.set_bipolar_reference

    anodes = [pair[0] for pair in bipolar_pairs]
    cathodes = [pair[1] for pair in bipolar_pairs]
    bipolar_raw = set_bipolar_reference_fn(
        raw,
        anode=anodes,
        cathode=cathodes,
        ch_name=list(names),
        drop_refs=True,
        copy=True,
    )
    available = set(str(channel) for channel in bipolar_raw.ch_names)
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Missing bipolar channels after conversion: {missing}")
    bipolar_raw.pick_channels(list(names), ordered=True)
    return bipolar_raw


def _load_raw_from_source(
    source_path: Path,
    *,
    csv_sr: float | None,
    csv_unit: str,
) -> tuple[Any, bool]:
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        if csv_sr is None or float(csv_sr) <= 0:
            raise ValueError("CSV import requires sr > 0.")
        df = pd.read_csv(source_path)
        return df2mne(df, sr=float(csv_sr), unit=csv_unit), False

    import mne

    # Deliberately not normalized here: load_import_channel_names() only needs
    # ch_names, and the import runner normalizes (and reports) before saving.
    raw = mne.io.read_raw(str(source_path), preload=True, verbose="ERROR")
    return raw, suffix == ".fif"


def parse_record_source(
    *,
    import_type: str,
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str], bool]:
    """Parse one import source by selected import type.

    Every branch funnels through `normalize_raw_timeline`, so downstream steps
    (sync, reset-reference, import) always see `first_samp == 0` without any
    change to the absolute time of samples or annotations.
    """
    normalized = str(import_type).strip()
    if normalized == "Medtronic":
        from lfptensorpipe.io.medtronic import parse as parse_medtronic

        raw, report = parse_medtronic(paths, options)
        is_fif_input = False
    elif normalized == "PINS":
        from lfptensorpipe.io.pins import parse as parse_pins

        raw, report = parse_pins(paths, options)
        is_fif_input = False
    elif normalized == "Sceneray":
        from lfptensorpipe.io.sceneray import parse as parse_sceneray

        raw, report = parse_sceneray(paths, options)
        is_fif_input = False
    elif normalized == "Legacy (MNE supported)":
        from lfptensorpipe.io.mne_supported import parse as parse_mne_supported

        raw, report = parse_mne_supported(paths, options)
        is_fif_input = _is_fif_like_path(Path(str(paths.get("file_path", ""))))
    elif normalized == "Legacy (CSV)":
        from lfptensorpipe.io.csv import parse as parse_legacy_csv

        raw, report = parse_legacy_csv(paths, options)
        is_fif_input = False
    else:
        raise ValueError(f"Unsupported import type: {import_type!r}")

    raw, timeline_report = normalize_raw_timeline(raw)
    summary = format_timeline_normalization(timeline_report)
    if summary and isinstance(report, dict):
        report["timeline"] = summary
    return raw, report, is_fif_input


def apply_reset_reference(
    raw: Any,
    reset_rows: tuple[tuple[str, str, str], ...],
) -> Any:
    """Apply reset-reference rows to one raw."""
    if not reset_rows:
        return raw

    import mne
    import numpy as np

    available = set(str(ch) for ch in raw.ch_names)
    seen_pairs: set[tuple[str, str]] = set()
    seen_names: set[str] = set()

    out_names: list[str] = []
    out_data: list[np.ndarray] = []
    row_raws: list[Any] = []
    bipolar_row_indices: list[int] = []
    source_bads = set(str(channel) for channel in raw.info.get("bads", []))
    out_bads: list[str] = []
    metadata_template = mne.io.RawArray(
        np.zeros((len(raw.ch_names), 1), dtype=float),
        raw.info.copy(),
        verbose="ERROR",
    )

    for idx, row in enumerate(reset_rows, start=1):
        if len(row) != 3:
            raise ValueError(f"Invalid reset reference row at {idx}: {row!r}")
        anode = str(row[0]).strip()
        cathode = str(row[1]).strip()
        name = str(row[2]).strip()

        if not anode and not cathode:
            raise ValueError("At least one of anode or cathode is required.")
        if anode and anode not in available:
            raise ValueError(f"Anode channel not found: {anode}")
        if cathode and cathode not in available:
            raise ValueError(f"Cathode channel not found: {cathode}")
        if not name:
            raise ValueError("Name is required.")
        if name in seen_names:
            raise ValueError(f"Duplicate output channel name: {name}")
        if anode and cathode and cathode == anode:
            raise ValueError("Anode and cathode cannot be identical.")

        key = (anode, cathode)
        if key in seen_pairs:
            raise ValueError("Duplicate pair is not allowed.")
        seen_pairs.add(key)
        seen_names.add(name)

        if anode and cathode:
            data_anode = raw.get_data(picks=[anode])[0]
            data_cathode = raw.get_data(picks=[cathode])[0]
            data_out = data_anode - data_cathode
            internal_name = name
            if internal_name in available:
                internal_name = f"__lfptensorpipe_reset_reference_{idx}__"
                while internal_name in available or internal_name in seen_names:
                    internal_name = f"_{internal_name}"
            row_raw = mne.set_bipolar_reference(
                metadata_template,
                anode=anode,
                cathode=cathode,
                ch_name=internal_name,
                drop_refs=False,
                copy=True,
                on_bad="ignore",
                verbose="ERROR",
            )
            row_raw.pick([internal_name])
            if internal_name != name:
                row_raw.rename_channels({internal_name: name})
            bipolar_row_indices.append(len(row_raws))
        elif anode:
            data_out = raw.get_data(picks=[anode])[0].copy()
            row_raw = metadata_template.copy().pick([anode])
            if anode != name:
                row_raw.rename_channels({anode: name})
        else:
            data_out = -raw.get_data(picks=[cathode])[0]
            row_raw = metadata_template.copy().pick([cathode])
            if cathode != name:
                row_raw.rename_channels({cathode: name})

        out_names.append(name)
        out_data.append(np.asarray(data_out, dtype=float))
        row_raws.append(row_raw)
        if anode in source_bads or cathode in source_bads:
            out_bads.append(name)

    mat = np.vstack(out_data)
    first_row_index = bipolar_row_indices[0] if bipolar_row_indices else 0
    merged = row_raws[first_row_index].copy()
    remaining_rows = [
        row_raw
        for row_index, row_raw in enumerate(row_raws)
        if row_index != first_row_index
    ]
    if remaining_rows:
        merged.add_channels(
            remaining_rows,
            force_update_info=bool(bipolar_row_indices),
        )
    merged.pick(out_names)
    info = merged.info.copy()
    info["bads"] = out_bads
    out = mne.io.RawArray(mat, info, verbose="ERROR")
    out.set_meas_date(raw.info.get("meas_date"))
    # `out` restarts at first_samp == 0, so onsets must be pulled back into the
    # record-relative frame or set_annotations() would crop them away.
    annotations = raw.annotations
    out.set_annotations(
        mne.Annotations(
            onset=raw_relative_onsets(raw),
            duration=np.asarray(annotations.duration, dtype=float),
            description=np.asarray(annotations.description, dtype=object),
            orig_time=out.info.get("meas_date"),
        )
    )
    return out


def load_import_channel_names(
    source_path: Path,
    *,
    csv_sr: float | None = None,
    csv_unit: str = "V",
) -> list[str]:
    """Read channel names for one import source."""
    raw, _ = _load_raw_from_source(source_path, csv_sr=csv_sr, csv_unit=csv_unit)
    return [str(name) for name in raw.ch_names]
