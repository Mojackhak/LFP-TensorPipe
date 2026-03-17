"""Source parsing and reference transform helpers for dataset imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lfptensorpipe.io.converter import df2mne


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

    raw = mne.io.read_raw(str(source_path), preload=True, verbose="ERROR")
    return raw, suffix == ".fif"


def parse_record_source(
    *,
    import_type: str,
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str], bool]:
    """Parse one import source by selected import type."""
    normalized = str(import_type).strip()
    if normalized == "Medtronic":
        from lfptensorpipe.io.medtronic import parse as parse_medtronic

        raw, report = parse_medtronic(paths, options)
        return raw, report, False
    if normalized == "PINS":
        from lfptensorpipe.io.pins import parse as parse_pins

        raw, report = parse_pins(paths, options)
        return raw, report, False
    if normalized == "Sceneray":
        from lfptensorpipe.io.sceneray import parse as parse_sceneray

        raw, report = parse_sceneray(paths, options)
        return raw, report, False
    if normalized == "Legacy (MNE supported)":
        from lfptensorpipe.io.mne_supported import parse as parse_mne_supported

        raw, report = parse_mne_supported(paths, options)
        file_path = Path(str(paths.get("file_path", "")))
        return raw, report, _is_fif_like_path(file_path)
    if normalized == "Legacy (CSV)":
        from lfptensorpipe.io.csv import parse as parse_legacy_csv

        raw, report = parse_legacy_csv(paths, options)
        return raw, report, False
    raise ValueError(f"Unsupported import type: {import_type!r}")


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
    out_types: list[str] = []
    out_data: list[np.ndarray] = []

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

        pick_channel = anode or cathode
        pick_type = str(raw.get_channel_types(picks=[pick_channel])[0])
        if anode and cathode:
            data_anode = raw.get_data(picks=[anode])[0]
            data_cathode = raw.get_data(picks=[cathode])[0]
            data_out = data_anode - data_cathode
        elif anode:
            data_out = raw.get_data(picks=[anode])[0].copy()
        else:
            data_out = -raw.get_data(picks=[cathode])[0]

        out_names.append(name)
        out_types.append(pick_type)
        out_data.append(np.asarray(data_out, dtype=float))

    mat = np.vstack(out_data)
    info = mne.create_info(
        ch_names=out_names, sfreq=float(raw.info["sfreq"]), ch_types=out_types
    )
    out = mne.io.RawArray(mat, info, verbose="ERROR")
    out.set_meas_date(raw.info.get("meas_date"))
    out.set_annotations(raw.annotations.copy())
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
