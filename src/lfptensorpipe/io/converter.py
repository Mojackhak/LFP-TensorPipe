"""converter.py

Sceneray LFP CSV conversion and audio-beep synchronization utilities.

Primary use cases
-----------------
1) Convert Sceneray "raw" LFP CSV (packet/block format) into a flat time-series CSV:
   - adds Time (seconds) and Time_BJS (absolute time)
   - reorganizes channel samples into a continuous sequence
   - expands per-packet tags into per-sample TagCode/TagAccurate

2) Convert the reorganized CSV into an MNE Raw (*.fif) file:
   - channels stored in Volts
   - TagCode converted to MNE Annotations

3) Synchronize LFP with behavioral video/audio using:
   - LFP tags (TagCode events) and audio beeps (peak detection)
   - alignment via pad/cut (no resampling)
   - optional sampling frequency drift correction via regression slope

This module is intentionally conservative:
- it does not resample any data,
- it avoids randomness in tag placement (inaccurate tags are clipped deterministically).
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Metadata parsing
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenerayTxtMetadata:
    """Metadata parsed from the Sceneray *.txt sidecar file."""

    sfreq_hz: float
    start_time_bjs: dt.datetime
    end_time_bjs: dt.datetime
    ipg_sn: str
    ipg_type: str


@dataclass(frozen=True)
class ScenerayCsvLayout:
    """Row layout needed to parse a specific Sceneray CSV format."""

    header_rows_to_skip: tuple[int, ...]
    channel_name_row: int


_CSV_LAYOUTS: dict[str, ScenerayCsvLayout] = {
    "1012P": ScenerayCsvLayout(header_rows_to_skip=tuple(range(6)), channel_name_row=2),
    "1030L": ScenerayCsvLayout(
        header_rows_to_skip=(0, 1, 2, 3, 4, 5, 7), channel_name_row=2
    ),
}


def _infer_ipg_type(ipg_sn: str) -> str:
    m = re.search(r"[A-Za-z]", ipg_sn)
    if not m:
        raise ValueError(f"Cannot infer IPG type from ipg_sn={ipg_sn!r}")
    return ipg_sn[: m.start() + 1]


def read_sceneray_txt_metadata(txt_path: str) -> ScenerayTxtMetadata:
    """Read Sceneray *.txt sidecar file.

    The file is expected to contain lines such as:
    - "IPG SN" followed by the serial number on the next line
    - "Sample Frequency" with the numeric value at the end of the line
    - "Start Time" and "End Time" followed by timestamps on the next line

    Returns
    -------
    ScenerayTxtMetadata
    """
    sample_freq: float | None = None
    start_time: dt.datetime | None = None
    end_time: dt.datetime | None = None
    ipg_sn: str | None = None

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]

    for i, line in enumerate(lines):
        if "IPG SN" in line and i + 1 < len(lines):
            ipg_sn = lines[i + 1].strip()

        if "Sample Frequency" in line:
            try:
                sample_freq = float(line.split()[-1])
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse Sample Frequency from line: {line!r}"
                ) from exc

        if "Start Time" in line and i + 1 < len(lines):
            start_time = dt.datetime.strptime(
                lines[i + 1].strip(), "%Y-%m-%d %H:%M:%S.%f"
            )

        if "End Time" in line and i + 1 < len(lines):
            end_time = dt.datetime.strptime(
                lines[i + 1].strip(), "%Y-%m-%d %H:%M:%S.%f"
            )

    if sample_freq is None:
        raise ValueError(f"Missing Sample Frequency in txt file: {txt_path}")
    if start_time is None or end_time is None:
        raise ValueError(f"Missing Start/End Time in txt file: {txt_path}")
    if ipg_sn is None:
        raise ValueError(f"Missing IPG SN in txt file: {txt_path}")

    ipg_type = _infer_ipg_type(ipg_sn)
    return ScenerayTxtMetadata(
        sfreq_hz=float(sample_freq),
        start_time_bjs=start_time,
        end_time_bjs=end_time,
        ipg_sn=ipg_sn,
        ipg_type=ipg_type,
    )


# Backwards-compatible name
def read_txt_file(file_path: str) -> tuple[float, dt.datetime, dt.datetime, str]:
    meta = read_sceneray_txt_metadata(file_path)
    return meta.sfreq_hz, meta.start_time_bjs, meta.end_time_bjs, meta.ipg_sn


# -----------------------------------------------------------------------------
# CSV reorganization (packet/block -> time series)
# -----------------------------------------------------------------------------


def _default_txt_path_from_csv(csv_path: str) -> str:
    p = Path(csv_path)
    if p.name.endswith("_uv.csv"):
        return str(p.with_name(p.name.replace("_uv.csv", ".txt")))
    return str(p.with_suffix(".txt"))


def _default_reorg_path_from_csv(csv_path: str) -> str:
    p = Path(csv_path)
    return str(p.with_suffix("").as_posix() + "_reorg.csv")


def _get_layout(ipg_type: str) -> ScenerayCsvLayout:
    if ipg_type not in _CSV_LAYOUTS:
        raise ValueError(
            f"Unknown ipg_type={ipg_type!r}. Known types: {sorted(_CSV_LAYOUTS)}"
        )
    return _CSV_LAYOUTS[ipg_type]


def _read_sceneray_csv_data(
    csv_path: str, layout: ScenerayCsvLayout
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path, skiprows=list(layout.header_rows_to_skip))
    df.columns = [str(c).strip() for c in df.columns]

    # Read only header line that contains channel labels.
    header_cols = pd.read_csv(
        csv_path, skiprows=list(range(layout.channel_name_row)), nrows=0
    ).columns
    header_cols = [str(c).strip() for c in header_cols]
    return df, header_cols


def _parse_channel_names(header_cols: list[str]) -> list[str]:
    # In some Sceneray exports, one channel label per channel contains a "~".
    raw_names = [c for c in header_cols if "~" in c]
    names: list[str] = []
    for c in raw_names:
        c2 = c.replace("~", "_")
        if "_" in c2:
            left, right = c2.split("_", 1)
            names.append(f"{right}_{left}")
        else:
            names.append(c2)
    return names


def _infer_n_channels_from_columns(df: pd.DataFrame) -> int:
    nums: list[int] = []
    for c in df.columns:
        m = re.match(r"^CH(\d+)", str(c))
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        raise ValueError("Could not infer channel count: no columns matching '^CH\\d+'")
    return max(nums)


def _normalize_channel_names(names: list[str], n_channels: int) -> list[str]:
    if len(names) == n_channels:
        return names
    if len(names) > n_channels:
        return names[:n_channels]
    # Pad
    padded = names[:]
    for i in range(len(names) + 1, n_channels + 1):
        padded.append(f"CH{i}")
    return padded


def _drop_and_fill_packet_index(df: pd.DataFrame) -> pd.DataFrame:
    packet_col = "Packet Index"
    if packet_col not in df.columns:
        raise ValueError(
            f"Missing required column: {packet_col!r}. Available columns: {list(df.columns)[:20]}..."
        )

    # Ensure integer packet indices
    df[packet_col] = pd.to_numeric(df[packet_col], errors="coerce").astype("Int64")
    if df[packet_col].isna().any():
        raise ValueError(
            "Packet Index contains NaNs after coercion. The CSV may be malformed."
        )

    # Drop all duplicated packet indices (matches legacy behavior)
    dup_mask = df.duplicated(subset=packet_col, keep=False)
    if dup_mask.any():
        logger.warning(
            "Dropping %d rows due to duplicated Packet Index (drop_all).",
            int(dup_mask.sum()),
        )
        df = df.loc[~dup_mask].copy()

    min_idx = int(df[packet_col].min())
    max_idx = int(df[packet_col].max())
    full_idx = np.arange(min_idx, max_idx + 1, dtype=int)

    df2 = df.set_index(packet_col).reindex(full_idx)
    fill_values: dict[str, Any] = {col: 0 for col in df2.columns}

    # Tag Code is sometimes stored as a hex string. Keep it as string during fill.
    if "Tag Code" in df2.columns:
        fill_values["Tag Code"] = "0x00"
    df2 = df2.fillna(fill_values).reset_index().rename(columns={"index": packet_col})

    # Restore packet index dtype
    df2[packet_col] = df2[packet_col].astype(int)
    return df2


def _block_sort_key(col_name: str) -> tuple[int, str]:
    """Return a sorting key for per-channel block columns.

    Sceneray CSV exports often contain repeated column names per channel.
    When pandas reads those, it disambiguates duplicates by appending ".1", ".2", ...

    Examples
    --------
    - "CH2"   -> first block within the packet (block index 0)
    - "CH2.1" -> second block within the packet (block index 1)
    - "CH2.4" -> fifth block within the packet (block index 4)

    IMPORTANT
    ---------
    We must *not* treat the channel number in "CH2" as a block index.
    The legacy implementation used a trailing-digit regex, which incorrectly sorted:
        "CH2.1" (1) before "CH2" (2)
    and therefore swapped the first two samples for CH2..CH8.
    """
    s = str(col_name).strip()

    # Base column (no suffix) is always the first block.
    if re.fullmatch(r"CH\d+", s):
        return (0, s)

    # pandas duplicate columns: ".<int>"
    m = re.fullmatch(r"CH\d+\.(\d+)", s)
    if m:
        return (int(m.group(1)), s)

    # Some exports may use "_<int>" instead of ".<int>"
    m = re.fullmatch(r"CH\d+_(\d+)", s)
    if m:
        return (int(m.group(1)), s)

    # Fallback: try to interpret a final separator + digits as a block index.
    m = re.search(r"[._](\d+)$", s)
    return (int(m.group(1)) if m else 0, s)


def _collect_channel_block_columns(
    df: pd.DataFrame, n_channels: int
) -> list[list[str]]:
    cols_by_ch: list[list[str]] = []
    for ch_i in range(1, n_channels + 1):
        # Avoid matching CH1 to CH10 by requiring the next char NOT to be a digit.
        pattern = re.compile(rf"^CH{ch_i}(?!\d)")
        cols = [c for c in df.columns if pattern.search(str(c))]
        if not cols:
            raise ValueError(f"No columns found for channel CH{ch_i}.")
        cols = sorted(cols, key=_block_sort_key)
        cols_by_ch.append(cols)

    n_blocks = len(cols_by_ch[0])
    for ch_i, cols in enumerate(cols_by_ch, start=1):
        if len(cols) != n_blocks:
            raise ValueError(
                f"Inconsistent block count: CH1 has {n_blocks} blocks, CH{ch_i} has {len(cols)} blocks."
            )
    return cols_by_ch


def _parse_tag_code(value: Any) -> int:
    """Parse Tag Code (hex string or numeric) to int."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip()
    if s == "":
        return 0
    # Sometimes CSV stores tag code like "0x1A" or "26.0"
    s = s.split(".")[0]
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(s)
    except Exception:
        # Last resort: try interpreting as float
        try:
            return int(float(s))
        except Exception:
            logger.warning("Failed to parse Tag Code %r -> treating as 0", value)
            return 0


def _parse_tag_index_ms(value: Any) -> int:
    """Parse Tag Index (ms) to integer."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    try:
        return int(float(value))
    except Exception:
        s = str(value).strip()
        if s == "":
            return 0
        try:
            return int(float(s))
        except Exception:
            logger.warning("Failed to parse Tag Index %r -> treating as 0", value)
            return 0


def uvcsv2vt(
    file_path: str,
    f: float | None = None,
    save_path: str | None = None,
    txt_path: str | None = None,
    ipg_type: str | None = None,
    correct_fs: bool = True,
    correct_idx: bool = True,
) -> str:
    """Convert Sceneray raw LFP CSV to a flat voltage-time CSV.

    Parameters
    ----------
    file_path:
        Path to the Sceneray raw CSV (often ends with *_uv.csv).
    f:
        Optional scaling factor applied to channel values (e.g., convert raw units to microvolts).
        If None, keep numeric values as-is.
    save_path:
        Output CSV path. If None, defaults to "<csv>_reorg.csv".
    txt_path:
        Sidecar metadata txt path. If None, inferred from csv name.
    ipg_type:
        Device type. If None, inferred from the IPG SN in the txt sidecar.
    correct_fs:
        If True, correct sampling frequency using (EndTime - StartTime) from the txt sidecar.
    correct_idx:
        If True, fill missing packet indices and drop duplicated packet indices (legacy behavior).

    Returns
    -------
    save_path:
        Path to the reorganized CSV.
    """
    csv_path = str(file_path)
    if txt_path is None:
        txt_path = _default_txt_path_from_csv(csv_path)

    meta = read_sceneray_txt_metadata(txt_path)
    if ipg_type is None:
        ipg_type = meta.ipg_type

    layout = _get_layout(ipg_type)
    df_raw, header_cols = _read_sceneray_csv_data(csv_path, layout)

    if correct_idx:
        df_raw = _drop_and_fill_packet_index(df_raw)

    channel_names = _parse_channel_names(header_cols)
    n_channels = _infer_n_channels_from_columns(df_raw)
    channel_names = _normalize_channel_names(channel_names, n_channels)

    cols_by_ch = _collect_channel_block_columns(df_raw, n_channels)
    n_packets = len(df_raw)
    n_blocks = len(cols_by_ch[0])

    # Assemble data cube: (packet, block, channel)
    cube = np.empty((n_packets, n_blocks, n_channels), dtype=float)
    for ch0, cols in enumerate(cols_by_ch):
        cube[:, :, ch0] = df_raw[cols].astype(float).to_numpy()

    # Flatten to time series: (time, channel)
    data2d = cube.reshape(n_packets * n_blocks, n_channels)

    df_out = pd.DataFrame(data2d, columns=channel_names)
    if f is not None:
        df_out *= float(f)

    # Determine effective sampling rate (optionally corrected by duration)
    sfreq = float(meta.sfreq_hz)
    if correct_fs:
        duration = (meta.end_time_bjs - meta.start_time_bjs).total_seconds()
        if duration > 0:
            sfreq = df_out.shape[0] / duration
            logger.info(
                "Corrected sfreq from %.6f to %.6f Hz using txt duration.",
                meta.sfreq_hz,
                sfreq,
            )

    time_s = np.arange(df_out.shape[0], dtype=float) / sfreq
    # Match legacy (converter good.py) formatting/rounding by generating python datetimes per-sample.
    time_bjs = [
        meta.start_time_bjs + pd.Timedelta(seconds=float(i) / sfreq)
        for i in range(df_out.shape[0])
    ]

    df_out.insert(0, "Time", time_s)
    df_out.insert(1, "Time_BJS", time_bjs)

    # Expand per-packet tag into per-sample arrays
    packet_index = df_raw["Packet Index"].to_numpy(dtype=int)
    pocket_index_ts = np.repeat(packet_index, n_blocks)

    tag_code_ts = np.zeros(df_out.shape[0], dtype=int)
    tag_acc_ts = np.full(df_out.shape[0], -1, dtype=int)

    has_tag = ("Tag Code" in df_raw.columns) and ("Tag Index" in df_raw.columns)
    if has_tag:
        pocket_ms = int(math.ceil(n_blocks * 1000.0 / sfreq))
        for i in range(n_packets):
            code = _parse_tag_code(df_raw.loc[i, "Tag Code"])
            if code == 0:
                continue
            idx_ms = _parse_tag_index_ms(df_raw.loc[i, "Tag Index"])

            # Legacy semantics: tag index is in milliseconds within the packet.
            # We map it to a block index and clip.
            if 1 <= idx_ms <= pocket_ms:
                acc = 1
            else:
                acc = 0

            # Convert ms -> sample index within packet (1-based -> 0-based), then clip.
            idx_ms_safe = max(1, idx_ms)
            block_1based = int(math.ceil((idx_ms_safe / 1000.0) * sfreq))
            block0 = max(0, min(n_blocks - 1, block_1based - 1))

            t0 = i * n_blocks + block0
            tag_code_ts[t0] = code
            tag_acc_ts[t0] = acc

    df_out["PocketIndex"] = pocket_index_ts.astype(float)
    df_out["TagCode"] = tag_code_ts.astype(float)
    df_out["TagAccurate"] = tag_acc_ts.astype(float)

    if save_path is None:
        save_path = _default_reorg_path_from_csv(csv_path)

    df_out.to_csv(save_path, index=False)
    logger.info("Saved reorganized CSV to %s", save_path)
    return str(save_path)


# -----------------------------------------------------------------------------
# CSV <-> MNE conversion
# -----------------------------------------------------------------------------


def _infer_sfreq_from_time(time_s: np.ndarray) -> float:
    if time_s.size < 2:
        raise ValueError("Time column must contain at least 2 samples.")
    dt_med = float(np.median(np.diff(time_s)))
    if dt_med <= 0:
        raise ValueError("Invalid Time column: non-positive median dt.")
    return 1.0 / dt_med


def _infer_lfp_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"Time", "Time_BJS", "PocketIndex", "TagCode", "TagAccurate"}
    return [c for c in df.columns if c not in exclude]


def csv2mne(csv_path: str, save_path: str | None = None) -> str:
    """Convert a reorganized LFP CSV to an MNE Raw FIF file.

    Assumptions
    -----------
    - Channel values in CSV are in microvolts (uV) and are converted to Volts (V) internally.
    - TagCode events are converted to MNE Annotations (duration=0).

    Returns
    -------
    save_path:
        Path to the saved FIF file.
    """
    import mne

    df = pd.read_csv(csv_path)

    if "Time" not in df.columns:
        raise ValueError("CSV must contain a 'Time' column.")
    sfreq = _infer_sfreq_from_time(df["Time"].to_numpy(dtype=float))

    ch_cols = _infer_lfp_columns(df)
    if not ch_cols:
        raise ValueError("No LFP channel columns found in CSV.")

    data = df[ch_cols].to_numpy(dtype=float).T  # (n_ch, n_times)
    data_v = data * 1e-6  # uV -> V

    info = mne.create_info(
        ch_names=list(ch_cols), sfreq=sfreq, ch_types=["dbs"] * len(ch_cols)
    )
    raw = mne.io.RawArray(data_v, info)

    # Convert TagCode to annotations if present
    if "TagCode" in df.columns:
        tagcodes = df["TagCode"].to_numpy()
        idx = np.flatnonzero(tagcodes != 0)
        if idx.size > 0:
            onsets = idx.astype(float) / sfreq
            durations = np.zeros_like(onsets)
            descriptions = [f"tag:0x{int(tagcodes[i]):X}" for i in idx]
            raw.set_annotations(
                mne.Annotations(
                    onset=onsets, duration=durations, description=descriptions
                )
            )

    if save_path is None:
        p = Path(csv_path)
        save_path = str(p.with_suffix("").as_posix() + "_raw.fif")

    raw.save(save_path, overwrite=True)
    logger.info(
        "Saved MNE Raw to %s (sfreq=%.3f Hz, channels=%d)",
        save_path,
        sfreq,
        len(ch_cols),
    )
    return str(save_path)


def matrix2mne(
    matrix: np.ndarray,
    sfreq: float,
    ch_names: list[str] | None = None,
    ch_types: list[str] | None = None,
) -> Any:
    """Convert a 2D matrix to an MNE Raw object.

    Parameters
    ----------
    matrix:
        2D array with shape (n_channels, n_times). If shape looks like (n_times, n_channels),
        it will be transposed.
    sfreq:
        Sampling frequency in Hz.
    ch_names:
        Channel names.
    ch_types:
        Channel types (default 'dbs').

    Returns
    -------
    raw:
        MNE RawArray object.
    """
    import mne

    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={mat.shape}")

    if mat.shape[0] > mat.shape[1]:
        logger.warning(
            "Matrix shape suggests (n_times, n_channels). Transposing to (n_channels, n_times)."
        )
        mat = mat.T

    n_ch = mat.shape[0]
    if ch_names is None:
        ch_names = [f"CH{i+1}" for i in range(n_ch)]
    if ch_types is None:
        ch_types = ["dbs"] * n_ch

    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types=ch_types)
    return mne.io.RawArray(mat, info)


def _voltage_unit_to_volt_scale(unit: str) -> float:
    unit_l = unit.strip().lower()
    unit_scales = {
        "v": 1.0,
        "volt": 1.0,
        "volts": 1.0,
        "mv": 1e-3,
        "millivolt": 1e-3,
        "millivolts": 1e-3,
        "uv": 1e-6,
        "microvolt": 1e-6,
        "microvolts": 1e-6,
        "μv": 1e-6,
        "µv": 1e-6,
    }
    if unit_l not in unit_scales:
        raise ValueError(
            f"Unsupported voltage unit {unit!r}. "
            f"Supported units: {sorted(unit_scales)}"
        )
    return unit_scales[unit_l]


def df2mne(
    df: pd.DataFrame,
    sr: float,
    ch_types: list[str] | None = None,
    unit: str = "V",
) -> Any:
    """Convert a channel-by-column DataFrame into an MNE Raw object.

    Parameters
    ----------
    df:
        Input DataFrame where each column is one channel signal and each row is one sample.
        Column names are used as MNE channel names.
        Non-numeric columns are ignored automatically.
    sr:
        Sampling rate in Hz.
    ch_types:
        Optional channel types. Defaults to ``'dbs'`` for all channels.
    unit:
        Unit for values stored in ``df``. Supported: V, mV, uV.
        Data are converted to Volts before creating the MNE Raw object.

    Returns
    -------
    raw:
        MNE RawArray object.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df)!r}")
    if df.empty:
        raise ValueError("df must contain at least one channel column and one sample.")
    if float(sr) <= 0:
        raise ValueError(f"sr must be > 0, got {sr}")
    if df.columns.duplicated().any():
        dup = list(df.columns[df.columns.duplicated()])
        raise ValueError(f"df contains duplicated channel names: {dup}")

    numeric_cols: list[str] = []
    ignored_cols: list[str] = []
    for col in df.columns:
        s_raw = df[col]
        s_num = pd.to_numeric(s_raw, errors="coerce")
        if int(s_num.notna().sum()) == int(s_raw.notna().sum()):
            numeric_cols.append(col)
        else:
            ignored_cols.append(col)

    if not numeric_cols:
        raise ValueError(
            "No numeric channel columns found in df. "
            f"Non-numeric columns: {ignored_cols}"
        )
    if ignored_cols:
        logger.warning("Ignoring non-numeric columns in df2mne: %s", ignored_cols)

    ch_names = [str(c).strip() for c in numeric_cols]
    data = (
        df.loc[:, numeric_cols]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
        .T
    )

    data_v = data * _voltage_unit_to_volt_scale(unit)
    import mne

    if ch_types is None:
        ch_types = ["dbs"] * len(ch_names)
    elif len(ch_types) != len(ch_names):
        raise ValueError(
            f"ch_types length ({len(ch_types)}) does not match number of numeric channels ({len(ch_names)})."
        )
    info = mne.create_info(ch_names=ch_names, sfreq=float(sr), ch_types=ch_types)
    return mne.io.RawArray(data_v, info)


def add_annotations_from_df(raw: Any, df_anno: pd.DataFrame) -> Any:
    """Append annotations from a DataFrame to an existing MNE Raw object.

    Parameters
    ----------
    raw:
        MNE Raw-like object to be updated.
    df_anno:
        DataFrame containing annotation rows with columns:
        ``description``, ``onset`` (seconds), ``duration`` (seconds).

    Returns
    -------
    raw:
        The same Raw object with annotations appended.
    """
    import mne

    if not isinstance(df_anno, pd.DataFrame):
        raise TypeError(f"df_anno must be a pandas.DataFrame, got {type(df_anno)!r}")
    required_cols = {"description", "onset", "duration"}
    missing = required_cols - set(df_anno.columns)
    if missing:
        raise ValueError(
            f"df_anno is missing required columns: {sorted(missing)}. "
            f"Required columns: {sorted(required_cols)}"
        )
    if df_anno.empty:
        return raw

    anno = df_anno.loc[:, ["description", "onset", "duration"]].copy()
    anno["description"] = anno["description"].astype(str)
    anno["onset"] = pd.to_numeric(anno["onset"], errors="coerce")
    anno["duration"] = pd.to_numeric(anno["duration"], errors="coerce")
    if anno[["onset", "duration"]].isna().any().any():
        raise ValueError("df_anno columns 'onset' and 'duration' must be numeric.")

    ann_new = mne.Annotations(
        onset=anno["onset"].to_numpy(dtype=float),
        duration=anno["duration"].to_numpy(dtype=float),
        description=anno["description"].to_list(),
    )
    raw.set_annotations(raw.annotations + ann_new)
    return raw
