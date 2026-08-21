"""DataFrame-to-MNE conversion used by current App source parsers."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InfiniteSignalValuesError(ValueError):
    """Raised when signal data contains positive or negative infinity."""


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
        "nv": 1e-9,
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
        Numeric column names are converted with ``str(...).strip()`` and used
        as MNE channel names; the resulting names must be non-empty and unique.
        Non-numeric columns are ignored automatically.
    sr:
        Sampling rate in Hz.
    ch_types:
        Optional channel types. Defaults to ``'dbs'`` for all channels.
    unit:
        Unit for values stored in ``df``. Supported: V, mV, uV, nV.
        Data are converted to Volts before creating the MNE Raw object.
        NaN values are preserved; positive or negative infinity raises
        ``InfiniteSignalValuesError``.

    Returns
    -------
    raw:
        MNE RawArray object.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas.DataFrame, got {type(df)!r}")
    if df.empty:
        raise ValueError("df must contain at least one channel column and one sample.")
    sr_value = float(sr)
    if not math.isfinite(sr_value) or sr_value <= 0:
        raise ValueError(f"sr must be finite and > 0, got {sr_value}")
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
    if any(not name for name in ch_names):
        raise ValueError("df channel names must be non-empty after trimming.")
    name_index = pd.Index(ch_names)
    duplicate_mask = name_index.duplicated()
    if duplicate_mask.any():
        dup = list(name_index[duplicate_mask])
        raise ValueError(f"df contains duplicated channel names: {dup}")
    data = (
        df.loc[:, numeric_cols]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
        .T
    )
    if np.isinf(data).any():
        raise InfiniteSignalValuesError("Signal data must not contain infinite values.")

    data_v = data * _voltage_unit_to_volt_scale(unit)
    import mne

    if ch_types is None:
        ch_types = ["dbs"] * len(ch_names)
    elif len(ch_types) != len(ch_names):
        raise ValueError(
            f"ch_types length ({len(ch_types)}) does not match number of numeric channels ({len(ch_names)})."
        )
    info = mne.create_info(ch_names=ch_names, sfreq=sr_value, ch_types=ch_types)
    return mne.io.RawArray(data_v, info)
