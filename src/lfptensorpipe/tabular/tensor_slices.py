from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd


def split_tensor4d_to_nested_df(
    tensor: np.ndarray,
    epoch: Sequence[Any],
    channel: Sequence[Any],
    freq: Sequence[Any],
    time: Sequence[Any],
) -> pd.DataFrame:
    """Split a 4D tensor into a long-form DataFrame with nested 2D DataFrames.

    Expected tensor shape: (n_epoch, n_channel, n_freq, n_time)

    Returns
    -------
    pd.DataFrame
        Columns: ["epoch", "channel", "value"]
        - epoch: epoch label
        - channel: channel label
        - value: nested pd.DataFrame (index=freq, columns=time)
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"tensor must be np.ndarray, got {type(tensor)}")
    if tensor.ndim != 4:
        raise ValueError(
            f"tensor must be 4D (epoch, channel, freq, time), got ndim={tensor.ndim}"
        )

    n_e, n_c, n_f, n_t = tensor.shape

    if len(epoch) != n_e:
        raise ValueError(
            f"len(epoch)={len(epoch)} does not match tensor.shape[0]={n_e}"
        )
    if len(channel) != n_c:
        raise ValueError(
            f"len(channel)={len(channel)} does not match tensor.shape[1]={n_c}"
        )
    if len(freq) != n_f:
        raise ValueError(f"len(freq)={len(freq)} does not match tensor.shape[2]={n_f}")
    if len(time) != n_t:
        raise ValueError(f"len(time)={len(time)} does not match tensor.shape[3]={n_t}")

    freq_idx = pd.Index(list(freq), name="freq")
    time_cols = pd.Index(list(time), name="time")

    rows: list[dict[str, Any]] = []
    for ei, e_label in enumerate(epoch):
        for ci, c_label in enumerate(channel):
            mat = tensor[ei, ci, :, :]  # (n_freq, n_time)
            val = pd.DataFrame(mat, index=freq_idx, columns=time_cols)

            rows.append(
                {
                    "epoch": e_label,
                    "channel": c_label,
                    "value": val,
                }
            )

    return pd.DataFrame(rows, columns=["epoch", "channel", "value"])
