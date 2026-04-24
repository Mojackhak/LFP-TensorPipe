"""General connectivity tensor helpers."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np


def swap_reciprocal_pairs_on_channel_axis(
    tensor: np.ndarray,
    pair_names: Sequence[Tuple[str, str]],
    *,
    strict: bool = True,
    return_perm: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Swap channel slices so that each (a,b) is exchanged with its (b,a).

    This is useful when you store directed connectivity in a channel-like axis.

    Args:
        tensor: Array of shape (n_epochs, n_channels, n_freqs, n_times).
        pair_names: pair_names[c] gives the ordered pair label for channel c, e.g. ('0_1','2_3').
        strict: If True, raise if any channel has no reciprocal.
        return_perm: If True, also return the permutation applied.

    Returns:
        tensor_out (and perm if requested).
    """
    if tensor.ndim != 4:
        raise ValueError(
            f"tensor must be 4D (epochs, channels, freqs, times); got shape {tensor.shape}"
        )

    n_channels = int(tensor.shape[1])
    if len(pair_names) != n_channels:
        raise ValueError(
            f"len(pair_names) ({len(pair_names)}) != n_channels ({n_channels})"
        )

    idx_of = {tuple(p): i for i, p in enumerate(pair_names)}

    perm = np.arange(n_channels)
    visited = np.zeros(n_channels, dtype=bool)

    for i, p in enumerate(pair_names):
        if visited[i]:
            continue
        a, b = p
        recip = (b, a)
        j = idx_of.get(recip, None)

        if j is None:
            if strict:
                raise KeyError(f"No reciprocal found for channel {i} labeled {p}")
            visited[i] = True
            continue

        if i == j:
            visited[i] = True
            continue

        perm[i], perm[j] = perm[j], perm[i]
        visited[i] = True
        visited[j] = True

    tensor_out = np.take(tensor, perm, axis=1)
    if return_perm:
        return tensor_out, perm
    return tensor_out
