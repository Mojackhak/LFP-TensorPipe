"""Build channel pairs for connectivity computations.

This file centralizes pair construction so that connectivity code can stay focused
on signal processing rather than bookkeeping.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Dict, List, Sequence, Tuple

import numpy as np


def build_within_group_pairs(
    channel_groups: Dict[str, Sequence[str]],
    *,
    ch_names: Sequence[str],
    ordered: bool = False,
) -> tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Build channel pairs within each group.

    Args:
        channel_groups: Mapping group_name -> list of channel names.
        ch_names: The channel names present in the data.
        ordered: If True, generate ordered pairs (A,B) and (B,A). If False,
            generate unordered combinations (each pair once).

    Returns:
        seeds_idx, targets_idx, pair_names

    Raises:
        ValueError: If no valid pairs could be formed.
    """
    seeds_idx: list[int] = []
    targets_idx: list[int] = []
    pairs: list[Tuple[str, str]] = []

    ch_list = list(ch_names)

    for _group_name, chs in channel_groups.items():
        present = [c for c in chs if c in ch_list]
        iterator = permutations(present, 2) if ordered else combinations(present, 2)
        for a, b in iterator:
            seeds_idx.append(ch_list.index(a))
            targets_idx.append(ch_list.index(b))
            pairs.append((a, b))

    if not pairs:
        raise ValueError("No valid within-group channel pairs found in the data.")

    return np.asarray(seeds_idx, dtype=int), np.asarray(targets_idx, dtype=int), pairs


def build_all_pairs(
    ch_names: Sequence[str],
    *,
    ordered: bool = False,
) -> tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Build all channel pairs from a channel list.

    Args:
        ch_names: Channel names in data order.
        ordered: If True, build permutations (A,B) and (B,A). If False, build
            combinations (each pair once).

    Returns:
        seeds_idx, targets_idx, pair_names
    """
    ch_list = list(ch_names)
    idx_of = {name: i for i, name in enumerate(ch_list)}
    iterator = permutations(ch_list, 2) if ordered else combinations(ch_list, 2)
    pairs = list(iterator)
    if not pairs:
        raise ValueError("No valid pairs could be formed from `ch_names`.")

    seeds_idx = [idx_of[a] for a, _b in pairs]
    targets_idx = [idx_of[b] for _a, b in pairs]
    return np.asarray(seeds_idx, dtype=int), np.asarray(targets_idx, dtype=int), pairs


def build_explicit_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    ch_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Build pair indices from an explicit pair list.

    Args:
        pairs: Explicit list of (seed, target) channel name pairs.
        ch_names: Channel names in data order.

    Returns:
        seeds_idx, targets_idx, pair_names

    Raises:
        ValueError: If any pair references a channel not in `ch_names`.
    """
    ch_list = list(ch_names)
    idx_of = {name: i for i, name in enumerate(ch_list)}
    seeds_idx: list[int] = []
    targets_idx: list[int] = []
    out_pairs: list[Tuple[str, str]] = []

    for a, b in pairs:
        if a not in idx_of:
            raise ValueError(f"Pair channel not found in data: {a!r}")
        if b not in idx_of:
            raise ValueError(f"Pair channel not found in data: {b!r}")
        seeds_idx.append(idx_of[a])
        targets_idx.append(idx_of[b])
        out_pairs.append((a, b))

    if not out_pairs:
        raise ValueError("`pairs` is empty; no connectivity pairs to compute.")

    return np.asarray(seeds_idx, dtype=int), np.asarray(targets_idx, dtype=int), out_pairs
