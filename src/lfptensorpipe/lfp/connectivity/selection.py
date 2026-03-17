"""Pair and channel selection helpers.

This module centralizes selection logic shared by different connectivity-like
computations (e.g., coherence grid, PSI grid).

Keeping this logic in one place avoids subtle inconsistencies in:
  - picked channel ordering
  - explicit pair validation
  - within-group vs all-to-all pair generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .pairs import build_all_pairs, build_explicit_pairs, build_within_group_pairs


def resolve_pairs(
    ch_names: Sequence[str],
    *,
    pairs: Sequence[Tuple[str, str]] | None,
    groups: Dict[str, Sequence[str]] | None,
    ordered_pairs: bool,
) -> tuple[np.ndarray, np.ndarray, List[Tuple[str, str]], Dict[str, Any]]:
    """Resolve connectivity pairs.

    Args:
        ch_names: Channel names in the data order.
        pairs: Optional explicit list of (seed, target) channel name pairs.
        groups: Optional mapping group_name -> list[channel_name]. Pairs are built
            within each group.
        ordered_pairs: If True, build ordered pairs (A,B) and (B,A) where applicable.

    Returns:
        seeds_idx: Indices of seed channels.
        targets_idx: Indices of target channels.
        pair_names: List of (seed_name, target_name) tuples.
        meta: JSON-safe dict describing the selection mode.

    Raises:
        ValueError: If both `pairs` and `groups` are provided.
    """
    if pairs is not None and groups is not None:
        raise ValueError("Provide only one of `pairs` or `groups`, not both.")

    if pairs is not None:
        seeds_idx, targets_idx, pair_names = build_explicit_pairs(pairs, ch_names=ch_names)
        meta = {"pair_mode": "explicit", "pairs": list(pair_names)}
        return seeds_idx, targets_idx, pair_names, meta

    if groups is not None:
        seeds_idx, targets_idx, pair_names = build_within_group_pairs(
            groups, ch_names=ch_names, ordered=bool(ordered_pairs)
        )
        meta = {"pair_mode": "groups", "groups": {k: list(v) for k, v in groups.items()}}
        return seeds_idx, targets_idx, pair_names, meta

    seeds_idx, targets_idx, pair_names = build_all_pairs(ch_names, ordered=bool(ordered_pairs))
    meta = {"pair_mode": "all", "pairs": list(pair_names)}
    return seeds_idx, targets_idx, pair_names, meta
