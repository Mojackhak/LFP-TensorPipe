"""Shared Burst tensor value semantics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

BURST_NATIVE_DECIM = 1
BURST_NATIVE_HOP_S = None
BURST_NOTCH_SPLIT_COMBINATION = "analytic_sum_before_magnitude"
BURST_VALUE_SEMANTICS: Mapping[str, Any] = {
    "non_burst_value": 0.0,
    "invalid_value": "nan",
    "burst_value": "hilbert_envelope_amplitude",
    "notch_split_combination": BURST_NOTCH_SPLIT_COMBINATION,
    "time_grid": "native_sampling_rate",
}


def burst_value_semantics() -> dict[str, Any]:
    """Return a serialization-safe copy of the current Burst value contract."""
    return dict(BURST_VALUE_SEMANTICS)


def has_current_burst_value_semantics(value: object) -> bool:
    """Return whether a serialized value declares the current Burst contract."""
    return isinstance(value, Mapping) and dict(value) == dict(BURST_VALUE_SEMANTICS)


def burst_bands_use_notch_split(value: object) -> bool | None:
    """Return whether normalized band segments contain a split band."""
    if not isinstance(value, Mapping) or not value:
        return None
    split = False
    for segments in value.values():
        if (
            isinstance(segments, Sequence)
            and not isinstance(segments, (str, bytes))
            and len(segments) == 2
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in segments
            )
        ):
            continue
        if (
            not isinstance(segments, Sequence)
            or isinstance(segments, (str, bytes))
            or not segments
        ):
            return None
        for segment in segments:
            if (
                not isinstance(segment, Sequence)
                or isinstance(segment, (str, bytes))
                or len(segment) != 2
                or not all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in segment
                )
            ):
                return None
        split = split or len(segments) > 1
    return split


def has_compatible_burst_value_semantics(
    value: object,
    *,
    bands_segments_hz: object,
) -> bool:
    """Accept current semantics or a numerically unchanged legacy single band."""
    if has_current_burst_value_semantics(value):
        return True
    if not isinstance(value, Mapping):
        return False
    legacy_semantics = dict(BURST_VALUE_SEMANTICS)
    legacy_semantics.pop("notch_split_combination")
    if dict(value) != legacy_semantics:
        return False
    return burst_bands_use_notch_split(bands_segments_hz) is False


__all__ = [
    "BURST_NATIVE_DECIM",
    "BURST_NATIVE_HOP_S",
    "BURST_NOTCH_SPLIT_COMBINATION",
    "BURST_VALUE_SEMANTICS",
    "burst_bands_use_notch_split",
    "burst_value_semantics",
    "has_compatible_burst_value_semantics",
    "has_current_burst_value_semantics",
]
