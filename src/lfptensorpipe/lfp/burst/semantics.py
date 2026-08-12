"""Shared Burst tensor value semantics."""

from __future__ import annotations

from typing import Any, Mapping

BURST_NATIVE_DECIM = 1
BURST_NATIVE_HOP_S = None
BURST_VALUE_SEMANTICS: Mapping[str, Any] = {
    "non_burst_value": 0.0,
    "invalid_value": "nan",
    "burst_value": "hilbert_envelope_amplitude",
    "time_grid": "native_sampling_rate",
}


def burst_value_semantics() -> dict[str, Any]:
    """Return a serialization-safe copy of the current Burst value contract."""
    return dict(BURST_VALUE_SEMANTICS)


def has_current_burst_value_semantics(value: object) -> bool:
    """Return whether a serialized value declares the current Burst contract."""
    return isinstance(value, Mapping) and dict(value) == dict(BURST_VALUE_SEMANTICS)


__all__ = [
    "BURST_NATIVE_DECIM",
    "BURST_NATIVE_HOP_S",
    "BURST_VALUE_SEMANTICS",
    "burst_value_semantics",
    "has_current_burst_value_semantics",
]
