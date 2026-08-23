"""Strict JSON boundary for reusable Burst threshold payloads."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lfptensorpipe.lfp.burst.semantics import (
    burst_value_semantics,
    has_current_burst_value_semantics,
    normalize_burst_estimator_signature,
)

BURST_THRESHOLD_KIND = "lfptensorpipe_burst_thresholds"

_PAYLOAD_KEYS = frozenset(
    {"kind", "value_semantics", "estimator", "channels", "bands", "values"}
)
_BAND_KEYS = frozenset({"name", "segments_hz"})


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}.")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate JSON object key: {key!r}.")
        out[key] = value
    return out


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    field: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if extra:
        details.append("unexpected " + ", ".join(extra))
    raise ValueError(f"{field} has invalid keys ({'; '.join(details)}).")


def _normalize_label(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace.")
    return value


def _normalize_number(value: Any, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number.")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite JSON number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite.")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{field} must be greater than or equal to 0.")
    return parsed


def _normalize_segments(value: Any, *, field: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list.")
    segments: list[tuple[float, float]] = []
    for index, segment in enumerate(value):
        segment_field = f"{field}[{index}]"
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError(f"{segment_field} must contain exactly two numbers.")
        low = _normalize_number(segment[0], field=f"{segment_field}[0]")
        high = _normalize_number(segment[1], field=f"{segment_field}[1]")
        if low >= high:
            raise ValueError(f"{segment_field} must satisfy low < high.")
        segments.append((low, high))

    segments.sort(key=lambda item: (item[0], item[1]))
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]
        if current[0] < previous[1]:
            raise ValueError(
                f"{field} contains duplicate or overlapping segments: "
                f"{previous!r} and {current!r}."
            )
    return [[float(low), float(high)] for low, high in segments]


def normalize_burst_threshold_payload(value: Any) -> dict[str, Any]:
    """Validate and return the canonical JSON-safe Burst threshold payload."""
    if not isinstance(value, dict):
        raise ValueError("Burst threshold payload must be a JSON object.")
    _require_exact_keys(
        value,
        expected=_PAYLOAD_KEYS,
        field="Burst threshold payload",
    )
    raw_semantics = value.get("value_semantics")
    if not has_current_burst_value_semantics(raw_semantics):
        raise ValueError(
            "Burst threshold value_semantics must declare threshold-normalized magnitude."
        )
    value_semantics = burst_value_semantics()
    estimator = normalize_burst_estimator_signature(value.get("estimator"))
    if value.get("kind") != BURST_THRESHOLD_KIND:
        raise ValueError(
            f"Burst threshold payload kind must be {BURST_THRESHOLD_KIND!r}."
        )

    channels_value = value.get("channels")
    if not isinstance(channels_value, list) or not channels_value:
        raise ValueError("channels must be a non-empty list.")
    channels = [
        _normalize_label(item, field=f"channels[{index}]")
        for index, item in enumerate(channels_value)
    ]
    if len(set(channels)) != len(channels):
        raise ValueError("channels must contain unique case-sensitive names.")

    bands_value = value.get("bands")
    if not isinstance(bands_value, list) or not bands_value:
        raise ValueError("bands must be a non-empty list.")
    bands: list[dict[str, Any]] = []
    band_names: list[str] = []
    for index, band_value in enumerate(bands_value):
        field = f"bands[{index}]"
        if not isinstance(band_value, dict):
            raise ValueError(f"{field} must be an object.")
        _require_exact_keys(band_value, expected=_BAND_KEYS, field=field)
        name = _normalize_label(band_value.get("name"), field=f"{field}.name")
        segments = _normalize_segments(
            band_value.get("segments_hz"),
            field=f"{field}.segments_hz",
        )
        bands.append({"name": name, "segments_hz": segments})
        band_names.append(name)
    if len(set(band_names)) != len(band_names):
        raise ValueError("bands must contain unique case-sensitive names.")

    values_value = value.get("values")
    if not isinstance(values_value, list) or len(values_value) != len(bands):
        raise ValueError(
            f"values must contain one row per band ({len(bands)} expected)."
        )
    values: list[list[float]] = []
    for band_index, row in enumerate(values_value):
        if not isinstance(row, list) or len(row) != len(channels):
            raise ValueError(
                f"values[{band_index}] must contain one value per channel "
                f"({len(channels)} expected)."
            )
        values.append(
            [
                _normalize_number(
                    item,
                    field=f"values[{band_index}][{channel_index}]",
                    nonnegative=False,
                )
                for channel_index, item in enumerate(row)
            ]
        )
        if any(item <= 0.0 for item in values[-1]):
            raise ValueError(
                f"values[{band_index}] must contain strictly positive thresholds."
            )

    return {
        "kind": BURST_THRESHOLD_KIND,
        "value_semantics": value_semantics,
        "estimator": estimator,
        "channels": channels,
        "bands": bands,
        "values": values,
    }


def load_burst_threshold_json(path: str | Path) -> dict[str, Any]:
    """Load one strict, non-executable Burst threshold JSON file."""
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("Burst threshold files must use the .json extension.")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    return normalize_burst_threshold_payload(payload)


def _runtime_segments(value: Any, *, field: str) -> list[list[float]]:
    if isinstance(value, tuple) and len(value) == 2:
        raw_segments = [[value[0], value[1]]]
    elif (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in value
        )
    ):
        raw_segments = [[value[0], value[1]]]
    else:
        try:
            raw_segments = [[segment[0], segment[1]] for segment in value]
        except (TypeError, IndexError) as exc:
            raise ValueError(f"{field} has invalid runtime segments.") from exc
    return _normalize_segments(raw_segments, field=field)


def build_burst_threshold_payload(
    *,
    channels: Sequence[str],
    bands: Mapping[str, Any],
    values: Any,
    estimator: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a canonical threshold payload from a completed Burst run."""
    band_rows = [
        {
            "name": str(name),
            "segments_hz": _runtime_segments(
                segments,
                field=f"bands[{str(name)!r}].segments_hz",
            ),
        }
        for name, segments in bands.items()
    ]
    array = np.asarray(values, dtype=float)
    payload = {
        "kind": BURST_THRESHOLD_KIND,
        "value_semantics": burst_value_semantics(),
        "estimator": normalize_burst_estimator_signature(estimator),
        "channels": [str(item) for item in channels],
        "bands": band_rows,
        "values": array.tolist(),
    }
    return normalize_burst_threshold_payload(payload)


def select_burst_threshold_subset(
    payload: Any,
    *,
    channels: Sequence[str],
    bands: Mapping[str, Any],
    estimator: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select and reorder a compatible payload to the requested runtime subset."""
    normalized = normalize_burst_threshold_payload(payload)
    expected_estimator = normalize_burst_estimator_signature(estimator)
    requested_channels = [str(item) for item in channels]
    requested_bands = [str(item) for item in bands]

    payload_channel_index = {
        name: index for index, name in enumerate(normalized["channels"])
    }
    payload_band_index = {
        band["name"]: index for index, band in enumerate(normalized["bands"])
    }
    payload_segments = {
        band["name"]: band["segments_hz"] for band in normalized["bands"]
    }

    missing_channels = [
        name for name in requested_channels if name not in payload_channel_index
    ]
    missing_bands = [name for name in requested_bands if name not in payload_band_index]
    segment_mismatches: list[str] = []
    runtime_segments: dict[str, list[list[float]]] = {}
    for name, segments in bands.items():
        normalized_segments = _runtime_segments(
            segments,
            field=f"runtime band {str(name)!r}",
        )
        runtime_segments[str(name)] = normalized_segments
        if (
            str(name) in payload_segments
            and payload_segments[str(name)] != normalized_segments
        ):
            segment_mismatches.append(str(name))

    problems: list[str] = []
    if missing_channels:
        problems.append("missing channels: " + ", ".join(missing_channels))
    if missing_bands:
        problems.append("missing bands: " + ", ".join(missing_bands))
    if segment_mismatches:
        problems.append("segment mismatches: " + ", ".join(segment_mismatches))
    if normalized["estimator"] != expected_estimator:
        problems.append("estimator mismatch")
    if problems:
        raise ValueError("Incompatible Burst thresholds (" + "; ".join(problems) + ").")

    values = np.asarray(normalized["values"], dtype=np.float64)
    band_indices = [payload_band_index[name] for name in requested_bands]
    channel_indices = [payload_channel_index[name] for name in requested_channels]
    selected = values[np.ix_(band_indices, channel_indices)]
    subset = build_burst_threshold_payload(
        channels=requested_channels,
        bands={name: runtime_segments[name] for name in requested_bands},
        values=selected,
        estimator=expected_estimator,
    )
    return selected, subset


def write_burst_threshold_json(payload: Any, path: str | Path) -> None:
    """Write one canonical Burst threshold payload as UTF-8 JSON."""
    normalized = normalize_burst_threshold_payload(payload)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


__all__ = [
    "BURST_THRESHOLD_KIND",
    "build_burst_threshold_payload",
    "load_burst_threshold_json",
    "normalize_burst_threshold_payload",
    "select_burst_threshold_subset",
    "write_burst_threshold_json",
]
