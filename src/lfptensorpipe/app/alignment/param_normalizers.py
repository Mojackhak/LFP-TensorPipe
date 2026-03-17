"""Alignment method parameter normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .method_specs import DEFAULT_DROP_FIELDS


def _normalize_sample_rate(value: Any, *, fallback: float) -> tuple[bool, float, str]:
    try:
        sample_rate = float(value)
    except Exception:  # noqa: BLE001
        return False, float(fallback), "sample_rate must be numeric."
    if not np.isfinite(sample_rate):
        return False, float(fallback), "sample_rate must be finite."
    if sample_rate <= 0.0:
        return False, float(fallback), "sample_rate must be > 0."
    return True, sample_rate, ""


def _normalize_nonnegative_float(
    value: Any,
    *,
    field_name: str,
    fallback: float,
) -> tuple[bool, float, str]:
    try:
        parsed = float(value)
    except Exception:  # noqa: BLE001
        return False, float(fallback), f"{field_name} must be numeric."
    if not np.isfinite(parsed):
        return False, float(fallback), f"{field_name} must be finite."
    if parsed < 0.0:
        return False, float(fallback), f"{field_name} must be >= 0."
    return True, parsed, ""


def _normalize_duration_range(
    value: Any,
    *,
    allow_none: bool,
) -> tuple[bool, list[float | None], str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return (
            False,
            [None, None] if allow_none else [0.0, 1_000_000.0],
            "duration range must have 2 values.",
        )
    out: list[float | None] = []
    for item in value:
        if item is None:
            if allow_none:
                out.append(None)
                continue
            return False, [0.0, 1_000_000.0], "duration range values cannot be null."
        try:
            parsed = float(item)
        except Exception:  # noqa: BLE001
            return False, [0.0, 1_000_000.0], "duration range values must be numbers."
        if parsed < 0.0:
            return False, [0.0, 1_000_000.0], "duration range values must be >= 0."
        out.append(parsed)
    if out[0] is not None and out[1] is not None and float(out[1]) < float(out[0]):
        return False, out, "duration max must be >= duration min."
    return True, out, ""


def _normalize_annotations(
    value: Any,
    *,
    allow_empty: bool = False,
) -> tuple[bool, list[str], str]:
    if not isinstance(value, list):
        return False, [], "annotations must be a list."
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        label = str(item).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    if not out and not allow_empty:
        return False, [], "Select at least one annotation label."
    return True, out, ""


def _normalize_drop_fields(value: Any) -> list[str]:
    if not isinstance(value, list):
        return list(DEFAULT_DROP_FIELDS)
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item).strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out or list(DEFAULT_DROP_FIELDS)


def _normalize_anchors(value: Any) -> tuple[bool, dict[float, str], str]:
    if not isinstance(value, dict):
        return False, {}, "anchors_percent must be a dict {percent: event_label}."
    parsed: list[tuple[float, str]] = []
    for key, raw_label in value.items():
        label = str(raw_label).strip()
        if not label:
            return False, {}, "anchor event labels cannot be empty."
        try:
            percent = float(key)
        except Exception:  # noqa: BLE001
            return False, {}, "anchor percents must be numeric."
        if percent < 0.0 or percent > 100.0:
            return False, {}, "anchor percents must be within [0, 100]."
        parsed.append((percent, label))
    if len(parsed) < 2:
        return False, {}, "At least 2 anchors are required."
    parsed = sorted(parsed, key=lambda item: item[0])
    percents = [item[0] for item in parsed]
    if percents[0] != 0.0 or percents[-1] != 100.0:
        return False, {}, "Anchors must start at 0 and end at 100."
    for idx in range(1, len(percents)):
        if percents[idx] <= percents[idx - 1]:
            return False, {}, "Anchor percents must be strictly increasing."
    return True, {percent: label for percent, label in parsed}, ""


def _normalize_slug(value: str) -> str:
    token = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value.strip().lower()
    )
    token = token.strip("-_")
    token = token.replace("_", "-")
    return token


__all__ = [
    "_normalize_anchors",
    "_normalize_annotations",
    "_normalize_drop_fields",
    "_normalize_duration_range",
    "_normalize_nonnegative_float",
    "_normalize_sample_rate",
    "_normalize_slug",
]
