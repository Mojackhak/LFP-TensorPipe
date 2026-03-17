"""Tensor parameter validation helpers."""

from __future__ import annotations

from typing import Any


def validate_bands(bands: list[dict[str, Any]]) -> tuple[bool, str]:
    if not bands:
        return False, "At least one band is required."

    normalized: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    for idx, band in enumerate(bands):
        name = str(band.get("name", "")).strip()
        if not name:
            return False, f"Band row {idx + 1} has empty name."
        if name in seen:
            return False, f"Duplicate band name: {name}"
        seen.add(name)
        try:
            start = float(band.get("start"))
            end = float(band.get("end"))
        except Exception:
            return False, f"Band row {idx + 1} has invalid numeric range."
        if start <= 0.0 or end <= start:
            return False, f"Band row {idx + 1} must satisfy 0 < start < end."
        normalized.append((name, start, end))

    return True, ""


__all__ = ["validate_bands"]
