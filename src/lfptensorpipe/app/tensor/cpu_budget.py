"""CPU percentage validation and Build Tensor slot derivation."""

from __future__ import annotations

import math
import os
from typing import Any

DEFAULT_TENSOR_CPU_PERCENT = 75.0


def normalize_tensor_cpu_percent(value: Any) -> float:
    """Validate and normalize the public Build Tensor CPU percentage."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CPU (%) must be a number between 0 and 100.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 100.0:
        raise ValueError("CPU (%) must be finite and between 0 and 100.")
    return parsed


def derive_global_compute_slots(
    cpu_percent: Any,
    *,
    detected_cpu_count: int | None = None,
) -> tuple[float, int, int]:
    """Return normalized percentage, detected CPUs, and the slot cap."""
    normalized_percent = normalize_tensor_cpu_percent(cpu_percent)
    cpu_count = max(
        (
            int(os.cpu_count() or 1)
            if detected_cpu_count is None
            else int(detected_cpu_count)
        ),
        1,
    )
    slots = max(math.floor(cpu_count * normalized_percent / 100.0), 1)
    return normalized_percent, cpu_count, slots


__all__ = [
    "DEFAULT_TENSOR_CPU_PERCENT",
    "derive_global_compute_slots",
    "normalize_tensor_cpu_percent",
]
