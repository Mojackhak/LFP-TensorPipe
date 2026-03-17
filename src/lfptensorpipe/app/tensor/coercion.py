"""Tensor parameter coercion and log-sanitization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return parsed


def _as_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return int(parsed)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _as_optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except Exception:
        return default
    if not np.isfinite(parsed):
        return default
    return float(parsed)


def _as_optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _sanitize_metric_params_for_logs(params: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
            continue
        if isinstance(value, (list, tuple)):
            sanitized[key] = list(value)
            continue
        if isinstance(value, dict):
            sanitized[key] = dict(value)
            continue
    return sanitized


def _normalize_metric_method(method: Any, *, metric_label: str) -> str:
    token = str(method).strip().lower() if method is not None else ""
    if not token:
        return "morlet"
    if token in {"morlet", "multitaper"}:
        return token
    raise ValueError(
        f"{metric_label} method must be 'morlet' or 'multitaper', got: {method!r}"
    )


__all__ = [
    "_as_bool",
    "_as_float",
    "_as_int",
    "_as_optional_float",
    "_as_optional_int",
    "_normalize_metric_method",
    "_sanitize_metric_params_for_logs",
]
