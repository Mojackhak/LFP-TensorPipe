"""Shared helpers for channel-pair and region-pair identifiers."""

from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np


def _canonical_sort_key(value: str) -> tuple[str, str]:
    text = str(value).strip()
    return (text.casefold(), text)


def normalize_ordered_pair(a: str, b: str) -> tuple[str, str]:
    return str(a).strip(), str(b).strip()


def normalize_undirected_pair(a: str, b: str) -> tuple[str, str]:
    left, right = normalize_ordered_pair(a, b)
    return tuple(sorted((left, right), key=_canonical_sort_key))


def normalize_region_pair_name(a: str, b: str) -> tuple[str, str]:
    left = str(a).strip()
    right = str(b).strip()
    return tuple(sorted((left, right), key=_canonical_sort_key))


def make_ordered_pair_key(a: str, b: str) -> str:
    return json.dumps(list(normalize_ordered_pair(a, b)), ensure_ascii=True)


def make_undirected_pair_key(a: str, b: str) -> str:
    return json.dumps(list(normalize_undirected_pair(a, b)), ensure_ascii=True)


def parse_pair_token(value: Any) -> tuple[str, str] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size != 2:
            return None
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            return None
        items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if not (
            (text.startswith("(") and text.endswith(")"))
            or (text.startswith("[") and text.endswith("]"))
        ):
            return None
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return None
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
            return None
        items = list(parsed)
    else:
        return None

    first = str(items[0]).strip()
    second = str(items[1]).strip()
    if not first or not second:
        return None
    return first, second


__all__ = [
    "make_ordered_pair_key",
    "make_undirected_pair_key",
    "normalize_ordered_pair",
    "normalize_region_pair_name",
    "normalize_undirected_pair",
    "parse_pair_token",
]
