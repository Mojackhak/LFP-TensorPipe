"""Axis normalization helpers for feature derivation."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _normalize_axis_rows(
    value: Any,
    *,
    allow_duplicate_names: bool = False,
) -> list[dict[str, float | str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if not allow_duplicate_names and name in seen:
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except Exception:
            continue
        if end <= start:
            continue
        if not allow_duplicate_names:
            seen.add(name)
        out.append({"name": name, "start": start, "end": end})
    out.sort(key=lambda row: float(row["start"]))
    return out


def _rows_to_interval_mapping(
    rows: list[dict[str, float | str]],
) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        start = float(row.get("start", 0.0))
        end = float(row.get("end", 0.0))
        if end <= start:
            continue
        out.setdefault(name, []).append([start, end])
    return out


def _bands_from_raw_value_index(payload: pd.DataFrame) -> dict[str, list[list[str]]]:
    if "Value" not in payload.columns:
        return {}
    source: Any = None
    for item in payload["Value"].tolist():
        if isinstance(item, (pd.Series, pd.DataFrame)):
            source = item
            break
    if source is None:
        return {}
    index_values = source.index.unique().tolist()
    out: dict[str, list[list[str]]] = {}
    for value in index_values:
        name = str(value).strip()
        if not name or name in out:
            continue
        out[name] = [[name]]
    return out
