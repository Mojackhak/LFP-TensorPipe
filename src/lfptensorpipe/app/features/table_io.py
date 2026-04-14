"""Table I/O and discovery helpers for features stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.io.pkl_io import load_pkl

from .indicator import _normalize_slug


def _flatten_value_for_xlsx(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return value.to_json(orient="split")
    if isinstance(value, pd.Series):
        return value.to_json(orient="split")
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _save_table_xlsx(df: pd.DataFrame, out_path: Path) -> tuple[bool, str]:
    """Save one table as `.xlsx` single sheet."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xlsx_df = df.copy()
    if "Value" in xlsx_df.columns:
        xlsx_df["Value"] = xlsx_df["Value"].map(_flatten_value_for_xlsx)
    if "Value_grid" in xlsx_df.columns:
        xlsx_df["Value_grid"] = xlsx_df["Value_grid"].map(_flatten_value_for_xlsx)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass
    try:
        xlsx_df.to_excel(tmp_path, index=False)
        with zipfile.ZipFile(tmp_path, mode="r") as archive:
            names = set(archive.namelist())
        if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
            raise ValueError("Generated xlsx payload is invalid.")
        tmp_path.replace(out_path)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        return False, str(exc)


def _detect_derived_type(df: pd.DataFrame) -> str:
    value_col = "Value"
    if value_col not in df.columns:
        if "Value_grid" in df.columns:
            value_col = "Value_grid"
        else:
            raise ValueError("Missing required column: Value")
    has_df = False
    has_series = False
    has_scalar = False
    for item in df[value_col].tolist():
        if item is None:
            continue
        if isinstance(item, float) and np.isnan(item):
            continue
        if isinstance(item, pd.DataFrame):
            has_df = True
            continue
        if isinstance(item, pd.Series):
            has_series = True
            continue
        if np.isscalar(item):
            has_scalar = True
            continue
        raise ValueError(f"Unsupported nested Value type: {type(item)!r}")
    kinds = int(has_df) + int(has_series) + int(has_scalar)
    if kinds == 0:
        raise ValueError("Value column has no valid cells.")
    if kinds > 1:
        raise ValueError("Mixed nested Value types are not supported.")
    if has_df:
        return "raw"
    if has_series:
        return "trace"
    return "scalar"


def _iter_raw_tables(
    resolver: PathResolver,
    *,
    paradigm_slug: str,
) -> list[tuple[str, str, Path]]:
    """
    Deprecated compatibility helper.

    New workflow reads alignment-side raw tables and maps each metric to one pseudo
    `subparam=default` row for compatibility with older tests.
    """
    rows: list[tuple[str, str, Path]] = []
    slug = _normalize_slug(paradigm_slug)
    if not slug:
        return rows
    root = resolver.alignment_root / slug
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/na-raw.pkl")):
        metric_key = path.parent.name
        rows.append((metric_key, "default", path))
    return rows


def _iter_feature_source_tables(
    root: Path,
) -> tuple[list[tuple[Path, Path, pd.DataFrame]], list[str]]:
    """Compatibility helper retained for tests and ad-hoc plotting scans."""
    tables: list[tuple[Path, Path, pd.DataFrame]] = []
    errors: list[str] = []
    if not root.exists():
        return tables, errors
    for path in sorted(root.glob("**/*.pkl")):
        rel = path.relative_to(root)
        try:
            payload = load_pkl(path)
            if not isinstance(payload, pd.DataFrame):
                continue
            derived_type = _detect_derived_type(payload)
            if derived_type == "scalar":
                continue
            tables.append((path, rel, payload))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rel}: {exc}")
    return tables, errors


def _iter_alignment_raw_tables(
    resolver: PathResolver,
    *,
    trial_slug: str,
) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    root = resolver.alignment_root / trial_slug
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/na-raw.pkl")):
        rows.append((path.parent.name, path))
    return rows
