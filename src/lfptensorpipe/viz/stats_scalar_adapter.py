"""Adapters that normalize scalar stats outputs for visualdf Tukey brackets."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .visualdf import p_to_stars


def prepare_scalar_tuk_from_single_var_stat(
    stats_result: Mapping[str, Any],
    *,
    x_var: str,
    context: Mapping[str, Any] | None = None,
) -> tuple[None, pd.DataFrame]:
    """Convert one-way ANOVA post hoc output into plot-ready Tukey rows."""
    raw = stats_result.get("multiple_comparisons_results")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return None, pd.DataFrame()

    tuk = raw.copy()
    rename_map: dict[str, str] = {}
    if "group1" not in tuk.columns and "A" in tuk.columns:
        rename_map["A"] = "group1"
    if "group2" not in tuk.columns and "B" in tuk.columns:
        rename_map["B"] = "group2"
    if rename_map:
        tuk = tuk.rename(columns=rename_map)

    if "group1" not in tuk.columns or "group2" not in tuk.columns:
        raise ValueError("Tukey results must include group1/group2 columns.")

    p_column = _resolve_p_column(tuk)
    tuk["p.value"] = pd.to_numeric(tuk[p_column], errors="coerce")
    tuk["stars"] = tuk["p.value"].map(p_to_stars)

    if context is not None:
        for key, value in context.items():
            tuk[key] = value

    ordered_cols = ["group1", "group2", "p.value", "stars"]
    context_cols = [key for key in (context or {}) if key not in ordered_cols]
    extra_cols = [
        column
        for column in tuk.columns
        if column not in ordered_cols and column not in context_cols
    ]
    tuk = tuk[ordered_cols + context_cols + extra_cols]
    return None, tuk


def _resolve_p_column(frame: pd.DataFrame) -> str:
    for name in ("p.value", "p-adj", "p_adj", "pval", "p_value", "p"):
        if name in frame.columns:
            return name
    raise ValueError("Could not resolve a p-value column from Tukey results.")
