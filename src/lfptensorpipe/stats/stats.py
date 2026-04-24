"""Lightweight scalar statistics helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def single_var_stat_test(
    df: pd.DataFrame,
    var: str,
    value: str,
    p: float = 0.05,
) -> dict[str, Any]:
    """Run one-way ANOVA plus Tukey HSD for one categorical variable."""
    data = df[[var, value]].dropna().copy()
    if data.empty:
        raise ValueError("Input data is empty after dropping NA rows.")

    groups = [
        group[value].astype(float).to_numpy()
        for _, group in data.groupby(var, observed=True, sort=False)
    ]
    if len(groups) < 2:
        raise ValueError("The variable must have at least 2 unique groups.")

    statistic, p_value = stats.f_oneway(*groups)
    tukey = pairwise_tukeyhsd(
        endog=data[value].astype(float).to_numpy(),
        groups=data[var].astype(str).to_numpy(),
        alpha=p,
    )
    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    for column in tukey_df.columns:
        try:
            tukey_df[column] = pd.to_numeric(tukey_df[column], errors="raise")
        except (TypeError, ValueError):
            continue

    return {
        "test": "One-way ANOVA",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "multiple_comparisons": "Tukey HSD",
        "multiple_comparisons_results": tukey_df,
    }
