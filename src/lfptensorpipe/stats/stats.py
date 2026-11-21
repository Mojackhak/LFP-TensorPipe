"""Lightweight scalar statistics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
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
    labels = tukey.groupsunique
    left, right = np.triu_indices(len(labels), k=1)
    tukey_df = pd.DataFrame(
        {
            "group1": labels[left],
            "group2": labels[right],
            "meandiff": tukey.meandiffs,
            "p-adj": tukey.pvalues,
            "lower": tukey.confint[:, 0],
            "upper": tukey.confint[:, 1],
            "reject": tukey.reject,
        }
    )

    return {
        "test": "One-way ANOVA",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "multiple_comparisons": "Tukey HSD",
        "multiple_comparisons_results": tukey_df,
    }
