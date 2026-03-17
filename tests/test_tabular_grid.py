"""Tests for tabular grid NumPy integration helpers."""

from __future__ import annotations

import numpy as np

from lfptensorpipe.tabular.grid import _trapezoid


def test_trapezoid_helper_uses_numpy_2_api() -> None:
    x = np.array([0.0, 1.0, 2.0], dtype=float)
    y = np.array([1.0, 3.0, 5.0], dtype=float)

    assert _trapezoid(y, x=x) == float(np.trapezoid(y, x=x))
