"""Connectivity runner facade.

Concrete implementations live in focused metric modules.
"""

from __future__ import annotations

from .connectivity_psi import run_psi_metric
from .connectivity_trgc import (
    run_trgc_backend_metric,
    run_trgc_finalize_metric,
    run_trgc_metric,
)
from .connectivity_undirected import run_undirected_connectivity_metric

__all__ = [
    "run_undirected_connectivity_metric",
    "run_trgc_backend_metric",
    "run_trgc_finalize_metric",
    "run_trgc_metric",
    "run_psi_metric",
]
