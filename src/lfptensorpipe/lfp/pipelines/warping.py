"""Warping helpers for time-aligned tensors.

This module provides :func:`warp_tensor` for a single tensor item:

.. code-block:: python

    tensor = {
        "tensor": <ndarray>,
        "meta": <dict>,
    }

Warping relies on a paradigm-specific ``warp_fn`` returned by warper builders
(e.g., linear_warper, pad_warper, concat_warper).

Conventions
-----------
Unwarped group meta must contain:

    meta['axes'] = {'epoch', 'channel', 'freq', 'time', ...}

Warped group meta will be rewritten to:

    meta['axes'] = {'annotation', 'epoch', 'channel', 'freq', 'time', 'percent', ...}

Notes
-----
- ``axes['freq']`` may be numeric (Hz) or strings (band/parameter names).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np

from ..common import infer_sfreq_from_times
from ..warp.metadata import build_warped_tensor_metadata


def warp_tensor(
    tensor: Mapping[str, Any],
    *,
    warp_fn,
    n_samples: int,
) -> dict[str, Any]:
    """Warp a single tensor item.

    The expected input structure is:

    .. code-block:: python

        tensor = {
            "tensor": <ndarray> | None,
            "meta": <dict>,
        }

    Args:
        tensor: Single tensor item.
        warp_fn: Warper callable.
        n_samples: Warped time-axis length.

    Returns:
        tensor_warped: Same structure, but ``tensor`` is warped and ``meta`` is
            replaced by warped metadata.
    """
    if not isinstance(tensor, Mapping):
        raise ValueError(f"`tensor` must be a mapping, got {type(tensor)}.")
    if "meta" not in tensor:
        raise ValueError("`tensor` is missing required key 'meta'.")
    if "tensor" not in tensor:
        raise ValueError("`tensor` is missing required key 'tensor'.")

    src_meta = dict(tensor["meta"])
    if "axes" not in src_meta or "time" not in src_meta["axes"]:
        raise ValueError("`tensor['meta']` must contain axes['time'].")

    if not isinstance(n_samples, int):
        raise ValueError(f"`n_samples` must be an int, got {type(n_samples)}.")

    # Infer sampling rate from the input time axis (ignore NaNs).
    sr = infer_sfreq_from_times(src_meta["axes"]["time"])

    arr = tensor["tensor"]
    if arr is None:
        return {
            "tensor": None,
            "meta": src_meta,
        }

    warped, percent_axis, meta_epochs = warp_fn(
        np.asarray(arr),
        sr=float(sr),
        n_samples=int(n_samples),
    )
    meta_warped: Dict[str, Any] = build_warped_tensor_metadata(
        src_meta["axes"],
        np.asarray(percent_axis, dtype=float),
        meta_epochs,
        source_meta=src_meta,
    )
    return {
        "tensor": np.asarray(warped, dtype=np.float64),
        "meta": meta_warped,
    }
