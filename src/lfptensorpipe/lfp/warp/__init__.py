"""Warping subpackage.

This subpackage contains utilities to map time-aligned tensors (TFR or
connectivity) onto a normalized axis, typically for averaging across events.

Warpers provided:
  - linear_warper: generic event-anchored, piecewise-linear time normalization
  - pad_warper   : crop+concat around annotations, with optional resampling
  - concat_warper: concatenate multiple allowed annotation intervals, then resample
  - stack_warper : keep each matched annotation as one epoch, then resample+stack
"""

from .event_anchored_linear import LinearEpoch, linear_warper
from .pad_concat import PadEpoch, pad_warper
from .concat import ConcatEpoch, concat_warper
from .stack import StackEpoch, stack_warper

from .metadata import build_warped_tensor_metadata

__all__ = [
    "LinearEpoch",
    "PadEpoch",
    "ConcatEpoch",
    "StackEpoch",
    "linear_warper",
    "pad_warper",
    "concat_warper",
    "stack_warper",
    "build_warped_tensor_metadata",
]
