"""Alignment method catalog and static defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentMethodSpec:
    """Display/runtime definition for one alignment method."""

    key: str
    display_name: str


ALIGNMENT_METHODS: tuple[AlignmentMethodSpec, ...] = (
    AlignmentMethodSpec("linear_warper", "Line Up Key Events"),
    AlignmentMethodSpec("pad_warper", "Clip Around Event"),
    AlignmentMethodSpec("stack_warper", "Stack Trials"),
    AlignmentMethodSpec("concat_warper", "Stitch Trials"),
)
ALIGNMENT_METHODS_BY_KEY = {item.key: item for item in ALIGNMENT_METHODS}
ALIGNMENT_METHODS_BY_LABEL = {item.display_name: item for item in ALIGNMENT_METHODS}

DEFAULT_DROP_MODE = "substring"
DEFAULT_DROP_FIELDS = ("bad", "edge")


__all__ = [
    "ALIGNMENT_METHODS",
    "ALIGNMENT_METHODS_BY_KEY",
    "ALIGNMENT_METHODS_BY_LABEL",
    "DEFAULT_DROP_FIELDS",
    "DEFAULT_DROP_MODE",
    "AlignmentMethodSpec",
]
