"""Configuration specs for the PD paper workspace."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

MergeSpecValue = str | Sequence[str]
MergeSpec = Mapping[str, Mapping[str, MergeSpecValue]]
TransformModeCfg = Mapping[str, Mapping[str, str]]

DEFAULT_PROJECT_ROOT = Path("/Users/mojackhu/Research/pd")
DEFAULT_STRICT_SELECTION = False

DEFAULT_MERGE_SPEC: dict[str, dict[str, MergeSpecValue]] = {
    "cycle": {"gait": "cycle_l"},
    "med": {"sit": "off", "sit_medon": "on"},
    "motor": {"gait": ["stand", "walk"]},
    "turn_stack": {"gait": "turn_stack"},
    "turn": {"gait": "turn"},
}

DEFAULT_TRANSFORM_MODE_CFG: dict[str, dict[str, str]] = {
    "periodic": {
        "default": "dB",
    },
    "raw_power": {
        "default": "dB",
    },
    "burst": {
        "rate": "asinh",
        "occupation": "none",
        "default": "log10",
    },
    "coherence": {
        "default": "fisherz_sqrt",
    },
    "ciplv": {
        "default": "logit",
    },
    "wpli": {
        "default": "logit",
    },
    "plv": {
        "default": "logit",
    },
    "pli": {
        "default": "logit",
    },
    "default": {
        "default": "none",
    },
}

DEFAULT_NORMALIZE_SPEC: dict[str, dict[str, object]] = {
    "cycle": {
        "baseline": [0, 100],
        "slice_mode": "percent",
        "mode": "mean",
    },
    "turn_stack": {
        "baseline": [0, 100],
        "slice_mode": "percent",
        "mode": "mean",
    },
    "turn": {
        "baseline": [0, 25],
        "slice_mode": "percent",
        "mode": "mean",
    },
}

DEFAULT_SCALAR_NORMALIZE_SPEC: dict[str, dict[str, object]] = {
    "cycle": {
        "baseline": {"Phase": "Off"},
    },
    "turn": {
        "baseline": {"Phase": "Pre"},
    },
    "med": {
        "baseline": {"Phase": "Off"},
    },
    "motor": {
        "baseline": {"Phase": "Stand"},
    },
}

SIMPLIFIED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "subject",
    "channel",
    "side",
    "mni_x",
    "mni_y",
    "mni_z",
    "region",
    "band",
    "phase",
    "lat",
    "value",
)

PHASE_FILTERS_BY_NAME: dict[str, frozenset[str]] = {
    "cycle": frozenset({"Strike", "Off"}),
    "turn": frozenset({"Pre", "Onset", "Offset", "Post"}),
}

NON_CONNECTIVITY_METRICS: frozenset[str] = frozenset(
    {
        "aperiodic",
        "periodic",
        "raw_power",
        "burst",
    }
)

UNORDERED_CONNECTIVITY_METRICS: frozenset[str] = frozenset(
    {
        "coherence",
        "plv",
        "pli",
        "ciplv",
        "wpli",
    }
)

ORDERED_CONNECTIVITY_METRICS: frozenset[str] = frozenset(
    {
        "trgc",
        "psi",
    }
)

SUMMARY_GROUP_COLS: tuple[str, ...] = (
    "Subject",
    "Channel",
    "Band",
    "Phase",
    "Lat",
)

SCALAR_NORMALIZE_GROUP_COLS: tuple[str, ...] = (
    "Subject",
    "Channel",
    "Band",
    "Lat",
)

LEFT_CHANNELS: frozenset[str] = frozenset({"0_1", "1_2", "2_3"})
RIGHT_CHANNELS: frozenset[str] = frozenset({"8_9", "9_10", "10_11"})

MERGE_PHASE_BY_NAME_TRIAL: dict[str, dict[str, str]] = {
    "med": {
        "off": "Off",
        "on": "On",
    },
    "motor": {
        "stand": "Stand",
        "walk": "Walk",
    },
}

SUMMARY_SUFFIX = "_summary"
TRANSFORM_SUFFIX = "_trans"
NORMALIZED_SUFFIX = "_normalized"
SHIFT_SUFFIX = "_shift"

TRACE_LIKE_FILE_NAMES = {
    "mean-trace.pkl",
    "na-raw.pkl",
}

SHIFT_ENABLED_NAMES: frozenset[str] = frozenset({"cycle"})
SHIFT_FILE_NAMES: frozenset[str] = frozenset(TRACE_LIKE_FILE_NAMES)

PASSTHROUGH_FILE_SUFFIXES = (
    "-scalar.pkl",
    "-spectral.pkl",
)
