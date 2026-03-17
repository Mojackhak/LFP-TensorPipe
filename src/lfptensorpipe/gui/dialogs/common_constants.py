"""Shared dialog constants and stage metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

INDICATOR_COLORS = {
    "gray": "#B0B0B0",
    "yellow": "#FFCC00",
    "green": "#34C759",
}

WINDOW_DEFAULT_SIZE = (900, 600)
WINDOW_MIN_SIZE = (900, 600)
ROOT_MARGIN = 8
ROOT_SPACING = 8
LEFT_WIDTH_RATIO = 0.33
LEFT_WIDTH_MIN = 200
LEFT_WIDTH_MAX = 600
STAGE_CONTENT_MIN_WIDTH = 400
PANEL_SPACING = 8
GRID_SPACING = 6
STAGE_PANEL_MARGIN = 8
STAGE_PANEL_SPACING = 6
PAGE_MARGIN = 6
PAGE_SPACING = 8
BUTTON_TEXT_HORIZONTAL_PADDING = 18
BUSY_FRAMES = ("Busy   ", "Busy.  ", "Busy.. ", "Busy...")
BUSY_INTERVAL_MS = 180
DISABLE_MATLAB_WARMUP_ENV = "LFPTP_DISABLE_MATLAB_WARMUP"
PREPROC_FILTER_DEFAULTS_KEY = "filter_advance_defaults"
PREPROC_FILTER_BASIC_DEFAULTS_KEY = "filter_basic_defaults"
PREPROC_VIZ_PSD_DEFAULTS_KEY = "viz_psd_defaults"
PREPROC_VIZ_TFR_DEFAULTS_KEY = "viz_tfr_defaults"
FEATURES_AXES_DEFAULTS_KEY = "features_axes_defaults"
TENSOR_SELECTOR_DEFAULTS_KEY = "selector_defaults"
TENSOR_BANDS_DEFAULTS_KEY = "bands_defaults"
TENSOR_METRIC_DEFAULTS_KEY = "metric_defaults"
TENSOR_PSI_BANDS_DEFAULTS_KEY = "psi_bands_defaults"
TENSOR_BURST_BANDS_DEFAULTS_KEY = "burst_bands_defaults"
PREPROC_VIZ_STEP_ORDER = (
    "raw",
    "filter",
    "annotations",
    "bad_segment_removal",
    "ecg_artifact_removal",
    "finish",
)
PREPROC_VIZ_STEP_LABELS = {
    "raw": "0. Raw",
    "filter": "1. Filter",
    "annotations": "2. Annotations",
    "bad_segment_removal": "3. Bad Segment Removal",
    "ecg_artifact_removal": "4. ECG Artifact Removal",
    "finish": "5. Finish",
}
TENSOR_CHANNEL_METRIC_KEYS = {"raw_power", "periodic_aperiodic", "burst"}
TENSOR_UNDIRECTED_METRIC_KEYS = {"coherence", "plv", "ciplv", "pli", "wpli"}
TENSOR_DIRECTED_METRIC_KEYS = {"trgc", "psi"}
TENSOR_COMMON_BASIC_METRIC_KEYS = {
    "raw_power",
    "periodic_aperiodic",
    "coherence",
    "plv",
    "ciplv",
    "pli",
    "wpli",
    "trgc",
}
TENSOR_BASIC_PARAM_ROWS_BY_METRIC = {
    "raw_power": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
        }
    ),
    "periodic_aperiodic": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
            "freq_range_hz",
        }
    ),
    "coherence": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
        }
    ),
    "plv": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
        }
    ),
    "ciplv": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
        }
    ),
    "pli": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
        }
    ),
    "wpli": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
        }
    ),
    "trgc": frozenset(
        {
            "low_freq_hz",
            "high_freq_hz",
            "freq_step_hz",
            "time_resolution_s",
            "hop_s",
        }
    ),
    "psi": frozenset({"bands", "time_resolution_s", "hop_s"}),
    "burst": frozenset({"bands", "percentile"}),
}
T = TypeVar("T")
LOCALIZE_PATH_CONFIG_FILENAME = "paths.yml"
LOCALIZE_PATH_FIELD_LABELS = {
    "leaddbs_dir": "Lead-DBS Directory",
    "matlab_root": "MATLAB Installation Path",
}
LOCALIZE_PATH_FIELD_TOOLTIPS = {
    "leaddbs_dir": "Lead-DBS root directory (used to discover spaces and atlases).",
    "matlab_root": (
        "Root directory of the local MATLAB installation. "
        "On macOS this is typically the MATLAB .app bundle root. "
        "The app derives MATLAB Engine bootstrap files automatically."
    ),
}
RECORD_IMPORT_DEFAULTS_KEY = "record_import_defaults"
RECORD_IMPORT_LAST_TYPE_KEY = "last_import_type"
RECORD_CONFIG_FILENAME = "record.yml"
RECORD_RESET_REFERENCE_DEFAULTS_KEY = "reset_reference_defaults"
RECORD_IMPORT_TYPES = (
    "Medtronic",
    "PINS",
    "Sceneray",
    "Legacy (MNE supported)",
    "Legacy (CSV)",
)
MNE_SUPPORTED_RECOMMENDED_EXTENSIONS = (
    ".fif",
    ".fif.gz",
    ".edf",
    ".bdf",
    ".vhdr",
    ".vmrk",
    ".eeg",
    ".set",
    ".fdt",
    ".cnt",
    ".gdf",
    ".lay",
    ".dat",
    ".nedf",
    ".ns1",
    ".ns2",
    ".ns3",
    ".ns4",
    ".ns5",
    ".ns6",
    ".ncs",
)
FEATURE_DERIVED_TYPES = ("raw", "spectral", "trace", "scalar")
FEATURE_PLOT_TRANSFORM_OPTIONS = (
    ("none", "none"),
    ("dB", "dB"),
    ("ln", "log"),
    ("log10", "log10"),
    ("fisherz", "fisherz"),
    ("fisherz_sqrt", "fisherz_sqrt"),
    ("logit", "logit"),
    ("asinh", "asinh"),
)
FEATURE_PLOT_TRANSFORM_MODES = tuple(
    value for _, value in FEATURE_PLOT_TRANSFORM_OPTIONS
)
FEATURE_PLOT_NORMALIZE_MODES = ("none", "mean", "ratio", "percent", "zscore")
FEATURE_PLOT_BASELINE_MODES = ("mean", "median", "max", "min")
FEATURE_PLOT_COLORMAPS = ("viridis", "cmcrameri.vik")
FEATURE_AUTO_BAND_METRICS = frozenset(
    {"aperiodic", "periodic_aperiodic", "psi", "burst"}
)


def normalize_feature_plot_transform_mode(value: object) -> str:
    """Normalize plot-transform aliases to canonical transform_df mode names."""
    token = str(value).strip()
    if not token:
        return "none"
    if token in FEATURE_PLOT_TRANSFORM_MODES:
        return token
    token_lower = token.lower()
    mapping = {
        "none": "none",
        "db": "dB",
        "log": "log",
        "ln": "log",
        "log10": "log10",
        "fisherz": "fisherz",
        "fisherz_sqrt": "fisherz_sqrt",
        "logit": "logit",
        "asinh": "asinh",
    }
    return mapping.get(token_lower, token)


@dataclass(frozen=True)
class StageSpec:
    """Display/internal route definition for one stage page."""

    key: str
    display_name: str
    route_key: str
    prerequisite_key: str | None = None


STAGE_SPECS: tuple[StageSpec, ...] = (
    StageSpec("preproc", "Preprocess Signal", "stage/preproc"),
    StageSpec("tensor", "Build Tensor", "stage/tensor", prerequisite_key="preproc"),
    StageSpec(
        "alignment", "Align Epochs", "stage/alignment", prerequisite_key="tensor"
    ),
    StageSpec(
        "features", "Extract Features", "stage/features", prerequisite_key="alignment"
    ),
)


__all__ = [
    name for name in globals() if not (name.startswith("__") and name.endswith("__"))
]
