"""Common utilities shared across TFR, connectivity, and warping."""

from .timefreq import (  # noqa: F401
    compute_decimation,
    decimated_times_from_raw,
    channel_names_after_picks,
    infer_sfreq_from_times,
    make_frequency_grid,
    multitaper_mask_radius_time_s_from_freqs_n_cycles,
    multitaper_n_cycles_from_time_fwhm_vector,
    morlet_n_cycles_from_time_fwhm,
    morlet_n_cycles_from_time_fwhm_vector,
    morlet_mask_radius_time_s_from_freqs_n_cycles,
)

__all__ = [
    "make_frequency_grid",
    "compute_decimation",
    "infer_sfreq_from_times",
    "multitaper_n_cycles_from_time_fwhm_vector",
    "multitaper_mask_radius_time_s_from_freqs_n_cycles",
    "morlet_n_cycles_from_time_fwhm",
    "morlet_n_cycles_from_time_fwhm_vector",
    "morlet_mask_radius_time_s_from_freqs_n_cycles",
    "decimated_times_from_raw",
    "channel_names_after_picks",
]
