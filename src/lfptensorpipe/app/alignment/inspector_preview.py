"""Display reductions for Epoch Inspector, separate from exported Features."""

import numpy as np

from lfptensorpipe.utils.transforms import apply_transform_array
from .inspector_qc import nan_stat

# Reductions retain their axes, so each metric can place its transform
# before or after either reduction. Omitting a reduction preserves individual values.
MEAN_STEPS = ("channel_mean", "epoch_mean")
PREVIEW_STEPS = {
    "raw_power": ("dB", *MEAN_STEPS),
    "periodic": ("dB", *MEAN_STEPS),
    "burst": ("log10", *MEAN_STEPS),
    "aperiodic": ("parameter_zscore", *MEAN_STEPS),
}
REDUCERS = {"mean": np.nanmean, "median": np.nanmedian}


def prepare_epoch_preview(metric, tensor, selected):
    """Return individual epoch planes and the four panels computed from them."""
    epochs = np.where(np.isfinite(tensor), tensor, np.nan)
    checked = np.isin(np.arange(len(epochs)), list(selected))
    groups = None
    for step in PREVIEW_STEPS.get(metric, MEAN_STEPS):
        if step.startswith("epoch_"):
            reducer = REDUCERS[step.removeprefix("epoch_")]
            groups = [
                (
                    nan_stat(reducer, epochs[mask], axis=0, keepdims=True)
                    if mask.any()
                    else np.full((1, *epochs.shape[1:]), np.nan)
                )
                for mask in (~checked, checked)
            ]
            continue
        if step.startswith("channel_"):
            reducer = REDUCERS[step.removeprefix("channel_")]

            def apply(values):
                return nan_stat(reducer, values, axis=1, keepdims=True)

        elif step == "parameter_zscore":
            # Each channel/parameter shares a reference across both View groups.
            center = nan_stat(np.nanmean, epochs, axis=(0, 3), keepdims=True)
            std = nan_stat(np.nanstd, epochs, axis=(0, 3), keepdims=True)
            scale = np.where(std > 0, std, np.nan)

            def apply(values):
                return (values - center) / scale

        else:

            def apply(values):
                return apply_transform_array(values, mode=step)

        epochs = apply(epochs)
        if groups is not None:
            groups = [apply(values) for values in groups]
    if epochs.shape[1] != 1:
        raise ValueError(
            "Select a single channel for a policy without channel reduction."
        )
    if groups is None:
        groups = [epochs[~checked], epochs[checked]]
    left, right = [
        group[0, 0] if len(group) == 1 else np.full(epochs.shape[2:], np.nan)
        for group in groups
    ]
    planes = epochs[:, 0]
    median = nan_stat(np.nanmedian, planes, axis=0)
    median[np.isfinite(planes).sum(axis=0) < 2] = np.nan
    deviation = nan_stat(np.nanmedian, np.abs(planes - median), axis=1)
    return planes, (left, right, left - right, deviation)


def preview_maps(metric, tensor, selected):
    """Return the four group comparison and deviation panels."""
    _, maps = prepare_epoch_preview(metric, tensor, selected)
    return maps
