"""Read-only spectral QC with channel-specific, discontinuous valid support."""

import numpy as np
from mne.time_frequency import (
    morlet,
    psd_array_welch,
    psd_array_multitaper,
    tfr_array_morlet,
    tfr_array_multitaper,
)
from lfptensorpipe.lfp.mask.annotations import (
    annotation_sample_support_by_channel,
    valid_segments_from_annotation_support,
)


def finite_mean(values, axis=0):
    """Equal-weight finite mean, retaining all-missing cells without warnings."""
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    return np.divide(
        np.where(valid, values, 0).sum(axis=axis),
        count,
        out=np.full(count.shape, np.nan),
        where=count > 0,
    )


def compute_spectral_qc(raw, picks, params, mode):
    """Compute normalized GUI settings; return axes, linear power and drops."""
    p = params
    fs = float(raw.info["sfreq"])
    if p["fmax"] > fs / 2:
        raise ValueError("fmax must not exceed Nyquist.")
    if p["method"] != "welch" and p["fmax"] >= fs / 2 and mode == "tfr":
        raise ValueError("TFR fmax must be below Nyquist.")
    if p["method"] == "multitaper" and p["bandwidth"] >= fs:
        raise ValueError("bandwidth must be below the sampling frequency.")
    start = max(0, int(np.ceil((p["tmin"] or 0) * fs)))
    stop = min(
        raw.n_times,
        int(np.ceil(p["tmax"] * fs)) if p["tmax"] is not None else raw.n_times,
    )
    if start >= stop:
        raise ValueError("Selected time range contains no samples.")
    if mode == "psd" and p["method"] == "welch":
        freqs = np.fft.rfftfreq(p["n_fft"], 1 / fs)
        freqs = freqs[(freqs >= p["fmin"]) & (freqs <= p["fmax"])]
        if not freqs.size:
            raise ValueError("Frequency range contains no Welch FFT bins.")
    else:
        grid = np.geomspace if p["spacing"] == "log" else np.linspace
        freqs = grid(p["fmin"], p["fmax"], p["n_freqs"])
    indices = np.arange(start, stop, p["decim"] if mode == "tfr" else 1)
    power = np.full(
        (
            (len(picks), len(freqs), len(indices))
            if mode == "tfr"
            else (len(picks), len(freqs))
        ),
        np.nan,
    )
    weights = np.zeros((len(picks), len(freqs)))
    totals = np.zeros_like(weights)
    if p["exclude_bad"]:
        invalid, points, _ = annotation_sample_support_by_channel(
            raw, channels=picks, keep=("bad", "edge"), mode="prefix"
        )
    else:
        invalid = np.zeros((len(picks), raw.n_times), dtype=bool)
        points = [() for _ in picks]
    wavelet = mode == "tfr" or p["method"] == "morlet"
    if wavelet:
        if p["method"] == "morlet":
            cycles = (
                np.maximum(2.0, freqs / 4)
                if p["cycles"] is None
                else np.full(len(freqs), p["cycles"])
            )
            lengths = np.array(
                [len(w) for w in morlet(fs, freqs, n_cycles=cycles, zero_mean=True)]
            )
        else:
            cycles = freqs * p["window_length_s"]
            lengths = np.array(
                [len(np.arange(0.0, c / f, 1 / fs)) for c, f in zip(cycles, freqs)]
            )
            if p["window_length_s"] * p["bandwidth"] / 2 >= lengths.min() / 2:
                raise ValueError(
                    "Multitaper time-bandwidth product exceeds kernel support."
                )
    drops = []
    for channel, name in enumerate(picks):
        for left, right in valid_segments_from_annotation_support(
            invalid[channel], points[channel]
        ):
            left, right = max(left, start), min(right, stop)
            if right <= left:
                continue
            n = right - left
            eligible = (
                (2 * (lengths // 2) + 1 <= n)
                if wavelet
                else np.ones(len(freqs), dtype=bool)
            )
            if not wavelet and (
                (p["method"] == "welch" and n < p["n_fft"])
                or (p["method"] == "multitaper" and n * p["bandwidth"] / fs < 1)
            ):
                eligible[:] = False
            if not eligible.all():
                drops.append(
                    dict(
                        channel=name,
                        start_sample=int(left),
                        stop_sample=int(right),
                        frequencies_hz=freqs[~eligible].tolist(),
                        reason="Insufficient estimator support",
                    )
                )
            if not eligible.any():
                continue
            data = raw.get_data(picks=[name], start=left, stop=right)[0]
            if wavelet:
                kwargs = dict(
                    sfreq=fs,
                    freqs=freqs[eligible],
                    n_cycles=cycles[eligible],
                    zero_mean=True,
                    output="power",
                    decim=1,
                    n_jobs=1,
                    verbose="ERROR",
                )
                if p["method"] == "morlet":
                    values = tfr_array_morlet(data[None, None, :], **kwargs)[0, 0]
                else:
                    values = tfr_array_multitaper(
                        data[None, None, :],
                        time_bandwidth=p["window_length_s"] * p["bandwidth"],
                        **kwargs,
                    )[0, 0]
                for row, fi in enumerate(np.flatnonzero(eligible)):
                    # Conservative symmetric guard for even-length kernels too.
                    guard = int(lengths[fi] // 2)
                    valid_left, valid_right = left + guard, right - guard
                    if mode == "tfr":
                        select = (indices >= valid_left) & (indices < valid_right)
                        power[channel, fi, select] = values[row, indices[select] - left]
                    else:
                        block = values[row, guard : n - guard] if guard else values[row]
                        finite = np.isfinite(block)
                        totals[channel, fi] += block[finite].sum()
                        weights[channel, fi] += finite.sum()
            else:
                if p["method"] == "welch":
                    used = n // p["n_fft"] * p["n_fft"]
                    values, native = psd_array_welch(
                        data[:used],
                        fs,
                        n_fft=p["n_fft"],
                        n_per_seg=p["n_fft"],
                        n_overlap=0,
                        verbose="ERROR",
                    )
                else:
                    used = n
                    values, native = psd_array_multitaper(
                        data,
                        fs,
                        bandwidth=p["bandwidth"],
                        normalization="full",
                        adaptive=False,
                        verbose="ERROR",
                    )
                mapped = np.interp(freqs, native, values, left=np.nan, right=np.nan)
                finite = np.isfinite(mapped)
                totals[channel, finite] += mapped[finite] * used
                weights[channel, finite] += used
    if mode == "psd":
        np.divide(totals, weights, out=power, where=weights > 0)
    return dict(
        frequencies=freqs,
        times=indices / fs if mode == "tfr" else None,
        power=power,
        dropped=drops,
        density=mode == "psd" and p["method"] != "morlet",
    )
