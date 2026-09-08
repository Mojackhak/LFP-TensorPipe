"""Python translation of SCCN's revised CleanLine (GPL version 2).

Tim Mullen, SCCN/INC/UCSD Copyright (C) 2011.
Source: https://github.com/sccn/cleanline,
commit 117bffa6e1fbc6d042a7e75cec9e289e1aa5be72.
Includes translated Chronux taper/spectral computations bundled by SCCN.
Python adaptation (2026): Ankang Hu. Added complete, end-aligned tail coverage
and optional per-line accumulation for adaptive background-based subtraction.
See LICENSES/THIRD_PARTY_NOTICES.txt and LICENSES/GPL-2.0.txt.
Distributed WITHOUT ANY WARRANTY, including MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE, under the upstream GNU General Public License.
"""

import numpy as np
from scipy.signal.windows import dpss
from scipy.special import expit
from scipy.stats import f as f_distribution


def clean_line(
    data,
    sfreq,
    frequencies,
    params,
    *,
    significance_thresholds=None,
    diagnostics=None,
    background_bounds=None,
):
    """Subtract significant multitaper sinusoidal fits using upstream iteration."""
    adaptive = params.get("limit_over_subtraction", False)
    if adaptive:
        ordered = np.sort(frequencies)
        if np.any(np.diff(ordered) <= 2 * params["search_radius_hz"]):
            raise ValueError(
                "Adaptive CleanLine target search/exclusion intervals must not overlap."
            )
    duration = params["window_length_s"]
    width = int(np.floor(sfreq * duration + 0.5))
    step = max(
        1,
        int(
            np.floor(
                sfreq * duration * (1 - params["window_overlap_percent"] / 100) + 0.5
            )
        ),
    )
    product = duration * params["multitaper_bandwidth_hz"] / 2
    count = int(np.floor(2 * product - 1))
    if count < 2 or count > width or product >= width / 2:
        raise ValueError(
            "CleanLine requires at least two tapers and time-bandwidth product below half the window samples."
        )
    tapers = dpss(width, product, Kmax=count) * np.sqrt(sfreq)
    nfft = 2 ** (int(np.ceil(np.log2(width))) + 2)
    grid = np.fft.rfftfreq(nfft, 1 / sfreq)
    h0 = tapers[::2].sum(axis=1)
    h0sq = np.sum(h0**2)
    radius = params["search_radius_hz"] if params["frequency_search_enabled"] else 0.0
    if any(
        center - radius <= 0 or center + radius >= sfreq / 2 for center in frequencies
    ):
        raise ValueError("CleanLine search bands must lie strictly within Nyquist.")
    length = data.shape[-1]
    starts = list(range(0, length - width + 1, step))
    if starts[-1] != length - width:
        starts.append(length - width)
    # Chronux createdatamatc floors event starts but rounds the window length.
    spectrum_starts = np.floor(
        np.arange(max(1, int(np.floor((length / sfreq - duration) / duration)) + 1))
        * duration
        * sfreq
    ).astype(int)

    def spectrum(signal):
        power = np.zeros(len(grid))
        for start in spectrum_starts:
            fft = (
                np.fft.rfft(tapers * signal[start : start + width], n=nfft, axis=-1)
                / sfreq
            )
            power += np.mean(np.abs(fft) ** 2, axis=0)
        with np.errstate(divide="ignore"):
            return 10 * np.log10(power / len(spectrum_starts))

    flat = np.asarray(data).reshape(-1, length)
    result = flat.copy()
    time = np.arange(width) / sfreq
    for channel in range(len(flat)):
        alpha = (
            params["significance_threshold"]
            if significance_thresholds is None
            else significance_thresholds[channel]
        )
        threshold = f_distribution.ppf(1 - alpha, 2, 2 * count - 2)
        signal = result[channel]
        centers = np.asarray(frequencies).copy()
        bins = np.array([np.argmin(abs(grid - center)) for center in centers])
        initial = spectrum(signal)
        if adaptive:
            components = np.zeros((len(frequencies), length))
            peaks = [set() for _ in frequencies]
            active_indices = np.arange(len(frequencies))
        for _ in range(10):
            significant = np.zeros(len(centers), dtype=bool)
            datafit = np.zeros(length)
            previous_fit = None
            if adaptive:
                component_fit = np.zeros_like(components)
                previous_components = None
            for window, start in enumerate(starts):
                fft = (
                    np.fft.rfft(tapers * signal[start : start + width], n=nfft, axis=-1)
                    / sfreq
                )
                amplitude = np.sum(fft[::2] * h0[:, None], axis=0) / h0sq
                residual = np.sum(
                    abs(fft[::2] - h0[:, None] * amplitude) ** 2, axis=0
                ) + np.sum(abs(fft[1::2]) ** 2, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    statistic = (count - 1) * abs(amplitude) ** 2 * h0sq / residual
                selected = np.zeros(len(grid), dtype=bool)
                if adaptive:
                    window_components = np.zeros((len(frequencies), width))
                for index, center in enumerate(centers):
                    low = np.argmin(abs(grid - (center - radius)))
                    high = np.argmin(abs(grid - (center + radius)))
                    candidates = statistic[low : high + 1]
                    passing = np.where(candidates >= threshold, candidates, 0)
                    if np.any(passing):
                        selected_bin = low + np.argmax(passing)
                        if adaptive:
                            if selected[selected_bin]:
                                raise ValueError(
                                    "Adaptive CleanLine targets resolve to the same Fourier bin."
                                )
                            owner = active_indices[index]
                            peaks[owner].add(float(grid[selected_bin]))
                            window_components[owner] = 2 * np.real(
                                np.exp(2j * np.pi * time * grid[selected_bin])
                                * amplitude[selected_bin]
                                * sfreq
                            )
                        selected[selected_bin] = True
                        significant[index] = True
                fit = 2 * np.real(
                    np.exp(2j * np.pi * time[:, None] * grid[selected])
                    @ (amplitude[selected] * sfreq)
                )
                if window:
                    overlap = starts[window - 1] + width - start
                    if overlap:
                        smooth = expit(
                            100 * (np.arange(1, overlap + 1) - overlap / 2) / overlap
                        )
                        fit[:overlap] = (
                            smooth * fit[:overlap]
                            + (1 - smooth) * previous_fit[-overlap:]
                        )
                if adaptive:
                    if window and overlap:
                        window_components[:, :overlap] = (
                            smooth * window_components[:, :overlap]
                            + (1 - smooth) * previous_components[:, -overlap:]
                        )
                    component_fit[:, start : start + width] = window_components
                    previous_components = window_components
                datafit[start : start + width] = fit
                previous_fit = fit
            signal -= datafit
            if adaptive:
                components += component_fit
            if significant.any():
                cleaned = spectrum(signal)
                with np.errstate(invalid="ignore"):
                    increasing = (initial - cleaned)[bins] < 0
                keep = significant & ~increasing
                centers, bins = centers[keep], bins[keep]
                if adaptive:
                    active_indices = active_indices[keep]
                initial = cleaned
            if not len(centers):
                break
        if adaptive:
            from .cleanline_adaptive import fit_subtraction

            result[channel], report = fit_subtraction(
                flat[channel],
                components,
                peaks,
                frequencies,
                sfreq,
                width,
                starts,
                params,
                background_bounds,
            )
            if diagnostics is not None:
                diagnostics.extend(
                    {"channel_index": channel, **entry} for entry in report
                )
    return result.reshape(data.shape)
