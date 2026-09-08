"""Python translation of Mohammad Reza Keshtkaran's removePLI.

Copyright (c) 2013, Mohammad Reza Keshtkaran <keshtkaran.github@gmail.com>
All rights reserved.
This program is provided "AS IS" for non-commercial, educational
and reseach purpose only. Any commercial use, of any kind, of
this program is prohibited. The Copyright notice should remain intact.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

See LICENSES/GPL-3.0.txt and LICENSES/THIRD_PARTY_NOTICES.txt. Both upstream
notices are retained, without resolving their inconsistency.
Source: https://github.com/mrezak/removePLI, commit 9cb446a4ea585adb5c19a1810313b75f7789074b.
Python adaptation (2026): Ankang Hu. Added V/mV conversion and mean restoration.
"""

import math

import numpy as np
from scipy.signal import butter, sosfilt


@np.errstate(divide="ignore")
def remove_pli(data, sfreq, params):
    """Run the original adaptive harmonic estimator independently per channel."""
    fundamental = params["fundamental_frequency_hz"]
    count = params["harmonic_count"]
    bandwidth = params["frequency_tracking_bandwidth"]
    settling = params["frequency_tracking_settling_time"]
    if fundamental <= 2 or fundamental + 2 >= sfreq / 2:
        raise ValueError("removePLI fundamental +/-2 Hz must lie within Nyquist.")
    if bandwidth["final_hz"] >= sfreq / 2:
        raise ValueError("removePLI final tracking bandwidth must be below Nyquist.")
    sos = butter(
        2, [fundamental - 2, fundamental + 2], fs=sfreq, btype="bandpass", output="sos"
    )
    flat = np.asarray(data).reshape(-1, data.shape[-1])
    result = np.empty_like(flat)
    for channel, signal in enumerate(flat):
        mean = signal.mean()
        x = (signal - mean) * 1000
        filtered = sosfilt(sos, x)
        differentiated = np.concatenate(([0.0], np.diff(filtered)))
        initial = math.atan(math.pi * bandwidth["initial_hz"] / sfreq)
        final = math.tan(math.pi * bandwidth["final_hz"] / sfreq)
        alpha = (1 - initial) / (1 + initial)
        alpha_final = (1 - final) / (1 + final)
        alpha_step = math.exp(
            math.log(0.05) / (bandwidth["transition_time_s"] * sfreq + 1)
        )
        forgetting = math.exp(math.log(0.05) / (settling["initial_s"] * sfreq + 1))
        forgetting_final = math.exp(math.log(0.05) / (settling["final_s"] * sfreq + 1))
        forgetting_step = math.exp(
            math.log(0.05) / (settling["transition_time_s"] * sfreq + 1)
        )
        smoothing = math.tan(0.5 * math.pi * min(90, sfreq / 2) / sfreq)
        smoothing = (1 - smoothing) / (1 + smoothing)
        amplitude_forgetting = math.exp(
            math.log(0.05) / (params["amplitude_phase_settling_time_s"] * sfreq + 1)
        )
        kappa = previous = previous2 = 0.0
        numerator, denominator = 5.0, 10.0
        u, up = np.ones(count), np.ones(count)
        r1, r4 = np.full(count, 10.0), np.full(count, 10.0)
        a, b = np.zeros(count), np.zeros(count)
        for index, sample in enumerate(x):
            lattice = (
                differentiated[index]
                + kappa * (1 + alpha) * previous
                - alpha * previous2
            )
            numerator = forgetting * numerator + (1 - forgetting) * previous * (
                lattice + previous2
            )
            denominator = forgetting * denominator + (1 - forgetting) * 2 * previous**2
            estimate = min(1.0, max(-1.0, numerator / denominator))
            kappa = smoothing * kappa + (1 - smoothing) * estimate
            previous2, previous = previous, lattice
            alpha = alpha_step * alpha + (1 - alpha_step) * alpha_final
            forgetting = (
                forgetting_step * forgetting + (1 - forgetting_step) * forgetting_final
            )
            old_cos, current_cos = kappa, 1.0
            error = sample
            for harmonic in range(count):
                cosine = 2 * kappa * current_cos - old_cos
                old_cos, current_cos = current_cos, cosine
                tmp = cosine * (up[harmonic] + u[harmonic])
                tmp2 = up[harmonic]
                up[harmonic] = tmp - u[harmonic]
                u[harmonic] = tmp + tmp2
                # At cosine=-1, upstream obtains -inf and resets the gain to one.
                gain = 1.5 - (
                    up[harmonic] ** 2 - (cosine - 1) / (cosine + 1) * u[harmonic] ** 2
                )
                if gain <= 0:
                    gain = 1.0
                up[harmonic] *= gain
                u[harmonic] *= gain
                error -= a[harmonic] * u[harmonic] + b[harmonic] * up[harmonic]
                r1[harmonic] = amplitude_forgetting * r1[harmonic] + u[harmonic] ** 2
                r4[harmonic] = amplitude_forgetting * r4[harmonic] + up[harmonic] ** 2
                a[harmonic] += u[harmonic] * error / r1[harmonic]
                b[harmonic] += up[harmonic] * error / r4[harmonic]
            result[channel, index] = error / 1000 + mean
    if not np.isfinite(result).all():
        raise ValueError("removePLI estimator produced nonfinite samples.")
    return result.reshape(data.shape)
