"""Background-constrained amplitude scaling of fixed line-noise components."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal.windows import dpss


def spectral_matrix(signals, sfreq, width, starts, bandwidth):
    """Equal-weight one-sided PSD/cross-PSD, averaged across identical windows."""
    product = width / sfreq * bandwidth / 2
    count = int(np.floor(2 * product - 1))
    if count < 2 or count > width or product >= width / 2:
        raise ValueError(
            "Background bandwidth and window length must support at least two tapers below Nyquist."
        )
    tapers = dpss(width, product, Kmax=count)
    gram = np.zeros((len(signals), len(signals), width // 2 + 1))
    for start in starts:
        block = signals[:, start : start + width]
        block = block - block.mean(axis=-1, keepdims=True)
        spectra = np.fft.rfft(block[:, None, :] * tapers, axis=-1)
        gram += np.einsum("itf,jtf->ijf", spectra, spectra.conj()).real
    gram /= len(starts) * count * sfreq
    gram[..., 1 : -1 if width % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(width, 1 / sfreq), gram


def candidate_power(gram, coefficients):
    """Power of x minus the fixed, scaled line components; includes cross terms."""
    weights = np.r_[1.0, -np.asarray(coefficients)]
    return np.einsum("i,ijf,j->f", weights, gram, weights)


def _interval_union(peaks, radius):
    intervals = []
    for peak in sorted(set(peaks)):
        left, right = float(peak - radius), float(peak + radius)
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(right, intervals[-1][1])
        else:
            intervals.append([left, right])
    return intervals


def _coordinate_candidates(gram, coefficients, index, objectives):
    """Bracket nonnegative search using power vertices and background crossings."""
    weights = np.r_[1.0, -coefficients.copy()]
    weights[index + 1] = 0.0
    quadratic = gram[index + 1, index + 1]
    cross = np.einsum("if,i->f", gram[index + 1], weights)
    constant = np.einsum("i,ijf,j->f", weights, gram, weights)
    critical = []
    for _, mask, target_db in objectives:
        active = quadratic[mask] > 0
        a, b, c = quadratic[mask][active], cross[mask][active], constant[mask][active]
        vertex = b / a
        discriminant = vertex**2 + (10 ** (target_db[active] / 10) - c) / a
        distance = np.sqrt(np.maximum(discriminant, 0))
        critical.extend((vertex, vertex - distance, vertex + distance))
    critical = np.concatenate(critical)
    critical = critical[critical > 0]
    upper = float(critical.max()) if critical.size else 0.0
    # Beyond the largest upper crossing/vertex, every contributing bin is above
    # its target and increasing. Sampling cost does not grow with alpha's size.
    return np.unique(
        np.r_[
            np.linspace(0, 1, 11),
            coefficients[index],
            critical,
            np.expm1(np.linspace(0, np.log1p(upper), 65)),
            upper,
        ]
    )


def fit_subtraction(
    data,
    components,
    peaks,
    frequencies,
    sfreq,
    width,
    starts,
    params,
    background_bounds=None,
    *,
    excluded_intervals=None,
):
    """Fit segment-constant coefficients and return auditable per-line results."""
    grid, gram = spectral_matrix(
        np.vstack((data, components)),
        sfreq,
        width,
        starts,
        params["background_bandwidth_hz"],
    )
    original = gram[0, 0]
    exclusions = (
        [_interval_union(found, params["search_radius_hz"]) for found in peaks]
        if excluded_intervals is None
        else excluded_intervals
    )
    masks = [
        (
            np.logical_or.reduce(
                [(grid >= left) & (grid <= right) for left, right in intervals]
            )
            if intervals
            else np.zeros(len(grid), dtype=bool)
        )
        for intervals in exclusions
    ]
    all_excluded = np.logical_or.reduce(masks)
    low, high = background_bounds if background_bounds is not None else (None, None)
    valid = (grid > 0) & (grid < sfreq / 2) & np.isfinite(original) & (original > 0)
    if low is not None:
        valid &= grid >= low
    if high is not None:
        valid &= grid <= high
    reports, objectives = [], []
    for index, center in enumerate(frequencies):
        report = {
            "target_hz": float(center),
            "coefficient": 0.0,
            "peak_range_hz": (
                [float(min(peaks[index])), float(max(peaks[index]))]
                if peaks[index]
                else None
            ),
            "excluded_intervals_hz": exclusions[index],
            "status": "no_detection",
            "reason": None,
            "converged": None,
            "sweeps": 0,
            "background_fit": None,
            "error_db_squared": None,
            "maximum_downward_db": None,
            "residual_peak_db": None,
        }
        reports.append(report)
        if not peaks[index]:
            continue
        left, right = exclusions[index][0][0], exclusions[index][-1][1]
        candidates = (
            valid
            & ~all_excluded
            & (abs(grid - center) <= params["background_radius_hz"])
        )
        donors_left, donors_right = (
            candidates & (grid < left),
            candidates & (grid > right),
        )
        evaluation = masks[index] & valid
        report["background_fit"] = {
            "center_hz": float(center),
            "left_slope_intercept_db": None,
            "right_slope_intercept_db": None,
            "boundary_hz": [left, right],
            "boundary_db": None,
            "left_bins": int(donors_left.sum()),
            "right_bins": int(donors_right.sum()),
        }
        if donors_left.sum() < 3 or donors_right.sum() < 3 or not evaluation.any():
            report.update(
                status="background_unavailable",
                reason="Insufficient valid background bins on both sides or no evaluation bins.",
            )
            continue
        # Center frequencies numerically, and use dB consistently on both sides.
        fit_left = np.polyfit(
            grid[donors_left] - center, 10 * np.log10(original[donors_left]), 1
        )
        fit_right = np.polyfit(
            grid[donors_right] - center, 10 * np.log10(original[donors_right]), 1
        )
        boundary_db = [
            float(np.polyval(fit_left, left - center)),
            float(np.polyval(fit_right, right - center)),
        ]
        target = np.interp(grid[evaluation], [left, right], boundary_db)
        report["status"] = "fitted"
        report["background_fit"].update(
            left_slope_intercept_db=fit_left.tolist(),
            right_slope_intercept_db=fit_right.tolist(),
            boundary_db=boundary_db,
        )
        objectives.append((index, evaluation, target))
    coefficients = np.zeros(len(frequencies))
    if not objectives:
        return data.copy(), reports
    # Numerical floor only, not a physiological background or a notch tolerance.
    floor = max(float(original.max()) * np.finfo(float).eps, np.finfo(float).tiny)

    def errors(values):
        power = np.maximum(candidate_power(gram, values), floor)
        return [10 * np.log10(power[mask]) - target for _, mask, target in objectives]

    def loss(values):
        return sum(float(np.mean(delta**2)) for delta in errors(values))

    converged = False
    order = sorted((index for index, _, _ in objectives), key=lambda i: frequencies[i])
    for sweep in range(10):
        previous = coefficients.copy()
        for index in order:

            def evaluate(value):
                trial = coefficients.copy()
                trial[index] = value
                return loss(trial)

            coarse = _coordinate_candidates(gram, coefficients, index, objectives)
            scores = np.array([evaluate(value) for value in coarse])
            choices = [
                (float(score), float(value)) for score, value in zip(scores, coarse)
            ]
            for point in range(len(coarse)):
                if (point == 0 or scores[point] <= scores[point - 1]) and (
                    point == len(coarse) - 1 or scores[point] <= scores[point + 1]
                ):
                    result = minimize_scalar(
                        evaluate,
                        bounds=(
                            coarse[max(0, point - 1)],
                            coarse[min(len(coarse) - 1, point + 1)],
                        ),
                        method="bounded",
                        options={"xatol": 1e-3},
                    )
                    choices.append((float(result.fun), float(result.x)))
            best = min(score for score, _ in choices)
            numerical_tie = 1e-12 * max(1.0, abs(best))
            coefficients[index] = min(
                value for score, value in choices if score <= best + numerical_tie
            )
        if np.max(abs(coefficients - previous)) < 1e-3:
            converged = True
            break
    for (index, _, _), delta in zip(objectives, errors(coefficients)):
        reports[index].update(
            coefficient=float(coefficients[index]),
            status="fitted" if converged else "not_converged",
            converged=converged,
            sweeps=sweep + 1,
            error_db_squared=float(np.mean(delta**2)),
            maximum_downward_db=float(max(0.0, -delta.min())),
            residual_peak_db=float(max(0.0, delta.max())),
        )
    return data - coefficients @ components, reports
