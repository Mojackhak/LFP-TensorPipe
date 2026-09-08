"""Windowed line-noise estimation shared by Filter preview and finalization."""

from copy import deepcopy
import math
from typing import Any

import numpy as np

MODEL_LABELS = {
    "mne_spectrum_fit": "MNE spectrum_fit",
    "sinusoidal_regression": "Sinusoidal regression",
    "cleanline": "CleanLine",
    "removepli": "removePLI",
}


def default_notch_model() -> dict[str, Any]:
    return {
        "enabled": False,
        "method": "cleanline",
        "params_by_method": {
            "sinusoidal_regression": {
                "window_length_s": 4.0,
                "window_overlap_percent": 50.0,
            },
            "mne_spectrum_fit": {
                "window_length_s": 4.0,
                "fit_width_hz": 1.0,
                "multitaper_bandwidth_hz": None,
            },
            "cleanline": {
                "window_length_s": 4.0,
                "window_overlap_percent": 50.0,
                "frequency_search_enabled": True,
                "search_radius_hz": 0.5,
                "significance_threshold": 0.01,
                "per_channel_thresholds_enabled": False,
                "significance_thresholds_by_channel": {},
                "multitaper_bandwidth_hz": 2.0,
                "limit_over_subtraction": False,
                "background_radius_hz": 10.0,
                "background_bandwidth_hz": 1.0,
            },
            "removepli": {
                "fundamental_frequency_hz": 50.0,
                "harmonic_count": 2,
                "amplitude_phase_settling_time_s": 1.0,
                "frequency_tracking_bandwidth": {
                    "initial_hz": 50.0,
                    "final_hz": 0.2,
                    "transition_time_s": 1.0,
                },
                "frequency_tracking_settling_time": {
                    "initial_s": 0.1,
                    "final_s": 4.0,
                    "transition_time_s": 1.0,
                },
            },
        },
    }


def normalize_notch_model(value: Any) -> dict[str, Any]:
    """Parse either a UI snapshot or the active configuration from a run log."""
    result = default_notch_model()
    if value is None:
        return result
    if not isinstance(value, dict):
        raise ValueError("notch_model must be an object.")
    enabled = value.get("enabled", False)
    method = value.get("method", result["method"])
    if not isinstance(enabled, bool):
        raise ValueError("notch_model.enabled must be true or false.")
    if not isinstance(method, str) or method not in MODEL_LABELS:
        raise ValueError(f"Unsupported notch_model.method: {method!r}.")
    saved = value.get("params_by_method", {})
    if not isinstance(saved, dict):
        raise ValueError("notch_model.params_by_method must be an object.")
    saved = deepcopy(saved)
    if "params" in value:
        saved[method] = value["params"]
    for key, defaults in result["params_by_method"].items():
        supplied = saved.get(key, {})
        if not isinstance(supplied, dict):
            raise ValueError(f"notch_model.{key} params must be an object.")
        if not enabled or key != method:
            for field, default in list(defaults.items()):
                raw = supplied.get(field, default)
                defaults[field] = (
                    {**default, **raw}
                    if isinstance(default, dict) and isinstance(raw, dict)
                    else deepcopy(raw)
                )
            continue
        for field, default in list(defaults.items()):
            raw = supplied.get(field, default)
            if key == "cleanline" and (
                (
                    field == "search_radius_hz"
                    and supplied.get(
                        "frequency_search_enabled", defaults["frequency_search_enabled"]
                    )
                    is False
                    and supplied.get("limit_over_subtraction", False) is False
                )
                or (
                    field in {"background_radius_hz", "background_bandwidth_hz"}
                    and supplied.get("limit_over_subtraction", False) is False
                )
                or (
                    field == "significance_thresholds_by_channel"
                    and supplied.get(
                        "per_channel_thresholds_enabled",
                        defaults["per_channel_thresholds_enabled"],
                    )
                    is False
                )
            ):
                defaults[field] = deepcopy(raw)
                continue
            if field == "significance_thresholds_by_channel":
                if not isinstance(raw, dict):
                    raise ValueError(
                        "notch_model.cleanline.significance_thresholds_by_channel must be an object."
                    )
                overrides = {}
                for name, threshold in raw.items():
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError(
                            "CleanLine threshold channel names must be nonempty strings."
                        )
                    if threshold is None or threshold == "":
                        continue
                    overrides[name] = _model_number(
                        threshold,
                        "cleanline.significance_thresholds_by_channel.significance_threshold",
                    )
                defaults[field] = overrides
            elif isinstance(default, dict):
                if not isinstance(raw, dict):
                    raise ValueError(f"notch_model.{key}.{field} must be an object.")
                for name, number in default.items():
                    default[name] = _model_number(
                        raw.get(name, number), f"{key}.{field}.{name}"
                    )
            elif isinstance(default, bool):
                if not isinstance(raw, bool):
                    raise ValueError(
                        f"notch_model.{key}.{field} must be true or false."
                    )
                defaults[field] = raw
            elif (
                key == "mne_spectrum_fit"
                and field == "multitaper_bandwidth_hz"
                and raw is None
            ):
                defaults[field] = None
            else:
                defaults[field] = _model_number(raw, f"{key}.{field}")
    if enabled and method == "cleanline":
        active = result["params_by_method"][method]
        if (
            active["limit_over_subtraction"]
            and active["window_length_s"] * active["background_bandwidth_hz"] < 3
        ):
            raise ValueError(
                "notch_model.cleanline.background_bandwidth_hz requires window length times bandwidth >= 3 for at least two tapers."
            )
    result.update(enabled=enabled, method=method)
    return result


def merge_notch_model_log(draft, logged):
    """Overlay effective run settings while retaining inactive UI drafts."""
    result = default_notch_model() if draft is None else deepcopy(draft)
    result["enabled"] = logged.get("enabled", False)
    if "method" in logged:
        result["method"] = logged["method"]
    if result["enabled"]:
        method = result["method"]
        values = result.setdefault("params_by_method", {}).setdefault(method, {})
        recorded = logged.get(
            "params", logged.get("params_by_method", {}).get(method, {})
        )
        if method == "cleanline":
            values["limit_over_subtraction"] = recorded.get(
                "limit_over_subtraction", False
            )
            values["per_channel_thresholds_enabled"] = recorded.get(
                "per_channel_thresholds_enabled", False
            )
        for key, value in recorded.items():
            if (
                key != "significance_thresholds_by_channel"
                and isinstance(value, dict)
                and isinstance(values.get(key), dict)
            ):
                values[key].update(deepcopy(value))
            else:
                values[key] = deepcopy(value)
    return normalize_notch_model(result)


def _model_number(raw, path):
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"notch_model.{path} must be numeric.") from exc
    if isinstance(raw, bool) or not math.isfinite(number):
        raise ValueError(f"notch_model.{path} must be finite.")
    field = path.rsplit(".", 1)[-1]
    if field == "window_overlap_percent":
        valid = 0 <= number < 100
    elif field == "fit_width_hz":
        valid = number >= 0
    elif field == "significance_threshold":
        valid = 0 < number < 1
    elif field == "harmonic_count":
        valid = number >= 1 and number.is_integer()
    else:
        valid = number > 0
    if not valid:
        raise ValueError(f"Invalid notch_model.{path}: {number}.")
    return int(number) if field == "harmonic_count" else number


def model_notch_frequencies(frequencies, model):
    """Resolve the active fundamental/harmonic controls without changing the draft."""
    if model and model["enabled"] and model["method"] == "removepli":
        params = model["params"]
        return [
            params["fundamental_frequency_hz"] * k
            for k in range(1, params["harmonic_count"] + 1)
        ]
    return list(frequencies) if frequencies is not None else []


def effective_notch_model(model: dict[str, Any]) -> dict[str, Any]:
    """Project a normalized UI model onto the parameters actually consumed."""
    if not model["enabled"]:
        return {"enabled": False}
    params = deepcopy(model["params_by_method"][model["method"]])
    if (
        model["method"] == "cleanline"
        and not params["frequency_search_enabled"]
        and not params["limit_over_subtraction"]
    ):
        params.pop("search_radius_hz", None)
    if model["method"] == "cleanline":
        if not params["limit_over_subtraction"]:
            for key in (
                "limit_over_subtraction",
                "background_radius_hz",
                "background_bandwidth_hz",
            ):
                params.pop(key, None)
        enabled = params.pop("per_channel_thresholds_enabled")
        overrides = params.pop("significance_thresholds_by_channel")
        overrides = overrides if enabled else {}
        overrides = {
            name: value
            for name, value in overrides.items()
            if value != params["significance_threshold"]
        }
        if enabled and overrides:
            params["per_channel_thresholds_enabled"] = True
            params["significance_thresholds_by_channel"] = overrides
    return {"enabled": True, "method": model["method"], "params": params}


def cleanline_channel_thresholds(model, channel_names):
    """Resolve active thresholds at the Raw channel-name boundary."""
    if not model or not model["enabled"] or model["method"] != "cleanline":
        return {}
    params = model["params"]
    overrides = (
        params.get("significance_thresholds_by_channel", {})
        if params.get("per_channel_thresholds_enabled", False)
        else {}
    )
    unknown = set(overrides) - set(channel_names)
    if unknown:
        raise ValueError(
            f"CleanLine threshold channels are absent from the Filter input: {', '.join(sorted(unknown))}."
        )
    return {
        name: overrides.get(name, params["significance_threshold"])
        for name in channel_names
    }


def model_support_samples(model: dict[str, Any], sfreq: float, n_times: int) -> int:
    """Bound the consumed span of the configured windowed estimator."""
    if model["method"] in {"cleanline", "removepli"}:
        return max(0, int(n_times) - 1)
    params = model["params"]
    width = max(1, int(np.ceil(params["window_length_s"] * sfreq)))
    if model["method"] == "mne_spectrum_fit":
        return width + width // 2 - 1
    return width - 1


def subtract_notch_model(
    data,
    sfreq,
    frequencies,
    model,
    *,
    significance_thresholds=None,
    diagnostics=None,
    background_bounds=None,
):
    """Subtract a validated active model from one independently valid segment."""
    x = np.asarray(data, dtype=float)
    freqs = np.asarray(model_notch_frequencies(frequencies, model), dtype=float)
    params = model["params"]
    if np.any(~np.isfinite(freqs)) or np.any((freqs <= 0) | (freqs >= sfreq / 2)):
        raise ValueError("Model notch frequencies must be positive and below Nyquist.")
    if model["method"] == "removepli":
        from .removepli import remove_pli

        return remove_pli(x, sfreq, params)
    samples = params["window_length_s"] * sfreq
    width = (
        int(np.floor(samples + 0.5))
        if model["method"] == "cleanline"
        else int(np.ceil(samples))
    )
    if width < 1 or x.shape[-1] < width:
        raise ValueError(
            f"{model['method']} requires at least {width} samples "
            f"({params['window_length_s']:g} s); segment has {x.shape[-1]}."
        )
    if model["method"] == "cleanline":
        from .cleanline import clean_line

        if (
            params.get("per_channel_thresholds_enabled", False)
            and significance_thresholds is None
        ):
            raise ValueError(
                "CleanLine per-channel thresholds require channel-name resolution through finalize_reviewed_lfp_filter."
            )
        return clean_line(
            x,
            sfreq,
            freqs,
            params,
            significance_thresholds=significance_thresholds,
            diagnostics=diagnostics,
            background_bounds=background_bounds,
        )
    if model["method"] == "mne_spectrum_fit":
        import mne

        half = params["fit_width_hz"] / 2
        if np.any(freqs - half <= 0) or np.any(freqs + half >= sfreq / 2):
            raise ValueError("Model fitting bands must lie strictly within Nyquist.")
        return mne.filter.notch_filter(
            x,
            sfreq,
            freqs=freqs,
            method="spectrum_fit",
            filter_length=width,
            notch_widths=params["fit_width_hz"],
            mt_bandwidth=params["multitaper_bandwidth_hz"],
            verbose="ERROR",
        )
    if model["method"] != "sinusoidal_regression":
        raise ValueError(f"Unsupported notch model: {model['method']}.")
    hop = max(1, int(round(width * (1 - params["window_overlap_percent"] / 100))))
    time = np.arange(width) / sfreq
    angles = 2 * np.pi * time[:, None] * freqs
    harmonics = np.column_stack((np.sin(angles), np.cos(angles)))
    design = np.column_stack((np.ones(width), harmonics))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Window cannot identify all requested sinusoidal components.")
    inverse = np.linalg.pinv(design)
    flat = x.reshape(-1, x.shape[-1])
    starts = list(range(0, x.shape[-1] - width + 1, hop))
    if starts[-1] != x.shape[-1] - width:
        starts.append(x.shape[-1] - width)
    estimate = np.zeros_like(flat)
    weights = np.zeros(x.shape[-1])
    for index, start in enumerate(starts):
        taper = np.hanning(width + 2)[1:-1] if hop < width else np.ones(width)
        if index == 0:
            taper[: max(1, width // 2)] = 1
        if index == len(starts) - 1:
            taper[width // 2 :] = 1
        fitted = (flat[:, start : start + width] @ inverse.T)[:, 1:] @ harmonics.T
        estimate[:, start : start + width] += fitted * taper
        weights[start : start + width] += taper
    return (flat - estimate / weights).reshape(x.shape)
