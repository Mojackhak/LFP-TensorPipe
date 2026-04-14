"""Pure helpers for GUI parameter/default normalization and nested dict ops."""

from __future__ import annotations

from typing import Any


def deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(dict(merged.get(key, {})), dict(value))
        else:
            merged[key] = value
    return merged


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for token in path:
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def nested_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if not path:
        return
    current: dict[str, Any] = payload
    for token in path[:-1]:
        node = current.get(token)
        if not isinstance(node, dict):
            node = {}
            current[token] = node
        current = node
    current[path[-1]] = value


def default_preproc_viz_psd_params() -> dict[str, Any]:
    return {
        "fmin": 1.0,
        "fmax": 200.0,
        "n_fft": 1024,
        "average": True,
    }


def default_preproc_filter_basic_params() -> dict[str, Any]:
    return {
        "notches": [50.0, 100.0],
        "l_freq": 1.0,
        "h_freq": 200.0,
    }


def normalize_filter_notches_config(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            return []
        parsed = [float(item) for item in parts]
    elif isinstance(value, (int, float)):
        parsed = [float(value)]
    elif isinstance(value, (list, tuple)):
        parsed = [float(item) for item in value]
    else:
        raise ValueError("notches must be a number, list, or comma-separated string.")
    if any(item <= 0.0 for item in parsed):
        raise ValueError("notches must be positive numbers.")
    return parsed


def normalize_preproc_filter_basic_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], str]:
    defaults = default_preproc_filter_basic_params()
    if params is None:
        return True, defaults, ""
    if not isinstance(params, dict):
        return False, defaults, "Filter basic params must be a dictionary."

    merged = dict(defaults)
    for key in ("notches", "l_freq", "h_freq"):
        if key in params:
            merged[key] = params[key]
    try:
        notches = normalize_filter_notches_config(merged["notches"])
        l_freq = float(merged["l_freq"])
        h_freq = float(merged["h_freq"])
    except Exception as exc:  # noqa: BLE001
        return False, defaults, str(exc)

    if l_freq < 0.0:
        return False, defaults, "l_freq must be >= 0."
    if h_freq <= l_freq:
        return False, defaults, "h_freq must be greater than l_freq."

    return True, {"notches": notches, "l_freq": l_freq, "h_freq": h_freq}, ""


def normalize_preproc_viz_psd_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], str]:
    defaults = default_preproc_viz_psd_params()
    if params is None:
        return True, defaults, ""
    if not isinstance(params, dict):
        return False, defaults, "PSD params must be a dictionary."

    merged = dict(defaults)
    merged.update({key: params[key] for key in defaults if key in params})
    try:
        fmin = float(merged["fmin"])
        fmax = float(merged["fmax"])
        n_fft = int(merged["n_fft"])
        average_raw = merged["average"]
        if isinstance(average_raw, str):
            average = average_raw.strip().lower() in {"1", "true", "yes", "on"}
        else:
            average = bool(average_raw)
    except Exception as exc:  # noqa: BLE001
        return False, defaults, str(exc)

    if fmin < 0.0:
        return False, defaults, "PSD fmin must be >= 0."
    if fmax <= fmin:
        return False, defaults, "PSD fmax must be greater than fmin."
    if n_fft < 16:
        return False, defaults, "PSD n_fft must be >= 16."

    return True, {"fmin": fmin, "fmax": fmax, "n_fft": n_fft, "average": average}, ""


def default_preproc_viz_tfr_params() -> dict[str, Any]:
    return {
        "fmin": 1.0,
        "fmax": 120.0,
        "n_freqs": 40,
        "decim": 4,
    }


def normalize_preproc_viz_tfr_params(
    params: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any], str]:
    defaults = default_preproc_viz_tfr_params()
    if params is None:
        return True, defaults, ""
    if not isinstance(params, dict):
        return False, defaults, "TFR params must be a dictionary."

    merged = dict(defaults)
    merged.update({key: params[key] for key in defaults if key in params})
    try:
        fmin = float(merged["fmin"])
        fmax = float(merged["fmax"])
        n_freqs = int(merged["n_freqs"])
        decim = int(merged["decim"])
    except Exception as exc:  # noqa: BLE001
        return False, defaults, str(exc)

    if fmin <= 0.0:
        return False, defaults, "TFR fmin must be > 0."
    if fmax <= fmin:
        return False, defaults, "TFR fmax must be greater than fmin."
    if n_freqs < 4:
        return False, defaults, "TFR n_freqs must be >= 4."
    if decim < 1:
        return False, defaults, "TFR decim must be >= 1."

    return (
        True,
        {
            "fmin": fmin,
            "fmax": fmax,
            "n_freqs": n_freqs,
            "decim": decim,
        },
        "",
    )


__all__ = [
    "deep_merge_dict",
    "nested_get",
    "nested_set",
    "default_preproc_viz_psd_params",
    "default_preproc_filter_basic_params",
    "normalize_filter_notches_config",
    "normalize_preproc_filter_basic_params",
    "normalize_preproc_viz_psd_params",
    "default_preproc_viz_tfr_params",
    "normalize_preproc_viz_tfr_params",
]
