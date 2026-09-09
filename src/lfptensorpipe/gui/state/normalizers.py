"""Pure helpers for GUI parameter/default normalization and nested dict ops."""

from __future__ import annotations

import math
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
    return dict(
        fmin=1.0,
        fmax=200.0,
        n_fft=1024,
        average=True,
        method="welch",
        tmin=None,
        tmax=None,
        exclude_bad=True,
        bandwidth=1.0,
        cycles=None,
        n_freqs=400,
        spacing="linear",
    )


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
    if any(not math.isfinite(item) or item <= 0.0 for item in parsed):
        raise ValueError("notches must contain positive finite numbers.")
    return parsed


def _normalize_optional_filter_frequency(
    value: Any,
    *,
    field_name: str,
) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite when provided.")
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
        l_freq = _normalize_optional_filter_frequency(
            merged["l_freq"], field_name="l_freq"
        )
        h_freq = _normalize_optional_filter_frequency(
            merged["h_freq"], field_name="h_freq"
        )
    except Exception as exc:  # noqa: BLE001
        return False, defaults, str(exc)

    if l_freq is not None and l_freq < 0.0:
        return False, defaults, "l_freq must be >= 0 when provided."
    if h_freq is not None and h_freq <= 0.0:
        return False, defaults, "h_freq must be > 0 when provided."
    if l_freq is not None and h_freq is not None and h_freq <= l_freq:
        return (
            False,
            defaults,
            "h_freq must be greater than l_freq when both are provided.",
        )

    return True, {"notches": notches, "l_freq": l_freq, "h_freq": h_freq}, ""


def default_preproc_viz_tfr_params() -> dict[str, Any]:
    return dict(
        fmin=1.0,
        fmax=120.0,
        n_freqs=40,
        decim=4,
        method="morlet",
        tmin=None,
        tmax=None,
        exclude_bad=True,
        average=True,
        cycles=None,
        window_length_s=4.0,
        bandwidth=1.0,
        spacing="log",
    )


def _normalize_viz(params, mode):
    defaults = (
        default_preproc_viz_psd_params()
        if mode == "psd"
        else default_preproc_viz_tfr_params()
    )
    if params is not None and not isinstance(params, dict):
        return False, defaults, f"{mode.upper()} params must be a dictionary."
    merged = {**defaults, **{k: v for k, v in (params or {}).items() if k in defaults}}
    try:
        methods = (
            ("welch", "multitaper", "morlet")
            if mode == "psd"
            else ("morlet", "multitaper")
        )
        if merged["method"] not in methods:
            raise ValueError("Unsupported method.")
        for key in ("average", "exclude_bad"):
            if not isinstance(merged[key], bool):
                raise ValueError(f"{key} must be true or false.")
        active = ["fmin", "fmax", "tmin", "tmax"]
        if mode == "psd" and merged["method"] == "welch":
            active += ["n_fft"]
        else:
            active += ["n_freqs"]
            if merged["spacing"] not in ("linear", "log"):
                raise ValueError("spacing must be linear or log.")
        if merged["method"] == "morlet":
            active += ["cycles"]
        if merged["method"] == "multitaper":
            active += ["bandwidth"]
            if mode == "tfr":
                active += ["window_length_s"]
        if mode == "tfr":
            active += ["decim"]
        for key in active:
            raw = merged[key]
            if key in ("tmin", "tmax", "cycles") and (raw is None or raw == ""):
                merged[key] = None
                continue
            if isinstance(raw, bool):
                raise ValueError(f"{key} must be numeric.")
            try:
                value = float(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{key} must be numeric.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite.")
            if key in ("n_fft", "n_freqs", "decim"):
                minimum = {"n_fft": 16, "n_freqs": 4, "decim": 1}[key]
                if not value.is_integer() or value < minimum:
                    raise ValueError(f"{key} must be an integer >= {minimum}.")
                value = int(value)
            elif value < 0 or (key not in ("fmin", "tmin") and value == 0):
                raise ValueError(f"{key} is outside the allowed range.")
            merged[key] = value
        if merged["fmax"] <= merged["fmin"]:
            raise ValueError("fmax must be greater than fmin.")
        if merged["fmin"] == 0 and (
            mode == "tfr"
            or merged["method"] == "morlet"
            or (merged["method"] != "welch" and merged["spacing"] == "log")
        ):
            raise ValueError(
                "fmin must be positive for wavelets or logarithmic spacing."
            )
        if merged["tmax"] is not None and merged["tmax"] <= (merged["tmin"] or 0):
            raise ValueError("tmax must be greater than tmin.")
        if (
            mode == "tfr"
            and merged["method"] == "multitaper"
            and merged["window_length_s"] * merged["bandwidth"] < 2
        ):
            raise ValueError("window_length_s times bandwidth must be >= 2.")
    except ValueError as exc:
        return False, defaults, str(exc)
    return True, merged, ""


def normalize_preproc_viz_psd_params(params):
    return _normalize_viz(params, "psd")


def normalize_preproc_viz_tfr_params(params):
    return _normalize_viz(params, "tfr")


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
