"""Alignment warper-construction helpers."""

from __future__ import annotations

from typing import Any, Callable


def _svc():
    from . import service as svc

    return svc


def _build_warper(
    raw: Any,
    *,
    method: str,
    method_params: dict[str, Any],
    linear_warper_fn: Callable[..., Any] | None = None,
    pad_warper_fn: Callable[..., Any] | None = None,
    concat_warper_fn: Callable[..., Any] | None = None,
    stack_warper_fn: Callable[..., Any] | None = None,
) -> tuple[dict[str, list[Any]], Callable[..., Any]]:
    svc = _svc()
    linear_warper_fn = linear_warper_fn or svc.linear_warper
    pad_warper_fn = pad_warper_fn or svc.pad_warper
    concat_warper_fn = concat_warper_fn or svc.concat_warper
    stack_warper_fn = stack_warper_fn or svc.stack_warper
    labels = sorted(
        set(str(item) for item in raw.annotations.description if str(item).strip())
    )
    mode = "exact"
    drop_mode = (
        str(method_params.get("drop_mode", svc.DEFAULT_DROP_MODE)).strip().lower()
    )
    if drop_mode not in {"exact", "substring"}:
        drop_mode = str(svc.DEFAULT_DROP_MODE)
    drop_bad = bool(method_params.get("drop_bad", True))
    drop_fields = svc._normalize_drop_fields(
        method_params.get("drop_fields", list(svc.DEFAULT_DROP_FIELDS))
    )
    anno_drop = tuple(drop_fields) if drop_bad else None

    if method == "linear_warper":
        anchors = method_params.get("anchors_percent", {})
        if not isinstance(anchors, dict) or len(anchors) < 2:
            if len(labels) >= 2:
                anchors = {0.0: labels[0], 100.0: labels[-1]}
            elif len(labels) == 1:
                anchors = {0.0: labels[0], 100.0: labels[0]}
            else:
                raise ValueError(
                    "No annotation labels available for linear warper anchors."
                )
        epoch_duration_range = svc._float_pair_list(
            method_params.get("epoch_duration_range"),
            (None, None),
        )
        return linear_warper_fn(
            raw,
            anchors_percent=anchors,
            mode=mode,
            drop_mode=drop_mode,
            epoch_duration_range=epoch_duration_range,
            linear_warp=bool(method_params.get("linear_warp", True)),
            percent_tolerance=float(method_params.get("percent_tolerance", 15.0)),
            anno_drop=anno_drop,
        )
    if method == "pad_warper":
        keep = method_params.get("annotations")
        if keep is None:
            keep = []
        if not isinstance(keep, list):
            keep = labels
        keep = [str(item) for item in keep if str(item).strip()]
        if not keep:
            raise ValueError("No annotation labels selected for pad warper.")
        pad_left = float(method_params.get("pad_left", 0.5))
        anno_left = float(method_params.get("anno_left", 0.5))
        anno_right = float(method_params.get("anno_right", 0.5))
        pad_right = float(method_params.get("pad_right", 0.5))
        anno_allowed = {
            label: (pad_left, anno_left, anno_right, pad_right) for label in keep
        }
        duration_range = svc._float_pair_list(
            method_params.get("duration_range"),
            (0.0, 1_000_000.0),
        )
        return pad_warper_fn(
            raw,
            anno_allowed=anno_allowed,
            mode=mode,
            drop_mode=drop_mode,
            duration_range=duration_range,
            anno_drop=anno_drop,
        )
    if method == "concat_warper":
        keep = method_params.get("annotations")
        if keep is None:
            keep = []
        if not isinstance(keep, list):
            keep = labels
        keep = [str(item) for item in keep if str(item).strip()]
        if not keep:
            raise ValueError("No annotation labels selected for concat warper.")
        return concat_warper_fn(
            raw,
            keep=keep,
            mode=mode,
            drop_mode=drop_mode,
            anno_drop=anno_drop,
            pad_s=0.0,
            clip_to_raw=True,
            require_match=True,
        )
    keep = method_params.get("annotations")
    if keep is None:
        keep = []
    if not isinstance(keep, list):
        keep = labels
    keep = [str(item) for item in keep if str(item).strip()]
    if not keep:
        raise ValueError("No annotation labels selected for stack warper.")
    duration_range = svc._float_pair_list(
        method_params.get("duration_range"),
        (0.0, 1_000_000.0),
    )
    return stack_warper_fn(
        raw,
        keep=keep,
        mode=mode,
        drop_mode=drop_mode,
        duration_range=duration_range,
        anno_drop=anno_drop,
        pad_s=0.0,
        clip_to_raw=True,
        require_match=True,
    )


def _resolve_target_n_samples(
    *,
    method: str,
    method_params: dict[str, Any],
    epochs_by_label: dict[str, list[Any]],
) -> int:
    svc = _svc()
    sample_rate = float(
        method_params.get(
            "sample_rate",
            svc.default_alignment_method_params(method).get("sample_rate", 5.0),
        )
    )
    if method in {"linear_warper", "stack_warper"}:
        n_samples = int(round(sample_rate * 100.0))
    elif method == "pad_warper":
        total_window_s = (
            float(method_params.get("pad_left", 0.5))
            + float(method_params.get("anno_left", 0.5))
            + float(method_params.get("anno_right", 0.5))
            + float(method_params.get("pad_right", 0.5))
        )
        n_samples = int(round(sample_rate * total_window_s))
    elif method == "concat_warper":
        all_epochs = epochs_by_label.get("ALL", [])
        if not all_epochs:
            raise ValueError("No epochs available to derive concat duration.")
        epoch0 = all_epochs[0]
        if hasattr(epoch0, "total_duration_s"):
            total_window_s = float(getattr(epoch0, "total_duration_s"))
        elif hasattr(epoch0, "intervals_s"):
            intervals = getattr(epoch0, "intervals_s")
            total_window_s = float(
                sum(
                    max(0.0, float(end) - float(start))
                    for start, end in (intervals or [])
                )
            )
        else:
            raise ValueError("Unable to infer concat total duration.")
        n_samples = int(round(sample_rate * total_window_s))
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    if n_samples < 2:
        raise ValueError(
            f"Derived n_samples={n_samples} is invalid (method={method}, sample_rate={sample_rate})."
        )
    return int(n_samples)


__all__ = ["_build_warper", "_resolve_target_n_samples"]
