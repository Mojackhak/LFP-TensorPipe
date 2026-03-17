"""Alignment method default and validation logic."""

from __future__ import annotations

from typing import Any

from .method_specs import (
    ALIGNMENT_METHODS_BY_KEY,
    DEFAULT_DROP_FIELDS,
)
from .param_normalizers import (
    _normalize_anchors,
    _normalize_annotations,
    _normalize_drop_fields,
    _normalize_duration_range,
    _normalize_nonnegative_float,
    _normalize_sample_rate,
)


def default_alignment_method_params(method_key: str) -> dict[str, Any]:
    key = str(method_key).strip()
    if key == "linear_warper":
        return {
            "anchors_percent": {},
            "epoch_duration_range": [None, None],
            "linear_warp": True,
            "percent_tolerance": 15.0,
            "drop_bad": True,
            "drop_fields": list(DEFAULT_DROP_FIELDS),
            "sample_rate": 5.0,
        }
    if key == "pad_warper":
        return {
            "annotations": [],
            "pad_left": 0.5,
            "anno_left": 0.5,
            "anno_right": 0.5,
            "pad_right": 0.5,
            "duration_range": [0.0, 1_000_000.0],
            "drop_bad": True,
            "drop_fields": list(DEFAULT_DROP_FIELDS),
            "sample_rate": 50.0,
        }
    if key == "concat_warper":
        return {
            "annotations": [],
            "drop_bad": True,
            "drop_fields": list(DEFAULT_DROP_FIELDS),
            "sample_rate": 50.0,
        }
    return {
        "annotations": [],
        "duration_range": [0.0, 1_000_000.0],
        "drop_bad": True,
        "drop_fields": list(DEFAULT_DROP_FIELDS),
        "sample_rate": 5.0,
    }


def validate_alignment_method_params(
    method_key: str,
    params: dict[str, Any] | None,
    *,
    annotation_labels: list[str] | None = None,
) -> tuple[bool, dict[str, Any], str]:
    key = str(method_key).strip()
    if key not in ALIGNMENT_METHODS_BY_KEY:
        return False, {}, f"Unknown alignment method: {key}"
    candidate = params if isinstance(params, dict) else {}
    defaults = default_alignment_method_params(key)
    labels = [
        str(item).strip() for item in (annotation_labels or []) if str(item).strip()
    ]

    ok_sample_rate, sample_rate, msg_sample_rate = _normalize_sample_rate(
        candidate.get("sample_rate", defaults["sample_rate"]),
        fallback=float(defaults.get("sample_rate", 5.0)),
    )
    if not ok_sample_rate:
        return False, defaults, msg_sample_rate
    drop_bad = bool(candidate.get("drop_bad", defaults.get("drop_bad", True)))
    drop_fields = _normalize_drop_fields(
        candidate.get("drop_fields", defaults.get("drop_fields"))
    )

    if key == "linear_warper":
        anchor_value = candidate.get("anchors_percent", defaults["anchors_percent"])
        if isinstance(anchor_value, dict) and len(anchor_value) == 0:
            anchors: dict[float, str] = {}
        else:
            ok_anchors, anchors, msg_anchors = _normalize_anchors(anchor_value)
            if not ok_anchors:
                return False, defaults, msg_anchors
        ok_range, epoch_duration_range, msg_range = _normalize_duration_range(
            candidate.get("epoch_duration_range", defaults["epoch_duration_range"]),
            allow_none=True,
        )
        if not ok_range:
            return False, defaults, msg_range
        try:
            percent_tolerance = float(
                candidate.get("percent_tolerance", defaults["percent_tolerance"])
            )
        except Exception:  # noqa: BLE001
            return False, defaults, "percent_tolerance must be numeric."
        if percent_tolerance < 0.0:
            return False, defaults, "percent_tolerance must be >= 0."
        if int(round(sample_rate * 100.0)) < 2:
            return False, defaults, "sample_rate is too small for linear_warper."
        return (
            True,
            {
                "anchors_percent": anchors,
                "epoch_duration_range": epoch_duration_range,
                "linear_warp": bool(
                    candidate.get("linear_warp", defaults["linear_warp"])
                ),
                "percent_tolerance": percent_tolerance,
                "drop_bad": drop_bad,
                "drop_fields": drop_fields,
                "sample_rate": sample_rate,
            },
            "",
        )

    if key == "pad_warper":
        annotation_value = candidate.get("annotations", defaults["annotations"])
        ok_annotations, annotations, msg_annotations = _normalize_annotations(
            annotation_value,
            allow_empty=True,
        )
        if not ok_annotations and labels:
            ok_annotations, annotations, msg_annotations = _normalize_annotations(
                labels,
                allow_empty=True,
            )
        if not ok_annotations and not labels:
            annotations = []
            ok_annotations = True
        if not ok_annotations:
            return False, defaults, msg_annotations
        ok_range, duration_range, msg_range = _normalize_duration_range(
            candidate.get("duration_range", defaults["duration_range"]),
            allow_none=False,
        )
        if not ok_range:
            return False, defaults, msg_range
        ok_pad_left, pad_left, msg_pad_left = _normalize_nonnegative_float(
            candidate.get("pad_left", defaults["pad_left"]),
            field_name="pad_left",
            fallback=float(defaults.get("pad_left", 0.5)),
        )
        if not ok_pad_left:
            return False, defaults, msg_pad_left
        ok_anno_left, anno_left, msg_anno_left = _normalize_nonnegative_float(
            candidate.get("anno_left", defaults["anno_left"]),
            field_name="anno_left",
            fallback=float(defaults.get("anno_left", 0.5)),
        )
        if not ok_anno_left:
            return False, defaults, msg_anno_left
        ok_anno_right, anno_right, msg_anno_right = _normalize_nonnegative_float(
            candidate.get("anno_right", defaults["anno_right"]),
            field_name="anno_right",
            fallback=float(defaults.get("anno_right", 0.5)),
        )
        if not ok_anno_right:
            return False, defaults, msg_anno_right
        ok_pad_right, pad_right, msg_pad_right = _normalize_nonnegative_float(
            candidate.get("pad_right", defaults["pad_right"]),
            field_name="pad_right",
            fallback=float(defaults.get("pad_right", 0.5)),
        )
        if not ok_pad_right:
            return False, defaults, msg_pad_right
        total_window_s = pad_left + anno_left + anno_right + pad_right
        if total_window_s <= 0.0:
            return (
                False,
                defaults,
                "pad_left + anno_left + anno_right + pad_right must be > 0.",
            )
        if int(round(sample_rate * total_window_s)) < 2:
            return False, defaults, "sample_rate is too small for pad_warper."
        return (
            True,
            {
                "annotations": annotations,
                "pad_left": pad_left,
                "anno_left": anno_left,
                "anno_right": anno_right,
                "pad_right": pad_right,
                "duration_range": duration_range,
                "drop_bad": drop_bad,
                "drop_fields": drop_fields,
                "sample_rate": sample_rate,
            },
            "",
        )

    annotation_value = candidate.get("annotations", defaults["annotations"])
    ok_annotations, annotations, msg_annotations = _normalize_annotations(
        annotation_value,
        allow_empty=True,
    )
    if not ok_annotations and labels:
        ok_annotations, annotations, msg_annotations = _normalize_annotations(
            labels,
            allow_empty=True,
        )
    if not ok_annotations and not labels:
        annotations = []
        ok_annotations = True
    if not ok_annotations:
        return False, defaults, msg_annotations
    if key == "concat_warper":
        return (
            True,
            {
                "annotations": annotations,
                "drop_bad": drop_bad,
                "drop_fields": drop_fields,
                "sample_rate": sample_rate,
            },
            "",
        )
    ok_range, duration_range, msg_range = _normalize_duration_range(
        candidate.get(
            "duration_range", defaults.get("duration_range", [0.0, 1_000_000.0])
        ),
        allow_none=False,
    )
    if not ok_range:
        return False, defaults, msg_range
    if key == "stack_warper" and int(round(sample_rate * 100.0)) < 2:
        return False, defaults, "sample_rate is too small for stack_warper."
    return (
        True,
        {
            "annotations": annotations,
            "duration_range": duration_range,
            "drop_bad": drop_bad,
            "drop_fields": drop_fields,
            "sample_rate": sample_rate,
        },
        "",
    )


__all__ = ["default_alignment_method_params", "validate_alignment_method_params"]
