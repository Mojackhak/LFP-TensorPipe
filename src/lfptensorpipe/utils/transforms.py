"""Element-wise numeric transforms for tensors and arrays.

This module centralizes small, reusable transform primitives used across the
project (stats, visualization, and tensor post-processing).

The key design goal is to keep transforms:
  - **pure** (no side effects),
  - **vectorized** (NumPy arrays in / arrays out),
  - **explicitly invertible** when possible.

Notes
-----
These transforms are intentionally *element-wise*.
They do not do any alignment / broadcasting based on metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Mapping, cast

import numpy as np

TransformMode = Literal[
    "dB",
    "log",
    "fisherz",
    "fisherz_sqrt",
    "logit",
    "asinh",
    "none",
    None,
]

TransformDomain = Literal["native", "transformed"]
VALUE_TRANSFORM_POLICY_KEY = "value_transform_policy"


@dataclass(frozen=True)
class TransformPolicy:
    """Declare the numerical domains used by interpolation and reduction."""

    mode: TransformMode
    interpolation_domain: TransformDomain
    reduction_domain: TransformDomain
    tensor_storage_domain: TransformDomain
    feature_storage_domain: TransformDomain


_TRANSFORM_POLICIES: dict[TransformMode, TransformPolicy] = {
    "dB": TransformPolicy(
        mode="dB",
        interpolation_domain="transformed",
        reduction_domain="transformed",
        tensor_storage_domain="native",
        feature_storage_domain="transformed",
    ),
    "log": TransformPolicy(
        mode="log",
        interpolation_domain="transformed",
        reduction_domain="transformed",
        tensor_storage_domain="native",
        feature_storage_domain="transformed",
    ),
    "fisherz": TransformPolicy(
        mode="fisherz",
        interpolation_domain="native",
        reduction_domain="native",
        tensor_storage_domain="native",
        feature_storage_domain="transformed",
    ),
    "fisherz_sqrt": TransformPolicy(
        mode="fisherz_sqrt",
        interpolation_domain="native",
        reduction_domain="native",
        tensor_storage_domain="native",
        feature_storage_domain="transformed",
    ),
    "logit": TransformPolicy(
        mode="logit",
        interpolation_domain="native",
        reduction_domain="native",
        tensor_storage_domain="native",
        feature_storage_domain="transformed",
    ),
    "asinh": TransformPolicy(
        mode="asinh",
        interpolation_domain="native",
        reduction_domain="native",
        tensor_storage_domain="native",
        feature_storage_domain="transformed",
    ),
    "none": TransformPolicy(
        mode="none",
        interpolation_domain="native",
        reduction_domain="native",
        tensor_storage_domain="native",
        feature_storage_domain="native",
    ),
    None: TransformPolicy(
        mode=None,
        interpolation_domain="native",
        reduction_domain="native",
        tensor_storage_domain="native",
        feature_storage_domain="native",
    ),
}


def get_transform_policy(mode: TransformMode) -> TransformPolicy:
    """Return the execution policy declared for one transform mode."""

    try:
        return _TRANSFORM_POLICIES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported transform mode: {mode}") from exc


def transform_policy_metadata(policy: TransformPolicy) -> dict[str, str | None]:
    """Serialize a transform policy into plain metadata values."""

    return dict(asdict(policy))


def attach_transform_policy(
    metadata: Mapping[str, Any] | None,
    policy: TransformPolicy,
) -> dict[str, Any]:
    """Return metadata carrying a serialized value-transform policy."""

    output = dict(metadata or {})
    output[VALUE_TRANSFORM_POLICY_KEY] = transform_policy_metadata(policy)
    return output


def _parse_transform_domain(value: Any, *, field: str) -> TransformDomain:
    """Validate one serialized transform-domain field."""

    if value not in {"native", "transformed"}:
        raise ValueError(
            f"Invalid transform policy {field}: expected 'native' or "
            f"'transformed', got {value!r}."
        )
    return cast(TransformDomain, value)


def transform_policy_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> TransformPolicy:
    """Read a transform policy, defaulting legacy metadata to native identity."""

    payload = (
        metadata.get(VALUE_TRANSFORM_POLICY_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(payload, Mapping):
        return get_transform_policy("none")
    mode = payload.get("mode", "none")
    if mode not in _TRANSFORM_POLICIES:
        raise ValueError(f"Unsupported transform mode in metadata: {mode!r}")
    policy = get_transform_policy(cast(TransformMode, mode))
    return TransformPolicy(
        mode=policy.mode,
        interpolation_domain=_parse_transform_domain(
            payload.get("interpolation_domain", policy.interpolation_domain),
            field="interpolation_domain",
        ),
        reduction_domain=_parse_transform_domain(
            payload.get("reduction_domain", policy.reduction_domain),
            field="reduction_domain",
        ),
        tensor_storage_domain=_parse_transform_domain(
            payload.get("tensor_storage_domain", policy.tensor_storage_domain),
            field="tensor_storage_domain",
        ),
        feature_storage_domain=_parse_transform_domain(
            payload.get("feature_storage_domain", policy.feature_storage_domain),
            field="feature_storage_domain",
        ),
    )


def get_transform_pair(
    mode: TransformMode,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return (forward, inverse) callables for a transform mode.

    Notes
    -----
    - Forward transform first checks the mathematical domain.
      Values outside the domain are set to NaN.
    """
    if mode is None or mode == "none":
        return (lambda x: x, lambda x: x)

    if mode == "dB":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            valid = np.isfinite(x) & (x > 0.0)
            if np.any(valid):
                out[valid] = 10.0 * np.log10(x[valid])
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.power(10.0, y / 10.0)

        return forward, inverse

    if mode == "log":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            valid = np.isfinite(x) & (x > 0.0)
            if np.any(valid):
                out[valid] = np.log(x[valid])
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.exp(y)

        return forward, inverse

    if mode == "fisherz":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            valid = np.isfinite(x) & (x > -1.0) & (x < 1.0)
            if np.any(valid):
                out[valid] = np.arctanh(x[valid])
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.tanh(y)

        return forward, inverse

    if mode == "fisherz_sqrt":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            # Domain: 0 <= x < 1
            valid = np.isfinite(x) & (x >= 0.0) & (x < 1.0)
            if np.any(valid):
                out[valid] = np.arctanh(np.sqrt(x[valid]))
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.square(np.tanh(y))

        return forward, inverse

    if mode == "logit":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)
            # Domain: 0 < x < 1
            valid = np.isfinite(x) & (x > 0.0) & (x < 1.0)
            if np.any(valid):
                out[valid] = np.log(x[valid] / (1.0 - x[valid]))
            return out

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            # Stable sigmoid (kept as-is; may warn on extreme values but returns finite 0/1)
            return 1.0 / (1.0 + np.exp(-y))

        return forward, inverse

    if mode == "asinh":

        def forward(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return np.arcsinh(x)

        def inverse(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            return np.sinh(y)

        return forward, inverse

    raise ValueError(f"Unsupported transform mode: {mode}")


def apply_transform_array(
    arr: np.ndarray,
    *,
    mode: TransformMode,
) -> np.ndarray:
    """Apply a forward transform to a NumPy array.

    This function always returns a floating array.
    """
    forward, _ = get_transform_pair(mode)
    x = np.asarray(arr, dtype=float)
    return forward(x)


def apply_inverse_transform_array(
    arr: np.ndarray,
    *,
    mode: TransformMode,
) -> np.ndarray:
    """Apply an inverse transform to a NumPy array.

    This function always returns a floating array.
    """
    _, inverse = get_transform_pair(mode)
    y = np.asarray(arr, dtype=float)
    return inverse(y)


def convert_transform_domain_array(
    arr: np.ndarray,
    *,
    mode: TransformMode,
    source_domain: TransformDomain,
    target_domain: TransformDomain,
) -> np.ndarray:
    """Convert values between a transform's native and transformed domains."""

    values = np.asarray(arr, dtype=float)
    if source_domain == target_domain:
        return values
    if source_domain == "native":
        return apply_transform_array(values, mode=mode)
    return apply_inverse_transform_array(values, mode=mode)
