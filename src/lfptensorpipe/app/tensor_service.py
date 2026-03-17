"""Backward-compatible static import surface for tensor service APIs."""

from __future__ import annotations

from .tensor import service as _module
from .tensor.indicator import tensor_metric_panel_state

_MODULE_EXPORTS = getattr(
    _module,
    "__all__",
    [name for name in dir(_module) if not name.startswith("_")],
)
__all__ = list(_MODULE_EXPORTS)
if "tensor_metric_panel_state" not in __all__:
    __all__.append("tensor_metric_panel_state")


def __getattr__(name: str):
    if name == "tensor_metric_panel_state":
        return tensor_metric_panel_state
    return getattr(_module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_module)) | {"tensor_metric_panel_state"})
