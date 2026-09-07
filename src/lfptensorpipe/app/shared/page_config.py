"""Shared page-export JSON contract, without GUI dependencies."""

from __future__ import annotations

from typing import Any

PAGE_SCHEMAS = {
    "localize": ("lfptensorpipe.localize-config", (1,)),
    "tensor": ("lfptensorpipe.tensor-config", (3, 4)),
    "alignment": ("lfptensorpipe.alignment-config", (1,)),
    "features": ("lfptensorpipe.features-config", (1,)),
}


def page_config_node(payload: Any, page: str) -> dict[str, Any]:
    """Validate the envelope and return the requested page node."""
    if not isinstance(payload, dict):
        raise ValueError(f"{page.title()} config must be a JSON object.")
    schema, versions = PAGE_SCHEMAS[page]
    if payload.get("schema") != schema:
        raise ValueError(
            f"Unsupported {page} config schema: {payload.get('schema')!r}."
        )
    if type(payload.get("version")) is not int or payload["version"] not in versions:
        raise ValueError(
            f"Unsupported {page} config version: {payload.get('version')!r}."
        )
    node = payload.get(page)
    if not isinstance(node, dict):
        raise ValueError(f"{page.title()} config is missing required `{page}` object.")
    return node


def json_value(value: Any) -> Any:
    """Return plain values accepted by the existing page exporters."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_value(item_method())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    raise TypeError(f"Unsupported page config value: {type(value).__name__}")
