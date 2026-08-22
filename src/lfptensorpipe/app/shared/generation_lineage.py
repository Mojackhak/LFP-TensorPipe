"""Accepted-result generation receipt primitives."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping
from uuid import uuid4

GENERATION_RECEIPT_SEMANTICS_KEY = "generation_receipt_semantics"
GENERATION_RECEIPT_SEMANTICS = "exact_direct_accepted_generation_ids"
RESULT_GENERATION_ID_KEY = "result_generation_id"
INPUT_GENERATIONS_KEY = "input_generations"

_GENERATION_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_LINEAGE_KEYS = {
    GENERATION_RECEIPT_SEMANTICS_KEY,
    RESULT_GENERATION_ID_KEY,
    INPUT_GENERATIONS_KEY,
}


@dataclass(frozen=True)
class GenerationLineage:
    """Parsed lineage fields for one run-log event."""

    result_generation_id: str | None
    input_generations: dict[str, str]


def new_result_generation_id() -> str:
    """Return one opaque ID for an accepted artifact transaction."""
    return uuid4().hex


def _is_generation_id(value: Any) -> bool:
    return isinstance(value, str) and _GENERATION_ID_RE.fullmatch(value) is not None


def _normalized_input_generations(
    value: Mapping[str, str | None],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_ref, generation_id in value.items():
        ref = str(raw_ref).strip()
        if not ref:
            raise ValueError("Generation producer references must be non-empty.")
        if generation_id is None:
            continue
        if not _is_generation_id(generation_id):
            raise ValueError(f"Invalid generation ID for producer {ref!r}.")
        normalized[ref] = generation_id
    return normalized


def params_with_generation_lineage(
    params: Mapping[str, Any] | None,
    *,
    result_generation_id: str,
    input_generations: Mapping[str, str | None],
) -> dict[str, Any]:
    """Return params carrying one accepted result and its direct receipts."""
    if not _is_generation_id(result_generation_id):
        raise ValueError("result_generation_id must be a lowercase UUID hex string.")
    payload = dict(params or {})
    payload[GENERATION_RECEIPT_SEMANTICS_KEY] = GENERATION_RECEIPT_SEMANTICS
    payload[RESULT_GENERATION_ID_KEY] = result_generation_id
    payload[INPUT_GENERATIONS_KEY] = _normalized_input_generations(input_generations)
    return payload


def params_with_input_generation_receipts(
    params: Mapping[str, Any] | None,
    *,
    input_generations: Mapping[str, str | None],
) -> dict[str, Any]:
    """Return pending-state params carrying direct receipts but no result ID."""
    payload = dict(params or {})
    payload[GENERATION_RECEIPT_SEMANTICS_KEY] = GENERATION_RECEIPT_SEMANTICS
    payload.pop(RESULT_GENERATION_ID_KEY, None)
    payload[INPUT_GENERATIONS_KEY] = _normalized_input_generations(input_generations)
    return payload


def parse_generation_lineage(
    entry: Mapping[str, Any] | None,
    *,
    require_result_generation: bool = True,
) -> GenerationLineage | None:
    """Parse exact generation lineage; return None when absent or malformed."""
    if not isinstance(entry, Mapping):
        return None
    params = entry.get("params")
    if not isinstance(params, Mapping):
        return None
    present = _LINEAGE_KEYS.intersection(params)
    if not present:
        return None
    if params.get(GENERATION_RECEIPT_SEMANTICS_KEY) != (GENERATION_RECEIPT_SEMANTICS):
        return None
    raw_inputs = params.get(INPUT_GENERATIONS_KEY)
    if not isinstance(raw_inputs, Mapping):
        return None
    try:
        input_generations = _normalized_input_generations(raw_inputs)
    except (TypeError, ValueError):
        return None
    result_generation_id = params.get(RESULT_GENERATION_ID_KEY)
    if require_result_generation:
        if not _is_generation_id(result_generation_id):
            return None
    elif result_generation_id is not None and not _is_generation_id(
        result_generation_id
    ):
        return None
    return GenerationLineage(
        result_generation_id=(
            result_generation_id if isinstance(result_generation_id, str) else None
        ),
        input_generations=input_generations,
    )


def accepted_result_generation_id(
    entry: Mapping[str, Any] | None,
) -> str | None:
    """Return the generation ID of one completed, current-format event."""
    if not isinstance(entry, Mapping) or entry.get("completed") is not True:
        return None
    lineage = parse_generation_lineage(entry)
    if lineage is None:
        return None
    return lineage.result_generation_id


def input_generation_receipts_match(
    entry: Mapping[str, Any] | None,
    *,
    expected: Mapping[str, str | None],
    require_result_generation: bool = True,
) -> bool:
    """Return whether one event carries the exact expected receipts."""
    lineage = parse_generation_lineage(
        entry,
        require_result_generation=require_result_generation,
    )
    if lineage is None:
        return False
    try:
        expected_present = _normalized_input_generations(expected)
    except (TypeError, ValueError):
        return False
    return lineage.input_generations == expected_present


def preproc_generation_ref(step: str) -> str:
    """Return the stable producer reference for one Preprocess step."""
    return f"preproc/{str(step).strip()}"


def tensor_generation_ref(metric_key: str) -> str:
    """Return the stable producer reference for one persisted Tensor output."""
    return f"tensor/{str(metric_key).strip()}"


def alignment_generation_ref(trial_slug: str, step: str) -> str:
    """Return the stable producer reference for one Alignment event."""
    return f"alignment/{str(trial_slug).strip()}/{str(step).strip()}"


LOCALIZE_GENERATION_REF = "localize/apply"


__all__ = [
    "GENERATION_RECEIPT_SEMANTICS",
    "GENERATION_RECEIPT_SEMANTICS_KEY",
    "INPUT_GENERATIONS_KEY",
    "LOCALIZE_GENERATION_REF",
    "RESULT_GENERATION_ID_KEY",
    "GenerationLineage",
    "accepted_result_generation_id",
    "alignment_generation_ref",
    "input_generation_receipts_match",
    "new_result_generation_id",
    "params_with_generation_lineage",
    "params_with_input_generation_receipts",
    "parse_generation_lineage",
    "preproc_generation_ref",
    "tensor_generation_ref",
]
