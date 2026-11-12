"""Content fingerprints of the accepted Finish Raw.

Every Finish Apply mints a new generation ID, yet a downstream result reads
only part of Finish:

- Build Tensor reads the signal, the record structure, and the BAD/EDGE mask.
- Burst additionally reads its baseline annotation labels.
- Align Epochs reads the record structure and each trial's own labels.

The fingerprint describes those parts by content, so re-applying Finish after
an annotation edit keeps every result whose inputs did not change.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from lfptensorpipe.app.path_resolver import PathResolver
from lfptensorpipe.app.runlog_store import read_run_log
from lfptensorpipe.app.shared.generation_lineage import (
    accepted_result_generation_id,
    parse_generation_lineage,
    preproc_generation_ref,
)

logger = logging.getLogger(__name__)

FINISH_FINGERPRINT_KEY = "finish_fingerprint"
FINISH_FINGERPRINT_SEMANTICS_KEY = "semantics"
FINISH_FINGERPRINT_SEMANTICS = "signal_source_structure_mask_and_label_digests"
LEGACY_FINISH_BRIDGE_KEY = "legacy_finish_bridge"
TENSOR_MASK_PATTERNS = ("bad", "edge")

_TIME_QUANTA_PER_SECOND = 1_000_000_000
_LABEL_DIGEST_BYTES = 8
_GENERATION_ID_BYTES = 16

AnnotationRow = tuple[str, int, int, tuple[str, ...]]


def _digest(payload: Any, *, size: int) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def _quantize_seconds(value: Any) -> int:
    return int(round(float(value) * _TIME_QUANTA_PER_SECOND))


def _annotation_rows(raw: Any) -> list[AnnotationRow]:
    """Return `(label, onset, duration, channels)` in the record-relative frame."""
    from lfptensorpipe.io.timeline import raw_relative_onsets

    annotations = getattr(raw, "annotations", None)
    if annotations is None or len(annotations) == 0:
        return []
    rows: list[AnnotationRow] = []
    for onset, duration, description, ch_names in zip(
        raw_relative_onsets(raw),
        annotations.duration,
        annotations.description,
        annotations.ch_names,
    ):
        rows.append(
            (
                str(description),
                _quantize_seconds(onset),
                _quantize_seconds(duration),
                tuple(sorted(str(name) for name in ch_names)),
            )
        )
    return rows


def _label_matches(label: str, pattern: str, *, substring: bool) -> bool:
    # Consumers disagree on case and on stripping the description before an
    # exact match; accepting every variant keeps each label any of them reads.
    folded = label.lower()
    if substring:
        return pattern in folded
    return folded == pattern or folded.strip() == pattern


def _normalized_patterns(patterns: Iterable[Any]) -> list[str]:
    return sorted({str(item).strip().lower() for item in patterns if str(item).strip()})


def annotation_label_digests(raw: Any) -> dict[str, str]:
    """Return one digest of onsets, durations, and channel scopes per label.

    Labels keep their case: a consumer that matches case-sensitively must see
    a row move between `Cue` and `cue` as a change.
    """
    grouped: dict[str, list[tuple[int, int, tuple[str, ...]]]] = {}
    for label, onset, duration, channels in _annotation_rows(raw):
        grouped.setdefault(label, []).append((onset, duration, channels))
    return {
        label: _digest(sorted(rows), size=_LABEL_DIGEST_BYTES)
        for label, rows in grouped.items()
    }


def tensor_mask_digest(raw: Any) -> str:
    """Digest the BAD/EDGE support Build Tensor masks.

    Label text is excluded: every matching annotation masks the same way, so
    renaming `BAD_a` to `BAD_b` leaves Tensor results unchanged.
    """
    rows = {
        (onset, duration, channels)
        for label, onset, duration, channels in _annotation_rows(raw)
        if any(pattern in label.lower() for pattern in TENSOR_MASK_PATTERNS)
    }
    return _digest(sorted(rows), size=_GENERATION_ID_BYTES)


def record_structure_digest(raw: Any) -> str:
    """Digest the channel and sample layout of one Raw."""
    info = raw.info
    return _digest(
        {
            "ch_names": [str(name) for name in raw.ch_names],
            "ch_types": [str(item) for item in raw.get_channel_types()],
            "bads": sorted(str(name) for name in info["bads"]),
            "sfreq": float(info["sfreq"]).hex(),
            "n_times": int(raw.n_times),
            "first_samp": int(raw.first_samp),
        },
        size=_GENERATION_ID_BYTES,
    )


def annotation_dependency_digest(
    label_digests: Mapping[str, str],
    *,
    exact: Iterable[Any] = (),
    substring: Iterable[Any] = (),
    everything: bool = False,
    structure_digest: str | None = None,
) -> str:
    """Digest every annotation a consumer selects by label.

    `everything` selects all labels, for a consumer whose selection depends on
    the label set itself. `structure_digest` adds the record layout for a
    consumer that also reads sample times.
    """
    exact_patterns = _normalized_patterns(exact)
    substring_patterns = _normalized_patterns(substring)
    selected = {
        label: digest
        for label, digest in label_digests.items()
        if everything
        or any(
            _label_matches(label, pattern, substring=False)
            for pattern in exact_patterns
        )
        or any(
            _label_matches(label, pattern, substring=True)
            for pattern in substring_patterns
        )
    }
    return _digest(
        {
            "everything": bool(everything),
            "exact": exact_patterns,
            "substring": substring_patterns,
            "labels": sorted(selected.items()),
            "structure": structure_digest,
        },
        size=_GENERATION_ID_BYTES,
    )


def _tensor_input_generation_id(
    *,
    signal_source: str,
    structure_digest: str,
    mask_digest: str,
) -> str:
    return _digest(
        {
            "signal_source": signal_source,
            "structure": structure_digest,
            "mask": mask_digest,
        },
        size=_GENERATION_ID_BYTES,
    )


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_run_log(path)
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _step_log_path(resolver: PathResolver, step: str) -> Path:
    return resolver.preproc_step_dir(step, create=False) / "lfptensorpipe_log.json"


def _finish_raw_path(resolver: PathResolver) -> Path:
    return resolver.preproc_step_dir("finish", create=False) / "raw.fif"


def _label_digest_mapping(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        return None
    return dict(value)


def finish_fingerprint(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the fingerprint of one completed Finish event, when it has one.

    A fingerprint names the Finish generation it describes. One copied into a
    later event is ignored, so that event falls back to exact generation
    matching instead of vouching for a Raw it never described.
    """
    if not isinstance(payload, Mapping) or payload.get("completed") is not True:
        return None
    generation_id = accepted_result_generation_id(payload)
    params = payload.get("params")
    fingerprint = (
        params.get(FINISH_FINGERPRINT_KEY) if isinstance(params, Mapping) else None
    )
    if (
        not isinstance(fingerprint, Mapping)
        or fingerprint.get(FINISH_FINGERPRINT_SEMANTICS_KEY)
        != FINISH_FINGERPRINT_SEMANTICS
        or generation_id is None
        or fingerprint.get("finish_generation_id") != generation_id
        or not isinstance(fingerprint.get("tensor_input_generation_id"), str)
        or not isinstance(fingerprint.get("record_structure_digest"), str)
        or not isinstance(fingerprint.get("tensor_mask_digest"), str)
        or _label_digest_mapping(fingerprint.get("annotation_label_digests")) is None
    ):
        return None
    return dict(fingerprint)


def _valid_bridge(value: Any) -> dict[str, Any] | None:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("finish_generation_id"), str)
        or not isinstance(value.get("source_step"), str)
        or not isinstance(value.get("record_structure_digest"), str)
        or not isinstance(value.get("tensor_mask_digest"), str)
        or _label_digest_mapping(value.get("annotation_label_digests")) is None
    ):
        return None
    return dict(value)


def finish_legacy_bridge(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the pre-fingerprint Finish this event is known to continue.

    Results built before fingerprints existed hold that Finish's generation ID.
    The bridge records what it contained, so they can still be compared.
    """
    if not isinstance(payload, Mapping):
        return None
    params = payload.get("params")
    if not isinstance(params, Mapping):
        return None
    return _valid_bridge(params.get(LEGACY_FINISH_BRIDGE_KEY))


def finish_tensor_input_receipts(
    payload: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return the Finish receipts a current Tensor result may hold.

    The first one is canonical and is what a new Build run records.
    """
    fingerprint = finish_fingerprint(payload)
    if fingerprint is None:
        generation_id = accepted_result_generation_id(payload)
        return () if generation_id is None else (generation_id,)
    receipts = [str(fingerprint["tensor_input_generation_id"])]
    bridge = finish_legacy_bridge(payload)
    if (
        bridge is not None
        and bridge["record_structure_digest"] == fingerprint["record_structure_digest"]
        and bridge["tensor_mask_digest"] == fingerprint["tensor_mask_digest"]
    ):
        receipts.append(str(bridge["finish_generation_id"]))
    return tuple(receipts)


def finish_current_annotation_view(
    payload: Mapping[str, Any] | None,
) -> tuple[str, dict[str, str]] | None:
    """Return `(structure digest, label digests)` of the current Finish."""
    fingerprint = finish_fingerprint(payload)
    if fingerprint is None:
        return None
    return (
        str(fingerprint["record_structure_digest"]),
        dict(fingerprint["annotation_label_digests"]),
    )


def _signal_source(
    resolver: PathResolver,
    *,
    source_step: str,
    source_generation_id: str | None,
) -> str | None:
    """Identify the samples Finish copies, ignoring annotation-only rewrites."""
    if source_generation_id is None:
        return None
    if source_step != "annotations":
        return f"{source_step}:{source_generation_id}"
    payload = _read_payload(_step_log_path(resolver, "annotations"))
    if accepted_result_generation_id(payload) != source_generation_id:
        return None
    params = payload.get("params") if payload is not None else None
    ancestor_step = params.get("source_step") if isinstance(params, Mapping) else None
    lineage = parse_generation_lineage(payload)
    if not isinstance(ancestor_step, str) or lineage is None:
        return None
    ancestor_generation_id = lineage.input_generations.get(
        preproc_generation_ref(ancestor_step)
    )
    if ancestor_generation_id is None:
        return None
    return f"annotations<-{ancestor_step}:{ancestor_generation_id}"


def _read_raw_header(
    raw_path: Path,
    read_raw_fif_fn: Callable[..., Any] | None,
) -> Any:
    if read_raw_fif_fn is None:
        import mne

        read_raw_fif_fn = mne.io.read_raw_fif
    return read_raw_fif_fn(str(raw_path), preload=False, verbose="ERROR")


def _raw_content_digests(
    raw_path: Path,
    read_raw_fif_fn: Callable[..., Any] | None,
) -> dict[str, Any]:
    raw = _read_raw_header(raw_path, read_raw_fif_fn)
    try:
        return {
            "record_structure_digest": record_structure_digest(raw),
            "tensor_mask_digest": tensor_mask_digest(raw),
            "annotation_label_digests": annotation_label_digests(raw),
        }
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()


def build_finish_fingerprint(
    resolver: PathResolver,
    *,
    source_step: str,
    input_generations: Mapping[str, str | None],
    result_generation_id: str,
    raw_path: Path,
    read_raw_fif_fn: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """Fingerprint one just-written Finish Raw; `None` when it cannot be read.

    Without a fingerprint Finish still works: downstream results then match
    its generation ID exactly, as they did before fingerprints existed.
    """
    try:
        signal_source = _signal_source(
            resolver,
            source_step=source_step,
            source_generation_id=input_generations.get(
                preproc_generation_ref(source_step)
            ),
        )
        if signal_source is None:
            return None
        digests = _raw_content_digests(raw_path, read_raw_fif_fn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fingerprint Finish Raw %s: %s", raw_path, exc)
        return None
    return {
        FINISH_FINGERPRINT_SEMANTICS_KEY: FINISH_FINGERPRINT_SEMANTICS,
        "finish_generation_id": result_generation_id,
        "signal_source": signal_source,
        **digests,
        "tensor_input_generation_id": _tensor_input_generation_id(
            signal_source=signal_source,
            structure_digest=digests["record_structure_digest"],
            mask_digest=digests["tensor_mask_digest"],
        ),
    }


def carried_legacy_finish_bridge(
    resolver: PathResolver,
    *,
    read_raw_fif_fn: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """Return the bridge the next Finish event should carry, read before it.

    Call this only while the samples behind Finish are known to be unchanged:
    before an annotation-only invalidation, or before Finish is re-applied.
    A completed pre-fingerprint Finish is described from its Raw on disk; any
    later event carries the bridge it already holds.
    """
    payload = _read_payload(_step_log_path(resolver, "finish"))
    if payload is None:
        return None
    if payload.get("completed") is not True or finish_fingerprint(payload) is not None:
        return finish_legacy_bridge(payload)
    generation_id = accepted_result_generation_id(payload)
    params = payload.get("params")
    source_step = params.get("source_step") if isinstance(params, Mapping) else None
    raw_path = _finish_raw_path(resolver)
    if (
        generation_id is None
        or not isinstance(source_step, str)
        or not raw_path.is_file()
    ):
        return None
    try:
        digests = _raw_content_digests(raw_path, read_raw_fif_fn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not describe legacy Finish Raw %s: %s", raw_path, exc)
        return None
    return {
        "finish_generation_id": generation_id,
        "source_step": source_step,
        **digests,
    }


def burst_baseline_annotation_digest(
    resolver: PathResolver,
    baseline_keep: Sequence[Any] | None,
) -> str | None:
    """Digest the current Finish annotations one Burst baseline selects."""
    if not baseline_keep:
        return None
    view = finish_current_annotation_view(
        _read_payload(_step_log_path(resolver, "finish"))
    )
    if view is None:
        return None
    return annotation_dependency_digest(view[1], exact=baseline_keep)


__all__ = [
    "FINISH_FINGERPRINT_KEY",
    "LEGACY_FINISH_BRIDGE_KEY",
    "TENSOR_MASK_PATTERNS",
    "annotation_dependency_digest",
    "annotation_label_digests",
    "build_finish_fingerprint",
    "burst_baseline_annotation_digest",
    "carried_legacy_finish_bridge",
    "finish_current_annotation_view",
    "finish_fingerprint",
    "finish_legacy_bridge",
    "finish_tensor_input_receipts",
    "record_structure_digest",
    "tensor_mask_digest",
]
