"""Annotations-step helpers for preprocess stage."""

from __future__ import annotations

from pathlib import Path
import csv
import math
import shutil
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.shared.atomic_outputs import AtomicOutputSet

from ..paths import (
    preproc_step_config_path,
    preproc_step_log_path,
    preproc_step_raw_path,
    write_preproc_step_config,
)

MarkStepFn = Callable[..., Any]
InvalidateFn = Callable[[RecordContext, str], list[Any]]

ANNOTATION_SUPPORT_SEMANTICS = "fully_within_half_open_source_support"
_ANNOTATION_SUPPORT_SEMANTICS_KEY = "annotation_support_semantics"


def annotation_log_has_current_support_semantics(payload: Any) -> bool:
    """Return whether one Annotations log uses the current support contract."""
    if not isinstance(payload, dict):
        return False
    params = payload.get("params")
    return bool(
        isinstance(params, dict)
        and params.get(_ANNOTATION_SUPPORT_SEMANTICS_KEY)
        == ANNOTATION_SUPPORT_SEMANTICS
    )


def load_annotations_csv_rows(csv_path: Path) -> tuple[bool, list[dict[str, Any]], str]:
    """Load annotations rows from csv header `description,onset,duration`."""
    if not csv_path.exists():
        return False, [], f"CSV file does not exist: {csv_path}"

    rows: list[dict[str, Any]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"description", "onset", "duration"}
            fieldnames = set(reader.fieldnames or [])
            if not required.issubset(fieldnames):
                return (
                    False,
                    [],
                    "CSV header must contain description,onset,duration.",
                )
            for row in reader:
                rows.append(
                    {
                        "description": str(row.get("description", "")).strip(),
                        "onset": str(row.get("onset", "")).strip(),
                        "duration": str(row.get("duration", "")).strip(),
                    }
                )
    except Exception as exc:
        return False, [], f"Failed to read CSV: {exc}"

    _, invalid_rows = _normalize_annotation_rows(rows)
    if invalid_rows:
        return False, [], f"CSV contains invalid rows: {invalid_rows}"

    return True, rows, "CSV loaded."


def _normalize_annotation_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    normalized: list[dict[str, Any]] = []
    invalid_rows: list[int] = []
    for idx, row in enumerate(rows):
        description = str(row.get("description", "")).strip()
        onset_raw = row.get("onset", "")
        duration_raw = row.get("duration", "")
        try:
            onset = float(onset_raw)
            duration = float(duration_raw)
        except Exception:
            invalid_rows.append(idx)
            continue
        if (
            not description
            or not math.isfinite(onset)
            or not math.isfinite(duration)
            or onset < 0.0
            or duration < 0.0
        ):
            invalid_rows.append(idx)
            continue
        normalized.append(
            {
                "description": description,
                "onset": onset,
                "duration": duration,
            }
        )
    normalized.sort(key=lambda item: float(item["onset"]))
    return normalized, invalid_rows


def _annotation_rows_outside_raw_support(
    rows: list[dict[str, Any]],
    raw: Any,
) -> list[int]:
    """Return original row indices outside the Raw-relative half-open support."""
    support_stop = float(raw.n_times) / float(raw.info["sfreq"])
    positive_stop_limit = math.nextafter(support_stop, math.inf)
    invalid_rows: list[int] = []
    for idx, row in enumerate(rows):
        onset = float(row["onset"])
        duration = float(row["duration"])
        if onset >= support_stop or (
            duration > 0.0 and onset + duration > positive_stop_limit
        ):
            invalid_rows.append(idx)
    return invalid_rows


def apply_annotations_step(
    context: RecordContext,
    *,
    source: tuple[str, Path] | None,
    rows: list[dict[str, Any]],
    mark_preproc_step_fn: MarkStepFn,
    invalidate_downstream_fn: InvalidateFn,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    copy2_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Apply annotations onto the nearest valid preceding artifact."""
    resolver = PathResolver(context)
    dst = preproc_step_raw_path(resolver, "annotations")
    csv_path = resolver.preproc_step_dir("annotations", create=True) / "annotations.csv"

    if source is None:
        mark_preproc_step_fn(
            resolver=resolver,
            step="annotations",
            completed=False,
            input_path="",
            output_path=str(dst),
            message="No valid preprocess input for annotations step.",
        )
        invalidate_downstream_fn(context, "annotations")
        return False, "No valid preprocess input for annotations step."

    source_step, src = source

    normalized_rows, invalid_rows = _normalize_annotation_rows(rows)
    if invalid_rows:
        mark_preproc_step_fn(
            resolver=resolver,
            step="annotations",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Invalid annotation rows: {invalid_rows}",
        )
        invalidate_downstream_fn(context, "annotations")
        return False, f"Invalid annotation rows: {invalid_rows}"

    try:
        import mne

        if read_raw_fif_fn is None:
            read_raw_fif_fn = mne.io.read_raw_fif
        runtime_copy2 = copy2_fn or shutil.copy2

        config_path = preproc_step_config_path(resolver, "annotations")
        log_path = preproc_step_log_path(resolver, "annotations")
        with AtomicOutputSet(
            [dst, csv_path, config_path, log_path],
            cleanup_stale_residues=True,
        ) as output_set:
            staged_raw = output_set.staged_path(dst)
            # Preserve the exact source file before applying annotations.
            runtime_copy2(src, staged_raw)
            raw = read_raw_fif_fn(str(staged_raw), preload=True, verbose="ERROR")
            outside_rows = _annotation_rows_outside_raw_support(rows, raw)
            if outside_rows:
                raise ValueError(f"Annotation rows outside Raw support: {outside_rows}")
            annotations = mne.Annotations(
                onset=[float(item["onset"]) for item in normalized_rows],
                duration=[float(item["duration"]) for item in normalized_rows],
                description=[str(item["description"]) for item in normalized_rows],
                orig_time=raw.annotations.orig_time,
            )
            inherited_annotations = raw.annotations.copy()
            raw.set_annotations(inherited_annotations + annotations)

            raw.save(str(staged_raw), overwrite=True)

            with output_set.staged_path(csv_path).open(
                "w",
                encoding="utf-8",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["description", "onset", "duration"],
                )
                writer.writeheader()
                for item in normalized_rows:
                    writer.writerow(
                        {
                            "description": str(item["description"]),
                            "onset": float(item["onset"]),
                            "duration": float(item["duration"]),
                        }
                    )

            write_preproc_step_config(
                resolver=resolver,
                step="annotations",
                path=output_set.staged_path(config_path),
                config={
                    "row_count": len(normalized_rows),
                    "csv_path": str(csv_path),
                    _ANNOTATION_SUPPORT_SEMANTICS_KEY: (ANNOTATION_SUPPORT_SEMANTICS),
                },
            )
            mark_preproc_step_fn(
                resolver=resolver,
                step="annotations",
                completed=True,
                params={
                    "row_count": len(normalized_rows),
                    _ANNOTATION_SUPPORT_SEMANTICS_KEY: (ANNOTATION_SUPPORT_SEMANTICS),
                },
                input_path=str(src),
                output_path=str(dst),
                message=f"Annotations step completed using source: {source_step}.",
                log_path=output_set.staged_path(log_path),
            )
            output_set.commit()
        invalidate_downstream_fn(context, "annotations")
    except Exception as exc:
        mark_preproc_step_fn(
            resolver=resolver,
            step="annotations",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message=f"Annotations step failed: {exc}",
        )
        invalidate_downstream_fn(context, "annotations")
        return False, f"Annotations step failed: {exc}"

    return True, "Annotations step completed."
