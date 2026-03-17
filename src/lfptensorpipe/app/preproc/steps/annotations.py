"""Annotations-step helpers for preprocess stage."""

from __future__ import annotations

from pathlib import Path
import csv
import shutil
from typing import Any, Callable

from lfptensorpipe.app.path_resolver import PathResolver, RecordContext

from ..paths import preproc_step_raw_path, write_preproc_step_config

MarkStepFn = Callable[..., Any]
InvalidateFn = Callable[[RecordContext, str], list[Any]]


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
        if not description or onset < 0.0 or duration < 0.0:
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


def apply_annotations_step(
    context: RecordContext,
    *,
    rows: list[dict[str, Any]],
    mark_preproc_step_fn: MarkStepFn,
    invalidate_downstream_fn: InvalidateFn,
    read_raw_fif_fn: Callable[..., Any] | None = None,
    copy2_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str]:
    """Apply annotations step by writing annotations onto filtered raw."""
    resolver = PathResolver(context)
    src = preproc_step_raw_path(resolver, "filter")
    dst = preproc_step_raw_path(resolver, "annotations")
    csv_path = resolver.preproc_step_dir("annotations", create=True) / "annotations.csv"

    if not src.exists():
        mark_preproc_step_fn(
            resolver=resolver,
            step="annotations",
            completed=False,
            input_path=str(src),
            output_path=str(dst),
            message="Missing filter raw input for annotations step.",
        )
        return False, "Missing filter raw input for annotations step."

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
        return False, f"Invalid annotation rows: {invalid_rows}"

    try:
        if read_raw_fif_fn is None:
            import mne

            read_raw_fif_fn = mne.io.read_raw_fif
        runtime_copy2 = copy2_fn or shutil.copy2

        dst.parent.mkdir(parents=True, exist_ok=True)
        # Explicitly inherit the latest filter artifact before mutating annotations.
        runtime_copy2(src, dst)
        raw = read_raw_fif_fn(str(dst), preload=True, verbose="ERROR")
        annotations = mne.Annotations(
            onset=[float(item["onset"]) for item in normalized_rows],
            duration=[float(item["duration"]) for item in normalized_rows],
            description=[str(item["description"]) for item in normalized_rows],
            orig_time=raw.annotations.orig_time,
        )
        inherited_annotations = raw.annotations.copy()
        raw.set_annotations(inherited_annotations + annotations)

        raw.save(str(dst), overwrite=True)

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["description", "onset", "duration"])
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
            config={
                "row_count": len(normalized_rows),
                "csv_path": str(csv_path),
            },
        )
        mark_preproc_step_fn(
            resolver=resolver,
            step="annotations",
            completed=True,
            params={"row_count": len(normalized_rows)},
            input_path=str(src),
            output_path=str(dst),
            message="Annotations step completed.",
        )
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
        return False, f"Annotations step failed: {exc}"

    return True, "Annotations step completed."
