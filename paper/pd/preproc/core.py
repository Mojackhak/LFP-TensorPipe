"""Backend helpers for PD paper table preprocessing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lfptensorpipe.stats.preproc.normalize import baseline_normalize, normalize_df
from lfptensorpipe.stats.preproc.transform import transform_df
from lfptensorpipe.tabular.nested_value import cell_is_empty_or_all_nan
from paper.pd.paths import resolve_project_root, summary_table_root
from paper.pd.preproc.aggregate import summarize_df
from paper.pd.specs import (
    DEFAULT_SCALAR_NORMALIZE_SPEC,
    NORMALIZED_SUFFIX,
    PASSTHROUGH_FILE_SUFFIXES,
    SCALAR_NORMALIZE_GROUP_COLS,
    SHIFT_ENABLED_NAMES,
    SHIFT_FILE_NAMES,
    SHIFT_SUFFIX,
    SUMMARY_GROUP_COLS,
    SUMMARY_SUFFIX,
    TRACE_LIKE_FILE_NAMES,
    TRANSFORM_SUFFIX,
    TransformModeCfg,
)
from paper.pd.table_io import is_scalar_table_name, save_table_outputs
from lfptensorpipe.io.pkl_io import load_pkl


@dataclass(slots=True)
class PreprocReport:
    """Structured preprocessing summary for interactive review."""

    project_root: Path
    raw_sources: int = 0
    summarized_outputs: list[Path] = field(default_factory=list)
    transformed_outputs: list[Path] = field(default_factory=list)
    normalized_outputs: list[Path] = field(default_factory=list)
    shifted_outputs: list[Path] = field(default_factory=list)
    passthrough_outputs: list[Path] = field(default_factory=list)
    skipped_normalization: list[str] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        """Return a compact numeric summary."""
        return {
            "raw_sources": self.raw_sources,
            "summarized_outputs": len(self.summarized_outputs),
            "transformed_outputs": len(self.transformed_outputs),
            "normalized_outputs": len(self.normalized_outputs),
            "shifted_outputs": len(self.shifted_outputs),
            "passthrough_outputs": len(self.passthrough_outputs),
            "skipped_normalization": len(self.skipped_normalization),
            "load_errors": len(self.load_errors),
        }


def _name_filter(table_names: Sequence[str] | None) -> set[str] | None:
    if table_names is None:
        return None
    return {str(name).strip() for name in table_names if str(name).strip()}


def _is_derived_path(path: Path) -> bool:
    stem = path.stem
    return (
        stem.endswith(SUMMARY_SUFFIX)
        or stem.endswith(TRANSFORM_SUFFIX)
        or stem.endswith(NORMALIZED_SUFFIX)
        or stem.endswith(SHIFT_SUFFIX)
    )


def _strip_file_suffix(name: str, suffix: str) -> str:
    stem = Path(name).stem
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return f"{stem}{Path(name).suffix}"


def _base_file_name(path: Path) -> str:
    name = path.name
    name = _strip_file_suffix(name, NORMALIZED_SUFFIX)
    name = _strip_file_suffix(name, TRANSFORM_SUFFIX)
    name = _strip_file_suffix(name, SUMMARY_SUFFIX)
    return name


def _summary_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}{SUMMARY_SUFFIX}{path.suffix}")


def _transform_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}{TRANSFORM_SUFFIX}{path.suffix}")


def _normalized_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}{NORMALIZED_SUFFIX}{path.suffix}")


def _shift_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}{SHIFT_SUFFIX}{path.suffix}")


def _relative_table_path(project_root: Path, path: Path) -> Path:
    return path.relative_to(summary_table_root(project_root))


def _metric_key(relative_path: Path) -> str:
    if len(relative_path.parts) < 3:
        raise ValueError(f"Expected at least name/metric/file layout, got {relative_path.as_posix()}.")
    return relative_path.parts[1]


def _reducer_key(file_name: str) -> str:
    return Path(file_name).stem.split("-", 1)[0]


def _resolve_transform_mode(
    relative_path: Path,
    transform_mode_cfg: TransformModeCfg,
) -> str:
    metric_key = _metric_key(relative_path)
    metric_cfg = transform_mode_cfg.get(metric_key, transform_mode_cfg.get("default", {}))
    fallback_cfg = transform_mode_cfg.get("default", {})
    reducer_key = _reducer_key(_base_file_name(relative_path))
    return str(
        metric_cfg.get(
            reducer_key,
            metric_cfg.get("default", fallback_cfg.get("default", "none")),
        )
    )


def _load_frame(path: Path) -> pd.DataFrame:
    payload = load_pkl(path)
    if not isinstance(payload, pd.DataFrame):
        raise TypeError(f"{path} does not contain a pandas.DataFrame payload.")
    return payload


def _is_passthrough_file(file_name: str) -> bool:
    return any(file_name.endswith(suffix) for suffix in PASSTHROUGH_FILE_SUFFIXES)


def _normalize_trace_like_table(
    frame: pd.DataFrame,
    *,
    baseline: Sequence[float] | Sequence[Sequence[float]],
    mode: str,
    slice_mode: str,
) -> pd.DataFrame:
    out = frame.copy()
    normalized_cells: list[Any] = []
    for cell in out["Value"].tolist():
        if cell_is_empty_or_all_nan(cell):
            normalized_cells.append(pd.NA)
            continue
        if isinstance(cell, (pd.Series, pd.DataFrame)):
            normalized_cells.append(
                baseline_normalize(
                    cell,
                    baseline=baseline,
                    mode=mode,
                    slice_mode=slice_mode,
                )
            )
            continue
        raise TypeError(f"x must be a pd.Series or pd.DataFrame, got {type(cell).__name__}")
    out["Value"] = normalized_cells
    return out


def _normalize_scalar_spectral_table(
    frame: pd.DataFrame,
    *,
    baseline: Mapping[str, Any],
) -> pd.DataFrame:
    return normalize_df(
        frame,
        group_cols=list(SCALAR_NORMALIZE_GROUP_COLS),
        baseline=baseline,
        value_col="Value",
        mode="mean",
        on_missing_baseline="drop",
    )


def _shift_cycle_trace_like_cell(cell: Any) -> Any:
    if cell_is_empty_or_all_nan(cell):
        return pd.NA
    if isinstance(cell, pd.Series):
        shift_n = len(cell.index) // 2
        if shift_n == 0:
            return cell.copy()
        values = np.roll(cell.to_numpy(copy=True), -shift_n)
        return pd.Series(values, index=cell.index.copy(), name=cell.name)
    if isinstance(cell, pd.DataFrame):
        shift_n = len(cell.columns) // 2
        if shift_n == 0:
            return cell.copy()
        values = np.roll(cell.to_numpy(copy=True), -shift_n, axis=1)
        return pd.DataFrame(
            values,
            index=cell.index.copy(),
            columns=cell.columns.copy(),
        )
    raise TypeError(f"x must be a pd.Series or pd.DataFrame, got {type(cell).__name__}")


def _shift_cycle_trace_like_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    shifted_values: list[Any] = []
    shifted_lat: list[Any] = []
    for lat, cell in zip(out["Lat"].tolist(), out["Value"].tolist(), strict=False):
        if lat == "Ipsi":
            shifted_values.append(_shift_cycle_trace_like_cell(cell))
            shifted_lat.append("Contra")
        else:
            shifted_values.append(cell)
            shifted_lat.append(lat)
    out["Value"] = shifted_values
    out["Lat"] = shifted_lat
    return out


def collect_preproc_sources(
    project_root: str | Path | None = None,
    *,
    table_names: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """Collect raw merged table sources eligible for preprocessing."""
    resolved_root = resolve_project_root(project_root)
    table_root = summary_table_root(resolved_root)
    selected_names = _name_filter(table_names)
    if not table_root.exists():
        return ()

    paths: list[Path] = []
    for path in sorted(table_root.glob("*/*.pkl")):
        # Keep compatibility with any shallow files, although current layout is deeper.
        rel = path.relative_to(table_root)
        if selected_names is not None and rel.parts[0] not in selected_names:
            continue
        if _is_derived_path(path):
            continue
        paths.append(path)

    for path in sorted(table_root.glob("*/*/*.pkl")):
        rel = path.relative_to(table_root)
        if selected_names is not None and rel.parts[0] not in selected_names:
            continue
        if _is_derived_path(path):
            continue
        paths.append(path)

    return tuple(sorted(set(paths)))


def preproc_source_frame(
    project_root: str | Path | None,
    sources: Iterable[Path],
) -> pd.DataFrame:
    """Return raw preprocessing sources as a compact review table."""
    resolved_root = resolve_project_root(project_root)
    rows: list[dict[str, str]] = []
    for path in sources:
        rel = _relative_table_path(resolved_root, path)
        rows.append(
            {
                "name": rel.parts[0],
                "metric": rel.parts[1] if len(rel.parts) > 2 else "",
                "file_name": path.name,
                "relative_path": rel.as_posix(),
            }
        )
    return pd.DataFrame(rows)


def export_summarized_tables(
    project_root: str | Path | None,
    *,
    table_names: Sequence[str] | None = None,
    report: PreprocReport | None = None,
) -> PreprocReport:
    """Summarize raw merged tables and write `_summary` outputs."""
    resolved_root = resolve_project_root(project_root)
    sources = collect_preproc_sources(resolved_root, table_names=table_names)
    current_report = report or PreprocReport(
        project_root=resolved_root,
        raw_sources=len(sources),
    )

    for path in sources:
        rel = _relative_table_path(resolved_root, path)
        try:
            frame = _load_frame(path)
            summarized = summarize_df(
                frame,
                group_cols=list(SUMMARY_GROUP_COLS),
                value_col="Value",
                mode="mean",
                drop_other_cols=False,
                align="reindex",
            )
        except Exception as exc:  # noqa: BLE001
            current_report.load_errors.append(
                f"summary:{rel.as_posix()}: {exc}"
            )
            continue

        out_path = _summary_output_path(path)
        save_table_outputs(
            summarized,
            out_path,
            export_xlsx=is_scalar_table_name(path.name),
        )
        current_report.summarized_outputs.append(out_path)

    return current_report


def export_transformed_tables(
    project_root: str | Path | None,
    *,
    transform_mode_cfg: TransformModeCfg,
    table_names: Sequence[str] | None = None,
    report: PreprocReport | None = None,
) -> PreprocReport:
    """Transform summarized tables and write `_summary_trans` outputs."""
    resolved_root = resolve_project_root(project_root)
    table_root = summary_table_root(resolved_root)
    selected_names = _name_filter(table_names)
    current_report = report or PreprocReport(project_root=resolved_root)

    sources = sorted(table_root.glob(f"*/*/*{SUMMARY_SUFFIX}.pkl"))

    for path in sources:
        rel = _relative_table_path(resolved_root, path)
        if selected_names is not None and rel.parts[0] not in selected_names:
            continue
        mode = _resolve_transform_mode(rel, transform_mode_cfg)
        try:
            frame = _load_frame(path)
            transformed = transform_df(frame, value_col="Value", mode=mode)
        except Exception as exc:  # noqa: BLE001
            current_report.load_errors.append(
                f"transform:{rel.as_posix()} ({mode}): {exc}"
            )
            continue
        out_path = _transform_output_path(path)
        save_table_outputs(
            transformed,
            out_path,
            export_xlsx=is_scalar_table_name(_base_file_name(path)),
        )
        current_report.transformed_outputs.append(out_path)

    return current_report


def export_normalized_tables(
    project_root: str | Path | None,
    *,
    normalize_spec: Mapping[str, Mapping[str, Any]],
    scalar_normalize_spec: Mapping[str, Mapping[str, Any]] = DEFAULT_SCALAR_NORMALIZE_SPEC,
    table_names: Sequence[str] | None = None,
    report: PreprocReport | None = None,
) -> PreprocReport:
    """Normalize transformed summary tables and write `_summary_trans_normalized` outputs."""
    resolved_root = resolve_project_root(project_root)
    table_root = summary_table_root(resolved_root)
    selected_names = _name_filter(table_names)
    current_report = report or PreprocReport(project_root=resolved_root)

    transformed_paths = sorted(table_root.glob(f"*/*/*{SUMMARY_SUFFIX}{TRANSFORM_SUFFIX}.pkl"))
    for path in transformed_paths:
        rel = path.relative_to(table_root)
        name = rel.parts[0]
        if selected_names is not None and name not in selected_names:
            continue

        file_name = _base_file_name(path)
        out_path = _normalized_output_path(path)

        try:
            frame = _load_frame(path)
            if file_name in TRACE_LIKE_FILE_NAMES:
                if name not in normalize_spec:
                    current_report.skipped_normalization.append(
                        f"{rel.as_posix()}: name '{name}' is not configured for trace normalization."
                    )
                    continue
                rule = normalize_spec[name]
                normalized = _normalize_trace_like_table(
                    frame,
                    baseline=rule["baseline"],
                    mode=str(rule.get("mode", "mean")),
                    slice_mode=str(rule.get("slice_mode", "percent")),
                )
                save_table_outputs(
                    normalized,
                    out_path,
                    export_xlsx=is_scalar_table_name(file_name),
                )
                current_report.normalized_outputs.append(out_path)
                continue

            if _is_passthrough_file(file_name):
                if name in scalar_normalize_spec:
                    normalized = _normalize_scalar_spectral_table(
                        frame,
                        baseline=scalar_normalize_spec[name]["baseline"],
                    )
                    save_table_outputs(
                        normalized,
                        out_path,
                        export_xlsx=is_scalar_table_name(file_name),
                    )
                    current_report.normalized_outputs.append(out_path)
                else:
                    save_table_outputs(
                        frame.copy(),
                        out_path,
                        export_xlsx=is_scalar_table_name(file_name),
                    )
                    current_report.passthrough_outputs.append(out_path)
                continue

            current_report.skipped_normalization.append(
                f"{rel.as_posix()}: unsupported file category '{file_name}'."
            )
        except Exception as exc:  # noqa: BLE001
            current_report.load_errors.append(
                f"normalize:{rel.as_posix()}: {exc}"
            )

    return current_report


def export_shifted_tables(
    project_root: str | Path | None,
    *,
    table_names: Sequence[str] | None = None,
    report: PreprocReport | None = None,
) -> PreprocReport:
    """Shift selected normalized trace-like tables and write `_shift` outputs."""
    resolved_root = resolve_project_root(project_root)
    table_root = summary_table_root(resolved_root)
    selected_names = _name_filter(table_names)
    current_report = report or PreprocReport(project_root=resolved_root)

    normalized_paths = sorted(
        table_root.glob(f"*/*/*{SUMMARY_SUFFIX}{TRANSFORM_SUFFIX}{NORMALIZED_SUFFIX}.pkl")
    )
    for path in normalized_paths:
        rel = path.relative_to(table_root)
        name = rel.parts[0]
        if selected_names is not None and name not in selected_names:
            continue

        file_name = _base_file_name(path)
        if name not in SHIFT_ENABLED_NAMES or file_name not in SHIFT_FILE_NAMES:
            continue

        try:
            frame = _load_frame(path)
            shifted = _shift_cycle_trace_like_table(frame)
        except Exception as exc:  # noqa: BLE001
            current_report.load_errors.append(
                f"shift:{rel.as_posix()}: {exc}"
            )
            continue

        out_path = _shift_output_path(path)
        save_table_outputs(
            shifted,
            out_path,
            export_xlsx=is_scalar_table_name(file_name),
        )
        current_report.shifted_outputs.append(out_path)

    return current_report


def export_preprocessed_tables(
    project_root: str | Path | None,
    *,
    transform_mode_cfg: TransformModeCfg,
    normalize_spec: Mapping[str, Mapping[str, Any]],
    scalar_normalize_spec: Mapping[str, Mapping[str, Any]] = DEFAULT_SCALAR_NORMALIZE_SPEC,
    table_names: Sequence[str] | None = None,
) -> PreprocReport:
    """Run the summary-then-transform-then-normalize-then-shift workflow."""
    resolved_root = resolve_project_root(project_root)
    report = export_summarized_tables(
        resolved_root,
        table_names=table_names,
        report=None,
    )
    export_transformed_tables(
        resolved_root,
        transform_mode_cfg=transform_mode_cfg,
        table_names=table_names,
        report=report,
    )
    export_normalized_tables(
        resolved_root,
        normalize_spec=normalize_spec,
        scalar_normalize_spec=scalar_normalize_spec,
        table_names=table_names,
        report=report,
    )
    export_shifted_tables(
        resolved_root,
        table_names=table_names,
        report=report,
    )
    return report
