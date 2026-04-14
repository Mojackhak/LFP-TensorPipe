"""Extract features runner logic."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lfptensorpipe.app.alignment_service import alignment_paradigm_log_path
from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import RunLogRecord, indicator_from_log, write_run_log
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl
from lfptensorpipe.tabular.grid import grid_nested_values, split_nested_values


@dataclass(frozen=True)
class _MetricExtractResult:
    """Result payload for one metric-level Extract Features task."""

    metric_key: str
    axes_signature: dict[str, list[dict[str, Any]]] | None
    saved: int
    total_targets: int
    errors: tuple[str, ...]
    xlsx_warnings: tuple[str, ...]


def _remove_stale_xlsx(path: Path) -> None:
    if path.exists():
        path.unlink()


def _should_export_xlsx(derived_type: str) -> bool:
    return derived_type == "scalar"


def _extract_metric_outputs(
    *,
    metric_key: str,
    src_path: Path,
    deriv_root: Path,
    derive_param_cfg: dict[str, Any],
    reducer_cfg: dict[str, Any],
    reducer_rule_by_method: dict[str, Any],
    collapse_base_cfg: dict[str, Any],
    alignment_method: str,
    axis_node: dict[str, Any] | None,
    override_outputs: dict[str, bool] | None,
    reducer_override: str,
) -> _MetricExtractResult:
    """Build every enabled output for one metric under its own subtree."""
    from . import service as svc

    try:
        payload = load_pkl(src_path)
        if not isinstance(payload, pd.DataFrame):
            raise ValueError("raw table payload must be a pandas.DataFrame")
    except Exception as exc:  # noqa: BLE001
        return _MetricExtractResult(
            metric_key=metric_key,
            axes_signature=None,
            saved=0,
            total_targets=0,
            errors=(f"{metric_key}: load failed ({exc})",),
            xlsx_warnings=(),
        )

    outputs = svc._resolve_enabled_outputs(derive_param_cfg, metric_key)
    reducer_list = svc._resolve_reducers(reducer_cfg, metric_key)
    if isinstance(override_outputs, dict):
        outputs = svc._normalize_enabled_outputs_map(override_outputs)
    reducers = [reducer_override] if reducer_override else list(reducer_list)
    method_rule_node = reducer_rule_by_method.get(metric_key, {})
    if not isinstance(method_rule_node, dict):
        method_rule_node = {}
    if alignment_method and alignment_method in method_rule_node:
        reducers = list(
            svc._normalize_reducer_list(method_rule_node.get(alignment_method))
        )

    axis_payload = axis_node if isinstance(axis_node, dict) else {}
    bands_rows = svc._normalize_axis_rows(
        axis_payload.get("bands"),
        allow_duplicate_names=False,
    )
    times_rows = svc._normalize_axis_rows(
        axis_payload.get("times"),
        allow_duplicate_names=True,
    )
    axes_signature = {
        "bands": (
            []
            if svc._metric_uses_auto_bands(metric_key)
            else [dict(item) for item in bands_rows]
        ),
        "times": [dict(item) for item in times_rows],
    }
    if svc._metric_uses_auto_bands(metric_key):
        freqs_axis = svc._bands_from_raw_value_index(payload)
    else:
        freqs_axis = svc._rows_to_interval_mapping(bands_rows)
    time_axis = svc._rows_to_interval_mapping(times_rows)

    metric_out_dir = deriv_root / metric_key
    metric_out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    total_targets = 0
    errors: list[str] = []
    xlsx_warnings: list[str] = []

    if outputs.get("raw", False):
        total_targets += 1
        out_pkl = metric_out_dir / "na-raw.pkl"
        out_xlsx = metric_out_dir / "na-raw.xlsx"
        try:
            _remove_stale_xlsx(out_xlsx)
            save_pkl(payload, out_pkl)
            saved += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{metric_key}/na-raw: {exc}")

    for reducer in reducers:
        for enabled_output in ("spectral", "trace", "scalar"):
            if not outputs.get(enabled_output, False):
                continue
            total_targets += 1
            stem = f"{reducer}-{enabled_output}"
            out_pkl = metric_out_dir / f"{stem}.pkl"
            out_xlsx = metric_out_dir / f"{stem}.xlsx"
            try:
                if not _should_export_xlsx(enabled_output):
                    _remove_stale_xlsx(out_xlsx)
                if enabled_output == "spectral":
                    if not time_axis:
                        raise ValueError("times axis is required for spectral.")
                    derived = split_nested_values(
                        payload,
                        bands=None,
                        times=time_axis,
                        axis="time",
                        reducer=reducer,
                        **collapse_base_cfg,
                    )
                elif enabled_output == "trace":
                    if not freqs_axis:
                        raise ValueError("bands axis is required for trace.")
                    derived = split_nested_values(
                        payload,
                        bands=freqs_axis,
                        times=None,
                        axis="freq",
                        reducer=reducer,
                        **collapse_base_cfg,
                    )
                else:
                    if not freqs_axis:
                        raise ValueError("bands axis is required for scalar.")
                    if not time_axis:
                        raise ValueError("times axis is required for scalar.")
                    derived = grid_nested_values(
                        payload,
                        bands=freqs_axis,
                        times=time_axis,
                        reducer=reducer,
                        **collapse_base_cfg,
                    )

                save_pkl(derived, out_pkl)
                if _should_export_xlsx(enabled_output):
                    xlsx_ok, xlsx_message = svc._save_table_xlsx(derived, out_xlsx)
                    if not xlsx_ok:
                        xlsx_warnings.append(f"{metric_key}/{stem}: {xlsx_message}")
                saved += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{metric_key}/{stem}: {exc}")

    return _MetricExtractResult(
        metric_key=metric_key,
        axes_signature=axes_signature,
        saved=saved,
        total_targets=total_targets,
        errors=tuple(errors),
        xlsx_warnings=tuple(xlsx_warnings),
    )


def run_extract_features(
    context: RecordContext,
    *,
    paradigm_slug: str,
    config_store: AppConfigStore | None = None,
    axes_by_metric: dict[str, dict[str, Any]] | None = None,
    enabled_outputs_by_metric: dict[str, dict[str, bool]] | None = None,
    reducer_by_metric: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Run Extract-Features over alignment raw-table inputs for one selected trial."""
    from . import service as svc

    resolver = PathResolver(context)
    resolver.ensure_record_roots()
    slug = svc._normalize_slug(paradigm_slug)
    if not slug:
        return False, "Select one trial first."

    if indicator_from_log(alignment_paradigm_log_path(resolver, slug)) != "green":
        return (
            False,
            "Selected alignment trial must be green before Extract Features.",
        )

    raw_tables = svc._iter_alignment_raw_tables(resolver, trial_slug=slug)
    if not raw_tables:
        return False, "No alignment raw-table inputs found for selected trial."

    derive_param_cfg = svc._load_derive_param_cfg(config_store)
    reducer_cfg = svc._load_reducer_cfg(config_store)
    reducer_rule_by_method = svc._load_reducer_rule_by_method(config_store)
    collapse_base_cfg = svc._load_collapse_base_cfg(config_store)
    alignment_method = svc._extract_alignment_method_from_log(resolver, trial_slug=slug)
    metric_axes = axes_by_metric if isinstance(axes_by_metric, dict) else {}
    metric_outputs = (
        enabled_outputs_by_metric if isinstance(enabled_outputs_by_metric, dict) else {}
    )
    metric_reducers = reducer_by_metric if isinstance(reducer_by_metric, dict) else {}
    axes_signature_by_metric: dict[str, dict[str, list[dict[str, Any]]]] = {}

    deriv_root = svc.features_derivatives_root(resolver, trial_slug=slug, create=True)
    saved = 0
    total_targets = 0
    errors: list[str] = []
    xlsx_warnings: list[str] = []
    metric_results_by_key: dict[str, _MetricExtractResult] = {}
    metric_items = list(raw_tables)
    if len(metric_items) >= 2:
        with ThreadPoolExecutor(max_workers=len(metric_items)) as executor:
            future_to_metric = {
                executor.submit(
                    _extract_metric_outputs,
                    metric_key=metric_key,
                    src_path=src_path,
                    deriv_root=deriv_root,
                    derive_param_cfg=derive_param_cfg,
                    reducer_cfg=reducer_cfg,
                    reducer_rule_by_method=reducer_rule_by_method,
                    collapse_base_cfg=collapse_base_cfg,
                    alignment_method=alignment_method,
                    axis_node=metric_axes.get(metric_key),
                    override_outputs=metric_outputs.get(metric_key),
                    reducer_override=str(metric_reducers.get(metric_key, "")).strip().lower(),
                ): metric_key
                for metric_key, src_path in metric_items
            }
            for future in as_completed(future_to_metric):
                metric_key = future_to_metric[future]
                try:
                    metric_results_by_key[metric_key] = future.result()
                except Exception as exc:  # noqa: BLE001
                    metric_results_by_key[metric_key] = _MetricExtractResult(
                        metric_key=metric_key,
                        axes_signature=None,
                        saved=0,
                        total_targets=0,
                        errors=(f"{metric_key}: worker failed ({exc})",),
                        xlsx_warnings=(),
                    )
    else:
        for metric_key, src_path in metric_items:
            metric_results_by_key[metric_key] = _extract_metric_outputs(
                metric_key=metric_key,
                src_path=src_path,
                deriv_root=deriv_root,
                derive_param_cfg=derive_param_cfg,
                reducer_cfg=reducer_cfg,
                reducer_rule_by_method=reducer_rule_by_method,
                collapse_base_cfg=collapse_base_cfg,
                alignment_method=alignment_method,
                axis_node=metric_axes.get(metric_key),
                override_outputs=metric_outputs.get(metric_key),
                reducer_override=str(metric_reducers.get(metric_key, "")).strip().lower(),
            )

    for metric_key, _src_path in metric_items:
        result = metric_results_by_key[metric_key]
        if result.axes_signature is not None:
            axes_signature_by_metric[metric_key] = result.axes_signature
        total_targets += result.total_targets
        saved += result.saved
        errors.extend(result.errors)
        xlsx_warnings.extend(result.xlsx_warnings)

    completed = total_targets > 0 and saved == total_targets and not errors
    params_payload = {
        "trial_slug": slug,
        "alignment_method": alignment_method,
        "metrics": [metric for metric, _ in raw_tables],
        "target_outputs": total_targets,
        "saved_outputs": saved,
        "errors": errors,
        "xlsx_warnings": xlsx_warnings,
        "axes_by_metric": axes_signature_by_metric,
    }
    write_run_log(
        svc.features_derivatives_log_path(resolver, trial_slug=slug),
        RunLogRecord(
            step="run_extract_features",
            completed=completed,
            params=params_payload,
            input_path=str(resolver.alignment_root / slug),
            output_path=str(deriv_root),
            message=(
                "Extract Features completed."
                if completed
                else f"Extract Features failed for {len(errors)} target(s)."
            ),
        ),
    )

    if completed:
        message = f"Extract Features completed. Saved {saved} table(s)."
        if xlsx_warnings:
            message += f" XLSX export failed for {len(xlsx_warnings)} table(s)."
        return True, message
    if total_targets == 0:
        return False, "Extract Features failed: no enabled derivation targets."
    return False, f"Extract Features failed. Saved {saved}, errors={len(errors)}."
