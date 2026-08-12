"""Extract features runner logic."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import multiprocessing
from pathlib import Path
from typing import Any

import pandas as pd

from lfptensorpipe.app.alignment_service import alignment_paradigm_log_path
from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    indicator_from_log,
    write_run_log,
)
from lfptensorpipe.app.tensor.atomic_io import write_outputs_atomically
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl
from lfptensorpipe.stats.preproc.transform import transform_df
from lfptensorpipe.tabular.grid import (
    ReducerKind,
    grid_nested_values,
    split_nested_values,
)
from lfptensorpipe.utils.transforms import (
    VALUE_TRANSFORM_POLICY_KEY,
    TransformDomain,
    TransformPolicy,
    get_transform_policy,
    transform_policy_metadata,
    transform_policy_from_metadata,
)

from .burst_native import (
    LEGACY_BURST_REDUCER_WARNING,
    build_burst_scalar_tables,
    cleanup_legacy_occupation_outputs,
    normalize_burst_reducers,
)


@dataclass(frozen=True)
class _MetricExtractResult:
    """Result payload for one metric-level Extract Features task."""

    metric_key: str
    axes_signature: dict[str, list[dict[str, Any]]] | None
    saved: int
    total_targets: int
    errors: tuple[str, ...]
    xlsx_warnings: tuple[str, ...]
    warnings: tuple[str, ...] = ()


_METRIC_VALUE_REDUCERS = frozenset({ReducerKind.MEAN, ReducerKind.MEDIAN})


def _remove_stale_xlsx(path: Path) -> None:
    if path.exists():
        path.unlink()


def _should_export_xlsx(derived_type: str) -> bool:
    return derived_type == "scalar"


def _attach_policy_attrs(
    frame: pd.DataFrame,
    policy: TransformPolicy,
) -> pd.DataFrame:
    frame.attrs[VALUE_TRANSFORM_POLICY_KEY] = transform_policy_metadata(policy)
    return frame


def _convert_payload_domain(
    payload: pd.DataFrame,
    policy: TransformPolicy,
    *,
    source_domain: TransformDomain,
    target_domain: TransformDomain,
) -> pd.DataFrame:
    """Convert nested Feature values between declared transform domains."""

    if source_domain == target_domain:
        return payload.copy()
    return transform_df(
        payload,
        mode=policy.mode,
        inverse=source_domain == "transformed",
    )


def _raw_feature_payload(
    payload: pd.DataFrame,
    policy: TransformPolicy,
) -> pd.DataFrame:
    converted = _convert_payload_domain(
        payload,
        policy,
        source_domain=policy.tensor_storage_domain,
        target_domain=policy.feature_storage_domain,
    )
    return _attach_policy_attrs(converted, policy)


def _reduction_payload(
    payload: pd.DataFrame,
    policy: TransformPolicy,
) -> pd.DataFrame:
    return _convert_payload_domain(
        payload,
        policy,
        source_domain=policy.tensor_storage_domain,
        target_domain=policy.reduction_domain,
    )


def _finalize_reduced_payload(
    derived: pd.DataFrame,
    policy: TransformPolicy,
    *,
    reducer: ReducerKind | str,
) -> pd.DataFrame:
    reducer_kind = (
        reducer
        if isinstance(reducer, ReducerKind)
        else ReducerKind(str(reducer).strip().lower())
    )
    if reducer_kind not in _METRIC_VALUE_REDUCERS:
        return _attach_policy_attrs(derived.copy(), get_transform_policy("none"))
    converted = _convert_payload_domain(
        derived,
        policy,
        source_domain=policy.reduction_domain,
        target_domain=policy.feature_storage_domain,
    )
    return _attach_policy_attrs(converted, policy)


def _feature_storage_policy(metadata: Any) -> TransformPolicy:
    """Use upstream computation domains with the current Feature storage target."""

    source_policy = transform_policy_from_metadata(metadata)
    default_policy = get_transform_policy(source_policy.mode)
    return replace(
        source_policy,
        feature_storage_domain=default_policy.feature_storage_domain,
    )


def _extract_burst_outputs(
    *,
    payload: pd.DataFrame,
    transform_policy: TransformPolicy,
    metric_out_dir: Path,
    outputs: dict[str, bool],
    reducers: list[str],
    time_axis: dict[str, list[list[float]]],
    axes_signature: dict[str, list[dict[str, Any]]],
    resolver: PathResolver | None,
    trial_slug: str,
) -> _MetricExtractResult:
    """Write Burst raw/scalar PKLs as one accepted metric-level set."""
    runtime_warnings = (
        (LEGACY_BURST_REDUCER_WARNING,)
        if any(str(value).strip().lower() == "occupation" for value in reducers)
        else ()
    )
    unsupported_outputs = [
        output_name
        for output_name in ("spectral", "trace")
        if outputs.get(output_name, False)
    ]
    if unsupported_outputs:
        names = ", ".join(unsupported_outputs)
        return _MetricExtractResult(
            metric_key="burst",
            axes_signature=axes_signature,
            saved=0,
            total_targets=len(unsupported_outputs),
            errors=(f"burst: unsupported output type(s): {names}",),
            xlsx_warnings=(),
            warnings=runtime_warnings,
        )
    try:
        requested_reducers = normalize_burst_reducers(reducers)
    except Exception as exc:  # noqa: BLE001
        return _MetricExtractResult(
            metric_key="burst",
            axes_signature=axes_signature,
            saved=0,
            total_targets=int(outputs.get("raw", False)) + len(reducers),
            errors=(f"burst: {exc}",),
            xlsx_warnings=(),
            warnings=runtime_warnings,
        )
    total_targets = int(outputs.get("raw", False)) + (
        len(requested_reducers) if outputs.get("scalar", False) else 0
    )
    if total_targets == 0:
        return _MetricExtractResult(
            metric_key="burst",
            axes_signature=axes_signature,
            saved=0,
            total_targets=0,
            errors=(),
            xlsx_warnings=(),
            warnings=runtime_warnings,
        )

    try:
        scalar_tables: dict[str, pd.DataFrame] = {}
        if outputs.get("scalar", False):
            if resolver is None or not trial_slug:
                raise ValueError("Burst scalar extraction requires record context.")
            scalar_tables = build_burst_scalar_tables(
                resolver,
                trial_slug=trial_slug,
                aligned_payload=payload,
                phases=time_axis,
                reducers=requested_reducers,
            )
        authoritative_outputs: list[tuple[Path, Any]] = []
        if outputs.get("raw", False):
            raw_payload = _raw_feature_payload(payload, transform_policy)
            authoritative_outputs.append(
                (
                    metric_out_dir / "na-raw.pkl",
                    lambda path, value=raw_payload: save_pkl(value, path),
                )
            )
        for reducer, table in scalar_tables.items():
            authoritative_outputs.append(
                (
                    metric_out_dir / f"{reducer}-scalar.pkl",
                    lambda path, value=table: save_pkl(value, path),
                )
            )
        write_outputs_atomically(authoritative_outputs)
    except Exception as exc:  # noqa: BLE001
        return _MetricExtractResult(
            metric_key="burst",
            axes_signature=axes_signature,
            saved=0,
            total_targets=total_targets,
            errors=(f"burst: {exc}",),
            xlsx_warnings=(),
            warnings=runtime_warnings,
        )

    xlsx_warnings: list[str] = []
    if outputs.get("raw", False):
        _remove_stale_xlsx(metric_out_dir / "na-raw.xlsx")
    for reducer, table in scalar_tables.items():
        xlsx_path = metric_out_dir / f"{reducer}-scalar.xlsx"
        _remove_stale_xlsx(xlsx_path)
        from . import service as svc

        xlsx_ok, xlsx_message = svc._save_table_xlsx(table, xlsx_path)
        if not xlsx_ok:
            xlsx_warnings.append(f"burst/{reducer}-scalar: {xlsx_message}")
    if "occupancy" in scalar_tables:
        cleanup_legacy_occupation_outputs(metric_out_dir)
    return _MetricExtractResult(
        metric_key="burst",
        axes_signature=axes_signature,
        saved=total_targets,
        total_targets=total_targets,
        errors=(),
        xlsx_warnings=tuple(xlsx_warnings),
        warnings=runtime_warnings,
    )


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
    resolver: PathResolver | None = None,
    trial_slug: str = "",
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
    transform_policy = _feature_storage_policy(payload.attrs)
    reduction_payload = _reduction_payload(payload, transform_policy)

    outputs = svc._resolve_enabled_outputs(derive_param_cfg, metric_key)
    reducer_list = svc._resolve_reducers(reducer_cfg, metric_key)
    if isinstance(override_outputs, dict):
        outputs = svc._normalize_enabled_outputs_map(override_outputs)
    reducers = [reducer_override] if reducer_override else list(reducer_list)
    method_rule_node = reducer_rule_by_method.get(metric_key, {})
    if not isinstance(method_rule_node, dict):
        method_rule_node = {}
    if (
        metric_key != "burst"
        and alignment_method
        and alignment_method in method_rule_node
    ):
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

    if metric_key == "burst":
        return _extract_burst_outputs(
            payload=payload,
            transform_policy=transform_policy,
            metric_out_dir=metric_out_dir,
            outputs=outputs,
            reducers=reducers,
            time_axis=time_axis,
            axes_signature=axes_signature,
            resolver=resolver,
            trial_slug=trial_slug,
        )

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
            save_pkl(_raw_feature_payload(payload, transform_policy), out_pkl)
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
                        reduction_payload,
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
                        reduction_payload,
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
                        reduction_payload,
                        bands=freqs_axis,
                        times=time_axis,
                        reducer=reducer,
                        **collapse_base_cfg,
                    )

                derived = _finalize_reduced_payload(
                    derived,
                    transform_policy,
                    reducer=reducer,
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
    selected_metrics: list[str] | tuple[str, ...] | None = None,
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
    if selected_metrics is not None:
        requested_metrics = {
            str(metric).strip() for metric in selected_metrics if str(metric).strip()
        }
        raw_tables = [item for item in raw_tables if item[0] in requested_metrics]
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
    warnings: list[str] = []
    xlsx_warnings: list[str] = []
    metric_results_by_key: dict[str, _MetricExtractResult] = {}
    metric_items = list(raw_tables)
    if len(metric_items) >= 2:
        with ProcessPoolExecutor(
            max_workers=len(metric_items),
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
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
                    reducer_override=str(metric_reducers.get(metric_key, ""))
                    .strip()
                    .lower(),
                    resolver=resolver,
                    trial_slug=slug,
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
                reducer_override=str(metric_reducers.get(metric_key, ""))
                .strip()
                .lower(),
                resolver=resolver,
                trial_slug=slug,
            )

    for metric_key, _src_path in metric_items:
        result = metric_results_by_key[metric_key]
        if result.axes_signature is not None:
            axes_signature_by_metric[metric_key] = result.axes_signature
        total_targets += result.total_targets
        saved += result.saved
        errors.extend(result.errors)
        warnings.extend(result.warnings)
        xlsx_warnings.extend(result.xlsx_warnings)

    completed = total_targets > 0 and saved == total_targets and not errors
    params_payload = {
        "trial_slug": slug,
        "alignment_method": alignment_method,
        "metrics": [metric for metric, _ in raw_tables],
        "target_outputs": total_targets,
        "saved_outputs": saved,
        "errors": errors,
        "warnings": warnings,
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
        if warnings:
            message += f" {len(warnings)} warning(s); see run log."
        if xlsx_warnings:
            message += f" XLSX export failed for {len(xlsx_warnings)} table(s)."
        return True, message
    if total_targets == 0:
        return False, "Extract Features failed: no enabled derivation targets."
    message = f"Extract Features failed. Saved {saved}, errors={len(errors)}."
    if warnings:
        message += f" {len(warnings)} warning(s); see run log."
    return False, message
