"""Extract-Features stage service helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from .extract_runner import run_extract_features as _run_extract_features_impl
from .normalization_runner import run_normalization as _run_normalization_impl

from .derive_config import (
    _bands_from_raw_value_index as _bands_from_raw_value_index_impl,
    _extract_alignment_method_from_log as _extract_alignment_method_from_log_impl,
    _load_collapse_base_cfg as _load_collapse_base_cfg_impl,
    _load_derive_param_cfg as _load_derive_param_cfg_impl,
    _load_plot_advance_defaults as _load_plot_advance_defaults_impl,
    _load_post_transform_modes as _load_post_transform_modes_impl,
    _load_reducer_cfg as _load_reducer_cfg_impl,
    _load_reducer_rule_by_method as _load_reducer_rule_by_method_impl,
    _metric_uses_auto_bands as _metric_uses_auto_bands_impl,
    _normalize_axis_rows as _normalize_axis_rows_impl,
    _normalize_enabled_outputs_map as _normalize_enabled_outputs_map_impl,
    _normalize_reducer_list as _normalize_reducer_list_impl,
    _read_derive_payload as _read_derive_payload_impl,
    _resolve_enabled_outputs as _resolve_enabled_outputs_impl,
    _resolve_reducers as _resolve_reducers_impl,
    _rows_to_interval_mapping as _rows_to_interval_mapping_impl,
    load_derive_defaults as _load_derive_defaults_impl,
)
from .indicator import (
    _aggregate_states as _aggregate_states_impl,
    _normalize_slug as _normalize_slug_impl,
    extract_features_indicator_state as _extract_features_indicator_state_impl,
    features_panel_state as _features_panel_state_impl,
    features_derivatives_log_path as _features_derivatives_log_path_impl,
    features_derivatives_root as _features_derivatives_root_impl,
    features_normalization_log_path as _features_normalization_log_path_impl,
    features_normalization_root as _features_normalization_root_impl,
    normalization_indicator_state as _normalization_indicator_state_impl,
)
from .table_io import (
    _detect_derived_type as _detect_derived_type_impl,
    _flatten_value_for_xlsx as _flatten_value_for_xlsx_impl,
    _iter_alignment_raw_tables as _iter_alignment_raw_tables_impl,
    _iter_feature_source_tables as _iter_feature_source_tables_impl,
    _iter_raw_tables as _iter_raw_tables_impl,
    _save_table_xlsx as _save_table_xlsx_impl,
)


def _normalize_slug(value: str) -> str:
    return _normalize_slug_impl(value)


def _aggregate_states(log_paths: list[Path]) -> str:
    return _aggregate_states_impl(log_paths)


def features_derivatives_root(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    transformed: bool | None = None,
    create: bool = False,
) -> Path:
    return _features_derivatives_root_impl(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
        transformed=transformed,
        create=create,
    )


def features_derivatives_log_path(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    transformed: bool | None = None,
) -> Path:
    return _features_derivatives_log_path_impl(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
        transformed=transformed,
    )


def features_normalization_root(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    transformed: bool | None = None,
    create: bool = False,
) -> Path:
    return _features_normalization_root_impl(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
        transformed=transformed,
        create=create,
    )


def features_normalization_log_path(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    transformed: bool | None = None,
) -> Path:
    return _features_normalization_log_path_impl(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
        transformed=transformed,
    )


def extract_features_indicator_state(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
) -> str:
    return _extract_features_indicator_state_impl(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
    )


def features_panel_state(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
    axes_by_metric: dict[str, dict[str, Any]] | None = None,
) -> str:
    return _features_panel_state_impl(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
        axes_by_metric=axes_by_metric,
    )


def normalization_indicator_state(
    resolver: PathResolver,
    *,
    trial_slug: str | None = None,
    paradigm_slug: str | None = None,
) -> str:
    return _normalization_indicator_state_impl(
        resolver,
        trial_slug=trial_slug,
        paradigm_slug=paradigm_slug,
    )


def _flatten_value_for_xlsx(value: Any) -> str:
    return _flatten_value_for_xlsx_impl(value)


def _save_table_xlsx(df: pd.DataFrame, out_path: Path) -> tuple[bool, str]:
    return _save_table_xlsx_impl(df, out_path)


def _detect_derived_type(df: pd.DataFrame) -> str:
    return _detect_derived_type_impl(df)


def _iter_raw_tables(
    resolver: PathResolver,
    *,
    paradigm_slug: str,
) -> list[tuple[str, str, Path]]:
    return _iter_raw_tables_impl(resolver, paradigm_slug=paradigm_slug)


def _iter_feature_source_tables(
    root: Path,
) -> tuple[list[tuple[Path, Path, pd.DataFrame]], list[str]]:
    return _iter_feature_source_tables_impl(root)


def _load_post_transform_modes(config_store: AppConfigStore | None) -> dict[str, str]:
    return _load_post_transform_modes_impl(config_store)


def load_derive_defaults(config_store: AppConfigStore | None) -> dict[str, Any]:
    return _load_derive_defaults_impl(config_store)


def _iter_alignment_raw_tables(
    resolver: PathResolver,
    *,
    trial_slug: str,
) -> list[tuple[str, Path]]:
    return _iter_alignment_raw_tables_impl(resolver, trial_slug=trial_slug)


def _read_derive_payload(config_store: AppConfigStore | None) -> dict[str, Any]:
    return _read_derive_payload_impl(config_store)


def _load_plot_advance_defaults(config_store: AppConfigStore | None) -> dict[str, Any]:
    return _load_plot_advance_defaults_impl(config_store)


def _normalize_enabled_outputs_map(value: Any) -> dict[str, bool]:
    return _normalize_enabled_outputs_map_impl(value)


def _load_derive_param_cfg(
    config_store: AppConfigStore | None,
) -> dict[str, dict[str, bool]]:
    return _load_derive_param_cfg_impl(config_store)


def _normalize_reducer_list(value: Any) -> list[str]:
    return _normalize_reducer_list_impl(value)


def _load_reducer_cfg(
    config_store: AppConfigStore | None,
) -> dict[str, dict[str, list[str]]]:
    return _load_reducer_cfg_impl(config_store)


def _load_reducer_rule_by_method(
    config_store: AppConfigStore | None,
) -> dict[str, dict[str, list[str]]]:
    return _load_reducer_rule_by_method_impl(config_store)


def _extract_alignment_method_from_log(
    resolver: PathResolver,
    *,
    trial_slug: str,
) -> str:
    return _extract_alignment_method_from_log_impl(resolver, trial_slug=trial_slug)


def _load_collapse_base_cfg(config_store: AppConfigStore | None) -> dict[str, Any]:
    return _load_collapse_base_cfg_impl(config_store)


def _resolve_enabled_outputs(
    derive_param_cfg: dict[str, dict[str, bool]],
    metric_key: str,
) -> dict[str, bool]:
    return _resolve_enabled_outputs_impl(derive_param_cfg, metric_key)


def _resolve_reducers(
    reducer_cfg: dict[str, dict[str, list[str]]],
    metric_key: str,
) -> list[str]:
    return _resolve_reducers_impl(reducer_cfg, metric_key)


def _normalize_axis_rows(
    value: Any,
    *,
    allow_duplicate_names: bool = False,
) -> list[dict[str, float | str]]:
    return _normalize_axis_rows_impl(
        value,
        allow_duplicate_names=allow_duplicate_names,
    )


def _rows_to_interval_mapping(
    rows: list[dict[str, float | str]],
) -> dict[str, list[list[float]]]:
    return _rows_to_interval_mapping_impl(rows)


def _metric_uses_auto_bands(metric_key: str) -> bool:
    return _metric_uses_auto_bands_impl(metric_key)


def _bands_from_raw_value_index(payload: pd.DataFrame) -> dict[str, list[list[str]]]:
    return _bands_from_raw_value_index_impl(payload)


def invalidate_normalization_logs(
    resolver: PathResolver,
    *,
    paradigm_slug: str,
    reason: str,
    message: str,
    write_when_missing: bool,
) -> bool:
    """Deprecated in new workflow: normalization is plot-time only."""
    _ = (resolver, paradigm_slug, reason, message, write_when_missing)
    return False


def run_extract_features(
    context: RecordContext,
    *,
    paradigm_slug: str,
    config_store: AppConfigStore | None = None,
    axes_by_metric: dict[str, dict[str, Any]] | None = None,
    enabled_outputs_by_metric: dict[str, dict[str, bool]] | None = None,
    reducer_by_metric: dict[str, str] | None = None,
) -> tuple[bool, str]:
    return _run_extract_features_impl(
        context,
        paradigm_slug=paradigm_slug,
        config_store=config_store,
        axes_by_metric=axes_by_metric,
        enabled_outputs_by_metric=enabled_outputs_by_metric,
        reducer_by_metric=reducer_by_metric,
    )


def run_normalization(
    context: RecordContext,
    *,
    paradigm_slug: str,
    baseline_phase: str,
    mode: str,
) -> tuple[bool, str]:
    return _run_normalization_impl(
        context,
        paradigm_slug=paradigm_slug,
        baseline_phase=baseline_phase,
        mode=mode,
    )
