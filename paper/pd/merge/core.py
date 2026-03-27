"""Backend helpers for merging PD feature tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from paper.pd.paths import derivatives_root, resolve_project_root, summary_table_root
from paper.pd.specs import (
    LEFT_CHANNELS,
    MergeSpec,
    MERGE_PHASE_BY_NAME_TRIAL,
    NON_CONNECTIVITY_METRICS,
    ORDERED_CONNECTIVITY_METRICS,
    PHASE_FILTERS_BY_NAME,
    RIGHT_CHANNELS,
    SIMPLIFIED_OUTPUT_COLUMNS,
    TRACE_LIKE_FILE_NAMES,
    UNORDERED_CONNECTIVITY_METRICS,
)
from paper.pd.table_io import is_scalar_table_name, save_table_outputs
from lfptensorpipe.io.pkl_io import load_pkl


@dataclass(frozen=True, slots=True)
class FeatureSourceKey:
    """One record-trial-relative-path bucket across subjects."""

    record: str
    trial: str
    relative_path: Path


FeatureInventory = dict[FeatureSourceKey, tuple[Path, ...]]


@dataclass(slots=True)
class MergeReport:
    """Structured merge summary for interactive review."""

    project_root: Path
    source_files: int = 0
    source_keys: int = 0
    named_outputs: list[Path] = field(default_factory=list)
    missing_selections: list[str] = field(default_factory=list)
    load_errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        """Return a compact numeric summary."""
        return {
            "source_files": self.source_files,
            "source_keys": self.source_keys,
            "named_outputs": len(self.named_outputs),
            "missing_selections": len(self.missing_selections),
            "load_errors": len(self.load_errors),
        }


def _normalize_token(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Record, trial, and merge names cannot be empty.")
    return text.lower().replace("_", "-")


def _normalize_trial_values(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    cleaned = tuple(str(item).strip() for item in values if str(item).strip())
    if not cleaned:
        raise ValueError("Each merge spec entry must contain at least one trial.")
    return cleaned


def normalize_merge_spec(
    merge_spec: MergeSpec,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Normalize the merge spec to a uniform mapping of trial tuples."""
    normalized: dict[str, dict[str, tuple[str, ...]]] = {}
    for name, record_map in merge_spec.items():
        merge_name = str(name).strip()
        if not merge_name:
            raise ValueError("Merge names cannot be empty.")
        normalized[merge_name] = {}
        for record, trials in record_map.items():
            record_name = str(record).strip()
            if not record_name:
                raise ValueError(f"Merge '{merge_name}' contains an empty record key.")
            normalized[merge_name][record_name] = _normalize_trial_values(trials)
    return normalized


def collect_feature_inventory(project_root: str | Path | None = None) -> FeatureInventory:
    """Scan the project derivatives tree and group feature tables by source key."""
    root = derivatives_root(project_root)
    grouped: defaultdict[FeatureSourceKey, list[Path]] = defaultdict(list)
    if not root.exists():
        return {}

    for path in sorted(root.glob("sub-*/**/features/**/*.pkl")):
        rel = path.relative_to(root)
        parts = rel.parts
        if "features" not in parts:
            continue
        feature_idx = parts.index("features")
        if feature_idx < 2 or len(parts) <= feature_idx + 2:
            continue
        key = FeatureSourceKey(
            record=parts[feature_idx - 1],
            trial=parts[feature_idx + 1],
            relative_path=Path(*parts[feature_idx + 2 :]),
        )
        grouped[key].append(path)

    return {
        key: tuple(sorted(paths))
        for key, paths in sorted(
            grouped.items(),
            key=lambda item: (
                item[0].record,
                item[0].trial,
                item[0].relative_path.as_posix(),
            ),
        )
    }


def inventory_frame(inventory: FeatureInventory) -> pd.DataFrame:
    """Return the inventory as a compact review table."""
    rows = [
        {
            "record": key.record,
            "trial": key.trial,
            "relative_path": key.relative_path.as_posix(),
            "source_files": len(paths),
        }
        for key, paths in inventory.items()
    ]
    return pd.DataFrame(rows)


def available_record_trials(inventory: FeatureInventory) -> dict[str, tuple[str, ...]]:
    """Return available trials for each record."""
    trials_by_record: defaultdict[str, set[str]] = defaultdict(set)
    for key in inventory:
        trials_by_record[key.record].add(key.trial)
    return {
        record: tuple(sorted(trials))
        for record, trials in sorted(trials_by_record.items())
    }


def cap_first(x: Any) -> Any:
    """Capitalize the first character of a string only when it is lowercase."""
    if isinstance(x, str):
        if x == "mni_x":
            return "MNI_x"
        if x == "mni_y":
            return "MNI_y"
        if x == "mni_z":
            return "MNI_z"
        if x and x[0].islower():
            return x[0].upper() + x[1:]
    return x


def capitalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Capitalize the first character of strings in values, columns, and index."""
    out = df.map(cap_first)
    out.columns = [cap_first(column) for column in out.columns]
    out.index = [cap_first(index) for index in out.index]
    return out


def _metric_key(relative_path: Path) -> str:
    if not relative_path.parts:
        raise ValueError("Relative feature path cannot be empty.")
    return relative_path.parts[0].lower()


def _required_series(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return df[name]
    raise KeyError(f"None of the required columns were found: {candidates!r}")


def _optional_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def _bool_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise KeyError(f"Column '{name}' is required for region filtering.")
    return df[name].fillna(False).astype(bool)


def _has_top_level_value(cell: Any) -> bool:
    if isinstance(cell, (pd.Series, pd.DataFrame)):
        return True
    try:
        return bool(pd.notna(cell))
    except Exception:  # noqa: BLE001
        return True


def _non_connectivity_region(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    snr = _bool_series(df, "SNr_in")
    stn = _bool_series(df, "STN_in")
    keep_snr = snr & ~stn
    keep_stn = ~snr & stn
    keep_mask = keep_snr | keep_stn
    region = pd.Series(pd.NA, index=df.index, dtype="object")
    region.loc[keep_snr] = "SNr"
    region.loc[keep_stn] = "STN"
    return keep_mask, region


def _unordered_connectivity_region(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    keep_mask = (
        _bool_series(df, "SNr-STN_in")
        & ~_bool_series(df, "SNr-SNr_in")
        & ~_bool_series(df, "STN-STN_in")
    )
    region = pd.Series(pd.NA, index=df.index, dtype="object")
    region.loc[keep_mask] = "SNr-STN"
    return keep_mask, region


def _ordered_connectivity_region(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    keep_mask = (
        _bool_series(df, "SNr-STN_in")
        & ~_bool_series(df, "STN-SNr_in")
        & ~_bool_series(df, "SNr-SNr_in")
        & ~_bool_series(df, "STN-STN_in")
    )
    region = pd.Series(pd.NA, index=df.index, dtype="object")
    region.loc[keep_mask] = "SNr->STN"
    return keep_mask, region


def _normalize_phase_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace("-", "_")


def _source_trial_from_path(path: Path) -> str:
    parts = path.parts
    if "features" not in parts:
        raise ValueError(f"Could not infer source trial from path: {path}")
    feature_idx = parts.index("features")
    if len(parts) <= feature_idx + 1:
        raise ValueError(f"Feature path is missing the trial component: {path}")
    return str(parts[feature_idx + 1])


def _channel_anchor(value: Any) -> tuple[str | None, bool]:
    is_pair = isinstance(value, (tuple, list)) and not isinstance(value, str)
    anchor = value[0] if is_pair and value else value
    if pd.isna(anchor):
        return None, is_pair
    text = str(anchor).strip()
    return text or None, is_pair


def _channel_side(value: Any) -> str | Any:
    anchor, is_pair = _channel_anchor(value)
    if anchor in LEFT_CHANNELS:
        return "L"
    if anchor in RIGHT_CHANNELS:
        return "R"
    if is_pair:
        return "R"
    return pd.NA


def _split_cycle_phase(value: Any, side: Any) -> tuple[str | Any, str | Any]:
    token = _normalize_phase_token(value)
    if token is None:
        return pd.NA, pd.NA
    head, sep, tail = token.partition("_")
    phase = cap_first(head.lower()) if head else pd.NA
    if not sep or not tail:
        return phase, pd.NA
    tail_upper = tail.upper()
    if tail_upper not in {"L", "R"} or pd.isna(side):
        return phase, pd.NA
    return phase, "Ipsi" if tail_upper == side else "Contra"


def _lat_from_side(side: Any) -> str | Any:
    if side == "L":
        return "Ipsi"
    if side == "R":
        return "Contra"
    return pd.NA


def _raw_phase_series(
    df: pd.DataFrame,
    *,
    merge_name: str,
    source_trial: str,
) -> pd.Series:
    name_token = merge_name.strip().lower()
    trial_token = _normalize_token(source_trial)
    phase_map = MERGE_PHASE_BY_NAME_TRIAL.get(name_token)
    if phase_map is not None and trial_token in phase_map:
        return pd.Series(phase_map[trial_token], index=df.index, dtype="object")
    return _optional_series(df, "Phase")


def _resolve_phase_and_lat(
    df: pd.DataFrame,
    *,
    side: pd.Series,
    file_name: str,
    merge_name: str,
    source_trial: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw_phase = _raw_phase_series(df, merge_name=merge_name, source_trial=source_trial)
    name_token = merge_name.strip().lower()
    skip_phase_filter = file_name.lower() in TRACE_LIKE_FILE_NAMES

    if name_token == "cycle":
        phase_values: list[Any] = []
        lat_values: list[Any] = []
        keep_values: list[bool] = []
        allowed = PHASE_FILTERS_BY_NAME.get(name_token, frozenset())
        for value, side_value in zip(raw_phase.tolist(), side.tolist(), strict=False):
            phase, lat = _split_cycle_phase(value, side_value)
            if skip_phase_filter:
                lat = _lat_from_side(side_value)
            phase_values.append(phase)
            lat_values.append(lat)
            if skip_phase_filter:
                keep_values.append(True)
            else:
                keep_values.append(isinstance(phase, str) and phase in allowed)
        return (
            pd.Series(keep_values, index=df.index, dtype=bool),
            pd.Series(phase_values, index=df.index, dtype="object"),
            pd.Series(lat_values, index=df.index, dtype="object"),
        )

    phase_values: list[Any] = []
    keep_values: list[bool] = []
    allowed = PHASE_FILTERS_BY_NAME.get(name_token)
    for value in raw_phase.tolist():
        token = _normalize_phase_token(value)
        if token is None:
            phase = pd.NA
        else:
            phase = cap_first(token.lower())
        phase_values.append(phase)
        if skip_phase_filter or allowed is None:
            keep_values.append(True)
        else:
            keep_values.append(isinstance(phase, str) and phase in allowed)

    return (
        pd.Series(keep_values, index=df.index, dtype=bool),
        pd.Series(phase_values, index=df.index, dtype="object"),
        pd.Series(pd.NA, index=df.index, dtype="object"),
    )


def _simplify_feature_table(
    df: pd.DataFrame,
    relative_path: Path,
    *,
    merge_name: str,
    source_trial: str,
) -> pd.DataFrame:
    metric_key = _metric_key(relative_path)
    if metric_key in ORDERED_CONNECTIVITY_METRICS:
        keep_mask, region = _ordered_connectivity_region(df)
    elif metric_key in UNORDERED_CONNECTIVITY_METRICS:
        keep_mask, region = _unordered_connectivity_region(df)
    elif metric_key in NON_CONNECTIVITY_METRICS:
        keep_mask, region = _non_connectivity_region(df)
    else:
        raise ValueError(f"Unsupported metric category for '{metric_key}'.")

    channel = _required_series(df, "Channel", "channel")
    side = channel.map(_channel_side)

    phase_keep_mask, phase, lat = _resolve_phase_and_lat(
        df,
        side=side,
        file_name=relative_path.name,
        merge_name=merge_name,
        source_trial=source_trial,
    )
    keep_mask = keep_mask & phase_keep_mask

    simplified = pd.DataFrame(
        {
            "subject": _required_series(df, "subject", "Subject"),
            "channel": channel,
            "side": side,
            "mni_x": _required_series(df, "mni_x"),
            "mni_y": _required_series(df, "mni_y"),
            "mni_z": _required_series(df, "mni_z"),
            "region": region,
            "band": _optional_series(df, "Band"),
            "phase": phase,
            "lat": lat,
            "value": _required_series(df, "Value"),
        },
        columns=list(SIMPLIFIED_OUTPUT_COLUMNS),
    )

    simplified = simplified.loc[keep_mask].reset_index(drop=True)
    value_mask = simplified["value"].map(_has_top_level_value)
    simplified = simplified.loc[value_mask].reset_index(drop=True)
    return capitalize_df(simplified)


def _merge_pickled_tables(
    paths: Sequence[Path],
    relative_path: Path,
    *,
    merge_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        payload = load_pkl(path)
        if not isinstance(payload, pd.DataFrame):
            raise TypeError(f"{path} does not contain a pandas.DataFrame payload.")
        frames.append(
            _simplify_feature_table(
                payload,
                relative_path,
                merge_name=merge_name,
                source_trial=_source_trial_from_path(path),
            )
        )
    if not frames:
        raise ValueError("No tables were provided for merging.")
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged.reset_index(drop=True)


def _ensure_report(
    project_root: Path,
    inventory: FeatureInventory,
    report: MergeReport | None,
) -> MergeReport:
    if report is not None:
        return report
    return MergeReport(
        project_root=project_root,
        source_files=sum(len(paths) for paths in inventory.values()),
        source_keys=len(inventory),
    )


def _pair_index(
    inventory: FeatureInventory,
) -> dict[tuple[str, str], dict[Path, tuple[Path, ...]]]:
    grouped: defaultdict[tuple[str, str], dict[Path, tuple[Path, ...]]] = defaultdict(dict)
    for key, paths in inventory.items():
        grouped[(key.record, key.trial)][key.relative_path] = paths
    return dict(grouped)


def export_named_tables(
    project_root: str | Path | None,
    merge_spec: MergeSpec,
    *,
    inventory: FeatureInventory | None = None,
    report: MergeReport | None = None,
    strict_selection: bool = False,
) -> MergeReport:
    """Export named paper-level tables that flatten selected record-trial groups."""
    resolved_root = resolve_project_root(project_root)
    normalized_spec = normalize_merge_spec(merge_spec)
    current_inventory = collect_feature_inventory(resolved_root) if inventory is None else inventory
    current_report = _ensure_report(resolved_root, current_inventory, report)
    table_root = summary_table_root(resolved_root, create=True)
    pairs = _pair_index(current_inventory)

    for name, record_map in normalized_spec.items():
        selected_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for record, trials in record_map.items():
            record_token = _normalize_token(record)
            for trial in trials:
                trial_token = _normalize_token(trial)
                matched_pairs = [
                    pair
                    for pair in pairs
                    if _normalize_token(pair[0]) == record_token
                    and _normalize_token(pair[1]) == trial_token
                ]
                if not matched_pairs:
                    message = (
                        f"{name}: no source tables found for record={record}, trial={trial}."
                    )
                    if strict_selection:
                        raise ValueError(message)
                    current_report.missing_selections.append(message)
                    continue
                for pair in sorted(matched_pairs):
                    if pair not in seen_pairs:
                        selected_pairs.append(pair)
                        seen_pairs.add(pair)

        grouped_paths: defaultdict[Path, list[Path]] = defaultdict(list)
        for pair in selected_pairs:
            for relative_path, paths in pairs[pair].items():
                grouped_paths[relative_path].extend(paths)

        for relative_path, paths in sorted(
            grouped_paths.items(), key=lambda item: item[0].as_posix()
        ):
            try:
                merged = _merge_pickled_tables(paths, relative_path, merge_name=name)
            except Exception as exc:  # noqa: BLE001
                current_report.load_errors.append(
                    f"name={name}, path={relative_path.as_posix()}: {exc}"
                )
                continue
            out_path = table_root / name / relative_path
            try:
                save_table_outputs(
                    merged,
                    out_path,
                    export_xlsx=is_scalar_table_name(relative_path.name),
                )
            except Exception as exc:  # noqa: BLE001
                current_report.load_errors.append(
                    f"name={name}, path={relative_path.as_posix()}: {exc}"
                )
                continue
            current_report.named_outputs.append(out_path)

    return current_report


def export_merge_tables(
    project_root: str | Path | None = None,
    *,
    merge_spec: MergeSpec,
    strict_selection: bool = False,
) -> MergeReport:
    """Run the named feature-table merge export."""
    resolved_root = resolve_project_root(project_root)
    inventory = collect_feature_inventory(resolved_root)
    report = _ensure_report(resolved_root, inventory, report=None)

    export_named_tables(
        resolved_root,
        merge_spec,
        inventory=inventory,
        report=report,
        strict_selection=strict_selection,
    )

    return report
