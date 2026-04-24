"""Alignment representative-coordinate merge helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lfptensorpipe.app.localize.paths import (
    localize_ordered_pair_representative_pkl_path,
    localize_representative_csv_path,
    localize_representative_pkl_path,
    localize_undirected_pair_representative_pkl_path,
)
from lfptensorpipe.app.path_resolver import RecordContext
from lfptensorpipe.app.tensor.params import (
    TENSOR_DIRECTED_SELECTOR_KEYS,
    TENSOR_UNDIRECTED_SELECTOR_KEYS,
)
from lfptensorpipe.io.pkl_io import load_pkl
from lfptensorpipe.utils.pair_keys import (
    make_ordered_pair_key,
    make_undirected_pair_key,
    parse_pair_token,
)


def _localize_representative_csv_path(context: RecordContext) -> Path:
    return localize_representative_csv_path(
        context.project_root,
        context.subject,
        context.record,
    )


def _localize_representative_pkl_path(context: RecordContext) -> Path:
    return localize_representative_pkl_path(
        context.project_root,
        context.subject,
        context.record,
    )


def _localize_ordered_pair_representative_pkl_path(context: RecordContext) -> Path:
    return localize_ordered_pair_representative_pkl_path(
        context.project_root,
        context.subject,
        context.record,
    )


def _localize_undirected_pair_representative_pkl_path(context: RecordContext) -> Path:
    return localize_undirected_pair_representative_pkl_path(
        context.project_root,
        context.subject,
        context.record,
    )


def _repcoord_conflict_column_name(column: str, existing: set[str]) -> str:
    base = f"{column}_repcoord"
    if base not in existing:
        return base
    idx = 2
    while f"{base}_{idx}" in existing:
        idx += 1
    return f"{base}_{idx}"


def _load_repcoord_frame(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file():
        return None, f"skip_repcoord_merge_missing_file:{path}"
    try:
        payload = load_pkl(path)
    except Exception as exc:  # noqa: BLE001
        return None, f"skip_repcoord_merge_read_failed:{exc}"
    if not isinstance(payload, pd.DataFrame):
        return None, "skip_repcoord_merge_invalid_payload"
    return payload.copy(), None


def _merge_repcoord_frame(
    frame: pd.DataFrame,
    rep_frame: pd.DataFrame,
    *,
    left_key: str,
    right_key: str,
    drop_left_key: bool = False,
    keep_right_key: bool = False,
) -> pd.DataFrame:
    existing = set(frame.columns)
    rename_map: dict[str, str] = {}
    for column in rep_frame.columns:
        if column == right_key:
            continue
        if column in existing:
            target = _repcoord_conflict_column_name(column, existing)
            rename_map[column] = target
            existing.add(target)
        else:
            existing.add(column)
    if rename_map:
        rep_frame = rep_frame.rename(columns=rename_map)

    merged = frame.merge(rep_frame, how="left", left_on=left_key, right_on=right_key)
    if right_key in merged.columns and not keep_right_key:
        merged = merged.drop(columns=[right_key])
    if drop_left_key and left_key in merged.columns:
        merged = merged.drop(columns=[left_key])
    return merged


def _merge_channel_representative_coords(
    frame: pd.DataFrame,
    context: RecordContext,
) -> tuple[pd.DataFrame, str | None]:
    rep_frame, warning = _load_repcoord_frame(
        _localize_representative_pkl_path(context)
    )
    if rep_frame is None:
        return frame, warning

    if "channel" not in rep_frame.columns:
        return frame, "skip_repcoord_merge_missing_channel_column"

    rep_frame["channel"] = rep_frame["channel"].astype(str).str.strip()
    rep_frame = rep_frame.loc[rep_frame["channel"] != ""]
    if rep_frame.empty:
        return frame, "skip_repcoord_merge_empty_channel_rows"

    rep_frame = rep_frame.drop_duplicates(subset=["channel"], keep="first")
    merged = _merge_repcoord_frame(
        frame,
        rep_frame,
        left_key="Channel",
        right_key="channel",
    )
    return merged, None


def _pair_key_series(
    values: pd.Series,
    *,
    directed: bool,
) -> tuple[pd.Series, int]:
    keys: list[str | None] = []
    valid = 0
    for value in values.tolist():
        pair = parse_pair_token(value)
        if pair is None:
            keys.append(None)
            continue
        valid += 1
        if directed:
            keys.append(make_ordered_pair_key(pair[0], pair[1]))
        else:
            keys.append(make_undirected_pair_key(pair[0], pair[1]))
    return pd.Series(keys, index=values.index, dtype=object), valid


def _merge_pair_representative_coords(
    frame: pd.DataFrame,
    context: RecordContext,
    *,
    directed: bool,
) -> tuple[pd.DataFrame, str | None]:
    path = (
        _localize_ordered_pair_representative_pkl_path(context)
        if directed
        else _localize_undirected_pair_representative_pkl_path(context)
    )
    key_column = "pair_key_ordered" if directed else "pair_key_undirected"
    rep_frame, warning = _load_repcoord_frame(path)
    if rep_frame is None:
        return frame, warning
    if key_column not in rep_frame.columns:
        return frame, f"skip_repcoord_merge_missing_{key_column}_column"

    rep_frame[key_column] = rep_frame[key_column].astype(str).str.strip()
    rep_frame = rep_frame.loc[rep_frame[key_column] != ""]
    if rep_frame.empty:
        return frame, "skip_repcoord_merge_empty_pair_rows"

    rep_frame = rep_frame.drop_duplicates(subset=[key_column], keep="first")

    merge_key = "__repcoord_pair_key__"
    pair_keys, valid_pairs = _pair_key_series(frame["Channel"], directed=directed)
    if valid_pairs == 0:
        return frame, "skip_repcoord_merge_unparseable_pair_channels"

    keyed = frame.copy()
    keyed[merge_key] = pair_keys
    merged = _merge_repcoord_frame(
        keyed,
        rep_frame,
        left_key=merge_key,
        right_key=key_column,
        drop_left_key=True,
        keep_right_key=True,
    )
    return merged, None


def _merge_representative_coords_for_metric(
    frame: pd.DataFrame,
    context: RecordContext,
    *,
    metric_key: str,
) -> tuple[pd.DataFrame, str | None]:
    key = str(metric_key).strip().lower()
    if key in TENSOR_DIRECTED_SELECTOR_KEYS:
        return _merge_pair_representative_coords(frame, context, directed=True)
    if key in TENSOR_UNDIRECTED_SELECTOR_KEYS:
        return _merge_pair_representative_coords(frame, context, directed=False)
    return _merge_channel_representative_coords(frame, context)


__all__ = [
    "_localize_ordered_pair_representative_pkl_path",
    "_localize_representative_csv_path",
    "_localize_representative_pkl_path",
    "_localize_undirected_pair_representative_pkl_path",
    "_merge_channel_representative_coords",
    "_merge_representative_coords_for_metric",
    "_repcoord_conflict_column_name",
]
