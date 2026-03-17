"""Representative-coordinate frame builders for channel and pair artifacts."""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any, Callable

import numpy as np
import pandas as pd

from lfptensorpipe.utils.pair_keys import (
    make_ordered_pair_key,
    make_undirected_pair_key,
    normalize_region_pair_name,
    normalize_undirected_pair,
)

_CHANNEL_BASE_COLUMNS = [
    "subject",
    "record",
    "space",
    "atlas",
    "channel",
    "anode",
    "cathode",
    "rep_coord",
    "mni_x",
    "mni_y",
    "mni_z",
]

_PAIR_BASE_COLUMNS = [
    "subject",
    "record",
    "space",
    "atlas",
    "channel",
    "channel_a",
    "channel_b",
    "pair_key",
    "pair_key_ordered",
    "pair_key_undirected",
    "mni_x",
    "mni_y",
    "mni_z",
]


def _as_xyz(value: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"Invalid {label} coordinate shape: {arr.shape}")
    return arr


def _build_contact_lookup(reconstruction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for lead in reconstruction.get("leads", []):
        contacts = lead.get("contacts", []) if isinstance(lead, dict) else []
        for contact in contacts:
            token = str(contact.get("token", "")).strip()
            if token:
                lookup[token] = contact
    return lookup


def _region_names_from_channel_frame(channel_frame: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for column in channel_frame.columns:
        if not column.endswith("_in"):
            continue
        names.append(column[:-3])
    return names


def _unique_channel_frame(channel_frame: pd.DataFrame) -> pd.DataFrame:
    if "channel" not in channel_frame.columns:
        raise ValueError("Representative frame is missing `channel`.")
    frame = channel_frame.copy()
    frame["channel"] = frame["channel"].astype(str).str.strip()
    frame = frame.loc[frame["channel"] != ""]
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=["channel"], keep="first").reset_index(drop=True)


def _pair_coordinate_tuple(
    row_a: pd.Series,
    row_b: pd.Series,
    axis: str,
) -> tuple[float, float]:
    return (float(row_a[axis]), float(row_b[axis]))


def _empty_pair_frame(region_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*_PAIR_BASE_COLUMNS, *region_columns])


def build_channel_representative_frame(
    *,
    subject: str,
    record: str,
    space: str,
    atlas: str,
    reconstruction: dict[str, Any],
    mappings: list[dict[str, Any]],
    map_native_to_mni_fn: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    lookup = _build_contact_lookup(reconstruction)

    rows: list[dict[str, Any]] = []
    mid_native_points: list[np.ndarray] = []
    mid_row_indices: list[int] = []

    for item in mappings:
        channel = str(item.get("channel", "")).strip()
        anode = str(item.get("anode", "")).strip()
        cathode_raw = str(item.get("cathode", "")).strip()
        cathode = cathode_raw.lower()
        rep_coord = str(item.get("rep_coord", "")).strip().title()
        if not channel:
            raise ValueError("Match entry missing channel.")
        if anode not in lookup:
            raise ValueError(f"Anode token not found in reconstruction: {anode}")

        anode_node = lookup[anode]
        anode_native = _as_xyz(anode_node.get("native", []), label=f"anode {anode}")
        anode_mni = _as_xyz(anode_node.get("mni", []), label=f"anode {anode}")

        is_special_cathode = cathode in {"case", "ground"}
        cathode_native = None
        cathode_mni = None
        if not is_special_cathode:
            if cathode_raw not in lookup:
                raise ValueError(
                    f"Cathode token not found in reconstruction: {cathode_raw}"
                )
            cathode_node = lookup[cathode_raw]
            cathode_native = _as_xyz(
                cathode_node.get("native", []),
                label=f"cathode {cathode_raw}",
            )
            cathode_mni = _as_xyz(
                cathode_node.get("mni", []),
                label=f"cathode {cathode_raw}",
            )

        if rep_coord not in {"Anode", "Cathode", "Mid"}:
            raise ValueError(f"Invalid rep_coord for channel {channel}: {rep_coord}")
        if is_special_cathode and rep_coord != "Anode":
            raise ValueError(
                f"rep_coord must be 'Anode' when cathode is case/ground: {channel}"
            )

        if rep_coord == "Anode":
            mni_xyz = anode_mni
        elif rep_coord == "Cathode":
            if cathode_mni is None:
                raise ValueError(f"Missing cathode for rep_coord Cathode: {channel}")
            mni_xyz = cathode_mni
        else:
            if cathode_native is None:
                raise ValueError(f"Missing cathode for rep_coord Mid: {channel}")
            mni_xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            mid_native_points.append((anode_native + cathode_native) / 2.0)
            mid_row_indices.append(len(rows))

        rows.append(
            {
                "subject": subject,
                "record": record,
                "space": space,
                "atlas": atlas,
                "channel": channel,
                "anode": anode,
                "cathode": cathode_raw,
                "rep_coord": rep_coord,
                "mni_x": float(mni_xyz[0]),
                "mni_y": float(mni_xyz[1]),
                "mni_z": float(mni_xyz[2]),
            }
        )

    if mid_native_points:
        native_mid = np.asarray(mid_native_points, dtype=float)
        mni_mid = np.asarray(map_native_to_mni_fn(native_mid), dtype=float)
        if mni_mid.ndim == 1:
            mni_mid = mni_mid.reshape(1, -1)
        if mni_mid.shape != (len(mid_row_indices), 3):
            raise ValueError("Midpoint mapping count mismatch.")
        for out_idx, row_idx in enumerate(mid_row_indices):
            rows[row_idx]["mni_x"] = float(mni_mid[out_idx, 0])
            rows[row_idx]["mni_y"] = float(mni_mid[out_idx, 1])
            rows[row_idx]["mni_z"] = float(mni_mid[out_idx, 2])

    return pd.DataFrame(rows, columns=_CHANNEL_BASE_COLUMNS)


def build_ordered_pair_representative_frame(
    channel_frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = _unique_channel_frame(channel_frame)
    region_names = _region_names_from_channel_frame(frame)
    region_columns = [
        f"{region_a}-{region_b}_in"
        for region_a in region_names
        for region_b in region_names
    ]
    if frame.shape[0] < 2:
        return _empty_pair_frame(region_columns)

    rows: list[dict[str, Any]] = []
    for left_idx, right_idx in permutations(range(frame.shape[0]), 2):
        row_a = frame.iloc[left_idx]
        row_b = frame.iloc[right_idx]
        channel_a = str(row_a["channel"]).strip()
        channel_b = str(row_b["channel"]).strip()
        row: dict[str, Any] = {
            "subject": row_a.get("subject"),
            "record": row_a.get("record"),
            "space": row_a.get("space"),
            "atlas": row_a.get("atlas"),
            "channel": (channel_a, channel_b),
            "channel_a": channel_a,
            "channel_b": channel_b,
            "pair_key": make_ordered_pair_key(channel_a, channel_b),
            "pair_key_ordered": make_ordered_pair_key(channel_a, channel_b),
            "pair_key_undirected": make_undirected_pair_key(channel_a, channel_b),
            "mni_x": _pair_coordinate_tuple(row_a, row_b, "mni_x"),
            "mni_y": _pair_coordinate_tuple(row_a, row_b, "mni_y"),
            "mni_z": _pair_coordinate_tuple(row_a, row_b, "mni_z"),
        }
        for region_a in region_names:
            left_hit = bool(row_a[f"{region_a}_in"])
            for region_b in region_names:
                row[f"{region_a}-{region_b}_in"] = left_hit and bool(
                    row_b[f"{region_b}_in"]
                )
        rows.append(row)

    return pd.DataFrame(rows, columns=[*_PAIR_BASE_COLUMNS, *region_columns])


def build_undirected_pair_representative_frame(
    channel_frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = _unique_channel_frame(channel_frame)
    region_names = _region_names_from_channel_frame(frame)
    ordered_region_names = sorted(
        region_names,
        key=lambda name: (name.casefold(), name),
    )
    region_columns = [
        f"{region_a}-{region_b}_in"
        for idx, region_a in enumerate(ordered_region_names)
        for region_b in ordered_region_names[idx:]
    ]
    if frame.shape[0] < 2:
        return _empty_pair_frame(region_columns)

    row_by_channel = {
        str(row["channel"]).strip(): row for _, row in frame.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for left_idx, right_idx in combinations(range(frame.shape[0]), 2):
        raw_a = frame.iloc[left_idx]
        raw_b = frame.iloc[right_idx]
        channel_left, channel_right = normalize_undirected_pair(
            str(raw_a["channel"]).strip(),
            str(raw_b["channel"]).strip(),
        )
        row_a = row_by_channel[channel_left]
        row_b = row_by_channel[channel_right]
        row: dict[str, Any] = {
            "subject": row_a.get("subject"),
            "record": row_a.get("record"),
            "space": row_a.get("space"),
            "atlas": row_a.get("atlas"),
            "channel": (channel_left, channel_right),
            "channel_a": channel_left,
            "channel_b": channel_right,
            "pair_key": make_undirected_pair_key(channel_left, channel_right),
            "pair_key_ordered": make_ordered_pair_key(channel_left, channel_right),
            "pair_key_undirected": make_undirected_pair_key(channel_left, channel_right),
            "mni_x": _pair_coordinate_tuple(row_a, row_b, "mni_x"),
            "mni_y": _pair_coordinate_tuple(row_a, row_b, "mni_y"),
            "mni_z": _pair_coordinate_tuple(row_a, row_b, "mni_z"),
        }
        for idx, region_a in enumerate(ordered_region_names):
            for region_b in ordered_region_names[idx:]:
                region_left, region_right = normalize_region_pair_name(region_a, region_b)
                if region_left == region_right:
                    value = bool(row_a[f"{region_left}_in"]) and bool(
                        row_b[f"{region_right}_in"]
                    )
                else:
                    value = (
                        bool(row_a[f"{region_left}_in"])
                        and bool(row_b[f"{region_right}_in"])
                    ) or (
                        bool(row_a[f"{region_right}_in"])
                        and bool(row_b[f"{region_left}_in"])
                    )
                row[f"{region_left}-{region_right}_in"] = value
        rows.append(row)

    return pd.DataFrame(rows, columns=[*_PAIR_BASE_COLUMNS, *region_columns])


__all__ = [
    "build_channel_representative_frame",
    "build_ordered_pair_representative_frame",
    "build_undirected_pair_representative_frame",
]
