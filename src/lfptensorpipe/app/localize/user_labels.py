"""Record-scoped, editable channel region membership."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lfptensorpipe.app.shared.atomic_outputs import write_outputs_atomically


@dataclass
class UserLabels:
    regions: tuple[str, ...] = ()
    # An absent channel is unassessed. An empty set is assessed outside all ROIs.
    assessed: dict[str, set[str]] = field(default_factory=dict)
    method: str = ""

    def signature(self, channels: tuple[str, ...]) -> dict:
        return {
            "channels": sorted(channels),
            "regions": sorted(self.regions),
            "membership": {ch: sorted(hits) for ch, hits in self.assessed.items()},
            "assignment_method": self.method.strip(),
        }

    def frame(
        self, subject: str, record: str, channels: tuple[str, ...]
    ) -> pd.DataFrame:
        frame = pd.DataFrame({"channel": list(channels)})
        frame["subject"] = subject
        frame["record"] = record
        for column in ("space", "atlas", "anode", "cathode", "rep_coord"):
            frame[column] = None
        frame["assessed"] = frame["channel"].isin(self.assessed)
        for axis in ("mni_x", "mni_y", "mni_z"):
            frame[axis] = np.nan
        for region in self.regions:
            frame[f"{region}_in"] = pd.array(
                [
                    region in self.assessed[ch] if ch in self.assessed else pd.NA
                    for ch in channels
                ],
                dtype="boolean",
            )
        return frame


def read_labels(
    path: Path, channels: tuple[str, ...], *, merge_into: UserLabels | None = None
) -> UserLabels:
    """Read a saved table or merge CSV cells into a separate editor draft."""
    base = merge_into or UserLabels()
    rows = {ch: hits.copy() for ch, hits in base.assessed.items()}
    known = set(channels)
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = [v.strip() for v in next(reader, [])]
        if "channel" not in header or len(set(header)) != len(header):
            raise ValueError("CSV needs a channel column and unique column names.")
        region_columns = []
        for index, name in enumerate(header):
            if name in ("channel", "assignment_method"):
                continue
            if not name.endswith("_in") or not name[:-3].strip():
                raise ValueError(
                    "Use channel, optional assignment_method, and <region>_in columns."
                )
            region_columns.append((index, name[:-3]))
        if not region_columns and merge_into is not None:
            raise ValueError("CSV needs at least one <region>_in column.")
        imported = [region for _, region in region_columns]
        channel_col = header.index("channel")
        method_col = (
            header.index("assignment_method") if "assignment_method" in header else None
        )
        method = None
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError(f"Row {reader.line_num}: incorrect number of columns.")
            ch = values[channel_col].strip()
            if ch not in known or ch in seen:
                raise ValueError(
                    f"Row {reader.line_num}: unknown or duplicate channel {ch!r}."
                )
            seen.add(ch)
            if method_col is not None:
                value = values[method_col].strip()
                if method is not None and value != method:
                    raise ValueError(
                        f"Row {reader.line_num}: inconsistent assignment method."
                    )
                method = value
            flags = [values[i].strip() for i, _ in region_columns]
            if not any(flags):
                rows.pop(ch, None)
            elif all(v in ("0", "1") for v in flags):
                hits = rows.get(ch, set()).difference(imported)
                hits.update(
                    region for region, flag in zip(imported, flags) if flag == "1"
                )
                rows[ch] = hits
            else:
                raise ValueError(
                    f"Row {reader.line_num}: use 0/1 or leave the entire row unassessed."
                )
    return UserLabels(
        tuple(dict.fromkeys((*base.regions, *imported))),
        rows,
        base.method if method is None else method,
    )


def save_labels(path: Path, labels: UserLabels, channels: tuple[str, ...]) -> None:
    def write(staged: Path) -> None:
        with staged.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["channel", *(r + "_in" for r in labels.regions), "assignment_method"]
            )
            for ch in channels:
                hits = labels.assessed.get(ch)
                writer.writerow(
                    [
                        ch,
                        *(
                            "" if hits is None else int(r in hits)
                            for r in labels.regions
                        ),
                        labels.method.strip(),
                    ]
                )

    write_outputs_atomically([(path, write)])
