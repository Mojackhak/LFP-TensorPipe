"""Representative med burst example workflow for Sub-014."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from lfptensorpipe.io.pkl_io import load_pkl, save_pkl
from lfptensorpipe.stats.preproc.transform import transform_df
from paper.pd.paths import resolve_project_root, summary_root, summary_table_root
from paper.pd.viz import defaults as cfg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()

EXAMPLE_SUBJECT = "Sub-014"
EXAMPLE_TIME_START_S = 60.0
EXAMPLE_TIME_END_S = 120.0
EXAMPLE_PANEL_LEVELS: tuple[str, str] = ("Off", "On")
EXAMPLE_SECTION_KEY = "med_burst_example"
EXAMPLE_PARAM_TYPE = "burst-mean"
EXAMPLE_DF_TYPE = "raw"
EXAMPLE_BAND_TICK_LABELS: dict[str, str] = {
    "delta": "δ",
    "theta": "θ",
    "alpha": "α",
    "beta_low": "β-low",
    "beta_high": "β-high",
    "gamma": "γ",
}


def _subject_slug(subject: str) -> str:
    return str(subject).strip().lower()


def burst_example_input_path(project_root: str | Path | None = None) -> Path:
    """Return the source med burst raw table path."""
    return summary_table_root(project_root) / "med" / "burst" / "na-raw.pkl"


def burst_example_output_dir(
    project_root: str | Path | None = None,
    *,
    create: bool = False,
) -> Path:
    """Return the med example output directory."""
    out = summary_root(project_root, create=create) / "eg" / "med"
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def burst_example_output_path(
    project_root: str | Path | None = None,
    *,
    subject: str = EXAMPLE_SUBJECT,
) -> Path:
    """Return the exported subject-level example pickle path."""
    return burst_example_output_dir(project_root, create=True) / (
        f"{_subject_slug(subject)}_burst_na-raw_log10_60_120.pkl"
    )


def _channel_filebase(subject: str, channel: str) -> str:
    return f"{_subject_slug(subject)}_channel-{channel}"


def _crop_time_window(cell: object, *, start_s: float, end_s: float) -> pd.DataFrame:
    if not isinstance(cell, pd.DataFrame):
        raise TypeError(
            "Representative med burst example expects DataFrame-valued cells; "
            f"got {type(cell).__name__}."
        )
    selected = [column for column in cell.columns if start_s <= float(column) <= end_s]
    if not selected:
        raise ValueError(
            f"No time columns found inside the requested window {start_s} <= t <= {end_s}."
        )
    return cell.loc[:, selected].copy()


def _phase_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    order = {name: idx for idx, name in enumerate(EXAMPLE_PANEL_LEVELS)}
    return order.get(text, len(order)), text


def _display_band_label(value: object) -> str:
    text = str(value)
    return EXAMPLE_BAND_TICK_LABELS.get(text, text)


def _display_time_label(value: object) -> float:
    return float(value) - EXAMPLE_TIME_START_S


def _with_display_band_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Value"] = [
        value.rename(index=_display_band_label).copy()
        if isinstance(value, pd.DataFrame)
        else value
        for value in out["Value"].tolist()
    ]
    return out


def _with_display_time_axis(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Value"] = [
        value.rename(columns=_display_time_label).copy()
        if isinstance(value, pd.DataFrame)
        else value
        for value in out["Value"].tolist()
    ]
    return out


def _ordered_subject_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    channel_order = {
        str(channel): idx for idx, channel in enumerate(pd.unique(ordered["Channel"]))
    }
    ordered["_channel_order"] = ordered["Channel"].map(
        lambda value: channel_order[str(value)]
    )
    ordered["_phase_order"] = ordered["Phase"].map(lambda value: _phase_sort_key(value)[0])
    ordered = ordered.sort_values(
        by=["_channel_order", "_phase_order", "Phase"],
        kind="stable",
    ).drop(columns=["_channel_order", "_phase_order"])
    ordered.reset_index(drop=True, inplace=True)
    return ordered


def export_subject_burst_example(
    project_root: str | Path | None = None,
    *,
    subject: str = EXAMPLE_SUBJECT,
) -> Path:
    """Export the representative Sub-014 med burst example pickle."""
    input_path = burst_example_input_path(project_root)
    payload = load_pkl(input_path)
    if not isinstance(payload, pd.DataFrame):
        raise TypeError(f"{input_path} does not contain a pandas.DataFrame payload.")

    frame = payload[payload["Subject"] == subject].copy()
    if frame.empty:
        raise ValueError(f"No med burst raw rows found for subject {subject!r}.")

    transformed = transform_df(frame, value_col="Value", mode="log10")
    transformed["Value"] = [
        _crop_time_window(
            cell,
            start_s=EXAMPLE_TIME_START_S,
            end_s=EXAMPLE_TIME_END_S,
        )
        for cell in transformed["Value"].tolist()
    ]
    transformed = _ordered_subject_frame(transformed)

    out_path = burst_example_output_path(project_root, subject=subject)
    save_pkl(transformed, out_path)
    return out_path


def run_subject_burst_example_viz(
    project_root: str | Path | None = None,
    *,
    subject: str = EXAMPLE_SUBJECT,
    source_path: str | Path | None = None,
) -> list[Path]:
    """Render one representative med burst PDF per channel."""
    project_root_i = resolve_project_root(project_root)
    if source_path is not None:
        source = Path(source_path).expanduser().resolve()
    else:
        source = burst_example_output_path(project_root_i, subject=subject)
    payload = load_pkl(source)
    if not isinstance(payload, pd.DataFrame):
        raise TypeError(f"{source} does not contain a pandas.DataFrame payload.")

    frame = _ordered_subject_frame(payload)
    channels = list(pd.unique(frame["Channel"]))
    if not channels:
        raise ValueError(f"No channel rows found in {source}.")

    outputs: list[Path] = []
    save_dir = source.parent
    for channel in channels:
        df_channel = frame[frame["Channel"] == channel].copy()
        phases = set(df_channel["Phase"].dropna().astype(str).tolist())
        missing = [phase for phase in EXAMPLE_PANEL_LEVELS if phase not in phases]
        if missing:
            raise ValueError(
                f"Channel {channel!r} is missing required phases {missing!r} in {source}."
            )
        df_channel = _with_display_band_labels(df_channel)
        df_channel = _with_display_time_axis(df_channel)
        fig = cfg.plot_raw_wrapper(
            df=df_channel,
            df_type=EXAMPLE_DF_TYPE,
            save_dir=save_dir,
            param_type=EXAMPLE_PARAM_TYPE,
            panel_var="Phase",
            panel_levels=list(EXAMPLE_PANEL_LEVELS),
            section=EXAMPLE_SECTION_KEY,
            filebase_override=_channel_filebase(subject, str(channel)),
        )
        outputs.append(
            save_dir / f"{_channel_filebase(subject, str(channel))}_{EXAMPLE_DF_TYPE}.pdf"
        )
        plt.close(fig)
    return outputs
