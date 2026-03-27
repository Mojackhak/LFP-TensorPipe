from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lfptensorpipe.io.pkl_io import load_pkl, save_pkl
from paper.pd.examples.med import workflow
from paper.pd.viz import defaults


def _value_frame(base: float) -> pd.DataFrame:
    columns = [30.0, 60.0, 90.0, 120.0, 150.0]
    index = ["delta", "theta", "alpha", "beta_low", "beta_high", "gamma"]
    data = np.array(
        [
            [base + 0.01 * row_idx + 0.001 * col_idx for col_idx in range(len(columns))]
            for row_idx in range(len(index))
        ],
        dtype=float,
    )
    return pd.DataFrame(data, index=index, columns=columns)


def _write_med_burst_raw_fixture(project_root: Path) -> None:
    rows: list[dict[str, object]] = []
    subject_rows = [
        ("Sub-014", "0_1", "SNr", "Off", 1.0),
        ("Sub-014", "2_3", "STN", "Off", 2.0),
        ("Sub-014", "8_9", "SNr", "Off", 3.0),
        ("Sub-014", "10_11", "STN", "Off", 4.0),
        ("Sub-014", "0_1", "SNr", "On", 5.0),
        ("Sub-014", "2_3", "STN", "On", 6.0),
        ("Sub-014", "8_9", "SNr", "On", 7.0),
        ("Sub-014", "10_11", "STN", "On", 8.0),
        ("Sub-015", "0_1", "SNr", "Off", 9.0),
    ]
    for subject, channel, region, phase, base in subject_rows:
        rows.append(
            {
                "Subject": subject,
                "Channel": channel,
                "Side": "L" if channel in {"0_1", "2_3"} else "R",
                "MNI_x": 1.0,
                "MNI_y": 2.0,
                "MNI_z": 3.0,
                "Region": region,
                "Band": pd.NA,
                "Phase": phase,
                "Lat": pd.NA,
                "Value": _value_frame(base),
            }
        )

    path = project_root / "summary" / "table" / "med" / "burst" / "na-raw.pkl"
    save_pkl(pd.DataFrame(rows), path)


def test_export_subject_burst_example_writes_cropped_log10_pickle(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_med_burst_raw_fixture(project_root)

    out_path = workflow.export_subject_burst_example(project_root=project_root)

    assert out_path == (
        project_root / "summary" / "eg" / "med" / "sub-014_burst_na-raw_log10_60_120.pkl"
    )

    exported = load_pkl(out_path)
    assert isinstance(exported, pd.DataFrame)
    assert exported.shape == (8, 11)
    assert exported["Subject"].dropna().unique().tolist() == ["Sub-014"]
    assert exported.groupby(["Channel", "Phase"]).size().to_dict() == {
        ("0_1", "Off"): 1,
        ("0_1", "On"): 1,
        ("2_3", "Off"): 1,
        ("2_3", "On"): 1,
        ("8_9", "Off"): 1,
        ("8_9", "On"): 1,
        ("10_11", "Off"): 1,
        ("10_11", "On"): 1,
    }
    assert exported["Channel"].tolist() == [
        "0_1",
        "0_1",
        "2_3",
        "2_3",
        "8_9",
        "8_9",
        "10_11",
        "10_11",
    ]
    assert exported["Phase"].tolist() == ["Off", "On", "Off", "On", "Off", "On", "Off", "On"]

    first_value = exported.iloc[0]["Value"]
    assert isinstance(first_value, pd.DataFrame)
    assert first_value.shape == (6, 3)
    assert list(first_value.columns) == [60.0, 90.0, 120.0]

    source = load_pkl(project_root / "summary" / "table" / "med" / "burst" / "na-raw.pkl")
    source_row = source[
        (source["Subject"] == "Sub-014")
        & (source["Channel"] == "0_1")
        & (source["Phase"] == "Off")
    ].iloc[0]
    expected = np.log10(source_row["Value"].loc[:, [60.0, 90.0, 120.0]].to_numpy(dtype=float))
    np.testing.assert_allclose(first_value.to_numpy(dtype=float), expected)


def test_plot_raw_wrapper_supports_panel_levels_and_filebase_override(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_plot_double_interaction_df(*, df, panel_var, panel_levels=None, **kwargs):
        captured["panel_var"] = panel_var
        captured["panel_levels"] = panel_levels
        captured["kwargs"] = kwargs
        captured["df_rows"] = len(df)
        return plt.figure()

    def fake_save_fig(fig, save_path):
        captured["save_path"] = Path(save_path)
        Path(save_path).write_text("pdf", encoding="utf-8")
        plt.close(fig)

    monkeypatch.setattr(defaults.vdf, "plot_double_interaction_df", fake_plot_double_interaction_df)
    monkeypatch.setattr(defaults, "save_fig", fake_save_fig)

    df = pd.DataFrame(
        {
            "Phase": ["Off", "On"],
            "Value": [_value_frame(1.0), _value_frame(2.0)],
        }
    )

    defaults.plot_raw_wrapper(
        df=df,
        df_type="raw",
        save_dir=tmp_path,
        param_type="burst-mean",
        panel_var="Phase",
        panel_levels=["Off", "On"],
        section="med_burst_example",
        filebase_override="sub-014_channel-0_1",
    )

    assert captured["panel_var"] == "Phase"
    assert captured["panel_levels"] == ["Off", "On"]
    assert captured["df_rows"] == 2
    kwargs = captured["kwargs"]
    assert kwargs["x_limits"] == [0, 60]
    assert kwargs["x_label"] == "Time (s)"
    assert kwargs["y_label"] == "Band"
    assert kwargs["y_log"] is False
    assert kwargs["y_limits"] is None
    assert kwargs["horizontal_lines"] is None
    assert kwargs["vertical_lines"] is None
    assert kwargs["cmap"] == "viridis"
    assert kwargs["vmode"] == "auto"
    assert captured["save_path"] == tmp_path / "sub-014_channel-0_1_raw.pdf"


def test_run_subject_burst_example_viz_writes_one_pdf_per_channel(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    _write_med_burst_raw_fixture(project_root)
    source_path = workflow.export_subject_burst_example(project_root=project_root)

    calls: list[dict[str, object]] = []

    def fake_plot_raw_wrapper(
        *,
        df: pd.DataFrame,
        df_type: str,
        save_dir: Path,
        param_type: str,
        panel_var: str | None = None,
        panel_levels: list[str] | None = None,
        section: str | None = None,
        filebase_override: str | None = None,
        **_: object,
    ):
        save_path = Path(save_dir) / f"{filebase_override}_{df_type}.pdf"
        save_path.write_text("pdf", encoding="utf-8")
        calls.append(
            {
                "channel": df["Channel"].iloc[0],
                "phases": df["Phase"].tolist(),
                "value_index": list(df.iloc[0]["Value"].index),
                "value_columns": list(df.iloc[0]["Value"].columns),
                "panel_var": panel_var,
                "panel_levels": panel_levels,
                "section": section,
                "filebase_override": filebase_override,
                "save_path": save_path,
                "rows": len(df),
            }
        )
        return plt.figure()

    monkeypatch.setattr(workflow.cfg, "plot_raw_wrapper", fake_plot_raw_wrapper)

    outputs = workflow.run_subject_burst_example_viz(
        project_root=project_root,
        source_path=source_path,
    )

    expected = [
        source_path.parent / "sub-014_channel-0_1_raw.pdf",
        source_path.parent / "sub-014_channel-2_3_raw.pdf",
        source_path.parent / "sub-014_channel-8_9_raw.pdf",
        source_path.parent / "sub-014_channel-10_11_raw.pdf",
    ]
    assert outputs == expected
    assert [call["channel"] for call in calls] == ["0_1", "2_3", "8_9", "10_11"]
    assert all(call["rows"] == 2 for call in calls)
    assert all(call["phases"] == ["Off", "On"] for call in calls)
    assert all(call["panel_var"] == "Phase" for call in calls)
    assert all(call["panel_levels"] == ["Off", "On"] for call in calls)
    assert all(call["section"] == "med_burst_example" for call in calls)
    assert all(path.exists() for path in expected)
    first_value = calls[0]["value_index"]
    assert first_value == ["δ", "θ", "α", "β-low", "β-high", "γ"]
    first_columns = calls[0]["value_columns"]
    assert first_columns == [0.0, 30.0, 60.0]
