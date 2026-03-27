from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pandas as pd

from paper.pd.viz import run_viz


class InlineExecutor:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> InlineExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - helper failure path
            future.set_exception(exc)
        return future


def _patch_scalar_fixture(tmp_path: Path, monkeypatch) -> Path:
    project = tmp_path / "project"
    summary_root = project / "summary" / "table"

    def write_scalar_workbook(section: str, phases: list[str]) -> None:
        metric_dir = summary_root / section / "periodic"
        metric_dir.mkdir(parents=True, exist_ok=True)

        workbook = metric_dir / "mean-scalar_summary_trans.xlsx"
        frame = pd.DataFrame(
            {
                "Phase": phases,
                "Region": ["SNr"] * len(phases),
                "Value": list(range(1, len(phases) + 1)),
            }
        )
        frame.to_excel(workbook, index=False)

        model_dir = metric_dir / workbook.stem / "Beta_low" / "lmer_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        frame.to_excel(model_dir.parent / "input_band.xlsx", index=False)

    write_scalar_workbook("med", ["Off", "On"])
    write_scalar_workbook("turn", ["Offset", "Pre", "Post", "Onset"])

    specs = {
        "med_scalar": {
            "family": "scalar",
            "section": "med",
            "metric_dirs": ["periodic"],
            "glob": "*scalar_summary_trans.xlsx",
            "recipes": [{"x_var": "Phase", "panel_var": "Region"}],
            "df_type": "raw",
            "param_mode": "prefixed",
            "stem_suffix": "_summary_trans",
        },
        "turn_scalar": {
            "family": "scalar",
            "section": "turn",
            "metric_dirs": ["periodic"],
            "glob": "*scalar_summary_trans.xlsx",
            "recipes": [{"x_var": "Phase", "panel_var": "Region"}],
            "df_type": "raw",
            "param_mode": "prefixed",
            "stem_suffix": "_summary_trans",
        },
    }

    monkeypatch.setattr(run_viz, "PROJECT", project)
    monkeypatch.setattr(run_viz, "VIZ_SPECS", specs)
    monkeypatch.setattr(run_viz, "SECTIONS", ("med", "turn"))

    def fake_plot_scalar_wrapper(
        *,
        df: pd.DataFrame,
        df_type: str,
        save_dir: Path,
        param_type: str,
        bd: str,
        x_var: str,
        x_levels: list[str] | None = None,
        panel_var: str | None = None,
        facet_var: str | None = None,
    ):
        parts = [x_var]
        if panel_var is not None:
            parts.append(panel_var)
        if facet_var is not None:
            parts.append(facet_var)
        save_path = Path(save_dir) / f"{'-'.join(parts)}_{df_type}.pdf"
        save_path.write_text(
            f"{param_type}|{bd}|rows={len(df)}|x_levels={x_levels}",
            encoding="utf-8",
        )
        return run_viz.plt.figure()

    monkeypatch.setattr(run_viz.cfg, "plot_scalar_wrapper", fake_plot_scalar_wrapper)
    return project


def _patch_series_fixture(tmp_path: Path, monkeypatch) -> Path:
    project = tmp_path / "project"
    summary_root = project / "summary" / "table"

    med_metric_dir = summary_root / "med" / "periodic"
    med_metric_dir.mkdir(parents=True, exist_ok=True)
    med_frame = pd.DataFrame(
        {
            "Phase": ["Off", "On"],
            "Region": ["SNr", "SNr"],
            "Value": [1.0, 2.0],
        }
    )
    med_frame.to_pickle(med_metric_dir / "mean-spectral_summary_trans.pkl")

    turn_metric_dir = summary_root / "turn" / "periodic"
    turn_metric_dir.mkdir(parents=True, exist_ok=True)
    turn_frame = pd.DataFrame(
        {
            "Phase": ["Offset", "Pre", "Post", "Onset"],
            "Region": ["SNr", "SNr", "SNr", "SNr"],
            "Value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    turn_frame.to_pickle(turn_metric_dir / "mean-spectral_summary_trans.pkl")

    specs = {
        "med_spectral": {
            "family": "spectral",
            "section": "med",
            "metric_dirs": ["periodic"],
            "glob": "*spectral_summary_trans.pkl",
            "recipes": [{"x_var": "Phase", "panel_var": "Region"}],
            "df_type": "raw",
            "derive_type": "spectral",
            "param_mode": "prefixed",
            "stem_suffix": "_summary_trans",
        },
        "turn_spectral_phase": {
            "family": "spectral",
            "section": "turn",
            "metric_dirs": ["periodic"],
            "glob": "*spectral_summary_trans.pkl",
            "recipes": [{"x_var": "Phase", "panel_var": "Region"}],
            "df_type": "raw",
            "derive_type": "spectral",
            "param_mode": "prefixed",
            "stem_suffix": "_summary_trans",
        },
        "turn_spectral_region": {
            "family": "spectral",
            "section": "turn",
            "metric_dirs": ["periodic"],
            "glob": "*spectral_summary_trans.pkl",
            "recipes": [{"x_var": "Region"}],
            "df_type": "raw",
            "derive_type": "spectral",
            "param_mode": "prefixed",
            "stem_suffix": "_summary_trans",
        },
    }

    monkeypatch.setattr(run_viz, "PROJECT", project)
    monkeypatch.setattr(run_viz, "VIZ_SPECS", specs)
    monkeypatch.setattr(run_viz, "SECTIONS", ("med", "turn"))
    return project


def _pdf_snapshot(project: Path) -> dict[str, str]:
    return {
        path.relative_to(project).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(project.rglob("*.pdf"))
    }


def test_build_parser_parses_jobs_and_collects_selected_specs(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_scalar_fixture(tmp_path, monkeypatch)

    args = run_viz._build_parser().parse_args(["--spec", "med_scalar", "--jobs", "3"])

    assert args.spec == "med_scalar"
    assert args.jobs == 3
    assert run_viz.collect_spec_names(spec_name=args.spec) == ["med_scalar"]
    assert run_viz.collect_spec_names(section="turn") == ["turn_scalar"]


def test_collect_workbook_tasks_expands_selected_specs(
    tmp_path: Path, monkeypatch
) -> None:
    project = _patch_scalar_fixture(tmp_path, monkeypatch)

    tasks = run_viz.collect_workbook_tasks(["med_scalar", "turn_scalar"])

    assert [(task.spec_name, task.df_path.relative_to(project).as_posix()) for task in tasks] == [
        ("med_scalar", "summary/table/med/periodic/mean-scalar_summary_trans.xlsx"),
        ("turn_scalar", "summary/table/turn/periodic/mean-scalar_summary_trans.xlsx"),
    ]


def test_run_selected_specs_matches_serial_and_parallel_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    project = _patch_scalar_fixture(tmp_path, monkeypatch)

    run_viz.run_selected_specs(["med_scalar", "turn_scalar"], jobs=1)
    serial_outputs = _pdf_snapshot(project)
    for pdf_path in project.rglob("*.pdf"):
        pdf_path.unlink()

    monkeypatch.setattr(run_viz, "ProcessPoolExecutor", InlineExecutor)
    run_viz.run_selected_specs(["med_scalar", "turn_scalar"], jobs=2)
    parallel_outputs = _pdf_snapshot(project)

    assert parallel_outputs == serial_outputs


def test_run_selected_specs_injects_turn_phase_levels_for_scalar(
    tmp_path: Path, monkeypatch
) -> None:
    project = _patch_scalar_fixture(tmp_path, monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_plot_scalar_wrapper(
        *,
        df: pd.DataFrame,
        df_type: str,
        save_dir: Path,
        param_type: str,
        bd: str,
        x_var: str,
        x_levels: list[str] | None = None,
        panel_var: str | None = None,
        facet_var: str | None = None,
    ):
        calls.append(
            {
                "save_dir": Path(save_dir),
                "x_var": x_var,
                "x_levels": x_levels,
                "panel_var": panel_var,
                "rows": len(df),
            }
        )
        return run_viz.plt.figure()

    monkeypatch.setattr(run_viz.cfg, "plot_scalar_wrapper", fake_plot_scalar_wrapper)

    run_viz.run_selected_specs(["turn_scalar"], jobs=1)

    assert calls == [
        {
            "save_dir": (
                project
                / "summary"
                / "table"
                / "turn"
                / "periodic"
                / "mean-scalar_summary_trans"
                / "Beta_low"
                / "lmer_model"
            ),
            "x_var": "Phase",
            "x_levels": ["Pre", "Onset", "Offset", "Post"],
            "panel_var": "Region",
            "rows": 4,
        }
    ]


def test_run_selected_specs_only_injects_phase_levels_for_turn_phase_series(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_series_fixture(tmp_path, monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_plot_series_wrapper(
        *,
        df: pd.DataFrame,
        df_type: str,
        save_dir: Path,
        param_type: str,
        x_var: str,
        x_levels: list[str] | None = None,
        bd: str | None = None,
        panel_var: str | None = None,
        facet_var: str | None = None,
        derive_type: str = "spectral",
        section: str | None = None,
    ):
        calls.append(
            {
                "section": section,
                "x_var": x_var,
                "x_levels": x_levels,
                "panel_var": panel_var,
                "rows": len(df),
            }
        )
        return run_viz.plt.figure()

    monkeypatch.setattr(run_viz.cfg, "plot_series_wrapper", fake_plot_series_wrapper)

    run_viz.run_selected_specs(
        ["med_spectral", "turn_spectral_phase", "turn_spectral_region"],
        jobs=1,
    )

    assert calls == [
        {
            "section": "med",
            "x_var": "Phase",
            "x_levels": None,
            "panel_var": "Region",
            "rows": 2,
        },
        {
            "section": "turn",
            "x_var": "Phase",
            "x_levels": ["Pre", "Onset", "Offset", "Post"],
            "panel_var": "Region",
            "rows": 4,
        },
        {
            "section": "turn",
            "x_var": "Region",
            "x_levels": None,
            "panel_var": None,
            "rows": 4,
        },
    ]
