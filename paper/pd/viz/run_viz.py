#%%
from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.ioff()

#%% path bootstrap
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[3]
_SRC_ROOT = _REPO_ROOT / "src"

for _path in (_REPO_ROOT, _SRC_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


#%% imports
from lfptensorpipe.io.pkl_io import load_pkl  # noqa: E402
from paper.pd.specs import DEFAULT_PROJECT_ROOT  # noqa: E402
from paper.pd.viz import defaults as cfg  # noqa: E402


#%% shared constants
PROJECT = Path(DEFAULT_PROJECT_ROOT)
DEFAULT_JOBS = max(1, os.cpu_count() or 1)
SINGLE_METRICS = ["raw_power", "periodic", "aperiodic", "burst"]
DOUBLE_METRICS = ["coherence", "ciplv", "pli", "plv", "psi", "trgc", "wpli"]

PREFIX_PARAM_MAP = {
    ("aperiodic", "mean"): "aperiod-mean",
    ("periodic", "mean"): "period-mean",
    ("raw_power", "mean"): "raw-mean",
    ("burst", "rate"): "burst-rate",
    ("burst", "duration"): "burst-duration",
    ("burst", "occupation"): "burst-occupation",
    ("burst", "mean"): "burst-mean",
    ("aperiodic", "na"): "aperiod-mean",
    ("periodic", "na"): "period-mean",
    ("raw_power", "na"): "raw-mean",
    ("burst", "na"): "burst-mean",
}

CONNECTIVITY_PARAM_MAP = {
    "coherence": "coh-mean",
    "ciplv": "ciplv-mean",
    "pli": "pli-mean",
    "plv": "plv-mean",
    "psi": "psi-mean",
    "trgc": "delta_net_gc-mean",
    "wpli": "wpli-mean",
}

FIGURE_X_LEVELS_BY_SECTION_AND_VAR: dict[tuple[str, str], tuple[str, ...]] = {
    ("turn", "Phase"): ("Pre", "Onset", "Offset", "Post"),
}


#%% spec builders
def _scalar_spec(
    *,
    section: str,
    metric_dirs: list[str],
    glob: str,
    recipes: list[dict[str, str]],
    param_mode: str,
) -> dict[str, Any]:
    return {
        "family": "scalar",
        "section": section,
        "metric_dirs": metric_dirs,
        "glob": glob,
        "recipes": recipes,
        "df_type": "raw",
        "param_mode": param_mode,
        "stem_suffix": "_summary_trans",
    }


def _spectral_spec(
    *,
    section: str,
    metric_dirs: list[str],
    recipes: list[dict[str, str]],
    param_mode: str,
) -> dict[str, Any]:
    return {
        "family": "spectral",
        "section": section,
        "metric_dirs": metric_dirs,
        "glob": "*spectral_summary_trans.pkl",
        "recipes": recipes,
        "df_type": "raw",
        "derive_type": "spectral",
        "param_mode": param_mode,
        "stem_suffix": "_summary_trans",
    }


def _trace_spec(
    *,
    section: str,
    metric_dirs: list[str],
    recipes: list[dict[str, str]],
    param_mode: str,
    glob: str = "*trace_summary_trans_normalized.pkl",
    stem_suffix: str = "_summary_trans_normalized",
) -> dict[str, Any]:
    return {
        "family": "trace",
        "section": section,
        "metric_dirs": metric_dirs,
        "glob": glob,
        "recipes": recipes,
        "df_type": "norm",
        "derive_type": "trace",
        "param_mode": param_mode,
        "stem_suffix": stem_suffix,
        "banded": True,
        "smooth_trace": True,
    }


def _raw_spec(
    *,
    section: str,
    metric_dirs: list[str],
    recipes: list[dict[str, str]],
    param_mode: str,
    glob: str = "*raw_summary_trans_normalized.pkl",
    stem_suffix: str = "_summary_trans_normalized",
) -> dict[str, Any]:
    return {
        "family": "raw",
        "section": section,
        "metric_dirs": metric_dirs,
        "glob": glob,
        "recipes": recipes,
        "df_type": "norm",
        "param_mode": param_mode,
        "stem_suffix": stem_suffix,
    }


VIZ_SPECS: dict[str, dict[str, Any]] = {
    # cycle
    "cycle_single_scalar": _scalar_spec(
        section="cycle",
        metric_dirs=SINGLE_METRICS,
        glob="*scalar_summary_trans.xlsx",
        recipes=[{"x_var": "Phase", "panel_var": "Lat", "facet_var": "Region"}],
        param_mode="prefixed",
    ),
    "cycle_double_scalar": _scalar_spec(
        section="cycle",
        metric_dirs=DOUBLE_METRICS,
        glob="*scalar_summary_trans.xlsx",
        recipes=[{"x_var": "Phase", "panel_var": "Lat"}],
        param_mode="metric",
    ),
    "cycle_single_spectral": _spectral_spec(
        section="cycle",
        metric_dirs=SINGLE_METRICS,
        recipes=[{"x_var": "Phase", "panel_var": "Lat", "facet_var": "Region"}],
        param_mode="prefixed",
    ),
    "cycle_double_spectral": _spectral_spec(
        section="cycle",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"x_var": "Phase", "panel_var": "Lat"}],
        param_mode="metric",
    ),
    "cycle_single_trace": _trace_spec(
        section="cycle",
        metric_dirs=["raw_power", "periodic", "aperiodic"],
        recipes=[{"x_var": "Lat", "panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "cycle_double_trace": _trace_spec(
        section="cycle",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"x_var": "Lat"}],
        param_mode="metric",
    ),
    "cycle_single_raw": _raw_spec(
        section="cycle",
        metric_dirs=["raw_power", "periodic", "aperiodic"],
        recipes=[{"panel_var": "Lat", "facet_var": "Region"}],
        param_mode="prefixed",
    ),
    "cycle_double_raw": _raw_spec(
        section="cycle",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"panel_var": "Lat"}],
        param_mode="metric",
    ),
    "cycle_single_trace_shift": _trace_spec(
        section="cycle",
        metric_dirs=["raw_power", "periodic", "aperiodic"],
        recipes=[{"x_var": "Region"}],
        param_mode="prefixed",
        glob="*trace_summary_trans_normalized_shift.pkl",
        stem_suffix="_summary_trans_normalized_shift",
    ),
    "cycle_double_trace_shift": _trace_spec(
        section="cycle",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"x_var": "Region"}],
        param_mode="metric",
        glob="*trace_summary_trans_normalized_shift.pkl",
        stem_suffix="_summary_trans_normalized_shift",
    ),
    "cycle_single_raw_shift": _raw_spec(
        section="cycle",
        metric_dirs=["raw_power", "periodic", "aperiodic"],
        recipes=[{"panel_var": "Region"}],
        param_mode="prefixed",
        glob="*raw_summary_trans_normalized_shift.pkl",
        stem_suffix="_summary_trans_normalized_shift",
    ),
    "cycle_double_raw_shift": _raw_spec(
        section="cycle",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"panel_var": "Region"}],
        param_mode="metric",
        glob="*raw_summary_trans_normalized_shift.pkl",
        stem_suffix="_summary_trans_normalized_shift",
    ),
    # med
    "med_single_scalar": _scalar_spec(
        section="med",
        metric_dirs=SINGLE_METRICS,
        glob="*scalar_summary_trans.xlsx",
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "med_double_scalar": _scalar_spec(
        section="med",
        metric_dirs=DOUBLE_METRICS,
        glob="*scalar_summary_trans.xlsx",
        recipes=[{"x_var": "Phase"}],
        param_mode="metric",
    ),
    "med_single_spectral": _spectral_spec(
        section="med",
        metric_dirs=SINGLE_METRICS,
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "med_double_spectral": _spectral_spec(
        section="med",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="metric",
    ),
    # motor
    "motor_single_scalar": _scalar_spec(
        section="motor",
        metric_dirs=SINGLE_METRICS,
        glob="*scalar_summary_trans.xlsx",
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "motor_double_scalar": _scalar_spec(
        section="motor",
        metric_dirs=DOUBLE_METRICS,
        glob="*scalar_summary_trans.xlsx",
        recipes=[{"x_var": "Phase"}],
        param_mode="metric",
    ),
    "motor_single_spectral": _spectral_spec(
        section="motor",
        metric_dirs=SINGLE_METRICS,
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "motor_double_spectral": _spectral_spec(
        section="motor",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="metric",
    ),
    # turn
    "turn_single_scalar": _scalar_spec(
        section="turn",
        metric_dirs=SINGLE_METRICS,
        glob="*scalar_summary_trans.xlsx",
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "turn_double_scalar": _scalar_spec(
        section="turn",
        metric_dirs=DOUBLE_METRICS,
        glob="*scalar_summary_trans.xlsx",
        recipes=[{"x_var": "Phase"}],
        param_mode="metric",
    ),
    "turn_single_spectral": _spectral_spec(
        section="turn",
        metric_dirs=SINGLE_METRICS,
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "turn_double_spectral": _spectral_spec(
        section="turn",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"x_var": "Phase", "panel_var": "Region"}],
        param_mode="metric",
    ),
    "turn_single_trace": _trace_spec(
        section="turn",
        metric_dirs=["raw_power", "periodic", "aperiodic"],
        recipes=[{"x_var": "Region"}],
        param_mode="prefixed",
    ),
    "turn_double_trace": _trace_spec(
        section="turn",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"x_var": "Region"}],
        param_mode="metric",
    ),
    "turn_single_raw": _raw_spec(
        section="turn",
        metric_dirs=["raw_power", "periodic", "aperiodic", "burst"],
        recipes=[{"panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "turn_double_raw": _raw_spec(
        section="turn",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"panel_var": "Region"}],
        param_mode="metric",
    ),
    # turn_stack
    "turn_stack_single_trace": _trace_spec(
        section="turn_stack",
        metric_dirs=["raw_power", "periodic", "aperiodic"],
        recipes=[{"x_var": "Region"}],
        param_mode="prefixed",
    ),
    "turn_stack_double_trace": _trace_spec(
        section="turn_stack",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"x_var": "Region"}],
        param_mode="metric",
    ),
    "turn_stack_single_raw": _raw_spec(
        section="turn_stack",
        metric_dirs=["raw_power", "periodic", "aperiodic"],
        recipes=[{"panel_var": "Region"}],
        param_mode="prefixed",
    ),
    "turn_stack_double_raw": _raw_spec(
        section="turn_stack",
        metric_dirs=DOUBLE_METRICS,
        recipes=[{"panel_var": "Region"}],
        param_mode="metric",
    ),
}

SECTIONS = tuple(sorted({spec["section"] for spec in VIZ_SPECS.values()}))


@dataclass(frozen=True)
class VizTask:
    spec_name: str
    df_path: Path


#%% helpers
def _parse_jobs(value: str) -> int:
    try:
        jobs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--jobs must be an integer.") from exc

    if jobs < 1:
        raise argparse.ArgumentTypeError("--jobs must be greater than or equal to 1.")
    return jobs


def _collect_df_paths(spec: dict[str, Any]) -> list[Path]:
    section_dir = PROJECT / "summary" / "table" / spec["section"]
    wk_dirs = [section_dir / metric_name for metric_name in spec["metric_dirs"]]
    return sorted(
        df_path
        for wk_dir in wk_dirs
        if wk_dir.exists()
        for df_path in wk_dir.glob(spec["glob"])
    )


def _resolve_param_type(df_path: Path, spec: dict[str, Any]) -> str:
    metric_name = df_path.parent.name
    param_mode = spec["param_mode"]

    if param_mode == "metric":
        try:
            return CONNECTIVITY_PARAM_MAP[metric_name]
        except KeyError as exc:
            raise KeyError(
                f"Unsupported connectivity workbook for visualization: {metric_name}/{df_path.name}"
            ) from exc

    if param_mode == "prefixed":
        prefix = df_path.stem.removesuffix(spec["stem_suffix"]).rsplit("-", 1)[0]
        try:
            return PREFIX_PARAM_MAP[(metric_name, prefix)]
        except KeyError as exc:
            raise KeyError(
                f"Unsupported workbook for visualization: {metric_name}/{df_path.name}"
            ) from exc

    raise ValueError(f"Unknown param_mode: {param_mode!r}")


def _load_scalar_band_frame(band_dir: Path) -> pd.DataFrame:
    df_band = pd.read_excel(band_dir / "input_band.xlsx")
    df_band.reset_index(drop=True, inplace=True)
    if "Value" in df_band.columns:
        df_band["Value"] = pd.to_numeric(df_band["Value"], errors="raise")
    return df_band


def _load_pkl_frame(df_path: Path) -> pd.DataFrame:
    df = load_pkl(df_path)
    df.reset_index(drop=True, inplace=True)
    return df


def _resolve_recipe_x_levels(
    *, spec: dict[str, Any], recipe: dict[str, str]
) -> list[str] | None:
    x_var = recipe.get("x_var")
    if x_var is None:
        return None

    levels = FIGURE_X_LEVELS_BY_SECTION_AND_VAR.get((spec["section"], x_var))
    if levels is None:
        return None
    return list(levels)


def _run_scalar_recipe(
    *,
    df: pd.DataFrame,
    save_dir: Path,
    param_type: str,
    spec: dict[str, Any],
    bd: str,
    recipe: dict[str, str],
) -> plt.Figure:
    return cfg.plot_scalar_wrapper(
        df=df,
        df_type="raw",
        save_dir=save_dir,
        param_type=param_type,
        bd=bd,
        x_var=recipe["x_var"],
        x_levels=_resolve_recipe_x_levels(spec=spec, recipe=recipe),
        panel_var=recipe.get("panel_var"),
        facet_var=recipe.get("facet_var"),
    )


def _run_series_recipe(
    *,
    df: pd.DataFrame,
    save_dir: Path,
    param_type: str,
    spec: dict[str, Any],
    recipe: dict[str, str],
    bd: str | None = None,
) -> plt.Figure:
    return cfg.plot_series_wrapper(
        df=df,
        df_type=spec["df_type"],
        save_dir=save_dir,
        param_type=param_type,
        bd=bd,
        x_var=recipe["x_var"],
        x_levels=_resolve_recipe_x_levels(spec=spec, recipe=recipe),
        panel_var=recipe.get("panel_var"),
        facet_var=recipe.get("facet_var"),
        derive_type=spec["derive_type"],
        section=spec["section"],
    )


def _run_raw_recipe(
    *,
    df: pd.DataFrame,
    save_dir: Path,
    param_type: str,
    spec: dict[str, Any],
    recipe: dict[str, str],
) -> plt.Figure:
    return cfg.plot_raw_wrapper(
        df=df,
        df_type=spec["df_type"],
        save_dir=save_dir,
        param_type=param_type,
        panel_var=recipe.get("panel_var"),
        facet_var=recipe.get("facet_var"),
        section=spec["section"],
    )


def collect_spec_names(
    *, section: str | None = None, spec_name: str | None = None
) -> list[str]:
    if spec_name is not None:
        if spec_name not in VIZ_SPECS:
            raise KeyError(f"Unknown visualization spec: {spec_name!r}")
        return [spec_name]

    if section is not None:
        if section not in SECTIONS:
            raise ValueError(f"Unknown visualization section: {section!r}")
        return [
            candidate_name
            for candidate_name, spec in VIZ_SPECS.items()
            if spec["section"] == section
        ]

    return list(VIZ_SPECS.keys())


def collect_workbook_tasks(spec_names: Iterable[str]) -> list[VizTask]:
    tasks: list[VizTask] = []
    for spec_name in spec_names:
        try:
            spec = VIZ_SPECS[spec_name]
        except KeyError as exc:
            raise KeyError(f"Unknown visualization spec: {spec_name!r}") from exc
        tasks.extend(
            VizTask(spec_name=spec_name, df_path=df_path)
            for df_path in _collect_df_paths(spec)
        )
    return tasks


def _run_scalar_workbook(spec_name: str, spec: dict[str, Any], df_path: Path) -> None:
    wk_dir = df_path.parent / df_path.stem
    if not wk_dir.exists():
        return

    param_type = _resolve_param_type(df_path, spec)
    print(f"[{spec_name}] Processing data: {df_path.stem}")

    band_dirs = sorted(
        band_dir
        for band_dir in wk_dir.iterdir()
        if band_dir.is_dir() and (band_dir / "input_band.xlsx").exists()
    )

    for band_dir in band_dirs:
        bd = band_dir.name
        df_band = _load_scalar_band_frame(band_dir)
        save_dir_model_list = sorted(
            save_dir_model
            for save_dir_model in band_dir.iterdir()
            if save_dir_model.is_dir() and save_dir_model.name.endswith("_model")
        )

        for save_dir_model in save_dir_model_list:
            print(f"[{spec_name}]   Processing model results in: {save_dir_model.stem}")
            for recipe in spec["recipes"]:
                _run_scalar_recipe(
                    df=df_band,
                    save_dir=save_dir_model,
                    param_type=param_type,
                    spec=spec,
                    bd=bd,
                    recipe=recipe,
                )
            plt.close("all")


def _run_series_workbook(spec_name: str, spec: dict[str, Any], df_path: Path) -> None:
    wk_dir = df_path.parent / df_path.stem
    wk_dir.mkdir(parents=True, exist_ok=True)

    param_type = _resolve_param_type(df_path, spec)
    print(f"[{spec_name}] Processing data: {df_path.stem}")

    df = _load_pkl_frame(df_path)
    if spec.get("smooth_trace", False):
        df = cfg.smooth_trace(df)

    if spec.get("banded", False):
        bands = sorted(df["Band"].dropna().unique().tolist())
        for bd in bands:
            df_band = df[df["Band"] == bd].copy()
            df_band.reset_index(drop=True, inplace=True)
            save_dir = wk_dir / bd
            save_dir.mkdir(parents=True, exist_ok=True)
            for recipe in spec["recipes"]:
                _run_series_recipe(
                    df=df_band,
                    save_dir=save_dir,
                    param_type=param_type,
                    spec=spec,
                    recipe=recipe,
                    bd=bd,
                )
            plt.close("all")
    else:
        for recipe in spec["recipes"]:
            _run_series_recipe(
                df=df,
                save_dir=wk_dir,
                param_type=param_type,
                spec=spec,
                recipe=recipe,
            )
        plt.close("all")


def _run_raw_workbook(spec_name: str, spec: dict[str, Any], df_path: Path) -> None:
    wk_dir = df_path.parent / df_path.stem
    wk_dir.mkdir(parents=True, exist_ok=True)

    param_type = _resolve_param_type(df_path, spec)
    print(f"[{spec_name}] Processing data: {df_path.stem}")

    df = _load_pkl_frame(df_path)
    for recipe in spec["recipes"]:
        _run_raw_recipe(
            df=df,
            save_dir=wk_dir,
            param_type=param_type,
            spec=spec,
            recipe=recipe,
        )
    plt.close("all")


def _run_task(task: VizTask) -> dict[str, Any]:
    try:
        spec = VIZ_SPECS[task.spec_name]
        family = spec["family"]
        if family == "scalar":
            _run_scalar_workbook(task.spec_name, spec, task.df_path)
        elif family in {"spectral", "trace"}:
            _run_series_workbook(task.spec_name, spec, task.df_path)
        elif family == "raw":
            _run_raw_workbook(task.spec_name, spec, task.df_path)
        else:
            raise ValueError(f"Unknown visualization family: {family!r}")

        return {
            "ok": True,
            "spec_name": task.spec_name,
            "df_path": task.df_path.as_posix(),
        }
    except Exception as exc:  # pragma: no cover - exercised in failure paths
        return {
            "ok": False,
            "spec_name": task.spec_name,
            "df_path": task.df_path.as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _report_task_result(result: dict[str, Any]) -> None:
    status = "completed" if result["ok"] else "failed"
    workbook_name = Path(result["df_path"]).name
    print(f"[{result['spec_name']}] {status}: {workbook_name}")


def _run_tasks(tasks: list[VizTask], jobs: int) -> list[dict[str, Any]]:
    if not tasks:
        return []

    worker_count = min(jobs, len(tasks))
    print(
        f"Running {len(tasks)} visualization workbook task(s) "
        f"with {worker_count} worker(s)"
    )

    if worker_count == 1:
        results = [_run_task(task) for task in tasks]
        for result in results:
            _report_task_result(result)
        return results

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - executor edge case
                result = {
                    "ok": False,
                    "spec_name": task.spec_name,
                    "df_path": task.df_path.as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _report_task_result(result)
            results.append(result)
    return results


def _raise_failures(results: list[dict[str, Any]]) -> None:
    failures = [result for result in results if not result["ok"]]
    if not failures:
        return

    failure_lines = [
        f"[{result['spec_name']}] {result['df_path']}: {result['error']}"
        for result in failures
    ]
    raise RuntimeError(
        "Visualization workbook tasks failed:\n" + "\n".join(failure_lines)
    )


def run_selected_specs(
    spec_names: Iterable[str], *, jobs: int = DEFAULT_JOBS
) -> list[dict[str, Any]]:
    tasks = collect_workbook_tasks(spec_names)
    if not tasks:
        print("No visualization workbook tasks matched the selection.")
        return []

    results = _run_tasks(tasks, jobs=jobs)
    _raise_failures(results)
    return results


def run_spec(spec_name: str, *, jobs: int = 1) -> list[dict[str, Any]]:
    return run_selected_specs([spec_name], jobs=jobs)


def run_section(section: str, *, jobs: int = 1) -> list[dict[str, Any]]:
    return run_selected_specs(collect_spec_names(section=section), jobs=jobs)


def run_all(*, jobs: int = 1) -> list[dict[str, Any]]:
    return run_selected_specs(collect_spec_names(), jobs=jobs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper/pd visualization specs. By default the script executes "
            "all configured specs."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--section",
        choices=SECTIONS,
        help="Run all visualization specs for one section.",
    )
    group.add_argument(
        "--spec",
        choices=tuple(VIZ_SPECS.keys()),
        help="Run one explicit visualization spec.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all visualization specs. This is also the default.",
    )
    parser.add_argument(
        "--jobs",
        type=_parse_jobs,
        default=DEFAULT_JOBS,
        help=(
            "Number of workbook worker processes to use. "
            f"Default: {DEFAULT_JOBS}."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.spec:
        run_selected_specs(collect_spec_names(spec_name=args.spec), jobs=args.jobs)
    elif args.section:
        run_selected_specs(collect_spec_names(section=args.section), jobs=args.jobs)
    else:
        run_selected_specs(collect_spec_names(), jobs=args.jobs)


if __name__ == "__main__":
    main()
