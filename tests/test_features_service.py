"""Tests for current Extract-Features service behavior."""

from __future__ import annotations

import sys
import threading
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lfptensorpipe.app.features.extract_runner as extract_runner_module
from lfptensorpipe.app.alignment_service import alignment_paradigm_log_path
from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.features_service import (
    _aggregate_states,
    _detect_derived_type,
    _flatten_value_for_xlsx,
    _iter_feature_source_tables,
    _iter_raw_tables,
    _load_post_transform_modes,
    _normalize_slug,
    _save_table_xlsx,
    features_derivatives_log_path,
    features_panel_state,
    load_derive_defaults,
    run_extract_features,
    run_normalization,
)
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    indicator_from_log,
    read_run_log,
    write_run_log,
)
from lfptensorpipe.gui.shell.features_subset import MainWindowFeaturesSubsetMixin
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl


def _context(project: Path) -> RecordContext:
    return RecordContext(project_root=project, subject="sub-001", record="runA")


def _nested_df() -> pd.DataFrame:
    return pd.DataFrame(
        np.ones((3, 5), dtype=float),
        index=np.array([4.0, 8.0, 12.0], dtype=float),
        columns=np.linspace(0.0, 100.0, 5, dtype=float),
    )


def _time_gradient_df() -> pd.DataFrame:
    coords = np.linspace(0.0, 100.0, 5, dtype=float)
    return pd.DataFrame(
        np.tile(coords, (3, 1)),
        index=np.array([4.0, 8.0, 12.0], dtype=float),
        columns=coords,
    )


def test_features_region_names_accept_pair_region_columns() -> None:
    payload = pd.DataFrame(
        {
            "SNr-STN_in": [True, False],
            "STN-SNr_in": [0, 1],
            "STN-STN_in": [False, False],
            "Band": ["alpha", "beta"],
        }
    )

    assert MainWindowFeaturesSubsetMixin._region_names_from_payload(payload) == [
        "SNr-STN",
        "STN-SNr",
    ]


def test_run_extract_features_builds_derivative_outputs_from_alignment_raw(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    raw_table_path = resolver.alignment_root / slug / "default" / "na-raw.pkl"
    frame = pd.DataFrame(
        [
            {
                "Subject": context.subject,
                "Record": context.record,
                "Trial": slug,
                "Metric": "default",
                "Epoch": "epoch_000",
                "Channel": "CH1",
                "Value": _nested_df(),
            }
        ]
    )
    save_pkl(frame, raw_table_path)
    write_run_log(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path=str(resolver.alignment_root / slug),
            message="alignment ready",
        ),
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    ok, message = run_extract_features(
        context,
        paradigm_slug=slug,
        config_store=store,
        axes_by_metric={
            "default": {
                "bands": [{"name": "theta", "start": 4.0, "end": 8.0}],
                "times": [{"name": "early", "start": 0.0, "end": 50.0}],
            }
        },
    )
    assert ok, message

    out_root = resolver.features_root / slug / "default"
    for stem in ("na-raw", "mean-spectral", "mean-trace", "mean-scalar"):
        assert (out_root / f"{stem}.pkl").exists()
    assert not (out_root / "na-raw.xlsx").exists()
    assert not (out_root / "mean-spectral.xlsx").exists()
    assert not (out_root / "mean-trace.xlsx").exists()
    assert (out_root / "mean-scalar.xlsx").exists()

    assert (
        indicator_from_log(features_derivatives_log_path(resolver, paradigm_slug=slug))
        == "green"
    )


def test_run_extract_features_keeps_successful_metric_outputs_when_another_fails(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    default_path = resolver.alignment_root / slug / "default" / "na-raw.pkl"
    psi_path = resolver.alignment_root / slug / "psi" / "na-raw.pkl"
    save_pkl(pd.DataFrame([{"Value": _nested_df()}]), default_path)
    psi_path.parent.mkdir(parents=True, exist_ok=True)
    psi_path.write_text("not-a-pickle", encoding="utf-8")
    write_run_log(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path=str(resolver.alignment_root / slug),
            message="alignment ready",
        ),
    )

    ok, message = run_extract_features(
        context,
        paradigm_slug=slug,
        axes_by_metric={
            "default": {
                "bands": [{"name": "theta", "start": 4.0, "end": 8.0}],
                "times": [{"name": "early", "start": 0.0, "end": 50.0}],
            },
            "psi": {
                "bands": [],
                "times": [{"name": "early", "start": 0.0, "end": 50.0}],
            },
        },
    )
    assert not ok
    assert "errors=1" in message

    default_root = resolver.features_root / slug / "default"
    assert (default_root / "na-raw.pkl").exists()
    assert (default_root / "mean-spectral.pkl").exists()
    assert (default_root / "mean-trace.pkl").exists()
    assert (default_root / "mean-scalar.pkl").exists()

    log_payload = read_run_log(
        features_derivatives_log_path(resolver, paradigm_slug=slug)
    )
    assert isinstance(log_payload, dict)
    assert log_payload["params"]["metrics"] == ["default", "psi"]
    assert list(log_payload["params"]["axes_by_metric"].keys()) == ["default"]
    assert len(log_payload["params"]["errors"]) == 1
    assert "psi: load failed" in log_payload["params"]["errors"][0]


def test_run_extract_features_parallelizes_metric_tasks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    for metric_key in ("default", "psi"):
        raw_table_path = resolver.alignment_root / slug / metric_key / "na-raw.pkl"
        save_pkl(pd.DataFrame([{"Value": _nested_df()}]), raw_table_path)
    write_run_log(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path=str(resolver.alignment_root / slug),
            message="alignment ready",
        ),
    )

    barrier = threading.Barrier(2, timeout=5.0)
    overlap = threading.Event()
    original = extract_runner_module._extract_metric_outputs

    def _wrapped_extract_metric_outputs(**kwargs: object) -> object:
        try:
            barrier.wait()
            overlap.set()
        except threading.BrokenBarrierError:
            pass
        return original(**kwargs)

    with patch.object(
        extract_runner_module,
        "_extract_metric_outputs",
        side_effect=_wrapped_extract_metric_outputs,
    ):
        ok, message = run_extract_features(
            context,
            paradigm_slug=slug,
            axes_by_metric={
                "default": {
                    "bands": [{"name": "theta", "start": 4.0, "end": 8.0}],
                    "times": [{"name": "early", "start": 0.0, "end": 50.0}],
                },
                "psi": {
                    "bands": [],
                    "times": [{"name": "early", "start": 0.0, "end": 50.0}],
                },
            },
        )

    assert ok, message
    assert overlap.is_set()


def test_run_extract_features_honors_metric_output_overrides(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    raw_table_path = resolver.alignment_root / slug / "default" / "na-raw.pkl"
    frame = pd.DataFrame([{"Value": _nested_df()}])
    save_pkl(frame, raw_table_path)
    write_run_log(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path=str(resolver.alignment_root / slug),
            message="alignment ready",
        ),
    )

    ok, message = run_extract_features(
        context,
        paradigm_slug=slug,
        axes_by_metric={"default": {"bands": [], "times": []}},
        enabled_outputs_by_metric={
            "default": {
                "raw": True,
                "spectral": False,
                "trace": False,
                "scalar": False,
            }
        },
        reducer_by_metric={"default": "occupation"},
    )
    assert ok, message

    out_root = resolver.features_root / slug / "default"
    assert (out_root / "na-raw.pkl").exists()
    assert not (out_root / "na-raw.xlsx").exists()
    assert not (out_root / "occupation-spectral.pkl").exists()
    assert not (out_root / "occupation-trace.pkl").exists()
    assert not (out_root / "occupation-scalar.pkl").exists()


def test_run_extract_features_removes_stale_non_scalar_xlsx_outputs(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    raw_table_path = resolver.alignment_root / slug / "default" / "na-raw.pkl"
    frame = pd.DataFrame([{"Value": _nested_df()}])
    save_pkl(frame, raw_table_path)
    write_run_log(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path=str(resolver.alignment_root / slug),
            message="alignment ready",
        ),
    )

    out_root = resolver.features_root / slug / "default"
    out_root.mkdir(parents=True, exist_ok=True)
    for stale_name in ("na-raw.xlsx", "mean-spectral.xlsx", "mean-trace.xlsx"):
        (out_root / stale_name).write_text("stale", encoding="utf-8")

    ok, message = run_extract_features(
        context,
        paradigm_slug=slug,
        axes_by_metric={
            "default": {
                "bands": [{"name": "theta", "start": 4.0, "end": 8.0}],
                "times": [{"name": "early", "start": 0.0, "end": 50.0}],
            }
        },
    )
    assert ok, message

    assert not (out_root / "na-raw.xlsx").exists()
    assert not (out_root / "mean-spectral.xlsx").exists()
    assert not (out_root / "mean-trace.xlsx").exists()
    assert (out_root / "mean-scalar.xlsx").exists()


def test_run_extract_features_auto_bands_from_raw_value_index_for_psi(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    raw_table_path = resolver.alignment_root / slug / "psi" / "na-raw.pkl"
    frame = pd.DataFrame([{"Value": _nested_df()}])
    save_pkl(frame, raw_table_path)
    write_run_log(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path=str(resolver.alignment_root / slug),
            message="alignment ready",
        ),
    )

    ok, message = run_extract_features(
        context,
        paradigm_slug=slug,
        axes_by_metric={
            "psi": {
                "bands": [],
                "times": [{"name": "early", "start": 0.0, "end": 50.0}],
            }
        },
    )
    assert ok, message

    out_root = resolver.features_root / slug / "psi"
    assert (out_root / "mean-trace.pkl").exists()
    assert (out_root / "mean-scalar.pkl").exists()


def test_run_extract_features_groups_repeated_time_rows_into_one_phase(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    raw_table_path = resolver.alignment_root / slug / "default" / "na-raw.pkl"
    frame = pd.DataFrame([{"Value": _time_gradient_df()}])
    save_pkl(frame, raw_table_path)
    write_run_log(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path=str(resolver.alignment_root / slug),
            message="alignment ready",
        ),
    )

    ok, message = run_extract_features(
        context,
        paradigm_slug=slug,
        axes_by_metric={
            "default": {
                "bands": [{"name": "beta", "start": 4.0, "end": 13.0}],
                "times": [
                    {"name": "strike_L", "start": 0.0, "end": 50.0},
                    {"name": "stance_L", "start": 0.0, "end": 75.0},
                    {"name": "strike_L", "start": 50.0, "end": 100.0},
                ],
            }
        },
    )
    assert ok, message

    scalar = load_pkl(resolver.features_root / slug / "default" / "mean-scalar.pkl")
    assert isinstance(scalar, pd.DataFrame)
    assert scalar["Phase"].astype(str).tolist() == ["strike_L", "stance_L"]
    strike_value = float(scalar.loc[scalar["Phase"] == "strike_L", "Value"].iloc[0])
    stance_value = float(scalar.loc[scalar["Phase"] == "stance_L", "Value"].iloc[0])
    assert np.isclose(strike_value, 37.5)
    assert np.isclose(stance_value, 25.0)

    log_payload = read_run_log(
        features_derivatives_log_path(resolver, paradigm_slug=slug)
    )
    assert isinstance(log_payload, dict)
    assert log_payload["params"]["axes_by_metric"]["default"]["times"] == [
        {"name": "strike_L", "start": 0.0, "end": 50.0},
        {"name": "stance_L", "start": 0.0, "end": 75.0},
        {"name": "strike_L", "start": 50.0, "end": 100.0},
    ]


def test_features_panel_indicator_tracks_axes_staleness_and_restore(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"
    axes_payload = {
        "raw_power": {
            "bands": [{"name": "beta", "start": 13.0, "end": 30.0}],
            "times": [{"name": "all", "start": 0.0, "end": 100.0}],
        },
        "psi": {
            "bands": [],
            "times": [{"name": "all", "start": 0.0, "end": 100.0}],
        },
    }
    write_run_log(
        features_derivatives_log_path(resolver, paradigm_slug=slug),
        RunLogRecord(
            step="run_extract_features",
            completed=True,
            params={"axes_by_metric": axes_payload},
            input_path="in",
            output_path="out",
            message="features ready",
        ),
    )
    assert (
        features_panel_state(
            resolver,
            paradigm_slug=slug,
            axes_by_metric=axes_payload,
        )
        == "green"
    )
    stale_axes = {
        **axes_payload,
        "raw_power": {
            "bands": [{"name": "beta", "start": 14.0, "end": 30.0}],
            "times": [{"name": "all", "start": 0.0, "end": 100.0}],
        },
    }
    assert (
        features_panel_state(
            resolver,
            paradigm_slug=slug,
            axes_by_metric=stale_axes,
        )
        == "yellow"
    )
    assert (
        features_panel_state(
            resolver,
            paradigm_slug=slug,
            axes_by_metric=axes_payload,
        )
        == "green"
    )


def test_run_extract_features_burst_reducer_matrix_by_alignment_method(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    expected_reducers_by_method = {
        "linear_warper": {"mean", "occupation"},
        "stack_warper": {"mean", "occupation"},
        "pad_warper": {"mean", "occupation", "rate", "duration"},
        "concat_warper": {"mean", "occupation", "rate", "duration"},
    }

    for method_key, expected_reducers in expected_reducers_by_method.items():
        slug = f"trial-{method_key.replace('_', '-')}"
        raw_table_path = resolver.alignment_root / slug / "burst" / "na-raw.pkl"
        frame = pd.DataFrame([{"Value": _nested_df()}])
        save_pkl(frame, raw_table_path)
        write_run_log(
            alignment_paradigm_log_path(resolver, slug),
            RunLogRecord(
                step="run_align_epochs",
                completed=True,
                params={
                    "trial_slug": slug,
                    "method": method_key,
                    "method_params": {"sample_rate": 0.4},
                },
                input_path="in",
                output_path=str(resolver.alignment_root / slug),
                message="alignment ready",
            ),
        )

        ok, message = run_extract_features(
            context,
            paradigm_slug=slug,
            axes_by_metric={
                "burst": {
                    "bands": [],
                    "times": [{"name": "early", "start": 0.0, "end": 50.0}],
                }
            },
        )
        assert ok, message

        out_root = resolver.features_root / slug / "burst"
        actual_files = {path.name for path in out_root.glob("*.pkl")}
        expected_files = {"na-raw.pkl"} | {
            f"{reducer}-scalar.pkl" for reducer in expected_reducers
        }
        assert actual_files == expected_files


def test_run_extract_features_alignment_and_axis_guards(tmp_path: Path) -> None:
    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    raw_table_path = resolver.alignment_root / slug / "default" / "na-raw.pkl"
    save_pkl(
        pd.DataFrame([{"Value": _nested_df()}]),
        raw_table_path,
    )

    # alignment log missing/gray
    ok, message = run_extract_features(
        context,
        paradigm_slug=slug,
        config_store=None,
        axes_by_metric={
            "default": {
                "bands": [{"name": "theta", "start": 4.0, "end": 8.0}],
                "times": [{"name": "early", "start": 0.0, "end": 50.0}],
            }
        },
    )
    assert not ok
    assert "alignment" in message.lower()

    write_run_log(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={},
            input_path="in",
            output_path=str(resolver.alignment_root / slug),
            message="alignment ready",
        ),
    )

    # missing times axis when default enables spectral/scalar
    ok, message = run_extract_features(
        context,
        paradigm_slug=slug,
        config_store=None,
        axes_by_metric={
            "default": {
                "bands": [{"name": "theta", "start": 4.0, "end": 8.0}],
                "times": [],
            }
        },
    )
    assert not ok
    assert "failed" in message.lower()


def test_helper_slug_and_compat_iterators(tmp_path: Path) -> None:
    assert _normalize_slug("  ") == ""
    assert _normalize_slug("Gait_Trial") == "gait-trial"

    context = _context(tmp_path)
    resolver = PathResolver(context)
    slug = "trial-gait"

    # compatibility helper now scans alignment/{trial}/{metric}/na-raw.pkl
    p = resolver.alignment_root / slug / "metric_a" / "na-raw.pkl"
    save_pkl(pd.DataFrame({"Value": [pd.Series([1.0])]}), p)
    assert _iter_raw_tables(resolver, paradigm_slug=slug) == [
        ("metric_a", "default", p)
    ]


def test_helper_detect_xlsx_and_modes(tmp_path: Path) -> None:
    assert _flatten_value_for_xlsx(pd.DataFrame({"a": [1]})).startswith("{")
    assert _flatten_value_for_xlsx(pd.Series([1, 2])).startswith("{")
    assert _flatten_value_for_xlsx(None) == ""
    assert _flatten_value_for_xlsx(float("nan")) == ""
    assert _flatten_value_for_xlsx(12) == "12"

    out_ok = tmp_path / "ok.xlsx"
    ok_save, message = _save_table_xlsx(pd.DataFrame({"Value": [1]}), out_ok)
    assert isinstance(ok_save, bool)
    assert isinstance(message, str)
    assert out_ok.exists()
    with zipfile.ZipFile(out_ok, "r") as archive:
        names = set(archive.namelist())
    assert "[Content_Types].xml" in names
    assert "xl/workbook.xml" in names

    assert (
        _detect_derived_type(pd.DataFrame({"Value": [pd.Series([1.0, 2.0])]}))
        == "trace"
    )
    assert (
        _detect_derived_type(pd.DataFrame({"Value": [pd.DataFrame(np.ones((2, 2)))]}))
        == "raw"
    )
    assert _detect_derived_type(pd.DataFrame({"Value": [1.0, 2.0]})) == "scalar"

    features_root = tmp_path / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    good = features_root / "a.pkl"
    bad = features_root / "bad.pkl"
    save_pkl(pd.DataFrame({"Value": [pd.Series([1.0])]}), good)
    bad.write_text("not-a-pickle", encoding="utf-8")
    tables, errors = _iter_feature_source_tables(features_root)
    assert len(tables) == 1
    assert errors

    defaults = _load_post_transform_modes(None)
    assert defaults == {
        "raw": "none",
        "trace": "none",
        "scalar": "none",
        "spectral": "none",
    }


def test_run_normalization_is_disabled(tmp_path: Path) -> None:
    context = _context(tmp_path)
    ok, message = run_normalization(
        context,
        paradigm_slug="trial-gait",
        baseline_phase="All",
        mode="mean",
    )
    assert not ok
    assert "disabled" in message.lower()


def test_load_derive_defaults_and_state_aggregator(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    payload = load_derive_defaults(store)
    assert "derive_param_cfg" in payload
    assert "reducer_cfg" in payload
    assert "collapse_base_cfg" in payload
    assert "post_transform_mode" in payload
    assert "plot_advance_defaults" in payload

    assert _aggregate_states([]) == "gray"
