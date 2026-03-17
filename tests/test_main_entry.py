"""Tests for the app main entrypoint and embedded-worker dispatch."""

from __future__ import annotations

import lfptensorpipe.main as main_module
from lfptensorpipe.main import main


def test_main_dispatches_tensor_worker_without_starting_gui() -> None:
    calls: dict[str, object] = {}

    def _fake_tensor_worker(argv: list[str] | None) -> int:
        calls["argv"] = argv
        return 7

    exit_code = main(
        ["--run-tensor-worker", "--request", "/tmp/request.json"],
        tensor_worker_main=_fake_tensor_worker,
    )

    assert exit_code == 7
    assert calls["argv"] == ["--request", "/tmp/request.json"]


def test_main_dispatches_runtime_plan_worker_without_starting_gui() -> None:
    calls: dict[str, object] = {}

    def _fake_runtime_worker(argv: list[str] | None) -> int:
        calls["argv"] = argv
        return 9

    exit_code = main(
        ["--run-runtime-plan-worker", "--request", "/tmp/request.json"],
        runtime_plan_worker_main=_fake_runtime_worker,
    )

    assert exit_code == 9
    assert calls["argv"] == ["--request", "/tmp/request.json"]


def test_main_dispatches_smoke_raw_plot_without_starting_main_window() -> None:
    calls: dict[str, object] = {}

    def _fake_smoke_runner(raw_fif_path: str, close_ms: int) -> int:
        calls["raw_fif_path"] = raw_fif_path
        calls["close_ms"] = close_ms
        return 11

    exit_code = main(
        [
            "--smoke-raw-plot-fif",
            "/tmp/raw.fif",
            "--smoke-raw-plot-close-ms",
            "250",
        ],
        smoke_raw_plot_main=_fake_smoke_runner,
    )

    assert exit_code == 11
    assert calls["raw_fif_path"] == "/tmp/raw.fif"
    assert calls["close_ms"] == 250


def test_main_dispatches_demo_record_smoke_without_starting_main_window() -> None:
    calls: dict[str, object] = {}

    def _fake_demo_records_runner(records_root: str) -> int:
        calls["records_root"] = records_root
        return 13

    exit_code = main(
        [
            "--smoke-demo-records-root",
            "/tmp/demo/records",
        ],
        smoke_demo_records_main=_fake_demo_records_runner,
    )

    assert exit_code == 13
    assert calls["records_root"] == "/tmp/demo/records"


def test_main_recovers_when_console_streams_are_none(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def _fake_demo_records_runner(records_root: str) -> int:
        calls["records_root"] = records_root
        return 31

    monkeypatch.setattr(main_module.sys, "stdout", None)
    monkeypatch.setattr(main_module.sys, "stderr", None)

    exit_code = main(
        [
            "--smoke-demo-records-root",
            "/tmp/demo/records",
        ],
        smoke_demo_records_main=_fake_demo_records_runner,
    )

    assert exit_code == 31
    assert calls["records_root"] == "/tmp/demo/records"
    assert main_module.sys.stdout is not None
    assert hasattr(main_module.sys.stdout, "write")
    assert main_module.sys.stderr is not None
    assert hasattr(main_module.sys.stderr, "write")


def test_main_dispatches_demo_record_import_smoke_without_starting_main_window() -> (
    None
):
    calls: dict[str, object] = {}

    def _fake_demo_record_imports_runner(records_root: str) -> int:
        calls["records_root"] = records_root
        return 17

    exit_code = main(
        [
            "--smoke-demo-record-imports-root",
            "/tmp/demo/records",
        ],
        smoke_demo_record_imports_main=_fake_demo_record_imports_runner,
    )

    assert exit_code == 17
    assert calls["records_root"] == "/tmp/demo/records"


def test_main_dispatches_demo_config_smoke_without_starting_main_window() -> None:
    calls: dict[str, object] = {}

    def _fake_demo_configs_runner(
        configs_root: str,
        project_root: str,
        subject: str,
        record: str,
    ) -> int:
        calls["configs_root"] = configs_root
        calls["project_root"] = project_root
        calls["subject"] = subject
        calls["record"] = record
        return 19

    exit_code = main(
        [
            "--smoke-demo-configs-root",
            "/tmp/demo/configs",
            "--smoke-project-root",
            "/tmp/project",
            "--smoke-subject",
            "sub-001",
            "--smoke-record",
            "gait",
        ],
        smoke_demo_configs_main=_fake_demo_configs_runner,
    )

    assert exit_code == 19
    assert calls == {
        "configs_root": "/tmp/demo/configs",
        "project_root": "/tmp/project",
        "subject": "sub-001",
        "record": "gait",
    }


def test_main_dispatches_preproc_ui_smoke_without_starting_main_window() -> None:
    calls: dict[str, object] = {}

    def _fake_preproc_ui_runner(project_root: str, subject: str, record: str) -> int:
        calls["project_root"] = project_root
        calls["subject"] = subject
        calls["record"] = record
        return 23

    exit_code = main(
        [
            "--smoke-preproc-ui",
            "--smoke-project-root",
            "/tmp/project",
            "--smoke-subject",
            "sub-001",
            "--smoke-record",
            "gait",
        ],
        smoke_preproc_ui_main=_fake_preproc_ui_runner,
    )

    assert exit_code == 23
    assert calls == {
        "project_root": "/tmp/project",
        "subject": "sub-001",
        "record": "gait",
    }


def test_main_dispatches_numerical_preproc_smoke_without_starting_main_window() -> None:
    calls: dict[str, object] = {}

    def _fake_numerical_preproc_runner(
        reference_root: str,
        project_root: str,
        subject: str,
        records_root: str,
    ) -> int:
        calls["reference_root"] = reference_root
        calls["project_root"] = project_root
        calls["subject"] = subject
        calls["records_root"] = records_root
        return 29

    exit_code = main(
        [
            "--smoke-numerical-preproc",
            "--smoke-reference-root",
            "/tmp/reference",
            "--smoke-project-root",
            "/tmp/project",
            "--smoke-subject",
            "sub-001",
            "--smoke-numerical-records-root",
            "/tmp/demo/records",
        ],
        smoke_numerical_preproc_main=_fake_numerical_preproc_runner,
    )

    assert exit_code == 29
    assert calls == {
        "reference_root": "/tmp/reference",
        "project_root": "/tmp/project",
        "subject": "sub-001",
        "records_root": "/tmp/demo/records",
    }


def test_main_dispatches_tensor_runtime_smoke_without_starting_main_window() -> None:
    calls: dict[str, object] = {}

    def _fake_tensor_runtime_runner(
        configs_root: str,
        project_root: str,
        subject: str,
        record: str,
    ) -> int:
        calls["configs_root"] = configs_root
        calls["project_root"] = project_root
        calls["subject"] = subject
        calls["record"] = record
        return 30

    exit_code = main(
        [
            "--smoke-tensor-runtime",
            "--smoke-demo-configs-root",
            "/tmp/demo/configs",
            "--smoke-project-root",
            "/tmp/project",
            "--smoke-subject",
            "sub-001",
            "--smoke-record",
            "gait",
        ],
        smoke_tensor_runtime_main=_fake_tensor_runtime_runner,
    )

    assert exit_code == 30
    assert calls == {
        "configs_root": "/tmp/demo/configs",
        "project_root": "/tmp/project",
        "subject": "sub-001",
        "record": "gait",
    }


def test_main_dispatches_numerical_full_pipeline_smoke_without_starting_main_window() -> (
    None
):
    calls: dict[str, object] = {}

    def _fake_numerical_full_pipeline_runner(
        reference_root: str,
        project_root: str,
        subject: str,
        records_root: str,
        configs_root: str,
    ) -> int:
        calls["reference_root"] = reference_root
        calls["project_root"] = project_root
        calls["subject"] = subject
        calls["records_root"] = records_root
        calls["configs_root"] = configs_root
        return 31

    exit_code = main(
        [
            "--smoke-numerical-full-pipeline",
            "--smoke-reference-root",
            "/tmp/reference",
            "--smoke-project-root",
            "/tmp/project",
            "--smoke-subject",
            "sub-001",
            "--smoke-numerical-records-root",
            "/tmp/demo/records",
            "--smoke-numerical-configs-root",
            "/tmp/demo/configs",
        ],
        smoke_numerical_full_pipeline_main=_fake_numerical_full_pipeline_runner,
    )

    assert exit_code == 31
    assert calls == {
        "reference_root": "/tmp/reference",
        "project_root": "/tmp/project",
        "subject": "sub-001",
        "records_root": "/tmp/demo/records",
        "configs_root": "/tmp/demo/configs",
    }
