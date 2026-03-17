"""Integration checks for dataset selection and stage sync in MainWindow."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LFPTP_DISABLE_MATLAB_WARMUP", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.runlog_store import RunLogRecord, write_run_log
from lfptensorpipe.gui.main_window import LocalizeAtlasDialog, MainWindow
import lfptensorpipe.gui.main_window as main_window_module


class OverrideMainWindow(MainWindow):
    def __init__(
        self, *args, overrides: dict[str, Any] | None = None, **kwargs
    ) -> None:
        self._overrides = dict(overrides or {})
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name: str) -> Any:
        if name not in {"_overrides", "__dict__", "__class__", "__getattribute__"}:
            try:
                overrides = object.__getattribute__(self, "_overrides")
            except AttributeError:
                overrides = {}
            if name in overrides:
                value = overrides[name]
                if hasattr(value, "__get__"):
                    return value.__get__(self, type(self))
                return value
        return super().__getattribute__(name)


def test_mainwindow_loads_recent_project_and_syncs_stage_states(tmp_path: Path) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    write_run_log(
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "finish"
        / "lfptensorpipe_log.json",
        RunLogRecord(step="finish", completed=True),
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(config_store=store, demo_data_root=tmp_path / "missing_demo")

    assert window._current_project == project
    assert window._current_subject == subject
    assert window._current_record is None
    assert window._stage_states["preproc"] == "gray"
    assert window._project_add_button.isEnabled()
    assert window._subject_add_button.isEnabled()
    assert window._record_add_button.isEnabled()
    assert not window._record_delete_button.isEnabled()
    assert not window._stage_buttons["tensor"].isEnabled()
    assert not window._stage_buttons["alignment"].isEnabled()
    assert not window._localize_apply_button.isEnabled()
    assert not window._localize_import_button.isEnabled()
    assert not window._localize_export_button.isEnabled()
    assert not window._contact_viewer_button.isEnabled()

    assert window._select_record_item(record)
    assert window._current_record == record
    assert window._stage_states["preproc"] == "green"
    assert window._record_delete_button.isEnabled()

    window.close()


def test_mainwindow_record_selection_upgrades_record_logs_before_refresh(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    write_run_log(
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "finish"
        / "lfptensorpipe_log.json",
        RunLogRecord(step="finish", completed=True),
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    calls: list[tuple[Path, str, str]] = []

    def _upgrade_record_run_logs_runtime(
        self,
        project_root: Path,
        selected_subject: str,
        selected_record: str,
    ) -> Any:
        calls.append((project_root, selected_subject, selected_record))
        return SimpleNamespace(scanned_count=1, upgraded_count=2, failed_count=0)

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        overrides={
            "_upgrade_record_run_logs_runtime": _upgrade_record_run_logs_runtime,
        },
    )

    assert window._select_record_item(record)
    assert calls == [(project, subject, record)]
    assert "Upgraded 2 log(s)" in window.statusBar().currentMessage()

    window.close()


def test_mainwindow_record_switch_autosaves_outgoing_record_snapshot(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    for record in ("runA", "runB"):
        (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(
            parents=True, exist_ok=True
        )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    persist_reasons: list[str] = []
    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        overrides={
            "_persist_record_params_snapshot": (
                lambda self, *, reason: persist_reasons.append(reason) or True
            ),
            "_persist_record_params_snapshot_on_close": lambda self: None,
        },
    )

    assert window._select_record_item("runA")
    assert window._current_record == "runA"
    assert persist_reasons == []

    assert window._select_record_item("runB")
    assert window._current_record == "runB"
    assert persist_reasons == ["record_context_switch"]

    window.close()


def test_mainwindow_stage_entry_states_block_downstream_green_results(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    write_run_log(
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "preproc"
        / "finish"
        / "lfptensorpipe_log.json",
        RunLogRecord(step="finish", completed=False),
    )
    write_run_log(
        project
        / "derivatives"
        / "lfptensorpipe"
        / subject
        / record
        / "tensor"
        / "raw_power"
        / "lfptensorpipe_log.json",
        RunLogRecord(step="raw_power", completed=True),
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    window = MainWindow(config_store=store, demo_data_root=tmp_path / "missing_demo")

    assert window._select_record_item(record)
    assert window._stage_raw_states["preproc"] == "yellow"
    assert window._stage_raw_states["tensor"] == "green"
    assert window._stage_states["preproc"] == "yellow"
    assert window._stage_states["tensor"] == "yellow"
    assert not window._stage_buttons["tensor"].isEnabled()
    assert not window._stage_buttons["alignment"].isEnabled()
    assert not window._stage_buttons["features"].isEnabled()

    window.close()


def test_localize_atlas_dialog_set_default_persists_localization_defaults(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    default_space = "MNI152NLin2009bAsym"
    default_atlas = "DISTAL Nano (Ewert 2017)"
    atlas_b = "Atlas B"
    region_names = {
        default_atlas: ("SNr", "STN"),
        atlas_b: ("GPe", "GPi"),
    }

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    dialog = LocalizeAtlasDialog(
        space=default_space,
        atlas_names=(default_atlas, atlas_b),
        current_atlas=default_atlas,
        current_selected_regions=("SNr",),
        region_loader=lambda atlas: region_names[atlas],
        config_store=store,
    )
    atlas_b_index = dialog._atlas_combo.findData(atlas_b)
    assert atlas_b_index >= 0
    dialog._atlas_combo.setCurrentIndex(atlas_b_index)

    assert dialog._current_atlas() == atlas_b
    assert dialog._current_selected_regions() == region_names[atlas_b]

    dialog._on_set_default()

    saved = store.read_yaml("localization.yml", default={})
    assert saved["space_localize_defaults"][default_space] == {
        "atlas": atlas_b,
        "selected_regions": ["GPe", "GPi"],
    }
    assert saved["match_defaults"] == {
        "channels": [],
        "lead_signature": [],
        "mappings": [],
    }


def test_localize_atlas_dialog_restore_default_restores_saved_or_fallback_atlas(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    default_space = "MNI152NLin2009bAsym"
    atlas_a = "Atlas A"
    atlas_b = "Atlas B"
    region_names = {
        atlas_a: ("SNr", "STN"),
        atlas_b: ("GPe", "GPi"),
    }

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "localization.yml",
        {
            "space_localize_defaults": {
                default_space: {
                    "atlas": atlas_b,
                    "selected_regions": ["GPi"],
                }
            }
        },
    )

    dialog = LocalizeAtlasDialog(
        space=default_space,
        atlas_names=(atlas_a, atlas_b),
        current_atlas=atlas_a,
        current_selected_regions=("SNr",),
        region_loader=lambda atlas: region_names[atlas],
        config_store=store,
    )
    assert dialog._current_atlas() == atlas_a
    assert dialog._current_selected_regions() == ("SNr",)

    dialog._on_restore_default()
    assert dialog._current_atlas() == atlas_b
    assert dialog._current_selected_regions() == ("GPi",)
    assert "restored" in dialog._status_label.text().lower()

    store.write_yaml(
        "localization.yml",
        {
            "space_localize_defaults": {
                default_space: {
                    "atlas": "Missing Atlas",
                    "selected_regions": ["Unknown"],
                }
            }
        },
    )
    dialog._on_restore_default()
    assert dialog._current_atlas() == atlas_a
    assert dialog._current_selected_regions() == region_names[atlas_a]
    assert "first available atlas" in dialog._status_label.text().lower()


def test_localize_atlas_dialog_manual_region_toggle_refreshes_status_and_save(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()

    dialog = LocalizeAtlasDialog(
        space="MNI152NLin2009bAsym",
        atlas_names=("Atlas A",),
        current_atlas="Atlas A",
        current_selected_regions=("SNr", "STN"),
        region_loader=lambda _atlas: ("SNr", "STN"),
        config_store=store,
    )

    dialog._on_clear()
    assert dialog._status_label.text() == (
        "Status: 0/2 selected At least one region must be selected."
    )
    assert not dialog._save_button.isEnabled()

    item = dialog._region_list.item(0)
    assert item is not None
    item.setCheckState(Qt.Checked)
    app.processEvents()

    assert dialog._status_label.text() == "Status: 1/2 selected"
    assert dialog._save_button.isEnabled()


def test_mainwindow_project_and_subject_add_action_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    project.mkdir(parents=True, exist_ok=True)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml("recent_projects.yml", {"recent_projects": []})

    warnings: list[str] = []
    selected_directory = {"value": ""}
    subject_prompt = {"value": (" sub-001 ", True)}
    create_subject_result = {"value": (False, "create failed")}
    subjects_state = {"value": ["sub-001", "sub-002"]}

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        overrides={
            "_show_warning": lambda self, _title, message: warnings.append(message)
            or 0,
            "_select_existing_directory": lambda self, _title, _start_dir: selected_directory[
                "value"
            ],
            "_prompt_text": lambda self, _title, _label: subject_prompt["value"],
            "_create_subject_runtime": lambda self, _project_root, _subject: create_subject_result[
                "value"
            ],
            "_discover_subjects_runtime": lambda self, _project_root: list(
                subjects_state["value"]
            ),
        },
    )

    missing_project = tmp_path / "missing_project"
    selected_directory["value"] = str(missing_project)
    window._on_project_add()
    assert any("does not exist" in item for item in warnings)

    selected_directory["value"] = str(project)
    window._on_project_add()
    assert window._project_combo is not None
    assert window._project_combo.currentData() == str(project.resolve())
    assert "Project added" in window.statusBar().currentMessage()

    window._current_project = None
    window._on_subject_add()
    assert any("Select a project first." in item for item in warnings)

    window._current_project = project
    create_subject_result["value"] = (False, "create failed")
    window._on_subject_add()
    assert any("create failed" in item for item in warnings)

    create_subject_result["value"] = (True, "Subject created: sub-001")
    window._on_subject_add()
    assert window._subject_combo is not None
    assert window._subject_combo.currentData() == "sub-001"
    assert "Subject created: sub-001" in window.statusBar().currentMessage()

    window.close()


def test_mainwindow_record_add_action_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    project.mkdir(parents=True, exist_ok=True)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    warnings: list[str] = []
    dialog_default_types: list[str] = []
    apply_calls: list[dict[str, object]] = []
    import_calls: list[dict[str, object]] = []
    preview_source = project / "demo.json"
    state: dict[str, object | None] = {
        "dialog_exec": QDialog.Rejected,
        "dialog_preview": None,
        "dialog_record_name": "runC",
        "dialog_import_type": "Medtronic",
        "dialog_use_reset": False,
        "dialog_reset_rows": (),
        "apply_error": None,
        "records": [],
        "import_result": SimpleNamespace(ok=False, message="import failed"),
    }

    def _fake_warning(parent, title, text, *args, **kwargs):  # noqa: ANN001
        _ = (parent, title, args, kwargs)
        message = str(text)
        warnings.append(message)
        return 0

    class _FakeImportDialog:
        def __init__(
            self,
            *,
            project_root: Path,
            existing_records: tuple[str, ...],
            default_import_type: str,
            config_store: AppConfigStore,
            parent=None,
        ) -> None:
            _ = (project_root, existing_records, config_store, parent)
            dialog_default_types.append(default_import_type)

        def exec(self) -> int:
            return int(state["dialog_exec"])

        @property
        def parsed_preview(self):
            return state["dialog_preview"]

        @property
        def selected_record_name(self) -> str:
            return str(state["dialog_record_name"])

        @property
        def selected_import_type(self) -> str:
            return str(state["dialog_import_type"])

        @property
        def use_reset_reference(self) -> bool:
            return bool(state["dialog_use_reset"])

        @property
        def reset_rows(self):
            return tuple(state["dialog_reset_rows"])

    def _fake_apply_reset_reference(raw, rows):
        apply_calls.append({"raw": raw, "rows": tuple(rows)})
        apply_error = state["apply_error"]
        if apply_error is not None:
            raise RuntimeError(str(apply_error))
        return "RAW_RESET"

    def _fake_import_record_from_raw(**kwargs):
        import_calls.append(dict(kwargs))
        return state["import_result"]

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        overrides={
            "_show_warning": lambda self, _title, message: _fake_warning(
                None, "", message
            ),
            "_create_record_import_dialog": lambda self, **kwargs: _FakeImportDialog(
                **kwargs
            ),
            "_apply_reset_reference_runtime": lambda self, raw, rows: _fake_apply_reset_reference(
                raw, rows
            ),
            "_import_record_from_raw_runtime": lambda self, **kwargs: _fake_import_record_from_raw(
                **kwargs
            ),
            "_discover_records_runtime": lambda self, _project_root, _subject: list(
                state["records"]
            ),
            "_run_with_busy": lambda self, _label, work: work(),
        },
    )
    window._current_project = project
    window._current_subject = subject

    window._current_project = None
    window._current_subject = None
    window._on_record_add()
    assert any("Select project and subject first." in item for item in warnings)
    window._current_project = project
    window._current_subject = subject

    state["dialog_exec"] = QDialog.Rejected
    window._on_record_add()
    assert dialog_default_types[-1] == "Medtronic"

    state["dialog_exec"] = QDialog.Accepted
    state["dialog_preview"] = None
    window._on_record_add()
    assert any("Parse result is missing." in item for item in warnings)

    state["dialog_preview"] = SimpleNamespace(
        raw="RAW_ORIGINAL",
        source_path=preview_source,
        is_fif_input=False,
    )
    state["dialog_record_name"] = "bad record"
    window._on_record_add()
    assert any("Record must match pattern" in item for item in warnings)

    state["dialog_record_name"] = "runA"
    state["dialog_use_reset"] = True
    state["dialog_reset_rows"] = (
        main_window_module.ResetReferenceRow(
            anode="A",
            cathode="B",
            name="A_B",
        ),
    )
    state["apply_error"] = "apply failed"
    window._on_record_add()
    assert any("Failed to apply reset reference" in item for item in warnings)
    assert apply_calls

    state["apply_error"] = None
    state["import_result"] = SimpleNamespace(ok=False, message="import failed")
    window._on_record_add()
    assert any("import failed" in item for item in warnings)

    state["dialog_use_reset"] = False
    state["dialog_reset_rows"] = ()
    state["dialog_import_type"] = "PINS"
    state["import_result"] = SimpleNamespace(ok=True, message="import ok")
    state["dialog_record_name"] = "runC"
    state["records"] = ["runA", "runB", "runC"]
    window._on_record_add()
    assert import_calls
    assert import_calls[-1]["record"] == "runC"
    assert import_calls[-1]["raw"] == "RAW_ORIGINAL"
    assert window._record_list is not None
    current_item = window._record_list.currentItem()
    assert current_item is not None
    assert current_item.text() == "runC"
    assert window._current_record == "runC"
    assert "import ok" in window.statusBar().currentMessage()
    saved_payload = store.read_yaml("recent_projects.yml", default={})
    assert saved_payload["record_import_defaults"]["last_import_type"] == "PINS"

    window.close()


def test_mainwindow_record_delete_action_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    project.mkdir(parents=True, exist_ok=True)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )

    warnings: list[str] = []
    question_queue: list[int] = []
    delete_calls: list[dict[str, object]] = []
    state: dict[str, object] = {
        "delete_result": SimpleNamespace(ok=False, message="delete failed"),
        "records": ["runB"],
    }
    empty_called = {"called": False}

    def _fake_warning(parent, title, text, *args, **kwargs):  # noqa: ANN001
        _ = (parent, title, args, kwargs)
        warnings.append(str(text))
        return 0

    def _fake_question(parent, title, text, *args, **kwargs):  # noqa: ANN001
        _ = (parent, title, text, args, kwargs)
        if question_queue:
            return question_queue.pop(0)
        return main_window_module.QMessageBox.No

    def _fake_delete_record(**kwargs):
        delete_calls.append(dict(kwargs))
        return state["delete_result"]

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        overrides={
            "_show_warning": lambda self, _title, message: _fake_warning(
                None, "", message
            ),
            "_ask_question": lambda self, _title, _message, **_kwargs: _fake_question(
                None, "", ""
            ),
            "_delete_record_runtime": lambda self, **kwargs: _fake_delete_record(
                **kwargs
            ),
            "_discover_records_runtime": lambda self, _project_root, _subject: list(
                state["records"]
            ),
            "_run_with_busy": lambda self, _label, work: work(),
            "_set_empty_record_context": lambda self: empty_called.__setitem__(
                "called", True
            ),
        },
    )
    window._current_project = project
    window._current_subject = subject
    window._current_record = "runA"

    window._current_project = None
    window._current_subject = None
    window._current_record = None
    window._on_record_delete()
    assert any(
        "Select project, subject, and record first." in item for item in warnings
    )

    window._current_project = project
    window._current_subject = subject
    window._current_record = "runA"
    question_queue[:] = [int(main_window_module.QMessageBox.No)]
    window._on_record_delete()
    assert not delete_calls

    question_queue[:] = [int(main_window_module.QMessageBox.Yes)]
    state["delete_result"] = SimpleNamespace(ok=False, message="delete failed")
    window._on_record_delete()
    assert any("delete failed" in item for item in warnings)

    question_queue[:] = [int(main_window_module.QMessageBox.Yes)]
    state["delete_result"] = SimpleNamespace(ok=True, message="delete ok")
    state["records"] = ["runB"]
    window._on_record_delete()
    assert delete_calls
    assert empty_called["called"]
    assert window._record_list is not None
    current_item = window._record_list.currentItem()
    assert current_item is None
    assert "delete ok" in window.statusBar().currentMessage()

    question_queue[:] = [int(main_window_module.QMessageBox.Yes)]
    state["records"] = []
    window._on_record_delete()
    assert empty_called["called"]

    window.close()


def test_mainwindow_dataset_context_helper_guard_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    demo_root = tmp_path / "demo_project"
    demo_root.mkdir(parents=True, exist_ok=True)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml("recent_projects.yml", {"recent_projects": []})

    appended: list[Path] = []

    def _append_recent_project(path: Path) -> list[str]:
        appended.append(path)
        return [str(path.resolve())]

    store.append_recent_project = _append_recent_project  # type: ignore[method-assign]
    window = OverrideMainWindow(
        config_store=store, demo_data_root=demo_root, enable_plots=False
    )

    assert appended and appended[-1] == demo_root

    assert window._project_combo is not None
    project_combo = window._project_combo
    window._project_combo = None
    window._on_project_changed(0)
    window._project_combo = project_combo

    window._on_project_changed(0)
    assert window._current_project is None
    assert window._current_subject is None

    missing_project = tmp_path / "missing_project"
    window._set_combo_values(
        window._project_combo, [str(missing_project)], "Select project"
    )
    window._on_project_changed(1)
    assert "Missing project path" in window.statusBar().currentMessage()

    assert window._subject_combo is not None
    subject_combo = window._subject_combo
    window._subject_combo = None
    window._on_subject_changed(0)
    window._subject_combo = subject_combo

    window._current_project = demo_root
    window._set_combo_values(window._subject_combo, [], "Select subject")
    window._on_subject_changed(0)
    assert window._current_subject is None

    assert window._record_list is not None
    record_list = window._record_list
    window._record_list = None
    window._on_record_changed()
    window._record_list = record_list

    window._current_project = demo_root
    window._current_subject = "sub-001"
    window._set_record_values([])
    window._on_record_changed()
    assert window._current_record is None

    window.close()


def test_mainwindow_localize_action_handler_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)
    (project / "rawdata" / subject / "ses-postop" / "lfp" / record / "raw").mkdir(
        parents=True
    )
    (project / "sourcedata" / subject / "lfp" / record).mkdir(parents=True)

    leaddbs_dir = tmp_path / "leaddbs"
    matlab_root = tmp_path / "matlab"
    (leaddbs_dir / "templates" / "space" / "MNI" / "atlases" / "AtlasA").mkdir(
        parents=True
    )
    matlab_root.mkdir(parents=True)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    store.write_yaml(
        "paths.yml",
        {
            "leaddbs_dir": str(leaddbs_dir),
            "matlab_root": str(matlab_root),
        },
    )

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
    )
    window._current_record = record

    # Non-dict defaults payload should safely fallback to empty defaults.
    window._config_store.read_yaml = (  # type: ignore[method-assign]
        lambda name, default=None: [] if name == "localization.yml" else default
    )
    assert window._load_localize_defaults() == {}

    # Atlas-config guard branch.
    window._localize_inferred_space = None
    window._localize_available_atlases = ()
    window._on_localize_atlas_configure()
    assert "Atlas Configure unavailable" in window.statusBar().currentMessage()

    # Missing context early-return branch.
    current_project = window._current_project
    window._current_project = None
    window._on_localize_apply()
    window._current_project = current_project

    # Invalid selector branch.
    window._on_localize_apply()

    # Valid selectors -> failure/success prefixes.
    window._localize_inferred_space = "MNI"
    window._localize_selected_atlas = "AtlasA"
    window._localize_selected_regions = ("SNr",)
    window._overrides["_ensure_matlab_ready_for_action"] = lambda self, _name: True
    window._overrides["_post_step_action_sync"] = lambda self, reason: None

    window._overrides["_run_with_busy"] = lambda self, _label, _work: (False, "err")
    window._on_localize_apply()
    assert "Localize Failed: err" in window.statusBar().currentMessage()

    window._overrides["_run_with_busy"] = lambda self, _label, _work: (True, "ok")
    window._on_localize_apply()
    assert "Localize OK: ok" in window.statusBar().currentMessage()

    # Contact viewer branches: invalid atlas, failed run, success run.
    window._localize_selected_atlas = None
    window._on_contact_viewer()
    assert "save atlas config first" in window.statusBar().currentMessage().lower()

    window._localize_selected_atlas = "AtlasA"

    window._overrides["_run_with_busy"] = lambda self, _label, _work: (
        False,
        "viewer fail",
    )
    window._on_contact_viewer()
    assert (
        "Contact Viewer unavailable: viewer fail" in window.statusBar().currentMessage()
    )

    window._overrides["_run_with_busy"] = lambda self, _label, _work: (
        True,
        "viewer ok",
    )
    window._on_contact_viewer()
    assert "viewer ok" in window.statusBar().currentMessage()

    window.close()


def test_mainwindow_settings_configs_save_updates_paths_and_refreshes(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)
    (project / "rawdata" / subject / "ses-postop" / "lfp" / record / "raw").mkdir(
        parents=True
    )
    (project / "sourcedata" / subject / "lfp" / record).mkdir(parents=True)

    old_leaddbs = tmp_path / "old_leaddbs"
    old_matlab = tmp_path / "old_matlab_engine"
    (old_leaddbs / "templates" / "space").mkdir(parents=True)
    old_matlab.mkdir(parents=True)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    store.write_yaml(
        "paths.yml",
        {
            "leaddbs_dir": str(old_leaddbs),
            "matlab_root": str(old_matlab),
        },
    )

    new_leaddbs = tmp_path / "new_leaddbs"
    new_matlab = tmp_path / "new_matlab_engine"
    (new_leaddbs / "templates" / "space" / "MNI" / "atlases" / "AtlasA").mkdir(
        parents=True
    )
    new_matlab.mkdir(parents=True)

    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(
            self, *, current_paths: dict[str, str], parent=None
        ) -> None:  # noqa: ANN001
            _ = parent
            captured["current_paths"] = dict(current_paths)
            self.selected_paths = {
                "leaddbs_dir": str(new_leaddbs),
                "matlab_root": str(new_matlab),
            }

        def exec(self) -> int:
            return QDialog.Accepted

    refresh_calls = {"count": 0}
    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
    )
    original_refresh = window._refresh_localize_controls

    def _tracked_refresh() -> None:
        refresh_calls["count"] += 1
        original_refresh()

    window._overrides["_create_paths_config_dialog"] = (
        lambda self, **kwargs: _FakeDialog(**kwargs)
    )
    window._overrides["_refresh_localize_controls"] = lambda self: _tracked_refresh()

    window._on_settings_configs()

    assert captured["current_paths"] == {
        "leaddbs_dir": str(old_leaddbs),
        "matlab_root": str(old_matlab),
    }
    saved = store.read_yaml("paths.yml", default={})
    assert saved["leaddbs_dir"] == str(new_leaddbs)
    assert saved["matlab_root"] == str(new_matlab)
    assert window._localize_paths.leaddbs_dir == new_leaddbs
    assert window._localize_paths.matlab_root == new_matlab
    assert refresh_calls["count"] == 1
    assert "Path settings saved to app storage." in window.statusBar().currentMessage()

    window.close()


def test_mainwindow_settings_configs_cancel_keeps_paths(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    original_payload = {
        "leaddbs_dir": str(tmp_path / "keep_leaddbs"),
        "matlab_root": str(tmp_path / "keep_matlab"),
    }
    store.write_yaml("paths.yml", dict(original_payload))
    store.write_yaml("recent_projects.yml", {"recent_projects": []})

    class _FakeDialog:
        def __init__(
            self, *, current_paths: dict[str, str], parent=None
        ) -> None:  # noqa: ANN001
            _ = (current_paths, parent)
            self.selected_paths = None

        def exec(self) -> int:
            return QDialog.Rejected

    refresh_calls = {"count": 0}
    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
    )
    window._overrides["_create_paths_config_dialog"] = (
        lambda self, **kwargs: _FakeDialog(**kwargs)
    )
    window._overrides["_refresh_localize_controls"] = (
        lambda self: refresh_calls.__setitem__("count", refresh_calls["count"] + 1)
    )

    window._on_settings_configs()

    saved = store.read_yaml("paths.yml", default={})
    assert saved == original_payload
    assert refresh_calls["count"] == 0

    window.close()


def test_mainwindow_stage_block_keeps_current_page_and_placeholder_block(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml("recent_projects.yml", {"recent_projects": []})

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
    )

    block = window._placeholder_block("Custom Placeholder")
    assert block.title() == "Custom Placeholder"
    assert block.layout() is not None
    assert block.layout().count() == 1

    # If the active stage is disabled by prerequisites, the current page stays visible.
    window._active_stage_key = "tensor"
    window._stage_states["preproc"] = "gray"
    window._refresh_stage_controls()
    assert window._active_stage_key == "tensor"
    assert not window._stage_buttons["tensor"].isEnabled()
    assert window._stage_buttons["tensor"].isChecked()

    # Cover combo helper no-op branch with `None`.
    window._set_combo_values(None, [], "placeholder")

    window._localize_selected_atlas = "AtlasA"
    window._localize_selected_regions = ("SNr",)
    window._localize_region_names_by_atlas = {"AtlasA": ("SNr", "STN")}
    window._refresh_localize_atlas_summary()
    assert window._localize_atlas_summary_label.text() == "1/2 regions selected"

    window.close()


def test_mainwindow_dataset_and_localize_cancel_guard_branches(
    tmp_path: Path,
) -> None:
    _ = QApplication.instance() or QApplication([])

    project = tmp_path / "demo_project"
    subject = "sub-001"
    record = "runA"
    (project / "derivatives" / "lfptensorpipe" / subject / record).mkdir(parents=True)
    (project / "rawdata" / subject / "ses-postop" / "lfp" / record / "raw").mkdir(
        parents=True
    )
    (project / "sourcedata" / subject / "lfp" / record).mkdir(parents=True)

    leaddbs_dir = tmp_path / "leaddbs"
    matlab_root = tmp_path / "matlab"
    (leaddbs_dir / "templates" / "space" / "MNI" / "atlases" / "AtlasA").mkdir(
        parents=True
    )
    matlab_root.mkdir(parents=True)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    store = AppConfigStore(repo_root=repo_root)
    store.ensure_core_files()
    store.write_yaml(
        "recent_projects.yml", {"recent_projects": [str(project.resolve())]}
    )
    store.write_yaml(
        "paths.yml",
        {
            "leaddbs_dir": str(leaddbs_dir),
            "matlab_root": str(matlab_root),
        },
    )

    window = OverrideMainWindow(
        config_store=store,
        demo_data_root=tmp_path / "missing_demo",
        overrides={
            "_select_existing_directory": lambda self, *_args, **_kwargs: "",
            "_prompt_text": lambda self, *_args, **_kwargs: ("sub-001", False),
        },
    )
    window._on_project_add()

    window._current_project = project
    window._on_subject_add()

    window._current_subject = subject

    class _RejectedImportDialog:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            _ = (args, kwargs)

        def exec(self) -> int:
            return QDialog.Rejected

    window._overrides["_create_record_import_dialog"] = (
        lambda self, **kwargs: _RejectedImportDialog(**kwargs)
    )
    window._on_record_add()

    window._overrides["_infer_subject_space_runtime"] = (
        lambda self, *_args, **_kwargs: ("MNI", "")
    )
    window._overrides["_discover_atlases_runtime"] = lambda self, *_args, **_kwargs: [
        "AtlasA"
    ]
    window._refresh_localize_controls()

    class _RejectedAtlasDialog:
        selected_payload = None

        def exec(self) -> int:
            return QDialog.Rejected

    window._current_record = record
    window._overrides["_create_localize_atlas_dialog"] = (
        lambda self, **kwargs: _RejectedAtlasDialog()
    )
    window._on_localize_atlas_configure()
    assert window._localize_selected_atlas is None

    window._current_project = None
    window._on_contact_viewer()

    window.close()
