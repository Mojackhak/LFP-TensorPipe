"""Internal packaged-app smoke runners."""

from __future__ import annotations

from collections import deque
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from lfptensorpipe.app.alignment_service import (
    alignment_paradigm_log_path,
    create_alignment_paradigm,
    delete_alignment_paradigm,
    update_alignment_paradigm,
)
from lfptensorpipe.app.config_store import AppConfigStore
from lfptensorpipe.app.dataset.source_parser import parse_record_source
from lfptensorpipe.app.path_resolver import PathResolver, RecordContext
from lfptensorpipe.app.preproc_service import (
    preproc_step_log_path,
    preproc_step_raw_path,
    rawdata_input_fif_path,
)
from lfptensorpipe.app.shared.runlog_store import read_ui_state, write_ui_state
from lfptensorpipe.app.tensor.orchestration import run_build_tensor
from lfptensorpipe.app.runlog_store import (
    RunLogRecord,
    append_run_log_event,
    write_run_log,
)
from lfptensorpipe.app.shared.config_store import CONFIG_FILE_BASENAMES
from lfptensorpipe.app.tensor_service import (
    tensor_metric_log_path,
    tensor_metric_tensor_path,
)
from lfptensorpipe.gui.dialogs.dataset_types import ParsedImportPreview
from lfptensorpipe.gui.main_window import MainWindow
from lfptensorpipe.gui.shell.preproc_plot_worker import run_preproc_plot_worker
from lfptensorpipe.io.pkl_io import load_pkl, save_pkl


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(["lfptensorpipe-smoke"])
    return app


def _drain_events(app: QApplication, *, ms: int = 200) -> None:
    deadline = time.monotonic() + (max(int(ms), 0) / 1000.0)
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


def _close_aux_windows(app: QApplication, keep: MainWindow | None = None) -> None:
    for widget in list(app.topLevelWidgets()):
        if keep is not None and widget is keep:
            continue
        with contextlib.suppress(Exception):
            widget.close()
    with contextlib.suppress(Exception):
        import matplotlib.pyplot as plt

        plt.close("all")
    _drain_events(app, ms=150)


def _copy_user_app_configs(target_store: AppConfigStore) -> None:
    source_store = AppConfigStore()
    target_store.ensure_core_files()
    for filename in CONFIG_FILE_BASENAMES:
        source_path = source_store.path_for(filename)
        if not source_path.exists():
            continue
        payload = source_store.read_yaml(filename, default={})
        target_store.write_yaml(filename, payload)


def _build_smoke_window(
    *,
    project_root: Path,
    enable_plots: bool,
) -> tuple[QApplication, MainWindow, AppConfigStore, tempfile.TemporaryDirectory[str]]:
    temp_dir = tempfile.TemporaryDirectory(prefix="lfptp-desktop-smoke-")
    store = AppConfigStore(repo_root=Path(temp_dir.name) / "app_store")
    _copy_user_app_configs(store)
    store.write_yaml(
        "recent_projects.yml",
        {"recent_projects": [str(project_root.expanduser().resolve())]},
    )
    app = _ensure_app()
    window = MainWindow(
        config_store=store,
        demo_data_root=Path(temp_dir.name) / "missing_demo",
        enable_plots=enable_plots,
    )
    window.show()
    _drain_events(app, ms=250)
    return app, window, store, temp_dir


def _select_window_context(
    window: MainWindow,
    *,
    project_root: Path,
    subject: str,
    record: str | None = None,
) -> None:
    project_value = str(project_root.expanduser().resolve())
    if window._project_combo is None or window._subject_combo is None:
        raise RuntimeError("Dataset context controls are unavailable.")
    project_idx = window._project_combo.findData(project_value)
    if project_idx < 0:
        raise RuntimeError(f"Project is not selectable: {project_value}")
    window._project_combo.setCurrentIndex(project_idx)
    _drain_events(_ensure_app(), ms=100)
    subject_idx = window._subject_combo.findData(subject)
    if subject_idx < 0:
        raise RuntimeError(f"Subject is not selectable: {subject}")
    window._subject_combo.setCurrentIndex(subject_idx)
    _drain_events(_ensure_app(), ms=100)
    if record is not None:
        if not window._select_record_item(record):
            raise RuntimeError(f"Record is not selectable: {record}")
        _drain_events(_ensure_app(), ms=150)
        if window._current_record != record:
            raise RuntimeError(f"Record selection did not stick: {record}")


def _override_dialog_sinks(
    window: MainWindow,
    *,
    warnings: list[str],
    infos: list[str],
) -> None:
    window._show_warning = (
        lambda title, message: warnings.append(  # type: ignore[method-assign]
            f"{title}: {message}"
        )
        or 0
    )
    window._show_information = (
        lambda title, message: infos.append(  # type: ignore[method-assign]
            f"{title}: {message}"
        )
        or 0
    )


def _set_open_file_queue(window: MainWindow, paths: list[Path]) -> None:
    queue: deque[Path] = deque(path.resolve() for path in paths)

    def _open_file_name(
        _title: str, _start_dir: str, _file_filter: str
    ) -> tuple[str, str]:
        if not queue:
            raise RuntimeError("Smoke open-file queue is empty.")
        return str(queue.popleft()), "JSON files (*.json)"

    window._open_file_name = _open_file_name  # type: ignore[method-assign]


def _ensure_subject_exists(
    window: MainWindow,
    *,
    project_root: Path,
    subject: str,
) -> None:
    subjects = window._discover_subjects_runtime(project_root)
    if subject in subjects:
        return
    created, message = window._create_subject_runtime(project_root, subject)
    if not created:
        raise RuntimeError(message)


def _delete_record_if_exists(
    window: MainWindow,
    *,
    project_root: Path,
    subject: str,
    record: str,
) -> None:
    existing = tuple(window._discover_records_runtime(project_root, subject))
    if record not in existing:
        return
    result = window._delete_record_runtime(
        project_root=project_root,
        subject=subject,
        record=record,
        read_only_project_root=window._demo_data_source_readonly,
    )
    if not result.ok:
        raise RuntimeError(result.message)


def _import_record_via_dialog(
    window: MainWindow,
    app: QApplication,
    *,
    import_type: str,
    record_name: str,
    source_path: Path,
    warnings: list[str],
    options: dict[str, Any] | None = None,
) -> None:
    raw, report, is_fif_input = parse_record_source(
        import_type=import_type,
        paths={"file_path": str(source_path)},
        options=options,
    )
    preview = ParsedImportPreview(
        raw=raw,
        report=report,
        source_path=source_path.expanduser().resolve(),
        is_fif_input=is_fif_input,
        import_type=import_type,
    )

    class _FakeRecordImportDialog:
        def exec(self) -> int:
            return QDialog.Accepted

        @property
        def parsed_preview(self) -> ParsedImportPreview | None:
            return preview

        @property
        def selected_record_name(self) -> str:
            return record_name

        @property
        def selected_import_type(self) -> str:
            return import_type

        @property
        def use_reset_reference(self) -> bool:
            return False

        @property
        def reset_rows(self) -> tuple[Any, ...]:
            return ()

    window._create_record_import_dialog = (  # type: ignore[method-assign]
        lambda **_kwargs: _FakeRecordImportDialog()
    )
    warnings_before = len(warnings)
    window._on_record_add()
    _drain_events(app, ms=300)
    if len(warnings) != warnings_before:
        raise RuntimeError(f"Record import failed: {warnings[-1]}")
    if window._current_record != record_name:
        raise RuntimeError(f"Imported record was not selected: {record_name}")


def _load_csv_demo_import_options(metadata_path: Path) -> dict[str, Any]:
    sr: float | None = None
    unit = "uV"
    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        key_lower = key.lower()
        if key_lower == "sampling rate":
            cleaned = value.lower().replace("hz", "").strip()
            sr = float(cleaned)
        elif key_lower == "unit":
            unit = value
    if sr is None or sr <= 0.0:
        raise RuntimeError(f"Invalid CSV sampling rate metadata: {metadata_path}")
    return {"sr": sr, "unit": unit}


def _load_ui_state_payload(path: Path) -> dict[str, Any]:
    payload = read_ui_state(path)
    return dict(payload) if isinstance(payload, dict) else {}


def _write_ui_state_payload(path: Path, payload: dict[str, Any]) -> None:
    write_ui_state(path, payload)


def _copy_snapshot_sections(
    *,
    source_context: RecordContext,
    target_context: RecordContext,
    sections: tuple[str, ...],
) -> None:
    source_payload = _load_ui_state_payload(
        PathResolver(source_context).record_ui_state_path()
    )
    target_path = PathResolver(target_context).record_ui_state_path(create=True)
    target_payload = _load_ui_state_payload(target_path)
    for section in sections:
        if section in source_payload:
            target_payload[section] = json.loads(json.dumps(source_payload[section]))
        else:
            target_payload.pop(section, None)
    _write_ui_state_payload(target_path, target_payload)


def _refresh_record_snapshot(window: MainWindow) -> None:
    window._record_param_dirty_keys.clear()
    window._sync_record_params_from_logs(include_master=True, clear_dirty=True)
    window._refresh_stage_states_from_context()
    window._refresh_preproc_controls()
    window._refresh_localize_controls()
    window._refresh_tensor_controls()
    window._reload_alignment_paradigms()
    window._refresh_alignment_controls()
    window._reload_features_paradigms()
    window._refresh_features_controls()


def _require_status_prefix(
    window: MainWindow,
    *,
    allowed_prefixes: tuple[str, ...],
    action_label: str,
) -> None:
    message = window.statusBar().currentMessage()
    if any(message.startswith(prefix) for prefix in allowed_prefixes):
        return
    raise RuntimeError(f"{action_label} failed: {message}")


def _bootstrap_raw_step(window: MainWindow, context: RecordContext) -> None:
    ok, message = window._run_with_busy(
        "Bootstrap Raw",
        lambda: window._bootstrap_raw_step_from_rawdata_runtime(context),
    )
    window._refresh_stage_states_from_context()
    window._refresh_preproc_controls()
    prefix = "Raw OK" if ok else "Raw failed"
    window.statusBar().showMessage(f"{prefix}: {message}")
    if not ok:
        raise RuntimeError(message)


def _run_reference_preproc_pipeline(
    window: MainWindow,
    *,
    context: RecordContext,
    preproc_snapshot: dict[str, Any],
) -> None:
    filter_snapshot = (
        preproc_snapshot.get("filter", {})
        if isinstance(preproc_snapshot.get("filter"), dict)
        else {}
    )
    basic = (
        filter_snapshot.get("basic", {})
        if isinstance(filter_snapshot.get("basic"), dict)
        else {}
    )
    advance = (
        filter_snapshot.get("advance", {})
        if isinstance(filter_snapshot.get("advance"), dict)
        else {}
    )
    ok_filter, message_filter = window._run_with_busy(
        "Filter Apply",
        lambda: window._apply_filter_step_runtime(
            context,
            advance_params=dict(advance),
            notches=list(basic.get("notches", [])),
            l_freq=float(basic.get("l_freq", 0.0)),
            h_freq=float(basic.get("h_freq", 0.0)),
        ),
    )
    window._refresh_stage_states_from_context()
    if not ok_filter:
        raise RuntimeError(f"Filter Apply failed: {message_filter}")

    annotations = (
        preproc_snapshot.get("annotations", {})
        if isinstance(preproc_snapshot.get("annotations"), dict)
        else {}
    )
    annotation_rows = list(annotations.get("rows", []))
    ok_annotations, message_annotations = window._run_with_busy(
        "Annotations Apply",
        lambda: window._apply_annotations_step_runtime(
            context,
            rows=annotation_rows,
        ),
    )
    window._refresh_stage_states_from_context()
    if not ok_annotations:
        raise RuntimeError(f"Annotations Apply failed: {message_annotations}")

    ok_bad_segment, message_bad_segment = window._run_with_busy(
        "Bad Segment Apply",
        lambda: window._apply_bad_segment_step_runtime(context),
    )
    window._refresh_stage_states_from_context()
    if not ok_bad_segment:
        raise RuntimeError(f"Bad Segment Apply failed: {message_bad_segment}")

    ecg_snapshot = (
        preproc_snapshot.get("ecg", {})
        if isinstance(preproc_snapshot.get("ecg"), dict)
        else {}
    )
    ok_ecg, message_ecg = window._run_with_busy(
        "ECG Apply",
        lambda: window._apply_ecg_step_runtime(
            context,
            method=str(ecg_snapshot.get("method", "svd")),
            picks=[
                str(item)
                for item in ecg_snapshot.get("selected_channels", [])
                if str(item).strip()
            ],
        ),
    )
    window._refresh_stage_states_from_context()
    if not ok_ecg:
        raise RuntimeError(f"ECG Apply failed: {message_ecg}")

    ok_finish, message_finish = window._run_with_busy(
        "Finish Apply",
        lambda: window._apply_finish_step_runtime(context),
    )
    window._refresh_stage_states_from_context()
    if not ok_finish:
        raise RuntimeError(f"Finish Apply failed: {message_finish}")


def _compare_scalar(
    left: Any,
    right: Any,
    *,
    location: str,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(left, (np.generic,)) and not isinstance(left, (np.ndarray,)):
        left = left.item()
    if isinstance(right, (np.generic,)) and not isinstance(right, (np.ndarray,)):
        right = right.item()
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        left_value = float(left)
        right_value = float(right)
        if np.isnan(left_value) and np.isnan(right_value):
            return
        if np.isclose(left_value, right_value, rtol=rtol, atol=atol, equal_nan=True):
            return
        raise RuntimeError(
            f"Numeric mismatch at {location}: {left_value!r} != {right_value!r}"
        )
    if left != right:
        raise RuntimeError(f"Value mismatch at {location}: {left!r} != {right!r}")


def _compare_payload(
    left: Any,
    right: Any,
    *,
    location: str,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        if list(left.columns) != list(right.columns):
            raise RuntimeError(
                f"Column mismatch at {location}: {list(left.columns)!r} != {list(right.columns)!r}"
            )
        if list(left.index) != list(right.index):
            raise RuntimeError(
                f"Index mismatch at {location}: {list(left.index)!r} != {list(right.index)!r}"
            )
        if left.shape != right.shape:
            raise RuntimeError(
                f"Shape mismatch at {location}: {left.shape!r} != {right.shape!r}"
            )
        for col_name in left.columns:
            _compare_payload(
                left[col_name],
                right[col_name],
                location=f"{location}[{col_name!r}]",
                rtol=rtol,
                atol=atol,
            )
        return

    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        if list(left.index) != list(right.index):
            raise RuntimeError(
                f"Series index mismatch at {location}: {list(left.index)!r} != {list(right.index)!r}"
            )
        if left.shape != right.shape:
            raise RuntimeError(
                f"Series shape mismatch at {location}: {left.shape!r} != {right.shape!r}"
            )
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
            right.dtype
        ):
            left_values = left.to_numpy()
            right_values = right.to_numpy()
            matches = np.isclose(
                left_values,
                right_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            if bool(np.all(matches)):
                return
            mismatch_positions = np.flatnonzero(~matches)
            if mismatch_positions.size <= 0:
                raise RuntimeError(f"Numeric mismatch at {location}")
            first_position = int(mismatch_positions[0])
            index_key = left.index[first_position]
            _compare_scalar(
                left_values[first_position],
                right_values[first_position],
                location=f"{location}[{index_key!r}]",
                rtol=rtol,
                atol=atol,
            )
            return
        for key in left.index:
            _compare_payload(
                left.loc[key],
                right.loc[key],
                location=f"{location}[{key!r}]",
                rtol=rtol,
                atol=atol,
            )
        return

    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            raise RuntimeError(
                f"Array shape mismatch at {location}: {left.shape!r} != {right.shape!r}"
            )
        if left.dtype == object or right.dtype == object:
            for idx in np.ndindex(left.shape):
                _compare_payload(
                    left[idx],
                    right[idx],
                    location=f"{location}{idx!r}",
                    rtol=rtol,
                    atol=atol,
                )
            return
        if not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
            max_abs = float(np.nanmax(np.abs(left - right)))
            raise RuntimeError(
                f"Array mismatch at {location}: shape={left.shape!r} max_abs_diff={max_abs}"
            )
        return

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            raise RuntimeError(
                f"Dict key mismatch at {location}: {sorted(left.keys())!r} != {sorted(right.keys())!r}"
            )
        for key in sorted(left.keys(), key=str):
            _compare_payload(
                left[key],
                right[key],
                location=f"{location}.{key}",
                rtol=rtol,
                atol=atol,
            )
        return

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(
                f"Sequence length mismatch at {location}: {len(left)} != {len(right)}"
            )
        for idx, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare_payload(
                left_item,
                right_item,
                location=f"{location}[{idx}]",
                rtol=rtol,
                atol=atol,
            )
        return

    _compare_scalar(left, right, location=location, rtol=rtol, atol=atol)


def _compare_raw_fif_outputs(
    *,
    reference_path: Path,
    candidate_path: Path,
    rtol: float,
    atol: float,
) -> None:
    import mne

    reference_raw = mne.io.read_raw_fif(reference_path, preload=True, verbose="ERROR")
    candidate_raw = mne.io.read_raw_fif(candidate_path, preload=True, verbose="ERROR")
    if list(reference_raw.ch_names) != list(candidate_raw.ch_names):
        raise RuntimeError(
            "Raw FIF channel mismatch: "
            f"{reference_raw.ch_names!r} != {candidate_raw.ch_names!r}"
        )
    if float(reference_raw.info["sfreq"]) != float(candidate_raw.info["sfreq"]):
        raise RuntimeError(
            "Raw FIF sampling-frequency mismatch: "
            f"{reference_raw.info['sfreq']!r} != {candidate_raw.info['sfreq']!r}"
        )
    if int(reference_raw.n_times) != int(candidate_raw.n_times):
        raise RuntimeError(
            f"Raw FIF n_times mismatch: {reference_raw.n_times} != {candidate_raw.n_times}"
        )
    reference_data = reference_raw.get_data()
    candidate_data = candidate_raw.get_data()
    if not np.allclose(
        reference_data,
        candidate_data,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        max_abs = float(np.nanmax(np.abs(reference_data - candidate_data)))
        raise RuntimeError(f"Raw FIF data mismatch: max_abs_diff={max_abs}")


def _is_metadata_sidecar_pickle(path: Path) -> bool:
    return path.name.startswith("._")


def _comparison_rtol_for_pickle(
    relative_path: Path,
    *,
    base_rtol: float,
    platform: str | None = None,
) -> float:
    resolved_platform = sys.platform if platform is None else platform
    if resolved_platform == "win32" and any(
        part in {"periodic", "aperiodic"} for part in relative_path.parts
    ):
        return 7e-3
    return base_rtol


def _compare_pkl_trees(
    *,
    reference_root: Path,
    candidate_root: Path,
    rtol: float,
    atol: float,
    platform: str | None = None,
) -> None:
    reference_files = sorted(
        path.relative_to(reference_root)
        for path in reference_root.glob("**/*.pkl")
        if path.is_file() and not _is_metadata_sidecar_pickle(path)
    )
    candidate_files = sorted(
        path.relative_to(candidate_root)
        for path in candidate_root.glob("**/*.pkl")
        if path.is_file() and not _is_metadata_sidecar_pickle(path)
    )
    if reference_files != candidate_files:
        raise RuntimeError(
            f"PKL file-set mismatch: {reference_files!r} != {candidate_files!r}"
        )
    for relative_path in reference_files:
        reference_payload = load_pkl(reference_root / relative_path)
        candidate_payload = load_pkl(candidate_root / relative_path)
        resolved_rtol = _comparison_rtol_for_pickle(
            relative_path,
            base_rtol=rtol,
            platform=platform,
        )
        _compare_payload(
            reference_payload,
            candidate_payload,
            location=str(relative_path),
            rtol=resolved_rtol,
            atol=atol,
        )


def _apply_localize_config(
    window: MainWindow, config_path: Path, warnings: list[str]
) -> None:
    _set_open_file_queue(window, [config_path])
    warnings_before = len(warnings)
    window._on_localize_import_config()
    if len(warnings) != warnings_before:
        raise RuntimeError(f"Localize config import failed: {warnings[-1]}")
    _require_status_prefix(
        window,
        allowed_prefixes=("Imported Localize config.",),
        action_label="Localize config import",
    )


def _apply_tensor_config(
    window: MainWindow, config_path: Path, warnings: list[str]
) -> None:
    _set_open_file_queue(window, [config_path])
    warnings_before = len(warnings)
    window._on_tensor_import_config()
    if len(warnings) != warnings_before:
        raise RuntimeError(f"Tensor config import failed: {warnings[-1]}")
    _require_status_prefix(
        window,
        allowed_prefixes=("Imported Tensor config:",),
        action_label="Tensor config import",
    )


def _create_and_select_trial(
    window: MainWindow,
    *,
    context: RecordContext,
    trial_name: str,
) -> str:
    created, message, entry = create_alignment_paradigm(
        window._config_store,
        name=trial_name,
        context=context,
    )
    if not created or not isinstance(entry, dict):
        raise RuntimeError(message)
    slug = str(entry.get("slug", "")).strip()
    if not slug:
        raise RuntimeError(f"Invalid trial slug for {trial_name!r}")
    window._reload_alignment_paradigms(preferred_slug=slug)
    window._reload_features_paradigms(preferred_slug=slug)
    if (
        window._alignment_paradigm_list is None
        or window._features_paradigm_list is None
        or window._alignment_row_for_slug(slug) < 0
        or window._features_trial_row_for_slug(slug) < 0
    ):
        raise RuntimeError(f"Trial is not selectable: {slug}")
    return slug


def _apply_alignment_config(
    window: MainWindow,
    *,
    config_path: Path,
    warnings: list[str],
) -> None:
    _set_open_file_queue(window, [config_path])
    warnings_before = len(warnings)
    window._on_alignment_import_config()
    if len(warnings) != warnings_before:
        raise RuntimeError(f"Alignment config import failed: {warnings[-1]}")
    _require_status_prefix(
        window,
        allowed_prefixes=("Imported Align Epochs config:",),
        action_label="Alignment config import",
    )


def _apply_features_config(
    window: MainWindow,
    *,
    config_path: Path,
    warnings: list[str],
) -> None:
    _set_open_file_queue(window, [config_path])
    warnings_before = len(warnings)
    window._on_features_import_config()
    if len(warnings) != warnings_before:
        raise RuntimeError(f"Features config import failed: {warnings[-1]}")
    _require_status_prefix(
        window,
        allowed_prefixes=("Imported Features config:",),
        action_label="Features config import",
    )


def _drop_first_epoch_pick(window: MainWindow) -> None:
    table = window._alignment_epoch_table
    if table is None or table.rowCount() <= 0:
        raise RuntimeError("Epoch Inspector did not expose any epochs to edit.")
    pick_item = table.item(0, 0)
    if pick_item is None:
        raise RuntimeError("Epoch Inspector first-row pick item is missing.")
    table.blockSignals(True)
    try:
        pick_item.setCheckState(Qt.Unchecked)
    finally:
        table.blockSignals(False)
    window._refresh_alignment_select_all_button_label()
    window._refresh_alignment_controls()
    window._persist_alignment_epoch_picks_state()


def run_smoke_numerical_preproc(
    reference_root: str,
    project_root: str,
    subject: str,
    records_root: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> int:
    """Validate CSV preprocess outputs against the approved `ecg` reference."""
    os.environ.setdefault("LFPTP_DISABLE_MATLAB_WARMUP", "1")
    reference_project = Path(reference_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    records_dir = Path(records_root).expanduser().resolve()
    reference_context = RecordContext(
        project_root=reference_project,
        subject=subject,
        record="ecg",
    )
    candidate_context = RecordContext(
        project_root=project,
        subject=subject,
        record="ecg",
    )
    reference_output = (
        PathResolver(reference_context).preproc_root / "finish" / "raw.fif"
    )
    candidate_output = (
        PathResolver(candidate_context).preproc_root / "finish" / "raw.fif"
    )
    app, window, _store, temp_handle = _build_smoke_window(
        project_root=project,
        enable_plots=False,
    )
    warnings: list[str] = []
    infos: list[str] = []
    try:
        _override_dialog_sinks(window, warnings=warnings, infos=infos)
        _ensure_subject_exists(window, project_root=project, subject=subject)
        _select_window_context(
            window, project_root=project, subject=subject, record=None
        )
        _delete_record_if_exists(
            window, project_root=project, subject=subject, record="ecg"
        )
        _select_window_context(
            window, project_root=project, subject=subject, record=None
        )
        _import_record_via_dialog(
            window,
            app,
            import_type="Legacy (CSV)",
            record_name="ecg",
            source_path=records_dir / "csv" / "ecg_contaminated.csv",
            warnings=warnings,
            options=_load_csv_demo_import_options(records_dir / "csv" / "metadata.txt"),
        )
        _copy_snapshot_sections(
            source_context=reference_context,
            target_context=candidate_context,
            sections=("preproc",),
        )
        _refresh_record_snapshot(window)
        reference_snapshot = _load_ui_state_payload(
            PathResolver(reference_context).record_ui_state_path()
        )
        preproc_snapshot = (
            dict(reference_snapshot.get("preproc", {}))
            if isinstance(reference_snapshot.get("preproc"), dict)
            else {}
        )
        _bootstrap_raw_step(window, candidate_context)
        _run_reference_preproc_pipeline(
            window,
            context=candidate_context,
            preproc_snapshot=preproc_snapshot,
        )
        if not candidate_output.exists():
            raise RuntimeError(
                f"Missing preprocess candidate output: {candidate_output}"
            )
        _compare_raw_fif_outputs(
            reference_path=reference_output,
            candidate_path=candidate_output,
            rtol=rtol,
            atol=atol,
        )
        print(f"Numerical preprocess validation: ok candidate={candidate_output}")
    finally:
        window.close()
        _close_aux_windows(app)
        temp_handle.cleanup()
    return 0


def run_smoke_numerical_full_pipeline(
    reference_root: str,
    project_root: str,
    subject: str,
    records_root: str,
    configs_root: str,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> int:
    """Validate the `gait` Localize/Tensor/Align/Features pipeline against `.pkl` reference outputs."""
    disabled_warmup = os.environ.pop("LFPTP_DISABLE_MATLAB_WARMUP", None)
    reference_project = Path(reference_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    records_dir = Path(records_root).expanduser().resolve()
    configs_dir = Path(configs_root).expanduser().resolve()
    candidate_context = RecordContext(
        project_root=project, subject=subject, record="gait"
    )
    reference_context = RecordContext(
        project_root=reference_project,
        subject=subject,
        record="gait",
    )
    app, window, _store, temp_handle = _build_smoke_window(
        project_root=project,
        enable_plots=False,
    )
    warnings: list[str] = []
    infos: list[str] = []
    trial_slugs = ("cycle-l", "turn", "turn-stack", "walk")
    try:
        _override_dialog_sinks(window, warnings=warnings, infos=infos)
        _ensure_subject_exists(window, project_root=project, subject=subject)
        _select_window_context(
            window, project_root=project, subject=subject, record=None
        )
        _delete_record_if_exists(
            window, project_root=project, subject=subject, record="gait"
        )
        _select_window_context(
            window, project_root=project, subject=subject, record=None
        )
        _import_record_via_dialog(
            window,
            app,
            import_type="Legacy (MNE supported)",
            record_name="gait",
            source_path=records_dir / "mne" / "gait.fif",
            warnings=warnings,
        )

        _bootstrap_raw_step(window, candidate_context)
        window._on_preproc_finish_apply()
        _require_status_prefix(
            window,
            allowed_prefixes=("Finish OK:",),
            action_label="Gait preprocess finish",
        )

        _apply_localize_config(
            window,
            configs_dir / "localize" / "lfptensorpipe_localize_config.json",
            warnings,
        )
        window._on_localize_apply()
        _require_status_prefix(
            window,
            allowed_prefixes=("Localize OK:",),
            action_label="Localize apply",
        )

        _apply_tensor_config(
            window,
            configs_dir / "tensor" / "lfptensorpipe_tensor_config.json",
            warnings,
        )
        selected_metrics, mask_edge_effects, metric_params_map = (
            window._collect_tensor_runtime_params(candidate_context)
        )
        ok_tensor, message_tensor = window._run_with_busy(
            "Build Tensor",
            lambda: run_build_tensor(
                candidate_context,
                selected_metrics=selected_metrics,
                mask_edge_effects=mask_edge_effects,
                metric_params_map=metric_params_map,
            ),
        )
        window._refresh_stage_states_from_context()
        window._refresh_tensor_controls()
        prefix = "Build Tensor OK" if ok_tensor else "Build Tensor failed"
        window.statusBar().showMessage(f"{prefix}: {message_tensor}")
        if not ok_tensor:
            raise RuntimeError(message_tensor)

        for trial_slug in trial_slugs:
            created_slug = _create_and_select_trial(
                window,
                context=candidate_context,
                trial_name=trial_slug,
            )
            _apply_alignment_config(
                window,
                config_path=configs_dir
                / "align"
                / f"lfptensorpipe_alignment_{trial_slug}_config.json",
                warnings=warnings,
            )
            window._on_alignment_run()
            _require_status_prefix(
                window,
                allowed_prefixes=("Align Epochs OK:",),
                action_label=f"Align Epochs run ({trial_slug})",
            )
            if trial_slug == "cycle-l":
                _drop_first_epoch_pick(window)
            window._on_alignment_finish()
            _require_status_prefix(
                window,
                allowed_prefixes=("Finish OK:",),
                action_label=f"Align Epochs finish ({trial_slug})",
            )
            window._reload_features_paradigms(preferred_slug=created_slug)
            if (
                window._features_paradigm_list is None
                or window._features_trial_row_for_slug(created_slug) < 0
            ):
                raise RuntimeError(f"Features trial is not selectable: {created_slug}")
            _apply_features_config(
                window,
                config_path=configs_dir
                / "feature"
                / f"lfptensorpipe_features_{trial_slug}_config.json",
                warnings=warnings,
            )
            window._on_features_run_extract()
            _require_status_prefix(
                window,
                allowed_prefixes=("Extract Features OK:",),
                action_label=f"Extract Features ({trial_slug})",
            )

        _compare_pkl_trees(
            reference_root=PathResolver(reference_context).features_root,
            candidate_root=PathResolver(candidate_context).features_root,
            rtol=rtol,
            atol=atol,
        )
        print(
            "Numerical full-pipeline validation: ok "
            f"candidate={PathResolver(candidate_context).features_root}"
        )
    finally:
        if disabled_warmup is not None:
            os.environ["LFPTP_DISABLE_MATLAB_WARMUP"] = disabled_warmup
        window.close()
        _close_aux_windows(app)
        temp_handle.cleanup()
    return 0


def _record_ui_state_backup(context: RecordContext) -> tuple[Path, bytes | None]:
    path = PathResolver(context).record_ui_state_path()
    if path.exists():
        return path, path.read_bytes()
    return path, None


def _restore_record_ui_state(path: Path, payload: bytes | None) -> None:
    if payload is None:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _assert_no_bundle_path_error(window: MainWindow, *, action_label: str) -> None:
    message = window.statusBar().currentMessage()
    if ".app/Contents/Frameworks" in message:
        raise RuntimeError(f"{action_label} resolved a frozen bundle path: {message}")
    if "Plot failed:" in message:
        raise RuntimeError(f"{action_label} failed: {message}")


def _expected_preproc_plot_path(
    context: RecordContext,
    step: str,
) -> Path:
    resolver = PathResolver(context)
    if step == "finish":
        return resolver.preproc_root / "finish" / "raw.fif"
    return preproc_step_raw_path(resolver, step)


def _run_raw_plot_subprocess(raw_path: Path, *, close_ms: int = 1200) -> None:
    if getattr(sys, "frozen", False):
        command = [
            sys.executable,
            "--smoke-raw-plot-fif",
            str(raw_path),
            "--smoke-raw-plot-close-ms",
            str(close_ms),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "lfptensorpipe.main",
            "--smoke-raw-plot-fif",
            str(raw_path),
            "--smoke-raw-plot-close-ms",
            str(close_ms),
        ]
    env = dict(os.environ)
    env.setdefault("LFPTP_DISABLE_MATLAB_WARMUP", "1")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        output = "\n".join(
            part.strip()
            for part in (result.stdout, result.stderr)
            if isinstance(part, str) and part.strip()
        )
        raise RuntimeError(f"Plot subprocess failed for {raw_path}:\n{output}")


def _write_smoke_mne_fixture(target_path: Path) -> Path:
    import mne

    target_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(
        ch_names=["demo-01", "demo-02"],
        sfreq=250.0,
        ch_types=["eeg", "eeg"],
    )
    time_axis = np.linspace(0.0, 2.0 * np.pi, 250, dtype=float)
    data = np.vstack(
        (
            np.sin(time_axis),
            np.cos(time_axis),
        )
    )
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")
    raw.save(target_path, overwrite=True, verbose="ERROR")
    return target_path


def _should_emit_smoke_output(
    *,
    platform: str | None = None,
    frozen: bool | None = None,
) -> bool:
    resolved_platform = sys.platform if platform is None else platform
    resolved_frozen = (
        bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    )
    return not (resolved_platform == "win32" and resolved_frozen)


def _smoke_print(message: str, *, emit_output: bool | None = None) -> None:
    if emit_output is None:
        emit_output = _should_emit_smoke_output()
    if emit_output:
        print(message)


def _legacy_mne_smoke_path(records_root: Path, *, temp_dir: Path) -> Path:
    checked_in_path = records_root / "mne" / "gait.fif"
    if checked_in_path.exists():
        return checked_in_path
    return _write_smoke_mne_fixture(temp_dir / "generated_demo.fif")


def _build_demo_record_cases(
    records_root: Path,
    *,
    temp_dir: Path,
) -> tuple[tuple[str, dict[str, str], dict[str, Any] | None], ...]:
    legacy_mne_path = _legacy_mne_smoke_path(records_root, temp_dir=temp_dir)
    return (
        (
            "Medtronic",
            {
                "file_path": str(
                    records_root
                    / "medrtronic"
                    / "Report_Json_Session_Report_20250227T105610.json"
                )
            },
            None,
        ),
        (
            "PINS",
            {
                "file_path": str(
                    records_root
                    / "PINS"
                    / "EEGRealTime_PATIENT_REDACTED_500Hz_2000-01-01-00-00-00.txt"
                )
            },
            None,
        ),
        (
            "Sceneray",
            {
                "file_path": str(
                    records_root
                    / "sceneray"
                    / "IPG_SERIAL_REDACTED_20000101000000_uv.csv"
                )
            },
            None,
        ),
        (
            "Legacy (CSV)",
            {"file_path": str(records_root / "csv" / "ecg_contaminated.csv")},
            {"sr": 250.0, "unit": "uV"},
        ),
        (
            "Legacy (MNE supported)",
            {"file_path": str(legacy_mne_path)},
            None,
        ),
    )


def _parse_record_source_for_smoke(
    *,
    import_type: str,
    paths: dict[str, str],
    options: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, str], bool]:
    import mne

    with mne.use_log_level("ERROR"):
        return parse_record_source(
            import_type=import_type,
            paths=paths,
            options=options,
        )


def _seed_alignment_trial_green(
    resolver: PathResolver,
    slug: str,
    *,
    metrics: tuple[str, ...],
) -> None:
    write_run_log(
        preproc_step_log_path(resolver, "finish"),
        RunLogRecord(
            step="finish",
            completed=True,
            params={},
            input_path="in",
            output_path=str(preproc_step_raw_path(resolver, "finish")),
            message="finish ready",
        ),
    )
    for metric_key in metrics:
        write_run_log(
            tensor_metric_log_path(resolver, metric_key),
            RunLogRecord(
                step=metric_key,
                completed=True,
                params={},
                input_path="in",
                output_path=str(tensor_metric_tensor_path(resolver, metric_key)),
                message=f"{metric_key} tensor ready",
            ),
        )
    trial_dir = resolver.alignment_paradigm_dir(slug, create=True)
    trial_cfg = {
        "name": slug,
        "slug": slug,
        "trial_slug": slug,
        "method": "stack_warper",
        "method_params": {
            "annotations": ["event"],
            "mode": "exact",
            "duration_range": [0.0, 100.0],
            "drop_bad": False,
            "pad_s": 0.0,
            "sample_rate": 0.4,
        },
        "annotation_filter": {},
    }
    append_run_log_event(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="run_align_epochs",
            completed=True,
            params={
                "trial_slug": slug,
                "name": slug,
                "method": "stack_warper",
                "method_params": dict(trial_cfg["method_params"]),
                "metrics": list(metrics),
            },
            input_path="in",
            output_path=str(trial_dir),
            message=f"{slug} alignment ready",
        ),
        state_patch={"trial_config": trial_cfg},
    )
    (trial_dir / "warp_fn.pkl").write_bytes(b"ok")
    (trial_dir / "warp_labels.pkl").write_bytes(b"ok")
    for metric_key in metrics:
        metric_dir = trial_dir / metric_key
        metric_dir.mkdir(parents=True, exist_ok=True)
        (metric_dir / "tensor_warped.pkl").write_bytes(b"ok")
        save_pkl(
            pd.DataFrame([{"Value": pd.Series([1.0, 2.0], index=[13.0, 30.0])}]),
            metric_dir / "na-raw.pkl",
        )
    append_run_log_event(
        alignment_paradigm_log_path(resolver, slug),
        RunLogRecord(
            step="build_raw_table",
            completed=True,
            params={
                "trial_slug": slug,
                "picked_epoch_indices": [0],
                "merge_location_info_ready": False,
            },
            input_path=str(trial_dir),
            output_path=str(trial_dir),
            message=f"{slug} finish ready",
        ),
    )


def run_smoke_raw_plot(raw_fif_path: str, *, close_ms: int = 1500) -> int:
    """Open one raw FIF browser and auto-close it for frozen-app validation."""
    raw_path = Path(raw_fif_path).expanduser().resolve()
    return run_preproc_plot_worker(
        str(raw_path),
        title=f"Smoke Raw Plot: {raw_path.name}",
        auto_close_ms=max(int(close_ms), 1),
    )


def run_smoke_demo_record_parsers(records_root: str) -> int:
    """Parse the repository demo records inside the packaged runtime."""
    root = Path(records_root).expanduser().resolve()
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lfptp-demo-record-parsers-") as temp_dir:
        cases = _build_demo_record_cases(root, temp_dir=Path(temp_dir))
        for import_type, paths, options in cases:
            try:
                raw, report, is_fif_input = _parse_record_source_for_smoke(
                    import_type=import_type,
                    paths=paths,
                    options=options,
                )
                _smoke_print(
                    f"{import_type}: ok"
                    f" channels={len(raw.ch_names)}"
                    f" n_times={raw.n_times}"
                    f" is_fif_input={is_fif_input}"
                    f" report={report}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{import_type}: {type(exc).__name__}: {exc}")
                _smoke_print(f"{import_type}: fail {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError("Demo record parser smoke failed:\n" + "\n".join(failures))
    return 0


def run_smoke_demo_record_imports(records_root: str) -> int:
    """Drive the packaged MainWindow record-import flow against demo records."""
    root = Path(records_root).expanduser().resolve()
    app = _ensure_app()
    with tempfile.TemporaryDirectory(prefix="lfptp-demo-imports-") as temp_dir:
        project_root = Path(temp_dir) / "project"
        subject = "sub-001"
        (project_root / "derivatives" / "lfptensorpipe" / subject).mkdir(
            parents=True,
            exist_ok=True,
        )
        window, temp_handle = None, None
        try:
            app, window, _store, temp_handle = _build_smoke_window(
                project_root=project_root,
                enable_plots=False,
            )
            warnings: list[str] = []
            infos: list[str] = []
            _override_dialog_sinks(window, warnings=warnings, infos=infos)
            _select_window_context(
                window,
                project_root=project_root,
                subject=subject,
                record=None,
            )

            cases = (
                (
                    import_type,
                    record_name,
                    paths,
                    options,
                )
                for (import_type, paths, options), record_name in zip(
                    _build_demo_record_cases(
                        root,
                        temp_dir=Path(temp_dir) / "fixtures",
                    ),
                    (
                        "medtronic-json",
                        "pins",
                        "sceneray",
                        "legacy-csv",
                        "legacy-fif",
                    ),
                    strict=True,
                )
            )

            for import_type, record_name, paths, options in cases:
                raw, report, is_fif_input = _parse_record_source_for_smoke(
                    import_type=import_type,
                    paths=paths,
                    options=options,
                )
                preview = ParsedImportPreview(
                    raw=raw,
                    report=report,
                    source_path=Path(str(paths["file_path"])).expanduser().resolve(),
                    is_fif_input=is_fif_input,
                    import_type=import_type,
                )

                class _FakeRecordImportDialog:
                    def exec(self) -> int:
                        return QDialog.Accepted

                    @property
                    def parsed_preview(self) -> ParsedImportPreview | None:
                        return preview

                    @property
                    def selected_record_name(self) -> str:
                        return record_name

                    @property
                    def selected_import_type(self) -> str:
                        return import_type

                    @property
                    def use_reset_reference(self) -> bool:
                        return False

                    @property
                    def reset_rows(self) -> tuple[Any, ...]:
                        return ()

                window._create_record_import_dialog = (  # type: ignore[method-assign]
                    lambda **_kwargs: _FakeRecordImportDialog()
                )
                warnings_before = len(warnings)
                window._on_record_add()
                _drain_events(app, ms=250)
                if len(warnings) != warnings_before:
                    raise RuntimeError(
                        f"Record import emitted warnings for {import_type}: {warnings[-1]}"
                    )
                if window._current_record != record_name:
                    raise RuntimeError(
                        f"Imported record was not selected for {import_type}: {record_name}"
                    )
                context = RecordContext(
                    project_root=project_root,
                    subject=subject,
                    record=record_name,
                )
                raw_path = rawdata_input_fif_path(context)
                if not raw_path.exists():
                    raise RuntimeError(
                        f"Imported raw artifact is missing for {import_type}: {raw_path}"
                    )
                _smoke_print(
                    f"{import_type}: imported record={record_name} raw={raw_path}"
                )
        finally:
            if window is not None:
                window.close()
            _close_aux_windows(app)
            if temp_handle is not None:
                temp_handle.cleanup()
    return 0


def run_smoke_demo_config_imports(
    configs_root: str,
    project_root: str,
    subject: str,
    record: str,
) -> int:
    """Import demo configs through the packaged MainWindow config handlers."""
    os.environ.setdefault("LFPTP_DISABLE_MATLAB_WARMUP", "1")
    configs_dir = Path(configs_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    context = RecordContext(project_root=project, subject=subject, record=record)
    resolver = PathResolver(context)
    ui_state_path, ui_state_payload = _record_ui_state_backup(context)
    app, window, store, temp_handle = _build_smoke_window(
        project_root=project,
        enable_plots=False,
    )
    warnings: list[str] = []
    infos: list[str] = []
    created_slug: str | None = None
    try:
        _override_dialog_sinks(window, warnings=warnings, infos=infos)
        _select_window_context(
            window,
            project_root=project,
            subject=subject,
            record=record,
        )

        _set_open_file_queue(
            window,
            [configs_dir / "localize" / "lfptensorpipe_localize_config.json"],
        )
        warnings_before = len(warnings)
        window._on_localize_import_config()
        if len(warnings) != warnings_before:
            raise RuntimeError(f"Localize config import failed: {warnings[-1]}")
        _smoke_print(window.statusBar().currentMessage())

        tensor_paths = sorted((configs_dir / "tensor").glob("*.json"))
        _set_open_file_queue(window, tensor_paths)
        for config_path in tensor_paths:
            warnings_before = len(warnings)
            window._on_tensor_import_config()
            if len(warnings) != warnings_before:
                raise RuntimeError(
                    f"Tensor config import failed for {config_path.name}: {warnings[-1]}"
                )
            _smoke_print(window.statusBar().currentMessage())

        created, message, entry = create_alignment_paradigm(
            store,
            name="Smoke Trial",
            context=context,
        )
        if not created or not isinstance(entry, dict):
            raise RuntimeError(message)
        created_slug = str(entry["slug"])
        updated, update_message = update_alignment_paradigm(
            store,
            slug=created_slug,
            method="stack_warper",
            method_params={
                "annotations": ["event"],
                "mode": "exact",
                "duration_range": [0.0, 100.0],
                "drop_bad": False,
                "pad_s": 0.0,
                "sample_rate": 0.4,
            },
            context=context,
        )
        if not updated:
            raise RuntimeError(update_message)

        feature_metric_keys = sorted(
            {
                str(metric_key)
                for config_path in (configs_dir / "feature").glob("*.json")
                for metric_key in (
                    __import__("json")
                    .loads(config_path.read_text(encoding="utf-8"))
                    .get("features", {})
                    .get("axes_by_metric", {})
                    .keys()
                )
            }
        )
        _seed_alignment_trial_green(
            resolver,
            created_slug,
            metrics=tuple(feature_metric_keys or ("raw_power",)),
        )
        window._reload_alignment_paradigms(preferred_slug=created_slug)
        row = window._alignment_row_for_slug(created_slug)
        if row < 0 or window._alignment_paradigm_list is None:
            raise RuntimeError("Smoke alignment trial is not selectable.")
        window._alignment_paradigm_list.setCurrentRow(row)
        _drain_events(app, ms=100)

        alignment_paths = sorted((configs_dir / "align").glob("*.json"))
        _set_open_file_queue(window, alignment_paths)
        for config_path in alignment_paths:
            warnings_before = len(warnings)
            window._on_alignment_import_config()
            if len(warnings) != warnings_before:
                raise RuntimeError(
                    f"Alignment config import failed for {config_path.name}: {warnings[-1]}"
                )
            _smoke_print(window.statusBar().currentMessage())

        window._reload_features_paradigms(preferred_slug=created_slug)
        row = window._features_trial_row_for_slug(created_slug)
        if row < 0 or window._features_paradigm_list is None:
            raise RuntimeError("Smoke features trial is not selectable.")
        window._features_paradigm_list.setCurrentRow(row)
        _drain_events(app, ms=100)

        feature_paths = sorted((configs_dir / "feature").glob("*.json"))
        _set_open_file_queue(window, feature_paths)
        for config_path in feature_paths:
            warnings_before = len(warnings)
            window._on_features_import_config()
            if len(warnings) != warnings_before:
                raise RuntimeError(
                    f"Features config import failed for {config_path.name}: {warnings[-1]}"
                )
            _smoke_print(window.statusBar().currentMessage())
    finally:
        if created_slug is not None:
            with contextlib.suppress(Exception):
                delete_alignment_paradigm(store, slug=created_slug, context=context)
        _restore_record_ui_state(ui_state_path, ui_state_payload)
        window.close()
        _close_aux_windows(app)
        temp_handle.cleanup()
    return 0


def run_smoke_tensor_runtime(
    configs_root: str,
    project_root: str,
    subject: str,
    record: str,
) -> int:
    """Run the packaged Build Tensor dependency-critical paths."""
    os.environ.setdefault("LFPTP_DISABLE_MATLAB_WARMUP", "1")
    configs_dir = Path(configs_root).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    context = RecordContext(project_root=project, subject=subject, record=record)
    tensor_config_path = (
        configs_dir / "tensor" / "lfptensorpipe_tensor_config.json"
    ).resolve()
    if not tensor_config_path.exists():
        raise RuntimeError(f"Smoke tensor config is missing: {tensor_config_path}")
    tensor_payload = json.loads(tensor_config_path.read_text(encoding="utf-8")).get(
        "tensor", {}
    )
    metric_params_map = dict(tensor_payload.get("metric_params", {}) or {})
    selected_metrics = ["psi", "periodic_aperiodic"]
    missing_metrics = [
        metric_key
        for metric_key in selected_metrics
        if metric_key not in metric_params_map
    ]
    if missing_metrics:
        raise RuntimeError(
            "Smoke tensor runtime is missing metric params for: "
            + ", ".join(missing_metrics)
        )

    resolver = PathResolver(context)
    stale_paths = {
        tensor_metric_tensor_path(resolver, "psi").parent,
        tensor_metric_tensor_path(resolver, "periodic_aperiodic").parent,
        tensor_metric_tensor_path(resolver, "aperiodic").parent,
    }
    for stale_path in stale_paths:
        if stale_path.is_dir():
            shutil.rmtree(stale_path, ignore_errors=True)
    with contextlib.suppress(FileNotFoundError):
        (resolver.tensor_root / "lfptensorpipe_log.json").unlink()

    ok, message = run_build_tensor(
        context,
        selected_metrics=selected_metrics,
        metric_params_map={
            metric_key: dict(metric_params_map[metric_key])
            for metric_key in selected_metrics
        },
        mask_edge_effects=bool(tensor_payload.get("mask_edge_effects", True)),
    )
    if not ok:
        raise RuntimeError(message)

    output_checks = (
        ("psi log", tensor_metric_log_path(resolver, "psi")),
        ("psi tensor", tensor_metric_tensor_path(resolver, "psi")),
        (
            "periodic log",
            tensor_metric_log_path(resolver, "periodic_aperiodic"),
        ),
        (
            "periodic tensor",
            tensor_metric_tensor_path(resolver, "periodic_aperiodic"),
        ),
        (
            "aperiodic tensor",
            tensor_metric_tensor_path(resolver, "aperiodic"),
        ),
    )
    for label, output_path in output_checks:
        if not output_path.exists():
            raise RuntimeError(
                f"Smoke tensor runtime did not write {label}: {output_path}"
            )
    report_dir = tensor_metric_tensor_path(resolver, "periodic_aperiodic").parent / (
        "specparam_report"
    )
    if not report_dir.exists():
        raise RuntimeError(
            f"Smoke tensor runtime did not write SpecParam reports: {report_dir}"
        )
    if not any(report_dir.glob("*.pdf")):
        raise RuntimeError(
            f"Smoke tensor runtime wrote no SpecParam PDF report: {report_dir}"
        )
    _smoke_print(
        "Tensor runtime smoke: ok "
        f"({tensor_metric_tensor_path(resolver, 'psi')}, {report_dir})"
    )
    return 0


def run_smoke_preproc_ui(project_root: str, subject: str, record: str) -> int:
    """Exercise the packaged Preprocess-page handlers on one real record context."""
    os.environ.setdefault("LFPTP_DISABLE_MATLAB_WARMUP", "1")
    project = Path(project_root).expanduser().resolve()
    context = RecordContext(project_root=project, subject=subject, record=record)
    app, window, _store, temp_handle = _build_smoke_window(
        project_root=project,
        enable_plots=True,
    )
    warnings: list[str] = []
    infos: list[str] = []

    class _AcceptedFilterAdvanceDialog:
        def __init__(self, *, session_params: dict[str, Any], **_kwargs: Any) -> None:
            self.selected_params = dict(session_params)

        def set_restore_callback(self, _callback: Callable[[], None]) -> None:
            return None

        def exec(self) -> int:
            return QDialog.Accepted

    class _AcceptedQcAdvanceDialog:
        def __init__(
            self, *, session_params: dict[str, Any], mode: str, **_kwargs: Any
        ) -> None:
            self.selected_action = "save"
            self.selected_params = dict(session_params)
            self.mode = mode

        def exec(self) -> int:
            return QDialog.Accepted

    class _AcceptedAnnotationDialog:
        def __init__(
            self, *, session_rows: list[dict[str, Any]], **_kwargs: Any
        ) -> None:
            self.selected_rows = tuple(session_rows) or (
                {
                    "description": "smoke",
                    "onset": 0.0,
                    "duration": 0.1,
                },
            )

        def exec(self) -> int:
            return QDialog.Accepted

    try:
        _override_dialog_sinks(window, warnings=warnings, infos=infos)
        window._create_filter_advance_dialog = (  # type: ignore[method-assign]
            lambda **kwargs: _AcceptedFilterAdvanceDialog(**kwargs)
        )
        window._create_qc_advance_dialog = (  # type: ignore[method-assign]
            lambda **kwargs: _AcceptedQcAdvanceDialog(**kwargs)
        )
        window._create_annotation_configure_dialog = (  # type: ignore[method-assign]
            lambda **kwargs: _AcceptedAnnotationDialog(**kwargs)
        )
        window._run_channel_selector = (  # type: ignore[method-assign]
            lambda *, available, **_kwargs: tuple(available[: min(2, len(available))])
        )

        _select_window_context(
            window,
            project_root=project,
            subject=subject,
            record=record,
        )

        window._enable_plots = False
        window._on_preproc_raw_plot()
        _drain_events(app, ms=300)
        _assert_no_bundle_path_error(window, action_label="Raw Plot")
        _run_raw_plot_subprocess(_expected_preproc_plot_path(context, "raw"))

        window._on_preproc_filter_advance()
        if "updated" not in window.statusBar().currentMessage().lower():
            raise RuntimeError(
                f"Filter Advance did not update session parameters: {window.statusBar().currentMessage()}"
            )

        window._on_preproc_filter_apply()
        filter_path = _expected_preproc_plot_path(context, "filter")
        if not filter_path.exists():
            raise RuntimeError(f"Filter output is missing: {filter_path}")
        if "Filter OK:" not in window.statusBar().currentMessage():
            raise RuntimeError(
                f"Filter Apply failed: {window.statusBar().currentMessage()}"
            )
        _run_raw_plot_subprocess(filter_path)

        window._on_preproc_annotations_edit()
        if warnings:
            raise RuntimeError(f"Annotations Configure failed: {warnings[-1]}")
        window._on_preproc_annotations_save()
        annotations_path = _expected_preproc_plot_path(context, "annotations")
        if not annotations_path.exists():
            raise RuntimeError(f"Annotations output is missing: {annotations_path}")
        if "Annotations OK:" not in window.statusBar().currentMessage():
            raise RuntimeError(
                f"Annotations Apply failed: {window.statusBar().currentMessage()}"
            )
        _run_raw_plot_subprocess(annotations_path)

        window._on_preproc_bad_segment_apply()
        bad_segment_path = _expected_preproc_plot_path(context, "bad_segment_removal")
        if not bad_segment_path.exists():
            raise RuntimeError(f"Bad Segment output is missing: {bad_segment_path}")
        if "Bad Segment OK:" not in window.statusBar().currentMessage():
            raise RuntimeError(
                f"Bad Segment Apply failed: {window.statusBar().currentMessage()}"
            )
        _run_raw_plot_subprocess(bad_segment_path)

        window._refresh_preproc_ecg_channel_state(context)
        if window._preproc_ecg_available_channels:
            window._on_preproc_ecg_channels_select()
            if not window._preproc_ecg_selected_channels:
                raise RuntimeError("ECG channel selection did not update.")
            warnings_before = len(warnings)
            window._on_preproc_ecg_apply()
            if len(warnings) != warnings_before:
                raise RuntimeError(f"ECG Apply failed: {warnings[-1]}")
            ecg_path = _expected_preproc_plot_path(context, "ecg_artifact_removal")
            if not ecg_path.exists():
                raise RuntimeError(f"ECG output is missing: {ecg_path}")
            if "ECG OK:" not in window.statusBar().currentMessage():
                raise RuntimeError(
                    f"ECG Apply failed: {window.statusBar().currentMessage()}"
                )
        else:
            warnings_before = len(warnings)
            window._on_preproc_ecg_channels_select()
            if (
                len(warnings) == warnings_before
                or "No channels available" not in warnings[-1]
            ):
                raise RuntimeError("ECG empty-channel state did not surface a warning.")
        if _expected_preproc_plot_path(context, "ecg_artifact_removal").exists():
            _run_raw_plot_subprocess(
                _expected_preproc_plot_path(context, "ecg_artifact_removal")
            )

        window._on_preproc_finish_apply()
        finish_path = _expected_preproc_plot_path(context, "finish")
        if not finish_path.exists():
            raise RuntimeError(f"Finish output is missing: {finish_path}")
        if "Finish OK:" not in window.statusBar().currentMessage():
            raise RuntimeError(
                f"Finish Apply failed: {window.statusBar().currentMessage()}"
            )
        _run_raw_plot_subprocess(finish_path)

        window._refresh_preproc_visualization_controls(context)
        if not window._preproc_viz_available_channels:
            raise RuntimeError("Visualization channels are unavailable.")
        window._on_preproc_viz_channels_select()
        if not window._preproc_viz_selected_channels:
            raise RuntimeError("Visualization channel selection did not update.")
        window._enable_plots = True
        warnings_before = len(warnings)
        window._on_preproc_viz_psd_advance()
        window._on_preproc_viz_psd_plot()
        _drain_events(app, ms=300)
        _close_aux_windows(app, keep=window)
        if len(warnings) != warnings_before:
            raise RuntimeError(f"PSD Plot failed: {warnings[-1]}")

        warnings_before = len(warnings)
        window._on_preproc_viz_tfr_advance()
        window._on_preproc_viz_tfr_plot()
        _drain_events(app, ms=300)
        _close_aux_windows(app, keep=window)
        if len(warnings) != warnings_before:
            raise RuntimeError(f"TFR Plot failed: {warnings[-1]}")

        _smoke_print("Preproc UI smoke: ok")
    finally:
        window.close()
        _close_aux_windows(app)
        temp_handle.cleanup()
    return 0
