"""Action helpers for the record import dialog."""

from __future__ import annotations

from typing import Any

from .common import (
    MNE_SUPPORTED_RECOMMENDED_EXTENSIONS,
    Path,
    QDialog,
    QFileDialog,
    QMessageBox,
    parse_record_source,
    validate_record_name,
)
from .dataset_types import ParsedImportPreview
from .record_import_state import (
    _invalidate_parse_state,
    _update_confirm_button_state,
)
from .reset_reference import ResetReferenceDialog


def _main_file_filter(dialog, import_type: str) -> str:
    if import_type == "Medtronic":
        return "Medtronic JSON (*.json);;All Files (*.*)"
    if import_type == "PINS":
        return "PINS TXT (EEGRealTime_*.txt);;Text Files (*.txt);;All Files (*.*)"
    if import_type == "Sceneray":
        return "Sceneray CSV (*_uv.csv);;CSV Files (*.csv);;All Files (*.*)"
    if import_type == "Legacy (CSV)":
        return "CSV Files (*.csv);;All Files (*.*)"
    ext_tokens = " ".join(f"*{ext}" for ext in MNE_SUPPORTED_RECOMMENDED_EXTENSIONS)
    return f"Recommended ({ext_tokens});;All Files (*.*)"


def _on_browse_main_file(dialog) -> None:
    selected, _ = QFileDialog.getOpenFileName(
        dialog,
        "Select source file",
        str(dialog._project_root),
        _main_file_filter(dialog, dialog.selected_import_type),
    )
    if not selected:
        return
    path = Path(selected).expanduser().resolve()
    dialog._file_path_edit.setText(str(path))
    if not dialog._record_name_edited and not dialog._record_name_edit.text().strip():
        dialog._record_name_edit.setText(path.stem)
    _invalidate_parse_state(dialog)


def _browse_sidecar(dialog, *, title: str) -> str:
    selected, _ = QFileDialog.getOpenFileName(
        dialog,
        title,
        str(dialog._project_root),
        "Text Files (*.txt);;All Files (*.*)",
    )
    return selected


def _on_browse_metadata(dialog) -> None:
    selected = _browse_sidecar(dialog, title="Select metadata sidecar")
    if not selected:
        return
    dialog._metadata_path_edit.setText(str(Path(selected).expanduser().resolve()))
    _invalidate_parse_state(dialog)


def _on_browse_marker(dialog) -> None:
    selected = _browse_sidecar(dialog, title="Select marker sidecar")
    if not selected:
        return
    dialog._marker_path_edit.setText(str(Path(selected).expanduser().resolve()))
    _invalidate_parse_state(dialog)


def _collect_parse_request(dialog) -> tuple[str, dict[str, str], dict[str, Any] | None, Path]:
    import_type = dialog.selected_import_type
    file_path_text = dialog._file_path_edit.text().strip()
    if not file_path_text:
        raise ValueError("File Path is required.")
    source_path = Path(file_path_text).expanduser().resolve()
    paths: dict[str, str] = {"file_path": str(source_path)}
    options: dict[str, Any] | None = None

    if import_type == "PINS" and dialog._advanced_check.isChecked():
        metadata = dialog._metadata_path_edit.text().strip()
        marker = dialog._marker_path_edit.text().strip()
        if metadata:
            paths["metadata_path"] = str(Path(metadata).expanduser().resolve())
        if marker:
            paths["marker_path"] = str(Path(marker).expanduser().resolve())
    elif import_type == "Sceneray" and dialog._advanced_check.isChecked():
        metadata = dialog._metadata_path_edit.text().strip()
        if metadata:
            paths["metadata_path"] = str(Path(metadata).expanduser().resolve())
    elif import_type == "Legacy (CSV)":
        sr_text = dialog._csv_sr_edit.text().strip()
        if not sr_text:
            raise ValueError("Sampling rate is required for Legacy (CSV).")
        try:
            sr_val = float(sr_text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Sampling rate must be numeric.") from exc
        if sr_val <= 0:
            raise ValueError("Sampling rate must be > 0.")
        options = {"sr": sr_val, "unit": dialog._csv_unit_combo.currentText().strip()}

    return import_type, paths, options, source_path


def _show_parse_error(dialog, exc: Exception) -> None:
    code = str(getattr(exc, "code", "PARSE_INTERNAL_ERROR"))
    vendor = str(getattr(exc, "vendor", "unknown"))
    version = str(getattr(exc, "version", "unknown"))
    message = str(getattr(exc, "message", str(exc)))
    title = dialog.ERROR_TITLE_BY_CODE.get(code, "Import Failed")
    popup_body = (
        f"{title}. {message}\n\n"
        f"code: {code}\n"
        f"vendor: {vendor}\n"
        f"version: {version}"
    )
    QMessageBox.warning(dialog, "Import Failed", popup_body)


def _format_parse_result(dialog, preview: ParsedImportPreview) -> str:
    raw = preview.raw
    report = preview.report
    sfreq = float(raw.info["sfreq"]) if raw is not None else 0.0
    duration = float(raw.n_times / sfreq) if sfreq > 0 else 0.0
    return "\n".join(
        [
            f"vendor: {report.get('vendor', 'unknown')}",
            f"version: {report.get('version', 'unknown')}",
            f"status: {report.get('status', 'ok')}",
            f"n_channels: {len(raw.ch_names)}",
            f"sfreq: {sfreq:.2f}",
            f"duration: {duration:.3f} s",
        ]
    )


def _on_parse(dialog) -> None:
    try:
        import_type, paths, options, source_path = _collect_parse_request(dialog)
    except Exception as exc:  # noqa: BLE001
        QMessageBox.warning(dialog, "Import Failed", str(exc))
        return

    try:
        raw, report, is_fif_input = parse_record_source(
            import_type=import_type,
            paths=paths,
            options=options,
        )
    except Exception as exc:  # noqa: BLE001
        _show_parse_error(dialog, exc)
        return

    current_signature = tuple(str(ch) for ch in raw.ch_names)
    if (
        dialog._parsed_channel_signature
        and current_signature != dialog._parsed_channel_signature
        and dialog._reset_rows
    ):
        dialog._reset_rows = ()
        dialog._reset_summary_label.setText("Pairs: No pairs configured")
        QMessageBox.warning(
            dialog,
            "Reset Reference",
            "Pair configuration is invalid after re-parse. Please configure again.",
        )

    dialog._parsed = ParsedImportPreview(
        raw=raw,
        report=report,
        source_path=source_path,
        is_fif_input=bool(is_fif_input),
        import_type=import_type,
    )
    dialog._parsed_channel_signature = current_signature
    dialog._result_label.setText(_format_parse_result(dialog, dialog._parsed))
    dialog._reset_configure_button.setEnabled(dialog._reset_check.isChecked())
    _update_confirm_button_state(dialog)


def _on_reset_configure(dialog) -> None:
    if dialog._parsed is None:
        return
    channels = tuple(str(ch) for ch in dialog._parsed.raw.ch_names)
    if not channels:
        QMessageBox.warning(dialog, "Reset Reference", "No channels detected.")
        return
    reset_dialog = ResetReferenceDialog(
        channel_names=channels,
        current_rows=dialog._reset_rows,
        default_rows=dialog._load_reset_reference_defaults(),
        set_default_callback=dialog._save_reset_reference_defaults,
        parent=dialog,
    )
    if reset_dialog.exec() != QDialog.Accepted:
        return
    dialog._reset_rows = reset_dialog.selected_rows
    dialog._set_reset_summary()
    _update_confirm_button_state(dialog)


def _on_confirm(dialog) -> None:
    if dialog._parsed is None:
        return
    ok, normalized = validate_record_name(dialog.selected_record_name)
    if not ok:
        QMessageBox.warning(dialog, "Import Failed", normalized)
        return
    if normalized in dialog._existing_records:
        QMessageBox.warning(
            dialog,
            "Import Failed",
            f"Record name already exists: {normalized}",
        )
        return
    if dialog._reset_check.isChecked() and len(dialog._reset_rows) == 0:
        QMessageBox.warning(
            dialog,
            "Import Failed",
            "No pair configured. Add at least one pair or disable Reset reference.",
        )
        return
    dialog.accept()
