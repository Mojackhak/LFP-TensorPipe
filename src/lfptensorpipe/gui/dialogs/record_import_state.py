"""State helpers for the record import dialog."""

from __future__ import annotations


def _set_result_placeholder(dialog) -> None:
    dialog._result_label.setText("No parse result yet.")


def _update_type_visibility(dialog) -> None:
    import_type = dialog.selected_import_type
    use_advanced = dialog._advanced_check.isChecked() and dialog._is_type_with_advanced(
        import_type
    )
    dialog._advanced_check.setVisible(True)

    show_metadata = import_type in {"PINS", "Sceneray"} and use_advanced
    show_marker = import_type == "PINS" and use_advanced
    show_csv = import_type == "Legacy (CSV)"
    for widget, visible in (
        (dialog._metadata_label, show_metadata),
        (dialog._metadata_row_widget, show_metadata),
        (dialog._marker_label, show_marker),
        (dialog._marker_row_widget, show_marker),
        (dialog._csv_sr_label, show_csv),
        (dialog._csv_sr_edit, show_csv),
        (dialog._csv_unit_label, show_csv),
        (dialog._csv_unit_combo, show_csv),
    ):
        widget.setVisible(visible)


def _is_parse_ready(dialog) -> bool:
    if not dialog._file_path_edit.text().strip():
        return False
    if dialog.selected_import_type == "Legacy (CSV)":
        if not dialog._csv_sr_edit.text().strip():
            return False
        if not dialog._csv_unit_combo.currentText().strip():
            return False
    return True


def _update_parse_button_state(dialog) -> None:
    dialog._parse_button.setEnabled(_is_parse_ready(dialog))


def _update_sync_configure_button_state(dialog) -> None:
    dialog._sync_configure_button.setEnabled(
        dialog._parsed is not None and dialog._sync_check.isChecked()
    )


def _update_confirm_button_state(dialog) -> None:
    if dialog._parsed is None:
        dialog._confirm_button.setEnabled(False)
        return
    if dialog._sync_check.isChecked() and dialog._sync_state is None:
        dialog._confirm_button.setEnabled(False)
        return
    if dialog._reset_check.isChecked() and len(dialog._reset_rows) == 0:
        dialog._confirm_button.setEnabled(False)
        return
    dialog._confirm_button.setEnabled(True)


def _invalidate_parse_state(dialog) -> None:
    dialog._parsed = None
    dialog._sync_state = None
    dialog._parsed_channel_signature = ()
    _set_result_placeholder(dialog)
    dialog._set_sync_summary()
    _update_sync_configure_button_state(dialog)
    dialog._reset_configure_button.setEnabled(False)
    _update_parse_button_state(dialog)
    _update_confirm_button_state(dialog)
