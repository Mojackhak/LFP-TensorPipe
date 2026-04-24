"""Record import dialog."""

from __future__ import annotations

from typing import Any

from .common import (
    AppConfigStore,
    Path,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    Qt,
    QVBoxLayout,
    QWidget,
    RECORD_CONFIG_FILENAME,
    RECORD_IMPORT_TYPES,
    RECORD_RESET_REFERENCE_DEFAULTS_KEY,
)
from .dataset_types import ParsedImportPreview, ResetReferenceRow
from .import_sync import ImportSyncDialog
from .record_import_actions import (
    _browse_sidecar as _browse_sidecar_impl,
    _collect_parse_request as _collect_parse_request_impl,
    _format_parse_result as _format_parse_result_impl,
    _main_file_filter as _main_file_filter_impl,
    _on_browse_main_file as _on_browse_main_file_impl,
    _on_browse_marker as _on_browse_marker_impl,
    _on_browse_metadata as _on_browse_metadata_impl,
    _on_confirm as _on_confirm_impl,
    _on_parse as _on_parse_impl,
    _on_reset_configure as _on_reset_configure_impl,
    _on_sync_configure as _on_sync_configure_impl,
    _show_parse_error as _show_parse_error_impl,
)
from .record_import_state import (
    _invalidate_parse_state as _invalidate_parse_state_impl,
    _is_parse_ready as _is_parse_ready_impl,
    _set_result_placeholder as _set_result_placeholder_impl,
    _update_confirm_button_state as _update_confirm_button_state_impl,
    _update_parse_button_state as _update_parse_button_state_impl,
    _update_sync_configure_button_state as _update_sync_configure_button_state_impl,
    _update_type_visibility as _update_type_visibility_impl,
)


class RecordImportDialog(QDialog):
    """Record import modal with Parse -> Confirm flow."""

    ERROR_TITLE_BY_CODE = {
        "PARSE_INPUT_MISSING_KEY": "Missing Input",
        "PARSE_INPUT_FILE_NOT_FOUND": "File Not Found",
        "PARSE_INPUT_FILE_TYPE_MISMATCH": "File Type Mismatch",
        "PARSE_SIDECAR_NOT_FOUND": "Required Sidecar Missing",
        "PARSE_SIDECAR_AMBIGUOUS": "Ambiguous Sidecar Files",
        "PARSE_VERSION_UNSUPPORTED": "Unsupported Version",
        "PARSE_SCHEMA_INVALID": "Invalid File Content",
        "PARSE_TIMELINE_INVALID": "Invalid Timeline",
        "PARSE_CHANNEL_MAP_INVALID": "Invalid Channel Mapping",
        "PARSE_UNIT_NORMALIZATION_FAILED": "Unit Conversion Failed",
        "PARSE_INTERNAL_ERROR": "Import Failed",
    }

    def __init__(
        self,
        *,
        project_root: Path,
        existing_records: tuple[str, ...],
        default_import_type: str,
        config_store: AppConfigStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Record")
        self.setMinimumSize(500, 300)
        self.resize(500, 300)
        self._config_store = config_store
        self._project_root = project_root
        self._existing_records = set(existing_records)
        self._parsed: ParsedImportPreview | None = None
        self._sync_state = None
        self._reset_rows: tuple[ResetReferenceRow, ...] = ()
        self._parsed_channel_signature: tuple[str, ...] = ()
        self._record_name_edited = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        left = QWidget()
        left_layout = QGridLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setHorizontalSpacing(6)
        left_layout.setVerticalSpacing(4)
        row = 0
        left_layout.addWidget(QLabel("Import Type"), row, 0)
        self._import_type_combo = QComboBox()
        self._import_type_combo.addItems(list(RECORD_IMPORT_TYPES))
        if default_import_type in RECORD_IMPORT_TYPES:
            self._import_type_combo.setCurrentText(default_import_type)
        self._import_type_combo.setToolTip(
            "Choose the parser that matches the source file format."
        )
        left_layout.addWidget(self._import_type_combo, row, 1)
        row += 1

        left_layout.addWidget(QLabel("Record Name"), row, 0)
        self._record_name_edit = QLineEdit()
        self._record_name_edit.textEdited.connect(self._on_record_name_edited)
        self._record_name_edit.setToolTip(
            "Record name to create under the current subject."
        )
        left_layout.addWidget(self._record_name_edit, row, 1)
        row += 1

        left_layout.addWidget(QLabel("File Path"), row, 0)
        file_row = QWidget()
        file_row_layout = QHBoxLayout(file_row)
        file_row_layout.setContentsMargins(0, 0, 0, 0)
        file_row_layout.setSpacing(4)
        self._file_path_edit = QLineEdit()
        self._file_path_edit.setToolTip("Source file path to parse and import.")
        self._file_path_browse_button = QPushButton("Browse")
        self._file_path_browse_button.clicked.connect(self._on_browse_main_file)
        self._file_path_browse_button.setToolTip(
            "Select the source file to parse and import."
        )
        file_row_layout.addWidget(self._file_path_edit, stretch=1)
        file_row_layout.addWidget(self._file_path_browse_button)
        left_layout.addWidget(file_row, row, 1)
        row += 1

        advanced_row = QWidget()
        advanced_row_layout = QHBoxLayout(advanced_row)
        advanced_row_layout.setContentsMargins(0, 0, 0, 0)
        advanced_row_layout.setSpacing(4)
        self._advanced_check = QCheckBox("Advanced")
        self._advanced_check.setToolTip(
            "Show optional sidecar inputs for supported import types."
        )
        advanced_row_layout.addWidget(self._advanced_check)
        advanced_row_layout.addStretch(1)
        left_layout.addWidget(advanced_row, row, 0, 1, 2)
        row += 1

        self._metadata_row_widget = QWidget()
        metadata_row_layout = QHBoxLayout(self._metadata_row_widget)
        metadata_row_layout.setContentsMargins(0, 0, 0, 0)
        metadata_row_layout.setSpacing(4)
        self._metadata_path_edit = QLineEdit()
        self._metadata_path_edit.setToolTip(
            "Optional metadata sidecar path for the selected import type."
        )
        self._metadata_browse_button = QPushButton("Meta")
        self._metadata_browse_button.clicked.connect(self._on_browse_metadata)
        self._metadata_browse_button.setToolTip("Select the metadata sidecar file.")
        metadata_row_layout.addWidget(self._metadata_path_edit, stretch=1)
        metadata_row_layout.addWidget(self._metadata_browse_button)
        self._metadata_label = QLabel("Metadata")
        left_layout.addWidget(self._metadata_label, row, 0)
        left_layout.addWidget(self._metadata_row_widget, row, 1)
        row += 1

        self._marker_row_widget = QWidget()
        marker_row_layout = QHBoxLayout(self._marker_row_widget)
        marker_row_layout.setContentsMargins(0, 0, 0, 0)
        marker_row_layout.setSpacing(4)
        self._marker_path_edit = QLineEdit()
        self._marker_path_edit.setToolTip(
            "Optional marker sidecar path for the selected import type."
        )
        self._marker_browse_button = QPushButton("Marker")
        self._marker_browse_button.clicked.connect(self._on_browse_marker)
        self._marker_browse_button.setToolTip("Select the marker sidecar file.")
        marker_row_layout.addWidget(self._marker_path_edit, stretch=1)
        marker_row_layout.addWidget(self._marker_browse_button)
        self._marker_label = QLabel("Marker")
        left_layout.addWidget(self._marker_label, row, 0)
        left_layout.addWidget(self._marker_row_widget, row, 1)
        row += 1

        self._csv_sr_edit = QLineEdit()
        self._csv_sr_edit.setToolTip("Sampling rate in Hz for Legacy (CSV) imports.")
        self._csv_sr_label = QLabel("Sampling rate")
        left_layout.addWidget(self._csv_sr_label, row, 0)
        left_layout.addWidget(self._csv_sr_edit, row, 1)
        row += 1

        self._csv_unit_combo = QComboBox()
        self._csv_unit_combo.addItems(["V", "mV", "uV", "nV"])
        self._csv_unit_combo.setToolTip("Signal unit for Legacy (CSV) imports.")
        self._csv_unit_label = QLabel("Unit")
        left_layout.addWidget(self._csv_unit_label, row, 0)
        left_layout.addWidget(self._csv_unit_combo, row, 1)
        row += 1

        self._parse_action_row = QWidget()
        parse_action_layout = QHBoxLayout(self._parse_action_row)
        parse_action_layout.setContentsMargins(0, 0, 0, 0)
        parse_action_layout.setSpacing(4)
        self._parse_button = QPushButton("Parse")
        self._parse_button.clicked.connect(self._on_parse)
        self._parse_button.setToolTip(
            "Parse the selected source and preview import metadata."
        )
        parse_action_layout.addStretch(1)
        parse_action_layout.addWidget(self._parse_button)
        left_layout.addWidget(self._parse_action_row, row, 0, 1, 2)
        row += 1

        self._sync_panel = QFrame()
        self._sync_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        sync_panel_layout = QVBoxLayout(self._sync_panel)
        sync_panel_layout.setContentsMargins(6, 6, 6, 6)
        sync_panel_layout.setSpacing(4)
        sync_row = QWidget()
        sync_row_layout = QHBoxLayout(sync_row)
        sync_row_layout.setContentsMargins(0, 0, 0, 0)
        sync_row_layout.setSpacing(4)
        self._sync_check = QCheckBox("Sync")
        self._sync_check.setToolTip(
            "Apply marker-based import synchronization before import."
        )
        self._sync_configure_button = QPushButton("Configure...")
        self._sync_configure_button.clicked.connect(self._on_sync_configure)
        self._sync_configure_button.setToolTip(
            "Configure synchronization markers, pairing, and preview."
        )
        sync_row_layout.addWidget(self._sync_check)
        sync_row_layout.addStretch(1)
        sync_row_layout.addWidget(self._sync_configure_button)
        self._sync_summary_label = QLabel("Sync: Off")
        sync_panel_layout.addWidget(sync_row)
        sync_panel_layout.addWidget(self._sync_summary_label)
        left_layout.addWidget(self._sync_panel, row, 0, 1, 2)
        row += 1

        self._reset_panel = QFrame()
        self._reset_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        reset_panel_layout = QVBoxLayout(self._reset_panel)
        reset_panel_layout.setContentsMargins(6, 6, 6, 6)
        reset_panel_layout.setSpacing(4)

        reset_row = QWidget()
        reset_row_layout = QHBoxLayout(reset_row)
        reset_row_layout.setContentsMargins(0, 0, 0, 0)
        reset_row_layout.setSpacing(4)
        self._reset_check = QCheckBox("Reset reference")
        self._reset_check.setToolTip(
            "Apply reset-reference montage before importing the record."
        )
        self._reset_configure_button = QPushButton("Configure...")
        self._reset_configure_button.clicked.connect(self._on_reset_configure)
        self._reset_configure_button.setToolTip(
            "Configure reset-reference pairs for the parsed channels."
        )
        reset_row_layout.addWidget(self._reset_check)
        reset_row_layout.addStretch(1)
        reset_row_layout.addWidget(self._reset_configure_button)
        self._reset_summary_label = QLabel("Pairs: No pairs configured")
        self._reset_summary_label.setToolTip(
            "Current reset-reference configuration summary."
        )
        reset_panel_layout.addWidget(reset_row)
        reset_panel_layout.addWidget(self._reset_summary_label)
        left_layout.addWidget(self._reset_panel, row, 0, 1, 2)
        row += 1
        left_layout.setRowStretch(row, 1)
        body_layout.addWidget(left, stretch=3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        right_layout.addWidget(QLabel("Parse Result"))
        self._result_label = QLabel()
        self._result_label.setWordWrap(True)
        self._result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._result_label.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self._result_label.setToolTip(
            "Parse summary for vendor, version, channels, sample rate, and duration."
        )
        right_layout.addWidget(self._result_label, stretch=1)
        body_layout.addWidget(right, stretch=2)
        root.addWidget(body, stretch=1)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)
        self._cancel_button = QPushButton("Cancel")
        self._confirm_button = QPushButton("Confirm Import")
        self._cancel_button.clicked.connect(self.reject)
        self._confirm_button.clicked.connect(self._on_confirm)
        self._cancel_button.setToolTip("Close without importing.")
        self._confirm_button.setToolTip(
            "Import the parsed record into the current subject."
        )
        footer_layout.addWidget(self._cancel_button)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self._confirm_button)
        root.addWidget(footer)

        self._import_type_combo.currentTextChanged.connect(self._on_import_type_changed)
        self._advanced_check.toggled.connect(self._on_advanced_toggled)
        for edit in (
            self._file_path_edit,
            self._metadata_path_edit,
            self._marker_path_edit,
            self._csv_sr_edit,
        ):
            edit.textChanged.connect(self._on_parse_inputs_changed)
        self._csv_unit_combo.currentTextChanged.connect(self._on_parse_inputs_changed)
        self._sync_check.toggled.connect(self._on_sync_toggled)
        self._reset_check.toggled.connect(self._on_reset_toggled)

        self._update_type_visibility()
        self._invalidate_parse_state()

    @property
    def selected_import_type(self) -> str:
        return self._import_type_combo.currentText().strip()

    @property
    def selected_record_name(self) -> str:
        return self._record_name_edit.text().strip()

    @property
    def parsed_preview(self) -> ParsedImportPreview | None:
        return self._parsed

    @property
    def reset_rows(self) -> tuple[ResetReferenceRow, ...]:
        return self._reset_rows

    @property
    def sync_state(self):
        return self._sync_state

    @property
    def use_sync(self) -> bool:
        return self._sync_check.isChecked()

    @property
    def use_reset_reference(self) -> bool:
        return self._reset_check.isChecked()

    @staticmethod
    def _clean_reset_reference_endpoint(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _load_reset_reference_defaults(self) -> tuple[ResetReferenceRow, ...]:
        payload = self._config_store.read_yaml(RECORD_CONFIG_FILENAME, default={})
        if not isinstance(payload, dict):
            return ()
        raw_rows = payload.get(RECORD_RESET_REFERENCE_DEFAULTS_KEY, [])
        if not isinstance(raw_rows, list):
            return ()
        rows: list[ResetReferenceRow] = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            anode = self._clean_reset_reference_endpoint(item.get("anode", ""))
            cathode = self._clean_reset_reference_endpoint(item.get("cathode", ""))
            name = str(item.get("name", "")).strip()
            if not name or (not anode and not cathode):
                continue
            rows.append(
                ResetReferenceRow(
                    anode=anode,
                    cathode=cathode,
                    name=name,
                )
            )
        return tuple(rows)

    def _save_reset_reference_defaults(
        self,
        rows: tuple[ResetReferenceRow, ...],
    ) -> None:
        payload = self._config_store.read_yaml(RECORD_CONFIG_FILENAME, default={})
        if not isinstance(payload, dict):
            payload = {}
        payload[RECORD_RESET_REFERENCE_DEFAULTS_KEY] = [
            {
                "anode": row.anode,
                "cathode": row.cathode,
                "name": row.name,
            }
            for row in rows
        ]
        self._config_store.write_yaml(RECORD_CONFIG_FILENAME, payload)

    def _set_reset_summary(self) -> None:
        if self._reset_rows:
            self._reset_summary_label.setText(
                f"Pairs: {len(self._reset_rows)} pairs configured"
            )
            return
        self._reset_summary_label.setText("Pairs: No pairs configured")

    def _set_sync_summary(self) -> None:
        if not self._sync_check.isChecked():
            self._sync_summary_label.setText("Sync: Off")
            return
        if self._sync_state is None:
            self._sync_summary_label.setText("Sync: Needs config")
            return
        estimate = self._sync_state.estimate
        self._sync_summary_label.setText(
            f"Sync: Ready ({estimate.pair_count} pairs, lag={estimate.lag_s:.6f} s)"
        )

    @staticmethod
    def _is_type_with_advanced(import_type: str) -> bool:
        return import_type in {"PINS", "Sceneray"}

    def _set_result_placeholder(self) -> None:
        _set_result_placeholder_impl(self)

    def _on_record_name_edited(self, _text: str) -> None:
        self._record_name_edited = True
        self._update_confirm_button_state()

    def _on_import_type_changed(self, _text: str) -> None:
        self._update_type_visibility()
        self._invalidate_parse_state()

    def _on_advanced_toggled(self, _checked: bool) -> None:
        self._update_type_visibility()
        self._invalidate_parse_state()

    def _on_parse_inputs_changed(self, _text: str) -> None:
        self._invalidate_parse_state()

    def _on_sync_toggled(self, _checked: bool) -> None:
        self._update_sync_configure_button_state()
        self._set_sync_summary()
        if self._parsed is not None:
            self._result_label.setText(self._format_parse_result(self._parsed))
        self._update_confirm_button_state()

    def _on_reset_toggled(self, _checked: bool) -> None:
        self._reset_configure_button.setEnabled(
            self._parsed is not None and self._reset_check.isChecked()
        )
        self._update_confirm_button_state()

    def _update_type_visibility(self) -> None:
        _update_type_visibility_impl(self)

    def _is_parse_ready(self) -> bool:
        return _is_parse_ready_impl(self)

    def _update_parse_button_state(self) -> None:
        _update_parse_button_state_impl(self)

    def _update_confirm_button_state(self) -> None:
        _update_confirm_button_state_impl(self)

    def _update_sync_configure_button_state(self) -> None:
        _update_sync_configure_button_state_impl(self)

    def _invalidate_parse_state(self) -> None:
        _invalidate_parse_state_impl(self)

    def _main_file_filter(self, import_type: str) -> str:
        return _main_file_filter_impl(self, import_type)

    def _on_browse_main_file(self) -> None:
        _on_browse_main_file_impl(self)

    def _browse_sidecar(self, *, title: str) -> str:
        return _browse_sidecar_impl(self, title=title)

    def _create_import_sync_dialog(self, *, raw: Any, current_state: Any) -> QDialog:
        return ImportSyncDialog(raw=raw, current_state=current_state, parent=self)

    def _build_import_synced_raw(self, raw: Any, estimate: Any) -> Any:
        from lfptensorpipe.app import build_import_synced_raw

        return build_import_synced_raw(raw, estimate)

    def _on_browse_metadata(self) -> None:
        _on_browse_metadata_impl(self)

    def _on_browse_marker(self) -> None:
        _on_browse_marker_impl(self)

    def _collect_parse_request(
        self,
    ) -> tuple[str, dict[str, str], dict[str, Any] | None, Path]:
        return _collect_parse_request_impl(self)

    def _show_parse_error(self, exc: Exception) -> None:
        _show_parse_error_impl(self, exc)

    def _format_parse_result(self, preview: ParsedImportPreview) -> str:
        return _format_parse_result_impl(self, preview)

    def _on_parse(self) -> None:
        _on_parse_impl(self)

    def _on_reset_configure(self) -> None:
        _on_reset_configure_impl(self)

    def _on_sync_configure(self) -> None:
        _on_sync_configure_impl(self)

    def _on_confirm(self) -> None:
        _on_confirm_impl(self)
